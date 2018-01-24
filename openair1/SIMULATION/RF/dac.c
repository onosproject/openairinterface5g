/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

//#define DEBUG_DAC
#include <math.h>
#include <stdio.h>
#include "PHY/TOOLS/defs.h"

void dac(double *s_re[2],
         double *s_im[2],
         uint32_t **input,
         uint32_t input_offset,
         uint32_t nb_tx_antennas,
         uint32_t length,
         double amp_dBm,
         uint8_t B,
         uint32_t meas_length,
         uint32_t meas_offset)
{

  int i;
  int aa;
  double V=0.0,amp;

  for (i=0; i<length; i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      s_re[aa][i] = ((double)(((short *)input[aa]))[((i+input_offset)<<1)])/(1<<(B-1));
      s_im[aa][i] = ((double)(((short *)input[aa]))[((i+input_offset)<<1)+1])/(1<<(B-1));

    }
  }

  for (i=meas_offset; i<meas_offset+meas_length; i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      V= V + (s_re[aa][i]*s_re[aa][i]) + (s_im[aa][i]*s_im[aa][i]);
    }
  }

  V /= (meas_length);
#ifdef DEBUG_DAC
  printf("DAC: 10*log10(V)=%f (%f)\n",10*log10(V),V);
#endif

  if (V) {
    amp = pow(10.0,.1*amp_dBm)/V;
    amp = sqrt(amp);
  } else {
    amp = 1;
  }

  for (i=0; i<length; i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      s_re[aa][i] *= amp;
      s_im[aa][i] *= amp;
    }
  }
}
#define dac_SSE
#ifdef  dac_SSE
double dac_fixed_gain(double *s_re[2],
                      double *s_im[2],
                      uint32_t **input,
                      uint32_t input_offset,
                      uint32_t nb_tx_antennas,
                      uint32_t length,
                      uint32_t input_offset_meas,
                      uint32_t length_meas,
                      uint8_t B,
                      double txpwr_dBm,
                      int NB_RE)
{

  int i;
  int aa;
  double amp,amp1,div;
  __m128d input_re128, input_im128;

  amp = //sqrt(NB_RE)*pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
    pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
  amp1 = 0;

  for (aa=0; aa<nb_tx_antennas; aa++) {
    amp1 += sqrt((double)signal_energy((int32_t*)&input[aa][input_offset_meas],length_meas)/NB_RE);
  }

  amp1/=nb_tx_antennas;

  div=amp/amp1;
  for (i=0; i<(length>>1); i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      input_re128=_mm_set_pd(((double)(((short *)input[aa]))[(((2*i+1)+input_offset)<<1)]),((double)(((short *)input[aa]))[((2*i+input_offset)<<1)]));
      input_im128=_mm_set_pd(((double)(((short *)input[aa]))[(((2*i+1)+input_offset)<<1)+1]),((double)(((short *)input[aa]))[((2*i+input_offset)<<1)+1]));
      input_re128=_mm_mul_pd(input_re128,_mm_set1_pd(div));
      input_im128=_mm_mul_pd(input_im128,_mm_set1_pd(div));
      _mm_storeu_pd(&s_re[aa][2*i],input_re128);
      _mm_storeu_pd(&s_im[aa][2*i],input_im128);
    }
  }

  return(signal_energy_fp(s_re,s_im,nb_tx_antennas,length_meas,0)/NB_RE);
}
#else
double dac_fixed_gain(double *s_re[2],
                      double *s_im[2],
                      uint32_t **input,
                      uint32_t input_offset,
                      uint32_t nb_tx_antennas,
                      uint32_t length,
                      uint32_t input_offset_meas,
                      uint32_t length_meas,
                      uint8_t B,
                      double txpwr_dBm,
                      int NB_RE)
{

  int i;
  int aa;
  double amp,amp1,div;

  amp = //sqrt(NB_RE)*pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
    pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
  amp1 = 0;

  for (aa=0; aa<nb_tx_antennas; aa++) {
    amp1 += sqrt((double)signal_energy((int32_t*)&input[aa][input_offset_meas],length_meas)/NB_RE);
  }

  amp1/=nb_tx_antennas;

  //  printf("DAC: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);

  /*
    if (nb_tx_antennas==2)
      amp1 = AMP/2;
    else if (nb_tx_antennas==4)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>16);
    else //assume (nb_tx_antennas==1)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>15);
    amp1 = amp1*sqrt(512.0/300.0); //account for loss due to null carriers
    //printf("DL: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);
  */

  div=amp/amp1;
  for (i=0; i<length; i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      s_re[aa][i] = div*((double)(((short *)input[aa]))[((i+input_offset)<<1)]); ///(1<<(B-1));
      s_im[aa][i] = div*((double)(((short *)input[aa]))[((i+input_offset)<<1)+1]); ///(1<<(B-1));
      //if (i<1024)
	//printf("s_re [%d]%e\n",i,s_re[aa][i]);
    }
  }      	
      /*for (i=0;i<length;i++)
      	printf(" s_re_out[%d] %e, s_im_out[%d] %e, input_re[%d] %e, input_im[%d] %e\n",i,s_re[0][i],i,s_im[0][i],i,((double)(((short *)input[0]))[((i+input_offset)<<1)]),i,((double)(((short *)input[0]))[((i+input_offset)<<1)+1]));*/
  //  printf("ener %e\n",signal_energy_fp(s_re,s_im,nb_tx_antennas,length,0));

  return(signal_energy_fp(s_re,s_im,nb_tx_antennas,length_meas,0)/NB_RE);
}
#endif
double dac_fixed_gain_SSE_float(float *s_re[2],
                      float *s_im[2],
                      uint32_t **input,
                      uint32_t input_offset,
                      uint32_t nb_tx_antennas,
                      uint32_t length,
                      uint32_t input_offset_meas,
                      uint32_t length_meas,
                      uint8_t B,
                      float txpwr_dBm,
                      int NB_RE)
{

  int i;
  int aa;
  float amp,amp1,div;
  __m128 input_re128, input_im128;

  amp = //sqrt(NB_RE)*pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
    pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
  amp1 = 0;

  for (aa=0; aa<nb_tx_antennas; aa++) {
    amp1 += sqrt((float)signal_energy((int32_t*)&input[aa][input_offset_meas],length_meas)/NB_RE);
  }

  amp1/=nb_tx_antennas;

  div=amp/amp1;
  for (i=0; i<(length>>2); i++) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      input_re128=_mm_set_ps(((float)(((short *)input[aa]))[(((4*i+3)+input_offset)<<1)]),((float)(((short *)input[aa]))[(((4*i+2)+input_offset)<<1)]),((float)(((short *)input[aa]))[(((4*i+1)+input_offset)<<1)]),((float)(((short *)input[aa]))[((4*i+input_offset)<<1)]));
      input_im128=_mm_set_ps(((float)(((short *)input[aa]))[(((4*i+3)+input_offset)<<1)+1]),((float)(((short *)input[aa]))[(((4*i+2)+input_offset)<<1)+1]),((float)(((short *)input[aa]))[(((4*i+1)+input_offset)<<1)+1]),((float)(((short *)input[aa]))[((4*i+input_offset)<<1)+1]));
      input_re128=_mm_mul_ps(input_re128,_mm_set1_ps(div));
      input_im128=_mm_mul_ps(input_im128,_mm_set1_ps(div));
      _mm_storeu_ps(&s_re[aa][4*i],input_re128);
      _mm_storeu_ps(&s_im[aa][4*i],input_im128);
    }
  }

  return(signal_energy_fp_SSE_float(s_re,s_im,nb_tx_antennas,length_meas,0)/NB_RE);
}
double dac_fixed_gain_prach(double *s_re[2],
                      double *s_im[2],
                      uint32_t *input,
                      uint32_t input_offset,
                      uint32_t nb_tx_antennas,
                      uint32_t length,
                      uint32_t input_offset_meas,
                      uint32_t length_meas,
                      uint8_t B,
                      double txpwr_dBm,
                      int NB_RE,
		      int ofdm_symbol_size)
{

  int i;
  int aa;
  double amp,amp1;
  int k;

  amp = //sqrt(NB_RE)*pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
    pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
  amp1 = 0;

  for (aa=0; aa<nb_tx_antennas; aa++) {
    amp1 += sqrt((double)signal_energy_prach((int32_t*)&input[input_offset_meas],length_meas)/NB_RE);
  }

  amp1/=nb_tx_antennas;

  //  printf("DAC: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);

  /*
    if (nb_tx_antennas==2)
      amp1 = AMP/2;
    else if (nb_tx_antennas==4)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>16);
    else //assume (nb_tx_antennas==1)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>15);
    amp1 = amp1*sqrt(512.0/300.0); //account for loss due to null carriers
    //printf("DL: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);
  */
#ifdef DEBUG_DAC
  printf("DAC: input_offset %d, amp %e, amp1 %e\n",input_offset,amp,amp1);
#endif
  k=input_offset;
  for (i=0; i<length*2; i+=2) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      s_re[aa][i/2] = amp*((double)(((short *)input))[((k))])/amp1; ///(1<<(B-1));
      s_im[aa][i/2] = amp*((double)(((short *)input))[((k))+1])/amp1; ///(1<<(B-1));
#ifdef DEBUG_DAC
      if(i<20)
      	printf("DAC[%d]: input (%d,%d). output (%e,%e)\n",i/2,(((short *)input))[((k))],(((short *)input))[((k))+1],s_re[aa][i/2],s_im[aa][i/2]);
      if (i>length*2-20&&i<length*2)
	printf("DAC[%d]: input (%d,%d). output (%e,%e)\n",i/2,(((short *)input))[((k))],(((short *)input))[((k))+1],s_re[aa][i/2],s_im[aa][i/2]);
#endif
      k+=2;
      if (k==12*2*ofdm_symbol_size)
 	k=0;
    }
  }

  //  printf("ener %e\n",signal_energy_fp(s_re,s_im,nb_tx_antennas,length,0));

  return(signal_energy_fp(s_re,s_im,nb_tx_antennas,length_meas,0)/NB_RE);
}
float dac_fixed_gain_prach_SSE_float(float *s_re[2],
                      float *s_im[2],
                      uint32_t *input,
                      uint32_t input_offset,
                      uint32_t nb_tx_antennas,
                      uint32_t length,
                      uint32_t input_offset_meas,
                      uint32_t length_meas,
                      uint8_t B,
                      float txpwr_dBm,
                      int NB_RE,
		      int ofdm_symbol_size)
{

  int i;
  int aa;
  float amp,amp1;
  int k;

  amp = //sqrt(NB_RE)*pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
    pow(10.0,.05*txpwr_dBm)/sqrt(nb_tx_antennas); //this is amp per tx antenna
  amp1 = 0;

  for (aa=0; aa<nb_tx_antennas; aa++) {
    amp1 += sqrt((float)signal_energy_prach((int32_t*)&input[input_offset_meas],length_meas)/NB_RE);
  }

  amp1/=nb_tx_antennas;

  //  printf("DAC: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);

  /*
    if (nb_tx_antennas==2)
      amp1 = AMP/2;
    else if (nb_tx_antennas==4)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>16);
    else //assume (nb_tx_antennas==1)
      amp1 = ((AMP*ONE_OVER_SQRT2_Q15)>>15);
    amp1 = amp1*sqrt(512.0/300.0); //account for loss due to null carriers
    //printf("DL: amp1 %f dB (%d,%d), tx_power %f\n",20*log10(amp1),input_offset,input_offset_meas,txpwr_dBm);
  */
#ifdef DEBUG_DAC
  printf("DAC: input_offset %d, amp %e, amp1 %e\n",input_offset,amp,amp1);
#endif
  k=input_offset;
  for (i=0; i<length*2; i+=2) {
    for (aa=0; aa<nb_tx_antennas; aa++) {
      s_re[aa][i/2] = amp*((float)(((short *)input))[((k))])/amp1; ///(1<<(B-1));
      s_im[aa][i/2] = amp*((float)(((short *)input))[((k))+1])/amp1; ///(1<<(B-1));
#ifdef DEBUG_DAC
      if(i<20)
      	printf("DAC[%d]: input (%d,%d). output (%e,%e)\n",i/2,(((short *)input))[((k))],(((short *)input))[((k))+1],s_re[aa][i/2],s_im[aa][i/2]);
      if (i>length*2-20&&i<length*2)
	printf("DAC[%d]: input (%d,%d). output (%e,%e)\n",i/2,(((short *)input))[((k))],(((short *)input))[((k))+1],s_re[aa][i/2],s_im[aa][i/2]);
#endif
      k+=2;
      if (k==12*2*ofdm_symbol_size)
 	k=0;
    }
  }

  //  printf("ener %e\n",signal_energy_fp(s_re,s_im,nb_tx_antennas,length,0));

  return(signal_energy_fp(s_re,s_im,nb_tx_antennas,length_meas,0)/NB_RE);
}
