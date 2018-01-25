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

#include <math.h>
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "PHY/TOOLS/defs.h"
#include "defs.h"

// NEW code with lookup table for sin/cos based on delay profile (TO BE TESTED)

static double **cos_lut=NULL,**sin_lut=NULL;


//#if 1

//#define abstraction_SSE
#ifdef  abstraction_SSE//abstraction_SSE is not working.
int init_freq_channel(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{

  static int first_run=1;
  double delta_f,freq,twopi;  // 90 kHz spacing
  double delay;
  int16_t f;
  uint8_t l;
  __m128d cos_lut128,sin_lut128;
  //static int count=0;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  delta_f = nb_rb*180000/(n_samples-1);

  if (first_run)
  {
	cos_lut = (double **)malloc16(n_samples*sizeof(double*));
	sin_lut = (double **)malloc16(n_samples*sizeof(double*));
	for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
	    cos_lut[f+(n_samples>>1)] = (double *)malloc16_clear((int)desc->nb_taps*sizeof(double));
	    sin_lut[f+(n_samples>>1)] = (double *)malloc16_clear((int)desc->nb_taps*sizeof(double));
	}
	first_run=0;
  }
  twopi=2*M_PI*1e-6*delta_f;
  for (f=-(n_samples>>2); f<0; f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
      cos_lut128=_mm_set_pd(cos(twopi*2*f*delay),cos(twopi*(2*f+1)*delay));
      sin_lut128=_mm_set_pd(sin(twopi*2*f*delay),sin(twopi*(2*f+1)*delay));
      _mm_storeu_pd(&cos_lut[2*f+(n_samples>>1)][l],cos_lut128);
      _mm_storeu_pd(&sin_lut[2*f+(n_samples>>1)][l],sin_lut128);
      //cos_lut[f+(n_samples>>1)][l] = cos(2*M_PI*freq*delay);
      //sin_lut[f+(n_samples>>1)][l] = sin(2*M_PI*freq*delay);
      //printf("values cos:%d, sin:%d\n", cos_lut[f][l], sin_lut[f][l]);

    }
  }
  for (l=0; l<(int)desc->nb_taps; l++) 
  {
      cos_lut[(n_samples>>1)][l] = 1;
      sin_lut[(n_samples>>1)][l] = 0;
      printf("[%d][%d] (cos,sin) (%e,%e):\n",2*f,l,cos_lut[(n_samples>>1)][l],sin_lut[(n_samples>>1)][l]);
  }

  for (f=1; f<=(n_samples>>2); f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
      cos_lut128=_mm_set_pd(cos(twopi*2*f*delay),cos(twopi*(2*f+1)*delay));
      sin_lut128=_mm_set_pd(sin(twopi*2*f*delay),sin(twopi*(2*f+1)*delay));
      _mm_storeu_pd(&cos_lut[2*f+(n_samples>>1)][l],cos_lut128);
      _mm_storeu_pd(&sin_lut[2*f+(n_samples>>1)][l],sin_lut128);
      //cos_lut[f+(n_samples>>1)][l] = cos(2*M_PI*freq*delay);
      //sin_lut[f+(n_samples>>1)][l] = sin(2*M_PI*freq*delay);
      //printf("values cos:%d, sin:%d\n", cos_lut[f][l], sin_lut[f][l]);

    }
  }
  for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
    for (l=0; l<(int)desc->nb_taps; l++) {  
      printf("[%d][%d] (cos,sin) (%e,%e):\n",f,l,cos_lut[f+(n_samples>>1)][l],sin_lut[f+(n_samples>>1)][l]);
    }
  }
  return(0);
}
#else
int init_freq_channel(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{

  static int first_run=1;
  double delta_f,freq;  // 90 kHz spacing
  double delay;
  int16_t f;
  uint8_t l;
  //static int count=0;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  delta_f = nb_rb*180000/(n_samples-1);

  if (first_run)
  {
	cos_lut = (double **)malloc16(n_samples*sizeof(double*));
	sin_lut = (double **)malloc16(n_samples*sizeof(double*));
	for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
	    cos_lut[f+(n_samples>>1)] = (double *)malloc16_clear((int)desc->nb_taps*sizeof(double));
	    sin_lut[f+(n_samples>>1)] = (double *)malloc16_clear((int)desc->nb_taps*sizeof(double));
	}
	first_run=0;
  }

  for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
    //count++;
    freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;

      cos_lut[f+(n_samples>>1)][l] = cos(2*M_PI*freq*delay);
      sin_lut[f+(n_samples>>1)][l] = sin(2*M_PI*freq*delay);
      //printf("[%d][%d] (cos,sin) (%e,%e):\n",f,l,cos_lut[f+(n_samples>>1)][l],sin_lut[f+(n_samples>>1)][l]);

    }
  }
  //printf("count %d\n",count);
  return(0);
}
#endif
#ifdef abstraction_SSE
int freq_channel(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{
  int16_t f,f2,d;
  uint8_t aarx,aatx,l;
  double *clut,*slut;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m128d clut128,slut128,chFx_128,chFy_128;

  // do some error checking
  // n_samples has to be a odd number because we assume the spectrum is symmetric around the DC and includes the DC
  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel: n_samples has to be odd\n");
    return(-1); 
  }

  // printf("no of taps:%d,",(int)desc->nb_taps);

  if (freq_channel_init == 0) {
    // we are initializing the lut for the largets possible n_samples=12*nb_rb+1
    // if called with n_samples<12*nb_rb+1, we decimate the lut
    n_samples_max=12*nb_rb+1;
    if (init_freq_channel(desc,nb_rb,n_samples_max)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  d=(n_samples_max-1)/(n_samples-1);

  //printf("no_samples=%d, n_samples_max=%d, d=%d\n",n_samples,n_samples_max,d);

  start_meas(&desc->interp_freq);

  for (f=-(n_samples_max>>2),f2=-(n_samples>>2); f<(n_samples_max>>2); f+=d,f2++) {
    //clut = cos_lut[(n_samples_max>>1)+f];
    //slut = sin_lut[(n_samples_max>>1)+f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
	chFx_128=_mm_setzero_pd();
	chFy_128=_mm_setzero_pd();
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x=0.0;
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y=0.0;
        for (l=0; l<(int)desc->nb_taps; l++) {
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
	  chFx_128=_mm_add_pd(chFx_128,_mm_add_pd(_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_pd(&cos_lut[(n_samples_max>>1)+2*f][l])),_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_pd(&sin_lut[(n_samples_max>>1)+2*f][l]))));  
	  chFy_128=_mm_add_pd(chFy_128,_mm_sub_pd(_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_pd(&cos_lut[(n_samples_max>>1)+2*f][l])),_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_pd(&sin_lut[(n_samples_max>>1)+2*f][l]))));  
        }
	_mm_storeu_pd(&desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+2*f2].x,chFx_128);
	_mm_storeu_pd(&desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+2*f2].y,chFy_128);
      }
    }
  }

  stop_meas(&desc->interp_freq);

  return(0);
}
#else
int freq_channel(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{


  int16_t f,f2,d;
  uint8_t aarx,aatx,l;
  double *clut,*slut;
  static int freq_channel_init=0;
  static int n_samples_max=0;

  // do some error checking
  // n_samples has to be a odd number because we assume the spectrum is symmetric around the DC and includes the DC
  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel: n_samples has to be odd\n");
    return(-1); 
  }

  // printf("no of taps:%d,",(int)desc->nb_taps);

  if (freq_channel_init == 0) {
    // we are initializing the lut for the largets possible n_samples=12*nb_rb+1
    // if called with n_samples<12*nb_rb+1, we decimate the lut
    n_samples_max=12*nb_rb+1;
    if (init_freq_channel(desc,nb_rb,n_samples_max)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  d=(n_samples_max-1)/(n_samples-1);

  //printf("no_samples=%d, n_samples_max=%d, d=%d\n",n_samples,n_samples_max,d);

  start_meas(&desc->interp_freq);

  for (f=-(n_samples_max>>1),f2=-(n_samples>>1); f<(n_samples_max>>1); f+=d,f2++) {
    clut = cos_lut[(n_samples_max>>1)+f];
    slut = sin_lut[(n_samples_max>>1)+f];

    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x=0.0;
        desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y=0.0;

        for (l=0; l<(int)desc->nb_taps; l++) {

          desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
        }
      }
    }
  }

  stop_meas(&desc->interp_freq);

  return(0);
}
#endif

int init_freq_channel_prach(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{


  double delta_f,freq;  // 90 kHz spacing
  double delay;
  int16_t f,f1;
  uint8_t l;
  int prach_samples, prach_pbr_offset_samples, max_nb_rb_samples;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  if (nb_rb-n_ra_prb<6) {
    fprintf(stderr, "freq_channel_init: Impossible to allocate PRACH, check n_ra_prb value (r_ra_prb=%d)\n",n_ra_prb);
    return(-1); 
  }
  prach_samples = (prach_fmt<4)?13+839+12:3+139+2;
  
  cos_lut = (double **)malloc(prach_samples*sizeof(double*));
  sin_lut = (double **)malloc(prach_samples*sizeof(double*));

  delta_f = (prach_fmt<4)?nb_rb*180000/((n_samples-1)*12):nb_rb*180000/((n_samples-1)*2);//1.25 khz for preamble format 1,2,3. 7.5 khz for preample format 4
  max_nb_rb_samples = nb_rb*180000/delta_f;//7200 if prach_fmt<4
  prach_pbr_offset_samples = (n_ra_prb+6)*180000/delta_f;//864 if n_ra_prb=0,7200 if n_ra_prb=44=50-6
  //printf("prach_samples = %d, delta_f = %e, max_nb_rb_samples= %d, prach_pbr_offset_samples = %d, nb_taps = %d\n",prach_samples,delta_f,max_nb_rb_samples,prach_pbr_offset_samples,desc->nb_taps);
  for (f=max_nb_rb_samples/2-prach_pbr_offset_samples,f1=0; f<max_nb_rb_samples/2-prach_pbr_offset_samples+prach_samples; f++,f1++) {//3600-864,3600-864+864|3600-7200,3600-7200+839
    freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    //printf("[init_freq_channel_prach] freq %e\n",freq);
    cos_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));
    sin_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));


    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;

      cos_lut[f1][l] = cos(2*M_PI*freq*delay);
      sin_lut[f1][l] = sin(2*M_PI*freq*delay);
      //if (f<max_nb_rb_samples/2-prach_pbr_offset_samples+10)
      	//printf("freq: %e, f1: %d, f: %d, arg_sin_cos = %e,  cos () = %e, sin () =n %e)\n",freq, f1,f, 2*M_PI*freq*delay, cos_lut[f1][l], sin_lut[f1][l]);
    }
  }

  return(0);
}
int freq_channel_prach(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  int16_t f;
  uint8_t aarx,aatx,l;
  double *clut,*slut;
  int prach_samples;
  static int freq_channel_init=0;
  static int n_samples_max=0;

  prach_samples = (prach_fmt<4)?13+839+12:3+139+2; 

  // do some error checking
  if (nb_rb-n_ra_prb<6) {
    fprintf(stderr, "freq_channel_init: Impossible to allocate PRACH, check r_ra_prb value (r_ra_prb=%d)\n",n_ra_prb);
    return(-1); 
  }
  if (freq_channel_init == 0) {
    // we are initializing the lut for the largets possible n_samples=12*nb_rb+1
    // if called with n_samples<12*nb_rb+1, we decimate the lut
    n_samples_max=12*nb_rb+1;
    if (init_freq_channel_prach(desc,nb_rb,n_samples_max,prach_fmt,n_ra_prb)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  start_meas(&desc->interp_freq);
  for (f=0; f<prach_samples; f++) {
    clut = cos_lut[f];
    slut = sin_lut[f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].x=0.0;
        desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].y=0.0;
        for (l=0; l<(int)desc->nb_taps; l++) {

          desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
        }
      }
    }
	//if (f<10 || (f>829&&f<839))
	//	printf("chF_prach[0][%d], (x,y) = (%e,%e)\n",f,desc->chF_prach[0][f].x,desc->chF_prach[0][f].y);
  }
  stop_meas(&desc->interp_freq);
  return(0);
}

double compute_pbch_sinr(channel_desc_t *desc,
                         channel_desc_t *desc_i1,
                         channel_desc_t *desc_i2,
                         double snr_dB,double snr_i1_dB,
                         double snr_i2_dB,
                         uint16_t nb_rb)
{

  double avg_sinr,snr=pow(10.0,.1*snr_dB),snr_i1=pow(10.0,.1*snr_i1_dB),snr_i2=pow(10.0,.1*snr_i2_dB);
  uint16_t f;
  uint8_t aarx,aatx;
  double S;
  struct complex S_i1;
  struct complex S_i2;

  avg_sinr=0.0;

  //  printf("nb_rb %d\n",nb_rb);
  for (f=(nb_rb-6); f<(nb_rb+6); f++) {
    S = 0.0;
    S_i1.x =0.0;
    S_i1.y =0.0;
    S_i2.x =0.0;
    S_i2.y =0.0;

    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        S    += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc->chF[aarx+(aatx*desc->nb_rx)][f].x +
                 desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc->chF[aarx+(aatx*desc->nb_rx)][f].y);
        //  printf("%d %d chF[%d] => (%f,%f)\n",aarx,aatx,f,desc->chF[aarx+(aatx*desc->nb_rx)][f].x,desc->chF[aarx+(aatx*desc->nb_rx)][f].y);

        if (desc_i1) {
          S_i1.x += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].x +
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].y);
          S_i1.y += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].y -
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].x);
        }

        if (desc_i2) {
          S_i2.x += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].x +
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].y);
          S_i2.y += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].y -
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].x);
        }
      }
    }

    //    printf("snr %f f %d : S %f, S_i1 %f, S_i2 %f\n",snr,f-nb_rb,S,snr_i1*sqrt(S_i1.x*S_i1.x + S_i1.y*S_i1.y),snr_i2*sqrt(S_i2.x*S_i2.x + S_i2.y*S_i2.y));
    avg_sinr += (snr*S/(desc->nb_tx+snr_i1*sqrt(S_i1.x*S_i1.x + S_i1.y*S_i1.y)+snr_i2*sqrt(S_i2.x*S_i2.x + S_i2.y*S_i2.y)));
  }

  //  printf("avg_sinr %f (%f,%f,%f)\n",avg_sinr/12.0,snr,snr_i1,snr_i2);
  return(10*log10(avg_sinr/12.0));
}


double compute_sinr(channel_desc_t *desc,
                    channel_desc_t *desc_i1,
                    channel_desc_t *desc_i2,
                    double snr_dB,double snr_i1_dB,
                    double snr_i2_dB,
                    uint16_t nb_rb)
{

  double avg_sinr,snr=pow(10.0,.1*snr_dB),snr_i1=pow(10.0,.1*snr_i1_dB),snr_i2=pow(10.0,.1*snr_i2_dB);
  uint16_t f;
  uint8_t aarx,aatx;
  double S;
  struct complex S_i1;
  struct complex S_i2;

  DevAssert( nb_rb > 0 );

  avg_sinr=0.0;

  //  printf("nb_rb %d\n",nb_rb);
  for (f=0; f<2*nb_rb; f++) {
    S = 0.0;
    S_i1.x =0.0;
    S_i1.y =0.0;
    S_i2.x =0.0;
    S_i2.y =0.0;

    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        S    += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc->chF[aarx+(aatx*desc->nb_rx)][f].x +
                 desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc->chF[aarx+(aatx*desc->nb_rx)][f].y);

        if (desc_i1) {
          S_i1.x += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].x +
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].y);
          S_i1.y += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].y -
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i1->chF[aarx+(aatx*desc->nb_rx)][f].x);
        }

        if (desc_i2) {
          S_i2.x += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].x +
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].y);
          S_i2.y += (desc->chF[aarx+(aatx*desc->nb_rx)][f].x*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].y -
                     desc->chF[aarx+(aatx*desc->nb_rx)][f].y*desc_i2->chF[aarx+(aatx*desc->nb_rx)][f].x);
        }
      }
    }

    //        printf("f %d : S %f, S_i1 %f, S_i2 %f\n",f-nb_rb,snr*S,snr_i1*sqrt(S_i1.x*S_i1.x + S_i1.y*S_i1.y),snr_i2*sqrt(S_i2.x*S_i2.x + S_i2.y*S_i2.y));
    avg_sinr += (snr*S/(desc->nb_tx+snr_i1*sqrt(S_i1.x*S_i1.x + S_i1.y*S_i1.y)+snr_i2*sqrt(S_i2.x*S_i2.x + S_i2.y*S_i2.y)));
  }

  //  printf("avg_sinr %f (%f,%f,%f)\n",avg_sinr/12.0,snr,snr_i1,snr_i2);
  return(10*log10(avg_sinr/(nb_rb*2)));
}

int pbch_polynomial_degree=6;
double pbch_awgn_polynomial[7]= {-7.2926e-05, -2.8749e-03, -4.5064e-02, -3.5301e-01, -1.4655e+00, -3.6282e+00, -6.6907e+00};

void load_pbch_desc(FILE *pbch_file_fd)
{

  int i, ret;
  char dummy[25];

  ret = fscanf(pbch_file_fd,"%d",&pbch_polynomial_degree);

  if (ret < 0) {
    printf("fscanf failed: %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  if (pbch_polynomial_degree>6) {
    printf("Illegal degree for pbch interpolation polynomial %d\n",pbch_polynomial_degree);
    exit(-1);
  }

  printf("PBCH polynomial : ");

  for (i=0; i<=pbch_polynomial_degree; i++) {
    ret = fscanf(pbch_file_fd,"%s",dummy);

    if (ret < 0) {
      printf("fscanf failed: %s\n", strerror(errno));
      exit(EXIT_FAILURE);
    }

    pbch_awgn_polynomial[i] = strtod(dummy,NULL);
    printf("%f ",pbch_awgn_polynomial[i]);
  }

  printf("\n");
}

double pbch_bler(double sinr)
{

  int i;
  double log10_bler=pbch_awgn_polynomial[pbch_polynomial_degree];
  double sinrpow=sinr;
  double bler=0.0;

  //  printf("log10bler %f\n",log10_bler);
  if (sinr<-10.0)
    bler= 1.0;
  else if (sinr>=0.0)
    bler=0.0;
  else  {
    for (i=1; i<=pbch_polynomial_degree; i++) {
      //    printf("sinrpow %f\n",sinrpow);
      log10_bler += (pbch_awgn_polynomial[pbch_polynomial_degree-i]*sinrpow);
      sinrpow *= sinr;
      //    printf("log10bler %f\n",log10_bler);
    }

    bler = pow(10.0,log10_bler);
  }

  //printf ("sinr %f bler %f\n",sinr,bler);
  return(bler);

}

