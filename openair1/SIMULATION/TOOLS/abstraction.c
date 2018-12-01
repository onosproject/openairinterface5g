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
 *-------------------------------------------------------------------------------
 * Optimization using SIMD instructions
 * Frecuency Domain Analysis
 * Luis Felipe Ariza Vesga, email:lfarizav@unal.edu.co
 * Functions: init_freq_channel_SSE_float(), freq_channel_SSE_float(),
 * init_freq_channel_prach(), init_freq_channel_prach_SSE_float(),
 * freq_channel_prach (), freq_channel_prach_SSE_float().
 *
 * sincos_ps(), log_ps() --> Functions mofified from Miloyip and Cephes sources.
 * More info https://github.com/miloyip/normaldist-benchmark.
 *-------------------------------------------------------------------------------
 */

#include <math.h>
#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "PHY/TOOLS/defs.h"
#include "defs.h"
#include "PHY/sse_intrin.h"

// NEW code with lookup table for sin/cos based on delay profile (TO BE TESTED)

static double **cos_lut=NULL,**sin_lut=NULL;
static float **cos_lut_f=NULL,**sin_lut_f=NULL,**cos_lut_f_prach=NULL,**sin_lut_f_prach=NULL;

//#if 1

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
    }
  }
  return(0);
}

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

int init_freq_channel_SSE_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{

  static int first_run=1;
  float delta_f,twopi;  // 90 kHz spacing
  float delay;
  int16_t f;
  uint8_t l;
  __m128 cos_lut128,sin_lut128;//,cos_lut128_tmp,sin_lut128_tmp;
  /*__m128 x128, log128, exp128;
  __m256 x256, log256, exp256;
  x128 = _mm_set_ps(1.0,2.0,3.0,4.0);
  x256 = _mm256_set_ps(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0);
  log128=log_ps(x128);
  log256=log256_ps(x256);
  exp128=exp_ps(x128);
  exp256=exp256_ps(x256);
  printf("log128 %e,%e,%e,%e\n",log128[0],log128[1],log128[2],log128[3]);
  printf("log %e,%e,%e,%e\n\n",log(1),log(2),log(3),log(4));
  printf("exp128 %e,%e,%e,%e\n",exp128[0],exp128[1],exp128[2],exp128[3]);
  printf("exp %e,%e,%e,%e\n\n",exp(1),exp(2),exp(3),exp(4));

  printf("log256 %e,%e,%e,%e,%e,%e,%e,%e\n",log256[0],log256[1],log256[2],log256[3],log256[4],log256[5],log256[6],log256[7]);
  printf("log %e,%e,%e,%e,%e,%e,%e,%e\n",log(1),log(2),log(3),log(4),log(5),log(6),log(7),log(8));
  printf("exp256 %e,%e,%e,%e,%e,%e,%e,%e\n",exp256[0],exp256[1],exp256[2],exp256[3],exp256[4],exp256[5],exp256[6],exp256[7]);
  printf("exp %e,%e,%e,%e,%e,%e,%e,%e\n",exp(1),exp(2),exp(3),exp(4),exp(5),exp(6),exp(7),exp(8));*/
  /*__m256 x256, sin256, cos256;
  __m128 x128, sin128, cos128;
  x128 = _mm_set_ps(1.0,2.0,3.0,4.0);
  x256 = _mm256_set_ps(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0);*/

  /*printf("sincos in abstraction.c\n");
  sincos256_ps(x256,&sin256,&cos256);
  sincos_ps(x128,&sin128,&cos128);
  printf("sin avx %e,%e,%e,%e,%e,%e,%e,%e\n",sin256[0],sin256[1],sin256[2],sin256[3],sin256[4],sin256[5],sin256[6],sin256[7]);
  printf("sin %e,%e,%e,%e,%e,%e,%e,%e\n",sin(1.0),sin(2.0),sin(3.0),sin(4.0),sin(5.0),sin(6.0),sin(7.0),sin(8.0));
  printf("cos avx %e,%e,%e,%e,%e,%e,%e,%e\n",cos256[0],cos256[1],cos256[2],cos256[3],cos256[4],cos256[5],cos256[6],cos256[7]);
  printf("cos %e,%e,%e,%e,%e,%e,%e,%e\n\n",cos(1.0),cos(2.0),cos(3.0),cos(4.0),cos(5.0),cos(6.0),cos(7.0),cos(8.0));
  printf("sin sse %e,%e,%e,%e\n",sin128[0],sin128[1],sin128[2],sin128[3]);
  printf("sin %e,%e,%e,%e\n",sin(1.0),sin(2.0),sin(3.0),sin(4.0));
  printf("cos sse %e,%e,%e,%e\n",cos128[0],cos128[1],cos128[2],cos128[3]);
  printf("cos %e,%e,%e,%e\n",cos(1.0),cos(2.0),cos(3.0),cos(4.0));*/

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  delta_f = nb_rb*180000/(n_samples-1);

  if (first_run)
  {
	cos_lut_f = (float **)malloc16(((int)desc->nb_taps)*sizeof(float*));
	sin_lut_f = (float **)malloc16(((int)desc->nb_taps)*sizeof(float*));
	for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
	    cos_lut_f[f+(n_samples>>1)] = (float *)malloc16_clear(n_samples*sizeof(float));
	    sin_lut_f[f+(n_samples>>1)] = (float *)malloc16_clear(n_samples*sizeof(float));
	}
	first_run=0;
  }
  twopi=2*M_PI*1e-6*delta_f;
  for (f=-(n_samples>>3); f<0; f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {

      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
        sincos_ps(_mm_set_ps(twopi*(4*f+3)*delay,twopi*(4*f+2)*delay,twopi*(4*f+1)*delay,twopi*(4*f)*delay), &sin_lut128, &cos_lut128);
        //cos_lut128=_mm_set_ps(cos(twopi*(4*f+3)*delay),cos(twopi*(4*f+2)*delay),cos(twopi*(4*f+1)*delay),cos(twopi*(4*f)*delay));
        //sin_lut128=_mm_set_ps(sin(twopi*(4*f+3)*delay),sin(twopi*(4*f+2)*delay),sin(twopi*(4*f+1)*delay),sin(twopi*(4*f)*delay));
        _mm_storeu_ps(&cos_lut_f[l][4*f+(n_samples>>1)],cos_lut128);
        _mm_storeu_ps(&sin_lut_f[l][4*f+(n_samples>>1)],sin_lut128);
	/*printf("sin128 %e,%e,%e,%e\n",sin_lut128_tmp[0],sin_lut128_tmp[1],sin_lut128_tmp[2],sin_lut128_tmp[3]);
	printf("cos128 %e,%e,%e,%e\n",cos_lut128_tmp[0],cos_lut128_tmp[1],cos_lut128_tmp[2],cos_lut128_tmp[3]);
	printf("sin %e,%e,%e,%e\n",sin_lut128[0],sin_lut128[1],sin_lut128[2],sin_lut128[3]);
	printf("cos %e,%e,%e,%e\n",cos_lut128[0],cos_lut128[1],cos_lut128[2],cos_lut128[3]);*/
    }
  }
  for (l=0; l<(int)desc->nb_taps; l++) 
  {
      cos_lut_f[l][(n_samples>>1)] = 1;
      sin_lut_f[l][(n_samples>>1)] = 0;
      //printf("f %d,l %d (cos,sin) (%e,%e):\n",4*f,l,cos_lut_f[(n_samples>>1)][l],sin_lut_f[(n_samples>>1)][l]);
  }

  for (f=1; f<=(n_samples>>3); f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
      sincos_ps(_mm_set_ps(twopi*(4*f)*delay,twopi*(4*f-1)*delay,twopi*(4*f-2)*delay,twopi*(4*f-3)*delay), &sin_lut128, &cos_lut128);
      //cos_lut128=_mm_set_ps(cos(twopi*(4*f)*delay),cos(twopi*(4*f-1)*delay),cos(twopi*(4*f-2)*delay),cos(twopi*(4*f-3)*delay));
      //sin_lut128=_mm_set_ps(sin(twopi*(4*f)*delay),sin(twopi*(4*f-1)*delay),sin(twopi*(4*f-2)*delay),sin(twopi*(4*f-3)*delay));
      _mm_storeu_ps(&cos_lut_f[l][4*f-3+(n_samples>>1)],cos_lut128);
      _mm_storeu_ps(&sin_lut_f[l][4*f-3+(n_samples>>1)],sin_lut128);
    }
  }
  /*for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
    for (l=0; l<(int)desc->nb_taps; l++) {  
      printf("f %d, l %d (cos,sin) (%e,%e):\n",f,l,cos_lut_f[l][f+(n_samples>>1)],sin_lut_f[l][f+(n_samples>>1)]);
    }
  }*/
  return(0);
}
int init_freq_channel_AVX_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{

  static int first_run=1;
  float delta_f,twopi;  // 90 kHz spacing
  float delay;
  int16_t f;
  uint8_t l;
  __m256 cos_lut256,sin_lut256;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  delta_f = nb_rb*180000/(n_samples-1);

  if (first_run)
  {
	cos_lut_f = (float **)malloc16(((int)desc->nb_taps)*sizeof(float*));
	sin_lut_f = (float **)malloc16(((int)desc->nb_taps)*sizeof(float*));
	for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
	    cos_lut_f[f+(n_samples>>1)] = (float *)malloc16_clear(n_samples*sizeof(float));
	    sin_lut_f[f+(n_samples>>1)] = (float *)malloc16_clear(n_samples*sizeof(float));
	}
	first_run=0;
  }
  twopi=2*M_PI*1e-6*delta_f;
  for (f=-(n_samples>>4); f<0; f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {

      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;

        sincos256_ps(_mm256_set_ps(twopi*(8*f+7)*delay,twopi*(8*f+6)*delay,twopi*(8*f+5)*delay,twopi*(8*f+4)*delay,twopi*(8*f+3)*delay,twopi*(8*f+2)*delay,twopi*(8*f+1)*delay,twopi*(8*f)*delay), &sin_lut256, &cos_lut256);
        //cos_lut256=_mm256_set_ps(cos(twopi*(8*f+7)*delay),cos(twopi*(8*f+6)*delay),cos(twopi*(8*f+5)*delay),cos(twopi*(8*f+4)*delay),cos(twopi*(8*f+3)*delay),cos(twopi*(8*f+2)*delay),cos(twopi*(8*f+1)*delay),cos(twopi*(8*f)*delay));
        //sin_lut256=_mm256_set_ps(sin(twopi*(8*f+7)*delay),sin(twopi*(8*f+6)*delay),sin(twopi*(8*f+5)*delay),sin(twopi*(8*f+4)*delay),sin(twopi*(8*f+3)*delay),sin(twopi*(8*f+2)*delay),sin(twopi*(8*f+1)*delay),sin(twopi*(8*f)*delay));
        _mm256_storeu_ps(&cos_lut_f[l][8*f+(n_samples>>1)],cos_lut256);
        _mm256_storeu_ps(&sin_lut_f[l][8*f+(n_samples>>1)],sin_lut256);

    }
  }
  for (l=0; l<(int)desc->nb_taps; l++) 
  {
      cos_lut_f[l][(n_samples>>1)] = 1;
      sin_lut_f[l][(n_samples>>1)] = 0;
      //printf("f %d,l %d (cos,sin) (%e,%e):\n",4*f,l,cos_lut_f[(n_samples>>1)][l],sin_lut_f[(n_samples>>1)][l]);
  }

  for (f=1; f<=(n_samples>>4)+1; f++) {
    //count++;
    //freq=delta_f*(double)f*1e-6;// due to the fact that delays is in mus
    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
        sincos256_ps(_mm256_set_ps(twopi*(8*f)*delay,twopi*(8*f-1)*delay,twopi*(8*f-2)*delay,twopi*(8*f-3)*delay,twopi*(8*f-4)*delay,twopi*(8*f-5)*delay,twopi*(8*f-6)*delay,twopi*(8*f-7)*delay), &sin_lut256, &cos_lut256);
      //cos_lut256=_mm256_set_ps(cos(twopi*(4*f)*delay),cos(twopi*(4*f-1)*delay),cos(twopi*(4*f-2)*delay),cos(twopi*(4*f-3)*delay),cos(twopi*(4*f-4)*delay),cos(twopi*(4*f-5)*delay),cos(twopi*(4*f-6)*delay),cos(twopi*(4*f-7)*delay));
      //sin_lut256=_mm256_set_ps(sin(twopi*(4*f)*delay),sin(twopi*(4*f-1)*delay),sin(twopi*(4*f-2)*delay),sin(twopi*(4*f-3)*delay),sin(twopi*(4*f-4)*delay),sin(twopi*(4*f-5)*delay),sin(twopi*(4*f-6)*delay),sin(twopi*(4*f-7)*delay));
      _mm256_storeu_ps(&cos_lut_f[l][8*f-7+(n_samples>>1)],cos_lut256);
      _mm256_storeu_ps(&sin_lut_f[l][8*f-7+(n_samples>>1)],sin_lut256);
    }
  }
  /*for (f=-(n_samples>>1); f<=(n_samples>>1); f++) {
    for (l=0; l<(int)desc->nb_taps; l++) {  
      printf("f %d, l %d (cos,sin) (%e,%e):\n",f,l,cos_lut_f[l][f+(n_samples>>1)],sin_lut_f[l][f+(n_samples>>1)]);
    }
  }*/
  return(0);
}
int freq_channel_SSE_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{
  int16_t f,f2,d;
  uint8_t aarx,aatx,l;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m128 chFx_128,chFy_128;

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
    if (init_freq_channel_SSE_float(desc,nb_rb,n_samples_max)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  d=(n_samples_max-1)/(n_samples-1);

  //printf("no_samples=%d, n_samples_max=%d, d=%d\n",n_samples,n_samples_max,d);

  start_meas(&desc->interp_freq);

  for (f=-(n_samples_max>>3),f2=-(n_samples>>3); f<(n_samples_max>>3); f+=d,f2++) {
    //clut = cos_lut[(n_samples_max>>1)+f];
    //slut = sin_lut[(n_samples_max>>1)+f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
	chFx_128=_mm_setzero_ps();
	chFy_128=_mm_setzero_ps();
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x=0.0;
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y=0.0;
        for (l=0; l<(int)desc->nb_taps; l++) {
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
	  chFx_128=_mm_add_ps(chFx_128,_mm_add_ps(_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_ps(&cos_lut_f[l][(n_samples_max>>1)+4*f])),_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_ps(&sin_lut_f[l][(n_samples_max>>1)+4*f]))));  
	  chFy_128=_mm_add_ps(chFy_128,_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_ps(&cos_lut_f[l][(n_samples_max>>1)+4*f])),_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_ps(&sin_lut_f[l][(n_samples_max>>1)+4*f]))));  
        }
	_mm_storeu_ps(&desc->chFf[aarx+(aatx*desc->nb_rx)].x[(n_samples>>1)+4*f],chFx_128);
	_mm_storeu_ps(&desc->chFf[aarx+(aatx*desc->nb_rx)].y[(n_samples>>1)+4*f],chFy_128);
	//printf("chFx %e,%e,%e,%e\n",chFx_128[0],chFx_128[1],chFx_128[2],chFx_128[3]);
	//printf("chFy %e,%e,%e,%e\n",chFy_128[0],chFy_128[1],chFy_128[2],chFy_128[3]);
      }
    }
  }
  stop_meas(&desc->interp_freq);

  /*for (f=-(n_samples>>1); f<(n_samples>>1); f++) { 
      printf("f %d, (chF.x,chF.y) (%e,%e):\n",f,desc->chFf[0].x[(n_samples>>1)+f],desc->chFf[0].y[(n_samples>>1)+f]);
  }*/
  return(0);
}
int freq_channel_AVX_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples)
{
  int16_t f,f2,d;
  uint8_t aarx,aatx,l;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m256 chFx_256,chFy_256;

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
    if (init_freq_channel_AVX_float(desc,nb_rb,n_samples_max)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  d=(n_samples_max-1)/(n_samples-1);

  //printf("no_samples=%d, n_samples_max=%d, d=%d\n",n_samples,n_samples_max,d);

  start_meas(&desc->interp_freq);

  for (f=-(n_samples_max>>4),f2=-(n_samples>>4); f<(n_samples_max>>4); f+=d,f2++) {
    //clut = cos_lut[(n_samples_max>>1)+f];
    //slut = sin_lut[(n_samples_max>>1)+f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
	chFx_256=_mm256_setzero_ps();
	chFy_256=_mm256_setzero_ps();
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x=0.0;
        //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y=0.0;
        for (l=0; l<(int)desc->nb_taps; l++) {
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          //desc->chF[aarx+(aatx*desc->nb_rx)][(n_samples>>1)+f2].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
	  chFx_256=_mm256_add_ps(chFx_256,_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm256_loadu_ps(&cos_lut_f[l][(n_samples_max>>1)+8*f])),_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm256_loadu_ps(&sin_lut_f[l][(n_samples_max>>1)+8*f]))));  
	  chFy_256=_mm256_add_ps(chFy_256,_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm256_loadu_ps(&cos_lut_f[l][(n_samples_max>>1)+8*f])),_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm256_loadu_ps(&sin_lut_f[l][(n_samples_max>>1)+8*f]))));  
        }
	_mm256_storeu_ps(&desc->chFf[aarx+(aatx*desc->nb_rx)].x[(n_samples>>1)+8*f],chFx_256);
	_mm256_storeu_ps(&desc->chFf[aarx+(aatx*desc->nb_rx)].y[(n_samples>>1)+8*f],chFy_256);
	//printf("chFx %e,%e,%e,%e\n",chFx_128[0],chFx_128[1],chFx_128[2],chFx_128[3]);
	//printf("chFy %e,%e,%e,%e\n",chFy_128[0],chFy_128[1],chFy_128[2],chFy_128[3]);
      }
    }
  }
  stop_meas(&desc->interp_freq);

  /*for (f=-(n_samples>>1); f<(n_samples>>1); f++) { 
      printf("f %d, (chF.x,chF.y) (%e,%e):\n",f,desc->chFf[0].x[(n_samples>>1)+f],desc->chFf[0].y[(n_samples>>1)+f]);
  }*/
  return(0);
}
int init_freq_channel_prach(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  static int first_run=1;
  float delta_f,freq;  // 90 kHz spacing
  float delay;
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
  if (first_run)
  {
	cos_lut_f_prach = (float **)malloc16((int)desc->nb_taps*sizeof(float*));
	sin_lut_f_prach = (float **)malloc16((int)desc->nb_taps*sizeof(float*));
        for (f=0; f<prach_samples; f++) {
	    cos_lut_f_prach[f] = (float *)malloc16_clear(prach_samples*sizeof(float));
	    sin_lut_f_prach[f] = (float *)malloc16_clear(prach_samples*sizeof(float));
	}
	first_run=0;
  }

  //cos_lut = (double **)malloc(prach_samples*sizeof(double*));
  //sin_lut = (double **)malloc(prach_samples*sizeof(double*));

  delta_f = (prach_fmt<4)?nb_rb*180000/((n_samples-1)*12):nb_rb*180000/((n_samples-1)*2);//1.25 khz for preamble format 1,2,3. 7.5 khz for preample format 4
  max_nb_rb_samples = nb_rb*180000/delta_f;//7200 if prach_fmt<4
  prach_pbr_offset_samples = (n_ra_prb+6)*180000/delta_f;//864 if n_ra_prb=0,7200 if n_ra_prb=44=50-6
  //printf("prach_samples = %d, delta_f = %e, max_nb_rb_samples= %d, prach_pbr_offset_samples = %d, nb_taps = %d\n",prach_samples,delta_f,max_nb_rb_samples,prach_pbr_offset_samples,desc->nb_taps);
  for (f=max_nb_rb_samples/2-prach_pbr_offset_samples,f1=0; f<max_nb_rb_samples/2-prach_pbr_offset_samples+prach_samples; f++,f1++) {//3600-864,3600-864+864|3600-7200,3600-7200+839
    freq=delta_f*(float)f*1e-6;// due to the fact that delays is in mus
    //printf("[init_freq_channel_prach] freq %e\n",freq);
    //cos_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));
    //sin_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));


    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;

      cos_lut_f_prach[l][f1] = cos(2*M_PI*freq*delay);
      sin_lut_f_prach[l][f1] = sin(2*M_PI*freq*delay);
      //if (f<max_nb_rb_samples/2-prach_pbr_offset_samples+10)
      	//printf("freq: %e, f1: %d, f: %d, arg_sin_cos = %e,  cos () = %e, sin () =n %e)\n",freq, f1,f, 2*M_PI*freq*delay, cos_lut[f1][l], sin_lut[f1][l]);
    }
  }

  return(0);
}
int init_freq_channel_prach_SSE_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  static int first_run=1;
  float delta_f,twopi;  // 90 kHz spacing
  float delay;
  int16_t f,f1;
  uint8_t l;
  int prach_samples, prach_pbr_offset_samples, max_nb_rb_samples;
  __m128 cos_lut128,sin_lut128;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  if (nb_rb-n_ra_prb<6) {
    fprintf(stderr, "freq_channel_init: Impossible to allocate PRACH, check n_ra_prb value (r_ra_prb=%d)\n",n_ra_prb);
    return(-1); 
  }
  prach_samples = (prach_fmt<4)?13+839+12:3+139+2;
  if (first_run)
  {
	cos_lut_f_prach = (float **)malloc16(prach_samples*sizeof(float*));
	sin_lut_f_prach = (float **)malloc16(prach_samples*sizeof(float*));
        for (f=max_nb_rb_samples/2-prach_pbr_offset_samples,f1=0; f<max_nb_rb_samples/2-prach_pbr_offset_samples+prach_samples; f++,f1++) {
	    cos_lut_f_prach[f1] = (float *)malloc16_clear((int)desc->nb_taps*sizeof(float));
	    sin_lut_f_prach[f1] = (float *)malloc16_clear((int)desc->nb_taps*sizeof(float));
	}
	first_run=0;
  }

  //cos_lut = (double **)malloc(prach_samples*sizeof(double*));
  //sin_lut = (double **)malloc(prach_samples*sizeof(double*));

  delta_f = (prach_fmt<4)?nb_rb*180000/((n_samples-1)*12):nb_rb*180000/((n_samples-1)*2);//1.25 khz for preamble format 1,2,3. 7.5 khz for preample format 4
  max_nb_rb_samples = nb_rb*180000/delta_f;//7200 if prach_fmt<4
  prach_pbr_offset_samples = (n_ra_prb+6)*180000/delta_f;//864 if n_ra_prb=0,7200 if n_ra_prb=44=50-6
  twopi=2*M_PI*1e-6*delta_f;
  //printf("prach_samples = %d, delta_f = %e, max_nb_rb_samples= %d, prach_pbr_offset_samples = %d, nb_taps = %d\n",prach_samples,delta_f,max_nb_rb_samples,prach_pbr_offset_samples,desc->nb_taps);
  for (f=((max_nb_rb_samples/2-prach_pbr_offset_samples)>>2),f1=0; f<((max_nb_rb_samples/2-prach_pbr_offset_samples+prach_samples)>>2); f++,f1++) {//3600-864,3600-864+864|3600-7200,3600-7200+839
    //freq=delta_f*(float)f*1e-6;// due to the fact that delays is in mus
    //printf("[init_freq_channel_prach] freq %e\n",freq);
    //cos_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));
    //sin_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));


    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
      cos_lut128=_mm_set_ps(cos(twopi*(4*f+3)*delay),cos(twopi*(4*f+2)*delay),cos(twopi*(4*f+1)*delay),cos(twopi*(4*f)*delay));
      sin_lut128=_mm_set_ps(sin(twopi*(4*f+3)*delay),sin(twopi*(4*f+2)*delay),sin(twopi*(4*f+1)*delay),sin(twopi*(4*f)*delay));
      _mm_storeu_ps(&cos_lut_f_prach[l][4*f1],cos_lut128);
      _mm_storeu_ps(&sin_lut_f_prach[l][4*f1],sin_lut128);

      //cos_lut[f1][l] = cos(2*M_PI*freq*delay);
      //sin_lut[f1][l] = sin(2*M_PI*freq*delay);
      //if (f<max_nb_rb_samples/2-prach_pbr_offset_samples+10)
      	//printf("freq: %e, f1: %d, f: %d, arg_sin_cos = %e,  cos () = %e, sin () =n %e)\n",freq, f1,f, 2*M_PI*freq*delay, cos_lut[f1][l], sin_lut[f1][l]);
    }
  }

  return(0);
}
static int first_run=1;
int init_freq_channel_prach_AVX_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  float delta_f,twopi;  // 90 kHz spacing
  float delay;
  int16_t f,f1;
  uint8_t l;
  int prach_samples, prach_pbr_offset_samples, max_nb_rb_samples;
  __m256 cos_lut256,sin_lut256, cos_256, sin_256;

  if ((n_samples&1)==0) {
    fprintf(stderr, "freq_channel_init: n_samples has to be odd\n");
    return(-1); 
  }
  if (nb_rb-n_ra_prb<6) {
    fprintf(stderr, "freq_channel_init: Impossible to allocate PRACH, check n_ra_prb value (r_ra_prb=%d)\n",n_ra_prb);
    return(-1); 
  }
  prach_samples = (prach_fmt<4)?13+839+12:3+139+2;
  if (first_run)
  {
	cos_lut_f_prach = (float **)malloc16((int)desc->nb_taps*sizeof(float*));
	sin_lut_f_prach = (float **)malloc16((int)desc->nb_taps*sizeof(float*));
        for (f=0; f<prach_samples; f++) {
	    cos_lut_f_prach[f] = (float *)malloc16_clear((int)prach_samples*sizeof(float));
	    sin_lut_f_prach[f] = (float *)malloc16_clear((int)prach_samples*sizeof(float));
	}
	first_run=0;
  }
  //cos_lut = (double **)malloc(prach_samples*sizeof(double*));
  //sin_lut = (double **)malloc(prach_samples*sizeof(double*));

  delta_f = (prach_fmt<4)?nb_rb*180000/((n_samples-1)*12):nb_rb*180000/((n_samples-1)*2);//1.25 khz for preamble format 1,2,3. 7.5 khz for preample format 4
  max_nb_rb_samples = nb_rb*180000/delta_f;//7200 if prach_fmt<4
  prach_pbr_offset_samples = (n_ra_prb+6)*180000/delta_f;//864 if n_ra_prb=0,7200 if n_ra_prb=44=50-6
  twopi=2*M_PI*1e-6*delta_f;
  //printf("prach_samples = %d, delta_f = %e, max_nb_rb_samples= %d, prach_pbr_offset_samples = %d, nb_taps = %d\n",prach_samples,delta_f,max_nb_rb_samples,prach_pbr_offset_samples,desc->nb_taps);
  for (f=((max_nb_rb_samples/2-prach_pbr_offset_samples)>>3),f1=0; f<((max_nb_rb_samples/2-prach_pbr_offset_samples+prach_samples)>>3); f++,f1++) {//3600-864,3600-864+864|3600-7200,3600-7200+839
    //freq=delta_f*(float)f*1e-6;// due to the fact that delays is in mus
    //printf("[init_freq_channel_prach] freq %e\n",freq);
    //cos_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));
    //sin_lut[f1] = (double *)malloc((int)desc->nb_taps*sizeof(double));


    for (l=0; l<(int)desc->nb_taps; l++) {
      if (desc->nb_taps==1)
        delay = desc->delays[l];
      else
        delay = desc->delays[l]+NB_SAMPLES_CHANNEL_OFFSET/desc->sampling_rate;
      //cos_lut256=_mm256_set_ps(cos(twopi*(8*f+3)*delay),cos(twopi*(8*f+2)*delay),cos(twopi*(8*f+1)*delay),cos(twopi*(8*f)*delay));
      //sin_lut256=_mm256_set_ps(sin(twopi*(8*f+3)*delay),sin(twopi*(8*f+2)*delay),sin(twopi*(8*f+1)*delay),sin(twopi*(8*f)*delay));
      sincos256_ps(_mm256_set_ps(twopi*(8*f+7)*delay,twopi*(8*f+6)*delay,twopi*(8*f+5)*delay,twopi*(8*f+4)*delay,twopi*(8*f+3)*delay,twopi*(8*f+2)*delay,twopi*(8*f+1)*delay,twopi*(8*f)*delay),&sin_256,&cos_256);
      cos_lut256=cos_256;
      sin_lut256=sin_256;
      _mm256_storeu_ps(&cos_lut_f_prach[l][8*f1],cos_lut256);
      _mm256_storeu_ps(&sin_lut_f_prach[l][8*f1],sin_lut256);

      //cos_lut[f1][l] = cos(2*M_PI*freq*delay);
      //sin_lut[f1][l] = sin(2*M_PI*freq*delay);
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
  //double *clut,*slut;
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
    if (init_freq_channel_prach_SSE_float(desc,nb_rb,n_samples_max,prach_fmt,n_ra_prb)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  start_meas(&desc->interp_freq_PRACH);
  for (f=0; f<prach_samples; f++) {
    //clut = cos_lut[f];
    //slut = sin_lut[f];1.26614
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]=0.0;
        desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]=0.0;
        for (l=0; l<(int)desc->nb_taps; l++) {

          desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*cos_lut_f_prach[l][f]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*sin_lut_f_prach[l][f]);
          desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*sin_lut_f_prach[l][f]+
              desc->a[l][aarx+(aatx*desc->nb_rx)].y*cos_lut_f_prach[l][f]);
        }
      }
    }
	//if (f<10 || (f>829&&f<839))
	//	printf("chF_prach[0][%d], (x,y) = (%e,%e)\n",f,desc->chF_prach[0][f].x,desc->chF_prach[0][f].y);
  }
  stop_meas(&desc->interp_freq_PRACH);
  return(0);
}

int freq_channel_prach_SSE_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  int16_t f;
  uint8_t aarx,aatx,l;
  int prach_samples;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m128 chFx_128,chFy_128;

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
    if (init_freq_channel_prach_SSE_float(desc,nb_rb,n_samples_max,prach_fmt,n_ra_prb)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  start_meas(&desc->interp_freq_PRACH);
  for (f=0; f<(prach_samples>>2); f++) {
    //clut = cos_lut[f];
    //slut = sin_lut[f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]=0.0;
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]=0.0;
	chFx_128=_mm_setzero_ps();
	chFy_128=_mm_setzero_ps();
        for (l=0; l<(int)desc->nb_taps; l++) {

          //desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*cos_lut_f_prach[l][f]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*sin_lut_f_prach[l][f]);
          //desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*sin_lut_f_prach[l][f]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*cos_lut_f_prach[l][f]);
	  chFx_128=_mm_add_ps(chFx_128,_mm_add_ps(_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_ps(&cos_lut_f_prach[l][4*f])),_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_ps(&sin_lut_f_prach[l][4*f]))));  
	  chFy_128=_mm_add_ps(chFy_128,_mm_sub_ps(_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_ps(&cos_lut_f_prach[l][4*f])),_mm_mul_ps(_mm_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_ps(&sin_lut_f_prach[l][4*f]))));  
        }
	_mm_storeu_ps(&desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[4*f],chFx_128);
	_mm_storeu_ps(&desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[4*f],chFy_128);
      }
    }
	//if (f<10 || (f>829&&f<839))
	//	printf("chF_prach[0][%d], (x,y) = (%e,%e)\n",f,desc->chF_prach[0][f].x,desc->chF_prach[0][f].y);
  }
  stop_meas(&desc->interp_freq_PRACH);
  return(0);
}
int freq_channel_prach_AVX_float(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  int16_t f;
  uint8_t aarx,aatx,l;
  int prach_samples;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m256 chFx_256,chFy_256;

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
    if (init_freq_channel_prach_AVX_float(desc,nb_rb,n_samples_max,prach_fmt,n_ra_prb)==0)
      freq_channel_init=1;
    else
      return(-1);
  }

  start_meas(&desc->interp_freq_PRACH);
  for (f=0; f<(prach_samples>>3); f++) {
    //clut = cos_lut[f];
    //slut = sin_lut[f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]=0.0;
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]=0.0;
	chFx_256=_mm256_setzero_ps();
	chFy_256=_mm256_setzero_ps();
        for (l=0; l<(int)desc->nb_taps; l++) {

          //desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[f]+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*cos_lut_f_prach[l][f]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*sin_lut_f_prach[l][f]);
          //desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[f]+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*sin_lut_f_prach[l][f]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*cos_lut_f_prach[l][f]);
	  chFx_256=_mm256_add_ps(chFx_256,_mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm256_loadu_ps(&cos_lut_f_prach[l][8*f])),_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm256_loadu_ps(&sin_lut_f_prach[l][8*f]))));  
	  chFy_256=_mm256_add_ps(chFy_256,_mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm256_loadu_ps(&cos_lut_f_prach[l][8*f])),_mm256_mul_ps(_mm256_set1_ps(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm256_loadu_ps(&sin_lut_f_prach[l][8*f]))));  
        }
	_mm256_storeu_ps(&desc->chF_prach[aarx+(aatx*desc->nb_rx)].x[8*f],chFx_256);
	_mm256_storeu_ps(&desc->chF_prach[aarx+(aatx*desc->nb_rx)].y[8*f],chFy_256);
      }
    }
	//if (f<10 || (f>829&&f<839))
	//	printf("chF_prach[0][%d], (x,y) = (%e,%e)\n",f,desc->chF_prach[0][f].x,desc->chF_prach[0][f].y);
  }
  stop_meas(&desc->interp_freq_PRACH);
  return(0);
}
//#endif
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


void sincos_ps(__m128 x, __m128 *s, __m128 *c) {
  __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
  __m128i emm0, emm2, emm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm_and_ps(x, _mm_castsi128_ps(_mm_set1_epi32(~0x80000000)));//_ps_inv_sign_mask
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm_and_ps(sign_bit_sin, _mm_castsi128_ps(_mm_set1_epi32(0x80000000)));//_ps_sign_mask
  
  /* scale by 4/Pi */
  y = _mm_mul_ps(x, _mm_set1_ps(1.27323954473516f));
    

  /* store the integer part of y in emm2 */
  emm2 = _mm_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  emm2 = _mm_add_epi32(emm2, _mm_set1_epi32(1));//_pi32_1
  emm2 = _mm_and_si128(emm2, _mm_set1_epi32(~1));//_pi32_inv1
  y = _mm_cvtepi32_ps(emm2);

  emm4 = emm2;

  /* get the swap sign flag for the sine */
  emm0 = _mm_and_si128(emm2, _mm_set1_epi32(4));//_pi32_4
  emm0 = _mm_slli_epi32(emm0, 29);
  __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

  /* get the polynom selection mask for the sine*/
  emm2 = _mm_and_si128(emm2, _mm_set1_epi32(2));//_pi32_2
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
  __m128 poly_mask = _mm_castsi128_ps(emm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */

  xmm1 = _mm_set1_ps(-0.78515625f);//_ps_minus_cephes_DP1
  xmm2 = _mm_set1_ps(-2.4187564849853515625e-4f);//_ps_minus_cephes_DP2
  xmm3 = _mm_set1_ps(-3.77489497744594108e-8f);//_ps_minus_cephes_DP3
  xmm1 = _mm_mul_ps(y, xmm1);
  xmm2 = _mm_mul_ps(y, xmm2);
  xmm3 = _mm_mul_ps(y, xmm3);
  x = _mm_add_ps(x, xmm1);
  x = _mm_add_ps(x, xmm2);
  x = _mm_add_ps(x, xmm3);


  emm4 = _mm_sub_epi32(emm4, _mm_set1_epi32(2));//_pi32_2
  emm4 = _mm_andnot_si128(emm4, _mm_set1_epi32(4));//_pi32_4
  emm4 = _mm_slli_epi32(emm4, 29);
  __m128 sign_bit_cos = _mm_castsi128_ps(emm4);

  sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  __m128 z = _mm_mul_ps(x,x);
  y = _mm_set1_ps( 2.443315711809948E-005f);//_ps_coscof_p0

  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, _mm_set1_ps(-1.388731625493765E-003f));//_ps_coscof_p1
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, _mm_set1_ps( 4.166664568298827E-002f));//_ps_coscof_p2
  y = _mm_mul_ps(y, z);
  y = _mm_mul_ps(y, z);
  __m128 tmp = _mm_mul_ps(z, _mm_set1_ps(0.5f));//_ps_0p5
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, _mm_set1_ps(1.f));//_ps_1
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m128 y2 = _mm_set1_ps(-1.9515295891E-4f);//*(__m128*)_ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, _mm_set1_ps( 8.3321608736E-3f));//*(__m128*)_ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, _mm_set1_ps(-1.6666654611E-1f));//_ps_sincof_p2
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_mul_ps(y2, x);
  y2 = _mm_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  __m128 ysin2 = _mm_and_ps(xmm3, y2);
  __m128 ysin1 = _mm_andnot_ps(xmm3, y);
  y2 = _mm_sub_ps(y2,ysin2);
  y = _mm_sub_ps(y, ysin1);

  xmm1 = _mm_add_ps(ysin1,ysin2);
  xmm2 = _mm_add_ps(y,y2);
 
  /* update the sign */
  *s = _mm_xor_ps(xmm1, sign_bit_sin);
  *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

void sincos256_ps(__m256 x, __m256 *s, __m256 *c) {
  __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  __m256i imm0, imm2, imm4;
  
  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000)));//_ps_inv_sign_mask
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));//_ps_sign_mask
  
  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, _mm256_set1_ps(1.27323954473516f));
    

  /* store the integer part of y in imm2 */
  imm2 = _mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = _mm256_add_epi32(imm2, _mm256_set1_epi32(1));//_pi32_1
  imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(~1));//_pi32_inv1
  y = _mm256_cvtepi32_ps(imm2);

  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = _mm256_and_si256(imm2, _mm256_set1_epi32(4));//_pi32_4
  imm0 = _mm256_slli_epi32(imm0, 29);

  /* get the polynom selection mask for the sine*/
  imm2 = _mm256_and_si256(imm2, _mm256_set1_epi32(2));//_pi32_2
  imm2 = _mm256_cmpeq_epi32(imm2, _mm256_setzero_si256());

  __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic" 
     x = ((x - y * DP1) - y * DP2) - y * DP3; */

  xmm1 = _mm256_set1_ps(-0.78515625f);//_ps_minus_cephes_DP1
  xmm2 = _mm256_set1_ps(-2.4187564849853515625e-4f);//_ps_minus_cephes_DP2
  xmm3 = _mm256_set1_ps(-3.77489497744594108e-8f);//_ps_minus_cephes_DP3
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);


  imm4 = _mm256_sub_epi32(imm4, _mm256_set1_epi32(2));//_pi32_2
  imm4 = _mm256_andnot_si256(imm4, _mm256_set1_epi32(4));//_pi32_4
  imm4 = _mm256_slli_epi32(imm4, 29);
  __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);

  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  
  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  __m256 z = _mm256_mul_ps(x,x);
  y = _mm256_set1_ps( 2.443315711809948E-005f);//_ps_coscof_p0

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, _mm256_set1_ps(-1.388731625493765E-003f));//_ps_coscof_p1
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, _mm256_set1_ps( 4.166664568298827E-002f));//_ps_coscof_p2
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  __m256 tmp = _mm256_mul_ps(z, _mm256_set1_ps(0.5f));//_ps_0p5
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.f));//_ps_1
  
  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  __m256 y2 = _mm256_set1_ps(-1.9515295891E-4f);//*(__m128*)_ps_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, _mm256_set1_ps( 8.3321608736E-3f));//*(__m128*)_ps_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, _mm256_set1_ps(-1.6666654611E-1f));//_ps_sincof_p2
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */  
  xmm3 = poly_mask;
  __m256 ysin2 = _mm256_and_ps(xmm3, y2);
  __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);

  xmm1 = _mm256_add_ps(ysin1,ysin2);
  xmm2 = _mm256_add_ps(y,y2);
 
  /* update the sign */
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

__m128 log_ps(__m128 x) {

  __m128i emm0 __attribute__((aligned(16)));
  __m128 one __attribute__((aligned(16)))=_mm_set1_ps(1.f);
  __m128 invalid_mask __attribute__((aligned(16))) = _mm_cmple_ps(x, _mm_setzero_ps());

  x = _mm_max_ps(x, _mm_castsi128_ps(_mm_set1_epi32(0x00800000)));  // cut off denormalized stuff
  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

  // keep only the fractional part 
  x = _mm_and_ps(x,_mm_castsi128_ps(_mm_set1_epi32(~0x7f800000)));
  //printf("ln inside x is %e,%e,%e,%e\n",x[0],x[1],x[2],x[3]);
  x = _mm_or_ps(x, _mm_set1_ps(0.5f));
  //printf("ln inside x is %e,%e,%e,%e\n",x[0],x[1],x[2],x[3]);


  // now e=mm0:mm1 contain the really base-2 exponent 
  emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(0x7f));
  __m128 e = _mm_cvtepi32_ps(emm0); 


  e = _mm_add_ps(e, one);


  __m128 mask = _mm_cmplt_ps(x, _mm_set1_ps(0.707106781186547524f));

  __m128 tmp = _mm_and_ps(x, mask);
  x = _mm_sub_ps(x, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));

  x = _mm_add_ps(x, tmp);


  __m128 z = _mm_mul_ps(x,x);

  __m128 y = _mm_set1_ps(7.0376836292E-2f);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(- 1.1514610310E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(1.1676998740E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(- 1.2420140846E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(1.4249322787E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(- 1.6668057665E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(2.0000714765E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(- 2.4999993993E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(3.3333331174E-1f));
  y = _mm_mul_ps(y, x);

  y = _mm_mul_ps(y, z);
  

  tmp = _mm_mul_ps(e, _mm_set1_ps(-2.12194440e-4f));
  y = _mm_add_ps(y, tmp);


  tmp = _mm_mul_ps(z, _mm_set1_ps(0.5f));
  y = _mm_sub_ps(y, tmp);

  tmp = _mm_mul_ps(e, _mm_set1_ps(0.693359375f));
  x = _mm_add_ps(x, y);
  x = _mm_add_ps(x, tmp);
  x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN

  return x;
}
__m256 log256_ps(__m256 x) {

  __m256i imm0 __attribute__((aligned(32)));
  __m256 one __attribute__((aligned(32)))=_mm256_set1_ps(1.f);
  __m256 invalid_mask __attribute__((aligned(32))) = _mm256_cmp_ps(x, _mm256_setzero_ps(),_CMP_LE_OS);

  x = _mm256_max_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x00800000)));  // cut off denormalized stuff
  imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

  // keep only the fractional part 
  x = _mm256_and_ps(x,_mm256_castsi256_ps(_mm256_set1_epi32(~0x7f800000)));
  //printf("ln inside x is %e,%e,%e,%e\n",x[0],x[1],x[2],x[3]);
  x = _mm256_or_ps(x, _mm256_set1_ps(0.5f));
  //printf("ln inside x is %e,%e,%e,%e\n",x[0],x[1],x[2],x[3]);


  // now e=mm0:mm1 contain the really base-2 exponent 
  imm0 = _mm256_sub_epi32(imm0, _mm256_set1_epi32(0x7f));
  __m256 e = _mm256_cvtepi32_ps(imm0); 


  e = _mm256_add_ps(e, one);


  __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(0.707106781186547524f),_CMP_LT_OS);

  __m256 tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));

  x = _mm256_add_ps(x, tmp);


  __m256 z = _mm256_mul_ps(x,x);

  __m256 y = _mm256_set1_ps(7.0376836292E-2f);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(- 1.1514610310E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.1676998740E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(- 1.2420140846E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.4249322787E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(- 1.6668057665E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(2.0000714765E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(- 2.4999993993E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(3.3333331174E-1f));
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);
  

  tmp = _mm256_mul_ps(e, _mm256_set1_ps(-2.12194440e-4f));
  y = _mm256_add_ps(y, tmp);


  tmp = _mm256_mul_ps(z, _mm256_set1_ps(0.5f));
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, _mm256_set1_ps(0.693359375f));
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN

  return x;
}
__m128 exp_ps(__m128 x) {
  __m128 tmp = _mm_setzero_ps(), fx;

  __m128i emm0;

  __m128 one = _mm_set1_ps(1.f);

  x = _mm_min_ps(x, _mm_set1_ps(88.3762626647949f));
  x = _mm_max_ps(x, _mm_set1_ps(-88.3762626647949f));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm_mul_ps(x, _mm_set1_ps(1.44269504088896341f));
  fx = _mm_add_ps(fx, _mm_set1_ps(0.5f));

  /* how to perform a floorf with SSE: just below */

  emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);

  /* if greater, substract 1 */
  __m128 mask = _mm_cmpgt_ps(tmp, fx);    
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = _mm_mul_ps(fx, _mm_set1_ps(0.693359375f));
  __m128 z = _mm_mul_ps(fx, _mm_set1_ps(-2.12194440e-4f));
  x = _mm_sub_ps(x, tmp);
  x = _mm_sub_ps(x, z);

  z = _mm_mul_ps(x,x);
  
  __m128 y = _mm_set1_ps(1.9875691500E-4f);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(1.3981999507E-3f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(8.3334519073E-3f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(4.1665795894E-2f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(1.6666665459E-1f));
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, _mm_set1_ps(5.0000001201E-1f));
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, x);
  y = _mm_add_ps(y, one);

  /* build 2^n */
  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, _mm_set1_epi32(0x7f));
  emm0 = _mm_slli_epi32(emm0, 23);
  __m128 pow2n = _mm_castsi128_ps(emm0);
  y = _mm_mul_ps(y, pow2n);
  return y;
}
__m256 exp256_ps(__m256 x) {
  __m256 tmp = _mm256_setzero_ps(), fx;

  __m256i imm0;

  __m256 one = _mm256_set1_ps(1.f);

  x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
  x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
  fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));

  /* how to perform a floorf with SSE: just below */

  /*emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);*/
  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f));
  __m256 z = _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f));
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);
  
  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
/*#ifdef abstraction_SSE
int freq_channel_prach(channel_desc_t *desc,uint16_t nb_rb,int16_t n_samples,int16_t prach_fmt,int16_t n_ra_prb)
{

  int16_t f;
  uint8_t aarx,aatx,l;
  double *clut,*slut;
  int prach_samples;
  static int freq_channel_init=0;
  static int n_samples_max=0;
  __m128d clut128,slut128,chFx_128,chFy_128;

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

  start_meas(&desc->interp_freq_PRACH);
  for (f=0; f<(prach_samples>>1); f++) {
    //clut = cos_lut[f];
    //slut = sin_lut[f];
    for (aarx=0; aarx<desc->nb_rx; aarx++) {
      for (aatx=0; aatx<desc->nb_tx; aatx++) {
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].x=0.0;
        //desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].y=0.0;
	chFx_128=_mm_setzero_pd();
	chFy_128=_mm_setzero_pd();
        for (l=0; l<(int)desc->nb_taps; l++) {

          //desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].x+=(desc->a[l][aarx+(aatx*desc->nb_rx)].x*clut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*slut[l]);
          //desc->chF_prach[aarx+(aatx*desc->nb_rx)][f].y+=(-desc->a[l][aarx+(aatx*desc->nb_rx)].x*slut[l]+
          //    desc->a[l][aarx+(aatx*desc->nb_rx)].y*clut[l]);
	  chFx_128=_mm_add_pd(chFx_128,_mm_add_pd(_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_pd(&cos_lut[2*f][l])),_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_pd(&sin_lut[2*f][l]))));  
	  chFy_128=_mm_add_pd(chFy_128,_mm_sub_pd(_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].y),_mm_loadu_pd(&cos_lut[2*f][l])),_mm_mul_pd(_mm_set1_pd(desc->a[l][aarx+(aatx*desc->nb_rx)].x),_mm_loadu_pd(&sin_lut[2*f][l]))));  
        }
	_mm_storeu_pd(&desc->chF_prach[aarx+(aatx*desc->nb_rx)][2*f].x,chFx_128);
	_mm_storeu_pd(&desc->chF_prach[aarx+(aatx*desc->nb_rx)][2*f].y,chFy_128);
      }
    }
	//if (f<10 || (f>829&&f<839))
	//	printf("chF_prach[0][%d], (x,y) = (%e,%e)\n",f,desc->chF_prach[0][f].x,desc->chF_prach[0][f].y);
  }
  stop_meas(&desc->interp_freq_PRACH);
  return(0);
}
#else*/

      //cos_lut[f+(n_samples>>1)][l] = cos(2*M_PI*freq*delay);
      //sin_lut[f+(n_samples>>1)][l] = sin(2*M_PI*freq*delay);
      //printf("cos %e,%e,%e,%e\n",cos_lut128[0],cos_lut128[1],cos_lut128[2],cos_lut128[3]);
      //printf("sin %e,%e,%e,%e\n",sin_lut128[0],sin_lut128[1],sin_lut128[2],sin_lut128[3]);
      //printf("arg %e, f %d, values cos:%e, sin:%e, cos# %e, sin# %e\n",twopi*(4*f)*delay,4*f+(n_samples>>1), cos_lut_f[l][4*f+(n_samples>>1)], sin_lut_f[l][4*f+(n_samples>>1)],cos(twopi*(4*f)*delay),sin(twopi*(4*f)*delay));
      //printf("arg %e, f %d, values cos:%e, sin:%e, cos# %e, sin# %e\n",twopi*(4*f+1)*delay,4*f+1+(n_samples>>1), cos_lut_f[l][4*f+1+(n_samples>>1)], sin_lut_f[l][4*f+1+(n_samples>>1)],cos(twopi*(4*f+1)*delay),sin(twopi*(4*f+1)*delay));
      //printf("arg %e, f %d, values cos:%e, sin:%e, cos# %e, sin# %e\n",twopi*(4*f+2)*delay,4*f+2+(n_samples>>1), cos_lut_f[l][4*f+2+(n_samples>>1)], sin_lut_f[l][4*f+2+(n_samples>>1)],cos(twopi*(4*f+2)*delay),sin(twopi*(4*f+2)*delay));
      //printf("arg %e, f %d, values cos:%e, sin:%e, cos# %e, sin# %e\n",twopi*(4*f+3)*delay,4*f+3+(n_samples>>1), cos_lut_f[l][4*f+3+(n_samples>>1)], sin_lut_f[l][4*f+3+(n_samples>>1)],cos(twopi*(4*f+3)*delay),sin(twopi*(4*f+3)*delay));
      //printf("f %d, cos0 %e, cos1 %e\n",2*f,(double) &cos_lut128[0],(double) &cos_lut128[1]);
      //printf("f %d, sin0 %e, sin1 %e\n",2*f+1,(double) &sin_lut128[0],(double) &sin_lut128[1]);


