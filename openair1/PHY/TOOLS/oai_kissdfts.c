/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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
/*! \file openair1/PHY/TPOOLS/oai_kissdfts.c
 * \brief: interface to kissfft, used to build libdfts_kiss.so
 *         alternative to oai implementation of dft and idft
 * \author Francois TABURET
 * \date 2020
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <execinfo.h>
#include <limits.h>
#define OAIDFTS_LIB
#include "PHY/defs_common.h"
#include "PHY/impl_defs_top.h"
#include "time_meas.h"
#include "LOG/log.h"


#include "kiss_fft130/kiss_fft.h"

static kiss_fft_cfg fftcfg[DFT_SIZE_IDXTABLESIZE];
static kiss_fft_cfg ifftcfg[IDFT_SIZE_IDXTABLESIZE];
static int fftsizes[] = DFT_SIZES;
static int ifftsizes[] = IDFT_SIZES;
/*----------------------------------------------------------------*/
/* dft library entry points:                                      */
static   int16_t *tmpbuff;
int dfts_autoinit(void)
{
  for (int i=0; i<DFT_SIZE_IDXTABLESIZE ; i++) {
  	  fftcfg[i]=kiss_fft_alloc(fftsizes[i],0,NULL,NULL);
  }
  for (int i=0; i<IDFT_SIZE_IDXTABLESIZE ; i++) {
  	  ifftcfg[i]=kiss_fft_alloc(ifftsizes[i],1,NULL,NULL);
  }
#ifdef FIXED_POINT
  tmpbuff=malloc(98304*2*sizeof(int16_t));
  olddfts_autoinit();
#endif
  return 0;
}






void convert_shorttofloat(int size,short *input,float *output,int factor){
	for (int i=0;i<(size-2);i++){
		output[2*i]=(float)(input[2*i]*factor);
		output[(2*i)+1]=(float)((input[(2*i)+1])*factor);
	}
}

void convert_floattoshort(int size,float *input,short *output,int factor){
	for (int i=0;i<(size-2);i++){
		output[2*i]=(int16_t)(((int)(roundf(input[2*i])))/factor);
		output[(2*i)+1]=(int16_t)(((int)(roundf(input[(2*i)+1])))/factor);
	}
}


void rescale_up_int16buff(int size,int16_t *input, int factor){
	for (int i=0;i<(size*2);i=i+1){
		input[i]=(input[i]*factor);
	}
}

void rescale_up_newint16buff(int size,int16_t *input, int16_t *output,int factor){
	for (int i=0;i<(size*2);i=i+1){
		output[i]=(input[i]*factor);
	}
}

void rescale_down_int16buff(int size,int16_t *input, int factor){
	for (int i=0;i<(size*2);i=i+1){
		input[i]=(input[i]/factor);
	}
}

void rescale_dft_int16buff(int size,int16_t *input, int factor){
	for (int i=0;i<(size*2);i=i+1){
		int32_t tmpi=input[i]*factor;
		input[i]=tmpi/4096;
	}
}


void print_minmax(int size,int16_t *buf,int scale_flag) {
  	int16_t vmin=0, vmax=0;
	for (int i=0;i<(size*2);i=i+1){
		if (buf[i]>vmax) vmax=buf[i];
		if (buf[i]<vmin) vmin=buf[i];
    }
    if (scale_flag == 0 || (vmax - vmin)>10)
      printf("%i: %i - %i\n",scale_flag,vmin,vmax);
}

void dft(uint8_t sizeidx,int16_t *input,int16_t *output,unsigned char scale_flag){
#ifndef FIXED_POINT
  float input_float[98304*2];
  float output_float[98304*2];
   switch(sizeidx) {
  	 case DFT_128:
  	 case DFT_256:
   	 case DFT_512:
 	 case DFT_1024:
 	 case DFT_1536:
  	 case DFT_2048:
	 case DFT_3072:
 	 case DFT_4096:
     case DFT_6144:
	 case DFT_8192:
	 case DFT_9216:
	 case DFT_12288:
	 case DFT_18432:
	 case DFT_24576:
	 case DFT_36864:
	 case DFT_49152:
	 case DFT_73728:
	 case DFT_98304: 
     convert_shorttofloat(fftsizes[sizeidx],input,input_float,8192);
     kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input_float,(kiss_fft_cpx *)output_float);
     if (scale_flag)  
       convert_floattoshort(fftsizes[sizeidx],output_float,output,8192);
     else
       convert_floattoshort(fftsizes[sizeidx],output_float,output,98304);
     break;
    default:
      olddft(sizeidx,input,output,scale_flag);
    break;
  }  
#else
  switch(sizeidx) {
  	 case DFT_128:
  	 case DFT_256:
   	 case DFT_512:
 	 case DFT_1024:
 	 case DFT_1536:
  	 case DFT_2048:
	 case DFT_3072:
 	 case DFT_4096:
     case DFT_6144:
	 case DFT_8192:
	 case DFT_9216:
	 case DFT_12288:
	 case DFT_18432:
	 case DFT_24576:
	 case DFT_36864:
	 case DFT_49152:
	 case DFT_73728:
	 case DFT_98304:
     if (scale_flag) 	 	 
       rescale_up_int16buff(fftsizes[sizeidx],input,16);
 
//     
       kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input,(kiss_fft_cpx *)output);
//      if (scale_flag) 
//         rescale_down_int16buff(fftsizes[sizeidx],output,16);
    break;
    default:
      olddft(sizeidx,input,output,scale_flag);
    break;
  }
//    print_minmax(fftsizes[sizeidx],output,scale_flag);
#endif
};

void idft(uint8_t sizeidx, int16_t *input,int16_t *output,unsigned char scale_flag){
#ifndef FIXED_POINT
  float input_float2[98304*2];
  float output_float2[98304*2];
  convert_shorttofloat(ifftsizes[sizeidx],input,input_float2, 4096);
  kiss_fft(ifftcfg[sizeidx],(kiss_fft_cpx *)input_float2,(kiss_fft_cpx *)output_float2);
  convert_floattoshort(ifftsizes[sizeidx],output_float2,output,98304);	
#else
  if (scale_flag)
    rescale_up_int16buff(ifftsizes[sizeidx],input,16);
  kiss_fft(ifftcfg[sizeidx],(kiss_fft_cpx *)input,(kiss_fft_cpx *)output);
//  oldidft(sizeidx,input,output,scale_flag);
//  print_minmax(ifftsizes[sizeidx],output,scale_flag);
#endif
};