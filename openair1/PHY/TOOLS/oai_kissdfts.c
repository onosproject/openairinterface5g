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

int dfts_autoinit(void)
{
  for (int i=0; i<DFT_SIZE_IDXTABLESIZE ; i++) {
  	  fftcfg[i]=kiss_fft_alloc(fftsizes[i],0,NULL,NULL);
  }
  for (int i=0; i<IDFT_SIZE_IDXTABLESIZE ; i++) {
  	  ifftcfg[i]=kiss_fft_alloc(ifftsizes[i],1,NULL,NULL);
  }  
  return 0;
}

#ifndef FIXED_POINT
static float input_float[98304*2];
static float output_float[98304*2];
static float input_float2[98304*2];
static float output_float2[98304*2];
#endif

void convert_shorttofloat(int sizeidx,short *input,float *output){
	for (int i=0;i<sizeidx;i=i+2){
		output[2*i]=(float)input[2*i];
		output[(2*i)+1]=(float)input[(2*i)+1];
	}
}

void convert_floattoshort(int sizeidx,float *input,short *output){
	for (int i=0;i<sizeidx;i=i+2){
		output[2*i]=(short)input[2*i];
		output[(2*i)+1]=(short)input[(2*i)+1];
	}
}
void idft_fixedpoint(uint8_t sizeidx, int16_t *input,int16_t *output,unsigned char scale_flag){
  kiss_fft(ifftcfg[sizeidx],(kiss_fft_cpx *)input,(kiss_fft_cpx *)output);
};

void dft(uint8_t sizeidx,int16_t *input,int16_t *output,unsigned char scale_flag){
#ifndef FIXED_POINT
  convert_shorttofloat(sizeidx,input,input_float);
  kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input_float,(kiss_fft_cpx *)output_float);
  convert_floattoshort(sizeidx,output_float,output);
#else
  kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input,(kiss_fft_cpx *)output);
#endif
};

void idft(uint8_t sizeidx, int16_t *input,int16_t *output,unsigned char scale_flag){
#ifndef FIXED_POINT
  convert_shorttofloat(sizeidx,input,input_float2);
  kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input_float2,(kiss_fft_cpx *)output_float2);
  convert_floattoshort(sizeidx,output_float2,output);	
#else	
  kiss_fft(fftcfg[sizeidx],(kiss_fft_cpx *)input,(kiss_fft_cpx *)output);
#endif
};