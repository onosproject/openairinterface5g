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

/*! \file PHY/LTE_TRANSPORT/ulsch_demodulation.c
* \brief Top-level routines for demodulating the PUSCH physical channel from 36.211 V8.6 2009-03
* \authors V. Savaux, M. Kanj
* \date 2018
* \version 0.1
* \company b<>com
* \email: vincent.savaux@b-com.com , matthieu.kanj@b-com.com
* \note
* \warning
*/

#include "PHY/defs_NB_IoT.h"
#include "PHY/extern_NB_IoT.h"
#include "defs_NB_IoT.h"
#include "extern_NB_IoT.h"
//#include "PHY/CODING/lte_interleaver2.h"
#include "PHY/CODING/extern.h"
//#define DEBUG_ULSCH
//#include "PHY/sse_intrin.h"
#include "PHY/LTE_ESTIMATION/defs_NB_IoT.h"

#include "openair1/SCHED/defs_NB_IoT.h"
//#include "openair1/PHY/LTE_TRANSPORT/sc_rotation_NB_IoT.h"

#include "T.h"

//extern char* namepointer_chMag ;
//eren
//extern int **ulchmag_eren;
//eren

static short jitter[8]  __attribute__ ((aligned(16))) = {1,0,0,1,0,1,1,0};
static short jitterc[8] __attribute__ ((aligned(16))) = {0,1,1,0,1,0,0,1};

#ifndef OFDMA_ULSCH
void lte_idft_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,uint32_t *z, uint16_t Msc_PUSCH)
{
#if defined(__x86_64__) || defined(__i386__)
  __m128i idft_in128[3][1200],idft_out128[3][1200];
  __m128i norm128;
#elif defined(__arm__)
  int16x8_t idft_in128[3][1200],idft_out128[3][1200];
  int16x8_t norm128;
#endif
  int16_t *idft_in0=(int16_t*)idft_in128[0],*idft_out0=(int16_t*)idft_out128[0];
  int16_t *idft_in1=(int16_t*)idft_in128[1],*idft_out1=(int16_t*)idft_out128[1];
  int16_t *idft_in2=(int16_t*)idft_in128[2],*idft_out2=(int16_t*)idft_out128[2];

  uint32_t *z0,*z1,*z2,*z3,*z4,*z5,*z6,*z7,*z8,*z9,*z10=NULL,*z11=NULL;
  int i,ip;

  //  printf("Doing lte_idft for Msc_PUSCH %d\n",Msc_PUSCH);

   // Normal prefix
    z0 = z;
    z1 = z0+(frame_parms->N_RB_DL*12);
    z2 = z1+(frame_parms->N_RB_DL*12);
    //pilot
    z3 = z2+(2*frame_parms->N_RB_DL*12);
    z4 = z3+(frame_parms->N_RB_DL*12);
    z5 = z4+(frame_parms->N_RB_DL*12);

    z6 = z5+(frame_parms->N_RB_DL*12);
    z7 = z6+(frame_parms->N_RB_DL*12);
    z8 = z7+(frame_parms->N_RB_DL*12);
    //pilot
    z9 = z8+(2*frame_parms->N_RB_DL*12);
    z10 = z9+(frame_parms->N_RB_DL*12);
    // srs
    z11 = z10+(frame_parms->N_RB_DL*12);
  
  // conjugate input
  for (i=0; i<(Msc_PUSCH>>2); i++) {
#if defined(__x86_64__)||defined(__i386__)
    *&(((__m128i*)z0)[i])=_mm_sign_epi16(*&(((__m128i*)z0)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z1)[i])=_mm_sign_epi16(*&(((__m128i*)z1)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z2)[i])=_mm_sign_epi16(*&(((__m128i*)z2)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z3)[i])=_mm_sign_epi16(*&(((__m128i*)z3)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z4)[i])=_mm_sign_epi16(*&(((__m128i*)z4)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z5)[i])=_mm_sign_epi16(*&(((__m128i*)z5)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z6)[i])=_mm_sign_epi16(*&(((__m128i*)z6)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z7)[i])=_mm_sign_epi16(*&(((__m128i*)z7)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z8)[i])=_mm_sign_epi16(*&(((__m128i*)z8)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z9)[i])=_mm_sign_epi16(*&(((__m128i*)z9)[i]),*(__m128i*)&conjugate2[0]);

    // if (frame_parms->Ncp==NORMAL_NB_IoT) {
      *&(((__m128i*)z10)[i])=_mm_sign_epi16(*&(((__m128i*)z10)[i]),*(__m128i*)&conjugate2[0]);
      *&(((__m128i*)z11)[i])=_mm_sign_epi16(*&(((__m128i*)z11)[i]),*(__m128i*)&conjugate2[0]);
    // }
#elif defined(__arm__)
    *&(((int16x8_t*)z0)[i])=vmulq_s16(*&(((int16x8_t*)z0)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z1)[i])=vmulq_s16(*&(((int16x8_t*)z1)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z2)[i])=vmulq_s16(*&(((int16x8_t*)z2)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z3)[i])=vmulq_s16(*&(((int16x8_t*)z3)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z4)[i])=vmulq_s16(*&(((int16x8_t*)z4)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z5)[i])=vmulq_s16(*&(((int16x8_t*)z5)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z6)[i])=vmulq_s16(*&(((int16x8_t*)z6)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z7)[i])=vmulq_s16(*&(((int16x8_t*)z7)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z8)[i])=vmulq_s16(*&(((int16x8_t*)z8)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z9)[i])=vmulq_s16(*&(((int16x8_t*)z9)[i]),*(int16x8_t*)&conjugate2[0]);


    // if (frame_parms->Ncp==NORMAL_NB_IoT) {
      *&(((int16x8_t*)z10)[i])=vmulq_s16(*&(((int16x8_t*)z10)[i]),*(int16x8_t*)&conjugate2[0]);
      *&(((int16x8_t*)z11)[i])=vmulq_s16(*&(((int16x8_t*)z11)[i]),*(int16x8_t*)&conjugate2[0]);
    // }

#endif
  }

  for (i=0,ip=0; i<Msc_PUSCH; i++,ip+=4) {
    ((uint32_t*)idft_in0)[ip+0] =  z0[i];
    ((uint32_t*)idft_in0)[ip+1] =  z1[i];
    ((uint32_t*)idft_in0)[ip+2] =  z2[i];
    ((uint32_t*)idft_in0)[ip+3] =  z3[i];
    ((uint32_t*)idft_in1)[ip+0] =  z4[i];
    ((uint32_t*)idft_in1)[ip+1] =  z5[i];
    ((uint32_t*)idft_in1)[ip+2] =  z6[i];
    ((uint32_t*)idft_in1)[ip+3] =  z7[i];
    ((uint32_t*)idft_in2)[ip+0] =  z8[i];
    ((uint32_t*)idft_in2)[ip+1] =  z9[i];

    // if (frame_parms->Ncp==0) {
      ((uint32_t*)idft_in2)[ip+2] =  z10[i];
      ((uint32_t*)idft_in2)[ip+3] =  z11[i];
    // }
  }


  switch (Msc_PUSCH) {
  case 12:
    dft12((int16_t *)idft_in0,(int16_t *)idft_out0);
    dft12((int16_t *)idft_in1,(int16_t *)idft_out1);
    dft12((int16_t *)idft_in2,(int16_t *)idft_out2);

#if defined(__x86_64__)||defined(__i386__)
    norm128 = _mm_set1_epi16(9459);
#elif defined(__arm__)
    norm128 = vdupq_n_s16(9459);
#endif
    for (i=0; i<12; i++) {
#if defined(__x86_64__)||defined(__i386__)
      ((__m128i*)idft_out0)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out0)[i],norm128),1);
      ((__m128i*)idft_out1)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out1)[i],norm128),1);
      ((__m128i*)idft_out2)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out2)[i],norm128),1);
#elif defined(__arm__)
      ((int16x8_t*)idft_out0)[i] = vqdmulhq_s16(((int16x8_t*)idft_out0)[i],norm128);
      ((int16x8_t*)idft_out1)[i] = vqdmulhq_s16(((int16x8_t*)idft_out1)[i],norm128);
      ((int16x8_t*)idft_out2)[i] = vqdmulhq_s16(((int16x8_t*)idft_out2)[i],norm128);
#endif
    }

    break;

  // case 24:
  //   dft24(idft_in0,idft_out0,1);
  //   dft24(idft_in1,idft_out1,1);
  //   dft24(idft_in2,idft_out2,1);
  //   break;

  // case 36:
  //   dft36(idft_in0,idft_out0,1);
  //   dft36(idft_in1,idft_out1,1);
  //   dft36(idft_in2,idft_out2,1);
  //   break;

  // case 48:
  //   dft48(idft_in0,idft_out0,1);
  //   dft48(idft_in1,idft_out1,1);
  //   dft48(idft_in2,idft_out2,1);
  //   break;

  // case 60:
  //   dft60(idft_in0,idft_out0,1);
  //   dft60(idft_in1,idft_out1,1);
  //   dft60(idft_in2,idft_out2,1);
  //   break;

  // case 72:
  //   dft72(idft_in0,idft_out0,1);
  //   dft72(idft_in1,idft_out1,1);
  //   dft72(idft_in2,idft_out2,1);
  //   break;

  // case 96:
  //   dft96(idft_in0,idft_out0,1);
  //   dft96(idft_in1,idft_out1,1);
  //   dft96(idft_in2,idft_out2,1);
  //   break;

  // case 108:
  //   dft108(idft_in0,idft_out0,1);
  //   dft108(idft_in1,idft_out1,1);
  //   dft108(idft_in2,idft_out2,1);
  //   break;

  // case 120:
  //   dft120(idft_in0,idft_out0,1);
  //   dft120(idft_in1,idft_out1,1);
  //   dft120(idft_in2,idft_out2,1);
  //   break;

  // case 144:
  //   dft144(idft_in0,idft_out0,1);
  //   dft144(idft_in1,idft_out1,1);
  //   dft144(idft_in2,idft_out2,1);
  //   break;

  // case 180:
  //   dft180(idft_in0,idft_out0,1);
  //   dft180(idft_in1,idft_out1,1);
  //   dft180(idft_in2,idft_out2,1);
  //   break;

  // case 192:
  //   dft192(idft_in0,idft_out0,1);
  //   dft192(idft_in1,idft_out1,1);
  //   dft192(idft_in2,idft_out2,1);
  //   break;

  // case 216:
  //   dft216(idft_in0,idft_out0,1);
  //   dft216(idft_in1,idft_out1,1);
  //   dft216(idft_in2,idft_out2,1);
  //   break;

  // case 240:
  //   dft240(idft_in0,idft_out0,1);
  //   dft240(idft_in1,idft_out1,1);
  //   dft240(idft_in2,idft_out2,1);
  //   break;

  // case 288:
  //   dft288(idft_in0,idft_out0,1);
  //   dft288(idft_in1,idft_out1,1);
  //   dft288(idft_in2,idft_out2,1);
  //   break;

  // case 300:
  //   dft300(idft_in0,idft_out0,1);
  //   dft300(idft_in1,idft_out1,1);
  //   dft300(idft_in2,idft_out2,1);
  //   break;

  // case 324:
  //   dft324((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft324((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft324((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 360:
  //   dft360((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft360((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft360((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 384:
  //   dft384((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft384((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft384((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 432:
  //   dft432((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft432((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft432((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 480:
  //   dft480((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft480((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft480((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 540:
  //   dft540((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft540((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft540((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 576:
  //   dft576((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft576((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft576((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 600:
  //   dft600((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft600((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft600((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 648:
  //   dft648((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft648((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft648((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 720:
  //   dft720((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft720((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft720((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 864:
  //   dft864((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft864((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft864((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 900:
  //   dft900((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft900((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft900((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 960:
  //   dft960((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft960((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft960((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 972:
  //   dft972((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft972((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft972((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 1080:
  //   dft1080((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft1080((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft1080((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 1152:
  //   dft1152((int16_t*)idft_in0,(int16_t*)idft_out0,1);
  //   dft1152((int16_t*)idft_in1,(int16_t*)idft_out1,1);
  //   dft1152((int16_t*)idft_in2,(int16_t*)idft_out2,1);
  //   break;

  // case 1200:
  //   dft1200(idft_in0,idft_out0,1);
  //   dft1200(idft_in1,idft_out1,1);
  //   dft1200(idft_in2,idft_out2,1);
  //   break;

  default:
    // should not be reached
    LOG_E( PHY, "Unsupported Msc_PUSCH value of %"PRIu16"\n", Msc_PUSCH );
    return;
  }



  for (i=0,ip=0; i<Msc_PUSCH; i++,ip+=4) {
    z0[i]     = ((uint32_t*)idft_out0)[ip];
    /*
      printf("out0 (%d,%d),(%d,%d),(%d,%d),(%d,%d)\n",
      ((int16_t*)&idft_out0[ip])[0],((int16_t*)&idft_out0[ip])[1],
      ((int16_t*)&idft_out0[ip+1])[0],((int16_t*)&idft_out0[ip+1])[1],
      ((int16_t*)&idft_out0[ip+2])[0],((int16_t*)&idft_out0[ip+2])[1],
      ((int16_t*)&idft_out0[ip+3])[0],((int16_t*)&idft_out0[ip+3])[1]);
    */
    z1[i]     = ((uint32_t*)idft_out0)[ip+1];
    z2[i]     = ((uint32_t*)idft_out0)[ip+2];
    z3[i]     = ((uint32_t*)idft_out0)[ip+3];
    z4[i]     = ((uint32_t*)idft_out1)[ip+0];
    z5[i]     = ((uint32_t*)idft_out1)[ip+1];
    z6[i]     = ((uint32_t*)idft_out1)[ip+2];
    z7[i]     = ((uint32_t*)idft_out1)[ip+3];
    z8[i]     = ((uint32_t*)idft_out2)[ip];
    z9[i]     = ((uint32_t*)idft_out2)[ip+1];

    // if (frame_parms->Ncp==0) {
      z10[i]    = ((uint32_t*)idft_out2)[ip+2];
      z11[i]    = ((uint32_t*)idft_out2)[ip+3];
    // }
  }

  // conjugate output
  for (i=0; i<(Msc_PUSCH>>2); i++) {
#if defined(__x86_64__) || defined(__i386__)
    ((__m128i*)z0)[i]=_mm_sign_epi16(((__m128i*)z0)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z1)[i]=_mm_sign_epi16(((__m128i*)z1)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z2)[i]=_mm_sign_epi16(((__m128i*)z2)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z3)[i]=_mm_sign_epi16(((__m128i*)z3)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z4)[i]=_mm_sign_epi16(((__m128i*)z4)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z5)[i]=_mm_sign_epi16(((__m128i*)z5)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z6)[i]=_mm_sign_epi16(((__m128i*)z6)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z7)[i]=_mm_sign_epi16(((__m128i*)z7)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z8)[i]=_mm_sign_epi16(((__m128i*)z8)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z9)[i]=_mm_sign_epi16(((__m128i*)z9)[i],*(__m128i*)&conjugate2[0]);

    // if (frame_parms->Ncp==NORMAL_NB_IoT) {
      ((__m128i*)z10)[i]=_mm_sign_epi16(((__m128i*)z10)[i],*(__m128i*)&conjugate2[0]);
      ((__m128i*)z11)[i]=_mm_sign_epi16(((__m128i*)z11)[i],*(__m128i*)&conjugate2[0]);
    // }
#elif defined(__arm__)
    *&(((int16x8_t*)z0)[i])=vmulq_s16(*&(((int16x8_t*)z0)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z1)[i])=vmulq_s16(*&(((int16x8_t*)z1)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z2)[i])=vmulq_s16(*&(((int16x8_t*)z2)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z3)[i])=vmulq_s16(*&(((int16x8_t*)z3)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z4)[i])=vmulq_s16(*&(((int16x8_t*)z4)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z5)[i])=vmulq_s16(*&(((int16x8_t*)z5)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z6)[i])=vmulq_s16(*&(((int16x8_t*)z6)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z7)[i])=vmulq_s16(*&(((int16x8_t*)z7)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z8)[i])=vmulq_s16(*&(((int16x8_t*)z8)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z9)[i])=vmulq_s16(*&(((int16x8_t*)z9)[i]),*(int16x8_t*)&conjugate2[0]);


    // if (frame_parms->Ncp==NORMAL_NB_IoT) {
      *&(((int16x8_t*)z10)[i])=vmulq_s16(*&(((int16x8_t*)z10)[i]),*(int16x8_t*)&conjugate2[0]);
      *&(((int16x8_t*)z11)[i])=vmulq_s16(*&(((int16x8_t*)z11)[i]),*(int16x8_t*)&conjugate2[0]);
    // }

#endif
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif

}
#endif


int32_t ulsch_bpsk_llr_NB_IoT(PHY_VARS_eNB *eNB, 
                              LTE_DL_FRAME_PARMS *frame_parms,
                              int32_t **rxdataF_comp,
                              int16_t *ulsch_llr, 
                              uint8_t symbol,
                              uint16_t ul_sc_start, 
                              uint8_t UE_id, 
                              int16_t **llrp)
{

  int16_t *rxF; 
 // uint32_t I_sc = 11;//eNB->ulsch_NB_IoT[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
 // uint16_t ul_sc_start; // subcarrier start index into UL RB 
  // int i; 

  //ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  rxF = (int16_t *)&rxdataF_comp[0][(symbol*frame_parms->N_RB_DL*12) + ul_sc_start]; 

  //  printf("qpsk llr for symbol %d (pos %d), llr offset %d\n",symbol,(symbol*frame_parms->N_RB_DL*12),llr128U-(__m128i*)ulsch_llr);

    //printf("%d,%d,%d,%d,%d,%d,%d,%d\n",((int16_t *)rxF)[0],((int16_t *)rxF)[1],((int16_t *)rxF)[2],((int16_t *)rxF)[3],((int16_t *)rxF)[4],((int16_t *)rxF)[5],((int16_t *)rxF)[6],((int16_t *)rxF)[7]);
    *(*llrp) = *rxF;
    //rxF++;
    (*llrp)++;

  return(0);

}

// int32_t ulsch_qpsk_llr_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
//                               int32_t **rxdataF_comp,
//                               int16_t *ulsch_llr,
//                               uint8_t symbol,
//                               uint16_t nb_rb,
//                               int16_t **llrp)
// {
// #if defined(__x86_64__) || defined(__i386__)
//   __m128i *rxF=(__m128i*)&rxdataF_comp[0][(symbol*frame_parms->N_RB_DL*12)];
//   __m128i **llrp128 = (__m128i **)llrp;
// #elif defined(__arm__)
//   int16x8_t *rxF= (int16x8_t*)&rxdataF_comp[0][(symbol*frame_parms->N_RB_DL*12)];
//   int16x8_t **llrp128 = (int16x8_t **)llrp;
// #endif

//   int i;

//   //  printf("qpsk llr for symbol %d (pos %d), llr offset %d\n",symbol,(symbol*frame_parms->N_RB_DL*12),llr128U-(__m128i*)ulsch_llr);

//   for (i=0; i<(nb_rb*3); i++) {
//     //printf("%d,%d,%d,%d,%d,%d,%d,%d\n",((int16_t *)rxF)[0],((int16_t *)rxF)[1],((int16_t *)rxF)[2],((int16_t *)rxF)[3],((int16_t *)rxF)[4],((int16_t *)rxF)[5],((int16_t *)rxF)[6],((int16_t *)rxF)[7]);
//     *(*llrp128) = *rxF;
//     rxF++;
//     (*llrp128)++;
//   }

// #if defined(__x86_64__) || defined(__i386__)
//   _mm_empty();
//   _m_empty();
// #endif

//   return(0);

// }

int32_t ulsch_qpsk_llr_NB_IoT(PHY_VARS_eNB *eNB, 
                              LTE_DL_FRAME_PARMS *frame_parms,
                              int32_t **rxdataF_comp,
                              int16_t *ulsch_llr, 
                              uint8_t symbol, 
                              uint8_t UE_id,
                              uint16_t ul_sc_start,
                              uint8_t Nsc_RU,
                              int16_t *llrp)
{

  int32_t *rxF; 
  int32_t *llrp32; // = (int32_t *)llrp; 
  //uint32_t I_sc = 11;//eNB->ulsch_NB_IoT[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  //uint16_t ul_sc_start; // subcarrier start index into UL RB 
  //uint8_t Nsc_RU = 1;//eNB->ulsch_NB_IoT[UE_id]->harq_process->N_sc_RU; // Vincent: number of sc 1,3,6,12 
  int i; 
  
  llrp32 = (int32_t *)&llrp[0];
  //ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  rxF = (int32_t *)&rxdataF_comp[0][(symbol*frame_parms->N_RB_DL*12) + ul_sc_start]; 

  //  printf("qpsk llr for symbol %d (pos %d), llr offset %d\n",symbol,(symbol*frame_parms->N_RB_DL*12),llr128U-(__m128i*)ulsch_llr);

  for (i=0; i<Nsc_RU; i++) {
    //printf("%d,%d,%d,%d,%d,%d,%d,%d\n",((int16_t *)rxF)[0],((int16_t *)rxF)[1],((int16_t *)rxF)[2],((int16_t *)rxF)[3],((int16_t *)rxF)[4],((int16_t *)rxF)[5],((int16_t *)rxF)[6],((int16_t *)rxF)[7]);
    /**(*llrp32) = *rxF;
    rxF++;
    (*llrp32)++;*/
    llrp32[i] = rxF[i]; 
  /*printf("\nin llr_%d === %d",ul_sc_start,(int32_t)llrp[i]); 
  printf("\n  in llr_%d === %d",ul_sc_start,llrp32[i]);*/
  }

  return(0);

}


void ulsch_detection_mrc_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,
                         int32_t **rxdataF_comp,
                         int32_t **ul_ch_mag,
                         int32_t **ul_ch_magb,
                         uint8_t symbol,
                         uint16_t nb_rb)
{

#if defined(__x86_64__) || defined(__i386__)

  __m128i *rxdataF_comp128_0,*ul_ch_mag128_0,*ul_ch_mag128_0b;
  __m128i *rxdataF_comp128_1,*ul_ch_mag128_1,*ul_ch_mag128_1b;
#elif defined(__arm__)

  int16x8_t *rxdataF_comp128_0,*ul_ch_mag128_0,*ul_ch_mag128_0b;
  int16x8_t *rxdataF_comp128_1,*ul_ch_mag128_1,*ul_ch_mag128_1b;

#endif
  int32_t i;

  if (frame_parms->nb_antennas_rx>1) {
#if defined(__x86_64__) || defined(__i386__)
    rxdataF_comp128_0   = (__m128i *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12];
    rxdataF_comp128_1   = (__m128i *)&rxdataF_comp[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0      = (__m128i *)&ul_ch_mag[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1      = (__m128i *)&ul_ch_mag[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0b     = (__m128i *)&ul_ch_magb[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1b     = (__m128i *)&ul_ch_magb[1][symbol*frame_parms->N_RB_DL*12];

    // MRC on each re of rb, both on MF output and magnitude (for 16QAM/64QAM llr computation)
    for (i=0; i<nb_rb*3; i++) {
      rxdataF_comp128_0[i] = _mm_adds_epi16(_mm_srai_epi16(rxdataF_comp128_0[i],1),_mm_srai_epi16(rxdataF_comp128_1[i],1));
      ul_ch_mag128_0[i]    = _mm_adds_epi16(_mm_srai_epi16(ul_ch_mag128_0[i],1),_mm_srai_epi16(ul_ch_mag128_1[i],1));
      ul_ch_mag128_0b[i]   = _mm_adds_epi16(_mm_srai_epi16(ul_ch_mag128_0b[i],1),_mm_srai_epi16(ul_ch_mag128_1b[i],1));
      rxdataF_comp128_0[i] = _mm_add_epi16(rxdataF_comp128_0[i],(*(__m128i*)&jitterc[0]));
    }

#elif defined(__arm__)
    rxdataF_comp128_0   = (int16x8_t *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12];
    rxdataF_comp128_1   = (int16x8_t *)&rxdataF_comp[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0      = (int16x8_t *)&ul_ch_mag[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1      = (int16x8_t *)&ul_ch_mag[1][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_0b     = (int16x8_t *)&ul_ch_magb[0][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128_1b     = (int16x8_t *)&ul_ch_magb[1][symbol*frame_parms->N_RB_DL*12];

    // MRC on each re of rb, both on MF output and magnitude (for 16QAM/64QAM llr computation)
    for (i=0; i<nb_rb*3; i++) {
      rxdataF_comp128_0[i] = vhaddq_s16(rxdataF_comp128_0[i],rxdataF_comp128_1[i]);
      ul_ch_mag128_0[i]    = vhaddq_s16(ul_ch_mag128_0[i],ul_ch_mag128_1[i]);
      ul_ch_mag128_0b[i]   = vhaddq_s16(ul_ch_mag128_0b[i],ul_ch_mag128_1b[i]);
      rxdataF_comp128_0[i] = vqaddq_s16(rxdataF_comp128_0[i],(*(int16x8_t*)&jitterc[0]));
    }


#endif
    
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

void ulsch_extract_rbs_single_NB_IoT(int32_t **rxdataF,
                                     int32_t **rxdataF_ext, 
                                     uint16_t UL_RB_ID_NB_IoT, // index of UL NB_IoT resource block !!! may be defined twice : in frame_parms and in NB_IoT_UL_eNB_HARQ_t
                                     uint16_t N_sc_RU, // number of subcarriers in UL 
                                     uint8_t l,
                                     uint8_t Ns,
                                     LTE_DL_FRAME_PARMS *frame_parms)
{
  uint16_t  nb_rb1; 
  // uint16_t  nb_rb2; 
  uint8_t   aarx,n;
  // int32_t   *rxF,*rxF_ext;
  //uint8_t symbol = l+Ns*frame_parms->symbols_per_tti/2;
  uint8_t   symbol = l+(7*(Ns&1)); ///symbol within sub-frame 
  // uint16_t ul_sc_start; // subcarrier start index into UL RB 
  //unsigned short UL_RB_ID_NB_IoT; 

  // ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); 
  //UL_RB_ID_NB_IoT = frame_parms->NB_IoT_RB_ID; 
  ////////////////////////////////////////////////////// if NB_IoT_start is used , no need for nb_rb1 and nb_rb2 

  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

    nb_rb1 = cmin(cmax((int)(frame_parms->N_RB_UL) - (int)(2*UL_RB_ID_NB_IoT),(int)0),(int)(2));    // 2 times no. RBs before the DC
                                 // 2 times no. RBs after the DC

    // rxF_ext   = &rxdataF_ext[aarx][(symbol*frame_parms->N_RB_UL*12)];
  /*  for (n=0;n<12;n++){

            rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12 + n] = rxdataF[aarx][NB_IoT_start_UL + (symbol)*frame_parms->ofdm_symbol_size + n]; 
        // since NB_IoT_start -1 is to apply for UPLINK, in this case the computation of start ul and dl should be done during the initializaiton
    }*/

    if (nb_rb1) { // RB NB-IoT is in the first half

      for (n=0;n<12;n++){ // extract whole RB of 12 subcarriers
        // Note that FFT splits the RBs 
        // !!! Note that frame_parms->N_RB_UL is the number of RB in LTE
        // rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12 + n] = rxdataF[aarx][UL_RB_ID_NB_IoT*12 + ul_sc_start + frame_parms->first_carrier_offset + symbol*frame_parms->ofdm_symbol_size + n];
        rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12 + n] = rxdataF[aarx][UL_RB_ID_NB_IoT*12 + frame_parms->first_carrier_offset + (symbol)*frame_parms->ofdm_symbol_size + n]; 
        //rxdataF_ext[aarx][symbol*12 + n] = rxdataF[aarx][UL_RB_ID_NB_IoT*12 + frame_parms->first_carrier_offset + symbol*frame_parms->ofdm_symbol_size + n];
        //// RB 22
      }

      // rxF = &rxdataF[aarx][(first_rb*12 + frame_parms->first_carrier_offset + symbol*frame_parms->ofdm_symbol_size)];
      // memcpy(rxF_ext, rxF, nb_rb1*6*sizeof(int));
      // rxF_ext += nb_rb1*6;

      // if (nb_rb2)  {
      //   //#ifdef OFDMA_ULSCH
      //   //  rxF = &rxdataF[aarx][(1 + symbol*frame_parms->ofdm_symbol_size)*2];
      //   //#else
      //   rxF = &rxdataF[aarx][(symbol*frame_parms->ofdm_symbol_size)];
      //   //#endif
      //   memcpy(rxF_ext, rxF, nb_rb2*6*sizeof(int));
      //   rxF_ext += nb_rb2*6;
      // }
    } else { // RB NB-IoT is in the second half 
              // RB 2
     
      for (n=0;n<12;n++){ // extract whole RB of 12 subcarriers
        // Note that FFT splits the RBs 
        // rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12 + n] = rxdataF[aarx][6*(2*UL_RB_ID_NB_IoT - frame_parms->N_RB_UL) +  ul_sc_start + symbol*frame_parms->ofdm_symbol_size + n]; 
        rxdataF_ext[aarx][symbol*frame_parms->N_RB_UL*12 + n] = rxdataF[aarx][6*(2*UL_RB_ID_NB_IoT - frame_parms->N_RB_UL) + (symbol)*frame_parms->ofdm_symbol_size + n]; 
        //printf("   rx_22_%d = %d   ",n,rxdataF[aarx][6*(2*UL_RB_ID_NB_IoT - frame_parms->N_RB_UL) + (subframe*14+symbol)*frame_parms->ofdm_symbol_size + n]); 
        //printf("   rx_20_%d = %d   ",n,rxdataF[aarx][6*(2*(UL_RB_ID_NB_IoT-7) - frame_parms->N_RB_UL) + (subframe*14+symbol)*frame_parms->ofdm_symbol_size + n]);
        //rxdataF_ext[aarx][symbol*12 + n] = rxdataF[aarx][6*(2*UL_RB_ID_NB_IoT - frame_parms->N_RB_UL) + symbol*frame_parms->ofdm_symbol_size + n];
      }
      //#ifdef OFDMA_ULSCH
      //      rxF = &rxdataF[aarx][(1 + 6*(2*first_rb - frame_parms->N_RB_UL) + symbol*frame_parms->ofdm_symbol_size)*2];
      //#else
      // rxF = &rxdataF[aarx][(6*(2*first_rb - frame_parms->N_RB_UL) + symbol*frame_parms->ofdm_symbol_size)]; 
      // //#endif
      // memcpy(rxF_ext, rxF, nb_rb2*6*sizeof(int));
      // rxF_ext += nb_rb2*6;
    }
  }

}


void ulsch_channel_compensation_NB_IoT(int32_t **rxdataF_ext,
                                int32_t **ul_ch_estimates_ext,
                                int32_t **ul_ch_mag,
                                int32_t **ul_ch_magb,
                                int32_t **rxdataF_comp,
                                LTE_DL_FRAME_PARMS *frame_parms,
                                uint8_t symbol,
                                uint8_t Qm,
                                uint16_t nb_rb,
                                uint8_t output_shift)
{

  // uint16_t rb;

#if defined(__x86_64__) || defined(__i386__)

  __m128i *ul_ch128,*ul_ch_mag128,*ul_ch_mag128b,*rxdataF128,*rxdataF_comp128;
  uint8_t aarx;//,symbol_mod;
  __m128i mmtmpU0,mmtmpU1,mmtmpU2,mmtmpU3;
#ifdef OFDMA_ULSCH
  __m128i QAM_amp128U,QAM_amp128bU;
#endif

#elif defined(__arm__)

  int16x4_t *ul_ch128,*rxdataF128;
  int16x8_t *ul_ch_mag128,*ul_ch_mag128b,*rxdataF_comp128;

  uint8_t aarx;//,symbol_mod;
  int32x4_t mmtmpU0,mmtmpU1,mmtmpU0b,mmtmpU1b;
#ifdef OFDMA_ULSCH
  int16x8_t mmtmpU2,mmtmpU3;
  int16x8_t QAM_amp128U,QAM_amp128bU;
#endif
  int16_t conj[4]__attribute__((aligned(16))) = {1,-1,1,-1};
  int32x4_t output_shift128 = vmovq_n_s32(-(int32_t)output_shift);



#endif



  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

#if defined(__x86_64__) || defined(__i386__)

    ul_ch128          = (__m128i *)&ul_ch_estimates_ext[aarx][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128      = (__m128i *)&ul_ch_mag[aarx][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128b     = (__m128i *)&ul_ch_magb[aarx][symbol*frame_parms->N_RB_DL*12];
    rxdataF128        = (__m128i *)&rxdataF_ext[aarx][symbol*frame_parms->N_RB_DL*12];
    rxdataF_comp128   = (__m128i *)&rxdataF_comp[aarx][symbol*frame_parms->N_RB_DL*12];

#elif defined(__arm__)


    ul_ch128          = (int16x4_t *)&ul_ch_estimates_ext[aarx][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128      = (int16x8_t *)&ul_ch_mag[aarx][symbol*frame_parms->N_RB_DL*12];
    ul_ch_mag128b     = (int16x8_t *)&ul_ch_magb[aarx][symbol*frame_parms->N_RB_DL*12];
    rxdataF128        = (int16x4_t *)&rxdataF_ext[aarx][symbol*frame_parms->N_RB_DL*12];
    rxdataF_comp128   = (int16x8_t *)&rxdataF_comp[aarx][symbol*frame_parms->N_RB_DL*12];

#endif


#if defined(__x86_64__) || defined(__i386__)
      mmtmpU0 = _mm_madd_epi16(ul_ch128[0],ul_ch128[0]);

      mmtmpU0 = _mm_srai_epi32(mmtmpU0,output_shift);
      mmtmpU1 = _mm_madd_epi16(ul_ch128[1],ul_ch128[1]);

      mmtmpU1 = _mm_srai_epi32(mmtmpU1,output_shift);

      mmtmpU0 = _mm_packs_epi32(mmtmpU0,mmtmpU1);

      ul_ch_mag128[0] = _mm_unpacklo_epi16(mmtmpU0,mmtmpU0);
      ul_ch_mag128[1] = _mm_unpackhi_epi16(mmtmpU0,mmtmpU0);

      mmtmpU0 = _mm_madd_epi16(ul_ch128[2],ul_ch128[2]);

      mmtmpU0 = _mm_srai_epi32(mmtmpU0,output_shift);
      mmtmpU1 = _mm_packs_epi32(mmtmpU0,mmtmpU0);
      ul_ch_mag128[2] = _mm_unpacklo_epi16(mmtmpU1,mmtmpU1);

      // printf("comp: symbol %d rb %d => %d,%d,%d (output_shift %d)\n",symbol,rb,*((int16_t*)&ul_ch_mag128[0]),*((int16_t*)&ul_ch_mag128[1]),*((int16_t*)&ul_ch_mag128[2]),output_shift);


#elif defined(__arm__)
          mmtmpU0 = vmull_s16(ul_ch128[0], ul_ch128[0]);
          mmtmpU0 = vqshlq_s32(vqaddq_s32(mmtmpU0,vrev64q_s32(mmtmpU0)),-output_shift128);
          mmtmpU1 = vmull_s16(ul_ch128[1], ul_ch128[1]);
          mmtmpU1 = vqshlq_s32(vqaddq_s32(mmtmpU1,vrev64q_s32(mmtmpU1)),-output_shift128);
          ul_ch_mag128[0] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));
          mmtmpU0 = vmull_s16(ul_ch128[2], ul_ch128[2]);
          mmtmpU0 = vqshlq_s32(vqaddq_s32(mmtmpU0,vrev64q_s32(mmtmpU0)),-output_shift128);
          mmtmpU1 = vmull_s16(ul_ch128[3], ul_ch128[3]);
          mmtmpU1 = vqshlq_s32(vqaddq_s32(mmtmpU1,vrev64q_s32(mmtmpU1)),-output_shift128);
          ul_ch_mag128[1] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));
          mmtmpU0 = vmull_s16(ul_ch128[4], ul_ch128[4]);
          mmtmpU0 = vqshlq_s32(vqaddq_s32(mmtmpU0,vrev64q_s32(mmtmpU0)),-output_shift128);
          mmtmpU1 = vmull_s16(ul_ch128[5], ul_ch128[5]);
          mmtmpU1 = vqshlq_s32(vqaddq_s32(mmtmpU1,vrev64q_s32(mmtmpU1)),-output_shift128);
          ul_ch_mag128[2] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));

#endif
// #endif

#if defined(__x86_64__) || defined(__i386__)
      // multiply by conjugated channel
      mmtmpU0 = _mm_madd_epi16(ul_ch128[0],rxdataF128[0]);
      //        print_ints("re",&mmtmpU0);

      // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
      mmtmpU1 = _mm_shufflelo_epi16(ul_ch128[0],_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)&conjugate[0]);

      mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[0]);
      //      print_ints("im",&mmtmpU1);
      // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
      mmtmpU0 = _mm_srai_epi32(mmtmpU0,output_shift);
      //  print_ints("re(shift)",&mmtmpU0);
      mmtmpU1 = _mm_srai_epi32(mmtmpU1,output_shift);
      //  print_ints("im(shift)",&mmtmpU1);
      mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
      mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);
      //        print_ints("c0",&mmtmpU2);
      //  print_ints("c1",&mmtmpU3);
      rxdataF_comp128[0] = _mm_packs_epi32(mmtmpU2,mmtmpU3);
      /*
              print_shorts("rx:",&rxdataF128[0]);
              print_shorts("ch:",&ul_ch128[0]);
              print_shorts("pack:",&rxdataF_comp128[0]);
      */
      // multiply by conjugated channel
      mmtmpU0 = _mm_madd_epi16(ul_ch128[1],rxdataF128[1]);
      // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
      mmtmpU1 = _mm_shufflelo_epi16(ul_ch128[1],_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)conjugate);
      mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[1]);
      // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
      mmtmpU0 = _mm_srai_epi32(mmtmpU0,output_shift);
      mmtmpU1 = _mm_srai_epi32(mmtmpU1,output_shift);
      mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
      mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);

      rxdataF_comp128[1] = _mm_packs_epi32(mmtmpU2,mmtmpU3);
      //        print_shorts("rx:",rxdataF128[1]);
      //        print_shorts("ch:",ul_ch128[1]);
      //        print_shorts("pack:",rxdataF_comp128[1]);
      //       multiply by conjugated channel
      mmtmpU0 = _mm_madd_epi16(ul_ch128[2],rxdataF128[2]);
      // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
      mmtmpU1 = _mm_shufflelo_epi16(ul_ch128[2],_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
      mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)conjugate);
      mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[2]);
      // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
      mmtmpU0 = _mm_srai_epi32(mmtmpU0,output_shift);
      mmtmpU1 = _mm_srai_epi32(mmtmpU1,output_shift);
      mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
      mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);

      rxdataF_comp128[2] = _mm_packs_epi32(mmtmpU2,mmtmpU3);
      //        print_shorts("rx:",rxdataF128[2]);
      //        print_shorts("ch:",ul_ch128[2]);
      //        print_shorts("pack:",rxdataF_comp128[2]);

      // Add a jitter to compensate for the saturation in "packs" resulting in a bias on the DC after IDFT
      rxdataF_comp128[0] = _mm_add_epi16(rxdataF_comp128[0],(*(__m128i*)&jitter[0]));
      rxdataF_comp128[1] = _mm_add_epi16(rxdataF_comp128[1],(*(__m128i*)&jitter[0]));
      rxdataF_comp128[2] = _mm_add_epi16(rxdataF_comp128[2],(*(__m128i*)&jitter[0]));

      ul_ch128+=3;
      ul_ch_mag128+=3;
      ul_ch_mag128b+=3;
      rxdataF128+=3;
      rxdataF_comp128+=3;
#elif defined(__arm__)
        mmtmpU0 = vmull_s16(ul_ch128[0], rxdataF128[0]);
        //mmtmpU0 = [Re(ch[0])Re(rx[0]) Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1]) Im(ch[1])Im(ch[1])] 
        mmtmpU1 = vmull_s16(ul_ch128[1], rxdataF128[1]);
        //mmtmpU1 = [Re(ch[2])Re(rx[2]) Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3]) Im(ch[3])Im(ch[3])] 
        mmtmpU0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0),vget_high_s32(mmtmpU0)),
                               vpadd_s32(vget_low_s32(mmtmpU1),vget_high_s32(mmtmpU1)));
        //mmtmpU0 = [Re(ch[0])Re(rx[0])+Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1])+Im(ch[1])Im(ch[1]) Re(ch[2])Re(rx[2])+Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3])+Im(ch[3])Im(ch[3])] 

        mmtmpU0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[0],*(int16x4_t*)conj)), rxdataF128[0]);
        //mmtmpU0 = [-Im(ch[0])Re(rx[0]) Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1]) Re(ch[1])Im(rx[1])]
        mmtmpU1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[1],*(int16x4_t*)conj)), rxdataF128[1]);
        //mmtmpU0 = [-Im(ch[2])Re(rx[2]) Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3]) Re(ch[3])Im(rx[3])]
        mmtmpU1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0b),vget_high_s32(mmtmpU0b)),
                               vpadd_s32(vget_low_s32(mmtmpU1b),vget_high_s32(mmtmpU1b)));
        //mmtmpU1 = [-Im(ch[0])Re(rx[0])+Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1])+Re(ch[1])Im(rx[1]) -Im(ch[2])Re(rx[2])+Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3])+Re(ch[3])Im(rx[3])]

        mmtmpU0 = vqshlq_s32(mmtmpU0,-output_shift128);
        mmtmpU1 = vqshlq_s32(mmtmpU1,-output_shift128);
        rxdataF_comp128[0] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));
        mmtmpU0 = vmull_s16(ul_ch128[2], rxdataF128[2]);
        mmtmpU1 = vmull_s16(ul_ch128[3], rxdataF128[3]);
        mmtmpU0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0),vget_high_s32(mmtmpU0)),
                               vpadd_s32(vget_low_s32(mmtmpU1),vget_high_s32(mmtmpU1)));
        mmtmpU0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[2],*(int16x4_t*)conj)), rxdataF128[2]);
        mmtmpU1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[3],*(int16x4_t*)conj)), rxdataF128[3]);
        mmtmpU1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0b),vget_high_s32(mmtmpU0b)),
                               vpadd_s32(vget_low_s32(mmtmpU1b),vget_high_s32(mmtmpU1b)));
        mmtmpU0 = vqshlq_s32(mmtmpU0,-output_shift128);
        mmtmpU1 = vqshlq_s32(mmtmpU1,-output_shift128);
        rxdataF_comp128[1] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));

        mmtmpU0 = vmull_s16(ul_ch128[4], rxdataF128[4]);
        mmtmpU1 = vmull_s16(ul_ch128[5], rxdataF128[5]);
        mmtmpU0 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0),vget_high_s32(mmtmpU0)),
                               vpadd_s32(vget_low_s32(mmtmpU1),vget_high_s32(mmtmpU1)));

        mmtmpU0b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[4],*(int16x4_t*)conj)), rxdataF128[4]);
        mmtmpU1b = vmull_s16(vrev32_s16(vmul_s16(ul_ch128[5],*(int16x4_t*)conj)), rxdataF128[5]);
        mmtmpU1 = vcombine_s32(vpadd_s32(vget_low_s32(mmtmpU0b),vget_high_s32(mmtmpU0b)),
                               vpadd_s32(vget_low_s32(mmtmpU1b),vget_high_s32(mmtmpU1b)));

              
        mmtmpU0 = vqshlq_s32(mmtmpU0,-output_shift128);
        mmtmpU1 = vqshlq_s32(mmtmpU1,-output_shift128);
        rxdataF_comp128[2] = vcombine_s16(vmovn_s32(mmtmpU0),vmovn_s32(mmtmpU1));
              
              // Add a jitter to compensate for the saturation in "packs" resulting in a bias on the DC after IDFT
        rxdataF_comp128[0] = vqaddq_s16(rxdataF_comp128[0],(*(int16x8_t*)&jitter[0]));
        rxdataF_comp128[1] = vqaddq_s16(rxdataF_comp128[1],(*(int16x8_t*)&jitter[0]));
        rxdataF_comp128[2] = vqaddq_s16(rxdataF_comp128[2],(*(int16x8_t*)&jitter[0]));

      
        ul_ch128+=6;
        ul_ch_mag128+=3;
        ul_ch_mag128b+=3;
        rxdataF128+=6;
        rxdataF_comp128+=3;
              
#endif
    // }
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

void fill_rbs_zeros_NB_IoT(PHY_VARS_eNB *eNB, 
                            LTE_DL_FRAME_PARMS *frame_parms,
                            int32_t **rxdataF_comp,
                            uint16_t ul_sc_start,
                            uint8_t UE_id,
                            uint8_t symbol)
{

  //uint32_t I_sc = 11;//eNB->ulsch[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  uint8_t Nsc_RU = 1;//eNB->ulsch[UE_id]->harq_process->N_sc_RU; // Vincent: number of sc 1,3,6,12 
  //uint16_t ul_sc_start; // subcarrier start index into UL RB 
  int32_t *rxdataF_comp32;   
  uint8_t m; // index of subcarrier

 // ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB 
  rxdataF_comp32   = (int32_t *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12]; 
  if (Nsc_RU != 12){
    for (m=0;m<12;m++)
    { // 12 is the number of subcarriers per RB
        if (m == ul_sc_start)
        {
            m = m + Nsc_RU; // skip non-zeros subcarriers
        }

        if(m<12)
        {
            rxdataF_comp32[m] = 0; 
        }   
    }  
  }


}

/*
void rotate_single_carrier_NB_IoT(PHY_VARS_eNB          *eNB, 
                                  LTE_DL_FRAME_PARMS    *frame_parms,
                                  int32_t               **rxdataF_comp, 
                                  uint8_t               eNB_id,                // to be removed ??? since not used
                                  uint8_t               l,                     //symbol within subframe
                                  uint8_t               counter_msg3,          ///  to be replaced by the number of received part
                                  uint16_t              ul_sc_start,
                                  uint8_t               Qm,
                                  uint8_t               option)  // 0 for data and 1 for ACK
{

  //uint32_t I_sc = 11;//eNB->ulsch_NB_IoT[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  //uint16_t ul_sc_start; // subcarrier start index into UL RB 
  int16_t pi_2_re[2] = {32767 , 0}; 
  int16_t pi_2_im[2] = {0 , 32767}; 
  //int16_t pi_4_re[2] = {32767 , 25735}; 
  //int16_t pi_4_im[2] = {0 , 25736}; 
  int16_t pi_4_re[2] = {32767 , 23170}; 
  int16_t pi_4_im[2] = {0 , 23170}; 
  int16_t e_phi_re[120] = {32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, -1, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621}; 
  int16_t e_phi_im[120] = {0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, -1, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009}; 
  int16_t e_phi_re_m6[120] = {32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, -1, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621};
  int16_t e_phi_im_m6[120] = {0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, -1, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, 0, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, -1, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010, -1, 21402, 32412, 27683, 9511, -13279, -29622, -32767, -24812, -4808, 17530, 31356, 29955, 14009, 0, -21403, -32413, -27684, -9512, 13278, 29621, 32767, 24811, 4807, -17531, -31357, -29956, -14010};
  int16_t *rxdataF_comp16; 
  int16_t rxdataF_comp16_re, rxdataF_comp16_im,rxdataF_comp16_re_2,rxdataF_comp16_im_2;    

  //ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  rxdataF_comp16   = (int16_t *)&rxdataF_comp[eNB_id][l*frame_parms->N_RB_DL*12 + ul_sc_start]; 
  rxdataF_comp16_re = rxdataF_comp16[0]; 
  rxdataF_comp16_im = rxdataF_comp16[1]; 
  rxdataF_comp16_re_2 = rxdataF_comp16_re; 
  rxdataF_comp16_im_2 = rxdataF_comp16_re;

  if (Qm == 1){
    rxdataF_comp16_re_2 = (int16_t)(((int32_t)pi_2_re[l%2] * (int32_t)rxdataF_comp16_re + 
                        (int32_t)pi_2_im[l%2] * (int32_t)rxdataF_comp16_im)>>15); 
    rxdataF_comp16_im_2 = (int16_t)(((int32_t)pi_2_re[l%2] * (int32_t)rxdataF_comp16_im - 
                        (int32_t)pi_2_im[l%2] * (int32_t)rxdataF_comp16_re)>>15); 
  }
  if(Qm == 2){
    rxdataF_comp16_re_2 = (int16_t)(((int32_t)pi_4_re[l%2] * (int32_t)rxdataF_comp16_re + 
                        (int32_t)pi_4_im[l%2] * (int32_t)rxdataF_comp16_im)>>15); 
    rxdataF_comp16_im_2 = (int16_t)(((int32_t)pi_4_re[l%2] * (int32_t)rxdataF_comp16_im - 
                        (int32_t)pi_4_im[l%2] * (int32_t)rxdataF_comp16_re)>>15); 
  }

  if (option ==0)
  {
        rxdataF_comp16[0] = (int16_t)(((int32_t)e_phi_re[14*(8-counter_msg3) + l] * (int32_t)rxdataF_comp16_re_2 + 
                              (int32_t)e_phi_im[14*(8-counter_msg3) + l] * (int32_t)rxdataF_comp16_im_2)>>15); 
        rxdataF_comp16[1] = (int16_t)(((int32_t)e_phi_re[14*(8-counter_msg3) + l] * (int32_t)rxdataF_comp16_im_2 - 
                              (int32_t)e_phi_im[14*(8-counter_msg3) + l] * (int32_t)rxdataF_comp16_re_2)>>15); 
  } else {
       
          rxdataF_comp16[0] = (int16_t)(((int32_t)e_phi_re_m6[14*(2-counter_msg3) + l] * (int32_t)rxdataF_comp16_re_2 + 
                              (int32_t)e_phi_im_m6[14*(2-counter_msg3) + l] * (int32_t)rxdataF_comp16_im_2)>>15); 
          rxdataF_comp16[1] = (int16_t)(((int32_t)e_phi_re_m6[14*(2-counter_msg3) + l] * (int32_t)rxdataF_comp16_im_2 - 
                              (int32_t)e_phi_im_m6[14*(2-counter_msg3) + l] * (int32_t)rxdataF_comp16_re_2)>>15); 
//  }
  

//}*/
//////////////////////////////////////////////////////////////////////
void rotate_single_carrier_NB_IoT(PHY_VARS_eNB          *eNB, 
                                  LTE_DL_FRAME_PARMS    *frame_parms,
                                  int32_t               **rxdataF_comp, 
                                  uint8_t               eNB_id,
                                  uint8_t               symbol, //symbol within subframe
                                  uint8_t               counter_msg3,          ///  to be replaced by the number of received part
                                  uint16_t              ul_sc_start,
                                  uint8_t               Qm,
                                  uint16_t              N_SF_per_word, 
                                  uint8_t               option)
{

  //uint32_t I_sc = 10;//eNB->ulsch_NB_IoT[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  //uint16_t ul_sc_start; // subcarrier start index into UL RB 
  int16_t pi_2_re[2] = {32767 , 0}; 
  int16_t pi_2_im[2] = {0 , 32767}; 
  int16_t pi_4_re[2] = {32767 , 23170}; 
  int16_t pi_4_im[2] = {0 , 23170}; 
  int16_t *e_phi_re,*e_phi_im;
  int16_t *rxdataF_comp16; 
  int16_t rxdataF_comp16_re, rxdataF_comp16_im,rxdataF_comp16_re_2,rxdataF_comp16_im_2;    
  
  int32_t sign_pm[2] = {1,-1}; 
  int8_t ind_sign_pm; // index for above table

  switch(ul_sc_start)
  {
    case 0: 
        e_phi_re = e_phi_re_m6; 
        e_phi_im = e_phi_im_m6; 
        break; 
    case 1: 
        e_phi_re = e_phi_re_m5; 
        e_phi_im = e_phi_im_m5; 
        break; 
    case 2: 
        e_phi_re = e_phi_re_m4; 
        e_phi_im = e_phi_im_m4; 
        break;
    case 3: 
        e_phi_re = e_phi_re_m3; 
        e_phi_im = e_phi_im_m3; 
        break;
    case 4: 
        e_phi_re = e_phi_re_m2; 
        e_phi_im = e_phi_im_m2; 
        break; 
    case 5: 
        e_phi_re = e_phi_re_m1; 
        e_phi_im = e_phi_im_m1; 
        break;
    case 6: 
        e_phi_re = e_phi_re_0; 
        e_phi_im = e_phi_im_0; 
        break;
    case 7: 
        e_phi_re = e_phi_re_p1; 
        e_phi_im = e_phi_im_p1; 
        break;
    case 8: 
        e_phi_re = e_phi_re_p2; 
        e_phi_im = e_phi_im_p2; 
        break;
    case 9: 
        e_phi_re = e_phi_re_p3; 
        e_phi_im = e_phi_im_p3; 
        break;
    case 10: 
        e_phi_re = e_phi_re_p4; 
        e_phi_im = e_phi_im_p4; 
        break;
    case 11: 
        e_phi_re = e_phi_re_p5; 
        e_phi_im = e_phi_im_p5; 
        break; 
  }
  ind_sign_pm = ((14*(N_SF_per_word-counter_msg3) + symbol)/14)%2;
  //ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  rxdataF_comp16   = (int16_t *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12 + ul_sc_start]; 
  rxdataF_comp16_re = rxdataF_comp16[0]; 
  rxdataF_comp16_im = rxdataF_comp16[1]; 
  rxdataF_comp16_re_2 = rxdataF_comp16_re; 
  rxdataF_comp16_im_2 = rxdataF_comp16_re;
    /// Apply two rotations, see section 10.1.5 in TS 36.211
  if (Qm == 1){ // rotation due to pi/2 BPSK
    rxdataF_comp16_re_2 = (int16_t)(((int32_t)pi_2_re[symbol%2] * (int32_t)rxdataF_comp16_re + 
                        (int32_t)pi_2_im[symbol%2] * (int32_t)rxdataF_comp16_im)>>15); 
    rxdataF_comp16_im_2 = (int16_t)(((int32_t)pi_2_re[symbol%2] * (int32_t)rxdataF_comp16_im - 
                        (int32_t)pi_2_im[symbol%2] * (int32_t)rxdataF_comp16_re)>>15); 
  }
  if(Qm == 2){ // rotation due to pi/4 QPSK
    rxdataF_comp16_re_2 = (int16_t)(((int32_t)pi_4_re[symbol%2] * (int32_t)rxdataF_comp16_re + 
                        (int32_t)pi_4_im[symbol%2] * (int32_t)rxdataF_comp16_im)>>15); 
    rxdataF_comp16_im_2 = (int16_t)(((int32_t)pi_4_re[symbol%2] * (int32_t)rxdataF_comp16_im - 
                        (int32_t)pi_4_im[symbol%2] * (int32_t)rxdataF_comp16_re)>>15); 
  }

      if(option==0) // rotation for msg3 (NPUSCH format 1)
    {
              rxdataF_comp16[0] = (int16_t)(((int32_t)e_phi_re[(14*(N_SF_per_word-counter_msg3) + symbol)%14] * sign_pm[ind_sign_pm] * (int32_t)rxdataF_comp16_re_2 + 
                        (int32_t)e_phi_im[(14*(N_SF_per_word-counter_msg3) + symbol)%14] * sign_pm[ind_sign_pm] * (int32_t)rxdataF_comp16_im_2)>>15); 
              rxdataF_comp16[1] = (int16_t)(((int32_t)e_phi_re[(14*(N_SF_per_word-counter_msg3) + symbol)%14] * sign_pm[ind_sign_pm] * (int32_t)rxdataF_comp16_im_2 - 
                        (int32_t)e_phi_im[(14*(N_SF_per_word-counter_msg3) + symbol)%14] * sign_pm[ind_sign_pm] * (int32_t)rxdataF_comp16_re_2)>>15); 
    
    }
      if(option==1) // rotation for msg5 (NPUSCH format 1)
    {
              rxdataF_comp16[0] = (int16_t)(((int32_t)e_phi_re[14*(2-counter_msg3) + symbol] * (int32_t)rxdataF_comp16_re_2 + 
                        (int32_t)e_phi_im[14*(2-counter_msg3) + symbol] * (int32_t)rxdataF_comp16_im_2)>>15); 
              rxdataF_comp16[1] = (int16_t)(((int32_t)e_phi_re[14*(2-counter_msg3) + symbol] * (int32_t)rxdataF_comp16_im_2 - 
                        (int32_t)e_phi_im[14*(2-counter_msg3) + symbol] * (int32_t)rxdataF_comp16_re_2)>>15); 
    }

}

//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

void rotate_bpsk_NB_IoT(PHY_VARS_eNB *eNB, 
                        LTE_DL_FRAME_PARMS *frame_parms,
                        int32_t **rxdataF_comp, 
                        uint16_t ul_sc_start,
                        uint8_t UE_id,
                        uint8_t symbol)
{

  //uint32_t I_sc = eNB->ulsch_NB_IoT[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  //uint16_t ul_sc_start; // subcarrier start index into UL RB 
  int16_t m_pi_4_re = 25735; // cos(pi/4) 
  int16_t m_pi_4_im = 25736; // sin(pi/4) 
  int16_t *rxdataF_comp16; 
  int16_t rxdataF_comp16_re, rxdataF_comp16_im; 

  //ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  rxdataF_comp16   = (int16_t *)&rxdataF_comp[0][symbol*frame_parms->N_RB_DL*12 + ul_sc_start]; 
  rxdataF_comp16_re = rxdataF_comp16[0]; 
  rxdataF_comp16_im = rxdataF_comp16[1]; 

  rxdataF_comp16[0] = (int16_t)(((int32_t)m_pi_4_re * (int32_t)rxdataF_comp16_re + 
                        (int32_t)m_pi_4_im * (int32_t)rxdataF_comp16_im)>>15); 
  rxdataF_comp16[1] = (int16_t)(((int32_t)m_pi_4_re * (int32_t)rxdataF_comp16_im -  
                        (int32_t)m_pi_4_im * (int32_t)rxdataF_comp16_re)>>15); 

} 





#if defined(__x86_64__) || defined(__i386__)
__m128i avg128U;
#elif defined(__arm__)
int32x4_t avg128U;
#endif

void ulsch_channel_level_NB_IoT(int32_t **drs_ch_estimates_ext,
                                LTE_DL_FRAME_PARMS *frame_parms,
                                int32_t *avg,
                                uint16_t nb_rb)
{

  // int16_t rb;
  uint8_t aarx;
#if defined(__x86_64__) || defined(__i386__)
  __m128i *ul_ch128;
#elif defined(__arm__)
  int16x4_t *ul_ch128;
#endif
  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {
    //clear average level
#if defined(__x86_64__) || defined(__i386__)
    avg128U = _mm_setzero_si128();
    ul_ch128=(__m128i *)drs_ch_estimates_ext[aarx];

    // for (rb=0; rb<nb_rb; rb++) {

      avg128U = _mm_add_epi32(avg128U,_mm_madd_epi16(ul_ch128[0],ul_ch128[0]));
      avg128U = _mm_add_epi32(avg128U,_mm_madd_epi16(ul_ch128[1],ul_ch128[1]));
      avg128U = _mm_add_epi32(avg128U,_mm_madd_epi16(ul_ch128[2],ul_ch128[2]));

      ul_ch128+=3;


    // }

#elif defined(__arm__)
    avg128U = vdupq_n_s32(0);
    ul_ch128=(int16x4_t *)drs_ch_estimates_ext[aarx];

    // for (rb=0; rb<nb_rb; rb++) {

       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[0],ul_ch128[0]));
       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[1],ul_ch128[1]));
       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[2],ul_ch128[2]));
       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[3],ul_ch128[3]));
       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[4],ul_ch128[4]));
       avg128U = vqaddq_s32(avg128U,vmull_s16(ul_ch128[5],ul_ch128[5]));
       ul_ch128+=6;


    // }

#endif

    DevAssert( nb_rb );
    avg[aarx] = (((int*)&avg128U)[0] +
                 ((int*)&avg128U)[1] +
                 ((int*)&avg128U)[2] +
                 ((int*)&avg128U)[3])/(nb_rb*12);

  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

//////////////////////////////////////////////////////////////////////////////////////
void get_pilots_position(uint8_t npusch_format,uint8_t  subcarrier_spacing,uint8_t *pilot_pos1,uint8_t *pilot_pos2,uint8_t *pilots_slot)
{

      uint8_t      pilot_pos1_format1_15k     = 3, pilot_pos2_format1_15k   = 10;      // holds for npusch format 1, and 15 kHz subcarrier bandwidth
      uint8_t      pilot_pos1_format2_15k     = 2, pilot_pos2_format2_15k   = 9;       // holds for npusch format 2, and 15 kHz subcarrier bandwidth 
      uint8_t      pilot_pos1_format1_3_75k   = 4, pilot_pos2_format1_3_75k = 11;      // holds for npusch format 1, and 3.75 kHz subcarrier bandwidth
      uint8_t      pilot_pos1_format2_3_75k   = 0, pilot_pos2_format2_3_75k = 7;       // holds for npusch format 2, and 3.75 kHz subcarrier bandwidth 

      switch(npusch_format + subcarrier_spacing*2)
      {
        case 0:    // data
                *pilot_pos1 = pilot_pos1_format1_3_75k; 
                *pilot_pos2 = pilot_pos2_format1_3_75k;
                *pilots_slot=1;
        break;

        case 1:    // ACK
                *pilot_pos1 = pilot_pos1_format2_3_75k; 
                *pilot_pos2 = pilot_pos2_format2_3_75k;
                *pilots_slot=3;
        break;

        case 2:   // data 
                *pilot_pos1 = pilot_pos1_format1_15k; 
                *pilot_pos2 = pilot_pos2_format1_15k;
                *pilots_slot=1;
        break;

        case 3:  // ACK
                *pilot_pos1 = pilot_pos1_format2_15k; 
                *pilot_pos2 = pilot_pos2_format2_15k;
                *pilots_slot=3;
        break;

        default:
         printf("Error in rx_ulsch_NB_IoT");
        break;
      }

}
//////////////////////////////////////////////////////////////////////////////////////
void UL_channel_estimation_NB_IoT(PHY_VARS_eNB        *eNB,
                                  LTE_DL_FRAME_PARMS  *fp,
                                  uint16_t            UL_RB_ID_NB_IoT,
                                  uint16_t            Nsc_RU,
                                  uint8_t             pilot_pos1,
                                  uint8_t             pilot_pos2,
                                  uint16_t            ul_sc_start,
                                  uint8_t             Qm,
                                  uint16_t            N_SF_per_word,
                                  uint8_t             rx_subframe)
{
     LTE_eNB_PUSCH           *pusch_vars      =  eNB->pusch_vars[0]; // UE_id
     LTE_eNB_COMMON          *common_vars     =  &eNB->common_vars;  
     NB_IoT_eNB_NULSCH_t     *ulsch_NB_IoT    =  eNB->ulsch_NB_IoT[0];

     int l=0;

     for (l=0; l<fp->symbols_per_tti; l++)
      { 
            
             ulsch_extract_rbs_single_NB_IoT(common_vars->rxdataF[0],      // common_vars->rxdataF[eNB_id],
                                             pusch_vars->rxdataF_ext[0],   // pusch_vars->rxdataF_ext[eNB_id]
                                             UL_RB_ID_NB_IoT,              //ulsch[UE_id]->harq_process->UL_RB_ID_NB_IoT, // index of UL NB_IoT resource block 
                                             Nsc_RU,                       //1, //ulsch_NB_IoT[0]->harq_process->N_sc_RU, // number of subcarriers in UL  //////////////// high level parameter
                                             l%(fp->symbols_per_tti/2),    // (0..13)
                                             l/(fp->symbols_per_tti/2),    // (0,1)
                                             fp);
            if(ulsch_NB_IoT->npusch_format == 0)      // format 1
            {
                      ul_chest_tmp_NB_IoT(pusch_vars->rxdataF_ext[0],       // pusch_vars->rxdataF_ext[eNB_id],
                                          pusch_vars->drs_ch_estimates[0],  // pusch_vars->drs_ch_estimates[eNB_id]
                                          l%(fp->symbols_per_tti/2),        //symbol within slot 
                                          l/(fp->symbols_per_tti/2),
                                          ulsch_NB_IoT->counter_sf,         // counter_msg
                                          pilot_pos1,
                                          pilot_pos2,
                                          ul_sc_start,
                                          Qm,
                                          N_SF_per_word,
                                          fp); 
            } else {
                    /// Channel Estimation (NPUSCH format 2)
                      ul_chest_tmp_f2_NB_IoT(pusch_vars->rxdataF_ext[0],
                                            pusch_vars->drs_ch_estimates[0],
                                            l%(fp->symbols_per_tti/2),        //symbol within slot 
                                            l/(fp->symbols_per_tti/2), 
                                            ulsch_NB_IoT->counter_sf,         //counter_msg, 
                                            ulsch_NB_IoT->npusch_format,      // proc->flag_msg5, 
                                            rx_subframe,
                                            Qm,                               // =1
                                            ul_sc_start,                      // = 0   
                                            fp); 
            }
      }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void get_llr_per_sf_NB_IoT(PHY_VARS_eNB        *eNB,
                           LTE_DL_FRAME_PARMS  *fp,
                           uint8_t             npusch_format,
                           uint8_t             counter_sf,
                           uint16_t            N_SF_per_word,
                           uint8_t             pilot_pos1,
                           uint8_t             pilot_pos2,
                           uint16_t            ul_sc_start,
                           uint16_t            Nsc_RU)
{
      LTE_eNB_PUSCH           *pusch_vars      =  eNB->pusch_vars[0]; // UE_id
      int16_t                 *llrp;
      uint32_t                l,ii=0; 

      if(npusch_format == 0)   // format 1
      { 
             llrp = (int16_t*)&pusch_vars->llr[0+ (N_SF_per_word-counter_sf)*24];  /// 24= 12 symbols/SF * 2 // since Real and im

      } else {                      // format 2        

              llrp = (int16_t*)&pusch_vars->llr[0+ (2-counter_sf)*16]; // 16 = 8 symbols/SF * 2 // since real and im
      }

      for (l=0; l<fp->symbols_per_tti; l++)
      {
          if (l==pilot_pos1 || l==pilot_pos2) // skip pilots  // option 0 pilots = x,y,  for option 1 pilots = 2,9 (subcarrier_spacing=1, npush_format=1)
          { 
              if(npusch_format == 0)
              {
                  l++;
              } else {
                  l=l+3;
              }
          }
 
          ulsch_qpsk_llr_NB_IoT(eNB, 
                                fp,
                                pusch_vars->rxdataF_comp[0],    // pusch_vars->rxdataF_comp[eNB_id],
                                pusch_vars->llr,
                                l, 
                                0, // UE ID
                                ul_sc_start,
                                Nsc_RU,
                                &llrp[ii*2]); 
          ii++;
      }
}


////////////////////////////////////descrambling NPUSCH //////////////////////////////////////////

void descrambling_NPUSCH_data_NB_IoT(LTE_DL_FRAME_PARMS  *fp,
                                     int16_t             *ulsch_llr,
                                     int16_t             *y,
                                     uint8_t             Qm,
                                     unsigned int        Cmux,
                                     uint32_t            rnti_tmp,
                                     uint8_t             rx_subframe,
                                     uint32_t            rx_frame)
{

      unsigned int    j,jj;
      uint32_t        x1, x2, s=0;
      uint8_t         reset;

      x2 =  (rnti_tmp<<14) + (rx_subframe<<9) + ((rx_frame%2)<<13) + fp->Nid_cell; //this is c_init in 36.211 Sec 10.1.3.1

      reset = 1; 
      switch (Qm)
      {
          case 1:
                  jj=0; 
                  for (j=0; j<Cmux; j++)
                  { 

                      if (j%32==0) 
                      {
                          s = lte_gold_generic(&x1, &x2, reset);
                          //      printf("lte_gold[%d]=%x\n",i,s);
                          reset = 0;
                      }

                      if (((s>>(j%32))&1)==0)
                      {
                          y[j] = (ulsch_llr[jj<<1]>>1) + (ulsch_llr[(jj<<1)+1]>>1);
                          jj+=2;
                      } else {

                          y[j] = -(ulsch_llr[jj<<1]>>1) + (ulsch_llr[(jj<<1)+1]>>1);
                          jj+=2;
                      }
                  }
          break; 

          case 2:
                  for (j=0; j<Cmux*2; j++)
                  {

                        if (j%32==0) 
                        {
                            s = lte_gold_generic(&x1, &x2, reset);
                            //      printf("lte_gold[%d]=%x\n",i,s);
                            reset = 0;
                        }
                        if (((s>>(j%32))&1)==0)
                        {
                            y[j] = -ulsch_llr[j];
                        } else {

                            y[j] = ulsch_llr[j];
                        }
                  }
          break;
      }

}
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void descrambling_NPUSCH_ack_NB_IoT(LTE_DL_FRAME_PARMS  *fp,
                                    int32_t             *y_msg5,
                                    int32_t             *llr_msg5,
                                    uint32_t            rnti_tmp,
                                    uint16_t            *counter_ack,
                                    uint8_t             rx_subframe,
                                    uint32_t            rx_frame)
{
        int             l = 0;
        uint32_t        x1, x2, s=0;
        uint8_t         reset = 1;

        x2     =  (rnti_tmp<<14) + (rx_subframe<<9) + ((rx_frame%2)<<13) + fp->Nid_cell;  
        s = lte_gold_generic(&x1, &x2, reset);

        reset = 0;

        for (l=0;l<16;l++)
        {
            if (((s>>(l%32))&1)==1) //xor
            {
                    y_msg5[l] = -llr_msg5[l];
            } else {
                    y_msg5[l] = llr_msg5[l];
            }
            *counter_ack += (y_msg5[l]>>31)&1; 
        }
}

//////////////////////////////////////////////////////////////////////////////////////////
uint32_t  turbo_decoding_NB_IoT(PHY_VARS_eNB           *eNB,
                                NB_IoT_eNB_NULSCH_t    *ulsch_NB_IoT,
                                eNB_rxtx_proc_t        *proc,
                                uint8_t                 npusch_format,
                                unsigned int            G,
                                uint8_t                 rvdx,
                                uint8_t                 Qm,
                                uint32_t                rx_frame,
                                uint8_t                 rx_subframe)
{  
          NB_IoT_UL_eNB_HARQ_t    *ulsch_harq       = ulsch_NB_IoT->harq_process;

          int            r = 0, Kr = 0;
          unsigned int   r_offset=0,Kr_bytes,iind=0;
          uint8_t        crc_type;
          int            offset = 0;
          int16_t        dummy_w[MAX_NUM_ULSCH_SEGMENTS_NB_IoT][3*(6144+64)];
          int            ret = 1;
          unsigned int   E; 

          uint8_t (*tc)(int16_t *y,
                        uint8_t *,
                        uint16_t,
                        uint16_t,
                        uint16_t,
                        uint8_t,
                        uint8_t,
                        uint8_t,
                        time_stats_t *,
                        time_stats_t *,
                        time_stats_t *,
                        time_stats_t *,
                        time_stats_t *,
                        time_stats_t *,
                        time_stats_t *);

          tc = phy_threegpplte_turbo_decoder16;

          for (r=0; r<ulsch_harq->C; r++)
          {
              // Get Turbo interleaver parameters
              if (r<ulsch_harq->Cminus)
              {
                  Kr = ulsch_harq->Kminus;
              } else{
                  Kr = ulsch_harq->Kplus;
              }

              Kr_bytes = Kr>>3;

              if (Kr_bytes<=64)
              {
                  iind = (Kr_bytes-5);

              } else if (Kr_bytes <=128) { 
                  
                  iind = 59 + ((Kr_bytes-64)>>1);

              } else if (Kr_bytes <= 256) {
                  
                  iind = 91 + ((Kr_bytes-128)>>2);

              } else if (Kr_bytes <= 768) { 
                  
                  iind = 123 + ((Kr_bytes-256)>>3);

              } else {
                  LOG_E(PHY,"ulsch_decoding: Illegal codeword size %d!!!\n",Kr_bytes);
              }

              memset(&dummy_w[r][0],0,3*(6144+64)*sizeof(short));
              ulsch_harq->RTC[r] = generate_dummy_w(4+(Kr_bytes*8),
                                                    (uint8_t*)&dummy_w[r][0],
                                                    (r==0) ? ulsch_harq->F : 0);

              if (lte_rate_matching_turbo_rx(ulsch_harq->RTC[r],
                                             G,
                                             ulsch_harq->w[r],
                                             (uint8_t*) &dummy_w[r][0],
                                             ulsch_harq->e+r_offset,
                                             ulsch_harq->C,
                                             1,                           ////// not used
                                             0,                           //Uplink
                                             1,
                                             rvdx,                        //ulsch_harq->rvidx,
                                             (ulsch_harq->round==0)?1:0,  // clear
                                             Qm,                          //2 //get_Qm_ul(ulsch_harq->mcs),
                                             1,
                                             r,
                                             &E)==-1) 
              {
                  LOG_E(PHY,"ulsch_decoding.c: Problem in rate matching\n");
              }

              r_offset += E;

              sub_block_deinterleaving_turbo(4+Kr,
                                             &ulsch_harq->d[r][96],
                                             ulsch_harq->w[r]); 

              if (ulsch_harq->C == 1)
              { 
                  crc_type = CRC24_A;
              }else{
                  crc_type = CRC24_B;
              }
              // turbo decoding and CRC 
              ret = tc(&ulsch_harq->d[r][96],
                       ulsch_harq->c[r],
                       Kr,
                       f1f2mat_old[iind*2],
                       f1f2mat_old[(iind*2)+1],
                       ulsch_NB_IoT->max_turbo_iterations, // MAX_TURBO_ITERATIONS,
                       crc_type,
                       (r==0) ? ulsch_harq->F : 0,
                       &eNB->ulsch_tc_init_stats,
                       &eNB->ulsch_tc_alpha_stats,
                       &eNB->ulsch_tc_beta_stats,
                       &eNB->ulsch_tc_gamma_stats,
                       &eNB->ulsch_tc_ext_stats,
                       &eNB->ulsch_tc_intl1_stats,
                       &eNB->ulsch_tc_intl2_stats); 
              ///////////////end decoding /////////////
              if (ret != (1+ulsch_NB_IoT->max_turbo_iterations)) 
              {   
                  if (r<ulsch_harq->Cminus)       
                  {
                      Kr = ulsch_harq->Kminus;
                  } else {                        
                      Kr = ulsch_harq->Kplus; 
                      Kr_bytes = Kr>>3;
                  }
                  if (r==0)                       
                  {
                      memcpy(ulsch_harq->b,
                            &ulsch_harq->c[0][(ulsch_harq->F>>3)],
                            Kr_bytes - (ulsch_harq->F>>3) - ((ulsch_harq->C>1)?3:0));
                            offset = Kr_bytes - (ulsch_harq->F>>3) - ((ulsch_harq->C>1)?3:0);
                  } else {
                      memcpy(ulsch_harq->b+offset,
                             ulsch_harq->c[r],
                             Kr_bytes - ((ulsch_harq->C>1)?3:0));
                             offset += (Kr_bytes- ((ulsch_harq->C>1)?3:0));
                  }
                  
                  fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,1); // indicate ACK to MAC
                  fill_rx_indication_NB_IoT(eNB,proc,npusch_format,1);
                  printf(" NPUSCH OK\n");
              } else { 

                  if (r<ulsch_harq->Cminus)       
                  {
                      Kr = ulsch_harq->Kminus;
                  } else {                        
                      Kr = ulsch_harq->Kplus; 
                      Kr_bytes = Kr>>3;
                  }
                  if (r==0)                       
                  {
                      memcpy(ulsch_harq->b,
                            &ulsch_harq->c[0][(ulsch_harq->F>>3)],
                            Kr_bytes - (ulsch_harq->F>>3) - ((ulsch_harq->C>1)?3:0));
                            offset = Kr_bytes - (ulsch_harq->F>>3) - ((ulsch_harq->C>1)?3:0);
                  } else {
                      memcpy(ulsch_harq->b+offset,
                             ulsch_harq->c[r],
                             Kr_bytes - ((ulsch_harq->C>1)?3:0));
                             offset += (Kr_bytes- ((ulsch_harq->C>1)?3:0));
                  }

                  int x = 0;
                  LOG_N(PHY,"Show the undecoded data: ");
                  for (x = 0; x < ulsch_harq->TBS; x ++)
                    printf("%02x ",ulsch_harq->b[x]);
                  printf("\n");

                  if (ulsch_harq->b[14] == 0x00 && ulsch_harq->b[15] == 0x07 && ulsch_harq->b[16] == 0x5e)
                  {
                    printf("Try to recovery Security mode complete, show the 11 th byte : %02x \n",ulsch_harq->b[11]);
                    //ulsch_harq->b[11] = ulsch_harq->b[11] + 0x08;
                    ulsch_harq->b[17] = 0x00;
                    fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,1); // indicate ACK to MAC
                    fill_rx_indication_NB_IoT(eNB,proc,npusch_format,1);                    
                  }else
                  {
                    fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,0);   // indicate NAK to MAC 
                    fill_rx_indication_NB_IoT(eNB,proc,npusch_format,0);
                    printf(" NPUSCH NOT OK\n");
                  }

              }
          }  ////////////  r loop end  ////////////

}

//////////////////////////////////////////////////////////////////////////////////////////
void deinterleaving_NPUSCH_data_NB_IoT(NB_IoT_UL_eNB_HARQ_t *ulsch_harq, int16_t *y, unsigned int G)
{
    
    unsigned int    j2=0;
    int16_t         *yp,*ep;
    int             iprime;

    for (iprime=0,yp=&y[j2],ep=&ulsch_harq->e[0]; iprime<G; iprime+=8,j2+=8,ep+=8,yp+=8)
    {
         ep[0] = yp[0];
         ep[1] = yp[1];
         ep[2] = yp[2];
         ep[3] = yp[3];
         ep[4] = yp[4];
         ep[5] = yp[5];
         ep[6] = yp[6];
         ep[7] = yp[7];
    }

}
 
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void decode_NPUSCH_msg_NB_IoT(PHY_VARS_eNB        *eNB,
                              LTE_DL_FRAME_PARMS  *fp,
                              eNB_rxtx_proc_t     *proc,
                              uint8_t             npusch_format,
                              uint16_t            N_SF_per_word,
                              uint16_t            Nsc_RU,
                              uint16_t            N_UL_slots,
                              uint8_t             Qm,
                              uint8_t             pilots_slot,
                              uint32_t            rnti_tmp,
                              uint8_t             rx_subframe,
                              uint32_t            rx_frame)
{
      LTE_eNB_PUSCH       *pusch_vars   =  eNB->pusch_vars[0];                  //  eNB->pusch_vars[UE_id];

      NB_IoT_eNB_NULSCH_t     *ulsch_NB_IoT     = eNB->ulsch_NB_IoT[0];
      NB_IoT_UL_eNB_HARQ_t    *ulsch_harq       = ulsch_NB_IoT->harq_process;

      unsigned int    A      = (ulsch_harq->TBS)*8;
      uint8_t         rvdx   = ulsch_harq->rvidx;
          
      if (npusch_format == 0)
      {
          int16_t         *ulsch_llr    = eNB->pusch_vars[0]->llr;  // eNB->pusch_vars[eNB_id]->llr;      //UE_id=0

          unsigned int    G,H,Hprime,Hpp,Cmux,Rmux_prime;
          int16_t         y[6*14*1200] __attribute__((aligned(32)));
          uint8_t         ytag[14*1200];
          G     =  (7-pilots_slot) * Qm * N_UL_slots * Nsc_RU; //(1 * Q_m) * 6 * 16; // Vincent : see 36.212, Section 5.1.4.1.2  // 16 slot(total number of slots) * 6 symboles (7-pilots_slot) * Qm*1 
          // x1 is set in lte_gold_generic
          // x2 should not reinitialized each subframe
          // x2 should be reinitialized according to 36.211 Sections 10.1.3.1 and 10.1.3.6
            if (ulsch_harq->round == 0)
          {
              // This is a new packet, so compute quantities regarding segmentation
              ulsch_harq->B = A+24;
              lte_segmentation_NB_IoT(NULL,
                                      NULL,
                                      ulsch_harq->B,
                                      &ulsch_harq->C,
                                      &ulsch_harq->Cplus,
                                      &ulsch_harq->Cminus,
                                      &ulsch_harq->Kplus,
                                      &ulsch_harq->Kminus,
                                      &ulsch_harq->F);
          }

          ulsch_harq->G  = G;
          H              = G ;
          Hprime         = H/Qm;
          Hpp            = Hprime;  // => Hprime = G/Qm
          Cmux           =  (7-pilots_slot) * N_UL_slots * Nsc_RU; 
          Rmux_prime     = Hpp/Cmux;
          // Clear "tag" interleaving matrix to allow for CQI/DATA identification
          memset(ytag,0,Cmux*Rmux_prime);
          memset(y,LTE_NULL_NB_IoT,Qm*Hpp);
          
          descrambling_NPUSCH_data_NB_IoT(fp,
                                          ulsch_llr,
                                          y,
                                          Qm,
                                          Cmux,
                                          rnti_tmp,
                                          ulsch_NB_IoT->Msg3_subframe,
                                          ulsch_NB_IoT->Msg3_frame);

          /// deinterleaving
          deinterleaving_NPUSCH_data_NB_IoT(ulsch_harq,y,G);

          ///  turbo decoding   NPUSCH data
          turbo_decoding_NB_IoT(eNB,
                                ulsch_NB_IoT,
                                proc,
                                npusch_format,
                                G,
                                rvdx,
                                Qm,
                                rx_frame,
                                rx_subframe);

      } else {   //////////////////// ACK ///////////////////

            int32_t      llr_msg5[16]; 
            int32_t      y_msg5[16];
            int16_t      *llrp2;
            int          l = 0;
            uint16_t     counter_ack = 0;   // ack counter for decision ack/nack
            
            llrp2 = (int16_t*)&pusch_vars->llr[0];

            for (l=0;l<16;l++) // putting reanl and im over 32 bits                   /// Add real and imaginary parts of BPSK constellation 
            {
                llr_msg5[l] = llrp2[l<<1] + llrp2[(l<<1)+1];
            }
            /////////////////////////////////////// descrambling + pre-decision /////////////////////////
            descrambling_NPUSCH_ack_NB_IoT(fp,
                                           y_msg5,
                                           llr_msg5,
                                           rnti_tmp,
                                           &counter_ack,
                                           ulsch_NB_IoT->Msg3_subframe,
                                           ulsch_NB_IoT->Msg3_frame);

            ///////////////////////////////// Decision ACK/NACK /////////////////////////////////////
            //printf("\n\n\n");
            if (counter_ack>8)   //hard decision
            {      
                  //fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,1);                               // indicate ACK to MAC
                  fill_rx_indication_NB_IoT(eNB,proc,npusch_format,1);
                  LOG_D(PHY,"  decoded ACK of DL Data (include MSG4)  \n");

            } else if (counter_ack<8) {     //hard decision

                  //fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,0);                              // indicate NAK to MAC
                  fill_rx_indication_NB_IoT(eNB,proc,npusch_format,0);
                  LOG_D(PHY,"  decoded ACK of DL Data (include MSG4)  \n"); 

            } else  {  //when equality (8 bits 0 vs 8 bits 1), soft decision
           
                  int32_t      counter_ack_soft = 0;

                  for (l=0;l<16;l++)
                  {
                        counter_ack_soft += y_msg5[l];   
                  }
                  if (counter_ack_soft>=0)            // decision 
                  {
                       // fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,1); // indicate ACK to MAC
                        fill_rx_indication_NB_IoT(eNB,proc,npusch_format,1);
                        LOG_I(PHY,"  decoded msg5 (soft): ACK  ");
                  } else {
                        //fill_crc_indication_NB_IoT(eNB,0,rx_frame,rx_subframe,0);   // indicate NAK to MAC
                        fill_rx_indication_NB_IoT(eNB,proc,npusch_format,0);
                        LOG_I(PHY,"  decoded msg5 (soft): NACK ");  
                  }
            }
            //printf("\n\n\n");  // end decision for ACK/NACK
      } 

      /////  if last sf of the word
      ulsch_NB_IoT->counter_repetitions--;

      if (npusch_format == 0)  // rvidx is used for data and not used otherwise
      {
          if(ulsch_NB_IoT->Msg3_flag == 1)   // case of msg3 
          {
              ulsch_harq->rvidx =  (ulsch_NB_IoT->counter_repetitions % 2)*2;        // rvidx toogle for new code word

          } else {                       /// other NPUSCH cases

              ulsch_harq->rvidx =  (((ulsch_harq->rvidx / 2) ^ 1) * 2);             // rvidx toogle for new code word
          }
      }

      if( (ulsch_NB_IoT->counter_sf == 1) && (ulsch_NB_IoT->counter_repetitions == 0) )
      {
          ulsch_NB_IoT->Msg3_active  = 0;
          ulsch_NB_IoT->Msg3_flag    = 0;
      } 

}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// generalization of RX procedures //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


uint8_t rx_ulsch_Gen_NB_IoT(PHY_VARS_eNB            *eNB,
                            eNB_rxtx_proc_t         *proc,
                            uint8_t                 eNB_id,                    // this is the effective sector id
                            uint8_t                 UE_id,
                            uint16_t                UL_RB_ID_NB_IoT,           // 22 , to be included in // to be replaced by NB_IoT_start ??     
                            uint8_t                 rx_subframe,               //  received subframe 
                            uint32_t                rx_frame)                  //  received frame 
{
      
      LTE_eNB_PUSCH       *pusch_vars   =  eNB->pusch_vars[UE_id];
      //LTE_eNB_COMMON      *common_vars  =  &eNB->common_vars;
      //NB_IoT_DL_FRAME_PARMS  *frame_parms  =  &eNB->frame_parms;
      LTE_DL_FRAME_PARMS     *fp  =  &eNB->frame_parms; 
      // NB_IoT_eNB_NULSCH_t    **ulsch_NB_IoT   =  &eNB->ulsch_NB_IoT[0];//[0][0]; 
      NB_IoT_eNB_NULSCH_t     *ulsch_NB_IoT     = eNB->ulsch_NB_IoT[0];
      NB_IoT_UL_eNB_HARQ_t    *ulsch_harq       = ulsch_NB_IoT->harq_process;

if(  (7 < ((rx_frame*10 + rx_subframe)%160)) && ( ((rx_frame*10 + rx_subframe)%160) < (8+6)) )
{ 
     return 0;

 } else {

  if (ulsch_NB_IoT->Msg3_active  == 1)    
  {    
      
      uint8_t      npusch_format         = ulsch_NB_IoT->npusch_format;           /// 0 or 1 -> format 1 or format 2         
      uint8_t      subcarrier_spacing    = ulsch_harq->subcarrier_spacing;        // can be set as fix value //values are OK // 0 (3.75 KHz) or 1 (15 KHz)
      uint16_t     I_sc                  = ulsch_harq->subcarrier_indication;     // Isc =0->18 , or 0->47 // format 2, 0->3 or 0->7
      uint16_t     I_mcs                 = ulsch_harq->mcs;                       // values 0->10
      uint16_t     Nsc_RU                = get_UL_N_ru_NB_IoT(I_mcs,ulsch_harq->resource_assignment,ulsch_NB_IoT->Msg3_flag);
      uint16_t     N_UL_slots            = get_UL_slots_per_RU_NB_IoT(subcarrier_spacing,I_sc,npusch_format)*Nsc_RU;           // N_UL_slots per word
      uint16_t     N_SF_per_word         = N_UL_slots/2;
      uint16_t     ul_sc_start           = 0;//nulsch->HARQ_ACK_resource

      if(ulsch_NB_IoT->flag_vars == 1)
      {
        ulsch_NB_IoT->counter_sf          = N_SF_per_word;
        ulsch_NB_IoT->counter_repetitions = get_UL_N_rep_NB_IoT(ulsch_harq->repetition_number);

        ulsch_NB_IoT->flag_vars = 0;
      }
      
      if(ulsch_NB_IoT->counter_sf == N_SF_per_word)                // initialization for scrambling
      {
          ulsch_NB_IoT->Msg3_subframe   =   rx_subframe;      // first received subframe 
          ulsch_NB_IoT->Msg3_frame      =   rx_frame;         // first received frame
      }

      uint8_t     pilot_pos1, pilot_pos2, pilots_slot;                                      // holds for npusch format 1, and 15 kHz subcarrier bandwidth
      uint32_t    l;
      uint32_t    rnti_tmp              = ulsch_NB_IoT->rnti;

      if( npusch_format == 1)    // format 2  // ACK part  
      { 
           ul_sc_start    =    get_UL_sc_ACK_NB_IoT(subcarrier_spacing,I_sc);
      } else {                   
           ul_sc_start    =    get_UL_sc_index_start_NB_IoT(subcarrier_spacing,I_sc,npusch_format); 
      }

      uint8_t      Qm     =    get_Qm_UL_NB_IoT(I_mcs,Nsc_RU,I_sc,ulsch_NB_IoT->Msg3_flag);              

      get_pilots_position(npusch_format, subcarrier_spacing, &pilot_pos1, &pilot_pos2, &pilots_slot);
      
                           ////////////////////// channel estimation per SF ////////////////////     
      UL_channel_estimation_NB_IoT(eNB, fp, UL_RB_ID_NB_IoT, Nsc_RU, pilot_pos1, pilot_pos2, ul_sc_start, Qm, N_SF_per_word, rx_subframe);
     
                           //////////////////////// Equalization  per SF ///////////////////////
      for (l=0; l<fp->symbols_per_tti; l++)
      { 
              ul_chequal_tmp_NB_IoT(pusch_vars->rxdataF_ext[eNB_id],
                                    pusch_vars->rxdataF_comp[eNB_id],
                                    pusch_vars->drs_ch_estimates[eNB_id],
                                    l%(fp->symbols_per_tti/2),               //symbol within slot 
                                    l/(fp->symbols_per_tti/2),
                                    fp);
      } 

                          ///////////////////// Rotation /////////////////
      for (l=0; l<fp->symbols_per_tti; l++)
      { 
           /// In case of 1 subcarrier: BPSK and QPSK should be rotated by pi/2 and pi/4, respectively 
            rotate_single_carrier_NB_IoT(eNB, 
                                         fp, 
                                         pusch_vars->rxdataF_comp[eNB_id], 
                                         UE_id, // UE ID
                                         l, 
                                         ulsch_NB_IoT->counter_sf,   //counter_msg,
                                         ul_sc_start,
                                         Qm,
                                         N_SF_per_word,
                                         npusch_format); // or data
      }
            ////////////////////// get LLR values per SF /////////////////////////
      get_llr_per_sf_NB_IoT(eNB,
                            fp,
                            npusch_format,
                            ulsch_NB_IoT->counter_sf,
                            N_SF_per_word,
                            pilot_pos1,
                            pilot_pos2,
                            ul_sc_start,
                            Nsc_RU);
      
      
      /////////////////////////////////////////////////  NPUSH DECOD //////////////////////////////////////
      if(ulsch_NB_IoT->counter_sf == 1)
      {   
          decode_NPUSCH_msg_NB_IoT(eNB,
                                   fp,
                                   proc,
                                   npusch_format,
                                   N_SF_per_word,
                                   Nsc_RU,
                                   N_UL_slots,
                                   Qm,
                                   pilots_slot,
                                   rnti_tmp,
                                   rx_subframe,
                                   rx_frame);
          
      } // NPUSH decode end

      /// update conter sf after every call
      ulsch_NB_IoT->counter_sf--;

      if( (ulsch_NB_IoT->counter_sf == 0) && (ulsch_NB_IoT->counter_repetitions > 0) )
      {
          ulsch_NB_IoT->counter_sf          = N_SF_per_word;
      }
      return 1;
      /////////////////////////////////////////END/////////////////////////////////////////////////////////////////////////////////////
  } else {
    return 0;     // create void function for NPUSCH ?
  }
}      
        
}