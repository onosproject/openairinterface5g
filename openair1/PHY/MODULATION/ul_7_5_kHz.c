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

#include "PHY/defs.h"
#include "PHY/extern.h"
#include "extern.h"
#include "kHz_7_5.h"
#include <math.h>
#include "PHY/sse_intrin.h"

short conjugate75[8]__attribute__((aligned(16))) = {-1,1,-1,1,-1,1,-1,1} ;
short conjugate75_2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;
short negate[8]__attribute__((aligned(16))) = {-1,-1,-1,-1,-1,-1,-1,-1};

#define print_shorts(s,x) printf("%s : %d %d %d %d %d %d %d %d\n",s,((int16_t*)x)[0],((int16_t*)x)[1],((int16_t*)x)[2],((int16_t*)x)[3],((int16_t*)x)[4],((int16_t*)x)[5],((int16_t*)x)[6],((int16_t*)x)[7])
#define print_dw(s,x) printf("%s : %d %d %d %d\n",s,((int32_t*)x)[0],((int32_t*)x)[1],((int32_t*)x)[2],((int32_t*)x)[3])

void apply_7_5_kHz(PHY_VARS_UE *ue,int32_t*txdata,uint8_t slot)
{


  uint16_t len;
  uint32_t *kHz7_5ptr;
#if defined(__x86_64__) || defined(__i386__)
  __m128i *txptr128,*kHz7_5ptr128,mmtmp_re,mmtmp_im,mmtmp_re2,mmtmp_im2;
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t *txptr128,*kHz7_5ptr128;
  int16x4x2_t mmtrn0, mmtrn1;
  int32x4_t mmtmp0,mmtmp1;
  int16x4_t tmpccgg0, tmpddhh0, tmpbafe0, tmpccgg1, tmpddhh1, tmpbafe1;
#endif
  uint32_t slot_offset;
  //   uint8_t aa;
  uint32_t i;
  LTE_DL_FRAME_PARMS *frame_parms=&ue->frame_parms;

  switch (frame_parms->N_RB_UL) {

  case 6:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s6n_kHz_7_5 : (uint32_t*)s6e_kHz_7_5;
    break;

  case 15:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s15n_kHz_7_5 : (uint32_t*)s15e_kHz_7_5;
    break;

  case 25:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s25n_kHz_7_5 : (uint32_t*)s25e_kHz_7_5;
    break;

  case 50:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s50n_kHz_7_5 : (uint32_t*)s50e_kHz_7_5;
    break;

  case 75:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s75n_kHz_7_5 : (uint32_t*)s75e_kHz_7_5;
    break;

  case 100:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s100n_kHz_7_5 : (uint32_t*)s100e_kHz_7_5;
    break;

  default:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s25n_kHz_7_5 : (uint32_t*)s25e_kHz_7_5;
    break;
  }

  slot_offset = (uint32_t)slot * frame_parms->samples_per_tti/2;
  len = frame_parms->samples_per_tti/2;

#if defined(__x86_64__) || defined(__i386__)
  txptr128 = (__m128i *)&txdata[slot_offset];
  kHz7_5ptr128 = (__m128i *)kHz7_5ptr;
#elif defined(__arm__) || defined(__aarch64__)
  txptr128 = (int16x8_t*)&txdata[slot_offset];
  kHz7_5ptr128 = (int16x8_t*)kHz7_5ptr;
#endif
  // apply 7.5 kHz
  for (i=0; i<(len>>2); i++) {
#if defined(__x86_64__) || defined(__i386__)
    mmtmp_re = _mm_madd_epi16(*txptr128,*kHz7_5ptr128);
    // Real part of complex multiplication (note: 7_5kHz signal is conjugated for this to work)
    mmtmp_im = _mm_shufflelo_epi16(*kHz7_5ptr128,_MM_SHUFFLE(2,3,0,1));
    mmtmp_im = _mm_shufflehi_epi16(mmtmp_im,_MM_SHUFFLE(2,3,0,1));
    mmtmp_im = _mm_sign_epi16(mmtmp_im,*(__m128i*)&conjugate75[0]);
    mmtmp_im = _mm_madd_epi16(mmtmp_im,txptr128[0]);
    mmtmp_re = _mm_srai_epi32(mmtmp_re,15);
    mmtmp_im = _mm_srai_epi32(mmtmp_im,15);
    mmtmp_re2 = _mm_unpacklo_epi32(mmtmp_re,mmtmp_im);
    mmtmp_im2 = _mm_unpackhi_epi32(mmtmp_re,mmtmp_im);

    txptr128[0] = _mm_packs_epi32(mmtmp_re2,mmtmp_im2);
    txptr128++;
    kHz7_5ptr128++;  
#elif defined(__arm__) || defined(__aarch64__)
    tmpddhh0 = ((int16x4_t*)kHz7_5ptr128)[0]; //c d g h
    tmpccgg0 = ((int16x4_t*)kHz7_5ptr128)[0]; //c d g h
    mmtrn0 = vtrn_s16(tmpccgg0, tmpddhh0); //c c g g mmtrn[0]    d d h h mmtrn[1]
    tmpddhh0 = vmul_s16(((int16x4_t*)&mmtrn0)[1], *(int16x4_t*)conjugate75); //-d d -h h
    tmpbafe0 = vrev32_s16(((int16x4_t*)txptr128)[0]); //b a f e
    mmtmp0 = vmull_s16(((int16x4_t*)txptr128)[0], ((int16x4_t*)&mmtrn0)[0]); //ac bc eg fg
    mmtmp0= vmlsl_s16(mmtmp0, tmpddhh0, tmpbafe0); //ac+bd bc-ad eg+fh fg-eh

    tmpddhh1 = ((int16x4_t*)kHz7_5ptr128)[1];
    tmpccgg1 = ((int16x4_t*)kHz7_5ptr128)[1];
    mmtrn1 = vtrn_s16(tmpccgg1, tmpddhh1);
    tmpddhh1 = vmul_s16(((int16x4_t*)&mmtrn1)[1], *(int16x4_t*)conjugate75);
    tmpbafe1 = vrev32_s16(((int16x4_t*)txptr128)[1]) ;
    mmtmp1 = vmull_s16(((int16x4_t*)txptr128)[1], ((int16x4_t*)&mmtrn1)[0]);
    mmtmp1 = vmlsl_s16(mmtmp1, tmpddhh1, tmpbafe1);

    txptr128[0] = vcombine_s16(vshrn_n_s32(mmtmp0, 15), vshrn_n_s32(mmtmp1, 15));
    /*if (i<4) {
        print_dw("mmtmp0",&mmtmp0);
        print_dw("mmtmp1",&mmtmp1);
        print_shorts("txp",txptr128);
    }*/
    txptr128++;
    kHz7_5ptr128++;
#endif
  }

  //}
}


void remove_7_5_kHz(RU_t *ru,uint8_t slot)
{


  int32_t **rxdata=ru->common.rxdata;
  int32_t **rxdata_7_5kHz=ru->common.rxdata_7_5kHz;
  uint16_t len;
  uint32_t *kHz7_5ptr;
#if defined(__x86_64__) || defined(__i386__)
  __m128i *rxptr128,*rxptr128_7_5kHz,*kHz7_5ptr128,kHz7_5_2,mmtmp_re,mmtmp_im,mmtmp_re2,mmtmp_im2;
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t *rxptr128,*kHz7_5ptr128,*rxptr128_7_5kHz;
  int16x4x2_t mmtrn0, mmtrn1;
  int32x4_t mmtmp0,mmtmp1;
  int16x4_t tmpccgg0, tmpddhh0, tmpbafe0, tmpccgg1, tmpddhh1, tmpbafe1;

#endif
  uint32_t slot_offset,slot_offset2;
  uint8_t aa;
  uint32_t i;
  LTE_DL_FRAME_PARMS *frame_parms=&ru->frame_parms;

  switch (frame_parms->N_RB_UL) {

  case 6:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s6n_kHz_7_5 : (uint32_t*)s6e_kHz_7_5;
    break;

  case 15:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s15n_kHz_7_5 : (uint32_t*)s15e_kHz_7_5;
    break;

  case 25:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s25n_kHz_7_5 : (uint32_t*)s25e_kHz_7_5;
    break;

  case 50:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s50n_kHz_7_5 : (uint32_t*)s50e_kHz_7_5;
    break;

  case 75:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s75n_kHz_7_5 : (uint32_t*)s75e_kHz_7_5;
    break;

  case 100:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s100n_kHz_7_5 : (uint32_t*)s100e_kHz_7_5;
    break;

  default:
    kHz7_5ptr = (frame_parms->Ncp_UL==0) ? (uint32_t*)s25n_kHz_7_5 : (uint32_t*)s25e_kHz_7_5;
    break;
  }


  slot_offset = ((uint32_t)slot * frame_parms->samples_per_tti/2)-ru->N_TA_offset;
  slot_offset2 = (uint32_t)(slot&1) * frame_parms->samples_per_tti/2;

  len = frame_parms->samples_per_tti/2;

  for (aa=0; aa<ru->nb_rx; aa++) {

#if defined(__x86_64__) || defined(__i386__)
    rxptr128        = (__m128i *)&rxdata[aa][slot_offset];
    rxptr128_7_5kHz = (__m128i *)&rxdata_7_5kHz[aa][slot_offset2];
    kHz7_5ptr128    = (__m128i *)kHz7_5ptr;
#elif defined(__arm__) || defined(__aarch64__)
    rxptr128        = (int16x8_t *)&rxdata[aa][slot_offset];
    rxptr128_7_5kHz = (int16x8_t *)&rxdata_7_5kHz[aa][slot_offset2];
    kHz7_5ptr128    = (int16x8_t *)kHz7_5ptr;
#endif
    // apply 7.5 kHz

    //      if (((slot>>1)&1) == 0) { // apply the sinusoid from the table directly
    for (i=0; i<(len>>2); i++) {

#if defined(__x86_64__) || defined(__i386__)
      kHz7_5_2 = _mm_sign_epi16(*kHz7_5ptr128,*(__m128i*)&conjugate75_2[0]);
      mmtmp_re = _mm_madd_epi16(*rxptr128,kHz7_5_2);
      // Real part of complex multiplication (note: 7_5kHz signal is conjugated for this to work)
      mmtmp_im = _mm_shufflelo_epi16(kHz7_5_2,_MM_SHUFFLE(2,3,0,1));
      mmtmp_im = _mm_shufflehi_epi16(mmtmp_im,_MM_SHUFFLE(2,3,0,1));
      mmtmp_im = _mm_sign_epi16(mmtmp_im,*(__m128i*)&conjugate75[0]);
      mmtmp_im = _mm_madd_epi16(mmtmp_im,rxptr128[0]);
      mmtmp_re = _mm_srai_epi32(mmtmp_re,15);
      mmtmp_im = _mm_srai_epi32(mmtmp_im,15);
      mmtmp_re2 = _mm_unpacklo_epi32(mmtmp_re,mmtmp_im);
      mmtmp_im2 = _mm_unpackhi_epi32(mmtmp_re,mmtmp_im);

      rxptr128_7_5kHz[0] = _mm_packs_epi32(mmtmp_re2,mmtmp_im2);
      rxptr128++;
      rxptr128_7_5kHz++;
      kHz7_5ptr128++;

#elif defined(__arm__) || defined(__aarch64__)

      kHz7_5ptr128[0] = vmulq_s16(kHz7_5ptr128[0],((int16x8_t*)conjugate75_2)[0]);
      tmpddhh0 = ((int16x4_t*)kHz7_5ptr128)[0]; //c d g h
      tmpccgg0 = ((int16x4_t*)kHz7_5ptr128)[0]; //c d g h
      mmtrn0 = vtrn_s16(tmpccgg0, tmpddhh0); //c c g g mmtrn[0]    d d h h mmtrn[1]
      tmpddhh0 = vmul_s16(((int16x4_t*)&mmtrn0)[1], *(int16x4_t*)conjugate75); //-d d -h h
      tmpbafe0 = vrev32_s16(((int16x4_t*)rxptr128)[0]); //b a f e
      mmtmp0 = vmull_s16(((int16x4_t*)rxptr128)[0], ((int16x4_t*)&mmtrn0)[0]); //ac bc eg fg
      mmtmp0= vmlsl_s16(mmtmp0, tmpddhh0, tmpbafe0); //ac+bd bc-ad eg+fh fg-eh

      tmpddhh1 = ((int16x4_t*)kHz7_5ptr128)[1];
      tmpccgg1 = ((int16x4_t*)kHz7_5ptr128)[1];
      mmtrn1 = vtrn_s16(tmpccgg1, tmpddhh1);
      tmpddhh1 = vmul_s16(((int16x4_t*)&mmtrn1)[1], *(int16x4_t*)conjugate75);
      tmpbafe1 = vrev32_s16(((int16x4_t*)rxptr128)[1]) ;
      mmtmp1 = vmull_s16(((int16x4_t*)rxptr128)[1], ((int16x4_t*)&mmtrn1)[0]);
      mmtmp1 = vmlsl_s16(mmtmp1, tmpddhh1, tmpbafe1);

      rxptr128_7_5kHz[0] = vcombine_s16(vshrn_n_s32(mmtmp0, 15), vshrn_n_s32(mmtmp1, 15));
      /*if (i<4) {
            print_dw("mmtmp0",&mmtmp0);
            print_dw("mmtmp1",&mmtmp1);
            print_shorts("txp",rxptr128_7_5kHz);
      }*/
      rxptr128_7_5kHz++;
      rxptr128++;
      kHz7_5ptr128++;
#endif


    }
  }
}

