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

#include "PHY/defs.h"
#include "PHY/extern.h"
#include "extern.h"
#include "fs_1_4.h"
#include "prach625Hz.h"
#ifdef USER_MODE
#include <math.h>
#else
#include "rtai_math.h"
#endif
#include "PHY/sse_intrin.h"

short conjugate14[8]__attribute__((aligned(16))) = {-1,1,-1,1,-1,1,-1,1} ;
short conjugate14_2[8]__attribute__((aligned(16))) = {1,-1,1,-1,1,-1,1,-1} ;


void remove_1_4_fs(PHY_VARS_eNB *eNB,uint8_t slot)
{
  int32_t **rxdata=eNB->common_vars.rxdata[0];
  int32_t **rxdata_1_4fs=eNB->common_vars.rxdata[0];
  uint16_t len;
  uint32_t *fs1_4ptr;
#if defined(__x86_64__) || defined(__i386__)
  __m128i *rxptr128,*rxptr128_1_4fs,*fs1_4ptr128,fs1_4_2,mmtmp_re,mmtmp_im,mmtmp_re2,mmtmp_im2;
#elif defined(__arm__)
  int16x8_t *rxptr128,*fs1_4ptr128,*rxptr128_1_4fs;
  int32x4_t mmtmp_re,mmtmp_im;
  int32x4_t mmtmp0,mmtmp1;

#endif
  uint32_t slot_offset,slot_offset2;
  uint8_t aa;
  uint32_t i;
  LTE_DL_FRAME_PARMS *frame_parms=&eNB->frame_parms;

  switch (frame_parms->N_RB_UL) {

  case 6:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s6n_fs_1_4 : (uint32_t*)s6e_fs_1_4;
    break;

  case 15:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s15n_fs_1_4 : (uint32_t*)s15e_fs_1_4;
    break;

  case 25:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s25n_fs_1_4 : (uint32_t*)s25e_fs_1_4;
    break;

  case 50:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s50n_fs_1_4 : (uint32_t*)s50e_fs_1_4;
    break;

  case 75:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s75n_fs_1_4 : (uint32_t*)s75e_fs_1_4;
    break;

  case 100:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s100n_fs_1_4 : (uint32_t*)s100e_fs_1_4;
    break;

  default:
    fs1_4ptr = (frame_parms->Ncp==0) ? (uint32_t*)s25n_fs_1_4 : (uint32_t*)s25e_fs_1_4;
    break;
  }


  slot_offset = (uint32_t)slot * frame_parms->samples_per_tti/2-eNB->N_TA_offset;
  //slot_offset2 = (uint32_t)(slot&1) * frame_parms->samples_per_tti/2;

  len = frame_parms->samples_per_tti/2;

  for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {

#if defined(__x86_64__) || defined(__i386__)
    rxptr128        = (__m128i *)&rxdata[aa][slot_offset];
    rxptr128_1_4fs = (__m128i *)&rxdata_1_4fs[aa][slot_offset];
    fs1_4ptr128    = (__m128i *)fs1_4ptr;
#elif defined(__arm__)
    rxptr128        = (int16x8_t *)&rxdata[aa][slot_offset];
    rxptr128_1_4fs = (int16x8_t *)&rxdata_1_4fs[aa][slot_offset];
    fs1_4ptr128    = (int16x8_t *)fs1_4ptr;
#endif
    // remove 7.5 kHz + 1/4*fs

    //      if (((slot>>1)&1) == 0) { // apply the sinusoid from the table directly
    for (i=0; i<(len>>2); i++) {

#if defined(__x86_64__) || defined(__i386__)
      fs1_4_2 = _mm_sign_epi16(*fs1_4ptr128,*(__m128i*)&conjugate14_2[0]);
      mmtmp_re = _mm_madd_epi16(*rxptr128,fs1_4_2);
      // Real part of complex multiplication (note: fs1_4 signal is conjugated for this to work)
      mmtmp_im = _mm_shufflelo_epi16(fs1_4_2,_MM_SHUFFLE(2,3,0,1));
      mmtmp_im = _mm_shufflehi_epi16(mmtmp_im,_MM_SHUFFLE(2,3,0,1));
      mmtmp_im = _mm_sign_epi16(mmtmp_im,*(__m128i*)&conjugate14[0]);
      mmtmp_im = _mm_madd_epi16(mmtmp_im,rxptr128[0]);
      mmtmp_re = _mm_srai_epi32(mmtmp_re,15);
      mmtmp_im = _mm_srai_epi32(mmtmp_im,15);
      mmtmp_re2 = _mm_unpacklo_epi32(mmtmp_re,mmtmp_im);
      mmtmp_im2 = _mm_unpackhi_epi32(mmtmp_re,mmtmp_im);

      rxptr128_1_4fs[0] = _mm_packs_epi32(mmtmp_re2,mmtmp_im2);
      rxptr128++;
      rxptr128_1_4fs++;
      fs1_4ptr128++;

#elif defined(__arm__)

      fs1_4ptr128[0] = vmulq_s16(fs1_4ptr128[0],((int16x8_t*)conjugate14_2)[0]);
      mmtmp0 = vmull_s16(((int16x4_t*)rxptr128)[0],((int16x4_t*)fs1_4ptr128)[0]);
        //mmtmp0 = [Re(ch[0])Re(rx[0]) Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1]) Im(ch[1])Im(ch[1])]
      mmtmp1 = vmull_s16(((int16x4_t*)rxptr128)[1],((int16x4_t*)fs1_4ptr128)[1]);
        //mmtmp1 = [Re(ch[2])Re(rx[2]) Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3]) Im(ch[3])Im(ch[3])]
      mmtmp_re = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
                              vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));
        //mmtmp_re = [Re(ch[0])Re(rx[0])+Im(ch[0])Im(ch[0]) Re(ch[1])Re(rx[1])+Im(ch[1])Im(ch[1]) Re(ch[2])Re(rx[2])+Im(ch[2])Im(ch[2]) Re(ch[3])Re(rx[3])+Im(ch[3])Im(ch[3])]

      mmtmp0 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)rxptr128)[0],*(int16x4_t*)conjugate14_2)), ((int16x4_t*)fs1_4ptr128)[0]);
        //mmtmp0 = [-Im(ch[0])Re(rx[0]) Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1]) Re(ch[1])Im(rx[1])]
      mmtmp1 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)rxptr128)[1],*(int16x4_t*)conjugate14_2)), ((int16x4_t*)fs1_4ptr128)[1]);
        //mmtmp1 = [-Im(ch[2])Re(rx[2]) Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3]) Re(ch[3])Im(rx[3])]
      mmtmp_im = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
                              vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));
        //mmtmp_im = [-Im(ch[0])Re(rx[0])+Re(ch[0])Im(rx[0]) -Im(ch[1])Re(rx[1])+Re(ch[1])Im(rx[1]) -Im(ch[2])Re(rx[2])+Re(ch[2])Im(rx[2]) -Im(ch[3])Re(rx[3])+Re(ch[3])Im(rx[3])]

      rxptr128_1_4fs[0] = vcombine_s16(vmovn_s32(mmtmp_re),vmovn_s32(mmtmp_im));
      rxptr128_1_4fs++;
      rxptr128++;
      fs1_4ptr128++;


#endif
    }
  }
}

