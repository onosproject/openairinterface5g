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

/*! \file PHY/LTE_TRANSPORT/pss.c
* \brief Top-level routines for generating primary synchronization signal (PSS) V8.6 2009-03
* \author F. Kaltenberger, O. Tonelli, R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: florian.kaltenberger@eurecom.fr, oscar.tonelli@yahoo.it,knopp@eurecom.fr
* \note
* \warning
*/
/* file: pss.c
   purpose: generate the primary synchronization signals of LTE
   author: florian.kaltenberger@eurecom.fr, oscar.tonelli@yahoo.it
   date: 21.10.2009
*/

//#include "defs.h"
#include "PHY/defs.h"
#include "PHY/extern.h"

#define PSBCH_A 40
#define PSBCH_E 1008 //12REs/PRB*6PRBs*7symbols*2 bits/RB

int generate_slbch(int32_t **txdataF,
		   short amp,
		   LTE_DL_FRAME_PARMS *frame_parms,
		   unsigned short symbol,
		   int subframe,
		   uint8_t *slmib) {
  
  uint8_t slbch_a[PSBCH_A>>3];
  uint32_t psbch_D;
  uint8_t psbch_d[96+(3*(16+PBCH_A))];
  uint8_t psbch_w[3*3*(16+PBCH_A)];
  uint8_t psbch_e[PSBCH_E];
  uint8_t RCC;
  int a;

  psbch_D    = 16+PSBCH_A;
  
  AssertFatal(frame_parms->Ncp==NORMAL,"Only Normal Prefix supported for Sidelink\n");
  AssertFatal(frame_parms->Ncp==NORMAL,"Only Normal Prefix supported for Sidelink\n");

	      
  for (int i=0; i<(PSBCH_A>>3); i++)
    slbch_a[(PSBCH_A>>3)-i-1] = slmib[i];

  ccodelte_encode(PSBCH_A,2,slbch_a,psbch_d+96,0);
  RCC = sub_block_interleaving_cc(psbch_D,psbch_d+96,psbch_w);
  
  lte_rate_matching_cc(RCC,PSBCH_E,psbch_w,psbch_e);

  pbch_scrambling(frame_parms,
		  psbch_e,
		  PSBCH_E,
		  1);
  int symb=0;
  uint8_t *eptr = psbch_e;
  int16_t *txptr;
  int k;

  a = (amp*SQRT_18_OVER_32_Q15)>>(15-2);
  int Nsymb=14;

  for (symb=0;symb<10;symb++) {
    k = frame_parms->ofdm_symbol_size-36;
    txptr = (int16_t*)&txdataF[0][subframe*Nsymb*frame_parms->ofdm_symbol_size+(symb*frame_parms->ofdm_symbol_size)];
    // first half (negative frequencies)
    for (int i=0;i<72;i++) {
      if (*eptr++ == 1) txptr[k] =-a;
      else              txptr[k] = a;
      
      k++;
    }    
    k=0;
    // second half (positive frequencies)
    for (int i=0;i<72;i++) {
      if (*eptr++ == 1) txptr[k] =-a;
      else              txptr[k] = a;
      
      k++;
    }
    if (symb==0) symb+=3;
  }

  // scale by sqrt(72/62)
  // note : we have to scale for TX power requirements too, beta_PSBCH !

  //  //printf("[PSS] amp=%d, a=%d\n",amp,a);
  
  
  return(0);
}
