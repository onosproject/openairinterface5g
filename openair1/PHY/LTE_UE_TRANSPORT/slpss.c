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
#include "PHY/defs_UE.h"
#include "PHY/phy_extern.h"

int generate_slpss(int32_t **txdataF,
		   short amp,
		   LTE_DL_FRAME_PARMS *frame_parms,
		   unsigned short symbol,
		   int subframe
		   )
{

  //unsigned int Nsymb;
  unsigned short k,m,a;
  uint8_t Nid2;
  short *primary_sync;


  Nid2 = frame_parms->Nid_SL / 168;
  AssertFatal(Nid2<2,"Nid2 %d >= 2\n",Nid2);

  switch (Nid2) {
  case 0:
    primary_sync = primary_synch0SL;
    break;

  case 1:
    primary_sync = primary_synch1SL;
    break;

  default:
    LOG_E(PHY,"[PSS] eNb_id has to be 0,1,2\n");
    return(-1);
  }
  // scale by sqrt(72/62)
  // note : we have to scale for TX power requirements too, beta_PSBCH !
  a = (amp*SQRT_18_OVER_32_Q15)>>(15-2);
  //printf("[PSS] amp=%d, a=%d\n",amp,a);

  LOG_D(PHY,"Generating PSS in subframe %d, symbol %d, amp %d (%d) => %p\n",
	subframe,symbol,a,amp,
	&((short*)txdataF[0])[subframe*frame_parms->samples_per_tti]);
  
  // The PSS occupies the inner 6 RBs, which start at
  k = frame_parms->ofdm_symbol_size-3*12+5;
  
  //printf("[PSS] k = %d\n",k);
  for (m=5; m<67; m++) {
    for (int aa=0;aa<frame_parms->nb_antennas_tx;aa++) {
      ((short*)txdataF[aa])[subframe*frame_parms->samples_per_tti + (2*(symbol*frame_parms->ofdm_symbol_size + k))] =
	(a * primary_sync[2*m]) >> 15;
      ((short*)txdataF[aa])[subframe*frame_parms->samples_per_tti + (2*(symbol*frame_parms->ofdm_symbol_size + k)) + 1] =
	(a * primary_sync[2*m+1]) >> 15;
    }    
    k+=1;
    
    if (k >= frame_parms->ofdm_symbol_size) {
      k-=frame_parms->ofdm_symbol_size;
    }
    
    
  }


  return(0);
}
