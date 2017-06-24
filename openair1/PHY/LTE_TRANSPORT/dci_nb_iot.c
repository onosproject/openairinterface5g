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

/*! \file PHY/LTE_TRANSPORT/dci.c
* \brief Implements PDCCH physical channel TX/RX procedures (36.211) and DCI encoding/decoding (36.212/36.213). Current LTE compliance V8.6 2009-03.
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "SCHED/defs.h"
#include "SIMULATION/TOOLS/defs.h" // for taus 
#include "PHY/sse_intrin.h"

#include "assertions.h" 
#include "T.h"

uint8_t generate_dci_top_NB(uint8_t Num_dci,
                         DCI_ALLOC_t *dci_alloc,
                         uint32_t n_rnti,
                         int16_t amp,
                         LTE_DL_FRAME_PARMS *frame_parms,
                         //NB_IoT_eNB_NPDCCH_t npdcch,
                         int32_t **txdataF,
                         uint32_t subframe)
{


  int i,L;
  /*
  **e_ptr : store the encoding result, and as a input to modulation
  *num_pdcch_symbols : to calculate the resource allocation for pdcch
  *L = aggregation level (there is 2 (at most) in NB-IoT) (Note this is not the real value but the index)
  *lprime,kprime,kprime_mod12,mprime,nsymb,symbol_offset,tti_offset,re_offset : used in the REG allocation
  *gain_lin_QPSK,yseq0[Msymb],yseq1[Msymb],*y[2] : used in the modulation
  *mi = used in interleaving
  *e = used to store the taus sequence (taus sequence is used to generate the first sequence for DCI)
  *wbar used in the interleaving and also REG allocation
  */

  //num_pdcch_symbols = get_num_pdcch_symbols(num_ue_spec_dci+num_common_dci,dci_alloc,frame_parms,subframe);


  // generate DCIs in order of decreasing aggregation level, then common/ue spec
  // MAC is assumed to have ordered the UE spec DCI according to the RNTI-based randomization
  // there is only 2 aggregation (0 = 1, 1 = 2)
  for (L=1; L>=0; L--) {
    for (i=0; i<Num_dci; i++) {

      if (dci_alloc[i].L == (uint8_t)L) {

        if (dci_alloc[i].firstCCE>=0) {

          //NB-IoT encoding
          /*npdcch_encoding_NB_IoT(dci_alloc[i].dci_pdu,
                                 frame_parms,
                                 npdcch, //see when function dci_top is called
                                 //no frame
                                subframe
                                //rm_stats, te_stats, i_stats
                                );*/


        }
      }
    }

  }

  // Scrambling
  //pdcch_scrambling(frame_parms,subframe,e,8*get_nquad(num_pdcch_symbols, frame_parms, mi));



  //NB-IoT--------------------------
  /*
   * switch(npdcch_start_index)
   * case 0
   * G = 272
   * case 1
   * G = 248
   * case 2
   * G = 224
   * case 3
   * G = 200
   */

  /*
  // NB-IoT scrambling
  npdcch_scrambling_NB_IoT(
              frame_parms,
          npdcch,
          //G,
          //q = nf mod 2 (TS 36.211 ch 10.2.3.1)  with nf = number of frame
          //slot_id
                    );



  //NB-IoT modulation
  npdcch_modulation_NB_IoT(
      txdataF,
      AMP,
      frame_parms,
      //no symbol
      //npdcch0???
      //RB_ID --> statically get from the higher layer (may included in the dl_frame params)
      );*/




  // This is the interleaving procedure defined in 36-211, first part of Section 6.8.5
  //pdcch_interleaving(frame_parms,&y[0],&wbar[0],num_pdcch_symbols,mi);
  //in NB-IoT the interleaving is done directly with the encoding procedure
  //there is no interleaving because we don't apply turbo coding


  // This is the REG allocation algorithm from 36-211, second part of Section 6.8.5
  // there is a function to do the resource mapping function

  return 0;
}
