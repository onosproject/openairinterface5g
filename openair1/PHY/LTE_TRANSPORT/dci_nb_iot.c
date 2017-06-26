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

  uint8_t *e_ptr,num_pdcch_symbols;
  int8_t L;
  uint32_t i, lprime;
  uint32_t gain_lin_QPSK,kprime,kprime_mod12,mprime,nsymb,symbol_offset,tti_offset;
  int16_t re_offset;
  uint8_t mi = get_mi(frame_parms,subframe);
  static uint8_t e[DCI_BITS_MAX];
  static int32_t yseq0[Msymb],yseq1[Msymb],wbar0[Msymb],wbar1[Msymb];

  int32_t *y[2];
  int32_t *wbar[2]; 

  int nushiftmod3 = frame_parms->nushift%3;

  int split_flag=0;

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

  wbar[0] = &wbar0[0];
  wbar[1] = &wbar1[0];
  y[0] = &yseq0[0];
  y[1] = &yseq1[0];

  // reset all bits to <NIL>, here we set <NIL> elements as 2
  // memset(e, 2, DCI_BITS_MAX);
  // here we interpret NIL as a random QPSK sequence. That makes power estimation easier.
  for (i=0; i<DCI_BITS_MAX; i++)
    e[i]=taus()&1;

  e_ptr = e;

  // generate DCIs in order of decreasing aggregation level, then common/ue spec
  // MAC is assumed to have ordered the UE spec DCI according to the RNTI-based randomization
  // there is only 2 aggregation (0 = 1, 1 = 2)
  for (L=1; L>=0; L--) {
    for (i=0; i<Num_dci; i++) {

      if (dci_alloc[i].L == (uint8_t)L) {

        if (dci_alloc[i].firstCCE>=0) {
          //encoding
          e_ptr = generate_dci0(
        		  dci_alloc[i].dci_pdu, //we should pass the two DCI pdu (if exist)
				  //second pdu
				  //aggregation level
        		  e+(72*dci_alloc[i].firstCCE),
				  dci_alloc[i].dci_length,
				  dci_alloc[i].L,
				  dci_alloc[i].rnti);

          //new NB-IoT
          npdcch_encoding_NB_IoT(
              dci_alloc[i].dci_pdu,
          frame_parms,
          npdcch, //see when function dci_top is called
          //no frame
          subframe
          //rm_stats, te_stats, i_stats
                      );


        }
      }
    }

  }

  // Scrambling
  //pdcch_scrambling(frame_parms,subframe,e,8*get_nquad(num_pdcch_symbols, frame_parms, mi));



  //NB-IoT--------------------------
  /*
   * switch(npdcch_start_index) (see mail)
   *
   * case 0
   * G = 304
   * case 1
   * G = 240
   * case 2
   * G = 224
   * case 3
   * G = 200
   */


  npdcch_scrambling_NB_IoT(
              frame_parms,
			  npdcch,
			  //G,
			  //q = nf mod 2 (TS 36.211 ch 10.2.3.1)  with nf = number of frame
			  //slot_id
                    );



  //NB-IoT
  npdcch_modulation_NB_IoT(
      txdataF,
      AMP,
      frame_parms,
      //no symbol
      //npdcch0???
      //RB_ID --> statically get from the higher layer (may included in the dl_frame params)
      );




  // This is the interleaving procedure defined in 36-211, first part of Section 6.8.5
  //pdcch_interleaving(frame_parms,&y[0],&wbar[0],num_pdcch_symbols,mi);
  //in NB-IoT the interleaving is done directly with the encoding procedure
  //there is no interleaving because we don't apply turbo coding


  // This is the REG allocation algorithm from 36-211, second part of Section 6.8.5
  // there is a function to do the resource mapping function
  //already done in the modulation in our NB-IoT implementaiton

  return 0;
}
