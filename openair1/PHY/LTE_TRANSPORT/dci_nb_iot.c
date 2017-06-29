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
                         DCI_ALLOC_NB_t *dci_alloc,
                         int16_t amp,
                         NB_DL_FRAME_PARMS *fp,
                         //NB_IoT_eNB_NPDCCH_t npdcch,
                         int32_t **txdataF,
                         uint32_t subframe)
{


  int i,L, G;
  int npdcch_start_index;

  /* PARAMETERS may not needed
  **e_ptr : store the encoding result, and as a input to modulation
  *num_pdcch_symbols : to calculate the resource allocation for pdcch
  *L = aggregation level (there is 2 (at most) in NB-IoT) (Note this is not the real value but the index)
  *lprime,kprime,kprime_mod12,mprime,nsymb,symbol_offset,tti_offset,re_offset : used in the REG allocation
  *gain_lin_QPSK,yseq0[Msymb],yseq1[Msymb],*y[2] : used in the modulation
  *mi = used in interleaving
  *e = used to store the taus sequence (taus sequence is used to generate the first sequence for DCI) Turbo coding
  *wbar used in the interleaving and also REG allocation
  */

  //num_pdcch_symbols = get_num_pdcch_symbols(num_ue_spec_dci+num_common_dci,dci_alloc,frame_parms,subframe);


  // generate DCIs in order of decreasing aggregation level, then common/ue spec
  // MAC is assumed to have ordered the UE spec DCI according to the RNTI-based randomization???

  // Value of aggregation level (FAPI/NFAPI specs v.9.0 pag 221 value 1,2)
  for (L=2; L>=1; L--) {
    for (i=0; i<Num_dci; i++) {

    	//XXX should be checked how the scheduler store the aggregation level for NB-IoT (value 1-2 or 0-1)
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


  //NB-IoT scrambling
  /*
   *
   * TS 36.213 ch 16.6.1
   * npdcch_start_index  indicate the starting OFDM symbol for NPDCCH in the first slot of a subframe k ad is determined as follow:
   * - if eutracontrolregionsize is present (defined for in-band operating mode (mode 0,1 for FAPI specs))
   * 	npdcch_start_index = eutracontrolregionsize (value 1,2,3) [units in number of OFDM symbol]
   * -otherwise
   * 	npdcch_start_index = 0
   *
   *Depending on npddch_start_index then we define different values for G
   */

  //XXX the setting of this npdcch_start_index parameter should be done in the MAC
//  if(fp->operating_mode == 0 || fp->operating_mode == 1) //in-band operating mode
//  {
//	  npdcch_start_index = fp->control_region_size;
//  }
//  else
//  {
//	  npdcch_start_index = 0;
//  }

  for(int i = 0; i <Num_dci; i++)
  {

	  switch(dci_alloc[i].npdcch_start_symbol) //mail Bcom matthieu
	  {
  	  	  case 0:
  	  		  G = 304;
  		 	break;
  	  	  case 1:
  	  		  G = 240;
  	  		  break;
  	  	  case 2:
  	  		  G = 224;
  	  		  break;
  	  	  case 3:
  	  		  G =200;
  	  		  break;
  	  	  default:
  	  		  LOG_E (PHY,"npdcch_start_index has unwanted value\n");
  	  		  break;
	  }



//  	  	  // NB-IoT scrambling
//  	  	  npdcch_scrambling_NB_IoT(
//  	  	              frame_parms,
//  	  				  npdcch,
//  	  				  //G,
//  	  				  //q = nf mod 2 (TS 36.211 ch 10.2.3.1)  with nf = number of frame
//  	  				  //slot_id
//  	  	                    );


  }



//  //NB-IoT modulation
//  npdcch_modulation_NB_IoT(
//      txdataF,
//      AMP,
//      frame_parms,
//      //no symbol
//      //npdcch0???
//      //RB_ID --> statically get from the higher layer (may included in the dl_frame params)
//      );




  //in NB-IoT the interleaving is done directly with the encoding procedure
  //there is no interleaving because we don't apply turbo coding


  // This is the REG allocation algorithm from 36-211
  //already done in the modulation in our NB-IoT implementaiton??

  return 0;
}
