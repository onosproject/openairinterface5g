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

/*! \file PHY/LTE_TRANSPORT/slss.c
 * \brief Functions to Generate and Received Sidelink PSS,SSS and PSBCH
 * \author R. Knopp
 * \date 2017
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */
#ifndef __LTE_TRANSPORT_SLSS__C__
#define __LTE_TRANSPORT_SLSS__C__
#include "PHY/defs.h"


void check_and_generate_slss(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {

  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);

  SLSS_t *slss = ue->slss;

  int tx_amp;

  if (slss->slmib == NULL) return;

  if ((((10*frame_tx) + subframe_tx)%40) != slss->SL_OffsetIndicator) return; 

  // here we have a transmission opportunity for SLSS
  ue->frame_parms.Nid_SL = slss->slss_id;

  // 6 PRBs => ceil(10*log10(6)) = 8 
  ue->tx_power_dBm[subframe_tx] = 8;
  ue->tx_total_RE[subframe_tx] = 72;

#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[subframe_tx],
		      ue->tx_power_max_dBm,
		      ue->frame_parms.N_RB_UL,
		      6);
#else
  tx_amp = AMP;
#endif  

  for (int aa=0; aa<ue->frame_parms.nb_antennas_tx; aa++) {
    memset(&ue->common_vars.txdataF[aa][subframe_tx*ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti],
           0,
	   ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti*sizeof(int32_t));
  }

  // PSS
  generate_slpss(ue->common_vars.txdataF,
                 tx_amp,
                 &ue->frame_parms,
                 1,
                 subframe_tx
		 );
  generate_slpss(ue->common_vars.txdataF,
                 tx_amp,
                 &ue->frame_parms,
                 2,
                 subframe_tx
		 );
  generate_slbch(ue->common_vars.txdataF,
                 tx_amp,
                 &ue->frame_parms,
		 2,
		 subframe_tx);

  ue->sl_chan = PSBCH;

  generate_drs_pusch(ue,
		     NULL,
		     0,
		     tx_amp,
		     subframe_tx,
		     (1+(ue->frame_parms.N_RB_UL/2))-3,
		     6,
                     0,
                     NULL,
                     0);
}
#endif
