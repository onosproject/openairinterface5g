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
#include "PHY/defs_UE.h"
#include "SCHED_UE/sched_UE.h"
#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"

void check_and_generate_slss(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {

  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);

  SLSS_t *slss = ue->slss;

  int tx_amp;

  AssertFatal(slss!=NULL,"slss is null\n");
  
  LOG_D(PHY,"check_and_generate_slss: frame_tx %d, subframe_tx %d : slss->SL_offsetIndicator %d, slss->slmib_length %d\n",
	frame_tx,subframe_tx,slss->SL_OffsetIndicator, slss->slmib_length);
 
  if (ue->is_SynchRef == 0) return;
 
  if ((((10*frame_tx) + subframe_tx)%40) != slss->SL_OffsetIndicator) return; 

  if (slss->slmib_length == 0) return;

  // here we have a transmission opportunity for SLSS
  ue->frame_parms.Nid_SL = slss->slss_id;

  if (ue->SLghinitialized ==0) {
    generate_sl_grouphop(ue);
    ue->SLghinitialized=1;
  }

  // 6 PRBs => ceil(10*log10(6)) = 8 
  ue->tx_power_dBm[subframe_tx] = -6;
  ue->tx_total_RE[subframe_tx] = 72;

#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[subframe_tx],
		      ue->tx_power_max_dBm,
		      ue->frame_parms.N_RB_UL,
		      6);
#else
  tx_amp = AMP;
#endif  
  if (frame_tx == 0) LOG_I(PHY, "slss: ue->tx_power_dBm: %d, tx_amp: %d\n", ue->tx_power_dBm[subframe_tx], tx_amp);

  if (ue->generate_ul_signal[subframe_tx][0] == 0)
    for (int aa=0; aa<ue->frame_parms.nb_antennas_tx; aa++) {
      LOG_D(PHY,"%d.%d: clearing ul signal\n",frame_tx,subframe_tx);
      memset(&ue->common_vars.txdataF[aa][subframe_tx*ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti],
	     0,
	     ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti*sizeof(int32_t));
    }


  // PSS
  generate_slpss(ue->common_vars.txdataF,
                 tx_amp<<1,
                 &ue->frame_parms,
                 1,
                 subframe_tx
		 );  
 
  generate_slpss(ue->common_vars.txdataF,
                 tx_amp<<1,
                 &ue->frame_parms,
                 2,
                 subframe_tx
		 ); 
          
  generate_slbch(ue->common_vars.txdataF,
                 tx_amp,
                 &ue->frame_parms,
		 subframe_tx,
		 ue->slss->slmib);
 
  
  generate_slsss(ue->common_vars.txdataF,
		 subframe_tx,
                 tx_amp<<2,
                 &ue->frame_parms,
		 11);
  generate_slsss(ue->common_vars.txdataF,
		 subframe_tx,
                 tx_amp<<2,
                 &ue->frame_parms,
		 12);
  
  
  ue->sl_chan = PSBCH;
    
  generate_drs_pusch(ue,
		     NULL,
		     0,
		     tx_amp<<2,
		     subframe_tx,
		     (ue->frame_parms.N_RB_UL/2)-3,
		     6,
                     0,
                     NULL,
                     0);

  
 
  LOG_D(PHY,"%d.%d : SLSS nbrb %d, first rb %d\n",frame_tx,subframe_tx,6,(ue->frame_parms.N_RB_UL/2)-3);
 
  ue->generate_ul_signal[subframe_tx][0] = 1;
  ue->slss_generated = 1;
  
  LOG_D(PHY,"ULSCH (after slss) : signal F energy %d dB (txdataF %p) at SFN/SF: %d/%d \n",dB_fixed(signal_energy(&ue->common_vars.txdataF[0][subframe_tx*14*ue->frame_parms.ofdm_symbol_size],14*ue->frame_parms.ofdm_symbol_size)),&ue->common_vars.txdataF[0][subframe_tx*14*ue->frame_parms.ofdm_symbol_size], frame_tx, subframe_tx);

}
#endif
