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
* \email: raymond.knopp@eurecom.fr
* \note
* \warning
*/
/* file: initial_syncSL.c
   purpose: Initial synchronization procedures for Sidelink (SPSS/SSSS/PSBCH detection)
   author: raymond.knopp@eurecom.fr
   date: 13.05.2018
*/

//#include "defs.h"
#include "PHY/defs_UE.h"
#include "PHY/defs_common.h"
#include "PHY/phy_extern_ue.h"

extern int lte_sync_timeSL(PHY_VARS_UE *ue, int *ind, int64_t *lev, int64_t *avg);
extern int rx_slsss(PHY_VARS_UE *ue,int32_t *tot_metric,uint8_t *phase_max,int Nid2);
extern void generate_sl_grouphop(PHY_VARS_UE *ue);
extern int rx_psbch(PHY_VARS_UE *ue,int frame_rx,int subframe_rx);

int initial_syncSL(PHY_VARS_UE *ue) {

  int index;
  int64_t psslevel;
  int64_t avglevel;
  int frame,subframe;

  ue->rx_offsetSL = lte_sync_timeSL(ue,
				    &index,
				    &psslevel,
				    &avglevel);
  LOG_I(PHY,"index %d, psslevel %d dB avglevel %d dB => %d sample offset\n",
	 index,dB_fixed64((uint64_t)psslevel),dB_fixed64((uint64_t)avglevel),ue->rx_offsetSL);
  if (ue->rx_offsetSL >= 0) {
    int32_t sss_metric;
    uint8_t phase_max;
    rx_slsss(ue,&sss_metric,&phase_max,index);
    generate_sl_grouphop(ue);
  
    if (rx_psbch(ue,0,0) == -1) {
      ue->slbch_errors++;
      LOG_I(PHY,"SLPBCH not decoded\n");
/*
      write_output("rxsig0.m","rxs0",&ue->common_vars.rxdata_syncSL[0][0],40*ue->frame_parms.samples_per_tti,1,1);
      write_output("corr0.m","rxsync0",sync_corr_ue0,40*ue->frame_parms.samples_per_tti,1,6);
      write_output("corr1.m","rxsync1",sync_corr_ue1,40*ue->frame_parms.samples_per_tti,1,6);

      exit(-1); */ 
      return(-1); 
    }
    else {
    // send payload to RRC
      LOG_I(PHY,"Synchronization with SyncREF UE found, sending MIB-SL to RRC\n");
      ue_decode_si(ue->Mod_id,
		   0, // CC_id
		   0, // frame
		   0, // eNB_index
		   NULL, // pdu, NULL for MIB-SL
		   0,    // len, 0 for MIB-SL
		   &ue->slss_rx,
		   &frame,
		   &subframe);

      for (int i=0;i<RX_NB_TH;i++){
	ue->proc.proc_rxtx[i].frame_rx = frame;
        ue->proc.proc_rxtx[i].subframe_rx = subframe;
      }
      LOG_I(PHY,"RRC returns MIB-SL for frame %d, subframe %d\n",frame,subframe);		   
      return(0);
    }
  }
  else return (-1);
}
