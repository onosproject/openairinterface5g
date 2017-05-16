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

/*! \file rrc_common.c
 * \brief rrc common procedures for eNB and UE
 * \author Navid Nikaein and Raymond Knopp
 * \date 2011 - 2014
 * \version 1.0
 * \company Eurecom
 * \email:  navid.nikaein@eurecom.fr and raymond.knopp@eurecom.fr
 */

#include "defs_nb_iot.h"
#include "extern.h"
#include "LAYER2/MAC/extern.h"
#include "COMMON/openair_defs.h"
#include "COMMON/platform_types.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "LAYER2/RLC/rlc.h"
#include "COMMON/mac_rrc_primitives.h"
#include "UTIL/LOG/log.h"
#include "asn1_msg.h"
#include "pdcp.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "rrc_eNB_UE_context.h"

#ifdef LOCALIZATION
#include <sys/time.h>
#endif

#define DEBUG_RRC 1
extern eNB_MAC_INST *eNB_mac_inst;
extern UE_MAC_INST *UE_mac_inst;

extern mui_t rrc_eNB_mui;

/*missed functions*/
//rrc_top_cleanup
//rrc_t310_expiration
//binary_search_init
//binary_search_float


//configure  BCCH & CCCH Logical Channels and associated rrc_buffers, configure associated SRBs
//called by openair_rrc_eNB_configuration_NB
//-----------------------------------------------------------------------------
void
openair_eNB_rrc_on_NB(
  const protocol_ctxt_t* const ctxt_pP
)
//-----------------------------------------------------------------------------
{
  unsigned short i;
  int            CC_id;

    LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" OPENAIR RRC-NB IN....\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
    for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
      rrc_config_buffer_NB (&eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SI, BCCH, 1);
      eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SI.Active = 1;
      rrc_config_buffer_NB (&eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0, CCCH, 1);
      eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Active = 1;
    }
    //no UE side
 }



//-----------------------------------------------------------------------------
void
rrc_config_buffer_NB(
  SRB_INFO_NB* Srb_info,
  uint8_t Lchan_type,
  uint8_t Role
)
//-----------------------------------------------------------------------------
{

  Srb_info->Rx_buffer.payload_size = 0;
  Srb_info->Tx_buffer.payload_size = 0;
}


//-----------------------------------------------------------------------------
//XXX NEW mplementation by Raymond: still used but no more called by MAC/main.c instead directly called by rrc_eNB_nb_iot.c
//XXX maybe this function is no more useful
int
rrc_init_global_param_NB(
  void
)
//-----------------------------------------------------------------------------
{

  rrc_rlc_register_rrc (rrc_data_ind, NULL); //register with rlc

  //XXX MP: most probably ALL of this stuff are no more needed (also the one not commented)

//  DCCH_LCHAN_DESC.transport_block_size = 4;
//  DCCH_LCHAN_DESC.max_transport_blocks = 16;
//  DCCH_LCHAN_DESC.Delay_class = 1;
//  DTCH_DL_LCHAN_DESC.transport_block_size = 52;
//  DTCH_DL_LCHAN_DESC.max_transport_blocks = 20;
//  DTCH_DL_LCHAN_DESC.Delay_class = 1;
//  DTCH_UL_LCHAN_DESC.transport_block_size = 52;
//  DTCH_UL_LCHAN_DESC.max_transport_blocks = 20;
//  DTCH_UL_LCHAN_DESC.Delay_class = 1;

  Rlc_info_am_config.rlc_mode = RLC_MODE_AM;
  Rlc_info_am_config.rlc.rlc_am_info_NB.max_retx_threshold = 50;
  Rlc_info_am_config.rlc.rlc_am_info_NB.t_poll_retransmit = 15;
  Rlc_info_am_config.rlc.rlc_am_info_NB.enableStatusReportSN_Gap = 0; //FIXME MP: should be disabled

#ifndef NO_RRM

  if (L3_xface_init_NB ()) {
    return (-1);
  }

#endif

  return 0;
}

//shoudl be changed?
#ifndef NO_RRM
//-----------------------------------------------------------------------------
int
L3_xface_init_NB(
  void
)
//-----------------------------------------------------------------------------
{

  int ret = 0;

#ifdef USER_MODE

  int sock;
  LOG_D(RRC, "[L3_XFACE] init de l'interface \n");

  if (open_socket (&S_rrc, RRC_RRM_SOCK_PATH, RRM_RRC_SOCK_PATH, 0) == -1) {
    return (-1);
  }

  if (S_rrc.s == -1) {
    return (-1);
  }

  socket_setnonblocking (S_rrc.s);
  msg ("Interface Connected... RRM-RRC\n");
  return 0;

#else

  ret=rtf_create(RRC2RRM_FIFO,32768);

  if (ret < 0) {
    msg("[openair][MAC][INIT] Cannot create RRC2RRM fifo %d (ERROR %d)\n",RRC2RRM_FIFO,ret);
    return(-1);
  } else {
    msg("[openair][MAC][INIT] Created RRC2RRM fifo %d\n",RRC2RRM_FIFO);
    rtf_reset(RRC2RRM_FIFO);
  }

  ret=rtf_create(RRM2RRC_FIFO,32768);

  if (ret < 0) {
    msg("[openair][MAC][INIT] Cannot create RRM2RRC fifo %d (ERROR %d)\n",RRM2RRC_FIFO,ret);
    return(-1);
  } else {
    msg("[openair][MAC][INIT] Created RRC2RRM fifo %d\n",RRM2RRC_FIFO);
    rtf_reset(RRM2RRC_FIFO);
  }

  return(0);

#endif
}
#endif

//------------------------------------------------------------------------------
//specialized function for the eNB initialization (NB-IoT)
//(OLD was called in MAC/main.c--> mac_top_init)(NEW is called in directly in "openair_rrc_eNB_configuration_NB")
void
openair_rrc_top_init_eNB_NB()//MP: XXX Raymond put this directly the definition on rrc_eNB.c file
//-----------------------------------------------------------------------------
{

  module_id_t         module_id;
  int                 CC_id;

  /* for no gcc warnings */
  (void)CC_id;

//not consider UE part

  if (NB_eNB_INST > 0) {
    eNB_rrc_inst_NB = (eNB_RRC_INST_NB*) malloc16(NB_eNB_INST*sizeof(eNB_RRC_INST_NB));
    memset (eNB_rrc_inst_NB, 0, NB_eNB_INST * sizeof(eNB_RRC_INST_NB));

//no CBA, no LOcalization, no MBMS flag

    LOG_D(RRC,
          "ALLOCATE %d Bytes for eNB_RRC_INST_NB @ %p\n", (unsigned int)(NB_eNB_INST*sizeof(eNB_RRC_INST_NB)), eNB_rrc_inst_NB);
  } else {
    eNB_rrc_inst_NB = NULL;
  }

#ifndef NO_RRM
#ifndef USER_MODE

  Header_buf=(char*)malloc16(sizeof(msg_head_t));
  Data=(char*)malloc16(2400);
  Header_read_idx=0;
  Data_read_idx=0;
  Header_size=sizeof(msg_head_t);

#endif //NO_RRM
  Data_to_read = 0;
#endif //USER_MODE
}

//-----------------------------------------------------------------------------
//XXX MP: most probably is not needed
RRC_status_t
rrc_rx_tx_NB(
  protocol_ctxt_t* const ctxt_pP,
  const uint8_t      enb_indexP,
  const int          CC_id
)
//-----------------------------------------------------------------------------
{
  //uint8_t        UE_id;
  int32_t        current_timestamp_ms, ref_timestamp_ms;
  struct timeval ts;
  struct rrc_eNB_ue_context_NB_s   *ue_context_p = NULL;
  struct rrc_eNB_ue_context_NB_s		  *ue_to_be_removed = NULL;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_IN);

  if(ctxt_pP->enb_flag == ENB_FLAG_NO) {  //is an UE
    // check timers

    if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_active == 1) {
      if ((UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_cnt % 10) == 0)
        LOG_D(RRC,
              "[UE %d][RAPROC] Frame %d T300 Count %d ms\n", ctxt_pP->module_id, ctxt_pP->frame, UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_cnt);

      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_cnt
          == T300[UE_rrc_inst[ctxt_pP->module_id].sib2[enb_indexP]->ue_TimersAndConstants.t300]) {
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_active = 0;
        // ALLOW CCCH to be used
        UE_rrc_inst[ctxt_pP->module_id].Srb0[enb_indexP].Tx_buffer.payload_size = 0;
        rrc_ue_generate_RRCConnectionRequest (ctxt_pP, enb_indexP);
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
        return (RRC_ConnSetup_failed);
      }

      UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T300_cnt++;
    }

    if ((UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].SIStatus&2)>0) {
      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].N310_cnt
          == N310[UE_rrc_inst[ctxt_pP->module_id].sib2[enb_indexP]->ue_TimersAndConstants.n310]) {
	LOG_I(RRC,"Activating T310\n");
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_active = 1;
      }
    } else { // in case we have not received SIB2 yet
      /*      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].N310_cnt == 100) {
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].N310_cnt = 0;

	}*/
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
      return RRC_OK;
    }

    if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_active == 1) {
      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].N311_cnt
          == N311[UE_rrc_inst[ctxt_pP->module_id].sib2[enb_indexP]->ue_TimersAndConstants.n311]) {
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_active = 0;
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].N311_cnt = 0;
      }

      if ((UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_cnt % 10) == 0) {
        LOG_D(RRC, "[UE %d] Frame %d T310 Count %d ms\n", ctxt_pP->module_id, ctxt_pP->frame, UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_cnt);
      }

      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_cnt    == T310[UE_rrc_inst[ctxt_pP->module_id].sib2[enb_indexP]->ue_TimersAndConstants.t310]) {
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_active = 0;
        rrc_t310_expiration (ctxt_pP, enb_indexP); //FIXME: maybe is required a NB_iot version of this function
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
	LOG_I(RRC,"Returning RRC_PHY_RESYNCH: T310 expired\n");
        return RRC_PHY_RESYNCH;
      }

      UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T310_cnt++;
    }

    if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_active==1) {
      if ((UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_cnt % 10) == 0)
        LOG_D(RRC,"[UE %d][RAPROC] Frame %d T304 Count %d ms\n",ctxt_pP->module_id,ctxt_pP->frame,
              UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_cnt);

      if (UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_cnt == 0) {
        UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_active = 0;
        UE_rrc_inst[ctxt_pP->module_id].HandoverInfoUe.measFlag = 1;
        LOG_E(RRC,"[UE %d] Handover failure..initiating connection re-establishment procedure... \n",
              ctxt_pP->module_id);
        //Implement 36.331, section 5.3.5.6 here
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
        return(RRC_Handover_failed);
      }

      UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].T304_cnt--;
    }

    // Layer 3 filtering of RRC measurements
    if (UE_rrc_inst[ctxt_pP->module_id].QuantityConfig[0] != NULL) {
      ue_meas_filtering(ctxt_pP,enb_indexP);
    }

    ue_measurement_report_triggering(ctxt_pP,enb_indexP);

    if (UE_rrc_inst[ctxt_pP->module_id].Info[0].handoverTarget > 0) {
      LOG_I(RRC,"[UE %d] Frame %d : RRC handover initiated\n", ctxt_pP->module_id, ctxt_pP->frame);
    }

    if((UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].State == RRC_HO_EXECUTION)   &&
        (UE_rrc_inst[ctxt_pP->module_id].HandoverInfoUe.targetCellId != 0xFF)) {
      UE_rrc_inst[ctxt_pP->module_id].Info[enb_indexP].State= RRC_IDLE;
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
      return(RRC_HO_STARTED);
    }

  } else { // eNB

	 //no handover in NB_IoT
    // counter, and get the value and aggregate

    // check for UL failure
    RB_FOREACH(ue_context_p, rrc_ue_tree_NB_s, &(eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head)) {
      if ((ctxt_pP->frame == 0) && (ctxt_pP->subframe==0)) {
	if (ue_context_p->ue_context.Initialue_identity_s_TMSI.presence == TRUE) {
	  LOG_I(RRC,"UE rnti %x:S-TMSI %x failure timer %d/20000\n",
		ue_context_p->ue_context.rnti,
		ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
		ue_context_p->ue_context.ul_failure_timer);
	}
	else {
	  LOG_I(RRC,"UE rnti %x failure timer %d/20000\n",
		ue_context_p->ue_context.rnti,
		ue_context_p->ue_context.ul_failure_timer);
	}
      }
      if (ue_context_p->ue_context.ul_failure_timer>0) {
	ue_context_p->ue_context.ul_failure_timer++;
	if (ue_context_p->ue_context.ul_failure_timer >= 20000) {
	  // remove UE after 20 seconds after MAC has indicated UL failure
	  LOG_I(RRC,"Removing UE %x instance\n",ue_context_p->ue_context.rnti);
	  ue_to_be_removed = ue_context_p;
	  break;
	}
      }
      if (ue_context_p->ue_context.ue_release_timer>0) {
	ue_context_p->ue_context.ue_release_timer++;
	if (ue_context_p->ue_context.ue_release_timer >=
	    ue_context_p->ue_context.ue_release_timer_thres) {
	  LOG_I(RRC,"Removing UE %x instance\n",ue_context_p->ue_context.rnti);
	  ue_to_be_removed = ue_context_p;
	  break;
	}
      }
    }
    if (ue_to_be_removed)
      rrc_eNB_free_UE(ctxt_pP->module_id,ue_to_be_removed); //FIXME: quite comple procedure, should use the same for NB-IoT?

//no localization in NB-IoT

    (void)ts; /* remove gcc warning "unused variable" */
    (void)ref_timestamp_ms; /* remove gcc warning "unused variable" */
    (void)current_timestamp_ms; /* remove gcc warning "unused variable" */
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX,VCD_FUNCTION_OUT);
  return (RRC_OK);
}



