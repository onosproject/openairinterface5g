/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file rrc_eNB.c
 * \brief rrc procedures for eNB
 * \author Navid Nikaein, Raymond Knopp and Michele Paffetti
 * \date 2011 - 2014
 * \version 1.0
 * \company Eurecom
 * \email: navid.nikaein@eurecom.fr, raymond.knopp@eurecom.fr, michele.paffetti@studio.unibo.it
 */
#define RRC_ENB
#define RRC_ENB_C

/*NB-IoT include files*/
#include "defs_nb_iot.h"
#include "RRC/LITE/MESSAGES/asn1_msg_nb_iot.h"
#include "RRCConnectionRequest-NB.h"
#include "RRCConnectionReestablishmentRequest-NB.h"
#include "ReestablishmentCause-NB-r13.h"
#include "UL-CCCH-Message-NB.h"
#include "DL-CCCH-Message-NB.h"
#include "UL-DCCH-Message-NB.h"
#include "DL-DCCH-Message-NB.h"

#include "extern.h"
#include "extern_nb_iot.h"
#include "assertions.h"
#include "asn1_conversions.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "LAYER2/RLC/rlc.h"
#include "LAYER2/MAC/proto.h"
#include "UTIL/LOG/log.h"
#include "COMMON/mac_rrc_primitives.h"
//#include "RRC/LITE/MESSAGES/asn1_msg.h"
//#include "RRCConnectionRequest.h"
//#include "RRCConnectionReestablishmentRequest.h"
//#include "ReestablishmentCause.h"
//#include "UL-CCCH-Message.h"
//#include "DL-CCCH-Message.h"
//#include "UL-DCCH-Message.h"
//#include "DL-DCCH-Message.h"
//#include "TDD-Config.h"
//#include "HandoverCommand.h"
#include "rlc.h"
#include "SIMULATION/ETH_TRANSPORT/extern.h"
#include "rrc_eNB_UE_context.h"
#include "platform_types.h"
#include "msc.h"
#include "T.h"

//FIXME: MP: once defined properly the MAC/defs_nb_iot this should be deleted
#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/defs_nb_iot.h"


#ifdef USER_MODE
#   include "RRC/NAS/nas_config.h"
#   include "RRC/NAS/rb_config.h"
#   include "OCG.h"
#   include "OCG_extern.h"
#endif

#if defined(ENABLE_SECURITY)
#   include "UTIL/OSA/osa_defs.h"
#endif

#if defined(ENABLE_USE_MME)
#   include "rrc_eNB_S1AP.h"
#   include "rrc_eNB_GTPV1U.h"
#   if defined(ENABLE_ITTI)
#   else
#      include "../../S1AP/s1ap_eNB.h"
#   endif
#endif

#include "pdcp.h"

#if defined(ENABLE_ITTI)
#   include "intertask_interface.h"
#endif

#include "SIMULATION/TOOLS/defs.h" // for taus

#if defined(FLEXRAN_AGENT_SB_IF)
#include "flexran_agent_extern.h"
#endif


#ifdef PHY_EMUL
extern EMULATION_VARS              *Emul_vars;
#endif
extern eNB_MAC_INST              *eNB_mac_inst;
extern UE_MAC_INST                 *UE_mac_inst;
#ifdef BIGPHYSAREA
extern void*                        bigphys_malloc(int);
#endif

extern uint16_t                     two_tier_hexagonal_cellIds[7];

mui_t                               rrc_eNB_mui = 0;





//-----------------------------------------------------------------------------
//actually is not used
void
rrc_eNB_generate_RRCConnectionRelease_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP
)
//-----------------------------------------------------------------------------
{

  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;

  T(T_ENB_RRC_CONNECTION_RELEASE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  memset(buffer, 0, RRC_BUF_SIZE);

  size = do_RRCConnectionRelease_NB(ctxt_pP->module_id,
		  	  	  	  	  	  	  	  buffer,
									  rrc_eNB_get_next_transaction_identifier_NB(ctxt_pP->module_id));
  // set release timer
  ue_context_pP->ue_context.ue_release_timer=1;
  // remove UE after 10 frames after RRCConnectionRelease is triggered
  ue_context_pP->ue_context.ue_release_timer_thres=100;

  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate RRCConnectionRelease-NB (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);

  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (rrcConnectionRelease-NB MUI %d) --->[PDCP][RB %u]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH1);//Through SRB1/or SRB1bis

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcConnectionRelease-NB UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  rrc_data_req(
	       ctxt_pP,
	       DCCH1,//Through SRB1/or SRB1bis
	       rrc_eNB_mui++,
	       SDU_CONFIRM_NO,
	       size,
	       buffer,
	       PDCP_TRANSMISSION_MODE_CONTROL);
}


//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionReestablishmentReject_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{ //exactly as in legacy LTE implementation

#ifdef RRC_MSG_PRINT
  int                                 cnt;
#endif

  T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_REJECT, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size =
    do_RRCConnectionReestablishmentReject_NB(ctxt_pP->module_id,
                          (uint8_t*) eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Payload);

#ifdef RRC_MSG_PRINT
  LOG_F(RRC,"[MSG] RRCConnectionReestablishmentReject\n");

  for (cnt = 0; cnt < eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size; cnt++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)eNB_rrc_inst_NB[ctxt_pP->module_id].Srb0.Tx_buffer.Payload)[cnt]);
  }

  LOG_F(RRC,"\n");
#endif

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Header,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size,
    MSC_AS_TIME_FMT" RRCConnectionReestablishmentReject UE %x size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP == NULL ? -1 : ue_context_pP->ue_context.rnti,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);

  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCConnectionReestablishmentReject (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_free_mem_UE_context_NB(
  const protocol_ctxt_t*               const ctxt_pP,
  struct rrc_eNB_ue_context_NB_s*         const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  LOG_T(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Clearing UE context 0x%p (free internal structs)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_context_pP);

  //no Scell in NB-IoT --> no DC (Dual Connectivity)

  if (ue_context_pP->ue_context.SRB_configList) {
    ASN_STRUCT_FREE(asn_DEF_SRB_ToAddModList_NB_r13, ue_context_pP->ue_context.SRB_configList);
    ue_context_pP->ue_context.SRB_configList = NULL;
  }

  if (ue_context_pP->ue_context.DRB_configList) {
    ASN_STRUCT_FREE(asn_DEF_DRB_ToAddModList_NB_r13, ue_context_pP->ue_context.DRB_configList);
    ue_context_pP->ue_context.DRB_configList = NULL;
  }

  memset(ue_context_pP->ue_context.DRB_active, 0, sizeof(ue_context_pP->ue_context.DRB_active));

  if (ue_context_pP->ue_context.physicalConfigDedicated_NB) {
    ASN_STRUCT_FREE(asn_DEF_PhysicalConfigDedicated_NB_r13, ue_context_pP->ue_context.physicalConfigDedicated_NB);
    ue_context_pP->ue_context.physicalConfigDedicated_NB = NULL;
  }


  if (ue_context_pP->ue_context.mac_MainConfig_NB) {
    ASN_STRUCT_FREE(asn_DEF_MAC_MainConfig_NB_r13, ue_context_pP->ue_context.mac_MainConfig_NB);
    ue_context_pP->ue_context.mac_MainConfig_NB = NULL;
  }

  //no sps in NB_IoT
  //no meas object
  //no report config
  //no quantity config
  //no meas gap config
  //no measConfig


}

//-----------------------------------------------------------------------------
// return the ue context if there is already an UE with ue_identityP, NULL otherwise
static struct rrc_eNB_ue_context_NB_s*
rrc_eNB_ue_context_random_exist_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const uint64_t               ue_identityP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_NB_s*        ue_context_p = NULL;
  //FIXME: there is a warning related to the new  type rrc_ue_tree_NB_s
  RB_FOREACH(ue_context_p, rrc_ue_tree_NB_s, &(eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head)) {
    if (ue_context_p->ue_context.random_ue_identity == ue_identityP)
      return ue_context_p;
  }
  return NULL;
}

//-----------------------------------------------------------------------------
// return a new ue context structure if ue_identityP, ctxt_pP->rnti not found in collection
static struct rrc_eNB_ue_context_NB_s*
rrc_eNB_get_next_free_ue_context_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const uint64_t               ue_identityP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_NB_s*        ue_context_p = NULL;
  ue_context_p = rrc_eNB_get_ue_context_NB(
                   &eNB_rrc_inst_NB[ctxt_pP->module_id],
                   ctxt_pP->rnti);

  if (ue_context_p == NULL) {
    RB_FOREACH(ue_context_p, rrc_ue_tree_NB_s, &(eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head)) {
      if (ue_context_p->ue_context.random_ue_identity == ue_identityP) {
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" Cannot create new UE context, already exist rand UE id 0x%"PRIx64", uid %u\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              ue_identityP,
              ue_context_p->local_uid);
        return NULL;
      }
    }
    ue_context_p = rrc_eNB_allocate_new_UE_context(&eNB_rrc_inst_NB[ctxt_pP->module_id]);

    if (ue_context_p == NULL) {
      LOG_E(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" Cannot create new UE context, no memory\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
      return NULL;
    }

    ue_context_p->ue_id_rnti                    = ctxt_pP->rnti; // here ue_id_rnti is just a key, may be something else
    ue_context_p->ue_context.rnti               = ctxt_pP->rnti; // yes duplicate, 1 may be removed
    ue_context_p->ue_context.random_ue_identity = ue_identityP;
    RB_INSERT(rrc_ue_tree_NB_s, &eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head, ue_context_p);
    LOG_D(RRC,
          PROTOCOL_RRC_CTXT_UE_FMT" Created new UE context uid %u\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          ue_context_p->local_uid);
    return ue_context_p;

  } else {
    LOG_E(RRC,
          PROTOCOL_RRC_CTXT_UE_FMT" Cannot create new UE context, already exist\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    return NULL;
  }
}

//-----------------------------------------------------------------------------
// return the ue context if there is already an UE with the same S-TMSI(MMEC+M-TMSI), NULL otherwise
static struct rrc_eNB_ue_context_NB_s*
rrc_eNB_ue_context_stmsi_exist_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const mme_code_t             mme_codeP,
  const m_tmsi_t               m_tmsiP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_NB_s*        ue_context_p = NULL;
  RB_FOREACH(ue_context_p, rrc_ue_tree_NB_s, &(eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head)) {
    LOG_I(RRC,"checking for UE S-TMSI %x, mme %x (%p): rnti %x",
	  m_tmsiP, mme_codeP, ue_context_p,
	  ue_context_p->ue_context.rnti);
    if (ue_context_p->ue_context.Initialue_identity_s_TMSI.presence == TRUE) {
      printf("=> S-TMSI %x, MME %x\n",
	    ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
	    ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code);
      if (ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi == m_tmsiP)
        if (ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code == mme_codeP)
          return ue_context_p;
    }
    else
      printf("\n");

  }
  return NULL;
}

//-----------------------------------------------------------------------------
uint8_t
rrc_eNB_get_next_transaction_identifier_NB(
  module_id_t enb_mod_idP
)
//-----------------------------------------------------------------------------
{
  static uint8_t                      rrc_transaction_identifier[NUMBER_OF_eNB_MAX];
  rrc_transaction_identifier[enb_mod_idP] = (rrc_transaction_identifier[enb_mod_idP] + 1) % RRC_TRANSACTION_IDENTIFIER_NUMBER;
  LOG_T(RRC,"generated xid is %d\n",rrc_transaction_identifier[enb_mod_idP]);
  return rrc_transaction_identifier[enb_mod_idP];
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionReject_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{
#ifdef RRC_MSG_PRINT
  int                                 cnt;
#endif

  T(T_ENB_RRC_CONNECTION_REJECT, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size =
    do_RRCConnectionReject_NB(ctxt_pP->module_id,
                          (uint8_t*) eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Payload);

#ifdef RRC_MSG_PRINT
  LOG_F(RRC,"[MSG] RRCConnectionReject-NB\n");

  for (cnt = 0; cnt < eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size; cnt++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)eNB_rrc_inst_NB[ctxt_pP->module_id].Srb0.Tx_buffer.Payload)[cnt]);
  }

  LOG_F(RRC,"\n");
#endif

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Header,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size,
    MSC_AS_TIME_FMT" RRCConnectionReject-NB UE %x size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP == NULL ? -1 : ue_context_pP->ue_context.rnti,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);

  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCConnectionReject-NB (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionSetup_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{

	//connection setup involve the establishment of SRB1 and SRB1bis (but srb1bis is established implicitly)
	//XXX: this message should go through SRB0 to see this--> uper_encode
	//XXX: they are assuming that 2 RLC-AM entities are used for SRB1 and SRB1bis



//  SRB_ToAddMod_NB_r13_t                     *SRB1_config; //may not needed now
//  LogicalChannelConfig_NB_r13_t             *SRB1_logicalChannelConfig;

  SRB_ToAddModList_NB_r13_t                **SRB_configList; //for both SRB1 and SRB1bis
  SRB_ToAddMod_NB_r13_t						*SRB1bis_config;
  LogicalChannelConfig_NB_r13_t				*SRB1bis_logicalChannelConfig;

  int  										cnt;

  //FIXME: MP: structure to be chaged at phy layer
  LTE_DL_FRAME_PARMS *fp = mac_xface->get_lte_frame_parms(ctxt_pP->module_id,CC_id);

  T(T_ENB_RRC_CONNECTION_SETUP, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  SRB_configList = &ue_context_pP->ue_context.SRB_configList;

  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size =
    do_RRCConnectionSetup_NB(ctxt_pP,
                          ue_context_pP,
                          CC_id,
                          (uint8_t*) eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Payload,
                          rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),
                          fp,
                          SRB_configList, //MP:should contain only SRb1bis for the moment
                          &ue_context_pP->ue_context.physicalConfigDedicated_NB);

#ifdef RRC_MSG_PRINT
  LOG_F(RRC,"[MSG] RRC Connection Setup\n");

  for (cnt = 0; cnt < eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size; cnt++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)eNB_rrc_inst_NB[ctxt_pP->module_id].Srb0.Tx_buffer.Payload)[cnt]);
  }

  LOG_F(RRC,"\n");
  //////////////////////////////////
#endif

  // configure SRB1bis, PhysicalConfigDedicated, MAC_MainConfig for UE

  if (*SRB_configList != NULL) {

	  //XXX MP: srb_identity is no more used in NB-IoT
	  //XXX MP: the list should contain just one element
	  for(cnt = 0; cnt < (*SRB_configList)->list.count; cnt++){

        SRB1bis_config = (*SRB_configList)->list.array[cnt];

        if (SRB1bis_config->logicalChannelConfig_r13) {
          if (SRB1bis_config->logicalChannelConfig_r13->present == SRB_ToAddMod_NB_r13__logicalChannelConfig_r13_PR_explicitValue)
          {
            SRB1bis_logicalChannelConfig = &SRB1bis_config->logicalChannelConfig_r13->choice.explicitValue;
          }
          else {
        	  //XXX MP: warning saying "assignment from incompatible pointer type"
            SRB1bis_logicalChannelConfig = &SRB1bis_NB_logicalChannelConfig_defaultValue;
          	  }
        } else {
          SRB1bis_logicalChannelConfig = &SRB1bis_NB_logicalChannelConfig_defaultValue;
        }

        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (SRB1bis/SRB1) ---> MAC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));


        //FIXME: Connection Setup could be called also after security activation

        //configure the MAC for SRB1bis/SRb1???? (but in principle this configuration should be not LCID dependent)
        NB_rrc_mac_config_req_eNB(
          ctxt_pP->module_id,
          ue_context_pP->ue_context.primaryCC_id,
          ue_context_pP->ue_context.rnti,
          0, //physCellID
		  0, //p_eNB
		  0, //Ncp
		  0, //eutraband
		  (NS_PmaxList_NB_r13_t*) NULL,
		  (MultiBandInfoList_NB_r13_t*) NULL,
		  (DL_Bitmap_NB_r13_t*) NULL,
		  (long*) NULL,
		  (long*)NULL,
		  (uint8_t*) NULL, //SIWindowSize
		  (uint16_t*)NULL, //SIperiod
		  0, //dl_carrierFrequency
		  0, //ul_carrierFrequency
		  (BCCH_BCH_Message_NB_t*) NULL,
          (RadioResourceConfigCommonSIB_NB_r13_t *) NULL,
          (PhysicalConfigDedicated_NB_r13_t*) ue_context_pP->ue_context.physicalConfigDedicated_NB,
          ue_context_pP->ue_context.mac_MainConfig_NB, //FIXME WHo have set this??
          DCCH0, //LCID  = 3 of SRB1bis //XXX for the moment have been defined in MAC/defs.h but should move to MAC/defs_nb_iot.h
          SRB1bis_logicalChannelConfig
		  );
        break;
    }
  }

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.Header, // LG WARNING
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size,
    MSC_AS_TIME_FMT" RRCConnectionSetup-NB UE %x size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);


  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCConnectionSetup-NB (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Srb0.Tx_buffer.payload_size);

  // activate release timer, if RRCSetupComplete-NB not received after 10 frames, remove UE
  ue_context_pP->ue_context.ue_release_timer=1;
  // remove UE after 10 frames after RRCConnectionRelease-NB is triggered
  ue_context_pP->ue_context.ue_release_timer_thres=100;

}

//-----------------------------------------------------------------------------
void
rrc_eNB_process_RRCConnectionReconfigurationComplete_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*        ue_context_pP,
  const uint8_t xid //transaction identifier
)
//-----------------------------------------------------------------------------
{
  int                                 i, drb_id;

#ifdef PDCP_USE_NETLINK
  int                                 oip_ifup = 0;
  int                                 dest_ip_offset = 0;
  module_id_t                         ue_module_id   = -1;
  /* avoid gcc warnings */
  (void)oip_ifup;
  (void)dest_ip_offset;
  (void)ue_module_id;
#endif

  uint8_t                            *kRRCenc = NULL;
  uint8_t                            *kRRCint = NULL;
  uint8_t                            *kUPenc = NULL;


  DRB_ToAddModList_NB_r13_t*                 DRB_configList2 = ue_context_pP->ue_context.DRB_configList2[xid];
  SRB_ToAddModList_NB_r13_t*                 SRB_configList2 = ue_context_pP->ue_context.SRB_configList2[xid];

  T(T_ENB_RRC_CONNECTION_RECONFIGURATION_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#if defined(ENABLE_SECURITY)

  /* Derive the keys from kenb */
  if (DRB_configList2 != NULL) {
    derive_key_up_enc(ue_context_pP->ue_context.ciphering_algorithm,
                      ue_context_pP->ue_context.kenb, &kUPenc);
  }

  derive_key_rrc_enc(ue_context_pP->ue_context.ciphering_algorithm,
                     ue_context_pP->ue_context.kenb, &kRRCenc);
  derive_key_rrc_int(ue_context_pP->ue_context.integrity_algorithm,
                     ue_context_pP->ue_context.kenb, &kRRCint);

#endif

  //no RAL

  // Refresh SRBs/DRBs for PDCP
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_PDCP_ENB,
    NULL,
    0,
    MSC_AS_TIME_FMT" CONFIG_REQ UE %x DRB (security unchanged)",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti);

  NB_rrc_pdcp_config_asn1_req(
    ctxt_pP,
    SRB_configList2,
    DRB_configList2,
    (DRB_ToReleaseList_NB_r13_t *) NULL,
    0xff, //security mode already configured during the securitymodecommand
    kRRCenc,
    kRRCint,
    kUPenc,
	NULL,
	DCCH1); //SRB1

  // Refresh SRBs/DRBs for RLC
  NB_rrc_rlc_config_asn1_req(
    ctxt_pP,
    SRB_configList2,
    DRB_configList2,
    (DRB_ToReleaseList_NB_r13_t *) NULL,
	SRB1BIS_FLAG_NO
  );

  // set the SRB active in Ue context

  if (SRB_configList2 != NULL) {
	  //MP: SRB_ToAddModList_NB_r13_t  size = 1 TS 36.331 V14.2.1 pag 615
	  //MP: may for loop is not needed --> should stop at first iteration

    for (i = 0; (i < SRB_configList2->list.count) && (i < 3); i++) {
    	//no need of checking the srb-Identity
    	//we should have only SRB1 in the list
    	ue_context_pP->ue_context.Srb1.Active=1;
	    ue_context_pP->ue_context.Srb1.Srb_info.Srb_id=DCCH1; //LCHAN_iD
	     LOG_I(RRC,"[eNB %d] Frame  %d CC %d : SRB1 is now active\n",
		ctxt_pP->module_id,
		ctxt_pP->frame,
		ue_context_pP->ue_context.primaryCC_id);
    }
  }


  // Loop through DRBs and establish if necessary

  if (DRB_configList2 != NULL) {
    for (i = 0; (i < DRB_configList2->list.count) && (i<2); i++) {  // num maxDRB-NB = 2
      if (DRB_configList2->list.array[i]) {
	drb_id = (int)DRB_configList2->list.array[i]->drb_Identity_r13;
        LOG_I(RRC,
              "[eNB %d] Frame  %d : Logical Channel UL-DCCH, Received RRCConnectionReconfigurationComplete-NB from UE rnti %x, reconfiguring DRB %d/LCID %d\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ctxt_pP->rnti,
              (int)DRB_configList2->list.array[i]->drb_Identity_r13, //x
              (int)*DRB_configList2->list.array[i]->logicalChannelIdentity_r13);//x+3
        // for pre-ci tests
        LOG_I(RRC,
              "[eNB %d] Frame  %d : Logical Channel UL-DCCH, Received RRCConnectionReconfigurationComplete-NB from UE %u, reconfiguring DRB %d/LCID %d\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              oai_emulation.info.eNB_ue_local_uid_to_ue_module_id[ctxt_pP->module_id][ue_context_pP->local_uid],
              (int)DRB_configList2->list.array[i]->drb_Identity_r13,
              (int)*DRB_configList2->list.array[i]->logicalChannelIdentity_r13);

        if (ue_context_pP->ue_context.DRB_active[drb_id] == 0) {

          ue_context_pP->ue_context.DRB_active[drb_id] = 1;

          LOG_D(RRC,
                "[eNB %d] Frame %d: Establish RLC UM Bidirectional??, DRB %d Active\n",
                ctxt_pP->module_id, ctxt_pP->frame, (int)DRB_configList2->list.array[i]->drb_Identity_r13);
          //FIXME: RLC UM not exist in NB-IoT and what means the stuff belows?
#if  defined(PDCP_USE_NETLINK) && !defined(LINK_ENB_PDCP_TO_GTPV1U)
          // can mean also IPV6 since ether -> ipv6 autoconf
#   if !defined(OAI_NW_DRIVER_TYPE_ETHERNET) && !defined(EXMIMO) && !defined(OAI_USRP) && !defined(OAI_BLADERF) && !defined(ETHERNET)
          LOG_I(OIP, "[eNB %d] trying to bring up the OAI interface oai%d\n",
                ctxt_pP->module_id,
                ctxt_pP->module_id);
          oip_ifup = nas_config(
                       ctxt_pP->module_id,   // interface index
                       ctxt_pP->module_id + 1,   // thrid octet
                       ctxt_pP->module_id + 1);  // fourth octet

          if (oip_ifup == 0) {    // interface is up --> send a config the DRB
#      ifdef OAI_EMU
            oai_emulation.info.oai_ifup[ctxt_pP->module_id] = 1;
            dest_ip_offset = NB_eNB_INST;
#      else
            dest_ip_offset = 8;
#      endif
            LOG_I(OIP,
                  "[eNB %d] Config the oai%d to send/receive pkt on DRB %d to/from the protocol stack\n",
                  ctxt_pP->module_id, ctxt_pP->module_id,
                  (ue_context_pP->local_uid * maxDRB_NB_r13) + DRB_configList2->list.array[i]->drb_Identity_r13);
            ue_module_id = oai_emulation.info.eNB_ue_local_uid_to_ue_module_id[ctxt_pP->module_id][ue_context_pP->local_uid];
            rb_conf_ipv4(0, //add
                         ue_module_id,  //cx
                         ctxt_pP->module_id,    //inst
                         (ue_module_id * maxDRB_NB_r13) + DRB_configList2->list.array[i]->drb_Identity_r13, // RB
                         0,    //dscp
                         ipv4_address(ctxt_pP->module_id + 1, ctxt_pP->module_id + 1),  //saddr
                         ipv4_address(ctxt_pP->module_id + 1, dest_ip_offset + ue_module_id + 1));  //daddr
            LOG_D(RRC, "[eNB %d] State = Attached (UE rnti %x module id %u)\n",
                  ctxt_pP->module_id, ue_context_pP->ue_context.rnti, ue_module_id);
          }

#   else
#      ifdef OAI_EMU
          oai_emulation.info.oai_ifup[ctxt_pP->module_id] = 1;
#      endif
#   endif
#endif

          LOG_D(RRC,
                PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_eNB\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

          if (DRB_configList2->list.array[i]->logicalChannelIdentity_r13) {
            DRB2LCHAN_NB[i] = (uint8_t) * DRB_configList2->list.array[i]->logicalChannelIdentity_r13; //set in RRCConnectionReconfiguration (x+3)
          }

          //for each DRB I send a rrc_mac_config_req
          NB_rrc_mac_config_req_eNB(
            ctxt_pP->module_id,
            ue_context_pP->ue_context.primaryCC_id,
            ue_context_pP->ue_context.rnti,
            0,//physCellId --> this parameters could be set to 0 since no MIB is present
			0,// p_eNB
			0, //Ncp
			0, //eutra_band
			(struct NS_PmaxList_NB_r13*) NULL,
			(struct MultiBandInfoList_NB_r13*) NULL,
			(struct DL_Bitmap_NB_r13*) NULL,
			(long*)NULL,//eutraControlRegionSize
			(long*)NULL, //nrs_CRS_PowerOffset
			(uint8_t*)NULL,
			(uint16_t*)NULL,
			0, //dl_CarrierFreq
			0, //ul_CarrierFreq
			(BCCH_BCH_Message_NB_t*)NULL,
			(RadioResourceConfigCommonSIB_NB_r13_t*)NULL,
            ue_context_pP->ue_context.physicalConfigDedicated_NB,
            ue_context_pP->ue_context.mac_MainConfig_NB,
            DRB2LCHAN_NB[i], //over the logical channel id of the DRB (>=4)
            DRB_configList2->list.array[i]->logicalChannelConfig_r13
			);

        } else {


        	/*MP:
        	 * This part of the code is wrong since a the list that we manage are ADD/MOD lists and not RELEASE lists
        	 * so it means that nothing should be deleted but only modified!!
        	 */


        	//remove LCHAN from MAC/PHY
        		  //MP: DRB_active could take values different from 0 and 1 for errors?
          if (ue_context_pP->ue_context.DRB_active[drb_id] == 1) {
            // DRB has just been removed (by UE) so remove RLC + PDCP for DRB
        	//FIXME: MP: why not from PDCP?

            NB_rrc_rlc_config_req(
              ctxt_pP,
              SRB_FLAG_NO,
              CONFIG_ACTION_REMOVE, //XXXmodify
              DRB2LCHAN_NB[i],
              Rlc_info_am); //XXX: should send a RLC_info_am--> RRC/vars_nb_iot.h //rlc_infoP
          }

          ue_context_pP->ue_context.DRB_active[drb_id] = 0;
          LOG_D(RRC,
                PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_eNB\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

          NB_rrc_mac_config_req_eNB(
                      ctxt_pP->module_id,
                      ue_context_pP->ue_context.primaryCC_id,
                      ue_context_pP->ue_context.rnti,
                      0,//physCellId --> this parameters could be set to 0 since no MIB is present
					  0,// p_eNB
					  0, //Ncp
					  0, //eutra_band
					  (struct NS_PmaxList_NB_r13*) NULL,
					  (struct MultiBandInfoList_NB_r13*) NULL,
					  (struct DL_Bitmap_NB_r13*) NULL,
					  (long*)NULL,//eutraControlRegionSize
					  (long*)NULL, //nrs_CRS_PowerOffset
					  (uint8_t*)NULL,
					  (uint16_t*)NULL,
					  0, //dl_CarrierFreq
					  0, //ul_CarrierFreq
					  (BCCH_BCH_Message_NB_t*)NULL,
					  (RadioResourceConfigCommonSIB_NB_r13_t*)NULL,
                      ue_context_pP->ue_context.physicalConfigDedicated_NB,
                      ue_context_pP->ue_context.mac_MainConfig_NB,
                      DRB2LCHAN_NB[i], //over the logical channel id of the DRB (>=4)
                      (LogicalChannelConfig_NB_r13_t*)NULL //MP: change this
          			);
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
void //was under ITTI
rrc_eNB_reconfigure_DRBs_NB(const protocol_ctxt_t* const ctxt_pP,
			       rrc_eNB_ue_context_NB_t*  ue_context_pP)
//------------------------------------------------------------------
{

  int i;
  int e_rab_done=0;

  //TODO: MP: to be check the number of e-rab and the parameter set over them

  for (i = 0; i < 2; i++) { //NB_RB_MAX - 3;  // S1AP_MAX_E_RAB //MP: why 3??--> i think at most 2 DRB = at most 2 e-rab

    if ( ue_context_pP->ue_context.e_rab[i].status < E_RAB_NB_STATUS_DONE){ // all those new e-rab ( E_RAB_NB_STATUS_NEW)
      ue_context_pP->ue_context.e_rab[i].status = E_RAB_NB_STATUS_NEW;
      ue_context_pP->ue_context.e_rab[i].param.e_rab_id = i + 1;
      ue_context_pP->ue_context.e_rab[i].param.qos.qci = (i % 9) + 5; //FIXME: if NB-IoT work on non-GBR range (5-9)
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.priority_level= i % PRIORITY_LEVEL_LOWEST;
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_capability= PRE_EMPTION_CAPABILITY_DISABLED;
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_vulnerability= PRE_EMPTION_VULNERABILITY_DISABLED;
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length = 0;
      //	memset (ue_context_pP->ue_context.e_rab[i].param.sgw_addr.buffer,0,20);
      ue_context_pP->ue_context.e_rab[i].param.sgw_addr.length = 0;
      ue_context_pP->ue_context.e_rab[i].param.gtp_teid=0;

      ue_context_pP->ue_context.nb_of_e_rabs++;
      e_rab_done++;
      LOG_I(RRC,"setting up the dedicated DRBs %d (index %d) status %d \n",
	    ue_context_pP->ue_context.e_rab[i].param.e_rab_id, i, ue_context_pP->ue_context.e_rab[i].status);
    }
  }
  ue_context_pP->ue_context.setup_e_rabs+=e_rab_done;

  //should be at most 2 DRBs set
  if(ue_context_pP->ue_context.setup_e_rabs > 2)
	  LOG_E(RRC, PROTOCOL_RRC_CTXT_FMT "e_context.setup_e_rabs > 2 but maxDRBs ==2", PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

  rrc_eNB_generate_dedicatedRRCConnectionReconfiguration_NB(ctxt_pP, ue_context_pP ); //no ho state
}
//-----------------------------------------------------------------------------
uint8_t qci_to_priority[9]={2,4,3,5,1,6,7,8,9}; //FIXME understand if used in NB_IoT (no GBR so only 5-9)

//-----------------------------------------------------------------------------
void //was under ITTI
rrc_eNB_generate_dedicatedRRCConnectionReconfiguration_NB(
		const protocol_ctxt_t* const ctxt_pP,
	    rrc_eNB_ue_context_NB_t*          const ue_context_pP
        //no ho state
	     )
//-----------------------------------------------------------------------------
{

  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;
  int i;

  struct DRB_ToAddMod_NB_r13                *DRB_config                       = NULL;
  struct RLC_Config_NB_r13                  *DRB_rlc_config                   = NULL;
  struct PDCP_Config_NB_r13                 *DRB_pdcp_config                  = NULL;
  //only RLC AM for NB-IoT
  struct LogicalChannelConfig_NB_r13        *DRB_lchan_config                 = NULL;

  //no UL specific parameters

  DRB_ToAddModList_NB_r13_t*                DRB_configList=ue_context_pP->ue_context.DRB_configList;
  DRB_ToAddModList_NB_r13_t**               DRB_configList2=NULL;

  struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13 *dedicatedInfoNASList_NB = NULL;
  DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;

  int drb_identity_index=0;
  int e_rab_done=0;


  uint8_t xid = rrc_eNB_get_next_transaction_identifier_NB(ctxt_pP->module_id);   //Transaction_id,

  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid];
  if (*DRB_configList2) {
    free(*DRB_configList2);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));

  /* Initialize NAS list */
  dedicatedInfoNASList_NB = CALLOC(1, sizeof(struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13));


  for ( i = 0  ; i < ue_context_pP->ue_context.setup_e_rabs ; i++){ //max must be 2 DRBs that are established

    // bypass the new and already configured erabs
    if (ue_context_pP->ue_context.e_rab[i].status >= E_RAB_NB_STATUS_DONE) {
      drb_identity_index++;
      continue; //skip to the next loop iteration
    }

    DRB_config = CALLOC(1, sizeof(*DRB_config));

    DRB_config->eps_BearerIdentity_r13 = CALLOC(1, sizeof(long));
    // allowed value 5..15, value : x+4 // e_rab_id setted previously in rrc_eNB_reconfigure_DRB_NB function
    *(DRB_config->eps_BearerIdentity_r13) = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;// x+4; // especial case generation

    DRB_config->drb_Identity_r13 =  1 + drb_identity_index + e_rab_done; //FIXME: max 2 DRBs could be established, x

    // allowed values (3..10) but 3 is reserved for SRB1bis so value : x+3
    DRB_config->logicalChannelIdentity_r13 = CALLOC(1, sizeof(long));
    *(DRB_config->logicalChannelIdentity_r13) = DRB_config->drb_Identity_r13 + 3;

    DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
    DRB_config->rlc_Config_r13 = DRB_rlc_config;

    DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
    DRB_config->pdcp_Config_r13 = DRB_pdcp_config;
    DRB_pdcp_config->discardTimer_r13 = CALLOC(1, sizeof(long));
    *(DRB_pdcp_config->discardTimer_r13) = PDCP_Config_NB_r13__discardTimer_r13_infinity;

    //must be checked in TS 36.413 and TS 23.401 if this parameter exist for NB_IoT
    switch (ue_context_pP->ue_context.e_rab[i].param.qos.qci){

    //GBR is not supported by NB-IoT

      /*
       * type: non-realtime data with low packer error rate
       * action: switch to RLC AM
       */
    case 5:  // 100ms, 10^-6, p1 , IMS signaling
    case 6:  // 300ms, 10^-6, p6
    case 8: // 300ms, 10^-6, p8
    case 69: // 60ms, 10^-6, p0.5, mission critical delay sensitive data, Lowest Priority
    case 70: // 200ms, 10^-6, p5.5, mission critical data
      // RLC
       DRB_rlc_config->present = RLC_Config_NB_r13_PR_am;
       DRB_rlc_config->choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = T_PollRetransmit_NB_r13_ms250; //random
       DRB_rlc_config->choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t8;
       DRB_rlc_config->choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 = CALLOC(1,sizeof(long));
       *(DRB_rlc_config->choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13)= DL_AM_RLC_NB_r13__enableStatusReportSN_Gap_r13_true;

       // PDCP FIXME: something to set for PDCP?

       break;

    default :
      LOG_E(RRC,"not supported qci %d\n", ue_context_pP->ue_context.e_rab[i].param.qos.qci);
      ue_context_pP->ue_context.e_rab[i].status = E_RAB_NB_STATUS_FAILED;
      ue_context_pP->ue_context.e_rab[i].xid = xid;
      continue;
    }

    DRB_pdcp_config->headerCompression_r13.present = PDCP_Config_NB_r13__headerCompression_r13_PR_notUsed;

    DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
    DRB_config->logicalChannelConfig_r13 = DRB_lchan_config;

    DRB_lchan_config->priority_r13 = CALLOC(1,sizeof(long));

    if (ue_context_pP->ue_context.e_rab[i].param.qos.qci < 9 )
    *(DRB_lchan_config->priority_r13) = qci_to_priority[ue_context_pP->ue_context.e_rab[i].param.qos.qci-1] + 3;//FIXME ??
    else
    	*(DRB_lchan_config->priority_r13)= 4L;

    //no prioritized bitrate
    //no bucketsize duration
    //no logical channel group

    ASN_SEQUENCE_ADD(&DRB_configList->list, DRB_config);
    ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);

    LOG_I(RRC,"EPS ID %ld, DRB ID %ld (index %d), QCI %d, priority %ld, LCID %ld\n",
	  *DRB_config->eps_BearerIdentity_r13,
	  DRB_config->drb_Identity_r13, i,
	  ue_context_pP->ue_context.e_rab[i].param.qos.qci,
	  *(DRB_lchan_config->priority_r13),
	  *(DRB_config->logicalChannelIdentity_r13)
	  );

    e_rab_done++;
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_NB_STATUS_DONE;
    ue_context_pP->ue_context.e_rab[i].xid = xid;

    {
      if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
	dedicatedInfoNas = CALLOC(1, sizeof(DedicatedInfoNAS_t));
	memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
	OCTET_STRING_fromBuf(dedicatedInfoNas,
			     (char*)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
			     ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
	ASN_SEQUENCE_ADD(&dedicatedInfoNASList_NB->list, dedicatedInfoNas);
	LOG_I(RRC,"add NAS info with size %d (rab id %d)\n",ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length, i);
      }
      else {
	LOG_W(RRC,"Not received activate dedicated EPS bearer context request\n");
      }
      /* TODO parameters yet to process ... */
      {
	//      ue_context_pP->ue_context.e_rab[i].param.qos;
	//      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
	//      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
      }
    }

  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList_NB != NULL) {
    if (dedicatedInfoNASList_NB->list.count == 0) {
      free(dedicatedInfoNASList_NB);
      dedicatedInfoNASList_NB = NULL;
      LOG_W(RRC,"dedicated NAS list is empty, free the list and reset the address\n");
    }
  } else {
    LOG_W(RRC,"dedicated NAS list is empty\n");
  }

  memset(buffer, 0, RRC_BUF_SIZE);

   size = do_RRCConnectionReconfiguration_NB(ctxt_pP,
					  buffer,
					  xid,
					  (SRB_ToAddModList_NB_r13_t*)NULL,
					  (DRB_ToAddModList_NB_r13_t*)*DRB_configList2,
					  (DRB_ToReleaseList_NB_r13_t*)NULL,  // DRB2_list,
					  NULL, NULL, //physical an MAC config dedicated
					  (struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13*)dedicatedInfoNASList_NB
                                        );


#ifdef RRC_MSG_PRINT
  LOG_F(RRC,"[MSG] RRC Connection Reconfiguration-NB\n");
  for (i = 0; i < size; i++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)buffer)[i]);
  }
  LOG_F(RRC,"\n");
  ////////////////////////////////////////
#endif

//#if defined(ENABLE_ITTI)....

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }


 LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate RRCConnectionReconfiguration-NB (bytes %d, UE RNTI %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);

  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH1); //through SRB1

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" dedicated rrcConnectionReconfiguration-NB UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  NB_rrc_data_req(
    ctxt_pP,
    DCCH1,//through SRB1
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);

}

//-----------------------------------------------------------------------------
void
rrc_eNB_process_RRCConnectionSetupComplete_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*         ue_context_pP,
  RRCConnectionSetupComplete_NB_r13_IEs_t * rrcConnectionSetupComplete_NB
)
//-----------------------------------------------------------------------------
{
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, " "processing RRCConnectionSetupComplete_NB from UE (SRB1bis/SRB1 Active)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

  ue_context_pP->ue_context.Srb1bis.Active=1;  //SRB1bis active for UE, from eNB point of view
  ue_context_pP->ue_context.Srb1.Active=1 //SRB1 active for UE (implicitly activated)

  T(T_ENB_RRC_CONNECTION_SETUP_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#if defined(ENABLE_USE_MME)

  if (EPC_MODE_ENABLED == 1) {
    // Forward message to S1AP layer
    rrc_eNB_send_S1AP_NAS_FIRST_REQ(
      ctxt_pP,
      ue_context_pP,
      rrcConnectionSetupComplete_NB); ///TODO: look at NAS messages
  } else
#endif
  {
    // RRC loop back (no S1AP), send SecurityModeCommand to UE
    rrc_eNB_generate_SecurityModeCommand_NB(
      ctxt_pP,
      ue_context_pP);
  }
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_SecurityModeCommand_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP
)
//-----------------------------------------------------------------------------
{
	/* R2-163262 3GPP NB-IOT Ad-hoc Meeting #2
	 * The SRB1-bis (no PDCP) is used for SecurityModeCommand and SecurityModeFailure messages
	 * The SRB1 (full PDCP) is used for SecurityModeComplete message
	 */


  uint8_t                             buffer[100];
  uint8_t                             size;

  T(T_ENB_RRC_SECURITY_MODE_COMMAND, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  size = do_SecurityModeCommand_NB(
           ctxt_pP,
           buffer,
           rrc_eNB_get_next_transaction_identifier_NB(ctxt_pP->module_id),
           ue_context_pP->ue_context.ciphering_algorithm,
           ue_context_pP->ue_context.integrity_algorithm);

#ifdef RRC_MSG_PRINT
  uint16_t i=0;
  LOG_F(RRC,"[MSG] RRC Security Mode Command\n");

  for (i = 0; i < size; i++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)buffer)[i]);
  }

  LOG_F(RRC,"\n");
#endif

  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate SecurityModeCommand-NB (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);

  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (securityModeCommand to UE MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH0); //MP: SRB1bis

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" securityModeCommand UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  NB_rrc_data_req( //to PDCP
	       ctxt_pP,
	       DCCH0,//MP:through SRB1bis
	       rrc_eNB_mui++,
	       SDU_CONFIRM_NO,
	       size,
	       buffer,
		   PDCP_TRANSMISSION_MODE_TRANSPARENT);

}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_UECapabilityEnquiry_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP
)
//-----------------------------------------------------------------------------
{
//TODO: Clarify: this message is also called when we receive SecurityModeFailure--> we should continue to use SRB1bis?

  uint8_t                             buffer[100];
  uint8_t                             size;

  T(T_ENB_RRC_UE_CAPABILITY_ENQUIRY, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  size = do_UECapabilityEnquiry_NB(
           ctxt_pP,
           buffer,
           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id));

  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate UECapabilityEnquiry-NB (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);

  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (UECapabilityEnquiry-NB MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH0);//through SRB1bis

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcUECapabilityEnquiry UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  NB_rrc_data_req( //to PDCP
	       ctxt_pP,
	       DCCH1, //FIXME SRB1 or SRB1bis? depends if successful security activation by UE
	       rrc_eNB_mui++,
	       SDU_CONFIRM_NO,
	       size,
	       buffer,
	       PDCP_TRANSMISSION_MODE_CONTROL);//XXX MP: no more transparent to PDCP // SRB1 was registered to the pdcp at the beginning

}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_defaultRRCConnectionReconfiguration_NB(const protocol_ctxt_t* const ctxt_pP,
						     rrc_eNB_ue_context_NB_t*          const ue_context_pP
						//no HO flag
						     )
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;
  int                                 i;

  // configure SRB1, PhysicalConfigDedicated, MAC_MainConfig for UE

  eNB_RRC_INST_NB*                       rrc_inst = &eNB_rrc_inst_NB[ctxt_pP->module_id];
  struct PhysicalConfigDedicated_NB_r13**    physicalConfigDedicated_NB = &ue_context_pP->ue_context.physicalConfigDedicated_NB;

  struct SRB_ToAddMod_NB_r13                           *SRB1_config                      = NULL;
  struct SRB_ToAddMod_NB_r13__rlc_Config_r13    	   *SRB1_rlc_config                  = NULL;
  struct SRB_ToAddMod_NB_r13__logicalChannelConfig_r13 *SRB1_lchan_config          		 = NULL;

  SRB_ToAddModList_NB_r13_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList; //both SRB1 and SRB1bis
  SRB_ToAddModList_NB_r13_t                 **SRB_configList2                 = NULL; //only SRB1

  struct DRB_ToAddMod_NB_r13                *DRB_config                       = NULL;
  struct RLC_Config_NB_r13                  *DRB_rlc_config                   = NULL;
  struct PDCP_Config_NB_r13                 *DRB_pdcp_config                  = NULL;
  //only AM is supported by NB-IoT
  struct LogicalChannelConfig_NB_r13        *DRB_lchan_config                 = NULL;

  DRB_ToAddModList_NB_r13_t**                DRB_configList = &ue_context_pP->ue_context.DRB_configList;
  DRB_ToAddModList_NB_r13_t**                DRB_configList2 = NULL;

   MAC_MainConfig_NB_r13_t                   *mac_MainConfig_NB               = NULL;

  long                              *periodicBSR_Timer;
  long								*enableStatusReportSN_Gap = NULL; //should be disabled
  long								*priority= NULL; //1 for SRB1 and 1 for SRB1bis
  BOOLEAN_t						    *logicalChannelSR_Prohibit = NULL;
  RSRP_Range_t                       *rsrp                             = NULL; //may not used

  struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13 *dedicatedInfoNASList_NB = NULL;
  DedicatedInfoNAS_t                 									   *dedicatedInfoNas        = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;

  uint8_t xid = rrc_eNB_get_next_transaction_identifier_NB(ctxt_pP->module_id);   //Transaction_id,

  T(T_ENB_RRC_CONNECTION_RECONFIGURATION, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  // Configure/	Refresh SRB1
  /// SRB1
  SRB_configList2=&ue_context_pP->ue_context.SRB_configList2[xid];
  if (*SRB_configList2) {
    free(*SRB_configList2);
  }
  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));
  memset(*SRB_configList2, 0, sizeof(**SRB_configList2));
  SRB1_config = CALLOC(1, sizeof(*SRB1_config));

  SRB1_rlc_config = CALLOC(1, sizeof(*SRB1_rlc_config));
  SRB1_config->rlc_Config_r13 = SRB1_rlc_config;

  //parameters values set as in the TS 36.331 specs ( pag 640)
  SRB1_rlc_config->present = SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB1_rlc_config->choice.explicitValue.present = RLC_Config_NB_r13_PR_am;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = T_PollRetransmit_NB_r13_ms25000;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t4;
  SRB1_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 = enableStatusReportSN_Gap;

  SRB1_lchan_config = CALLOC(1, sizeof(*SRB1_lchan_config));
  SRB1_config->logicalChannelConfig_r13 = SRB1_lchan_config;

  SRB1_lchan_config->present = SRB_ToAddMod_NB_r13__logicalChannelConfig_r13_PR_explicitValue;

  priority = CALLOC(1, sizeof(long));
  *priority = 1;
  SRB1_lchan_config->choice.explicitValue.priority_r13 =priority;

  logicalChannelSR_Prohibit = CALLOC(1, sizeof(BOOLEAN_t));
  *logicalChannelSR_Prohibit = 1;
  SRB1_lchan_config->choice.explicitValue.logicalChannelSR_Prohibit_r13 = logicalChannelSR_Prohibit;

  // this list has the configuration for SRB1 and SRB1bis
  //XXX: Problems ? because SRB_ToAddModList_NB_r13_t max size = 1
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB1_config);

  // this list has only the configuration for SRB1
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB1_config);


  /// Configure DRB

  //list of all DRBs
  if (*DRB_configList) {
    free(*DRB_configList);
  }
  *DRB_configList = CALLOC(1, sizeof(**DRB_configList));
  memset(*DRB_configList, 0, sizeof(**DRB_configList));

  // list for the configured DRB for this xid (transaction id)
  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid];
  if (*DRB_configList2) {
    free(*DRB_configList2);
  }
  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  memset(*DRB_configList2, 0, sizeof(**DRB_configList2));


  /// DRB (maxDRBs = 2)

  DRB_config = CALLOC(1, sizeof(*DRB_config));
  DRB_config->eps_BearerIdentity_r13 = CALLOC(1, sizeof(long));
  //FIXME: MP: I'm not sure that in NB-IoT the first 0..4 eps-bearerIdentity will be reserved for NSAPI (NEtwork Service Access Point Identifier)
  *(DRB_config->eps_BearerIdentity_r13) = 5L; // LW set to first value, allowed value 5..15, value : x+4
  // NN/MP: this is the 1st DRB for this ue, so set it to 1
  DRB_config->drb_Identity_r13 = (DRB_Identity_t) 1; //allowed values INTEGER (1..32), value: x
  DRB_config->logicalChannelIdentity_r13 = CALLOC(1, sizeof(long));
  //MP: logical channel identity =3 is reserved for SRB1bis, seee TS 36.331 pag 616
  *(DRB_config->logicalChannelIdentity_r13) = (long)4; //allowed value (4..10), value : x+3

  DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
  DRB_config->rlc_Config_r13 = DRB_rlc_config;


  //set as TS 36.331 specs
  ///RLC-AM
  DRB_rlc_config->present = RLC_Config_NB_r13_PR_am;
  DRB_rlc_config->choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = T_PollRetransmit_NB_r13_ms25000;
  DRB_rlc_config->choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t4;
  DRB_rlc_config->choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 = enableStatusReportSN_Gap;

  ///PDCP
  DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
  DRB_config->pdcp_Config_r13 = DRB_pdcp_config;
  DRB_pdcp_config->discardTimer_r13 = CALLOC(1, sizeof(long));
  *(DRB_pdcp_config->discardTimer_r13) = PDCP_Config_NB_r13__discardTimer_r13_infinity;

//no UM modality for RLC in NB-IoT

  DRB_pdcp_config->headerCompression_r13.present = PDCP_Config_NB_r13__headerCompression_r13_PR_notUsed;

  DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
  DRB_config->logicalChannelConfig_r13 = DRB_lchan_config;

  DRB_lchan_config->priority_r13 = CALLOC(1, sizeof(long));
  *(DRB_lchan_config->priority_r13) = 12; // lower priority than srb1, srb1bis and other dedicated bearer
  DRB_lchan_config->logicalChannelSR_Prohibit_r13 = logicalChannelSR_Prohibit;//FIXME: for DRB should be used?

  //Add the DRB in both list
  ASN_SEQUENCE_ADD(&(*DRB_configList)->list, DRB_config);
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);


  ///Mac_MainConfig
  mac_MainConfig_NB = CALLOC(1, sizeof(*mac_MainConfig_NB));
  ue_context_pP->ue_context.mac_MainConfig_NB = mac_MainConfig_NB;

  mac_MainConfig_NB->ul_SCH_Config_r13 = CALLOC(1, sizeof(*mac_MainConfig_NB->ul_SCH_Config_r13));

  //no maxARQtx

  periodicBSR_Timer = CALLOC(1, sizeof(long));
  *periodicBSR_Timer = PeriodicBSR_Timer_NB_r13_pp8;
  mac_MainConfig_NB->ul_SCH_Config_r13->periodicBSR_Timer_r13 = periodicBSR_Timer;
  mac_MainConfig_NB->ul_SCH_Config_r13->retxBSR_Timer_r13 = RetxBSR_Timer_NB_r13_infinity;

  mac_MainConfig_NB->timeAlignmentTimerDedicated_r13 = TimeAlignmentTimer_infinity;

  mac_MainConfig_NB->drx_Config_r13 = NULL;

  //no phr_config

  mac_MainConfig_NB->logicalChannelSR_Config_r13 = CALLOC(1, sizeof(struct MAC_MainConfig_NB_r13__logicalChannelSR_Config_r13));
  mac_MainConfig_NB->logicalChannelSR_Config_r13->present = MAC_MainConfig_NB_r13__logicalChannelSR_Config_r13_PR_setup;
  //depends if previously activated
  mac_MainConfig_NB->logicalChannelSR_Config_r13->choice.setup.logicalChannelSR_ProhibitTimer_r13 =
		  MAC_MainConfig_NB_r13__logicalChannelSR_Config_r13__setup__logicalChannelSR_ProhibitTimer_r13_pp2; //value in PP=PDCCH periods


  struct PhysicalConfigDedicated ciao;

  if (*physicalConfigDedicated_NB) {

	  //TODO: which value should be configured of phisical config dedicated?
	  //antennaInfo not present in PhysicalConfigDedicated for NB_IoT
      //cqi reporting is not present in PhysicalConfigDedicated for NB_IoT
	  /*
	   * carrierConfigDedicated_r13
	   * npdcch_ConfigDedicated_r13
	   * npusch_ConfigDedicated_r13
	   * uplinkPowerControlDedicated_r13
	   */
  }
  else {
    LOG_E(RRC,"physical_config_dedicated not present in RRCConnectionReconfiguration-NB. Not reconfiguring!\n");
  }

  //no Measurement ID list
  //no measurement Object--> no cellToAddModList
  //no report config_list
  //no Sparams
  //no handover

//#if defined(ENABLE_ITTI).....

  /* Initialize NAS list */
  dedicatedInfoNASList_NB = CALLOC(1, sizeof(struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13));

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      dedicatedInfoNas = CALLOC(1, sizeof(DedicatedInfoNAS_t));
      memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
      OCTET_STRING_fromBuf(dedicatedInfoNas,
			   (char*)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                           ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
      ASN_SEQUENCE_ADD(&dedicatedInfoNASList_NB->list, dedicatedInfoNas);
    }

    /*OLD  TODO parameters yet to process ... */
    {
      //      ue_context_pP->ue_context.e_rab[i].param.qos;
      //      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      //      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
    }

    /* TODO should test if e RAB are Ok before! */
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_NB_STATUS_DONE;
    LOG_D(RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
	  i, ue_context_pP->ue_context.e_rab[i].status, "E_RAB_NB_STATUS_DONE");
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList_NB->list.count == 0) {
    free(dedicatedInfoNASList_NB);
    dedicatedInfoNASList_NB = NULL;
  }


  memset(buffer, 0, RRC_BUF_SIZE);

  size = do_RRCConnectionReconfiguration_NB(ctxt_pP,
                                         buffer,
                                         xid,   //Transaction_id,
                                         (SRB_ToAddModList_NB_r13_t*)*SRB_configList2, //only SRB1
                                         (DRB_ToAddModList_NB_r13_t*)*DRB_configList,
                                         (DRB_ToReleaseList_NB_r13_t*)NULL,  // DRB2_list,
                                         (struct PhysicalConfigDedicated_NB_r13*)*physicalConfigDedicated_NB,
                                         (MAC_MainConfig_t*)mac_MainConfig_NB,
                                         (struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13*)dedicatedInfoNASList_NB
                                        );

#ifdef RRC_MSG_PRINT
  LOG_F(RRC,"[MSG] RRCConnectionReconfiguration-NB\n");
  for (i = 0; i < size; i++) {
    LOG_F(RRC,"%02x ", ((uint8_t*)buffer)[i]);
  }
  LOG_F(RRC,"\n");
  ////////////////////////////////////////
#endif

/// was inside ITTI but I think it is needed
  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }


  LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate RRCConnectionReconfiguration-NB (bytes %d, UE id %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);

  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration-NB to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH1);//through SRB1

  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcConnectionReconfiguration-NB UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  NB_rrc_data_req( //to PDCP
	       ctxt_pP,
	       DCCH1,//through SRB1
	       rrc_eNB_mui++,
	       SDU_CONFIRM_NO,
	       size,
	       buffer,
	       PDCP_TRANSMISSION_MODE_CONTROL);
}


/*-----------------CONFIGURATION-------------------*/

//-----------------------------------------------------------------------------
static void
init_SI_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const int              CC_id,
  RrcConfigurationReq* configuration
)
//-----------------------------------------------------------------------------
{

	//FIXME: MP: may this two  parameters are not needed
  uint8_t                             SIwindowsize = 1; //frame
  uint16_t                            SIperiod = 8; // in frames


  //copy basic parameters
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].physCellId      = configuration->Nid_cell[CC_id];
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].p_eNB           = configuration->nb_antenna_ports[CC_id];
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Ncp             = configuration->prefix_type[CC_id];
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].dl_CarrierFreq  = configuration->downlink_frequency[CC_id];
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].ul_CarrierFreq  = configuration->downlink_frequency[CC_id]+ configuration->uplink_frequency_offset[CC_id];


  //TODO: verify who allocate memory for sib1_NB, sib2_NB, sib3_NB and mib_nb in the carrier before being passed as parameter

  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB1_NB = 0;
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB23_NB = 0;
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_MIB_NB = 0;

  //MIB
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].MIB_NB = (uint8_t*) malloc16(32); //MIB is 34 bits=5bytes needed

  //FIXME do_MIB_NB parameters
  if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].MIB_NB)
  {
	  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_MIB_NB =
	  			  do_MIB_NB(&eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id],
	  					  	configuration->N_RB_DL,
					        0
					        );
  }
  else {
      LOG_E(RRC, PROTOCOL_RRC_CTXT_FMT" init_SI: FATAL, no memory for MIB_NB allocated\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
      mac_xface->macphy_exit("[RRC][init_SI] FATAL, no memory for MIB_NB allocated");
    }


  if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_MIB_NB == 255) {
      mac_xface->macphy_exit("[RRC][init_SI] FATAL, eNB_rrc_inst_NB[enb_mod_idP].carrier[CC_id].sizeof_MIB_NB == 255");
    }

 //SIB1_NB
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SIB1_NB = (uint8_t*) malloc16(32);//allocation of buffer memory for SIB1_NB

  if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SIB1_NB)
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB1_NB = do_SIB1_NB( //follow the new implementation
    				ctxt_pP->module_id,
					CC_id,
					&eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id],
					configuration
    				);

  else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_FMT" init_SI: FATAL, no memory for SIB1_NB allocated\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
    mac_xface->macphy_exit("[RRC][init_SI] FATAL, no memory for SIB1_NB allocated");
  }

  if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB1_NB == 255) {
    mac_xface->macphy_exit("[RRC][init_SI] FATAL, eNB_rrc_inst_NB[enb_mod_idP].carrier[CC_id].sizeof_SIB1_NB == 255");
  }

  //SIB23_NB
  eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SIB23_NB = (uint8_t*) malloc16(64);

  if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].SIB23_NB) {
    eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB23_NB = do_SIB23_NB(
          ctxt_pP->module_id,
          CC_id,
		  &eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id],
		  configuration
        );

    if (eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sizeof_SIB23_NB == 255) {
      mac_xface->macphy_exit("[RRC][init_SI] FATAL, eNB_rrc_inst_NB[mod].carrier[CC_id].sizeof_SIB23_NB == 255");
    }

    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" SIB2/3 Contents (partial)\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

    /*FIXME: decide which parameter to show....*/

    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" TODO: some parameter of SIB23-NB to show\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

    //Use the following as an example
//    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" npusch_config_common.groupAssignmentNPUSCH = %ld\n",
//          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
//          eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.
//		  ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13);


    //Configure MAC
    LOG_D(RRC,
          PROTOCOL_RRC_CTXT_FMT" RRC_UE --- MAC_CONFIG_REQ (SIB1_NB & SIB2_NB params) ---> MAC_UE\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

    //
    NB_rrc_mac_config_req_eNB(ctxt_pP->module_id,
    							CC_id,
								0,//rnti
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].physCellId,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].p_eNB,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].Ncp,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->freqBandIndicator_r13,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->freqBandInfo_r13,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->multiBandInfoList_r13,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->downlinkBitmap_r13,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->eutraControlRegionSize_r13,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib1_NB->nrs_CRS_PowerOffset_r13,
								&SIwindowsize,
								&SIperiod,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].dl_CarrierFreq,
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].ul_CarrierFreq,
								(BCCH_BCH_Message_NB_t*) &
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].mib_NB,
								(RadioResourceConfigCommonSIB_NB_r13_t *) &
								eNB_rrc_inst_NB[ctxt_pP->module_id].carrier[CC_id].sib2_NB->radioResourceConfigCommon_r13,
								(struct PhysicalConfigDedicated_NB_r13_t *)NULL,
								(MAC_MainConfig_NB_r13_t *) NULL,
								0,// MP:logicalChannelID  //TODO still have to be properly managed in the interface
								(struct LogicalChannelConfig_NB_r13_t *)NULL
								);
  } else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_FMT" init_SI: FATAL, no memory for SIB2/3_NB allocated\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
    mac_xface->macphy_exit("[RRC][init_SI] FATAL, no memory for SIB2/3_NB allocated");
  }
}

//-----------------------------------------------------------------------------
//aka openair_rrc_eNB_init
char
openair_rrc_eNB_configuration_NB(
  const module_id_t enb_mod_idP,
  RrcConfigurationReq* configuration //FIXME MP: previously was insiede ITTI but actually I put it out
)
//-----------------------------------------------------------------------------
{
  protocol_ctxt_t ctxt;
  int             CC_id;

  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, enb_mod_idP, ENB_FLAG_YES, NOT_A_RNTI, 0, 0,enb_mod_idP);
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_FMT" Init...\n",
        PROTOCOL_RRC_CTXT_ARGS(&ctxt));

#if OCP_FRAMEWORK
while ( eNB_rrc_inst_NB == NULL ) {
  LOG_E(RRC, "eNB_rrc_inst_NB not yet initialized, waiting 1 second\n");
  sleep(1);
}
#endif
  AssertFatal(eNB_rrc_inst_NB != NULL, "eNB_rrc_nb_iot_inst not initialized!");
  AssertFatal(NUMBER_OF_UE_MAX < (module_id_t)0xFFFFFFFFFFFFFFFF, " variable overflow");

  eNB_rrc_inst_NB[ctxt.module_id].Nb_ue = 0;

  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    eNB_rrc_inst_NB[ctxt.module_id].carrier[CC_id].Srb0.Active = 0;
  }

  uid_linear_allocator_init_NB(&eNB_rrc_inst_NB[ctxt.module_id].uid_allocator); //rrc_eNB_UE_context
  RB_INIT(&eNB_rrc_inst_NB[ctxt.module_id].rrc_ue_head);

  eNB_rrc_inst_NB[ctxt.module_id].initial_id2_s1ap_ids = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);
  eNB_rrc_inst_NB[ctxt.module_id].s1ap_id2_s1ap_ids    = hashtable_create (NUMBER_OF_UE_MAX * 2, NULL, NULL);

  memcpy(&eNB_rrc_inst_NB[ctxt.module_id].configuration,configuration,sizeof(RrcConfigurationReq));

  /// System Information INIT

  LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Checking release \n",
        PROTOCOL_RRC_CTXT_ARGS(&ctxt));

  LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" NB-IoT Rel13 detected\n",PROTOCOL_RRC_CTXT_ARGS(&ctxt));

  //no CBA

  //init SI
  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    init_SI_NB(&ctxt, CC_id,configuration);
  }

  /*New implementation Raymond*/
  //MP: NEW defined in rrc_common_nb_iot.h // no more called in MAC/main.c but directly here
  //rrc_init_global_param_NB();

  //XXX following the old implementation: openair_rrc_top_init is called in MAC/main.c

  //FIXME: this probably would create some bug for the following reaon:
  //in the onld implementation this function was called in the MAC/main.c
  //actually is called here --> we are allocating memory so maybe we generate some problems

  //openair_rrc_top_init_eNB_NB();

  //init ch SRB0, SRB1 & BDTCH
  openair_eNB_rrc_on_NB(&ctxt);

  return 0;

}


/*-----------------STATE MACHINE---------------*/

//-----------------------------------------------------------------------------
int
rrc_eNB_decode_ccch_NB(
  protocol_ctxt_t* const ctxt_pP,
  const SRB_INFO_NB*        const Srb_info, //SRB0
  const int              CC_id
)
//-----------------------------------------------------------------------------
{
  module_id_t                                   Idx;
  asn_dec_rval_t                      dec_rval;
  UL_CCCH_Message_NB_t                  *ul_ccch_msg_NB = NULL;
  RRCConnectionRequest_NB_r13_IEs_t                *rrcConnectionRequest_NB = NULL;
  RRCConnectionReestablishmentRequest_NB_r13_IEs_t* rrcConnectionReestablishmentRequest_NB = NULL;
  RRCConnectionResumeRequest_NB_r13_IEs_t *rrcConnectionResumeRequest_NB= NULL;
  int                                 i, rval;
  struct rrc_eNB_ue_context_NB_s*                  ue_context_p = NULL;
  uint64_t                                      random_value = 0;
  int                                           stmsi_received = 0;

  T(T_ENB_RRC_UL_CCCH_DATA_IN, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  //memset(ul_ccch_msg,0,sizeof(UL_CCCH_Message_t));

  LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Decoding UL CCCH %x.%x.%x.%x.%x.%x (%p)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ((uint8_t*) Srb_info->Rx_buffer.Payload)[0],
        ((uint8_t *) Srb_info->Rx_buffer.Payload)[1],
        ((uint8_t *) Srb_info->Rx_buffer.Payload)[2],
        ((uint8_t *) Srb_info->Rx_buffer.Payload)[3],
        ((uint8_t *) Srb_info->Rx_buffer.Payload)[4],
        ((uint8_t *) Srb_info->Rx_buffer.Payload)[5], (uint8_t *) Srb_info->Rx_buffer.Payload);

  dec_rval = uper_decode(
               NULL,
               &asn_DEF_UL_CCCH_Message_NB,
               (void**)&ul_ccch_msg_NB,
               (uint8_t*) Srb_info->Rx_buffer.Payload,
               100,
               0,
               0);

//#if defined(ENABLE_ITTI)....
//#   if defined(DISABLE_ITTI_XER_PRINT)...


  for (i = 0; i < 8; i++) { //MP: FIXME why 8?
    LOG_T(RRC, "%x.", ((uint8_t *) & ul_ccch_msg_NB)[i]);
  }

  if (dec_rval.consumed == 0) {
    LOG_E(RRC,
          PROTOCOL_RRC_CTXT_UE_FMT" FATAL Error in receiving CCCH\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    return -1;
  }

  if (ul_ccch_msg_NB->message.present == UL_CCCH_MessageType_NB_PR_c1) {

    switch (ul_ccch_msg_NB->message.choice.c1.present) {

    case UL_CCCH_MessageType_NB__c1_PR_NOTHING:
      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_FMT" Received PR_NOTHING on UL-CCCH-Message-NB\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
      break;

    case UL_CCCH_MessageType_NB__c1_PR_rrcConnectionReestablishmentRequest_r13:
      T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_REQUEST, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRCConnectionReestablishementRequest-NB\n");

      for (i = 0; i < Srb_info->Rx_buffer.payload_size; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Srb_info->Rx_buffer.Payload)[i]);
      }

      LOG_F(RRC,"\n");
#endif
      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT"MAC_eNB--- MAC_DATA_IND (rrcConnectionReestablishmentRequest-NB on SRB0) --> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
      rrcConnectionReestablishmentRequest_NB =
        &ul_ccch_msg_NB->message.choice.c1.choice.rrcConnectionReestablishmentRequest_r13.criticalExtensions.choice.rrcConnectionReestablishmentRequest_r13;

      //for NB-IoT only "reconfiguration Failure" and "other failure" are allowed as reestablishment cause
      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionReestablishmentRequest-NB cause %s\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            ((rrcConnectionReestablishmentRequest_NB->reestablishmentCause_r13 == ReestablishmentCause_NB_r13_otherFailure) ? "Other Failure" :
             "reconfigurationFailure"));


     //FIXME:connection reestablishment to be implemented
      /*{
      uint64_t                            c_rnti = 0;

      memcpy(((uint8_t *) & c_rnti) + 3, rrcConnectionReestablishmentRequest.UE_identity.c_RNTI.buf,
      rrcConnectionReestablishmentRequest.UE_identity.c_RNTI.size);
      ue_mod_id = rrc_eNB_get_UE_index(enb_mod_idP, c_rnti);
      }

      if ((eNB_rrc_inst_NB[enb_mod_idP].phyCellId == rrcConnectionReestablishmentRequest.UE_identity.physCellId) &&
      (ue_mod_id != UE_INDEX_INVALID)){
      rrc_eNB_generate_RRCConnectionReestablishment(enb_mod_idP, frameP, ue_mod_id);
      }else {
      rrc_eNB_generate_RRCConnectionReestablishmentReject(enb_mod_idP, frameP, ue_mod_id);
      }
      */
      /* reject all reestablishment attempts for the moment */

      //for the moment we only reject
      rrc_eNB_generate_RRCConnectionReestablishmentReject_NB(ctxt_pP,
                       rrc_eNB_get_ue_context_NB(&eNB_rrc_inst_NB[ctxt_pP->module_id], ctxt_pP->rnti),
                       CC_id);

      break;

    case UL_CCCH_MessageType_NB__c1_PR_rrcConnectionRequest_r13: //MSG3
      T(T_ENB_RRC_CONNECTION_REQUEST, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRCConnectionRequest-NB\n");

      for (i = 0; i < Srb_info->Rx_buffer.payload_size; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Srb_info->Rx_buffer.Payload)[i]);
      }

      LOG_F(RRC,"\n");
#endif
      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT"MAC_eNB --- MAC_DATA_IND  (rrcConnectionRequest-NB on SRB0) --> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

      ue_context_p = rrc_eNB_get_ue_context_NB( //XXX define new function in rrc_eNB_UE_context
                       &eNB_rrc_inst_NB[ctxt_pP->module_id],
                       ctxt_pP->rnti);

      if (ue_context_p != NULL) { //MP: i receive a ConnectionRequest from a UE which have already a context
        // erase content
        rrc_eNB_free_mem_UE_context_NB(ctxt_pP, ue_context_p);

        MSC_LOG_RX_DISCARDED_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Srb_info->Rx_buffer.Payload,
          dec_rval.consumed,
          MSC_AS_TIME_FMT" RRCConnectionRequest-NB UE %x size %u (UE already in context)",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          dec_rval.consumed);
      }
      else {
        rrcConnectionRequest_NB = &ul_ccch_msg_NB->message.choice.c1.choice.rrcConnectionRequest_r13.criticalExtensions.choice.rrcConnectionRequest_r13;
        {
          if (InitialUE_Identity_PR_randomValue == rrcConnectionRequest_NB->ue_Identity_r13.present) {

        	  //InitialUE-Identity randomValue size should be 40bits = 5 byte
            AssertFatal(rrcConnectionRequest_NB->ue_Identity_r13.choice.randomValue.size == 5,
                        "wrong InitialUE-Identity randomValue size, expected 5, provided %d",
                        rrcConnectionRequest_NB->ue_Identity_r13.choice.randomValue.size);

            memcpy(((uint8_t*) & random_value) + 3,
                   rrcConnectionRequest_NB->ue_Identity_r13.choice.randomValue.buf,
                   rrcConnectionRequest_NB->ue_Identity_r13.choice.randomValue.size);

            /* if there is already a registered UE (with another RNTI) with this random_value,
             * the current one must be removed from MAC/PHY (zombie UE)
             */
            if ((ue_context_p = rrc_eNB_ue_context_random_exist_NB(ctxt_pP, random_value))) {
              LOG_W(RRC, "new UE rnti %x (coming with random value) is already there as UE %x, removing %x from MAC/PHY\n",
                    ctxt_pP->rnti, ue_context_p->ue_context.rnti, ctxt_pP->rnti);

	      NB_rrc_mac_remove_ue(ctxt_pP->module_id, ctxt_pP->rnti);
              ue_context_p = NULL;
              return 0;
            } else {
              ue_context_p = rrc_eNB_get_next_free_ue_context_NB(ctxt_pP, random_value);
            }
          } else if (InitialUE_Identity_PR_s_TMSI == rrcConnectionRequest_NB->ue_Identity_r13.present) {
            /* Save s-TMSI */
            S_TMSI_t   s_TMSI = rrcConnectionRequest_NB->ue_Identity_r13.choice.s_TMSI;
            mme_code_t mme_code = BIT_STRING_to_uint8(&s_TMSI.mmec);
            m_tmsi_t   m_tmsi   = BIT_STRING_to_uint32(&s_TMSI.m_TMSI);
            random_value = (((uint64_t)mme_code) << 32) | m_tmsi;

            if ((ue_context_p = rrc_eNB_ue_context_stmsi_exist_NB(ctxt_pP, mme_code, m_tmsi))) {
	      LOG_I(RRC," S-TMSI exists, ue_context_p %p, old rnti %x => %x\n",ue_context_p,ue_context_p->ue_context.rnti,ctxt_pP->rnti);
	      NB_rrc_mac_remove_ue(ctxt_pP->module_id, ue_context_p->ue_context.rnti);
	      stmsi_received=1;

              /* replace rnti in the context */
              /* for that, remove the context from the RB tree */
	      ///FIXME MP: warning --> implicit declaration because I insert the new type "rrc_ue_tree_NB_s"
              RB_REMOVE(rrc_ue_tree_NB_s, &eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head, ue_context_p);
              /* and insert again, after changing rnti everywhere it has to be changed */
              ue_context_p->ue_id_rnti = ctxt_pP->rnti;
	      ue_context_p->ue_context.rnti = ctxt_pP->rnti;
              RB_INSERT(rrc_ue_tree_NB_s, &eNB_rrc_inst_NB[ctxt_pP->module_id].rrc_ue_head, ue_context_p);
              /* reset timers */
              ue_context_p->ue_context.ul_failure_timer = 0;
              ue_context_p->ue_context.ue_release_timer = 0;
            } else {
	      LOG_I(RRC," S-TMSI doesn't exist, setting Initialue_identity_s_TMSI.m_tmsi to %p => %x\n",ue_context_p,m_tmsi);
              ue_context_p = rrc_eNB_get_next_free_ue_context_NB(ctxt_pP, NOT_A_RANDOM_UE_IDENTITY);
              if (ue_context_p == NULL)
                LOG_E(RRC, "%s:%d:%s: rrc_eNB_get_next_free_ue_context returned NULL\n", __FILE__, __LINE__, __FUNCTION__);
              if (ue_context_p != NULL) {
	        ue_context_p->ue_context.Initialue_identity_s_TMSI.presence = TRUE;
	        ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code = mme_code;
	        ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi = m_tmsi;
              } else {
                /* TODO: do we have to break here? */
                //break;
              }
            }

            MSC_LOG_RX_MESSAGE(
              MSC_RRC_ENB,
              MSC_RRC_UE,
              Srb_info->Rx_buffer.Payload,
              dec_rval.consumed,
              MSC_AS_TIME_FMT" RRCConnectionRequest-NB UE %x size %u (s-TMSI mmec %u m_TMSI %u random UE id (0x%" PRIx64 ")",
              MSC_AS_TIME_ARGS(ctxt_pP),
              ue_context_p->ue_context.rnti,
              dec_rval.consumed,
              ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
              ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
              ue_context_p->ue_context.random_ue_identity);
          } else {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionRequest-NB without random UE identity or S-TMSI not supported, let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

            rrc_eNB_generate_RRCConnectionReject_NB(ctxt_pP,
                             rrc_eNB_get_ue_context_NB(&eNB_rrc_inst_NB[ctxt_pP->module_id], ctxt_pP->rnti),
                             CC_id);
            break;
          }

        }
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" UE context: %p\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              ue_context_p);

        if (ue_context_p != NULL) {


//#if defined(ENABLE_ITTI)
          ue_context_p->ue_context.establishment_cause_NB = rrcConnectionRequest_NB->establishmentCause_r13;
	  if (stmsi_received==0){
	    LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept new connection from UE random UE identity (0x%" PRIx64 ") MME code %u TMSI %u cause %ld\n",
		  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
		  ue_context_p->ue_context.random_ue_identity,
		  ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
		  ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
		  ue_context_p->ue_context.establishment_cause_NB);
	  }
	  else{
	    LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept new connection from UE  MME code %u TMSI %u cause %ld\n",
		  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
		  ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
		  ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
		  ue_context_p->ue_context.establishment_cause_NB);
	  }
//#else
//          LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept new connection for UE random UE identity (0x%" PRIx64 ")\n",
//                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
//                ue_context_p->ue_context.random_ue_identity);
//#endif
          if (stmsi_received == 0)
	    eNB_rrc_inst_NB[ctxt_pP->module_id].Nb_ue++;

        } else {
          // no context available
          LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Can't create new context for UE random UE identity (0x%" PRIx64 ")\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                random_value);
          NB_rrc_mac_remove_ue(ctxt_pP->module_id,ctxt_pP->rnti);
          return -1;
        }
      }

//XXX MP: RRM not used
//#ifndef NO_RRM
//      send_msg(&S_rrc, msg_rrc_MR_attach_ind(ctxt_pP->module_id, Mac_id));
//#else

      ue_context_p->ue_context.primaryCC_id = CC_id;

      // SRB1bis (LCID = 3 = DCCH0)
      ue_context_p->ue_context.Srb1bis.Active = 1;
      ue_context_p->ue_context.Srb1bis.Srb_info.Srb_id = DCCH0;
      memcpy(&ue_context_p->ue_context.Srb1bis.Srb_info.Lchan_desc[0],
             &DCCH_LCHAN_DESC,//FIXME MP: LCHAN_DESC (mac_rrc_primitives) to be changed
             LCHAN_DESC_SIZE);
      memcpy(&ue_context_p->ue_context.Srb1bis.Srb_info.Lchan_desc[1],
             &DCCH_LCHAN_DESC,
             LCHAN_DESC_SIZE);

      //generate RRCConnectionSetup-NB
      rrc_eNB_generate_RRCConnectionSetup_NB(ctxt_pP, ue_context_p, CC_id);

      LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT"CALLING RLC CONFIG SRB1 (rbid %d)\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            Idx);

      MSC_LOG_TX_MESSAGE(
        MSC_RRC_ENB,
        MSC_PDCP_ENB,
        NULL,
        0,
        MSC_AS_TIME_FMT" CONFIG_REQ UE %x SRB",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti);

      //XXX we should not configure PDCP for SRB1bis but only for SRB1 (we do it now?)
      NB_rrc_pdcp_config_asn1_req(ctxt_pP,
                               ue_context_p->ue_context.SRB_configList, //contain SRB1bis but used as SRB1
                               (DRB_ToAddModList_NB_r13_t *) NULL,
                               (DRB_ToReleaseList_NB_r13_t*) NULL,
                               0xff,
                               NULL,
                               NULL,
                               NULL,
                               NULL,
							   DCCH1);

      ///MP: Configure RLC for SRB1bis
      NB_rrc_rlc_config_asn1_req(ctxt_pP,
                              ue_context_p->ue_context.SRB_configList,
                              (DRB_ToAddModList_NB_r13_t*) NULL,
                              (DRB_ToReleaseList_NB_r13_t*) NULL,
							  SRB1BIS_FLAG_YES
                             );
      //MP: Configure RLC for SRB1
            NB_rrc_rlc_config_asn1_req(ctxt_pP,
                                    ue_context_p->ue_context.SRB_configList,
                                    (DRB_ToAddModList_NB_r13_t*) NULL,
                                    (DRB_ToReleaseList_NB_r13_t*) NULL,
      							  SRB1BIS_FLAG_NO
                                   );
//#endif //NO_RRM

      break; //RRCConnectionSetup-NB

    case UL_CCCH_MessageType_NB__c1_PR_rrcConnectionResumeRequest_r13:
    	// TS 36.331 accept the resume, or switch back to a Connection setup or reject the request

    	T(T_ENB_RRC_CONNECTION_RESUME_REQUEST, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    	        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

    	#ifdef RRC_MSG_PRINT
    	      LOG_F(RRC,"[MSG] RRCConnectionResumeRequest-NB\n");

    	      for (i = 0; i < Srb_info->Rx_buffer.payload_size; i++) {
    	        LOG_F(RRC,"%02x ", ((uint8_t*)Srb_info->Rx_buffer.Payload)[i]);
    	      }

    	      LOG_F(RRC,"\n");
    	#endif

    LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT"MAC_eNB--- MAC_DATA_IND (rrcConnectionResumeRequest-NB on SRB0) --> RRC_eNB\n",
    	                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    rrcConnectionResumeRequest_NB=
    	              &ul_ccch_msg_NB->message.choice.c1.choice.rrcConnectionResumeRequest_r13.criticalExtensions.choice.rrcConnectionResumeRequest_r13;

    LOG_I(RRC,PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionResumeRequest-NB cause %s\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                ((rrcConnectionResumeRequest_NB->resumeCause_r13 == EstablishmentCause_NB_r13_mt_Access) ?    "mt Access" :
                 (rrcConnectionResumeRequest_NB->resumeCause_r13 == EstablishmentCause_NB_r13_mo_Signalling) ? "mo Signalling" :
                 (rrcConnectionResumeRequest_NB->resumeCause_r13 == EstablishmentCause_NB_r13_mo_Data) ? "mo Data":
                (rrcConnectionResumeRequest_NB->resumeCause_r13 == EstablishmentCause_NB_r13_mo_ExceptionData) ? "mo Exception data" :
                "delay tollerant Access v1330"));

    //only reject for now
    rrc_eNB_generate_RRCConnectionReject_NB(ctxt_pP,
                                 rrc_eNB_get_ue_context_NB(&eNB_rrc_inst_NB[ctxt_pP->module_id], ctxt_pP->rnti),
                                 CC_id);
      break;

    default:
      LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Unknown message\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
      rval = -1;
      break;
    }

    rval = 0;
  } else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT"  Unknown error \n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    rval = -1;
  }

  return rval;
}

//-----------------------------------------------------------------------------
int
rrc_eNB_decode_dcch_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const rb_id_t                Srb_id, //Dcch_index
  const uint8_t*    const      Rx_sdu,
  const sdu_size_t             sdu_sizeP
)
//-----------------------------------------------------------------------------
{

  asn_dec_rval_t                      dec_rval;
  UL_DCCH_Message_NB_t                  *ul_dcch_msg_NB = NULL;
  UE_Capability_NB_r13_t              *UE_Capability_NB = NULL;
  int i;
  struct rrc_eNB_ue_context_NB_s*        ue_context_p = NULL;

  int dedicated_DRB=0;

  T(T_ENB_RRC_UL_DCCH_DATA_IN, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  //FIXME : depends on how SRBs are managed
  if ((Srb_id != 1) && (Srb_id != 3)) {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received message on SRB%d, should not have ...\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          Srb_id);
  } else {
    LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received message on SRB%d\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          Srb_id);
  }

  LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Decoding UL-DCCH Message-NB\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
  dec_rval = uper_decode(
               NULL,
               &asn_DEF_UL_DCCH_Message_NB,
               (void**)&ul_dcch_msg_NB,
               Rx_sdu,
               sdu_sizeP,
               0,
               0);

  //TODO:delete?
#if defined(ENABLE_ITTI)
#   if defined(DISABLE_ITTI_XER_PRINT)
  {
    MessageDef                         *message_p;

    message_p = itti_alloc_new_message(TASK_RRC_ENB, RRC_UL_DCCH_MESSAGE);
    memcpy(&message_p->ittiMsg, (void *)ul_dcch_msg, sizeof(RrcUlDcchMessage));

    itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, message_p);
  }
#   else
  {
    char                                message_string[10000];
    size_t                              message_string_size;

    if ((message_string_size =
           xer_sprint(message_string, sizeof(message_string), &asn_DEF_UL_DCCH_Message_NB, (void *)ul_dcch_msg_NB)) >= 0) {
      MessageDef                         *msg_p;

      msg_p = itti_alloc_new_message_sized(TASK_RRC_ENB, RRC_UL_DCCH, message_string_size + sizeof(IttiMsgText));
      msg_p->ittiMsg.rrc_ul_dcch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_ul_dcch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);
    }
  }
#   endif
#endif


  {
    for (i = 0; i < sdu_sizeP; i++) {
      LOG_T(RRC, "%x.", Rx_sdu[i]);
    }

    LOG_T(RRC, "\n");
  }

  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Failed to decode UL-DCCH (%zu bytes)\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          dec_rval.consumed);
    return -1;
  }

  ue_context_p = rrc_eNB_get_ue_context_NB(
                   &eNB_rrc_inst_NB[ctxt_pP->module_id],
                   ctxt_pP->rnti);

  if (ul_dcch_msg_NB->message.present == UL_DCCH_MessageType_NB_PR_c1) {

    switch (ul_dcch_msg_NB->message.choice.c1.present) {
    case UL_DCCH_MessageType_NB__c1_PR_NOTHING:   /* No components present */
      break;

      //no measurement reports

    case UL_DCCH_MessageType_NB__c1_PR_rrcConnectionReconfigurationComplete_r13:

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRCConnectionReconfigurationComplete-NB\n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif
      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" RRCConnectionReconfigurationComplete-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(RRCConnectionReconfigurationComplete-NB) ---> RRC_eNB]\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH1, //through SRB1
            sdu_sizeP);

      if (ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionReconfigurationComplete_r13.criticalExtensions.
          present == RRCConnectionReconfigurationComplete_NB__criticalExtensions_PR_rrcConnectionReconfigurationComplete_r13) {
	/*NN: revise the condition */
    	  //XXX MP: RRC_RECONFIGURED indicate if the default/dedicated bearer has been/has been not established
        if (ue_context_p->ue_context.Status == RRC_RECONFIGURED){ // a dedicated bearers has been established
	  dedicated_DRB = 1;
	  LOG_I(RRC,
		PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (dedicated DRB, xid %ld)\n",
		PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionReconfigurationComplete_r13.rrc_TransactionIdentifier);
	}else { //a default bearer has been established
	  dedicated_DRB = 0;
	  ue_context_p->ue_context.Status = RRC_RECONFIGURED;
	  LOG_I(RRC,
		PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (default DRB, xid %ld)\n",
		PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionReconfigurationComplete_r13.rrc_TransactionIdentifier);
	}
	rrc_eNB_process_RRCConnectionReconfigurationComplete_NB(
          ctxt_pP,
          ue_context_p,
	  ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionReconfigurationComplete_r13.rrc_TransactionIdentifier);

#if defined(FLEXRAN_AGENT_SB_IF)
	//WARNING:Inform the controller about the UE activation. Should be moved to RRC agent in the future
	if (mac_agent_registered[ctxt_pP->module_id]) {
	  agent_mac_xface[ctxt_pP->eNB_index]->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
										ue_context_p->ue_id_rnti,
										PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_UPDATED);
	}
#endif
      }

//#if defined(ENABLE_ITTI)...
#if defined(ENABLE_USE_MME)
      if (EPC_MODE_ENABLED == 1) {
	if (dedicated_DRB == 1){
	  rrc_eNB_send_S1AP_E_RAB_SETUP_RESP(ctxt_pP,
					     ue_context_p,
					     ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionReconfigurationComplete_r13.rrc_TransactionIdentifier);
	}else {
	  rrc_eNB_send_S1AP_INITIAL_CONTEXT_SETUP_RESP(ctxt_pP,
						       ue_context_p);
	}
      }
#else  // MP: ENABLE_USE_MME ??
      if (dedicated_DRB == 0 ) {
	rrc_eNB_reconfigure_DRBs_NB(ctxt_pP,ue_context_p); //MP: establish a dedicated DRB
      }
#endif

      break;

    case UL_DCCH_MessageType_NB__c1_PR_rrcConnectionReestablishmentComplete_r13:
    	//for the moment will be not possible since we reject any connectionReestablishmentRequest

      T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRCConnectionReestablishmentComplete-NB \n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif

      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" rrcConnectionReestablishmentComplete-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(rrcConnectionReestablishmentComplete-NB) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH,
            sdu_sizeP);
      break;

    case UL_DCCH_MessageType_NB__c1_PR_rrcConnectionSetupComplete_r13:
    	//XXX MP: Ts 36.331 V14.2.1 RRCConnectionSetupComplete is transmitted over SRB1bis (pag 585)

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRCConnectionSetupComplete-NB\n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif

      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" RRCConnectionSetupComplete-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(RRCConnectionSetupComplete-NB) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH0,//SRB1bis
            sdu_sizeP);

      if (ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionSetupComplete_r13.criticalExtensions.present ==
    		  RRCConnectionSetupComplete_NB__criticalExtensions_PR_rrcConnectionSetupComplete_r13) {

          rrc_eNB_process_RRCConnectionSetupComplete_NB(
            ctxt_pP,
            ue_context_p,
            &ul_dcch_msg_NB->message.choice.c1.choice.rrcConnectionSetupComplete_r13.criticalExtensions.choice.rrcConnectionSetupComplete_r13);

          ue_context_p->ue_context.Status = RRC_CONNECTED;

          LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_CONNECTED \n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

#if defined(FLEXRAN_AGENT_SB_IF)
	  //WARNING:Inform the controller about the UE activation. Should be moved to RRC agent in the future
	  if (mac_agent_registered[ctxt_pP->module_id]) {
	    agent_mac_xface[ctxt_pP->eNB_index]->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
										  ue_context_p->ue_id_rnti,
										  PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_ACTIVATED);
	  }
#endif

      }

      ue_context_p->ue_context.ue_release_timer=0;

      break;

    case UL_DCCH_MessageType_NB__c1_PR_securityModeComplete_r13:

    	/* R2-163262 3GPP NB-IOT Ad-hoc Meeting #2
    	 * After receiving the SMC and performing the security activation, the UE shall use the SRB1.
    	 * If the UE fails the security activation, the UE shall use the SRB1-bis for Security Mode Complete message failure
    	 */

      T(T_ENB_RRC_SECURITY_MODE_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRC SecurityModeComplete-NB\n");

      for (i = 0; i < sdu_sizeP; i++) eNB->pusch_vars[UE_id]{
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif

      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" securityModeComplete UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" received securityModeComplete-NB on UL-DCCH %d from UE\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH);
      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(securityModeComplete-NB) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH0,//through SRB1bis
            sdu_sizeP);
#ifdef XER_PRINT
      xer_fprint(stdout, &asn_DEF_UL_DCCH_Message_NB, (void *)ul_dcch_msg_NB);
#endif
      //XXX MP: ???
      // confirm with PDCP about the security mode for DCCH
      //rrc_pdcp_config_req (enb_mod_idP, frameP, 1,CONFIG_ACTION_SET_SECURITY_MODE, (ue_mod_idP * NB_RB_MAX) + DCCH, 0x77);
      // continue the procedure
      rrc_eNB_generate_UECapabilityEnquiry_NB(ctxt_pP, ue_context_p);

      break;

    case UL_DCCH_MessageType_NB__c1_PR_securityModeFailure_r13:

    	//MP: Security Mode failure should be received over SRB1bis since the security activation fails
    	/*(see R2-163262)
    	The SRB1-bis (no PDCP) is used for SecurityModeCommand and SecurityModeFailure messages,
    	The SRB1 (full PDCP) is used for SecurityModeComplete message.*/

    	/*
    	 * Furthermore from TS 36.331 ch:5.3.4.3 Reception of the SecurityModeCommand by the UE
    	 * if the SecurityModeCommand message passes the integrity protection check
    	 * ....
    	 * else
    	 * 2> continue using the configuration used prior to the reception of the SecurityModeCommand message,
    	 *  i.e. neither apply integrity protection nor ciphering.
    	 * 2> submit the SecurityModeFailure message to lower layers for transmission, upon which the procedure ends
    	 */

      T(T_ENB_RRC_SECURITY_MODE_FAILURE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRC SecurityModeFailure-NB\n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif

      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" securityModeFailure-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_W(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(securityModeFailure-NB) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH0,
            sdu_sizeP);

#ifdef XER_PRINT
      xer_fprint(stdout, &asn_DEF_UL_DCCH_Message_NB, (void *)ul_dcch_msg_NB);
#endif
      // cancel the security mode in PDCP??
      // followup with the remaining procedure
//#warning "LG Removed rrc_eNB_generate_UECapabilityEnquiry after receiving securityModeFailure"
//XXX MP: make sense to go ahead with the procedure without security? --> means that we should continue to use SRB1bis?
      rrc_eNB_generate_UECapabilityEnquiry_NB(ctxt_pP, ue_context_p);

      break;

    case UL_DCCH_MessageType_NB__c1_PR_ueCapabilityInformation_r13:

    	//XXX message received over SRB1

      T(T_ENB_RRC_UE_CAPABILITY_INFORMATION, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRC UECapablilityInformation-NB \n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif

      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" ueCapabilityInformation-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" received ueCapabilityInformation-NB on UL-DCCH %d from UE\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH1);
      LOG_D(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(UECapabilityInformation) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH1,
            sdu_sizeP);
#ifdef XER_PRINT
      xer_fprint(stdout, &asn_DEF_UL_DCCH_Message_NB, (void *)ul_dcch_msg);
#endif
      LOG_I(RRC, "got UE capabilities for UE %x\n", ctxt_pP->rnti);


      //FIXME MP: ueCapabilityInformation different w.r.t LTE --> how to decode it?
      dec_rval = uper_decode(NULL,
                             &asn_DEF_UE_Capability_NB_r13,
                             (void **)&UE_Capability_NB,
							 Rx_sdu, //*buffer//FIXME may this is not the best way (may incorrect)
							 sdu_sizeP,//*size //FIXME may this is not the best way (may incorrect)
							 0, //skip bits
							 0); //unused bits

#if defined(ENABLE_USE_MME)

      if (EPC_MODE_ENABLED == 1) {
        rrc_eNB_send_S1AP_UE_CAPABILITIES_IND(ctxt_pP,
                                              ue_context_p,
                                              ul_dcch_msg_NB);
      }
#else
      ue_context_p->ue_context.nb_of_e_rabs = 1;
      //FIXME may no more present in NB_IoT or different parameter to set
      for (i = 0; i < ue_context_p->ue_context.nb_of_e_rabs; i++){
	ue_context_p->ue_context.e_rab[i].status = E_RAB_NB_STATUS_NEW;
	ue_context_p->ue_context.e_rab[i].param.e_rab_id = 1+i;
	ue_context_p->ue_context.e_rab[i].param.qos.qci=9; //Non-GBR the default value (TS
      }
      ue_context_p->ue_context.setup_e_rabs =ue_context_p->ue_context.nb_of_e_rabs;
#endif

      rrc_eNB_generate_defaultRRCConnectionReconfiguration_NB(ctxt_pP, ue_context_p);//no HO_FLAG

      break;

  ///no ULHandoverPreparationTransfer

    case UL_DCCH_MessageType_NB__c1_PR_ulInformationTransfer_r13:
      T(T_ENB_RRC_UL_INFORMATION_TRANSFER, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

#ifdef RRC_MSG_PRINT
      LOG_F(RRC,"[MSG] RRC ULInformationTransfer-NB \n");

      for (i = 0; i < sdu_sizeP; i++) {
        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
      }

      LOG_F(RRC,"\n");
#endif


      MSC_LOG_RX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        Rx_sdu,
        sdu_sizeP,
        MSC_AS_TIME_FMT" ulInformationTransfer-NB UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_p->ue_context.rnti,
        sdu_sizeP);

#if defined(ENABLE_USE_MME)

      if (EPC_MODE_ENABLED == 1) {
        rrc_eNB_send_S1AP_UPLINK_NAS(ctxt_pP,
                                     ue_context_p,
                                     ul_dcch_msg_NB);
      }

#endif
      break;

  ///no CounterCheckResponse

    case UL_DCCH_MessageType_NB__c1_PR_rrcConnectionResumeComplete_r13:

    	//will be not possible because for the moment we reject any ConnectionResumeRequest

    	T(T_ENB_RRC_CONNECTION_RESUME_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    	        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

    	#ifdef RRC_MSG_PRINT
    	      LOG_F(RRC,"[MSG] RRCConnectionResumeComplete-NB \n");

    	      for (i = 0; i < sdu_sizeP; i++) {
    	        LOG_F(RRC,"%02x ", ((uint8_t*)Rx_sdu)[i]);
    	      }

    	      LOG_F(RRC,"\n");
    	#endif

    	      MSC_LOG_RX_MESSAGE(
    	        MSC_RRC_ENB,
    	        MSC_RRC_UE,
    	        Rx_sdu,
    	        sdu_sizeP,
    	        MSC_AS_TIME_FMT" rrcConnectionResumeComplete-NB UE %x size %u",
    	        MSC_AS_TIME_ARGS(ctxt_pP),
    	        ue_context_p->ue_context.rnti,
    	        sdu_sizeP);

    	      LOG_I(RRC,
    	            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
    	            "(rrcConnectionResumeComplete-NB) ---> RRC_eNB\n",
    	            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
    	            DCCH, //check
    	            sdu_sizeP);

	break;

    default:

      T(T_ENB_RRC_UNKNOW_MESSAGE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
        T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

      LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Unknown message %s:%u\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            __FILE__, __LINE__);
      return -1;
    }

    return 0;
  } else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Unknown error %s:%u\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          __FILE__, __LINE__);
    return -1;
  }

}

//-----------------------------------------------------------------------------
//put out from the ITTI FIXME is completely based on ITTI--> must be changed (msg_p, itti_receive_msg ecc...)
void*
rrc_enb_task_NB(
  void* args_p
)
//-----------------------------------------------------------------------------
{
  MessageDef                         *msg_p;
  const char                         *msg_name_p;
  instance_t                          instance;
  int                                 result;
  SRB_INFO_NB                           *srb_info_p;
  int                                 CC_id;

  protocol_ctxt_t                     ctxt;
  itti_mark_task_ready(TASK_RRC_ENB);

  while (1) {
    // Wait for a message
    itti_receive_msg(TASK_RRC_ENB, &msg_p);

    msg_name_p = ITTI_MSG_NAME(msg_p);
    instance = ITTI_MSG_INSTANCE(msg_p);

    switch (ITTI_MSG_ID(msg_p)) {
    case TERMINATE_MESSAGE:
      itti_exit_task();
      break;

    case MESSAGE_TEST:
      LOG_I(RRC, "[eNB %d] Received %s\n", instance, msg_name_p);
      break;

      /* Messages from MAC */
    case RRC_MAC_CCCH_DATA_IND:
      PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                    instance,
                                    ENB_FLAG_YES,
                                    RRC_MAC_CCCH_DATA_IND(msg_p).rnti,
                                    msg_p->ittiMsgHeader.lte_time.frame,
                                    msg_p->ittiMsgHeader.lte_time.slot);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received %s\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(&ctxt),
            msg_name_p);

      CC_id = RRC_MAC_CCCH_DATA_IND(msg_p).CC_id;

      ///SRB0
      srb_info_p = &eNB_rrc_inst_NB[instance].carrier[CC_id].Srb0;

      memcpy(srb_info_p->Rx_buffer.Payload,
             RRC_MAC_CCCH_DATA_IND(msg_p).sdu,
             RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size);

      srb_info_p->Rx_buffer.payload_size = RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size;

      rrc_eNB_decode_ccch_NB(&ctxt, srb_info_p, CC_id); //SRB0

      break;

      /* Messages from PDCP */
    case RRC_DCCH_DATA_IND:
      PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                    instance,
                                    ENB_FLAG_YES,
                                    RRC_DCCH_DATA_IND(msg_p).rnti,
                                    msg_p->ittiMsgHeader.lte_time.frame,
                                    msg_p->ittiMsgHeader.lte_time.slot);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received on DCCH %d %s\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(&ctxt),
            RRC_DCCH_DATA_IND(msg_p).dcch_index,
            msg_name_p);
      rrc_eNB_decode_dcch_NB(&ctxt,
                          RRC_DCCH_DATA_IND(msg_p).dcch_index, //3 if SRB1bis and 1 for SRB1
                          RRC_DCCH_DATA_IND(msg_p).sdu_p, //rx_sdu
                          RRC_DCCH_DATA_IND(msg_p).sdu_size);

      // Message buffer has been processed, free it now.
      result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), RRC_DCCH_DATA_IND(msg_p).sdu_p);
      AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
      break;

#if defined(ENABLE_USE_MME)

      /* Messages from S1AP */
    case S1AP_DOWNLINK_NAS:
      rrc_eNB_process_S1AP_DOWNLINK_NAS(msg_p, msg_name_p, instance, &rrc_eNB_mui);
      break;

    case S1AP_INITIAL_CONTEXT_SETUP_REQ:
      rrc_eNB_process_S1AP_INITIAL_CONTEXT_SETUP_REQ(msg_p, msg_name_p, instance);
      break;

    case S1AP_UE_CTXT_MODIFICATION_REQ:
      rrc_eNB_process_S1AP_UE_CTXT_MODIFICATION_REQ(msg_p, msg_name_p, instance);
      break;

    case S1AP_PAGING_IND:
      LOG_E(RRC, "[eNB %d] Received not yet implemented message %s\n", instance, msg_name_p);
      break;

    case S1AP_E_RAB_SETUP_REQ:
      rrc_eNB_process_S1AP_E_RAB_SETUP_REQ(msg_p, msg_name_p, instance);
      LOG_D(RRC, "[eNB %d] Received the message %s\n", instance, msg_name_p);
      break;

    case S1AP_UE_CONTEXT_RELEASE_REQ:
      rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_REQ(msg_p, msg_name_p, instance);
      break;

    case S1AP_UE_CONTEXT_RELEASE_COMMAND:
      rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_COMMAND(msg_p, msg_name_p, instance);
      break;

    case GTPV1U_ENB_DELETE_TUNNEL_RESP:
      /* Nothing to do. Apparently everything is done in S1AP processing */
      //LOG_I(RRC, "[eNB %d] Received message %s, not processed because procedure not synched\n",
      //instance, msg_name_p);
      break;

#   endif

      /* Messages from eNB app */
    case RRC_CONFIGURATION_REQ:
      LOG_I(RRC, "[eNB %d] Received %s\n", instance, msg_name_p);
      openair_rrc_eNB_configuration_NB(ENB_INSTANCE_TO_MODULE_ID(instance), &RRC_CONFIGURATION_REQ(msg_p));
      break;

    default:
      LOG_E(RRC, "[eNB %d] Received unexpected message %s\n", instance, msg_name_p);
      break;
    }

    result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
    msg_p = NULL;
  }
}


#ifndef USER_MODE
EXPORT_SYMBOL(Rlc_info_am_config);
#endif
