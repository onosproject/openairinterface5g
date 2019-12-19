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

#include "nr_pdcp_ue_manager.h"

/* from OAI */
#include "pdcp.h"

#define TODO do { \
    printf("%s:%d:%s: todo\n", __FILE__, __LINE__, __FUNCTION__); \
    exit(1); \
  } while (0)

static nr_pdcp_ue_manager_t *nr_pdcp_ue_manager;

/* necessary globals for OAI, not used internally */
hash_table_t  *pdcp_coll_p;
static uint64_t pdcp_optmask;

/****************************************************************************/
/* rlc_data_req queue - begin                                               */
/****************************************************************************/

static void init_nr_rlc_data_req_queue(void)
{
}

/****************************************************************************/
/* rlc_data_req queue - end                                                 */
/****************************************************************************/

#include "LAYER2/MAC/mac_extern.h"

void nr_ip_over_LTE_DRB_preconfiguration(void)
{
          // Addition for the use-case of 4G stack on top of 5G-NR.
          // We need to configure pdcp and rlc instances without having an actual
          // UE RRC Connection. In order to be able to test the NR PHY with some injected traffic
          // on top of the LTE stack.
          protocol_ctxt_t ctxt;
          LTE_DRB_ToAddModList_t*                DRB_configList=NULL;
          DRB_configList = CALLOC(1, sizeof(LTE_DRB_ToAddModList_t));
          struct LTE_LogicalChannelConfig        *DRB_lchan_config                                 = NULL;
          struct LTE_RLC_Config                  *DRB_rlc_config                   = NULL;
          struct LTE_PDCP_Config                 *DRB_pdcp_config                  = NULL;
          struct LTE_PDCP_Config__rlc_UM         *PDCP_rlc_UM                      = NULL;

          struct LTE_DRB_ToAddMod                *DRB_config                       = NULL;
          struct LTE_LogicalChannelConfig__ul_SpecificParameters *DRB_ul_SpecificParameters        = NULL;
          long  *logicalchannelgroup_drb;


          //Static preconfiguration of DRB
          DRB_config = CALLOC(1, sizeof(*DRB_config));

          DRB_config->eps_BearerIdentity = CALLOC(1, sizeof(long));
          // allowed value 5..15, value : x+4
          *(DRB_config->eps_BearerIdentity) = 1; //ue_context_pP->ue_context.e_rab[i].param.e_rab_id;//+ 4; // especial case generation
          //   DRB_config->drb_Identity =  1 + drb_identity_index + e_rab_done;// + i ;// (DRB_Identity_t) ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
          // 1 + drb_identiy_index;
          DRB_config->drb_Identity = 1;
          DRB_config->logicalChannelIdentity = CALLOC(1, sizeof(long));
          *(DRB_config->logicalChannelIdentity) = DRB_config->drb_Identity + 2; //(long) (ue_context_pP->ue_context.e_rab[i].param.e_rab_id + 2); // value : x+2

          DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
          DRB_config->rlc_Config = DRB_rlc_config;

          DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
          DRB_config->pdcp_Config = DRB_pdcp_config;
          DRB_pdcp_config->discardTimer = CALLOC(1, sizeof(long));
          *DRB_pdcp_config->discardTimer = LTE_PDCP_Config__discardTimer_infinity;
          DRB_pdcp_config->rlc_AM = NULL;
          DRB_pdcp_config->rlc_UM = NULL;

          DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
          DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
          DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
          DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
          // PDCP
          PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
          DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
          PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;

          DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;

          DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
          DRB_config->logicalChannelConfig = DRB_lchan_config;
          DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
          DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;

          DRB_ul_SpecificParameters->priority= 4;

          DRB_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8;
          //LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
          DRB_ul_SpecificParameters->bucketSizeDuration =
          LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;

          logicalchannelgroup_drb = CALLOC(1, sizeof(long));
          *logicalchannelgroup_drb = 1;//(i+1) % 3;
          DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;

          ASN_SEQUENCE_ADD(&DRB_configList->list,DRB_config);

          if (ENB_NAS_USE_TUN){
                  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, 0, ENB_FLAG_YES, 0x1234, 0, 0,0);
          }
          else{
                  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, 0, ENB_FLAG_NO, 0x1234, 0, 0,0);
          }

          rrc_pdcp_config_asn1_req(&ctxt,
                       (LTE_SRB_ToAddModList_t *) NULL,
                       DRB_configList,
                       (LTE_DRB_ToReleaseList_t *) NULL,
                       0xff, NULL, NULL, NULL
                       , (LTE_PMCH_InfoList_r9_t *) NULL,
                       &DRB_config->drb_Identity);

        rrc_rlc_config_asn1_req(&ctxt,
                       (LTE_SRB_ToAddModList_t*)NULL,
                       DRB_configList,
                       (LTE_DRB_ToReleaseList_t*)NULL
        //#if (RRC_VERSION >= MAKE_VERSION(10, 0, 0))
                       ,(LTE_PMCH_InfoList_r9_t *)NULL
                       , 0, 0
        //#endif
                 );
}

int pdcp_fifo_flush_sdus(const protocol_ctxt_t *const ctxt_pP)
{
  TODO;
  return 0;
}

void pdcp_layer_init(void)
{
  /* be sure to initialize only once */
  static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  static int initialized = 0;
  if (pthread_mutex_lock(&m) != 0) abort();
  if (initialized) {
    if (pthread_mutex_unlock(&m) != 0) abort();
    return;
  }
  initialized = 1;
  if (pthread_mutex_unlock(&m) != 0) abort();

  nr_pdcp_ue_manager = new_nr_pdcp_ue_manager(1);
  init_nr_rlc_data_req_queue();
}

uint64_t pdcp_module_init(uint64_t _pdcp_optmask)
{
  pdcp_optmask = _pdcp_optmask;
  return pdcp_optmask;
}

static void deliver_sdu_drb(void *_ue, nr_pdcp_entity_t *entity,
                            char *buf, int size)
{
}

static void deliver_pdu_drb(void *_ue, nr_pdcp_entity_t *entity,
                            char *buf, int size, int sdu_id)
{
}

boolean_t pdcp_data_ind(
  const protocol_ctxt_t *const  ctxt_pP,
  const srb_flag_t srb_flagP,
  const MBMS_flag_t MBMS_flagP,
  const rb_id_t rb_id,
  const sdu_size_t sdu_buffer_size,
  mem_block_t *const sdu_buffer)
{
  TODO;
  return 1;
}

void pdcp_run(const protocol_ctxt_t *const  ctxt_pP)
{
  MessageDef      *msg_p;
  int             result;
  protocol_ctxt_t ctxt;

  while (1) {
    itti_poll_msg(ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE, &msg_p);
    if (msg_p == NULL)
      break;
    switch (ITTI_MSG_ID(msg_p)) {
    case RRC_DCCH_DATA_REQ:
      PROTOCOL_CTXT_SET_BY_MODULE_ID(
          &ctxt,
          RRC_DCCH_DATA_REQ(msg_p).module_id,
          RRC_DCCH_DATA_REQ(msg_p).enb_flag,
          RRC_DCCH_DATA_REQ(msg_p).rnti,
          RRC_DCCH_DATA_REQ(msg_p).frame,
          0,
          RRC_DCCH_DATA_REQ(msg_p).eNB_index);
      result = pdcp_data_req(&ctxt,
                             SRB_FLAG_YES,
                             RRC_DCCH_DATA_REQ(msg_p).rb_id,
                             RRC_DCCH_DATA_REQ(msg_p).muip,
                             RRC_DCCH_DATA_REQ(msg_p).confirmp,
                             RRC_DCCH_DATA_REQ(msg_p).sdu_size,
                             RRC_DCCH_DATA_REQ(msg_p).sdu_p,
                             RRC_DCCH_DATA_REQ(msg_p).mode,
                             NULL, NULL);

      if (result != TRUE)
        LOG_E(PDCP, "PDCP data request failed!\n");
      result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), RRC_DCCH_DATA_REQ(msg_p).sdu_p);
      AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
      break;
    default:
      LOG_E(PDCP, "Received unexpected message %s\n", ITTI_MSG_NAME(msg_p));
      break;
    }
  }
}

static void add_srb(int rnti, struct LTE_SRB_ToAddMod *s)
{
  TODO;
}

static void add_drb_am(int rnti, struct LTE_DRB_ToAddMod *s)
{
  nr_pdcp_entity_t *pdcp_drb;
  nr_pdcp_ue_t *ue;

  int drb_id = s->drb_Identity;

printf("\n\n################# add drb %d\n\n\n", drb_id);

  if (drb_id != 1) {
    LOG_E(PDCP, "%s:%d:%s: fatal, bad drb id %d\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id);
    exit(1);
  }

  nr_pdcp_manager_lock(nr_pdcp_ue_manager);
  ue = nr_pdcp_manager_get_ue(nr_pdcp_ue_manager, rnti);
  if (ue->drb[drb_id-1] != NULL) {
    LOG_D(PDCP, "%s:%d:%s: warning DRB %d already exist for ue %d, do nothing\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  } else {
    pdcp_drb = new_nr_pdcp_entity_drb_am(drb_id, deliver_sdu_drb, ue, deliver_pdu_drb, ue);
    nr_pdcp_ue_add_drb_pdcp_entity(ue, drb_id, pdcp_drb);

    LOG_D(PDCP, "%s:%d:%s: added drb %d to ue %d\n",
          __FILE__, __LINE__, __FUNCTION__, drb_id, rnti);
  }
  nr_pdcp_manager_unlock(nr_pdcp_ue_manager);
}

static void add_drb(int rnti, struct LTE_DRB_ToAddMod *s)
{
  switch (s->rlc_Config->present) {
  case LTE_RLC_Config_PR_am:
    add_drb_am(rnti, s);
    break;
  case LTE_RLC_Config_PR_um_Bi_Directional:
    //add_drb_um(rnti, s);
    /* hack */
    add_drb_am(rnti, s);
    break;
  default:
    LOG_E(PDCP, "%s:%d:%s: fatal: unhandled DRB type\n",
          __FILE__, __LINE__, __FUNCTION__);
    exit(1);
  }
}

boolean_t rrc_pdcp_config_asn1_req(
  const protocol_ctxt_t *const  ctxt_pP,
  LTE_SRB_ToAddModList_t  *const srb2add_list,
  LTE_DRB_ToAddModList_t  *const drb2add_list,
  LTE_DRB_ToReleaseList_t *const drb2release_list,
  const uint8_t                   security_modeP,
  uint8_t                  *const kRRCenc,
  uint8_t                  *const kRRCint,
  uint8_t                  *const kUPenc
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
  ,LTE_PMCH_InfoList_r9_t  *pmch_InfoList_r9
#endif
  ,rb_id_t                 *const defaultDRB)
{
  int rnti = ctxt_pP->rnti;
  int i;

  if (ctxt_pP->enb_flag != 1 ||
      ctxt_pP->module_id != 0 ||
      ctxt_pP->instance != 0 ||
      ctxt_pP->eNB_index != 0 ||
      //ctxt_pP->configured != 2 ||
      //srb2add_list == NULL ||
      //drb2add_list != NULL ||
      drb2release_list != NULL ||
      security_modeP != 255 ||
      //kRRCenc != NULL ||
      //kRRCint != NULL ||
      //kUPenc != NULL ||
      pmch_InfoList_r9 != NULL /*||
      defaultDRB != NULL */) {
    TODO;
  }

  if (srb2add_list != NULL) {
    for (i = 0; i < srb2add_list->list.count; i++) {
      add_srb(rnti, srb2add_list->list.array[i]);
    }
  }

  if (drb2add_list != NULL) {
    for (i = 0; i < drb2add_list->list.count; i++) {
      add_drb(rnti, drb2add_list->list.array[i]);
    }
  }

  /* update security */
  if (kRRCint != NULL) {
    /* todo */
  }

  free(kRRCenc);
  free(kRRCint);
  free(kUPenc);

  return 0;
}

uint64_t get_pdcp_optmask(void)
{
  return pdcp_optmask;
}

boolean_t pdcp_remove_UE(
  const protocol_ctxt_t *const  ctxt_pP)
{
  TODO;
  return 1;
}

void pdcp_config_set_security(const protocol_ctxt_t* const  ctxt_pP, pdcp_t *pdcp_pP, rb_id_t rb_id,
                              uint16_t lc_idP, uint8_t security_modeP, uint8_t *kRRCenc_pP, uint8_t *kRRCint_pP, uint8_t *kUPenc_pP)
{
  TODO;
}

boolean_t pdcp_data_req(
  protocol_ctxt_t  *ctxt_pP,
  const srb_flag_t srb_flagP,
  const rb_id_t rb_id,
  const mui_t muiP,
  const confirm_t confirmP,
  const sdu_size_t sdu_buffer_size,
  unsigned char *const sdu_buffer,
  const pdcp_transmission_mode_t mode
#if (LTE_RRC_VERSION >= MAKE_VERSION(14, 0, 0))
  ,const uint32_t *const sourceL2Id
  ,const uint32_t *const destinationL2Id
#endif
  )
{
  TODO;
  return 1;
}

void pdcp_set_pdcp_data_ind_func(pdcp_data_ind_func_t pdcp_data_ind)
{
  /* nothing to do */
}

void pdcp_set_rlc_data_req_func(send_rlc_data_req_func_t send_rlc_data_req)
{
  /* nothing to do */
}
