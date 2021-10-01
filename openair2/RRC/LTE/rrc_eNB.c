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

/*! \file rrc_eNB.c
 * \brief rrc procedures for eNB
 * \author Navid Nikaein and  Raymond Knopp
 * \date 2011 - 2014
 * \version 1.0
 * \company Eurecom
 * \email: navid.nikaein@eurecom.fr and raymond.knopp@eurecom.fr
 */
#define RRC_ENB
#define RRC_ENB_C
#include <asn_application.h>
#include <asn_internal.h> /* for _ASN_DEFAULT_STACK_MAX */
#include <per_encoder.h>
#include "rrc_defs.h"
#include "rrc_extern.h"
#include "assertions.h"
#include "common/ran_context.h"
#include "asn1_conversions.h"
#include "asn_internal.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
#include "LAYER2/RLC/rlc.h"
#include "LAYER2/MAC/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "COMMON/mac_rrc_primitives.h"
#include "RRC/LTE/MESSAGES/asn1_msg.h"
#include "LTE_RRCConnectionRequest.h"
#include "LTE_RRCConnectionReestablishmentRequest.h"
//#include "ReestablishmentCause.h"
#include "LTE_BCCH-BCH-Message.h"
#include "LTE_UL-CCCH-Message.h"
#include "LTE_DL-CCCH-Message.h"
#include "LTE_UL-DCCH-Message.h"
#include "LTE_DL-DCCH-Message.h"
#include "LTE_TDD-Config.h"
#include "LTE_HandoverPreparationInformation.h"
#include "LTE_HandoverCommand.h"
#include "rlc.h"
#include "rrc_eNB_UE_context.h"
#include "platform_types.h"
#include "msc.h"
#include "LTE_SL-CommConfig-r12.h"
#include "LTE_PeriodicBSR-Timer-r12.h"
#include "LTE_RetxBSR-Timer-r12.h"

#include "LTE_BCCH-BCH-Message-MBMS.h"
#include "LTE_BCCH-DL-SCH-Message-MBMS.h"
#include "LTE_SystemInformationBlockType1-MBMS-r14.h"
#include "LTE_NonMBSFN-SubframeConfig-r14.h"

#include "common/utils/LOG/vcd_signal_dumper.h"
#include "x2ap_eNB.h"

#include "T.h"
#include "LTE_MeasResults.h"

#include "RRC/NAS/nas_config.h"
#include "RRC/NAS/rb_config.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "UTIL/OSA/osa_defs.h"

#include "rrc_eNB_S1AP.h"
#include "rrc_eNB_GTPV1U.h"
#include "rrc_eNB_M2AP.h"

#include "pdcp.h"
#include "gtpv1u_eNB_task.h"

#include "intertask_interface.h"

#if ENABLE_RAL
  #include "rrc_eNB_ral.h"
#endif

#include "SIMULATION/TOOLS/sim.h" // for taus

#ifdef ENABLE_RAN_SLICING
#include "ric_agent.h"
#endif

#define ASN_MAX_ENCODE_SIZE 4096
#define NUMBEROF_DRBS_TOBE_ADDED 1
static int encode_CG_ConfigInfo(char *buffer,int buffer_size,rrc_eNB_ue_context_t *const ue_context_pP,int *enc_size);
static int is_en_dc_supported(LTE_UE_EUTRA_Capability_t *c);

extern RAN_CONTEXT_t RC;

#ifdef PHY_EMUL
  extern EMULATION_VARS              *Emul_vars;
#endif
extern eNB_MAC_INST                *eNB_mac_inst;
extern UE_MAC_INST                 *UE_mac_inst;

extern uint16_t                     two_tier_hexagonal_cellIds[7];

mui_t                               rrc_eNB_mui = 0;

extern uint32_t to_earfcn_DL(int eutra_bandP, uint32_t dl_CarrierFreq, uint32_t bw);
extern int rrc_eNB_process_security(const protocol_ctxt_t *const ctxt_pP, rrc_eNB_ue_context_t *const ue_context_pP, security_capabilities_t *security_capabilities_pP);
extern void process_eNB_security_key (const protocol_ctxt_t *const ctxt_pP, rrc_eNB_ue_context_t *const ue_context_pP, uint8_t *security_key_pP);
extern int derive_keNB_star(const uint8_t *kenb_32, const uint16_t pci, const uint32_t earfcn_dl, const bool is_rel8_only, uint8_t *kenb_star);
extern int rrc_eNB_generate_RRCConnectionReconfiguration_endc(protocol_ctxt_t *ctxt, rrc_eNB_ue_context_t *ue_context, unsigned char *buffer, int buffer_size, OCTET_STRING_t *scg_group_config,
    OCTET_STRING_t *scg_RB_config);
extern struct rrc_eNB_ue_context_s *get_first_ue_context(eNB_RRC_INST *rrc_instance_pP);

pthread_mutex_t      rrc_release_freelist;
RRC_release_list_t   rrc_release_info;
pthread_mutex_t      lock_ue_freelist;

#ifdef ENABLE_RIC_AGENT
eNB_RRC_KPI_STATS    rrc_kpi_stats;
#endif

#ifdef ENABLE_RAN_SLICING
extern uint8_t rsm_emm_event_trigger;
#endif

void
openair_rrc_on(
  const protocol_ctxt_t *const ctxt_pP
)
//-----------------------------------------------------------------------------
{
  int            CC_id;
  LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" ENB:OPENAIR RRC IN....\n",
        PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    rrc_config_buffer (&RC.rrc[ctxt_pP->module_id]->carrier[CC_id].SI, BCCH, 1);
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].SI.Active = 1;
  }
}

//-----------------------------------------------------------------------------
static void
init_SI(
  const protocol_ctxt_t *const ctxt_pP,
  const int              CC_id,
  RrcConfigurationReq *configuration
)
//-----------------------------------------------------------------------------
{
  int                                 i;
  LTE_SystemInformationBlockType1_v1310_IEs_t *sib1_v13ext=(LTE_SystemInformationBlockType1_v1310_IEs_t *)NULL;
  LOG_D(RRC,"%s()\n\n\n\n",__FUNCTION__);

  if(configuration->radioresourceconfig[CC_id].mbms_dedicated_serving_cell == TRUE) {
    LOG_I(RRC, "Configuring MIB FeMBMS (N_RB_DL %d)\n",
          (int)configuration->N_RB_DL[CC_id]);
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].MIB_FeMBMS = (uint8_t *) malloc16(4);
    do_MIB_FeMBMS(&RC.rrc[ctxt_pP->module_id]->carrier[CC_id],
                  configuration->N_RB_DL[CC_id],
                  0, //additionalNonMBSFN
                  0);
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sizeof_SIB1_MBMS = 0;
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].SIB1_MBMS = (uint8_t *) malloc16(32);
    AssertFatal(RC.rrc[ctxt_pP->module_id]->carrier[CC_id].SIB1_MBMS!=NULL,PROTOCOL_RRC_CTXT_FMT" init_SI: FATAL, no memory for SIB1_MBMS allocated\n",
                PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sizeof_SIB1_MBMS = do_SIB1_MBMS(&RC.rrc[ctxt_pP->module_id]->carrier[CC_id],ctxt_pP->module_id,CC_id,
        configuration
                                                                              );
    LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Contents of SIB1-MBMS\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP)
         );
    LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS freqBandIndicator_r14 %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->freqBandIndicator_r14
         );

    for (i = 0; i < RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->schedulingInfoList_MBMS_r14.list.count; i++) {
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS contents for Scheduling Info List %d/%d(partial)\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            i,
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->schedulingInfoList_MBMS_r14.list.count);
    }

    LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS SIB13-r14 contents for MBSFN subframe allocation (partial)\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP)
         );

    for (i = 0; i < RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.count; i++) {
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS SIB13-r14 contents for MBSFN sync area %d/%d (partial)\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            i,
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.count);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS MCCH Repetition Period: %ld (just index number, not real value)\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.mcch_RepetitionPeriod_r9);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS MCCH Offset: %ld\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.mcch_Offset_r9);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS MCCH Modification Period: %ld\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.mcch_ModificationPeriod_r9);
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS MCCH Signalling MCS: %ld\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.signallingMCS_r9);
      //LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS SIB13 sf_AllocInfo is = %x\n",
      //    PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
      //    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.sf_AllocInfo_r9.buf);

      if(RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->ext1) {
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS Subcarrier Spacing MBMS: %s\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              (*RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->systemInformationBlockType13_r14->mbsfn_AreaInfoList_r9.list.array[i]->ext1->subcarrierSpacingMBMS_r14 ==
               LTE_MBSFN_AreaInfo_r9__ext1__subcarrierSpacingMBMS_r14_khz_1dot25 ? "khz_1dot25": "khz_7dot5"));
      }
    }

    if(RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->nonMBSFN_SubframeConfig_r14) {
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS non MBSFN Subframe Config radioFrameAllocationPeriod-r14 %ld\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->nonMBSFN_SubframeConfig_r14->radioFrameAllocationPeriod_r14
           );
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS non MBSFN Subframe Config radioFrameAllocationOffset-r14 %ld\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->nonMBSFN_SubframeConfig_r14->radioFrameAllocationOffset_r14
           );
      LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB1-MBMS non MBSFN Subframe Config subframeAllocation-r14 is = %s\n",
            PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
            RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib1_MBMS->nonMBSFN_SubframeConfig_r14->subframeAllocation_r14.buf);
    }

    //AssertFatal(RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sizeof_SIB1 != 255,"FATAL, RC.rrc[enb_mod_idP].carrier[CC_id].sizeof_SIB1 == 255");
  }

  eNB_RRC_INST *rrc = RC.rrc[ctxt_pP->module_id];
  rrc_eNB_carrier_data_t *carrier=&rrc->carrier[CC_id];
  carrier->MIB = (uint8_t *) malloc16(4);
  carrier->sizeof_SIB1 = 0;
  carrier->sizeof_SIB23 = 0;
  carrier->SIB1 = (uint8_t *) malloc16(32);
  AssertFatal(carrier->SIB1!=NULL,PROTOCOL_RRC_CTXT_FMT" init_SI: FATAL, no memory for SIB1 allocated\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
  LOG_I(RRC,"[eNB %d] Node type %d \n ", ctxt_pP->module_id, rrc->node_type);

  if (NODE_IS_DU(rrc->node_type) || NODE_IS_MONOLITHIC(rrc->node_type)) {
    // copy basic Cell parameters
    carrier->physCellId      = configuration->Nid_cell[CC_id];
    carrier->p_eNB           = configuration->nb_antenna_ports[CC_id];
    carrier->Ncp             = configuration->prefix_type[CC_id];
    carrier->dl_CarrierFreq  = configuration->downlink_frequency[CC_id];
    carrier->ul_CarrierFreq  = configuration->downlink_frequency[CC_id]+ configuration->uplink_frequency_offset[CC_id];
    carrier->eutra_band      = configuration->eutra_band[CC_id];
    carrier->N_RB_DL         = configuration->N_RB_DL[CC_id];
    carrier->pbch_repetition = configuration->pbch_repetition[CC_id];
    LOG_I(RRC, "configuration->schedulingInfoSIB1_BR_r13[CC_id] %d\n",(int)configuration->schedulingInfoSIB1_BR_r13[CC_id]);
    LOG_I(RRC, "Configuring MIB (N_RB_DL %d,phich_Resource %d,phich_Duration %d)\n",
          (int)configuration->N_RB_DL[CC_id],
          (int)configuration->radioresourceconfig[CC_id].phich_resource,
          (int)configuration->radioresourceconfig[CC_id].phich_duration);
    carrier->sizeof_MIB = do_MIB(&rrc->carrier[CC_id],
                                 configuration->N_RB_DL[CC_id],
                                 configuration->radioresourceconfig[CC_id].phich_resource,
                                 configuration->radioresourceconfig[CC_id].phich_duration,
                                 0,
                                 configuration->schedulingInfoSIB1_BR_r13[CC_id]
                                );
    carrier->sizeof_SIB1 = do_SIB1(&rrc->carrier[CC_id],
                                   ctxt_pP->module_id,
                                   CC_id,
                                   FALSE,
                                   configuration
                                  );
    AssertFatal(carrier->sizeof_SIB1 != 255,"FATAL, RC.rrc[enb_mod_idP].carrier[CC_id].sizeof_SIB1 == 255");
    RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sizeof_SIB1_BR = 0;

    if (configuration->schedulingInfoSIB1_BR_r13[CC_id] > 0) {
      RC.rrc[ctxt_pP->module_id]->carrier[CC_id].SIB1_BR = (uint8_t *) malloc16(32);
      RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sizeof_SIB1_BR = do_SIB1(&RC.rrc[ctxt_pP->module_id]->carrier[CC_id],
          ctxt_pP->module_id,
          CC_id, TRUE, configuration);
    }
  }

  if (!NODE_IS_DU(rrc->node_type)) {
    carrier->SIB23 = (uint8_t *) malloc16(64);
    AssertFatal(carrier->SIB23!=NULL,"cannot allocate memory for SIB");
    carrier->sizeof_SIB23 = do_SIB23(ctxt_pP->module_id,
                                     CC_id,
                                     FALSE,
                                     configuration
                                    );
    LOG_I(RRC,"do_SIB23, size %d \n ", carrier->sizeof_SIB23);
    AssertFatal(carrier->sizeof_SIB23 != 255,"FATAL, RC.rrc[mod].carrier[CC_id].sizeof_SIB23 == 255");
    carrier->sizeof_SIB23_BR = 0;

    if (configuration->schedulingInfoSIB1_BR_r13[CC_id]>0) {
      carrier->SIB23_BR = (uint8_t *) malloc16(64);
      AssertFatal(carrier->SIB23_BR!=NULL,"cannot allocate memory for SIB");
      carrier->sizeof_SIB23_BR = do_SIB23(ctxt_pP->module_id, CC_id, TRUE, configuration);
    }

    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" SIB2/3 Contents (partial)\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.n_SB = %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.pusch_ConfigBasic.n_SB);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.hoppingMode = %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.pusch_ConfigBasic.hoppingMode);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.pusch_HoppingOffset = %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.pusch_ConfigBasic.pusch_HoppingOffset);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.enable64QAM = %d\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          (int)carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.pusch_ConfigBasic.enable64QAM);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.groupHoppingEnabled = %d\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          (int)carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.ul_ReferenceSignalsPUSCH.groupHoppingEnabled);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.groupAssignmentPUSCH = %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.ul_ReferenceSignalsPUSCH.groupAssignmentPUSCH);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.sequenceHoppingEnabled = %d\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          (int)carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.ul_ReferenceSignalsPUSCH.sequenceHoppingEnabled);
    LOG_T(RRC, PROTOCOL_RRC_CTXT_FMT" pusch_config_common.cyclicShift  = %ld\n",
          PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
          carrier->sib2->radioResourceConfigCommon.pusch_ConfigCommon.ul_ReferenceSignalsPUSCH.cyclicShift);

    if (carrier->MBMS_flag > 0) {
      for (i = 0; i < carrier->sib2->mbsfn_SubframeConfigList->list.count; i++) {
        // SIB 2
        //   LOG_D(RRC, "[eNB %d] mbsfn_SubframeConfigList.list.count = %ld\n", enb_mod_idP, RC.rrc[enb_mod_idP].sib2->mbsfn_SubframeConfigList->list.count);
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" SIB13 contents for MBSFN subframe allocation %d/%d(partial)\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              i,
              carrier->sib2->mbsfn_SubframeConfigList->list.count);
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" mbsfn_Subframe_pattern is  = %x\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib2->mbsfn_SubframeConfigList->list.array[i]->subframeAllocation.choice.oneFrame.buf[0] >> 0);
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" radioframe_allocation_period  = %ld (just index number, not the real value)\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib2->mbsfn_SubframeConfigList->list.array[i]->radioframeAllocationPeriod);   // need to display the real value, using array of char (like in dumping SIB2)
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" radioframe_allocation_offset  = %ld\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib2->mbsfn_SubframeConfigList->list.array[i]->radioframeAllocationOffset);
      }

      //   SIB13
      for (i = 0; i < carrier->sib13->mbsfn_AreaInfoList_r9.list.count; i++) {
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" SIB13 contents for MBSFN sync area %d/%d (partial)\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              i,
              carrier->sib13->mbsfn_AreaInfoList_r9.list.count);
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" MCCH Repetition Period: %ld (just index number, not real value)\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib13->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.mcch_RepetitionPeriod_r9);
        LOG_D(RRC, PROTOCOL_RRC_CTXT_FMT" MCCH Offset: %ld\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib13->mbsfn_AreaInfoList_r9.list.array[i]->mcch_Config_r9.mcch_Offset_r9);
      }
    } else memset((void *)&RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib13,0,sizeof(RC.rrc[ctxt_pP->module_id]->carrier[CC_id].sib13));

    //TTN - SIB 18
    if (configuration->SL_configured > 0) {
      for (int j = 0; j < carrier->sib18->commConfig_r12->commRxPool_r12.list.count; j++) {
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Contents of SIB18 %d/%d \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              j+1,
              carrier->sib18->commConfig_r12->commRxPool_r12.list.count);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 rxPool_sc_CP_Len: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_CP_Len_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 sc_Period_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_Period_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 data_CP_Len_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->data_CP_Len_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 prb_Num_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_TF_ResourceConfig_r12.prb_Num_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 prb_Start_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_TF_ResourceConfig_r12.prb_Start_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 prb_End_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_TF_ResourceConfig_r12.prb_End_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 offsetIndicator: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_TF_ResourceConfig_r12.offsetIndicator_r12.choice.small_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB18 subframeBitmap_choice_bs_buf: %s \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib18->commConfig_r12->commRxPool_r12.list.array[j]->sc_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs16_r12.buf);
      }

      //TTN - SIB 19
      for (int j = 0; j < carrier->sib19->discConfig_r12->discRxPool_r12.list.count; j++) {
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Contents of SIB19 %d/%d \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              j+1,
              carrier->sib19->discConfig_r12->discRxPool_r12.list.count);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 cp_Len_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->cp_Len_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 discPeriod_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->discPeriod_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 numRetx_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->numRetx_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 numRepetition_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->numRepetition_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 tf_ResourceConfig_r12 prb_Num_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->tf_ResourceConfig_r12.prb_Num_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 tf_ResourceConfig_r12 prb_Start_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->tf_ResourceConfig_r12.prb_Start_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 tf_ResourceConfig_r12 prb_End_r12: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->tf_ResourceConfig_r12.prb_End_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 tf_ResourceConfig_r12 offsetIndicator: %ld \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->tf_ResourceConfig_r12.offsetIndicator_r12.choice.small_r12);
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" SIB19 tf_ResourceConfig_r12 subframeBitmap_choice_bs_buf: %s \n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP),
              carrier->sib19->discConfig_r12->discRxPool_r12.list.array[j]->tf_ResourceConfig_r12.subframeBitmap_r12.choice.bs16_r12.buf);
      }
    }
  }

  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_FMT" RRC_UE --- MAC_CONFIG_REQ (SIB1.tdd & SIB2 params) ---> MAC_UE\n",
        PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));

  // LTE-M stuff here (take out CU-DU for now)
  if (NODE_IS_MONOLITHIC(rrc->node_type)) {
    if ((carrier->mib.message.schedulingInfoSIB1_BR_r13>0) &&
        (carrier->sib1_BR!=NULL)) {
      AssertFatal(carrier->sib1_BR->nonCriticalExtension!=NULL,
                  "sib2_br->nonCriticalExtension is null (v8.9)\n");
      AssertFatal(carrier->sib1_BR->nonCriticalExtension->nonCriticalExtension!=NULL,
                  "sib2_br->nonCriticalExtension is null (v9.2)\n");
      AssertFatal(carrier->sib1_BR->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL,
                  "sib2_br->nonCriticalExtension is null (v11.3)\n");
      AssertFatal(carrier->sib1_BR->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL,
                  "sib2_br->nonCriticalExtension is null (v12.5)\n");
      AssertFatal(carrier->sib1_BR->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL,
                  "sib2_br->nonCriticalExtension is null (v13.10)\n");
      sib1_v13ext = carrier->sib1_BR->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension;
      // Basic Asserts for CE_level0 PRACH configuration
      LTE_RadioResourceConfigCommonSIB_t *radioResourceConfigCommon_BR = &carrier[CC_id].sib2_BR->radioResourceConfigCommon;
      struct LTE_PRACH_ConfigSIB_v1310 *ext4_prach=radioResourceConfigCommon_BR->ext4->prach_ConfigCommon_v1310;
      LTE_PRACH_ParametersListCE_r13_t   *prach_ParametersListCE_r13 = &ext4_prach->prach_ParametersListCE_r13;
      AssertFatal(prach_ParametersListCE_r13->list.count>0,"prach_ParametersListCE_r13 is empty\n");
      LTE_PRACH_ParametersCE_r13_t *p = prach_ParametersListCE_r13->list.array[0];
      AssertFatal(p->prach_StartingSubframe_r13 != NULL, "prach_StartingSubframe_r13 celevel0 is null\n");
      AssertFatal((1<<p->numRepetitionPerPreambleAttempt_r13)<=(2<<*p->prach_StartingSubframe_r13),
                  "prachce0->numReptitionPerPreambleAttempt_r13 %d > prach_StartingSubframe_r13 %d\n",
                  1<<p->numRepetitionPerPreambleAttempt_r13,
                  2<<*p->prach_StartingSubframe_r13);
    }
  }

  if (NODE_IS_MONOLITHIC(rrc->node_type)) {
    LOG_D(RRC, "About to call rrc_mac_config_req_eNB for ngran_eNB\n");
    rrc_mac_config_req_eNB(ctxt_pP->module_id, CC_id,
                           carrier->physCellId,
                           carrier->p_eNB,
                           carrier->Ncp,
                           carrier->sib1->freqBandIndicator,
                           carrier->dl_CarrierFreq,
                           carrier->pbch_repetition,
                           0, // rnti
                           (LTE_BCCH_BCH_Message_t *) &carrier->mib,
                           (LTE_RadioResourceConfigCommonSIB_t *) &carrier->sib2->radioResourceConfigCommon,
                           (LTE_RadioResourceConfigCommonSIB_t *) &carrier->sib2_BR->radioResourceConfigCommon,
                           (struct LTE_PhysicalConfigDedicated *)NULL,
                           (LTE_SCellToAddMod_r10_t *)NULL,
                           (LTE_MeasObjectToAddMod_t **) NULL,
                           (LTE_MAC_MainConfig_t *) NULL, 0,
                           (struct LTE_LogicalChannelConfig *)NULL,
                           (LTE_MeasGapConfig_t *) NULL,
                           carrier->sib1->tdd_Config,
                           NULL,
                           &carrier->sib1->schedulingInfoList,
                           carrier->ul_CarrierFreq,
                           carrier->sib2->freqInfo.ul_Bandwidth,
                           &carrier->sib2->freqInfo.additionalSpectrumEmission,
                           (LTE_MBSFN_SubframeConfigList_t *) carrier->sib2->mbsfn_SubframeConfigList,
                           carrier->MBMS_flag,
                           (LTE_MBSFN_AreaInfoList_r9_t *) & carrier->sib13->mbsfn_AreaInfoList_r9,
                           (LTE_PMCH_InfoList_r9_t *) NULL,
                           sib1_v13ext,
                           RC.rrc[ctxt_pP->module_id]->carrier[CC_id].FeMBMS_flag,
                           (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                           (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                           (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                           (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                           (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                          );
  }

  /* set flag to indicate that cell information is configured. This is required
   * in DU to trigger F1AP_SETUP procedure */
  pthread_mutex_lock(&rrc->cell_info_mutex);
  rrc->cell_info_configured=1;
  pthread_mutex_unlock(&rrc->cell_info_mutex);
}

/*------------------------------------------------------------------------------*/
static void
init_MCCH(
  module_id_t enb_mod_idP,
  int CC_id
)
//-----------------------------------------------------------------------------
{
  int                                 sync_area = 0;
  // initialize RRC_eNB_INST MCCH entry
  eNB_RRC_INST *rrc = RC.rrc[enb_mod_idP];
  RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESSAGE =
    malloc(RC.rrc[enb_mod_idP]->carrier[CC_id].num_mbsfn_sync_area * sizeof(uint8_t *));

  for (sync_area = 0; sync_area < RC.rrc[enb_mod_idP]->carrier[CC_id].num_mbsfn_sync_area; sync_area++) {
    RC.rrc[enb_mod_idP]->carrier[CC_id].sizeof_MCCH_MESSAGE[sync_area] = 0;
    RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESSAGE[sync_area] = (uint8_t *) malloc16(32);
    AssertFatal(RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESSAGE[sync_area] != NULL,
                "[eNB %d]init_MCCH: FATAL, no memory for MCCH MESSAGE allocated \n", enb_mod_idP);
    RC.rrc[enb_mod_idP]->carrier[CC_id].sizeof_MCCH_MESSAGE[sync_area] = do_MBSFNAreaConfig(enb_mod_idP,
        sync_area,
        (uint8_t *)RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESSAGE[sync_area],
        &RC.rrc[enb_mod_idP]->carrier[CC_id].mcch,
        &RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message);
    LOG_I(RRC, "mcch message pointer %p for sync area %d \n",
          RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESSAGE[sync_area],
          sync_area);
    LOG_D(RRC, "[eNB %d] MCCH_MESSAGE  contents for Sync Area %d (partial)\n", enb_mod_idP, sync_area);
    LOG_D(RRC, "[eNB %d] CommonSF_AllocPeriod_r9 %ld\n", enb_mod_idP,
          RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->commonSF_AllocPeriod_r9);
    LOG_D(RRC,
          "[eNB %d] CommonSF_Alloc_r9.list.count (number of MBSFN Subframe Pattern) %d\n",
          enb_mod_idP, RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->commonSF_Alloc_r9.list.count);
    LOG_D(RRC, "[eNB %d] MBSFN Subframe Pattern: %02x (in hex)\n",
          enb_mod_idP,
          RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->commonSF_Alloc_r9.list.array[0]->subframeAllocation.
          choice.oneFrame.buf[0]);
    AssertFatal(RC.rrc[enb_mod_idP]->carrier[CC_id].sizeof_MCCH_MESSAGE[sync_area] != 255,
                "RC.rrc[enb_mod_idP]->carrier[CC_id].sizeof_MCCH_MESSAGE[sync_area] == 255");
    RC.rrc[enb_mod_idP]->carrier[CC_id].MCCH_MESS[sync_area].Active = 1;
  }

  //Set the RC.rrc[enb_mod_idP]->MCCH_MESS.Active to 1 (allow to  transfer MCCH message RRC->MAC in function mac_rrc_data_req)
  // ??Configure MCCH logical channel
  // call mac_config_req with appropriate structure from ASN.1 description
  //  LOG_I(RRC, "DUY: serviceID is %d\n",RC.rrc[enb_mod_idP]->mcch_message->pmch_InfoList_r9.list.array[0]->mbms_SessionInfoList_r9.list.array[0]->tmgi_r9.serviceId_r9.buf[2]);
  //  LOG_I(RRC, "DUY: session ID is %d\n",RC.rrc[enb_mod_idP]->mcch_message->pmch_InfoList_r9.list.array[0]->mbms_SessionInfoList_r9.list.array[0]->sessionId_r9->buf[0]);
  if (NODE_IS_MONOLITHIC(rrc->node_type)) {
    rrc_mac_config_req_eNB(enb_mod_idP, CC_id,
                           0,0,0,0,0, 0, 0,//rnti
                           (LTE_BCCH_BCH_Message_t *)NULL,
                           (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                           (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                           (struct LTE_PhysicalConfigDedicated *)NULL,
                           (LTE_SCellToAddMod_r10_t *)NULL,
                           //(struct LTE_PhysicalConfigDedicatedSCell_r10 *)NULL,
                           (LTE_MeasObjectToAddMod_t **) NULL,
                           (LTE_MAC_MainConfig_t *) NULL,
                           0,
                           (struct LTE_LogicalChannelConfig *)NULL,
                           (LTE_MeasGapConfig_t *) NULL,
                           (LTE_TDD_Config_t *) NULL,
                           (LTE_MobilityControlInfo_t *)NULL,
                           (LTE_SchedulingInfoList_t *) NULL,
                           0, NULL, NULL, (LTE_MBSFN_SubframeConfigList_t *) NULL, 0,
                           (LTE_MBSFN_AreaInfoList_r9_t *) NULL,
                           (LTE_PMCH_InfoList_r9_t *) & (RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->pmch_InfoList_r9),
                           (LTE_SystemInformationBlockType1_v1310_IEs_t *)NULL,
                           0,
                           (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                           (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                           (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                           (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                           (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                          );
  }

  //LOG_I(RRC,"DUY: lcid after rrc_mac_config_req is %02d\n",RC.rrc[enb_mod_idP]->mcch_message->pmch_InfoList_r9.list.array[0]->mbms_SessionInfoList_r9.list.array[0]->logicalChannelIdentity_r9);
}

//-----------------------------------------------------------------------------
static void init_MBMS(
  module_id_t enb_mod_idP,
  int         CC_id,
  frame_t frameP
)
//-----------------------------------------------------------------------------
{
  // init the configuration for MTCH
  protocol_ctxt_t               ctxt;

  if (RC.rrc[enb_mod_idP]->carrier[CC_id].MBMS_flag > 0) {
    PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, enb_mod_idP, ENB_FLAG_YES, NOT_A_RNTI, frameP, 0,enb_mod_idP);
    LOG_D(RRC, "[eNB %d] Frame %d : Radio Bearer config request for MBMS\n", enb_mod_idP, frameP);   //check the lcid
    // Configuring PDCP and RLC for MBMS Radio Bearer
    rrc_pdcp_config_asn1_req(&ctxt,
                             (LTE_SRB_ToAddModList_t *)NULL,   // LTE_SRB_ToAddModList
                             (LTE_DRB_ToAddModList_t *)NULL,   // LTE_DRB_ToAddModList
                             (LTE_DRB_ToReleaseList_t *)NULL,
                             0,     // security mode
                             NULL,  // key rrc encryption
                             NULL,  // key rrc integrity
                             NULL   // key encryption
                             , &(RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->pmch_InfoList_r9)
                             ,NULL);

    if (!NODE_IS_CU(RC.rrc[enb_mod_idP]->node_type)) {
      rrc_rlc_config_asn1_req(&ctxt,
                              NULL, // LTE_SRB_ToAddModList
                              NULL,   // LTE_DRB_ToAddModList
                              NULL,   // DRB_ToReleaseList
                              &(RC.rrc[enb_mod_idP]->carrier[CC_id].mcch_message->pmch_InfoList_r9),0, 0
                             );
    }

    //rrc_mac_config_req();
  }
}

//-----------------------------------------------------------------------------
uint8_t
rrc_eNB_get_next_transaction_identifier(
  module_id_t enb_mod_idP
)
//-----------------------------------------------------------------------------
{
  static uint8_t                      rrc_transaction_identifier[NUMBER_OF_eNB_MAX];
  rrc_transaction_identifier[enb_mod_idP] = (rrc_transaction_identifier[enb_mod_idP] + 1) % RRC_TRANSACTION_IDENTIFIER_NUMBER;
  LOG_T(RRC,"generated xid is %d\n",rrc_transaction_identifier[enb_mod_idP]);
  return rrc_transaction_identifier[enb_mod_idP];
}
/*------------------------------------------------------------------------------*/
/* Functions to handle UE index in eNB UE list */


////-----------------------------------------------------------------------------
//static module_id_t
//rrc_eNB_get_UE_index(
//                module_id_t enb_mod_idP,
//                uint64_t    UE_identity
//)
////-----------------------------------------------------------------------------
//{
//
//    boolean_t      reg = FALSE;
//    module_id_t    i;
//
//    AssertFatal(enb_mod_idP < NB_eNB_INST, "eNB index invalid (%d/%d)!", enb_mod_idP, NB_eNB_INST);
//
//    for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
//        if (RC.rrc[enb_mod_idP]->Info.UE_info[i] == UE_identity) {
//            // UE_identity already registered
//            reg = TRUE;
//            break;
//        }
//    }
//
//    if (reg == FALSE) {
//        return (UE_MODULE_INVALID);
//    } else
//        return (i);
//}


//-----------------------------------------------------------------------------
// return the ue context if there is already an UE with ue_identityP, NULL otherwise
static struct rrc_eNB_ue_context_s *
rrc_eNB_ue_context_random_exist(
  const protocol_ctxt_t *const ctxt_pP,
  const uint64_t               ue_identityP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
  RB_FOREACH(ue_context_p, rrc_ue_tree_s, &(RC.rrc[ctxt_pP->module_id]->rrc_ue_head)) {
    if (ue_context_p->ue_context.random_ue_identity == ue_identityP)
      return ue_context_p;
  }
  return NULL;
}
//-----------------------------------------------------------------------------
// return the ue context if there is already an UE with the same S-TMSI(MMEC+M-TMSI), NULL otherwise
static struct rrc_eNB_ue_context_s *
rrc_eNB_ue_context_stmsi_exist(
  const protocol_ctxt_t *const ctxt_pP,
  const mme_code_t             mme_codeP,
  const m_tmsi_t               m_tmsiP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
  RB_FOREACH(ue_context_p, rrc_ue_tree_s, &(RC.rrc[ctxt_pP->module_id]->rrc_ue_head)) {
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
    } else
      printf("\n");
  }
  return NULL;
}

//-----------------------------------------------------------------------------
// return a new ue context structure if ue_identityP, ctxt_pP->rnti not found in collection
static struct rrc_eNB_ue_context_s *
rrc_eNB_get_next_free_ue_context(
  const protocol_ctxt_t *const ctxt_pP,
  const uint64_t               ue_identityP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
  ue_context_p = rrc_eNB_get_ue_context(
                   RC.rrc[ctxt_pP->module_id],
                   ctxt_pP->rnti);

  if (ue_context_p == NULL) {
    ue_context_p = rrc_eNB_allocate_new_UE_context(RC.rrc[ctxt_pP->module_id]);

    if (ue_context_p == NULL) {
      LOG_E(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" Cannot create new UE context, no memory\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
      return NULL;
    }

    ue_context_p->ue_id_rnti                    = ctxt_pP->rnti; // here ue_id_rnti is just a key, may be something else
    ue_context_p->ue_context.rnti               = ctxt_pP->rnti; // yes duplicate, 1 may be removed
    ue_context_p->ue_context.random_ue_identity = ue_identityP;
    RB_INSERT(rrc_ue_tree_s, &RC.rrc[ctxt_pP->module_id]->rrc_ue_head, ue_context_p);
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

  return(ue_context_p);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_free_mem_UE_context(
  const protocol_ctxt_t               *const ctxt_pP,
  struct rrc_eNB_ue_context_s         *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  int i;
  LOG_T(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Clearing UE context 0x%p (free internal structs)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_context_pP);
  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_LTE_SCellToAddMod_r10, &ue_context_pP->ue_context.sCell_config[0]);
  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_LTE_SCellToAddMod_r10, &ue_context_pP->ue_context.sCell_config[1]);

  if (ue_context_pP->ue_context.SRB_configList) {
    ASN_STRUCT_FREE(asn_DEF_LTE_SRB_ToAddModList, ue_context_pP->ue_context.SRB_configList);
    ue_context_pP->ue_context.SRB_configList = NULL;
  }

  for(i = 0; i < RRC_TRANSACTION_IDENTIFIER_NUMBER; i++) {
    if (ue_context_pP->ue_context.SRB_configList2[i]) {
      free(ue_context_pP->ue_context.SRB_configList2[i]);
      ue_context_pP->ue_context.SRB_configList2[i] = NULL;
    }
  }

  if (ue_context_pP->ue_context.DRB_configList) {
    ASN_STRUCT_FREE(asn_DEF_LTE_DRB_ToAddModList, ue_context_pP->ue_context.DRB_configList);
    ue_context_pP->ue_context.DRB_configList = NULL;
  }

  for(i = 0; i < RRC_TRANSACTION_IDENTIFIER_NUMBER; i++) {
    if (ue_context_pP->ue_context.DRB_configList2[i]) {
      free(ue_context_pP->ue_context.DRB_configList2[i]);
      ue_context_pP->ue_context.DRB_configList2[i] = NULL;
    }

    if (ue_context_pP->ue_context.DRB_Release_configList2[i]) {
      free(ue_context_pP->ue_context.DRB_Release_configList2[i]);
      ue_context_pP->ue_context.DRB_Release_configList2[i] = NULL;
    }
  }

  memset(ue_context_pP->ue_context.DRB_active, 0, sizeof(ue_context_pP->ue_context.DRB_active));

  if (ue_context_pP->ue_context.physicalConfigDedicated) {
    ASN_STRUCT_FREE(asn_DEF_LTE_PhysicalConfigDedicated, ue_context_pP->ue_context.physicalConfigDedicated);
    ue_context_pP->ue_context.physicalConfigDedicated = NULL;
  }

  if (ue_context_pP->ue_context.sps_Config) {
    ASN_STRUCT_FREE(asn_DEF_LTE_SPS_Config, ue_context_pP->ue_context.sps_Config);
    ue_context_pP->ue_context.sps_Config = NULL;
  }

  for (i=0; i < MAX_MEAS_OBJ; i++) {
    if (ue_context_pP->ue_context.MeasObj[i] != NULL) {
      ASN_STRUCT_FREE(asn_DEF_LTE_MeasObjectToAddMod, ue_context_pP->ue_context.MeasObj[i]);
      ue_context_pP->ue_context.MeasObj[i] = NULL;
    }
  }

  for (i=0; i < MAX_MEAS_CONFIG; i++) {
    if (ue_context_pP->ue_context.ReportConfig[i] != NULL) {
      ASN_STRUCT_FREE(asn_DEF_LTE_ReportConfigToAddMod, ue_context_pP->ue_context.ReportConfig[i]);
      ue_context_pP->ue_context.ReportConfig[i] = NULL;
    }
  }

  if (ue_context_pP->ue_context.QuantityConfig) {
    ASN_STRUCT_FREE(asn_DEF_LTE_QuantityConfig, ue_context_pP->ue_context.QuantityConfig);
    ue_context_pP->ue_context.QuantityConfig = NULL;
  }

  if (ue_context_pP->ue_context.mac_MainConfig) {
    ASN_STRUCT_FREE(asn_DEF_LTE_MAC_MainConfig, ue_context_pP->ue_context.mac_MainConfig);
    ue_context_pP->ue_context.mac_MainConfig = NULL;
  }

  /*  if (ue_context_pP->ue_context.measGapConfig) {
      ASN_STRUCT_FREE(asn_DEF_LTE_MeasGapConfig, ue_context_pP->ue_context.measGapConfig);
      ue_context_pP->ue_context.measGapConfig = NULL;
    }*/
  if (ue_context_pP->ue_context.handover_info) {
    /* TODO: be sure free is enough here (check memory leaks) */
    free(ue_context_pP->ue_context.handover_info);
    ue_context_pP->ue_context.handover_info = NULL;
  }

  if (ue_context_pP->ue_context.measurement_info) {
    /* TODO: be sure free is enough here (check memory leaks) */
    if (ue_context_pP->ue_context.measurement_info->events) {
      if (ue_context_pP->ue_context.measurement_info->events->a3_event) {
        free(ue_context_pP->ue_context.measurement_info->events->a3_event);
        ue_context_pP->ue_context.measurement_info->events->a3_event = NULL;
      }

      free(ue_context_pP->ue_context.measurement_info->events);
      ue_context_pP->ue_context.measurement_info->events = NULL;
    }

    free(ue_context_pP->ue_context.measurement_info);
    ue_context_pP->ue_context.measurement_info = NULL;
  }

  //SRB_INFO                           SI;
  //SRB_INFO                           Srb0;
  //SRB_INFO_TABLE_ENTRY               Srb1;
  //SRB_INFO_TABLE_ENTRY               Srb2;
  if (ue_context_pP->ue_context.measConfig) {
    ASN_STRUCT_FREE(asn_DEF_LTE_MeasConfig, ue_context_pP->ue_context.measConfig);
    ue_context_pP->ue_context.measConfig = NULL;
  }

  if (ue_context_pP->ue_context.measConfig) {
    ASN_STRUCT_FREE(asn_DEF_LTE_MeasConfig, ue_context_pP->ue_context.measConfig);
    ue_context_pP->ue_context.measConfig = NULL;
  }

#if 0
  //HANDOVER_INFO                     *handover_info;
  //uint8_t kenb[32];
  //e_SecurityAlgorithmConfig__cipheringAlgorithm     ciphering_algorithm;
  //e_SecurityAlgorithmConfig__integrityProtAlgorithm integrity_algorithm;
  //uint8_t                            Status;
  //rnti_t                             rnti;
  //uint64_t                           random_ue_identity;
  //UE_S_TMSI                          Initialue_identity_s_TMSI;
  //EstablishmentCause_t               establishment_cause;
  //ReestablishmentCause_t             reestablishment_cause;
  //uint16_t                           ue_initial_id;
  //uint32_t                           eNB_ue_s1ap_id :24;
  //security_capabilities_t            security_capabilities;
  //uint8_t                            nb_of_e_rabs;
  //e_rab_param_t                      e_rab[S1AP_MAX_E_RAB];
  //uint32_t                           enb_gtp_teid[S1AP_MAX_E_RAB];
  //transport_layer_addr_t             enb_gtp_addrs[S1AP_MAX_E_RAB];
  //rb_id_t                            enb_gtp_ebi[S1AP_MAX_E_RAB];
#endif
}

//-----------------------------------------------------------------------------
/*
* Should be called when UE context in eNB should be released
* or when S1 command UE_CONTEXT_RELEASE_REQ should be sent
*/
void
rrc_eNB_free_UE(
  const module_id_t enb_mod_idP,
  const struct rrc_eNB_ue_context_s *const ue_context_pP)
//-----------------------------------------------------------------------------
{
  rnti_t rnti = ue_context_pP->ue_context.rnti;

  if (enb_mod_idP >= NB_eNB_INST) {
    LOG_E(RRC, "eNB instance invalid (%d/%d) for UE %x!\n",
          enb_mod_idP,
          NB_eNB_INST,
          rnti);
    return;
  }

  if(EPC_MODE_ENABLED) {
    if (!NODE_IS_DU(RC.rrc[enb_mod_idP]->node_type)) {
      if((ue_context_pP->ue_context.ul_failure_timer >= 20000) && (mac_eNB_get_rrc_status(enb_mod_idP, rnti) >= RRC_CONNECTED)) {
        LOG_I(RRC, "[eNB %d] S1AP_UE_CONTEXT_RELEASE_REQ sent for RNTI %x, cause 21, radio connection with ue lost\n",
              enb_mod_idP,
              rnti);
        rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_REQ(enb_mod_idP,
            ue_context_pP,
            S1AP_CAUSE_RADIO_NETWORK,
            21); // send cause 21: radio connection with ue lost
        /* From 3GPP 36300v10 p129 : 19.2.2.2.2 S1 UE Context Release Request (eNB triggered)
         * If the E-UTRAN internal reason is a radio link failure detected in the eNB, the eNB shall wait a sufficient time before
         *  triggering the S1 UE Context Release Request procedure in order to allow the UE to perform the NAS recovery
         *  procedure, see TS 23.401 [17].
         */
        return;
      }

      if((ue_context_pP->ue_context.ue_rrc_inactivity_timer >= RC.rrc[enb_mod_idP]->configuration.rrc_inactivity_timer_thres) &&
          (mac_eNB_get_rrc_status(enb_mod_idP, rnti) >= RRC_CONNECTED) &&
          (RC.rrc[enb_mod_idP]->configuration.rrc_inactivity_timer_thres > 0)) {
        LOG_I(RRC, "[eNB %d] S1AP_UE_CONTEXT_RELEASE_REQ sent for RNTI %x, cause 20, user inactivity\n",
              enb_mod_idP,
              rnti);
        rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_REQ(enb_mod_idP,
            ue_context_pP,
            S1AP_CAUSE_RADIO_NETWORK,
            20); // send cause 20: user inactivity
        return;
      }
    }
  }

  LOG_W(RRC, "[eNB %d] Removing UE RNTI %x\n",
        enb_mod_idP,
        rnti);
  // add UE info to freeList
  LOG_I(RRC, "Put UE %x into freeList\n",
        rnti);
  put_UE_in_freelist(enb_mod_idP, rnti, 1);
}

void remove_UE_from_freelist(module_id_t mod_id, rnti_t rnti) {
  eNB_MAC_INST                             *eNB_MAC = RC.mac[mod_id];
  pthread_mutex_lock(&lock_ue_freelist);
  UE_free_list_t                           *free_list = &eNB_MAC->UE_free_list;
  free_list->UE_free_ctrl[free_list->head_freelist].rnti = 0;
  free_list->head_freelist = (free_list->head_freelist + 1) % (NUMBER_OF_UE_MAX+1);
  free_list->num_UEs--;
  pthread_mutex_unlock(&lock_ue_freelist);
}

void put_UE_in_freelist(module_id_t mod_id, rnti_t rnti, boolean_t removeFlag) {
  UE_free_list_t                           *free_list = NULL;
  eNB_MAC_INST                             *eNB_MAC = RC.mac[mod_id];
  pthread_mutex_lock(&lock_ue_freelist);
  free_list = &eNB_MAC->UE_free_list;
  free_list->UE_free_ctrl[free_list->tail_freelist].rnti = rnti;
  free_list->UE_free_ctrl[free_list->tail_freelist].removeContextFlg = removeFlag;
  free_list->num_UEs++;
  eNB_MAC->UE_release_req.ue_release_request_body.ue_release_request_TLVs_list[eNB_MAC->UE_release_req.ue_release_request_body.number_of_TLVs].rnti = rnti;
  eNB_MAC->UE_release_req.ue_release_request_body.number_of_TLVs++;
  free_list->tail_freelist = (free_list->tail_freelist + 1) % (NUMBER_OF_UE_MAX+1);
  pthread_mutex_unlock(&lock_ue_freelist);
}

void release_UE_in_freeList(module_id_t mod_id) {
  int i, j, CC_id, pdu_number;
  protocol_ctxt_t                           ctxt;
  LTE_eNB_ULSCH_t                          *ulsch = NULL;
  LTE_eNB_DLSCH_t                          *dlsch = NULL;
  nfapi_ul_config_request_body_t           *ul_req_tmp = NULL;
  PHY_VARS_eNB                             *eNB_PHY = NULL;
  struct rrc_eNB_ue_context_s              *ue_context_pP = NULL;
  eNB_MAC_INST                             *eNB_MAC = RC.mac[mod_id];
  boolean_t                                 remove_UEContext;
  rnti_t                                    rnti;
  int                                       head, tail, ue_num;
  pthread_mutex_lock(&lock_ue_freelist);
  head = eNB_MAC->UE_free_list.head_freelist;
  tail = eNB_MAC->UE_free_list.tail_freelist;

  if(head == tail) {
    pthread_mutex_unlock(&lock_ue_freelist);
    return;
  }

  if(tail < head) {
    tail = head + eNB_MAC->UE_free_list.num_UEs;
  }

  pthread_mutex_unlock(&lock_ue_freelist);

  for(ue_num = head; ue_num < tail; ue_num++) {
    ue_num = ue_num % (NUMBER_OF_UE_MAX+1);
    rnti = eNB_MAC->UE_free_list.UE_free_ctrl[ue_num].rnti;

    if(rnti != 0) {
      remove_UEContext = eNB_MAC->UE_free_list.UE_free_ctrl[ue_num].removeContextFlg;
      PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, mod_id, ENB_FLAG_YES, rnti, 0, 0,mod_id);

      for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
        eNB_PHY = RC.eNB[mod_id][CC_id];

        for (i=0; i<MAX_MOBILES_PER_ENB; i++) {
          ulsch = eNB_PHY->ulsch[i];

          if((ulsch != NULL) && (ulsch->rnti == rnti)) {
            void clean_eNb_ulsch(LTE_eNB_ULSCH_t *ulsch);
            LOG_I(RRC, "clean_eNb_ulsch ulsch[%d] UE %x\n", i, rnti);
            clean_eNb_ulsch(ulsch);
          }

          dlsch = eNB_PHY->dlsch[i][0];

          if((dlsch != NULL) && (dlsch->rnti == rnti)) {
            void clean_eNb_dlsch(LTE_eNB_DLSCH_t *dlsch);
            LOG_I(RRC, "clean_eNb_dlsch dlsch[%d] UE %x \n", i, rnti);
            clean_eNb_dlsch(dlsch);
          }
        }

        ulsch = eNB_PHY->ulsch[i];

        if((ulsch != NULL) && (ulsch->rnti == rnti)) {
          void clean_eNb_ulsch(LTE_eNB_ULSCH_t *ulsch);
          LOG_I(RRC, "clean_eNb_ulsch ulsch[%d] UE %x\n", i, rnti);
          clean_eNb_ulsch(ulsch);
        }

        for (i=0; i<NUMBER_OF_UCI_VARS_MAX; i++) {
          if(eNB_PHY->uci_vars[i].rnti == rnti) {
            LOG_I(MAC, "clean eNb uci_vars[%d] UE %x \n",i, rnti);
            memset(&eNB_PHY->uci_vars[i],0,sizeof(LTE_eNB_UCI));
          }
        }

        if (flexran_agent_get_rrc_xface(mod_id)) {
          flexran_agent_get_rrc_xface(mod_id)->flexran_agent_notify_ue_state_change(
            mod_id, rnti, PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_DEACTIVATED);
        }

        for(j = 0; j < 10; j++) {
          ul_req_tmp = &eNB_MAC->UL_req_tmp[CC_id][j].ul_config_request_body;

          if(ul_req_tmp) {
            pdu_number = ul_req_tmp->number_of_pdus;

            for(int pdu_index = pdu_number-1; pdu_index >= 0; pdu_index--) {
              if((ul_req_tmp->ul_config_pdu_list[pdu_index].ulsch_pdu.ulsch_pdu_rel8.rnti == rnti) ||
                  (ul_req_tmp->ul_config_pdu_list[pdu_index].uci_harq_pdu.ue_information.ue_information_rel8.rnti == rnti) ||
                  (ul_req_tmp->ul_config_pdu_list[pdu_index].uci_cqi_pdu.ue_information.ue_information_rel8.rnti == rnti) ||
                  (ul_req_tmp->ul_config_pdu_list[pdu_index].uci_sr_pdu.ue_information.ue_information_rel8.rnti == rnti) ||
                  (ul_req_tmp->ul_config_pdu_list[pdu_index].srs_pdu.srs_pdu_rel8.rnti == rnti)) {
                LOG_I(RRC, "remove UE %x from ul_config_pdu_list %d/%d\n", rnti, pdu_index, pdu_number);

                if(pdu_index < pdu_number -1) {
                  memcpy(&ul_req_tmp->ul_config_pdu_list[pdu_index], &ul_req_tmp->ul_config_pdu_list[pdu_index+1], (pdu_number-1-pdu_index) * sizeof(nfapi_ul_config_request_pdu_t));
                }

                ul_req_tmp->number_of_pdus--;
              }
            }
          }
        }
      }

      if (!NODE_IS_CU(RC.rrc[mod_id]->node_type)) {
        rrc_mac_remove_ue(mod_id,rnti);
        rrc_rlc_remove_ue(&ctxt);
        pdcp_remove_UE(&ctxt);
      } else {
        MessageDef *m = itti_alloc_new_message(TASK_RRC_ENB, F1AP_UE_CONTEXT_RELEASE_CMD);
        F1AP_UE_CONTEXT_RELEASE_CMD(m).rnti = rnti;
        F1AP_UE_CONTEXT_RELEASE_CMD(m).cause = F1AP_CAUSE_RADIO_NETWORK;
        F1AP_UE_CONTEXT_RELEASE_CMD(m).cause_value = 10; // 10 = F1AP_CauseRadioNetwork_normal_release
        F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container = NULL;
        F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container_length = 0;
        itti_send_msg_to_task(TASK_CU_F1, mod_id, m);
      }

      if(remove_UEContext) {
        ue_context_pP = rrc_eNB_get_ue_context(RC.rrc[mod_id],rnti);

        if(ue_context_pP) {
          rrc_eNB_remove_ue_context(&ctxt,RC.rrc[mod_id],
                                    (struct rrc_eNB_ue_context_s *) ue_context_pP);
        }
      }

      LOG_I(RRC, "[release_UE_in_freeList] remove UE %x from freeList\n", rnti);
      remove_UE_from_freelist(mod_id, rnti);
    }
  }
}

int rrc_eNB_previous_SRB2(rrc_eNB_ue_context_t         *ue_context_pP) {
  struct LTE_SRB_ToAddMod                *SRB2_config = NULL;
  uint8_t i;
  LTE_SRB_ToAddModList_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  LTE_SRB_ToAddModList_t                **SRB_configList2 = &ue_context_pP->ue_context.SRB_configList2[ue_context_pP->ue_context.reestablishment_xid];

  if (*SRB_configList2 != NULL) {
    if((*SRB_configList2)->list.count!=0) {
      LOG_D(RRC, "rrc_eNB_previous_SRB2 SRB_configList2(%p) count is %d\n           SRB_configList2->list.array[0] addr is %p",
            SRB_configList2, (*SRB_configList2)->list.count,  (*SRB_configList2)->list.array[0]);
    }

    for (i = 0; (i < (*SRB_configList2)->list.count) && (i < 3); i++) {
      if ((*SRB_configList2)->list.array[i]->srb_Identity == 2 ) {
        SRB2_config = (*SRB_configList2)->list.array[i];
        break;
      }
    }
  } else {
    LOG_E(RRC, "rrc_eNB_previous_SRB2 SRB_configList2 NULL\n");
  }

  if (SRB2_config != NULL) {
    ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);
  } else {
    LOG_E(RRC, "rrc_eNB_previous_SRB2 SRB2_config NULL\n");
  }

  return 0;
}
//-----------------------------------------------------------------------------
/*
* Process the rrc connection setup complete message from UE (SRB1 Active)
*/
void
rrc_eNB_process_RRCConnectionSetupComplete(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t *ue_context_pP,
  LTE_RRCConnectionSetupComplete_r8_IEs_t *rrcConnectionSetupComplete
)
//-----------------------------------------------------------------------------
{
  LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, " "processing LTE_RRCConnectionSetupComplete from UE (SRB1 Active)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
  ue_context_pP->ue_context.Srb1.Active = 1;
  ue_context_pP->ue_context.Status = RRC_CONNECTED;
  ue_context_pP->ue_context.ue_rrc_inactivity_timer = 1; // set rrc inactivity timer when UE goes into RRC_CONNECTED
  T(T_ENB_RRC_CONNECTION_SETUP_COMPLETE,
    T_INT(ctxt_pP->module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(ctxt_pP->rnti));

#ifdef ENABLE_RIC_AGENT
    rrc_kpi_stats.rrc_conn_estab_succ_sum++;
#endif        
  if (EPC_MODE_ENABLED == 1) {
    // Forward message to S1AP layer
    rrc_eNB_send_S1AP_NAS_FIRST_REQ(ctxt_pP, ue_context_pP, rrcConnectionSetupComplete);
  } else {
    // RRC loop back (no S1AP), send SecurityModeCommand to UE
    rrc_eNB_generate_SecurityModeCommand(ctxt_pP, ue_context_pP);
  }
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_SecurityModeCommand(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t          *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[100];
  uint8_t                             size;
  T(T_ENB_RRC_SECURITY_MODE_COMMAND, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  size = do_SecurityModeCommand(
           ctxt_pP,
           buffer,
           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),
           ue_context_pP->ue_context.ciphering_algorithm,
           ue_context_pP->ue_context.integrity_algorithm);
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,"[MSG] RRC Security Mode Command\n");
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate SecurityModeCommand (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (securityModeCommand to UE MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH);
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

  if (!NODE_IS_DU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    LOG_I(RRC,"calling rrc_data_req :securityModeCommand\n");
    rrc_data_req(ctxt_pP,
                 DCCH,
                 rrc_eNB_mui++,
                 SDU_CONFIRM_NO,
                 size,
                 buffer,
                 PDCP_TRANSMISSION_MODE_CONTROL);
  }
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_UECapabilityEnquiry(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t          *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[100];
  uint8_t                             size;
  T(T_ENB_RRC_UE_CAPABILITY_ENQUIRY, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  size = do_UECapabilityEnquiry(
           ctxt_pP,
           buffer,
           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id));
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate UECapabilityEnquiry (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (UECapabilityEnquiry MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH);
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
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_NR_UECapabilityEnquiry(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t          *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[100];
  uint8_t                             size;
  T(T_ENB_RRC_UE_CAPABILITY_ENQUIRY, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  size = do_NR_UECapabilityEnquiry(
           ctxt_pP,
           buffer,
           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id));
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate NR UECapabilityEnquiry (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (NR UECapabilityEnquiry MUI %d) --->[PDCP][RB %02d]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcNRUECapabilityEnquiry UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionReject(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t          *const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{
  T(T_ENB_RRC_CONNECTION_REJECT, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  eNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  ue_p->Srb0.Tx_buffer.payload_size =
    do_RRCConnectionReject(ctxt_pP->module_id,
                           (uint8_t *) ue_p->Srb0.Tx_buffer.Payload);
  LOG_DUMPMSG(RRC,DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRCConnectionReject\n");
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    ue_p->Srb0.Tx_buffer.Header,
    ue_p->Srb0.Tx_buffer.payload_size,
    MSC_AS_TIME_FMT" LTE_RRCConnectionReject UE %x size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP == NULL ? -1 : ue_context_pP->ue_context.rnti,
    ue_p->Srb0.Tx_buffer.payload_size);
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating LTE_RRCConnectionReject (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_p->Srb0.Tx_buffer.payload_size);
}

//-----------------------------------------------------------------------------
/*
 * Generate a RCC Connection Reestablishment after requested
 */
void
rrc_eNB_generate_RRCConnectionReestablishment(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t  *const ue_context_pP,
  const int             CC_id)
//-----------------------------------------------------------------------------
{
  int UE_id = -1;
  LTE_LogicalChannelConfig_t *SRB1_logicalChannelConfig = NULL;
  LTE_SRB_ToAddModList_t     **SRB_configList;
  LTE_SRB_ToAddMod_t         *SRB1_config = NULL;
  rrc_eNB_carrier_data_t     *carrier = NULL;
  eNB_RRC_UE_t               *ue_context = NULL;
  module_id_t module_id = ctxt_pP->module_id;
  uint16_t rnti = ctxt_pP->rnti;
  T(T_ENB_RRC_CONNECTION_REESTABLISHMENT,
    T_INT(module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(rnti));
  SRB_configList = &(ue_context_pP->ue_context.SRB_configList);
  carrier = &(RC.rrc[ctxt_pP->module_id]->carrier[CC_id]);
  ue_context = &(ue_context_pP->ue_context);
  ue_context->Srb0.Tx_buffer.payload_size = do_RRCConnectionReestablishment(ctxt_pP,
      ue_context_pP,
      CC_id,
      (uint8_t *) ue_context->Srb0.Tx_buffer.Payload,
      (uint8_t) carrier->p_eNB, // at this point we do not have the UE capability information, so it can only be TM1 or TM2
      rrc_eNB_get_next_transaction_identifier(module_id),
      SRB_configList,
      &(ue_context->physicalConfigDedicated));
  LOG_DUMPMSG(RRC, DEBUG_RRC,
              (char *)(ue_context->Srb0.Tx_buffer.Payload),
              ue_context->Srb0.Tx_buffer.payload_size,
              "[MSG] RRCConnectionReestablishment \n");

  /* Configure SRB1 for UE */
  if (*SRB_configList != NULL) {
    for (int cnt = 0; cnt < (*SRB_configList)->list.count; cnt++) {
      if ((*SRB_configList)->list.array[cnt]->srb_Identity == 1) {
        SRB1_config = (*SRB_configList)->list.array[cnt];

        if (SRB1_config->logicalChannelConfig) {
          if (SRB1_config->logicalChannelConfig->present == LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue) {
            SRB1_logicalChannelConfig = &(SRB1_config->logicalChannelConfig->choice.explicitValue);
          } else {
            SRB1_logicalChannelConfig = &(SRB1_logicalChannelConfig_defaultValue);
          }
        } else {
          SRB1_logicalChannelConfig = &(SRB1_logicalChannelConfig_defaultValue);
        }

        LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (SRB1) ---> MAC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

        if (NODE_IS_MONOLITHIC(RC.rrc[ctxt_pP->module_id]->node_type)) {
          rrc_mac_config_req_eNB(module_id,
                                 ue_context->primaryCC_id,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 0,
                                 rnti,
                                 (LTE_BCCH_BCH_Message_t *) NULL,
                                 (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                 (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                 (struct LTE_PhysicalConfigDedicated * ) ue_context->physicalConfigDedicated,
                                 (LTE_SCellToAddMod_r10_t *)NULL,
                                 (LTE_MeasObjectToAddMod_t **) NULL,
                                 ue_context->mac_MainConfig,
                                 1,
                                 SRB1_logicalChannelConfig,
                                 ue_context->measGapConfig,
                                 (LTE_TDD_Config_t *) NULL,
                                 NULL,
                                 (LTE_SchedulingInfoList_t *) NULL,
                                 0,
                                 NULL,
                                 NULL,
                                 (LTE_MBSFN_SubframeConfigList_t *) NULL, 0,
                                 (LTE_MBSFN_AreaInfoList_r9_t *) NULL,
                                 (LTE_PMCH_InfoList_r9_t *) NULL,
                                 (LTE_SystemInformationBlockType1_v1310_IEs_t *) NULL,
                                 0,
                                 (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                                 (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                                 (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                                 (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                                 (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                                );
          break;
        }
      }  // if ((*SRB_configList)->list.array[cnt]->srb_Identity == 1)
    }  // for (int cnt = 0; cnt < (*SRB_configList)->list.count; cnt++)
  }  // if (*SRB_configList != NULL)

  MSC_LOG_TX_MESSAGE(MSC_RRC_ENB,
                     MSC_RRC_UE,
                     ue_context->Srb0.Tx_buffer.Header,
                     ue_context->Srb0.Tx_buffer.payload_size,
                     MSC_AS_TIME_FMT" LTE_RRCConnectionReestablishment UE %x size %u",
                     MSC_AS_TIME_ARGS(ctxt_pP),
                     ue_context->rnti,
                     ue_context->Srb0.Tx_buffer.payload_size);
  LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating LTE_RRCConnectionReestablishment (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_context->Srb0.Tx_buffer.payload_size);
  UE_id = find_UE_id(module_id, rnti);

  if (UE_id != -1) {
    /* Activate reject timer, if RRCComplete not received after 10 frames, reject UE */
    RC.mac[module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 1;
    /* Reject UE after 10 frames, LTE_RRCConnectionReestablishmentReject is triggered */
    RC.mac[module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer_thres = 100;
  } else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Generating LTE_RRCConnectionReestablishment without UE_id(MAC) rnti %x\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          rnti);
  }
}

//-----------------------------------------------------------------------------
void
rrc_eNB_process_RRCConnectionReestablishmentComplete(
  const protocol_ctxt_t *const ctxt_pP,
  const rnti_t reestablish_rnti,
  rrc_eNB_ue_context_t         *ue_context_pP,
  const uint8_t xid,
  LTE_RRCConnectionReestablishmentComplete_r8_IEs_t *LTE_RRCConnectionReestablishmentComplete
)
//-----------------------------------------------------------------------------
{
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, processing LTE_RRCConnectionReestablishmentComplete from UE (SRB1 Active)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
  T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  LTE_DRB_ToAddModList_t                 *DRB_configList = ue_context_pP->ue_context.DRB_configList;
  LTE_SRB_ToAddModList_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  LTE_SRB_ToAddModList_t                **SRB_configList2 = NULL;
  LTE_DRB_ToAddModList_t                **DRB_configList2 = NULL;
  struct LTE_SRB_ToAddMod                *SRB2_config = NULL;
  struct LTE_DRB_ToAddMod                *DRB_config = NULL;
  int i = 0;
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;
  LTE_MeasObjectToAddModList_t       *MeasObj_list                     = NULL;
  LTE_MeasObjectToAddMod_t           *MeasObj                          = NULL;
  LTE_ReportConfigToAddModList_t     *ReportConfig_list                = NULL;
  LTE_ReportConfigToAddMod_t         *ReportConfig_per, *ReportConfig_A1,
                                     *ReportConfig_A2, *ReportConfig_A3, *ReportConfig_A4, *ReportConfig_A5;
  LTE_MeasIdToAddModList_t           *MeasId_list                      = NULL;
  LTE_MeasIdToAddMod_t               *MeasId0, *MeasId1, *MeasId2, *MeasId3, *MeasId4, *MeasId5;
  LTE_RSRP_Range_t                   *rsrp                             = NULL;
  struct LTE_MeasConfig__speedStatePars  *Sparams                          = NULL;
  LTE_QuantityConfig_t                   *quantityConfig                   = NULL;
  LTE_CellsToAddMod_t                    *CellToAdd                        = NULL;
  LTE_CellsToAddModList_t                *CellsToAddModList                = NULL;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;
  LTE_C_RNTI_t                           *cba_RNTI                         = NULL;
  int                                    measurements_enabled;
  uint8_t next_xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);
  int ret = 0;
  ue_context_pP->ue_context.Status = RRC_CONNECTED;
  ue_context_pP->ue_context.ue_rrc_inactivity_timer = 1; // set rrc inactivity when UE goes into RRC_CONNECTED
  ue_context_pP->ue_context.reestablishment_xid = next_xid;
  SRB_configList2 = &ue_context_pP->ue_context.SRB_configList2[xid];

  // get old configuration of SRB2
  if (*SRB_configList2 != NULL) {
    if((*SRB_configList2)->list.count!=0) {
      LOG_D(RRC, "SRB_configList2(%p) count is %d\n           SRB_configList2->list.array[0] addr is %p",
            SRB_configList2, (*SRB_configList2)->list.count,  (*SRB_configList2)->list.array[0]);
    }

    for (i = 0; (i < (*SRB_configList2)->list.count) && (i < 3); i++) {
      if ((*SRB_configList2)->list.array[i]->srb_Identity == 2 ) {
        LOG_D(RRC, "get SRB2_config from (ue_context_pP->ue_context.SRB_configList2[%d])\n", xid);
        SRB2_config = (*SRB_configList2)->list.array[i];
        break;
      }
    }
  }

  SRB_configList2 = &(ue_context_pP->ue_context.SRB_configList2[next_xid]);
  DRB_configList2 = &(ue_context_pP->ue_context.DRB_configList2[next_xid]);

  if (*SRB_configList2) {
    free(*SRB_configList2);
    LOG_D(RRC, "free(ue_context_pP->ue_context.SRB_configList2[%d])\n", next_xid);
  }

  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));

  if (SRB2_config != NULL) {
    // Add SRB2 to SRB configuration list
    ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);
    ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);
    LOG_D(RRC, "Add SRB2_config (srb_Identity:%ld) to ue_context_pP->ue_context.SRB_configList\n",
          SRB2_config->srb_Identity);
    LOG_D(RRC, "Add SRB2_config (srb_Identity:%ld) to ue_context_pP->ue_context.SRB_configList2[%d]\n",
          SRB2_config->srb_Identity, next_xid);
  } else {
    // SRB configuration list only contains SRB1.
    LOG_W(RRC,"SRB2 configuration does not exist in SRB configuration list\n");
  }

  if (*DRB_configList2) {
    free(*DRB_configList2);
    LOG_D(RRC, "free(ue_context_pP->ue_context.DRB_configList2[%d])\n", next_xid);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));

  if (DRB_configList != NULL) {
    LOG_D(RRC, "get DRB_config from (ue_context_pP->ue_context.DRB_configList)\n");

    for (i = 0; (i < DRB_configList->list.count) && (i < 3); i++) {
      DRB_config = DRB_configList->list.array[i];
      // Add DRB to DRB configuration list, for LTE_RRCConnectionReconfigurationComplete
      ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
    }
  }

  ue_context_pP->ue_context.Srb1.Active = 1;
  //ue_context_pP->ue_context.Srb2.Srb_info.Srb_id = 2;

  if (EPC_MODE_ENABLED) {
    hashtable_rc_t    h_rc;
    int               j;
    rrc_ue_s1ap_ids_t *rrc_ue_s1ap_ids_p = NULL;
    uint16_t ue_initial_id = ue_context_pP->ue_context.ue_initial_id;
    uint32_t eNB_ue_s1ap_id = ue_context_pP->ue_context.eNB_ue_s1ap_id;
    eNB_RRC_INST *rrc_instance_p = RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)];

    if (eNB_ue_s1ap_id > 0) {
      h_rc = hashtable_get(rrc_instance_p->s1ap_id2_s1ap_ids, (hash_key_t)eNB_ue_s1ap_id, (void **)&rrc_ue_s1ap_ids_p);

      if  (h_rc == HASH_TABLE_OK) {
        rrc_ue_s1ap_ids_p->ue_rnti = ctxt_pP->rnti;
      }
    }

    if (ue_initial_id != 0) {
      h_rc = hashtable_get(rrc_instance_p->initial_id2_s1ap_ids, (hash_key_t)ue_initial_id, (void **)&rrc_ue_s1ap_ids_p);

      if  (h_rc == HASH_TABLE_OK) {
        rrc_ue_s1ap_ids_p->ue_rnti = ctxt_pP->rnti;
      }
    }

    gtpv1u_enb_create_tunnel_req_t  create_tunnel_req;
    /* Save e RAB information for later */
    memset(&create_tunnel_req, 0, sizeof(create_tunnel_req));

    for ( j = 0, i = 0; i < NB_RB_MAX; i++) {
      if (ue_context_pP->ue_context.e_rab[i].status == E_RAB_STATUS_ESTABLISHED || ue_context_pP->ue_context.e_rab[i].status == E_RAB_STATUS_DONE) {
        create_tunnel_req.eps_bearer_id[j]   = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
        create_tunnel_req.sgw_S1u_teid[j]  = ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
        memcpy(&create_tunnel_req.sgw_addr[j],
               &ue_context_pP->ue_context.e_rab[i].param.sgw_addr,
               sizeof(transport_layer_addr_t));
        j++;
      }
    }

    create_tunnel_req.rnti       = ctxt_pP->rnti; // warning put zero above
    create_tunnel_req.num_tunnels    = j;
    ret = gtpv1u_update_s1u_tunnel(
            ctxt_pP->instance,
            &create_tunnel_req,
            reestablish_rnti);

    if ( ret != 0 ) {
      LOG_E(RRC,"gtpv1u_update_s1u_tunnel failed,start to release UE %x\n",reestablish_rnti);

      // update s1u tunnel failed,reset rnti?
      if (eNB_ue_s1ap_id > 0) {
        h_rc = hashtable_get(rrc_instance_p->s1ap_id2_s1ap_ids, (hash_key_t)eNB_ue_s1ap_id, (void **)&rrc_ue_s1ap_ids_p);

        if (h_rc == HASH_TABLE_OK ) {
          rrc_ue_s1ap_ids_p->ue_rnti = reestablish_rnti;
        }
      }

      if (ue_initial_id != 0) {
        h_rc = hashtable_get(rrc_instance_p->initial_id2_s1ap_ids, (hash_key_t)ue_initial_id, (void **)&rrc_ue_s1ap_ids_p);

        if (h_rc == HASH_TABLE_OK ) {
          rrc_ue_s1ap_ids_p->ue_rnti = reestablish_rnti;
        }
      }

      ue_context_pP->ue_context.ue_release_timer_s1 = 1;
      ue_context_pP->ue_context.ue_release_timer_thres_s1 = 100;
      ue_context_pP->ue_context.ue_release_timer = 0;
      ue_context_pP->ue_context.ue_reestablishment_timer = 0;
      ue_context_pP->ue_context.ul_failure_timer = 20000; // set ul_failure to 20000 for triggering rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_REQ
      rrc_eNB_free_UE(ctxt_pP->module_id,ue_context_pP);
      ue_context_pP->ue_context.ul_failure_timer = 0;
      put_UE_in_freelist(ctxt_pP->module_id, ctxt_pP->rnti, 0);
      return;
    }
  } /* EPC_MODE_ENABLED */

  /* Update RNTI in ue_context */
  ue_context_pP->ue_id_rnti                    = ctxt_pP->rnti; // here ue_id_rnti is just a key, may be something else
  ue_context_pP->ue_context.rnti               = ctxt_pP->rnti;

  if (EPC_MODE_ENABLED) {
    uint8_t send_security_mode_command = FALSE;
    rrc_pdcp_config_security(
      ctxt_pP,
      ue_context_pP,
      send_security_mode_command);
    LOG_D(RRC, "set security successfully \n");
  }

  // Measurement ID list
  MeasId_list = CALLOC(1, sizeof(*MeasId_list));
  memset((void *)MeasId_list, 0, sizeof(*MeasId_list));
  MeasId0 = CALLOC(1, sizeof(*MeasId0));
  MeasId0->measId = 1;
  MeasId0->measObjectId = 1;
  MeasId0->reportConfigId = 1;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId0);
  MeasId1 = CALLOC(1, sizeof(*MeasId1));
  MeasId1->measId = 2;
  MeasId1->measObjectId = 1;
  MeasId1->reportConfigId = 2;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId1);
  MeasId2 = CALLOC(1, sizeof(*MeasId2));
  MeasId2->measId = 3;
  MeasId2->measObjectId = 1;
  MeasId2->reportConfigId = 3;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId2);
  MeasId3 = CALLOC(1, sizeof(*MeasId3));
  MeasId3->measId = 4;
  MeasId3->measObjectId = 1;
  MeasId3->reportConfigId = 4;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId3);
  MeasId4 = CALLOC(1, sizeof(*MeasId4));
  MeasId4->measId = 5;
  MeasId4->measObjectId = 1;
  MeasId4->reportConfigId = 5;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId4);
  MeasId5 = CALLOC(1, sizeof(*MeasId5));
  MeasId5->measId = 6;
  MeasId5->measObjectId = 1;
  MeasId5->reportConfigId = 6;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId5);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measIdToAddModList = MeasId_list;
  // Add one EUTRA Measurement Object
  MeasObj_list = CALLOC(1, sizeof(*MeasObj_list));
  memset((void *)MeasObj_list, 0, sizeof(*MeasObj_list));
  // Configure MeasObject
  MeasObj = CALLOC(1, sizeof(*MeasObj));
  memset((void *)MeasObj, 0, sizeof(*MeasObj));
  MeasObj->measObjectId = 1;
  MeasObj->measObject.present = LTE_MeasObjectToAddMod__measObject_PR_measObjectEUTRA;
  MeasObj->measObject.choice.measObjectEUTRA.carrierFreq = 3350; //band 7, 2.68GHz
  //MeasObj->measObject.choice.measObjectEUTRA.carrierFreq = 36090; //band 33, 1.909GHz
  MeasObj->measObject.choice.measObjectEUTRA.allowedMeasBandwidth = LTE_AllowedMeasBandwidth_mbw25;
  MeasObj->measObject.choice.measObjectEUTRA.presenceAntennaPort1 = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf = CALLOC(1, sizeof(uint8_t));
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf[0] = 0;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.size = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.bits_unused = 6;
  MeasObj->measObject.choice.measObjectEUTRA.offsetFreq = NULL;   // Default is 15 or 0dB
  MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList =
    (LTE_CellsToAddModList_t *) CALLOC(1, sizeof(*CellsToAddModList));
  CellsToAddModList = MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList;

  // Add adjacent cell lists (6 per eNB)
  for (i = 0; i < 6; i++) {
    CellToAdd = (LTE_CellsToAddMod_t *) CALLOC(1, sizeof(*CellToAdd));
    CellToAdd->cellIndex = i + 1;
    CellToAdd->physCellId = get_adjacent_cell_id(ctxt_pP->module_id, i);
    CellToAdd->cellIndividualOffset = LTE_Q_OffsetRange_dB0;
    ASN_SEQUENCE_ADD(&CellsToAddModList->list, CellToAdd);
  }

  ASN_SEQUENCE_ADD(&MeasObj_list->list, MeasObj);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measObjectToAddModList = MeasObj_list;
  // Report Configurations for periodical, A1-A5 events
  ReportConfig_list = CALLOC(1, sizeof(*ReportConfig_list));
  ReportConfig_per = CALLOC(1, sizeof(*ReportConfig_per));
  ReportConfig_A1 = CALLOC(1, sizeof(*ReportConfig_A1));
  ReportConfig_A2 = CALLOC(1, sizeof(*ReportConfig_A2));
  ReportConfig_A3 = CALLOC(1, sizeof(*ReportConfig_A3));
  ReportConfig_A4 = CALLOC(1, sizeof(*ReportConfig_A4));
  ReportConfig_A5 = CALLOC(1, sizeof(*ReportConfig_A5));
  ReportConfig_per->reportConfigId = 1;
  ReportConfig_per->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_periodical;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.choice.periodical.purpose =
    LTE_ReportConfigEUTRA__triggerType__periodical__purpose_reportStrongestCells;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_per);
  ReportConfig_A1->reportConfigId = 2;
  ReportConfig_A1->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA1;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A1);

  if (RC.rrc[ctxt_pP->module_id]->HO_flag == 1 /*HO_MEASUREMENT */ ) {
    LOG_I(RRC, "[eNB %d] frame %d: requesting A2, A3, A4, A5, and A6 event reporting\n",
          ctxt_pP->module_id, ctxt_pP->frame);
    ReportConfig_A2->reportConfigId = 3;
    ReportConfig_A2->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.present =
      LTE_ReportConfigEUTRA__triggerType_PR_event;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
      LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA2;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA2.a2_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA2.a2_Threshold.choice.threshold_RSRP = 10;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
      LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
    ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
    ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A2);
    ReportConfig_A3->reportConfigId = 4;
    ReportConfig_A3->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.present =
      LTE_ReportConfigEUTRA__triggerType_PR_event;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
      LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA3;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset = 1;   //10;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA3.reportOnLeave = 1;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
      LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis = 0.5; // FIXME ...hysteresis is of type long!
    ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger =
      LTE_TimeToTrigger_ms40;
    ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A3);
    ReportConfig_A4->reportConfigId = 5;
    ReportConfig_A4->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.present =
      LTE_ReportConfigEUTRA__triggerType_PR_event;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
      LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA4;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA4.a4_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA4.a4_Threshold.choice.threshold_RSRP = 10;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
      LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
    ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
    ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A4);
    ReportConfig_A5->reportConfigId = 6;
    ReportConfig_A5->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.present =
      LTE_ReportConfigEUTRA__triggerType_PR_event;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
      LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA5;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA5.a5_Threshold1.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA5.a5_Threshold2.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA5.a5_Threshold1.choice.threshold_RSRP = 10;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
    eventA5.a5_Threshold2.choice.threshold_RSRP = 10;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
      LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
    ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
    ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A5);
    //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->reportConfigToAddModList = ReportConfig_list;
#if 0
    /* TODO: set a proper value.
     * 20 means: UE does not report if RSRP of serving cell is higher
     * than -120 dB (see 36.331 5.5.3.1).
     * This is too low for the X2 handover experiment.
     */
    rsrp = CALLOC(1, sizeof(LTE_RSRP_Range_t));
    *rsrp = 20;
#endif
    Sparams = CALLOC(1, sizeof(*Sparams));
    Sparams->present = LTE_MeasConfig__speedStatePars_PR_setup;
    Sparams->choice.setup.timeToTrigger_SF.sf_High = LTE_SpeedStateScaleFactors__sf_Medium_oDot75;
    Sparams->choice.setup.timeToTrigger_SF.sf_Medium = LTE_SpeedStateScaleFactors__sf_High_oDot5;
    Sparams->choice.setup.mobilityStateParameters.n_CellChangeHigh = 10;
    Sparams->choice.setup.mobilityStateParameters.n_CellChangeMedium = 5;
    Sparams->choice.setup.mobilityStateParameters.t_Evaluation = LTE_MobilityStateParameters__t_Evaluation_s60;
    Sparams->choice.setup.mobilityStateParameters.t_HystNormal = LTE_MobilityStateParameters__t_HystNormal_s120;
    quantityConfig = CALLOC(1, sizeof(*quantityConfig));
    memset((void *)quantityConfig, 0, sizeof(*quantityConfig));
    quantityConfig->quantityConfigEUTRA = CALLOC(1, sizeof(struct LTE_QuantityConfigEUTRA));
    memset((void *)quantityConfig->quantityConfigEUTRA, 0, sizeof(*quantityConfig->quantityConfigEUTRA));
    quantityConfig->quantityConfigCDMA2000 = NULL;
    quantityConfig->quantityConfigGERAN = NULL;
    quantityConfig->quantityConfigUTRA = NULL;
    quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP =
      CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP)));
    quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ =
      CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ)));
    *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = LTE_FilterCoefficient_fc4;
    *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = LTE_FilterCoefficient_fc4;
    LOG_I(RRC,
          "[eNB %d] Frame %d: potential handover preparation: store the information in an intermediate structure in case of failure\n",
          ctxt_pP->module_id, ctxt_pP->frame);
    // store the information in an intermediate structure for Hanodver management
    //rrc_inst->handover_info.as_config.sourceRadioResourceConfig.srb_ToAddModList = CALLOC(1,sizeof());
    ue_context_pP->ue_context.handover_info = CALLOC(1, sizeof(*(ue_context_pP->ue_context.handover_info)));
    //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.srb_ToAddModList,(void *)SRB_list,sizeof(LTE_SRB_ToAddModList_t));
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.srb_ToAddModList = *SRB_configList2;
    //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.drb_ToAddModList,(void *)DRB_list,sizeof(LTE_DRB_ToAddModList_t));
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.drb_ToAddModList = DRB_configList;
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.drb_ToReleaseList = NULL;
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig =
      CALLOC(1, sizeof(*ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig));
    memcpy((void *)ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig,
           (void *)ue_context_pP->ue_context.mac_MainConfig, sizeof(LTE_MAC_MainConfig_t));
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.physicalConfigDedicated =
      CALLOC(1, sizeof(LTE_PhysicalConfigDedicated_t));
    memcpy((void *)ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.physicalConfigDedicated,
           (void *)ue_context_pP->ue_context.physicalConfigDedicated, sizeof(LTE_PhysicalConfigDedicated_t));
    ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.sps_Config = NULL;
    //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.sps_Config,(void *)rrc_inst->sps_Config[ue_mod_idP],sizeof(SPS_Config_t));
  }

  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
      memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
      OCTET_STRING_fromBuf(dedicatedInfoNas,
                           (char *)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                           ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
      LOG_D(RRC, "Add LTE_DedicatedInfoNAS(%d) to LTE_DedicatedInfoNASList\n", i);
      ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
    }

    /* TODO parameters yet to process ... */
    {
      //      ue_context_pP->ue_context.e_rab[i].param.qos;
      //      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      //      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
    }
    /* TODO should test if e RAB are Ok before! */
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_DONE;
    ue_context_pP->ue_context.e_rab[i].xid    = xid;
    LOG_D(RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
          i, ue_context_pP->ue_context.e_rab[i].status, "E_RAB_STATUS_DONE");
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList->list.count == 0) {
    free(dedicatedInfoNASList);
    dedicatedInfoNASList = NULL;
  }

  measurements_enabled = RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2 ||
                         RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_measurement_reports;
  // send LTE_RRCConnectionReconfiguration
  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         next_xid,   //Transaction_id,
                                         (LTE_SRB_ToAddModList_t *)*SRB_configList2, // SRB_configList
                                         (LTE_DRB_ToAddModList_t *)DRB_configList,
                                         (LTE_DRB_ToReleaseList_t *)NULL, // DRB2_list,
                                         (struct LTE_SPS_Config *)NULL,   // maybe ue_context_pP->ue_context.sps_Config,
                                         (struct LTE_PhysicalConfigDedicated *)ue_context_pP->ue_context.physicalConfigDedicated,
                                         //#ifdef EXMIMO_IOT
                                         //                                         NULL, NULL, NULL,NULL,
                                         //#else
                                         measurements_enabled ? (LTE_MeasObjectToAddModList_t *)MeasObj_list : NULL, // MeasObj_list,
                                         measurements_enabled ? (LTE_ReportConfigToAddModList_t *)ReportConfig_list : NULL, // ReportConfig_list,
                                         measurements_enabled ? (LTE_QuantityConfig_t *)quantityConfig : NULL, //quantityConfig,
                                         (LTE_MeasIdToAddModList_t *)NULL,
                                         //#endif
                                         (LTE_MAC_MainConfig_t *)ue_context_pP->ue_context.mac_MainConfig,
                                         (LTE_MeasGapConfig_t *)NULL,
                                         (LTE_MobilityControlInfo_t *)NULL,
                                         (LTE_SecurityConfigHO_t *)NULL,
                                         (struct LTE_MeasConfig__speedStatePars *)Sparams, // Sparams,
                                         (LTE_RSRP_Range_t *)rsrp, // rsrp,
                                         (LTE_C_RNTI_t *)cba_RNTI, // cba_RNTI
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)dedicatedInfoNASList, //dedicatedInfoNASList
                                         (LTE_SL_CommConfig_r12_t *)NULL,
                                         (LTE_SL_DiscConfig_r12_t *)NULL,
                                         (LTE_SCellToAddMod_r10_t *)NULL
                                        );
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  if(size==65535) {
    LOG_E(RRC,"RRC decode err!!! do_RRCConnectionReconfiguration\n");
    put_UE_in_freelist(ctxt_pP->module_id, reestablish_rnti, 0);
    return;
  } else {
    LOG_I(RRC,
          "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE id %x)\n",
          ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
    LOG_D(RRC,
          "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
          ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH);
    MSC_LOG_TX_MESSAGE(
      MSC_RRC_ENB,
      MSC_RRC_UE,
      buffer,
      size,
      MSC_AS_TIME_FMT" LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
      MSC_AS_TIME_ARGS(ctxt_pP),
      ue_context_pP->ue_context.rnti,
      rrc_eNB_mui,
      size);
    rrc_data_req(
      ctxt_pP,
      DCCH,
      rrc_eNB_mui++,
      SDU_CONFIRM_NO,
      size,
      buffer,
      PDCP_TRANSMISSION_MODE_CONTROL);
  }

  // delete UE data of prior RNTI.  UE use current RNTI.
  //  protocol_ctxt_t ctxt_prior = *ctxt_pP;
  //  ctxt_prior.rnti = reestablish_rnti;
  //
  //  LTE_eNB_ULSCH_t *ulsch = NULL;
  //  nfapi_ul_config_request_body_t *ul_req_tmp = NULL;
  //  PHY_VARS_eNB *eNB_PHY = NULL;
  //  eNB_MAC_INST *eNB_MAC = RC.mac[ctxt_prior.module_id];
  //  for (int CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
  //    eNB_PHY = RC.eNB[ctxt_prior.module_id][CC_id];
  //    for (int i=0; i<MAX_MOBILES_PER_ENB; i++) {
  //      ulsch = eNB_PHY->ulsch[i];
  //      if((ulsch != NULL) && (ulsch->rnti == ctxt_prior.rnti)){
  //        void clean_eNb_ulsch(LTE_eNB_ULSCH_t *ulsch);
  //        LOG_I(RRC, "clean_eNb_ulsch UE %x \n", ctxt_prior.rnti);
  //        clean_eNb_ulsch(ulsch);
  //        break;
  //      }
  //    }
  //
  //    for(int j = 0; j < 10; j++){
  //      ul_req_tmp = &eNB_MAC->UL_req_tmp[CC_id][j].ul_config_request_body;
  //      if(ul_req_tmp){
  //        int pdu_number = ul_req_tmp->number_of_pdus;
  //        for(int pdu_index = pdu_number-1; pdu_index >= 0; pdu_index--){
  //          if(ul_req_tmp->ul_config_pdu_list[pdu_index].ulsch_pdu.ulsch_pdu_rel8.rnti == ctxt_prior.rnti){
  //            LOG_I(RRC, "remove UE %x from ul_config_pdu_list %d/%d\n", ctxt_prior.rnti, pdu_index, pdu_number);
  //            if(pdu_index < pdu_number -1){
  //               memcpy(&ul_req_tmp->ul_config_pdu_list[pdu_index], &ul_req_tmp->ul_config_pdu_list[pdu_index+1], (pdu_number-1-pdu_index) * sizeof(nfapi_ul_config_request_pdu_t));
  //            }
  //            ul_req_tmp->number_of_pdus--;
  //          }
  //        }
  //      }
  //    }
  //  }
  //  rrc_mac_remove_ue(ctxt_prior.module_id, ctxt_prior.rnti);
  //  rrc_rlc_remove_ue(&ctxt_prior);
  //  pdcp_remove_UE(&ctxt_prior);
  // add UE info to freeList for RU_thread to remove the UE instead of remove it here
  LOG_I(RRC, "[RRCConnectionReestablishment]put UE %x into freeList\n", reestablish_rnti);
  put_UE_in_freelist(ctxt_pP->module_id, reestablish_rnti, 0);
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionReestablishmentReject(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t          *const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{
  if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    int UE_id = find_UE_id(ctxt_pP->module_id, ctxt_pP->rnti);

    if(UE_id != -1) {
      RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 1;
      RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer_thres = 20;
    } else {
      LOG_E(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" Generating LTE_RRCConnectionReestablishmentReject without UE_id(MAC) rnti %x\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ctxt_pP->rnti);
    }
  }

  T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_REJECT, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  eNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  ue_p->Srb0.Tx_buffer.payload_size =
    do_RRCConnectionReestablishmentReject(ctxt_pP->module_id,
                                          (uint8_t *) ue_p->Srb0.Tx_buffer.Payload);
  LOG_DUMPMSG(RRC,DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRCConnectionReestablishmentReject\n");
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    ue_p->Srb0.Tx_buffer.Header,
    ue_p->Srb0.Tx_buffer.payload_size,
    MSC_AS_TIME_FMT" LTE_RRCConnectionReestablishmentReject UE %x size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP == NULL ? -1 : ue_context_pP->ue_context.rnti,
    ue_p->Srb0.Tx_buffer.payload_size);
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating LTE_RRCConnectionReestablishmentReject (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ue_p->Srb0.Tx_buffer.payload_size);
}
#if 0
void rrc_generate_SgNBReleaseRequest(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  MessageDef      *msg;
  msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_ENDC_SGNB_RELEASE_REQ);
  memset(&(X2AP_ENDC_SGNB_RELEASE_REQ(msg)), 0, sizeof(x2ap_ENDC_sgnb_release_req_t));
  // X2AP_ENDC_SGNB_RELEASE_REQ(msg).MeNB_ue_x2_id = ;
  // X2AP_ENDC_SGNB_RELEASE_REQ(msg).SgNB_ue_x2_id = ;
  // X2AP_ENDC_SGNB_RELEASE_REQ(msg).cause = ;
  // X2AP_ENDC_SGNB_RELEASE_REQ(msg).target_physCellId = ;
  LOG_I(RRC,
        "[eNB %d] frame %d UE rnti %x transmitting sgnb release request to sgnb \n",
        ctxt_pP->module_id,ctxt_pP->frame,ctxt_pP->rnti);
  itti_send_msg_to_task(TASK_X2AP, ENB_MODULE_ID_TO_INSTANCE(ctxt_pP->module_id), msg);
  return;
}
#endif
//-----------------------------------------------------------------------------
/*
* Generate the RRC Connection Release to UE.
* If received, UE should switch to RRC_IDLE mode.
*/
void
rrc_eNB_generate_RRCConnectionRelease(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t *const ue_context_pP
)
//-----------------------------------------------------------------------------
{
  uint8_t buffer[RRC_BUF_SIZE];
  uint16_t size = 0;
  int release_num;

  memset(buffer, 0, RRC_BUF_SIZE);
  T(T_ENB_RRC_CONNECTION_RELEASE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
#if 0

  if(ue_context_pP != NULL) {
    if(ue_context_pP->ue_context.Status == RRC_NR_NSA) {
      //rrc_eNB_generate_SgNBReleaseRequest(ctxt_pP,ue_context_pP);
    }
  }

#endif
  size = do_RRCConnectionRelease(ctxt_pP->module_id, buffer,rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id));
  ue_context_pP->ue_context.ue_reestablishment_timer = 0;
  ue_context_pP->ue_context.ue_release_timer = 0;
  ue_context_pP->ue_context.ue_rrc_inactivity_timer = 0;
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Logical Channel DL-DCCH, Generate RRCConnectionRelease (bytes %d)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size);
  LOG_D(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" --- PDCP_DATA_REQ/%d Bytes (rrcConnectionRelease MUI %d) --->[PDCP][RB %u]\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        size,
        rrc_eNB_mui,
        DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" LTE_RRCConnectionRelease UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  pthread_mutex_lock(&rrc_release_freelist);

  for (release_num = 0; release_num < NUMBER_OF_UE_MAX; release_num++) {
    if (rrc_release_info.RRC_release_ctrl[release_num].flag == 0) {
      if (ue_context_pP->ue_context.ue_release_timer_s1 > 0) {
        rrc_release_info.RRC_release_ctrl[release_num].flag = 1;
      } else {
        rrc_release_info.RRC_release_ctrl[release_num].flag = 2;
      }

      rrc_release_info.RRC_release_ctrl[release_num].rnti = ctxt_pP->rnti;
      rrc_release_info.RRC_release_ctrl[release_num].rrc_eNB_mui = rrc_eNB_mui;
      rrc_release_info.num_UEs++;
      LOG_D(RRC, "Generate DLSCH Release send: index %d rnti %x mui %d flag %d \n",
            release_num,
            ctxt_pP->rnti,
            rrc_eNB_mui,
            rrc_release_info.RRC_release_ctrl[release_num].flag);
      break;
    }
  }

  /* TODO: what to do if RRC_release_ctrl is full? For now, exit. */
  if (release_num == NUMBER_OF_UE_MAX) {
    LOG_E(RRC, "fatal: rrc_release_info.RRC_release_ctrl is full\n");
    exit(1);
  }

  pthread_mutex_unlock(&rrc_release_freelist);

  if (NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    MessageDef *m = itti_alloc_new_message(TASK_RRC_ENB, F1AP_UE_CONTEXT_RELEASE_CMD);
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rnti = ctxt_pP->rnti;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).cause = F1AP_CAUSE_RADIO_NETWORK;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).cause_value = 10; // 10 = F1AP_CauseRadioNetwork_normal_release
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container = buffer;
    F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container_length = size;
    itti_send_msg_to_task(TASK_CU_F1, ctxt_pP->module_id, m);
  } else {
    rrc_data_req(ctxt_pP,
                 DCCH,
                 rrc_eNB_mui++,
                 SDU_CONFIRM_NO,
                 size,
                 buffer,
                 PDCP_TRANSMISSION_MODE_CONTROL);
  }
}

uint8_t qci_to_priority[9]= {2,4,3,5,1,6,7,8,9};

// TBD: this directive can be remived if we create a similar e_rab_param_t structure in RRC context
//-----------------------------------------------------------------------------
void
rrc_eNB_generate_dedicatedRRCConnectionReconfiguration(const protocol_ctxt_t *const ctxt_pP,
    rrc_eNB_ue_context_t          *const ue_context_pP,
    const uint8_t                ho_state
                                                      )
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;
  int i;
  struct LTE_DRB_ToAddMod                *DRB_config                       = NULL;
  struct LTE_RLC_Config                  *DRB_rlc_config                   = NULL;
  struct LTE_PDCP_Config                 *DRB_pdcp_config                  = NULL;
  struct LTE_PDCP_Config__rlc_AM         *PDCP_rlc_AM                      = NULL;
  struct LTE_PDCP_Config__rlc_UM         *PDCP_rlc_UM                      = NULL;
  struct LTE_LogicalChannelConfig        *DRB_lchan_config                 = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *DRB_ul_SpecificParameters        = NULL;
  //  LTE_DRB_ToAddModList_t**                DRB_configList=&ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t                *DRB_configList=ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t                **DRB_configList2=NULL;
  //DRB_ToAddModList_t**                RRC_DRB_configList=&ue_context_pP->ue_context.DRB_configList;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;
  long  *logicalchannelgroup_drb;
  //  int drb_identity_index=0;
  uint8_t xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);   //Transaction_id,
  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid];

  if (*DRB_configList2) {
    free(*DRB_configList2);
  }

  //*DRB_configList = CALLOC(1, sizeof(*DRB_configList));
  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));
  int e_rab_done=0;

  for ( i = 0  ;
        i < ue_context_pP->ue_context.setup_e_rabs ;
        i++) {
    if (e_rab_done >= ue_context_pP->ue_context.nb_of_e_rabs) {
      break;
    }

    // bypass the new and already configured erabs
    if (ue_context_pP->ue_context.e_rab[i].status >= E_RAB_STATUS_DONE) {
      //      drb_identity_index++;
      continue;
    }

    DRB_config = CALLOC(1, sizeof(*DRB_config));
    DRB_config->eps_BearerIdentity = CALLOC(1, sizeof(long));
    // allowed value 5..15, value : x+4
    *(DRB_config->eps_BearerIdentity) = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;//+ 4; // especial case generation
    //   DRB_config->drb_Identity =  1 + drb_identity_index + e_rab_done;// + i ;// (LTE_DRB_Identity_t) ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
    // 1 + drb_identiy_index;
    DRB_config->drb_Identity = i+1;
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

    switch (ue_context_pP->ue_context.e_rab[i].param.qos.qci) {
      /*
       * type: realtime data with medium packer error rate
       * action: swtich to RLC UM
       */
      case 1: // 100ms, 10^-2, p2, GBR
      case 2: // 150ms, 10^-3, p4, GBR
      case 3: // 50ms, 10^-3, p3, GBR
      case 4:  // 300ms, 10^-6, p5
      case 7: // 100ms, 10^-3, p7, GBR
      case 9: // 300ms, 10^-6, p9
      case 65: // 75ms, 10^-2, p0.7, mission critical voice, GBR
      case 66: // 100ms, 10^-2, p2, non-mission critical  voice , GBR
        // RLC
        DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
        DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
        DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
        DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
        // PDCP
        PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
        DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
        PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
        break;

      /*
       * type: non-realtime data with low packer error rate
       * action: swtich to RLC AM
       */
      case 5:  // 100ms, 10^-6, p1 , IMS signaling
      case 6:  // 300ms, 10^-6, p6
      case 8: // 300ms, 10^-6, p8
      case 69: // 60ms, 10^-6, p0.5, mission critical delay sensitive data, Lowest Priority
      case 70: // 200ms, 10^-6, p5.5, mision critical data
        // RLC
        DRB_rlc_config->present = LTE_RLC_Config_PR_am;
        DRB_rlc_config->choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
        DRB_rlc_config->choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
        DRB_rlc_config->choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
        DRB_rlc_config->choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
        DRB_rlc_config->choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
        DRB_rlc_config->choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;
        // PDCP
        PDCP_rlc_AM = CALLOC(1, sizeof(*PDCP_rlc_AM));
        DRB_pdcp_config->rlc_AM = PDCP_rlc_AM;
        PDCP_rlc_AM->statusReportRequired = FALSE;
        break;

      default :
        LOG_E(RRC,"not supported qci %d\n", ue_context_pP->ue_context.e_rab[i].param.qos.qci);
        ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_FAILED;
        ue_context_pP->ue_context.e_rab[i].xid = xid;
        e_rab_done++;
        free(DRB_pdcp_config->discardTimer);
        free(DRB_pdcp_config);
        free(DRB_rlc_config);
        free(DRB_config->logicalChannelIdentity);
        free(DRB_config->eps_BearerIdentity);
        free(DRB_config);
        continue;
    }

    DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
    DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
    DRB_config->logicalChannelConfig = DRB_lchan_config;
    DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
    DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;

    if (ue_context_pP->ue_context.e_rab[i].param.qos.qci < 9 )
      DRB_ul_SpecificParameters->priority = qci_to_priority[ue_context_pP->ue_context.e_rab[i].param.qos.qci-1] + 3;
    // ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.priority_level;
    else
      DRB_ul_SpecificParameters->priority= 4;

    DRB_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8;
    //LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
    DRB_ul_SpecificParameters->bucketSizeDuration =
      LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
    logicalchannelgroup_drb = CALLOC(1, sizeof(long));
    *logicalchannelgroup_drb = 1;//(i+1) % 3;
    DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;
    ASN_SEQUENCE_ADD(&DRB_configList->list, DRB_config);
    ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
    //ue_context_pP->ue_context.DRB_configList2[drb_identity_index] = &(*DRB_configList);
    LOG_I(RRC,"EPS ID %ld, DRB ID %ld (index %d), QCI %d, priority %ld, LCID %ld LCGID %ld \n",
          *DRB_config->eps_BearerIdentity,
          DRB_config->drb_Identity, i,
          ue_context_pP->ue_context.e_rab[i].param.qos.qci,
          DRB_ul_SpecificParameters->priority,
          *(DRB_config->logicalChannelIdentity),
          *DRB_ul_SpecificParameters->logicalChannelGroup
         );
    e_rab_done++;
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_DONE;
    ue_context_pP->ue_context.e_rab[i].xid = xid;
    {
      if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
        dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
        memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
        OCTET_STRING_fromBuf(dedicatedInfoNas,
                             (char *)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                             ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
        ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
        LOG_I(RRC,"add NAS info with size %d (rab id %d)\n",ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length, i);
      } else {
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
  if (dedicatedInfoNASList != NULL) {
    if (dedicatedInfoNASList->list.count == 0) {
      free(dedicatedInfoNASList);
      dedicatedInfoNASList = NULL;
      LOG_W(RRC,"dedlicated NAS list is empty, free the list and reset the address\n");
    }
  } else {
    LOG_W(RRC,"dedlicated NAS list is empty\n");
  }

  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         xid,
                                         (LTE_SRB_ToAddModList_t *)NULL,
                                         (LTE_DRB_ToAddModList_t *)*DRB_configList2,
                                         (LTE_DRB_ToReleaseList_t *)NULL, // DRB2_list,
                                         (struct LTE_SPS_Config *)NULL,   // *sps_Config,
                                         NULL, NULL, NULL, NULL,NULL,
                                         NULL, NULL,  NULL, NULL, NULL, NULL, NULL,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)dedicatedInfoNASList,
                                         (LTE_SL_CommConfig_r12_t *)NULL,
                                         (LTE_SL_DiscConfig_r12_t *)NULL,
                                         (LTE_SCellToAddMod_r10_t *)NULL
                                        );
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,"[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE RNTI %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" dedicated LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
}
int
rrc_eNB_modify_dedicatedRRCConnectionReconfiguration(const protocol_ctxt_t *const ctxt_pP,
    rrc_eNB_ue_context_t          *const ue_context_pP,
    const uint8_t                ho_state
                                                    )
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size;
  int i, j;
  struct LTE_DRB_ToAddMod                *DRB_config                       = NULL;
  struct LTE_RLC_Config                  *DRB_rlc_config                   = NULL;
  struct LTE_PDCP_Config                 *DRB_pdcp_config                  = NULL;
  struct LTE_PDCP_Config__rlc_AM         *PDCP_rlc_AM                      = NULL;
  struct LTE_PDCP_Config__rlc_UM         *PDCP_rlc_UM                      = NULL;
  struct LTE_LogicalChannelConfig        *DRB_lchan_config                 = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *DRB_ul_SpecificParameters        = NULL;
  LTE_DRB_ToAddModList_t                 *DRB_configList = ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t                *DRB_configList2 = NULL;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;
  uint8_t xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);   // Transaction_id,
  DRB_configList2 = CALLOC(1, sizeof(*DRB_configList2));
  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));

  for (i = 0; i < ue_context_pP->ue_context.nb_of_modify_e_rabs; i++) {
    // bypass the new and already configured erabs
    if (ue_context_pP->ue_context.modify_e_rab[i].status >= E_RAB_STATUS_DONE) {
      ue_context_pP->ue_context.modify_e_rab[i].xid = xid;
      continue;
    }

    if (ue_context_pP->ue_context.modify_e_rab[i].cause != S1AP_CAUSE_NOTHING) {
      // set xid of failure RAB
      ue_context_pP->ue_context.modify_e_rab[i].xid = xid;
      ue_context_pP->ue_context.modify_e_rab[i].status = E_RAB_STATUS_FAILED;
      continue;
    }

    DRB_config = NULL;

    // search exist DRB_config
    for (j = 0; j < DRB_configList->list.count; j++) {
      if((uint8_t)*(DRB_configList->list.array[j]->eps_BearerIdentity) == ue_context_pP->ue_context.modify_e_rab[i].param.e_rab_id) {
        DRB_config = DRB_configList->list.array[j];
        break;
      }
    }

    if (NULL == DRB_config) {
      ue_context_pP->ue_context.modify_e_rab[i].xid = xid;
      ue_context_pP->ue_context.modify_e_rab[i].status = E_RAB_STATUS_FAILED;
      // TODO use which cause
      ue_context_pP->ue_context.modify_e_rab[i].cause = S1AP_CAUSE_RADIO_NETWORK;
      ue_context_pP->ue_context.modify_e_rab[i].cause_value = 0;//S1ap_CauseRadioNetwork_unspecified;
      ue_context_pP->ue_context.nb_of_failed_e_rabs++;
      continue;
    }

    DRB_rlc_config = DRB_config->rlc_Config;
    DRB_pdcp_config = DRB_config->pdcp_Config;
    *DRB_pdcp_config->discardTimer = LTE_PDCP_Config__discardTimer_infinity;

    switch (ue_context_pP->ue_context.modify_e_rab[i].param.qos.qci) {
      /*
       * type: realtime data with medium packer error rate
       * action: swtich to RLC UM
       */
      case 1: // 100ms, 10^-2, p2, GBR
      case 2: // 150ms, 10^-3, p4, GBR
      case 3: // 50ms, 10^-3, p3, GBR
      case 4:  // 300ms, 10^-6, p5
      case 7: // 100ms, 10^-3, p7, GBR
      case 9: // 300ms, 10^-6, p9
      case 65: // 75ms, 10^-2, p0.7, mission critical voice, GBR
      case 66: // 100ms, 10^-2, p2, non-mission critical  voice , GBR
        // RLC
        DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
        DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
        DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
        DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;

        // PDCP
        if (DRB_pdcp_config->rlc_AM) {
          free(DRB_pdcp_config->rlc_AM);
          DRB_pdcp_config->rlc_AM = NULL;
        }

        if (DRB_pdcp_config->rlc_UM) {
          free(DRB_pdcp_config->rlc_UM);
          DRB_pdcp_config->rlc_UM = NULL;
        }

        PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
        DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
        PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
        break;

      /*
       * type: non-realtime data with low packer error rate
       * action: swtich to RLC AM
       */
      case 5:  // 100ms, 10^-6, p1 , IMS signaling
      case 6:  // 300ms, 10^-6, p6
      case 8: // 300ms, 10^-6, p8
      case 69: // 60ms, 10^-6, p0.5, mission critical delay sensitive data, Lowest Priority
      case 70: // 200ms, 10^-6, p5.5, mision critical data
        // RLC
        DRB_rlc_config->present = LTE_RLC_Config_PR_am;
        DRB_rlc_config->choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
        DRB_rlc_config->choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
        DRB_rlc_config->choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
        DRB_rlc_config->choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
        DRB_rlc_config->choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
        DRB_rlc_config->choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;

        // PDCP
        if (DRB_pdcp_config->rlc_AM) {
          free(DRB_pdcp_config->rlc_AM);
          DRB_pdcp_config->rlc_AM = NULL;
        }

        if (DRB_pdcp_config->rlc_UM) {
          free(DRB_pdcp_config->rlc_UM);
          DRB_pdcp_config->rlc_UM = NULL;
        }

        PDCP_rlc_AM = CALLOC(1, sizeof(*PDCP_rlc_AM));
        DRB_pdcp_config->rlc_AM = PDCP_rlc_AM;
        PDCP_rlc_AM->statusReportRequired = FALSE;
        break;

      default :
        LOG_E(RRC, "not supported qci %d\n", ue_context_pP->ue_context.modify_e_rab[i].param.qos.qci);
        ue_context_pP->ue_context.modify_e_rab[i].status = E_RAB_STATUS_FAILED;
        ue_context_pP->ue_context.modify_e_rab[i].xid = xid;
        ue_context_pP->ue_context.modify_e_rab[i].cause = S1AP_CAUSE_RADIO_NETWORK;
        ue_context_pP->ue_context.modify_e_rab[i].cause_value = 37;//S1ap_CauseRadioNetwork_not_supported_QCI_value;
        ue_context_pP->ue_context.nb_of_failed_e_rabs++;
        continue;
    }

    DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
    DRB_lchan_config = DRB_config->logicalChannelConfig;
    DRB_ul_SpecificParameters = DRB_lchan_config->ul_SpecificParameters;

    if (ue_context_pP->ue_context.modify_e_rab[i].param.qos.qci < 9 )
      DRB_ul_SpecificParameters->priority = qci_to_priority[ue_context_pP->ue_context.modify_e_rab[i].param.qos.qci-1] + 3;
    else
      DRB_ul_SpecificParameters->priority= 4;

    DRB_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8;
    DRB_ul_SpecificParameters->bucketSizeDuration =
      LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
    ASN_SEQUENCE_ADD(&(DRB_configList2)->list, DRB_config);
    LOG_I(RRC, "EPS ID %ld, DRB ID %ld (index %d), QCI %d, priority %ld, LCID %ld LCGID %ld \n",
          *DRB_config->eps_BearerIdentity,
          DRB_config->drb_Identity, i,
          ue_context_pP->ue_context.modify_e_rab[i].param.qos.qci,
          DRB_ul_SpecificParameters->priority,
          *(DRB_config->logicalChannelIdentity),
          *DRB_ul_SpecificParameters->logicalChannelGroup
         );
    //e_rab_done++;
    ue_context_pP->ue_context.modify_e_rab[i].status = E_RAB_STATUS_DONE;
    ue_context_pP->ue_context.modify_e_rab[i].xid = xid;
    {
      if (ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.buffer != NULL) {
        dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
        memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
        OCTET_STRING_fromBuf(dedicatedInfoNas,
                             (char *)ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.buffer,
                             ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.length);
        ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
        LOG_I(RRC, "add NAS info with size %d (rab id %d)\n",ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.length, i);
      } else {
        LOG_W(RRC, "Not received activate dedicated EPS bearer context request\n");
      }
    }
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList != NULL) {
    if (dedicatedInfoNASList->list.count == 0) {
      free(dedicatedInfoNASList);
      dedicatedInfoNASList = NULL;
      LOG_W(RRC,"dedlicated NAS list is empty, free the list and reset the address\n");
    }
  } else {
    LOG_W(RRC,"dedlicated NAS list is empty\n");
  }

  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         xid,
                                         (LTE_SRB_ToAddModList_t *)NULL,
                                         (LTE_DRB_ToAddModList_t *)DRB_configList2,
                                         (LTE_DRB_ToReleaseList_t *)NULL, // DRB2_list,
                                         (struct LTE_SPS_Config *)NULL,   // *sps_Config,
                                         NULL, NULL, NULL, NULL,NULL,
                                         NULL, NULL,  NULL, NULL, NULL, NULL, NULL,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)dedicatedInfoNASList,
                                         (LTE_SL_CommConfig_r12_t *)NULL,
                                         (LTE_SL_DiscConfig_r12_t *)NULL,
                                         (LTE_SCellToAddMod_r10_t *)NULL
                                        );
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_modify_e_rabs; i++) {
    if (ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.modify_e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE RNTI %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" dedicated LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
  return 0;
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_dedicatedRRCConnectionReconfiguration_release(  const protocol_ctxt_t   *const ctxt_pP,
    rrc_eNB_ue_context_t    *const ue_context_pP,
    uint8_t                  xid,
    uint32_t                 nas_length,
    uint8_t                 *nas_buffer)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  int                                 i;
  uint16_t                            size  = 0;
  LTE_DRB_ToReleaseList_t                **DRB_Release_configList2=NULL;
  LTE_DRB_Identity_t *DRB_release;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  DRB_Release_configList2=&ue_context_pP->ue_context.DRB_Release_configList2[xid];

  if (*DRB_Release_configList2) {
    free(*DRB_Release_configList2);
  }

  *DRB_Release_configList2 = CALLOC(1, sizeof(**DRB_Release_configList2));

  for(i = 0; i < NB_RB_MAX; i++) {
    if((ue_context_pP->ue_context.e_rab[i].status == E_RAB_STATUS_TORELEASE) && ue_context_pP->ue_context.e_rab[i].xid == xid) {
      DRB_release = CALLOC(1, sizeof(LTE_DRB_Identity_t));
      *DRB_release = i+1;
      ASN_SEQUENCE_ADD(&(*DRB_Release_configList2)->list, DRB_release);
      //free(DRB_release);
    }
  }

  /* If list is empty free the list and reset the address */
  if (nas_length > 0) {
    LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
    dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));
    dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
    memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
    OCTET_STRING_fromBuf(dedicatedInfoNas,
                         (char *)nas_buffer,
                         nas_length);
    ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
    LOG_I(RRC,"add NAS info with size %d\n",nas_length);
  } else {
    LOG_W(RRC,"dedlicated NAS list is empty\n");
  }

  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         xid,
                                         NULL,
                                         NULL,
                                         (LTE_DRB_ToReleaseList_t *)*DRB_Release_configList2,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         NULL,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)dedicatedInfoNASList,
                                         (LTE_SL_CommConfig_r12_t *)NULL,
                                         (LTE_SL_DiscConfig_r12_t *)NULL,
                                         (LTE_SCellToAddMod_r10_t *)NULL
                                        );
  ue_context_pP->ue_context.e_rab_release_command_flag = 1;
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  if (nas_length > 0) {
    /* Free the NAS PDU buffer and invalidate it */
    free(nas_buffer);
  }

  LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE RNTI %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" dedicated LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
}

//-----------------------------------------------------------------------------
/*
 * Generate the basic (first) RRC Connection Reconfiguration (non BR UE)
 */
void rrc_eNB_generate_defaultRRCConnectionReconfiguration(const protocol_ctxt_t *const ctxt_pP,
    rrc_eNB_ue_context_t  *const ue_context_pP,
    const uint8_t                ho_state)
//-----------------------------------------------------------------------------
{
  uint8_t   buffer[RRC_BUF_SIZE];
  uint16_t  size;
  int       i;
  MessageDef *message_p = NULL;
  /* Configure SRB1/SRB2, PhysicalConfigDedicated, LTE_MAC_MainConfig for UE */
  eNB_RRC_INST                           *rrc_inst = RC.rrc[ctxt_pP->module_id];
  struct LTE_PhysicalConfigDedicated    **physicalConfigDedicated = &ue_context_pP->ue_context.physicalConfigDedicated;
  struct LTE_SRB_ToAddMod                *SRB2_config                      = NULL;
  struct LTE_SRB_ToAddMod__rlc_Config    *SRB2_rlc_config                  = NULL;
  struct LTE_SRB_ToAddMod__logicalChannelConfig *SRB2_lchan_config         = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *SRB2_ul_SpecificParameters       = NULL;
  LTE_SRB_ToAddModList_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  LTE_SRB_ToAddModList_t                 **SRB_configList2                 = NULL;
  struct LTE_DRB_ToAddMod                *DRB_config                       = NULL;
  struct LTE_RLC_Config                  *DRB_rlc_config                   = NULL;
  struct LTE_PDCP_Config                 *DRB_pdcp_config                  = NULL;
  struct LTE_PDCP_Config__rlc_AM         *PDCP_rlc_AM                      = NULL;
  struct LTE_PDCP_Config__rlc_UM         *PDCP_rlc_UM                      = NULL;
  struct LTE_LogicalChannelConfig        *DRB_lchan_config                 = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *DRB_ul_SpecificParameters        = NULL;
  LTE_DRB_ToAddModList_t                **DRB_configList = &ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t                **DRB_configList2                  = NULL;
  LTE_MAC_MainConfig_t                   *mac_MainConfig                   = NULL;
  LTE_MeasObjectToAddModList_t           *MeasObj_list                     = NULL;
  LTE_MeasObjectToAddMod_t               *MeasObj                          = NULL;
  LTE_MeasObjectToAddMod_t               *MeasObj2                         = NULL;
  LTE_ReportConfigToAddModList_t         *ReportConfig_list                = NULL;
  LTE_ReportConfigToAddMod_t             *ReportConfig_per, *ReportConfig_A1,
                                         *ReportConfig_A2, *ReportConfig_A3, *ReportConfig_A4, *ReportConfig_A5;
  LTE_ReportConfigToAddMod_t             *ReportConfig_NR                  = NULL;
  LTE_MeasIdToAddModList_t               *MeasId_list                      = NULL;
  LTE_MeasIdToAddMod_t                   *MeasId0, *MeasId1, *MeasId2, *MeasId3, *MeasId4, *MeasId5;
  LTE_MeasIdToAddMod_t                   *MeasId6;
  long                                   *sr_ProhibitTimer_r9              = NULL;
  long                                   *logicalchannelgroup              = NULL;
  long                                   *logicalchannelgroup_drb          = NULL;
  long                                   *maxHARQ_Tx                       = NULL;
  long                                   *periodicBSR_Timer                = NULL;
  LTE_RSRP_Range_t                       *rsrp                             = NULL;
  struct LTE_MeasConfig__speedStatePars  *Sparams                          = NULL;
  LTE_QuantityConfig_t                   *quantityConfig                   = NULL;
  LTE_CellsToAddMod_t                    *CellToAdd                        = NULL;
  LTE_CellsToAddModList_t                *CellsToAddModList                = NULL;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* For no gcc warnings */
  (void) dedicatedInfoNas;
  LTE_C_RNTI_t                           *cba_RNTI                         = NULL;
  int                                    measurements_enabled;
  uint8_t xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);   //Transaction_id,
  uint8_t cc_id = ue_context_pP->ue_context.primaryCC_id;
  LTE_UE_EUTRA_Capability_t *UEcap = ue_context_pP->ue_context.UE_Capability;
  T(T_ENB_RRC_CONNECTION_RECONFIGURATION,
    T_INT(ctxt_pP->module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(ctxt_pP->rnti));
  /* Configure SRB2 */
  SRB_configList2 = &(ue_context_pP->ue_context.SRB_configList2[xid]);

  if (*SRB_configList2) {
    free(*SRB_configList2);
  }

  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));
  memset(*SRB_configList2, 0, sizeof(**SRB_configList2));
  SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  SRB2_config->srb_Identity = 2;
  SRB2_rlc_config = CALLOC(1, sizeof(*SRB2_rlc_config));
  SRB2_config->rlc_Config = SRB2_rlc_config;
  SRB2_rlc_config->present = LTE_SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB2_rlc_config->choice.explicitValue.present = LTE_RLC_Config_PR_am;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms15;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p8;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kB1000;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t32;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms10;
  SRB2_lchan_config = CALLOC(1, sizeof(*SRB2_lchan_config));
  SRB2_config->logicalChannelConfig = SRB2_lchan_config;
  SRB2_lchan_config->present = LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue;
  SRB2_ul_SpecificParameters = CALLOC(1, sizeof(*SRB2_ul_SpecificParameters));
  SRB2_ul_SpecificParameters->priority = 3; // let some priority for SRB1 and dedicated DRBs
  SRB2_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  SRB2_ul_SpecificParameters->bucketSizeDuration = LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  /* LCG for CCCH and DCCH is 0 as defined in 36331 */
  logicalchannelgroup = CALLOC(1, sizeof(long));
  *logicalchannelgroup = 0;
  SRB2_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup;
  SRB2_lchan_config->choice.explicitValue.ul_SpecificParameters = SRB2_ul_SpecificParameters;
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config); // this list has the configuration for SRB1 and SRB2
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config); // this list has only the configuration for SRB2

  /* Configure DRB */
  // list for all the configured DRB
  if (*DRB_configList) {
    free(*DRB_configList);
  }

  *DRB_configList = CALLOC(1, sizeof(**DRB_configList));
  memset(*DRB_configList, 0, sizeof(**DRB_configList));
  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid]; // list for the configured DRB for a this xid

  if (*DRB_configList2) {
    free(*DRB_configList2);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  memset(*DRB_configList2, 0, sizeof(**DRB_configList2));
  DRB_config = CALLOC(1, sizeof(*DRB_config));
  DRB_config->eps_BearerIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->eps_BearerIdentity) = 5L; // LW set to first value, allowed value 5..15, value : x+4
  // NN: this is the 1st DRB for this ue, so set it to 1
  DRB_config->drb_Identity = (LTE_DRB_Identity_t) 1;  // (ue_mod_idP+1); //allowed values 1..32, value: x
  DRB_config->logicalChannelIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->logicalChannelIdentity) = (long)3; // value : x+2
  DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
  DRB_config->rlc_Config = DRB_rlc_config;
#ifdef RRC_DEFAULT_RAB_IS_AM
  DRB_rlc_config->present = LTE_RLC_Config_PR_am;
  DRB_rlc_config->choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
  DRB_rlc_config->choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;
#else
  DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
  DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
#endif
  DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
  DRB_config->pdcp_Config = DRB_pdcp_config;
  DRB_pdcp_config->discardTimer = CALLOC(1, sizeof(long));
  *DRB_pdcp_config->discardTimer = LTE_PDCP_Config__discardTimer_infinity;
  DRB_pdcp_config->rlc_AM = NULL;
  DRB_pdcp_config->rlc_UM = NULL;
  /* Avoid gcc warnings */
  (void)PDCP_rlc_AM;
  (void)PDCP_rlc_UM;
#ifdef RRC_DEFAULT_RAB_IS_AM // EXMIMO_IOT
  PDCP_rlc_AM = CALLOC(1, sizeof(*PDCP_rlc_AM));
  DRB_pdcp_config->rlc_AM = PDCP_rlc_AM;
  PDCP_rlc_AM->statusReportRequired = FALSE;
#else
  PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
  DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
  PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
#endif
  DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
  DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
  DRB_config->logicalChannelConfig = DRB_lchan_config;
  DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
  DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;
  DRB_ul_SpecificParameters->priority = 12; // lower priority than srb1, srb2 and other dedicated bearer
  DRB_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8; // LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  DRB_ul_SpecificParameters->bucketSizeDuration = LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for DTCH can take the value from 1 to 3 as defined in 36331: normally controlled by upper layers (like RRM)
  logicalchannelgroup_drb = CALLOC(1, sizeof(long));
  *logicalchannelgroup_drb = 1;
  DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;
  ASN_SEQUENCE_ADD(&(*DRB_configList)->list, DRB_config);
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
  /* MAC Main Config */
  // The different parts of MAC main config are set below
  mac_MainConfig = CALLOC(1, sizeof(*mac_MainConfig));
  ue_context_pP->ue_context.mac_MainConfig = mac_MainConfig;
  mac_MainConfig->ul_SCH_Config = CALLOC(1, sizeof(*mac_MainConfig->ul_SCH_Config));
  maxHARQ_Tx = CALLOC(1, sizeof(long));
  *maxHARQ_Tx = LTE_MAC_MainConfig__ul_SCH_Config__maxHARQ_Tx_n5;
  mac_MainConfig->ul_SCH_Config->maxHARQ_Tx = maxHARQ_Tx;
  /* BSR reconfiguration */
  periodicBSR_Timer = CALLOC(1, sizeof(long));
  *periodicBSR_Timer = LTE_PeriodicBSR_Timer_r12_sf64; //LTE_PeriodicBSR_Timer_r12_infinity; // LTE_PeriodicBSR_Timer_r12_sf64; // LTE_PeriodicBSR_Timer_r12_sf20
  mac_MainConfig->ul_SCH_Config->periodicBSR_Timer = periodicBSR_Timer;
  mac_MainConfig->ul_SCH_Config->retxBSR_Timer = LTE_RetxBSR_Timer_r12_sf320; // LTE_RetxBSR_Timer_r12_sf320; // LTE_RetxBSR_Timer_r12_sf5120
  mac_MainConfig->ul_SCH_Config->ttiBundling = 0; // FALSE
  mac_MainConfig->timeAlignmentTimerDedicated = LTE_TimeAlignmentTimer_infinity;
  /* PHR reconfiguration */
  mac_MainConfig->phr_Config = CALLOC(1, sizeof(*mac_MainConfig->phr_Config));
  mac_MainConfig->phr_Config->present = LTE_MAC_MainConfig__phr_Config_PR_setup;
  mac_MainConfig->phr_Config->choice.setup.periodicPHR_Timer =
    LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_sf500; // sf20 = 20 subframes // LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_infinity
  mac_MainConfig->phr_Config->choice.setup.prohibitPHR_Timer =
    LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf200; // sf20 = 20 subframes // LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf1000
  mac_MainConfig->phr_Config->choice.setup.dl_PathlossChange = LTE_MAC_MainConfig__phr_Config__setup__dl_PathlossChange_dB3;  // Value dB1 =1 dB, dB3 = 3 dB
  /* CDRX Configuration */
  mac_MainConfig->drx_Config = NULL;
  rnti_t rnti = ue_context_pP->ue_id_rnti;
  module_id_t module_id = ctxt_pP->module_id;
  LOG_D(RRC, "Processing the DRX configuration in RRC Connection Reconfiguration\n");

  /* Process the IE drx_Config */
  if (NODE_IS_MONOLITHIC(rrc_inst->node_type)) {
    mac_MainConfig->drx_Config = do_DrxConfig(cc_id, &rrc_inst->configuration, UEcap); // drx_Config IE

    if (mac_MainConfig->drx_Config == NULL) {
      LOG_W(RRC, "drx_Configuration parameter is NULL, cannot configure local UE parameters or CDRX is deactivated\n");
    } else {
      /* Send DRX configuration to MAC task to configure timers of local UE context */
      message_p = itti_alloc_new_message(TASK_RRC_ENB, RRC_MAC_DRX_CONFIG_REQ);
      RRC_MAC_DRX_CONFIG_REQ(message_p).rnti = rnti;
      RRC_MAC_DRX_CONFIG_REQ(message_p).drx_Configuration = mac_MainConfig->drx_Config;
      itti_send_msg_to_task(TASK_MAC_ENB, module_id, message_p);
      LOG_D(RRC, "DRX configured in MAC Main Configuration for RRC Connection Reconfiguration\n");
    }
  }

  /* End of CDRX configuration */
  sr_ProhibitTimer_r9 = CALLOC(1, sizeof(long));
  *sr_ProhibitTimer_r9 = 0;   // SR tx on PUCCH, Value in number of SR period(s). Value 0 = no timer for SR, Value 2 = 2*SR
  mac_MainConfig->ext1 = CALLOC(1, sizeof(struct LTE_MAC_MainConfig__ext1));
  mac_MainConfig->ext1->sr_ProhibitTimer_r9 = sr_ProhibitTimer_r9;

  // change the transmission mode for the primary component carrier
  // TODO: add codebook subset restriction here
  // TODO: change TM for secondary CC in SCelltoaddmodlist
  if (*physicalConfigDedicated) {
    if ((*physicalConfigDedicated)->antennaInfo) {
      (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.transmissionMode = rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode;
      LOG_D(RRC,"Setting transmission mode to %ld+1\n",rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode);

      if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm3) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm3;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf[0] = 0xc0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.bits_unused=6;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm4) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm4;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf[0] = 0xfc;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.bits_unused=2;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm5) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm5;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf[0] = 0xf0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.bits_unused=4;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm6) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm6;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf[0] = 0xf0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.bits_unused=4;
      }
    } else {
      LOG_E(RRC,"antenna_info not present in physical_config_dedicated. Not reconfiguring!\n");
    }

    /* CSI RRC Reconfiguration */
    if ((*physicalConfigDedicated)->cqi_ReportConfig != NULL) {
      if ((rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm4) ||
          (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm5) ||
          (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm6)) {
        // feedback mode needs to be set as well
        // TODO: I think this is taken into account in the PHY automatically based on the transmission mode variable
        LOG_I(RRC, "Setting cqi reporting mode to rm31 (hardcoded)\n");
        *((*physicalConfigDedicated)->cqi_ReportConfig->cqi_ReportModeAperiodic) = LTE_CQI_ReportModeAperiodic_rm31; // HLC CQI, single PMI
      }
    } else {
      LOG_E(RRC,"cqi_ReportConfig not present in physical_config_dedicated. Not reconfiguring!\n");
    }
  } else {
    LOG_E(RRC,"physical_config_dedicated not present in LTE_RRCConnectionReconfiguration. Not reconfiguring!\n");
  }

  // Measurement ID list
  MeasId_list = CALLOC(1, sizeof(*MeasId_list));
  memset((void *)MeasId_list, 0, sizeof(*MeasId_list));
  MeasId0 = CALLOC(1, sizeof(*MeasId0));
  MeasId0->measId = 1;
  MeasId0->measObjectId = 1;
  MeasId0->reportConfigId = 1;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId0);
  MeasId1 = CALLOC(1, sizeof(*MeasId1));
  MeasId1->measId = 2;
  MeasId1->measObjectId = 1;
  MeasId1->reportConfigId = 2;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId1);
  MeasId2 = CALLOC(1, sizeof(*MeasId2));
  MeasId2->measId = 3;
  MeasId2->measObjectId = 1;
  MeasId2->reportConfigId = 3;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId2);
  MeasId3 = CALLOC(1, sizeof(*MeasId3));
  MeasId3->measId = 4;
  MeasId3->measObjectId = 1;
  MeasId3->reportConfigId = 4;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId3);
  MeasId4 = CALLOC(1, sizeof(*MeasId4));
  MeasId4->measId = 5;
  MeasId4->measObjectId = 1;
  MeasId4->reportConfigId = 5;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId4);
  MeasId5 = CALLOC(1, sizeof(*MeasId5));
  MeasId5->measId = 6;
  MeasId5->measObjectId = 1;
  MeasId5->reportConfigId = 6;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId5);

  if (ue_context_pP->ue_context.does_nr) {
    MeasId6 = calloc(1, sizeof(LTE_MeasIdToAddMod_t));

    if (MeasId6 == NULL) exit(1);

    MeasId6->measId = 7;
    MeasId6->measObjectId = 2;
    MeasId6->reportConfigId = 7;
    ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId6);
  }

  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measIdToAddModList = MeasId_list;
  // Add one EUTRA Measurement Object
  MeasObj_list = CALLOC(1, sizeof(*MeasObj_list));
  memset((void *)MeasObj_list, 0, sizeof(*MeasObj_list));
  // Configure MeasObject
  MeasObj = CALLOC(1, sizeof(*MeasObj));
  memset((void *)MeasObj, 0, sizeof(*MeasObj));
  MeasObj->measObjectId = 1;
  MeasObj->measObject.present = LTE_MeasObjectToAddMod__measObject_PR_measObjectEUTRA;
  MeasObj->measObject.choice.measObjectEUTRA.carrierFreq =
    to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->configuration.eutra_band[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.downlink_frequency[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.N_RB_DL[0]);
  MeasObj->measObject.choice.measObjectEUTRA.allowedMeasBandwidth = LTE_AllowedMeasBandwidth_mbw25;
  MeasObj->measObject.choice.measObjectEUTRA.presenceAntennaPort1 = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf = CALLOC(1, sizeof(uint8_t));
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf[0] = 0;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.size = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.bits_unused = 6;
  MeasObj->measObject.choice.measObjectEUTRA.offsetFreq = NULL;   // Default is 15 or 0dB

  if (RC.rrc[ctxt_pP->module_id]->num_neigh_cells > 0) {
    MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList =
      (LTE_CellsToAddModList_t *) CALLOC(1, sizeof(*CellsToAddModList));
    CellsToAddModList = MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList;
  }

  if (!ue_context_pP->ue_context.measurement_info) {
    ue_context_pP->ue_context.measurement_info = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info)));
  }

  //TODO: Assign proper values
  ue_context_pP->ue_context.measurement_info->offsetFreq = 0;
  ue_context_pP->ue_context.measurement_info->cellIndividualOffset[0] = LTE_Q_OffsetRange_dB0;

  /* TODO: Extend to multiple carriers */
  // Add adjacent cell lists (max 6 per eNB)
  for (i = 0; i < RC.rrc[ctxt_pP->module_id]->num_neigh_cells; i++) {
    CellToAdd = (LTE_CellsToAddMod_t *) CALLOC(1, sizeof(*CellToAdd));
    CellToAdd->cellIndex = i + 1;
    CellToAdd->physCellId = RC.rrc[ctxt_pP->module_id]->neigh_cells_id[i][0];//get_adjacent_cell_id(ctxt_pP->module_id, i);
    CellToAdd->cellIndividualOffset = LTE_Q_OffsetRange_dB0;
    ue_context_pP->ue_context.measurement_info->cellIndividualOffset[i+1] = CellToAdd->cellIndividualOffset;
    ASN_SEQUENCE_ADD(&CellsToAddModList->list, CellToAdd);
  }

  ASN_SEQUENCE_ADD(&MeasObj_list->list, MeasObj);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measObjectToAddModList = MeasObj_list;

  if (ue_context_pP->ue_context.does_nr) {
    MeasObj2 = calloc(1, sizeof(LTE_MeasObjectToAddMod_t));

    if (MeasObj2 == NULL) exit(1);

    MeasObj2->measObjectId = 2;
    MeasObj2->measObject.present = LTE_MeasObjectToAddMod__measObject_PR_measObjectNR_r15;
    MeasObj2->measObject.choice.measObjectNR_r15.carrierFreq_r15 =641272; //634000; //(634000 = 3.51GHz) (640000 = 3.6GHz) (641272 = 3619.08MHz = 3600 + 30/1000*106*12/2) (642256 is for 3.6GHz and absoluteFrequencySSB = 642016)
    MeasObj2->measObject.choice.measObjectNR_r15.rs_ConfigSSB_r15.measTimingConfig_r15.periodicityAndOffset_r15.present = LTE_MTC_SSB_NR_r15__periodicityAndOffset_r15_PR_sf20_r15;
    MeasObj2->measObject.choice.measObjectNR_r15.rs_ConfigSSB_r15.measTimingConfig_r15.periodicityAndOffset_r15.choice.sf20_r15 = 0;
    MeasObj2->measObject.choice.measObjectNR_r15.rs_ConfigSSB_r15.measTimingConfig_r15.ssb_Duration_r15 = LTE_MTC_SSB_NR_r15__ssb_Duration_r15_sf4;
    MeasObj2->measObject.choice.measObjectNR_r15.rs_ConfigSSB_r15.subcarrierSpacingSSB_r15 = LTE_RS_ConfigSSB_NR_r15__subcarrierSpacingSSB_r15_kHz30;
    MeasObj2->measObject.choice.measObjectNR_r15.quantityConfigSet_r15 = 1;
    MeasObj2->measObject.choice.measObjectNR_r15.ext1 = calloc(1, sizeof(struct LTE_MeasObjectNR_r15__ext1));

    if (MeasObj2->measObject.choice.measObjectNR_r15.ext1 == NULL) exit(1);

    MeasObj2->measObject.choice.measObjectNR_r15.ext1->bandNR_r15 = calloc(1, sizeof(struct LTE_MeasObjectNR_r15__ext1__bandNR_r15));

    if (MeasObj2->measObject.choice.measObjectNR_r15.ext1->bandNR_r15 == NULL) exit(1);

    MeasObj2->measObject.choice.measObjectNR_r15.ext1->bandNR_r15->present = LTE_MeasObjectNR_r15__ext1__bandNR_r15_PR_setup;
    MeasObj2->measObject.choice.measObjectNR_r15.ext1->bandNR_r15->choice.setup = 78;
    ASN_SEQUENCE_ADD(&MeasObj_list->list, MeasObj2);
  }

  if (!ue_context_pP->ue_context.measurement_info->events) {
    ue_context_pP->ue_context.measurement_info->events = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info->events)));
  }

  // Report Configurations for periodical, A1-A5 events
  ReportConfig_list = CALLOC(1, sizeof(*ReportConfig_list));
  ReportConfig_per = CALLOC(1, sizeof(*ReportConfig_per));
  ReportConfig_A1 = CALLOC(1, sizeof(*ReportConfig_A1));
  ReportConfig_A2 = CALLOC(1, sizeof(*ReportConfig_A2));
  ReportConfig_A3 = CALLOC(1, sizeof(*ReportConfig_A3));
  ReportConfig_A4 = CALLOC(1, sizeof(*ReportConfig_A4));
  ReportConfig_A5 = CALLOC(1, sizeof(*ReportConfig_A5));

  if (ue_context_pP->ue_context.does_nr) {
    ReportConfig_NR = CALLOC(1, sizeof(*ReportConfig_NR));
  }

  ReportConfig_per->reportConfigId = 1;
  ReportConfig_per->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_periodical;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.choice.periodical.purpose =
    LTE_ReportConfigEUTRA__triggerType__periodical__purpose_reportStrongestCells;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_per);
  ReportConfig_A1->reportConfigId = 2;
  ReportConfig_A1->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA1;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A1);
  //if (ho_state == 1 /*HO_MEASURMENT */ ) {
  LOG_I(RRC, "[eNB %d] frame %d: requesting A2, A3, A4, and A5 event reporting\n",
        ctxt_pP->module_id, ctxt_pP->frame);
  ReportConfig_A2->reportConfigId = 3;
  ReportConfig_A2->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A2);
  ReportConfig_A3->reportConfigId = 4;
  ReportConfig_A3->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA3;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset = 0;   //10;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA3.reportOnLeave = 1;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis = 0; // FIXME ...hysteresis is of type long!
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger =
    LTE_TimeToTrigger_ms40;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A3);
  ReportConfig_A4->reportConfigId = 5;
  ReportConfig_A4->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA4;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A4);
  ReportConfig_A5->reportConfigId = 6;
  ReportConfig_A5->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA5;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A5);

  if (ue_context_pP->ue_context.does_nr) {
    ReportConfig_NR->reportConfigId = 7;
    ReportConfig_NR->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigInterRAT;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.present = LTE_ReportConfigInterRAT__triggerType_PR_event;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.eventId.present = LTE_ReportConfigInterRAT__triggerType__event__eventId_PR_eventB1_NR_r15;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.eventId.choice.eventB1_NR_r15.b1_ThresholdNR_r15.present = LTE_ThresholdNR_r15_PR_nr_RSRP_r15;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.eventId.choice.eventB1_NR_r15.b1_ThresholdNR_r15.choice.nr_RSRP_r15 = 0;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.eventId.choice.eventB1_NR_r15.reportOnLeave_r15 = FALSE;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.hysteresis = 2;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.triggerType.choice.event.timeToTrigger = LTE_TimeToTrigger_ms80;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.maxReportCells = 4;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.reportInterval = LTE_ReportInterval_ms120;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.reportAmount = LTE_ReportConfigInterRAT__reportAmount_infinity;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7 = calloc(1, sizeof(struct LTE_ReportConfigInterRAT__ext7));

    if (ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7 == NULL) exit(1);

    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7->reportQuantityCellNR_r15 = calloc(1, sizeof(struct LTE_ReportQuantityNR_r15));

    if (ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7->reportQuantityCellNR_r15 == NULL) exit(1);

    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7->reportQuantityCellNR_r15->ss_rsrp = TRUE;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7->reportQuantityCellNR_r15->ss_rsrq = TRUE;
    ReportConfig_NR->reportConfig.choice.reportConfigInterRAT.ext7->reportQuantityCellNR_r15->ss_sinr = TRUE;
    ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_NR);
  }

  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->reportConfigToAddModList = ReportConfig_list;

  /* A3 event update */
  if (!ue_context_pP->ue_context.measurement_info->events->a3_event) {
    ue_context_pP->ue_context.measurement_info->events->a3_event = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info->events->a3_event)));
  }

  ue_context_pP->ue_context.measurement_info->events->a3_event->a3_offset = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset;
  ue_context_pP->ue_context.measurement_info->events->a3_event->reportOnLeave = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.reportOnLeave;
  ue_context_pP->ue_context.measurement_info->events->a3_event->hysteresis = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis;
  ue_context_pP->ue_context.measurement_info->events->a3_event->timeToTrigger = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger;
  ue_context_pP->ue_context.measurement_info->events->a3_event->maxReportCells = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells;
#if 0
  /* TODO: set a proper value.
   * 20 means: UE does not report if RSRP of serving cell is higher
   * than -120 dB (see 36.331 5.5.3.1).
   * This is too low for the X2 handover experiment.
   */
  rsrp = CALLOC(1, sizeof(LTE_RSRP_Range_t));
  *rsrp = 20;
#endif
  Sparams = CALLOC(1, sizeof(*Sparams));
  Sparams->present = LTE_MeasConfig__speedStatePars_PR_setup;
  Sparams->choice.setup.timeToTrigger_SF.sf_High = LTE_SpeedStateScaleFactors__sf_Medium_oDot75;
  Sparams->choice.setup.timeToTrigger_SF.sf_Medium = LTE_SpeedStateScaleFactors__sf_High_oDot5;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeHigh = 10;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeMedium = 5;
  Sparams->choice.setup.mobilityStateParameters.t_Evaluation = LTE_MobilityStateParameters__t_Evaluation_s60;
  Sparams->choice.setup.mobilityStateParameters.t_HystNormal = LTE_MobilityStateParameters__t_HystNormal_s120;
  quantityConfig = CALLOC(1, sizeof(*quantityConfig));
  memset((void *)quantityConfig, 0, sizeof(*quantityConfig));
  quantityConfig->quantityConfigEUTRA = CALLOC(1, sizeof(struct LTE_QuantityConfigEUTRA));
  memset((void *)quantityConfig->quantityConfigEUTRA, 0, sizeof(*quantityConfig->quantityConfigEUTRA));
  quantityConfig->quantityConfigCDMA2000 = NULL;
  quantityConfig->quantityConfigGERAN = NULL;
  quantityConfig->quantityConfigUTRA = NULL;
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP)));
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ)));
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = LTE_FilterCoefficient_fc4;
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = LTE_FilterCoefficient_fc4;
  ue_context_pP->ue_context.measurement_info->filterCoefficientRSRP = *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP;
  ue_context_pP->ue_context.measurement_info->filterCoefficientRSRQ = *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ;
  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
      memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
      OCTET_STRING_fromBuf(dedicatedInfoNas,
                           (char *)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                           ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
      ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
    }

    /* TODO parameters yet to process ... */
    {
      //      ue_context_pP->ue_context.e_rab[i].param.qos;
      //      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      //      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
    }
    /* TODO should test if e RAB are OK before! */
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_DONE;
    LOG_D(RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
          i, ue_context_pP->ue_context.e_rab[i].status, "E_RAB_STATUS_DONE");
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList->list.count == 0) {
    free(dedicatedInfoNASList);
    dedicatedInfoNASList = NULL;
  }

  measurements_enabled = RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2 ||
                         RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_measurement_reports;
  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         xid, // Transaction_id,
                                         (LTE_SRB_ToAddModList_t *) *SRB_configList2, // SRB_configList
                                         (LTE_DRB_ToAddModList_t *) *DRB_configList,
                                         (LTE_DRB_ToReleaseList_t *) NULL, // DRB2_list,
                                         (struct LTE_SPS_Config *) NULL,   // *sps_Config,
                                         (struct LTE_PhysicalConfigDedicated *) *physicalConfigDedicated,
                                         measurements_enabled ? (LTE_MeasObjectToAddModList_t *) MeasObj_list : NULL,
                                         measurements_enabled ? (LTE_ReportConfigToAddModList_t *) ReportConfig_list : NULL,
                                         measurements_enabled ? (LTE_QuantityConfig_t *) quantityConfig : NULL,
                                         measurements_enabled ? (LTE_MeasIdToAddModList_t *) MeasId_list : NULL,
                                         (LTE_MAC_MainConfig_t *) mac_MainConfig,
                                         (LTE_MeasGapConfig_t *) NULL,
                                         (LTE_MobilityControlInfo_t *) NULL,
                                         (LTE_SecurityConfigHO_t *) NULL,
                                         (struct LTE_MeasConfig__speedStatePars *) Sparams,
                                         (LTE_RSRP_Range_t *) rsrp,
                                         (LTE_C_RNTI_t *) cba_RNTI,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *) dedicatedInfoNASList,
                                         (LTE_SL_CommConfig_r12_t *) NULL,
                                         (LTE_SL_DiscConfig_r12_t *) NULL,
                                         (LTE_SCellToAddMod_r10_t *) NULL
                                        );
  LOG_DUMPMSG(RRC, DEBUG_RRC,(char *)buffer, size, "[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  LOG_I(RRC, "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE id %x)\n",
        ctxt_pP->module_id,
        ctxt_pP->frame,
        size,
        ue_context_pP->ue_context.rnti);
  LOG_D(RRC, "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame,
        ctxt_pP->module_id,
        size,
        ue_context_pP->ue_context.rnti,
        rrc_eNB_mui,
        ctxt_pP->module_id,
        DCCH);
  MSC_LOG_TX_MESSAGE(MSC_RRC_ENB,
                     MSC_RRC_UE,
                     buffer,
                     size,
                     MSC_AS_TIME_FMT" LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
                     MSC_AS_TIME_ARGS(ctxt_pP),
                     ue_context_pP->ue_context.rnti,
                     rrc_eNB_mui,
                     size);
  rrc_data_req(ctxt_pP,
               DCCH,
               rrc_eNB_mui++,
               SDU_CONFIRM_NO,
               size,
               buffer,
               PDCP_TRANSMISSION_MODE_CONTROL);

  /* Refresh SRBs/DRBs */
  rrc_pdcp_config_asn1_req(ctxt_pP,
                          *SRB_configList2, // NULL,
                          *DRB_configList,
                          NULL,
                          0xff, // already configured during the securitymodecommand
                          NULL,
                          NULL,
                          NULL
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
                          , (LTE_PMCH_InfoList_r9_t *) NULL
#endif
                          , NULL);

  /* Refresh SRBs/DRBs */
  if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    rrc_rlc_config_asn1_req(ctxt_pP,
                            *SRB_configList2, // NULL,
                            *DRB_configList,
                            NULL
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
                            , (LTE_PMCH_InfoList_r9_t *) NULL,
                            0,
                            0
#endif
                            );
  }

  free(Sparams);
  Sparams = NULL;
  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = NULL;
  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = NULL;
  free(quantityConfig->quantityConfigEUTRA);
  quantityConfig->quantityConfigEUTRA = NULL;
  free(quantityConfig);
  quantityConfig = NULL;
}

//-----------------------------------------------------------------------------
void
flexran_rrc_eNB_generate_defaultRRCConnectionReconfiguration(const protocol_ctxt_t *const ctxt_pP,
    rrc_eNB_ue_context_t          *const ue_context_pP,
    const uint8_t                ho_state)
//-----------------------------------------------------------------------------
{
  uint8_t   buffer[RRC_BUF_SIZE];
  uint16_t  size;
  int       i;
  MessageDef *message_p = NULL;
  /* Configure SRB1/SRB2, PhysicalConfigDedicated, LTE_MAC_MainConfig for UE */
  eNB_RRC_INST                           *rrc_inst = RC.rrc[ctxt_pP->module_id];
  struct LTE_PhysicalConfigDedicated    **physicalConfigDedicated = &ue_context_pP->ue_context.physicalConfigDedicated;
  struct LTE_SRB_ToAddMod                *SRB2_config                      = NULL;
  struct LTE_SRB_ToAddMod__rlc_Config    *SRB2_rlc_config                  = NULL;
  struct LTE_SRB_ToAddMod__logicalChannelConfig *SRB2_lchan_config         = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *SRB2_ul_SpecificParameters       = NULL;
  LTE_SRB_ToAddModList_t                 *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  LTE_SRB_ToAddModList_t                 **SRB_configList2                 = NULL;
  struct LTE_DRB_ToAddMod                *DRB_config                       = NULL;
  struct LTE_RLC_Config                  *DRB_rlc_config                   = NULL;
  struct LTE_PDCP_Config                 *DRB_pdcp_config                  = NULL;
  struct LTE_PDCP_Config__rlc_AM         *PDCP_rlc_AM                      = NULL;
  struct LTE_PDCP_Config__rlc_UM         *PDCP_rlc_UM                      = NULL;
  struct LTE_LogicalChannelConfig        *DRB_lchan_config                 = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *DRB_ul_SpecificParameters        = NULL;
  LTE_DRB_ToAddModList_t                **DRB_configList = &ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t                **DRB_configList2                  = NULL;
  LTE_MAC_MainConfig_t                   *mac_MainConfig                   = NULL;
  LTE_MeasObjectToAddModList_t           *MeasObj_list                     = NULL;
  LTE_MeasObjectToAddMod_t               *MeasObj                          = NULL;
  LTE_ReportConfigToAddModList_t         *ReportConfig_list                = NULL;
  LTE_ReportConfigToAddMod_t             *ReportConfig_per, *ReportConfig_A1,
                                         *ReportConfig_A2, *ReportConfig_A3, *ReportConfig_A4, *ReportConfig_A5;
  LTE_MeasIdToAddModList_t               *MeasId_list                      = NULL;
  LTE_MeasIdToAddMod_t                   *MeasId0, *MeasId1, *MeasId2, *MeasId3, *MeasId4, *MeasId5;
  long                                   *sr_ProhibitTimer_r9              = NULL;
  long                                   *logicalchannelgroup              = NULL;
  long                                   *logicalchannelgroup_drb          = NULL;
  long                                   *maxHARQ_Tx                       = NULL;
  long                                   *periodicBSR_Timer                = NULL;
  LTE_RSRP_Range_t                       *rsrp                             = NULL;
  struct LTE_MeasConfig__speedStatePars  *Sparams                          = NULL;
  LTE_QuantityConfig_t                   *quantityConfig                   = NULL;
  LTE_CellsToAddMod_t                    *CellToAdd                        = NULL;
  LTE_CellsToAddModList_t                *CellsToAddModList                = NULL;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t                 *dedicatedInfoNas                 = NULL;
  /* For no gcc warnings */
  (void) dedicatedInfoNas;
  LTE_C_RNTI_t                           *cba_RNTI                         = NULL;
  int                                    measurements_enabled;
  uint8_t xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);   //Transaction_id,
#ifdef CBA // Contention Based Access
  uint8_t                            *cba_RNTI_buf;
  cba_RNTI = CALLOC(1, sizeof(LTE_C_RNTI_t));
  cba_RNTI_buf = CALLOC(1, 2 * sizeof(uint8_t));
  cba_RNTI->buf = cba_RNTI_buf;
  cba_RNTI->size = 2;
  cba_RNTI->bits_unused = 0;

  /* Associate UEs to the CBA groups as a function of their UE id */
  if (rrc_inst->num_active_cba_groups) {
    cba_RNTI->buf[0] = rrc_inst->cba_rnti[ue_mod_idP % rrc_inst->num_active_cba_groups] & 0xff;
    cba_RNTI->buf[1] = 0xff;
    LOG_D(RRC, "[eNB %d] Frame %d: cba_RNTI = %x in group %d is attribued to UE %d\n",
          enb_mod_idP,
          frameP,
          rrc_inst->cba_rnti[ue_mod_idP % rrc_inst->num_active_cba_groups],
          ue_mod_idP % rrc_inst->num_active_cba_groups, ue_mod_idP);
  } else {
    cba_RNTI->buf[0] = 0x0;
    cba_RNTI->buf[1] = 0x0;
    LOG_D(RRC, "[eNB %d] Frame %d: no cba_RNTI is configured for UE %d\n",
          enb_mod_idP,
          frameP,
          ue_mod_idP);
  }

#endif
  T(T_ENB_RRC_CONNECTION_RECONFIGURATION,
    T_INT(ctxt_pP->module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(ctxt_pP->rnti));
  /* Configure SRB2 */
  SRB_configList2 = &(ue_context_pP->ue_context.SRB_configList2[xid]);

  if (*SRB_configList2) {
    free(*SRB_configList2);
  }

  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));
  memset(*SRB_configList2, 0, sizeof(**SRB_configList2));
  SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  SRB2_config->srb_Identity = 2;
  SRB2_rlc_config = CALLOC(1, sizeof(*SRB2_rlc_config));
  SRB2_config->rlc_Config = SRB2_rlc_config;
  SRB2_rlc_config->present = LTE_SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB2_rlc_config->choice.explicitValue.present = LTE_RLC_Config_PR_am;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms15;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p8;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kB1000;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t32;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms10;
  SRB2_lchan_config = CALLOC(1, sizeof(*SRB2_lchan_config));
  SRB2_config->logicalChannelConfig = SRB2_lchan_config;
  SRB2_lchan_config->present = LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue;
  SRB2_ul_SpecificParameters = CALLOC(1, sizeof(*SRB2_ul_SpecificParameters));
  SRB2_ul_SpecificParameters->priority = 3; // let some priority for SRB1 and dedicated DRBs
  SRB2_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  SRB2_ul_SpecificParameters->bucketSizeDuration = LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  /* LCG for CCCH and DCCH is 0 as defined in 36331 */
  logicalchannelgroup = CALLOC(1, sizeof(long));
  *logicalchannelgroup = 0;
  SRB2_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup;
  SRB2_lchan_config->choice.explicitValue.ul_SpecificParameters = SRB2_ul_SpecificParameters;
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config); // this list has the configuration for SRB1 and SRB2
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config); // this list has only the configuration for SRB2

  /* Configure DRB */
  // list for all the configured DRB
  if (*DRB_configList) {
    free(*DRB_configList);
  }

  *DRB_configList = CALLOC(1, sizeof(**DRB_configList));
  memset(*DRB_configList, 0, sizeof(**DRB_configList));
  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid]; // list for the configured DRB for a this xid

  if (*DRB_configList2) {
    free(*DRB_configList2);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  memset(*DRB_configList2, 0, sizeof(**DRB_configList2));
  DRB_config = CALLOC(1, sizeof(*DRB_config));
  DRB_config->eps_BearerIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->eps_BearerIdentity) = 5L; // LW set to first value, allowed value 5..15, value : x+4
  // NN: this is the 1st DRB for this ue, so set it to 1
  DRB_config->drb_Identity = (LTE_DRB_Identity_t) 1;  // (ue_mod_idP+1); //allowed values 1..32, value: x
  DRB_config->logicalChannelIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->logicalChannelIdentity) = (long)3; // value : x+2
  DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
  DRB_config->rlc_Config = DRB_rlc_config;
#ifdef RRC_DEFAULT_RAB_IS_AM
  DRB_rlc_config->present = LTE_RLC_Config_PR_am;
  DRB_rlc_config->choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
  DRB_rlc_config->choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;
#else
  DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
  DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
#endif
  DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
  DRB_config->pdcp_Config = DRB_pdcp_config;
  DRB_pdcp_config->discardTimer = CALLOC(1, sizeof(long));
  *DRB_pdcp_config->discardTimer = LTE_PDCP_Config__discardTimer_infinity;
  DRB_pdcp_config->rlc_AM = NULL;
  DRB_pdcp_config->rlc_UM = NULL;
  /* Avoid gcc warnings */
  (void)PDCP_rlc_AM;
  (void)PDCP_rlc_UM;
#ifdef RRC_DEFAULT_RAB_IS_AM // EXMIMO_IOT
  PDCP_rlc_AM = CALLOC(1, sizeof(*PDCP_rlc_AM));
  DRB_pdcp_config->rlc_AM = PDCP_rlc_AM;
  PDCP_rlc_AM->statusReportRequired = FALSE;
#else
  PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
  DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
  PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
#endif
  DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
  DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
  DRB_config->logicalChannelConfig = DRB_lchan_config;
  DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
  DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;
  DRB_ul_SpecificParameters->priority = 12; // lower priority than srb1, srb2 and other dedicated bearer
  DRB_ul_SpecificParameters->prioritisedBitRate = LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8; // LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  DRB_ul_SpecificParameters->bucketSizeDuration = LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for DTCH can take the value from 1 to 3 as defined in 36331: normally controlled by upper layers (like RRM)
  logicalchannelgroup_drb = CALLOC(1, sizeof(long));
  *logicalchannelgroup_drb = 1;
  DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;
  ASN_SEQUENCE_ADD(&(*DRB_configList)->list, DRB_config);
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
  /* MAC Main Config */
  // The different parts of MAC main config are set below
  mac_MainConfig = CALLOC(1, sizeof(*mac_MainConfig));
  mac_MainConfig->ul_SCH_Config = CALLOC(1, sizeof(*mac_MainConfig->ul_SCH_Config));
  maxHARQ_Tx = CALLOC(1, sizeof(long));
  *maxHARQ_Tx = LTE_MAC_MainConfig__ul_SCH_Config__maxHARQ_Tx_n5;
  mac_MainConfig->ul_SCH_Config->maxHARQ_Tx = maxHARQ_Tx;
  /* BSR reconfiguration */
  periodicBSR_Timer = CALLOC(1, sizeof(long));
  *periodicBSR_Timer = LTE_PeriodicBSR_Timer_r12_sf64; //LTE_PeriodicBSR_Timer_r12_infinity; // LTE_PeriodicBSR_Timer_r12_sf64; // LTE_PeriodicBSR_Timer_r12_sf20
  mac_MainConfig->ul_SCH_Config->periodicBSR_Timer = periodicBSR_Timer;
  mac_MainConfig->ul_SCH_Config->retxBSR_Timer = LTE_RetxBSR_Timer_r12_sf320; // LTE_RetxBSR_Timer_r12_sf320; // LTE_RetxBSR_Timer_r12_sf5120
  mac_MainConfig->ul_SCH_Config->ttiBundling = 0; // FALSE
  mac_MainConfig->timeAlignmentTimerDedicated = LTE_TimeAlignmentTimer_infinity;
  /* PHR reconfiguration */
  mac_MainConfig->phr_Config = CALLOC(1, sizeof(*mac_MainConfig->phr_Config));
  mac_MainConfig->phr_Config->present = LTE_MAC_MainConfig__phr_Config_PR_setup;
  mac_MainConfig->phr_Config->choice.setup.periodicPHR_Timer =
    LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_sf500; // sf20 = 20 subframes // LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_infinity
  mac_MainConfig->phr_Config->choice.setup.prohibitPHR_Timer =
    LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf200; // sf20 = 20 subframes // LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf1000
  mac_MainConfig->phr_Config->choice.setup.dl_PathlossChange = LTE_MAC_MainConfig__phr_Config__setup__dl_PathlossChange_dB3;  // Value dB1 =1 dB, dB3 = 3 dB
  /* CDRX Configuration */
  mac_MainConfig->drx_Config = NULL;
  rnti_t rnti = ue_context_pP->ue_id_rnti;
  module_id_t module_id = ctxt_pP->module_id;
  uint8_t cc_id = ue_context_pP->ue_context.primaryCC_id;
  LTE_UE_EUTRA_Capability_t *UEcap = ue_context_pP->ue_context.UE_Capability;
  LOG_D(RRC, "Processing the DRX configuration in RRC Connection Reconfiguration\n");

  /* Process the IE drx_Config */
  if (NODE_IS_MONOLITHIC(rrc_inst->node_type)) {
    mac_MainConfig->drx_Config = do_DrxConfig(cc_id, &rrc_inst->configuration, UEcap); // drx_Config IE

    if (mac_MainConfig->drx_Config == NULL) {
      LOG_W(RRC, "drx_Configuration parameter is NULL, cannot configure local UE parameters or CDRX is deactivated\n");
    } else {
      /* Send DRX configuration to MAC task to configure timers of local UE context */
      message_p = itti_alloc_new_message(TASK_RRC_ENB, RRC_MAC_DRX_CONFIG_REQ);
      RRC_MAC_DRX_CONFIG_REQ(message_p).rnti = rnti;
      RRC_MAC_DRX_CONFIG_REQ(message_p).drx_Configuration = mac_MainConfig->drx_Config;
      itti_send_msg_to_task(TASK_MAC_ENB, module_id, message_p);
      LOG_D(RRC, "DRX configured in MAC Main Configuration for RRC Connection Reconfiguration\n");
    }
  }

  /* End of CDRX configuration */
  sr_ProhibitTimer_r9 = CALLOC(1, sizeof(long));
  *sr_ProhibitTimer_r9 = 0;   // SR tx on PUCCH, Value in number of SR period(s). Value 0 = no timer for SR, Value 2 = 2*SR
  mac_MainConfig->ext1 = CALLOC(1, sizeof(struct LTE_MAC_MainConfig__ext1));
  mac_MainConfig->ext1->sr_ProhibitTimer_r9 = sr_ProhibitTimer_r9;
  // free the old LTE_MAC_MainConfig_t: get a pointer to "old" memory, assign
  // the new values in the ue_context, then free it
  // Note: can not completely avoid race condition with FlexRAN
  LTE_MAC_MainConfig_t *old_mac_MainConfig = ue_context_pP->ue_context.mac_MainConfig;
  ue_context_pP->ue_context.mac_MainConfig = mac_MainConfig;
  free(old_mac_MainConfig->ul_SCH_Config->periodicBSR_Timer);
  free(old_mac_MainConfig->ul_SCH_Config->maxHARQ_Tx);
  free(old_mac_MainConfig->ul_SCH_Config);
  free(old_mac_MainConfig->phr_Config);
  free(old_mac_MainConfig->ext1->sr_ProhibitTimer_r9);
  free(old_mac_MainConfig->ext1);
  free(old_mac_MainConfig);

  // change the transmission mode for the primary component carrier
  // TODO: add codebook subset restriction here
  // TODO: change TM for secondary CC in SCelltoaddmodlist
  if (*physicalConfigDedicated) {
    if ((*physicalConfigDedicated)->antennaInfo) {
      (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.transmissionMode = rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode;
      LOG_D(RRC,"Setting transmission mode to %ld+1\n",rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode);

      if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm3) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm3;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf[0] = 0xc0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.bits_unused=6;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm4) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm4;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf[0] = 0xfc;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.bits_unused=2;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm5) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm5;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf[0] = 0xf0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.bits_unused=4;
      } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm6) {
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
          CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
          LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm6;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf= MALLOC(1);
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf[0] = 0xf0;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.size=1;
        (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.bits_unused=4;
      }
    } else {
      LOG_E(RRC,"antenna_info not present in physical_config_dedicated. Not reconfiguring!\n");
    }

    /* CSI RRC Reconfiguration */
    if ((*physicalConfigDedicated)->cqi_ReportConfig != NULL) {
      if ((rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm4) ||
          (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm5) ||
          (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode == LTE_AntennaInfoDedicated__transmissionMode_tm6)) {
        // feedback mode needs to be set as well
        // TODO: I think this is taken into account in the PHY automatically based on the transmission mode variable
        LOG_I(RRC, "Setting cqi reporting mode to rm31 (hardcoded)\n");
        *((*physicalConfigDedicated)->cqi_ReportConfig->cqi_ReportModeAperiodic) = LTE_CQI_ReportModeAperiodic_rm31; // HLC CQI, single PMI
      }
    } else {
      LOG_E(RRC,"cqi_ReportConfig not present in physical_config_dedicated. Not reconfiguring!\n");
    }
  } else {
    LOG_E(RRC,"physical_config_dedicated not present in LTE_RRCConnectionReconfiguration. Not reconfiguring!\n");
  }

  // Measurement ID list
  MeasId_list = CALLOC(1, sizeof(*MeasId_list));
  memset((void *)MeasId_list, 0, sizeof(*MeasId_list));
  MeasId0 = CALLOC(1, sizeof(*MeasId0));
  MeasId0->measId = 1;
  MeasId0->measObjectId = 1;
  MeasId0->reportConfigId = 1;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId0);
  MeasId1 = CALLOC(1, sizeof(*MeasId1));
  MeasId1->measId = 2;
  MeasId1->measObjectId = 1;
  MeasId1->reportConfigId = 2;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId1);
  MeasId2 = CALLOC(1, sizeof(*MeasId2));
  MeasId2->measId = 3;
  MeasId2->measObjectId = 1;
  MeasId2->reportConfigId = 3;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId2);
  MeasId3 = CALLOC(1, sizeof(*MeasId3));
  MeasId3->measId = 4;
  MeasId3->measObjectId = 1;
  MeasId3->reportConfigId = 4;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId3);
  MeasId4 = CALLOC(1, sizeof(*MeasId4));
  MeasId4->measId = 5;
  MeasId4->measObjectId = 1;
  MeasId4->reportConfigId = 5;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId4);
  MeasId5 = CALLOC(1, sizeof(*MeasId5));
  MeasId5->measId = 6;
  MeasId5->measObjectId = 1;
  MeasId5->reportConfigId = 6;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId5);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measIdToAddModList = MeasId_list;
  // Add one EUTRA Measurement Object
  MeasObj_list = CALLOC(1, sizeof(*MeasObj_list));
  memset((void *)MeasObj_list, 0, sizeof(*MeasObj_list));
  // Configure MeasObject
  MeasObj = CALLOC(1, sizeof(*MeasObj));
  memset((void *)MeasObj, 0, sizeof(*MeasObj));
  MeasObj->measObjectId = 1;
  MeasObj->measObject.present = LTE_MeasObjectToAddMod__measObject_PR_measObjectEUTRA;
  MeasObj->measObject.choice.measObjectEUTRA.carrierFreq =
    to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->configuration.eutra_band[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.downlink_frequency[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.N_RB_DL[0]);
  MeasObj->measObject.choice.measObjectEUTRA.allowedMeasBandwidth = LTE_AllowedMeasBandwidth_mbw25;
  MeasObj->measObject.choice.measObjectEUTRA.presenceAntennaPort1 = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf = CALLOC(1, sizeof(uint8_t));
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf[0] = 0;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.size = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.bits_unused = 6;
  //<<<<<<< HEAD
  MeasObj->measObject.choice.measObjectEUTRA.offsetFreq = NULL;   // Default is 15 or 0dB

  if (rrc_inst->carrier[0].sib1->tdd_Config!=NULL) {
    MeasObj->measObject.choice.measObjectEUTRA.ext1 = CALLOC(1, sizeof(struct LTE_MeasObjectEUTRA__ext1));
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measCycleSCell_r10 = NULL;
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10 = CALLOC(1, sizeof(struct LTE_MeasSubframePatternConfigNeigh_r10));
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->present=LTE_MeasSubframePatternConfigNeigh_r10_PR_setup;
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.present=LTE_MeasSubframePattern_r10_PR_subframePatternTDD_r10;
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.present=
      LTE_MeasSubframePattern_r10__subframePatternTDD_r10_PR_subframeConfig1_5_r10;
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf=CALLOC(3,
        sizeof(uint8_t));
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.size=3;
    MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.bits_unused=4;

    switch (rrc_inst->carrier[0].sib1->tdd_Config->subframeAssignment) {
      case 1: //subframe 0,4,5,9
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[0]=0x8C;
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[1]=0x63;
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[2]=0x10;
        break;

      default: //subframe 0 , 5
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[0]=0x84;
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[1]=0x21;
        MeasObj->measObject.choice.measObjectEUTRA.ext1->measSubframePatternConfigNeigh_r10->choice.setup.measSubframePatternNeigh_r10.choice.subframePatternTDD_r10.choice.subframeConfig1_5_r10.buf[2]=0x00;
        break;
    }
  }

  MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList =
    (LTE_CellsToAddModList_t *) CALLOC(1, sizeof(*CellsToAddModList));
  CellsToAddModList = MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList;
  //=======
  //  MeasObj->measObject.choice.measObjectEUTRA.offsetFreq = (LTE_Q_OffsetRange_t *) CALLOC(1,sizeof(LTE_Q_OffsetRange_t));
  //  *(MeasObj->measObject.choice.measObjectEUTRA.offsetFreq) = ue_context_pP->ue_context.measurement_info->offsetFreq;   // Default is 15 or 0dB
  //>>>>>>> origin/OAI_develop

  if (RC.rrc[ctxt_pP->module_id]->num_neigh_cells > 0) {
    MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList =
      (LTE_CellsToAddModList_t *) CALLOC(1, sizeof(*CellsToAddModList));
    CellsToAddModList = MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList;
  }

  /* TODO: Extend to multiple carriers */
  // Add adjacent cell lists (max 6 per eNB)
  for (i = 0; i < RC.rrc[ctxt_pP->module_id]->num_neigh_cells; i++) {
    CellToAdd = (LTE_CellsToAddMod_t *) CALLOC(1, sizeof(*CellToAdd));
    CellToAdd->cellIndex = i + 1;
    CellToAdd->physCellId = RC.rrc[ctxt_pP->module_id]->neigh_cells_id[i][0];//get_adjacent_cell_id(ctxt_pP->module_id, i);
    CellToAdd->cellIndividualOffset = ue_context_pP->ue_context.measurement_info->cellIndividualOffset[i+1];
    ASN_SEQUENCE_ADD(&CellsToAddModList->list, CellToAdd);
  }

  ASN_SEQUENCE_ADD(&MeasObj_list->list, MeasObj);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measObjectToAddModList = MeasObj_list;
  // Report Configurations for periodical, A1-A5 events
  ReportConfig_list = CALLOC(1, sizeof(*ReportConfig_list));
  ReportConfig_per = CALLOC(1, sizeof(*ReportConfig_per));
  ReportConfig_A1 = CALLOC(1, sizeof(*ReportConfig_A1));
  ReportConfig_A2 = CALLOC(1, sizeof(*ReportConfig_A2));
  ReportConfig_A3 = CALLOC(1, sizeof(*ReportConfig_A3));
  ReportConfig_A4 = CALLOC(1, sizeof(*ReportConfig_A4));
  ReportConfig_A5 = CALLOC(1, sizeof(*ReportConfig_A5));
  ReportConfig_per->reportConfigId = 1;
  ReportConfig_per->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_periodical;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.choice.periodical.purpose =
    LTE_ReportConfigEUTRA__triggerType__periodical__purpose_reportStrongestCells;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_per);
  ReportConfig_A1->reportConfigId = 2;
  ReportConfig_A1->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA1;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A1);
  //if (ho_state == 1 /*HO_MEASURMENT */ ) {
  LOG_I(RRC, "[eNB %d] frame %d: requesting A2, A3, A4, and A5 event reporting\n",
        ctxt_pP->module_id, ctxt_pP->frame);
  ReportConfig_A2->reportConfigId = 3;
  ReportConfig_A2->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A2);
  ReportConfig_A3->reportConfigId = 4;
  ReportConfig_A3->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA3;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset = ue_context_pP->ue_context.measurement_info->events->a3_event->a3_offset;//10;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA3.reportOnLeave = ue_context_pP->ue_context.measurement_info->events->a3_event->reportOnLeave;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells = ue_context_pP->ue_context.measurement_info->events->a3_event->maxReportCells;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis = ue_context_pP->ue_context.measurement_info->events->a3_event->hysteresis;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger =
    ue_context_pP->ue_context.measurement_info->events->a3_event->timeToTrigger;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A3);
  ReportConfig_A4->reportConfigId = 5;
  ReportConfig_A4->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA4;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A4);
  ReportConfig_A5->reportConfigId = 6;
  ReportConfig_A5->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA5;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A5);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->reportConfigToAddModList = ReportConfig_list;
#if 0
  /* TODO: set a proper value.
   * 20 means: UE does not report if RSRP of serving cell is higher
   * than -120 dB (see 36.331 5.5.3.1).
   * This is too low for the X2 handover experiment.
   */
  rsrp = CALLOC(1, sizeof(LTE_RSRP_Range_t));
  *rsrp = 20;
#endif
  Sparams = CALLOC(1, sizeof(*Sparams));
  Sparams->present = LTE_MeasConfig__speedStatePars_PR_setup;
  Sparams->choice.setup.timeToTrigger_SF.sf_High = LTE_SpeedStateScaleFactors__sf_Medium_oDot75;
  Sparams->choice.setup.timeToTrigger_SF.sf_Medium = LTE_SpeedStateScaleFactors__sf_High_oDot5;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeHigh = 10;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeMedium = 5;
  Sparams->choice.setup.mobilityStateParameters.t_Evaluation = LTE_MobilityStateParameters__t_Evaluation_s60;
  Sparams->choice.setup.mobilityStateParameters.t_HystNormal = LTE_MobilityStateParameters__t_HystNormal_s120;
  quantityConfig = CALLOC(1, sizeof(*quantityConfig));
  memset((void *)quantityConfig, 0, sizeof(*quantityConfig));
  quantityConfig->quantityConfigEUTRA = CALLOC(1, sizeof(struct LTE_QuantityConfigEUTRA));
  memset((void *)quantityConfig->quantityConfigEUTRA, 0, sizeof(*quantityConfig->quantityConfigEUTRA));
  quantityConfig->quantityConfigCDMA2000 = NULL;
  quantityConfig->quantityConfigGERAN = NULL;
  quantityConfig->quantityConfigUTRA = NULL;
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP)));
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ)));
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = ue_context_pP->ue_context.measurement_info->filterCoefficientRSRP;
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = ue_context_pP->ue_context.measurement_info->filterCoefficientRSRQ;
  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
      memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
      OCTET_STRING_fromBuf(dedicatedInfoNas,
                           (char *)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                           ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
      ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
    }

    /* TODO parameters yet to process ... */
    {
      //      ue_context_pP->ue_context.e_rab[i].param.qos;
      //      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      //      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
    }
    /* TODO should test if e RAB are OK before! */
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_DONE;
    LOG_D(RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
          i, ue_context_pP->ue_context.e_rab[i].status, "E_RAB_STATUS_DONE");
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList->list.count == 0) {
    free(dedicatedInfoNASList);
    dedicatedInfoNASList = NULL;
  }

  measurements_enabled = RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2 ||
                         RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_measurement_reports;
  memset(buffer, 0, RRC_BUF_SIZE);
  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         xid, // Transaction_id,
                                         (LTE_SRB_ToAddModList_t *) *SRB_configList2, // SRB_configList
                                         (LTE_DRB_ToAddModList_t *) *DRB_configList,
                                         (LTE_DRB_ToReleaseList_t *) NULL, // DRB2_list,
                                         (struct LTE_SPS_Config *) NULL,   // *sps_Config,
                                         (struct LTE_PhysicalConfigDedicated *) *physicalConfigDedicated,
                                         measurements_enabled ? (LTE_MeasObjectToAddModList_t *) MeasObj_list : NULL,
                                         measurements_enabled ? (LTE_ReportConfigToAddModList_t *) ReportConfig_list : NULL,
                                         measurements_enabled ? (LTE_QuantityConfig_t *) quantityConfig : NULL,
                                         measurements_enabled ? (LTE_MeasIdToAddModList_t *) MeasId_list : NULL,
                                         (LTE_MAC_MainConfig_t *) mac_MainConfig,
                                         (LTE_MeasGapConfig_t *) NULL,
                                         (LTE_MobilityControlInfo_t *) NULL,
                                         (LTE_SecurityConfigHO_t *) NULL,
                                         (struct LTE_MeasConfig__speedStatePars *) Sparams,
                                         (LTE_RSRP_Range_t *) rsrp,
                                         (LTE_C_RNTI_t *) cba_RNTI,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *) dedicatedInfoNASList,
                                         (LTE_SL_CommConfig_r12_t *) NULL,
                                         (LTE_SL_DiscConfig_r12_t *) NULL,
                                         (LTE_SCellToAddMod_r10_t *) NULL
                                        );
  LOG_DUMPMSG(RRC, DEBUG_RRC,(char *)buffer, size, "[MSG] RRC Connection Reconfiguration\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  LOG_I(RRC, "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE id %x)\n",
        ctxt_pP->module_id,
        ctxt_pP->frame,
        size,
        ue_context_pP->ue_context.rnti);
  LOG_D(RRC, "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame,
        ctxt_pP->module_id,
        size,
        ue_context_pP->ue_context.rnti,
        rrc_eNB_mui,
        ctxt_pP->module_id,
        DCCH);
  MSC_LOG_TX_MESSAGE(MSC_RRC_ENB,
                     MSC_RRC_UE,
                     buffer,
                     size,
                     MSC_AS_TIME_FMT" LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
                     MSC_AS_TIME_ARGS(ctxt_pP),
                     ue_context_pP->ue_context.rnti,
                     rrc_eNB_mui,
                     size);
  rrc_data_req(ctxt_pP,
               DCCH,
               rrc_eNB_mui++,
               SDU_CONFIRM_NO,
               size,
               buffer,
               PDCP_TRANSMISSION_MODE_CONTROL);
  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = NULL;
  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = NULL;
  free(quantityConfig->quantityConfigEUTRA);
  quantityConfig->quantityConfigEUTRA = NULL;
  free(quantityConfig);
  quantityConfig = NULL;
}

//-----------------------------------------------------------------------------
int
rrc_eNB_generate_RRCConnectionReconfiguration_SCell(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t *const ue_context_pP,
  uint32_t dl_CarrierFreq_r10
)
//-----------------------------------------------------------------------------
{
  uint8_t size;
  uint8_t buffer[100];
  uint8_t sCellIndexToAdd = 0; //one SCell so far

  //   uint8_t sCellIndexToAdd;
  //   sCellIndexToAdd = rrc_find_free_SCell_index(enb_mod_idP, ue_mod_idP, 1);
  //  if (RC.rrc[enb_mod_idP]->sCell_config[ue_mod_idP][sCellIndexToAdd] ) {
  if (ue_context_pP->ue_context.sCell_config != NULL) {
    ue_context_pP->ue_context.sCell_config[sCellIndexToAdd].cellIdentification_r10->dl_CarrierFreq_r10 = dl_CarrierFreq_r10;
  } else {
    LOG_E(RRC,"Scell not configured!\n");
    return(-1);
  }

  size = do_RRCConnectionReconfiguration(ctxt_pP,
                                         buffer,
                                         rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),//Transaction_id,
                                         (LTE_SRB_ToAddModList_t *)NULL,
                                         (LTE_DRB_ToAddModList_t *)NULL,
                                         (LTE_DRB_ToReleaseList_t *)NULL,
                                         (struct LTE_SPS_Config *)NULL,
                                         (struct LTE_PhysicalConfigDedicated *)NULL,
                                         (LTE_MeasObjectToAddModList_t *)NULL,
                                         (LTE_ReportConfigToAddModList_t *)NULL,
                                         (LTE_QuantityConfig_t *)NULL,
                                         (LTE_MeasIdToAddModList_t *)NULL,
                                         (LTE_MAC_MainConfig_t *)NULL,
                                         (LTE_MeasGapConfig_t *)NULL,
                                         (LTE_MobilityControlInfo_t *)NULL,
                                         (LTE_SecurityConfigHO_t *)NULL,
                                         (struct LTE_MeasConfig__speedStatePars *)NULL,
                                         (LTE_RSRP_Range_t *)NULL,
                                         (LTE_C_RNTI_t *)NULL,
                                         (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)NULL,
                                         (LTE_SL_CommConfig_r12_t *)NULL,
                                         (LTE_SL_DiscConfig_r12_t *)NULL,
                                         ue_context_pP->ue_context.sCell_config
                                        );
  LOG_I(RRC,"[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration (bytes %d, UE id %x)\n",
        ctxt_pP->module_id,ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" LTE_RRCConnectionReconfiguration UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
  return(0);
}



//-----------------------------------------------------------------------------
/**
 * @fn    :encode_CG_ConfigInfo
 * @param   :enc_buf to store the encoded bits
 * @param   :ue_context_pP ue context used to fill CG-ConfigInfo
 * @param   :enc_size to store thre size of encoded size
 *      this api is to fill and encode CG-ConfigInfo
 */
static int encode_CG_ConfigInfo(
  char *buffer,
  int buffer_size,
  rrc_eNB_ue_context_t *const ue_context_pP,
  int *enc_size
) {
  struct NR_CG_ConfigInfo *cg_configinfo = NULL;
  struct NR_RadioBearerConfig *rb_config = NULL;
  asn_enc_rval_t enc_rval;
  int RRC_OK = 1;
  int index = 0;
  char temp_buff[ASN_MAX_ENCODE_SIZE];
  LTE_UE_CapabilityRAT_ContainerList_t *ue_cap_rat_container_list = NULL;
  LTE_UE_CapabilityRAT_Container_t ue_cap_rat_container_MRDC;
  LTE_UE_CapabilityRAT_Container_t ue_cap_rat_container_nr;
  int RAT_Container_count = 0;
  rb_config = calloc(1,sizeof(struct NR_RadioBearerConfig));
  AssertFatal(rb_config != NULL,"failed to allocate memory for rb_config");

  if(ue_context_pP->ue_context.DRB_configList->list.count != 0) {
    rb_config->drb_ToAddModList = calloc(1,sizeof(struct NR_DRB_ToAddModList ));
    AssertFatal(rb_config->drb_ToAddModList != NULL,"failed to allocated memory for drbtoaddmodlist");
    rb_config->drb_ToAddModList->list.count = NUMBEROF_DRBS_TOBE_ADDED;
    rb_config->drb_ToAddModList->list.array
      = calloc(NUMBEROF_DRBS_TOBE_ADDED, sizeof(struct NR_DRB_ToAddMod *));
    AssertFatal( rb_config->drb_ToAddModList->list.array != NULL,
                 "falied to allocate memory for list.array");

    for(index = 0; index < NUMBEROF_DRBS_TOBE_ADDED; index++) {
      rb_config->drb_ToAddModList->list.array[index]
        = calloc(1,sizeof(struct NR_DRB_ToAddMod));
      AssertFatal(rb_config->drb_ToAddModList->list.array[index] != NULL,
                  "failed to allocate memory for drb_toaddmod");
      rb_config->drb_ToAddModList->list.array[index]->drb_Identity
        = ue_context_pP->ue_context.DRB_configList->list.array[index]->drb_Identity;

      if(ue_context_pP->ue_context.DRB_configList->list.array[index]->eps_BearerIdentity) {
        rb_config->drb_ToAddModList->list.array[index]->cnAssociation
          = calloc(1,sizeof(struct NR_DRB_ToAddMod__cnAssociation));
        AssertFatal(rb_config->drb_ToAddModList->list.array[index]->cnAssociation != NULL,
                    "failed to allocate memory cnAssociation");
        rb_config->drb_ToAddModList->list.array[index]->cnAssociation->present
          = NR_DRB_ToAddMod__cnAssociation_PR_eps_BearerIdentity;
        rb_config->drb_ToAddModList->list.array[index]->cnAssociation->choice.eps_BearerIdentity
          = *(ue_context_pP->ue_context.DRB_configList->list.array[index]->eps_BearerIdentity);
      }

      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config = calloc(1,sizeof(struct NR_PDCP_Config));
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb = calloc(1,sizeof(struct NR_PDCP_Config__drb));
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->discardTimer = calloc(1,sizeof(long));
      *rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->discardTimer
        = *(ue_context_pP->ue_context.DRB_configList->list.array[index]->pdcp_Config->discardTimer);
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->pdcp_SN_SizeUL = calloc(1,sizeof(long));
      *rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->pdcp_SN_SizeUL
        = NR_PDCP_Config__drb__pdcp_SN_SizeUL_len18bits;
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->pdcp_SN_SizeDL = calloc(1,sizeof(long));
      *rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->pdcp_SN_SizeDL
        = NR_PDCP_Config__drb__pdcp_SN_SizeDL_len18bits;
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->headerCompression.present
        = NR_PDCP_Config__drb__headerCompression_PR_notUsed;
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->drb->headerCompression.choice.notUsed = 0;
      rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->t_Reordering = calloc(1,sizeof(long));
      *rb_config->drb_ToAddModList->list.array[index]->pdcp_Config->t_Reordering
        = NR_PDCP_Config__t_Reordering_ms0;
    }

    rb_config->securityConfig = calloc(1,sizeof(struct NR_SecurityConfig ));
    rb_config->securityConfig->securityAlgorithmConfig = calloc(1,sizeof(struct NR_SecurityAlgorithmConfig));
    rb_config->securityConfig->securityAlgorithmConfig->cipheringAlgorithm = NR_CipheringAlgorithm_nea0;
    rb_config->securityConfig->securityAlgorithmConfig->integrityProtAlgorithm = NULL;
    rb_config->securityConfig->keyToUse = calloc(1,sizeof(long));
    *rb_config->securityConfig->keyToUse = NR_SecurityConfig__keyToUse_master;
  }

  cg_configinfo = calloc(1,sizeof(struct NR_CG_ConfigInfo));
  AssertFatal(cg_configinfo != NULL,"failed to allocate memory for cg_configinfo");
  cg_configinfo->criticalExtensions.present = NR_CG_ConfigInfo__criticalExtensions_PR_c1;
  cg_configinfo->criticalExtensions.choice.c1
    = calloc(1, sizeof(struct NR_CG_ConfigInfo__criticalExtensions__c1));
  AssertFatal(cg_configinfo->criticalExtensions.choice.c1 != NULL,
              "failed to allocate memory for cg_configinfo->criticalExtensions.choice.c1");
  cg_configinfo->criticalExtensions.choice.c1->present
    = NR_CG_ConfigInfo__criticalExtensions__c1_PR_cg_ConfigInfo;
  cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo
    = calloc(1,sizeof(struct NR_CG_ConfigInfo_IEs));
  AssertFatal(cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo != NULL,
              "failed to allocate memory for cg_configinfo_IEs");
  cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->ue_CapabilityInfo
    = calloc(1,sizeof( OCTET_STRING_t));
  AssertFatal(cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->
              ue_CapabilityInfo != NULL, "failed to allocate memory for ue_capabilityinfo");

  if(ue_context_pP->ue_context.UE_Capability_MRDC) {
    RAT_Container_count++;
    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UE_MRDC_Capability,NULL,
                                     (void *)ue_context_pP->ue_context.UE_Capability_MRDC,temp_buff,ASN_MAX_ENCODE_SIZE);
    AssertFatal(enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n",
                enc_rval.failed_type->name, enc_rval.encoded);
    ue_cap_rat_container_MRDC.rat_Type = LTE_RAT_Type_eutra_nr;
    OCTET_STRING_fromBuf(&ue_cap_rat_container_MRDC.ueCapabilityRAT_Container,
                         (const char *)temp_buff,(enc_rval.encoded+7)>>3);
    memset((void *)temp_buff,0,sizeof(temp_buff));
  }

  if(ue_context_pP->ue_context.UE_Capability_nr) {
    RAT_Container_count++;
    enc_rval = uper_encode_to_buffer(&asn_DEF_NR_UE_NR_Capability,NULL,
                                     (void *)ue_context_pP->ue_context.UE_Capability_nr,temp_buff,ASN_MAX_ENCODE_SIZE);
    AssertFatal(enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n",
                enc_rval.failed_type->name, enc_rval.encoded);
    ue_cap_rat_container_nr.rat_Type = LTE_RAT_Type_nr;
    OCTET_STRING_fromBuf(&ue_cap_rat_container_nr.ueCapabilityRAT_Container,
                         (const char *)temp_buff,(enc_rval.encoded+7)>>3);
    memset((void *)temp_buff,0,sizeof(temp_buff));
  }

  ue_cap_rat_container_list = calloc(1,sizeof(LTE_UE_CapabilityRAT_ContainerList_t));
  ue_cap_rat_container_list->list.count = RAT_Container_count;
  ue_cap_rat_container_list->list.size = RAT_Container_count * sizeof(LTE_UE_CapabilityRAT_Container_t);
  ue_cap_rat_container_list->list.array = calloc(RAT_Container_count,sizeof(LTE_UE_CapabilityRAT_Container_t *));

  if(ue_context_pP->ue_context.UE_Capability_MRDC) {
    ue_cap_rat_container_list->list.array[0] = calloc(1,sizeof(LTE_UE_CapabilityRAT_Container_t));
    memcpy(ue_cap_rat_container_list->list.array[0],&ue_cap_rat_container_MRDC,sizeof(LTE_UE_CapabilityRAT_Container_t));
  }

  if(ue_context_pP->ue_context.UE_Capability_nr) {
    ue_cap_rat_container_list->list.array[1] = calloc(1,sizeof(LTE_UE_CapabilityRAT_Container_t));
    memcpy(ue_cap_rat_container_list->list.array[1],&ue_cap_rat_container_nr,sizeof(LTE_UE_CapabilityRAT_Container_t));
  }

  //this xer_fprint logs can be enabled for additional debugging logs
  //xer_fprint(stdout,&asn_DEF_LTE_UE_CapabilityRAT_ContainerList,ue_cap_rat_container_list);
  enc_rval = uper_encode_to_buffer(&asn_DEF_LTE_UE_CapabilityRAT_ContainerList,NULL,
                                   (void *)ue_cap_rat_container_list,temp_buff,ASN_MAX_ENCODE_SIZE);
  AssertFatal(enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n",
              enc_rval.failed_type->name, enc_rval.encoded);
  OCTET_STRING_fromBuf(cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->ue_CapabilityInfo,
                       (const char *)temp_buff, (enc_rval.encoded+7)>>3);
  cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->mcg_RB_Config
    = calloc(1,sizeof(OCTET_STRING_t));
  AssertFatal(cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->
              mcg_RB_Config != NULL, "failed to allocate memory for mcg_rb_config");
  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_RadioBearerConfig,NULL,(void *)rb_config,temp_buff,ASN_MAX_ENCODE_SIZE);
  AssertFatal(enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %jd)!\n",
              enc_rval.failed_type->name, enc_rval.encoded);
  OCTET_STRING_fromBuf(cg_configinfo->criticalExtensions.choice.c1->choice.cg_ConfigInfo->mcg_RB_Config,
                       (const char *)temp_buff, (enc_rval.encoded+7)>>3);
  // this xer_fprint can be enabled for additional debugging messages
  // xer_fprint(stdout,&asn_DEF_NR_CG_ConfigInfo,(void *)cg_configinfo);
  enc_rval = uper_encode_to_buffer(&asn_DEF_NR_CG_ConfigInfo,NULL,(void *)cg_configinfo,
                                   buffer,buffer_size);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);
  *enc_size = (enc_rval.encoded+7)/8;
  ASN_STRUCT_FREE(asn_DEF_NR_RadioBearerConfig,rb_config);
  ASN_STRUCT_FREE(asn_DEF_NR_CG_ConfigInfo,cg_configinfo);
  ASN_STRUCT_FREE(asn_DEF_LTE_UE_CapabilityRAT_ContainerList,ue_cap_rat_container_list);
  return RRC_OK;
}
//-----------------------------------------------------------------------------






//-----------------------------------------------------------------------------
void
rrc_eNB_process_MeasurementReport(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t         *ue_context_pP,
  const LTE_MeasResults_t   *const measResults2
)
//-----------------------------------------------------------------------------
{
  int i=0;
  int neighboring_cells=-1;
  int ncell_index = 0;
  long ncell_max = -150;
  uint32_t earfcn_dl;
  uint8_t KeNB_star[32] = { 0 };
  char enc_buf[ASN_MAX_ENCODE_SIZE] = {0};
  int enc_size = 0;
  T(T_ENB_RRC_MEASUREMENT_REPORT, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  if (measResults2 == NULL )
    return;

  if (measResults2->measId > 0 ) {
    if (ue_context_pP->ue_context.measResults == NULL) {
      ue_context_pP->ue_context.measResults = CALLOC(1, sizeof(LTE_MeasResults_t));
    }

    ue_context_pP->ue_context.measResults->measId=measResults2->measId;

    switch (measResults2->measId) {
      case 1:
        LOG_D(RRC,"Periodic report at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      case 2:
        LOG_D(RRC,"A1 event report (Serving becomes better than absolute threshold) at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      case 3:
        LOG_D(RRC,"A2 event report (Serving becomes worse than absolute threshold) at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      case 4:
        LOG_D(RRC,"A3 event report (Neighbour becomes amount of offset better than PCell) at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      case 5:
        LOG_D(RRC,"A4 event report (Neighbour becomes better than absolute threshold) at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      case 6:
        LOG_D(RRC,"A5 event report (PCell becomes worse than absolute threshold1 AND Neighbour becomes better than another absolute threshold2) at frame %d and subframe %d \n", ctxt_pP->frame,
              ctxt_pP->subframe);
        break;

      case 7:
        LOG_D(RRC, "NR event frame %d subframe %d\n", ctxt_pP->frame, ctxt_pP->subframe);
        break;

      default:
        LOG_D(RRC,"Other event report frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
        break;
    }

    ue_context_pP->ue_context.measResults->measResultPCell.rsrpResult=measResults2->measResultPCell.rsrpResult;
    ue_context_pP->ue_context.measResults->measResultPCell.rsrqResult=measResults2->measResultPCell.rsrqResult;
    LOG_D(RRC, "[eNB %d]Frame %d: UE %x (Measurement Id %d): RSRP of Source %ld\n", ctxt_pP->module_id, ctxt_pP->frame, ctxt_pP->rnti, (int)measResults2->measId,
          ue_context_pP->ue_context.measResults->measResultPCell.rsrpResult-140);
    LOG_D(RRC, "[eNB %d]Frame %d: UE %x (Measurement Id %d): RSRQ of Source %ld\n", ctxt_pP->module_id, ctxt_pP->frame, ctxt_pP->rnti, (int)measResults2->measId,
          ue_context_pP->ue_context.measResults->measResultPCell.rsrqResult/2 - 20);
  }

  /* TODO: improve NR triggering */
  if (measResults2->measId == 7) {
    if ((ue_context_pP->ue_context.Status != RRC_NR_NSA) && (ue_context_pP->ue_context.Status != RRC_NR_NSA_RECONFIGURED)) {
      MessageDef      *msg;
      ue_context_pP->ue_context.Status = RRC_NR_NSA;

      if(is_en_dc_supported(ue_context_pP->ue_context.UE_Capability)) {
        /** to add gNB as Secondary node CG-ConfigInfo to be added as per 36.423 r15 **/
        if(encode_CG_ConfigInfo(enc_buf,sizeof(enc_buf),ue_context_pP,&enc_size) == RRC_OK)
          LOG_I(RRC,"CG-ConfigInfo encoded successfully\n");

        msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_ENDC_SGNB_ADDITION_REQ);
        memset(&(X2AP_ENDC_SGNB_ADDITION_REQ(msg)), 0, sizeof(x2ap_ENDC_sgnb_addition_req_t));
        X2AP_ENDC_SGNB_ADDITION_REQ(msg).rnti = ctxt_pP->rnti;
        memcpy(X2AP_ENDC_SGNB_ADDITION_REQ(msg).rrc_buffer,enc_buf,enc_size);
        X2AP_ENDC_SGNB_ADDITION_REQ(msg).rrc_buffer_size = enc_size;
        X2AP_ENDC_SGNB_ADDITION_REQ(msg).target_physCellId
          = measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[0]->physCellId;
        //For the moment we have a single E-RAB which will be the one to be added to the gNB
        //Not sure how to select bearers to be added if there are multiple.
        X2AP_ENDC_SGNB_ADDITION_REQ(msg).nb_e_rabs_tobeadded = 1;

        for (int e_rab=0; e_rab < X2AP_ENDC_SGNB_ADDITION_REQ(msg).nb_e_rabs_tobeadded; e_rab++) {
          X2AP_ENDC_SGNB_ADDITION_REQ(msg).e_rabs_tobeadded[e_rab].e_rab_id = ue_context_pP->ue_context.e_rab[e_rab].param.e_rab_id;
          X2AP_ENDC_SGNB_ADDITION_REQ(msg).e_rabs_tobeadded[e_rab].gtp_teid = ue_context_pP->ue_context.e_rab[e_rab].param.gtp_teid;
          X2AP_ENDC_SGNB_ADDITION_REQ(msg).e_rabs_tobeadded[e_rab].drb_ID = ue_context_pP->ue_context.DRB_configList->list.array[e_rab]->drb_Identity; 
          memcpy(&X2AP_ENDC_SGNB_ADDITION_REQ(msg).e_rabs_tobeadded[e_rab].sgw_addr,
                 &ue_context_pP->ue_context.e_rab[e_rab].param.sgw_addr,
                 sizeof(transport_layer_addr_t));
        }

        LOG_I(RRC,
              "[eNB %d] frame %d subframe %d: UE rnti %x switching to NSA mode\n",
              ctxt_pP->module_id, ctxt_pP->frame, ctxt_pP->subframe, ctxt_pP->rnti);
        itti_send_msg_to_task(TASK_X2AP, ENB_MODULE_ID_TO_INSTANCE(ctxt_pP->module_id), msg);
        return;
      }
    }
  }

  if (measResults2->measResultNeighCells == NULL)
    return;

  if (measResults2->measResultNeighCells->choice.measResultListEUTRA.list.count > 0) {
    neighboring_cells = measResults2->measResultNeighCells->choice.measResultListEUTRA.list.count;

    if (ue_context_pP->ue_context.measResults->measResultNeighCells == NULL) {
      ue_context_pP->ue_context.measResults->measResultNeighCells = CALLOC(1, sizeof(*ue_context_pP->ue_context.measResults->measResultNeighCells));
    }

    for (i=0; i < neighboring_cells; i++) {
      if (i>=ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.count) {
        //printf("NeighCells number: %d \n", ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.count);
        ASN_SEQUENCE_ADD(&ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list,measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]);
      }

      ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->physCellId =
        measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->physCellId;
      ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrpResult =
        measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrpResult;
      ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrqResult =
        measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrqResult;
      LOG_D(RRC, "Physical Cell Id %d\n",
            (int)ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->physCellId);
      LOG_D(RRC, "RSRP of Target %ld\n",
            (*ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrpResult)-140);
      LOG_D(RRC, "RSRQ of Target %ld\n",
            (*ue_context_pP->ue_context.measResults->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrqResult)/2 - 20);

      if ( *measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrpResult >= ncell_max ) {
        ncell_max = *measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->measResult.rsrpResult;
        ncell_index = i;
      }

      //LOG_D(RRC, "Physical Cell Id2 %d\n",
      //(int)measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->physCellId);
      //LOG_D(RRC, "RSRP of Target2 %ld\n",
      //(*(measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->
      //measResult.rsrpResult))-140);
      //LOG_D(RRC, "RSRQ of Target2 %ld\n",
      //(*(measResults2->measResultNeighCells->choice.measResultListEUTRA.list.array[i]->
      //measResult.rsrqResult))/2 - 20);
    }
  }

  /* Decide whether to trigger HO or not */
  if (!(measResults2->measId == 4))
    return;

  /* if X2AP is disabled, do nothing */
  if (!RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2)
    return;

  if (RC.rrc[ctxt_pP->module_id]->x2_ho_net_control)
    return;

  LOG_D(RRC, "A3 event is triggered...\n");

  /* if the UE is not in handover mode, start handover procedure */
  if (ue_context_pP->ue_context.Status != RRC_HO_EXECUTION) {
    MessageDef      *msg;
    LOG_I(RRC, "Send HO preparation message at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
    /* HO info struct may not be needed anymore */
    ue_context_pP->ue_context.handover_info = CALLOC(1, sizeof(*(ue_context_pP->ue_context.handover_info)));
    ue_context_pP->ue_context.Status = RRC_HO_EXECUTION;
    ue_context_pP->ue_context.handover_info->state = HO_REQUEST;
    /* HO Preparation message */
    msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_HANDOVER_REQ);
    rrc_eNB_generate_HandoverPreparationInformation(
      ue_context_pP,
      X2AP_HANDOVER_REQ(msg).rrc_buffer,
      &X2AP_HANDOVER_REQ(msg).rrc_buffer_size);
    X2AP_HANDOVER_REQ(msg).rnti = ctxt_pP->rnti;
    X2AP_HANDOVER_REQ(msg).target_physCellId = measResults2->measResultNeighCells->choice.
        measResultListEUTRA.list.array[ncell_index]->physCellId;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mcc = ue_context_pP->ue_context.ue_gummei.mcc;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mnc = ue_context_pP->ue_context.ue_gummei.mnc;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mnc_len = ue_context_pP->ue_context.ue_gummei.mnc_len;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mme_code = ue_context_pP->ue_context.ue_gummei.mme_code;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mme_group_id = ue_context_pP->ue_context.ue_gummei.mme_group_id;
    // Don't know how to get this ID?
    X2AP_HANDOVER_REQ(msg).mme_ue_s1ap_id = ue_context_pP->ue_context.mme_ue_s1ap_id;
    X2AP_HANDOVER_REQ(msg).security_capabilities = ue_context_pP->ue_context.security_capabilities;
    // compute keNB*
    earfcn_dl = (uint32_t)to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->carrier[0].eutra_band, RC.rrc[ctxt_pP->module_id]->carrier[0].dl_CarrierFreq,
                                       RC.rrc[ctxt_pP->module_id]->carrier[0].N_RB_DL);
    derive_keNB_star(ue_context_pP->ue_context.kenb, X2AP_HANDOVER_REQ(msg).target_physCellId, earfcn_dl, true, KeNB_star);
    memcpy(X2AP_HANDOVER_REQ(msg).kenb, KeNB_star, 32);
    X2AP_HANDOVER_REQ(msg).kenb_ncc = ue_context_pP->ue_context.kenb_ncc;
    //X2AP_HANDOVER_REQ(msg).ue_ambr=ue_context_pP->ue_context.ue_ambr;
    X2AP_HANDOVER_REQ(msg).nb_e_rabs_tobesetup = ue_context_pP->ue_context.setup_e_rabs;

    for (int i=0; i<ue_context_pP->ue_context.setup_e_rabs; i++) {
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].e_rab_id = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].eNB_addr = ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].gtp_teid = ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.qci = ue_context_pP->ue_context.e_rab[i].param.qos.qci;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.priority_level = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.priority_level;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_capability = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_capability;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_vulnerability = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_vulnerability;
    }

    /* TODO: don't do that, X2AP should find the target by itself */
    //X2AP_HANDOVER_REQ(msg).target_mod_id = 0;
    LOG_I(RRC,
          "[eNB %d] Frame %d: potential handover preparation: store the information in an intermediate structure in case of failure\n",
          ctxt_pP->module_id, ctxt_pP->frame);
    itti_send_msg_to_task(TASK_X2AP, ENB_MODULE_ID_TO_INSTANCE(ctxt_pP->module_id), msg);
  } else {
    LOG_D(RRC, "[eNB %d] Frame %d: Ignoring MeasReport from UE %x as Handover is in progress... \n", ctxt_pP->module_id, ctxt_pP->frame,
          ctxt_pP->rnti);
  }
}


//-----------------------------------------------------------------------------
void
rrc_eNB_generate_HandoverPreparationInformation(
  //const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_t *const ue_context_pP,
  uint8_t                     *buffer,
  int                          *_size
) {
  memset(buffer, 0, 8192);
  char *ho_buf = (char *) buffer;
  int ho_size;
  ho_size = do_HandoverPreparation(ho_buf, 8192, ue_context_pP->ue_context.UE_Capability, ue_context_pP->ue_context.UE_Capability_size);
  *_size = ho_size;
}

void rrc_eNB_process_x2_setup_request(int mod_id, x2ap_setup_req_t *m) {
  if (RC.rrc[mod_id]->num_neigh_cells > MAX_NUM_NEIGH_CELLs) {
    LOG_E(RRC, "Error: number of neighbouring cells is exceeded \n");
    return;
  }

  if (m->num_cc > MAX_NUM_CCs) {
    LOG_E(RRC, "Error: number of neighbouring cells carriers is exceeded \n");
    return;
  }

  RC.rrc[mod_id]->num_neigh_cells++;
  RC.rrc[mod_id]->num_neigh_cells_cc[RC.rrc[mod_id]->num_neigh_cells-1] = m->num_cc;

  for (int i=0; i<m->num_cc; i++) {
    RC.rrc[mod_id]->neigh_cells_id[RC.rrc[mod_id]->num_neigh_cells-1][i] = m->Nid_cell[i];
  }
}

void rrc_eNB_process_x2_setup_response(int mod_id, x2ap_setup_resp_t *m) {
  if (RC.rrc[mod_id]->num_neigh_cells > MAX_NUM_NEIGH_CELLs) {
    LOG_E(RRC, "Error: number of neighbouring cells is exceeded \n");
    return;
  }

  if (m->num_cc > MAX_NUM_CCs) {
    LOG_E(RRC, "Error: number of neighbouring cells carriers is exceeded \n");
    return;
  }

  RC.rrc[mod_id]->num_neigh_cells++;
  RC.rrc[mod_id]->num_neigh_cells_cc[RC.rrc[mod_id]->num_neigh_cells-1] = m->num_cc;

  for (int i=0; i<m->num_cc; i++) {
    RC.rrc[mod_id]->neigh_cells_id[RC.rrc[mod_id]->num_neigh_cells-1][i] = m->Nid_cell[i];
  }
}

void rrc_eNB_process_handoverPreparationInformation(int mod_id, x2ap_handover_req_t *m) {
  struct rrc_eNB_ue_context_s        *ue_context_target_p = NULL;
  /* TODO: get proper UE rnti */
  int rnti = taus() & 0xffff;
  int i;
  //global_rnti = rnti;
  LTE_HandoverPreparationInformation_t *ho = NULL;
  LTE_HandoverPreparationInformation_r8_IEs_t *ho_info;
  asn_dec_rval_t                      dec_rval;
  ue_context_target_p = rrc_eNB_get_ue_context(RC.rrc[mod_id], rnti);

  if (ue_context_target_p != NULL) {
    LOG_E(RRC, "\nError in obtaining free UE id in target eNB for handover \n");
    return;
  }

  ue_context_target_p = rrc_eNB_allocate_new_UE_context(RC.rrc[mod_id]);

  if (ue_context_target_p == NULL) {
    LOG_E(RRC, "Cannot create new UE context\n");
    return;
  }

  ue_context_target_p->ue_id_rnti = rnti;
  ue_context_target_p->ue_context.rnti = rnti;
  RB_INSERT(rrc_ue_tree_s, &RC.rrc[mod_id]->rrc_ue_head, ue_context_target_p);
  LOG_D(RRC, "eNB %d: Created new UE context uid %u\n", mod_id, ue_context_target_p->local_uid);
  ue_context_target_p->ue_context.handover_info = CALLOC(1, sizeof(*(ue_context_target_p->ue_context.handover_info)));
  //ue_context_target_p->ue_context.Status = RRC_HO_EXECUTION;
  //ue_context_target_p->ue_context.handover_info->state = HO_ACK;
  ue_context_target_p->ue_context.handover_info->x2_id = m->x2_id;
  ue_context_target_p->ue_context.handover_info->assoc_id = m->target_assoc_id;
  memset (ue_context_target_p->ue_context.nh, 0, 32);
  ue_context_target_p->ue_context.nh_ncc = -1;
  memcpy (ue_context_target_p->ue_context.kenb, m->kenb, 32);
  ue_context_target_p->ue_context.kenb_ncc = m->kenb_ncc;
  ue_context_target_p->ue_context.security_capabilities.encryption_algorithms = m->security_capabilities.encryption_algorithms;
  ue_context_target_p->ue_context.security_capabilities.integrity_algorithms = m->security_capabilities.integrity_algorithms;
  dec_rval = uper_decode(NULL,
                         &asn_DEF_LTE_HandoverPreparationInformation,
                         (void **)&ho,
                         m->rrc_buffer,
                         m->rrc_buffer_size, 0, 0);

  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_LTE_HandoverPreparationInformation, ho);
  }

  if (dec_rval.code != RC_OK ||
      ho->criticalExtensions.present != LTE_HandoverPreparationInformation__criticalExtensions_PR_c1 ||
      ho->criticalExtensions.choice.c1.present != LTE_HandoverPreparationInformation__criticalExtensions__c1_PR_handoverPreparationInformation_r8) {
    LOG_E(RRC, "could not decode Handover Preparation\n");
    abort();
  }

  ho_info = &ho->criticalExtensions.choice.c1.choice.handoverPreparationInformation_r8;

  if (ue_context_target_p->ue_context.UE_Capability) {
    LOG_I(RRC, "freeing old UE capabilities for UE %x\n", rnti);
    ASN_STRUCT_FREE(asn_DEF_LTE_UE_EUTRA_Capability,
                    ue_context_target_p->ue_context.UE_Capability);
    ue_context_target_p->ue_context.UE_Capability = 0;
  }

  dec_rval = uper_decode(NULL,
                         &asn_DEF_LTE_UE_EUTRA_Capability,
                         (void **)&ue_context_target_p->ue_context.UE_Capability,
                         ho_info->ue_RadioAccessCapabilityInfo.list.array[0]->ueCapabilityRAT_Container.buf,
                         ho_info->ue_RadioAccessCapabilityInfo.list.array[0]->ueCapabilityRAT_Container.size, 0, 0);
  ue_context_target_p->ue_context.UE_Capability_size = ho_info->ue_RadioAccessCapabilityInfo.list.array[0]->ueCapabilityRAT_Container.size;

  if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
    xer_fprint(stdout, &asn_DEF_LTE_UE_EUTRA_Capability, ue_context_target_p->ue_context.UE_Capability);
  }

  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    LOG_E(RRC, "Failed to decode UE capabilities (%zu bytes)\n", dec_rval.consumed);
    ASN_STRUCT_FREE(asn_DEF_LTE_UE_EUTRA_Capability,
                    ue_context_target_p->ue_context.UE_Capability);
    ue_context_target_p->ue_context.UE_Capability = 0;
  }

  ue_context_target_p->ue_context.nb_of_e_rabs = m->nb_e_rabs_tobesetup;
  ue_context_target_p->ue_context.setup_e_rabs = m->nb_e_rabs_tobesetup;
  ue_context_target_p->ue_context.mme_ue_s1ap_id = m->mme_ue_s1ap_id;
  ue_context_target_p->ue_context.ue_gummei.mcc = m->ue_gummei.mcc;
  ue_context_target_p->ue_context.ue_gummei.mnc = m->ue_gummei.mnc;
  ue_context_target_p->ue_context.ue_gummei.mnc_len = m->ue_gummei.mnc_len;
  ue_context_target_p->ue_context.ue_gummei.mme_code = m->ue_gummei.mme_code;
  ue_context_target_p->ue_context.ue_gummei.mme_group_id = m->ue_gummei.mme_group_id;
  LOG_I(RRC, "eNB %d: Update the E-RABS %u\n", mod_id, ue_context_target_p->ue_context.nb_of_e_rabs);

  for (i = 0; i < ue_context_target_p->ue_context.nb_of_e_rabs; i++) {
    ue_context_target_p->ue_context.e_rab[i].status = E_RAB_STATUS_NEW;
    ue_context_target_p->ue_context.e_rab[i].param.e_rab_id = m->e_rabs_tobesetup[i].e_rab_id;
    ue_context_target_p->ue_context.e_rab[i].param.sgw_addr = m->e_rabs_tobesetup[i].eNB_addr;
    ue_context_target_p->ue_context.e_rab[i].param.gtp_teid= m->e_rabs_tobesetup[i].gtp_teid;
    LOG_I(RRC, "eNB %d: Update the UE context after HO, e_rab_id %u gtp_teid %u\n", mod_id,
          ue_context_target_p->ue_context.e_rab[i].param.e_rab_id,
          ue_context_target_p->ue_context.e_rab[i].param.gtp_teid);
  }

  rrc_eNB_process_X2AP_TUNNEL_SETUP_REQ(mod_id, ue_context_target_p);
  ue_context_target_p->ue_context.Status = RRC_HO_EXECUTION;
  ue_context_target_p->ue_context.handover_info->state = HO_ACK;
}

void rrc_eNB_process_handoverCommand(
  int                         mod_id,
  struct rrc_eNB_ue_context_s *ue_context,
  x2ap_handover_req_ack_t     *m) {
  asn_dec_rval_t dec_rval;
  LTE_HandoverCommand_t *ho = NULL;
  dec_rval = uper_decode(
               NULL,
               &asn_DEF_LTE_HandoverCommand,
               (void **)&ho,
               m->rrc_buffer,
               m->rrc_buffer_size,
               0,
               0);

  if (dec_rval.code != RC_OK ||
      ho->criticalExtensions.present != LTE_HandoverCommand__criticalExtensions_PR_c1 ||
      ho->criticalExtensions.choice.c1.present != LTE_HandoverCommand__criticalExtensions__c1_PR_handoverCommand_r8) {
    LOG_E(RRC, "could not decode Handover Command\n");
    abort();
  }

  unsigned char *buf = ho->criticalExtensions.choice.c1.choice.handoverCommand_r8.handoverCommandMessage.buf;
  int size = ho->criticalExtensions.choice.c1.choice.handoverCommand_r8.handoverCommandMessage.size;

  if (size > RRC_BUF_SIZE) {
    printf("%s:%d: fatal\n", __FILE__, __LINE__);
    abort();
  }

  memcpy(ue_context->ue_context.handover_info->buf, buf, size);
  ue_context->ue_context.handover_info->size = size;
}

void rrc_eNB_handover_ue_context_release(
  protocol_ctxt_t *const ctxt_pP,
  struct rrc_eNB_ue_context_s *ue_context_p) {
  int e_rab = 0;
  //MessageDef *msg_release_p = NULL;
  MessageDef *msg_delete_tunnels_p = NULL;
  uint32_t eNB_ue_s1ap_id = ue_context_p->ue_context.eNB_ue_s1ap_id;
  //msg_release_p = itti_alloc_new_message(TASK_RRC_ENB, S1AP_UE_CONTEXT_RELEASE);
  //itti_send_msg_to_task(TASK_S1AP, ctxt_pP->module_id, msg_release_p);
  s1ap_ue_context_release(ctxt_pP->instance, ue_context_p->ue_context.eNB_ue_s1ap_id);
  //MSC_LOG_TX_MESSAGE(MSC_RRC_ENB, MSC_GTPU_ENB, NULL,0, "0 GTPV1U_ENB_DELETE_TUNNEL_REQ rnti %x ", eNB_ue_s1ap_id);
  msg_delete_tunnels_p = itti_alloc_new_message(TASK_RRC_ENB, GTPV1U_ENB_DELETE_TUNNEL_REQ);
  memset(&GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p), 0, sizeof(GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p)));
  GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).rnti = ue_context_p->ue_context.rnti;

  for (e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
    GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).eps_bearer_id[GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).num_erab++] =
      ue_context_p->ue_context.enb_gtp_ebi[e_rab];
    // erase data
    ue_context_p->ue_context.enb_gtp_teid[e_rab] = 0;
    memset(&ue_context_p->ue_context.enb_gtp_addrs[e_rab], 0, sizeof(ue_context_p->ue_context.enb_gtp_addrs[e_rab]));
    ue_context_p->ue_context.enb_gtp_ebi[e_rab] = 0;
  }

  itti_send_msg_to_task(TASK_GTPV1_U, ctxt_pP->module_id, msg_delete_tunnels_p);
  struct rrc_ue_s1ap_ids_s *rrc_ue_s1ap_ids = NULL;
  rrc_ue_s1ap_ids = rrc_eNB_S1AP_get_ue_ids(RC.rrc[ctxt_pP->module_id], 0, eNB_ue_s1ap_id);

  if (rrc_ue_s1ap_ids != NULL) {
    rrc_eNB_S1AP_remove_ue_ids(RC.rrc[ctxt_pP->module_id], rrc_ue_s1ap_ids);
  }
}

/* This function may be incorrect. */
void rrc_eNB_handover_cancel(
  protocol_ctxt_t              *const ctxt_pP,
  struct rrc_eNB_ue_context_s  *ue_context_p) {
  int s1_cause = 1;                        /* 1 = tx2relocoverall-expiry */
  rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_REQ(ctxt_pP->module_id, ue_context_p,
      S1AP_CAUSE_RADIO_NETWORK, s1_cause);
}


int
flexran_rrc_eNB_trigger_handover (int mod_id,
                                  const protocol_ctxt_t *const ctxt_pP,
                                  rrc_eNB_ue_context_t  *ue_context_pP,
                                  int target_cell_id) {
  uint32_t earfcn_dl;
  uint8_t KeNB_star[32] = { 0 };
  int cell_found = 0;

  /* if X2AP is disabled, do nothing */
  if (!is_x2ap_enabled()) {
    LOG_E(RRC, "X2 is disabled\n");
    return -1;
  }

  /* Check if eNB id belongs to the supported ones-Extend for multiple carrieres */
  for (int i=0; i < RC.rrc[mod_id]->num_neigh_cells; i++) {
    if (RC.rrc[mod_id]->neigh_cells_id[i][0] == target_cell_id) {
      cell_found = 1;
      break;
    }
  }

  /* Check if eNB id was found */
  if (!cell_found) {
    LOG_E(RRC, "%s(): cannot find target eNB with phyCellId %d\n", __func__, target_cell_id);
    return -1;
  }

  /* Handover process is following */
  LOG_D(RRC, "Handover is triggered by FlexRAN controller...\n");

  /* if the UE is not in handover mode, start handover procedure */
  if (ue_context_pP->ue_context.Status != RRC_HO_EXECUTION) {
    MessageDef      *msg;
    LOG_I(RRC, "Send HO preparation message at frame %d and subframe %d \n", ctxt_pP->frame, ctxt_pP->subframe);
    /* Check memory leakage for handover info */
    //if (ue_context_pP->ue_context.handover_info) {
    //free(ue_context_pP->ue_context.handover_info);
    //}
    ue_context_pP->ue_context.handover_info = CALLOC(1, sizeof(*(ue_context_pP->ue_context.handover_info)));
    ue_context_pP->ue_context.Status = RRC_HO_EXECUTION;
    ue_context_pP->ue_context.handover_info->state = HO_REQUEST;
    /* HO Preparation message */
    msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_HANDOVER_REQ);
    rrc_eNB_generate_HandoverPreparationInformation(
      ue_context_pP,
      X2AP_HANDOVER_REQ(msg).rrc_buffer,
      &X2AP_HANDOVER_REQ(msg).rrc_buffer_size);
    X2AP_HANDOVER_REQ(msg).rnti = ctxt_pP->rnti;
    X2AP_HANDOVER_REQ(msg).target_physCellId = target_cell_id;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mcc = ue_context_pP->ue_context.ue_gummei.mcc;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mnc = ue_context_pP->ue_context.ue_gummei.mnc;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mnc_len = ue_context_pP->ue_context.ue_gummei.mnc_len;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mme_code = ue_context_pP->ue_context.ue_gummei.mme_code;
    X2AP_HANDOVER_REQ(msg).ue_gummei.mme_group_id = ue_context_pP->ue_context.ue_gummei.mme_group_id;
    // Don't know how to get this ID?
    X2AP_HANDOVER_REQ(msg).mme_ue_s1ap_id = ue_context_pP->ue_context.mme_ue_s1ap_id;
    X2AP_HANDOVER_REQ(msg).security_capabilities = ue_context_pP->ue_context.security_capabilities;
    // compute keNB*
    earfcn_dl = (uint32_t)to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->carrier[0].eutra_band, RC.rrc[ctxt_pP->module_id]->carrier[0].dl_CarrierFreq,
                                       RC.rrc[ctxt_pP->module_id]->carrier[0].N_RB_DL);
    derive_keNB_star(ue_context_pP->ue_context.kenb, X2AP_HANDOVER_REQ(msg).target_physCellId, earfcn_dl, true, KeNB_star);
    memcpy(X2AP_HANDOVER_REQ(msg).kenb, KeNB_star, 32);
    X2AP_HANDOVER_REQ(msg).kenb_ncc = ue_context_pP->ue_context.kenb_ncc;
    //X2AP_HANDOVER_REQ(msg).ue_ambr=ue_context_pP->ue_context.ue_ambr;
    X2AP_HANDOVER_REQ(msg).nb_e_rabs_tobesetup = ue_context_pP->ue_context.setup_e_rabs;

    for (int i=0; i<ue_context_pP->ue_context.setup_e_rabs; i++) {
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].e_rab_id = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].eNB_addr = ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      X2AP_HANDOVER_REQ(msg).e_rabs_tobesetup[i].gtp_teid = ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.qci = ue_context_pP->ue_context.e_rab[i].param.qos.qci;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.priority_level = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.priority_level;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_capability = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_capability;
      X2AP_HANDOVER_REQ(msg).e_rab_param[i].qos.allocation_retention_priority.pre_emp_vulnerability = ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_vulnerability;
    }

    /* TODO: don't do that, X2AP should find the target by itself */
    //X2AP_HANDOVER_REQ(msg).target_mod_id = 0;
    LOG_I(RRC,
          "[eNB %d] Frame %d: potential handover preparation: store the information in an intermediate structure in case of failure\n",
          ctxt_pP->module_id, ctxt_pP->frame);
    itti_send_msg_to_task(TASK_X2AP, ENB_MODULE_ID_TO_INSTANCE(ctxt_pP->module_id), msg);
  } else {
    LOG_D(RRC, "[eNB %d] Frame %d: Ignoring MeasReport from UE %x as Handover is in progress... \n", ctxt_pP->module_id, ctxt_pP->frame,
          ctxt_pP->rnti);
  }

  return 0;
}

void
check_handovers(
  protocol_ctxt_t *const ctxt_pP
)
//-----------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_s        *ue_context_p;
  RB_FOREACH(ue_context_p, rrc_ue_tree_s, &(RC.rrc[ctxt_pP->module_id]->rrc_ue_head)) {
    ctxt_pP->rnti  = ue_context_p->ue_id_rnti;

    if (ue_context_p->ue_context.Status == RRC_HO_EXECUTION && ue_context_p->ue_context.handover_info != NULL) {
      /* in the source, UE in HO_PREPARE mode */
      if (ue_context_p->ue_context.handover_info->state == HO_PREPARE) {
        LOG_D(RRC,
              "[eNB %d] Frame %d: Incoming handover detected for new UE_id %x) \n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ctxt_pP->rnti);
        // source eNB generates rrcconnectionreconfiguration to prepare the HO
        LOG_I(RRC,
              "[eNB %d] Frame %d : Logical Channel UL-DCCH, processing RRCHandoverPreparationInformation, sending RRCConnectionReconfiguration to UE %d \n",
              ctxt_pP->module_id, ctxt_pP->frame, ue_context_p->ue_context.rnti);
        rrc_data_req(
          ctxt_pP,
          DCCH,
          rrc_eNB_mui++,
          SDU_CONFIRM_NO,
          ue_context_p->ue_context.handover_info->size,
          ue_context_p->ue_context.handover_info->buf,
          PDCP_TRANSMISSION_MODE_CONTROL);
        ue_context_p->ue_context.handover_info->state = HO_COMPLETE;
        LOG_I(RRC, "RRC Sends RRCConnectionReconfiguration to UE %d  at frame %d and subframe %d \n", ue_context_p->ue_context.rnti, ctxt_pP->frame,ctxt_pP->subframe);
      }

      /* in the target, UE in HO_ACK mode */
      if (ue_context_p->ue_context.handover_info->state == HO_ACK) {
        MessageDef *msg;
        // Configure target
        ue_context_p->ue_context.handover_info->state = HO_FORWARDING;
        msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_HANDOVER_REQ_ACK);
        rrc_eNB_generate_HO_RRCConnectionReconfiguration(ctxt_pP, ue_context_p, X2AP_HANDOVER_REQ_ACK(msg).rrc_buffer,
            &X2AP_HANDOVER_REQ_ACK(msg).rrc_buffer_size);
        rrc_eNB_configure_rbs_handover(ue_context_p,ctxt_pP);
        X2AP_HANDOVER_REQ_ACK(msg).rnti = ue_context_p->ue_context.rnti;
        X2AP_HANDOVER_REQ_ACK(msg).x2_id_target = ue_context_p->ue_context.handover_info->x2_id;
        X2AP_HANDOVER_REQ_ACK(msg).source_assoc_id = ue_context_p->ue_context.handover_info->assoc_id;
        /* Call admission control not implemented yet */
        X2AP_HANDOVER_REQ_ACK(msg).nb_e_rabs_tobesetup = ue_context_p->ue_context.setup_e_rabs;

        for (int i=0; i<ue_context_p->ue_context.setup_e_rabs; i++) {
          /* set gtpv teid info */
          X2AP_HANDOVER_REQ_ACK(msg).e_rabs_tobesetup[i].e_rab_id = ue_context_p->ue_context.e_rab[i].param.e_rab_id;
          X2AP_HANDOVER_REQ_ACK(msg).e_rabs_tobesetup[i].gtp_teid = ue_context_p->ue_context.enb_gtp_x2u_teid[i];
          X2AP_HANDOVER_REQ_ACK(msg).e_rabs_tobesetup[i].eNB_addr = ue_context_p->ue_context.enb_gtp_x2u_addrs[i];
        }

        itti_send_msg_to_task(TASK_X2AP, ENB_MODULE_ID_TO_INSTANCE(ctxt_pP->module_id), msg);
        LOG_I(RRC, "RRC Sends X2 HO ACK to the source eNB at frame %d and subframe %d \n", ctxt_pP->frame,ctxt_pP->subframe);
      }
    }

    if (ue_context_p->ue_context.Status == RRC_RECONFIGURED
        && ue_context_p->ue_context.handover_info != NULL &&
        ue_context_p->ue_context.handover_info->forwarding_state == FORWARDING_NO_EMPTY ) {
      MessageDef   *msg_p;
      int    result;
      protocol_ctxt_t  ctxt;

      do {
        // Checks if a message has been sent to PDCP sub-task
        itti_poll_msg (TASK_DATA_FORWARDING, &msg_p);

        if (msg_p != NULL) {
          switch (ITTI_MSG_ID(msg_p)) {
            case GTPV1U_ENB_DATA_FORWARDING_IND:
              PROTOCOL_CTXT_SET_BY_MODULE_ID(
                &ctxt,
                GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).module_id,
                GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).enb_flag,
                GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rnti,
                GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).frame,
                0,
                GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).eNB_index);
              LOG_D(RRC, PROTOCOL_CTXT_FMT"[check_handovers]Received %s from %s: instance %d, rb_id %ld, muiP %d, confirmP %d, mode %d\n",
                    PROTOCOL_CTXT_ARGS(&ctxt),
                    ITTI_MSG_NAME (msg_p),
                    ITTI_MSG_ORIGIN_NAME(msg_p),
                    ITTI_MSG_INSTANCE (msg_p),
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).muip,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).confirmp,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).mode);
              LOG_I(RRC, "Before calling pdcp_data_req from check_handovers! GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id: %ld \n", GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id);
              result = pdcp_data_req (&ctxt,
                                      SRB_FLAG_NO,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).muip,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).confirmp,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).sdu_size,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).sdu_p,
                                      GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).mode, NULL, NULL
                                     );

              if (result != TRUE) {
                LOG_E(RRC, "target enb send data forwarding buffer to PDCP request failed!\n");
              } else {
                LOG_D(RRC, "target enb send data forwarding buffer to PDCP!\n");
              }

              // Message buffer has been processed, free it now.
              result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).sdu_p);
              AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
              break;

            default:
              LOG_E(RRC, "Received unexpected message %s\n", ITTI_MSG_NAME (msg_p));
              break;
          }

          result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
          AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
        }
      } while(msg_p != NULL);

      ue_context_p->ue_context.handover_info->forwarding_state = FORWARDING_EMPTY;
    }

    if( ue_context_p->ue_context.Status == RRC_RECONFIGURED &&
        ue_context_p->ue_context.handover_info != NULL &&
        ue_context_p->ue_context.handover_info->forwarding_state == FORWARDING_EMPTY &&
        ue_context_p->ue_context.handover_info->endmark_state == ENDMARK_NO_EMPTY &&
        ue_context_p->ue_context.handover_info->state == HO_END_MARKER    ) {
      MessageDef   *msg_p;
      int    result;
      protocol_ctxt_t  ctxt;

      do {
        // Checks if a message has been sent to PDCP sub-task
        itti_poll_msg (TASK_END_MARKER, &msg_p);

        if (msg_p != NULL) {
          switch (ITTI_MSG_ID(msg_p)) {
            case GTPV1U_ENB_END_MARKER_IND:
              PROTOCOL_CTXT_SET_BY_MODULE_ID(
                &ctxt,
                GTPV1U_ENB_END_MARKER_IND (msg_p).module_id,
                GTPV1U_ENB_END_MARKER_IND (msg_p).enb_flag,
                GTPV1U_ENB_END_MARKER_IND (msg_p).rnti,
                GTPV1U_ENB_END_MARKER_IND (msg_p).frame,
                0,
                GTPV1U_ENB_END_MARKER_IND (msg_p).eNB_index);
              LOG_I(RRC, PROTOCOL_CTXT_FMT"[check_handovers]Received %s from %s: instance %d, rb_id %ld, muiP %d, confirmP %d, mode %d\n",
                    PROTOCOL_CTXT_ARGS(&ctxt),
                    ITTI_MSG_NAME (msg_p),
                    ITTI_MSG_ORIGIN_NAME(msg_p),
                    ITTI_MSG_INSTANCE (msg_p),
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).muip,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).confirmp,
                    GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).mode);
              LOG_D(RRC, "Before calling pdcp_data_req from check_handovers! GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id: %ld \n", GTPV1U_ENB_DATA_FORWARDING_IND (msg_p).rb_id);
              result = pdcp_data_req (&ctxt,
                                      SRB_FLAG_NO,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).rb_id,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).muip,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).confirmp,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).sdu_size,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).sdu_p,
                                      GTPV1U_ENB_END_MARKER_IND (msg_p).mode, NULL, NULL
                                     );

              if (result != TRUE) {
                LOG_E(RRC, "target enb send spgw buffer to PDCP request failed!\n");
              } else {
                LOG_D(RRC, "target enb send spgw buffer to PDCP!\n");
              }

              // Message buffer has been processed, free it now.
              result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), GTPV1U_ENB_END_MARKER_IND (msg_p).sdu_p);
              AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
              break;

            default:
              LOG_E(RRC, "Received unexpected message %s\n", ITTI_MSG_NAME (msg_p));
              break;
          }

          result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
          AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
        }
      } while(msg_p != NULL);

      ue_context_p->ue_context.handover_info->endmark_state = ENDMARK_EMPTY;
      ue_context_p->ue_context.handover_info->state = HO_FORWARDING_COMPLETE;
    }
  }
}

void
rrc_eNB_generate_HO_RRCConnectionReconfiguration(const protocol_ctxt_t *const ctxt_pP,
    rrc_eNB_ue_context_t  *const ue_context_pP,
    uint8_t               *buffer,
    int                    *_size
    //const uint8_t        ho_state
                                                )
//-----------------------------------------------------------------------------
{
  uint16_t                            size;
  int                                 i;
  uint8_t                             rv[2];
  // configure SRB1/SRB2, PhysicalConfigDedicated, MAC_MainConfig for UE
  eNB_RRC_INST                       *rrc_inst = RC.rrc[ctxt_pP->module_id];
  struct LTE_PhysicalConfigDedicated **physicalConfigDedicated = &ue_context_pP->ue_context.physicalConfigDedicated;
  // phy config dedicated
  LTE_PhysicalConfigDedicated_t      *physicalConfigDedicated2;
  // srb 1: for HO
  struct LTE_SRB_ToAddMod            *SRB1_config                      = NULL;
  struct LTE_SRB_ToAddMod__rlc_Config *SRB1_rlc_config                 = NULL;
  struct LTE_SRB_ToAddMod__logicalChannelConfig *SRB1_lchan_config     = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *SRB1_ul_SpecificParameters       = NULL;
  struct LTE_SRB_ToAddMod            *SRB2_config                      = NULL;
  struct LTE_SRB_ToAddMod__rlc_Config *SRB2_rlc_config                 = NULL;
  struct LTE_SRB_ToAddMod__logicalChannelConfig *SRB2_lchan_config     = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *SRB2_ul_SpecificParameters       = NULL;
  LTE_SRB_ToAddModList_t             *SRB_configList = ue_context_pP->ue_context.SRB_configList;
  LTE_SRB_ToAddModList_t             **SRB_configList2                 = NULL;
  struct LTE_DRB_ToAddMod            *DRB_config                       = NULL;
  struct LTE_RLC_Config              *DRB_rlc_config                   = NULL;
  struct LTE_PDCP_Config             *DRB_pdcp_config                  = NULL;
  struct LTE_PDCP_Config__rlc_AM     *PDCP_rlc_AM                      = NULL;
  struct LTE_PDCP_Config__rlc_UM     *PDCP_rlc_UM                      = NULL;
  struct LTE_LogicalChannelConfig    *DRB_lchan_config                 = NULL;
  struct LTE_LogicalChannelConfig__ul_SpecificParameters
    *DRB_ul_SpecificParameters        = NULL;
  LTE_DRB_ToAddModList_t            **DRB_configList = &ue_context_pP->ue_context.DRB_configList;
  LTE_DRB_ToAddModList_t            **DRB_configList2 = NULL;
  LTE_MAC_MainConfig_t               *mac_MainConfig                   = NULL;
  LTE_MeasObjectToAddModList_t       *MeasObj_list                     = NULL;
  LTE_MeasObjectToAddMod_t           *MeasObj                          = NULL;
  LTE_ReportConfigToAddModList_t     *ReportConfig_list                = NULL;
  LTE_ReportConfigToAddMod_t         *ReportConfig_per, *ReportConfig_A1,
                                     *ReportConfig_A2, *ReportConfig_A3, *ReportConfig_A4, *ReportConfig_A5;
  LTE_MeasIdToAddModList_t           *MeasId_list                      = NULL;
  LTE_MeasIdToAddMod_t               *MeasId0, *MeasId1, *MeasId2, *MeasId3, *MeasId4, *MeasId5;
  long                               *sr_ProhibitTimer_r9              = NULL;
  long                               *logicalchannelgroup, *logicalchannelgroup_drb;
  long                               *maxHARQ_Tx, *periodicBSR_Timer;
  LTE_RSRP_Range_t                   *rsrp                             = NULL;
  struct LTE_MeasConfig__speedStatePars *Sparams                       = NULL;
  LTE_QuantityConfig_t               *quantityConfig                   = NULL;
  LTE_MobilityControlInfo_t          *mobilityInfo                     = NULL;
  LTE_SecurityConfigHO_t             *securityConfigHO                 = NULL;
  LTE_CellsToAddMod_t                *CellToAdd                        = NULL;
  LTE_CellsToAddModList_t            *CellsToAddModList                = NULL;
  struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *dedicatedInfoNASList = NULL;
  LTE_DedicatedInfoNAS_t             *dedicatedInfoNas                 = NULL;
  /* for no gcc warnings */
  (void)dedicatedInfoNas;
  LTE_C_RNTI_t                       *cba_RNTI                         = NULL;
  int                                measurements_enabled;
  uint8_t xid = rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id);   //Transaction_id,
  T(T_ENB_RRC_CONNECTION_RECONFIGURATION, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  rv[0] = (ue_context_pP->ue_context.rnti >> 8) & 255;
  rv[1] = ue_context_pP->ue_context.rnti & 255;
  LOG_I(RRC, "target UE rnti = %x (decimal: %d)\n", ue_context_pP->ue_context.rnti, ue_context_pP->ue_context.rnti);
  LOG_D(RRC, "[eNB %d] Frame %d : handover preparation: add target eNB SRB1 and PHYConfigDedicated reconfiguration\n",
        ctxt_pP->module_id, ctxt_pP->frame);

  if (SRB_configList) {
    free(SRB_configList);
  }

  SRB_configList = CALLOC(1, sizeof(*SRB_configList));
  memset(SRB_configList, 0, sizeof(*SRB_configList));
  SRB1_config = CALLOC(1, sizeof(*SRB1_config));
  SRB1_config->srb_Identity = 1;
  SRB1_rlc_config = CALLOC(1, sizeof(*SRB1_rlc_config));
  SRB1_config->rlc_Config = SRB1_rlc_config;
  SRB1_rlc_config->present = LTE_SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB1_rlc_config->choice.explicitValue.present = LTE_RLC_Config_PR_am;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms15;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p8;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kB1000;
  SRB1_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t16;
  SRB1_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  SRB1_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms10;
  SRB1_lchan_config = CALLOC(1, sizeof(*SRB1_lchan_config));
  SRB1_config->logicalChannelConfig = SRB1_lchan_config;
  SRB1_lchan_config->present = LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue;
  SRB1_ul_SpecificParameters = CALLOC(1, sizeof(*SRB1_ul_SpecificParameters));
  SRB1_lchan_config->choice.explicitValue.ul_SpecificParameters = SRB1_ul_SpecificParameters;
  SRB1_ul_SpecificParameters->priority = 1;
  //assign_enum(&SRB1_ul_SpecificParameters->prioritisedBitRate,LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity);
  SRB1_ul_SpecificParameters->prioritisedBitRate =
    LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  //assign_enum(&SRB1_ul_SpecificParameters->bucketSizeDuration,LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50);
  SRB1_ul_SpecificParameters->bucketSizeDuration =
    LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  logicalchannelgroup = CALLOC(1, sizeof(long));
  *logicalchannelgroup = 0;
  SRB1_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup;
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB1_config);
  ue_context_pP->ue_context.SRB_configList = SRB_configList;
  // Configure SRB2
  /// SRB2
  SRB_configList2=&ue_context_pP->ue_context.SRB_configList2[xid];

  if (*SRB_configList2) {
    free(*SRB_configList2);
  }

  *SRB_configList2 = CALLOC(1, sizeof(**SRB_configList2));
  memset(*SRB_configList2, 0, sizeof(**SRB_configList2));
  SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  SRB2_config->srb_Identity = 2;
  SRB2_rlc_config = CALLOC(1, sizeof(*SRB2_rlc_config));
  SRB2_config->rlc_Config = SRB2_rlc_config;
  SRB2_rlc_config->present = LTE_SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB2_rlc_config->choice.explicitValue.present = LTE_RLC_Config_PR_am;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms15;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p8;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kB1000;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t32;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms10;
  SRB2_lchan_config = CALLOC(1, sizeof(*SRB2_lchan_config));
  SRB2_config->logicalChannelConfig = SRB2_lchan_config;
  SRB2_lchan_config->present = LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue;
  SRB2_ul_SpecificParameters = CALLOC(1, sizeof(*SRB2_ul_SpecificParameters));
  SRB2_ul_SpecificParameters->priority = 3; // let some priority for SRB1 and dedicated DRBs
  SRB2_ul_SpecificParameters->prioritisedBitRate =
    LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  SRB2_ul_SpecificParameters->bucketSizeDuration =
    LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for CCCH and DCCH is 0 as defined in 36331
  logicalchannelgroup = CALLOC(1, sizeof(long));
  *logicalchannelgroup = 0;
  SRB2_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup;
  SRB2_lchan_config->choice.explicitValue.ul_SpecificParameters = SRB2_ul_SpecificParameters;
  // this list has the configuration for SRB1 and SRB2
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);
  // this list has only the configuration for SRB2
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);

  // Configure DRB
  //*DRB_configList = CALLOC(1, sizeof(*DRB_configList));
  // list for all the configured DRB
  if (*DRB_configList) {
    free(*DRB_configList);
  }

  *DRB_configList = CALLOC(1, sizeof(**DRB_configList));
  memset(*DRB_configList, 0, sizeof(**DRB_configList));
  // list for the configured DRB for a this xid
  DRB_configList2=&ue_context_pP->ue_context.DRB_configList2[xid];

  if (*DRB_configList2) {
    free(*DRB_configList2);
  }

  *DRB_configList2 = CALLOC(1, sizeof(**DRB_configList2));
  memset(*DRB_configList2, 0, sizeof(**DRB_configList2));
  /// DRB
  DRB_config = CALLOC(1, sizeof(*DRB_config));
  DRB_config->eps_BearerIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->eps_BearerIdentity) = 5L; // LW set to first value, allowed value 5..15, value : x+4
  // DRB_config->drb_Identity = (DRB_Identity_t) 1; //allowed values 1..32
  // NN: this is the 1st DRB for this ue, so set it to 1
  DRB_config->drb_Identity = (LTE_DRB_Identity_t) 1;  // (ue_mod_idP+1); //allowed values 1..32, value: x
  DRB_config->logicalChannelIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->logicalChannelIdentity) = (long)3; // value : x+2
  DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
  DRB_config->rlc_Config = DRB_rlc_config;
#ifdef RRC_DEFAULT_RAB_IS_AM
  DRB_rlc_config->present = LTE_RLC_Config_PR_am;
  DRB_rlc_config->choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms50;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p16;
  DRB_rlc_config->choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kBinfinity;
  DRB_rlc_config->choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t8;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  DRB_rlc_config->choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms25;
#else
  DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
  DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
#endif
  DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
  DRB_config->pdcp_Config = DRB_pdcp_config;
  DRB_pdcp_config->discardTimer = CALLOC(1, sizeof(long));
  *DRB_pdcp_config->discardTimer = LTE_PDCP_Config__discardTimer_infinity;
  DRB_pdcp_config->rlc_AM = NULL;
  DRB_pdcp_config->rlc_UM = NULL;
  /* avoid gcc warnings */
  (void)PDCP_rlc_AM;
  (void)PDCP_rlc_UM;
#ifdef RRC_DEFAULT_RAB_IS_AM // EXMIMO_IOT
  PDCP_rlc_AM = CALLOC(1, sizeof(*PDCP_rlc_AM));
  DRB_pdcp_config->rlc_AM = PDCP_rlc_AM;
  PDCP_rlc_AM->statusReportRequired = FALSE;
#else
  PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
  DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
  PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
#endif
  DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
  DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
  DRB_config->logicalChannelConfig = DRB_lchan_config;
  DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
  DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;
  DRB_ul_SpecificParameters->priority = 12;    // lower priority than srb1, srb2 and other dedicated bearer
  DRB_ul_SpecificParameters->prioritisedBitRate =LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_kBps8 ;
  //LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  DRB_ul_SpecificParameters->bucketSizeDuration =
    LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for DTCH can take the value from 1 to 3 as defined in 36331: normally controlled by upper layers (like RRM)
  logicalchannelgroup_drb = CALLOC(1, sizeof(long));
  *logicalchannelgroup_drb = 1;
  DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;
  ASN_SEQUENCE_ADD(&(*DRB_configList)->list, DRB_config);
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
  //ue_context_pP->ue_context.DRB_configList2[0] = &(*DRB_configList);
  mac_MainConfig = CALLOC(1, sizeof(*mac_MainConfig));
  ue_context_pP->ue_context.mac_MainConfig = mac_MainConfig;
  mac_MainConfig->ul_SCH_Config = CALLOC(1, sizeof(*mac_MainConfig->ul_SCH_Config));
  maxHARQ_Tx = CALLOC(1, sizeof(long));
  *maxHARQ_Tx = LTE_MAC_MainConfig__ul_SCH_Config__maxHARQ_Tx_n5;
  mac_MainConfig->ul_SCH_Config->maxHARQ_Tx = maxHARQ_Tx;
  periodicBSR_Timer = CALLOC(1, sizeof(long));
  *periodicBSR_Timer = LTE_PeriodicBSR_Timer_r12_sf64;
  mac_MainConfig->ul_SCH_Config->periodicBSR_Timer = periodicBSR_Timer;
  mac_MainConfig->ul_SCH_Config->retxBSR_Timer = LTE_RetxBSR_Timer_r12_sf320;
  mac_MainConfig->ul_SCH_Config->ttiBundling = 0; // FALSE
  mac_MainConfig->timeAlignmentTimerDedicated = LTE_TimeAlignmentTimer_infinity;
  mac_MainConfig->drx_Config = NULL;
  mac_MainConfig->phr_Config = CALLOC(1, sizeof(*mac_MainConfig->phr_Config));
  mac_MainConfig->phr_Config->present = LTE_MAC_MainConfig__phr_Config_PR_setup;
  mac_MainConfig->phr_Config->choice.setup.periodicPHR_Timer = LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_sf20; // sf20 = 20 subframes
  mac_MainConfig->phr_Config->choice.setup.prohibitPHR_Timer = LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf20; // sf20 = 20 subframes
  mac_MainConfig->phr_Config->choice.setup.dl_PathlossChange = LTE_MAC_MainConfig__phr_Config__setup__dl_PathlossChange_dB1;  // Value dB1 =1 dB, dB3 = 3 dB
  sr_ProhibitTimer_r9 = CALLOC(1, sizeof(long));
  *sr_ProhibitTimer_r9 = 0;   // SR tx on PUCCH, Value in number of SR period(s). Value 0 = no timer for SR, Value 2= 2*SR
  mac_MainConfig->ext1 = CALLOC(1, sizeof(struct LTE_MAC_MainConfig__ext1));
  mac_MainConfig->ext1->sr_ProhibitTimer_r9 = sr_ProhibitTimer_r9;
  //sps_RA_ConfigList_rlola = NULL;

  //change the transmission mode for the primary component carrier
  //TODO: add codebook subset restriction here
  //TODO: change TM for secondary CC in SCelltoaddmodlist
  /// now reconfigure phy config dedicated
  if (*physicalConfigDedicated) {
    free(*physicalConfigDedicated);
  }

  //if (*physicalConfigDedicated) {
  physicalConfigDedicated2 = CALLOC(1, sizeof(*physicalConfigDedicated2));
  *physicalConfigDedicated = physicalConfigDedicated2;
  physicalConfigDedicated2->pdsch_ConfigDedicated =
    CALLOC(1, sizeof(*physicalConfigDedicated2->pdsch_ConfigDedicated));
  physicalConfigDedicated2->pucch_ConfigDedicated =
    CALLOC(1, sizeof(*physicalConfigDedicated2->pucch_ConfigDedicated));
  physicalConfigDedicated2->pusch_ConfigDedicated =
    CALLOC(1, sizeof(*physicalConfigDedicated2->pusch_ConfigDedicated));
  physicalConfigDedicated2->uplinkPowerControlDedicated =
    CALLOC(1, sizeof(*physicalConfigDedicated2->uplinkPowerControlDedicated));
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH =
    CALLOC(1, sizeof(*physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH));
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH =
    CALLOC(1, sizeof(*physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH));
  physicalConfigDedicated2->cqi_ReportConfig = NULL;  //CALLOC(1,sizeof(*physicalConfigDedicated2->cqi_ReportConfig));
  physicalConfigDedicated2->soundingRS_UL_ConfigDedicated = NULL; //CALLOC(1,sizeof(*physicalConfigDedicated2->soundingRS_UL_ConfigDedicated));
  physicalConfigDedicated2->antennaInfo = CALLOC(1, sizeof(*physicalConfigDedicated2->antennaInfo));
  physicalConfigDedicated2->schedulingRequestConfig =
    CALLOC(1, sizeof(*physicalConfigDedicated2->schedulingRequestConfig));
  // PDSCH
  //assign_enum(&physicalConfigDedicated2->pdsch_ConfigDedicated->p_a,
  //          PDSCH_ConfigDedicated__p_a_dB0);
  physicalConfigDedicated2->pdsch_ConfigDedicated->p_a = LTE_PDSCH_ConfigDedicated__p_a_dB0;
  // PUCCH
  physicalConfigDedicated2->pucch_ConfigDedicated->ackNackRepetition.present =
    LTE_PUCCH_ConfigDedicated__ackNackRepetition_PR_release;
  physicalConfigDedicated2->pucch_ConfigDedicated->ackNackRepetition.choice.release = 0;
  physicalConfigDedicated2->pucch_ConfigDedicated->tdd_AckNackFeedbackMode = NULL;    //PUCCH_ConfigDedicated__tdd_AckNackFeedbackMode_multiplexing;
  // Pusch_config_dedicated
  physicalConfigDedicated2->pusch_ConfigDedicated->betaOffset_ACK_Index = 0;  // 2.00
  physicalConfigDedicated2->pusch_ConfigDedicated->betaOffset_RI_Index = 0;   // 1.25
  physicalConfigDedicated2->pusch_ConfigDedicated->betaOffset_CQI_Index = 8;  // 2.25
  // UplinkPowerControlDedicated
  physicalConfigDedicated2->uplinkPowerControlDedicated->p0_UE_PUSCH = 0; // 0 dB
  //assign_enum(&physicalConfigDedicated2->uplinkPowerControlDedicated->deltaMCS_Enabled,
  // UplinkPowerControlDedicated__deltaMCS_Enabled_en1);
  physicalConfigDedicated2->uplinkPowerControlDedicated->deltaMCS_Enabled =
    LTE_UplinkPowerControlDedicated__deltaMCS_Enabled_en1;
  physicalConfigDedicated2->uplinkPowerControlDedicated->accumulationEnabled = 1; // should be TRUE in order to have 0dB power offset
  physicalConfigDedicated2->uplinkPowerControlDedicated->p0_UE_PUCCH = 0; // 0 dB
  physicalConfigDedicated2->uplinkPowerControlDedicated->pSRS_Offset = 0; // 0 dB
  physicalConfigDedicated2->uplinkPowerControlDedicated->filterCoefficient =
    CALLOC(1, sizeof(*physicalConfigDedicated2->uplinkPowerControlDedicated->filterCoefficient));
  //  assign_enum(physicalConfigDedicated2->uplinkPowerControlDedicated->filterCoefficient,FilterCoefficient_fc4); // fc4 dB
  *physicalConfigDedicated2->uplinkPowerControlDedicated->filterCoefficient = LTE_FilterCoefficient_fc4;  // fc4 dB
  // TPC-PDCCH-Config
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->present = LTE_TPC_PDCCH_Config_PR_setup;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_Index.present = LTE_TPC_Index_PR_indexOfFormat3;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_Index.choice.indexOfFormat3 = 1;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_RNTI.buf = CALLOC(1, 2);
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_RNTI.size = 2;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_RNTI.buf[0] = 0x12;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_RNTI.buf[1] = 0x34 + ue_context_pP->local_uid;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUCCH->choice.setup.tpc_RNTI.bits_unused = 0;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->present = LTE_TPC_PDCCH_Config_PR_setup;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_Index.present = LTE_TPC_Index_PR_indexOfFormat3;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_Index.choice.indexOfFormat3 = 1;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_RNTI.buf = CALLOC(1, 2);
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_RNTI.size = 2;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_RNTI.buf[0] = 0x22;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_RNTI.buf[1] = 0x34 + ue_context_pP->local_uid;
  physicalConfigDedicated2->tpc_PDCCH_ConfigPUSCH->choice.setup.tpc_RNTI.bits_unused = 0;
  //AntennaInfoDedicated
  physicalConfigDedicated2->antennaInfo = CALLOC(1, sizeof(*physicalConfigDedicated2->antennaInfo));
  physicalConfigDedicated2->antennaInfo->present = LTE_PhysicalConfigDedicated__antennaInfo_PR_explicitValue;
  //if ((*physicalConfigDedicated)->antennaInfo) {
  (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.transmissionMode = rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode;
  LOG_D(RRC,"Setting transmission mode to %ld+1\n",rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode);

  if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm3) {
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
      CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
      LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm3;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf= MALLOC(1);
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.buf[0] = 0xc0;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.size=1;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm3.bits_unused=6;
  } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm4) {
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
      CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
      LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm4;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf= MALLOC(1);
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.buf[0] = 0xfc;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.size=1;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm4.bits_unused=2;
  } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm5) {
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
      CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
      LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm5;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf= MALLOC(1);
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.buf[0] = 0xf0;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.size=1;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm5.bits_unused=4;
  } else if (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm6) {
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction=
      CALLOC(1,sizeof(LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR));
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->present =
      LTE_AntennaInfoDedicated__codebookSubsetRestriction_PR_n2TxAntenna_tm6;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf= MALLOC(1);
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.buf[0] = 0xf0;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.size=1;
    (*physicalConfigDedicated)->antennaInfo->choice.explicitValue.codebookSubsetRestriction->choice.n2TxAntenna_tm6.bits_unused=4;
  }

  physicalConfigDedicated2->antennaInfo->choice.explicitValue.ue_TransmitAntennaSelection.present =
    LTE_AntennaInfoDedicated__ue_TransmitAntennaSelection_PR_release;
  physicalConfigDedicated2->antennaInfo->choice.explicitValue.ue_TransmitAntennaSelection.choice.release = 0;
  //}
  //else {
  //LOG_E(RRC,"antenna_info not present in physical_config_dedicated. Not reconfiguring!\n");
  //}
  // CQI ReportConfig
  physicalConfigDedicated2->cqi_ReportConfig = CALLOC(1,sizeof(*physicalConfigDedicated2->cqi_ReportConfig));
  physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportModeAperiodic = CALLOC(1,sizeof(*physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportModeAperiodic));
  physicalConfigDedicated2->cqi_ReportConfig->nomPDSCH_RS_EPRE_Offset = 0; // 0 dB
  //physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportPeriodic=NULL;
  physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportPeriodic=CALLOC(1,sizeof(*physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportPeriodic));
  physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportPeriodic->present =  LTE_CQI_ReportPeriodic_PR_release;

  //if ((*physicalConfigDedicated)->cqi_ReportConfig) {
  if ((rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm4) ||
      (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm5) ||
      (rrc_inst->configuration.radioresourceconfig[0].ue_TransmissionMode==LTE_AntennaInfoDedicated__transmissionMode_tm6)) {
    //feedback mode needs to be set as well
    //TODO: I think this is taken into account in the PHY automatically based on the transmission mode variable
    printf("setting cqi reporting mode to rm31\n");
    *((*physicalConfigDedicated)->cqi_ReportConfig->cqi_ReportModeAperiodic)=LTE_CQI_ReportModeAperiodic_rm31;
  } else {
    *physicalConfigDedicated2->cqi_ReportConfig->cqi_ReportModeAperiodic= LTE_CQI_ReportModeAperiodic_rm30;
  }

  //}
  //else {
  //LOG_E(RRC,"cqi_ReportConfig not present in physical_config_dedicated. Not reconfiguring!\n");
  //}
  // SchedulingRequestConfig
  physicalConfigDedicated2->schedulingRequestConfig->present = LTE_SchedulingRequestConfig_PR_setup;
  physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_PUCCH_ResourceIndex = ue_context_pP->local_uid;

  if (rrc_inst->carrier[0].sib1->tdd_Config==NULL) {  // FDD
    physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_ConfigIndex = 5 + (ue_context_pP->local_uid %
        10);   // Isr = 5 (every 10 subframes, offset=2+UE_id mod3)
  } else {
    switch (rrc_inst->carrier[0].sib1->tdd_Config->subframeAssignment) {
      case 1:
        physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_ConfigIndex = 7 + (ue_context_pP->local_uid & 1) + ((
              ue_context_pP->local_uid & 3) >> 1) * 5;    // Isr = 5 (every 10 subframes, offset=2 for UE0, 3 for UE1, 7 for UE2, 8 for UE3 , 2 for UE4 etc..)
        break;

      case 3:
        physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_ConfigIndex = 7 + (ue_context_pP->local_uid %
            3);    // Isr = 5 (every 10 subframes, offset=2 for UE0, 3 for UE1, 3 for UE2, 2 for UE3 , etc..)
        break;

      case 4:
        physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_ConfigIndex = 7 + (ue_context_pP->local_uid &
            1);    // Isr = 5 (every 10 subframes, offset=2 for UE0, 3 for UE1, 3 for UE2, 2 for UE3 , etc..)
        break;

      default:
        physicalConfigDedicated2->schedulingRequestConfig->choice.setup.sr_ConfigIndex = 7; // Isr = 5 (every 10 subframes, offset=2 for all UE0 etc..)
        break;
    }
  }

  //  assign_enum(&physicalConfigDedicated2->schedulingRequestConfig->choice.setup.dsr_TransMax,
  //SchedulingRequestConfig__setup__dsr_TransMax_n4);
  //  assign_enum(&physicalConfigDedicated2->schedulingRequestConfig->choice.setup.dsr_TransMax = SchedulingRequestConfig__setup__dsr_TransMax_n4;
  physicalConfigDedicated2->schedulingRequestConfig->choice.setup.dsr_TransMax =
    LTE_SchedulingRequestConfig__setup__dsr_TransMax_n4;
  LOG_D(RRC,
        "handover_config [FRAME %05d][RRC_eNB][MOD %02d][][--- MAC_CONFIG_REQ  (SRB1 UE %x) --->][MAC_eNB][MOD %02d][]\n",
        ctxt_pP->frame, ctxt_pP->module_id, ue_context_pP->ue_context.rnti, ctxt_pP->module_id);
  rrc_mac_config_req_eNB(
    ctxt_pP->module_id,
    ue_context_pP->ue_context.primaryCC_id,
    0,0,0,0,0,0,
    ue_context_pP->ue_context.rnti,
    (LTE_BCCH_BCH_Message_t *) NULL,
    (LTE_RadioResourceConfigCommonSIB_t *) NULL,
    (LTE_RadioResourceConfigCommonSIB_t *) NULL,
    ue_context_pP->ue_context.physicalConfigDedicated,
    (LTE_SCellToAddMod_r10_t *)NULL,
    (LTE_MeasObjectToAddMod_t **) NULL,
    ue_context_pP->ue_context.mac_MainConfig,
    1,
    NULL,//SRB1_logicalChannelConfig,
    ue_context_pP->ue_context.measGapConfig,
    (LTE_TDD_Config_t *) NULL,
    (LTE_MobilityControlInfo_t *) NULL,
    (LTE_SchedulingInfoList_t *) NULL,
    0,
    NULL,
    NULL,
    (LTE_MBSFN_SubframeConfigList_t *) NULL,
    0, (LTE_MBSFN_AreaInfoList_r9_t *) NULL, (LTE_PMCH_InfoList_r9_t *) NULL,
    (LTE_SystemInformationBlockType1_v1310_IEs_t *)NULL,
    0,
    (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
    (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
    (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
    (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
    (LTE_MBSFN_AreaInfoList_r9_t *) NULL
  );
  // Configure target eNB SRB2
  /// SRB2
  SRB2_config = CALLOC(1, sizeof(*SRB2_config));
  SRB_configList2 = CALLOC(1, sizeof(*SRB_configList2));
  memset(SRB_configList2, 0, sizeof(*SRB_configList2));
  SRB2_config->srb_Identity = 2;
  SRB2_rlc_config = CALLOC(1, sizeof(*SRB2_rlc_config));
  SRB2_config->rlc_Config = SRB2_rlc_config;
  SRB2_rlc_config->present = LTE_SRB_ToAddMod__rlc_Config_PR_explicitValue;
  SRB2_rlc_config->choice.explicitValue.present = LTE_RLC_Config_PR_am;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.t_PollRetransmit = LTE_T_PollRetransmit_ms15;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollPDU = LTE_PollPDU_p8;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.pollByte = LTE_PollByte_kB1000;
  SRB2_rlc_config->choice.explicitValue.choice.am.ul_AM_RLC.maxRetxThreshold = LTE_UL_AM_RLC__maxRetxThreshold_t32;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  SRB2_rlc_config->choice.explicitValue.choice.am.dl_AM_RLC.t_StatusProhibit = LTE_T_StatusProhibit_ms10;
  SRB2_lchan_config = CALLOC(1, sizeof(*SRB2_lchan_config));
  SRB2_config->logicalChannelConfig = SRB2_lchan_config;
  SRB2_lchan_config->present = LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue;
  SRB2_ul_SpecificParameters = CALLOC(1, sizeof(*SRB2_ul_SpecificParameters));
  SRB2_ul_SpecificParameters->priority = 1;
  SRB2_ul_SpecificParameters->prioritisedBitRate =
    LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  SRB2_ul_SpecificParameters->bucketSizeDuration =
    LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for CCCH and DCCH is 0 as defined in 36331
  logicalchannelgroup = CALLOC(1, sizeof(long));
  *logicalchannelgroup = 0;
  SRB2_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup;
  SRB2_lchan_config->choice.explicitValue.ul_SpecificParameters = SRB2_ul_SpecificParameters;
  ASN_SEQUENCE_ADD(&SRB_configList->list, SRB2_config);
  ASN_SEQUENCE_ADD(&(*SRB_configList2)->list, SRB2_config);
  // Configure target eNB DRB
  DRB_configList2 = CALLOC(1, sizeof(*DRB_configList2));
  /// DRB
  DRB_config = CALLOC(1, sizeof(*DRB_config));
  //DRB_config->drb_Identity = (LTE_DRB_Identity_t) 1; //allowed values 1..32
  // NN: this is the 1st DRB for this ue, so set it to 1
  DRB_config->drb_Identity = (LTE_DRB_Identity_t) 1;  // (ue_mod_idP+1); //allowed values 1..32
  DRB_config->logicalChannelIdentity = CALLOC(1, sizeof(long));
  *(DRB_config->logicalChannelIdentity) = (long)3;
  DRB_rlc_config = CALLOC(1, sizeof(*DRB_rlc_config));
  DRB_config->rlc_Config = DRB_rlc_config;
  DRB_rlc_config->present = LTE_RLC_Config_PR_um_Bi_Directional;
  DRB_rlc_config->choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = LTE_SN_FieldLength_size10;
  DRB_rlc_config->choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = LTE_T_Reordering_ms35;
  DRB_pdcp_config = CALLOC(1, sizeof(*DRB_pdcp_config));
  DRB_config->pdcp_Config = DRB_pdcp_config;
  DRB_pdcp_config->discardTimer = NULL;
  DRB_pdcp_config->rlc_AM = NULL;
  PDCP_rlc_UM = CALLOC(1, sizeof(*PDCP_rlc_UM));
  DRB_pdcp_config->rlc_UM = PDCP_rlc_UM;
  PDCP_rlc_UM->pdcp_SN_Size = LTE_PDCP_Config__rlc_UM__pdcp_SN_Size_len12bits;
  DRB_pdcp_config->headerCompression.present = LTE_PDCP_Config__headerCompression_PR_notUsed;
  DRB_lchan_config = CALLOC(1, sizeof(*DRB_lchan_config));
  DRB_config->logicalChannelConfig = DRB_lchan_config;
  DRB_ul_SpecificParameters = CALLOC(1, sizeof(*DRB_ul_SpecificParameters));
  DRB_lchan_config->ul_SpecificParameters = DRB_ul_SpecificParameters;
  DRB_ul_SpecificParameters->priority = 2;    // lower priority than srb1, srb2
  DRB_ul_SpecificParameters->prioritisedBitRate =
    LTE_LogicalChannelConfig__ul_SpecificParameters__prioritisedBitRate_infinity;
  DRB_ul_SpecificParameters->bucketSizeDuration =
    LTE_LogicalChannelConfig__ul_SpecificParameters__bucketSizeDuration_ms50;
  // LCG for DTCH can take the value from 1 to 3 as defined in 36331: normally controlled by upper layers (like RRM)
  logicalchannelgroup_drb = CALLOC(1, sizeof(long));
  *logicalchannelgroup_drb = 1;
  DRB_ul_SpecificParameters->logicalChannelGroup = logicalchannelgroup_drb;
  ASN_SEQUENCE_ADD(&(*DRB_configList2)->list, DRB_config);
  mac_MainConfig = CALLOC(1, sizeof(*mac_MainConfig));
  ue_context_pP->ue_context.mac_MainConfig = mac_MainConfig;
  mac_MainConfig->ul_SCH_Config = CALLOC(1, sizeof(*mac_MainConfig->ul_SCH_Config));
  maxHARQ_Tx = CALLOC(1, sizeof(long));
  *maxHARQ_Tx = LTE_MAC_MainConfig__ul_SCH_Config__maxHARQ_Tx_n5;
  mac_MainConfig->ul_SCH_Config->maxHARQ_Tx = maxHARQ_Tx;
  periodicBSR_Timer = CALLOC(1, sizeof(long));
  *periodicBSR_Timer = LTE_PeriodicBSR_Timer_r12_sf64;
  mac_MainConfig->ul_SCH_Config->periodicBSR_Timer = periodicBSR_Timer;
  mac_MainConfig->ul_SCH_Config->retxBSR_Timer = LTE_RetxBSR_Timer_r12_sf320;
  mac_MainConfig->ul_SCH_Config->ttiBundling = 0; // FALSE
  mac_MainConfig->drx_Config = NULL;
  mac_MainConfig->phr_Config = CALLOC(1, sizeof(*mac_MainConfig->phr_Config));
  mac_MainConfig->phr_Config->present = LTE_MAC_MainConfig__phr_Config_PR_setup;
  mac_MainConfig->phr_Config->choice.setup.periodicPHR_Timer = LTE_MAC_MainConfig__phr_Config__setup__periodicPHR_Timer_sf20; // sf20 = 20 subframes
  mac_MainConfig->phr_Config->choice.setup.prohibitPHR_Timer = LTE_MAC_MainConfig__phr_Config__setup__prohibitPHR_Timer_sf20; // sf20 = 20 subframes
  mac_MainConfig->phr_Config->choice.setup.dl_PathlossChange = LTE_MAC_MainConfig__phr_Config__setup__dl_PathlossChange_dB1;  // Value dB1 =1 dB, dB3 = 3 dB
  sr_ProhibitTimer_r9 = CALLOC(1, sizeof(long));
  *sr_ProhibitTimer_r9 = 0;   // SR tx on PUCCH, Value in number of SR period(s). Value 0 = no timer for SR, Value 2= 2*SR
  mac_MainConfig->ext1 = CALLOC(1, sizeof(struct LTE_MAC_MainConfig__ext1));
  mac_MainConfig->ext1->sr_ProhibitTimer_r9 = sr_ProhibitTimer_r9;
  // Measurement ID list
  MeasId_list = CALLOC(1, sizeof(*MeasId_list));
  memset((void *)MeasId_list, 0, sizeof(*MeasId_list));
  MeasId0 = CALLOC(1, sizeof(*MeasId0));
  MeasId0->measId = 1;
  MeasId0->measObjectId = 1;
  MeasId0->reportConfigId = 1;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId0);
  MeasId1 = CALLOC(1, sizeof(*MeasId1));
  MeasId1->measId = 2;
  MeasId1->measObjectId = 1;
  MeasId1->reportConfigId = 2;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId1);
  MeasId2 = CALLOC(1, sizeof(*MeasId2));
  MeasId2->measId = 3;
  MeasId2->measObjectId = 1;
  MeasId2->reportConfigId = 3;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId2);
  MeasId3 = CALLOC(1, sizeof(*MeasId3));
  MeasId3->measId = 4;
  MeasId3->measObjectId = 1;
  MeasId3->reportConfigId = 4;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId3);
  MeasId4 = CALLOC(1, sizeof(*MeasId4));
  MeasId4->measId = 5;
  MeasId4->measObjectId = 1;
  MeasId4->reportConfigId = 5;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId4);
  MeasId5 = CALLOC(1, sizeof(*MeasId5));
  MeasId5->measId = 6;
  MeasId5->measObjectId = 1;
  MeasId5->reportConfigId = 6;
  ASN_SEQUENCE_ADD(&MeasId_list->list, MeasId5);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measIdToAddModList = MeasId_list;
  // Add one EUTRA Measurement Object
  MeasObj_list = CALLOC(1, sizeof(*MeasObj_list));
  memset((void *)MeasObj_list, 0, sizeof(*MeasObj_list));
  // Configure MeasObject
  MeasObj = CALLOC(1, sizeof(*MeasObj));
  memset((void *)MeasObj, 0, sizeof(*MeasObj));
  MeasObj->measObjectId = 1;
  MeasObj->measObject.present = LTE_MeasObjectToAddMod__measObject_PR_measObjectEUTRA;
  MeasObj->measObject.choice.measObjectEUTRA.carrierFreq =
    to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->configuration.eutra_band[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.downlink_frequency[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.N_RB_DL[0]);
  MeasObj->measObject.choice.measObjectEUTRA.allowedMeasBandwidth = LTE_AllowedMeasBandwidth_mbw25;
  MeasObj->measObject.choice.measObjectEUTRA.presenceAntennaPort1 = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf = CALLOC(1, sizeof(uint8_t));
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.buf[0] = 0;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.size = 1;
  MeasObj->measObject.choice.measObjectEUTRA.neighCellConfig.bits_unused = 6;
  MeasObj->measObject.choice.measObjectEUTRA.offsetFreq = NULL;   // Default is 15 or 0dB

  if (RC.rrc[ctxt_pP->module_id]->num_neigh_cells > 0) {
    MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList =
      (LTE_CellsToAddModList_t *) CALLOC(1, sizeof(*CellsToAddModList));
    CellsToAddModList = MeasObj->measObject.choice.measObjectEUTRA.cellsToAddModList;
  }

  if (!ue_context_pP->ue_context.measurement_info) {
    ue_context_pP->ue_context.measurement_info = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info)));
  }

  //TODO: Assign proper values
  ue_context_pP->ue_context.measurement_info->offsetFreq = 0;
  ue_context_pP->ue_context.measurement_info->cellIndividualOffset[0] = LTE_Q_OffsetRange_dB0;

  /* TODO: Extend to multiple carriers */
  // Add adjacent cell lists (max 6 per eNB)
  for (i = 0; i < RC.rrc[ctxt_pP->module_id]->num_neigh_cells; i++) {
    CellToAdd = (LTE_CellsToAddMod_t *) CALLOC(1, sizeof(*CellToAdd));
    CellToAdd->cellIndex = i + 1;
    CellToAdd->physCellId = RC.rrc[ctxt_pP->module_id]->neigh_cells_id[i][0];//get_adjacent_cell_id(ctxt_pP->module_id, i);
    CellToAdd->cellIndividualOffset = LTE_Q_OffsetRange_dB0;
    ue_context_pP->ue_context.measurement_info->cellIndividualOffset[i+1] = CellToAdd->cellIndividualOffset;
    ASN_SEQUENCE_ADD(&CellsToAddModList->list, CellToAdd);
  }

  ASN_SEQUENCE_ADD(&MeasObj_list->list, MeasObj);
  //  LTE_RRCConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->measObjectToAddModList = MeasObj_list;

  if (!ue_context_pP->ue_context.measurement_info->events) {
    ue_context_pP->ue_context.measurement_info->events = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info->events)));
  }

  // Report Configurations for periodical, A1-A5 events
  ReportConfig_list = CALLOC(1, sizeof(*ReportConfig_list));
  ReportConfig_per = CALLOC(1, sizeof(*ReportConfig_per));
  ReportConfig_A1 = CALLOC(1, sizeof(*ReportConfig_A1));
  ReportConfig_A2 = CALLOC(1, sizeof(*ReportConfig_A2));
  ReportConfig_A3 = CALLOC(1, sizeof(*ReportConfig_A3));
  ReportConfig_A4 = CALLOC(1, sizeof(*ReportConfig_A4));
  ReportConfig_A5 = CALLOC(1, sizeof(*ReportConfig_A5));
  ReportConfig_per->reportConfigId = 1;
  ReportConfig_per->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_periodical;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerType.choice.periodical.purpose =
    LTE_ReportConfigEUTRA__triggerType__periodical__purpose_reportStrongestCells;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_per->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_per);
  ReportConfig_A1->reportConfigId = 2;
  ReportConfig_A1->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA1;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA1.
  a1_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.triggerQuantity = LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A1->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A1);
  //  if (ho_state == 1 /*HO_MEASUREMENT */ ) {
  LOG_I(RRC, "[eNB %d] frame %d: requesting A2, A3, A4, and A5 event reporting\n",
        ctxt_pP->module_id, ctxt_pP->frame);
  ReportConfig_A2->reportConfigId = 3;
  ReportConfig_A2->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA2.a2_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A2->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A2);
  ReportConfig_A3->reportConfigId = 4;
  ReportConfig_A3->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA3;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset = 0;   //10;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA3.reportOnLeave = 1;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis = 0; // FIXME ...hysteresis is of type long!
  ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger =
    LTE_TimeToTrigger_ms40;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A3);
  ReportConfig_A4->reportConfigId = 5;
  ReportConfig_A4->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA4;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA4.a4_Threshold.choice.threshold_RSRP = 10;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A4->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A4);
  ReportConfig_A5->reportConfigId = 6;
  ReportConfig_A5->reportConfig.present = LTE_ReportConfigToAddMod__reportConfig_PR_reportConfigEUTRA;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.present =
    LTE_ReportConfigEUTRA__triggerType_PR_event;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.present =
    LTE_ReportConfigEUTRA__triggerType__event__eventId_PR_eventA5;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.present = LTE_ThresholdEUTRA_PR_threshold_RSRP;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold1.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.
  eventA5.a5_Threshold2.choice.threshold_RSRP = 10;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.triggerQuantity =
    LTE_ReportConfigEUTRA__triggerQuantity_rsrp;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportQuantity = LTE_ReportConfigEUTRA__reportQuantity_both;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.maxReportCells = 2;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportInterval = LTE_ReportInterval_ms120;
  ReportConfig_A5->reportConfig.choice.reportConfigEUTRA.reportAmount = LTE_ReportConfigEUTRA__reportAmount_infinity;
  ASN_SEQUENCE_ADD(&ReportConfig_list->list, ReportConfig_A5);
  //  rrcConnectionReconfiguration->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r8.measConfig->reportConfigToAddModList = ReportConfig_list;

  /* A3 event update */
  if (!ue_context_pP->ue_context.measurement_info->events->a3_event) {
    ue_context_pP->ue_context.measurement_info->events->a3_event = CALLOC(1,sizeof(*(ue_context_pP->ue_context.measurement_info->events->a3_event)));
  }

  ue_context_pP->ue_context.measurement_info->events->a3_event->a3_offset = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.a3_Offset;
  ue_context_pP->ue_context.measurement_info->events->a3_event->reportOnLeave = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.eventId.choice.eventA3.reportOnLeave;
  ue_context_pP->ue_context.measurement_info->events->a3_event->hysteresis = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.hysteresis;
  ue_context_pP->ue_context.measurement_info->events->a3_event->timeToTrigger = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.triggerType.choice.event.timeToTrigger;
  ue_context_pP->ue_context.measurement_info->events->a3_event->maxReportCells = ReportConfig_A3->reportConfig.choice.reportConfigEUTRA.maxReportCells;
#if 0
  /* TODO: set a proper value.
   * 20 means: UE does not report if RSRP of serving cell is higher
   * than -120 dB (see 36.331 5.5.3.1).
   * This is too low for the X2 handover experiment.
   */
  rsrp = CALLOC(1, sizeof(RSRP_Range_t));
  *rsrp = 20;
#endif
  Sparams = CALLOC(1, sizeof(*Sparams));
  Sparams->present = LTE_MeasConfig__speedStatePars_PR_setup;
  Sparams->choice.setup.timeToTrigger_SF.sf_High = LTE_SpeedStateScaleFactors__sf_Medium_oDot75;
  Sparams->choice.setup.timeToTrigger_SF.sf_Medium = LTE_SpeedStateScaleFactors__sf_High_oDot5;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeHigh = 10;
  Sparams->choice.setup.mobilityStateParameters.n_CellChangeMedium = 5;
  Sparams->choice.setup.mobilityStateParameters.t_Evaluation = LTE_MobilityStateParameters__t_Evaluation_s60;
  Sparams->choice.setup.mobilityStateParameters.t_HystNormal = LTE_MobilityStateParameters__t_HystNormal_s120;
  quantityConfig = CALLOC(1, sizeof(*quantityConfig));
  memset((void *)quantityConfig, 0, sizeof(*quantityConfig));
  quantityConfig->quantityConfigEUTRA = CALLOC(1, sizeof(struct LTE_QuantityConfigEUTRA));
  memset((void *)quantityConfig->quantityConfigEUTRA, 0, sizeof(*quantityConfig->quantityConfigEUTRA));
  quantityConfig->quantityConfigCDMA2000 = NULL;
  quantityConfig->quantityConfigGERAN = NULL;
  quantityConfig->quantityConfigUTRA = NULL;
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP)));
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ =
    CALLOC(1, sizeof(*(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ)));
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = LTE_FilterCoefficient_fc4;
  *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = LTE_FilterCoefficient_fc4;
  ue_context_pP->ue_context.measurement_info->filterCoefficientRSRP = *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP;
  ue_context_pP->ue_context.measurement_info->filterCoefficientRSRQ = *quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ;
  /* mobilityinfo  */
  mobilityInfo = ue_context_pP->ue_context.mobilityInfo;

  if (mobilityInfo) {
    free(mobilityInfo);
  }

  mobilityInfo = CALLOC(1, sizeof(*mobilityInfo));
  memset((void *)mobilityInfo, 0, sizeof(*mobilityInfo));
  mobilityInfo->targetPhysCellId = RC.rrc[ctxt_pP->module_id]->carrier[0].physCellId;
  //(PhysCellId_t) two_tier_hexagonal_cellIds[ue_context_pP->ue_context.handover_info->modid_t];
  LOG_D(RRC, "[eNB %d] Frame %d: handover preparation: targetPhysCellId: %ld mod_id: %d ue: %x \n",
        ctxt_pP->module_id,
        ctxt_pP->frame,
        mobilityInfo->targetPhysCellId,
        ctxt_pP->module_id, // get_adjacent_cell_mod_id(mobilityInfo->targetPhysCellId),
        ue_context_pP->ue_context.rnti);
  mobilityInfo->additionalSpectrumEmission = CALLOC(1, sizeof(*mobilityInfo->additionalSpectrumEmission));
  *mobilityInfo->additionalSpectrumEmission = (LTE_AdditionalSpectrumEmission_t) 1;  //Check this value!
  mobilityInfo->t304 = LTE_MobilityControlInfo__t304_ms200;    // need to configure an appropriate value here
  // New UE Identity (C-RNTI) to identify an UE uniquely in a cell
  mobilityInfo->newUE_Identity.size = 2;
  mobilityInfo->newUE_Identity.bits_unused = 0;
  mobilityInfo->newUE_Identity.buf = rv;
  mobilityInfo->newUE_Identity.buf[0] = rv[0];
  mobilityInfo->newUE_Identity.buf[1] = rv[1];
  //memset((void *)&mobilityInfo->radioResourceConfigCommon,(void *)&rrc_inst->sib2->radioResourceConfigCommon,sizeof(RadioResourceConfigCommon_t));
  //memset((void *)&mobilityInfo->radioResourceConfigCommon,0,sizeof(RadioResourceConfigCommon_t));
  // Configuring radioResourceConfigCommon
  mobilityInfo->radioResourceConfigCommon.rach_ConfigCommon =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.rach_ConfigCommon));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.rach_ConfigCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.rach_ConfigCommon, sizeof(LTE_RACH_ConfigCommon_t));
  mobilityInfo->radioResourceConfigCommon.prach_Config.prach_ConfigInfo =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.prach_Config.prach_ConfigInfo));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.prach_Config.prach_ConfigInfo,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.prach_Config.prach_ConfigInfo,
         sizeof(LTE_PRACH_ConfigInfo_t));
  mobilityInfo->radioResourceConfigCommon.prach_Config.rootSequenceIndex =
    rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.prach_Config.rootSequenceIndex;
  mobilityInfo->radioResourceConfigCommon.pdsch_ConfigCommon =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.pdsch_ConfigCommon));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.pdsch_ConfigCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.pdsch_ConfigCommon, sizeof(LTE_PDSCH_ConfigCommon_t));
  memcpy((void *)&mobilityInfo->radioResourceConfigCommon.pusch_ConfigCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.pusch_ConfigCommon, sizeof(LTE_PUSCH_ConfigCommon_t));
  mobilityInfo->radioResourceConfigCommon.phich_Config = NULL;
  mobilityInfo->radioResourceConfigCommon.pucch_ConfigCommon =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.pucch_ConfigCommon));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.pucch_ConfigCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.pucch_ConfigCommon, sizeof(LTE_PUCCH_ConfigCommon_t));
  mobilityInfo->radioResourceConfigCommon.soundingRS_UL_ConfigCommon =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.soundingRS_UL_ConfigCommon));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.soundingRS_UL_ConfigCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.soundingRS_UL_ConfigCommon,
         sizeof(LTE_SoundingRS_UL_ConfigCommon_t));
  mobilityInfo->radioResourceConfigCommon.uplinkPowerControlCommon =
    CALLOC(1, sizeof(*mobilityInfo->radioResourceConfigCommon.uplinkPowerControlCommon));
  memcpy((void *)mobilityInfo->radioResourceConfigCommon.uplinkPowerControlCommon,
         (void *)&rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.uplinkPowerControlCommon,
         sizeof(LTE_UplinkPowerControlCommon_t));
  mobilityInfo->radioResourceConfigCommon.antennaInfoCommon = NULL;
  mobilityInfo->radioResourceConfigCommon.p_Max = NULL;   // CALLOC(1,sizeof(*mobilityInfo->radioResourceConfigCommon.p_Max));
  //memcpy((void *)mobilityInfo->radioResourceConfigCommon.p_Max,(void *)rrc_inst->sib1->p_Max,sizeof(P_Max_t));
  mobilityInfo->radioResourceConfigCommon.tdd_Config = NULL;  //CALLOC(1,sizeof(TDD_Config_t));
  //memcpy((void *)mobilityInfo->radioResourceConfigCommon.tdd_Config,(void *)rrc_inst->sib1->tdd_Config,sizeof(TDD_Config_t));
  mobilityInfo->radioResourceConfigCommon.ul_CyclicPrefixLength =
    rrc_inst->carrier[0] /* CROUX TBC */.sib2->radioResourceConfigCommon.ul_CyclicPrefixLength;
  //End of configuration of radioResourceConfigCommon
  mobilityInfo->carrierFreq = CALLOC(1, sizeof(*mobilityInfo->carrierFreq));  //CALLOC(1,sizeof(CarrierFreqEUTRA_t)); 36090
  mobilityInfo->carrierFreq->dl_CarrierFreq =
    to_earfcn_DL(RC.rrc[ctxt_pP->module_id]->configuration.eutra_band[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.downlink_frequency[0],
                 RC.rrc[ctxt_pP->module_id]->configuration.N_RB_DL[0]);
  mobilityInfo->carrierFreq->ul_CarrierFreq = NULL;
  mobilityInfo->carrierBandwidth = CALLOC(1, sizeof(
      *mobilityInfo->carrierBandwidth));    //CALLOC(1,sizeof(struct CarrierBandwidthEUTRA));  AllowedMeasBandwidth_mbw25
  mobilityInfo->carrierBandwidth->dl_Bandwidth = LTE_CarrierBandwidthEUTRA__dl_Bandwidth_n25;
  mobilityInfo->carrierBandwidth->ul_Bandwidth = NULL;
  mobilityInfo->rach_ConfigDedicated = NULL;
  ue_context_pP->ue_context.mobilityInfo = mobilityInfo;
#if 0
  LOG_I(RRC,
        "[eNB %d] Frame %d: potential handover preparation: store the information in an intermediate structure in case of failure\n",
        ctxt_pP->module_id, ctxt_pP->frame);
  // store the information in an intermediate structure for Hanodver management
  //rrc_inst->handover_info.as_config.sourceRadioResourceConfig.srb_ToAddModList = CALLOC(1,sizeof());
  ue_context_pP->ue_context.handover_info = CALLOC(1, sizeof(*(ue_context_pP->ue_context.handover_info)));
  //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.srb_ToAddModList,(void *)SRB_list,sizeof(SRB_ToAddModList_t));
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.srb_ToAddModList = *SRB_configList2;
  //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.drb_ToAddModList,(void *)DRB_list,sizeof(DRB_ToAddModList_t));
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.drb_ToAddModList = *DRB_configList;
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.drb_ToReleaseList = NULL;
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig =
    CALLOC(1, sizeof(*ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig));
  memcpy((void *)ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.mac_MainConfig,
         (void *)mac_MainConfig, sizeof(MAC_MainConfig_t));
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.physicalConfigDedicated =
    CALLOC(1, sizeof(PhysicalConfigDedicated_t));
  memcpy((void *)ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.physicalConfigDedicated,
         (void *)ue_context_pP->ue_context.physicalConfigDedicated, sizeof(PhysicalConfigDedicated_t));
  ue_context_pP->ue_context.handover_info->as_config.sourceRadioResourceConfig.sps_Config = NULL;
  //memcpy((void *)rrc_inst->handover_info[ue_mod_idP]->as_config.sourceRadioResourceConfig.sps_Config,(void *)rrc_inst->sps_Config[ue_mod_idP],sizeof(SPS_Config_t));
#endif
  //  }
  securityConfigHO = CALLOC(1, sizeof(*securityConfigHO));
  memset((void *)securityConfigHO, 0, sizeof(*securityConfigHO));
  securityConfigHO->handoverType.present = LTE_SecurityConfigHO__handoverType_PR_intraLTE;
  securityConfigHO->handoverType.choice.intraLTE.securityAlgorithmConfig = NULL; /* TODO: to be checked */
  securityConfigHO->handoverType.choice.intraLTE.keyChangeIndicator = 0;
  securityConfigHO->handoverType.choice.intraLTE.nextHopChainingCount = 0;
  /* Initialize NAS list */
  dedicatedInfoNASList = CALLOC(1, sizeof(struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList));

  /* Add all NAS PDUs to the list */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      dedicatedInfoNas = CALLOC(1, sizeof(LTE_DedicatedInfoNAS_t));
      memset(dedicatedInfoNas, 0, sizeof(OCTET_STRING_t));
      OCTET_STRING_fromBuf(dedicatedInfoNas,
                           (char *)ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer,
                           ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length);
      ASN_SEQUENCE_ADD(&dedicatedInfoNASList->list, dedicatedInfoNas);
    }

    /* TODO parameters yet to process ... */
    {
      //      ue_context_pP->ue_context.e_rab[i].param.qos;
      //      ue_context_pP->ue_context.e_rab[i].param.sgw_addr;
      //      ue_context_pP->ue_context.e_rab[i].param.gtp_teid;
    }
    /* TODO should test if e RAB are Ok before! */
    ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_DONE;
    LOG_D(RRC, "setting the status for the default DRB (index %d) to (%d,%s)\n",
          i, ue_context_pP->ue_context.e_rab[i].status, "E_RAB_STATUS_DONE");
  }

  /* If list is empty free the list and reset the address */
  if (dedicatedInfoNASList->list.count == 0) {
    free(dedicatedInfoNASList);
    dedicatedInfoNASList = NULL;
  }

  measurements_enabled = RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2 ||
                         RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_measurement_reports;
  memset(buffer, 0, RRC_BUF_SIZE);
  char rrc_buf[1000 /* arbitrary, should be big enough, has to be less than size of return buf by a few bits/bytes */];
  int rrc_size;
  rrc_size = do_RRCConnectionReconfiguration(ctxt_pP,
             (unsigned char *)rrc_buf,
             xid,   //Transaction_id,
             NULL, // SRB_configList
             NULL,
             NULL,  // DRB2_list,
             (struct LTE_SPS_Config *)NULL,   // *sps_Config,
             (struct LTE_PhysicalConfigDedicated *)*physicalConfigDedicated,
             //#ifdef EXMIMO_IOT
             //                                         NULL, NULL, NULL,NULL,
             //#else
             measurements_enabled ? (LTE_MeasObjectToAddModList_t *)MeasObj_list : NULL,
             measurements_enabled ? (LTE_ReportConfigToAddModList_t *)ReportConfig_list : NULL,
             measurements_enabled ? (LTE_QuantityConfig_t *)quantityConfig : NULL,
             measurements_enabled ? (LTE_MeasIdToAddModList_t *)MeasId_list : NULL,
             //#endif
             (LTE_MAC_MainConfig_t *)mac_MainConfig,
             (LTE_MeasGapConfig_t *)NULL,
             (LTE_MobilityControlInfo_t *)mobilityInfo,
             (LTE_SecurityConfigHO_t *)securityConfigHO,
             (struct LTE_MeasConfig__speedStatePars *)Sparams,
             (LTE_RSRP_Range_t *)rsrp,
             (LTE_C_RNTI_t *)cba_RNTI,
             (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)dedicatedInfoNASList,
             (LTE_SL_CommConfig_r12_t *)NULL,
             (LTE_SL_DiscConfig_r12_t *)NULL,
             (LTE_SCellToAddMod_r10_t *)NULL
                                            );

  if (rrc_size <= 0) {
    printf("%s:%d: fatal\n", __FILE__, __LINE__);
    abort();
  }

  char *ho_buf = (char *)buffer;
  int ho_size;
  ho_size = do_HandoverCommand(
              ho_buf, 1024 /* TODO: this is the value found in struct x2ap_handover_req_ack_s for array rrc_buffer */,
              rrc_buf,
              rrc_size);
  *_size = size = ho_size;
  LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,size,
              "[MSG] RRC Connection Reconfiguration handover\n");

  /* Free all NAS PDUs */
  for (i = 0; i < ue_context_pP->ue_context.nb_of_e_rabs; i++) {
    if (ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer != NULL) {
      /* Free the NAS PDU buffer and invalidate it */
      free(ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer);
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
    }
  }

  LOG_I(RRC,
        "[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate RRCConnectionReconfiguration handover (bytes %d, UE id %x)\n",
        ctxt_pP->module_id, ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  LOG_D(RRC,
        "[FRAME %05d][RRC_eNB][MOD %u][][--- PDCP_DATA_REQ/%d Bytes (rrcConnectionReconfiguration handover to UE %x MUI %d) --->][PDCP][MOD %u][RB %u]\n",
        ctxt_pP->frame, ctxt_pP->module_id, size, ue_context_pP->ue_context.rnti, rrc_eNB_mui, ctxt_pP->module_id, DCCH);
  MSC_LOG_TX_MESSAGE(
    MSC_RRC_ENB,
    MSC_RRC_UE,
    buffer,
    size,
    MSC_AS_TIME_FMT" rrcConnectionReconfiguration handover UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ue_context_pP->ue_context.rnti,
    rrc_eNB_mui,
    size);

  /* Refresh SRBs/DRBs */
  rrc_pdcp_config_asn1_req(ctxt_pP,
                          *SRB_configList2, // NULL,
                          *DRB_configList,
                          NULL,
                          0xff, // already configured during the securitymodecommand
                          NULL,
                          NULL,
                          NULL
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
                          , (LTE_PMCH_InfoList_r9_t *) NULL
#endif
                          , NULL);

  /* Refresh SRBs/DRBs */
  if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    rrc_rlc_config_asn1_req(ctxt_pP,
                            *SRB_configList2, // NULL,
                            *DRB_configList,
                            NULL
#if (LTE_RRC_VERSION >= MAKE_VERSION(9, 0, 0))
                            , (LTE_PMCH_InfoList_r9_t *) NULL,
                            0,
                            0
#endif
                            );
  }

  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRQ = NULL;
  free(quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP);
  quantityConfig->quantityConfigEUTRA->filterCoefficientRSRP = NULL;
  free(quantityConfig->quantityConfigEUTRA);
  quantityConfig->quantityConfigEUTRA = NULL;
  free(quantityConfig);
  quantityConfig = NULL;
  free(securityConfigHO);
  securityConfigHO = NULL;
  free(Sparams);
  Sparams = NULL;
}

void
rrc_eNB_configure_rbs_handover(struct rrc_eNB_ue_context_s *ue_context_p, protocol_ctxt_t *const ctxt_pP) {
  uint16_t                            Idx;
  Idx = DCCH;
#if 1
  // Activate the radio bearers
  // SRB1
  ue_context_p->ue_context.Srb1.Active = 1;
  ue_context_p->ue_context.Srb1.Srb_info.Srb_id = Idx;
  memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[0], &DCCH_LCHAN_DESC, LCHAN_DESC_SIZE);
  memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[1], &DCCH_LCHAN_DESC, LCHAN_DESC_SIZE);
  // SRB2
  ue_context_p->ue_context.Srb2.Active = 1;
  ue_context_p->ue_context.Srb2.Srb_info.Srb_id = Idx;
  memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[0], &DCCH_LCHAN_DESC, LCHAN_DESC_SIZE);
  memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[1], &DCCH_LCHAN_DESC, LCHAN_DESC_SIZE);
#endif
  LOG_I(RRC, "[eNB %d] CALLING RLC CONFIG SRB1 (rbid %d) for UE %x\n",
        ctxt_pP->module_id, Idx, ue_context_p->ue_context.rnti);
  // Configure PDCP/RLC for the target
  rrc_pdcp_config_asn1_req(ctxt_pP,
                           ue_context_p->ue_context.SRB_configList,
                           (LTE_DRB_ToAddModList_t *) NULL,
                           (LTE_DRB_ToReleaseList_t *) NULL,
                           0xff,
                           NULL,
                           NULL,
                           NULL,
                           (LTE_PMCH_InfoList_r9_t *) NULL, NULL);
  if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    rrc_rlc_config_asn1_req(ctxt_pP,
                            ue_context_p->ue_context.SRB_configList,
                            (LTE_DRB_ToAddModList_t *) NULL,
                            (LTE_DRB_ToReleaseList_t *) NULL,
                            (LTE_PMCH_InfoList_r9_t *) NULL, 0, 0
                           );
  }

  if (EPC_MODE_ENABLED) {
    rrc_eNB_process_security (
      ctxt_pP,
      ue_context_p,
      &ue_context_p->ue_context.security_capabilities);
    process_eNB_security_key (
      ctxt_pP,
      ue_context_p,
      ue_context_p->ue_context.kenb);
    rrc_pdcp_config_security(
      ctxt_pP,
      ue_context_p,
      FALSE);
  }

  // Add a new user (called during the HO procedure)
  LOG_I(RRC, "rrc_eNB_target_add_ue_handover module_id %d rnti %d\n", ctxt_pP->module_id, ctxt_pP->rnti);
  // Configure MAC for the target
  rrc_mac_config_req_eNB(
    ctxt_pP->module_id,
    ue_context_p->ue_context.primaryCC_id,
    0,0,0,0,0, 0,
    ue_context_p->ue_context.rnti,
    (LTE_BCCH_BCH_Message_t *) NULL,
    (LTE_RadioResourceConfigCommonSIB_t *) NULL,
    (LTE_RadioResourceConfigCommonSIB_t *) NULL,
    ue_context_p->ue_context.physicalConfigDedicated,
    (LTE_SCellToAddMod_r10_t *)NULL,
    (LTE_MeasObjectToAddMod_t **) NULL,
    ue_context_p->ue_context.mac_MainConfig,
    1,
    NULL,//SRB1_logicalChannelConfig,
    ue_context_p->ue_context.measGapConfig,
    (LTE_TDD_Config_t *) NULL,
    (LTE_MobilityControlInfo_t *) ue_context_p->ue_context.mobilityInfo,
    (LTE_SchedulingInfoList_t *) NULL,
    0,
    NULL,
    NULL,
    (LTE_MBSFN_SubframeConfigList_t *) NULL, 0,
    (LTE_MBSFN_AreaInfoList_r9_t *) NULL, (LTE_PMCH_InfoList_r9_t *) NULL,
    (LTE_SystemInformationBlockType1_v1310_IEs_t *)NULL,
    0,
    (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
    (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
    (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
    (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
    (LTE_MBSFN_AreaInfoList_r9_t *) NULL
  );
}

//-----------------------------------------------------------------------------
/*
* Process the RRC Connection Reconfiguration Complete from the UE
*/
void
rrc_eNB_process_RRCConnectionReconfigurationComplete(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t  *ue_context_pP,
  const uint8_t xid
)
//-----------------------------------------------------------------------------
{
  int                                 drb_id;
  int                                 oip_ifup = 0;
  int                                 dest_ip_offset = 0;
  uint8_t                            *kRRCenc = NULL;
  uint8_t                            *kRRCint = NULL;
  uint8_t                            *kUPenc = NULL;
  LTE_DRB_ToAddModList_t             *DRB_configList = ue_context_pP->ue_context.DRB_configList2[xid];
  LTE_SRB_ToAddModList_t             *SRB_configList = ue_context_pP->ue_context.SRB_configList2[xid];
  LTE_DRB_ToReleaseList_t            *DRB_Release_configList2 = ue_context_pP->ue_context.DRB_Release_configList2[xid];
  LTE_DRB_Identity_t                 *drb_id_p      = NULL;
  ue_context_pP->ue_context.ue_reestablishment_timer = 0;
  ue_context_pP->ue_context.ue_rrc_inactivity_timer = 1; // reset rrc inactivity timer
  /* CDRX: activated when RRC Connection Reconfiguration Complete is received */
  rnti_t rnti = ue_context_pP->ue_id_rnti;
  module_id_t module_id = ctxt_pP->module_id;

#ifdef ENABLE_RAN_SLICING
  eventTrigger CUEventTriggerToRic;
  ueStatusInd *ueAttachInd;
#endif

  if (NODE_IS_MONOLITHIC(RC.rrc[module_id]->node_type)) {
    int UE_id_mac = find_UE_id(module_id, rnti);

    if (UE_id_mac == -1) {
      LOG_E(RRC, "Can't find UE_id(MAC) of UE rnti %x\n", rnti);
      return;
    }

    UE_sched_ctrl_t *UE_scheduling_control = &(RC.mac[module_id]->UE_info.UE_sched_ctrl[UE_id_mac]);

    if (UE_scheduling_control->cdrx_waiting_ack == TRUE) {
      UE_scheduling_control->cdrx_waiting_ack = FALSE;
      UE_scheduling_control->cdrx_configured = TRUE; // Set to TRUE when RRC Connection Reconfiguration is received
      LOG_I(RRC, "CDRX configuration activated after RRC Connection Reconfiguration Complete reception\n");
    }
  }

  /* End of CDRX processing */
  T(T_ENB_RRC_CONNECTION_RECONFIGURATION_COMPLETE,
    T_INT(ctxt_pP->module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(ctxt_pP->rnti));

  /* Derive the keys from kenb */
  if (DRB_configList != NULL) {
    derive_key_up_enc(ue_context_pP->ue_context.ciphering_algorithm,
                      ue_context_pP->ue_context.kenb,
                      &kUPenc);
  }

  derive_key_rrc_enc(ue_context_pP->ue_context.ciphering_algorithm,
                     ue_context_pP->ue_context.kenb,
                     &kRRCenc);
  derive_key_rrc_int(ue_context_pP->ue_context.integrity_algorithm,
                     ue_context_pP->ue_context.kenb,
                     &kRRCint);
  /* Refresh SRBs/DRBs */
  MSC_LOG_TX_MESSAGE(MSC_RRC_ENB, MSC_PDCP_ENB, NULL, 0, MSC_AS_TIME_FMT" CONFIG_REQ UE %x DRB (security unchanged)",
                     MSC_AS_TIME_ARGS(ctxt_pP),
                     ue_context_pP->ue_context.rnti);
  rrc_pdcp_config_asn1_req(ctxt_pP,
                           SRB_configList, // NULL,
                           DRB_configList,
                           DRB_Release_configList2,
                           0xff, // already configured during the securitymodecommand
                           kRRCenc,
                           kRRCint,
                           kUPenc,
                           (LTE_PMCH_InfoList_r9_t *) NULL,
                           NULL);
  /* Refresh SRBs/DRBs */
  if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
    rrc_rlc_config_asn1_req(ctxt_pP,
                            SRB_configList, // NULL,
                            DRB_configList,
                            DRB_Release_configList2,
                            (LTE_PMCH_InfoList_r9_t *) NULL,
                            0,
                            0
                            );
  }

  /* Set the SRB active in UE context */
  if (SRB_configList != NULL) {
    for (int i = 0; (i < SRB_configList->list.count) && (i < 3); i++) {
      if (SRB_configList->list.array[i]->srb_Identity == 1) {
        ue_context_pP->ue_context.Srb1.Active = 1;
      } else if (SRB_configList->list.array[i]->srb_Identity == 2) {
        ue_context_pP->ue_context.Srb2.Active = 1;
        ue_context_pP->ue_context.Srb2.Srb_info.Srb_id = 2;
        LOG_I(RRC,"[eNB %d] Frame  %d CC %d : SRB2 is now active\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ue_context_pP->ue_context.primaryCC_id);
      } else {
        LOG_W(RRC,"[eNB %d] Frame  %d CC %d : invalide SRB identity %ld\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ue_context_pP->ue_context.primaryCC_id,
              SRB_configList->list.array[i]->srb_Identity);
      }
    }

    free(SRB_configList);
    ue_context_pP->ue_context.SRB_configList2[xid] = NULL;
  }

  /* Loop through DRBs and establish if necessary */
  if (DRB_configList != NULL) 
  {
    for (int i = 0; i < DRB_configList->list.count; i++) // num max DRB (11-3-8)
    {
      if (DRB_configList->list.array[i]) 
      {
        drb_id = (int)DRB_configList->list.array[i]->drb_Identity;
        LOG_I(RRC, "[eNB %d] Frame  %d : Logical Channel UL-DCCH, Received LTE_RRCConnectionReconfigurationComplete from UE rnti %x, reconfiguring DRB %d/LCID %d\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              ctxt_pP->rnti,
              (int)DRB_configList->list.array[i]->drb_Identity,
              (int)*DRB_configList->list.array[i]->logicalChannelIdentity);
        /* For pre-ci tests */
        LOG_I(RRC, "[eNB %d] Frame  %d : Logical Channel UL-DCCH, Received LTE_RRCConnectionReconfigurationComplete from UE %u, reconfiguring DRB %d/LCID %d\n",
              ctxt_pP->module_id,
              ctxt_pP->frame,
              oai_emulation.info.eNB_ue_local_uid_to_ue_module_id[ctxt_pP->module_id][ue_context_pP->local_uid],
              (int)DRB_configList->list.array[i]->drb_Identity,
              (int)*DRB_configList->list.array[i]->logicalChannelIdentity);

#ifdef ENABLE_RAN_SLICING
	if (rsm_emm_event_trigger == 1)
    {
        /* Prepare and Send Attach Event Trigger for each DRB */
        MessageDef *m = itti_alloc_new_message(TASK_RRC_ENB, CU_EVENT_TRIGGER);
        CUEventTriggerToRic.eventTriggerType = UE_ATTACH_EVENT_TRIGGER;
        CUEventTriggerToRic.eventTriggerSize = sizeof(ueStatusInd);

        ueAttachInd = (ueStatusInd *)CUEventTriggerToRic.eventTriggerBuff;
        ueAttachInd->rnti = ue_context_pP->ue_context.rnti;
        ueAttachInd->eNB_ue_s1ap_id = ue_context_pP->ue_context.eNB_ue_s1ap_id; 
        ueAttachInd->mme_ue_s1ap_id = ue_context_pP->ue_context.mme_ue_s1ap_id;
        ueAttachInd->e_rab_id = ue_context_pP->ue_context.e_rab[i].param.e_rab_id;
        ueAttachInd->qci = ue_context_pP->ue_context.e_rab[i].param.qos.qci;
        ueAttachInd->cu_ue_f1ap_id = 0;//f1ap_get_cu_ue_f1ap_id(&f1ap_cu_inst[ctxt_pP->instance],
                                       //                     ueAttachInd->rnti); These will be populated in e2sm_rsm.c
        ueAttachInd->du_ue_f1ap_id = 0;//f1ap_get_du_ue_f1ap_id(&f1ap_cu_inst[ctxt_pP->instance],
                                       //                     ueAttachInd->rnti);

        CU_EVENT_TRIGGER(m).eventTriggerType = CUEventTriggerToRic.eventTriggerType;
        CU_EVENT_TRIGGER(m).eventTriggerSize = CUEventTriggerToRic.eventTriggerSize;
        
        memcpy(CU_EVENT_TRIGGER(m).eventTriggerBuff, 
               CUEventTriggerToRic.eventTriggerBuff, 
               CUEventTriggerToRic.eventTriggerSize);

        itti_send_msg_to_task(TASK_RIC_AGENT, module_id, m);
	}
#endif
    
        if (ue_context_pP->ue_context.DRB_active[drb_id] == 0) {
          ue_context_pP->ue_context.DRB_active[drb_id] = 1;
          LOG_D(RRC, "[eNB %d] Frame %d: Establish RLC UM Bidirectional, DRB %d Active\n",
                ctxt_pP->module_id, ctxt_pP->frame, (int)DRB_configList->list.array[i]->drb_Identity);

          if (!EPC_MODE_ENABLED && !ENB_NAS_USE_TUN) {
            LOG_I(OIP, "[eNB %d] trying to bring up the OAI interface oai%d\n",
                  ctxt_pP->module_id,
                  ctxt_pP->module_id);
            oip_ifup = nas_config(ctxt_pP->module_id,   // interface index
                                  ctxt_pP->module_id + 1,   // third octet
                                  ctxt_pP->module_id + 1,   // fourth octet
                                  "oai");

            if (oip_ifup == 0) { // interface is up --> send a config the DRB
              module_id_t ue_module_id;
              dest_ip_offset = 8;
              LOG_I(OIP,
                    "[eNB %d] Config the oai%d to send/receive pkt on DRB %ld to/from the protocol stack\n",
                    ctxt_pP->module_id, ctxt_pP->module_id,
                    (long int)((ue_context_pP->local_uid * LTE_maxDRB) + DRB_configList->list.array[i]->drb_Identity));
              ue_module_id = oai_emulation.info.eNB_ue_local_uid_to_ue_module_id[ctxt_pP->module_id][ue_context_pP->local_uid];
              rb_conf_ipv4(0, //add
                           ue_module_id,  //cx
                           ctxt_pP->module_id,    //inst
                           (ue_module_id * LTE_maxDRB) + DRB_configList->list.array[i]->drb_Identity, // RB
                           0,    //dscp
                           ipv4_address(ctxt_pP->module_id + 1, ctxt_pP->module_id + 1),  //saddr
                           ipv4_address(ctxt_pP->module_id + 1, dest_ip_offset + ue_module_id + 1));  //daddr
              LOG_D(RRC, "[eNB %d] State = Attached (UE rnti %x module id %u)\n",
                    ctxt_pP->module_id, ue_context_pP->ue_context.rnti, ue_module_id);
            } /* oip_ifup */
          } /* !EPC_MODE_ENABLED && ENB_NAS_USE_TUN*/

          LOG_D(RRC,
                PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_eNB\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

          if (DRB_configList->list.array[i]->logicalChannelIdentity) {
            DRB2LCHAN[i] = (uint8_t) * DRB_configList->list.array[i]->logicalChannelIdentity;
          }

          if (NODE_IS_MONOLITHIC(RC.rrc[ctxt_pP->module_id]->node_type)) {
            rrc_mac_config_req_eNB(ctxt_pP->module_id,
                                   ue_context_pP->ue_context.primaryCC_id,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   ue_context_pP->ue_context.rnti,
                                   (LTE_BCCH_BCH_Message_t *) NULL,
                                   (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                   (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                   ue_context_pP->ue_context.physicalConfigDedicated,
                                   (LTE_SCellToAddMod_r10_t *)NULL,
                                   (LTE_MeasObjectToAddMod_t **) NULL,
                                   ue_context_pP->ue_context.mac_MainConfig,
                                   DRB2LCHAN[i],
                                   DRB_configList->list.array[i]->logicalChannelConfig,
                                   ue_context_pP->ue_context.measGapConfig,
                                   (LTE_TDD_Config_t *) NULL,
                                   NULL,
                                   (LTE_SchedulingInfoList_t *) NULL,
                                   0,
                                   NULL,
                                   NULL,
                                   (LTE_MBSFN_SubframeConfigList_t *) NULL, 0,
                                   (LTE_MBSFN_AreaInfoList_r9_t *) NULL,
                                   (LTE_PMCH_InfoList_r9_t *) NULL,
                                   (LTE_SystemInformationBlockType1_v1310_IEs_t *) NULL,
                                   0,
                                   (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                                   (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                                   (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                                   (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                                   (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                                  );
          }
        } else {        // remove LCHAN from MAC/PHY
          if (ue_context_pP->ue_context.DRB_active[drb_id] == 1) {
            // DRB has just been removed so remove RLC + PDCP for DRB
            /*      rrc_pdcp_config_req (ctxt_pP->module_id, frameP, 1, CONFIG_ACTION_REMOVE,
               (ue_mod_idP * NB_RB_MAX) + DRB2LCHAN[i],UNDEF_SECURITY_MODE);
             */
            if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
              rrc_rlc_config_req(ctxt_pP,
                                 SRB_FLAG_NO,
                                 MBMS_FLAG_NO,
                                 CONFIG_ACTION_REMOVE,
                                 DRB2LCHAN[i],
                                 Rlc_info_um);
            }
          }

          ue_context_pP->ue_context.DRB_active[drb_id] = 0;
          LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (DRB) ---> MAC_eNB\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

          if (NODE_IS_MONOLITHIC(RC.rrc[ctxt_pP->module_id]->node_type)) {
            rrc_mac_config_req_eNB(ctxt_pP->module_id,
                                   ue_context_pP->ue_context.primaryCC_id,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   0,
                                   ue_context_pP->ue_context.rnti,
                                   (LTE_BCCH_BCH_Message_t *) NULL,
                                   (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                   (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                   ue_context_pP->ue_context.physicalConfigDedicated,
                                   (LTE_SCellToAddMod_r10_t *) NULL,
                                   (LTE_MeasObjectToAddMod_t **) NULL,
                                   ue_context_pP->ue_context.mac_MainConfig,
                                   DRB2LCHAN[i],
                                   (LTE_LogicalChannelConfig_t *) NULL,
                                   (LTE_MeasGapConfig_t *) NULL,
                                   (LTE_TDD_Config_t *) NULL,
                                   NULL,
                                   (LTE_SchedulingInfoList_t *) NULL,
                                   0,
                                   NULL,
                                   NULL,
                                   NULL,
                                   0,
                                   (LTE_MBSFN_AreaInfoList_r9_t *) NULL,
                                   (LTE_PMCH_InfoList_r9_t *) NULL,
                                   (LTE_SystemInformationBlockType1_v1310_IEs_t *) NULL,
                                   0,
                                   (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                                   (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                                   (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                                   (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                                   (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                                  );
          }
        } // end else of if (ue_context_pP->ue_context.DRB_active[drb_id] == 0)
      } // end if (DRB_configList->list.array[i])
    } // end for (int i = 0; i < DRB_configList->list.count; i++)

    free(DRB_configList);
    ue_context_pP->ue_context.DRB_configList2[xid] = NULL;
  } // end if DRB_configList != NULL

  if(DRB_Release_configList2 != NULL) {
    for (int i = 0; i < DRB_Release_configList2->list.count; i++) {
      if (DRB_Release_configList2->list.array[i]) {
        drb_id_p = DRB_Release_configList2->list.array[i];
        drb_id = *drb_id_p;

        if (ue_context_pP->ue_context.DRB_active[drb_id] == 1) {
          ue_context_pP->ue_context.DRB_active[drb_id] = 0;
        }
      }
    }

    free(DRB_Release_configList2);
    ue_context_pP->ue_context.DRB_Release_configList2[xid] = NULL;
  }

  /* let's request NR capabilities if the UE supports NR
   * maybe not the right place/time to request
   */
  if (ue_context_pP->ue_context.does_nr &&
      !ue_context_pP->ue_context.nr_capabilities_requested) {
    ue_context_pP->ue_context.nr_capabilities_requested = 1;
    rrc_eNB_generate_NR_UECapabilityEnquiry(ctxt_pP, ue_context_pP);
  }
}

//-----------------------------------------------------------------------------
void
rrc_eNB_generate_RRCConnectionSetup(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t  *const ue_context_pP,
  const int                    CC_id
)
//-----------------------------------------------------------------------------
{
  boolean_t is_mtc = ctxt_pP->brOption;
  LTE_LogicalChannelConfig_t             *SRB1_logicalChannelConfig;  //,*SRB2_logicalChannelConfig;
  LTE_SRB_ToAddModList_t                **SRB_configList;
  LTE_SRB_ToAddMod_t                     *SRB1_config;
  MessageDef                             *message_p;
  T(T_ENB_RRC_CONNECTION_SETUP,
    T_INT(ctxt_pP->module_id),
    T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe),
    T_INT(ctxt_pP->rnti));
  eNB_RRC_UE_t *ue_p = &ue_context_pP->ue_context;
  SRB_configList = &ue_p->SRB_configList;

  if (is_mtc) {
    ue_p->Srb0.Tx_buffer.payload_size =
      do_RRCConnectionSetup_BR(ctxt_pP,
                               ue_context_pP,
                               CC_id,
                               (uint8_t *) ue_p->Srb0.Tx_buffer.Payload,
                               (const uint8_t) RC.rrc[ctxt_pP->module_id]->carrier[CC_id].p_eNB, //at this point we do not have the UE capability information, so it can only be TM1 or TM2
                               rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),
                               SRB_configList,
                               &ue_context_pP->ue_context.physicalConfigDedicated);
  } else {
    ue_p->Srb0.Tx_buffer.payload_size =
      do_RRCConnectionSetup(ctxt_pP,
                            ue_context_pP,
                            CC_id,
                            (uint8_t *) ue_p->Srb0.Tx_buffer.Payload,
                            (uint8_t) RC.rrc[ctxt_pP->module_id]->carrier[CC_id].p_eNB, //at this point we do not have the UE capability information, so it can only be TM1 or TM2
                            rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),
                            SRB_configList,
                            &ue_context_pP->ue_context.physicalConfigDedicated);
  }

  LOG_DUMPMSG(RRC,DEBUG_RRC,
              (char *)(ue_p->Srb0.Tx_buffer.Payload),
              ue_p->Srb0.Tx_buffer.payload_size,
              "[MSG] RRC Connection Setup\n");

  // configure SRB1/SRB2, PhysicalConfigDedicated, MAC_MainConfig for UE
  switch (RC.rrc[ctxt_pP->module_id]->node_type) {
    case ngran_eNB_CU    :
    case ngran_ng_eNB_CU :
    case ngran_gNB_CU    :
      // create an ITTI message
      /* TODO: F1 IDs ar missing in RRC */
      message_p = itti_alloc_new_message (TASK_RRC_ENB, F1AP_DL_RRC_MESSAGE);
      F1AP_DL_RRC_MESSAGE (message_p).rrc_container =  (uint8_t *)ue_p->Srb0.Tx_buffer.Payload;
      F1AP_DL_RRC_MESSAGE (message_p).rrc_container_length = ue_p->Srb0.Tx_buffer.payload_size;
      F1AP_DL_RRC_MESSAGE (message_p).gNB_CU_ue_id     = 0;
      F1AP_DL_RRC_MESSAGE (message_p).gNB_DU_ue_id = 0;
      F1AP_DL_RRC_MESSAGE (message_p).old_gNB_DU_ue_id  = 0xFFFFFFFF; // unknown
      F1AP_DL_RRC_MESSAGE (message_p).rnti = ue_p->rnti;
      F1AP_DL_RRC_MESSAGE (message_p).srb_id = CCCH;
      F1AP_DL_RRC_MESSAGE (message_p).execute_duplication      = 1;
      F1AP_DL_RRC_MESSAGE (message_p).RAT_frequency_priority_information.en_dc      = 0;
      itti_send_msg_to_task (TASK_CU_F1, ctxt_pP->module_id, message_p);
      LOG_D(RRC, "Send F1AP_DL_RRC_MESSAGE with ITTI\n");
      break;

    case ngran_eNB_DU    :
    case ngran_gNB_DU  :
      // nothing to do for DU
      AssertFatal(1==0,"nothing to do for DU\n");
      break;

    case ngran_eNB:
    case ngran_ng_eNB :
    case ngran_gNB  :
      if (*SRB_configList != NULL) {
        for (int cnt = 0; cnt < (*SRB_configList)->list.count; cnt++) {
          if ((*SRB_configList)->list.array[cnt]->srb_Identity == 1) {
            SRB1_config = (*SRB_configList)->list.array[cnt];

            if (SRB1_config->logicalChannelConfig) {
              if (SRB1_config->logicalChannelConfig->present ==
                  LTE_SRB_ToAddMod__logicalChannelConfig_PR_explicitValue) {
                SRB1_logicalChannelConfig = &SRB1_config->logicalChannelConfig->choice.explicitValue;
              } else {
                SRB1_logicalChannelConfig = &SRB1_logicalChannelConfig_defaultValue;
              }
            } else {
              SRB1_logicalChannelConfig = &SRB1_logicalChannelConfig_defaultValue;
            }

            LOG_D(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" RRC_eNB --- MAC_CONFIG_REQ  (SRB1) ---> MAC_eNB\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

            if (RC.rrc[ctxt_pP->module_id]->node_type == ngran_eNB) {
              rrc_mac_config_req_eNB(ctxt_pP->module_id,
                                     ue_context_pP->ue_context.primaryCC_id,
                                     0,0,0,0,0, 0,
                                     ue_context_pP->ue_context.rnti,
                                     (LTE_BCCH_BCH_Message_t *) NULL,
                                     (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                     (LTE_RadioResourceConfigCommonSIB_t *) NULL,
                                     ue_context_pP->ue_context.physicalConfigDedicated,
                                     (LTE_SCellToAddMod_r10_t *)NULL,
                                     (LTE_MeasObjectToAddMod_t **) NULL,
                                     ue_context_pP->ue_context.mac_MainConfig,
                                     1,
                                     SRB1_logicalChannelConfig,
                                     ue_context_pP->ue_context.measGapConfig,
                                     (LTE_TDD_Config_t *) NULL,
                                     NULL,
                                     (LTE_SchedulingInfoList_t *) NULL,
                                     0, NULL, NULL, (LTE_MBSFN_SubframeConfigList_t *) NULL,
                                     0, (LTE_MBSFN_AreaInfoList_r9_t *) NULL, (LTE_PMCH_InfoList_r9_t *) NULL,
                                     (LTE_SystemInformationBlockType1_v1310_IEs_t *)NULL,
                                     0,
                                     (LTE_BCCH_DL_SCH_Message_MBMS_t *) NULL,
                                     (LTE_SchedulingInfo_MBMS_r14_t *) NULL,
                                     (struct LTE_NonMBSFN_SubframeConfig_r14 *) NULL,
                                     (LTE_SystemInformationBlockType1_MBMS_r14_t *) NULL,
                                     (LTE_MBSFN_AreaInfoList_r9_t *) NULL
                                    );
              break;
            }
          }
        }

        break;

      default :
        LOG_W(RRC, "Unknown node type %d\n", RC.rrc[ctxt_pP->module_id]->node_type);
      }

      MSC_LOG_TX_MESSAGE(
        MSC_RRC_ENB,
        MSC_RRC_UE,
        ue_p->Srb0.Tx_buffer.Header, // LG WARNING
        ue_p->Srb0.Tx_buffer.payload_size,
        MSC_AS_TIME_FMT" RRCConnectionSetup UE %x size %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        ue_context_pP->ue_context.rnti,
        ue_p->Srb0.Tx_buffer.payload_size);
      LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel DL-CCCH, Generating RRCConnectionSetup (bytes %d)\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            ue_p->Srb0.Tx_buffer.payload_size);
      // activate release timer, if RRCSetupComplete not received after 100 frames, remove UE
      ue_context_pP->ue_context.ue_release_timer = 1;
      // remove UE after 10 frames after RRCConnectionRelease is triggered
      ue_context_pP->ue_context.ue_release_timer_thres = 1000;
      /* init timers */
      ue_context_pP->ue_context.ue_rrc_inactivity_timer = 0;
  }
}

void setup_ngran_CU(eNB_RRC_INST *rrc) {
}

//-----------------------------------------------------------------------------
char openair_rrc_eNB_configuration(
  const module_id_t enb_mod_idP,
  RrcConfigurationReq *configuration
)
//-----------------------------------------------------------------------------
{
  protocol_ctxt_t ctxt;
  int             CC_id;
  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, enb_mod_idP, ENB_FLAG_YES, NOT_A_RNTI, 0, 0,enb_mod_idP);
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_FMT" Init...\n",
        PROTOCOL_RRC_CTXT_ARGS(&ctxt));
  AssertFatal(RC.rrc[enb_mod_idP] != NULL, "RC.rrc not initialized!");
  AssertFatal(MAX_MOBILES_PER_ENB < (module_id_t)0xFFFFFFFFFFFFFFFF, " variable overflow");
  AssertFatal(configuration!=NULL,"configuration input is null\n");
  RC.rrc[ctxt.module_id]->Nb_ue = 0;
  pthread_mutex_init(&RC.rrc[ctxt.module_id]->cell_info_mutex,NULL);
  RC.rrc[ctxt.module_id]->cell_info_configured = 0;
  uid_linear_allocator_init(&RC.rrc[ctxt.module_id]->uid_allocator);
  RB_INIT(&RC.rrc[ctxt.module_id]->rrc_ue_head);
  //    for (j = 0; j < (MAX_MOBILES_PER_ENB + 1); j++) {
  //        RC.rrc[enb_mod_idP]->Srb2[j].Active = 0;
  //    }
  RC.rrc[ctxt.module_id]->initial_id2_s1ap_ids = hashtable_create (MAX_MOBILES_PER_ENB * 2, NULL, NULL);
  RC.rrc[ctxt.module_id]->s1ap_id2_s1ap_ids    = hashtable_create (MAX_MOBILES_PER_ENB * 2, NULL, NULL);
  /// System Information INIT
  LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Checking release \n",
        PROTOCOL_RRC_CTXT_ARGS(&ctxt));
  // can clear it at runtime
  RC.rrc[ctxt.module_id]->carrier[0].MBMS_flag = 0;
  // This has to come from some top-level configuration
  // only CC_id 0 is logged
  LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Rel14 RRC detected, MBMS flag %d\n",
        PROTOCOL_RRC_CTXT_ARGS(&ctxt),
        RC.rrc[ctxt.module_id]->carrier[0].MBMS_flag);

  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    init_SI(&ctxt, CC_id, configuration);

    for (int ue_id = 0; ue_id < MAX_MOBILES_PER_ENB; ue_id++) {
      RC.rrc[ctxt.module_id]->carrier[CC_id].sizeof_paging[ue_id] = 0;
      RC.rrc[ctxt.module_id]->carrier[CC_id].paging[ue_id] = (uint8_t *) malloc16(256);
    }
  }

  rrc_init_global_param();

  for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
    switch (RC.rrc[ctxt.module_id]->carrier[CC_id].MBMS_flag) {
      case 1:
      case 2:
      case 3:
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Configuring 1 MBSFN sync area\n", PROTOCOL_RRC_CTXT_ARGS(&ctxt));
        RC.rrc[ctxt.module_id]->carrier[CC_id].num_mbsfn_sync_area = 1;
        break;

      case 4:
        LOG_I(RRC, PROTOCOL_RRC_CTXT_FMT" Configuring 2 MBSFN sync area\n", PROTOCOL_RRC_CTXT_ARGS(&ctxt));
        RC.rrc[ctxt.module_id]->carrier[CC_id].num_mbsfn_sync_area = 2;
        break;

      default:
        RC.rrc[ctxt.module_id]->carrier[CC_id].num_mbsfn_sync_area = 0;
        break;
    }

    // if we are here the RC.rrc[enb_mod_idP]->MBMS_flag > 0,
    /// MCCH INIT
    if (RC.rrc[ctxt.module_id]->carrier[CC_id].MBMS_flag > 0) {
      init_MCCH(ctxt.module_id, CC_id);
      /// MTCH data bearer init
      init_MBMS(ctxt.module_id, CC_id, 0);
    }

    openair_rrc_top_init_eNB(RC.rrc[ctxt.module_id]->carrier[CC_id].MBMS_flag,0);
  }

  openair_rrc_on(&ctxt);

  /*
    RC.rrc[ctxt.module_id]->mcc= rrc_configuration_req->mcc;
    RC.rrc[ctxt.module_id]->mnc= rrc_configuration_req->mnc;
    RC.rrc[ctxt.module_id]->mnc_digit_length= rrc_configuration_req->mnc_digit_length;
    RC.rrc[ctxt.module_id]->tac= rrc_configuration_req->tac;

    LOG_W(RRC, "[inst %d] RRC->MCC/MSG->MCC %d/%d \n", ctxt.module_id, RC.rrc[ctxt.module_id]->mcc, rrc_configuration_req->mcc);
    */
  if (NODE_IS_CU(RC.rrc[ctxt.module_id]->node_type))
    // msg_p = itti_alloc_new_message (TASK_ENB_APP, F1AP_SCTP_REQ);
    // RCconfig_CU_F1(msg_p, enb_id);
    setup_ngran_CU(RC.rrc[ctxt.module_id]);

  return 0;
}

/*------------------------------------------------------------------------------*/
int
rrc_eNB_decode_ccch(
  protocol_ctxt_t *const ctxt_pP,
  const uint8_t          *buffer,
  int                    buffer_length,
  const int              CC_id
)
//-----------------------------------------------------------------------------
{
  module_id_t                                       Idx;
  asn_dec_rval_t                                    dec_rval;
  LTE_UL_CCCH_Message_t                             *ul_ccch_msg = NULL;
  LTE_RRCConnectionRequest_r8_IEs_t                *rrcConnectionRequest = NULL;
  LTE_RRCConnectionReestablishmentRequest_r8_IEs_t *rrcConnectionReestablishmentRequest = NULL;
  int                                 i, rval;
  struct rrc_eNB_ue_context_s                  *ue_context_p = NULL;
  uint64_t                                      random_value = 0;
  int                                           stmsi_received = 0;
  T(T_ENB_RRC_UL_CCCH_DATA_IN, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
  //memset(ul_ccch_msg,0,sizeof(UL_CCCH_Message_t));
  LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Decoding UL CCCH %x.%x.%x.%x.%x.%x (%p)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
        ((uint8_t *) buffer)[0],
        ((uint8_t *) buffer)[1],
        ((uint8_t *) buffer)[2],
        ((uint8_t *) buffer)[3],
        ((uint8_t *) buffer)[4],
        ((uint8_t *) buffer)[5], (uint8_t *) buffer);
  dec_rval = uper_decode(
               NULL,
               &asn_DEF_LTE_UL_CCCH_Message,
               (void **)&ul_ccch_msg,
               (uint8_t *) buffer,
               100,
               0,
               0);

  for (i = 0; i < 8; i++) {
    LOG_T(RRC, "%x.", ((uint8_t *) & ul_ccch_msg)[i]);
  }

  if (dec_rval.consumed == 0) {
    LOG_E(RRC,
          PROTOCOL_RRC_CTXT_UE_FMT" FATAL Error in receiving CCCH\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    return -1;
  }

  if (ul_ccch_msg->message.present == LTE_UL_CCCH_MessageType_PR_c1) {
    switch (ul_ccch_msg->message.choice.c1.present) {
      case LTE_UL_CCCH_MessageType__c1_PR_NOTHING:
        LOG_I(RRC,
              PROTOCOL_RRC_CTXT_FMT" Received PR_NOTHING on UL-CCCH-Message\n",
              PROTOCOL_RRC_CTXT_ARGS(ctxt_pP));
        break;

      case LTE_UL_CCCH_MessageType__c1_PR_rrcConnectionReestablishmentRequest:
        T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_REQUEST, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)(buffer), buffer_length,
                    "[MSG] RRC Connection Reestablishment Request\n");
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT"MAC_eNB--- MAC_DATA_IND (rrcConnectionReestablishmentRequest on SRB0) --> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
        rrcConnectionReestablishmentRequest =
          &ul_ccch_msg->message.choice.c1.choice.rrcConnectionReestablishmentRequest.criticalExtensions.choice.rrcConnectionReestablishmentRequest_r8;
#ifdef ENABLE_RIC_AGENT
        rrc_kpi_stats.rrc_conn_reestab_att_sum++;
#endif        
        LOG_I(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest cause %s\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              ((rrcConnectionReestablishmentRequest->reestablishmentCause == LTE_ReestablishmentCause_otherFailure) ?    "Other Failure" :
               (rrcConnectionReestablishmentRequest->reestablishmentCause == LTE_ReestablishmentCause_handoverFailure) ? "Handover Failure" :
               "reconfigurationFailure"));
        {
          uint16_t                          c_rnti = 0;

          if (rrcConnectionReestablishmentRequest->ue_Identity.physCellId != RC.rrc[ctxt_pP->module_id]->carrier[CC_id].physCellId) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest ue_Identity.physCellId(%ld) is not equal to current physCellId(%d), let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                  rrcConnectionReestablishmentRequest->ue_Identity.physCellId,
                  RC.rrc[ctxt_pP->module_id]->carrier[CC_id].physCellId);
            rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
            break;
          }

          LOG_D(RRC, "physCellId is %ld\n", rrcConnectionReestablishmentRequest->ue_Identity.physCellId);

          for (i = 0; i < rrcConnectionReestablishmentRequest->ue_Identity.shortMAC_I.size; i++) {
            LOG_D(RRC, "rrcConnectionReestablishmentRequest->ue_Identity.shortMAC_I.buf[%d] = %x\n",
                  i, rrcConnectionReestablishmentRequest->ue_Identity.shortMAC_I.buf[i]);
          }

          if (rrcConnectionReestablishmentRequest->ue_Identity.c_RNTI.size == 0 ||
              rrcConnectionReestablishmentRequest->ue_Identity.c_RNTI.size > 2) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest c_RNTI range error, let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
            rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
            break;
          }

          c_rnti = BIT_STRING_to_uint16(&rrcConnectionReestablishmentRequest->ue_Identity.c_RNTI);
          LOG_D(RRC, "c_rnti is %x\n", c_rnti);
          ue_context_p = rrc_eNB_get_ue_context(RC.rrc[ctxt_pP->module_id], c_rnti);

          if (ue_context_p == NULL) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest without UE context, let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
            //rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
            break;
          }

          //int UE_id = find_UE_id(ctxt_pP->module_id, c_rnti);
          int UE_id = -1; //find_UE_id shouldn't be invoked from CU
          if(UE_id == -1) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest without UE_id(MAC) rnti %x, let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),c_rnti);
            rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
            break;
          }

          if((RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer > 0) &&
              (RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer_thres > 20)) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" RCConnectionReestablishmentComplete(Previous) don't receive, delete the c-rnti UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
            RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 1000;
            rrc_eNB_previous_SRB2(ue_context_p);
            ue_context_p->ue_context.ue_reestablishment_timer = 0;
          }

          //previous rnti
          rnti_t previous_rnti = 0;

          for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
            if (reestablish_rnti_map[i][1] == c_rnti) {
              previous_rnti = reestablish_rnti_map[i][0];
              break;
            }
          }

          if(previous_rnti != 0) {
            UE_id = find_UE_id(ctxt_pP->module_id, previous_rnti);

            if(UE_id == -1) {
              LOG_E(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionReestablishmentRequest without UE_id(MAC) previous rnti %x, let's reject the UE\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),previous_rnti);
              rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
              break;
            }

            if((RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer > 0) &&
                (RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer_thres > 20)) {
              LOG_E(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" RCConnectionReestablishmentComplete(Previous) don't receive, delete the Previous UE\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
              RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 1000;
              rrc_eNB_previous_SRB2(ue_context_p);
              ue_context_p->ue_context.ue_reestablishment_timer = 0;
            }
          }

          //c-plane not end
          if((ue_context_p->ue_context.Status != RRC_RECONFIGURED) && (ue_context_p->ue_context.reestablishment_cause == LTE_ReestablishmentCause_spare1)) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentRequest (UE %x c-plane is not end), let's reject the UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),c_rnti);
            rrc_eNB_generate_RRCConnectionReestablishmentReject(ctxt_pP, ue_context_p, CC_id);
            break;
          }

          if(ue_context_p->ue_context.ue_reestablishment_timer > 0) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" RRRCConnectionReconfigurationComplete(Previous) don't receive, delete the Previous UE\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
            ue_context_p->ue_context.Status = RRC_RECONFIGURED;
            protocol_ctxt_t  ctxt_old_p;
            PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt_old_p,
                                          ctxt_pP->instance,
                                          ENB_FLAG_YES,
                                          c_rnti,
                                          ctxt_pP->frame,
                                          ctxt_pP->subframe);
            rrc_eNB_process_RRCConnectionReconfigurationComplete(&ctxt_old_p,
                ue_context_p,
                ue_context_p->ue_context.reestablishment_xid);

            for (uint8_t e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
              if (ue_context_p->ue_context.e_rab[e_rab].status == E_RAB_STATUS_DONE) {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_ESTABLISHED;
              } else {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_FAILED;
              }
            }
          }

          LOG_D(RRC,
                PROTOCOL_RRC_CTXT_UE_FMT" UE context: %p\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                ue_context_p);
          /* reset timers */
          ue_context_p->ue_context.ul_failure_timer = 0;
          ue_context_p->ue_context.ue_release_timer = 0;
          ue_context_p->ue_context.ue_reestablishment_timer = 0;
          ue_context_p->ue_context.ue_release_timer_s1 = 0;
          ue_context_p->ue_context.ue_release_timer_rrc = 0;
          ue_context_p->ue_context.reestablishment_xid = -1;

          // insert C-RNTI to map
          for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
            if (reestablish_rnti_map[i][0] == 0) {
              reestablish_rnti_map[i][0] = ctxt_pP->rnti;
              reestablish_rnti_map[i][1] = c_rnti;
              LOG_D(RRC, "reestablish_rnti_map[%d] [0] %x, [1] %x\n",
                    i, reestablish_rnti_map[i][0], reestablish_rnti_map[i][1]);
              break;
            }
          }

          ue_context_p->ue_context.reestablishment_cause = rrcConnectionReestablishmentRequest->reestablishmentCause;
          LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept connection reestablishment request from UE physCellId %ld cause %ld\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                rrcConnectionReestablishmentRequest->ue_Identity.physCellId,
                ue_context_p->ue_context.reestablishment_cause);
#ifndef NO_RRM
          send_msg(&S_rrc, msg_rrc_MR_attach_ind(ctxt_pP->module_id, Mac_id));
#else
          ue_context_p->ue_context.primaryCC_id = CC_id;
          //LG COMMENT Idx = (ue_mod_idP * NB_RB_MAX) + DCCH;
          Idx = DCCH;
          // SRB1
          ue_context_p->ue_context.Srb1.Active = 1;
          ue_context_p->ue_context.Srb1.Srb_info.Srb_id = Idx;
          memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[0],
                 &DCCH_LCHAN_DESC,
                 LCHAN_DESC_SIZE);
          memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[1],
                 &DCCH_LCHAN_DESC,
                 LCHAN_DESC_SIZE);
          // SRB2: set  it to go through SRB1 with id 1 (DCCH)
          ue_context_p->ue_context.Srb2.Active = 1;
          ue_context_p->ue_context.Srb2.Srb_info.Srb_id = Idx;
          memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[0],
                 &DCCH_LCHAN_DESC,
                 LCHAN_DESC_SIZE);
          memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[1],
                 &DCCH_LCHAN_DESC,
                 LCHAN_DESC_SIZE);
          rrc_eNB_generate_RRCConnectionReestablishment(ctxt_pP, ue_context_p, CC_id);
          LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT"CALLING RLC CONFIG SRB1 (rbid %d)\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                Idx);
          MSC_LOG_TX_MESSAGE(MSC_RRC_ENB,
                             MSC_PDCP_ENB,
                             NULL,
                             0,
                             MSC_AS_TIME_FMT" CONFIG_REQ UE %x SRB",
                             MSC_AS_TIME_ARGS(ctxt_pP),
                             ue_context_p->ue_context.rnti);
          rrc_pdcp_config_asn1_req(ctxt_pP,
                                   ue_context_p->ue_context.SRB_configList,
                                   (LTE_DRB_ToAddModList_t *) NULL,
                                   (LTE_DRB_ToReleaseList_t *) NULL,
                                   0xff,
                                   NULL,
                                   NULL,
                                   NULL
                                   , (LTE_PMCH_InfoList_r9_t *) NULL
                                   ,NULL);

          if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
            rrc_rlc_config_asn1_req(ctxt_pP,
                                    ue_context_p->ue_context.SRB_configList,
                                    (LTE_DRB_ToAddModList_t *) NULL,
                                    (LTE_DRB_ToReleaseList_t *) NULL
                                    , (LTE_PMCH_InfoList_r9_t *) NULL,
                                    0,0
                                   );
          }

#endif //NO_RRM
        }
        break;

      case LTE_UL_CCCH_MessageType__c1_PR_rrcConnectionRequest:
        T(T_ENB_RRC_CONNECTION_REQUEST, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)buffer,
                    buffer_length,
                    "[MSG] RRC Connection Request\n");
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT"MAC_eNB --- MAC_DATA_IND  (rrcConnectionRequest on SRB0) --> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
        ue_context_p = rrc_eNB_get_ue_context(
                         RC.rrc[ctxt_pP->module_id],
                         ctxt_pP->rnti);

#ifdef ENABLE_RIC_AGENT
        rrc_kpi_stats.rrc_conn_estab_att_sum++;
#endif        
        if (ue_context_p != NULL) {
          // erase content
          rrc_eNB_free_mem_UE_context(ctxt_pP, ue_context_p);
          MSC_LOG_RX_DISCARDED_MESSAGE(
            MSC_RRC_ENB,
            MSC_RRC_UE,
            buffer,
            dec_rval.consumed,
            MSC_AS_TIME_FMT" LTE_RRCConnectionRequest UE %x size %u (UE already in context)",
            MSC_AS_TIME_ARGS(ctxt_pP),
            ue_context_p->ue_context.rnti,
            dec_rval.consumed);
        } else {
          rrcConnectionRequest = &ul_ccch_msg->message.choice.c1.choice.rrcConnectionRequest.criticalExtensions.choice.rrcConnectionRequest_r8;
          {
            if (LTE_InitialUE_Identity_PR_randomValue == rrcConnectionRequest->ue_Identity.present) {
              if(rrcConnectionRequest->ue_Identity.choice.randomValue.size != 5) {
                LOG_I(RRC, "wrong InitialUE-Identity randomValue size, expected 5, provided %lu",
                      (long unsigned int)rrcConnectionRequest->ue_Identity.choice.randomValue.size);
                return -1;
              }

              memcpy(((uint8_t *) & random_value) + 3,
                     rrcConnectionRequest->ue_Identity.choice.randomValue.buf,
                     rrcConnectionRequest->ue_Identity.choice.randomValue.size);

              /* if there is already a registered UE (with another RNTI) with this random_value,
               * the current one must be removed from MAC/PHY (zombie UE)
               */
              if ((ue_context_p = rrc_eNB_ue_context_random_exist(ctxt_pP, random_value))) {
                LOG_W(RRC, "new UE rnti %x (coming with random value) is already there as UE %x, removing %x from MAC/PHY\n",
                      ctxt_pP->rnti, ue_context_p->ue_context.rnti, ue_context_p->ue_context.rnti);
                ue_context_p->ue_context.ul_failure_timer = 20000;
              }

              ue_context_p = rrc_eNB_get_next_free_ue_context(ctxt_pP, random_value);
              ue_context_p->ue_context.Srb0.Srb_id = 0;
              ue_context_p->ue_context.Srb0.Active = 1;
              memcpy(ue_context_p->ue_context.Srb0.Rx_buffer.Payload,
                     buffer,
                     buffer_length);
              ue_context_p->ue_context.Srb0.Rx_buffer.payload_size = buffer_length;
            } else if (LTE_InitialUE_Identity_PR_s_TMSI == rrcConnectionRequest->ue_Identity.present) {
              /* Save s-TMSI */
              LTE_S_TMSI_t   s_TMSI = rrcConnectionRequest->ue_Identity.choice.s_TMSI;
              mme_code_t mme_code = BIT_STRING_to_uint8(&s_TMSI.mmec);
              m_tmsi_t   m_tmsi   = BIT_STRING_to_uint32(&s_TMSI.m_TMSI);
              random_value = (((uint64_t)mme_code) << 32) | m_tmsi;

              if ((ue_context_p = rrc_eNB_ue_context_stmsi_exist(ctxt_pP, mme_code, m_tmsi))) {
                LOG_I(RRC," S-TMSI exists, ue_context_p %p, old rnti %x => %x\n",ue_context_p,ue_context_p->ue_context.rnti,ctxt_pP->rnti);

                if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
                  rrc_mac_remove_ue(ctxt_pP->module_id, ue_context_p->ue_context.rnti);
                } else {
                  MessageDef *m = itti_alloc_new_message(TASK_RRC_ENB, F1AP_UE_CONTEXT_RELEASE_CMD);
                  F1AP_UE_CONTEXT_RELEASE_CMD(m).rnti = ctxt_pP->rnti;
                  F1AP_UE_CONTEXT_RELEASE_CMD(m).cause = F1AP_CAUSE_RADIO_NETWORK;
                  F1AP_UE_CONTEXT_RELEASE_CMD(m).cause_value = 10; // 10 = F1AP_CauseRadioNetwork_normal_release
                  F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container = NULL;
                  F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container_length = 0;
                  itti_send_msg_to_task(TASK_CU_F1, ctxt_pP->module_id, m);
                }

                stmsi_received=1;
                /* replace rnti in the context */
                /* for that, remove the context from the RB tree */
                RB_REMOVE(rrc_ue_tree_s, &RC.rrc[ctxt_pP->module_id]->rrc_ue_head, ue_context_p);
                /* and insert again, after changing rnti everywhere it has to be changed */
                ue_context_p->ue_id_rnti = ctxt_pP->rnti;
                ue_context_p->ue_context.rnti = ctxt_pP->rnti;
                RB_INSERT(rrc_ue_tree_s, &RC.rrc[ctxt_pP->module_id]->rrc_ue_head, ue_context_p);
                /* reset timers */
                ue_context_p->ue_context.ul_failure_timer = 0;
                ue_context_p->ue_context.ue_release_timer = 0;
                ue_context_p->ue_context.ue_reestablishment_timer = 0;
                ue_context_p->ue_context.ue_release_timer_s1 = 0;
                ue_context_p->ue_context.ue_release_timer_rrc = 0;
                ue_context_p->ue_context.reestablishment_xid = -1;
              } else {
                LOG_I(RRC," S-TMSI doesn't exist, setting Initialue_identity_s_TMSI.m_tmsi to %p => %x\n",ue_context_p,m_tmsi);
                //              ue_context_p = rrc_eNB_get_next_free_ue_context(ctxt_pP, NOT_A_RANDOM_UE_IDENTITY);
                ue_context_p = rrc_eNB_get_next_free_ue_context(ctxt_pP,random_value);

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
                buffer,
                dec_rval.consumed,
                MSC_AS_TIME_FMT" RRCConnectionRequest UE %x size %u (s-TMSI mmec %u m_TMSI %u random UE id (0x%" PRIx64 ")",
                MSC_AS_TIME_ARGS(ctxt_pP),
                ue_context_p->ue_context.rnti,
                dec_rval.consumed,
                ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
                ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
                ue_context_p->ue_context.random_ue_identity);
            } else {
              LOG_E(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionRequest without random UE identity or S-TMSI not supported, let's reject the UE\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
              rrc_eNB_generate_RRCConnectionReject(ctxt_pP,
                                                   rrc_eNB_get_ue_context(RC.rrc[ctxt_pP->module_id], ctxt_pP->rnti),
                                                   CC_id);
              break;
            }
          }
          LOG_D(RRC,
                PROTOCOL_RRC_CTXT_UE_FMT" UE context: %p\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                ue_context_p);

          if (ue_context_p != NULL) {
            ue_context_p->ue_context.establishment_cause = rrcConnectionRequest->establishmentCause;
            ue_context_p->ue_context.reestablishment_cause = LTE_ReestablishmentCause_spare1;

            if (stmsi_received==0)
              LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept new connection from UE random UE identity (0x%" PRIx64 ") MME code %u TMSI %u cause %ld\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                    ue_context_p->ue_context.random_ue_identity,
                    ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
                    ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
                    ue_context_p->ue_context.establishment_cause);
            else
              LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Accept new connection from UE  MME code %u TMSI %u cause %ld\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                    ue_context_p->ue_context.Initialue_identity_s_TMSI.mme_code,
                    ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
                    ue_context_p->ue_context.establishment_cause);

            if (stmsi_received == 0)
              RC.rrc[ctxt_pP->module_id]->Nb_ue++;
          } else {
            // no context available
            if (flexran_agent_get_rrc_xface(ctxt_pP->module_id)) {
              flexran_agent_get_rrc_xface(ctxt_pP->module_id)->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
                  ctxt_pP->rnti,
                  PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_DEACTIVATED);
            }

            LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Can't create new context for UE random UE identity (0x%" PRIx64 ")\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                  random_value);

            if (NODE_IS_MONOLITHIC(RC.rrc[ctxt_pP->module_id]->node_type))
              rrc_mac_remove_ue(ctxt_pP->module_id,ctxt_pP->rnti);
            else if (NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
              MessageDef *m = itti_alloc_new_message(TASK_RRC_ENB, F1AP_UE_CONTEXT_RELEASE_CMD);
              F1AP_UE_CONTEXT_RELEASE_CMD(m).rnti = ctxt_pP->rnti;
              F1AP_UE_CONTEXT_RELEASE_CMD(m).cause = F1AP_CAUSE_RADIO_NETWORK;
              F1AP_UE_CONTEXT_RELEASE_CMD(m).cause_value = 10; // 10 = F1AP_CauseRadioNetwork_normal_release
              F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container = NULL;
              F1AP_UE_CONTEXT_RELEASE_CMD(m).rrc_container_length = 0;
              itti_send_msg_to_task(TASK_CU_F1, ctxt_pP->module_id, m);
            }

            return -1;
          }
        }

#ifndef NO_RRM
        send_msg(&S_rrc, msg_rrc_MR_attach_ind(ctxt_pP->module_id, Mac_id));
#else
        ue_context_p->ue_context.primaryCC_id = CC_id;
        //LG COMMENT Idx = (ue_mod_idP * NB_RB_MAX) + DCCH;
        Idx = DCCH;
        // SRB1
        ue_context_p->ue_context.Srb1.Active = 1;
        ue_context_p->ue_context.Srb1.Srb_info.Srb_id = Idx;
        memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[0],
               &DCCH_LCHAN_DESC,
               LCHAN_DESC_SIZE);
        memcpy(&ue_context_p->ue_context.Srb1.Srb_info.Lchan_desc[1],
               &DCCH_LCHAN_DESC,
               LCHAN_DESC_SIZE);
        // SRB2: set  it to go through SRB1 with id 1 (DCCH)
        ue_context_p->ue_context.Srb2.Active = 1;
        ue_context_p->ue_context.Srb2.Srb_info.Srb_id = Idx;
        memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[0],
               &DCCH_LCHAN_DESC,
               LCHAN_DESC_SIZE);
        memcpy(&ue_context_p->ue_context.Srb2.Srb_info.Lchan_desc[1],
               &DCCH_LCHAN_DESC,
               LCHAN_DESC_SIZE);
        rrc_eNB_generate_RRCConnectionSetup(ctxt_pP, ue_context_p, CC_id);
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
        rrc_pdcp_config_asn1_req(ctxt_pP,
                                 ue_context_p->ue_context.SRB_configList,
                                 (LTE_DRB_ToAddModList_t *) NULL,
                                 (LTE_DRB_ToReleaseList_t *) NULL,
                                 0xff,
                                 NULL,
                                 NULL,
                                 NULL,
                                 (LTE_PMCH_InfoList_r9_t *) NULL,NULL);

        if (!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type)) {
          rrc_rlc_config_asn1_req(ctxt_pP,
                                  ue_context_p->ue_context.SRB_configList,
                                  (LTE_DRB_ToAddModList_t *) NULL,
                                  (LTE_DRB_ToReleaseList_t *) NULL,
                                  (LTE_PMCH_InfoList_r9_t *) NULL, 0, 0
                                 );
        }

#endif //NO_RRM
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
static int
is_en_dc_supported(
  LTE_UE_EUTRA_Capability_t *c
)
//-----------------------------------------------------------------------------
{
  /* to be refined - check that the eNB is connected to a gNB, check that
   * the bands supported by the UE include the band of the gNB
   */
#define NCE nonCriticalExtension
  return c != NULL
         && c->NCE != NULL
         && c->NCE->NCE != NULL
         && c->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->irat_ParametersNR_r15 != NULL
         && c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->irat_ParametersNR_r15->en_DC_r15 != NULL
         && *c->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->NCE->irat_ParametersNR_r15->en_DC_r15 ==
         LTE_IRAT_ParametersNR_r15__en_DC_r15_supported;
#undef NCE
}

//-----------------------------------------------------------------------------
int
rrc_eNB_decode_dcch(
  const protocol_ctxt_t *const ctxt_pP,
  const rb_id_t                Srb_id,
  const uint8_t    *const      Rx_sdu,
  const sdu_size_t             sdu_sizeP
)
//-----------------------------------------------------------------------------
{
  asn_dec_rval_t                      dec_rval;
  //UL_DCCH_Message_t uldcchmsg;
  LTE_UL_DCCH_Message_t               *ul_dcch_msg = NULL; //&uldcchmsg;
  int i;
  struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
  MessageDef                         *msg_delete_tunnels_p = NULL;
  uint8_t                             xid;
  int dedicated_DRB=0;
  T(T_ENB_RRC_UL_DCCH_DATA_IN, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
    T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

  if ((Srb_id != 1) && (Srb_id != 2)) {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received message on SRB%ld, should not have ...\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          Srb_id);
  } else {
    LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received message on SRB%ld\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          Srb_id);
  }

  //memset(ul_dcch_msg,0,sizeof(UL_DCCH_Message_t));
  LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Decoding UL-DCCH Message\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
  dec_rval = uper_decode(
               NULL,
               &asn_DEF_LTE_UL_DCCH_Message,
               (void **)&ul_dcch_msg,
               Rx_sdu,
               sdu_sizeP,
               0,
               0);
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

  ue_context_p = rrc_eNB_get_ue_context(
                   RC.rrc[ctxt_pP->module_id],
                   ctxt_pP->rnti);

  if (ul_dcch_msg->message.present == LTE_UL_DCCH_MessageType_PR_c1) {
    switch (ul_dcch_msg->message.choice.c1.present) {
      case LTE_UL_DCCH_MessageType__c1_PR_NOTHING:   /* No components present */
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_csfbParametersRequestCDMA2000:
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_measurementReport:

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing measurementReport UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND "
              "%d bytes (measurementReport) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);
        rrc_eNB_process_MeasurementReport(
          ctxt_pP,
          ue_context_p,
          &ul_dcch_msg->message.choice.c1.choice.measurementReport.
          criticalExtensions.choice.c1.choice.measurementReport_r8.measResults);
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_rrcConnectionReconfigurationComplete:

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing LTE_RRCConnectionReconfigurationComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)(Rx_sdu),sdu_sizeP,
                    "[MSG] RRC Connection Reconfiguration Complete\n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" LTE_RRCConnectionReconfigurationComplete UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(RRCConnectionReconfigurationComplete) ---> RRC_eNB]\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if (ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.
            present ==
            LTE_RRCConnectionReconfigurationComplete__criticalExtensions_PR_rrcConnectionReconfigurationComplete_r8) {
          /*NN: revise the condition */
          /*FK: left the condition as is for the case MME is used (S1 mode) but setting  dedicated_DRB = 1 otherwise (noS1 mode) so that no second RRCReconfiguration message activationg more DRB is sent as this causes problems with the nasmesh driver.*/
          int flexran_agent_handover = 0;

          if (EPC_MODE_ENABLED) {
            if (ue_context_p->ue_context.Status == RRC_RECONFIGURED) {
              dedicated_DRB = 1;
              LOG_I(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (dedicated DRB, xid %ld)\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
              //clear
              int16_t UE_id = find_UE_id(ctxt_pP->module_id, ctxt_pP->rnti);

              if(UE_id == -1) {
                LOG_E(RRC,
                      PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionReconfigurationComplete without rnti %x, fault\n",
                      PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ctxt_pP->rnti);
                break;
              }

              AssertFatal(!NODE_IS_CU(RC.rrc[ctxt_pP->module_id]->node_type),
                          "CU cannot decode DCCH: no access to RC.mac[]\n");

              if(RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].crnti_reconfigurationcomplete_flag == 1) {
                LOG_I(RRC,
                      PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (dedicated DRB, xid %ld) C-RNTI Complete\n",
                      PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
                dedicated_DRB = 2;
                RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].crnti_reconfigurationcomplete_flag = 0;
              }
            } else if (ue_context_p->ue_context.Status == RRC_HO_EXECUTION) {
              int16_t UE_id = find_UE_id(ctxt_pP->module_id, ctxt_pP->rnti);

              if(UE_id == -1) {
                LOG_E(RRC,
                      PROTOCOL_RRC_CTXT_UE_FMT" RRCConnectionReconfigurationComplete without rnti %x, fault\n",
                      PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ctxt_pP->rnti);
                break;
              }

              flexran_agent_handover = 1;
              RC.rrc[ctxt_pP->module_id]->Nb_ue++;
              dedicated_DRB = 3;
              RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].crnti_reconfigurationcomplete_flag = 0;
              ue_context_p->ue_context.Status = RRC_RECONFIGURED;

              if(ue_context_p->ue_context.handover_info) {
                ue_context_p->ue_context.handover_info->state = HO_CONFIGURED;
              }

              LOG_I(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_HO_EXECUTION (xid %ld)\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
            }
            else if(ue_context_p->ue_context.Status == RRC_NR_NSA){
              //Looking for a condition to trigger S1AP E-RAB-Modification-indication, based on the reception of RRCConnectionReconfigurationComplete
              //including NR specific elements.
              if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                  nonCriticalExtension!=NULL) {
                if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                    nonCriticalExtension->nonCriticalExtension!=NULL) {
                  if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                      nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL) {
                    if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                        nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL) {
                      if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                          nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL) {
                        if (ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                            nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension!=NULL) {
                          if(ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.criticalExtensions.choice.rrcConnectionReconfigurationComplete_r8.
                              nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension->nonCriticalExtension
                              ->scg_ConfigResponseNR_r15!=NULL) {
                            ue_context_p->ue_context.Status = RRC_NR_NSA_RECONFIGURED;
                            /*Trigger E-RAB Modification Indication */
                            rrc_eNB_send_E_RAB_Modification_Indication(ctxt_pP, ue_context_p);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            else {
              dedicated_DRB = 0;
              ue_context_p->ue_context.Status = RRC_RECONFIGURED;
              LOG_I(RRC,
                    PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (default DRB, xid %ld)\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
            }

            ue_context_p->ue_context.reestablishment_xid = -1;

          } else {
            dedicated_DRB = 1;
            ue_context_p->ue_context.Status = RRC_RECONFIGURED;
            LOG_I(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_RECONFIGURED (dedicated DRB, xid %ld)\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
          }

          rrc_eNB_process_RRCConnectionReconfigurationComplete(
            ctxt_pP,
            ue_context_p,
            ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);

          //WARNING:Inform the controller about the UE activation. Should be moved to RRC agent in the future
          if (flexran_agent_get_rrc_xface(ctxt_pP->module_id)) {
            flexran_agent_get_rrc_xface(ctxt_pP->module_id)->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
                ue_context_p->ue_id_rnti,
                flexran_agent_handover?PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_ACTIVATED:PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_UPDATED);
          }
        }

        if (EPC_MODE_ENABLED) {
          if (dedicated_DRB == 1) {
            //    rrc_eNB_send_S1AP_E_RAB_SETUP_RESP(ctxt_pP,
            //               ue_context_p,
            //               ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
            if (ue_context_p->ue_context.nb_of_modify_e_rabs > 0) {
              rrc_eNB_send_S1AP_E_RAB_MODIFY_RESP(ctxt_pP,
                                                  ue_context_p,
                                                  ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
              ue_context_p->ue_context.nb_of_modify_e_rabs = 0;
              ue_context_p->ue_context.nb_of_failed_e_rabs = 0;
              memset(ue_context_p->ue_context.modify_e_rab, 0, sizeof(ue_context_p->ue_context.modify_e_rab));

              for(int i = 0; i < NB_RB_MAX; i++) {
                ue_context_p->ue_context.modify_e_rab[i].xid = -1;
              }
            } else if(ue_context_p->ue_context.e_rab_release_command_flag == 1) {
              xid = ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier;
              ue_context_p->ue_context.e_rab_release_command_flag = 0;
              //gtp tunnel delete
              msg_delete_tunnels_p = itti_alloc_new_message(TASK_RRC_ENB, GTPV1U_ENB_DELETE_TUNNEL_REQ);
              memset(&GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p), 0, sizeof(GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p)));
              GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).rnti = ue_context_p->ue_context.rnti;

              for(i = 0; i < NB_RB_MAX; i++) {
                if(xid == ue_context_p->ue_context.e_rab[i].xid) {
                  GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).eps_bearer_id[GTPV1U_ENB_DELETE_TUNNEL_REQ(msg_delete_tunnels_p).num_erab++] = ue_context_p->ue_context.enb_gtp_ebi[i];
                  ue_context_p->ue_context.enb_gtp_teid[i] = 0;
                  memset(&ue_context_p->ue_context.enb_gtp_addrs[i], 0, sizeof(ue_context_p->ue_context.enb_gtp_addrs[i]));
                  ue_context_p->ue_context.enb_gtp_ebi[i]  = 0;
                }
              }

              itti_send_msg_to_task(TASK_GTPV1_U, ctxt_pP->instance, msg_delete_tunnels_p);
              //S1AP_E_RAB_RELEASE_RESPONSE
              rrc_eNB_send_S1AP_E_RAB_RELEASE_RESPONSE(ctxt_pP,
                  ue_context_p,
                  xid);
            } else {
              rrc_eNB_send_S1AP_E_RAB_SETUP_RESP(ctxt_pP,
                                                 ue_context_p,
                                                 ul_dcch_msg->message.choice.c1.choice.rrcConnectionReconfigurationComplete.rrc_TransactionIdentifier);
            }
          } else if(dedicated_DRB == 0) {
            if(ue_context_p->ue_context.reestablishment_cause == LTE_ReestablishmentCause_spare1) {
              rrc_eNB_send_S1AP_INITIAL_CONTEXT_SETUP_RESP(ctxt_pP,
                  ue_context_p);
            } else {
              ue_context_p->ue_context.reestablishment_cause = LTE_ReestablishmentCause_spare1;

              for (uint8_t e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
                if (ue_context_p->ue_context.e_rab[e_rab].status == E_RAB_STATUS_DONE) {
                  ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_ESTABLISHED;
                } else {
                  ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_FAILED;
                }
              }
            }
          } else if(dedicated_DRB == 2) {
            for (uint8_t e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
              if (ue_context_p->ue_context.e_rab[e_rab].status == E_RAB_STATUS_DONE) {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_ESTABLISHED;
              } else {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_FAILED;
              }
            }
          } else if(dedicated_DRB == 3) { //x2 path switch
            for (uint8_t e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
              if (ue_context_p->ue_context.e_rab[e_rab].status == E_RAB_STATUS_DONE) {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_ESTABLISHED;
              } else {
                ue_context_p->ue_context.e_rab[e_rab].status = E_RAB_STATUS_FAILED;
              }
            }

            LOG_I(RRC,"issue rrc_eNB_send_PATH_SWITCH_REQ \n");
            rrc_eNB_send_PATH_SWITCH_REQ(ctxt_pP,ue_context_p);
          }
        } /* EPC_MODE_ENABLED */

        break;

      case LTE_UL_DCCH_MessageType__c1_PR_rrcConnectionReestablishmentComplete:
        T(T_ENB_RRC_CONNECTION_REESTABLISHMENT_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC Connection Reestablishment Complete\n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" LTE_RRCConnectionReestablishmentComplete UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);
        LOG_I(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(rrcConnectionReestablishmentComplete) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);
        {
          rnti_t reestablish_rnti = 0;

          // select C-RNTI from map
          for (i = 0; i < MAX_MOBILES_PER_ENB; i++) {
            if (reestablish_rnti_map[i][0] == ctxt_pP->rnti) {
              reestablish_rnti = reestablish_rnti_map[i][1];
              ue_context_p = rrc_eNB_get_ue_context(
                               RC.rrc[ctxt_pP->module_id],
                               reestablish_rnti);
              // clear currentC-RNTI from map
              reestablish_rnti_map[i][0] = 0;
              reestablish_rnti_map[i][1] = 0;
              LOG_D(RRC, "reestablish_rnti_map[%d] [0] %x, [1] %x\n",
                    i, reestablish_rnti_map[i][0], reestablish_rnti_map[i][1]);
              break;
            }
          }

          if (!ue_context_p) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentComplete without UE context, falt\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
            break;
          }

          //clear
          int UE_id = find_UE_id(ctxt_pP->module_id, ctxt_pP->rnti);

          if(UE_id == -1) {
            LOG_E(RRC,
                  PROTOCOL_RRC_CTXT_UE_FMT" LTE_RRCConnectionReestablishmentComplete without UE_id(MAC) rnti %x, fault\n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),ctxt_pP->rnti);
            break;
          }

          RC.mac[ctxt_pP->module_id]->UE_info.UE_sched_ctrl[UE_id].ue_reestablishment_reject_timer = 0;
          ue_context_p->ue_context.reestablishment_xid = -1;

          if (ul_dcch_msg->message.choice.c1.choice.rrcConnectionReestablishmentComplete.criticalExtensions.present ==
              LTE_RRCConnectionReestablishmentComplete__criticalExtensions_PR_rrcConnectionReestablishmentComplete_r8) {
            rrc_eNB_process_RRCConnectionReestablishmentComplete(ctxt_pP, reestablish_rnti, ue_context_p,
                ul_dcch_msg->message.choice.c1.choice.rrcConnectionReestablishmentComplete.rrc_TransactionIdentifier,
                &ul_dcch_msg->message.choice.c1.choice.rrcConnectionReestablishmentComplete.criticalExtensions.choice.rrcConnectionReestablishmentComplete_r8);

            //WARNING:Inform the controller about the UE activation. Should be moved to RRC agent in the future
            if (flexran_agent_get_rrc_xface(ctxt_pP->module_id)) {
              flexran_agent_get_rrc_xface(ctxt_pP->module_id)->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
                  ue_context_p->ue_id_rnti,
                  PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_ACTIVATED);
            }
          }

          //ue_context_p->ue_context.ue_release_timer = 0;
          ue_context_p->ue_context.ue_reestablishment_timer = 1;
          // remove UE after 100 frames after LTE_RRCConnectionReestablishmentRelease is triggered
          ue_context_p->ue_context.ue_reestablishment_timer_thres = 1000;
        }
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_rrcConnectionSetupComplete:

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing LTE_RRCConnectionSetupComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC Connection SetupComplete\n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" LTE_RRCConnectionSetupComplete UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(RRCConnectionSetupComplete) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if (ul_dcch_msg->message.choice.c1.choice.rrcConnectionSetupComplete.criticalExtensions.present ==
            LTE_RRCConnectionSetupComplete__criticalExtensions_PR_c1) {
          if (ul_dcch_msg->message.choice.c1.choice.rrcConnectionSetupComplete.criticalExtensions.choice.c1.
              present ==
              LTE_RRCConnectionSetupComplete__criticalExtensions__c1_PR_rrcConnectionSetupComplete_r8) {
            AssertFatal(!NODE_IS_DU(RC.rrc[ctxt_pP->module_id]->node_type),
                        "should not be reached in DU\n");
            rrc_eNB_process_RRCConnectionSetupComplete(
              ctxt_pP,
              ue_context_p,
              &ul_dcch_msg->message.choice.c1.choice.rrcConnectionSetupComplete.criticalExtensions.choice.c1.choice.rrcConnectionSetupComplete_r8);
            LOG_I(RRC, PROTOCOL_RRC_CTXT_UE_FMT" UE State = RRC_CONNECTED \n",
                  PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

            //WARNING:Inform the controller about the UE activation. Should be moved to RRC agent in the future
            if (flexran_agent_get_rrc_xface(ctxt_pP->module_id)) {
              flexran_agent_get_rrc_xface(ctxt_pP->module_id)->flexran_agent_notify_ue_state_change(ctxt_pP->module_id,
                  ue_context_p->ue_id_rnti,
                  PROTOCOL__FLEX_UE_STATE_CHANGE_TYPE__FLUESC_ACTIVATED);
            }
          }
        }

        ue_context_p->ue_context.ue_release_timer=0;
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_securityModeComplete:
        T(T_ENB_RRC_SECURITY_MODE_COMPLETE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing securityModeComplete UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC Security Mode Complete\n");
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
              PROTOCOL_RRC_CTXT_UE_FMT" received securityModeComplete on UL-DCCH %d from UE\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH);
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(securityModeComplete) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
          xer_fprint(stdout, &asn_DEF_LTE_UL_DCCH_Message, (void *)ul_dcch_msg);
        }

        // confirm with PDCP about the security mode for DCCH
        //rrc_pdcp_config_req (enb_mod_idP, frameP, 1,CONFIG_ACTION_SET_SECURITY_MODE, (ue_mod_idP * NB_RB_MAX) + DCCH, 0x77);
        // continue the procedure
        rrc_eNB_generate_UECapabilityEnquiry(
          ctxt_pP,
          ue_context_p);
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_securityModeFailure:
        T(T_ENB_RRC_SECURITY_MODE_FAILURE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC Security Mode Failure\n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" securityModeFailure UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);
        LOG_W(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(securityModeFailure) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
          xer_fprint(stdout, &asn_DEF_LTE_UL_DCCH_Message, (void *)ul_dcch_msg);
        }

        // cancel the security mode in PDCP
        // followup with the remaining procedure
        //#warning "LG Removed rrc_eNB_generate_UECapabilityEnquiry after receiving securityModeFailure"
        rrc_eNB_generate_UECapabilityEnquiry(ctxt_pP, ue_context_p);
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_ueCapabilityInformation:
        T(T_ENB_RRC_UE_CAPABILITY_INFORMATION, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing ueCapabilityInformation UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC UECapablility Information\n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" ueCapabilityInformation UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);
        LOG_I(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" received ueCapabilityInformation on UL-DCCH %d from UE\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH);
        LOG_D(RRC,
              PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
              "(UECapabilityInformation) ---> RRC_eNB\n",
              PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
              DCCH,
              sdu_sizeP);

        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
          xer_fprint(stdout, &asn_DEF_LTE_UL_DCCH_Message, (void *)ul_dcch_msg);
        }

        LOG_I(RRC, "got UE capabilities for UE %x\n", ctxt_pP->rnti);
        int eutra_index = -1;

        for (i = 0; i < ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.count; i++) {
          if (ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->rat_Type ==
              LTE_RAT_Type_nr) {
            if(ue_context_p->ue_context.UE_Capability_nr) {
              ASN_STRUCT_FREE(asn_DEF_NR_UE_NR_Capability,ue_context_p->ue_context.UE_Capability_nr);
              ue_context_p->ue_context.UE_Capability_nr = 0;
            }

            dec_rval = uper_decode(NULL,
                                   &asn_DEF_NR_UE_NR_Capability,
                                   (void **)&ue_context_p->ue_context.UE_Capability_nr,
                                   ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.buf,
                                   ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.size,
                                   0, 0);

            if (LOG_DEBUGFLAG(DEBUG_ASN1)) {
              xer_fprint(stdout, &asn_DEF_NR_UE_NR_Capability, ue_context_p->ue_context.UE_Capability_nr);
            }

            if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
              LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Failed to decode nr UE capabilities (%zu bytes)\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),dec_rval.consumed);
              ASN_STRUCT_FREE(asn_DEF_NR_UE_NR_Capability,ue_context_p->ue_context.UE_Capability_nr);
              ue_context_p->ue_context.UE_Capability_nr = 0;
            }

            ue_context_p->ue_context.UE_NR_Capability_size =
              ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.size;
          }

          if (ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->rat_Type ==
              LTE_RAT_Type_eutra_nr) {
            if(ue_context_p->ue_context.UE_Capability_MRDC) {
              ASN_STRUCT_FREE(asn_DEF_NR_UE_MRDC_Capability,ue_context_p->ue_context.UE_Capability_MRDC);
              ue_context_p->ue_context.UE_Capability_MRDC = 0;
            }

            dec_rval = uper_decode(NULL,
                                   &asn_DEF_NR_UE_MRDC_Capability,
                                   (void **)&ue_context_p->ue_context.UE_Capability_MRDC,
                                   ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.buf,
                                   ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.size,
                                   0, 0);

            if (LOG_DEBUGFLAG(DEBUG_ASN1)) {
              xer_fprint(stdout, &asn_DEF_NR_UE_MRDC_Capability, ue_context_p->ue_context.UE_Capability_MRDC);
            }

            if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
              LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Failed to decode MRDC UE capabilities (%zu bytes)\n",
                    PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),dec_rval.consumed);
              ASN_STRUCT_FREE(asn_DEF_NR_UE_MRDC_Capability,ue_context_p->ue_context.UE_Capability_MRDC);
              ue_context_p->ue_context.UE_Capability_MRDC = 0;
            }

            ue_context_p->ue_context.UE_MRDC_Capability_size =
              ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->ueCapabilityRAT_Container.size;
          }

          if (ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[i]->rat_Type ==
              LTE_RAT_Type_eutra) {
            if (eutra_index != -1) {
              LOG_E(RRC, "fatal: more than 1 eutra capability\n");
              exit(1);
            }

            eutra_index = i;
          }
        }

        /* do nothing if no EUTRA capabilities (TODO: may be NR capabilities, to be processed somehow) */
        if (eutra_index == -1)
          break;

        if (ue_context_p->ue_context.UE_Capability) {
          LOG_I(RRC, "freeing old UE capabilities for UE %x\n", ctxt_pP->rnti);
          ASN_STRUCT_FREE(asn_DEF_LTE_UE_EUTRA_Capability,
                          ue_context_p->ue_context.UE_Capability);
          ue_context_p->ue_context.UE_Capability = 0;
        }

        dec_rval = uper_decode(NULL,
                               &asn_DEF_LTE_UE_EUTRA_Capability,
                               (void **)&ue_context_p->ue_context.UE_Capability,
                               ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[eutra_index]->ueCapabilityRAT_Container.buf,
                               ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[eutra_index]->ueCapabilityRAT_Container.size,
                               0, 0);
        ue_context_p->ue_context.UE_Capability_size =
          ul_dcch_msg->message.choice.c1.choice.ueCapabilityInformation.criticalExtensions.choice.c1.choice.ueCapabilityInformation_r8.ue_CapabilityRAT_ContainerList.list.array[eutra_index]->ueCapabilityRAT_Container.size;

        if ( LOG_DEBUGFLAG(DEBUG_ASN1) ) {
          xer_fprint(stdout, &asn_DEF_LTE_UE_EUTRA_Capability, ue_context_p->ue_context.UE_Capability);
        }

        if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
          LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Failed to decode UE capabilities (%zu bytes)\n",
                PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
                dec_rval.consumed);
          ASN_STRUCT_FREE(asn_DEF_LTE_UE_EUTRA_Capability,
                          ue_context_p->ue_context.UE_Capability);
          ue_context_p->ue_context.UE_Capability = 0;
        }

        if (dec_rval.code == RC_OK)
          ue_context_p->ue_context.does_nr = is_en_dc_supported(ue_context_p->ue_context.UE_Capability);

        if (EPC_MODE_ENABLED) {
          rrc_eNB_send_S1AP_UE_CAPABILITIES_IND(ctxt_pP,
                                                ue_context_p,
                                                ul_dcch_msg);
        } else {
          ue_context_p->ue_context.nb_of_e_rabs = 1;

          for (i = 0; i < ue_context_p->ue_context.nb_of_e_rabs; i++) {
            ue_context_p->ue_context.e_rab[i].status = E_RAB_STATUS_NEW;
            ue_context_p->ue_context.e_rab[i].param.e_rab_id = 1+i;
            ue_context_p->ue_context.e_rab[i].param.qos.qci=9;
          }

          ue_context_p->ue_context.setup_e_rabs =ue_context_p->ue_context.nb_of_e_rabs;
        }

        rrc_eNB_generate_defaultRRCConnectionReconfiguration(ctxt_pP,
            ue_context_p,
            RC.rrc[ctxt_pP->module_id]->HO_flag);
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_ulHandoverPreparationTransfer:
        T(T_ENB_RRC_UL_HANDOVER_PREPARATION_TRANSFER, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_ulInformationTransfer:
        T(T_ENB_RRC_UL_INFORMATION_TRANSFER, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));

        // to avoid segmentation fault
        if(!ue_context_p) {
          LOG_I(RRC, "Processing ulInformationTransfer UE %x, ue_context_p is NULL\n", ctxt_pP->rnti);
          break;
        }

        LOG_D(RRC,"[MSG] RRC UL Information Transfer \n");
        LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC UL Information Transfer \n");
        MSC_LOG_RX_MESSAGE(
          MSC_RRC_ENB,
          MSC_RRC_UE,
          Rx_sdu,
          sdu_sizeP,
          MSC_AS_TIME_FMT" ulInformationTransfer UE %x size %u",
          MSC_AS_TIME_ARGS(ctxt_pP),
          ue_context_p->ue_context.rnti,
          sdu_sizeP);

        if (EPC_MODE_ENABLED == 1) {
          rrc_eNB_send_S1AP_UPLINK_NAS(ctxt_pP,
                                       ue_context_p,
                                       ul_dcch_msg);
        }

        break;

      case LTE_UL_DCCH_MessageType__c1_PR_counterCheckResponse:
        T(T_ENB_RRC_COUNTER_CHECK_RESPONSE, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_ueInformationResponse_r9:
        T(T_ENB_RRC_UE_INFORMATION_RESPONSE_R9, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_proximityIndication_r9:
        T(T_ENB_RRC_PROXIMITY_INDICATION_R9, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_rnReconfigurationComplete_r10:
        T(T_ENB_RRC_RECONFIGURATION_COMPLETE_R10, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_mbmsCountingResponse_r10:
        T(T_ENB_RRC_MBMS_COUNTING_RESPONSE_R10, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
        break;

      case LTE_UL_DCCH_MessageType__c1_PR_interFreqRSTDMeasurementIndication_r10:
        T(T_ENB_RRC_INTER_FREQ_RSTD_MEASUREMENT_INDICATION, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->frame),
          T_INT(ctxt_pP->subframe), T_INT(ctxt_pP->rnti));
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
    //TTN for D2D
  } else if (ul_dcch_msg->message.present == LTE_UL_DCCH_MessageType_PR_messageClassExtension) {
    LOG_I(RRC, "[LTE_UL_DCCH_MessageType_PR_messageClassExtension]\n");

    switch (ul_dcch_msg->message.choice.messageClassExtension.present) {
      case LTE_UL_DCCH_MessageType__messageClassExtension_PR_NOTHING: /* No components present */
        break;

      case LTE_UL_DCCH_MessageType__messageClassExtension_PR_c2: //SidelinkUEInformation
        if(ul_dcch_msg->message.choice.messageClassExtension.choice.c2.present ==
          LTE_UL_DCCH_MessageType__messageClassExtension__c2_PR_scgFailureInformationNR_r15){
          if (ul_dcch_msg->message.choice.messageClassExtension.choice.c2.choice.scgFailureInformationNR_r15.
            criticalExtensions.present == LTE_SCGFailureInformationNR_r15__criticalExtensions_PR_c1){
            if (ul_dcch_msg->message.choice.messageClassExtension.choice.c2.choice.scgFailureInformationNR_r15.criticalExtensions.
              choice.c1.present == LTE_SCGFailureInformationNR_r15__criticalExtensions__c1_PR_scgFailureInformationNR_r15){
              if (ul_dcch_msg->message.choice.messageClassExtension.choice.c2.choice.scgFailureInformationNR_r15.criticalExtensions.
                choice.c1.choice.scgFailureInformationNR_r15.failureReportSCG_NR_r15!=NULL) {
                LOG_E(RRC, "Received NR scgFailureInformation from UE, failure type: %ld \n",
                  ul_dcch_msg->message.choice.messageClassExtension.choice.c2.choice.scgFailureInformationNR_r15.criticalExtensions.
                  choice.c1.choice.scgFailureInformationNR_r15.failureReportSCG_NR_r15->failureType_r15);
                xer_fprint(stdout, &asn_DEF_LTE_UL_DCCH_Message, (void *)ul_dcch_msg);
              }
            }
          }
        }
        else if(ul_dcch_msg->message.choice.messageClassExtension.choice.c2.present == LTE_UL_DCCH_MessageType__messageClassExtension__c2_PR_sidelinkUEInformation_r12){
          //case UL_DCCH_MessageType__messageClassExtension__c2_PR_sidelinkUEInformation_r12: //SidelinkUEInformation
          LOG_I(RRC,"THINH [LTE_UL_DCCH_MessageType__messageClassExtension_PR_c2]\n");
          LOG_DUMPMSG(RRC,DEBUG_RRC,(char *)Rx_sdu,sdu_sizeP,
                    "[MSG] RRC SidelinkUEInformation \n");
          MSC_LOG_RX_MESSAGE(
              MSC_RRC_ENB,
              MSC_RRC_UE,
              Rx_sdu,
              sdu_sizeP,
              MSC_AS_TIME_FMT" SidelinkUEInformation UE %x size %u",
              MSC_AS_TIME_ARGS(ctxt_pP),
              ue_context_p->ue_context.rnti,
              sdu_sizeP);
          LOG_I(RRC,
            PROTOCOL_RRC_CTXT_UE_FMT" RLC RB %02d --- RLC_DATA_IND %d bytes "
            "(SidelinkUEInformation) ---> RRC_eNB\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
            DCCH,
            sdu_sizeP);
          rrc_eNB_process_SidelinkUEInformation(
            ctxt_pP,
            ue_context_p,
            &ul_dcch_msg->message.choice.messageClassExtension.choice.c2.choice.sidelinkUEInformation_r12);
          }
        break;

      default:
        break;
    }

    //end TTN
  } else {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Unknown error %s:%u\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP),
          __FILE__, __LINE__);
    return -1;
  }

  return 0;
}

void rrc_eNB_reconfigure_DRBs (const protocol_ctxt_t *const ctxt_pP,
                               rrc_eNB_ue_context_t  *ue_context_pP) {
  int i;
  int e_rab_done=0;

  for (i = 0;
       i < 3;//NB_RB_MAX - 3;  // S1AP_MAX_E_RAB
       i++) {
    if ( ue_context_pP->ue_context.e_rab[i].status < E_RAB_STATUS_DONE) {
      ue_context_pP->ue_context.e_rab[i].status = E_RAB_STATUS_NEW;
      ue_context_pP->ue_context.e_rab[i].param.e_rab_id = i + 1;
      ue_context_pP->ue_context.e_rab[i].param.qos.qci = i % 9;
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.priority_level= i % PRIORITY_LEVEL_LOWEST;
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_capability= PRE_EMPTION_CAPABILITY_DISABLED;
      ue_context_pP->ue_context.e_rab[i].param.qos.allocation_retention_priority.pre_emp_vulnerability= PRE_EMPTION_VULNERABILITY_DISABLED;
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.buffer = NULL;
      ue_context_pP->ue_context.e_rab[i].param.nas_pdu.length = 0;
      //  memset (ue_context_pP->ue_context.e_rab[i].param.sgw_addr.buffer,0,20);
      ue_context_pP->ue_context.e_rab[i].param.sgw_addr.length = 0;
      ue_context_pP->ue_context.e_rab[i].param.gtp_teid=0;
      ue_context_pP->ue_context.nb_of_e_rabs++;
      e_rab_done++;
      LOG_I(RRC,"setting up the dedicated DRBs %d (index %d) status %d \n",
            ue_context_pP->ue_context.e_rab[i].param.e_rab_id, i, ue_context_pP->ue_context.e_rab[i].status);
    }
  }

  ue_context_pP->ue_context.setup_e_rabs+=e_rab_done;
  rrc_eNB_generate_dedicatedRRCConnectionReconfiguration(ctxt_pP, ue_context_pP, 0);
}

void handle_f1_setup_req(f1ap_setup_req_t *f1_setup_req) {
  LOG_I(RRC,"Received F1 Setup Request from gNB_DU %llu (%s)\n",(unsigned long long int)f1_setup_req->gNB_DU_id,f1_setup_req->gNB_DU_name);
  //uint16_t num_cells_to_activate = 0;
  int cu_cell_ind=0;
  MessageDef                         *msg_p = NULL;

  //LOG_W(RRC,"num_cells_available %d \n", f1_setup_req->num_cells_available);
  for (int i=0; i<f1_setup_req->num_cells_available; i++) {
    // check that mcc/mnc match and grab MIB/SIB1
    int found_cell=0;

    for (int j=0; j<RC.nb_inst; j++) {
      eNB_RRC_INST *rrc = RC.rrc[j];

      if (rrc->configuration.mcc[0] == f1_setup_req->mcc[i] &&
          rrc->configuration.mnc[0] == f1_setup_req->mnc[i] &&
          rrc->nr_cellid == f1_setup_req->nr_cellid[i]) {
        // check that CU rrc instance corresponds to mcc/mnc/cgi (normally cgi should be enough, but just in case)
        rrc->carrier[0].MIB = malloc(f1_setup_req->mib_length[i]);
        rrc->carrier[0].sizeof_MIB = f1_setup_req->mib_length[i];
        LOG_W(RRC, "instance %d mib length %d\n", i, f1_setup_req->mib_length[i]);
        LOG_W(RRC, "instance %d sib1 length %d\n", i, f1_setup_req->sib1_length[i]);
        memcpy((void *)rrc->carrier[0].MIB,f1_setup_req->mib[i],f1_setup_req->mib_length[i]);
        asn_dec_rval_t dec_rval = uper_decode_complete(NULL,
                                  &asn_DEF_LTE_BCCH_BCH_Message,
                                  (void **)&rrc->carrier[0].mib_DU,
                                  f1_setup_req->mib[i],
                                  f1_setup_req->mib_length[i]);
        AssertFatal(dec_rval.code == RC_OK,
                    "[eNB_DU %"PRIu8"] Failed to decode LTE_BCCH_BCH_MESSAGE (%zu bits)\n",
                    j,
                    dec_rval.consumed );
        LTE_BCCH_BCH_Message_t *mib = &rrc->carrier[0].mib;
        LTE_BCCH_BCH_Message_t *mib_DU = rrc->carrier[0].mib_DU;
        mib->message.dl_Bandwidth = mib_DU->message.dl_Bandwidth;
        mib->message.phich_Config.phich_Resource = mib_DU->message.phich_Config.phich_Resource;
        mib->message.phich_Config.phich_Duration = mib_DU->message.phich_Config.phich_Duration;
        rrc->carrier[0].SIB1 = malloc(f1_setup_req->sib1_length[i]);
        rrc->carrier[0].sizeof_SIB1 = f1_setup_req->sib1_length[i];
        memcpy((void *)rrc->carrier[0].SIB1,f1_setup_req->sib1[i],f1_setup_req->sib1_length[i]);
        dec_rval = uper_decode_complete(NULL,
                                        &asn_DEF_LTE_BCCH_DL_SCH_Message,
                                        (void **)&rrc->carrier[0].siblock1_DU,
                                        f1_setup_req->sib1[i],
                                        f1_setup_req->sib1_length[i]);
        AssertFatal(dec_rval.code == RC_OK,
                    "[eNB_DU %"PRIu8"] Failed to decode LTE_BCCH_DLSCH_MESSAGE (%zu bits)\n",
                    j,
                    dec_rval.consumed );
        // Parse message and extract SystemInformationBlockType1 field
        LTE_BCCH_DL_SCH_Message_t *bcch_message = rrc->carrier[0].siblock1_DU;
        AssertFatal(bcch_message->message.present == LTE_BCCH_DL_SCH_MessageType_PR_c1,
                    "bcch_message->message.present != LTE_BCCH_DL_SCH_MessageType_PR_c1\n");
        AssertFatal(bcch_message->message.choice.c1.present == LTE_BCCH_DL_SCH_MessageType__c1_PR_systemInformationBlockType1,
                    "bcch_message->message.choice.c1.present != LTE_BCCH_DL_SCH_MessageType__c1_PR_systemInformationBlockType1\n");
        rrc->carrier[0].sib1 = &bcch_message->message.choice.c1.choice.systemInformationBlockType1;
        rrc->carrier[0].physCellId = f1_setup_req->nr_pci[i];
        // prepare F1_SETUP_RESPONSE

        if (msg_p == NULL) {
          msg_p = itti_alloc_new_message (TASK_CU_F1,F1AP_SETUP_RESP);
        }

        F1AP_SETUP_RESP (msg_p).gNB_CU_name                                = rrc->node_name;
        F1AP_SETUP_RESP (msg_p).mcc[cu_cell_ind]                           = rrc->configuration.mcc[0];
        F1AP_SETUP_RESP (msg_p).mnc[cu_cell_ind]                           = rrc->configuration.mnc[0];
        F1AP_SETUP_RESP (msg_p).mnc_digit_length[cu_cell_ind]              = rrc->configuration.mnc_digit_length[0];
        F1AP_SETUP_RESP (msg_p).nr_cellid[cu_cell_ind]                     = rrc->nr_cellid;
        F1AP_SETUP_RESP (msg_p).nrpci[cu_cell_ind]                         = f1_setup_req->nr_pci[i];
        int num_SI= 0;

        if (rrc->carrier[0].SIB23) {
          F1AP_SETUP_RESP (msg_p).SI_container[cu_cell_ind][num_SI]        = rrc->carrier[0].SIB23;
          F1AP_SETUP_RESP (msg_p).SI_container_length[cu_cell_ind][num_SI] = rrc->carrier[0].sizeof_SIB23;
          //printf("SI %d size %d: ", 0, F1AP_SETUP_RESP(msg_p).SI_container_length[j][num_SI]);
          //for (int n = 0; n < F1AP_SETUP_RESP(msg_p).SI_container_length[j][num_SI]; n++)
          //  printf("%02x ", F1AP_SETUP_RESP(msg_p).SI_container[0][num_SI][n]);
          //printf("\n");
          num_SI++;
        }

        F1AP_SETUP_RESP (msg_p).num_SI[cu_cell_ind] = num_SI;
        cu_cell_ind++;
        found_cell=1;
        F1AP_SETUP_RESP (msg_p).num_cells_to_activate = cu_cell_ind;
        // send ITTI message to F1AP-CU task
        itti_send_msg_to_task (TASK_CU_F1, ENB_MODULE_ID_TO_INSTANCE(j), msg_p);
        break;
      } else {// setup_req mcc/mnc match rrc internal list element
        LOG_W(RRC,"[Inst %d] No matching MCC/MNC: rrc->mcc/f1_setup_req->mcc %d/%d rrc->mnc/f1_setup_req->mnc %d/%d \n",
              j, rrc->configuration.mcc[0], f1_setup_req->mcc[i],rrc->configuration.mnc[0], f1_setup_req->mnc[i]);
      }
    }// for (int j=0;j<RC.nb_inst;j++)

    if (found_cell==0) {
      AssertFatal(1==0,"No cell found\n");
      /*msg_p = itti_alloc_new_message (TASK_CU_F1,F1AP_SETUP_FAILURE);
      F1AP_SETUP_RESP (msg_p).cause                             = rrc->node_name;
      F1AP_SETUP_RESP (msg_p).time_to_wait                      = rrc->node_id;
      F1AP_SETUP_RESP (msg_p).criticality_diagnostics           = rrc->node_name;*/
    }

    // handle other failure cases
  }//for (int i=0;i<f1_setup_req->num_cells_available;i++)
}


// ignore 5GNR fields for now, just take MIB and SIB1
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
void rrc_enb_init(void) {
  pthread_mutex_init(&lock_ue_freelist, NULL);
  pthread_mutex_init(&rrc_release_freelist, NULL);
  memset(&rrc_release_info,0,sizeof(RRC_release_list_t));
#ifdef ENABLE_RIC_AGENT
  rrc_kpi_stats.rrc_conn_estab_att_sum = 0;
  rrc_kpi_stats.rrc_conn_estab_succ_sum = 0;
  rrc_kpi_stats.rrc_conn_reestab_att_sum = 0;
  rrc_kpi_stats.rrc_conn_mean = 0;
  rrc_kpi_stats.rrc_conn_max = 0;
#endif        
}

//-----------------------------------------------------------------------------
void process_successful_rlc_sdu_indication(int instance,
                                           int rnti,
                                           int message_id)
{
  int release_num;
  int release_total;
  RRC_release_ctrl_t *release_ctrl;

  /* Check if the message sent was RRC Connection Release.
   * If yes then continue the release process.
   */

  pthread_mutex_lock(&rrc_release_freelist);

  if (rrc_release_info.num_UEs > 0) {
    release_total = 0;

    for (release_num = 0, release_ctrl = &rrc_release_info.RRC_release_ctrl[0];
         release_num < NUMBER_OF_UE_MAX;
         release_num++, release_ctrl++) {
      if(release_ctrl->flag > 0) {
        release_total++;
      } else {
        continue;
      }

      if (release_ctrl->flag == 1 && release_ctrl->rnti == rnti && release_ctrl->rrc_eNB_mui == message_id) {
        release_ctrl->flag = 3;
        LOG_D(MAC,"DLSCH Release send:index %d rnti %x mui %d flag 1->3\n",
              release_num,
              rnti,
              message_id);
        break;
      }

      if (release_ctrl->flag == 2 && release_ctrl->rnti == rnti && release_ctrl->rrc_eNB_mui == message_id) {
        release_ctrl->flag = 4;
        LOG_D(MAC, "DLSCH Release send:index %d rnti %x mui %d flag 2->4\n",
              release_num,
              rnti,
              message_id);
        break;
      }

      if(release_total >= rrc_release_info.num_UEs)
        break;
    }
  }

  pthread_mutex_unlock(&rrc_release_freelist);
}

//-----------------------------------------------------------------------------
void process_unsuccessful_rlc_sdu_indication(int instance, int rnti)
{
  int release_num;
  int release_total;
  RRC_release_ctrl_t *release_ctrl;

  /* radio link failure detected by RLC layer, remove UE properly */

  pthread_mutex_lock(&rrc_release_freelist);

  /* first, check if the rnti is in the list rrc_release_info.RRC_release_ctrl */

  if (rrc_release_info.num_UEs > 0) {
    release_total = 0;

    for (release_num = 0, release_ctrl = &rrc_release_info.RRC_release_ctrl[0];
         release_num < NUMBER_OF_UE_MAX;
         release_num++, release_ctrl++) {
      if(release_ctrl->flag > 0) {
        release_total++;
      } else {
        continue;
      }

      if (release_ctrl->flag == 1 && release_ctrl->rnti == rnti) {
        release_ctrl->flag = 3;
        LOG_D(MAC,"DLSCH Release send:index %d rnti %x flag 1->3\n",
              release_num,
              rnti);
        goto done;
      }

      if (release_ctrl->flag == 2 && release_ctrl->rnti == rnti) {
        release_ctrl->flag = 4;
        LOG_D(MAC, "DLSCH Release send:index %d rnti %x flag 2->4\n",
              release_num,
              rnti);
        goto done;
      }

      if(release_total >= rrc_release_info.num_UEs)
        break;
    }
  }

  /* it's not in the list, put it with flag = 4 */
  for (release_num = 0; release_num < NUMBER_OF_UE_MAX; release_num++) {
    if (rrc_release_info.RRC_release_ctrl[release_num].flag == 0) {
      rrc_release_info.RRC_release_ctrl[release_num].flag = 4;
      rrc_release_info.RRC_release_ctrl[release_num].rnti = rnti;
      rrc_release_info.RRC_release_ctrl[release_num].rrc_eNB_mui = -1;     /* not defined */
      rrc_release_info.num_UEs++;
      LOG_I(RRC, "RLC-MAX-RETX [RLF detected]: index %d rnti %x flag %d RRC Rel UEs %d\n",
            release_num,
            rnti,
            rrc_release_info.RRC_release_ctrl[release_num].flag,
        rrc_release_info.num_UEs);
      break;
    }
  }

  /* TODO: what to do if rrc_release_info.RRC_release_ctrl is full? */
  if (release_num == NUMBER_OF_UE_MAX) {
    LOG_E(RRC, "fatal: radio link failure: rrc_release_info.RRC_release_ctrl is full\n");
    exit(1);
  }

done:
  pthread_mutex_unlock(&rrc_release_freelist);
}

//-----------------------------------------------------------------------------
void process_rlc_sdu_indication(int instance,
                                int rnti,
                                int is_successful,
                                int srb_id,
                                int message_id)
{
  if (is_successful)
    process_successful_rlc_sdu_indication(instance, rnti, message_id);
  else
    process_unsuccessful_rlc_sdu_indication(instance, rnti);
}

//-----------------------------------------------------------------------------
int add_ue_to_remove(struct rrc_eNB_ue_context_s **ue_to_be_removed,
                     int removed_ue_count,
                     struct rrc_eNB_ue_context_s *ue_context_p) {
  int i;

  /* is it already here? */
  for (i = 0; i < removed_ue_count; i++)
    if (ue_to_be_removed[i] == ue_context_p)
      return removed_ue_count;

  if (removed_ue_count == NUMBER_OF_UE_MAX) {
    LOG_E(RRC, "fatal: ue_to_be_removed is full\n");
    exit(1);
  }

  ue_to_be_removed[removed_ue_count] = ue_context_p;
  removed_ue_count++;
  return removed_ue_count;
}

//-----------------------------------------------------------------------------
void rrc_subframe_process(protocol_ctxt_t *const ctxt_pP, const int CC_id) 
{
  int32_t current_timestamp_ms = 0;
  int32_t ref_timestamp_ms = 0;
  struct timeval ts;
  struct rrc_eNB_ue_context_s *ue_context_p = NULL;
  struct rrc_eNB_ue_context_s *ue_to_be_removed[NUMBER_OF_UE_MAX];
  int removed_ue_count = 0;
  int cur_ue;
#ifdef LOCALIZATION
  double estimated_distance = 0;
  protocol_ctxt_t                     ctxt;
#endif
  MessageDef *msg;
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX, VCD_FUNCTION_IN);

  if (RC.rrc[ENB_INSTANCE_TO_MODULE_ID(ctxt_pP->instance)]->configuration.enable_x2) {
    /* send a tick to x2ap */
    msg = itti_alloc_new_message(TASK_RRC_ENB, X2AP_SUBFRAME_PROCESS);
    itti_send_msg_to_task(TASK_X2AP, ctxt_pP->module_id, msg);
    check_handovers(ctxt_pP); // counter, get the value and aggregate
  }

  // check for UL failure or for UE to be released
  RB_FOREACH(ue_context_p, rrc_ue_tree_s, &(RC.rrc[ctxt_pP->module_id]->rrc_ue_head)) {
    ctxt_pP->rnti = ue_context_p->ue_id_rnti;

    if ((ctxt_pP->frame == 0) && (ctxt_pP->subframe == 0)) {
      if (ue_context_p->ue_context.Initialue_identity_s_TMSI.presence == TRUE) {
        LOG_I(RRC, "UE rnti %x: S-TMSI %x failure timer %d/8\n",
              ue_context_p->ue_context.rnti,
              ue_context_p->ue_context.Initialue_identity_s_TMSI.m_tmsi,
              ue_context_p->ue_context.ul_failure_timer);
      } else {
        LOG_I(RRC, "UE rnti %x failure timer %d/8\n",
              ue_context_p->ue_context.rnti,
              ue_context_p->ue_context.ul_failure_timer);
      }
    }

    if (ue_context_p->ue_context.ul_failure_timer > 0) {
      ue_context_p->ue_context.ul_failure_timer++;

      if (ue_context_p->ue_context.ul_failure_timer >= 20000) {
        // remove UE after 20 seconds after MAC (or else) has indicated UL failure
        LOG_I(RRC, "Removing UE %x instance, because of uplink failure timer timeout\n",
              ue_context_p->ue_context.rnti);
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        break; // break RB_FOREACH
      }
    }

    if (ue_context_p->ue_context.ue_release_timer_s1 > 0) {
      ue_context_p->ue_context.ue_release_timer_s1++;

      if (ue_context_p->ue_context.ue_release_timer_s1 >= ue_context_p->ue_context.ue_release_timer_thres_s1) {
        LOG_I(RRC, "Removing UE %x instance, because of UE_CONTEXT_RELEASE_COMMAND not received after %d ms from sending request\n",
              ue_context_p->ue_context.rnti,
              ue_context_p->ue_context.ue_release_timer_thres_s1);

        if (EPC_MODE_ENABLED && !NODE_IS_DU(RC.rrc[ctxt_pP->module_id]->node_type))
          rrc_eNB_generate_RRCConnectionRelease(ctxt_pP, ue_context_p);
        else
          removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);

        ue_context_p->ue_context.ue_release_timer_s1 = 0;
        break; // break RB_FOREACH
      } // end if timer_s1 timeout
    } // end if timer_s1 > 0 (S1 UE_CONTEXT_RELEASE_REQ ongoing)

    if (ue_context_p->ue_context.ue_release_timer_rrc > 0) {
      ue_context_p->ue_context.ue_release_timer_rrc++;

      if (ue_context_p->ue_context.ue_release_timer_rrc >= ue_context_p->ue_context.ue_release_timer_thres_rrc) {
        LOG_I(RRC, "Removing UE %x instance after UE_CONTEXT_RELEASE_Complete (ue_release_timer_rrc timeout)\n",
              ue_context_p->ue_context.rnti);
        ue_context_p->ue_context.ue_release_timer_rrc = 0;
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        break; // break RB_FOREACH
      }
    }

    if (ue_context_p->ue_context.handover_info != NULL) {
      if (ue_context_p->ue_context.handover_info->state == HO_RELEASE) {
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        rrc_eNB_handover_ue_context_release(ctxt_pP, ue_context_p);
        break; //break RB_FOREACH (why to break ?)
      }

      if (ue_context_p->ue_context.handover_info->state == HO_CANCEL) {
        rrc_eNB_handover_cancel(ctxt_pP, ue_context_p);
        /* freeing handover_info and setting it to NULL to let
         * RRC wait for MME to later on release the UE
         */
        free(ue_context_p->ue_context.handover_info);
        ue_context_p->ue_context.handover_info = NULL;
      }
    }

    pthread_mutex_lock(&rrc_release_freelist);

    if (rrc_release_info.num_UEs > 0) {
      uint16_t release_total = 0;

      for (uint16_t release_num = 0; release_num < NUMBER_OF_UE_MAX; release_num++) {
        if (rrc_release_info.RRC_release_ctrl[release_num].flag > 0) {
          release_total++;
        }

        if ((rrc_release_info.RRC_release_ctrl[release_num].flag > 2) &&
            (rrc_release_info.RRC_release_ctrl[release_num].rnti == ue_context_p->ue_context.rnti)) 
        {

          if (ue_context_p->ue_context.ue_release_timer_rrc == 0) /*setting ue_release_timer_rrc for the first time only */
          {
            ue_context_p->ue_context.ue_release_timer_rrc = 1;
            ue_context_p->ue_context.ue_release_timer_thres_rrc = 100;

            if ( (NODE_IS_DU(RC.rrc[ctxt_pP->module_id]->node_type)) &&
                   (rrc_release_info.RRC_release_ctrl[release_num].rrc_eNB_mui == -1) )
              {
                /* This will trigger F1AP context cleanup procedure at DU & CU, only in case of RLC-MAX-RETX */
                MessageDef *m = itti_alloc_new_message(TASK_MAC_ENB, F1AP_UE_CONTEXT_RELEASE_REQ);
                F1AP_UE_CONTEXT_RELEASE_REQ(m).rnti = rrc_release_info.RRC_release_ctrl[release_num].rnti;
                F1AP_UE_CONTEXT_RELEASE_REQ(m).cause = F1AP_CAUSE_RADIO_NETWORK;
                F1AP_UE_CONTEXT_RELEASE_REQ(m).cause_value = 1; // 1 = F1AP_CauseRadioNetwork_rl_failure
                F1AP_UE_CONTEXT_RELEASE_REQ(m).rrc_container = NULL;
                F1AP_UE_CONTEXT_RELEASE_REQ(m).rrc_container_length = 0;
                itti_send_msg_to_task(TASK_DU_F1, ctxt_pP->module_id, m);
                
                LOG_I(RRC, "[RLC-MAX-RETX]*** rnti %x: F1AP_UE_CONTEXT_RELEASE_REQ ==> RLF ********** \n",
                              rrc_release_info.RRC_release_ctrl[release_num].rnti);
              }
          }
                
          LOG_D(RRC, "[%s,%u] Rel_num %d RNTI %x,%x:Flag %d ue_release_timer_rrc %d\n",
            __func__,__LINE__, release_num, rrc_release_info.RRC_release_ctrl[release_num].rnti,
            ue_context_p->ue_context.rnti, rrc_release_info.RRC_release_ctrl[release_num].flag,
            ue_context_p->ue_context.ue_release_timer_rrc);

          if (EPC_MODE_ENABLED && !NODE_IS_DU(RC.rrc[ctxt_pP->module_id]->node_type)) {
            if (rrc_release_info.RRC_release_ctrl[release_num].flag == 4) { // if timer_s1 == 0
              rrc_eNB_send_S1AP_UE_CONTEXT_RELEASE_CPLT(ctxt_pP->module_id,
                  ue_context_p->ue_context.eNB_ue_s1ap_id);
            }

            rrc_eNB_send_GTPV1U_ENB_DELETE_TUNNEL_REQ(ctxt_pP->module_id,
                ue_context_p);

            // erase data of GTP tunnels in UE context
            for (int e_rab = 0; e_rab < ue_context_p->ue_context.nb_of_e_rabs; e_rab++) {
              ue_context_p->ue_context.enb_gtp_teid[e_rab] = 0;
              memset(&ue_context_p->ue_context.enb_gtp_addrs[e_rab],
                     0, sizeof(ue_context_p->ue_context.enb_gtp_addrs[e_rab]));
              ue_context_p->ue_context.enb_gtp_ebi[e_rab]  = 0;
            }

            struct rrc_ue_s1ap_ids_s *rrc_ue_s1ap_ids = NULL;

            rrc_ue_s1ap_ids = rrc_eNB_S1AP_get_ue_ids(RC.rrc[ctxt_pP->module_id], 0,
                              ue_context_p->ue_context.eNB_ue_s1ap_id);

            if (rrc_ue_s1ap_ids != NULL) {
              rrc_eNB_S1AP_remove_ue_ids(RC.rrc[ctxt_pP->module_id], rrc_ue_s1ap_ids);
            }
          } /* EPC_MODE_ENABLED && !NODE_IS_DU */

          rrc_release_info.RRC_release_ctrl[release_num].flag = 0;
          rrc_release_info.num_UEs--;
          break; // break for (release_num)
        } // end if ((rrc_release_info.RRC_release_ctrl[release_num].flag > 2) && ...

        if (release_total >= rrc_release_info.num_UEs) {
          break; // break for (release_num)
        }
      } // end for (release_num)
    } // end if (rrc_release_info.num_UEs > 0)

    pthread_mutex_unlock(&rrc_release_freelist);

    if ((ue_context_p->ue_context.ue_rrc_inactivity_timer > 0) && (RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres > 0)) {
      ue_context_p->ue_context.ue_rrc_inactivity_timer++;

      if (ue_context_p->ue_context.ue_rrc_inactivity_timer >= RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres) {
        LOG_I(RRC, "Removing UE %x instance because of rrc_inactivity_timer timeout\n",
              ue_context_p->ue_context.rnti);
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        break; // break RB_FOREACH
      }
    }

    if (ue_context_p->ue_context.ue_reestablishment_timer > 0) {
      ue_context_p->ue_context.ue_reestablishment_timer++;

      if (ue_context_p->ue_context.ue_reestablishment_timer >= ue_context_p->ue_context.ue_reestablishment_timer_thres) {
        LOG_I(RRC, "Removing UE %x instance because of reestablishment_timer timeout\n",
              ue_context_p->ue_context.rnti);
        ue_context_p->ue_context.ul_failure_timer = 20000; // lead to send S1 UE_CONTEXT_RELEASE_REQ
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        ue_context_p->ue_context.ue_reestablishment_timer = 0;
        break; // break RB_FOREACH
      }
    }

    if (ue_context_p->ue_context.ue_release_timer > 0) {
      ue_context_p->ue_context.ue_release_timer++;

      if (ue_context_p->ue_context.ue_release_timer >= ue_context_p->ue_context.ue_release_timer_thres) {
        LOG_I(RRC, "Removing UE %x instance because of RRC Connection Setup timer timeout\n",
              ue_context_p->ue_context.rnti);
        /*
        * TODO: Naming problem here: ue_release_timer seems to have been used when RRC Connection Release was sent.
        * It is no more the case.
        * The timer should be renamed.
        */
        removed_ue_count = add_ue_to_remove(ue_to_be_removed, removed_ue_count, ue_context_p);
        ue_context_p->ue_context.ue_release_timer = 0;
        break; // break RB_FOREACH
      }
    }
  } // end RB_FOREACH

  for (cur_ue = 0; cur_ue < removed_ue_count; cur_ue++) {
    if ((ue_to_be_removed[cur_ue]->ue_context.ul_failure_timer >= 20000) ||
        ((ue_to_be_removed[cur_ue]->ue_context.ue_rrc_inactivity_timer >= RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres) &&
         (RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres > 0))) {
      ue_to_be_removed[cur_ue]->ue_context.ue_release_timer_s1 = 1;
      ue_to_be_removed[cur_ue]->ue_context.ue_release_timer_thres_s1 = 100;
      ue_to_be_removed[cur_ue]->ue_context.ue_release_timer = 0;
      ue_to_be_removed[cur_ue]->ue_context.ue_reestablishment_timer = 0;
    }

    rrc_eNB_free_UE(ctxt_pP->module_id, ue_to_be_removed[cur_ue]);

    if (ue_to_be_removed[cur_ue]->ue_context.ul_failure_timer >= 20000) {
      ue_to_be_removed[cur_ue]->ue_context.ul_failure_timer = 0;
    }

    if ((ue_to_be_removed[cur_ue]->ue_context.ue_rrc_inactivity_timer >= RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres) &&
        (RC.rrc[ctxt_pP->module_id]->configuration.rrc_inactivity_timer_thres > 0)) {
      ue_to_be_removed[cur_ue]->ue_context.ue_rrc_inactivity_timer = 0; //reset timer after S1 command UE context release request is sent
    }
  }

#ifdef RRC_LOCALIZATION
  /* for the localization, only primary CC_id might be relevant*/
  gettimeofday(&ts, NULL);
  current_timestamp_ms = ts.tv_sec * 1000 + ts.tv_usec / 1000;
  ref_timestamp_ms = RC.rrc[ctxt_pP->module_id]->reference_timestamp_ms;
  RB_FOREACH(ue_context_p, rrc_ue_tree_s, &(RC.rrc[ctxt_pP->module_id]->rrc_ue_head)) {
    ctxt = *ctxt_pP;
    ctxt.rnti = ue_context_p->ue_context.rnti;
    estimated_distance = rrc_get_estimated_ue_distance(&ctxt, CC_id, RC.rrc[ctxt_pP->module_id]->loc_type);

    if ((current_timestamp_ms - ref_timestamp_ms > RC.rrc[ctxt_pP->module_id]->aggregation_period_ms) &&
        estimated_distance != -1) {
      LOG_D(LOCALIZE, "RRC [UE/id %d -> eNB/id %d] timestamp %d frame %d estimated r = %f\n",
            ctxt.rnti,
            ctxt_pP->module_id,
            current_timestamp_ms,
            ctxt_pP->frame,
            estimated_distance);
      LOG_D(LOCALIZE, "RRC status %d\n",
            ue_context_p->ue_context.Status);
      push_front(&RC.rrc[ctxt_pP->module_id]->loc_list, estimated_distance);
      RC.rrc[ctxt_pP->module_id]->reference_timestamp_ms = current_timestamp_ms;
    } // end if
  } // end RB_FOREACH
#endif
  (void)ts; /* remove gcc warning "unused variable" */
  (void)ref_timestamp_ms; /* remove gcc warning "unused variable" */
  (void)current_timestamp_ms; /* remove gcc warning "unused variable" */
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_RX_TX, VCD_FUNCTION_OUT);
}

void rrc_eNB_process_AdditionResponseInformation(const module_id_t enb_mod_idP, x2ap_ENDC_sgnb_addition_req_ACK_t *m) {
  NR_CG_Config_t *CG_Config = NULL;
  {
    int i;
    printf("%d: ", m->rrc_buffer_size);

    for (i=0; i<m->rrc_buffer_size; i++) printf("%2.2x", (unsigned char)m->rrc_buffer[i]);

    printf("\n");
  }
  asn_dec_rval_t dec_rval = uper_decode_complete( NULL,
                            &asn_DEF_NR_CG_Config,
                            (void **)&CG_Config,
                            (uint8_t *)m->rrc_buffer,
                            (int) m->rrc_buffer_size);//m->rrc_buffer_size);

  if ((dec_rval.code != RC_OK) && (dec_rval.consumed == 0)) {
    AssertFatal(1==0,"NR_UL_DCCH_MESSAGE decode error\n");
    // free the memory
    SEQUENCE_free( &asn_DEF_NR_CG_Config, CG_Config, 1 );
    return;
  }

  xer_fprint(stdout,&asn_DEF_NR_CG_Config, CG_Config);
  // recreate enough of X2 EN-DC Container
  AssertFatal(CG_Config->criticalExtensions.choice.c1->present == NR_CG_Config__criticalExtensions__c1_PR_cg_Config,
              "CG_Config not present\n");
  OCTET_STRING_t *scg_CellGroupConfig = NULL;
  OCTET_STRING_t *nr1_conf = NULL;

  if(CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_CellGroupConfig) {
    scg_CellGroupConfig = CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_CellGroupConfig;
#ifdef DEBUG_SCG_CONFIG
    {
      int size_s = CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_CellGroupConfig->size;
      int i;
      LOG_I(RRC, "Dumping scg_CellGroupConfig: %d", size_s);

      for (i=0; i<size_s; i++) printf("%2.2x", (unsigned char)CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_CellGroupConfig->buf[i]);

      printf("\n");
    }
#endif
  } else {
    LOG_W(RRC, "SCG Cell group configuration is not present in the Addition Response message \n");
    return;
  }

  if(CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_RB_Config) {
    nr1_conf = CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_RB_Config;
#ifdef DEBUG_SCG_CONFIG
    {
      int size_s = CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_RB_Config->size;
      int i;
      LOG_I(RRC, "Dumping scg_RB_Config: %d", size_s);

      for (i=0; i<size_s; i++) printf("%2.2x", (unsigned char)CG_Config->criticalExtensions.choice.c1->choice.cg_Config->scg_RB_Config->buf[i]);

      printf("\n");
    }
#endif
  } else {
    LOG_W(RRC, "SCG RB configuration is not present in the Addition Response message \n");
    return;
  }

  protocol_ctxt_t ctxt;
  rrc_eNB_ue_context_t *ue_context;
  unsigned char buffer[8192];
  int size;
  ue_context = rrc_eNB_get_ue_context(RC.rrc[enb_mod_idP], m->rnti);
  ue_context->ue_context.nb_of_modify_endc_e_rabs = m->nb_e_rabs_admitted_tobeadded;
  int j=0;

  while(j < m->nb_e_rabs_admitted_tobeadded) {
    for (int e_rab_idx=0; e_rab_idx<ue_context->ue_context.setup_e_rabs; e_rab_idx++) {
      //Update ue_context information with gNB's address and new GTP tunnel ID
      if( ue_context->ue_context.e_rab[e_rab_idx].param.e_rab_id == m->e_rabs_admitted_tobeadded[j].e_rab_id) {
        memcpy(ue_context->ue_context.gnb_gtp_endc_addrs[e_rab_idx].buffer,
               m->e_rabs_admitted_tobeadded[j].gnb_addr.buffer,
               m->e_rabs_admitted_tobeadded[j].gnb_addr.length);
        ue_context->ue_context.gnb_gtp_endc_addrs[e_rab_idx].length = m->e_rabs_admitted_tobeadded[j].gnb_addr.length;
        ue_context->ue_context.gnb_gtp_endc_teid[e_rab_idx] = m->e_rabs_admitted_tobeadded[j].gtp_teid;
        ue_context->ue_context.e_rab[e_rab_idx].status = E_RAB_STATUS_TOMODIFY;
        break;
      }
    }

    j++;
  }

  PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                0,
                                ENB_FLAG_YES,
                                m->rnti,
                                0, 0);
  size = rrc_eNB_generate_RRCConnectionReconfiguration_endc(&ctxt, ue_context, buffer, 8192, scg_CellGroupConfig, nr1_conf);
  rrc_data_req(&ctxt,
               DCCH,
               rrc_eNB_mui++,
               SDU_CONFIRM_NO,
               size,
               buffer,
               PDCP_TRANSMISSION_MODE_CONTROL);
}


//-----------------------------------------------------------------------------
void *rrc_enb_process_itti_msg(void *notUsed) {
  MessageDef                         *msg_p;
  const char                         *msg_name_p;
  instance_t                          instance;
  int                                 result;
  protocol_ctxt_t                     ctxt;
  memset(&ctxt, 0, sizeof(ctxt));
  // Wait for a message
  itti_receive_msg(TASK_RRC_ENB, &msg_p);
  msg_name_p = ITTI_MSG_NAME(msg_p);
  instance = ITTI_MSG_INSTANCE(msg_p);

  /* RRC_SUBFRAME_PROCESS is sent every subframe, do not log it */
  if (ITTI_MSG_ID(msg_p) != RRC_SUBFRAME_PROCESS)
    LOG_D(RRC,"Received message %s\n",msg_name_p);

  switch (ITTI_MSG_ID(msg_p)) {
    case TERMINATE_MESSAGE:
      LOG_W(RRC, " *** Exiting RRC thread\n");
      itti_exit_task();
      break;

    case MESSAGE_TEST:
      LOG_I(RRC, "[eNB %d] Received %s\n", instance, msg_name_p);
      break;

    /* Messages from MAC */
    case RRC_MAC_CCCH_DATA_IND:
      PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                    RRC_MAC_CCCH_DATA_IND(msg_p).enb_index,
                                    ENB_FLAG_YES,
                                    RRC_MAC_CCCH_DATA_IND(msg_p).rnti,
                                    msg_p->ittiMsgHeader.lte_time.frame,
                                    msg_p->ittiMsgHeader.lte_time.slot);
      LOG_I(RRC,"Decoding CCCH : inst %d, CC_id %d, ctxt %p, sib_info_p->Rx_buffer.payload_size %d\n",
            instance,
            RRC_MAC_CCCH_DATA_IND(msg_p).CC_id,
            &ctxt,
            RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size);

      if (RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size >= CCCH_SDU_SIZE) {
        LOG_I(RRC, "CCCH message has size %d > %d\n",
              RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size,CCCH_SDU_SIZE);
        break;
      }

      rrc_eNB_decode_ccch(&ctxt,
                          (uint8_t *)RRC_MAC_CCCH_DATA_IND(msg_p).sdu,
                          RRC_MAC_CCCH_DATA_IND(msg_p).sdu_size,
                          RRC_MAC_CCCH_DATA_IND(msg_p).CC_id);
      break;

    /* Messages from PDCP */
    case RRC_DCCH_DATA_IND:
      PROTOCOL_CTXT_SET_BY_INSTANCE(&ctxt,
                                    instance,
                                    ENB_FLAG_YES,
                                    RRC_DCCH_DATA_IND(msg_p).rnti,
                                    msg_p->ittiMsgHeader.lte_time.frame,
                                    msg_p->ittiMsgHeader.lte_time.slot);
      LOG_D(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Received on DCCH %d %s\n",
            PROTOCOL_RRC_CTXT_UE_ARGS(&ctxt),
            RRC_DCCH_DATA_IND(msg_p).dcch_index,
            msg_name_p);
      rrc_eNB_decode_dcch(&ctxt,
                          RRC_DCCH_DATA_IND(msg_p).dcch_index,
                          RRC_DCCH_DATA_IND(msg_p).sdu_p,
                          RRC_DCCH_DATA_IND(msg_p).sdu_size);
      // Message buffer has been processed, free it now.
      result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), RRC_DCCH_DATA_IND(msg_p).sdu_p);

      if (result != EXIT_SUCCESS) {
        LOG_I(RRC, "Failed to free memory (%d)!\n",result);
        break;
      }

      break;

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
      LOG_D(RRC, "[eNB %d] Received Paging message from S1AP: %s\n", instance, msg_name_p);
      rrc_eNB_process_PAGING_IND(msg_p, msg_name_p, instance);
      break;

    case S1AP_E_RAB_SETUP_REQ:
      rrc_eNB_process_S1AP_E_RAB_SETUP_REQ(msg_p, msg_name_p, instance);
      LOG_D(RRC, "[eNB %d] Received the message %s\n", instance, msg_name_p);
      break;

    case S1AP_E_RAB_MODIFY_REQ:
      rrc_eNB_process_S1AP_E_RAB_MODIFY_REQ(msg_p, msg_name_p, instance);
      break;

    case S1AP_E_RAB_RELEASE_COMMAND:
      rrc_eNB_process_S1AP_E_RAB_RELEASE_COMMAND(msg_p, msg_name_p, instance);
      break;

    case S1AP_UE_CONTEXT_RELEASE_REQ:
      rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_REQ(msg_p, msg_name_p, instance);
      break;

    case S1AP_UE_CONTEXT_RELEASE_COMMAND:
      rrc_eNB_process_S1AP_UE_CONTEXT_RELEASE_COMMAND(msg_p, msg_name_p, instance);
      break;

    case GTPV1U_ENB_DELETE_TUNNEL_RESP: {
      rrc_eNB_ue_context_t *ue = rrc_eNB_get_ue_context(RC.rrc[instance], GTPV1U_ENB_DELETE_TUNNEL_RESP(msg_p).rnti);

      if (ue != NULL
          && ue->ue_context.ue_release_timer_rrc > 0
          && (ue->ue_context.handover_info == NULL ||
              (ue->ue_context.handover_info->state != HO_RELEASE &&
               ue->ue_context.handover_info->state != HO_CANCEL))) {
        ue->ue_context.ue_release_timer_rrc = ue->ue_context.ue_release_timer_thres_rrc;
      }

      break;
    }

    case S1AP_PATH_SWITCH_REQ_ACK:
      LOG_I(RRC, "[eNB %d] received path switch ack %s\n", instance, msg_name_p);
      rrc_eNB_process_S1AP_PATH_SWITCH_REQ_ACK(msg_p, msg_name_p, instance);
      break;

    case X2AP_SETUP_REQ:
      rrc_eNB_process_x2_setup_request(instance, &X2AP_SETUP_REQ(msg_p));
      break;

    case X2AP_SETUP_RESP:
      rrc_eNB_process_x2_setup_response(instance, &X2AP_SETUP_RESP(msg_p));
      break;

    case X2AP_HANDOVER_REQ:
      LOG_I(RRC, "[eNB %d] target eNB Receives X2 HO Req %s\n", instance, msg_name_p);
      rrc_eNB_process_handoverPreparationInformation(instance, &X2AP_HANDOVER_REQ(msg_p));
      break;

    case X2AP_HANDOVER_REQ_ACK: {
      struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
      x2ap_handover_req_ack_t         *x2ap_handover_req_ack = NULL;
      hashtable_rc_t                    hash_rc      = HASH_TABLE_KEY_NOT_EXISTS;
      gtpv1u_ue_data_t                  *gtpv1u_ue_data_p = NULL;
      ue_context_p = rrc_eNB_get_ue_context(RC.rrc[instance], X2AP_HANDOVER_REQ_ACK(msg_p).rnti);

      if (ue_context_p == NULL) {
        /* is it possible? */
        LOG_E(RRC, "could not find UE (rnti %x) while processing X2AP_HANDOVER_REQ_ACK\n",
              X2AP_HANDOVER_REQ_ACK(msg_p).rnti);
        exit(1);
      }

      LOG_I(RRC, "[eNB %d] source eNB receives the X2 HO ACK %s\n", instance, msg_name_p);
      DevAssert(ue_context_p != NULL);

      if (ue_context_p->ue_context.handover_info->state != HO_REQUEST) abort();

      hash_rc = hashtable_get(RC.gtpv1u_data_g->ue_mapping, ue_context_p->ue_context.rnti, (void **)&gtpv1u_ue_data_p);

      /* set target enb gtp teid */
      if (hash_rc == HASH_TABLE_KEY_NOT_EXISTS) {
        LOG_E(RRC, "X2AP_HANDOVER_REQ_ACK func(), hashtable_get failed: while getting ue rnti %x in hashtable ue_mapping\n", ue_context_p->ue_context.rnti);
      } else {
        uint8_t nb_e_rabs_tobesetup = 0;
        ebi_t   eps_bearer_id       = 0;
        int     ip_offset           = 0;
        in_addr_t  in_addr;
        x2ap_handover_req_ack = &X2AP_HANDOVER_REQ_ACK(msg_p);
        nb_e_rabs_tobesetup = x2ap_handover_req_ack->nb_e_rabs_tobesetup;
        ue_context_p->ue_context.nb_x2u_e_rabs = nb_e_rabs_tobesetup;

        for(int i=0; i< nb_e_rabs_tobesetup; i++) {
          ip_offset               = 0;
          eps_bearer_id = x2ap_handover_req_ack->e_rabs_tobesetup[i].e_rab_id;
          ue_context_p->ue_context.enb_gtp_x2u_ebi[i] = eps_bearer_id;
          ue_context_p->ue_context.enb_gtp_x2u_teid[i] = x2ap_handover_req_ack->e_rabs_tobesetup[i].gtp_teid;
          gtpv1u_ue_data_p->bearers[eps_bearer_id - GTPV1U_BEARER_OFFSET].teid_teNB = x2ap_handover_req_ack->e_rabs_tobesetup[i].gtp_teid;

          if ((x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.length == 4) ||
              (x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.length == 20)) {
            in_addr = *((in_addr_t *)x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.buffer);
            ip_offset = 4;
            gtpv1u_ue_data_p->bearers[eps_bearer_id - GTPV1U_BEARER_OFFSET].tenb_ip_addr = in_addr;
            ue_context_p->ue_context.enb_gtp_x2u_addrs[i] = x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr;
          }

          if ((x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.length == 16) ||
              (x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.length == 20)) {
            memcpy(gtpv1u_ue_data_p->bearers[eps_bearer_id - GTPV1U_BEARER_OFFSET].tenb_ip6_addr.s6_addr,
                   &x2ap_handover_req_ack->e_rabs_tobesetup[i].eNB_addr.buffer[ip_offset],
                   16);
          }
        }
      }

      rrc_eNB_process_handoverCommand(instance, ue_context_p, &X2AP_HANDOVER_REQ_ACK(msg_p));
      ue_context_p->ue_context.handover_info->state = HO_PREPARE;
      break;
    }

    case X2AP_UE_CONTEXT_RELEASE: {
      struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
      ue_context_p = rrc_eNB_get_ue_context(RC.rrc[instance], X2AP_UE_CONTEXT_RELEASE(msg_p).rnti);
      LOG_I(RRC, "[eNB %d] source eNB receives the X2 UE CONTEXT RELEASE %s\n", instance, msg_name_p);
      DevAssert(ue_context_p != NULL);

      if (ue_context_p->ue_context.handover_info->state != HO_COMPLETE) abort();

      ue_context_p->ue_context.handover_info->state = HO_RELEASE;
      break;
    }

    case X2AP_HANDOVER_CANCEL: {
      struct rrc_eNB_ue_context_s        *ue_context_p = NULL;
      char *cause;

      switch (X2AP_HANDOVER_CANCEL(msg_p).cause) {
        case X2AP_T_RELOC_PREP_TIMEOUT:
          cause = "T_RelocPrep timeout";
          break;

        case X2AP_TX2_RELOC_OVERALL_TIMEOUT:
          cause = "Tx2_RelocOverall timeout";
          break;

        default:
          /* cannot come here */
          exit(1);
      }

      ue_context_p = rrc_eNB_get_ue_context(RC.rrc[instance], X2AP_HANDOVER_CANCEL(msg_p).rnti);

      if (ue_context_p != NULL &&
          ue_context_p->ue_context.handover_info != NULL) {
        LOG_I(RRC, "[eNB %d] eNB receives X2 HANDOVER CANCEL for rnti %x, cause %s [%s]\n",
              instance,
              X2AP_HANDOVER_CANCEL(msg_p).rnti,
              cause,
              msg_name_p);

        if (X2AP_HANDOVER_CANCEL(msg_p).cause == X2AP_T_RELOC_PREP_TIMEOUT) {
          /* for prep timeout, simply return to normal state */
          /* TODO: be sure that it's correct to set Status to RRC_RECONFIGURED */
          ue_context_p->ue_context.Status = RRC_RECONFIGURED;
          /* TODO: be sure free is enough here (check memory leaks) */
          free(ue_context_p->ue_context.handover_info);
          ue_context_p->ue_context.handover_info = NULL;
        } else {
          /* for overall timeout, remove UE entirely */
          ue_context_p->ue_context.handover_info->state = HO_CANCEL;
        }
      } else {
        char *failure_cause;

        if (ue_context_p == NULL)
          failure_cause = "no UE found";
        else
          failure_cause = "UE not in handover";

        LOG_W(RRC, "[eNB %d] cannot process (%s) X2 HANDOVER CANCEL for rnti %x, cause %s, ignoring\n",
              instance, failure_cause, X2AP_HANDOVER_CANCEL(msg_p).rnti, cause);
      }

      break;
    }

    case X2AP_ENDC_SGNB_ADDITION_REQ_ACK: {
      rrc_eNB_process_AdditionResponseInformation(ENB_INSTANCE_TO_MODULE_ID(instance), &X2AP_ENDC_SGNB_ADDITION_REQ_ACK(msg_p));
      break;
    }

    /* Messages from eNB app */
    case RRC_CONFIGURATION_REQ:
      LOG_I(RRC, "[eNB %d] Received %s : %p\n", instance, msg_name_p, &RRC_CONFIGURATION_REQ(msg_p));
      openair_rrc_eNB_configuration(ENB_INSTANCE_TO_MODULE_ID(instance), &RRC_CONFIGURATION_REQ(msg_p));
      break;

    /* Messages from F1AP task */
    case F1AP_SETUP_REQ:
      AssertFatal(NODE_IS_CU(RC.rrc[instance]->node_type),
                  "should not receive F1AP_SETUP_REQUEST, need call by CU!\n");
      LOG_I(RRC,"[eNB %d] Received %s : %p\n", instance, msg_name_p, &F1AP_SETUP_REQ(msg_p));
      handle_f1_setup_req(&F1AP_SETUP_REQ(msg_p));
      break;

    case RRC_SUBFRAME_PROCESS:
      rrc_subframe_process(&RRC_SUBFRAME_PROCESS(msg_p).ctxt, RRC_SUBFRAME_PROCESS(msg_p).CC_id);
      break;

    case M2AP_SETUP_RESP:
      rrc_eNB_process_M2AP_SETUP_RESP(&ctxt,0/*CC_id*/,ENB_INSTANCE_TO_MODULE_ID(instance),&M2AP_SETUP_RESP(msg_p));
      break;

    case M2AP_MBMS_SCHEDULING_INFORMATION:
      rrc_eNB_process_M2AP_MBMS_SCHEDULING_INFORMATION(&ctxt,0/*CC_id*/,ENB_INSTANCE_TO_MODULE_ID(instance),&M2AP_MBMS_SCHEDULING_INFORMATION(msg_p));
      break;

    case M2AP_MBMS_SESSION_START_REQ:
      rrc_eNB_process_M2AP_MBMS_SESSION_START_REQ(&ctxt,0/*CC_id*/,ENB_INSTANCE_TO_MODULE_ID(instance),&M2AP_MBMS_SESSION_START_REQ(msg_p));
      break;

    case M2AP_MBMS_SESSION_STOP_REQ:
      rrc_eNB_process_M2AP_MBMS_SESSION_STOP_REQ(&ctxt,&M2AP_MBMS_SESSION_STOP_REQ(msg_p));
      break;

    case M2AP_RESET:
      rrc_eNB_process_M2AP_RESET(&ctxt,&M2AP_RESET(msg_p));
      break;

    case M2AP_ENB_CONFIGURATION_UPDATE_ACK:
      rrc_eNB_process_M2AP_ENB_CONFIGURATION_UPDATE_ACK(&ctxt,&M2AP_ENB_CONFIGURATION_UPDATE_ACK(msg_p));
      break;

    case M2AP_ERROR_INDICATION:
      rrc_eNB_process_M2AP_ERROR_INDICATION(&ctxt,&M2AP_ERROR_INDICATION(msg_p));
      break;

    case M2AP_MBMS_SERVICE_COUNTING_REQ:
      rrc_eNB_process_M2AP_MBMS_SERVICE_COUNTING_REQ(&ctxt,&M2AP_MBMS_SERVICE_COUNTING_REQ(msg_p));
      break;

    case M2AP_MCE_CONFIGURATION_UPDATE:
       rrc_eNB_process_M2AP_MCE_CONFIGURATION_UPDATE(&ctxt,&M2AP_MCE_CONFIGURATION_UPDATE(msg_p));
       break;

    case RLC_SDU_INDICATION:
      process_rlc_sdu_indication(instance,
                                 RLC_SDU_INDICATION(msg_p).rnti,
                                 RLC_SDU_INDICATION(msg_p).is_successful,
                                 RLC_SDU_INDICATION(msg_p).srb_id,
                                 RLC_SDU_INDICATION(msg_p).message_id);
      break;

    default:
      LOG_E(RRC, "[eNB %d] Received unexpected message %s\n", instance, msg_name_p);
      break;
  }

  result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), msg_p);

  if (result != EXIT_SUCCESS) {
    LOG_I(RRC, "Failed to free memory (%d)!\n",result);
  }

  msg_p = NULL;
  return NULL;
}

//-----------------------------------------------------------------------------
void *
rrc_enb_task(
  void *args_p
)
//-----------------------------------------------------------------------------
{
  rrc_enb_init();
  itti_mark_task_ready(TASK_RRC_ENB);
  LOG_I(RRC,"Entering main loop of RRC message task\n");

  while (1) {
    (void) rrc_enb_process_itti_msg(NULL);
    {
      //extern volatile int go_nr;
      void rrc_go_nr(void);
      //if (go_nr) rrc_go_nr();
    }
  }
  }

/*------------------------------------------------------------------------------*/
void
openair_rrc_top_init_eNB(int eMBMS_active,uint8_t HO_active)
//-----------------------------------------------------------------------------
{
  module_id_t         module_id;
  int                 CC_id;
  /* for no gcc warnings */
  (void)CC_id;
  LOG_D(RRC, "[OPENAIR][INIT] Init function start: NB_eNB_INST=%d\n", RC.nb_inst);

  if (RC.nb_inst > 0) {
    LOG_I(RRC,"[eNB] handover active state is %d \n", HO_active);

    for (module_id=0; module_id<NB_eNB_INST; module_id++) {
      RC.rrc[module_id]->HO_flag   = (uint8_t)HO_active;
    }

    LOG_I(RRC,"[eNB] eMBMS active state is %d \n", eMBMS_active);

    for (module_id=0; module_id<NB_eNB_INST; module_id++) {
      for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
        RC.rrc[module_id]->carrier[CC_id].MBMS_flag = (uint8_t)eMBMS_active;
      }
    }
  }
}

//-----------------------------------------------------------------------------
void
rrc_top_cleanup_eNB(
  void
)
//-----------------------------------------------------------------------------
{
  for (int i=0; i<RC.nb_inst; i++) free (RC.rrc[i]);

  free(RC.rrc);
}


//-----------------------------------------------------------------------------
//TTN - for D2D
uint8_t
rrc_eNB_process_SidelinkUEInformation(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t         *ue_context_pP,
  LTE_SidelinkUEInformation_r12_t *sidelinkUEInformation
)
//-----------------------------------------------------------------------------
{
  LTE_SL_DestinationInfoList_r12_t  *destinationInfoList;
  int n_destinations = 0;
  int n_discoveryMessages = 0;
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" [RAPROC] Logical Channel UL-DCCH, " "processing SidelinkUEInformation from UE (SRB1 Active)\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));

  //For SL Communication
  if (sidelinkUEInformation->criticalExtensions.present == LTE_SidelinkUEInformation_r12__criticalExtensions_PR_c1) {
    if (sidelinkUEInformation->criticalExtensions.choice.c1.present == LTE_SidelinkUEInformation_r12__criticalExtensions__c1_PR_sidelinkUEInformation_r12) {
      // express its interest to receive SL communication
      if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commRxInterestedFreq_r12 !=  NULL) {
      }

      // express its interest to transmit  non-relay one-to-many SL communication
      if ((sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commTxResourceReq_r12 != NULL) &&
          (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commTxResourceReq_r12->carrierFreq_r12 != NULL)) {
        n_destinations = sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commTxResourceReq_r12->destinationInfoList_r12.list.count;
        destinationInfoList = CALLOC(1, sizeof(LTE_SL_DestinationInfoList_r12_t));

        for (int i=0; i< n_destinations; i++ ) {
          //sl_DestinationIdentityList[i] = *(sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commTxResourceReq_r12->destinationInfoList_r12.list.array[i]);
          ASN_SEQUENCE_ADD(&destinationInfoList->list, sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.commTxResourceReq_r12->destinationInfoList_r12.list.array[i]);
        }

        //generate RRC Reconfiguration
        rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, destinationInfoList, 0);
        free(destinationInfoList);
        destinationInfoList = NULL;
        return 0;
      }

      // express its interest to transmit  non-relay one-to-one SL communication
      if ((sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension != NULL) &&
          (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceReqUC_r13 != NULL)) {
        if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceReqUC_r13->carrierFreq_r12 != NULL) {
          n_destinations = sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceReqUC_r13->destinationInfoList_r12.list.count;
          destinationInfoList = CALLOC(1, sizeof(LTE_SL_DestinationInfoList_r12_t));

          for (int i=0; i< n_destinations; i++ ) {
            //sl_DestinationIdentityList[i] = *(sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceReqUC_r13->destinationInfoList_r12.list.array[i]);
            ASN_SEQUENCE_ADD(&destinationInfoList->list,
                             sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceReqUC_r13->destinationInfoList_r12.list.array[i]);
          }

          //generate RRC Reconfiguration
          rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, destinationInfoList, 0);
          free(destinationInfoList);
          destinationInfoList = NULL;
          return 0;
        }
      }

      // express its interest to transmit relay related one-to-one SL communication
      if ((sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension != NULL) &&
          (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelayUC_r13 != NULL)) {
        if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelayUC_r13->destinationInfoList_r12.list.count
            > 0) {
          n_destinations =
            sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelayUC_r13->destinationInfoList_r12.list.count;
          destinationInfoList = CALLOC(1, sizeof(LTE_SL_DestinationInfoList_r12_t));

          for (int i=0; i< n_destinations; i++ ) {
            //sl_DestinationIdentityList[i] = *(sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelayUC_r13->destinationInfoList_r12.list.array[i]);
            ASN_SEQUENCE_ADD(&destinationInfoList->list,
                             sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelayUC_r13->destinationInfoList_r12.list.array[i]);
          }

          //generate RRC Reconfiguration
          rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, destinationInfoList, 0);
          free(destinationInfoList);
          destinationInfoList = NULL;
          return 0;
        }
      }

      //express its interest to transmit relay related one-to-many SL communication
      if ((sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension != NULL) &&
          (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13 != NULL)) {
        if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelay_r13->destinationInfoList_r12.list.count
            > 0) {
          n_destinations =
            sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelay_r13->destinationInfoList_r12.list.count;
          destinationInfoList = CALLOC(1, sizeof(LTE_SL_DestinationInfoList_r12_t));

          for (int i=0; i< n_destinations; i++ ) {
            //sl_DestinationIdentityList[i] = *(sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelay_r13->destinationInfoList_r12.list.array[i]);
            ASN_SEQUENCE_ADD(&destinationInfoList->list,
                             sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->commTxResourceInfoReqRelay_r13->commTxResourceReqRelay_r13->destinationInfoList_r12.list.array[i]);
          }

          //generate RRC Reconfiguration
          rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, destinationInfoList, 0);
          free(destinationInfoList);
          destinationInfoList = NULL;
          return 0;
        }
      }

      //For SL Discovery
      //express its interest to receive SL discovery announcements
      //express its interest to transmit non-PS related discovery announcements
      if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.discTxResourceReq_r12 != NULL) {
        n_discoveryMessages = *(sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.discTxResourceReq_r12);
        //generate RRC Reconfiguration
        rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, NULL, n_discoveryMessages);
        return 0;
      }

      //express its interest to transmit PS related discovery announcements
      if ((sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension != NULL) &&
          (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->discTxResourceReqPS_r13 !=NULL)) {
        if (sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->discTxResourceReqPS_r13->discTxResourceReq_r13 > 0) {
          n_discoveryMessages = sidelinkUEInformation->criticalExtensions.choice.c1.choice.sidelinkUEInformation_r12.nonCriticalExtension->discTxResourceReqPS_r13->discTxResourceReq_r13;
          //generate RRC Reconfiguration
          rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(ctxt_pP, ue_context_pP, NULL, n_discoveryMessages);
          return 0;
        }
      }
    }
  }

  return 0;
}

//-----------------------------------------------------------------------------
int
rrc_eNB_generate_RRCConnectionReconfiguration_Sidelink(
  const protocol_ctxt_t *const ctxt_pP,
  rrc_eNB_ue_context_t *const ue_context_pP,
  LTE_SL_DestinationInfoList_r12_t  *destinationInfoList,
  int n_discoveryMessages
)
//-----------------------------------------------------------------------------
{
  uint8_t                             buffer[RRC_BUF_SIZE];
  uint16_t                            size = 0;
  memset(buffer, 0, RRC_BUF_SIZE);

  // allocate dedicated pools for UE -sl-CommConfig/sl-DiscConfig (sl-V2X-ConfigDedicated)
  //populate dedicated resources for SL communication (sl-CommConfig)
  if ((destinationInfoList != NULL) && (destinationInfoList->list.count > 0)) {
    LOG_I(RRC,"[eNB %d] Frame %d, Generate LTE_RRCConnectionReconfiguration_Sidelink (bytes %d, UE id %x), number of destinations %d\n",
          ctxt_pP->module_id,ctxt_pP->frame, size, ue_context_pP->ue_context.rnti,destinationInfoList->list.count );
    //get dedicated resources from available pool and assign to the UE
    LTE_SL_CommConfig_r12_t  sl_CommConfig[destinationInfoList->list.count];
    //get a RP from the available RPs
    sl_CommConfig[0] = rrc_eNB_get_sidelink_commTXPool(ctxt_pP, ue_context_pP, destinationInfoList);
    size = do_RRCConnectionReconfiguration(ctxt_pP,
                                           buffer,
                                           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),   //Transaction_id
                                           (LTE_SRB_ToAddModList_t *)NULL,
                                           (LTE_DRB_ToAddModList_t *)NULL,
                                           (LTE_DRB_ToReleaseList_t *)NULL, // DRB2_list,
                                           (struct LTE_SPS_Config *)NULL,   // *sps_Config,
                                           NULL, NULL, NULL, NULL,NULL,
                                           NULL, NULL,  NULL, NULL, NULL, NULL, NULL,
                                           (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)NULL,
                                           (LTE_SL_CommConfig_r12_t *)&sl_CommConfig,
                                           (LTE_SL_DiscConfig_r12_t *)NULL,
                                           (LTE_SCellToAddMod_r10_t *)NULL
                                          );
    //
  }

  //populate dedicated resources for SL discovery (sl-DiscConfig)
  if (n_discoveryMessages > 0) {
    LTE_SL_DiscConfig_r12_t sl_DiscConfig[n_discoveryMessages];
    //get a RP from the available RPs
    sl_DiscConfig[0] = rrc_eNB_get_sidelink_discTXPool(ctxt_pP, ue_context_pP, n_discoveryMessages );
    size = do_RRCConnectionReconfiguration(ctxt_pP,
                                           buffer,
                                           rrc_eNB_get_next_transaction_identifier(ctxt_pP->module_id),   //Transaction_id
                                           (LTE_SRB_ToAddModList_t *)NULL,
                                           (LTE_DRB_ToAddModList_t *)NULL,
                                           (LTE_DRB_ToReleaseList_t *)NULL, // DRB2_list,
                                           (struct LTE_SPS_Config *)NULL,   // *sps_Config,
                                           NULL, NULL, NULL, NULL,NULL,
                                           NULL, NULL,  NULL, NULL, NULL, NULL, NULL,
                                           (struct LTE_RRCConnectionReconfiguration_r8_IEs__dedicatedInfoNASList *)NULL,
                                           (LTE_SL_CommConfig_r12_t *)NULL,
                                           (LTE_SL_DiscConfig_r12_t *)&sl_DiscConfig,
                                           (LTE_SCellToAddMod_r10_t *)NULL
                                          );
  }

  LOG_I(RRC,"[eNB %d] Frame %d, Logical Channel DL-DCCH, Generate LTE_RRCConnectionReconfiguration_Sidelink (bytes %d, UE id %x)\n",
        ctxt_pP->module_id,ctxt_pP->frame, size, ue_context_pP->ue_context.rnti);
  rrc_data_req(
    ctxt_pP,
    DCCH,
    rrc_eNB_mui++,
    SDU_CONFIRM_NO,
    size,
    buffer,
    PDCP_TRANSMISSION_MODE_CONTROL);
  // rrc_data_req();
  return size;
}

LTE_SL_CommConfig_r12_t rrc_eNB_get_sidelink_commTXPool( const protocol_ctxt_t *const ctxt_pP, rrc_eNB_ue_context_t *const ue_context_pP, LTE_SL_DestinationInfoList_r12_t  *destinationInfoList ) {
  // for the moment, use scheduled resource allocation
  LTE_SL_CommConfig_r12_t sl_CommConfig_r12;
  LTE_SL_CommConfig_r12_t  *sl_CommConfig = &sl_CommConfig_r12;
  LTE_SL_CommResourcePool_r12_t    *sc_CommTxConfig;
  memset(sl_CommConfig,0,sizeof(LTE_SL_CommConfig_r12_t));
  sl_CommConfig->commTxResources_r12 = CALLOC(1, sizeof(*sl_CommConfig->commTxResources_r12));
  sl_CommConfig->commTxResources_r12->present = LTE_SL_CommConfig_r12__commTxResources_r12_PR_setup;
  sl_CommConfig->commTxResources_r12->choice.setup.present = LTE_SL_CommConfig_r12__commTxResources_r12__setup_PR_scheduled_r12;
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sl_RNTI_r12.size = 2;
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sl_RNTI_r12.buf = CALLOC(1,2);
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sl_RNTI_r12.buf[0] = 0x00;
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sl_RNTI_r12.buf[1] = 0x01;//ctxt_pP->rnti;//rnti
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sl_RNTI_r12.bits_unused = 0;
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.mcs_r12 = CALLOC(1,sizeof(*sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.mcs_r12));
  //*sl_CommConfig_test->commTxResources_r12->choice.setup.choice.scheduled_r12.mcs_r12 = 12; //Msc
  sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.mac_MainConfig_r12.retx_BSR_TimerSL = LTE_RetxBSR_Timer_r12_sf320; //MacConfig, for testing only
  //sl_CommConfig_test->commTxResources_r12->choice.setup.choice.scheduled_r12.sc_CommTxConfig_r12;
  sc_CommTxConfig = & sl_CommConfig->commTxResources_r12->choice.setup.choice.scheduled_r12.sc_CommTxConfig_r12;
  sc_CommTxConfig->sc_CP_Len_r12 = LTE_SL_CP_Len_r12_normal;
  sc_CommTxConfig->sc_Period_r12 = LTE_SL_PeriodComm_r12_sf40;
  sc_CommTxConfig->data_CP_Len_r12 = LTE_SL_CP_Len_r12_normal;
  //sc_TF_ResourceConfig_r12
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.prb_Num_r12 = 20;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.prb_Start_r12 = 5;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.prb_End_r12 = 44;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.offsetIndicator_r12.present = LTE_SL_OffsetIndicator_r12_PR_small_r12;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.offsetIndicator_r12.choice.small_r12 = 0;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.subframeBitmap_r12.present = LTE_SubframeBitmapSL_r12_PR_bs40_r12;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs40_r12.size = 5;
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs40_r12.buf  = CALLOC(1,5);
  sc_CommTxConfig->sc_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs40_r12.bits_unused = 0;
  //dataHoppingConfig_r12
  sc_CommTxConfig->dataHoppingConfig_r12.hoppingParameter_r12 = 0;
  sc_CommTxConfig->dataHoppingConfig_r12.numSubbands_r12  = LTE_SL_HoppingConfigComm_r12__numSubbands_r12_ns1;
  sc_CommTxConfig->dataHoppingConfig_r12.rb_Offset_r12 = 0;
  //ue_SelectedResourceConfig_r12
  sc_CommTxConfig->ue_SelectedResourceConfig_r12 = CALLOC (1, sizeof (*sc_CommTxConfig->ue_SelectedResourceConfig_r12));
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.prb_Num_r12 = 20;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.prb_Start_r12 = 5;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.prb_End_r12 = 44;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.offsetIndicator_r12.present = LTE_SL_OffsetIndicator_r12_PR_small_r12;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.offsetIndicator_r12.choice.small_r12 = 0 ;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.present = LTE_SubframeBitmapSL_r12_PR_bs40_r12;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.size = 5;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf  = CALLOC(1,5);
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.bits_unused = 0;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf[0] = 0xF0;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf[1] = 0xFF;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf[2] = 0xFF;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf[3] = 0xFF;
  sc_CommTxConfig->ue_SelectedResourceConfig_r12->data_TF_ResourceConfig_r12.subframeBitmap_r12.choice.bs4_r12.buf[4] = 0xFF;
  //rxParametersNCell_r12
  sc_CommTxConfig->rxParametersNCell_r12 = CALLOC (1, sizeof (*sc_CommTxConfig->rxParametersNCell_r12));
  sc_CommTxConfig->rxParametersNCell_r12->tdd_Config_r12 = CALLOC (1, sizeof (*sc_CommTxConfig->rxParametersNCell_r12->tdd_Config_r12 ));
  sc_CommTxConfig->rxParametersNCell_r12->tdd_Config_r12->subframeAssignment = 0 ;
  sc_CommTxConfig->rxParametersNCell_r12->tdd_Config_r12->specialSubframePatterns = 0;
  sc_CommTxConfig->rxParametersNCell_r12->syncConfigIndex_r12 = 0;
  //txParameters_r12
  sc_CommTxConfig->txParameters_r12 = CALLOC (1, sizeof (*sc_CommTxConfig->txParameters_r12));
  sc_CommTxConfig->txParameters_r12->sc_TxParameters_r12.alpha_r12 = LTE_Alpha_r12_al0;
  sc_CommTxConfig->txParameters_r12->sc_TxParameters_r12.p0_r12 = 0;
  sc_CommTxConfig->ext1 = NULL ;
  return *sl_CommConfig;
}


LTE_SL_DiscConfig_r12_t rrc_eNB_get_sidelink_discTXPool( const protocol_ctxt_t *const ctxt_pP, rrc_eNB_ue_context_t *const ue_context_pP,  int n_discoveryMessages ) {
  //TODO
  LTE_SL_DiscConfig_r12_t  sl_DiscConfig;
  memset(&sl_DiscConfig,0,sizeof(LTE_SL_DiscConfig_r12_t));
  sl_DiscConfig.discTxResources_r12 = CALLOC(1,sizeof(*sl_DiscConfig.discTxResources_r12));
  sl_DiscConfig.discTxResources_r12->present = LTE_SL_DiscConfig_r12__discTxResources_r12_PR_setup;
  sl_DiscConfig.discTxResources_r12->choice.setup.present = LTE_SL_DiscConfig_r12__discTxResources_r12__setup_PR_scheduled_r12;
  //sl_DiscConfig.discTxResources_r12->choice.setup.choice.scheduled_r12.discHoppingConfig_r12;
  //sl_DiscConfig.discTxResources_r12->choice.setup.choice.scheduled_r12.discTF_IndexList_r12;
  //sl_DiscConfig.discTxResources_r12->choice.setup.choice.scheduled_r12.discTxConfig_r12;
  return sl_DiscConfig;
}

RRC_status_t
rrc_rx_tx(
  protocol_ctxt_t *const ctxt_pP,
  const int        CC_id
)
//-----------------------------------------------------------------------------
{
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_RRC_ENB, RRC_SUBFRAME_PROCESS);
  RRC_SUBFRAME_PROCESS(message_p).ctxt  = *ctxt_pP;
  RRC_SUBFRAME_PROCESS(message_p).CC_id = CC_id;
  itti_send_msg_to_task(TASK_RRC_ENB, ctxt_pP->module_id, message_p);
  return RRC_OK;
}
