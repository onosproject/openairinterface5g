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

/*! \file openair2/NR_PHY_INTERFACE/NR_IF_Module.c
* \brief data structures for PHY/MAC interface modules
* \author EURECOM/NTUST
* \date 2018
* \version 0.1
* \company Eurecom, NTUST
* \email: raymond.knopp@eurecom.fr, kroempa@gmail.com
* \note
* \warning
*/

#include "openair1/PHY/defs_eNB.h"
#include "openair1/PHY/phy_extern.h"
#include "openair1/SCHED_NR/fapi_nr_l1.h"
#include "openair2/NR_PHY_INTERFACE/NR_IF_Module.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_extern.h"
#include "LAYER2/MAC/mac_proto.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"
#include "common/ran_context.h"
#include "executables/softmodem-common.h"

#define MAX_IF_MODULES 100

NR_IF_Module_t *if_inst[MAX_IF_MODULES];
NR_Sched_Rsp_t Sched_INFO[MAX_IF_MODULES][MAX_NUM_CCs];

extern int oai_nfapi_harq_indication(nfapi_harq_indication_t *harq_ind);
extern int oai_nfapi_crc_indication(nfapi_crc_indication_t *crc_ind);
extern int oai_nfapi_cqi_indication(nfapi_cqi_indication_t *cqi_ind);
extern int oai_nfapi_sr_indication(nfapi_sr_indication_t *ind);
extern int oai_nfapi_rx_ind(nfapi_rx_indication_t *ind);
extern uint8_t nfapi_mode;
extern uint16_t sf_ahead;
extern uint16_t sl_ahead;

void handle_nr_rach(NR_UL_IND_t *UL_info) {
  if (UL_info->rach_ind.number_of_pdus>0) {
    LOG_I(MAC,"UL_info[Frame %d, Slot %d] Calling initiate_ra_proc RACH:SFN/SLOT:%d/%d\n",UL_info->frame,UL_info->slot, UL_info->rach_ind.sfn,UL_info->rach_ind.slot);
   for(int i = 0; i < UL_info->rach_ind.number_of_pdus; i++) {
    if (UL_info->rach_ind.pdu_list[i].num_preamble>0)
    AssertFatal(UL_info->rach_ind.pdu_list[i].num_preamble==1,
		"More than 1 preamble not supported\n");
    
    nr_initiate_ra_proc(UL_info->module_id,
                        UL_info->CC_id,
                        UL_info->rach_ind.sfn,
                        UL_info->rach_ind.slot,
                        UL_info->rach_ind.pdu_list[i].preamble_list[0].preamble_index,
                        UL_info->rach_ind.pdu_list[i].freq_index,
                        UL_info->rach_ind.pdu_list[i].symbol_index,
                        UL_info->rach_ind.pdu_list[i].preamble_list[0].timing_advance);
   }
  }
}

//!TODO : smae function can be written to handle csi_resources
uint8_t get_ssb_resources (NR_CSI_MeasConfig_t *csi_MeasConfig, 
		NR_CSI_ResourceConfigId_t csi_ResourceConfigId, 
		NR_CSI_ReportConfig__reportQuantity_PR reportQuantity_type) {
  uint8_t idx = 0;
  //uint8_t csi_ssb_idx =0;

  for ( idx = 0; idx < csi_MeasConfig->csi_ResourceConfigToAddModList->list.count; idx++) {
    if ( csi_MeasConfig->csi_ResourceConfigToAddModList->list.array[idx]->csi_ResourceConfigId == csi_ResourceConfigId) {
    //Finding the CSI_RS or SSB Resources
      if ( csi_MeasConfig->csi_ResourceConfigToAddModList->list.array[idx]->csi_RS_ResourceSetList.present == NR_CSI_ResourceConfig__csi_RS_ResourceSetList_PR_nzp_CSI_RS_SSB) {

        if (NR_CSI_ReportConfig__reportQuantity_PR_ssb_Index_RSRP == reportQuantity_type )
          return (csi_MeasConfig->csi_ResourceConfigToAddModList->list.array[idx]->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->csi_SSB_ResourceSetList->list.count);

	else if (NR_CSI_ReportConfig__reportQuantity_PR_cri_RSRP == reportQuantity_type)
	  return (csi_MeasConfig->csi_ResourceConfigToAddModList->list.array[idx]->csi_RS_ResourceSetList.choice.nzp_CSI_RS_SSB->nzp_CSI_RS_ResourceSetList->list.count);

      } else {
        //TODO: find the CSI_RS IM resources
      }
    } else {
      AssertFatal(csi_ResourceConfigId, "csi_ResourceConfigId is not configured");
    }
  }

  return -1;
}


void extract_pucch_csi_report ( NR_CSI_MeasConfig_t *csi_MeasConfig,
                                nfapi_nr_uci_pucch_pdu_format_2_3_4_t *uci_pdu,
                                NR_UE_sched_ctrl_t *sched_ctrl,
                                frame_t frame,
                                slot_t slot,
				NR_SubcarrierSpacing_t scs
                              ) {
  /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
  //uint8_t bitlen_cri = (log(csi_MeasConfig->csi_ResourceConfigToAddModList->list.count)/log(2));
  uint8_t payload_size = ceil(uci_pdu->csi_part1.csi_part1_bit_len/8);
  uint16_t *payload = calloc (1, payload_size);
  NR_CSI_ReportConfig__reportQuantity_PR reportQuantity_type = NR_CSI_ReportConfig__reportQuantity_PR_NOTHING;
  uint8_t UE_id = 0;
  uint8_t csi_report_id = 0;
  memcpy ( payload, uci_pdu->csi_part1.csi_part1_payload, payload_size);

  for ( csi_report_id =0; csi_report_id < csi_MeasConfig->csi_ReportConfigToAddModList->list.count; csi_report_id++ ) {
    //Assuming in periodic reporting for one slot can be configured with only one CSI-ReportConfig
    if (csi_MeasConfig->csi_ReportConfigToAddModList->list.array[csi_report_id]->reportConfigType.present == NR_CSI_ReportConfig__reportConfigType_PR_periodic) {
      //considering 30khz scs and
      //Has to implement according to reportSlotConfig type
      LOG_I(PHY,"SFN/SF:%d%d \n", frame,slot);
      if (((NR_SubcarrierSpacing_kHz30 == scs) && 
		      (0 == ((frame*20) + (slot+1)) % csi_MeasConfig->csi_ReportConfigToAddModList->list.array[csi_report_id]->reportConfigType.choice.periodic->reportSlotConfig.choice.slots320))
		      ||((NR_SubcarrierSpacing_kHz120 == scs)&&
			      (0 == ((frame*80) + (slot+1)) % csi_MeasConfig->csi_ReportConfigToAddModList->list.array[csi_report_id]->reportConfigType.choice.periodic->reportSlotConfig.choice.slots320))){
        reportQuantity_type = csi_MeasConfig->csi_ReportConfigToAddModList->list.array[csi_report_id]->reportQuantity.present;
        LOG_I(PHY,"SFN/SF:%d%d reportQuantity type = %d\n",
              frame,slot,reportQuantity_type);
      }
    }
  }

  if ( !(reportQuantity_type)) 
    AssertFatal(reportQuantity_type, "reportQuantity is not configured");

  if ( NR_CSI_ReportConfig__reportQuantity_PR_ssb_Index_RSRP == reportQuantity_type ) {
    uint8_t nb_ssb_resource_set = get_ssb_resources(csi_MeasConfig,
                               csi_MeasConfig->csi_ReportConfigToAddModList->list.array[csi_report_id]->resourcesForChannelMeasurement,
			       reportQuantity_type);//csi_MeasConfig->csi_SSB_ResourceSetToAddModList->list.array[0]->CSI_SSB_ResourceList.list.count;
    uint8_t idx = 0;
    uint8_t ssb_idx = 0;
    uint8_t diff_rsrp_idx = 0;
    uint8_t bitlen_ssbri = log (nb_ssb_resource_set)/log (2);
    sched_ctrl->nr_of_csi_report[UE_id] = nb_ssb_resource_set;

    /*! As per the spec 38.212 and table:  6.3.1.1.2-12 in a single UCI sequence we can have multiple CSI_report
     * the number of CSI_report will depend on number of CSI resource sets that are configured in CSI-ResourceConfig RRC IE
     * From spec 38.331 from the IE CSI-ResourceConfig for SSB RSRP reporting we can configure only one resource set
     * From spec 38.214 section 5.2.1.2 For periodic and semi-persistent CSI Resource Settings, the number of CSI-RS Resource Sets configured is limited to S=1
     */
    for (idx = 0; idx < nb_ssb_resource_set; idx++) {
      
      /** from 38.214 sec 5.2.1.4.2
      - if the UE is configured with the higher layer parameter groupBasedBeamReporting set to 'disabled', the UE is
        not required to update measurements for more than 64 CSI-RS and/or SSB resources, and the UE shall report in
        a single report nrofReportedRS (higher layer configured) different CRI or SSBRI for each report setting

      - if the UE is configured with the higher layer parameter groupBasedBeamReporting set to 'enabled', the UE is not
      required to update measurements for more than 64 CSI-RS and/or SSB resources, and the UE shall report in a
      single reporting instance two different CRI or SSBRI for each report setting, where CSI-RS and/or SSB
      resources can be received simultaneously by the UE either with a single spatial domain receive filter, or with
      multiple simultaneous spatial domain receive filter
      */

      if (NR_CSI_ReportConfig__groupBasedBeamReporting_PR_disabled ==
          csi_MeasConfig->csi_ReportConfigToAddModList->list.array[0]->groupBasedBeamReporting.present ) {

        if ((NULL != csi_MeasConfig->csi_ReportConfigToAddModList->list.array[0]->groupBasedBeamReporting.choice.disabled->nrofReportedRS) && 
		      *(csi_MeasConfig->csi_ReportConfigToAddModList->list.array[0]->groupBasedBeamReporting.choice.disabled->nrofReportedRS)) 
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.nr_ssbri_cri = *(csi_MeasConfig->csi_ReportConfigToAddModList->list.array[0]->groupBasedBeamReporting.choice.disabled->nrofReportedRS);

        else 
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.nr_ssbri_cri = NR_CSI_ReportConfig__groupBasedBeamReporting__disabled__nrofReportedRS_n1;

        for (ssb_idx = 0; ssb_idx <= sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.nr_ssbri_cri ; ssb_idx++) {
          /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.CRI_SSBRI [ssb_idx] = (*payload) & ~(~1<<(bitlen_ssbri-1));
          *payload >>= bitlen_ssbri;
        }

        sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.RSRP = (*payload) & 0x7f;

        for ( diff_rsrp_idx =0; diff_rsrp_idx <= sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.nr_ssbri_cri - 1; diff_rsrp_idx++ ) {
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.diff_RSRP[diff_rsrp_idx] = (*payload) & 0x0f;
          *payload >>= 4;
        }

      } else if (NR_CSI_ReportConfig__groupBasedBeamReporting_PR_disabled !=
                 csi_MeasConfig->csi_ReportConfigToAddModList->list.array[0]->groupBasedBeamReporting.present ) {
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.nr_ssbri_cri = 2;
        for (ssb_idx = 0; ssb_idx < 2; ssb_idx++) {
          /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
          sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.CRI_SSBRI[ssb_idx] = (*payload) & ~(~1<<(bitlen_ssbri-1));
          *payload >>= bitlen_ssbri;
        }

        sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.RSRP = (*payload) & 0x7f;
        /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
        sched_ctrl->CSI_report[UE_id][idx].choice.ssb_cri_report.diff_RSRP[0] = (*payload) & 0x0f;
        *payload >>= 4;
      }
    }
  }

#if 0

  if ( NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_PMI_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_i1 == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_i1_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_LI_PMI_CQI== reportQuantity_type) {
    // Handling of extracting cri
    sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report.cri = calloc ( 1, ceil(bitlen_cri/8));
    *(sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report.cri) = *((uint32_t *)payload) & ~(~1<<(bitlen_cri-1));
    *payload >>= bitlen_cri;

    if ( 1 == RC.nrrrc[gnb_mod_idP]->carrier.pdsch_AntennaPorts ) {
      /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
      sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report->ri = NULL;
    } else {
      //Handling for the ri for multiple csi ports
    }
  }

  if (NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_LI_PMI_CQI== reportQuantity_type) {
    if ( 1 == RC.nrrrc[gnb_mod_idP]->carrier.pdsch_AntennaPorts )
      /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
      sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report->li = NULL;
    else {
      //Handle for li for multiple CSI ports
    }
  }

  //TODO: check for zero padding if available shift payload to the number of zero padding bits

  if ( NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_PMI_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_LI_PMI_CQI== reportQuantity_type) {
    if ( 1 == RC.nrrrc[gnb_mod_idP]->carrier.pdsch_AntennaPorts ) {
      /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
      sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report->pmi_x1 = NULL;
      sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report->pmi_x2 = NULL;
    }
  }

  if ( NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_PMI_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_i1_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_CQI == reportQuantity_type ||
       NR_CSI_ReportConfig__reportQuantity_PR_cri_RI_LI_PMI_CQI== reportQuantity_type) {
    /** From Table 6.3.1.1.2-3: RI, LI, CQI, and CRI of codebookType=typeI-SinglePanel */
    *(sched_ctrl->CSI_report[UE_id][cqi_idx].choice.cri_ri_li_pmi_cqi_report->cqi) = *(payload) & 0x0f;
    *(payload) >>= 4;
  }

#endif
}

void handle_nr_uci(NR_UL_IND_t *UL_info, NR_UE_sched_ctrl_t *sched_ctrl) {
  // TODO
  int max_harq_rounds = 4; // TODO define macro
  int num_ucis = UL_info->uci_ind.num_ucis;
  nfapi_nr_uci_t *uci_list = UL_info->uci_ind.uci_list;
  uint8_t UE_id = 0;

  for (int i = 0; i < num_ucis; i++) {
    switch (uci_list[i].pdu_type) {
      case NFAPI_NR_UCI_PDCCH_PDU_TYPE: break;

      case NFAPI_NR_UCI_FORMAT_0_1_PDU_TYPE: {
        if (get_softmodem_params()->phy_test == 0) {
          nfapi_nr_uci_pucch_pdu_format_0_1_t *uci_pdu = &uci_list[i].pucch_pdu_format_0_1;
          // handle harq
          int harq_idx_s = 0;

        if ( uci_pdu->sr->sr_indication && !(uci_pdu->sr->sr_confidence_level)) {
          sched_ctrl->sr_req.nr_of_srs =1;
          sched_ctrl->sr_req.ul_SR[0] = 1;
        }

          // iterate over received harq bits
          for (int harq_bit = 0; harq_bit < uci_pdu->harq->num_harq; harq_bit++) {
            // search for the right harq process
            for (int harq_idx = harq_idx_s; harq_idx < NR_MAX_NB_HARQ_PROCESSES-1; harq_idx++) {
              if ((UL_info->slot-1) == sched_ctrl->harq_processes[harq_idx].feedback_slot) {
                if (uci_pdu->harq->harq_list[harq_bit].harq_value == 0)
                  sched_ctrl->harq_processes[harq_idx].round++;
                if ((uci_pdu->harq->harq_list[harq_bit].harq_value == 1) ||
                   (sched_ctrl->harq_processes[harq_idx].round == max_harq_rounds)) {
                  sched_ctrl->harq_processes[harq_idx].ndi ^= 1;
                  sched_ctrl->harq_processes[harq_idx].round = 0;
                }
                sched_ctrl->harq_processes[harq_idx].is_waiting = 0;
                harq_idx_s = harq_idx + 1;
                break;
              }
              // if gNB fails to receive a ACK/NACK
              else if (((UL_info->slot-1) > sched_ctrl->harq_processes[harq_idx].feedback_slot) &&
                      (sched_ctrl->harq_processes[harq_idx].is_waiting)) {
                sched_ctrl->harq_processes[harq_idx].round++;
                if (sched_ctrl->harq_processes[harq_idx].round == max_harq_rounds) {
                  sched_ctrl->harq_processes[harq_idx].ndi ^= 1;
                  sched_ctrl->harq_processes[harq_idx].round = 0;
                }
                sched_ctrl->harq_processes[harq_idx].is_waiting = 0;
              }
            }
          }
        }
        break;
      }

      case NFAPI_NR_UCI_FORMAT_2_3_4_PDU_TYPE: {
        nfapi_nr_uci_pucch_pdu_format_2_3_4_t *uci_pdu = &uci_list[i].pucch_pdu_format_2_3_4;
        module_id_t Mod_idP = UL_info -> module_id;
        NR_CSI_MeasConfig_t *csi_MeasConfig = RC.nrmac[Mod_idP]->UE_list.secondaryCellGroup[UE_id]->spCellConfig->spCellConfigDedicated->csi_MeasConfig->choice.setup;

        if ( uci_pdu -> pduBitmap & 0x01 ) {
          ///Handle SR PDU
          uint8_t sr_id = 0;
          uint8_t payload_size = ceil(uci_pdu->sr.sr_bit_len/8);
          uint8_t *payload = calloc (payload_size, sizeof (uint8_t));
          memcpy (payload, uci_pdu->sr.sr_payload, payload_size);

          for (sr_id = 0; sr_id < uci_pdu->sr.sr_bit_len; sr_id++) {
            sched_ctrl->sr_req.ul_SR[sr_id] = *payload & 1;
            *payload >>= 1;
          }

          sched_ctrl->sr_req.nr_of_srs = uci_pdu->sr.sr_bit_len;
        }

        if (uci_pdu -> pduBitmap & 0x02) {
          ///Hadle Harq
          int harq_idx_s = 0;

          // iterate over received harq bits
          for (int harq_bit = 0; harq_bit < uci_pdu->harq.harq_bit_len; harq_bit++) {
            // search for the right harq process
            for (int harq_idx = harq_idx_s; harq_idx < NR_MAX_NB_HARQ_PROCESSES-1; harq_idx++) {
              if ((UL_info->slot-1) == sched_ctrl->harq_processes[harq_idx].feedback_slot) {
                if ((*(uci_pdu->harq.harq_payload)&(1<<harq_bit)) == 0)
                  sched_ctrl->harq_processes[harq_idx].round++;

                if (((*(uci_pdu->harq.harq_payload) & (1 << harq_bit)) == 1) ||
                    (sched_ctrl->harq_processes[harq_idx].round == max_harq_rounds)) {
                  sched_ctrl->harq_processes[harq_idx].ndi ^= 1;
                  sched_ctrl->harq_processes[harq_idx].round = 0;
                }

                sched_ctrl->harq_processes[harq_idx].is_waiting = 0;
                harq_idx_s = harq_idx + 1;
                break;
              }
              // if gNB fails to receive a ACK/NACK
              else if (((UL_info->slot-1) > sched_ctrl->harq_processes[harq_idx].feedback_slot) &&
                       (sched_ctrl->harq_processes[harq_idx].is_waiting)) {
                sched_ctrl->harq_processes[harq_idx].round++;

                if (sched_ctrl->harq_processes[harq_idx].round == max_harq_rounds) {
                  sched_ctrl->harq_processes[harq_idx].ndi ^= 1;
                  sched_ctrl->harq_processes[harq_idx].round = 0;
                }

                sched_ctrl->harq_processes[harq_idx].is_waiting = 0;
              }
            }
          }
        }

        if (uci_pdu -> pduBitmap & 0x04) {
          int bwp_id =1;
          NR_BWP_Uplink_t *ubwp=RC.nrmac[Mod_idP]->UE_list.secondaryCellGroup[UE_id]->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.array[bwp_id-1];
          NR_SubcarrierSpacing_t scs=ubwp->bwp_Common->genericParameters.subcarrierSpacing;
          LOG_I(PHY,"SFN/SF:%d%d scs %d \n",
            UL_info->frame,UL_info->slot,
            scs);
          //API to parse the csi report and store it into sched_ctrl
          extract_pucch_csi_report (csi_MeasConfig, uci_pdu, sched_ctrl,UL_info->frame, UL_info->slot, scs);
        }

        if (uci_pdu -> pduBitmap & 0x08) {
          ///Handle CSI Report 2
        }

        break;
      }
    }
  }

  UL_info->uci_ind.num_ucis = 0;
}


void handle_nr_ulsch(NR_UL_IND_t *UL_info) {
  if(nfapi_mode == 1) {
    if (UL_info->crc_ind.number_crcs>0) {
      //LOG_D(PHY,"UL_info->crc_ind.crc_indication_body.number_of_crcs:%d CRC_IND:SFN/SF:%d\n", UL_info->crc_ind.crc_indication_body.number_of_crcs, NFAPI_SFNSF2DEC(UL_info->crc_ind.sfn_sf));
      //      oai_nfapi_crc_indication(&UL_info->crc_ind);

      UL_info->crc_ind.number_crcs = 0;
    }

    if (UL_info->rx_ind.number_of_pdus>0) {
      //LOG_D(PHY,"UL_info->rx_ind.number_of_pdus:%d RX_IND:SFN/SF:%d\n", UL_info->rx_ind.rx_indication_body.number_of_pdus, NFAPI_SFNSF2DEC(UL_info->rx_ind.sfn_sf));
      //      oai_nfapi_rx_ind(&UL_info->rx_ind);
      UL_info->rx_ind.number_of_pdus = 0;
    }
  } else {

    if (UL_info->rx_ind.number_of_pdus>0 && UL_info->crc_ind.number_crcs>0) {
      for (int i=0; i<UL_info->rx_ind.number_of_pdus; i++) {
        for (int j=0; j<UL_info->crc_ind.number_crcs; j++) {
          // find crc_indication j corresponding rx_indication i
          LOG_D(PHY,"UL_info->crc_ind.crc_indication_body.crc_pdu_list[%d].rx_ue_information.rnti:%04x UL_info->rx_ind.rx_indication_body.rx_pdu_list[%d].rx_ue_information.rnti:%04x\n", j,
                UL_info->crc_ind.crc_list[j].rnti, i, UL_info->rx_ind.pdu_list[i].rnti);

          if (UL_info->crc_ind.crc_list[j].rnti ==
              UL_info->rx_ind.pdu_list[i].rnti) {
            LOG_D(PHY, "UL_info->crc_ind.crc_indication_body.crc_pdu_list[%d].crc_indication_rel8.crc_flag:%d\n", j, UL_info->crc_ind.crc_list[j].tb_crc_status);

            if (UL_info->crc_ind.crc_list[j].tb_crc_status == 1) { // CRC error indication
              LOG_D(MAC,"Frame %d, Slot %d Calling rx_sdu (CRC error) \n",UL_info->frame,UL_info->slot);

              nr_rx_sdu(UL_info->module_id,
                        UL_info->CC_id,
                        UL_info->rx_ind.sfn, //UL_info->frame,
                        UL_info->rx_ind.slot, //UL_info->slot,
                        UL_info->rx_ind.pdu_list[i].rnti,
                        (uint8_t *)NULL,
                        UL_info->rx_ind.pdu_list[i].pdu_length,
                        UL_info->rx_ind.pdu_list[i].timing_advance,
                        UL_info->rx_ind.pdu_list[i].ul_cqi);
            } else {
              LOG_D(MAC,"Frame %d, Slot %d Calling rx_sdu (CRC ok) \n",UL_info->frame,UL_info->slot);
              nr_rx_sdu(UL_info->module_id,
                        UL_info->CC_id,
                        UL_info->rx_ind.sfn, //UL_info->frame,
                        UL_info->rx_ind.slot, //UL_info->slot,
                        UL_info->rx_ind.pdu_list[i].rnti,
                        UL_info->rx_ind.pdu_list[i].pdu,
                        UL_info->rx_ind.pdu_list[i].pdu_length,
                        UL_info->rx_ind.pdu_list[i].timing_advance,
                        UL_info->rx_ind.pdu_list[i].ul_cqi);
            }
            break;
          }
        } //    for (j=0;j<UL_info->crc_ind.number_crcs;j++)
      } //   for (i=0;i<UL_info->rx_ind.number_of_pdus;i++)

      UL_info->crc_ind.number_crcs=0;
      UL_info->rx_ind.number_of_pdus = 0;
    }
    else if (UL_info->rx_ind.number_of_pdus!=0 || UL_info->crc_ind.number_crcs!=0) {
      LOG_E(PHY,"hoping not to have mis-match between CRC ind and RX ind - hopefully the missing message is coming shortly rx_ind:%d(SFN/SL:%d/%d) crc_ind:%d(SFN/SL:%d/%d) \n",
            UL_info->rx_ind.number_of_pdus, UL_info->rx_ind.sfn, UL_info->rx_ind.slot,
            UL_info->crc_ind.number_crcs, UL_info->rx_ind.sfn, UL_info->rx_ind.slot);
    }
  }
}

void NR_UL_indication(NR_UL_IND_t *UL_info) {
  AssertFatal(UL_info!=NULL,"UL_INFO is null\n");
#ifdef DUMP_FAPI
  dump_ul(UL_info);
#endif
  module_id_t      module_id   = UL_info->module_id;
  int              CC_id       = UL_info->CC_id;
  NR_Sched_Rsp_t   *sched_info = &Sched_INFO[module_id][CC_id];
  NR_IF_Module_t   *ifi        = if_inst[module_id];
  gNB_MAC_INST     *mac        = RC.nrmac[module_id];

  LOG_I(PHY,"SFN/SF:%d%d module_id:%d CC_id:%d UL_info[rach_pdus:%d rx_ind:%d crcs:%d]\n",
        UL_info->frame,UL_info->slot,
        module_id,CC_id, UL_info->rach_ind.number_of_pdus,
        UL_info->rx_ind.number_of_pdus, UL_info->crc_ind.number_crcs);

  if (nfapi_mode != 1) {
    if (ifi->CC_mask==0) {
      ifi->current_frame    = UL_info->frame;
      ifi->current_slot = UL_info->slot;
    } else {
      AssertFatal(UL_info->frame != ifi->current_frame,"CC_mask %x is not full and frame has changed\n",ifi->CC_mask);
      AssertFatal(UL_info->slot != ifi->current_slot,"CC_mask %x is not full and slot has changed\n",ifi->CC_mask);
    }

    ifi->CC_mask |= (1<<CC_id);
  }

  // clear DL/UL info for new scheduling round
  clear_nr_nfapi_information(mac,CC_id,UL_info->frame,UL_info->slot);
  handle_nr_rach(UL_info);
  handle_nr_uci(UL_info, &mac->UE_list.UE_sched_ctrl[0]);
  // clear HI prior to handling ULSCH
  mac->UL_dci_req[CC_id].numPdus = 0;
  handle_nr_ulsch(UL_info);

  if (nfapi_mode != 1) {
    if (ifi->CC_mask == ((1<<MAX_NUM_CCs)-1)) {
      /*
      eNB_dlsch_ulsch_scheduler(module_id,
          (UL_info->frame+((UL_info->slot>(9-sl_ahead))?1:0)) % 1024,
          (UL_info->slot+sl_ahead)%10);
      */
      nfapi_nr_config_request_scf_t *cfg = &mac->config[CC_id];
      int spf = get_spf(cfg);
      gNB_dlsch_ulsch_scheduler(module_id,
				UL_info->frame,
				UL_info->slot,
				(UL_info->frame+((UL_info->slot>(spf-1-sl_ahead))?1:0)) % 1024,
				(UL_info->slot+sl_ahead)%spf);
      
      ifi->CC_mask            = 0;
      sched_info->module_id   = module_id;
      sched_info->CC_id       = CC_id;
      sched_info->frame       = (UL_info->frame + ((UL_info->slot>(spf-1-sl_ahead)) ? 1 : 0)) % 1024;
      sched_info->slot        = (UL_info->slot+sl_ahead)%spf;
      sched_info->DL_req      = &mac->DL_req[CC_id];
      sched_info->UL_dci_req  = &mac->UL_dci_req[CC_id];

      if ((mac->common_channels[CC_id].ServingCellConfigCommon->tdd_UL_DL_ConfigurationCommon==NULL) ||
          (is_nr_UL_slot(mac->common_channels[CC_id].ServingCellConfigCommon,UL_info->slot)>0)) {
	//printf("NR_UL_indication: this is an UL slot. UL_info: frame %d, slot %d. UL_tti_req: frame %d, slot %d\n",UL_info->frame,UL_info->slot,mac->UL_tti_req[CC_id].SFN,mac->UL_tti_req[CC_id].Slot);
        sched_info->UL_tti_req      = &mac->UL_tti_req[CC_id];
      }
      else
        sched_info->UL_tti_req      = NULL;

      sched_info->TX_req      = &mac->TX_req[CC_id];
#ifdef DUMP_FAPI
      dump_dl(sched_info);
#endif

      if (ifi->NR_Schedule_response) {
        AssertFatal(ifi->NR_Schedule_response!=NULL,
                    "nr_schedule_response is null (mod %d, cc %d)\n",
                    module_id,
                    CC_id);
        ifi->NR_Schedule_response(sched_info);
      }

      LOG_D(PHY,"NR_Schedule_response: SFN_SF:%d%d dl_pdus:%d\n",
	    sched_info->frame,
	    sched_info->slot,
	    sched_info->DL_req->dl_tti_request_body.nPDUs);
    }
  }
}

NR_IF_Module_t *NR_IF_Module_init(int Mod_id) {
  AssertFatal(Mod_id<MAX_MODULES,"Asking for Module %d > %d\n",Mod_id,MAX_IF_MODULES);
  LOG_I(PHY,"Installing callbacks for IF_Module - UL_indication\n");

  if (if_inst[Mod_id]==NULL) {
    if_inst[Mod_id] = (NR_IF_Module_t*)malloc(sizeof(NR_IF_Module_t));
    memset((void*)if_inst[Mod_id],0,sizeof(NR_IF_Module_t));

    LOG_I(MAC,"Allocating shared L1/L2 interface structure for instance %d @ %p\n",Mod_id,if_inst[Mod_id]);

    if_inst[Mod_id]->CC_mask=0;
    if_inst[Mod_id]->NR_UL_indication = NR_UL_indication;
    AssertFatal(pthread_mutex_init(&if_inst[Mod_id]->if_mutex,NULL)==0,
                "allocation of if_inst[%d]->if_mutex fails\n",Mod_id);
  }

  return if_inst[Mod_id];
}
