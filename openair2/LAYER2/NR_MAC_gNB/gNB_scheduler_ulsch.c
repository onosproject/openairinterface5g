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

/*! \file gNB_scheduler_ulsch.c
 * \brief gNB procedures for the ULSCH transport channel
 * \author Navid Nikaein and Raymond Knopp, Guido Casati
 * \date 2019
 * \email: guido.casati@iis.fraunhofer.de
 * \version 1.0
 * @ingroup _mac
 */

#include "LAYER2/NR_MAC_gNB/mac_proto.h"

/*
* When data are received on PHY and transmitted to MAC
*/
void nr_rx_sdu(const module_id_t gnb_mod_idP,
               const int CC_idP,
               const frame_t frameP,
               const sub_frame_t subframeP,
               const rnti_t rntiP,
               uint8_t *sduP,
               const uint16_t sdu_lenP,
               const uint16_t timing_advance,
               const uint8_t ul_cqi){
  int current_rnti = 0, UE_id = -1, harq_pid = 0;
  gNB_MAC_INST *gNB_mac = NULL;
  NR_UE_list_t *UE_list = NULL;
  UE_sched_ctrl_t *UE_scheduling_control = NULL;

  current_rnti = rntiP;
  UE_id = find_nr_UE_id(gnb_mod_idP, current_rnti);
  gNB_mac = RC.nrmac[gnb_mod_idP];
  UE_list = &gNB_mac->UE_list;

  if (UE_id != -1) {
    UE_scheduling_control = &(UE_list->UE_sched_ctrl[UE_id]);

    LOG_D(MAC, "[gNB %d][PUSCH %d] CC_id %d %d.%d Received ULSCH sdu round %d from PHY (rnti %x, UE_id %d) ul_cqi %d\n",
          gnb_mod_idP,
          harq_pid,
          CC_idP,
          frameP,
          subframeP,
          UE_scheduling_control->round_UL[CC_idP][harq_pid],
          current_rnti,
          UE_id,
          ul_cqi);

    if (sduP != NULL)
      UE_scheduling_control->ta_update = timing_advance;
  }
}

void nr_schedule_ulsch_rnti(module_id_t module_idP, frame_t frameP, sub_frame_t slotP) {
  gNB_MAC_INST *nr_mac = RC.nrmac[module_idP];
  NR_COMMON_channels_t *cc = nr_mac->common_channels;
  NR_ServingCellConfigCommon_t *scc = cc->ServingCellConfigCommon;

  const int bwp_id = 1;
  const int UE_id = 0;

  NR_UE_list_t *UE_list = &RC.nrmac[module_idP]->UE_list;
  AssertFatal(UE_list->active[UE_id] >= 0,
              "Cannot find UE_id %d is not active\n",
              UE_id);

  NR_CellGroupConfig_t *secondaryCellGroup = UE_list->secondaryCellGroup[UE_id];
  AssertFatal(secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count == 1,
              "downlinkBWP_ToAddModList has %d BWP!\n",
              secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.count);
  NR_BWP_Uplink_t *ubwp = secondaryCellGroup->spCellConfig->spCellConfigDedicated->uplinkConfig->uplinkBWP_ToAddModList->list.array[bwp_id - 1];
  NR_BWP_Downlink_t *bwp = secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList->list.array[bwp_id - 1];

  uint16_t rnti = UE_list->rnti[UE_id];
  nfapi_nr_ul_dci_request_pdus_t *ul_dci_request_pdu;

  nfapi_nr_ul_tti_request_t *UL_tti_req = &RC.nrmac[module_idP]->UL_tti_req[0];
  UL_tti_req->SFN = frameP;
  UL_tti_req->Slot = slotP;
  UL_tti_req->n_pdus = 1;
  UL_tti_req->pdus_list[0].pdu_type = NFAPI_NR_UL_CONFIG_PUSCH_PDU_TYPE;
  UL_tti_req->pdus_list[0].pdu_size = sizeof(nfapi_nr_pusch_pdu_t);
  nfapi_nr_pusch_pdu_t *pusch_pdu = &UL_tti_req->pdus_list[0].pusch_pdu;
  memset(pusch_pdu, 0, sizeof(nfapi_nr_pusch_pdu_t));

  LOG_D(MAC, "Scheduling UE specific PUSCH\n");
  // UL_tti_req = &nr_mac->UL_tti_req[CC_id];
  /*
  // original configuration
  rel15_ul->rnti                           = 0x1234;
  rel15_ul->ulsch_pdu_rel15.start_rb       = 30;
  rel15_ul->ulsch_pdu_rel15.number_rbs     = 50;
  rel15_ul->ulsch_pdu_rel15.start_symbol   = 2;
  rel15_ul->ulsch_pdu_rel15.number_symbols = 12;
  rel15_ul->ulsch_pdu_rel15.nb_re_dmrs     = 6;
  rel15_ul->ulsch_pdu_rel15.length_dmrs    = 1;
  rel15_ul->ulsch_pdu_rel15.Qm             = 2;
  rel15_ul->ulsch_pdu_rel15.mcs            = 9;
  rel15_ul->ulsch_pdu_rel15.rv             = 0;
  rel15_ul->ulsch_pdu_rel15.n_layers       = 1;
  */
  pusch_pdu->pdu_bit_map = PUSCH_PDU_BITMAP_PUSCH_DATA;
  pusch_pdu->rnti = rnti;
  pusch_pdu->handle = 0; // not yet used

  pusch_pdu->bwp_size = NRRIV2BW(ubwp->bwp_Common->genericParameters.locationAndBandwidth, 275);
  pusch_pdu->bwp_start = NRRIV2PRBOFFSET(ubwp->bwp_Common->genericParameters.locationAndBandwidth, 275);
  pusch_pdu->subcarrier_spacing = ubwp->bwp_Common->genericParameters.subcarrierSpacing;
  pusch_pdu->cyclic_prefix = 0;
  // pusch information always include
  // this informantion seems to be redundant. with the mcs_index and the
  // modulation table, the mod_order and target_code_rate can be determined.
  pusch_pdu->mcs_index = 9;
  // 0: notqam256 [TS38.214, table 5.1.3.1-1] - corresponds to nr_target_code_rate_table1 in PHY
  pusch_pdu->mcs_table = 0;
  pusch_pdu->target_code_rate = nr_get_code_rate_ul(pusch_pdu->mcs_index, pusch_pdu->mcs_table);
  pusch_pdu->qam_mod_order = nr_get_Qm_ul(pusch_pdu->mcs_index, pusch_pdu->mcs_table);
  pusch_pdu->transform_precoding = 0;
  // It equals the higher-layer parameter Data-scrambling-Identity if
  // configured and the RNTI equals the C-RNTI, otherwise L2 needs to set it to
  // physical cell id.;
  pusch_pdu->data_scrambling_id = 0;
  pusch_pdu->nrOfLayers = 1; // DMRS
  pusch_pdu->ul_dmrs_symb_pos = 1;
  pusch_pdu->dmrs_config_type = 0; // dmrs-type 1 (single DMRS symbol in the beginning)
  // If provided and the PUSCH is not a msg3 PUSCH, otherwise, L2 should set
  // this to physical cell id.
  pusch_pdu->ul_dmrs_scrambling_id = 0;
  // DMRS sequence initialization [TS38.211, sec 6.4.1.1.1]. Should match what
  // is sent in DCI 0_1, otherwise set to 0.
  pusch_pdu->scid = 0;
  // pusch_pdu->num_dmrs_cdm_grps_no_data;
  // pusch_pdu->dmrs_ports; //DMRS ports. [TS38.212 7.3.1.1.2] provides
  // description between DCI 0-1 content and DMRS ports. Bitmap occupying the 11
  // LSBs with: bit 0: antenna port 1000 bit 11: antenna port 1011 and for each
  // bit 0: DMRS port not used 1: DMRS port used Pusch Allocation in frequency
  // domain [TS38.214, sec 6.1.2.2]
  pusch_pdu->resource_alloc = 1; // type 1
  // pusch_pdu->rb_bitmap;// for ressource alloc type 0
  pusch_pdu->rb_start = 0;
  pusch_pdu->rb_size = 50;
  pusch_pdu->vrb_to_prb_mapping = 0;
  pusch_pdu->frequency_hopping = 0;
  // pusch_pdu->tx_direct_current_location;//The uplink Tx Direct Current
  // location for the carrier. Only values in the value range of this field
  // between 0 and 3299, which indicate the subcarrier index within the carrier
  // corresponding 1o the numerology of the corresponding uplink BWP and value
  // 3300, which indicates "Outside the carrier" and value 3301, which indicates
  // "Undetermined position within the carrier" are used. [TS38.331,
  // UplinkTxDirectCurrentBWP IE]
  pusch_pdu->uplink_frequency_shift_7p5khz = 0; // Resource Allocation in time domain
  pusch_pdu->start_symbol_index = 0;
  pusch_pdu->nr_of_symbols = 12;
  // Optional Data only included if indicated in pduBitmap
  pusch_pdu->pusch_data.rv_index = 0;
  pusch_pdu->pusch_data.harq_process_id = 0;
  pusch_pdu->pusch_data.new_data_indicator = 0;
  pusch_pdu->pusch_data.tb_size =
      nr_compute_tbs(pusch_pdu->mcs_index,
                     pusch_pdu->target_code_rate,
                     pusch_pdu->rb_size,
                     pusch_pdu->nr_of_symbols,
                     6, // nb_re_dmrs - not sure where this is coming from - its
                        // not in the FAPI
                     0, // nb_rb_oh
                     pusch_pdu->nrOfLayers = 1);
  pusch_pdu->pusch_data.num_cb = 0; // CBG not supported
  // pusch_pdu->pusch_data.cb_present_and_position;
  // pusch_pdu->pusch_uci;
  // pusch_pdu->pusch_ptrs;
  // pusch_pdu->dfts_ofdm;
  // beamforming
  // pusch_pdu->beamforming; //not used for now

  nfapi_nr_ul_dci_request_t *UL_dci_req = &RC.nrmac[module_idP]->UL_dci_req[0];
  ul_dci_request_pdu = &UL_dci_req->ul_dci_pdu_list[UL_dci_req->numPdus];
  memset(ul_dci_request_pdu, 0, sizeof(nfapi_nr_ul_dci_request_pdus_t));
  ul_dci_request_pdu->PDUType = NFAPI_NR_DL_TTI_PDCCH_PDU_TYPE;
  ul_dci_request_pdu->PDUSize = (uint8_t)(2 + sizeof(nfapi_nr_dl_tti_pdcch_pdu));

  int CCEIndex = allocate_nr_CCEs(nr_mac,
                                  1, // bwp_id
                                  0, // coreset_id
                                  4, // aggregation,
                                  1, // search_space, 0 common, 1 ue-specific
                                  UE_id,
                                  0); // m

  nfapi_nr_dl_tti_pdcch_pdu_rel15_t *pdcch_pdu_rel15 =
      &ul_dci_request_pdu->pdcch_pdu.pdcch_pdu_rel15;
  LOG_D(MAC, "Configuring ULDCI/PDCCH in %d.%d\n", frameP, slotP);
  nr_configure_pdcch(pdcch_pdu_rel15,
                     1, // ue-specific,
                     scc,
                     bwp);

  dci_pdu_rel15_t dci_pdu_rel15[MAX_DCI_CORESET];

  AssertFatal(CCEIndex >= 0, "CCEIndex is negative \n");
  pdcch_pdu_rel15->CceIndex[pdcch_pdu_rel15->numDlDci] = CCEIndex;

  LOG_D(PHY,
        "CCEIndex %d\n",
        pdcch_pdu_rel15->CceIndex[pdcch_pdu_rel15->numDlDci]);

  int dci_formats[2] = {NR_UL_DCI_FORMAT_0_0, 0};
  int rnti_types[2] = {NR_RNTI_C, 0};
  config_uldci(ubwp,
               pusch_pdu,
               pdcch_pdu_rel15,
               &dci_pdu_rel15[0],
               dci_formats,
               rnti_types);

  pdcch_pdu_rel15->PayloadSizeBits[0] =
      nr_dci_size(dci_formats[0], rnti_types[0], pdcch_pdu_rel15->BWPSize);
  fill_dci_pdu_rel15( pdcch_pdu_rel15, &dci_pdu_rel15[0], dci_formats, rnti_types);
}
