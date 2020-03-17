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

/*! \file       gNB_scheduler_dlsch.c
 * \brief       procedures related to gNB for the DLSCH transport channel
 * \author      Guido Casati
 * \date        2019
 * \email:      guido.casati@iis.fraunhofe.de
 * \version     1.0
 * @ingroup     _mac

 */

/*PHY*/
#include "PHY/CODING/coding_defs.h"
#include "PHY/defs_nr_common.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
/*MAC*/
#include "LAYER2/NR_MAC_COMMON/nr_mac.h"
#include "LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "LAYER2/NR_MAC_COMMON/nr_mac_extern.h"
#include "LAYER2/NR_MAC_gNB/mac_proto.h"

/*NFAPI*/
#include "nfapi_nr_interface.h"
/*TAG*/
#include "NR_TAG-Id.h"


int nr_generate_dlsch_pdu(module_id_t module_idP,
                          unsigned char *sdus_payload,
                          unsigned char *mac_pdu,
                          unsigned char num_sdus,
                          unsigned short *sdu_lengths,
                          unsigned char *sdu_lcids,
                          unsigned char drx_cmd,
                          unsigned char *ue_cont_res_id,
                          unsigned short post_padding){

  gNB_MAC_INST *gNB = RC.nrmac[module_idP];

  NR_MAC_SUBHEADER_FIXED *mac_pdu_ptr = (NR_MAC_SUBHEADER_FIXED *) mac_pdu;
  unsigned char * dlsch_buffer_ptr = sdus_payload;
  uint8_t last_size = 0;
  int offset = 0, mac_ce_size, i, timing_advance_cmd, tag_id = 0;

  // MAC CEs 
  uint8_t mac_header_control_elements[16], *ce_ptr;
  ce_ptr = &mac_header_control_elements[0];

  // 1) Compute MAC CE and related subheaders 

  // DRX command subheader (MAC CE size 0)
  if (drx_cmd != 255) {
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_DRX;
    //last_size = 1;
    mac_pdu_ptr++;
  }

  // Timing Advance subheader
  /* This was done only when timing_advance_cmd != 31
  // now TA is always send when ta_timer resets regardless of its value
  // this is done to avoid issues with the timeAlignmentTimer which is
  // supposed to monitor if the UE received TA or not */
  if (gNB->ta_len){
    mac_pdu_ptr->R = 0;
    mac_pdu_ptr->LCID = DL_SCH_LCID_TA_COMMAND;
    //last_size = 1;
    mac_pdu_ptr++;

    // TA MAC CE (1 octet)
    timing_advance_cmd = gNB->ta_command;
    AssertFatal(timing_advance_cmd < 64,"timing_advance_cmd %d > 63\n", timing_advance_cmd);
    ((NR_MAC_CE_TA *) ce_ptr)->TA_COMMAND = timing_advance_cmd;    //(timing_advance_cmd+31)&0x3f;
    if (gNB->tag->tag_Id != 0){
       tag_id = gNB->tag->tag_Id;
      ((NR_MAC_CE_TA *) ce_ptr)->TAGID = tag_id;
    }

    LOG_D(MAC, "NR MAC CE timing advance command = %d (%d) TAG ID = %d\n", timing_advance_cmd, ((NR_MAC_CE_TA *) ce_ptr)->TA_COMMAND, tag_id);
    mac_ce_size = sizeof(NR_MAC_CE_TA);

    // Copying  bytes for MAC CEs to the mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *) ce_ptr, mac_ce_size);
    ce_ptr += mac_ce_size;
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }


  // Contention resolution fixed subheader and MAC CE
  if (ue_cont_res_id) {
    mac_pdu_ptr->R = 0;
  	mac_pdu_ptr->LCID = DL_SCH_LCID_CON_RES_ID;
    mac_pdu_ptr++;
    //last_size = 1;

    // contention resolution identity MAC ce has a fixed 48 bit size
    // this contains the UL CCCH SDU. If UL CCCH SDU is longer than 48 bits, 
    // it contains the first 48 bits of the UL CCCH SDU
    LOG_T(MAC, "[gNB ][RAPROC] Generate contention resolution msg: %x.%x.%x.%x.%x.%x\n",
        ue_cont_res_id[0], ue_cont_res_id[1], ue_cont_res_id[2],
        ue_cont_res_id[3], ue_cont_res_id[4], ue_cont_res_id[5]);

    // Copying bytes (6 octects) to CEs pointer
    mac_ce_size = 6;
    memcpy(ce_ptr, ue_cont_res_id, mac_ce_size);
    
    // Copying bytes for MAC CEs to mac pdu pointer
    memcpy((void *) mac_pdu_ptr, (void *) ce_ptr, mac_ce_size);
    ce_ptr += mac_ce_size;
    mac_pdu_ptr += (unsigned char) mac_ce_size;
  }


  // 2) Generation of DLSCH MAC SDU subheaders
  for (i = 0; i < num_sdus; i++) {
    LOG_D(MAC, "[gNB] Generate DLSCH header num sdu %d len sdu %d\n", num_sdus, sdu_lengths[i]);

    if (sdu_lengths[i] < 128) {
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->R = 0;
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->F = 0;
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->LCID = sdu_lcids[i];
      ((NR_MAC_SUBHEADER_SHORT *) mac_pdu_ptr)->L = (unsigned char) sdu_lengths[i];
      last_size = 2;
    } else {
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->R = 0;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->F = 1;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->LCID = sdu_lcids[i];
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->L1 = ((unsigned short) sdu_lengths[i] >> 8) & 0x7f;
      ((NR_MAC_SUBHEADER_LONG *) mac_pdu_ptr)->L2 = (unsigned short) sdu_lengths[i] & 0xff;
      last_size = 3;
    }

    mac_pdu_ptr += last_size;

    // 3) cycle through SDUs, compute each relevant and place dlsch_buffer in   
    memcpy((void *) mac_pdu_ptr, (void *) dlsch_buffer_ptr, sdu_lengths[i]);
    dlsch_buffer_ptr+= sdu_lengths[i]; 
    mac_pdu_ptr += sdu_lengths[i];
  }

  // 4) Compute final offset for padding
  if (post_padding > 0) {    
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->R = 0;
    ((NR_MAC_SUBHEADER_FIXED *) mac_pdu_ptr)->LCID = DL_SCH_LCID_PADDING;
    mac_pdu_ptr++;

  } else {            
    // no MAC subPDU with padding
  }

  // compute final offset
  offset = ((unsigned char *) mac_pdu_ptr - mac_pdu);
    
  //printf("Offset %d \n", ((unsigned char *) mac_pdu_ptr - mac_pdu));

  return offset;
}

uint16_t getBWPsize(module_id_t Mod_id, int UE_id, int bwp_id, int N_RB) {
  NR_UE_list_t *UE_list = &RC.nrmac[Mod_id]->UE_list;
  NR_CellGroupConfig_t *secondaryCellGroup = UE_list->secondaryCellGroup[UE_id];
  struct NR_ServingCellConfig__downlinkBWP_ToAddModList *BWP_ToAddModList =
    secondaryCellGroup->spCellConfig->spCellConfigDedicated->downlinkBWP_ToAddModList;
  AssertFatal(BWP_ToAddModList->list.count == 1,
              "downlinkBWP_ToAddModList has %d BWP!\n",
              BWP_ToAddModList->list.count);
  NR_BWP_Downlink_t *bwp = BWP_ToAddModList->list.array[bwp_id - 1];
  return NRRIV2BW(bwp->bwp_Common->genericParameters.locationAndBandwidth, N_RB);
}

void nr_dlsch_buffer_update(module_id_t Mod_id, frame_t frame, sub_frame_t slot) {
  NR_UE_list_t *UE_list = &RC.nrmac[Mod_id]->UE_list;

  for (int UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id]) {
    const rnti_t rnti = UE_list->rnti[UE_id];
    UE_sched_ctrl_t *sched_ctrl = &UE_list->UE_sched_ctrl[UE_id];
    sched_ctrl->dl_buffer_total = 0;

    for (int i = 0; i < sched_ctrl->dl_lc_num; i++) {
      const uint8_t lcid = sched_ctrl->dl_lc_ids[i];
      mac_rlc_status_resp_t resp = mac_rlc_status_ind(Mod_id,
                                                      rnti,
                                                      Mod_id,
                                                      frame,
                                                      slot,
                                                      ENB_FLAG_YES,
                                                      MBMS_FLAG_NO,
                                                      lcid,
                                                      0,
                                                      0);
      sched_ctrl->dl_buffer_total += resp.bytes_in_buffer;
      sched_ctrl->dl_lc_bytes[lcid] = resp.bytes_in_buffer;

      LOG_D(MAC,
            "[gNB %d] %d.%d, UE %d RNTI %04x, LC %d, total buffer %d\n",
            Mod_id,
            frame,
            slot,
            UE_id,
            rnti,
            lcid,
            sched_ctrl->dl_buffer_total);
    }
  }
}

void nr_dlsch_pre_processor(module_id_t Mod_id, frame_t f, sub_frame_t s) {
  NR_UE_list_t *UE_list = &RC.nrmac[Mod_id]->UE_list;
  AssertFatal(UE_list->num_UEs == 1,
              "the scheduler handles only one UE at the moment, but found %d\n",
              UE_list->num_UEs);

  const int N_RB = 275;
  const int UE_id = 0;
  const int CC_id = 0;
  const int bwp_id = 1;
  UE_sched_ctrl_t *sched_ctrl = &UE_list->UE_sched_ctrl[UE_id];

  sched_ctrl->pre_nb_available_rbs[CC_id] = 0;
  if (sched_ctrl->dl_buffer_total == 0)
    return;

  // give all available RBs to this UE, start is implicit (0)
  const uint16_t rbSize = getBWPsize(Mod_id, UE_id, bwp_id, N_RB);
  sched_ctrl->pre_nb_available_rbs[CC_id] = rbSize;
  sched_ctrl->dlsch_mcs = 9;
}

void nr_schedule_ue_spec(module_id_t module_idP, frame_t frameP, sub_frame_t slotP){
  const int bwp_id = 1;
  const int CC_id = 0;

  gNB_MAC_INST *gNB_mac = RC.nrmac[module_idP];
  NR_UE_list_t *UE_list = &gNB_mac->UE_list;
  if (UE_list->num_UEs == 0)
    return;

  nr_dlsch_buffer_update(module_idP, frameP, slotP);

  /* the preprocessor determines resource block allocation, MCS, etc. for all
   * UEs.  Currently, these values are kept in the (LTE) UE_sched_ctrl_t
   * structure */
  nr_dlsch_pre_processor(module_idP, frameP, slotP);

  LOG_D(MAC, "Scheduling UE specific search space DCI type 1\n");
  int rbStart = 0;
  for (int UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id]) {
    UE_sched_ctrl_t *sched_ctrl = &UE_list->UE_sched_ctrl[UE_id];
    if (sched_ctrl->pre_nb_available_rbs[CC_id] == 0)
      continue;

    int CCEIndex = allocate_nr_CCEs(gNB_mac,
                                    bwp_id, // bwp_id
                                    0, // coreset_id
                                    4, // aggregation,
                                    1, // search_space, 0 common, 1 ue-specific
                                    UE_id,
                                    0); // m
    if (CCEIndex < 0) {
      LOG_E(MAC, "%d.%d can not allocate CCE for UE %d\n", frameP, slotP, UE_id);
      continue;
    }

    /* this pre-fills the PDCCH and PDSCH PDUs of FAPI. At the same time, it
     * calculates the TBS. We could do the latter in a separate step first, but
     * this involves multiple variables (like code rate, etc) that we'd need to
     * carry along the TBS so we avoid this for the moment since at this point
     * we can be sure that (a) resource blocks have been allocated and (b) this
     * UE's CCEs can be allocated, hence this will run until the end */
    int TBS_bytes = configure_fapi_dl_pdu(gNB_mac,
                                          CC_id,
                                          UE_id,
                                          bwp_id,
                                          CCEIndex,
                                          sched_ctrl->dlsch_mcs,
                                          sched_ctrl->pre_nb_available_rbs[CC_id],
                                          rbStart);

    unsigned char sdu_lcids[NB_RB_MAX] = {0};
    uint16_t sdu_lengths[NB_RB_MAX] = {0};
    uint8_t mac_sdus[MAX_NR_DLSCH_PAYLOAD_BYTES];
    const uint16_t rnti = UE_list->rnti[UE_id];
    int ta_len = gNB_mac->ta_len; /* TODO needs to be UE specific? */
    int header_length_total = 0;
    int sdu_length_total = 0;
    int num_sdus = 0;
    for (int i = 0; i < sched_ctrl->dl_lc_num; i++) {
      int lcid = sched_ctrl->dl_lc_ids[i];
      // if nothing is allocated for this RB, continue
      if (sched_ctrl->dl_lc_bytes[lcid] == 0)
        continue;

      int req_data =
          min(TBS_bytes - ta_len - header_length_total - sdu_length_total - 3,
              sched_ctrl->dl_lc_bytes[lcid]);
      LOG_D(MAC,
            "[gNB %d][USER-PLANE DEFAULT DRB] Frame %d : DTCH->DLSCH, Requesting "
            "%d bytes from RLC (lcid %d total hdr len %d), TBS_bytes: %d \n \n",
            module_idP,
            frameP,
            req_data,
            lcid,
            header_length_total,
            TBS_bytes);

      sdu_lengths[num_sdus] = mac_rlc_data_req(module_idP,
                                               rnti,
                                               module_idP,
                                               frameP,
                                               ENB_FLAG_YES,
                                               MBMS_FLAG_NO,
                                               lcid,
                                               req_data,
                                               (char *)&mac_sdus[sdu_length_total],
                                               0,
                                               0);

      LOG_D(MAC,
            "[gNB %d][USER-PLANE DEFAULT DRB] Got %d bytes for DTCH %d \n",
            module_idP,
            sdu_lengths[num_sdus],
            lcid);

      sdu_lcids[num_sdus] = lcid;
      sdu_length_total += sdu_lengths[num_sdus];
      header_length_total += 1 + 1 + (sdu_lengths[num_sdus] >= 128);

      num_sdus++;

      //ue_sched_ctl->uplane_inactivity_timer = 0;

      if (TBS_bytes - ta_len - header_length_total - sdu_length_total - 3 <= 0)
        break;
    }

    /* check if there is at least one SDU or TA command. Actually this should
     * never be true (a UE is only allocated resource blocks if it has data to
     * transmit), but leave it as a final check */
    if (ta_len + sdu_length_total + header_length_total <= 0)
      continue;

    // Check if we have to consider padding
    int post_padding = TBS_bytes >= 2 + header_length_total + sdu_length_total + ta_len;

    int offset = nr_generate_dlsch_pdu(
        module_idP,
        (unsigned char *)mac_sdus,
        (unsigned char *)UE_list->DLSCH_pdu[0][UE_id].payload[0],
        num_sdus, // num_sdus
        sdu_lengths,
        sdu_lcids,
        255, // no drx
        NULL, // contention res id
        post_padding);

    // Padding: fill remainder of DLSCH with 0
    if (post_padding > 0)
      for (int j = 0; j < (TBS_bytes - offset); j++)
        UE_list->DLSCH_pdu[0][UE_id].payload[0][offset + j] = 0;

    configure_fapi_dl_Tx(gNB_mac,
                         CC_id,
                         frameP,
                         slotP,
                         TBS_bytes,
                         UE_list->DLSCH_pdu[0][UE_id].payload[0]);

#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
    LOG_I(MAC, "%d.%d first 10 payload bytes UE %d TBSsize %d:", frameP, slotP, UE_id, TBS_bytes);
    for(int i = 0; i < 10; i++)
      printf(" %02x.", UE_list->DLSCH_pdu[0][UE_id].payload[0][i]);
    prinf("..\n");
#endif
  }
}
