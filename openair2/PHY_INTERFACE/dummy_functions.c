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
#include "LAYER2/MAC/mac_proto.h"

void initiate_ra_proc(module_id_t module_idP, int CC_id, frame_t frameP,
			sub_frame_t subframeP, uint16_t preamble_index,
			int16_t timing_offset, uint16_t rnti
#if (LTE_RRC_VERSION >= MAKE_VERSION(14, 0, 0))
			, uint8_t rach_resource_type
#endif
			) {;}

void SR_indication(module_id_t module_idP, int CC_id, frame_t frameP,
		     sub_frame_t subframe, rnti_t rnti, uint8_t ul_cqi) {;}


void cqi_indication(module_id_t mod_idP, int CC_idP, frame_t frameP,
		      sub_frame_t subframeP, rnti_t rntiP,
		      nfapi_cqi_indication_rel9_t * rel9, uint8_t * pdu,
		      nfapi_ul_cqi_information_t * ul_cqi_information) {;}

void harq_indication(module_id_t mod_idP, int CC_idP, frame_t frameP,
		       sub_frame_t subframeP,
		       nfapi_harq_indication_pdu_t * harq_pdu) {;}

void rx_sdu(const module_id_t enb_mod_idP,
	      const int CC_idP,
	      const frame_t frameP,
	      const sub_frame_t subframeP,
	      const rnti_t rntiP,
	      uint8_t * sduP,
	      const uint16_t sdu_lenP,
	      const uint16_t timing_advance, const uint8_t ul_cqi) {;}

void clear_nfapi_information(eNB_MAC_INST * eNB, int CC_idP,
			       frame_t frameP, sub_frame_t subframeP) {;}

void eNB_dlsch_ulsch_scheduler(module_id_t module_idP, frame_t frameP, sub_frame_t subframeP) {;}

int is_UL_sf(COMMON_channels_t * ccP, sub_frame_t subframeP) {return(0);}
