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

/*! \file LAYER2/MAC/proto_NB_IoT.h
 * \brief MAC functions prototypes for eNB and UE
 * \author Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email navid.nikaein@eurecom.fr
 * \version 1.0
 */

#ifndef __LAYER2_MAC_PROTO_NB_IoT_H__
#define __LAYER2_MAC_PROTO_NB_IoT_H__

#include "openair1/PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "COMMON/platform_types.h"
/** \addtogroup _mac
 *  @{
 */

/*for NB-IoT*/

void init_tool_sib1(eNB_MAC_INST_NB_IoT *mac_inst);

void init_dlsf_info(eNB_MAC_INST_NB_IoT *mac_inst, DLSF_INFO_t *DLSF_info);

void init_mac_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst);

int is_dlsf(eNB_MAC_INST_NB_IoT *mac_inst, int abs_subframe);

void init_dl_list(eNB_MAC_INST_NB_IoT *mac_inst);

void setting_nprach(void);

void init_rrc_NB_IoT(void);

void add_UL_Resource_node(available_resource_UL_t **head, uint32_t *end_subframe, uint32_t ce_level);

void add_UL_Resource(void);

void Initialize_Resource(void);

void extend_available_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, int max_subframe);

void rrc_mac_config_req_NB_IoT(rrc_config_NB_IoT_t *mac_config,
							   uint8_t mib_flag,
							   uint8_t sib_flag,
							   uint8_t ded_flag,
							   uint8_t ue_list_ded_num);
// schedule functinons
void schedule_sibs_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t sibs_order, int start_subframe1);

void fill_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, available_resource_DL_t *node, int start_subframe, int end_subframe, schedule_result_t *new_node);

available_resource_DL_t *check_sibs_resource(eNB_MAC_INST_NB_IoT *mac_inst, int check_start_subframe, int check_end_subframe, int num_subframe, int *residual_subframe, int *out_last_subframe, int *out_first_subframe);

uint32_t calculate_DLSF(eNB_MAC_INST_NB_IoT *mac_inst, int abs_start_subframe, int abs_end_subframe);
#endif
