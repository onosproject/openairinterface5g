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

#ifndef _RIC_AGENT_RRC_H
#define _RIC_AGENT_RRC_H

#include "ric_agent_defs.h"
//#include "rrc_defs.h"
#include "common/ngran_types.h"

int ric_rrc_get_node_type(ranid_t ranid,ngran_node_t *node_type);
int ric_rcc_get_nb_id(ranid_t ranid,uint32_t *nb_id);
int ric_rrc_get_plmn_len(ranid_t ranid,uint8_t *len);
int ric_rrc_get_mcc_mnc(ranid_t ranid,uint8_t index,
			uint16_t *mcc,uint16_t *mnc,uint8_t *mnc_digit_len);
#endif /* _RIC_AGENT_RRC_H */
