/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: LicenseRef-ONF-Member-1.0
 */

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

#ifndef _E2AP_GENERATE_MESSAGES_H
#define _E2AP_GENERATE_MESSAGES_H

#include <stdint.h>

#include "ric_agent_defs.h"
#include "e2ap_common.h"

int e2ap_generate_e2_setup_request(ric_agent_info_t *ric,
				   uint8_t **buffer,uint32_t *len);
int e2ap_generate_ric_subscription_response(ric_agent_info_t *ric,
					    ric_subscription_t *rs,
					    uint8_t **buffer,uint32_t *len);
int e2ap_generate_ric_subscription_failure(ric_agent_info_t *ric,
					   ric_subscription_t *rs,
					   uint8_t **buffer,uint32_t *len);
int e2ap_generate_ric_subscription_delete_response(
  ric_agent_info_t *ric,long request_id,long instance_id,
  ric_ran_function_id_t function_id,uint8_t **buffer,uint32_t *len);
int e2ap_generate_ric_subscription_delete_failure(
  ric_agent_info_t *ric,long request_id,long instance_id,
  ric_ran_function_id_t function_id,long cause,long cause_detail,
  uint8_t **buffer,uint32_t *len);
int e2ap_generate_ric_service_update(ric_agent_info_t *ric,
				     uint8_t **buffer,uint32_t *len);
int e2ap_generate_reset_response(ric_agent_info_t *ric,
				 uint8_t **buffer,uint32_t *len);

#endif /* _E2AP_GENERATE_MESSAGES_H */
