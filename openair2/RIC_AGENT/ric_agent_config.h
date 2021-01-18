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

#ifndef _RIC_AGENT_CONFIG_H
#define _RIC_AGENT_CONFIG_H

#include "common/ran_context.h"

#define PLMN_LIST_MAX_SIZE  6

typedef enum {
    E2NODE_TYPE_NONE,
    E2NODE_TYPE_ENB,
    E2NODE_TYPE_NG_ENB,
    E2NODE_TYPE_GNB,
    E2NODE_TYPE_ENB_CU,
    E2NODE_TYPE_NG_ENB_CU,
    E2NODE_TYPE_GNB_CU,
    E2NODE_TYPE_ENB_DU,
    E2NODE_TYPE_GNB_DU,
    E2NODE_TYPE_ENB_MBMS_STA
} e2node_type_t;

typedef struct e2_conf {
    e2node_type_t e2node_type;
    char *node_name;
    uint32_t cell_identity;
    uint16_t mcc;
    uint16_t mnc;
    uint8_t mnc_digit_length;

    char *remote_ipv4_addr;
    uint16_t remote_port;
} e2_conf_t;

extern e2_conf_t **e2_conf;

void RCconfig_ric_agent(void);
int ric_agent_is_enabled(void);

#endif /* _RIC_AGENT_CONFIG_H */
