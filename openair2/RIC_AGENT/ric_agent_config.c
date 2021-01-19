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

#include "common/ran_context.h"
#include "ric_agent_config.h"
#include "ric_agent_common.h"

extern RAN_CONTEXT_t RC;
e2_conf_t **e2_conf;

#define RIC_CONFIG_STRING_ENABLED "enabled"
#define RIC_CONFIG_STRING_REMOTE_IPV4_ADDR "remote_ipv4_addr"
#define RIC_CONFIG_STRING_REMOTE_PORT "remote_port"
#define RIC_CONFIG_STRING_FUNCTIONS_ENABLED "functions_enabled"

#define RIC_CONFIG_IDX_ENABLED          0
#define RIC_CONFIG_IDX_REMOTE_IPV4_ADDR 1
#define RIC_CONFIG_IDX_REMOTE_PORT      2
#define RIC_CONFIG_IDX_FUNCTIONS_ENABLED 3

#define RICPARAMS_DESC { \
    { RIC_CONFIG_STRING_ENABLED, \
        "yes/no", 0, strptr:NULL, defstrval:"no", TYPE_STRING, 0 }, \
    { RIC_CONFIG_STRING_REMOTE_IPV4_ADDR, \
        NULL, 0, strptr:NULL, defstrval: "127.0.0.1", TYPE_STRING, 0 }, \
    { RIC_CONFIG_STRING_REMOTE_PORT, \
        NULL, 0, uptr:NULL, defintval:E2AP_PORT, TYPE_UINT, 0 },	\
    { RIC_CONFIG_STRING_FUNCTIONS_ENABLED, \
        NULL, 0, strptr:NULL, defstrval:"ORAN-E2SM-KPM", TYPE_STRING, 0 } \
}

void RCconfig_ric_agent(void) {
    uint16_t i;
    int j;
    char buf[16];
    paramdef_t ric_params[] = RICPARAMS_DESC;

    RC.ric = (ric_agent_info_t **)calloc(RC.nb_inst,sizeof(*RC.ric));
    e2_conf = (e2_conf_t **)calloc(RC.nb_inst, sizeof(e2_conf_t));

    for (i = 0; i < RC.nb_inst; ++i) {
        /* Get RIC configuration. */
        snprintf(buf, sizeof(buf), "%s.[%u].RIC", ENB_CONFIG_STRING_ENB_LIST, i);
        config_get(ric_params, sizeof(ric_params)/sizeof(paramdef_t), buf);
        if (ric_params[RIC_CONFIG_IDX_ENABLED].strptr != NULL
                && strcmp(*ric_params[RIC_CONFIG_IDX_ENABLED].strptr, "yes") == 0) {
            RIC_AGENT_INFO("enabled for NB %u\n",i);

            RC.ric[i] = (ric_agent_info_t *)calloc(1,sizeof(**RC.ric));
            RC.ric[i]->assoc_id = -1;
            RC.ric[i]->enabled = 1;
            RC.ric[i]->functions_enabled_str = strdup(*ric_params[RIC_CONFIG_IDX_FUNCTIONS_ENABLED].strptr);
            for (j = 0; j < strlen(RC.ric[i]->functions_enabled_str); ++j) {
                /* We want a space-delimited list, but be forgiving. */
                if (RC.ric[i]->functions_enabled_str[j] == ','
                        || RC.ric[i]->functions_enabled_str[j] == ';'
                        || RC.ric[i]->functions_enabled_str[j] == '\t') {
                    RC.ric[i]->functions_enabled_str[j] = ' ';
                }
            }

            e2_conf[i] = (e2_conf_t *)calloc(1,sizeof(e2_conf_t));
            e2_conf[i]->node_name = strdup(RC.rrc[i]->node_name);
            e2_conf[i]->cell_identity = RC.rrc[i]->configuration.cell_identity;
            e2_conf[i]->mcc = RC.rrc[i]->configuration.mcc[0];
            e2_conf[i]->mnc = RC.rrc[i]->configuration.mnc[0];
            e2_conf[i]->mnc_digit_length = RC.rrc[i]->configuration.mnc_digit_length[0];
            switch (RC.rrc[i]->node_type) {
                case ngran_eNB:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB;
                    break;
                case ngran_ng_eNB:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_NG_ENB;
                    break;
                case ngran_gNB:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_GNB;
                    break;
                case ngran_eNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB_CU;
                    break;
                case ngran_ng_eNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_NG_ENB_CU;
                    break;
                case ngran_gNB_CU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_GNB_CU;
                    break;
                case ngran_eNB_DU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB_DU;
                    break;
                case ngran_gNB_DU:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_GNB_DU;
                    break;
                case ngran_eNB_MBMS_STA:
                    e2_conf[i]->e2node_type = E2NODE_TYPE_ENB_MBMS_STA;
                    break;
                default:
                    break;
            }
            e2_conf[i]->remote_ipv4_addr = strdup(*ric_params[RIC_CONFIG_IDX_REMOTE_IPV4_ADDR].strptr);
            e2_conf[i]->remote_port = *ric_params[RIC_CONFIG_IDX_REMOTE_PORT].uptr;
        }
        else {
            RIC_AGENT_INFO("not enabled for NB %u\n",i);
            RC.ric[i]->enabled = 0;
        }
        RC.ric[i]->state = RIC_UNINITIALIZED;
    }
}
