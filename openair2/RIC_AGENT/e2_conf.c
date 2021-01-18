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
#include "ric_agent_common.h"
#include "e2_conf.h"

e2_conf_t **e2_conf;

void e2_conf_init(RAN_CONTEXT_t *RC) {
    uint32_t i;

    if (e2_conf) {
        // LOG
        return;
    } 

    e2_conf = (e2_conf_t **)calloc(RC->nb_inst, sizeof(e2_conf_t));
    for (i = 0; i < RC->nb_inst; ++i) {
        e2_conf[i] = (e2_conf_t *)calloc(1,sizeof(e2_conf_t));
        e2_conf[i]->node_name = strdup(RC->rrc[i]->node_name);
        e2_conf[i]->cell_identity = RC->rrc[i]->configuration.cell_identity;
        switch (RC->rrc[i]->node_type) {
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
    }
}
