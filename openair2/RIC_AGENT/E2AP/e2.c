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

#include <stdlib.h>
#include <string.h>
#include "e2.h"
#include "ric_agent_defs.h"

e2_conf_t **e2_conf;

void e2_init(int index, e2_conf_t conf) {

    if (!e2_conf) {
        e2_conf = (e2_conf_t **)calloc(256, sizeof(e2_conf_t));
    }

    e2_conf[index] = (e2_conf_t *)calloc(1,sizeof(e2_conf_t));
    memcpy(e2_conf[index], &conf, sizeof(e2_conf_t));

    if (!ric_agent_info) {
        ric_agent_info = (ric_agent_info_t **)calloc(250, sizeof(ric_agent_info_t));
    }
    ric_agent_info[index] = (ric_agent_info_t *)calloc(1, sizeof(ric_agent_info_t));
    ric_agent_info[index]->assoc_id = -1;
}
