/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef _E2AP_HANDLER_H
#define _E2AP_HANDLER_H

#include <stdint.h>

extern int e2ap_handle_message(
        ric_agent_info_t *ric,
        int32_t stream,
        const uint8_t * const buf,
        const uint32_t buflen,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *assoc_id);

extern int e2ap_handle_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        void* arg,
        uint8_t **outbuf,
        uint32_t *outlen);

extern int e2ap_handle_gp_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        void* arg,
        uint8_t **outbuf,
        uint32_t *outlen);

extern int du_e2ap_handle_message(
        du_ric_agent_info_t *ric,
        int32_t stream,
        const uint8_t * const buf,
        const uint32_t buflen,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *du_assoc_id);

extern void du_e2ap_prepare_ric_control_response(
        du_ric_agent_info_t *ric,
        apiMsg   *sliceResp,
        uint8_t **outbuf,
        uint32_t *outlen,
        uint32_t *du_assoc_id);

#endif /* _E2AP_ENB_HANDLER_H */
