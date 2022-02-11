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

#ifndef _RIC_AGENT_H
#define _RIC_AGENT_H

#include <stdint.h>
#include <stdbool.h> 
#include "list.h"
#include "common/utils/LOG/log.h"

#define E2AP_SCTP_PPID 70 /*< E2AP SCTP Payload Protocol Identifier (PPID) */
#define UE_ATTACH_EVENT_TRIGGER 100
#define UE_DETACH_EVENT_TRIGGER 200

typedef uint16_t ranid_t;

#define RIC_AGENT_ERROR(msg, args...) LOG_E(RIC_AGENT, msg, ##args)
#define RIC_AGENT_INFO(msg, args...)  LOG_I(RIC_AGENT, msg, ##args)
#define RIC_AGENT_WARN(msg, args...)  LOG_W(RIC_AGENT, msg, ##args)
#define RIC_AGENT_DEBUG(msg, args...) LOG_D(RIC_AGENT, msg, ##args)
#define DISABLE_SCTP_MULTIHOMING 1

/**
 * These are local function IDs.  Each service model might expose many
 * functions.  E2SM functions do not currently have global IDs, unless
 * you concat the E2SM OID and the function name.  There is no
 * requirement that function IDs be the same for different E2Setup/Reset
 * sessions, so we allow e2sm modules to register functions.
 */
typedef long ric_ran_function_id_t;

typedef struct ric_action {
    long id;
    long type;
    long error_cause;
    long error_cause_detail;
    size_t def_size;
    uint8_t *def_buf;
    long subsequent_action;
    long time_to_wait;

    int enabled;
    void *state;

    LIST_ENTRY(ric_action) actions;
} ric_action_t;

typedef struct ric_event_trigger {
    uint8_t *buf;
    size_t size;
} ric_event_trigger_t;

typedef struct ric_subscription {
    long request_id;
    long instance_id;
    ric_ran_function_id_t function_id;
    ric_event_trigger_t event_trigger;

    int enabled;
    void *state;
    LIST_HEAD(ric_subscription_action_list,ric_action) action_list;
    LIST_ENTRY(ric_subscription) subscriptions;
} ric_subscription_t;

typedef struct ric_event_trigger ric_control_header_t;
typedef struct ric_event_trigger ric_control_msg_t;

typedef struct ric_control {
    long request_id;
    long instance_id;
    ric_ran_function_id_t function_id;
    ric_control_header_t control_hdr;
    long    failure_cause;
    uint16_t control_req_type;
    ric_control_msg_t control_msg;
} ric_control_t;

typedef struct {
    int32_t assoc_id;
    int32_t data_conn_assoc_id;

    ranid_t ranid;

    uint16_t ric_mcc;
    uint16_t ric_mnc;
    uint16_t ric_mnc_digit_len;
    uint32_t ric_id;

    long e2sm_kpm_timer_id;
    long gran_prd_timer_id;
    long ric_connect_timer_id;

	ric_ran_function_id_t e2sm_rsm_function_id;
    long e2sm_rsm_request_id;
    long e2sm_rsm_instance_id;

    LIST_HEAD(ric_subscription_list, ric_subscription) subscription_list;
} ric_agent_info_t;

typedef struct {
    int32_t du_assoc_id;
    int32_t du_data_conn_assoc_id;

    ranid_t ranid;

    uint16_t ric_mcc;
    uint16_t ric_mnc;
    uint16_t ric_mnc_digit_len;
    uint32_t ric_id;

    long du_ric_connect_timer_id;
} du_ric_agent_info_t;

typedef struct ric_ran_function_requestor_info {
    ric_ran_function_id_t function_id;
    long request_id;
    long instance_id;
    long action_id;
} ric_ran_function_requestor_info_t;

/**
 * An abstraction that describes an E2 service model.
 */

typedef struct {
    long     meas_type_id;
    char     *meas_type_name;
    uint16_t meas_data;
    bool     subscription_status;
} kmp_meas_info_t;

typedef struct {
    char *name;
    char *oid;
    int (*handle_subscription_add)(ric_agent_info_t *ric, ric_subscription_t *sub);
    int (*handle_subscription_del)(ric_agent_info_t *ric, ric_subscription_t *sub,
            int force, long *cause, long *cause_detail);
    int (*handle_control)(ric_agent_info_t *ric,ric_control_t *control);
    int (*handle_ricInd_timer_expiry)(
            ric_agent_info_t *ric,
            long timer_id,
            ric_ran_function_id_t function_id,
            long request_id,
            long instance_id,
            long action_id,
            uint8_t **outbuf,
            uint32_t *outlen);
    int (*handle_gp_timer_expiry)(
            ric_agent_info_t *ric,
            long timer_id,
            ric_ran_function_id_t function_id,
            long request_id,
            long instance_id,
            long action_id,
            uint8_t **outbuf,
            uint32_t *outlen);
} ric_service_model_t;

typedef struct ric_ran_function {
    ric_ran_function_id_t function_id;
    ric_service_model_t *model;
    long revision;
    char *name;
    char *description;

    uint8_t *enc_definition;
    size_t enc_definition_len;

    int enabled;
    void *definition;
} ric_ran_function_t;

typedef enum {
    E2NODE_TYPE_NONE,
    E2NODE_TYPE_ENB_CU,
    E2NODE_TYPE_NG_ENB_CU,
    E2NODE_TYPE_GNB_CU,
    E2NODE_TYPE_ENB_DU
} e2node_type_t;

typedef struct e2_conf {
    int enabled;
    e2node_type_t e2node_type;
    char *node_name;
    uint32_t cell_identity;
    uint16_t mcc;
    uint16_t mnc;
    uint8_t mnc_digit_length;

    char *remote_ipv4_addr;
    uint16_t remote_port;

    char data_conn_remote_ipv4[16];
	uint16_t data_conn_remote_port;
} e2_conf_t;

extern ric_agent_info_t **ric_agent_info;
extern du_ric_agent_info_t **du_ric_agent_info;
extern e2_conf_t **e2_conf;

void *ric_agent_task(void *args);
void *du_ric_agent_task(void *args);
void RCconfig_ric_agent(void);
int ric_agent_reset(ric_agent_info_t *ric);
int ric_agent_register_ran_function(ric_ran_function_t *func);
ric_ran_function_t *ric_agent_lookup_ran_function(
        ric_ran_function_id_t function_id);
ric_ran_function_t *ric_agent_lookup_ran_function_by_name(char *name);
ric_subscription_t *ric_agent_lookup_subscription(ric_agent_info_t *ric,
        long request_id, long instance_id,
        ric_ran_function_id_t function_id);
void ric_free_action(ric_action_t *action);
void ric_free_subscription(ric_subscription_t *sub);

#endif /* _RIC_AGENT_H */
