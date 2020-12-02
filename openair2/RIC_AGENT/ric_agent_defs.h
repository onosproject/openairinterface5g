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

#ifndef _RIC_AGENT_DEFS_H
#define _RIC_AGENT_DEFS_H

#include <stdint.h>

#include "list.h"
#include "ric_agent.h"

#define E2AP_PORT 36422
#define E2AP_SCTP_PPID 70 /*< E2AP SCTP Payload Protocol Identifier (PPID) */

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

typedef struct ric_control {
} ric_control_t;

typedef enum {
  RIC_UNINITIALIZED = 0,
  RIC_CONNECTING = 1,
  RIC_CONNECTED = 2,
  RIC_ESTABLISHED = 3,
  RIC_FAILURE = 4,
  RIC_DISCONNECTED = 5,
  RIC_DISABLED = 6,
} ric_nb_state_t;

typedef struct {
  int enabled;
  ric_nb_state_t state;

  char *remote_ipv4_addr;
  uint16_t remote_port;
  int32_t assoc_id;
  char *functions_enabled_str;
  ric_ran_function_id_t *functions_enabled;
  size_t functions_enabled_len;

  ranid_t ranid;
  uint16_t mcc;
  uint16_t mnc;
  uint8_t mnc_digit_len;

  uint16_t ric_mcc;
  uint16_t ric_mnc;
  uint16_t ric_mnc_digit_len;
  uint32_t ric_id;

  long e2sm_kpm_timer_id;
  long ric_connect_timer_id;

  LIST_HEAD(ric_subscription_list,ric_subscription) subscription_list;
  
} ric_agent_info_t;

/**
 * These are generic service mechanisms.
 */
typedef enum {
  RIC_REPORT = 1,
  RIC_INSERT = 2,
  RIC_CONTROL = 3,
  RIC_POLICY = 4,
} ric_service_t;

/**
 * An abstraction that describes an E2 service model.
 */
typedef struct {
  char *name;
  char *oid;
  int (*handle_subscription_add)(ric_agent_info_t *ric,ric_subscription_t *sub);
  int (*handle_subscription_del)(ric_agent_info_t *ric,ric_subscription_t *sub,
				 int force,long *cause,long *cause_detail);
  int (*handle_control)(ric_agent_info_t *ric,ric_control_t *control);
  int (*handle_timer_expiry)(ric_agent_info_t *ric, long timer_id, ric_ran_function_id_t function_id);
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

int ric_agent_reset(ric_agent_info_t *ric);
int ric_agent_register_ran_function(ric_ran_function_t *func);
ric_ran_function_t *ric_agent_lookup_ran_function(
  ric_ran_function_id_t function_id);
ric_ran_function_t *ric_agent_lookup_ran_function_by_name(char *name);
ric_subscription_t *ric_agent_lookup_subscription(
  ric_agent_info_t *ric,long request_id,long instance_id,
  ric_ran_function_id_t function_id);
void ric_free_action(ric_action_t *action);
void ric_free_subscription(ric_subscription_t *sub);
void ric_agent_send_sctp_data(ric_agent_info_t *ric,uint16_t stream,
			      uint8_t *buf,uint32_t len);

#endif /* _RIC_AGENT_DEFS_H */
