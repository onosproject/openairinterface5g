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

#include <pthread.h>

#include "common/ran_context.h"
#include "ric_agent_common.h"

extern RAN_CONTEXT_t RC;

static volatile int ric_config_loaded = 0;
static pthread_mutex_t ric_config_mutex = PTHREAD_MUTEX_INITIALIZER;

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
    "yes/no",0,strptr:NULL,defstrval:"no",TYPE_STRING,0 }, \
  { RIC_CONFIG_STRING_REMOTE_IPV4_ADDR, \
    NULL,0,strptr:NULL,defstrval:"127.0.0.1",TYPE_STRING,0 }, \
  { RIC_CONFIG_STRING_REMOTE_PORT, \
    NULL,0,uptr:NULL,defintval:E2AP_PORT,TYPE_UINT,0 },	\
  { RIC_CONFIG_STRING_FUNCTIONS_ENABLED, \
    NULL,0,strptr:NULL,defstrval:"ORAN-E2SM-KPM",TYPE_STRING,0 } \
}

static void RCconfig_ric_agent_init(void)
{
  uint32_t i;

  /* Allocate RIC state in the ran_context_t RC. */
  if (!RC.ric) {
    RC.ric = (ric_agent_info_t **)calloc(RC.nb_inst,sizeof(*RC.ric));
    for (i = 0; i < RC.nb_inst; ++i) {
      RC.ric[i] = (ric_agent_info_t *)calloc(1,sizeof(**RC.ric));
      RC.ric[i]->assoc_id = -1;
    }
  }
}

static void RCconfig_ric_agent_ric(void)
{
    uint16_t i;
    int j;
    char buf[16];
    paramdef_t ric_params[] = RICPARAMS_DESC;

    for (i = 0; i < RC.nb_inst; ++i) {
        /* Get RIC configuration. */
        snprintf(buf, sizeof(buf), "%s.[%u].RIC", ENB_CONFIG_STRING_ENB_LIST, i);
        config_get(ric_params, sizeof(ric_params)/sizeof(paramdef_t), buf);
        if (ric_params[RIC_CONFIG_IDX_ENABLED].strptr != NULL
                && strcmp(*ric_params[RIC_CONFIG_IDX_ENABLED].strptr, "yes") == 0) {
            RIC_AGENT_INFO("enabled for NB %u\n",i);
            RC.ric[i]->enabled = 1;
            RC.ric[i]->remote_ipv4_addr = strdup(*ric_params[RIC_CONFIG_IDX_REMOTE_IPV4_ADDR].strptr);
            RC.ric[i]->remote_port = *ric_params[RIC_CONFIG_IDX_REMOTE_PORT].uptr;
            RC.ric[i]->functions_enabled_str = strdup(*ric_params[RIC_CONFIG_IDX_FUNCTIONS_ENABLED].strptr);
            for (j = 0; j < strlen(RC.ric[i]->functions_enabled_str); ++j) {
                /* We want a space-delimited list, but be forgiving. */
                if (RC.ric[i]->functions_enabled_str[j] == ','
                        || RC.ric[i]->functions_enabled_str[j] == ';'
                        || RC.ric[i]->functions_enabled_str[j] == '\t') {
                    RC.ric[i]->functions_enabled_str[j] = ' ';
                }
            }
        }
        else {
            RIC_AGENT_INFO("not enabled for NB %u\n",i);
            RC.ric[i]->enabled = 0;
        }
        RC.ric[i]->state = RIC_UNINITIALIZED;
    }
}

/**
 * Should only be called from eNB/gNB init, after RRC config (because we
 * assume RC.nb_inst has been initialized), prior to task start.  (It
 * could also be called as a side-effect of running ric_agent_task, but
 * that is not preferred since then the eNB/gNB would throw config
 * errors after having started many threads and possibly initializing
 * hardware.)
 */
void RCconfig_ric_agent(void)
{
  if (pthread_mutex_lock(&ric_config_mutex))
    goto mutex_error;

  if (ric_config_loaded) {
    if (pthread_mutex_unlock(&ric_config_mutex))
      goto mutex_error;
    return;
  }

  RCconfig_ric_agent_init();
  RCconfig_ric_agent_ric();

  ric_config_loaded = 1;

  if (pthread_mutex_unlock(&ric_config_mutex))
    goto mutex_error;

  return;

 mutex_error:
  RIC_AGENT_ERROR("mutex error (ric_config_mutex)\n");
  exit(1);
}

int ric_agent_is_enabled_for_nb(ranid_t ranid)
{
  if (ranid >= RC.nb_inst) {
    RIC_AGENT_ERROR("invalid NB %u (%u total)\n",ranid,RC.nb_inst);
    return 0;
  }

  return RC.ric[ranid]->enabled;
}
