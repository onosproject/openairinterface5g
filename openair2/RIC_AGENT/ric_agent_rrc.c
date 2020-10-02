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

#include "assertions.h"
#include "common/ran_context.h"

extern RAN_CONTEXT_t RC;

static inline int ric_rrc_is_present(ranid_t ranid)
{
  DevAssert(ranid < RC.nb_inst);
  return RC.rrc && RC.rrc[ranid];
}

int ric_rrc_get_node_type(ranid_t ranid,ngran_node_t *node_type)
{
  if (!ric_rrc_is_present(ranid))
    return 1;
  if (node_type)
    *node_type = RC.rrc[ranid]->node_type;
  return 0;
}

int ric_rcc_get_nb_id(ranid_t ranid,uint32_t *nb_id)
{
  if (!ric_rrc_is_present(ranid))
    return 1;
  if (nb_id)
    *nb_id = RC.rrc[ranid]->configuration.cell_identity;
  return 0;
}

int ric_rrc_get_plmn_len(ranid_t ranid,uint8_t *len)
{
  if (!ric_rrc_is_present(ranid))
    return 1;
  if (len)
    *len = RC.rrc[ranid]->configuration.num_plmn;
  return 0;
}

int ric_rrc_get_mcc_mnc(ranid_t ranid,uint8_t index,
			uint16_t *mcc,uint16_t *mnc,uint8_t *mnc_digit_len)
{
  uint8_t plen;

  if (!ric_rrc_is_present(ranid))
    return 1;
  ric_rrc_get_plmn_len(ranid,&plen);
  if (index >= plen)
    return 1;

  if (mcc)
    *mcc = RC.rrc[ranid]->configuration.mcc[index];
  if (mnc)
    *mnc = RC.rrc[ranid]->configuration.mnc[index];
  if (mnc_digit_len)
    *mcc = RC.rrc[ranid]->configuration.mnc_digit_length[index];

  return 0;
}

