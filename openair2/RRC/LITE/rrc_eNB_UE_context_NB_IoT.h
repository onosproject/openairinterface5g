/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*! \file rrc_eNB_UE_context.h
 * \brief rrc procedures for UE context
 * \author Lionel GAUTHIER
 * \date 2015
 * \version 1.0
 * \company Eurecom
 * \email: lionel.gauthier@eurecom.fr
 */
#ifndef __RRC_ENB_UE_CONTEXT_NB_IoT_H__
#define __RRC_ENB_UE_CONTEXT_NB_IoT_H__


#include "collection/tree.h"
#include "COMMON/platform_types.h"
//#include "defs.h"
#include "defs_NB_IoT.h"


void
uid_linear_allocator_init_NB_IoT(
  uid_allocator_NB_IoT_t* const uid_pP
);


uid_t
uid_linear_allocator_new_NB_IoT(
  eNB_RRC_INST_NB_IoT* const rrc_instance_pP
);

void
uid_linear_allocator_free_NB_IoT(
  eNB_RRC_INST_NB_IoT* rrc_instance_pP,
  uid_t uidP
);


int rrc_eNB_compare_ue_rnti_id_NB_IoT(
  struct rrc_eNB_ue_context_NB_IoT_s* c1_pP, struct rrc_eNB_ue_context_NB_IoT_s* c2_pP);

RB_PROTOTYPE(rrc_ue_tree_NB_IoT_s, rrc_eNB_ue_context_NB_IoT_s, entries, rrc_eNB_compare_ue_rnti_id_NB_IoT);

struct rrc_eNB_ue_context_NB_IoT_s*
rrc_eNB_allocate_new_UE_context_NB_IoT(
  eNB_RRC_INST_NB_IoT* rrc_instance_pP
);

struct rrc_eNB_ue_context_NB_IoT_s*
rrc_eNB_get_ue_context_NB_IoT(
  eNB_RRC_INST_NB_IoT* rrc_instance_pP,
  rnti_t rntiP);

void rrc_eNB_remove_ue_context_NB_IoT(
  const protocol_ctxt_t* const ctxt_pP,
  eNB_RRC_INST_NB_IoT*                rrc_instance_pP,
  struct rrc_eNB_ue_context_NB_IoT_s* ue_context_pP);

#endif
