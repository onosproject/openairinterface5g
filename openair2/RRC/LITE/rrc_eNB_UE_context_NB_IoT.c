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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "UTIL/LOG/log.h"
#include "rrc_eNB_UE_context_NB_IoT.h"
#include "msc.h"

//------------------------------------------------------------------------------
void uid_linear_allocator_init_NB_IoT(

  uid_allocator_NB_IoT_t* const uid_pP
)
//------------------------------------------------------------------------------
{
  memset(uid_pP, 0, sizeof(uid_allocator_NB_IoT_t));
}


//------------------------------------------------------------------------------
uid_t uid_linear_allocator_new_NB_IoT(

  eNB_RRC_INST_NB_IoT* const rrc_instance_pP
)
//------------------------------------------------------------------------------
{
  unsigned int i;
  unsigned int bit_index = 1;
  uid_t        uid = 0;
  uid_allocator_NB_IoT_t* uia_p = &rrc_instance_pP->uid_allocator;

  for (i=0; i < UID_LINEAR_ALLOCATOR_BITMAP_SIZE_NB_IoT; i++) {
    if (uia_p->bitmap[i] != UINT_MAX) {
      bit_index = 1;
      uid       = 0;

      while ((uia_p->bitmap[i] & bit_index) == bit_index) {
        bit_index = bit_index << 1;
        uid       += 1;
      }

      uia_p->bitmap[i] |= bit_index;
      return uid + (i*sizeof(unsigned int)*8);
    }
  }

  return UINT_MAX;
}

//------------------------------------------------------------------------------
void uid_linear_allocator_free_NB_IoT(

  eNB_RRC_INST_NB_IoT*   rrc_instance_pP,
  uid_t                  uidP
)
//------------------------------------------------------------------------------
{
  unsigned int i      =  uidP/sizeof(unsigned int)/8;
  unsigned int bit    =  uidP % (sizeof(unsigned int) * 8);
  unsigned int value  =  ~(0x00000001 << bit);

  if (i < UID_LINEAR_ALLOCATOR_BITMAP_SIZE_NB_IoT) {
    rrc_instance_pP->uid_allocator.bitmap[i] &=  value;
  }
}



int rrc_eNB_compare_ue_rnti_id_NB_IoT(

  struct rrc_eNB_ue_context_NB_IoT_s   *c1_pP,
  struct rrc_eNB_ue_context_NB_IoT_s   *c2_pP)
//------------------------------------------------------------------------------
{
  if (c1_pP->ue_id_rnti > c2_pP->ue_id_rnti) {
    return 1;
  }

  if (c1_pP->ue_id_rnti < c2_pP->ue_id_rnti) {
    return -1;
  }

  return 0;
}

/* Generate the tree management functions for NB-IoT structures */
RB_GENERATE(rrc_ue_tree_NB_IoT_s, rrc_eNB_ue_context_NB_IoT_s, entries,
            rrc_eNB_compare_ue_rnti_id_NB_IoT);


//------------------------------------------------------------------------------
struct rrc_eNB_ue_context_NB_IoT_s*
rrc_eNB_allocate_new_UE_context_NB_IoT(
  eNB_RRC_INST_NB_IoT* rrc_instance_pP
)
//------------------------------------------------------------------------------
{
  struct rrc_eNB_ue_context_NB_IoT_s* new_p;
  new_p = malloc(sizeof(struct rrc_eNB_ue_context_NB_IoT_s));

  if (new_p == NULL) {
    LOG_E(RRC, "Cannot allocate new ue context\n");
    return NULL;
  }

  memset(new_p, 0, sizeof(struct rrc_eNB_ue_context_NB_IoT_s));
  new_p->local_uid = uid_linear_allocator_new_NB_IoT(rrc_instance_pP);
  return new_p;
}


struct rrc_eNB_ue_context_NB_IoT_s*
rrc_eNB_get_ue_context_NB_IoT(
  eNB_RRC_INST_NB_IoT* rrc_instance_pP,
  rnti_t rntiP)
//------------------------------------------------------------------------------
{
  rrc_eNB_ue_context_NB_IoT_t temp;
  memset(&temp, 0, sizeof(struct rrc_eNB_ue_context_NB_IoT_s));
  /* eNB ue rrc id = 24 bits wide */
  temp.ue_id_rnti = rntiP;
  return RB_FIND(rrc_ue_tree_NB_IoT_s, &rrc_instance_pP->rrc_ue_head, &temp);
}

//------------------------------------------------------------------------------
void rrc_eNB_remove_ue_context_NB_IoT(
  const protocol_ctxt_t* const ctxt_pP,
  eNB_RRC_INST_NB_IoT*                rrc_instance_pP,
  struct rrc_eNB_ue_context_NB_IoT_s* ue_context_pP)
//------------------------------------------------------------------------------
{
  if (rrc_instance_pP == NULL) {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Bad RRC instance\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    return;
  }

  if (ue_context_pP == NULL) {
    LOG_E(RRC, PROTOCOL_RRC_CTXT_UE_FMT" Trying to free a NULL UE context\n",
          PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
    return;
  }

  RB_REMOVE(rrc_ue_tree_NB_IoT_s, &rrc_instance_pP->rrc_ue_head, ue_context_pP);

  MSC_LOG_EVENT(
    MSC_RRC_ENB,
    "0 Removed UE %"PRIx16" ",
    ue_context_pP->ue_context.rnti);

  rrc_eNB_free_mem_UE_context_NB_IoT(ctxt_pP, ue_context_pP);
  uid_linear_allocator_free_NB_IoT(rrc_instance_pP, ue_context_pP->local_uid);
  free(ue_context_pP);
  LOG_I(RRC,
        PROTOCOL_RRC_CTXT_UE_FMT" Removed UE context\n",
        PROTOCOL_RRC_CTXT_UE_ARGS(ctxt_pP));
}



