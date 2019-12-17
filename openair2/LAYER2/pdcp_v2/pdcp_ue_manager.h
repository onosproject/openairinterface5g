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

#ifndef _PDCP_UE_MANAGER_H_
#define _PDCP_UE_MANAGER_H_

#include "pdcp_entity.h"

typedef void pdcp_ue_manager_t;

typedef struct pdcp_ue_t {
  int rnti;
  pdcp_entity_t *srb[2];
  pdcp_entity_t *drb[5];
} pdcp_ue_t;

/***********************************************************************/
/* manager functions                                                   */
/***********************************************************************/

pdcp_ue_manager_t *new_pdcp_ue_manager(int enb_flag);

int pdcp_manager_get_enb_flag(pdcp_ue_manager_t *m);

void pdcp_manager_lock(pdcp_ue_manager_t *m);
void pdcp_manager_unlock(pdcp_ue_manager_t *m);

pdcp_ue_t *pdcp_manager_get_ue(pdcp_ue_manager_t *m, int rnti);
void pdcp_manager_remove_ue(pdcp_ue_manager_t *m, int rnti);

/***********************************************************************/
/* ue functions                                                        */
/***********************************************************************/

void pdcp_ue_add_srb_pdcp_entity(pdcp_ue_t *ue, int srb_id,
                                 pdcp_entity_t *entity);
void pdcp_ue_add_drb_pdcp_entity(pdcp_ue_t *ue, int drb_id,
                                 pdcp_entity_t *entity);

#endif /* _PDCP_UE_MANAGER_H_ */
