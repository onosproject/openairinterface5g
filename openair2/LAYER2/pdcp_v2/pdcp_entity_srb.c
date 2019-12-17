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

#include "pdcp_entity_srb.h"

#include <stdlib.h>

void pdcp_entity_srb_recv_pdu(pdcp_entity_t *_entity, char *buffer, int size)
{
  pdcp_entity_srb_t *entity = (pdcp_entity_srb_t *)_entity;

  entity->common.deliver_sdu(entity->common.deliver_sdu_data,
                             (pdcp_entity_t *)entity, buffer+1, size-5);
}

#include <string.h>
#include <stdint.h>
#include "UTIL/OSA/osa_defs.h"

void pdcp_entity_srb_recv_sdu(pdcp_entity_t *_entity, char *buffer, int size,
                              int sdu_id)
{
  pdcp_entity_srb_t *entity = (pdcp_entity_srb_t *)_entity;
  int sn;
  char buf[size+5];

  sn = entity->common.next_pdcp_tx_sn;

  entity->common.next_pdcp_tx_sn++;
  if (entity->common.next_pdcp_tx_sn > entity->common.maximum_pdcp_sn) {
    entity->common.next_pdcp_tx_sn = 0;
    entity->common.tx_hfn++;
  }

  buf[0] = sn & 0x1f;
  memcpy(buf+1, buffer, size);

  if (entity->integrity_active) {
    stream_cipher_t params;
    params.message = (unsigned char *)buf;
    params.blength = (size + 1) << 3;
    params.key = (unsigned char *)entity->key_integrity + 16;
    params.key_length = 16;
    params.count = (entity->common.tx_hfn << 5) | sn;
    params.bearer = entity->rb_id - 1;
    params.direction = SECU_DIRECTION_DOWNLINK;
printf("call stream_compute_integrity\n");
    stream_compute_integrity(EIA2_128_ALG_ID, &params,
                             (unsigned char *)&buf[size+1]);
  } else {
printf("no integrity\n");
    buf[size+1] = 0;
    buf[size+2] = 0;
    buf[size+3] = 0;
    buf[size+4] = 0;
  }

  entity->common.deliver_pdu(entity->common.deliver_pdu_data,
                             (pdcp_entity_t *)entity, buf, size+5, sdu_id);
}

void pdcp_entity_srb_set_integrity_key(pdcp_entity_t *_entity, char *key)
{
printf("activate integrity\n");
  pdcp_entity_srb_t *entity = (pdcp_entity_srb_t *)_entity;

  memcpy(entity->key_integrity, key, 32);
  entity->integrity_active = 1;
}

void pdcp_entity_srb_delete(pdcp_entity_t *_entity)
{
  pdcp_entity_srb_t *entity = (pdcp_entity_srb_t *)_entity;
  free(entity);
}
