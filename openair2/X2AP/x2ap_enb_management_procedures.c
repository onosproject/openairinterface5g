/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2014 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is
   included in this distribution in the file called "COPYING". If not,
   see <http://www.gnu.org/licenses/>.

  Contact Information
  OpenAirInterface Admin: openair_admin@eurecom.fr
  OpenAirInterface Tech : openair_tech@eurecom.fr
  OpenAirInterface Dev  : openair4g-devel@lists.eurecom.fr

  Address      : Eurecom, Compus SophiaTech 450, route des chappes, 06451 Biot, France.

 *******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "intertask_interface.h"

#include "assertions.h"
#include "conversions.h"

#include "x2ap_common.h"
#include "x2ap_eNB_defs.h"
#include "x2ap.h"

x2ap_eNB_internal_data_t x2ap_eNB_internal_data;

RB_GENERATE(x2ap_enb_map, x2ap_eNB_data_s, entry, x2ap_eNB_compare_assoc_id);

int x2ap_eNB_compare_assoc_id(
  struct x2ap_eNB_data_s *p1, struct x2ap_eNB_data_s *p2)
{
  if (p1->assoc_id == -1) {
    if (p1->cnx_id < p2->cnx_id) {
      return -1;
    }

    if (p1->cnx_id > p2->cnx_id) {
      return 1;
    }
  } else {
    if (p1->assoc_id < p2->assoc_id) {
      return -1;
    }

    if (p1->assoc_id > p2->assoc_id) {
      return 1;
    }
  }

  /* Matching reference */
  return 0;
}

uint16_t x2ap_eNB_fetch_add_global_cnx_id(void)
{
  return ++x2ap_eNB_internal_data.global_cnx_id;
}

void x2ap_eNB_prepare_internal_data(void)
{
  memset(&x2ap_eNB_internal_data, 0, sizeof(x2ap_eNB_internal_data));
  STAILQ_INIT(&x2ap_eNB_internal_data.x2ap_eNB_instances_head);
}

void x2ap_eNB_insert_new_instance(x2ap_eNB_instance_t *new_instance_p)
{
  DevAssert(new_instance_p != NULL);

  STAILQ_INSERT_TAIL(&x2ap_eNB_internal_data.x2ap_eNB_instances_head,
                     new_instance_p, x2ap_eNB_entries);
}

struct s1ap_eNB_mme_data_s *x2ap_eNB_get_taregt_eNB(
  x2ap_eNB_instance_t *instance_p,
  int32_t assoc_id, uint16_t cnx_id)
{
  struct x2ap_eNB_data_s  temp;
  struct x2ap_eNB_data_s *found;

  memset(&temp, 0, sizeof(struct x2ap_eNB_data_s));

  temp.assoc_id = assoc_id;
  temp.cnx_id   = cnx_id;

  if (instance_p == NULL) {
    STAILQ_FOREACH(instance_p, &x2ap_eNB_internal_data.x2ap_eNB_instances_head,
                   x2ap_eNB_entries) {
      found = RB_FIND(x2ap_enb_map, &instance_p->x2ap_enb_head, &temp);

      if (found != NULL) {
        return found;
      }
    }
  } else {
    return RB_FIND(x2ap_enb_map, &instance_p->x2ap_enb_head, &temp);
  }

  return NULL;
}

x2ap_eNB_instance_t *x2ap_eNB_get_instance(instance_t instance)
{
  x2ap_eNB_instance_t *temp = NULL;

  STAILQ_FOREACH(temp, &x2ap_eNB_internal_data.x2ap_eNB_instances_head,
                 x2ap_eNB_entries) {
    if (temp->instance == instance) {
      /* Matching occurence */
      return temp;
    }
  }

  return NULL;
}
