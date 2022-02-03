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

#ifndef _E2AP_ENCODER_H
#define _E2AP_ENCODER_H

#include "E2AP_E2AP-PDU.h"

ssize_t e2ap_encode(const struct asn_TYPE_descriptor_s *td,
		    const asn_per_constraints_t *constraints,void *sptr,
		    uint8_t **buf)
  __attribute__ ((warn_unused_result));
ssize_t e2ap_encode_pdu(E2AP_E2AP_PDU_t *pdu,uint8_t **buf,uint32_t *len)
  __attribute__ ((warn_unused_result));

#endif /* _E2AP_ENCODER_H */
