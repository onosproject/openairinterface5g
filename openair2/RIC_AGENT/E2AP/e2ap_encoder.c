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

#include "assertions.h"
#include "conversions.h"
#include "intertask_interface.h"
#include "e2ap_encoder.h"

#include "E2AP_E2AP-PDU.h"
#include "E2AP_ProcedureCode.h"
#include "per_encoder.h"
#include "asn_application.h"
#include "per_support.h"

ssize_t e2ap_encode(const struct asn_TYPE_descriptor_s *td,
        const asn_per_constraints_t *constraints,
        void *sptr,
        uint8_t **buf)
{
    ssize_t encoded;

    DevAssert(td != NULL);
    DevAssert(buf != NULL);

    xer_fprint(stdout, td, sptr);

    encoded = aper_encode_to_new_buffer(td, constraints, sptr, (void **)buf);
    if (encoded < 0) {
        return -1;
    }

    ASN_STRUCT_FREE_CONTENTS_ONLY((*td), sptr);

    return encoded;
}

ssize_t e2ap_encode_pdu(E2AP_E2AP_PDU_t *pdu, uint8_t **buf, uint32_t *len)
{
    ssize_t encoded;

    DevAssert(pdu != NULL);
    DevAssert(buf != NULL);
    DevAssert(len != NULL);

    encoded = e2ap_encode(&asn_DEF_E2AP_E2AP_PDU,0,pdu,buf);
    if (encoded < 0) {
        return -1;
    }

    *len = encoded;

    return encoded;
}
