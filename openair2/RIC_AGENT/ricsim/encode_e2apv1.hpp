/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef ENCODE_E2APV1_HPP
#define ENCODE_E2APV1_HPP

#include "encode_kpm.hpp"


extern "C" {
#include "E2AP-PDU.h"
}

void buildSubsReq(E2AP_PDU_t *pdu);

void generate_e2apv1_setup_request(E2AP_PDU_t *setup_req_pdu);

void generate_e2apv1_setup_response(E2AP_PDU_t *setup_resp_pdu);

void generate_e2apv1_subscription_delete(E2AP_PDU_t *sub_req_pdu);

void generate_e2apv1_subscription_request(E2AP_PDU_t *sub_req_pdu);

void generate_e2apv1_subscription_response(E2AP_PDU_t *sub_resp_pdu, E2AP_PDU_t *sub_req_pdu);

void generate_e2apv1_indication_request(E2AP_PDU_t *ind_req_pdu);

void generate_e2apv1_subscription_response_success(E2AP_PDU *e2ap_pdu, long reqActionIdsAccepted[], long reqActionIdsRejected[], int accept_size, int reject_size, long reqRequestorId, long reqInstanceId);

void generate_e2apv1_indication_request_parameterized(E2AP_PDU *e2ap_pdu, long requestorId, long instanceId, long ranFunctionId, long actionId, long seqNum, uint8_t *ind_header_buf, int header_length, uint8_t *ind_message_buf, int message_length);

#endif
