/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*****************************************************************************
#                                                                            *
# Copyright 2019 AT&T Intellectual Property                                  *
# Copyright 2019 Nokia                                                       *
#                                                                            *
# Licensed under the Apache License, Version 2.0 (the "License");            *
# you may not use this file except in compliance with the License.           *
# You may obtain a copy of the License at                                    *
#                                                                            *
#      http://www.apache.org/licenses/LICENSE-2.0                            *
#                                                                            *
# Unless required by applicable law or agreed to in writing, software        *
# distributed under the License is distributed on an "AS IS" BASIS,          *
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# See the License for the specific language governing permissions and        *
# limitations under the License.                                             *
#                                                                            *
******************************************************************************/
#ifndef E2AP_MESSAGE_HANDLER_HPP
#define E2AP_MESSAGE_HANDLER_HPP


#include "ricsim_sctp.hpp"


extern "C" {
  #include "ricsim_defs.h"
  #include "e2ap_asn1c_codec.h"
}

void e2ap_handle_sctp_data(int &socket_fd, sctp_buffer_t &data, bool xmlenc);

void e2ap_handle_X2SetupRequest(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_X2SetupResponse(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_ENDCX2SetupRequest(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_E2SetupRequest(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_RICSubscriptionRequest(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_RICSubscriptionRequest_securityDemo(E2AP_PDU_t* pdu, int &socket_fd);

void e2ap_handle_ResourceStatusRequest(E2AP_PDU_t* pdu, int &socket_fd);

#endif
