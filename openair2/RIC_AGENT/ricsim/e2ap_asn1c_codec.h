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
#ifndef E2AP_ASN1C_CODEC_H
#define E2AP_ASN1C_CODEC_H

#include "ricsim_defs.h"
#include "E2AP-PDU.h"
#include "InitiatingMessage.h"
#include "SuccessfulOutcome.h"
#include "UnsuccessfulOutcome.h"

#include "GlobalE2node-ID.h"
#include "E2setupRequest.h"

#define ASN1C_PDU_PRINT_BUFFER     4096
#define MAX_XML_BUFFER             10000
#define E2AP_XML_DIR               "/src/E2AP/XML/"

void e2ap_asn1c_print_pdu(const E2AP_PDU_t* pdu);

void asn1c_xer_print(asn_TYPE_descriptor_t *typeDescriptor, void *data);

E2AP_PDU_t* e2ap_xml_to_pdu(char const* xml_message);
E2setupRequest_t* smaller_e2ap_xml_to_pdu(char const* xml_message);

int e2ap_asn1c_encode_pdu(E2AP_PDU_t* pdu, unsigned char **buffer);

void e2ap_asn1c_decode_pdu(E2AP_PDU_t* pdu, unsigned char *buffer, int len);

int e2ap_asn1c_get_procedureCode(E2AP_PDU_t* pdu);

#endif
