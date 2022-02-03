/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "E2AP-PDU.h"

void encode_and_send_sctp_data(E2AP_PDU_t* pdu);

void encode_and_send_sctp_data(E2AP_PDU_t* pdu, int socket_fd);
