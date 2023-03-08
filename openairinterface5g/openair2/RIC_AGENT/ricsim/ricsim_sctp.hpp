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

#ifndef ricsim_SCTP_HPP
#define ricsim_SCTP_HPP

#include "ricsim_defs.h"

const int SERVER_LISTEN_QUEUE_SIZE  = 10;

int sctp_start_server(const char *server_ip_str, const int server_port);

int sctp_start_client(const char *server_ip_str, const int server_port);

int sctp_accept_connection(const char *server_ip_str, const int server_fd);

int sctp_send_data(int &socket_fd, sctp_buffer_t &data);

int sctp_send_data_X2AP(int &socket_fd, sctp_buffer_t &data);

int sctp_receive_data(int &socket_fd, sctp_buffer_t &data);

#endif
