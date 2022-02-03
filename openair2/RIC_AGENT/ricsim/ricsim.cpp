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

#include <stdio.h>
#include <unistd.h>
#include <assert.h>

#include "ricsim_sctp.hpp"
#include "e2ap_message_handler.hpp"

#include "encode_e2apv1.hpp"
#include "encode_kpm.hpp"

extern "C" {
  #include "ricsim_defs.h"
  #include "E2AP-PDU.h"
  #include "e2ap_asn1c_codec.h"
  #include "GlobalE2node-ID.h"
  #include "GlobalE2node-gNB-ID.h"
  #include "GlobalgNB-ID.h"
  #include "OCTET_STRING.h"
  #include "asn_application.h"
  #include "GNB-ID-Choice.h"
  #include "ProtocolIE-Field.h"
  #include "E2setupRequest.h"
  #include "RICaction-ToBeSetup-Item.h"
  #include "RICactions-ToBeSetup-List.h"
  #include "RICeventTriggerDefinition.h"
  #include "RICsubscriptionRequest.h"
  #include "ProtocolIE-SingleContainer.h"
  #include "RANfunctions-List.h"
  #include "RICindication.h"
  #include "RICsubsequentActionType.h"
  #include "RICsubsequentAction.h"  
  #include "RICtimeToWait.h"
}

using namespace std;

/*
struct {
  type **array;
  int count;
  int size;
  void (*free)(decltype(*array));
} 
*/

int client_fd = 0;

void encode_and_send_sctp_data(E2AP_PDU_t* pdu, int client_fd)
{
  uint8_t       *buf;
  sctp_buffer_t data;

  data.len = e2ap_asn1c_encode_pdu(pdu, &buf);
  memcpy(data.buffer, buf, min(data.len, MAX_SCTP_BUFFER));

  sctp_send_data(client_fd, data);
}


void encode_and_send_sctp_data(E2AP_PDU_t* pdu)
{
  uint8_t       *buf;
  sctp_buffer_t data;

  data.len = e2ap_asn1c_encode_pdu(pdu, &buf);
  memcpy(data.buffer, buf, min(data.len, MAX_SCTP_BUFFER));

  sctp_send_data(client_fd, data);
}

void wait_for_sctp_data(int client_fd)
{
  sctp_buffer_t recv_buf;
  if(sctp_receive_data(client_fd, recv_buf) > 0)
  {
    LOG_I("[SCTP] Received new data of size %d", recv_buf.len);
    e2ap_handle_sctp_data(client_fd, recv_buf, false);
  }
}



int main(int argc, char* argv[]){
  LOG_I("Start RIC Simulator");

  bool xmlenc = true;

  options_t ops = read_input_options(argc, argv);

  int server_fd = sctp_start_server(ops.server_ip, ops.server_port);
  client_fd = sctp_accept_connection(ops.server_ip, server_fd);

  sctp_buffer_t recv_buf;

  LOG_I("[SCTP] Waiting for SCTP data");

  int outer_loop_index = 3;
  int inner_loop_index = 10;

  while (outer_loop_index) {
      while (inner_loop_index) //constantly looking for data on SCTP interface    
      {
        LOG_I("in while loop");
        if(sctp_receive_data(client_fd, recv_buf) <= 0)
          break;

        LOG_I("[SCTP] Received new data of size %d", recv_buf.len);

        e2ap_handle_sctp_data(client_fd, recv_buf, xmlenc);
        inner_loop_index--;

        if (xmlenc)
          xmlenc = false;
      }

      // Send subscription delete
      {
          E2AP_PDU_t* pdu_sub = (E2AP_PDU_t*)calloc(1, sizeof(E2AP_PDU));

          generate_e2apv1_subscription_delete(pdu_sub);

          xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu_sub);

          auto buffer_size2 = MAX_SCTP_BUFFER;
          unsigned char buffer2[MAX_SCTP_BUFFER];

          sctp_buffer_t data2;

          auto er2 = asn_encode_to_buffer(nullptr, ATS_ALIGNED_BASIC_PER, &asn_DEF_E2AP_PDU, pdu_sub, buffer2, buffer_size2);

          data2.len = er2.encoded;
          memcpy(data2.buffer, buffer2, er2.encoded);

          fprintf(stderr, "er encded is %zu\n", er2.encoded);

          if(sctp_send_data(client_fd, data2) > 0) {
            LOG_I("[SCTP] Sent E2-SUBSCRIPTION-DELETE");
          } else {
            LOG_E("[SCTP] Unable to send E2-SUBSCRIPTION-DELETE to peer");
          }
      }


      //Sending Subscription Request
      {
          E2AP_PDU_t* pdu_sub = (E2AP_PDU_t*)calloc(1,sizeof(E2AP_PDU));
          generate_e2apv1_subscription_request(pdu_sub);
          xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu_sub);
          auto buffer_size2 = MAX_SCTP_BUFFER;
          unsigned char buffer2[MAX_SCTP_BUFFER];
          sctp_buffer_t data2;
          auto er2 = asn_encode_to_buffer(nullptr, ATS_ALIGNED_BASIC_PER, &asn_DEF_E2AP_PDU, pdu_sub, buffer2, buffer_size2);
          data2.len = er2.encoded;
          memcpy(data2.buffer, buffer2, er2.encoded);
          fprintf(stderr, "er encded is %zu\n", er2.encoded);
          if(sctp_send_data(client_fd, data2) > 0) {
            LOG_I("[SCTP] Sent E2-SUBSCRIPTION-REQUEST");
          } else {
            LOG_E("[SCTP] Unable to send E2-SUBSCRIPTION-REQUEST to peer");
          }
      }

      outer_loop_index--;
  }

  while (1) //constantly looking for data on SCTP interface    
  {
    LOG_I("in while loop");
    if(sctp_receive_data(client_fd, recv_buf) <= 0)
      break;

    LOG_I("[SCTP] Received new data of size %d", recv_buf.len);

    e2ap_handle_sctp_data(client_fd, recv_buf, xmlenc);

    if (xmlenc)
      xmlenc = false;
  }

  return 0;
}
