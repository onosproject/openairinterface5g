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
#include "e2ap_message_handler.hpp"

//#include <iostream>
//#include <vector>
#include "encode_e2apv1.hpp"

#include <unistd.h>

void e2ap_handle_sctp_data(int &socket_fd, sctp_buffer_t &data, bool xmlenc)
{
  fprintf(stderr, "in e2ap_handle_sctp_data()\n");
  //decode the data into E2AP-PDU
  E2AP_PDU_t* pdu = (E2AP_PDU_t*)calloc(1, sizeof(E2AP_PDU));
  ASN_STRUCT_RESET(asn_DEF_E2AP_PDU, pdu);

  fprintf(stderr, "decoding...\n");

  asn_transfer_syntax syntax;
  

  syntax = ATS_ALIGNED_BASIC_PER;
  

  fprintf(stderr, "full buffer\n%s\n", data.buffer);
  //  e2ap_asn1c_decode_pdu(pdu, data.buffer, data.len);

  auto rval = asn_decode(nullptr, syntax, &asn_DEF_E2AP_PDU, (void **) &pdu,
		    data.buffer, data.len);
  

  int index = (int)pdu->present;
  fprintf(stderr, "length of data %d\n", rval.consumed);
  fprintf(stderr, "result %d\n", rval.code);
  fprintf(stderr, "index is %d\n", index);
  
  fprintf(stderr, "showing xer of data\n");  
  
  xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu);
  
  int procedureCode = e2ap_asn1c_get_procedureCode(pdu);
  index = (int)pdu->present;

  LOG_D("[E2AP] Unpacked E2AP-PDU: index = %d, procedureCode = %d\n",
                            index, procedureCode);

  switch(procedureCode)
    {
      
    case ProcedureCode_id_E2setup:
      switch(index)
	{
        case E2AP_PDU_PR_initiatingMessage:
	  e2ap_handle_E2SetupRequest(pdu, socket_fd);
          LOG_I("[E2AP] Received SETUP-REQUEST");
          break;
	  
        case E2AP_PDU_PR_successfulOutcome:
          LOG_I("[E2AP] Received SETUP-RESPONSE-SUCCESS");
          break;
	  
        case E2AP_PDU_PR_unsuccessfulOutcome:
          LOG_I("[E2AP] Received SETUP-RESPONSE-FAILURE");
          break;
	  
        default:
          LOG_E("[E2AP] Invalid message index=%d in E2AP-PDU", index);
          break;
	}
      break;    
      
    case ProcedureCode_id_Reset: //reset = 7
      switch(index)
	{
        case E2AP_PDU_PR_initiatingMessage:
          LOG_I("[E2AP] Received RESET-REQUEST");
          break;
	  
        case E2AP_PDU_PR_successfulOutcome:
          break;
	  
        case E2AP_PDU_PR_unsuccessfulOutcome:
          break;
	  
        default:
          LOG_E("[E2AP] Invalid message index=%d in E2AP-PDU", index);
          break;
	}
      break;
      
    case ProcedureCode_id_RICsubscription: //RIC SUBSCRIPTION = 201
      switch(index)
	{
        case E2AP_PDU_PR_initiatingMessage: //initiatingMessage
          LOG_I("[E2AP] Received RIC-SUBSCRIPTION-REQUEST");
          break;
	  
        case E2AP_PDU_PR_successfulOutcome:
          LOG_I("[E2AP] Received RIC-SUBSCRIPTION-RESPONSE");
          break;
	  
        case E2AP_PDU_PR_unsuccessfulOutcome:
          LOG_I("[E2AP] Received RIC-SUBSCRIPTION-FAILURE");
          break;
	  
        default:
          LOG_E("[E2AP] Invalid message index=%d in E2AP-PDU", index);
          break;
	}
      break;
    case ProcedureCode_id_RICsubscriptionDelete:
          switch(index) {
              case E2AP_PDU_PR_initiatingMessage:
                  LOG_I("[E2AP] Received RIC-SUBSCRIPTION-DELETE-REQUEST");
                  break;
              case E2AP_PDU_PR_successfulOutcome:
                  LOG_I("[E2AP] Received RIC-SUBSCRIPTION-DELETE-RESPONSE");
                  break;
              case E2AP_PDU_PR_unsuccessfulOutcome:
                  LOG_I("[E2AP] Received RIC-SUBSCRIPTION-DELETE-FAILURE");
                  break;
              default:
                  LOG_E("[E2AP] Invalid message index=%d in E2AP-PDU", index);
                  break;
          }
          break;

    case ProcedureCode_id_RICindication: // 205
      switch(index)
	{
        case E2AP_PDU_PR_initiatingMessage: //initiatingMessage
          LOG_I("[E2AP] Received RIC-INDICATION-REQUEST");
          // e2ap_handle_RICSubscriptionRequest(pdu, socket_fd);
          break;
        case E2AP_PDU_PR_successfulOutcome:
          LOG_I("[E2AP] Received RIC-INDICATION-RESPONSE");
          break;
	  
        case E2AP_PDU_PR_unsuccessfulOutcome:
          LOG_I("[E2AP] Received RIC-INDICATION-FAILURE");
          break;
	  
        default:
          LOG_E("[E2AP] Invalid message index=%d in E2AP-PDU %d", index,
                                    (int)ProcedureCode_id_RICindication);
          break;
	}
      break;
      
    default:
      
      LOG_E("[E2AP] No available handler for procedureCode=%d", procedureCode);

      break;
    }
}

void e2ap_handle_E2SetupRequest(E2AP_PDU_t* pdu, int &socket_fd) {

  
  E2AP_PDU_t* res_pdu = (E2AP_PDU_t*)calloc(1, sizeof(E2AP_PDU));
  generate_e2apv1_setup_response(res_pdu);

  
  LOG_D("[E2AP] Created E2-SETUP-RESPONSE");

  e2ap_asn1c_print_pdu(res_pdu);


  auto buffer_size = MAX_SCTP_BUFFER;
  unsigned char buffer[MAX_SCTP_BUFFER];
  
  sctp_buffer_t data;
  auto er = asn_encode_to_buffer(nullptr, ATS_BASIC_XER, &asn_DEF_E2AP_PDU, res_pdu, buffer, buffer_size);

  data.len = er.encoded;
  fprintf(stderr, "er encoded is %d\n", er.encoded);  
  
  //data.len = e2ap_asn1c_encode_pdu(res_pdu, &buf);
  memcpy(data.buffer, buffer, er.encoded);

  //send response data over sctp
  if(sctp_send_data(socket_fd, data) > 0) {
    LOG_I("[SCTP] Sent E2-SETUP-RESPONSE");
  } else {
    LOG_E("[SCTP] Unable to send E2-SETUP-RESPONSE to peer");
  }

  sleep(5);

  //Sending Subscription Request

  E2AP_PDU_t* pdu_sub = (E2AP_PDU_t*)calloc(1,sizeof(E2AP_PDU));

  generate_e2apv1_subscription_request(pdu_sub);

  xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu_sub);

  auto buffer_size2 = MAX_SCTP_BUFFER;
  unsigned char buffer2[MAX_SCTP_BUFFER];
  
  sctp_buffer_t data2;

  auto er2 = asn_encode_to_buffer(nullptr, ATS_ALIGNED_BASIC_PER, &asn_DEF_E2AP_PDU, pdu_sub, buffer2, buffer_size2);
  
  data2.len = er2.encoded;
  memcpy(data2.buffer, buffer2, er2.encoded);
  
  fprintf(stderr, "er encded is %d\n", er2.encoded);

  if(sctp_send_data(socket_fd, data2) > 0) {
    LOG_I("[SCTP] Sent E2-SUBSCRIPTION-REQUEST");
  } else {
    LOG_E("[SCTP] Unable to send E2-SUBSCRIPTION-REQUEST to peer");
  }  


}


void e2ap_handle_RICSubscriptionRequest(E2AP_PDU_t* pdu, int &socket_fd)
{

  //Send back Subscription Success Response

  E2AP_PDU_t* pdu_resp = (E2AP_PDU_t*)calloc(1,sizeof(E2AP_PDU));

  generate_e2apv1_subscription_response(pdu_resp, pdu);

  fprintf(stderr, "Subscription Response\n");

  xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu_resp);

  auto buffer_size2 = MAX_SCTP_BUFFER;
  unsigned char buffer2[MAX_SCTP_BUFFER];
  
  sctp_buffer_t data2;

  auto er2 = asn_encode_to_buffer(nullptr, ATS_ALIGNED_BASIC_PER, &asn_DEF_E2AP_PDU, pdu_resp, buffer2, buffer_size2);
  data2.len = er2.encoded;

  fprintf(stderr, "er encded is %d\n", er2.encoded);

  memcpy(data2.buffer, buffer2, er2.encoded);

  if(sctp_send_data(socket_fd, data2) > 0) {
    LOG_I("[SCTP] Sent RIC-SUBSCRIPTION-RESPONSE");
  } else {
    LOG_E("[SCTP] Unable to send RIC-SUBSCRIPTION-RESPONSE to peer");
  }
  
  
  //Send back an Indication

  E2AP_PDU_t* pdu_ind = (E2AP_PDU_t*)calloc(1,sizeof(E2AP_PDU));

  generate_e2apv1_indication_request(pdu_ind);

  xer_fprint(stderr, &asn_DEF_E2AP_PDU, pdu_ind);

  auto buffer_size = MAX_SCTP_BUFFER;
  unsigned char buffer[MAX_SCTP_BUFFER];
  
  sctp_buffer_t data;

  auto er = asn_encode_to_buffer(nullptr, ATS_ALIGNED_BASIC_PER, &asn_DEF_E2AP_PDU, pdu_ind, buffer, buffer_size);
  data.len = er.encoded;

  fprintf(stderr, "er encded is %d\n", er.encoded);

  memcpy(data.buffer, buffer, er.encoded);

  if(sctp_send_data(socket_fd, data) > 0) {
    LOG_I("[SCTP] Sent RIC-INDICATION-REQUEST");
  } else {
    LOG_E("[SCTP] Unable to send RIC-INDICATION-REQUEST to peer");
  }  

}




