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

/*! \file x2ap_eNB_encoder.c
 * \brief x2ap pdu encode procedures for eNB
 * \author Navid Nikaein
 * \date 2015- 2016
 * \version 0.1
 */

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "conversions.h"

#include "intertask_interface.h"

#include "x2ap_common.h"
#include "x2ap_ies_defs.h"
#include "x2ap_eNB_encoder.h"

static inline int x2ap_encode_initiating(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length);
static inline int x2ap_encode_successful(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length);
static inline int x2ap_encode_unsuccessful(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length);

int 
x2ap_eNB_encode_pdu(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length) {

  
  switch(x2ap_message_p->direction) {
  case X2AP_PDU_PR_initiatingMessage:
    return x2ap_eNB_encode_initiating(x2ap_message_p, buf, length);
  case X2AP_PDU_PR_successfulOutcome:
    return x2ap_eNB_encode_successful(x2ap_message_p, buf, length);
  case X2AP_PDU_PR_unsuccessfulOutcome:
    return x2ap_eNB_encode_unsuccessful(x2ap_message_p, buf, length);
  default:
    X2AP_DEBUG("Unknown message outcome (%d) or not implemented", (int)x2ap_message_p->direction);
    break;
  }
  return -1;
}

static inline int 
x2ap_eNB_encode_initiating(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length){
  
  int ret = -1;
  MessageDef *message_p;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;

  DevAssert(x2ap_message_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;

  switch(x2ap_message_p->procedureCode) {
  case ProcedureCode_id_reset:
    ret = x2ap_reset_request_encoder(&x2ap_message_p->msg.resetRequest_IEs, buf, length);
  
#warning "do the same for the other messages" 
    x2ap_xer_print_x2ap_resetrequest_(x2ap_xer__print2sp, message_string, x2ap_message_p);
    message_id = X2AP_RESET_REQUST_LOG;
    message_p = itti_alloc_new_message_sized(TASK_X2AP, message_id, message_string_size + sizeof (IttiMsgText));
    message_p->ittiMsg.x2ap_reset_request_log.size = message_string_size;
    memcpy(&message_p->ittiMsg.x2ap_reset_request_log.text, message_string, message_string_size);
    itti_send_msg_to_task(TASK_UNKNOWN, INSTANCE_DEFAULT, message_p);
    free(message_string);
    break;
  
  case ProcedureCode_id_loadIndication:
    ret = x2ap_load_information_encoder(&x2ap_message_p->msg.loadInformation_IEs, buf, length);
    break;
    
  case ProcedureCode_id_resourceStatusReportingInitiation:
    ret =  x2ap_resource_status_request_encoder(&x2ap_message_p->msg.resourceStatusRequest_IEs, buf, length);
    break;
    
  case ProcedureCode_id_resourceStatusReporting:
    ret = x2ap_resource_status_update_encoder(&x2ap_message_p->msg.resourceStatusUpdate_IEs, buf, length);
    break;
    
  case ProcedureCode_id_mobilitySettingsChange:
    ret = x2ap_mobility_change_request_encoder(&x2ap_message_p->msg.mobilityChangeRequest_IEs, buf, length);
    break;
    
  case ProcedureCode_id_x2Setup:
    ret = x2_setup_request_encoder(&x2ap_message_p->msg.x2SetupRequest_IEs, buf, length);
    break;
  case ProcedureCode_id_handoverPreparation:
    ret = x2ap_handover_request_encoder(&x2ap_message_p->msg.handoverRequest_IEs, buf, length);
    break;
  case ProcedureCode_id_errorIndication:
    ret = x2ap_error_indication_encoder(&x2ap_message_p->msg.errorIndication_IEs, buf, length);
    break;
  case ProcedureCode_id_handoverCancel:
    ret = x2ap_handover_cancel_encoder(&x2ap_message_p->msg.handoverCancel_IEs, buf, length);
    break;
  case ProcedureCode_id_handoverReport:
    ret = x2ap_handover_report_encoder(&x2ap_message_p->msg.handoverReport_IEs, buf, length);
    break;
  case ProcedureCode_id_eNBConfigurationUpdate:
    ret =  x2ap_eNB_configuration_update_request_encoder(&x2ap_message_p->msg.enbConfigurationUpdate_IEs,buf, length);
    break;
  case ProcedureCode_id_uEContextRelease:
    ret = x2ap_ue_context_release_encoder(&x2ap_message_p->msg.ueContextRelease_IEs, buf, length );
    break;
  case ProcedureCode_id_snStatusTransfer:
    ret = x2ap_sn_status_transfer_encoder(&x2ap_message_p->msg.snStatusTransfer_IEs,  buf, length);
    break;
  case ProcedureCode_id_rLFIndication:
    ret = x2ap_rlf_indication_encoder(&x2ap_message_p->msg.rlfIndication_IEs,  buf, length);
    break;
  case ProcedureCode_id_cellActivation:
    ret =  x2ap_cell_activation_request_encoder(&x2ap_message_p->msg.cellActivationRequest_IEs, buf, length);
    break;
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  
  return ret; 
}

static inline int
x2ap_eNB_encode_successful(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length){

  int ret = -1;
  MessageDef *message_p;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;
   
  DevAssert(x2ap_message_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;
  
  switch(x2ap_message_p->procedureCode) {
  
  case ProcedureCode_id_resourceStatusReportingInitiation:
    ret =  x2ap_resource_status_response_encoder(&x2ap_message_p->msg.resourceStatusResponse_IEs, buf, length);

#warning "do the same for the other messages" 
    x2ap_xer_print_x2ap_resourcestatusresponse_(x2ap_xer__print2sp, message_string, x2ap_message_p);
    message_id = X2AP_RESOURCE_STATUS_RESPONSE_LOG;
    message_p = itti_alloc_new_message_sized(TASK_X2AP, message_id, message_string_size + sizeof (IttiMsgText));
    message_p->ittiMsg.x2ap_resource_status_response_log.size = message_string_size;
    memcpy(&message_p->ittiMsg.x2ap_resource_status_response_log.text, message_string, message_string_size);
    itti_send_msg_to_task(TASK_UNKNOWN, INSTANCE_DEFAULT, message_p);
    free(message_string);
    break;
  case ProcedureCode_id_mobilitySettingsChange:
    ret = x2ap_mobility_change_acknowledge_encoder(&x2ap_message_p->msg.mobilityChangeAcknowledge_IEs, buf, length);
     break;
  case ProcedureCode_id_reset:
    ret =  x2ap_reset_response_encoder(&x2ap_message_p->msg.resetResponse_IEs, buf, length);	
    break;	
  case ProcedureCode_id_x2Setup:
    ret =  x2_setup_response_encoder(&x2ap_message_p->msg.x2SetupResponse_IEs, buf, length);	
    break;
  case ProcedureCode_id_handoverPreparation:
    ret = x2ap_handover_request_acknowledge_encoder(&x2ap_message_p->msg.handoverRequestAcknowledge_IEs, buf, length);
    break;
  case ProcedureCode_id_eNBConfigurationUpdate:
    ret = x2ap_eNB_configuration_update_acknowledge_encoder(&x2ap_message_p->msg.enbConfigurationUpdateAcknowledge_IEs, buf, length);
    break;
  case ProcedureCode_id_cellActivation:
    ret =  x2ap_cell_activation_response_encoder(&x2ap_message_p->msg.cellActivationResponse_IEs, buf, length);
     break;
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  return ret; 
}

static inline 
int x2ap_eNB_encode_unsuccessful(x2ap_message *x2ap_message_p, uint8_t **buf, uint32_t *length){

  int ret = -1;
  MessageDef *message_p;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;

  DevAssert(x2ap_message_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;
  
  switch(x2ap_message_p->procedureCode) {
  case ProcedureCode_id_resourceStatusReportingInitiation:
    ret = x2ap_resource_status_failure_encoder(&x2ap_message_p->msg.resourceStatusFailure_IEs,  buf, length);
#warning "do the same for the other messages" 
    x2ap_xer_print_x2ap_resourcestatusfailure_(x2ap_xer__print2sp, message_string, x2ap_message_p);
    message_id = X2AP_RESOURCE_STATUS_FAILURE_LOG;
    message_p = itti_alloc_new_message_sized(TASK_X2AP, message_id, message_string_size + sizeof (IttiMsgText));
    message_p->ittiMsg.x2ap_resource_status_failure_log.size = message_string_size;
    memcpy(&message_p->ittiMsg.x2ap_resource_status_failure_log.text, message_string, message_string_size);
    itti_send_msg_to_task(TASK_UNKNOWN, INSTANCE_DEFAULT, message_p);
    free(message_string);
    break;
    
  case ProcedureCode_id_mobilitySettingsChange:
    ret =  x2ap_mobility_change_failure_encoder(&x2ap_message_p->msg.mobilityChangeFailure_IEs,  buf, length);
    break;
  case ProcedureCode_id_x2Setup:
    ret = x2_setup_failure_encoder(&x2ap_message_p->msg.x2SetupFailure_IEs, buf, length);
    break;
  case ProcedureCode_id_handoverPreparation:
    ret = x2ap_handover_preparation_failure_encoder(&x2ap_message_p->msg.handoverPreparationFailure_IEs, buf, length);
    break;
  case ProcedureCode_id_eNBConfigurationUpdate:
    ret = x2ap_eNB_configuration_update_failure_encoder(&x2ap_message_p->msg.enbConfigurationUpdateFailure_IEs, buf, length);
   break;
  case ProcedureCode_id_cellActivation:
    ret = x2ap_cell_activation_failure_encoder(&x2ap_message_p->msg.cellActivationFailure_IEs, buf, length);
    break;
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  return ret; 
}


static inline 
int x2ap_reset_request_encoder(ResetRequest_IEs_t *resetRequest_IEs, uint8_t **buf, uint32_t *length){

	int i;
	ResetRequest_t resetRequest;
	memset (&resetRequest,0, sizeof(ResetRequest_t));

	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding Reset request message	
	if (x2ap_encode_resetrequest_ies(&resetRequest, resetRequest_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_reset
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_reset, Criticality_reject, &asn_DEF_ResetRequest, &resetRequest) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

int x2_setup_response_encoder(X2SetupResponse_IEs_t *x2SetupResponse_IEs, uint8_t **buf, uint32_t *length){

	int i;	
	X2SetupResponse_t x2SetupResponse;
	memset (&x2SetupResponse,0, sizeof(X2SetupResponse_t));

	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding X2 Setup response message
	if (x2ap_encode_x2setupresponse_ies(&x2SetupResponse, x2SetupResponse_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_x2Setup
	if (x2ap_generate_successfull_outcome (buf, length, ProcedureCode_id_x2Setup, Criticality_reject, &asn_DEF_X2SetupResponse, &x2SetupResponse) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

int x2_setup_failure_encoder(X2SetupFailure_IEs_t *x2SetupFailure_IEs, uint8_t **buf, uint32_t *length){

	int i;
	X2SetupFailure_t x2SetupFailure;
	memset (&x2SetupFailure,0, sizeof(X2SetupFailure_t));

	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding X2 Setup failure message
	if (x2ap_encode_x2setupfailure_ies(&x2SetupFailure, x2SetupFailure_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_x2Setup
	if (x2ap_generate_unsuccessfull_outcome (buf, length, ProcedureCode_id_x2Setup, Criticality_reject, &asn_DEF_X2SetupFailure, &x2SetupFailure) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

int x2_setup_request_encoder(X2SetupRequest_IEs_t *x2SetupRequest_IEs, uint8_t **buf, uint32_t *length){

	int i;
	X2SetupRequest_t x2SetupRequest;
	memset (&x2SetupRequest,0, sizeof(X2SetupRequest_t));

	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding X2 Setup request message
	if (x2ap_encode_x2setuprequest_ies(&x2SetupRequest, x2SetupRequest_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_x2Setup
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_x2Setup, Criticality_reject, &asn_DEF_X2SetupRequest, &x2SetupRequest) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_reset_response_encoder(ResetResponse_IEs_t *resetResponse_IEs, uint8_t **buf, uint32_t *length){
	
	int i;
	ResetResponse_t resetResponse;
	memset (&resetResponse,0, sizeof(ResetResponse_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding Reset Response message
	if (x2ap_encode_resetresponse_ies(&resetResponse, resetResponse_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_reset
	if (x2ap_generate_successfull_outcome(buf, length, ProcedureCode_id_reset, Criticality_reject, &asn_DEF_ResetResponse, &resetResponse) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

static inline 
int x2ap_mobility_change_failure_encoder(MobilityChangeFailure_IEs_t *mobilityChangeFailure_IEs, uint8_t **buf, uint32_t *length){
	
	int i;
	MobilityChangeFailure_t mobilityChangeFailure;
	memset (&mobilityChangeFailure,0, sizeof(MobilityChangeFailure_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding Mobility Change Failure message
	if (x2ap_encode_mobilitychangefailure_ies(&mobilityChangeFailure, mobilityChangeFailure_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_mobilitySettingsChange
	if (x2ap_generate_unsuccessfull_outcome (buf, length, ProcedureCode_id_mobilitySettingsChange, Criticality_reject, &asn_DEF_MobilityChangeFailure, &mobilityChangeFailure) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

static inline 
int x2ap_mobility_change_acknowledge_encoder(MobilityChangeAcknowledge_IEs_t *mobilityChangeAcknowledge_IEs, uint8_t **buf, uint32_t *length){
	
	int i;
	MobilityChangeAcknowledge_t mobilityChangeAcknowledge;
	memset (&mobilityChangeAcknowledge,0, sizeof(MobilityChangeAcknowledge_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding Mobility Change Acknowledge message
	if (x2ap_encode_mobilitychangeacknowledge_ies(&mobilityChangeAcknowledge, mobilityChangeAcknowledge_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_mobilitySettingsChange
	if (x2ap_generate_successfull_outcome (buf, length, ProcedureCode_id_mobilitySettingsChange, Criticality_reject, &asn_DEF_MobilityChangeAcknowledge, &mobilityChangeAcknowledge) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_mobility_change_request_encoder(MobilityChangeRequest_IEs_t *mobilityChangeRequest_IEs, uint8_t **buf, uint32_t *length){

	int i;
	MobilityChangeRequest_t mobilityChangeRequest;
	memset (&mobilityChangeRequest,0, sizeof(MobilityChangeRequest_t));

	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Mobility Change Request message
	if (x2ap_encode_mobilitychangerequest_ies(&mobilityChangeRequest, mobilityChangeRequest_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_mobilitySettingsChange
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_mobilitySettingsChange, Criticality_reject, &asn_DEF_MobilityChangeRequest, &mobilityChangeRequest) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_resource_status_update_encoder(ResourceStatusUpdate_IEs_t *resourceStatusUpdate_IEs, uint8_t **buf, uint32_t *length){

	int i;
	ResourceStatusUpdate_t resourceStatusUpdate;
	memset (&resourceStatusUpdate,0, sizeof(ResourceStatusUpdate_t));

	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Resource Status Update message
	if (x2ap_encode_resourcestatusupdate_ies(&resourceStatusUpdate, resourceStatusUpdate_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_resourceStatusReporting
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_resourceStatusReporting, Criticality_ignore, &asn_DEF_ResourceStatusUpdate, &resourceStatusUpdate) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

static inline 
int x2ap_resource_status_failure_encoder(ResourceStatusFailure_IEs_t *resourceStatusFailure_IEs, uint8_t **buf, uint32_t *length){

	int i;
	ResourceStatusFailure_t resourceStatusFailure;
	memset (&resourceStatusFailure,0, sizeof(ResourceStatusFailure_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Resource Status Failure message	
	if (x2ap_encode_resourcestatusfailure_ies(&resourceStatusFailure, resourceStatusFailure_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_resourceStatusReportingInitiation
	if (x2ap_generate_unsuccessfull_outcome (buf, length, ProcedureCode_id_resourceStatusReportingInitiation, Criticality_reject, &asn_DEF_ResourceStatusFailure, &resourceStatusFailure) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_resource_status_response_encoder(ResourceStatusResponse_IEs_t *resourceStatusResponse_IEs, uint8_t **buf, uint32_t *length){
	
	int i;
	ResourceStatusResponse_t resourceStatusResponse;
	memset (&resourceStatusResponse,0, sizeof(ResourceStatusResponse_t));

	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Resource Status Response message	
	if (x2ap_encode_resourcestatusresponse_ies(&resourceStatusResponse, resourceStatusResponse_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_resourceStatusReportingInitiation
	if (x2ap_generate_successfull_outcome (buf, length, ProcedureCode_id_resourceStatusReportingInitiation, Criticality_reject, &asn_DEF_ResourceStatusResponse, &resourceStatusResponse) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_resource_status_request_encoder(ResourceStatusRequest_IEs_t *resourceStatusRequest_IEs, uint8_t **buf, uint32_t *length){

	int i;
	ResourceStatusRequest_t resourceStatusRequest;
	memset (&resourceStatusRequest,0, sizeof(ResourceStatusRequest_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding  Resource Status Resquest message	
	if (x2ap_encode_resourcestatusrequest_ies(&resourceStatusRequest, resourceStatusRequest_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_resourceStatusReportingInitiation
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_resourceStatusReportingInitiation, Criticality_reject, &asn_DEF_ResourceStatusRequest, &resourceStatusRequest) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_load_information_encoder(LoadInformation_IEs_t *loadInformation_IEs, uint8_t **buf, uint32_t *length){

	int i;
	LoadInformation_t loadInformation;
	memset (&loadInformation,0, sizeof(LoadInformation_t));

	asn1_xer_print = 0;
	asn_debug = 0;

	// encoding Reset request message	
	if (x2ap_encode_loadinformation_ies(&loadInformation, loadInformation_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_loadIndication, Criticality_ignore, &asn_DEF_LoadInformation, &loadInformation) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_handover_request_encoder(HandoverRequest_IEs_t *handoverRequest_IEs, uint8_t **buf, uint32_t *length){
	
	int i;
	HandoverRequest_t x2HandoverRequest;
	memset (&x2HandoverRequest,0, sizeof(HandoverRequest_t));
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_handoverrequest_ies(&x2HandoverRequest, handoverRequest_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_handoverPreparation, Criticality_reject, &asn_DEF_HandoverRequest, &x2HandoverRequest) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}	

static inline 
int x2ap_handover_request_acknowledge_encoder(HandoverRequestAcknowledge_IEs_t *handoverRequestAcknowledge_IEs, uint8_t **buf, uint32_t *length){

	int i;
	HandoverRequestAcknowledge_t x2HandoverRequestAcknowledge;
	memset (&x2HandoverRequestAcknowledge,0, sizeof(HandoverRequestAcknowledge_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_handoverrequestacknowledge_ies(&x2HandoverRequestAcknowledge, handoverRequestAcknowledge_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	
	if (x2ap_generate_successfull_outcome(buf, length, ProcedureCode_id_handoverPreparation, Criticality_reject, &asn_DEF_HandoverRequestAcknowledge, &x2HandoverRequestAcknowledge) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_handover_preparation_failure_encoder(HandoverPreparationFailure_IEs_t *handoverPreparationFailure_IEs, uint8_t **buf, uint32_t *length){
	int i;
	HandoverPreparationFailure_t x2HandoverPreparationFailure;
	memset (&x2HandoverPreparationFailure,0, sizeof(HandoverPreparationFailure_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_handoverpreparationfailure_ies(&x2HandoverPreparationFailure, handoverPreparationFailure_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_unsuccessfull_outcome(buf, length, ProcedureCode_id_handoverPreparation, Criticality_reject, &asn_DEF_HandoverPreparationFailure, &x2HandoverPreparationFailure) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_error_indication_encoder(ErrorIndication_IEs_t *errorIndication_IEs, uint8_t **buf, uint32_t *length){
	int i;
	ErrorIndication_t x2Error_indication;
	memset (&x2Error_indication,0, sizeof(ErrorIndication_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_errorindication_ies(&x2Error_indication, errorIndication_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message(buf, length, ProcedureCode_id_errorIndication, Criticality_ignore, &asn_DEF_ErrorIndication, &x2Error_indication) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_handover_cancel_encoder(HandoverCancel_IEs_t *handoverCancel_IEs, uint8_t **buf, uint32_t *length){
	int i;
	HandoverCancel_t x2HandoverCancel;
	memset (&x2HandoverCancel,0, sizeof(HandoverCancel_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_handovercancel_ies(&x2HandoverCancel, handoverCancel_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message(buf, length, ProcedureCode_id_handoverCancel, Criticality_ignore, &asn_DEF_HandoverCancel, &x2HandoverCancel) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	

}

static inline 
int x2ap_handover_report_encoder( HandoverReport_IEs_t *handoverReport_IEs, uint8_t **buf, uint32_t *length){
	int i;
	HandoverReport_t HandoverReport;
	memset (&HandoverReport,0, sizeof(HandoverReport_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_handoverreport_ies(&HandoverReport, handoverReport_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message(buf, length, ProcedureCode_id_handoverReport, Criticality_ignore, &asn_DEF_HandoverReport, &HandoverReport) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_eNB_configuration_update_request_encoder(ENBConfigurationUpdate_IEs_t *enbConfigurationUpdate_IEs, uint8_t **buf, uint32_t *length){
	int i;
	ENBConfigurationUpdate_t x2ENBConfigurationUpdate;
	memset (&x2ENBConfigurationUpdate,0, sizeof(ENBConfigurationUpdate_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_enbconfigurationupdate_ies(&x2ENBConfigurationUpdate, enbConfigurationUpdate_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_initiating_message(buf, length, ProcedureCode_id_eNBConfigurationUpdate, Criticality_reject, &asn_DEF_ENBConfigurationUpdate, &x2ENBConfigurationUpdate) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
	
}

static inline 
int x2ap_eNB_configuration_update_acknowledge_encoder(ENBConfigurationUpdateAcknowledge_IEs_t *enbConfigurationUpdateAcknowledge_IEs, uint8_t **buf, uint32_t *length){

	int i;

	ENBConfigurationUpdateAcknowledge_t x2ENBConfigurationUpdateAcknowledge;
	memset (&x2ENBConfigurationUpdateAcknowledge,0, sizeof(ENBConfigurationUpdateAcknowledge_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_enbconfigurationupdateacknowledge_ies(&x2ENBConfigurationUpdateAcknowledge, enbConfigurationUpdateAcknowledge_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_loadIndication
	if (x2ap_generate_successfull_outcome(buf, length, ProcedureCode_id_eNBConfigurationUpdate, Criticality_reject, &asn_DEF_ENBConfigurationUpdateAcknowledge, &x2ENBConfigurationUpdateAcknowledge) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");
}

static inline 
int x2ap_eNB_configuration_update_failure_encoder( ENBConfigurationUpdateFailure_IEs_t *enbConfigurationUpdateFailure_IEs, uint8_t **buf, uint32_t *length){
	int i;

	ENBConfigurationUpdateFailure_t x2ENBConfigurationUpdateFailure;
	memset (&x2ENBConfigurationUpdateFailure,0, sizeof(ENBConfigurationUpdateFailure_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Reset request message	
	if (x2ap_encode_enbconfigurationupdatefailure_ies(&x2ENBConfigurationUpdateFailure, enbConfigurationUpdateFailure_IEs) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	// encoding ProcedureCode_id_eNBConfigurationUpdate
	if (x2ap_generate_unsuccessfull_outcome(buf, length, ProcedureCode_id_eNBConfigurationUpdate, Criticality_reject, &asn_DEF_ENBConfigurationUpdateFailure, &x2ENBConfigurationUpdateFailure) <0) {
	printf ("Encode procedure failes\n");
	return -1;
	}
	for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	printf ("\n");

}

static inline 
int x2ap_cell_activation_failure_encoder(CellActivationFailure_IEs_t *cellActivationFailure_IEs,  uint8_t **buf, uint32_t *length){

int i;
CellActivationFailure_t cellActivationFailure;
	memset (&cellActivationFailure,0, sizeof(CellActivationFailure_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding cell activationfailure message
if (x2ap_encode_cellactivationfailure_ies(&cellActivationFailure,cellActivationFailure_IEs)<0) {

	printf ("Encode procedure failes\n");
    		return -1;
    	}
	// encoding ProcedureCode_id_cellActivation
	
    if (x2ap_generate_unsuccessfull_outcome (buf, length, ProcedureCode_id_cellActivation , Criticality_reject, &asn_DEF_CellActivationFailure, &cellActivationFailure) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
		for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
		printf ("\n");

}
    	

static inline 
int x2ap_cell_activation_response_encoder(CellActivationResponse_IEs_t *cellActivationResponse_IEs, uint8_t **buf, uint32_t *length) {
 
     int i;


	CellActivationResponse_t cellActivationResponse;;
	memset (&cellActivationResponse,0, sizeof(CellActivationResponse_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding Cell Activation Response message
	if (x2ap_encode_cellactivationresponse_ies(&cellActivationResponse,cellActivationResponse_IEs)<0) {

	printf ("Encode procedure failes\n");
    		return -1;
    	}
	// encoding ProcedureCode_id_cellActivation
    if (x2ap_generate_successfull_outcome (buf, length, ProcedureCode_id_cellActivation , Criticality_reject, &asn_DEF_CellActivationResponse, &cellActivationResponse) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
	    for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	    printf ("\n");
}   

static inline 
int x2ap_cell_activation_request_encoder(CellActivationRequest_IEs_t *cellActivationRequest_IEs,  uint8_t **buf, uint32_t *length)  {

int i;

	
	CellActivationRequest_t cellActivationRequest;
	memset (&cellActivationRequest,0, sizeof(CellActivationRequest_t));

	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding cell Activation Request message

if (x2ap_encode_cellactivationrequest_ies(&cellActivationRequest,cellActivationRequest_IEs)<0) {

	printf ("Encode procedure failes\n");
    		return -1;
    	}
	// encoding ProcedureCode_id_cellActivation
    if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_cellActivation , Criticality_reject, &asn_DEF_CellActivationRequest, &cellActivationRequest) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
	    for (i=0;i< *length; i++)
		{
		printf ("0x%02x ", (*buf)[i]);
		}
	    printf ("\n");
}   
    	
static inline 	
int x2ap_rlf_indication_encoder(RLFIndication_IEs_t *rlfIndication_IEs, uint8_t **buf, uint32_t *length) {
int i;

	
	RLFIndication_t rlfIndication;
	memset (&rlfIndication,0, sizeof(RLFIndication_t));
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding RLF Indication  message
	
if (x2ap_encode_rlfindication_ies(&rlfIndication,rlfIndication_IEs)<0) {

	printf ("Encode procedure failes\n");
    		return -1;
    
	}
	// encoding ProcedureCode_id_rlfIndication
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_rLFIndication , Criticality_reject, &asn_DEF_RLFIndication, &rlfIndication) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
    for (i=0;i< *length; i++)
        {
        printf ("0x%02x ", (*buf)[i]);
        }
    printf ("\n");
   
}    

static inline 
int x2ap_sn_status_transfer_encoder(SNStatusTransfer_IEs_t *snStatusTransfer_IEs, uint8_t **buf, uint32_t *length){

	int i;

	SNStatusTransfer_t snStatusTransfer;
	memset (&snStatusTransfer,0, sizeof(SNStatusTransfer_t));
	
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding sn status transfer message

	if (x2ap_encode_snstatustransfer_ies(&snStatusTransfer,snStatusTransfer_IEs)<0) {
	printf ("Encode procedure failes\n");
    	return -1;
    	}
	
	if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_snStatusTransfer , Criticality_reject, &asn_DEF_SNStatusTransfer, &snStatusTransfer) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
	for (i=0;i< *length; i++)
        	{
        	printf ("0x%02x ", (*buf)[i]);
        	}
	printf ("\n");
   
}   

static inline 
int x2ap_ue_context_release_encoder(UEContextRelease_IEs_t *uecontext_rel, uint8_t **buf, uint32_t *length){

int i;

	UEContextRelease_t ueContextRelease;
	memset (&ueContextRelease,0, sizeof(UEContextRelease_t));
	asn1_xer_print = 0;
	asn_debug = 0;
	// encoding ue context release message
if (x2ap_encode_uecontextrelease_ies(&ueContextRelease, uecontext_rel) <0)
	{
		printf ("Encode procedure failes\n");
    		return -1;
    	}
	
	//Procedure code for UE Context Release = "5"
    if (x2ap_generate_initiating_message (buf, length, ProcedureCode_id_uEContextRelease , Criticality_reject, &asn_DEF_UEContextRelease, &ueContextRelease) <0)
	{
	    	printf ("Initiating Message for Encode procedure failes\n");
    		return -1;
    	}
	//Printing Buff values on terminal
    for (i=0;i< *length; i++)
        {
        printf ("0x%02x ", (*buf)[i]);
        }
    printf ("\n");
       
}


