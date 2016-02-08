/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2015 Eurecom

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

/*! \file x2ap_eNB_decoder.c
 * \brief x2ap pdu decode procedures for eNB
 * \author Navid Nikaein
 * \date 2015- 2016
 * \version 0.1
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>

#include "x2ap_eNB_decoder.h"
#include "x2ap_common.h"
#include "x2ap_ies_defs.h"

#include "X2ap-ProcedureCode.h"

#include "intertask_interface.h"

#include "assertions.h"

int x2ap_eNB_decode_initiating(x2ap_message *x2ap_message_p, X2ap_InitiatingMessage_t *initiating_p);
int x2ap_eNB_decode_successful(x2ap_message *x2ap_message_p, X2ap_SuccessfulOutcome_t *successful_p);
int x2ap_eNB_decode_unsuccessful(x2ap_message *x2ap_message_p, X2ap_UnsuccessfulOutcome_t *unsuccessful_p);

int x2ap_eNB_decode_pdu(x2ap_message *x2ap_message_p, uint8_t *buffer, uint32_t len) {
  X2AP_PDU_t  pdu;
  X2AP_PDU_t *pdu_p = &pdu;
  asn_dec_rval_t dec_ret;
  
  DevAssert(buffer != NULL);
  
  memset((void *)pdu_p, 0, sizeof(X2AP_PDU_t));
  
  dec_ret = aper_decode(NULL,
			&asn_DEF_X2AP_PDU,
			(void**)&pdu_p,
			buffer,
			len,
			0,
			0);
  if (dec_ret.code != RC_OK){
    X2AP_ERROR("Failed to decode X2AP pdu\n");
    return -1;
  }
  
  x2ap_message_p->direction = pdu_p->present;
  asn1_xer_print = 1;
  asn_debug = 0;
  switch(pdu_p->present) {
  case X2AP_PDU_PR_initiatingMessage:
    return x2ap_eNB_decode_initiating(x2ap_message_p, &pdu_p->choice.initiatingMessage);
  
  case X2AP_PDU_PR_successfulOutcome:
    return x2ap_eNB_decode_successful(x2ap_message_p, &pdu_p->choice.successfulOutcome);
  
  case X2AP_PDU_PR_unsuccessfulOutcome:
    return x2ap_eNB_decode_unsuccessful(x2ap_message_p, &pdu_p->choice.unsuccessfulOutcome);
  
  default:
    X2AP_DEBUG("Unknown message outcome (%d) or not implemented", (int)pdu_p->present);
    break;
  }
  return -1;
}


int 
x2ap_eNB_decode_initiating(x2ap_message *x2ap_message_p, X2ap_InitiatingMessage_t *initiating_p) {
  
  int         ret = -1;
  MessageDef *message;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;

  DevAssert(initiating_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;

 
  x2ap_message_p->procedureCode = initiating_p->procedureCode;
  x2ap_message_p->criticality   = initiating_p->criticality;

  switch(x2ap_message_p->procedureCode) {
  case X2ap_ProcedureCode_id_x2Setup :
    ret = x2ap_decode_x2setuprequest_ies(&x2ap_message_p->msg.x2SetupRequest_IEs, &initiating_p->value);
    x2ap_xer_print_x2setuprequest_(x2ap_xer__print2sp,message_string,message);
    message_id          = X2AP_SETUP_REQUEST_LOG;
    message_string_size = strlen(message_string);
    message           = itti_alloc_new_message_sized(TASK_X2AP,
						       message_id,
						       message_string_size + sizeof (IttiMsgText));
    message->ittiMsg.x2ap_setup_request_log.size = message_string_size;
    memcpy(&message->ittiMsg.x2ap_setup_request_log.text, message_string, message_string_size);
    itti_send_msg_to_task(TASK_UNKNOWN, INSTANCE_DEFAULT, message);
    free(message_string);
    break;
  
  case  X2ap_ProcedureCode_id_reset:
    ret =  x2ap_decode_x2ap_resetrequest_ies(&x2ap_message_p->msg.x2ap_ResetRequest_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_resourceStatusReportingInitiation:
    ret =  x2ap_decode_x2ap_resourcestatusrequest_ies(&x2ap_message_p->msg.x2ap_ResourceStatusRequest_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_resourceStatusReporting:
    ret =  x2ap_decode_x2ap_resourcestatusupdate_ies(&x2ap_message_p->msg.x2ap_ResourceStatusUpdate_IEs, &initiating_p->value);
     break;
  case  X2ap_ProcedureCode_id_loadIndication:
    ret =  x2ap_decode_x2ap_loadinformation_ies(&x2ap_message_p->msg.x2ap_LoadInformation_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_mobilitySettingsChange:
    ret =  x2ap_decode_x2ap_mobilitychangerequest_ies(&x2ap_message_p->msg.x2ap_MobilityChangeRequest_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_eNBConfigurationUpdate:
    ret =  x2ap_decode_x2ap_enbconfigurationupdate_ies(&x2ap_message_p->msg.x2ap_ENBConfigurationUpdate_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_errorIndication:
    ret =  x2ap_decode_x2ap_errorindication_ies(&x2ap_message_p->msg.x2ap_ErrorIndication_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_handoverCancel:
    ret =  x2ap_decode_x2ap_handovercancel_ies(&x2ap_message_p->msg.x2ap_HandoverCancel_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_handoverPreparation:
    ret =  x2ap_decode_x2ap_handoverrequest_ies(&x2ap_message_p->msg.x2ap_HandoverRequest_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_uEContextRelease:
    ret =  x2ap_decode_x2ap_uecontextrelease_ies(&x2ap_message_p->msg.x2ap_UEContextRelease_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_snStatusTransfer:
    ret =  x2ap_decode_x2ap_snstatustransfer_ies(&x2ap_message_p->msg.x2ap_SNStatusTransfer_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_rLFIndication:
    ret =  x2ap_decode_x2ap_rlfindication_ies(&x2ap_message_p->msg.x2ap_RLFIndication_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_cellActivation:      
    ret =  x2ap_decode_x2ap_cellactivationrequest_ies(&x2ap_message_p->msg.x2ap_CellActivationRequest_IEs, &initiating_p->value);
    break;
  case  X2ap_ProcedureCode_id_handoverReport: 
    ret = x2ap_decode_x2ap_handoverreport_ies(&x2ap_message_p->msg.x2ap_HandoverReport_IEs, &initiating_p->value);
    break;
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  return ret; 
}

int 
x2ap_eNB_decode_successful(x2ap_message *x2ap_message_p, X2ap_SuccessfulOutcome_t *successful_p) {
 
  int         ret = -1;
  MessageDef *message_p;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;

  DevAssert(successful_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;

 
  x2ap_message_p->procedureCode = successful_p->procedureCode;
  x2ap_message_p->criticality   = successful_p->criticality;

  switch(x2ap_message_p->procedureCode) {
  case X2ap_ProcedureCode_id_x2Setup:
    ret = x2ap_decode_x2setupresponse_ies(&x2ap_message_p->msg.x2SetupResponse_IEs, &successful_p->value);

  case X2ap_ProcedureCode_id_reset:
    ret =  x2ap_decode_x2ap_resetresponse_ies(&x2ap_message_p->msg.x2ap_ResetResponse_IEs, &successful_p->value);

  case X2ap_ProcedureCode_id_resourceStatusReportingInitiation:
    ret =  x2ap_decode_x2ap_resourcestatusresponse_ies(&x2ap_message_p->msg.x2ap_ResourceStatusResponse_IEs, &successful_p->value);
    
  case X2ap_ProcedureCode_id_mobilitySettingsChange:
    ret = x2ap_decode_x2ap_mobilitychangeacknowledge_ies(&x2ap_message_p->msg.x2ap_MobilityChangeAcknowledge_IEs, &successful_p->value);
    
  case X2ap_ProcedureCode_id_eNBConfigurationUpdate:
    ret =  x2ap_decode_x2ap_enbconfigurationupdateacknowledge_ies(&x2ap_message_p->msg.x2ap_ENBConfigurationUpdateAcknowledge_IEs, &successful_p->value);
    
  case X2ap_ProcedureCode_id_handoverPreparation:
    ret = x2ap_decode_x2ap_handoverrequestacknowledge_ies(&x2ap_message_p->msg.x2ap_HandoverRequestAcknowledge_IEs, &successful_p->value);

  case X2ap_ProcedureCode_id_cellActivation:
    ret =  x2ap_decode_x2ap_cellactivationresponse_ies(&x2ap_message_p->msg.x2ap_CellActivationResponse_IEs, &successful_p->value);
    
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  return ret; 
}

int 
x2ap_eNB_decode_unsuccessful(x2ap_message *x2ap_message_p, X2ap_UnsuccessfulOutcome_t *unsuccessful_p) {

  int         ret = -1;
  MessageDef *message;
  char       *message_string = NULL;
  size_t      message_string_size;
  MessagesIds message_id;

  DevAssert(unsuccessful_p != NULL);

  message_string = calloc(10000, sizeof(char));

  x2ap_string_total_size = 0;

 
  x2ap_message_p->procedureCode = unsuccessful_p->procedureCode;
  x2ap_message_p->criticality   = unsuccessful_p->criticality;

  switch(x2ap_message_p->procedureCode) {
  case X2ap_ProcedureCode_id_x2Setup:
    ret =  x2ap_decode_x2setupfailure_ies(&x2ap_message_p->msg.x2SetupFailure_IEs, &unsuccessful_p->value);

  case X2ap_ProcedureCode_id_resourceStatusReportingInitiation:
    ret =  x2ap_decode_x2ap_resourcestatusfailure_ies(&x2ap_message_p->msg.x2ap_ResourceStatusFailure_IEs, &unsuccessful_p->value);
    
  case X2ap_ProcedureCode_id_mobilitySettingsChange:
    ret =  x2ap_decode_x2ap_mobilitychangefailure_ies(&x2ap_message_p->msg.x2ap_MobilityChangeFailure_IEs, &unsuccessful_p->value);
    
  case X2ap_ProcedureCode_id_eNBConfigurationUpdate:
    ret =  x2ap_decode_x2ap_enbconfigurationupdatefailure_ies(&x2ap_message_p->msg.x2ap_ENBConfigurationUpdateFailure_IEs, &unsuccessful_p->value);
    
  case X2ap_ProcedureCode_id_handoverPreparation:
    ret = x2ap_decode_x2ap_handoverpreparationfailure_ies(&x2ap_message_p->msg.x2ap_HandoverPreparationFailure_IEs, &unsuccessful_p->value);

  case X2ap_ProcedureCode_id_cellActivation:
    ret = x2ap_decode_x2ap_cellactivationfailure_ies(&x2ap_message_p->msg.x2ap_CellActivationFailure_IEs, &unsuccessful_p->value);			
  default:
    X2AP_DEBUG("Unknown procedure (%d) or not implemented", (int)x2ap_message_p->procedureCode);
    break;
  }
  return ret; 
}
