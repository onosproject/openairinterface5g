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

/*! \file x2ap_eNB_handlers.c
 * \brief x2ap messages handlers for eNB part
 * \author Navid Nikaein
 * \date 2016
 * \version 0.1
 */

#include <stdint.h>

#include "intertask_interface.h"

#include "asn1_conversions.h"

#include "x2ap_common.h"
#include "x2ap_ies_defs.h"
#include "x2ap_eNB_defs.h"
#include "x2ap_eNB_handler.h"
#include "x2ap_eNB_decoder.h"

#include "x2ap_eNB_management_procedures.h"
#include "x2ap_eNB_generate_messages.h"

//#include "x2ap_eNB_default_values.h"

#include "assertions.h"
#include "conversions.h"
#include "msc.h"

static 
int x2ap_eNB_handle_x2_setup_request (uint32_t assoc_id,
				      uint32_t stream,
				      struct x2ap_message_s *message);
static
int x2ap_eNB_handle_x2_setup_response(uint32_t               assoc_id,
                                      uint32_t               stream,
                                      struct x2ap_message_s *message_p);
static
int x2ap_eNB_handle_x2_setup_failure(uint32_t               assoc_id,
                                     uint32_t               stream,
                                     struct x2ap_message_s *message_p);

static
int x2ap_eNB_handle_error_indication(uint32_t               assoc_id,
				     uint32_t               stream,
				     struct x2ap_message_s *message_p);

int x2ap_eNB_handle_initial_context_request(uint32_t               assoc_id,
					    uint32_t               stream,
					    struct x2ap_message_s *message_p);

static
int x2ap_eNB_handle_ue_context_release_command(uint32_t               assoc_id,
					       uint32_t               stream,
					       struct x2ap_message_s *message_p);

/* Handlers matrix. Only eNB related procedure present here */
// x2ap_messages_callback[message.procedureCode][message.direction]
x2ap_message_decoded_callback x2ap_messages_callback[][3] = {
  //  { x2ap_eNB_handle_handover_preparation, 0, 0 }, /* HandoverPreparation */
  { 0, 0, 0 }, /* HandoverPreparation */
  { 0, 0, 0 }, /* HandoverCancel */
  { 0, 0, 0 }, /* loadIndication */
  { x2ap_eNB_handle_error_indication, 0, 0 }, /* errorIndication */
  { 0, 0, 0 }, /* snStatusTransfer */
  { 0, 0, 0 }, /* uEContextRelease */
  { x2ap_eNB_handle_x2_setup_request, x2ap_eNB_handle_x2_setup_response, x2ap_eNB_handle_x2_setup_failure }, /* x2Setup */
  { 0, 0, 0 }, /* reset */
  { 0, 0, 0 }, /* eNBConfigurationUpdate */
  { 0, 0, 0 }, /* resourceStatusReportingInitiation */
  { 0, 0, 0 }, /* resourceStatusReporting */
  { 0, 0, 0 }, /* privateMessage */
  { 0, 0, 0 }, /* mobilitySettingsChange */
  { 0, 0, 0 }, /* rLFIndication */
  { 0, 0, 0 }, /* handoverReport */
  { 0, 0, 0 }  /* cellActivation */
  
};

static const char *x2ap_direction2String[] = {
  "", /* Nothing */
  "Originating message", /* originating message */
  "Successfull outcome", /* successfull outcome */
  "UnSuccessfull outcome", /* successfull outcome */
};

int x2ap_eNB_handle_message(uint32_t assoc_id, int32_t stream,
                            const uint8_t * const data, const uint32_t data_length)
{
  struct x2ap_message_s message;

  DevAssert(data != NULL);

  memset(&message, 0, sizeof(struct x2ap_message_s));

  if (x2ap_eNB_decode_pdu(&message, data, data_length) < 0) {
    X2AP_ERROR("Failed to decode X2AP PDU\n");
    return -1;
  }

  /* Checking procedure Code and direction of message */
  if (message.procedureCode > sizeof(x2ap_messages_callback) / (3 * sizeof(x2ap_message_decoded_callback))
      || (message.direction > X2AP_PDU_PR_unsuccessfulOutcome)) {
    X2AP_ERROR("[SCTP %d] Either procedureCode %d or direction %d exceed expected\n",
               assoc_id, message.procedureCode, message.direction);
    return -1;
  }
  
  /* No handler present.
   * This can mean not implemented or no procedure for eNB (wrong direction).
   */
  if (x2ap_messages_callback[message.procedureCode][message.direction-1] == NULL) {
    X2AP_ERROR("[SCTP %d] No handler for procedureCode %d in %s\n",
               assoc_id, message.procedureCode,
               x2ap_direction2String[message.direction]);
    return -1;
  }

  /* Calling the right handler */
  return (*x2ap_messages_callback[message.procedureCode][message.direction-1])
         (assoc_id, stream, &message);
}


void x2ap_handle_x2_setup_message(x2ap_eNB_data_t *enb_desc_p, int sctp_shutdown)
{
  if (sctp_shutdown) {
    /* A previously connected eNB has been shutdown */

    /* TODO check if it was used by some eNB and send a message to inform these eNB if there is no more associated MME */
    if (enb_desc_p->state == X2AP_ENB_STATE_CONNECTED) {
      enb_desc_p->state = X2AP_ENB_STATE_DISCONNECTED;

      if (enb_desc_p->x2ap_eNB_instance-> x2_target_enb_associated_nb > 0) {
        /* Decrease associated eNB number */
        enb_desc_p->x2ap_eNB_instance-> x2_target_enb_associated_nb --;
      }
      
      /* If there are no more associated eNB, inform eNB app */
      if (enb_desc_p->x2ap_eNB_instance->x2_target_enb_associated_nb == 0) {
        MessageDef                 *message_p;
	
        message_p = itti_alloc_new_message(TASK_X2AP, X2AP_DEREGISTERED_ENB_IND);
        X2AP_DEREGISTERED_ENB_IND(message_p).nb_x2 = 0;
        itti_send_msg_to_task(TASK_ENB_APP, enb_desc_p->x2ap_eNB_instance->instance, message_p);
      }
    }
  } else {
    /* Check that at least one setup message is pending */
    DevCheck(enb_desc_p->x2ap_eNB_instance->x2_target_enb_pending_nb > 0, 
	     enb_desc_p->x2ap_eNB_instance->instance,
             enb_desc_p->x2ap_eNB_instance->x2_target_enb_pending_nb, 0);

    if (enb_desc_p->x2ap_eNB_instance->x2_target_enb_pending_nb > 0) {
      /* Decrease pending messages number */
      enb_desc_p->x2ap_eNB_instance->x2_target_enb_pending_nb --;
    }
    
    /* If there are no more pending messages, inform eNB app */
    if (enb_desc_p->x2ap_eNB_instance->x2_target_enb_pending_nb == 0) {
      MessageDef                 *message_p;

      message_p = itti_alloc_new_message(TASK_X2AP, X2AP_REGISTER_ENB_CNF);
      X2AP_REGISTER_ENB_CNF(message_p).nb_x2 = enb_desc_p->x2ap_eNB_instance->x2_target_enb_associated_nb;
      itti_send_msg_to_task(TASK_ENB_APP, enb_desc_p->x2ap_eNB_instance->instance, message_p);
    }
  }
}


int
x2ap_eNB_handle_x2_setup_request (uint32_t assoc_id,
				  uint32_t stream,
				  struct x2ap_message_s *message)

{
  
  X2SetupRequest_IEs_t               *x2SetupRequest;
  x2ap_eNB_data_t                    *x2ap_eNB_data;
  uint32_t                           eNB_id = 0;
  int                                ta_ret;
  //uint16_t                                max_enb_connected;

  DevAssert (message != NULL);
  x2SetupRequest = &message->msg.x2SetupRequest_IEs;
    
  MSC_LOG_RX_MESSAGE (MSC_X2AP_TARGET_ENB, 
		      MSC_X2AP_SRC_ENB, NULL, 0, 
		      "0 X2Setup/%s assoc_id %u stream %u", 
		      x2ap_direction2String[message->direction], 
		      assoc_id, stream);
  /*
   * We received a new valid X2 Setup Request on a stream != 0.
   * * * * This should not happen -> reject eNB x2 setup request.
   */

  if (stream != 0) {
    X2AP_ERROR ("Received new x2 setup request on stream != 0\n");
      /*
       * Send a x2 setup failure with protocol cause unspecified
       */
    return x2ap_eNB_generate_x2_setup_failure (assoc_id, 
					       X2ap_Cause_PR_protocol, 
					       X2ap_CauseProtocol_unspecified, 
					       -1);
  }
  
  X2AP_DEBUG ("Received a new X2 setup request\n");
  
  if (x2SetupRequest->globalENB_ID.eNB_ID.present == X2ap_ENB_ID_PR_home_eNB_ID) {
    // Home eNB ID = 28 bits
    uint8_t  *eNB_id_buf = x2SetupRequest->globalENB_ID.eNB_ID.choice.home_eNB_ID.buf;
    
    if (x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.size != 28) {
      //TODO: handle case were size != 28 -> notify ? reject ?
    }
    
    eNB_id = (eNB_id_buf[0] << 20) + (eNB_id_buf[1] << 12) + (eNB_id_buf[2] << 4) + ((eNB_id_buf[3] & 0xf0) >> 4);
    X2AP_DEBUG ("Home eNB id: %07x\n", eNB_id);
  } else {
    // Macro eNB = 20 bits
    uint8_t *eNB_id_buf = x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf;
    
    if (x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.size != 20) {
      //TODO: handle case were size != 20 -> notify ? reject ?
    }
    
    eNB_id = (eNB_id_buf[0] << 12) + (eNB_id_buf[1] << 4) + ((eNB_id_buf[2] & 0xf0) >> 4);
    X2AP_DEBUG ("macro eNB id: %05x\n", eNB_id);
  }
  
  /*
   * If none of the provided PLMNs/TAC match the one configured in MME,
   * * * * the x2 setup should be rejected with a cause set to Unknown PLMN.
   */
  // ta_ret = x2ap_eNB_compare_ta_lists (&x2SetupRequest_p->supportedTAs);
  
  /*
   * Source and Target eNBs have no common PLMN
   */
  /*
  if (ta_ret != TA_LIST_RET_OK) {
    X2AP_ERROR ("No Common PLMN with the target eNB, generate_x2_setup_failure\n");
      return x2ap_eNB_generate_x2_setup_failure (assoc_id, 
						 X2ap_Cause_PR_misc, 
						 X2ap_CauseMisc_unknown_PLMN, 
						 X2ap_TimeToWait_v20s);
  }
  */
  X2AP_DEBUG ("Adding eNB to the list of associated eNBs\n");

  if ((x2ap_eNB_data = x2ap_is_eNB_id_in_list (eNB_id)) == NULL) {
      /*
       * eNB has not been fount in list of associated eNB,
       * * * * Add it to the tail of list and initialize data
       */
    if ((x2ap_eNB_data = x2ap_is_eNB_assoc_id_in_list (assoc_id)) == NULL) {
      /*
       * ??
       */
      return -1;
    } else {
      x2ap_eNB_data->state = X2AP_ENB_STATE_RESETTING;
      x2ap_eNB_data->eNB_id = eNB_id;
    } 


  } else {
    x2ap_eNB_data->state = X2AP_ENB_STATE_RESETTING;
    
    /*
     * eNB has been fount in list, consider the x2 setup request as a reset connection,
     * * * * reseting any previous UE state if sctp association is != than the previous one
     */
    if (x2ap_eNB_data->assoc_id != assoc_id) {
      X2SetupFailure_IEs_t                x2SetupFailure;
      
      memset (&x2SetupFailure, 0, sizeof (x2SetupFailure));
      /*
       * Send an overload cause...
       */
      X2AP_ERROR ("Rejeting x2 setup request as eNB id %d is already associated to an active sctp association" "Previous known: %d, new one: %d\n", eNB_id, x2ap_eNB_data->assoc_id, assoc_id);
      x2ap_eNB_generate_x2_setup_failure (assoc_id, 
					  X2ap_Cause_PR_protocol, 
					  X2ap_CauseProtocol_unspecified, 
					  -1);
      return -1;
    }
 
    /*
     * TODO: call the reset procedure
     */
  } 
  
  return x2ap_generate_x2_setup_response (x2ap_eNB_data);
  
}

static
int x2ap_eNB_handle_x2_setup_failure(uint32_t               assoc_id,
                                     uint32_t               stream,
                                     struct x2ap_message_s *message_p)
{

  X2SetupFailure_IEs_t   *x2_setup_failure;
  x2ap_eNB_data_t        *x2ap_eNB_data;

  DevAssert(message_p != NULL);

  x2_setup_failure = &message_p->msg.x2SetupFailure_IEs;

  /* x2 Setup Failure == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    X2AP_WARN("[SCTP %d] Received x2 setup failure on stream != 0 (%d)\n",
    assoc_id, stream);
  }

  if ((x2ap_eNB_data = x2ap_get_eNB (NULL, assoc_id, 0)) == NULL) {
    X2AP_ERROR("[SCTP %d] Received X2 setup response for non existing "
    "eNB context\n", assoc_id);
    return -1;
}
  
  // need a FSM to handle all cases 
  if ((x2_setup_failure->cause.present == X2ap_Cause_PR_misc) &&
      (x2_setup_failure->cause.choice.misc == X2ap_CauseMisc_unspecified)) {
    X2AP_WARN("Received X2 setup failure for eNB ... eNB is not ready\n");
  } else {
    X2AP_ERROR("Received x2 setup failure for eNB... please check your parameters\n");
  }

  x2ap_eNB_data->state = X2AP_ENB_STATE_WAITING;
  x2ap_handle_x2_setup_message(x2ap_eNB_data, 0);
 
  return 0;
}

static
int x2ap_eNB_handle_x2_setup_response(uint32_t               assoc_id,
                                      uint32_t               stream,
                                      struct x2ap_message_s *message)
{

  X2SetupResponse_IEs_t *x2SetupResponse;
  x2ap_eNB_data_t       *x2ap_eNB_data;
  uint32_t                           eNB_id = 0;
  
  DevAssert(message != NULL);
  x2SetupResponse = &message->msg.x2SetupResponse_IEs;
 
  MSC_LOG_RX_MESSAGE (MSC_X2AP_TARGET_ENB, 
		      MSC_X2AP_SRC_ENB, NULL, 0, 
		      "0 X2Setup/%s assoc_id %u stream %u", 
		      x2ap_direction2String[message->direction], 
		      assoc_id, stream);

  /* X2 Setup Response == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    X2AP_ERROR("[SCTP %d] Received X2 setup response on stream != 0 (%d)\n",
               assoc_id, stream);
    return -1;
  }

  if ((x2ap_eNB_data = x2ap_get_eNB(NULL, assoc_id, 0)) == NULL) {
    X2AP_ERROR("[SCTP %d] Received X2 setup response for non existing "
               "MME context\n", assoc_id);
    return -1;
  }
  
  /* check and store eNB info here*/ 
  if (x2SetupResponse->globalENB_ID.eNB_ID.present == X2ap_ENB_ID_PR_home_eNB_ID) {
    // Home eNB ID = 28 bits
    uint8_t  *eNB_id_buf = x2SetupResponse->globalENB_ID.eNB_ID.choice.home_eNB_ID.buf;
    
    if (x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.size != 28) {
      //TODO: handle case were size != 28 -> notify ? reject ?
    }
    
    eNB_id = (eNB_id_buf[0] << 20) + (eNB_id_buf[1] << 12) + (eNB_id_buf[2] << 4) + ((eNB_id_buf[3] & 0xf0) >> 4);
    X2AP_DEBUG ("Home eNB id: %07x\n", eNB_id);
  } else {
    // Macro eNB = 20 bits
    uint8_t *eNB_id_buf = x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf;
    
    if (x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.size != 20) {
      //TODO: handle case were size != 20 -> notify ? reject ?
    }
    
    eNB_id = (eNB_id_buf[0] << 12) + (eNB_id_buf[1] << 4) + ((eNB_id_buf[2] & 0xf0) >> 4);
    X2AP_DEBUG ("macro eNB id: %05x\n", eNB_id);
  }
  
  if ((x2ap_eNB_data = x2ap_is_eNB_id_in_list (eNB_id)) == NULL) {
      /*
       * eNB has not been fount in list of associated eNB,
       * * * * Add it to the tail of list and initialize data
       */
    if ((x2ap_eNB_data = x2ap_is_eNB_assoc_id_in_list (assoc_id)) == NULL) {
      /*
       * ??
       */
      return -1;
    } else {
      x2ap_eNB_data->state = X2AP_ENB_STATE_RESETTING;
      x2ap_eNB_data->eNB_id = eNB_id;
    } 


  } else {
    x2ap_eNB_data->state = X2AP_ENB_STATE_RESETTING;
    /* 
    if (x2ap_eNB_data->assoc_id != assoc_id) {
      X2SetupFailure_IEs_t                x2SetupFailure;
      
      memset (&x2SetupFailure, 0, sizeof (x2SetupFailure));
      // Send an overload cause
      X2AP_ERROR ("Rejeting x2 setup response as eNB id %d is already associated to an active sctp association" "Previous known: %d, new one: %d\n", eNB_id, x2ap_eNB_data->assoc_id, assoc_id);
      x2ap_eNB_generate_x2_setup_failure (assoc_id, 
					  X2ap_Cause_PR_protocol, 
					  X2ap_CauseProtocol_unspecified, 
					  -1);
      return -1;
    }
    */ 
    
    /*
     * TODO: call the reset procedure
     */
  }


  /* Optionaly set the target eNB name */

 

 /* The association is now ready as source and target eNBs know parameters of each other.
   * Mark the association as connected.
   */
  x2ap_eNB_data->state = X2AP_ENB_STATE_READY;
  x2ap_eNB_data->x2ap_eNB_instance->x2_target_enb_associated_nb ++;
  x2ap_handle_x2_setup_message(x2ap_eNB_data, 0);


  return 0;
}


static
int x2ap_eNB_handle_error_indication(uint32_t               assoc_id,
                                     uint32_t               stream,
                                     struct x2ap_message_s *message_p)
{

  X2ap_ErrorIndication_IEs_t  *x2_error_indication;
  x2ap_eNB_data_t        *x2ap_eNB_data;
  char       *message_string = NULL;

  DevAssert(message_p != NULL);

  x2_error_indication = &message_p->msg.x2ap_ErrorIndication_IEs;

  /* X2 Setup Failure == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    X2AP_WARN("[SCTP %d] Received X2 Error indication on stream != 0 (%d)\n",
              assoc_id, stream);
  }

  x2ap_xer_print_x2ap_errorindication_(x2ap_xer__print2sp,
				       message_string,
				       message_p);
  
  if ( x2_error_indication->presenceMask & X2AP_ERRORINDICATION_IES_OLD_ENB_UE_X2AP_ID_PRESENT ) {
    X2AP_WARN("Received X2 Error indication OLD ENB UE X2AP ID 0x%x\n", x2_error_indication->old_eNB_UE_X2AP_ID);
  }

  if ( x2_error_indication->presenceMask & X2AP_ERRORINDICATION_IES_NEW_ENB_UE_X2AP_ID_PRESENT) {
    X2AP_WARN("Received X2 Error indication NEW eNB UE X2AP ID 0x%x\n", x2_error_indication->new_eNB_UE_X2AP_ID);
  }

  if ( x2_error_indication->presenceMask & X2AP_ERRORINDICATION_IES_CAUSE_PRESENT) {
    switch(x2_error_indication->cause.present) {
      case X2ap_Cause_PR_NOTHING:
    	X2AP_WARN("Received X2 Error indication cause NOTHING\n");
      break;
      case X2ap_Cause_PR_radioNetwork:
      	switch (x2_error_indication->cause.choice.radioNetwork) {
	      case X2ap_CauseRadioNetwork_handover_desirable_for_radio_reasons:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_handover_desirable_for_radio_reasons\n");
            break;
  	      case X2ap_CauseRadioNetwork_time_critical_handover:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_time_critical_handover\n");
            break;
  	      case X2ap_CauseRadioNetwork_resource_optimisation_handover:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_resource_optimisation_handover\n");
            break;
  	      case X2ap_CauseRadioNetwork_reduce_load_in_serving_cell:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_reduce_load_in_serving_cell\n");
            break;
  	      case X2ap_CauseRadioNetwork_partial_handover:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_partial_handover\n");
            break;
  	      case X2ap_CauseRadioNetwork_unknown_new_eNB_UE_X2AP_ID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unknown_new_eNB_UE_X2AP_ID\n");
            break;
  	      case X2ap_CauseRadioNetwork_unknown_old_eNB_UE_X2AP_ID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unknown_old_eNB_UE_X2AP_ID\n");
            break;
  	      case X2ap_CauseRadioNetwork_unknown_pair_of_UE_X2AP_ID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unknown_pair_of_UE_X2AP_ID\n");
            break;
  	      case X2ap_CauseRadioNetwork_ho_target_not_allowed:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_ho_target_not_allowed\n");
            break;
  	      case X2ap_CauseRadioNetwork_tx2relocoverall_expiry:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_tx2relocoverall_expiry\n");
            break;
  	      case X2ap_CauseRadioNetwork_trelocprep_expiry:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_trelocprep_expiry\n");
            break;
  	      case X2ap_CauseRadioNetwork_cell_not_available:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_cell_not_available\n");
            break;
  	      case X2ap_CauseRadioNetwork_no_radio_resources_available_in_target_cell:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_no_radio_resources_available_in_target_cell\n");
            break;
  	      case X2ap_CauseRadioNetwork_invalid_MME_GroupID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_invalid_MME_GroupID\n");
            break;
  	      case X2ap_CauseRadioNetwork_unknown_MME_Code:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unknown_MME_Code\n");
            break;
  	      case X2ap_CauseRadioNetwork_encryption_and_or_integrity_protection_algorithms_not_supported:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_encryption_and_or_integrity_protection_algorithms_not_supported\n");
            break;
  	      case X2ap_CauseRadioNetwork_reportCharacteristicsEmpty:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_reportCharacteristicsEmpty\n");
            break;
  	      case X2ap_CauseRadioNetwork_noReportPeriodicity:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_noReportPeriodicity\n");
            break;
  	      case X2ap_CauseRadioNetwork_existingMeasurementID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_existingMeasurementID\n");
            break;
  	      case X2ap_CauseRadioNetwork_unknown_eNB_Measurement_ID:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unknown_eNB_Measurement_ID\n");
            break;
  	      case X2ap_CauseRadioNetwork_measurement_temporarily_not_available:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_measurement_temporarily_not_available\n");
            break;
  	      case X2ap_CauseRadioNetwork_unspecified:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_unspecified\n");
            break;
  	      case X2ap_CauseRadioNetwork_load_balancing:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_load_balancing\n");
            break;
  	      case X2ap_CauseRadioNetwork_handover_optimisation:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_handover_optimisation\n");
            break;
  	      case X2ap_CauseRadioNetwork_value_out_of_allowed_range:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_value_out_of_allowed_range\n");
            break;
  	      case X2ap_CauseRadioNetwork_multiple_E_RAB_ID_instances:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_multiple_E_RAB_ID_instances\n");
            break;
  	      case X2ap_CauseRadioNetwork_switch_off_ongoing:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_switch_off_ongoing\n");
            break;
  	      case X2ap_CauseRadioNetwork_not_supported_QCI_value:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_not_supported_QCI_value\n");
            break;
  	      case X2ap_CauseRadioNetwork_measurement_not_supported_for_the_object:
            X2AP_WARN("Received X2 Error indication X2ap_CauseRadioNetwork_measurement_not_supported_for_the_object\n");
	   
      	  default:
            X2AP_WARN("Received X2 Error indication cause radio network case not handled\n");
      	}
      break;

      case X2ap_Cause_PR_transport:
      	switch (x2_error_indication->cause.choice.transport) {
    	  case X2ap_CauseTransport_transport_resource_unavailable:
            X2AP_WARN("Received X2 Error indication X2ap_CauseTransport_transport_resource_unavailable\n");
            break;
    	  case X2ap_CauseTransport_unspecified:
            X2AP_WARN("Received X2 Error indication SX2ap_CauseTransport_unspecified\n");
            break;
      	  default:
            X2AP_WARN("Received X2 Error indication cause transport case not handled\n");
      	}
      break;

    case X2ap_Cause_PR_protocol:
      	switch (x2_error_indication->cause.choice.protocol) {
      	  case X2ap_CauseProtocol_transfer_syntax_error:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_transfer_syntax_error\n");
            break;
      	  case X2ap_CauseProtocol_abstract_syntax_error_reject:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_abstract_syntax_error_reject\n");
            break;
      	  case X2ap_CauseProtocol_abstract_syntax_error_ignore_and_notify:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_abstract_syntax_error_ignore_and_notify\n");
            break;
      	  case X2ap_CauseProtocol_message_not_compatible_with_receiver_state:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_message_not_compatible_with_receiver_state\n");
            break;
      	  case X2ap_CauseProtocol_semantic_error:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_semantic_error\n");
            break;
      	   case X2ap_CauseProtocol_unspecified:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_unspecified\n");
            break;
      	  case X2ap_CauseProtocol_abstract_syntax_error_falsely_constructed_message:
            X2AP_WARN("Received X2 Error indication X2ap_CauseProtocol_abstract_syntax_error_falsely_constructed_message\n");
            break;
      	 default:
            X2AP_WARN("Received X2 Error indication cause protocol case not handled\n");
      	}
      break;

      case X2ap_Cause_PR_misc:
        switch (x2_error_indication->cause.choice.protocol) {
          case X2ap_CauseMisc_control_processing_overload:
            X2AP_WARN("Received X2 Error indication X2ap_CauseMisc_control_processing_overload\n");
            break;
          case X2ap_CauseMisc_hardware_failure:
        	X2AP_WARN("Received X2 Error indication X2ap_CauseMisc_hardware_failure\n");
        	break;
          case X2ap_CauseMisc_om_intervention:
        	X2AP_WARN("Received X2 Error indication X2ap_CauseMisc_om_intervention\n");
        	break;
          case X2ap_CauseMisc_not_enough_user_plane_processing_resources:
        	X2AP_WARN("Received X2 Error indication X2ap_CauseMisc_not_enough_user_plane_processing_resources\n");
        	break;
          case X2ap_CauseMisc_unspecified:
        	X2AP_WARN("Received X2 Error indication X2ap_CauseMisc_unspecified\n");
        	break;
	default:
            X2AP_WARN("Received X2 Error indication cause misc case not handled\n");
        }
      break;
    }
  }
  if ( x2_error_indication->presenceMask &X2AP_ERRORINDICATION_IES_CRITICALITYDIAGNOSTICS_PRESENT) {
    // TODO continue
  }
  // TODO continue
  return 0;
}



