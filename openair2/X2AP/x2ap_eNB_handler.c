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

/*! \file s1ap_eNB_handlers.c
 * \brief s1ap messages handlers for eNB part
 * \author Sebastien ROUX <sebastien.roux@eurecom.fr>
 * \date 2013
 * \version 0.1
 */

#include <stdint.h>

#include "intertask_interface.h"

#include "asn1_conversions.h"

#include "x2ap_common.h"
#include "x2ap_ies_defs.h"
// #include "s1ap_eNB.h"
#include "x2ap_eNB_defs.h"
#include "x2ap_eNB_handlers.h"
#include "x2ap_eNB_decoder.h"

#include "x2ap_eNB_management_procedures.h"

//#include "x2ap_eNB_default_values.h"

#include "assertions.h"
#include "conversions.h"
#include "msc.h"

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
					       struct s1ap_message_s *message_p);

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
  if (message.procedureCode > sizeof(messages_callback) / (3 * sizeof(x2ap_message_decoded_callback))
      || (message.direction > X2AP_PDU_PR_unsuccessfulOutcome)) {
    X2AP_ERROR("[SCTP %d] Either procedureCode %d or direction %d exceed expected\n",
               assoc_id, message.procedureCode, message.direction);
    return -1;
  }
  
  /* No handler present.
   * This can mean not implemented or no procedure for eNB (wrong direction).
   */
  if (x2ap_messages_callback[message.procedureCode][message.direction-1] == NULL) {
    S1AP_ERROR("[SCTP %d] No handler for procedureCode %d in %s\n",
               assoc_id, message.procedureCode,
               s1ap_direction2String[message.direction]);
    return -1;
  }

  /* Calling the right handler */
  return (*x2ap_messages_callback[message.procedureCode][message.direction-1])
         (assoc_id, stream, &message);
}


void x2ap_handle_x2_setup_message(x2ap_enb_data_t *enb_desc_p, int sctp_shutdown)
{
  if (sctp_shutdown) {
    /* A previously connected eNB has been shutdown */

    /* TODO check if it was used by some eNB and send a message to inform these eNB if there is no more associated MME */
    if (enb_desc_p->state == X2AP_ENB_STATE_CONNECTED) {
      enb_desc_p->state = X2AP_ENB_STATE_DISCONNECTED;

      if (enb_desc_p->x2ap_eNB_instance->x2ap_enb_associated_nb > 0) {
        /* Decrease associated eNB number */
        enb_desc_p->x2ap_eNB_instance->x2ap_enb_associated_nb --;
      }
      
      /* If there are no more associated MME, inform eNB app */
      if (enb_desc_p->x2ap_eNB_instance->x2ap_enb_associated_nb == 0) {
        MessageDef                 *message_p;
	
        message_p = itti_alloc_new_message(TASK_X2AP, X2AP_DEREGISTERED_ENB_IND);
        X2AP_DEREGISTERED_ENB_IND(message_p).nb_x2 = 0;
        itti_send_msg_to_task(TASK_ENB_APP, enb_desc_p->x2ap_eNB_instance->instance, message_p);
      }
    }
  } else {
    /* Check that at least one setup message is pending */
    DevCheck(enb_desc_p->x2ap_eNB_instance->x2ap_enb_pending_nb > 0, 
	     enb_desc_p->x2ap_eNB_instance->instance,
             enb_desc_p->x2ap_eNB_instance->x2ap_enb_pending_nb, 0);

    if (enb_desc_p->x2ap_eNB_instance->x2ap_enb_pending_nb > 0) {
      /* Decrease pending messages number */
      enb_desc_p->x2ap_eNB_instance->x2ap_enb_pending_nb --;
    }
    
    /* If there are no more pending messages, inform eNB app */
    if (enb_desc_p->x2ap_eNB_instance->x2ap_enb_pending_nb == 0) {
      MessageDef                 *message_p;

      message_p = itti_alloc_new_message(TASK_X2AP, X2AP_REGISTER_ENB_CNF);
      X2AP_REGISTER_ENB_CNF(message_p).nb_x2 = enb_desc_p->x2ap_eNB_instance->x2ap_enb_associated_nb;
      itti_send_msg_to_task(TASK_ENB_APP, enb_desc_p->x2ap_eNB_instance->instance, message_p);
    }
  }
}


int
x2ap_eNB_handle_x2_setup_request (uint32_t assoc_id,
				  uint32_t stream,
				  struct s1ap_message_s *message)
{
 
  X2SetupRequestIEs_t               *x2SetupRequest_p;
  eNB_description_t                      *eNB_association;
  uint32_t                                eNB_id = 0;
  char                                   *eNB_name = NULL;
  int                                     ta_ret;
  uint16_t                                max_enb_connected;

  DevAssert (message != NULL);
  x2SetupRequest_p = &message->msg.X2SetupRequestIEs;
    /*
     * We received a new valid X2 Setup Request on a stream != 0.
     * * * * This should not happen -> reject eNB s1 setup request.
     */
    MSC_LOG_RX_MESSAGE (MSC_X2AP_TARGET_ENB, MSC_X2AP_SRC_ENB, NULL, 0, "0 X2Setup/%s assoc_id %u stream %u", x2ap_direction2String[message->direction], assoc_id, stream);

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

    if (x2SetupRequest_p->global_ENB_ID.eNB_ID.present == X2ap_ENB_ID_PR_homeENB_ID) {
      // Home eNB ID = 28 bits
      uint8_t  *eNB_id_buf = x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.homeENB_ID.buf;

      if (x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.size != 28) {
        //TODO: handle case were size != 28 -> notify ? reject ?
      }
      
      eNB_id = (eNB_id_buf[0] << 20) + (eNB_id_buf[1] << 12) + (eNB_id_buf[2] << 4) + ((eNB_id_buf[3] & 0xf0) >> 4);
      X2AP_DEBUG ("eNB id: %07x\n", eNB_id);
    } else {
      // Macro eNB = 20 bits
      uint8_t                                *eNB_id_buf = s1SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.buf;

      if (s1SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.size != 20) {
        //TODO: handle case were size != 20 -> notify ? reject ?
      }

      eNB_id = (eNB_id_buf[0] << 12) + (eNB_id_buf[1] << 4) + ((eNB_id_buf[2] & 0xf0) >> 4);
      S1AP_DEBUG ("macro eNB id: %05x\n", eNB_id);
    }

    config_read_lock (&mme_config);
    max_enb_connected = mme_config.max_eNBs;
    config_unlock (&mme_config);

    if (nb_eNB_associated == max_enb_connected) {
      S1AP_ERROR ("There is too much eNB connected to MME, rejecting the association\n");
      S1AP_DEBUG ("Connected = %d, maximum allowed = %d\n", nb_eNB_associated, max_enb_connected);
      /*
       * Send an overload cause...
       */
      return s1ap_mme_generate_s1_setup_failure (assoc_id, S1ap_Cause_PR_misc, S1ap_CauseMisc_control_processing_overload, S1ap_TimeToWait_v20s);
    }

    /*
     * If none of the provided PLMNs/TAC match the one configured in MME,
     * * * * the s1 setup should be rejected with a cause set to Unknown PLMN.
     */
    ta_ret = s1ap_mme_compare_ta_lists (&s1SetupRequest_p->supportedTAs);

    /*
     * eNB and MME have no common PLMN
     */
    if (ta_ret != TA_LIST_RET_OK) {
      S1AP_ERROR ("No Common PLMN with eNB, generate_s1_setup_failure\n");
      return s1ap_mme_generate_s1_setup_failure (assoc_id, S1ap_Cause_PR_misc, S1ap_CauseMisc_unknown_PLMN, S1ap_TimeToWait_v20s);
    }

    S1AP_DEBUG ("Adding eNB to the list of served eNBs\n");

    if ((eNB_association = s1ap_is_eNB_id_in_list (eNB_id)) == NULL) {
      /*
       * eNB has not been fount in list of associated eNB,
       * * * * Add it to the tail of list and initialize data
       */
      if ((eNB_association = s1ap_is_eNB_assoc_id_in_list (assoc_id)) == NULL) {
        /*
         * ??
         */
        return -1;
      } else {
        eNB_association->s1_state = S1AP_RESETING;
        eNB_association->eNB_id = eNB_id;
        eNB_association->default_paging_drx = s1SetupRequest_p->defaultPagingDRX;

        if (eNB_name != NULL) {
          memcpy (eNB_association->eNB_name, s1SetupRequest_p->eNBname.buf, s1SetupRequest_p->eNBname.size);
          eNB_association->eNB_name[s1SetupRequest_p->eNBname.size] = '\0';
        }
      }
    } else {
      eNB_association->s1_state = S1AP_RESETING;

      /*
       * eNB has been fount in list, consider the s1 setup request as a reset connection,
       * * * * reseting any previous UE state if sctp association is != than the previous one
       */
      if (eNB_association->sctp_assoc_id != assoc_id) {
        S1ap_S1SetupFailureIEs_t                s1SetupFailure;

        memset (&s1SetupFailure, 0, sizeof (s1SetupFailure));
        /*
         * Send an overload cause...
         */
        s1SetupFailure.cause.present = S1ap_Cause_PR_misc;      //TODO: send the right cause
        s1SetupFailure.cause.choice.misc = S1ap_CauseMisc_control_processing_overload;
        S1AP_ERROR ("Rejeting s1 setup request as eNB id %d is already associated to an active sctp association" "Previous known: %d, new one: %d\n", eNB_id, eNB_association->sctp_assoc_id, assoc_id);
        //             s1ap_mme_encode_s1setupfailure(&s1SetupFailure,
        //                                            receivedMessage->msg.s1ap_sctp_new_msg_ind.assocId);
        return -1;
      }

      /*
       * TODO: call the reset procedure
       */
    }

    s1ap_dump_eNB (eNB_association);
    return s1ap_generate_s1_setup_response (eNB_association);
  } else {
    /*
     * Can not process the request, MME is not connected to HSS
     */
    S1AP_ERROR ("Rejecting s1 setup request Can not process the request, MME is not connected to HSS\n");
    return s1ap_mme_generate_s1_setup_failure (assoc_id, S1ap_Cause_PR_misc, S1ap_CauseMisc_unspecified, -1);
  }
}


static
int x2ap_eNB_handle_x2_setup_failure(uint32_t               assoc_id,
                                     uint32_t               stream,
                                     struct x2ap_message_s *message_p)
{
#ifdef 0 
  X2SetupFailureIEs_t   *s1_setup_failure_p;
  x2ap_eNB_data_t        *enb_desc_p;

  DevAssert(message_p != NULL);

  s1_setup_failure_p = &message_p->msg.s1ap_S1SetupFailureIEs;

  /* S1 Setup Failure == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    S1AP_WARN("[SCTP %d] Received s1 setup failure on stream != 0 (%d)\n",
              assoc_id, stream);
  }

  if ((mme_desc_p = s1ap_eNB_get_MME(NULL, assoc_id, 0)) == NULL) {
    S1AP_ERROR("[SCTP %d] Received S1 setup response for non existing "
               "MME context\n", assoc_id);
    return -1;
  }

  if ((s1_setup_failure_p->cause.present == S1ap_Cause_PR_misc) &&
      (s1_setup_failure_p->cause.choice.misc == S1ap_CauseMisc_unspecified)) {
    S1AP_WARN("Received s1 setup failure for MME... MME is not ready\n");
  } else {
    S1AP_ERROR("Received s1 setup failure for MME... please check your parameters\n");
  }

  mme_desc_p->state = S1AP_ENB_STATE_WAITING;
  s1ap_handle_s1_setup_message(mme_desc_p, 0);
#endif 
  return 0;
}

static
int x2ap_eNB_handle_x2_setup_response(uint32_t               assoc_id,
                                      uint32_t               stream,
                                      struct x2ap_message_s *message_p)
{
#ifdef 0 
  S1ap_S1SetupResponseIEs_t *s1SetupResponse_p;
  s1ap_eNB_mme_data_t       *mme_desc_p;
  int i;

  DevAssert(message_p != NULL);

  s1SetupResponse_p = &message_p->msg.s1ap_S1SetupResponseIEs;

  /* S1 Setup Response == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    S1AP_ERROR("[SCTP %d] Received s1 setup response on stream != 0 (%d)\n",
               assoc_id, stream);
    return -1;
  }

  if ((mme_desc_p = s1ap_eNB_get_MME(NULL, assoc_id, 0)) == NULL) {
    S1AP_ERROR("[SCTP %d] Received S1 setup response for non existing "
               "MME context\n", assoc_id);
    return -1;
  }

  /* The list of served gummei can contain at most 8 elements.
   * LTE related gummei is the first element in the list, i.e with an id of 0.
   */
  S1AP_DEBUG("servedGUMMEIs.list.count %d\n",s1SetupResponse_p->servedGUMMEIs.list.count); 
  DevAssert(s1SetupResponse_p->servedGUMMEIs.list.count > 0);
  DevAssert(s1SetupResponse_p->servedGUMMEIs.list.count <= 8);


  for (i = 0; i < s1SetupResponse_p->servedGUMMEIs.list.count; i++) {
    struct S1ap_ServedGUMMEIsItem *gummei_item_p;
    struct served_gummei_s        *new_gummei_p;
    int j;

    gummei_item_p = (struct S1ap_ServedGUMMEIsItem *)
                    s1SetupResponse_p->servedGUMMEIs.list.array[i];
    new_gummei_p = calloc(1, sizeof(struct served_gummei_s));

    STAILQ_INIT(&new_gummei_p->served_plmns);
    STAILQ_INIT(&new_gummei_p->served_group_ids);
    STAILQ_INIT(&new_gummei_p->mme_codes);
    
    S1AP_DEBUG("servedPLMNs.list.count %d\n",gummei_item_p->servedPLMNs.list.count);
    for (j = 0; j < gummei_item_p->servedPLMNs.list.count; j++) {
      S1ap_PLMNidentity_t *plmn_identity_p;
      struct plmn_identity_s *new_plmn_identity_p;
      
      plmn_identity_p = gummei_item_p->servedPLMNs.list.array[j];
      new_plmn_identity_p = calloc(1, sizeof(struct plmn_identity_s));
      TBCD_TO_MCC_MNC(plmn_identity_p, new_plmn_identity_p->mcc,
                      new_plmn_identity_p->mnc, new_plmn_identity_p->mnc_digit_length);
      STAILQ_INSERT_TAIL(&new_gummei_p->served_plmns, new_plmn_identity_p, next);
      new_gummei_p->nb_served_plmns++;
    }

    for (j = 0; j < gummei_item_p->servedGroupIDs.list.count; j++) {
      S1ap_MME_Group_ID_t           *mme_group_id_p;
      struct served_group_id_s *new_group_id_p;

      mme_group_id_p = gummei_item_p->servedGroupIDs.list.array[j];
      new_group_id_p = calloc(1, sizeof(struct served_group_id_s));
      OCTET_STRING_TO_INT16(mme_group_id_p, new_group_id_p->mme_group_id);
      STAILQ_INSERT_TAIL(&new_gummei_p->served_group_ids, new_group_id_p, next);
      new_gummei_p->nb_group_id++;
    }

    for (j = 0; j < gummei_item_p->servedMMECs.list.count; j++) {
      S1ap_MME_Code_t        *mme_code_p;
      struct mme_code_s *new_mme_code_p;

      mme_code_p = gummei_item_p->servedMMECs.list.array[j];
      new_mme_code_p = calloc(1, sizeof(struct mme_code_s));

      OCTET_STRING_TO_INT8(mme_code_p, new_mme_code_p->mme_code);
      STAILQ_INSERT_TAIL(&new_gummei_p->mme_codes, new_mme_code_p, next);
      new_gummei_p->nb_mme_code++;
    }

    STAILQ_INSERT_TAIL(&mme_desc_p->served_gummei, new_gummei_p, next);
  }

  /* Free contents of the list */
  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_S1ap_ServedGUMMEIs,
                                (void *)&s1SetupResponse_p->servedGUMMEIs);
  /* Set the capacity of this MME */
  mme_desc_p->relative_mme_capacity = s1SetupResponse_p->relativeMMECapacity;

  /* Optionaly set the mme name */
  if (s1SetupResponse_p->presenceMask & S1AP_S1SETUPRESPONSEIES_MMENAME_PRESENT) {
    mme_desc_p->mme_name = calloc(s1SetupResponse_p->mmEname.size + 1, sizeof(char));
    memcpy(mme_desc_p->mme_name, s1SetupResponse_p->mmEname.buf,
           s1SetupResponse_p->mmEname.size);
    /* Convert the mme name to a printable string */
    mme_desc_p->mme_name[s1SetupResponse_p->mmEname.size] = '\0';
  }

  /* The association is now ready as eNB and MME know parameters of each other.
   * Mark the association as UP to enable UE contexts creation.
   */
  mme_desc_p->state = S1AP_ENB_STATE_CONNECTED;
  mme_desc_p->s1ap_eNB_instance->s1ap_mme_associated_nb ++;
  s1ap_handle_s1_setup_message(mme_desc_p, 0);

#endif

  return 0;
}


static
int x2ap_eNB_handle_error_indication(uint32_t               assoc_id,
                                     uint32_t               stream,
                                     struct x2ap_message_s *message_p)
{


#ifdef 0
  X2ap_ErrorIndicationIEs_t   *x2_error_indication_p;
  x2ap_eNB_data_t        *enb_desc_p;

  DevAssert(message_p != NULL);

  s1_error_indication_p = &message_p->msg.s1ap_ErrorIndicationIEs;

  /* S1 Setup Failure == Non UE-related procedure -> stream 0 */
  if (stream != 0) {
    S1AP_WARN("[SCTP %d] Received s1 Error indication on stream != 0 (%d)\n",
              assoc_id, stream);
  }

  if ((mme_desc_p = s1ap_eNB_get_MME(NULL, assoc_id, 0)) == NULL) {
    S1AP_ERROR("[SCTP %d] Received S1 Error indication for non existing "
               "MME context\n", assoc_id);
    return -1;
  }
  if ( s1_error_indication_p->presenceMask & S1AP_ERRORINDICATIONIES_MME_UE_S1AP_ID_PRESENT) {
	  	S1AP_WARN("Received S1 Error indication MME UE S1AP ID 0x%x\n", s1_error_indication_p->mme_ue_s1ap_id);
  }
  if ( s1_error_indication_p->presenceMask & S1AP_ERRORINDICATIONIES_ENB_UE_S1AP_ID_PRESENT) {
  	S1AP_WARN("Received S1 Error indication eNB UE S1AP ID 0x%x\n", s1_error_indication_p->eNB_UE_S1AP_ID);
  }

  if ( s1_error_indication_p->presenceMask & S1AP_ERRORINDICATIONIES_CAUSE_PRESENT) {
    switch(s1_error_indication_p->cause.present) {
      case S1ap_Cause_PR_NOTHING:
    	S1AP_WARN("Received S1 Error indication cause NOTHING\n");
      break;
      case S1ap_Cause_PR_radioNetwork:
      	switch (s1_error_indication_p->cause.choice.radioNetwork) {
	      case S1ap_CauseRadioNetwork_unspecified:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unspecified\n");
            break;
  	      case S1ap_CauseRadioNetwork_tx2relocoverall_expiry:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_tx2relocoverall_expiry\n");
            break;
  	      case S1ap_CauseRadioNetwork_successful_handover:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_successful_handover\n");
            break;
  	      case S1ap_CauseRadioNetwork_release_due_to_eutran_generated_reason:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_release_due_to_eutran_generated_reason\n");
            break;
  	      case S1ap_CauseRadioNetwork_handover_cancelled:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_handover_cancelled\n");
            break;
  	      case S1ap_CauseRadioNetwork_partial_handover:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_partial_handover\n");
            break;
  	      case S1ap_CauseRadioNetwork_ho_failure_in_target_EPC_eNB_or_target_system:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_ho_failure_in_target_EPC_eNB_or_target_system\n");
            break;
  	      case S1ap_CauseRadioNetwork_ho_target_not_allowed:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_ho_target_not_allowed\n");
            break;
  	      case S1ap_CauseRadioNetwork_tS1relocoverall_expiry:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_tS1relocoverall_expiry\n");
            break;
  	      case S1ap_CauseRadioNetwork_tS1relocprep_expiry:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_tS1relocprep_expiry\n");
            break;
  	      case S1ap_CauseRadioNetwork_cell_not_available:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_cell_not_available\n");
            break;
  	      case S1ap_CauseRadioNetwork_unknown_targetID:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unknown_targetID\n");
            break;
  	      case S1ap_CauseRadioNetwork_no_radio_resources_available_in_target_cell:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_no_radio_resources_available_in_target_cell\n");
            break;
  	      case S1ap_CauseRadioNetwork_unknown_mme_ue_s1ap_id:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unknown_mme_ue_s1ap_id\n");
            break;
  	      case S1ap_CauseRadioNetwork_unknown_enb_ue_s1ap_id:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unknown_enb_ue_s1ap_id\n");
            break;
  	      case S1ap_CauseRadioNetwork_unknown_pair_ue_s1ap_id:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unknown_pair_ue_s1ap_id\n");
            break;
  	      case S1ap_CauseRadioNetwork_handover_desirable_for_radio_reason:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_handover_desirable_for_radio_reason\n");
            break;
  	      case S1ap_CauseRadioNetwork_time_critical_handover:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_time_critical_handover\n");
            break;
  	      case S1ap_CauseRadioNetwork_resource_optimisation_handover:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_resource_optimisation_handover\n");
            break;
  	      case S1ap_CauseRadioNetwork_reduce_load_in_serving_cell:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_reduce_load_in_serving_cell\n");
            break;
  	      case S1ap_CauseRadioNetwork_user_inactivity:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_user_inactivity\n");
            break;
  	      case S1ap_CauseRadioNetwork_radio_connection_with_ue_lost:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_radio_connection_with_ue_lost\n");
            break;
  	      case S1ap_CauseRadioNetwork_load_balancing_tau_required:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_load_balancing_tau_required\n");
            break;
  	      case S1ap_CauseRadioNetwork_cs_fallback_triggered:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_cs_fallback_triggered\n");
            break;
  	      case S1ap_CauseRadioNetwork_ue_not_available_for_ps_service:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_ue_not_available_for_ps_service\n");
            break;
  	      case S1ap_CauseRadioNetwork_radio_resources_not_available:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_radio_resources_not_available\n");
            break;
  	      case S1ap_CauseRadioNetwork_failure_in_radio_interface_procedure:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_failure_in_radio_interface_procedure\n");
            break;
  	      case S1ap_CauseRadioNetwork_invals1ap_id_qos_combination:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_invals1ap_id_qos_combination\n");
            break;
  	      case S1ap_CauseRadioNetwork_interrat_redirection:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_interrat_redirection\n");
            break;
  	      case S1ap_CauseRadioNetwork_interaction_with_other_procedure:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_interaction_with_other_procedure\n");
            break;
  	      case S1ap_CauseRadioNetwork_unknown_E_RAB_ID:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_unknown_E_RAB_ID\n");
            break;
  	      case S1ap_CauseRadioNetwork_multiple_E_RAB_ID_instances:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_multiple_E_RAB_ID_instances\n");
            break;
  	      case S1ap_CauseRadioNetwork_encryption_and_or_integrity_protection_algorithms_not_supported:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_encryption_and_or_integrity_protection_algorithms_not_supported\n");
            break;
  	      case S1ap_CauseRadioNetwork_s1_intra_system_handover_triggered:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_s1_intra_system_handover_triggered\n");
            break;
  	      case S1ap_CauseRadioNetwork_s1_inter_system_handover_triggered:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_s1_inter_system_handover_triggered\n");
            break;
  	      case S1ap_CauseRadioNetwork_x2_handover_triggered:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_x2_handover_triggered\n");
            break;
  	      case S1ap_CauseRadioNetwork_redirection_towards_1xRTT:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_redirection_towards_1xRTT\n");
            break;
  	      case S1ap_CauseRadioNetwork_not_supported_QCI_value:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_not_supported_QCI_value\n");
            break;
  	      case S1ap_CauseRadioNetwork_invals1ap_id_CSG_Id:
            S1AP_WARN("Received S1 Error indication S1ap_CauseRadioNetwork_invals1ap_id_CSG_Id\n");
            break;
      	  default:
            S1AP_WARN("Received S1 Error indication cause radio network case not handled\n");
      	}
      break;

      case S1ap_Cause_PR_transport:
      	switch (s1_error_indication_p->cause.choice.transport) {
    	  case S1ap_CauseTransport_transport_resource_unavailable:
            S1AP_WARN("Received S1 Error indication S1ap_CauseTransport_transport_resource_unavailable\n");
            break;
    	  case S1ap_CauseTransport_unspecified:
            S1AP_WARN("Received S1 Error indication S1ap_CauseTransport_unspecified\n");
            break;
      	  default:
            S1AP_WARN("Received S1 Error indication cause transport case not handled\n");
      	}
      break;

      case S1ap_Cause_PR_nas:
      	switch (s1_error_indication_p->cause.choice.nas) {
    	  case S1ap_CauseNas_normal_release:
            S1AP_WARN("Received S1 Error indication S1ap_CauseNas_normal_release\n");
            break;
      	  case S1ap_CauseNas_authentication_failure:
            S1AP_WARN("Received S1 Error indication S1ap_CauseNas_authentication_failure\n");
            break;
      	  case S1ap_CauseNas_detach:
            S1AP_WARN("Received S1 Error indication S1ap_CauseNas_detach\n");
            break;
      	  case S1ap_CauseNas_unspecified:
            S1AP_WARN("Received S1 Error indication S1ap_CauseNas_unspecified\n");
            break;
      	  case S1ap_CauseNas_csg_subscription_expiry:
            S1AP_WARN("Received S1 Error indication S1ap_CauseNas_csg_subscription_expiry\n");
            break;
      	  default:
            S1AP_WARN("Received S1 Error indication cause nas case not handled\n");
      	}
      break;

      case S1ap_Cause_PR_protocol:
      	switch (s1_error_indication_p->cause.choice.protocol) {
      	  case S1ap_CauseProtocol_transfer_syntax_error:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_transfer_syntax_error\n");
            break;
      	  case S1ap_CauseProtocol_abstract_syntax_error_reject:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_abstract_syntax_error_reject\n");
            break;
      	  case S1ap_CauseProtocol_abstract_syntax_error_ignore_and_notify:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_abstract_syntax_error_ignore_and_notify\n");
            break;
      	  case S1ap_CauseProtocol_message_not_compatible_with_receiver_state:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_message_not_compatible_with_receiver_state\n");
            break;
      	  case S1ap_CauseProtocol_semantic_error:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_semantic_error\n");
            break;
      	  case S1ap_CauseProtocol_abstract_syntax_error_falsely_constructed_message:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_abstract_syntax_error_falsely_constructed_message\n");
            break;
      	  case S1ap_CauseProtocol_unspecified:
            S1AP_WARN("Received S1 Error indication S1ap_CauseProtocol_unspecified\n");
            break;
      	  default:
            S1AP_WARN("Received S1 Error indication cause protocol case not handled\n");
      	}
      break;

      case S1ap_Cause_PR_misc:
        switch (s1_error_indication_p->cause.choice.protocol) {
          case S1ap_CauseMisc_control_processing_overload:
            S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_control_processing_overload\n");
            break;
          case S1ap_CauseMisc_not_enough_user_plane_processing_resources:
        	S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_not_enough_user_plane_processing_resources\n");
        	break;
          case S1ap_CauseMisc_hardware_failure:
        	S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_hardware_failure\n");
        	break;
          case S1ap_CauseMisc_om_intervention:
        	S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_om_intervention\n");
        	break;
          case S1ap_CauseMisc_unspecified:
        	S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_unspecified\n");
        	break;
          case S1ap_CauseMisc_unknown_PLMN:
        	S1AP_WARN("Received S1 Error indication S1ap_CauseMisc_unknown_PLMN\n");
        	break;
          default:
            S1AP_WARN("Received S1 Error indication cause misc case not handled\n");
        }
      break;
    }
  }
  if ( s1_error_indication_p->presenceMask & S1AP_ERRORINDICATIONIES_CRITICALITYDIAGNOSTICS_PRESENT) {
    // TODO continue
  }
  // TODO continue
#endif 
  return 0;
}

