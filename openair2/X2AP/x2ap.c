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

  Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

*******************************************************************************/

/*! \file x2ap.c
 * \brief x2ap protocol
 * \author Navid Nikaein 
 * \date 2014 - 2015
 * \version 1.0
 * \company Eurecom
 * \email: navid.nikaein@eurecom.fr
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


#include "intertask_interface.h"

#include "x2ap.h"

#include "msc.h"


#include "assertions.h"
#include "conversions.h"

static
void x2ap_eNB_handle_register_eNB(instance_t instance, 
				  x2ap_register_enb_req_t *x2ap_register_eNB);

static 
void x2ap_eNB_register_eNB(x2ap_eNB_instance_t *instance_p,
			   net_ip_address_t    *target_eNB_ip_addr,
			   net_ip_address_t    *local_ip_addr, 
			   uint16_t             in_streams,
			   uint16_t             out_streams);
static
void x2ap_eNB_handle_sctp_association_resp(instance_t instance, 
					   sctp_new_association_resp_t *sctp_new_association_resp);


static 
int x2ap_eNB_generate_x2_setup_request(x2ap_eNB_instance_t *instance_p, 
				       x2ap_enb_data_t *x2ap_enb_data_p);

static 
int x2ap_eNB_generate_x2_setup_response(x2ap_eNB_instance_t *instance_p, 
				       x2ap_enb_data_t *x2ap_enb_data_p);

static 
int x2ap_eNB_generate_x2_setup_failure(x2ap_eNB_instance_t *instance_p, 
				       x2ap_enb_data_t *x2ap_enb_data_p);




static
void x2ap_eNB_handle_sctp_data_ind(instance_t instance, sctp_data_ind_t *sctp_data_ind) {

  int result;

  DevAssert(sctp_data_ind != NULL);
  
  x2ap_eNB_handle_message(sctp_data_ind->assoc_id, sctp_data_ind->stream,
                          sctp_data_ind->buffer, sctp_data_ind->buffer_length);
  
  result = itti_free(TASK_UNKNOWN, sctp_data_ind->buffer);
  AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);

}

static
void x2ap_eNB_handle_sctp_association_resp(instance_t instance, sctp_new_association_resp_t *sctp_new_association_resp)
{
  x2ap_eNB_instance_t *instance_p;
  x2ap_eNB_mme_data_t *x2ap_enb_data_p;

  DevAssert(sctp_new_association_resp != NULL);

  instance_p = x2ap_eNB_get_instance(instance);
  DevAssert(instance_p != NULL);

  x2ap_enb_data_p = x2ap_eNB_get_eNB(instance_p, -1,
                                     sctp_new_association_resp->ulp_cnx_id);
  DevAssert(x2ap_enb_data_p != NULL);

  if (sctp_new_association_resp->sctp_state != SCTP_STATE_ESTABLISHED) {
    S1AP_WARN("Received unsuccessful result for SCTP association (%u), instance %d, cnx_id %u\n",
              sctp_new_association_resp->sctp_state,
              instance,
              sctp_new_association_resp->ulp_cnx_id);

    x2ap_handle_x2_setup_message(x2ap_enb_data_p, sctp_new_association_resp->sctp_state == SCTP_STATE_SHUTDOWN);

    return;
  }

  /* Update parameters */
  x2ap_enb_data_p->assoc_id    = sctp_new_association_resp->assoc_id;
  x2ap_enb_data_p->in_streams  = sctp_new_association_resp->in_streams;
  x2ap_enb_data_p->out_streams = sctp_new_association_resp->out_streams;

  /* Prepare new x2 Setup Request */
  x2ap_eNB_generate_x2_setup_request(instance_p, x2ap_enb_data_p);
}

  
static void x2ap_eNB_register_eNB(x2ap_eNB_instance_t *instance_p,
                                  net_ip_address_t    *target_eNB_ip_address,
                                  net_ip_address_t    *local_ip_addr,
                                  uint16_t             in_streams,
                                  uint16_t             out_streams)
{

  MessageDef                 *message_p                   = NULL;
  sctp_new_association_req_t *sctp_new_association_req_p  = NULL;
  x2ap_eNB_data_t            *x2ap_enb_data_p             = NULL;

  DevAssert(instance_p != NULL);
  DevAssert(target_eNB_ip_address != NULL);

  message_p = itti_alloc_new_message(TASK_X2AP, SCTP_NEW_ASSOCIATION_REQ);

  sctp_new_association_req_p = &message_p->ittiMsg.sctp_new_association_req;

  sctp_new_association_req_p->port = X2AP_PORT_NUMBER;
  sctp_new_association_req_p->ppid = X2AP_SCTP_PPID;

  sctp_new_association_req_p->in_streams  = in_streams;
  sctp_new_association_req_p->out_streams = out_streams;

  memcpy(&sctp_new_association_req_p->remote_address,
         target_eNB_ip_address,
         sizeof(*target_eNB_ip_address));

  memcpy(&sctp_new_association_req_p->local_address,
         local_ip_addr,
         sizeof(*local_ip_addr));

  /* Create new MME descriptor */
  x2ap_enb_data_p = calloc(1, sizeof(*x2ap_enb_data_p));
  DevAssert(x2ap_enb_data_p != NULL);

  x2ap_enb_data_p->cnx_id                = x2ap_eNB_fetch_add_global_cnx_id();
  sctp_new_association_req_p->ulp_cnx_id = x2ap_enb_data_p->cnx_id;

  x2ap_enb_data_p->assoc_id          = -1;
  x2ap_enb_data_p->x2ap_eNB_instance = instance_p;

  /* Insert the new descriptor in list of known eNB
   * but not yet associated.
   */
  RB_INSERT(x2ap_enb_map, &instance_p->x2ap_enb_head, x2ap_enb_data_p);
  s1ap_mme_data_p->state = X2AP_ENB_STATE_WAITING;
  instance_p->x2ap_enb_nb ++;
  instance_p->x2ap_enb_pending_nb ++;

  itti_send_msg_to_task(TASK_SCTP, instance_p->instance, message_p);
}

static
void x2ap_eNB_handle_register_eNB(instance_t instance, x2ap_register_enb_req_t *x2ap_register_eNB)
{

  x2ap_eNB_instance_t *new_instance;
  uint8_t index;

  DevAssert(x2ap_register_eNB != NULL);

  /* Look if the provided instance already exists */
  new_instance = x2ap_eNB_get_instance(instance);

  if (new_instance != NULL) {
    /* Checks if it is a retry on the same eNB */
    DevCheck(new_instance->eNB_id == x2ap_register_eNB->eNB_id, new_instance->eNB_id, x2ap_register_eNB->eNB_id, 0);
    DevCheck(new_instance->cell_type == x2ap_register_eNB->cell_type, new_instance->cell_type, x2ap_register_eNB->cell_type, 0);
    DevCheck(new_instance->tac == x2ap_register_eNB->tac, new_instance->tac, x2ap_register_eNB->tac, 0);
    DevCheck(new_instance->mcc == x2ap_register_eNB->mcc, new_instance->mcc, x2ap_register_eNB->mcc, 0);
    DevCheck(new_instance->mnc == x2ap_register_eNB->mnc, new_instance->mnc, x2ap_register_eNB->mnc, 0);

  } 
  else {
    new_instance = calloc(1, sizeof(x2ap_eNB_instance_t));
    DevAssert(new_instance != NULL);

    RB_INIT(&new_instance->x2ap_enb_head);
    //RB_INIT(&new_instance->x2ap_ue_head);

    /* Copy usefull parameters */
    new_instance->instance         = instance;
    new_instance->eNB_name         = x2ap_register_eNB->eNB_name;
    new_instance->eNB_id           = x2ap_register_eNB->eNB_id;
    new_instance->cell_type        = x2ap_register_eNB->cell_type;
    new_instance->tac              = x2ap_register_eNB->tac;
    new_instance->mcc              = x2ap_register_eNB->mcc;
    new_instance->mnc              = x2ap_register_eNB->mnc;
    new_instance->mnc_digit_length = x2ap_register_eNB->mnc_digit_length;

    /* Add the new instance to the list of eNB (meaningfull in virtual mode) */
    x2ap_eNB_insert_new_instance(new_instance);

    X2AP_INFO("Registered new eNB[%d] and %s eNB id %u\n",
               instance,
               x2ap_register_eNB->cell_type == CELL_MACRO_ENB ? "macro" : "home",
               x2ap_register_eNB->eNB_id);
  }

  DevCheck(x2ap_register_eNB->nb_mme <= X2AP_MAX_NB_ENB_IP_ADDRESS,
           X2AP_MAX_NB_ENB_IP_ADDRESS, x2ap_register_eNB->nb_x2, 0);

  /* Trying to connect to the provided list of eNB ip address */
  for (index = 0; index < x2ap_register_eNB->nb_x2; index++) {
    x2ap_eNB_register_eNB(new_instance,
			  &x2ap_register_eNB->target_enb_x2_ip_address[index],
                          &x2ap_register_eNB->enb_x2_ip_address,
                          x2ap_register_eNB->sctp_in_streams,
                          x2ap_register_eNB->sctp_out_streams);
  }

}


void *x2ap_task(void *arg)
{
  MessageDef *received_msg = NULL;
  int         result;

  X2AP_DEBUG("Starting X2AP layer\n");

  x2ap_prepare_internal_data();

  itti_mark_task_ready(TASK_X2AP);

  while (1) {
    itti_receive_msg(TASK_X2AP, &received_msg);

    switch (ITTI_MSG_ID(received_msg)) {
    case TERMINATE_MESSAGE:
      itti_exit_task();
      break;

    case X2AP_REGISTER_ENB_REQ: 
      
      x2ap_eNB_handle_register_eNB(ITTI_MESSAGE_GET_INSTANCE(received_msg),
                                   &X2AP_REGISTER_ENB_REQ(received_msg));
    
      break;
 
    case SCTP_NEW_ASSOCIATION_RESP: 
      x2ap_eNB_handle_sctp_association_resp(ITTI_MESSAGE_GET_INSTANCE(received_msg),
					    &received_msg->ittiMsg.sctp_new_association_resp);
      
    case SCTP_DATA_IND: {
      x2ap_eNB_handle_sctp_data_ind(ITTI_MESSAGE_GET_INSTANCE(received_msg),
				    &received_msg->ittiMsg.sctp_data_ind);
    } 
      
    break;
      
    default:
      X2AP_ERROR("Received unhandled message: %d:%s\n",
                 ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
      break;
    }

    result = itti_free (ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);

    received_msg = NULL;
  }

  return NULL;
}


