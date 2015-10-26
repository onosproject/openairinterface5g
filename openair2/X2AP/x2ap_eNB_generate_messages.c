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

/*! \file x2ap_eNB_generate_messages.c
 * \brief x2ap message generator
 * \author Navid Nikaein 
 * \date 2015 - 2016
 * \version 1.0
 * \company Eurecom
 * \email: navid.nikaein@eurecom.fr
 */

#include "intertask_interface.h"

#include "x2ap.h"

#include "msc.h"

#include "assertions.h"


static 
int x2ap_eNB_generate_x2_setup_request(x2ap_eNB_instance_t *instance_p, 
				       x2ap_enb_data_t *x2ap_enb_data_p){
 
  x2ap_message               message;

  X2SetupRequest_IEs_t       *x2SetupRequest_p;
  
  X2ap_PLMN_Identity_t       plmnIdentity;

  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_1;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_2;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_3;
  
  ServedCellItem_t *served_cell= malloc(sizeof(ServedCellItem_t));;
  
  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(x2ap_enb_data_p != NULL);

  memset(&message, 0, sizeof(x2ap_message));

  message.direction     = X2AP_PDU_PR_initiatingMessage;
  message.procedureCode = X2ap_ProcedureCode_id_X2Setup;
  message.criticality   = X2ap_Criticality_reject;

  x2SetupRequest_p = &message.msg.x2SetupRequest_IEs;
  memset((void *)&plmnIdentity, 0, sizeof(X2ap_PLMN_Identity_t));

  memset((void *)&broadcast_plmnIdentity_1, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_2, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_3, 0, sizeof(X2ap_PLMN_Identity_t));

  
  x2ap_enb_data_p->state = X2AP_ENB_STATE_WAITING;

  //----globalENB_ID------
  x2SetupRequest_p->global_ENB_ID.eNB_ID.present = X2ap_ENB_ID_PR_macro_eNB_ID;
  MACRO_ENB_ID_TO_BIT_STRING(instance_p->eNB_id,
                             &x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID);
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length,
                    &x2SetupRequest_p->global_ENB_ID.pLMNidentity);

  X2AP_INFO("%d -> %02x%02x%02x\n", instance_p->eNB_id, 
	    x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.buf[0], 
	    x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.buf[1],
            x2SetupRequest_p->global_ENB_ID.eNB_ID.choice.macroENB_ID.buf[2]);

  //----served cells------
#warning update the value of the message
  served_cell->servedCellInfo.pCI = 6;
  served_cell->servedCellInfo.eUTRA_Mode_Info.present = EUTRA_Mode_Info_PR_fDD; 
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_EARFCN = 3350; 
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_EARFCN = 3350;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_Transmission_Bandwidth = 0;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_Transmission_Bandwidth = 0;

  MCC_MNC_TO_PLMNID(instance_p->mcc,instance_p->mnc,&served_cell->servedCellInfo.cellId.pLMN_Identity);
  MCC_MNC_TO_PLMNID(0,0,&broadcast_plmnid_1);
  MCC_MNC_TO_PLMNID(0,0,&broadcast_plmnid_2);
  MCC_MNC_TO_PLMNID(0,0,&broadcast_plmnid_3);

  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnid_1);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnid_2);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnid_3);

  ECI_TO_BIT_STRING(instance_p->eNB_id, &served_cell->servedCellInfo.cellId.eUTRANcellIdentifier);
  TAC_TO_OCTET_STRING(instance_p->tac, &served_cell->servedCellInfo.tAC);
  ASN_SEQUENCE_ADD(&x2SetupRequest_p->servedCells.list, served_cell);

  if (x2ap_eNB_encode_pdu(&message, &buffer, &len) < 0) {
    X2AP_ERROR("Failed to encode X2 setup request\n");
    return -1;
  }

  /* Non UE-Associated signalling -> stream = 0 */
  x2ap_eNB_itti_send_sctp_data_req(instance_p->instance, x2ap_enb_data_p->assoc_id, buffer, len, 0);

  return ret;

}

int x2ap_eNB_generate_x2_setup_failure ( uint32_t assoc_id,
					 X2ap_Cause_PR cause_type,
					 long cause_value,
					 long time_to_waitx){ 
  
  uint8_t                                *buffer_p;
  uint32_t                                length;
  x2ap_message                            message;
  X2SetupFailure_IEs_t                    *x2_setup_failure_p;

  memset (&message, 0, sizeof (x2ap_message));
  x2_setup_failure_p = &message.msg.x2SetupFailure_IEs;
  message.procedureCode = X2ap_ProcedureCode_id_X2Setup;
  message.direction = X2AP_PDU_PR_unsuccessfulOutcome;
  x2ap_eNB_set_cause (&x2_setup_failure_p->cause, cause_type, cause_value);
  
  if (time_to_wait > -1) {
    x2_setup_failure_p->presenceMask |= X2SETUPFAILURE_IES_TIMETOWAIT_PRESENT;
    x2_setup_failure_p->timeToWait = time_to_wait;
  }
 
  if (x2ap_eNB_encode_pdu (&message, &buffer_p, &length) < 0) {
    X2AP_ERROR ("Failed to encode x2 setup failure\n");
    return -1;
  } 
  MSC_LOG_TX_MESSAGE (MSC_X2AP_SRC_ENB, 
		      MSC_X2AP_TARGET_ENB, NULL, 0, 
		      "0 X2Setup/unsuccessfulOutcome  assoc_id %u cause %u value %u", 
		      assoc_id, cause_type, cause_value);
  return x2ap_eNB_itti_send_sctp_request (buffer_p, length, assoc_id, 0);
}
























int x2ap_eB_set_cause (S1ap_Cause_t * cause_p,
		       S1ap_Cause_PR cause_type,
		       long cause_value)
{

  DevAssert (cause_p != NULL);
  cause_p->present = cause_type;

  switch (cause_type) {
  case X2ap_Cause_PR_radioNetwork:
    cause_p->choice.misc = cause_value;
    break;
    
  case X2ap_Cause_PR_transport:
    cause_p->choice.transport = cause_value;
    break;
    
  case X2ap_Cause_PR_protocol:
    cause_p->choice.protocol = cause_value;
    break;
    
  case X2ap_Cause_PR_misc:
    cause_p->choice.misc = cause_value;
    break;
    
  default:
    return -1;
  }
  
  return 0;
}
