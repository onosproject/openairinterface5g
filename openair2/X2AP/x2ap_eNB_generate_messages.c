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

#include "x2ap_eNB.h"
#include "x2ap_eNB_generate_messages.h"
#include "x2ap_eNB_encoder.h"
#include "x2ap_eNB_itti_messaging.h"

#include "msc.h"
#include "assertions.h"
#include "conversions.h"


int x2ap_eNB_generate_x2_setup_request(x2ap_eNB_instance_t *instance_p,
				       x2ap_eNB_data_t *x2ap_enb_data_p)
{
  x2ap_message               message;

  X2SetupRequest_IEs_t       *x2SetupRequest;

  X2ap_PLMN_Identity_t       plmnIdentity;

  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_1;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_2;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_3;

  X2ap_ServedCellItem_t *served_cell= malloc(sizeof(X2ap_ServedCellItem_t));

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(x2ap_enb_data_p != NULL);

  memset(&message, 0, sizeof(x2ap_message));

  message.direction     = X2AP_PDU_PR_initiatingMessage;
  message.procedureCode = X2ap_ProcedureCode_id_x2Setup;
  message.criticality   = X2ap_Criticality_reject;

  x2SetupRequest = &message.msg.x2SetupRequest_IEs;
  memset((void *)&plmnIdentity, 0, sizeof(X2ap_PLMN_Identity_t));

  memset((void *)&broadcast_plmnIdentity_1, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_2, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_3, 0, sizeof(X2ap_PLMN_Identity_t));


  x2ap_enb_data_p->state = X2AP_ENB_STATE_WAITING;

  //----globalENB_ID------
  x2SetupRequest->globalENB_ID.eNB_ID.present = X2ap_ENB_ID_PR_macro_eNB_ID ;
  MACRO_ENB_ID_TO_BIT_STRING(instance_p->eNB_id,
                             &x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID);
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length,
                    &x2SetupRequest->globalENB_ID.pLMN_Identity);

  X2AP_INFO("%d -> %02x%02x%02x\n", instance_p->eNB_id,
	    x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[0],
	    x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[1],
            x2SetupRequest->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[2]);

 //----served cells------
#warning update the value of the message
  served_cell->servedCellInfo.pCI = 6;
  served_cell->servedCellInfo.eUTRA_Mode_Info.present = X2ap_EUTRA_Mode_Info_PR_fDD;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_EARFCN = 3350;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_EARFCN = 3350;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_Transmission_Bandwidth = 0;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_Transmission_Bandwidth = 0;

  MCC_MNC_TO_PLMNID(instance_p->mcc,instance_p->mnc,instance_p->mnc_digit_length,
		    &served_cell->servedCellInfo.cellId.pLMN_Identity);
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc,instance_p->mnc_digit_length,&broadcast_plmnIdentity_1);
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc,instance_p->mnc_digit_length,&broadcast_plmnIdentity_2);
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc,instance_p->mnc_digit_length,&broadcast_plmnIdentity_3);

  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_1);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_2);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_3);

  ECI_TO_BIT_STRING(instance_p->eNB_id, &served_cell->servedCellInfo.cellId.eUTRANcellIdentifier);
  TAC_TO_OCTET_STRING(instance_p->tac, &served_cell->servedCellInfo.tAC);
  ASN_SEQUENCE_ADD(&x2SetupRequest->servedCells.list, served_cell);

  if (x2ap_eNB_encode_pdu(&message, &buffer, &len) < 0) {
    X2AP_ERROR("Failed to encode X2 setup request\n");
    return -1;
  }

  MSC_LOG_TX_MESSAGE (MSC_X2AP_SRC_ENB, MSC_X2AP_TARGET_ENB, NULL, 0, "0 X2Setup/initiatingMessage assoc_id %u", x2ap_enb_data_p->assoc_id);

  /* Non UE-Associated signalling -> stream = 0 */
  x2ap_eNB_itti_send_sctp_data_req(instance_p->instance, x2ap_enb_data_p->assoc_id, buffer, len, 0);

  return ret;

}

int
x2ap_generate_x2_setup_response (x2ap_eNB_data_t * eNB_association)
{

  x2ap_eNB_instance_t      *instance=eNB_association->x2ap_eNB_instance;

  x2ap_message              message;

  X2SetupResponse_IEs_t    *x2SetupResponse;

  //  X2ap_PLMN_Identity_t       plmnIdentity;

  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_1;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_2;
  X2ap_PLMN_Identity_t       broadcast_plmnIdentity_3;

  X2ap_ServedCellItem_t *served_cell = calloc(1, sizeof(X2ap_ServedCellItem_t));;

  uint8_t                                *buffer;
  uint32_t                                len;
  int                                      ret = 0;
  // get the eNB instance
  //
  DevAssert (eNB_association != NULL);


  // Generating response
  memset (&message, 0, sizeof (x2ap_message));
  message.direction     = X2AP_PDU_PR_successfulOutcome;
  message.procedureCode = X2ap_ProcedureCode_id_x2Setup;
  message.criticality   = X2ap_Criticality_reject;

  x2SetupResponse = &message.msg.x2SetupResponse_IEs;

  //  memset((void *)&plmnIdentity, 0, sizeof(X2ap_PLMN_Identity_t));

  memset((void *)&broadcast_plmnIdentity_1, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_2, 0, sizeof(X2ap_PLMN_Identity_t));
  memset((void *)&broadcast_plmnIdentity_3, 0, sizeof(X2ap_PLMN_Identity_t));

  //----globalENB_ID------
  x2SetupResponse->globalENB_ID.eNB_ID.present = X2ap_ENB_ID_PR_macro_eNB_ID;
  MACRO_ENB_ID_TO_BIT_STRING(instance->eNB_id,
                             &x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID);
  MCC_MNC_TO_PLMNID(instance->mcc, instance->mnc, instance->mnc_digit_length,
                    &x2SetupResponse->globalENB_ID.pLMN_Identity);

  X2AP_INFO("%d -> %02x%02x%02x\n", instance->eNB_id,
	    x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[0],
	    x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[1],
            x2SetupResponse->globalENB_ID.eNB_ID.choice.macro_eNB_ID.buf[2]);

  //----served cells------
#warning update the value of the message
  served_cell->servedCellInfo.pCI = 6;
  served_cell->servedCellInfo.eUTRA_Mode_Info.present = X2ap_EUTRA_Mode_Info_PR_fDD;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_EARFCN = 3350;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_EARFCN = 3350;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.uL_Transmission_Bandwidth = 0;
  served_cell->servedCellInfo.eUTRA_Mode_Info.choice.fDD.dL_Transmission_Bandwidth = 0;

  MCC_MNC_TO_PLMNID(instance->mcc,instance->mnc,instance->mnc_digit_length,&served_cell->servedCellInfo.cellId.pLMN_Identity);
  MCC_MNC_TO_PLMNID(instance->mcc, instance->mnc, instance->mnc_digit_length,&broadcast_plmnIdentity_1);
  MCC_MNC_TO_PLMNID(instance->mcc, instance->mnc, instance->mnc_digit_length,&broadcast_plmnIdentity_2);
  MCC_MNC_TO_PLMNID(instance->mcc, instance->mnc, instance->mnc_digit_length,&broadcast_plmnIdentity_3);

  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_1);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_2);
  ASN_SEQUENCE_ADD(&served_cell->servedCellInfo.broadcastPLMNs.list, &broadcast_plmnIdentity_3);

  ECI_TO_BIT_STRING(instance->eNB_id, &served_cell->servedCellInfo.cellId.eUTRANcellIdentifier);
  TAC_TO_OCTET_STRING(instance->tac, &served_cell->servedCellInfo.tAC);
  ASN_SEQUENCE_ADD(&x2SetupResponse->servedCells.list, served_cell);

  if (x2ap_eNB_encode_pdu(&message, &buffer, &len) < 0) {
    X2AP_ERROR("Failed to encode X2 setup request\n");
    return -1;
  }

  eNB_association->state = X2AP_ENB_STATE_READY;

   MSC_LOG_TX_MESSAGE (MSC_X2AP_TARGET_ENB, MSC_X2AP_SRC_ENB, NULL, 0, "0 X2Setup/successfulOutcome assoc_id %u", eNB_association->assoc_id);
  /*
   * Non-UE signalling -> stream 0
   */
  x2ap_eNB_itti_send_sctp_data_req (instance->instance, eNB_association->assoc_id, buffer, len, 0);

  return ret;
}

int x2ap_eNB_set_cause (X2ap_Cause_t * cause_p,
		       X2ap_Cause_PR cause_type,
		       long cause_value);

int x2ap_eNB_generate_x2_setup_failure ( uint32_t assoc_id,
					 X2ap_Cause_PR cause_type,
					 long cause_value,
					 long time_to_wait){

  uint8_t                                *buffer_p;
  uint32_t                                length;
  x2ap_message                            message;
  X2SetupFailure_IEs_t                    *x2_setup_failure_p;
  int                                     ret = 0;

  memset (&message, 0, sizeof (x2ap_message));
  x2_setup_failure_p = &message.msg.x2SetupFailure_IEs;
  message.procedureCode = X2ap_ProcedureCode_id_x2Setup;
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

  x2ap_eNB_itti_send_sctp_data_req (/* instance? */ 0, assoc_id, buffer_p, length, 0);

  return ret;
}

int x2ap_eNB_set_cause (X2ap_Cause_t * cause_p,
		       X2ap_Cause_PR cause_type,
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

int x2ap_eNB_generate_x2_handover_request(x2ap_eNB_instance_t *instance_p,
				          x2ap_eNB_data_t *x2ap_enb_data_p,
                                          int source_x2id)
{
  /* TODO: set correct values in there */
  x2ap_message               message;
  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(x2ap_enb_data_p != NULL);

  memset(&message, 0, sizeof(x2ap_message));

  message.direction     = X2AP_PDU_PR_initiatingMessage;
  message.procedureCode = X2ap_ProcedureCode_id_handoverPreparation;
  message.criticality   = X2ap_Criticality_reject;

  message.msg.x2ap_HandoverRequest_IEs.old_eNB_UE_X2AP_ID = source_x2id;

  message.msg.x2ap_HandoverRequest_IEs.cause.present = X2ap_Cause_PR_radioNetwork;
  message.msg.x2ap_HandoverRequest_IEs.cause.choice.radioNetwork = X2ap_CauseRadioNetwork_handover_desirable_for_radio_reasons;

  MCC_MNC_TO_PLMNID(123, 456, 3, &message.msg.x2ap_HandoverRequest_IEs.targetCell_ID.pLMN_Identity);

  int eci1 = 79;
  ECI_TO_BIT_STRING(eci1, &message.msg.x2ap_HandoverRequest_IEs.targetCell_ID.eUTRANcellIdentifier);

  MCC_MNC_TO_PLMNID(456, 123, 3, &message.msg.x2ap_HandoverRequest_IEs.gummei_id.gU_Group_ID.pLMN_Identity);
  MMEGID_TO_OCTET_STRING(26, &message.msg.x2ap_HandoverRequest_IEs.gummei_id.gU_Group_ID.mME_Group_ID);
  MMEC_TO_OCTET_STRING(12, &message.msg.x2ap_HandoverRequest_IEs.gummei_id.mME_Code);
  message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.mME_UE_S1AP_ID = 12;
  ENCRALG_TO_BIT_STRING(0,&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.uESecurityCapabilities.encryptionAlgorithms);

  INTPROTALG_TO_BIT_STRING(0,&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.uESecurityCapabilities.integrityProtectionAlgorithms);

  char KeNB_star[32] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 };

  KENB_STAR_TO_BIT_STRING(KeNB_star,&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.aS_SecurityInformation.key_eNodeB_star);
  message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.aS_SecurityInformation.nextHopChainingCount = 1;

  UEAGMAXBITRTD_TO_ASN_PRIMITIVES(3L,&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.uEaggregateMaximumBitRate.uEaggregateMaximumBitRateDownlink);
  UEAGMAXBITRTU_TO_ASN_PRIMITIVES(6L,&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.uEaggregateMaximumBitRate.uEaggregateMaximumBitRateUplink);

  X2ap_E_RABs_ToBeSetup_Item_t *e_RABs_ToBeSetup_Item1 = calloc(1, sizeof(X2ap_E_RABs_ToBeSetup_Item_t));

  e_RABs_ToBeSetup_Item1->e_RAB_ID=10;
  e_RABs_ToBeSetup_Item1->e_RAB_Level_QoS_Parameters.qCI=1;
  e_RABs_ToBeSetup_Item1->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.priorityLevel = 1;
  e_RABs_ToBeSetup_Item1->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.pre_emptionCapability = 0;
  e_RABs_ToBeSetup_Item1->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.pre_emptionVulnerability = 0;
  e_RABs_ToBeSetup_Item1->e_RAB_Level_QoS_Parameters.allocationAndRetentionPriority.iE_Extensions = NULL;

  TRLA_TO_BIT_STRING(1, &e_RABs_ToBeSetup_Item1->uL_GTPtunnelEndpoint.transportLayerAddress); // IPv4
  GTP_TEID_TO_OCTET_STRING (12, &e_RABs_ToBeSetup_Item1->uL_GTPtunnelEndpoint.gTP_TEID);

  X2ap_E_RABs_ToBeSetup_ListIEs_t *e_RABs_ToBeSetup_List1 = calloc(1, sizeof(X2ap_E_RABs_ToBeSetup_ListIEs_t));

  ASN_SEQUENCE_ADD(e_RABs_ToBeSetup_List1, e_RABs_ToBeSetup_Item1);
  x2ap_encode_x2ap_e_rabs_tobesetup_list(&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.e_RABs_ToBeSetup_List, e_RABs_ToBeSetup_List1);

#if 0
  char RRC[81] = { 0x0a,0x10,0x00,0x00,0x03,0x41,0x60,0x08,0xcf,0x50,0x4a,0x0e,0x07,0x00,0x8c,0xf5,0x04,0xa0,0xe0,0x03,0xc0,0x51,0xc2,0x28,
                   0xb8,0x56,0xd1,0x80,0x4a,0x00,0x00,0x08,0x18,0x02,0x20,0x42,0x08,0x00,0x80,0x60,0x00,0x20,0x00,0x00,0x03,0x82,0xca,0x04,
                   0x06,0x14,0x08,0x0f,0x59,0x95,0x64,0x80,0x00,0x02,0x0a,0x5a,0x00,0x20,0x00,0x08,0x04,0x01,0x2f,0x39,0x57,0x96,0xde,0xc0,
                   0x00,0x88,0xea,0x46,0x7d,0x10,0x00,0x03,0x40 };
#endif
  char RRC[] = {
0x0A,0x10,0x31,0xC5,0x20,0x00,0x05,0x00,0x10,0x40,0xC1,0xC9,0x85,0x8B,0xF8,0xDF,0xE2,0xFE,0x37,0xF8,0xBF,0x8D,0xFE,0x2F,0xE3,0x7F,0x8B,0xF8,0xDF,0xE2,0xFE,0x37,0xF7,0xF0,0xFF,0xE9,0x88,0x81,0x00,0x87,0x0C,0xA7,0x4A,0x92,0x20,0x20,0x58,0x00,0x00,0x00,0x00,0x00,0x15,0xD8,0x00,0x00,0x06,0x8B,0x02,0x00,0x02,0x90,0x21,0x59,0x70,0x00,0x00,0x72,0xA2,0x40,0x37,0xB0,0x11,0xFA,0x9C,0x0B,0x81,0x1F,0xA9,0xC0,0x83,0xEA,0x26,0xE0,0x25,0x75,0x38,0xA1,0xD8,0x84,0xF9,0x80,0x3E,0x55,0x8F,0xFD,0x15,0x35,0x86,0x1C,0x86,0xC8,0xA1,0x82,0x40,0xA1,0x35,0x00,0x00,0x00,0x24,0x10,0x92,0x80,0x02,0x00,0x00,0x10,0x2C,0x00,0x30,0x00,0x40,0xF6,0x07,0xCA,0xCB,0xB2,0x4C,0x00,0x6C,0x05,0x35,0x00,0x10,0x00,0x04,0x01,0x00,0xF7,0x52,0xAB,0xCA,0xF7,0x20,0x07,0x43,0x45,0xA0,0x1E,0xB8,0xE0,
};

  OCTET_STRING_fromBuf(&message.msg.x2ap_HandoverRequest_IEs.uE_ContextInformation.rRC_Context, (char*) RRC, sizeof(RRC));

  X2ap_LastVisitedCell_Item_t *lastVisitedCell_Item1 = calloc(1, sizeof(X2ap_LastVisitedCell_Item_t));
  lastVisitedCell_Item1->present = X2ap_LastVisitedCell_Item_PR_e_UTRAN_Cell;
  lastVisitedCell_Item1->choice.e_UTRAN_Cell.cellType.cell_Size=1;
  lastVisitedCell_Item1->choice.e_UTRAN_Cell.time_UE_StayedInCell=2;

  MCC_MNC_TO_PLMNID(987,765, 3, &lastVisitedCell_Item1->choice.e_UTRAN_Cell.global_Cell_ID.pLMN_Identity);

  int eci2 = 55;
  ECI_TO_BIT_STRING(eci2,&lastVisitedCell_Item1->choice.e_UTRAN_Cell.global_Cell_ID.eUTRANcellIdentifier);

  ASN_SEQUENCE_ADD(&message.msg.x2ap_HandoverRequest_IEs.uE_HistoryInformation.list,lastVisitedCell_Item1);

  if (x2ap_eNB_encode_pdu(&message, &buffer, &len) < 0) {
    X2AP_ERROR("Failed to encode X2 setup request\n");
abort();
    return -1;
  }

  /* TODO: use correct stream, 1 for the moment */
  x2ap_eNB_itti_send_sctp_data_req(instance_p->instance, x2ap_enb_data_p->assoc_id, buffer, len, 1);

  return ret;
}

int x2ap_eNB_generate_x2_handover_response(x2ap_eNB_instance_t *instance,
				           x2ap_eNB_data_t *x2ap_enb_data_p,
                                           int source_x2id)
{

  x2ap_message              message;
  uint8_t                                *buffer;
  uint32_t                                len;
  int                                      ret = 0;

  // Generating response
  memset (&message, 0, sizeof (x2ap_message));

  X2ap_E_RABs_Admitted_ListIEs_t *e_RABs_Admitted_List1;
  X2ap_E_RABs_Admitted_Item_t *e_RABs_Admitted_Item1;

  e_RABs_Admitted_Item1 = calloc(1, sizeof(X2ap_E_RABs_Admitted_Item_t));
  e_RABs_Admitted_List1 = calloc(1, sizeof(X2ap_E_RABs_Admitted_ListIEs_t));
  asn1_xer_print = 1;
  asn_debug = 0;
  X2ap_ProcedureCode_t procedure = X2ap_ProcedureCode_id_handoverPreparation;
  X2ap_Criticality_t criticality = X2ap_Criticality_reject;
  X2AP_PDU_PR present;
  present = X2AP_PDU_PR_successfulOutcome;
  message.procedureCode = procedure;
  message.criticality= criticality;
  message.direction = present;
  //data is in file RRC_Context_acknowledge.txt
  uint8_t RRC[63] = { 0x01,0xe9,0x00,0x90,0xa8,0x00,0x00,0x22,0x33,0xe9,0x42,0x80,0x02,0xf0,0x80,0x9e,0x20,0x23,0xc6,0x05,0x79,0x00,0xef,0x28,
                      0x21,0xe1,0x01,0x24,0x38,0x40,0x05,0x00,0x12,0x1c,0xa0,0x00,0x02,0x00,0x88,0x02,0x18,0x06,0x40,0x10,0xa0,0x2b,0x43,0x81,
                      0x1d,0xd9,0xc0,0x30,0x70,0x00,0xe0,0x21,0xc3,0x17,0x01,0x74,0x60,0x12,0x80 };
  message.msg.x2ap_HandoverRequestAcknowledge_IEs.old_eNB_UE_X2AP_ID = source_x2id;
  message.msg.x2ap_HandoverRequestAcknowledge_IEs.new_eNB_UE_X2AP_ID= 2001;

  e_RABs_Admitted_Item1->e_RAB_ID=12;

  e_RABs_Admitted_Item1->iE_Extensions = NULL;
  ASN_SEQUENCE_ADD(e_RABs_Admitted_List1, e_RABs_Admitted_Item1);
  memcpy(&message.msg.x2ap_HandoverRequestAcknowledge_IEs.e_RABs_Admitted_List, e_RABs_Admitted_List1, sizeof(X2ap_E_RABs_Admitted_ListIEs_t));
  OCTET_STRING_fromBuf(&message.msg.x2ap_HandoverRequestAcknowledge_IEs.targeteNBtoSource_eNBTransparentContainer, (char*) RRC, sizeof(RRC));

  if (x2ap_eNB_encode_pdu(&message, &buffer, &len) < 0) {
    X2AP_ERROR("Failed to encode X2 handover response\n");
abort();
    return -1;
  }

  //eNB_association->state = X2AP_ENB_STATE_READY;

  /* TODO: use correct stream, 1 for the moment */
  x2ap_eNB_itti_send_sctp_data_req(instance->instance, x2ap_enb_data_p->assoc_id, buffer, len, 1);

  return ret;
}
