/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: LicenseRef-ONF-Member-1.0
 */

/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "assertions.h"
#include "conversions.h"
#include "common/ngran_types.h"

#include "ric_agent.h"
#include "e2ap_generate_messages.h"
#include "e2ap_encoder.h"
#include "e2ap_decoder.h"
#include "e2sm_kpm.h"

#include "E2AP_Cause.h"
#include "E2AP_ProtocolIE-Field.h"
#include "E2AP_InitiatingMessage.h"
#include "E2AP_SuccessfulOutcome.h"
#include "E2AP_UnsuccessfulOutcome.h"
#include "E2AP_E2setupRequest.h"
#include "E2AP_GlobalE2node-ID.h"

extern int global_e2_node_id(ranid_t ranid, E2AP_GlobalE2node_ID_t* node_id);

extern unsigned int ran_functions_len;
extern ric_ran_function_t **ran_functions;

int e2ap_generate_e2_setup_request(ric_agent_info_t *ric,
				   uint8_t **buffer,uint32_t *len)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_E2setupRequest_t *req;
  E2AP_E2setupRequestIEs_t *ie;
  E2AP_RANfunction_ItemIEs_t *ran_function_item_ie;
  ric_ran_function_t *func;
  unsigned int i;

  DevAssert(ric != NULL);
  DevAssert(buffer != NULL && len != NULL);

  memset(&pdu,0,sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage.procedureCode = E2AP_ProcedureCode_id_E2setup;
  pdu.choice.initiatingMessage.criticality = E2AP_Criticality_reject;
  pdu.choice.initiatingMessage.value.present = E2AP_InitiatingMessage__value_PR_E2setupRequest;
  req = &pdu.choice.initiatingMessage.value.choice.E2setupRequest;

  /* GlobalE2node_ID */
  ie = (E2AP_E2setupRequestIEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_GlobalE2node_ID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_E2setupRequestIEs__value_PR_GlobalE2node_ID;

  global_e2_node_id(ric->ranid, &ie->value.choice.GlobalE2node_ID);

  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  /* "Optional" RANfunctions_List. */
  ie = (E2AP_E2setupRequestIEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionsAdded;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_E2setupRequestIEs__value_PR_RANfunctions_List;

  for (i = 0; i < ran_functions_len; ++i) 
  {
    func = ran_functions[i];
    DevAssert(func != NULL);

    ran_function_item_ie = (E2AP_RANfunction_ItemIEs_t *) \
      calloc(1,sizeof(*ran_function_item_ie));
    ran_function_item_ie->id = E2AP_ProtocolIE_ID_id_RANfunction_Item;
    ran_function_item_ie->criticality = E2AP_Criticality_reject;
    
    ran_function_item_ie->value.present = E2AP_RANfunction_ItemIEs__value_PR_RANfunction_Item; 
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionID = func->function_id;
	//RIC_AGENT_INFO("RAN Function Def Len:%u\n",func->enc_definition_len);
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionDefinition.buf = (uint8_t *)malloc(func->enc_definition_len);
    memcpy(ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionDefinition.buf,
	       func->enc_definition, 
           func->enc_definition_len);
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionDefinition.size = func->enc_definition_len;
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionRevision = func->revision;

    int oid_len = strlen(func->model->oid);
    E2AP_RANfunctionOID_t* oid = (E2AP_RANfunctionOID_t*)calloc(1, sizeof(E2AP_RANfunctionOID_t));
    oid->buf = (uint8_t *)calloc(oid_len, sizeof(uint8_t));
    memcpy(oid->buf, func->model->oid, oid_len);
    oid->size = oid_len;
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionOID = oid;

    ASN_SEQUENCE_ADD(&ie->value.choice.RANfunctions_List.list,
		     ran_function_item_ie);
  }
  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
    RIC_AGENT_ERROR("Failed to encode E2setupRequest\n");
    return 1;
  }
  /*
  E2AP_E2AP_PDU_t pdud;
  memset(&pdud,0,sizeof(pdud));
  if (e2ap_decode_pdu(&pdud,*buffer,*len) < 0) {
    RIC_AGENT_WARN("Failed to encode E2setupRequest\n");
  }
  */

  return 0;
}

int e2ap_generate_ric_subscription_response(
        ric_agent_info_t *ric,
        ric_subscription_t *rs,
        uint8_t **buffer,
        uint32_t *len)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_RICsubscriptionResponse_t *out;
  E2AP_RICsubscriptionResponse_IEs_t *ie;
  E2AP_RICaction_NotAdmitted_Item_t *nai;
  ric_action_t *action;

  memset(&pdu, 0, sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_successfulOutcome;
  pdu.choice.successfulOutcome.procedureCode = E2AP_ProcedureCode_id_RICsubscription;
  pdu.choice.successfulOutcome.criticality = E2AP_Criticality_reject;
  pdu.choice.successfulOutcome.value.present = E2AP_SuccessfulOutcome__value_PR_RICsubscriptionResponse;
  out = &pdu.choice.successfulOutcome.value.choice.RICsubscriptionResponse;

  ie = (E2AP_RICsubscriptionResponse_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionResponse_IEs__value_PR_RICrequestID;
  ie->value.choice.RICrequestID.ricRequestorID = rs->request_id;
  ie->value.choice.RICrequestID.ricInstanceID = rs->instance_id;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICsubscriptionResponse_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionResponse_IEs__value_PR_RANfunctionID;
  ie->value.choice.RANfunctionID = rs->function_id;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICsubscriptionResponse_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RICactions_Admitted;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionResponse_IEs__value_PR_RICaction_Admitted_List;
  LIST_FOREACH(action, &rs->action_list, actions) {
    if (!action->enabled) {
      continue;
    }
    E2AP_RICaction_Admitted_ItemIEs_t* ais = (E2AP_RICaction_Admitted_ItemIEs_t*)calloc(1, sizeof(E2AP_RICaction_Admitted_ItemIEs_t));
    ais->id = E2AP_ProtocolIE_ID_id_RICaction_Admitted_Item;
    ais->criticality = E2AP_Criticality_reject;
    ais->value.present = E2AP_RICaction_Admitted_ItemIEs__value_PR_RICaction_Admitted_Item;
    E2AP_RICaction_Admitted_Item_t *ai = &ais->value.choice.RICaction_Admitted_Item;
    ai->ricActionID = action->id;
    ASN_SEQUENCE_ADD(&ie->value.choice.RICaction_Admitted_List.list,ais);
  }
  xer_fprint(stdout, &asn_DEF_E2AP_RICsubscriptionResponse_IEs, ie);
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICsubscriptionResponse_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RICactions_NotAdmitted;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionResponse_IEs__value_PR_RICaction_NotAdmitted_List;
  LIST_FOREACH(action,&rs->action_list,actions) {
    if (action->enabled) {
      continue;
    }
    nai = (E2AP_RICaction_NotAdmitted_Item_t *)calloc(1,sizeof(*nai));
    nai->ricActionID = action->id;
    nai->cause.present = action->error_cause;
    switch (nai->cause.present) {
    case E2AP_Cause_PR_NOTHING:
      break;
    case E2AP_Cause_PR_ricRequest:
      nai->cause.choice.ricRequest = action->error_cause_detail;
      break;
    case E2AP_Cause_PR_ricService:
      nai->cause.choice.ricService = action->error_cause_detail;
      break;
    case E2AP_Cause_PR_transport:
      nai->cause.choice.transport = action->error_cause_detail;
      break;
    case E2AP_Cause_PR_protocol:
      nai->cause.choice.protocol = action->error_cause_detail;
      break;
    case E2AP_Cause_PR_misc:
      nai->cause.choice.misc = action->error_cause_detail;
      break;
    default:
      break;
    }
    ASN_SEQUENCE_ADD(&ie->value.choice.RICaction_NotAdmitted_List.list,nai);
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  if (e2ap_encode_pdu(&pdu, buffer, len) < 0) {
    RIC_AGENT_ERROR("Failed to encode RICsubscriptionResponse\n");
    return -1;
  }

  return 0;
}

int e2ap_generate_ric_subscription_failure(ric_agent_info_t *ric,
        ric_subscription_t *rs, uint8_t **buffer,uint32_t *len)
{
    E2AP_E2AP_PDU_t pdu;
    E2AP_RICsubscriptionFailure_t *out;
    E2AP_RICsubscriptionFailure_IEs_t *ie;
    ric_action_t *action;

    memset(&pdu, 0, sizeof(pdu));
    pdu.present = E2AP_E2AP_PDU_PR_unsuccessfulOutcome;
    pdu.choice.unsuccessfulOutcome.procedureCode
        = E2AP_ProcedureCode_id_RICsubscription;
    pdu.choice.unsuccessfulOutcome.criticality = E2AP_Criticality_reject;
    pdu.choice.unsuccessfulOutcome.value.present
        = E2AP_UnsuccessfulOutcome__value_PR_RICsubscriptionFailure;
    out = &pdu.choice.unsuccessfulOutcome.value.choice.RICsubscriptionFailure;

    ie = (E2AP_RICsubscriptionFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICsubscriptionFailure_IEs__value_PR_RICrequestID;
    ie->value.choice.RICrequestID.ricRequestorID = rs->request_id;
    ie->value.choice.RICrequestID.ricInstanceID = rs->instance_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

    ie = (E2AP_RICsubscriptionFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICsubscriptionFailure_IEs__value_PR_RANfunctionID;
    ie->value.choice.RANfunctionID = rs->function_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

    ie = (E2AP_RICsubscriptionFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RICactions_NotAdmitted;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present
        = E2AP_RICsubscriptionFailure_IEs__value_PR_RICaction_NotAdmitted_List;

    LIST_FOREACH(action, &rs->action_list, actions) {
        E2AP_RICaction_NotAdmitted_ItemIEs_t* ais
            = (E2AP_RICaction_NotAdmitted_ItemIEs_t*)calloc(1,
                    sizeof(E2AP_RICaction_NotAdmitted_ItemIEs_t));
        ais->id = E2AP_ProtocolIE_ID_id_RICaction_NotAdmitted_Item;
        ais->criticality = E2AP_Criticality_reject;
        ais->value.present
            = E2AP_RICaction_NotAdmitted_ItemIEs__value_PR_RICaction_NotAdmitted_Item;
        E2AP_RICaction_NotAdmitted_Item_t *ai
            = &ais->value.choice.RICaction_NotAdmitted_Item;
        ai->ricActionID = action->id;
        // TODO
        //ai->cause.present = action->error_cause;
        ai->cause.present = E2AP_Cause_PR_ricRequest;
        switch (ai->cause.present) {
            case E2AP_Cause_PR_NOTHING:
                break;
            case E2AP_Cause_PR_ricRequest:
                // TODO
                //ai->cause.choice.ricRequest = action->error_cause_detail;
                ai->cause.choice.ricRequest
                    = E2AP_CauseRIC_ran_function_id_Invalid;
                break;
            case E2AP_Cause_PR_ricService:
                ai->cause.choice.ricService = action->error_cause_detail;
                break;
            case E2AP_Cause_PR_transport:
                ai->cause.choice.transport = action->error_cause_detail;
                break;
            case E2AP_Cause_PR_protocol:
                ai->cause.choice.protocol = action->error_cause_detail;
                break;
            case E2AP_Cause_PR_misc:
                ai->cause.choice.misc = action->error_cause_detail;
                break;
            default:
                break;
        }
        ASN_SEQUENCE_ADD(
                &ie->value.choice.RICaction_NotAdmitted_List.list, ais);
    }

    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

    if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
        RIC_AGENT_ERROR("Failed to encode RICsubscriptionFailure\n");
        return -1;
    }

    return 0;
}

int e2ap_generate_ric_subscription_delete_response(
        ric_agent_info_t *ric,
        long request_id,
        long instance_id,
        ric_ran_function_id_t function_id,
        uint8_t **buffer,
        uint32_t *len)
{
    E2AP_E2AP_PDU_t pdu;
    E2AP_RICsubscriptionDeleteResponse_t *out;
    E2AP_RICsubscriptionDeleteResponse_IEs_t *ie;

    memset(&pdu, 0, sizeof(pdu));
    pdu.present = E2AP_E2AP_PDU_PR_successfulOutcome;
    pdu.choice.successfulOutcome.procedureCode = E2AP_ProcedureCode_id_RICsubscriptionDelete;
    pdu.choice.successfulOutcome.criticality = E2AP_Criticality_reject;
    pdu.choice.successfulOutcome.value.present = E2AP_SuccessfulOutcome__value_PR_RICsubscriptionDeleteResponse;
    out = &pdu.choice.successfulOutcome.value.choice.RICsubscriptionDeleteResponse;

    ie = (E2AP_RICsubscriptionDeleteResponse_IEs_t *)calloc(1, sizeof(E2AP_RICsubscriptionDeleteResponse_IEs_t));
    ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICsubscriptionDeleteResponse_IEs__value_PR_RICrequestID;
    ie->value.choice.RICrequestID.ricRequestorID = request_id;
    ie->value.choice.RICrequestID.ricInstanceID = instance_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

    ie = (E2AP_RICsubscriptionDeleteResponse_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICsubscriptionDeleteResponse_IEs__value_PR_RANfunctionID;
    ie->value.choice.RANfunctionID = function_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list, ie);

    if (e2ap_encode_pdu(&pdu, buffer, len) < 0) {
        RIC_AGENT_ERROR("Failed to encode RICsubscriptionDeleteResponse\n");
        return -1;
    }

    return 0;
}

int e2ap_generate_ric_subscription_delete_failure(
  ric_agent_info_t *ric,long request_id,long instance_id,
  ric_ran_function_id_t function_id,long cause,long cause_detail,
  uint8_t **buffer,uint32_t *len)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_RICsubscriptionDeleteFailure_t *out;
  E2AP_RICsubscriptionDeleteFailure_IEs_t *ie;

  memset(&pdu, 0, sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_unsuccessfulOutcome;
  pdu.choice.unsuccessfulOutcome.procedureCode = E2AP_ProcedureCode_id_RICsubscription;
  pdu.choice.unsuccessfulOutcome.criticality = E2AP_Criticality_reject;
  pdu.choice.unsuccessfulOutcome.value.present = E2AP_UnsuccessfulOutcome__value_PR_RICsubscriptionDeleteFailure;
  out = &pdu.choice.unsuccessfulOutcome.value.choice.RICsubscriptionDeleteFailure;

  ie = (E2AP_RICsubscriptionDeleteFailure_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionDeleteFailure_IEs__value_PR_RICrequestID;
  ie->value.choice.RICrequestID.ricRequestorID = request_id;
  ie->value.choice.RICrequestID.ricInstanceID = instance_id;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICsubscriptionDeleteFailure_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionDeleteFailure_IEs__value_PR_RANfunctionID;
  ie->value.choice.RANfunctionID = function_id;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICsubscriptionDeleteFailure_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RICactions_NotAdmitted;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICsubscriptionDeleteFailure_IEs__value_PR_Cause;
  ie->value.choice.Cause.present = cause;
  switch (cause) {
  case E2AP_Cause_PR_NOTHING:
    break;
  case E2AP_Cause_PR_ricRequest:
    ie->value.choice.Cause.choice.ricRequest = cause_detail;
    break;
  case E2AP_Cause_PR_ricService:
    ie->value.choice.Cause.choice.ricService = cause_detail;
    break;
  case E2AP_Cause_PR_transport:
    ie->value.choice.Cause.choice.transport = cause_detail;
    break;
  case E2AP_Cause_PR_protocol:
    ie->value.choice.Cause.choice.protocol = cause_detail;
    break;
  case E2AP_Cause_PR_misc:
    ie->value.choice.Cause.choice.misc = cause_detail;
    break;
  default:
    break;
  }
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
    RIC_AGENT_ERROR("Failed to encode RICsubscriptionDeleteFailure\n");
    return -1;
  }

  return 0;
}

int e2ap_generate_ric_service_update(ric_agent_info_t *ric,
				     uint8_t **buffer,uint32_t *len)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_RICserviceUpdate_t *out;
  E2AP_RICserviceUpdate_IEs_t *ie;

  /*
   * NB: we never add, modify, or remove ran functions, so this is a noop.
   */

  memset(&pdu, 0, sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage.procedureCode = E2AP_ProcedureCode_id_RICserviceUpdate;
  pdu.choice.initiatingMessage.criticality = E2AP_Criticality_reject;
  pdu.choice.initiatingMessage.value.present = E2AP_InitiatingMessage__value_PR_RICserviceUpdate;
  out = &pdu.choice.initiatingMessage.value.choice.RICserviceUpdate;

  ie = (E2AP_RICserviceUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionsAdded;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICserviceUpdate_IEs__value_PR_RANfunctions_List;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICserviceUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionsModified;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICserviceUpdate_IEs__value_PR_RANfunctions_List_1;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  ie = (E2AP_RICserviceUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_RANfunctionsDeleted;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_RICserviceUpdate_IEs__value_PR_RANfunctionsID_List;
  ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

  if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
    RIC_AGENT_ERROR("Failed to encode RICserviceUpdate\n");
    return -1;
  }

  return 0;
}

int e2ap_generate_reset_response(ric_agent_info_t *ric,
				 uint8_t **buffer,uint32_t *len)
{
  E2AP_E2AP_PDU_t pdu;

  memset(&pdu, 0, sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_successfulOutcome;
  pdu.choice.successfulOutcome.procedureCode = E2AP_ProcedureCode_id_Reset;
  pdu.choice.successfulOutcome.criticality = E2AP_Criticality_reject;
  pdu.choice.successfulOutcome.value.present = E2AP_SuccessfulOutcome__value_PR_ResetResponse;

  if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
    RIC_AGENT_ERROR("Failed to encode ResetResponse\n");
    return -1;
  }

  return 0;
}

int global_e2_node_id(ranid_t ranid, E2AP_GlobalE2node_ID_t* node_id) {
    e2node_type_t node_type;

    node_type = e2_conf[ranid]->e2node_type;

    if (node_type == E2NODE_TYPE_ENB_CU) {
        node_id->present = E2AP_GlobalE2node_ID_PR_eNB;

        MCC_MNC_TO_PLMNID(
                e2_conf[ranid]->mcc,
                e2_conf[ranid]->mnc,
                e2_conf[ranid]->mnc_digit_length,
                &node_id->choice.eNB.global_eNB_ID.pLMN_Identity);

        node_id->choice.eNB.global_eNB_ID.eNB_ID.present = E2AP_ENB_ID_PR_macro_eNB_ID;

        MACRO_ENB_ID_TO_BIT_STRING(
                e2_conf[ranid]->cell_identity,
                &node_id->choice.eNB.global_eNB_ID.eNB_ID.choice.macro_eNB_ID);

    }
#if 0
    else if (node_type == E2NODE_TYPE_NG_ENB) {
        node_id->present = E2AP_GlobalE2node_ID_PR_ng_eNB;

        MCC_MNC_TO_PLMNID(
                e2_conf[ranid]->mcc,
                e2_conf[ranid]->mnc,
                e2_conf[ranid]->mnc_digit_length,
                &node_id->choice.ng_eNB.global_ng_eNB_ID.plmn_id);

        node_id->choice.ng_eNB.global_ng_eNB_ID.enb_id.present
            = E2AP_ENB_ID_Choice_PR_enb_ID_macro;

        MACRO_ENB_ID_TO_BIT_STRING(
                e2_conf[ranid]->cell_identity,
                &node_id->choice.ng_eNB.global_ng_eNB_ID.enb_id.choice.enb_ID_macro);

    } else if (node_type == E2NODE_TYPE_GNB) {

        node_id->present = E2AP_GlobalE2node_ID_PR_gNB;

        MCC_MNC_TO_PLMNID(
                e2_conf[ranid]->mcc,
                e2_conf[ranid]->mnc,
                e2_conf[ranid]->mnc_digit_length,
                &node_id->choice.gNB.global_gNB_ID.plmn_id);

        node_id->choice.gNB.global_gNB_ID.gnb_id.present = E2AP_GNB_ID_Choice_PR_gnb_ID;

        /* XXX: GNB version? */

        MACRO_ENB_ID_TO_BIT_STRING(
                e2_conf[ranid]->cell_identity,
                &node_id->choice.gNB.global_gNB_ID.gnb_id.choice.gnb_ID);
    } else {
        RIC_AGENT_ERROR("unsupported eNB/gNB ngran_node_t %d; aborting!\n", node_type);
        exit(1);
    }
#endif
    return 0;
}
