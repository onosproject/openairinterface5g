/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
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
#include "E2AP_E2nodeComponentID.h"

extern int global_e2_node_id(ranid_t ranid, E2AP_GlobalE2node_ID_t* node_id);

extern unsigned int ran_functions_len;
extern ric_ran_function_t **ran_functions;

int e2ap_generate_e2_setup_request(ranid_t  ranid,
                   uint8_t **buffer,uint32_t *len,
                   e2node_type_t e2node_type)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_E2setupRequest_t *req;
  E2AP_E2setupRequestIEs_t *ie;
  E2AP_RANfunction_ItemIEs_t *ran_function_item_ie;
  E2AP_E2nodeComponentConfigAddition_ItemIEs_t *e2node_comp_cfg_update_ie;
  ric_ran_function_t *func;
  unsigned int i;

  //DevAssert(ric != NULL);
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

  global_e2_node_id(ranid, &ie->value.choice.GlobalE2node_ID);

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
    ran_function_item_ie->value.choice.RANfunction_Item.ranFunctionOID = *oid;

    ASN_SEQUENCE_ADD(&ie->value.choice.RANfunctions_List.list,
             ran_function_item_ie);
  }
  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  /*to be re-done during E2AP 2.0.1 integration */
  /* E2nodeComponentConfigUpdate_List */
  E2AP_E2nodeComponentID_t *e2NodeCompId;
  ie = (E2AP_E2setupRequestIEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_E2nodeComponentConfigAddition;
  ie->criticality = E2AP_Criticality_reject;
  //ie->value.present = E2AP_E2setupRequestIEs__value_PR_NOTHING;
  ie->value.present = E2AP_E2setupRequestIEs__value_PR_E2nodeComponentConfigAddition_List;

#if 1
  e2node_comp_cfg_update_ie = (E2AP_E2nodeComponentConfigAddition_ItemIEs_t *)calloc(1,sizeof(*e2node_comp_cfg_update_ie));
  e2node_comp_cfg_update_ie->id = E2AP_ProtocolIE_ID_id_E2nodeComponentConfigAddition_Item;
  e2node_comp_cfg_update_ie->criticality = E2AP_Criticality_reject;
  e2node_comp_cfg_update_ie->value.present = E2AP_E2nodeComponentConfigAddition_ItemIEs__value_PR_E2nodeComponentConfigAddition_Item;
  
  e2NodeCompId = &(e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentID);

  if (e2node_type == E2NODE_TYPE_ENB_CU)
  {
    e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentInterfaceType = E2AP_E2nodeComponentInterfaceType_e1; //E2AP_E2nodeComponentType_ng_eNB_CU 
    e2NodeCompId->present = E2AP_E2nodeComponentID_PR_e2nodeComponentInterfaceTypeE1;
    if (asn_umax2INTEGER(&e2NodeCompId->choice.e2nodeComponentInterfaceTypeE1.gNB_CU_CP_ID, 100) != 0)
        RIC_AGENT_ERROR("gNB_CU_UP_ID encoding failed\n");
    //e2NodeCompId->choice.e2nodeComponentTypeGNB_CU_UP.gNB_CU_UP_ID.size = strlen("100");//sizeof(uint64_t);
    //e2NodeCompId->choice.e2nodeComponentTypeGNB_CU_UP.gNB_CU_UP_ID.buf = (uint8_t *)strdup("100");
  }
  else if (e2node_type == E2NODE_TYPE_ENB_DU)
  {
    e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentInterfaceType = E2AP_E2nodeComponentInterfaceType_f1; //E2AP_E2nodeComponentType_ng_eNB_DU
    e2NodeCompId->present = E2AP_E2nodeComponentID_PR_e2nodeComponentInterfaceTypeF1;
    if (asn_umax2INTEGER(&e2NodeCompId->choice.e2nodeComponentInterfaceTypeF1.gNB_DU_ID, 200) != 0)
        RIC_AGENT_ERROR("gNB_DU_ID encoding failed\n");
    //e2NodeCompId->choice.e2nodeComponentTypeGNB_DU.gNB_DU_ID.size = strlen("200");
    //e2NodeCompId->choice.e2nodeComponentTypeGNB_DU.gNB_DU_ID.buf = (uint8_t *)strdup("200");
  }

  e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentConfiguration.e2nodeComponentRequestPart.size = strlen("100");
  e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentConfiguration.e2nodeComponentRequestPart.buf = (uint8_t *)strdup("100");
  e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentConfiguration.e2nodeComponentResponsePart.size = strlen("200");
  e2node_comp_cfg_update_ie->value.choice.E2nodeComponentConfigAddition_Item.e2nodeComponentConfiguration.e2nodeComponentResponsePart.buf = (uint8_t *)strdup("200");
  
  ASN_SEQUENCE_ADD(&ie->value.choice.E2nodeComponentConfigAddition_List.list,
                   e2node_comp_cfg_update_ie);
#endif
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

int e2ap_generate_e2_config_update(ranid_t  ranid,
                   uint8_t **buffer,uint32_t *len,
                   e2node_type_t e2node_type)
{
  E2AP_E2AP_PDU_t pdu;
  E2AP_E2nodeConfigurationUpdate_t *req;
  E2AP_E2nodeConfigurationUpdate_IEs_t *ie;
  E2AP_E2nodeComponentConfigUpdate_ItemIEs_t *updateItemIe;
  E2AP_E2nodeComponentConfigUpdate_Item_t *updateItem;
  //E2AP_RANfunction_ItemIEs_t *ran_function_item_ie;
  //E2AP_E2nodeComponentConfigUpdate_ItemIEs_t *e2node_comp_cfg_update_ie;
  //ric_ran_function_t *func;
  //unsigned int i;

  //DevAssert(ric != NULL);
  DevAssert(buffer != NULL && len != NULL);

  memset(&pdu,0,sizeof(pdu));
  pdu.present = E2AP_E2AP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage.procedureCode = E2AP_ProcedureCode_id_E2nodeConfigurationUpdate;
  pdu.choice.initiatingMessage.criticality = E2AP_Criticality_reject;
  pdu.choice.initiatingMessage.value.present = E2AP_InitiatingMessage__value_PR_E2nodeConfigurationUpdate;
  req = &pdu.choice.initiatingMessage.value.choice.E2nodeConfigurationUpdate;

  /* Transaction Id */
  ie = (E2AP_E2nodeConfigurationUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_TransactionID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_E2nodeConfigurationUpdate_IEs__value_PR_TransactionID;
  ie->value.choice.TransactionID = 10;

  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  /* GlobalE2node_ID */
  ie = (E2AP_E2nodeConfigurationUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_GlobalE2node_ID;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_E2nodeConfigurationUpdate_IEs__value_PR_GlobalE2node_ID;
  global_e2_node_id(ranid, &ie->value.choice.GlobalE2node_ID);

  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  /* Component Configuration Update List */
  ie = (E2AP_E2nodeConfigurationUpdate_IEs_t *)calloc(1,sizeof(*ie));
  ie->id = E2AP_ProtocolIE_ID_id_E2nodeComponentConfigUpdate;
  ie->criticality = E2AP_Criticality_reject;
  ie->value.present = E2AP_E2nodeConfigurationUpdate_IEs__value_PR_E2nodeComponentConfigUpdate_List;

  
  if (e2node_type == E2NODE_TYPE_ENB_CU)
  { 
    /* S1 Interface Component Config */
    updateItemIe = (E2AP_E2nodeComponentConfigUpdate_ItemIEs_t *)calloc(1,sizeof(*updateItemIe));
    updateItemIe->id = E2AP_ProtocolIE_ID_id_E2nodeComponentConfigUpdate_Item;
    updateItemIe->criticality = E2AP_Criticality_reject;
    updateItemIe->value.present = E2AP_E2nodeComponentConfigUpdate_ItemIEs__value_PR_E2nodeComponentConfigUpdate_Item;

    updateItem = (E2AP_E2nodeComponentConfigUpdate_Item_t *)&updateItemIe->value.choice.E2nodeComponentConfigUpdate_Item;
    updateItem->e2nodeComponentInterfaceType = E2AP_E2nodeComponentInterfaceType_s1;
    updateItem->e2nodeComponentID.present = E2AP_E2nodeComponentID_PR_e2nodeComponentInterfaceTypeS1;
    updateItem->e2nodeComponentID.choice.e2nodeComponentInterfaceTypeS1.mme_name.buf = (uint8_t *)strdup("MME1");
    updateItem->e2nodeComponentID.choice.e2nodeComponentInterfaceTypeS1.mme_name.size = strlen("MME1");
    updateItem->e2nodeComponentConfiguration.e2nodeComponentRequestPart.size = strlen("100");
    updateItem->e2nodeComponentConfiguration.e2nodeComponentRequestPart.buf = (uint8_t *)strdup("100");
    updateItem->e2nodeComponentConfiguration.e2nodeComponentResponsePart.size = strlen("200");
    updateItem->e2nodeComponentConfiguration.e2nodeComponentResponsePart.buf = (uint8_t *)strdup("200");

    ASN_SEQUENCE_ADD(&ie->value.choice.E2nodeComponentConfigUpdate_List.list,updateItemIe);
  }

  /* E1/F1 Interface Component Config */
  updateItemIe = (E2AP_E2nodeComponentConfigUpdate_ItemIEs_t *)calloc(1,sizeof(*updateItemIe));
  updateItemIe->id = E2AP_ProtocolIE_ID_id_E2nodeComponentConfigUpdate_Item;
  updateItemIe->criticality = E2AP_Criticality_reject;
  updateItemIe->value.present = E2AP_E2nodeComponentConfigUpdate_ItemIEs__value_PR_E2nodeComponentConfigUpdate_Item;

  updateItem = (E2AP_E2nodeComponentConfigUpdate_Item_t *)&updateItemIe->value.choice.E2nodeComponentConfigUpdate_Item;
  if (e2node_type == E2NODE_TYPE_ENB_CU)
  {
    updateItem->e2nodeComponentInterfaceType = E2AP_E2nodeComponentInterfaceType_e1;
    updateItem->e2nodeComponentID.present = E2AP_E2nodeComponentID_PR_e2nodeComponentInterfaceTypeE1;
    if (asn_umax2INTEGER(&updateItem->e2nodeComponentID.choice.e2nodeComponentInterfaceTypeE1.gNB_CU_CP_ID, 100) != 0)
        RIC_AGENT_ERROR("gNB_CU_UP_ID encoding failed\n");
  }
  else if (e2node_type == E2NODE_TYPE_ENB_DU)
  {
    updateItem->e2nodeComponentInterfaceType = E2AP_E2nodeComponentInterfaceType_f1;
    updateItem->e2nodeComponentID.present = E2AP_E2nodeComponentID_PR_e2nodeComponentInterfaceTypeF1;
    if (asn_umax2INTEGER(&updateItem->e2nodeComponentID.choice.e2nodeComponentInterfaceTypeF1.gNB_DU_ID, 200) != 0)
        RIC_AGENT_ERROR("gNB_DU_ID encoding failed\n");
  }

  updateItem->e2nodeComponentConfiguration.e2nodeComponentRequestPart.size = strlen("100");
  updateItem->e2nodeComponentConfiguration.e2nodeComponentRequestPart.buf = (uint8_t *)strdup("100");
  updateItem->e2nodeComponentConfiguration.e2nodeComponentResponsePart.size = strlen("200");
  updateItem->e2nodeComponentConfiguration.e2nodeComponentResponsePart.buf = (uint8_t *)strdup("200");

  ASN_SEQUENCE_ADD(&ie->value.choice.E2nodeComponentConfigUpdate_List.list,updateItemIe);

  ASN_SEQUENCE_ADD(&req->protocolIEs.list,ie);

  if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
    RIC_AGENT_ERROR("Failed to encode E2ConfigUpdate\n");
    return 1;
  }

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

int du_e2ap_generate_ric_control_failure(du_ric_agent_info_t *ric,
        ric_control_t *rc, uint8_t **buffer,uint32_t *len)
{
    E2AP_E2AP_PDU_t pdu;
    E2AP_RICcontrolFailure_t *out;
    E2AP_RICcontrolFailure_IEs_t *ie;

    memset(&pdu, 0, sizeof(pdu));
    pdu.present = E2AP_E2AP_PDU_PR_unsuccessfulOutcome;
    pdu.choice.unsuccessfulOutcome.procedureCode = 
                            E2AP_ProcedureCode_id_RICcontrol;
    pdu.choice.unsuccessfulOutcome.criticality = E2AP_Criticality_reject;
    pdu.choice.unsuccessfulOutcome.value.present = 
                    E2AP_UnsuccessfulOutcome__value_PR_RICcontrolFailure;
    out = &pdu.choice.unsuccessfulOutcome.value.choice.RICcontrolFailure;

    ie = (E2AP_RICcontrolFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICcontrolFailure_IEs__value_PR_RICrequestID;
    ie->value.choice.RICrequestID.ricRequestorID = rc->request_id;
    ie->value.choice.RICrequestID.ricInstanceID = rc->instance_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

    ie = (E2AP_RICcontrolFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICcontrolFailure_IEs__value_PR_RANfunctionID;
    ie->value.choice.RANfunctionID = rc->function_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

    ie = (E2AP_RICcontrolFailure_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_Cause;
    ie->criticality = E2AP_Criticality_ignore;
    ie->value.present = E2AP_RICcontrolFailure_IEs__value_PR_Cause;
    ie->value.choice.Cause.present = E2AP_Cause_PR_ricRequest;
    ie->value.choice.Cause.choice.ricRequest = rc->failure_cause;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

     if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
        RIC_AGENT_ERROR("Failed to encode RICcontrolFailure\n");
        return -1;
    }

    return 0;
}

int du_e2ap_generate_ric_control_acknowledge(du_ric_agent_info_t *ric,
        ric_control_t *rc, uint8_t **buffer,uint32_t *len)
{   
    E2AP_E2AP_PDU_t pdu;
    E2AP_RICcontrolAcknowledge_t *out;
    E2AP_RICcontrolAcknowledge_IEs_t *ie;

    memset(&pdu, 0, sizeof(pdu));
    pdu.present = E2AP_E2AP_PDU_PR_successfulOutcome;
    pdu.choice.successfulOutcome.procedureCode = 
                            E2AP_ProcedureCode_id_RICcontrol;
    pdu.choice.successfulOutcome.criticality = E2AP_Criticality_reject;
    pdu.choice.successfulOutcome.value.present = 
                    E2AP_SuccessfulOutcome__value_PR_RICcontrolAcknowledge;
    out = &pdu.choice.successfulOutcome.value.choice.RICcontrolAcknowledge;
    
    ie = (E2AP_RICcontrolAcknowledge_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICcontrolAcknowledge_IEs__value_PR_RICrequestID;
    ie->value.choice.RICrequestID.ricRequestorID = rc->request_id;
    ie->value.choice.RICrequestID.ricInstanceID = rc->instance_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);
    
    ie = (E2AP_RICcontrolAcknowledge_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present = E2AP_RICcontrolAcknowledge_IEs__value_PR_RANfunctionID;
    ie->value.choice.RANfunctionID = rc->function_id;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);

#if 0
    ie = (E2AP_RICcontrolAcknowledge_IEs_t *)calloc(1,sizeof(*ie));
    ie->id = E2AP_ProtocolIE_ID_id_RICcontrolStatus;
    ie->criticality = E2AP_Criticality_ignore;
    ie->value.present = E2AP_RICcontrolAcknowledge_IEs__value_PR_RICcontrolOutcome;
    ie->value.choice.RICcontrolOutcome = E2AP_RICcontrolStatus_success;
    ASN_SEQUENCE_ADD(&out->protocolIEs.list,ie);
#endif

     if (e2ap_encode_pdu(&pdu,buffer,len) < 0) {
        RIC_AGENT_ERROR("Failed to encode RICcontrolAcknowledge\n");
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
    //ric_action_t *action;

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
    ie->id = E2AP_ProtocolIE_ID_id_Cause;
    ie->criticality = E2AP_Criticality_reject;
    ie->value.present
        = E2AP_RICsubscriptionFailure_IEs__value_PR_Cause;
    ie->value.choice.Cause.present = E2AP_Cause_PR_misc;
    ie->value.choice.Cause.choice.misc = E2AP_CauseMisc_control_processing_overload;

#if 0
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
#endif

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

    if ( (node_type == E2NODE_TYPE_ENB_CU) || (node_type == E2NODE_TYPE_ENB_DU) ) 
    {
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

void generate_e2apv1_indication_request_parameterized(E2AP_E2AP_PDU_t *e2ap_pdu,
        long requestorId, long instanceId, long ranFunctionId, long actionId,
        long seqNum, uint8_t *ind_header_buf, int header_length,
        uint8_t *ind_message_buf, int message_length)
{

    e2ap_pdu->present = E2AP_E2AP_PDU_PR_initiatingMessage;

    E2AP_InitiatingMessage_t* initmsg = &e2ap_pdu->choice.initiatingMessage;
    initmsg->procedureCode = E2AP_ProcedureCode_id_RICindication;
    initmsg->criticality = E2AP_Criticality_ignore;
    initmsg->value.present = E2AP_InitiatingMessage__value_PR_RICindication;

    /*
     * Encode RICrequestID IE
     */
    E2AP_RICindication_IEs_t *ricrequestid_ie = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricrequestid_ie->id = E2AP_ProtocolIE_ID_id_RICrequestID;
    ricrequestid_ie->criticality = 0;
    ricrequestid_ie->value.present = E2AP_RICsubscriptionRequest_IEs__value_PR_RICrequestID;
    ricrequestid_ie->value.choice.RICrequestID.ricRequestorID = requestorId;
    ricrequestid_ie->value.choice.RICrequestID.ricInstanceID = instanceId;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricrequestid_ie);

    /*
     * Encode RANfunctionID IE
     */
    E2AP_RICindication_IEs_t *ricind_ies2 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies2->id = E2AP_ProtocolIE_ID_id_RANfunctionID;
    ricind_ies2->criticality = 0;
    ricind_ies2->value.present = E2AP_RICindication_IEs__value_PR_RANfunctionID;
    ricind_ies2->value.choice.RANfunctionID = ranFunctionId;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies2);


    /*
     * Encode RICactionID IE
     */
    E2AP_RICindication_IEs_t *ricind_ies3 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies3->id = E2AP_ProtocolIE_ID_id_RICactionID;
    ricind_ies3->criticality = 0;
    ricind_ies3->value.present = E2AP_RICindication_IEs__value_PR_RICactionID;
    ricind_ies3->value.choice.RICactionID = actionId;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies3);


    /*
     * Encode RICindicationSN IE
     */
    E2AP_RICindication_IEs_t *ricind_ies4 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies4->id = E2AP_ProtocolIE_ID_id_RICindicationSN;
    ricind_ies4->criticality = 0;
    ricind_ies4->value.present = E2AP_RICindication_IEs__value_PR_RICindicationSN;
    ricind_ies4->value.choice.RICindicationSN = seqNum;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies4);

    /*
     * Encode RICindicationType IE
     */
    E2AP_RICindication_IEs_t *ricind_ies5 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies5->id = E2AP_ProtocolIE_ID_id_RICindicationType;
    ricind_ies5->criticality = 0;
    ricind_ies5->value.present = E2AP_RICindication_IEs__value_PR_RICindicationType;
    ricind_ies5->value.choice.RICindicationType = E2AP_RICindicationType_report;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies5);


#if 0
    //uint8_t *buf2 = (uint8_t *)"reportheader";
    OCTET_STRING_t *hdr_str = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));

    hdr_str->buf = (uint8_t*)calloc(1,header_length);
    hdr_str->size = header_length;
    memcpy(hdr_str->buf, ind_header_buf, header_length);
#endif

    /*
     * Encode RICindicationHeader IE
     */
    E2AP_RICindication_IEs_t *ricind_ies6 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies6->id = E2AP_ProtocolIE_ID_id_RICindicationHeader;
    ricind_ies6->criticality = 0;
    ricind_ies6->value.present = E2AP_RICindication_IEs__value_PR_RICindicationHeader;
    ricind_ies6->value.choice.RICindicationHeader.buf = (uint8_t*)calloc(1, header_length);
    ricind_ies6->value.choice.RICindicationHeader.size = header_length;
    memcpy(ricind_ies6->value.choice.RICindicationHeader.buf, ind_header_buf, header_length);
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies6);

    /*
     * Encode RICindicationMessage IE
     */
    E2AP_RICindication_IEs_t *ricind_ies7 = (E2AP_RICindication_IEs_t*)calloc(1, sizeof(E2AP_RICindication_IEs_t));
    ricind_ies7->id = E2AP_ProtocolIE_ID_id_RICindicationMessage;
    ricind_ies7->criticality = 0;
    ricind_ies7->value.present = E2AP_RICindication_IEs__value_PR_RICindicationMessage;
    ricind_ies7->value.choice.RICindicationMessage.buf = (uint8_t*)calloc(1, 8192);
    memcpy(ricind_ies7->value.choice.RICindicationMessage.buf, ind_message_buf, message_length);
    ricind_ies7->value.choice.RICindicationMessage.size = message_length;
    ASN_SEQUENCE_ADD(&initmsg->value.choice.RICindication.protocolIEs.list, ricind_ies7);


    char *error_buf = (char*)calloc(300, sizeof(char));
    size_t errlen;

    asn_check_constraints(&asn_DEF_E2AP_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
    //printf(" E2AP PDU error length %zu\n", errlen);
    //printf("E2AP PDU error buf %s\n", error_buf);
    free(error_buf);

    //xer_fprint(stderr, &asn_DEF_E2AP_E2AP_PDU, e2ap_pdu);
}

int e2ap_asn1c_encode_pdu(E2AP_E2AP_PDU_t* pdu, unsigned char **buffer)
{
    int len;

    *buffer = NULL;
    assert(pdu != NULL);
    assert(buffer != NULL);

    len = aper_encode_to_new_buffer(&asn_DEF_E2AP_E2AP_PDU, 0, pdu, (void **)buffer);

    if (len < 0)  {
        RIC_AGENT_ERROR("Unable to aper encode");
        exit(1);
    }
    else {
        RIC_AGENT_INFO("Encoded succesfully, encoded size = %d\n", len);
    }

    ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU, pdu);

    return len;
}

