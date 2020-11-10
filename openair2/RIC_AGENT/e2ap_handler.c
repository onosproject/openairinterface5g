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

#include <stdint.h>
#include <string.h>

#include "intertask_interface.h"
#include "conversions.h"

#include "ric_agent_common.h"
#include "ric_agent_defs.h"
#include "e2ap_handler.h"
#include "e2ap_decoder.h"
#include "e2ap_encoder.h"
#include "e2ap_generate_messages.h"
#include "e2sm_common.h"

#include "E2AP_Cause.h"
#include "E2AP_E2AP-PDU.h"
#include "E2AP_ProtocolIE-Field.h"
#include "E2AP_E2setupRequest.h"
#include "E2AP_RICsubsequentAction.h"

int e2ap_handle_e2_setup_response(ric_agent_info_t *ric,uint32_t stream,
				  E2AP_E2AP_PDU_t *pdu)
{
  E2AP_E2setupResponse_t *resp;
  E2AP_E2setupResponseIEs_t *rie,**ptr;

  DevAssert(pdu != NULL);
  resp = &pdu->choice.successfulOutcome.value.choice.E2setupResponse;

  E2AP_INFO("Received E2SetupResponse (ranid %u)\n",ric->ranid);

  for (ptr = resp->protocolIEs.list.array;
       ptr < &resp->protocolIEs.list.array[resp->protocolIEs.list.count];
       ptr++) {
    rie = (E2AP_E2setupResponseIEs_t *)*ptr;
    if (rie->id == E2AP_ProtocolIE_ID_id_GlobalRIC_ID) {
      PLMNID_TO_MCC_MNC(&rie->value.choice.GlobalRIC_ID.pLMN_Identity,
			ric->ric_mcc,ric->ric_mnc,ric->ric_mnc_digit_len);
      //BIT_STRING_TO_INT32(&rie->value.choice.GlobalRIC_ID.ric_ID,ric->ric_id);
    }
    /* XXX: handle RANfunction IEs once we have some E2SM support. */
  }

  E2AP_INFO("E2SetupResponse (ranid %u) from RIC (mcc=%u,mnc=%u,id=%u)\n",
	    ric->ranid,ric->ric_mcc,ric->ric_mnc,ric->ric_id);

  ric->state = RIC_ESTABLISHED;

  return 0;
}

int e2ap_handle_e2_setup_failure(ric_agent_info_t *ric,uint32_t stream,
				 E2AP_E2AP_PDU_t *pdu)
{
  E2AP_E2setupFailure_t *resp;
  E2AP_E2setupFailureIEs_t *rie,**ptr;
  long cause,cause_detail;

  DevAssert(pdu != NULL);
  resp = &pdu->choice.unsuccessfulOutcome.value.choice.E2setupFailure;

  E2AP_INFO("Received E2SetupFailure (ranid %u)\n",ric->ranid);

  for (ptr = resp->protocolIEs.list.array;
       ptr < &resp->protocolIEs.list.array[resp->protocolIEs.list.count];
       ptr++) {
    rie = (E2AP_E2setupFailureIEs_t *)*ptr;
    if (rie->id == E2AP_ProtocolIE_ID_id_Cause) {
	cause = rie->value.choice.Cause.present;
	cause_detail = rie->value.choice.Cause.choice.misc;
    }
    /* XXX: handle RANfunction IEs once we have some E2SM support. */
  }

  E2AP_INFO("E2SetupFailure (ranid %u) from RIC (cause=%ld,detail=%ld)\n",
	    ric->ranid,cause,cause_detail);

  ric->state = RIC_FAILURE;

  return 0;
}

int e2ap_handle_ric_subscription_request(ric_agent_info_t *ric,uint32_t stream,
					 E2AP_E2AP_PDU_t *pdu)
{
  E2AP_RICsubscriptionRequest_t *req;
  E2AP_RICsubscriptionRequest_IEs_t *rie,**ptr;
  ric_subscription_t *rs;
  ric_action_t *ra,*rat;
  int ret;
  uint8_t *buf;
  uint32_t len;
  ric_ran_function_t *func;

  DevAssert(pdu != NULL);
  req = &pdu->choice.initiatingMessage.value.choice.RICsubscriptionRequest;

  E2AP_INFO("Received RICsubscriptionRequest from ranid %u\n",ric->ranid);

  /* We need to create an ric_subscription to generate errors. */
  rs = (ric_subscription_t *)calloc(1,sizeof(*rs));
  LIST_INIT(&rs->action_list);

  for (ptr = req->protocolIEs.list.array;
       ptr < &req->protocolIEs.list.array[req->protocolIEs.list.count];
       ptr++) {
    rie = (E2AP_RICsubscriptionRequest_IEs_t *)*ptr;
    if (rie->id == E2AP_ProtocolIE_ID_id_RICrequestID) {
      rs->request_id = rie->value.choice.RICrequestID.ricRequestorID;
      rs->instance_id = rie->value.choice.RICrequestID.ricInstanceID;
    }
    else if (rie->id == E2AP_ProtocolIE_ID_id_RANfunctionID) {
      rs->function_id = rie->value.choice.RANfunctionID;
    }
    else if (rie->id == E2AP_ProtocolIE_ID_id_RICsubscriptionDetails) {
      E2AP_RICeventTriggerDefinition_t *rtd = &rie->value.choice.RICsubscriptionDetails.ricEventTriggerDefinition;
      E2AP_RICactions_ToBeSetup_List_t *ral = &rie->value.choice.RICsubscriptionDetails.ricAction_ToBeSetup_List;
      E2AP_RICaction_ToBeSetup_Item_t *rai;

      if (rtd->size > 0 && rtd->size < E2SM_MAX_DEF_SIZE) {
	rs->event_trigger.size = rie->value.choice.RICsubscriptionDetails.ricEventTriggerDefinition.size;
	rs->event_trigger.buf = (uint8_t *)malloc(rs->event_trigger.size);
	memcpy(rs->event_trigger.buf,
	       rie->value.choice.RICsubscriptionDetails.ricEventTriggerDefinition.buf,
	       rs->event_trigger.size);
      }
      else if (rtd->size > E2SM_MAX_DEF_SIZE) {
	E2AP_ERROR("RICsubscriptionRequest eventTriggerDefinition too long!");
	// XXX: protocol error?
      }

#ifdef SHAD
      for (int i = 0; i < ral->list.count; ++i) {
	rai = (E2AP_RICaction_ToBeSetup_Item_t *)ral->list.array[i];
	ra = (ric_action_t *)calloc(1,sizeof(*ra));
	ra->id = rai->ricActionID;
	ra->type = rai->ricActionType;
	if (rai->ricActionDefinition && rai->ricActionDefinition->size > 0) {
	  ra->def_size = rai->ricActionDefinition->size;
	  ra->def_buf = (uint8_t *)malloc(ra->def_size);
	  memcpy(ra->def_buf,rai->ricActionDefinition->buf,ra->def_size);
	}
	if (rai->ricSubsequentAction) {
	  ra->subsequent_action = rai->ricSubsequentAction->ricSubsequentActionType;
	  ra->time_to_wait = rai->ricSubsequentAction->ricTimeToWait;
	}

	if (LIST_EMPTY(&rs->action_list) == 0) {
	  LIST_INSERT_HEAD(&rs->action_list,ra,actions);
	}
	else {
	  LIST_INSERT_BEFORE(LIST_FIRST(&rs->action_list),ra,actions);
	}
      }
#endif
    }
  }

#ifdef SHAD
  func = ric_agent_lookup_ran_function(rs->function_id);
  if (!func) {
    E2AP_ERROR("failed to find ran_function %ld\n",rs->function_id);
    goto errout;
  }

  ret = func->model->handle_subscription_add(ric,rs);
  if (ret) {
    E2AP_ERROR("failed to subscribe to ran_function %ld\n",rs->function_id);
    goto errout;
  }
#endif

  ret = e2ap_generate_ric_subscription_response(ric,rs,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICsubscriptionResponse (ranid %u)\n",
	       ric->ranid);
    goto errout;
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

  return 0;

 errout:
  ret = e2ap_generate_ric_subscription_failure(ric,rs,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICsubscriptionFailure (ranid %u)\n",
	       ric->ranid);
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

#ifdef SHAD
  ra = LIST_FIRST(&rs->action_list);
  while (ra != NULL) {
    rat = LIST_NEXT(ra,actions);
    if (ra->def_buf)
      free(ra->def_buf);
    free(ra);
    ra = rat;
  }
  if (rs->event_trigger.buf)
    free(rs->event_trigger.buf);
  free(rs);
#endif

  return ret;
}

int e2ap_handle_ric_subscription_delete_request(
  ric_agent_info_t *ric,uint32_t stream,E2AP_E2AP_PDU_t *pdu)
{
  E2AP_RICsubscriptionDeleteRequest_t *req;
  E2AP_RICsubscriptionDeleteRequest_IEs_t *rie,**ptr;
  long request_id;
  long instance_id;
  ric_ran_function_id_t function_id;
  ric_subscription_t *rs;
  int ret;
  uint8_t *buf;
  uint32_t len;
  ric_ran_function_t *func;
  long cause;
  long cause_detail;

  DevAssert(pdu != NULL);
  req = &pdu->choice.initiatingMessage.value.choice.RICsubscriptionDeleteRequest;

  E2AP_INFO("Received RICsubscriptionDeleteRequest from ranid %u\n",ric->ranid);

  for (ptr = req->protocolIEs.list.array;
       ptr < &req->protocolIEs.list.array[req->protocolIEs.list.count];
       ptr++) {
    rie = (E2AP_RICsubscriptionDeleteRequest_IEs_t *)*ptr;
    if (rie->id == E2AP_ProtocolIE_ID_id_RICrequestID) {
      request_id = rie->value.choice.RICrequestID.ricRequestorID;
      instance_id = rie->value.choice.RICrequestID.ricInstanceID;
    }
    else if (rie->id == E2AP_ProtocolIE_ID_id_RANfunctionID) {
      function_id = rie->value.choice.RANfunctionID;
    }
  }

  func = ric_agent_lookup_ran_function(function_id);
  if (!func) {
    E2AP_ERROR("failed to find ran_function %ld\n",function_id);
    cause = E2AP_Cause_PR_ricRequest;
    cause_detail = E2AP_CauseRIC_ran_function_id_Invalid;
    goto errout;
  }

  rs = ric_agent_lookup_subscription(ric,request_id,instance_id,function_id);
  if (!rs) {
    E2AP_ERROR("failed to find subscription %ld/%ld/%ld\n",
	       request_id,instance_id,function_id);
    cause = E2AP_Cause_PR_ricRequest;
    cause_detail = E2AP_CauseRIC_request_id_unknown;
    goto errout;
  }

  ret = func->model->handle_subscription_del(ric,rs,0,&cause,&cause_detail);
  if (ret) {
    E2AP_ERROR("failed to remove subscription to ran_function %ld\n",
	       rs->function_id);
    goto errout;
  }

  ret = e2ap_generate_ric_subscription_delete_response(
    ric,request_id,instance_id,function_id,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICsubscriptionDeleteResponse (ranid %u)\n",
	       ric->ranid);
    cause = E2AP_Cause_PR_protocol;
    cause_detail = E2AP_CauseProtocol_unspecified;
    goto errout;
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

  return 0;

 errout:
  ret = e2ap_generate_ric_subscription_delete_failure(
    ric,request_id,instance_id,function_id,cause,cause_detail,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICsubscriptionDeleteFailure (ranid %u)\n",
	       ric->ranid);
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

  return ret;
}

int e2ap_handle_ric_service_query(ric_agent_info_t *ric,uint32_t stream,
				  E2AP_E2AP_PDU_t *pdu)
{
  uint8_t *buf;
  uint32_t len;
  int ret;

  E2AP_INFO("Received RICserviceQuery from ranid %u\n",ric->ranid);

  /*
   * NB: we never add, modify, or remove service models or functions, so
   * this is a noop for us.
   */
  ret = e2ap_generate_ric_service_update(ric,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICserviceUpdate (ranid %u)\n",
	       ric->ranid);
    return -1;
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

  return 0;
}

int e2ap_handle_reset_request(ric_agent_info_t *ric,uint32_t stream,
			      E2AP_E2AP_PDU_t *pdu)
{
  uint8_t *buf;
  uint32_t len;
  int ret;

  E2AP_INFO("Received RICresetRequest from ranid %u\n",ric->ranid);

  ric_agent_reset(ric);

  ret = e2ap_generate_reset_response(ric,&buf,&len);
  if (ret) {
    E2AP_ERROR("failed to generate RICresetResponse (ranid %u)\n",
	       ric->ranid);
    return -1;
  }
  else {
    ric_agent_send_sctp_data(ric,stream,buf,len);
  }

  return 0;
}

int e2ap_handle_message(ric_agent_info_t *ric,int32_t stream,
			const uint8_t * const buf,const uint32_t buflen)
{
  E2AP_E2AP_PDU_t pdu;
  int ret;

  DevAssert(buf != NULL);

  memset(&pdu,0,sizeof(pdu));
  ret = e2ap_decode_pdu(&pdu,buf,buflen);
  if (ret < 0) {
    E2AP_ERROR("failed to decode PDU\n");
    return -1;
  }

  switch (pdu.present) {
  case E2AP_E2AP_PDU_PR_initiatingMessage:
    switch (pdu.choice.initiatingMessage.procedureCode) {
    case E2AP_ProcedureCode_id_RICsubscription:
      ret = e2ap_handle_ric_subscription_request(ric,stream,&pdu);
      break;
    case E2AP_ProcedureCode_id_RICsubscriptionDelete:
      ret = e2ap_handle_ric_subscription_delete_request(ric,stream,&pdu);
      break;
    case E2AP_ProcedureCode_id_RICserviceQuery:
      ret = e2ap_handle_ric_service_query(ric,stream,&pdu);
      break;
    case E2AP_ProcedureCode_id_Reset:
      ret = e2ap_handle_reset_request(ric,stream,&pdu);
      break;
    default:
      E2AP_WARN("unsupported initiatingMessage procedure %ld (ranid %u)\n",
		pdu.choice.initiatingMessage.procedureCode,ric->ranid);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU,&pdu);
      return -1;
    };
    break;
  case E2AP_E2AP_PDU_PR_successfulOutcome:
    switch (pdu.choice.successfulOutcome.procedureCode) {
    case E2AP_ProcedureCode_id_E2setup:
      ret = e2ap_handle_e2_setup_response(ric,stream,&pdu);
      break;
    default:
      E2AP_WARN("unsupported successfulOutcome procedure %ld (ranid %u)\n",
		pdu.choice.initiatingMessage.procedureCode,ric->ranid);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU,&pdu);
      return -1;
    };
    break;
  case E2AP_E2AP_PDU_PR_unsuccessfulOutcome:
    switch (pdu.choice.unsuccessfulOutcome.procedureCode) {
    case E2AP_ProcedureCode_id_E2setup:
      ret = e2ap_handle_e2_setup_failure(ric,stream,&pdu);
      break;
    default:
      E2AP_WARN("unsupported unsuccessfulOutcome procedure %ld (ranid %u)\n",
		pdu.choice.initiatingMessage.procedureCode,ric->ranid);
      ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU,&pdu);
      return -1;
    };
    break;
  default:
    E2AP_ERROR("unsupported presence %u (ranid %u)\n",
	       pdu.present,ric->ranid);
    ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU,&pdu);
    return -1;
  }

  ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2AP_E2AP_PDU,&pdu);
  return ret;
}
