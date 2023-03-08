/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <unistd.h>

#include <iterator>
#include <vector>


#include "encode_e2apv1.hpp"

extern "C" {
#include "E2SM-KPM-RANfunction-Description.h"

#include "e2ap_asn1c_codec.h"
#include "GlobalE2node-ID.h"
#include "GlobalE2node-gNB-ID.h"
#include "GlobalgNB-ID.h"
#include "OCTET_STRING.h"
#include "asn_application.h"
#include "GNB-ID-Choice.h"
#include "ProtocolIE-Field.h"
#include "E2setupRequest.h"
#include "RICaction-ToBeSetup-Item.h"
#include "RICactions-ToBeSetup-List.h"
#include "RICeventTriggerDefinition.h"
#include "RICsubscriptionRequest.h"
#include "RICsubscriptionResponse.h"
#include "ProtocolIE-SingleContainer.h"
#include "RANfunctions-List.h"
#include "RICindication.h"
#include "RICsubsequentActionType.h"
#include "RICsubsequentAction.h"  
#include "RICtimeToWait.h"
#include "E2SM-KPM-ActionDefinition.h"
#include "E2SM-KPM-EventTriggerDefinition.h"
#include "RT-Period-IE.h"
//#include "E2SM-KPM-IndicationHeader-Format1.h"
//#include "E2SM-KPM-EventTriggerDefinition-Format1.h"
//#include "Trigger-ConditionIE-Item.h"
}

static ssize_t e2sm_encode_ric_event_trigger_definition(void *buffer, size_t buf_size, size_t event_trigger_count, long *RT_periods);

void generate_e2apv1_setup_request(E2AP_PDU_t *e2ap_pdu) {
  
  //  uint8_t *buf = (uint8_t *)"gnb1"

  BIT_STRING_t *gnb_bstring = (BIT_STRING_t*)calloc(1, sizeof(BIT_STRING_t));;
  gnb_bstring->buf = (uint8_t*)calloc(1,4);
  gnb_bstring->size = 4;
  gnb_bstring->buf[0] = 0xB5;
  gnb_bstring->buf[1] = 0xC6;
  gnb_bstring->buf[2] = 0x77;
  gnb_bstring->buf[3] = 0x88;

  gnb_bstring->bits_unused = 3;

  uint8_t *buf2 = (uint8_t *)"747";
  OCTET_STRING_t *plmn = (OCTET_STRING_t*)calloc(1, sizeof(OCTET_STRING_t));
  plmn->buf = (uint8_t*)calloc(1,3);
  memcpy(plmn->buf, buf2, 3);
  plmn->size = 3;

  GNB_ID_Choice_t *gnbchoice = (GNB_ID_Choice_t*)calloc(1,sizeof(GNB_ID_Choice_t));
  GNB_ID_Choice_PR pres2 = GNB_ID_Choice_PR_gnb_ID;
  gnbchoice->present = pres2;
  gnbchoice->choice.gnb_ID = *gnb_bstring;

  GlobalgNB_ID_t *gnb = (GlobalgNB_ID_t*)calloc(1, sizeof(GlobalgNB_ID_t));
  gnb->plmn_id = *plmn;
  gnb->gnb_id = *gnbchoice;

  GlobalE2node_gNB_ID_t *e2gnb = (GlobalE2node_gNB_ID_t*)calloc(1, sizeof(GlobalE2node_gNB_ID_t));
  e2gnb->global_gNB_ID = *gnb;

  GlobalE2node_ID_t *globale2nodeid = (GlobalE2node_ID_t*)calloc(1, sizeof(GlobalE2node_ID_t));
  GlobalE2node_ID_PR pres;
  pres = GlobalE2node_ID_PR_gNB;
  globale2nodeid->present = pres;
  globale2nodeid->choice.gNB = e2gnb;
  
  E2setupRequestIEs_t *e2setuprid = (E2setupRequestIEs_t*)calloc(1, sizeof(E2setupRequestIEs_t));
  E2setupRequestIEs__value_PR pres3;
  pres3 = E2setupRequestIEs__value_PR_GlobalE2node_ID;
  e2setuprid->id = 3;
  e2setuprid->criticality = 0;
  e2setuprid->value.choice.GlobalE2node_ID = *globale2nodeid;
  e2setuprid->value.present = pres3;


  auto *ranFlistIEs = (E2setupRequestIEs_t *)calloc(1, sizeof(E2setupRequestIEs_t));
  ASN_STRUCT_RESET(asn_DEF_E2setupRequestIEs, ranFlistIEs);
  ranFlistIEs->criticality = 0;
  ranFlistIEs->id = ProtocolIE_ID_id_RANfunctionsAdded;
  ranFlistIEs->value.present = E2setupRequestIEs__value_PR_RANfunctions_List;

  auto *itemIes = (RANfunction_ItemIEs_t *)calloc(1, sizeof(RANfunction_ItemIEs_t));
  itemIes->id = ProtocolIE_ID_id_RANfunction_Item;
  itemIes->criticality = Criticality_reject;
  itemIes->value.present = RANfunction_ItemIEs__value_PR_RANfunction_Item;
  itemIes->value.choice.RANfunction_Item.ranFunctionID = 1;

  E2SM_KPM_RANfunction_Description_t *ranfunc_desc =
    (E2SM_KPM_RANfunction_Description_t*)calloc(1,sizeof(E2SM_KPM_RANfunction_Description_t));
  encode_kpm_function_description(ranfunc_desc);

  uint8_t e2smbuffer[8192];
  size_t e2smbuffer_size = 8192;

  asn_codec_ctx_t *opt_cod;  

  asn_enc_rval_t er =
    asn_encode_to_buffer(opt_cod,
			 ATS_ALIGNED_BASIC_PER,
			 &asn_DEF_E2SM_KPM_RANfunction_Description,
			 ranfunc_desc, e2smbuffer, e2smbuffer_size);
  
  fprintf(stderr, "er encded is %zu\n", er.encoded);
  fprintf(stderr, "after encoding message\n");

  OCTET_STRING_t *ranfuncdesc_str = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));
  ranfuncdesc_str->buf = (uint8_t*)calloc(1,er.encoded);
  ranfuncdesc_str->size = er.encoded;
  memcpy(ranfuncdesc_str->buf, e2smbuffer, er.encoded);
  

  itemIes->value.choice.RANfunction_Item.ranFunctionDefinition = *ranfuncdesc_str;
  itemIes->value.choice.RANfunction_Item.ranFunctionRevision = (long)2;

  ASN_SEQUENCE_ADD(&ranFlistIEs->value.choice.RANfunctions_List.list, itemIes);

  E2setupRequest_t *e2setupreq = (E2setupRequest_t*)calloc(1, sizeof(E2setupRequest_t));
  ASN_SEQUENCE_ADD(&e2setupreq->protocolIEs.list, e2setuprid);
  ASN_SEQUENCE_ADD(&e2setupreq->protocolIEs.list, ranFlistIEs);

  InitiatingMessage__value_PR pres4;
  pres4 = InitiatingMessage__value_PR_E2setupRequest;
  InitiatingMessage_t *initmsg = (InitiatingMessage_t*)calloc(1, sizeof(InitiatingMessage_t));

  initmsg->procedureCode = ProcedureCode_id_E2setup;
  initmsg->criticality = Criticality_reject;
  initmsg->value.present = pres4;
  initmsg->value.choice.E2setupRequest = *e2setupreq;

  E2AP_PDU_PR pres5;
  pres5 = E2AP_PDU_PR_initiatingMessage;
  

  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.initiatingMessage = initmsg;

}


void generate_e2apv1_setup_response(E2AP_PDU_t *e2ap_pdu) {

  E2setupResponseIEs *resp_ies1 = (E2setupResponseIEs_t*)calloc(1, sizeof(E2setupResponseIEs_t));
  E2setupResponseIEs *resp_ies2 = (E2setupResponseIEs_t*)calloc(1, sizeof(E2setupResponseIEs_t));
  E2setupResponseIEs *resp_ies3 = (E2setupResponseIEs_t*)calloc(1, sizeof(E2setupResponseIEs_t));

  uint8_t *buf = (uint8_t *)"gnb1";

  BIT_STRING_t *ricid_bstring = (BIT_STRING_t*)calloc(1,sizeof(BIT_STRING_t));
  ricid_bstring->buf = buf;
  ricid_bstring->size = 4;
  ricid_bstring->bits_unused = 0;

  uint8_t *buf2 = (uint8_t *)"plmn3";
  OCTET_STRING_t *plmn = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));
  plmn->buf = buf2;
  plmn->size = 5;

  GlobalRIC_ID_t *globalricid = (GlobalRIC_ID_t*)calloc(1,sizeof(GlobalRIC_ID_t));
  globalricid->pLMN_Identity = *plmn;
  globalricid->ric_ID = *ricid_bstring;

  E2setupResponseIEs__value_PR pres1;
  pres1 = E2setupResponseIEs__value_PR_GlobalRIC_ID;
  
  resp_ies1->id = ProtocolIE_ID_id_GlobalRIC_ID;
  resp_ies1->criticality = 0;
  resp_ies1->value.present = pres1;
  resp_ies1->value.choice.GlobalRIC_ID = *globalricid;

  E2setupResponse_t *e2setupresp = (E2setupResponse_t*)calloc(1,sizeof(E2setupResponse_t));
  int ret = ASN_SEQUENCE_ADD(&e2setupresp->protocolIEs.list, resp_ies1);


  SuccessfulOutcome__value_PR pres;
  pres = SuccessfulOutcome__value_PR_E2setupResponse;
  SuccessfulOutcome_t *successoutcome = (SuccessfulOutcome_t*)calloc(1, sizeof(SuccessfulOutcome_t));
  successoutcome->procedureCode = 1;
  successoutcome->criticality = 0;
  successoutcome->value.present = pres;
  successoutcome->value.choice.E2setupResponse = *e2setupresp;

  E2AP_PDU_PR pres5 = E2AP_PDU_PR_successfulOutcome;
  
  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.successfulOutcome = successoutcome;
  
}


void generate_e2apv1_subscription_delete(E2AP_PDU *e2ap_pdu) {

    // Encode RICRequestID 
    RICsubscriptionDeleteRequest_IEs_t *ricreqid = (RICsubscriptionDeleteRequest_IEs_t*)calloc(1, sizeof(RICsubscriptionDeleteRequest_IEs_t));
    ASN_STRUCT_RESET(asn_DEF_RICsubscriptionDeleteRequest_IEs, ricreqid);
    ricreqid->id = ProtocolIE_ID_id_RICrequestID;
    ricreqid->criticality = 0;
    ricreqid->value.present = RICsubscriptionDeleteRequest_IEs__value_PR_RICrequestID;
    ricreqid->value.choice.RICrequestID.ricRequestorID = 22;
    ricreqid->value.choice.RICrequestID.ricInstanceID = 6;

    // Encode RANFunctionID
    RICsubscriptionDeleteRequest_IEs_t *ranfuncid = (RICsubscriptionDeleteRequest_IEs_t*)calloc(1, sizeof(RICsubscriptionDeleteRequest_IEs_t));
    ASN_STRUCT_RESET(asn_DEF_RICsubscriptionDeleteRequest_IEs, ranfuncid);
    ranfuncid->id = ProtocolIE_ID_id_RANfunctionID;
    ranfuncid->criticality = 0;
    ranfuncid->value.present = RICsubscriptionDeleteRequest_IEs__value_PR_RANfunctionID;
    ranfuncid->value.choice.RANfunctionID = 1;

    // Encode RICsubscriptionDeleteRequest
    RICsubscriptionDeleteRequest_t *ricsubreq = (RICsubscriptionDeleteRequest_t*)calloc(1, sizeof(RICsubscriptionDeleteRequest_t));
    ASN_SEQUENCE_ADD(&ricsubreq->protocolIEs.list, ricreqid);
    ASN_SEQUENCE_ADD(&ricsubreq->protocolIEs.list, ranfuncid);
    InitiatingMessage__value_PR pres4;
    pres4 = InitiatingMessage__value_PR_RICsubscriptionDeleteRequest;
    InitiatingMessage_t *initmsg = (InitiatingMessage_t*)calloc(1, sizeof(InitiatingMessage_t));
    initmsg->procedureCode = ProcedureCode_id_RICsubscriptionDelete;
    initmsg->criticality = Criticality_reject;
    initmsg->value.present = pres4;
    initmsg->value.choice.RICsubscriptionDeleteRequest = *ricsubreq;

    /// Encode E2AP_PDU
    E2AP_PDU_PR pres5;
    pres5 = E2AP_PDU_PR_initiatingMessage;
    e2ap_pdu->present = pres5;
    e2ap_pdu->choice.initiatingMessage = initmsg;

    char *error_buf = (char*)calloc(300, sizeof(char));;
    size_t errlen;
    asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
    printf("error length %zu\n", errlen);
    printf("error buf %s\n", error_buf);
}

void generate_e2apv1_subscription_request(E2AP_PDU *e2ap_pdu) {

  RICsubscriptionRequest_IEs_t *ricreqid = (RICsubscriptionRequest_IEs_t*)calloc(1, sizeof(RICsubscriptionRequest_IEs_t));
  ASN_STRUCT_RESET(asn_DEF_RICsubscriptionRequest_IEs, ricreqid);

  auto *ricsubrid = (RICsubscriptionRequest_IEs_t*)calloc(1, sizeof(RICsubscriptionRequest_IEs_t));
  ASN_STRUCT_RESET(asn_DEF_RICsubscriptionRequest_IEs, ricsubrid);

  RICsubscriptionRequest_IEs_t *ranfuncid = (RICsubscriptionRequest_IEs_t*)calloc(1, sizeof(RICsubscriptionRequest_IEs_t));
  ASN_STRUCT_RESET(asn_DEF_RICsubscriptionRequest_IEs, ranfuncid);
  ranfuncid->id = ProtocolIE_ID_id_RANfunctionID;
  ranfuncid->criticality = 0;
  ranfuncid->value.present = RICsubscriptionRequest_IEs__value_PR_RANfunctionID;
  ranfuncid->value.choice.RANfunctionID = 1;

  uint8_t buf2[200];
  OCTET_STRING_t *triggerdef = (OCTET_STRING_t*)calloc(1, sizeof(OCTET_STRING_t));
  RT_Period_IE_t period = RT_Period_IE_ms1024;

  triggerdef->size = e2sm_encode_ric_event_trigger_definition(buf2, 200, 1, &period);
  assert(triggerdef->size != -1);
  triggerdef->buf = (uint8_t *)calloc(1, triggerdef->size);
  memcpy(triggerdef->buf, buf2, triggerdef->size);

  ProtocolIE_ID_t proto_id= ProtocolIE_ID_id_RICaction_ToBeSetup_Item;

  RICaction_ToBeSetup_ItemIEs__value_PR pres6;
  pres6 = RICaction_ToBeSetup_ItemIEs__value_PR_RICaction_ToBeSetup_Item;


  OCTET_STRING_t *actdef = (OCTET_STRING_t*)calloc(1, sizeof(OCTET_STRING_t));
  actdef->buf = (uint8_t *)calloc(1,9);
  actdef->size = 9;

  auto *sa = (RICsubsequentAction_t *) calloc(1, sizeof(RICsubsequentAction_t));
  ASN_STRUCT_RESET(asn_DEF_RICsubsequentAction, sa);

  sa->ricTimeToWait = RICtimeToWait_w500ms;
  sa->ricSubsequentActionType = RICsubsequentActionType_continue;

  /*
  RICaction_ToBeSetup_Item_t *action_item = (RICaction_ToBeSetup_Item_t*)calloc(1, sizeof(RICaction_ToBeSetup_Item_t));
  action_item->ricActionID = 5;
  action_item->ricActionType = 9;
  action_item->ricActionDefinition = actdef;
  action_item->ricSubsequentAction = sa;
  */

  RICaction_ToBeSetup_ItemIEs_t *action_item_ies = (RICaction_ToBeSetup_ItemIEs_t *)calloc(1, sizeof(RICaction_ToBeSetup_Item_t));
  action_item_ies->id = proto_id;
  action_item_ies->criticality = 0;

  action_item_ies->value.present = pres6;
  action_item_ies->value.choice.RICaction_ToBeSetup_Item.ricActionID = 5;
  action_item_ies->value.choice.RICaction_ToBeSetup_Item.ricActionType = RICactionType_report;
  action_item_ies->value.choice.RICaction_ToBeSetup_Item.ricActionDefinition = actdef;
  action_item_ies->value.choice.RICaction_ToBeSetup_Item.ricSubsequentAction = sa;


  /*
  RICsubscriptionDetails_t *ricsubdetails = (RICsubscriptionDetails_t*)calloc(1, sizeof(RICsubscriptionDetails_t));
  printf("sub5.5\n");

  ASN_SEQUENCE_ADD(&ricsubdetails->ricAction_ToBeSetup_List.list, action_item_ies);
  ricsubdetails->ricEventTriggerDefinition = *triggerdef;

  printf("sub6\n");
  */

  RICsubscriptionRequest_IEs__value_PR pres3;
  pres3 = RICsubscriptionRequest_IEs__value_PR_RICsubscriptionDetails;
  ricsubrid->id = ProtocolIE_ID_id_RICsubscriptionDetails;

  ricsubrid->criticality = 0;
  ricsubrid->value.present = pres3;

  ricsubrid->value.choice.RICsubscriptionDetails.ricEventTriggerDefinition = *triggerdef;

  ASN_SEQUENCE_ADD(&ricsubrid->value.choice.RICsubscriptionDetails.ricAction_ToBeSetup_List.list, action_item_ies);

  ricreqid->id = ProtocolIE_ID_id_RICrequestID;
  ricreqid->criticality = 0;
  ricreqid->value.present = RICsubscriptionRequest_IEs__value_PR_RICrequestID;
  ricreqid->value.choice.RICrequestID.ricRequestorID = 22;
  ricreqid->value.choice.RICrequestID.ricInstanceID = 6;

  RICsubscriptionRequest_t *ricsubreq = (RICsubscriptionRequest_t*)calloc(1, sizeof(RICsubscriptionRequest_t));

  ASN_SEQUENCE_ADD(&ricsubreq->protocolIEs.list,ricreqid);
  ASN_SEQUENCE_ADD(&ricsubreq->protocolIEs.list,ranfuncid);
  ASN_SEQUENCE_ADD(&ricsubreq->protocolIEs.list,ricsubrid);

  InitiatingMessage__value_PR pres4;
  pres4 = InitiatingMessage__value_PR_RICsubscriptionRequest;
  InitiatingMessage_t *initmsg = (InitiatingMessage_t*)calloc(1, sizeof(InitiatingMessage_t));
  initmsg->procedureCode = ProcedureCode_id_RICsubscription;
  initmsg->criticality = Criticality_reject;
  initmsg->value.present = pres4;
  initmsg->value.choice.RICsubscriptionRequest = *ricsubreq;

  E2AP_PDU_PR pres5;
  pres5 = E2AP_PDU_PR_initiatingMessage;

  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.initiatingMessage = initmsg;

  char *error_buf = (char*)calloc(300, sizeof(char));;
  size_t errlen;

  asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
  printf("error length %zu\n", errlen);
  printf("error buf %s\n", error_buf);

  //  xer_fprint(stderr, &asn_DEF_E2AP_PDU, e2ap_pdu);

}

void generate_e2apv1_subscription_response_success(E2AP_PDU *e2ap_pdu, long reqActionIdsAccepted[],
						   long reqActionIdsRejected[], int accept_size, int reject_size,
						   long reqRequestorId, long reqInstanceId) {

  RICsubscriptionResponse_IEs_t *respricreqid =
    (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
  
  respricreqid->id = ProtocolIE_ID_id_RICrequestID;
  respricreqid->criticality = 0;
  respricreqid->value.present = RICsubscriptionResponse_IEs__value_PR_RICrequestID;
  respricreqid->value.choice.RICrequestID.ricRequestorID = reqRequestorId;
  
  respricreqid->value.choice.RICrequestID.ricInstanceID = reqInstanceId;


  RICsubscriptionResponse_IEs_t *ricactionadmitted =
    (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
  ricactionadmitted->id = ProtocolIE_ID_id_RICactions_Admitted;
  ricactionadmitted->criticality = 0;
  ricactionadmitted->value.present = RICsubscriptionResponse_IEs__value_PR_RICaction_Admitted_List;

  RICaction_Admitted_List_t* admlist = 
    (RICaction_Admitted_List_t*)calloc(1,sizeof(RICaction_Admitted_List_t));
  ricactionadmitted->value.choice.RICaction_Admitted_List = *admlist;

  //  int numAccept = sizeof(reqActionIdsAccepted);
  int numAccept = accept_size;
  int numReject = reject_size;
  //  int numReject = sizeof(reqActionIdsRejected);

  
  for (int i=0; i < numAccept ; i++) {
    fprintf(stderr, "in for loop i = %d\n", i);

    long aid = reqActionIdsAccepted[i];

    RICaction_Admitted_ItemIEs_t *admitie = (RICaction_Admitted_ItemIEs_t*)calloc(1,sizeof(RICaction_Admitted_ItemIEs_t));
    admitie->id = ProtocolIE_ID_id_RICaction_Admitted_Item;
    admitie->criticality = 0;
    admitie->value.present = RICaction_Admitted_ItemIEs__value_PR_RICaction_Admitted_Item;
    admitie->value.choice.RICaction_Admitted_Item.ricActionID = aid;
    
    ASN_SEQUENCE_ADD(&ricactionadmitted->value.choice.RICaction_Admitted_List.list, admitie);

  }

  RICsubscriptionResponse_t *ricsubresp = (RICsubscriptionResponse_t*)calloc(1,sizeof(RICsubscriptionResponse_t));
  ASN_SEQUENCE_ADD(&ricsubresp->protocolIEs.list, respricreqid);
  ASN_SEQUENCE_ADD(&ricsubresp->protocolIEs.list, ricactionadmitted);
  

  if (numReject > 0) {

    RICsubscriptionResponse_IEs_t *ricactionrejected =
      (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
    ricactionrejected->id = ProtocolIE_ID_id_RICactions_NotAdmitted;
    ricactionrejected->criticality = 0;
    ricactionrejected->value.present = RICsubscriptionResponse_IEs__value_PR_RICaction_NotAdmitted_List;
    
    RICaction_NotAdmitted_List_t* rejectlist = 
      (RICaction_NotAdmitted_List_t*)calloc(1,sizeof(RICaction_NotAdmitted_List_t));
    ricactionadmitted->value.choice.RICaction_NotAdmitted_List = *rejectlist;
    
    for (int i=0; i < numReject; i++) {
      fprintf(stderr, "in for loop i = %d\n", i);
      
      long aid = reqActionIdsRejected[i];
      
      RICaction_NotAdmitted_ItemIEs_t *noadmitie = (RICaction_NotAdmitted_ItemIEs_t*)calloc(1,sizeof(RICaction_NotAdmitted_ItemIEs_t));
      noadmitie->id = ProtocolIE_ID_id_RICaction_NotAdmitted_Item;
      noadmitie->criticality = 0;
      noadmitie->value.present = RICaction_NotAdmitted_ItemIEs__value_PR_RICaction_NotAdmitted_Item;
      noadmitie->value.choice.RICaction_NotAdmitted_Item.ricActionID = aid;
      
      ASN_SEQUENCE_ADD(&ricactionrejected->value.choice.RICaction_NotAdmitted_List.list, noadmitie);
      ASN_SEQUENCE_ADD(&ricsubresp->protocolIEs.list, ricactionrejected);      
    }
  }


  SuccessfulOutcome__value_PR pres2;
  pres2 = SuccessfulOutcome__value_PR_RICsubscriptionResponse;
  SuccessfulOutcome_t *successoutcome = (SuccessfulOutcome_t*)calloc(1, sizeof(SuccessfulOutcome_t));
  successoutcome->procedureCode = ProcedureCode_id_RICsubscription;
  successoutcome->criticality = 0;
  successoutcome->value.present = pres2;
  successoutcome->value.choice.RICsubscriptionResponse = *ricsubresp;

  E2AP_PDU_PR pres5 = E2AP_PDU_PR_successfulOutcome;
  
  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.successfulOutcome = successoutcome;

  char *error_buf = (char*)calloc(300, sizeof(char));
  size_t errlen;

  asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
  printf("error length %zu\n", errlen);
  printf("error buf %s\n", error_buf);

  
}

void generate_e2apv1_subscription_response(E2AP_PDU *e2ap_pdu, E2AP_PDU *sub_req_pdu) {

  //Gather details of the request

  RICsubscriptionRequest_t orig_req =
    sub_req_pdu->choice.initiatingMessage->value.choice.RICsubscriptionRequest;
  
  RICsubscriptionResponse_IEs_t *ricreqid =
    (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
					   
  int count = orig_req.protocolIEs.list.count;
  int size = orig_req.protocolIEs.list.size;
  
  RICsubscriptionRequest_IEs_t **ies = (RICsubscriptionRequest_IEs_t**)orig_req.protocolIEs.list.array;

  fprintf(stderr, "count%d\n", count);
  fprintf(stderr, "size%d\n", size);

  RICsubscriptionRequest_IEs__value_PR pres;

  long responseRequestorId;
  long responseInstanceId;
  long responseActionId;

  std::vector<long> actionIds;

  for (int i=0; i < count; i++) {
    RICsubscriptionRequest_IEs_t *next_ie = ies[i];
    pres = next_ie->value.present;
    
    fprintf(stderr, "next present value %d\n", pres);

    switch(pres) {
    case RICsubscriptionRequest_IEs__value_PR_RICrequestID:
      {
	RICrequestID_t reqId = next_ie->value.choice.RICrequestID;
	long requestorId = reqId.ricRequestorID;
	long instanceId = reqId.ricInstanceID;
	fprintf(stderr, "requestorId %ld\n", requestorId);
	fprintf(stderr, "instanceId %ld\n", instanceId);
	responseRequestorId = requestorId;
	responseInstanceId = instanceId;
		
	break;
      }
    case RICsubscriptionRequest_IEs__value_PR_RANfunctionID:
      break;
    case RICsubscriptionRequest_IEs__value_PR_RICsubscriptionDetails:
      {
	RICsubscriptionDetails_t subDetails = next_ie->value.choice.RICsubscriptionDetails; 
	RICeventTriggerDefinition_t triggerDef = subDetails.ricEventTriggerDefinition;
	RICactions_ToBeSetup_List_t actionList = subDetails.ricAction_ToBeSetup_List;
	
	int actionCount = actionList.list.count;
	fprintf(stderr, "action count%d\n", actionCount);

	auto **item_array = actionList.list.array;

	for (int i=0; i < actionCount; i++) {
	  //RICaction_ToBeSetup_Item_t
	  auto *next_item = item_array[i];
	  RICactionID_t actionId = ((RICaction_ToBeSetup_ItemIEs*)next_item)->value.choice.RICaction_ToBeSetup_Item.ricActionID;
	  fprintf(stderr, "Next Action ID %ld\n", actionId);
	  responseActionId = actionId;
	  actionIds.push_back(responseActionId);
	}
	
	break;
      }
    }
    
  }

  fprintf(stderr, "After Processing Subscription Request\n");

  fprintf(stderr, "requestorId %ld\n", responseRequestorId);
  fprintf(stderr, "instanceId %ld\n", responseInstanceId);


  for (int i=0; i < actionIds.size(); i++) {
    fprintf(stderr, "Action ID %d %ld\n", i, actionIds.at(i));
    
  }


  RICsubscriptionResponse_IEs_t *respricreqid =
    (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
  
  respricreqid->id = ProtocolIE_ID_id_RICrequestID;
  respricreqid->criticality = 0;
  respricreqid->value.present = RICsubscriptionResponse_IEs__value_PR_RICrequestID;
  respricreqid->value.choice.RICrequestID.ricRequestorID = responseRequestorId;
  
  respricreqid->value.choice.RICrequestID.ricInstanceID = responseInstanceId;


  RICsubscriptionResponse_IEs_t *ricactionadmitted =
    (RICsubscriptionResponse_IEs_t*)calloc(1, sizeof(RICsubscriptionResponse_IEs_t));
  ricactionadmitted->id = ProtocolIE_ID_id_RICactions_Admitted;
  ricactionadmitted->criticality = 0;
  ricactionadmitted->value.present = RICsubscriptionResponse_IEs__value_PR_RICaction_Admitted_List;

  RICaction_Admitted_List_t* admlist = 
    (RICaction_Admitted_List_t*)calloc(1,sizeof(RICaction_Admitted_List_t));
  ricactionadmitted->value.choice.RICaction_Admitted_List = *admlist;

  for (int i=0; i < actionIds.size(); i++) {
    fprintf(stderr, "in for loop i = %d\n", i);

    long aid = actionIds.at(i);

    RICaction_Admitted_ItemIEs_t *admitie = (RICaction_Admitted_ItemIEs_t*)calloc(1,sizeof(RICaction_Admitted_ItemIEs_t));
    admitie->id = ProtocolIE_ID_id_RICaction_Admitted_Item;
    admitie->criticality = 0;
    admitie->value.present = RICaction_Admitted_ItemIEs__value_PR_RICaction_Admitted_Item;
    admitie->value.choice.RICaction_Admitted_Item.ricActionID = aid;
    
    ASN_SEQUENCE_ADD(&ricactionadmitted->value.choice.RICaction_Admitted_List.list, admitie);

  }


  RICsubscriptionResponse_t *ricsubresp = (RICsubscriptionResponse_t*)calloc(1,sizeof(RICsubscriptionResponse_t));
  
  ASN_SEQUENCE_ADD(&ricsubresp->protocolIEs.list, respricreqid);
  ASN_SEQUENCE_ADD(&ricsubresp->protocolIEs.list, ricactionadmitted);


  SuccessfulOutcome__value_PR pres2;
  pres2 = SuccessfulOutcome__value_PR_RICsubscriptionResponse;
  SuccessfulOutcome_t *successoutcome = (SuccessfulOutcome_t*)calloc(1, sizeof(SuccessfulOutcome_t));
  successoutcome->procedureCode = ProcedureCode_id_RICsubscription;
  successoutcome->criticality = 0;
  successoutcome->value.present = pres2;
  successoutcome->value.choice.RICsubscriptionResponse = *ricsubresp;

  E2AP_PDU_PR pres5 = E2AP_PDU_PR_successfulOutcome;
  
  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.successfulOutcome = successoutcome;

  char *error_buf = (char*)calloc(300, sizeof(char));
  size_t errlen;

  asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
  printf("error length %zu\n", errlen);
  printf("error buf %s\n", error_buf);
  
}

void generate_e2apv1_indication_request_parameterized(E2AP_PDU *e2ap_pdu,
						      long requestorId,
						      long instanceId,
						      long ranFunctionId,
						      long actionId,
						      long seqNum,
						      uint8_t *ind_header_buf,
						      int header_length,
						      uint8_t *ind_message_buf,
						      int message_length) {

  fprintf(stderr, "ind1\n");
  RICindication_IEs_t *ricind_ies = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies2 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies3 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies4 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies5 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies6 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies7 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies8 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));

  RICindication_IEs__value_PR pres3;

  pres3 = RICindication_IEs__value_PR_RICrequestID;
  ricind_ies->id = ProtocolIE_ID_id_RICrequestID;
  ricind_ies->criticality = 0;
  ricind_ies->value.present = pres3;
  ricind_ies->value.choice.RICrequestID.ricRequestorID = requestorId;
  ricind_ies->value.choice.RICrequestID.ricInstanceID = instanceId;

  fprintf(stderr, "ind2\n");

  pres3 = RICindication_IEs__value_PR_RANfunctionID;
  ricind_ies2->id = ProtocolIE_ID_id_RANfunctionID;
  ricind_ies2->criticality = 0;
  ricind_ies2->value.present = pres3;
  ricind_ies2->value.choice.RANfunctionID = ranFunctionId;

  
  ricind_ies3->id = ProtocolIE_ID_id_RICactionID;
  ricind_ies3->criticality = 0;
  pres3 =  RICindication_IEs__value_PR_RICactionID;
  ricind_ies3->value.present = pres3;
  ricind_ies3->value.choice.RICactionID = actionId;


  pres3 = RICindication_IEs__value_PR_RICindicationSN;
  ricind_ies4->id = ProtocolIE_ID_id_RICindicationSN;
  ricind_ies4->criticality = 0;
  ricind_ies4->value.present = pres3;
  ricind_ies4->value.choice.RICindicationSN = seqNum;

  //Indication type is REPORT
  pres3 = RICindication_IEs__value_PR_RICindicationType;
  ricind_ies5->id = ProtocolIE_ID_id_RICindicationType;
  ricind_ies5->criticality = 0;
  ricind_ies5->value.present = pres3;
  ricind_ies5->value.choice.RICindicationType = 0;


  uint8_t *buf2 = (uint8_t *)"reportheader";
  OCTET_STRING_t *hdr_str = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));

  hdr_str->buf = (uint8_t*)calloc(1,header_length);
  hdr_str->size = header_length;
  memcpy(hdr_str->buf, ind_header_buf, header_length);

  fprintf(stderr, "ind3\n");

  ricind_ies6->value.choice.RICindicationHeader.buf = (uint8_t*)calloc(1,header_length);

  pres3 = RICindication_IEs__value_PR_RICindicationHeader;
  ricind_ies6->id = ProtocolIE_ID_id_RICindicationHeader;
  ricind_ies6->criticality = 0;
  ricind_ies6->value.present = pres3;
  ricind_ies6->value.choice.RICindicationHeader.size = header_length;
  memcpy(ricind_ies6->value.choice.RICindicationHeader.buf, ind_header_buf, header_length);
  
  ricind_ies7->value.choice.RICindicationMessage.buf = (uint8_t*)calloc(1,8192);


  

  pres3 = RICindication_IEs__value_PR_RICindicationMessage;
  ricind_ies7->id = ProtocolIE_ID_id_RICindicationMessage;
  fprintf(stderr, "after encoding message 1\n");

  ricind_ies7->criticality = 0;
  ricind_ies7->value.present = pres3;

  fprintf(stderr, "after encoding message 2\n");

  fprintf(stderr, "after encoding message 3\n");      
  ricind_ies7->value.choice.RICindicationMessage.size = message_length;

  fprintf(stderr, "after encoding message 4\n");
  memcpy(ricind_ies7->value.choice.RICindicationMessage.buf, ind_message_buf, message_length);

  fprintf(stderr, "after encoding message 5\n");      

  uint8_t *cpid_buf = (uint8_t *)"cpid";
  OCTET_STRING_t cpid_str;

  printf("5.1\n");

  int cpid_buf_len = strlen((char*)cpid_buf);
  pres3 = RICindication_IEs__value_PR_RICcallProcessID;
  ricind_ies8->id = ProtocolIE_ID_id_RICcallProcessID;

  ricind_ies8->criticality = 0;
  ricind_ies8->value.present = pres3;

  ricind_ies8->value.choice.RICcallProcessID.buf = (uint8_t*)calloc(1,cpid_buf_len);
  ricind_ies8->value.choice.RICcallProcessID.size = cpid_buf_len;

  memcpy(ricind_ies8->value.choice.RICcallProcessID.buf, cpid_buf, cpid_buf_len);

  printf("5.2\n");

  RICindication_t *ricindication = (RICindication_t*)calloc(1, sizeof(RICindication_t));

  
  int ret;

  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies);
  
  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies2);

  printf("5.3\n");

  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies3);

  printf("5.35\n");
  
  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies4);

  printf("5.36\n");
  
  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies5);

  printf("5.4\n");
  
  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies6);

  printf("5.5\n");

  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies7);  
  
  //  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies8);    


  InitiatingMessage__value_PR pres4;
  pres4 = InitiatingMessage__value_PR_RICindication;
  InitiatingMessage_t *initmsg = (InitiatingMessage_t*)calloc(1, sizeof(InitiatingMessage_t));
  initmsg->procedureCode = 5;
  initmsg->criticality = 1;
  initmsg->value.present = pres4;
  initmsg->value.choice.RICindication = *ricindication;

  E2AP_PDU_PR pres5;
  pres5 = E2AP_PDU_PR_initiatingMessage;
  
  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.initiatingMessage = initmsg;

  char *error_buf = (char*)calloc(300, sizeof(char));
  size_t errlen;

  asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
  printf("error length %zu\n", errlen);
  printf("error buf %s\n", error_buf);

  xer_fprint(stderr, &asn_DEF_E2AP_PDU, e2ap_pdu);  

}

void generate_e2apv1_indication_request(E2AP_PDU *e2ap_pdu) {
  fprintf(stderr, "ind1\n");
  RICindication_IEs_t *ricind_ies = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies2 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies3 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies4 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies5 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies6 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies7 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));
  RICindication_IEs_t *ricind_ies8 = (RICindication_IEs_t*)calloc(1, sizeof(RICindication_IEs_t));

  RICindication_IEs__value_PR pres3;

  pres3 = RICindication_IEs__value_PR_RICrequestID;
  ricind_ies->id = ProtocolIE_ID_id_RICrequestID;
  ricind_ies->criticality = 0;
  ricind_ies->value.present = pres3;
  ricind_ies->value.choice.RICrequestID.ricRequestorID = 25;
  ricind_ies->value.choice.RICrequestID.ricInstanceID = 3;

  fprintf(stderr, "ind2\n");  

  pres3 = RICindication_IEs__value_PR_RANfunctionID;
  ricind_ies2->id = ProtocolIE_ID_id_RANfunctionID;
  ricind_ies2->criticality = 0;
  ricind_ies2->value.present = pres3;
  ricind_ies2->value.choice.RANfunctionID = 70;

  
  ricind_ies3->id = ProtocolIE_ID_id_RICactionID;
  ricind_ies3->criticality = 0;
  pres3 =  RICindication_IEs__value_PR_RICactionID;
  ricind_ies3->value.present = pres3;
  ricind_ies3->value.choice.RICactionID = 80;


  pres3 = RICindication_IEs__value_PR_RICindicationSN;
  ricind_ies4->id = ProtocolIE_ID_id_RICindicationSN;
  ricind_ies4->criticality = 0;
  ricind_ies4->value.present = pres3;
  ricind_ies4->value.choice.RICindicationSN = 45;

  pres3 = RICindication_IEs__value_PR_RICindicationType;
  ricind_ies5->id = ProtocolIE_ID_id_RICindicationType;
  ricind_ies5->criticality = 0;
  ricind_ies5->value.present = pres3;
  ricind_ies5->value.choice.RICindicationType = 0;


  uint8_t *buf2 = (uint8_t *)"reportheader";
  OCTET_STRING_t *hdr_str = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));
  hdr_str->buf = (uint8_t*)calloc(1,12);
  hdr_str->size = 12;
  memcpy(hdr_str->buf, buf2, 12);

  fprintf(stderr, "ind3\n");

  ricind_ies6->value.choice.RICindicationHeader.buf = (uint8_t*)calloc(1,12);

  pres3 = RICindication_IEs__value_PR_RICindicationHeader;
  ricind_ies6->id = ProtocolIE_ID_id_RICindicationHeader;
  ricind_ies6->criticality = 0;
  ricind_ies6->value.present = pres3;
  ricind_ies6->value.choice.RICindicationHeader.size = 12;
  memcpy(ricind_ies6->value.choice.RICindicationHeader.buf, buf2, 12);
  
  ricind_ies7->value.choice.RICindicationMessage.buf = (uint8_t*)calloc(1,8192);
  //  uint8_t *buf9 = (uint8_t *)"reportmsg";

  /*
  E2SM_KPM_IndicationMessage_t *e2sm_ind_msg =
    (E2SM_KPM_IndicationMessage_t*)calloc(1,sizeof(E2SM_KPM_IndicationMessage_t));

  encode_kpm(e2sm_ind_msg);
  */

  E2SM_KPM_RANfunction_Description_t *e2sm_desc =
    (E2SM_KPM_RANfunction_Description_t*)calloc(1,sizeof(E2SM_KPM_RANfunction_Description_t));

  encode_kpm_function_description(e2sm_desc);
  
  
  uint8_t e2smbuffer[8192];
  size_t e2smbuffer_size = 8192;

  asn_codec_ctx_t *opt_cod;

  
  asn_enc_rval_t er =
    asn_encode_to_buffer(opt_cod,
			 ATS_ALIGNED_BASIC_PER,
			 &asn_DEF_E2SM_KPM_RANfunction_Description,
			 e2sm_desc, e2smbuffer, e2smbuffer_size);
    
    /*
    asn_encode_to_buffer(opt_cod,
			 ATS_ALIGNED_BASIC_PER,
			 &asn_DEF_E2SM_KPM_IndicationMessage,
			 e2sm_ind_msg, e2smbuffer, e2smbuffer_size);    
    */


  
  fprintf(stderr, "er encded is %zu\n", er.encoded);
  fprintf(stderr, "after encoding message\n");
  
  OCTET_STRING_t *msg_str = (OCTET_STRING_t*)calloc(1,sizeof(OCTET_STRING_t));
  msg_str->buf = (uint8_t*)calloc(1,er.encoded);
  msg_str->size = er.encoded;
  memcpy(msg_str->buf, e2smbuffer, er.encoded);
  

  pres3 = RICindication_IEs__value_PR_RICindicationMessage;
  ricind_ies7->id = ProtocolIE_ID_id_RICindicationMessage;
  fprintf(stderr, "after encoding message 1\n");

  ricind_ies7->criticality = 0;
  ricind_ies7->value.present = pres3;

  fprintf(stderr, "after encoding message 2\n");

  fprintf(stderr, "after encoding message 3\n");      
  ricind_ies7->value.choice.RICindicationMessage.size = er.encoded;

  fprintf(stderr, "after encoding message 4\n");      
  memcpy(ricind_ies7->value.choice.RICindicationMessage.buf, e2smbuffer, er.encoded);

  fprintf(stderr, "after encoding message 5\n");      

  uint8_t *buf4 = (uint8_t *)"cpid";
  OCTET_STRING_t cpid_str;
  cpid_str.buf = buf4;
  cpid_str.size = 4;      


  pres3 = RICindication_IEs__value_PR_RICcallProcessID;
  ricind_ies8->id = ProtocolIE_ID_id_RICcallProcessID;

  ricind_ies8->criticality = 0;
  ricind_ies8->value.present = pres3;

  ricind_ies8->value.choice.RICcallProcessID = cpid_str;


  RICindication_t *ricindication = (RICindication_t*)calloc(1, sizeof(RICindication_t));

  
  int ret;
  /*
    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies);

    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies2);

    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies3);  
    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies4);
    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies5);  

    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies6);
  */
    ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies7);  

  ret = ASN_SEQUENCE_ADD(&ricindication->protocolIEs.list, ricind_ies8);    


  InitiatingMessage__value_PR pres4;
  pres4 = InitiatingMessage__value_PR_RICindication;
  InitiatingMessage_t *initmsg = (InitiatingMessage_t*)calloc(1, sizeof(InitiatingMessage_t));
  initmsg->procedureCode = 5;
  initmsg->criticality = 1;
  initmsg->value.present = pres4;
  initmsg->value.choice.RICindication = *ricindication;

  E2AP_PDU_PR pres5;
  pres5 = E2AP_PDU_PR_initiatingMessage;
  
  e2ap_pdu->present = pres5;
  e2ap_pdu->choice.initiatingMessage = initmsg;

  char *error_buf = (char*)calloc(300, sizeof(char));;
  size_t errlen;  

  asn_check_constraints(&asn_DEF_E2AP_PDU, e2ap_pdu, error_buf, &errlen);
  printf("error length %zu\n", errlen);
  printf("error buf %s\n", error_buf);

  xer_fprint(stderr, &asn_DEF_E2AP_PDU, e2ap_pdu);
}


void generate_e2apv1_indication_response(E2AP_PDU *e2ap_pdu) {


}

ssize_t e2sm_encode_ric_event_trigger_definition(void *buffer, size_t buf_size, size_t event_trigger_count, long *RT_periods) {
	E2SM_KPM_EventTriggerDefinition_t *eventTriggerDef = (E2SM_KPM_EventTriggerDefinition_t *)calloc(1, sizeof(E2SM_KPM_EventTriggerDefinition_t));
	if(!eventTriggerDef) {
        fprintf(stderr, "alloc EventTriggerDefinition failed\n");
        return -1;
	}

	eventTriggerDef->present = E2SM_KPM_EventTriggerDefinition_PR_eventDefinition_Format1;

	struct E2SM_KPM_EventTriggerDefinition_Format1__policyTest_List *policyTestList = (struct E2SM_KPM_EventTriggerDefinition_Format1__policyTest_List *)calloc(1, sizeof(struct E2SM_KPM_EventTriggerDefinition_Format1__policyTest_List));

	int index = 0;
	while(index < event_trigger_count) {
        Trigger_ConditionIE_Item_t *triggerCondition = (Trigger_ConditionIE_Item_t *)calloc(1, sizeof(Trigger_ConditionIE_Item_t));
        assert(triggerCondition != 0);
        triggerCondition->report_Period_IE = RT_periods[index];
        ASN_SEQUENCE_ADD(&policyTestList->list, triggerCondition); index++;
	}

    eventTriggerDef->choice.eventDefinition_Format1.policyTest_List = policyTestList;

	asn_enc_rval_t encode_result;
    encode_result = aper_encode_to_buffer(&asn_DEF_E2SM_KPM_EventTriggerDefinition, NULL, eventTriggerDef, buffer, buf_size);
    ASN_STRUCT_FREE(asn_DEF_E2SM_KPM_EventTriggerDefinition, eventTriggerDef);
    if(encode_result.encoded == -1) {
        fprintf(stderr, "Cannot encode %s: %s\n", encode_result.failed_type->name, strerror(errno));
        return -1;
    } else {
	    return encode_result.encoded;
	}
}

ssize_t e2sm_encode_ric_action_definition(void *buffer, size_t buf_size, long ric_style_type) {
	E2SM_KPM_ActionDefinition_t *actionDef = (E2SM_KPM_ActionDefinition_t *)calloc(1, sizeof(E2SM_KPM_ActionDefinition_t));
	if(!actionDef) {
		fprintf(stderr, "alloc RIC ActionDefinition failed\n");
		return -1;
	}

	actionDef->ric_Style_Type = ric_style_type;

	asn_enc_rval_t encode_result;
    encode_result = aper_encode_to_buffer(&asn_DEF_E2SM_KPM_ActionDefinition, NULL, actionDef, buffer, buf_size);
    ASN_STRUCT_FREE(asn_DEF_E2SM_KPM_ActionDefinition, actionDef);
	if(encode_result.encoded == -1) {
	    fprintf(stderr, "Cannot encode %s: %s\n", encode_result.failed_type->name, strerror(errno));
	    return -1;
	} else {
    	return encode_result.encoded;
    }
}
