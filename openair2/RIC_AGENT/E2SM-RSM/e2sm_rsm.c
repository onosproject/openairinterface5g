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

#include <sys/time.h>
#include <arpa/inet.h>
#include <stdbool.h>
#include <string.h>
#include "common/utils/assertions.h"
#include "f1ap_common.h"
#include "ric_agent.h"
#include "e2ap_encoder.h"
#include "e2ap_generate_messages.h"
#include "e2sm_rsm.h"
#include "E2SM_RSM_E2SM-RSM-RANfunction-Description.h"
#include "E2SM_RSM_NodeSlicingCapability-Item.h"
#include "E2SM_RSM_SupportedSlicingConfig-Item.h"
#include "E2SM_RSM_E2SM-RSM-IndicationHeader.h"
#include "E2SM_RSM_UE-Identity.h"
#include "E2SM_RSM_E2SM-RSM-IndicationMessage-Format2.h"
#include "E2SM_RSM_E2SM-RSM-IndicationMessage.h"
#include "E2SM_RSM_Bearer-ID.h"

static int e2sm_rsm_subscription_add(ric_agent_info_t *ric, ric_subscription_t *sub);
static int e2sm_rsm_subscription_del(ric_agent_info_t *ric, ric_subscription_t *sub, int force,long *cause,long *cause_detail);
extern f1ap_cudu_inst_t f1ap_cu_inst[MAX_eNB];

static ric_service_model_t e2sm_rsm_model = {
    .name = "e2sm_rsm-v1",
    /* iso(1) identified-organization(3) dod(6) internet(1) private(4) enterprise(1) oran(53148) e2(1) version1 (1) e2sm(2) e2sm-RSM-IEs (102) */
    .oid = "1.3.6.1.4.1.53148.1.1.2.102",
    .handle_subscription_add = e2sm_rsm_subscription_add,
    .handle_subscription_del = e2sm_rsm_subscription_del,
//    .handle_control = e2sm_rsm_control,
//    .handle_ricInd_timer_expiry = e2sm_rsm_ricInd_timer_expiry,
//    .handle_gp_timer_expiry = e2sm_rsm_gp_timer_expiry
};

uint8_t rsm_emm_event_trigger;

int e2sm_rsm_init(e2node_type_t e2node_type)
{
    //uint16_t i;
    ric_ran_function_t *func;   
    E2SM_RSM_E2SM_RSM_RANfunction_Description_t *func_def;
    E2SM_RSM_NodeSlicingCapability_Item_t       *rsm_node_slicing_cap_item;
    E2SM_RSM_SupportedSlicingConfig_Item_t      *rsm_supported_slicing_cfg_create;
    E2SM_RSM_SupportedSlicingConfig_Item_t      *rsm_supported_slicing_cfg_update;
    E2SM_RSM_SupportedSlicingConfig_Item_t      *rsm_supported_slicing_cfg_delete;
    E2SM_RSM_SupportedSlicingConfig_Item_t      *rsm_supported_slicing_cfg_ueAssoc;
    E2SM_RSM_SupportedSlicingConfig_Item_t      *rsm_supported_slicing_cfg_eventTrigger;

    func = (ric_ran_function_t *)calloc(1, sizeof(*func));
    func->model = &e2sm_rsm_model;
    func->revision = 1;
    func->name = "ORAN-E2SM-RSM";
    func->description = "RAN Slicing";

    func_def = (E2SM_RSM_E2SM_RSM_RANfunction_Description_t *)calloc(1, sizeof(*func_def));
    /* RAN Function Name */
    func_def->ranFunction_Name.ranFunction_ShortName.buf = (uint8_t *)strdup(func->name);
    func_def->ranFunction_Name.ranFunction_ShortName.size = strlen(func->name);
    func_def->ranFunction_Name.ranFunction_E2SM_OID.buf = (uint8_t *)strdup(func->model->oid);
    func_def->ranFunction_Name.ranFunction_E2SM_OID.size = strlen(func->model->oid);
    func_def->ranFunction_Name.ranFunction_Description.buf = (uint8_t *)strdup(func->description);
    func_def->ranFunction_Name.ranFunction_Description.size = strlen(func->description);
    
    /* Hack for E2t crash */
    long *ranFuncInst;
    ranFuncInst = (long *)calloc(1,sizeof(*func_def->ranFunction_Name.ranFunction_Instance));
    *ranFuncInst = 0;
    func_def->ranFunction_Name.ranFunction_Instance = ranFuncInst;

    rsm_node_slicing_cap_item = (E2SM_RSM_NodeSlicingCapability_Item_t *)calloc(1, sizeof(*rsm_node_slicing_cap_item));
    /* Node Slicing Capability */
    rsm_node_slicing_cap_item->maxNumberOfSlicesDL = 4; 
    rsm_node_slicing_cap_item->maxNumberOfSlicesUL = 4;
    rsm_node_slicing_cap_item->slicingType = E2SM_RSM_SlicingType_static;
    rsm_node_slicing_cap_item->maxNumberOfUEsPerSlice = 4;

    /* Supported Slicing Configgurations */
    if (e2node_type == E2NODE_TYPE_ENB_DU)
    {
        rsm_supported_slicing_cfg_create = (E2SM_RSM_SupportedSlicingConfig_Item_t *)calloc(1, sizeof(*rsm_supported_slicing_cfg_create));
        rsm_supported_slicing_cfg_create->slicingConfigType = E2SM_RSM_E2SM_RSM_Command_sliceCreate;
        ASN_SEQUENCE_ADD(&rsm_node_slicing_cap_item->supportedConfig.list, rsm_supported_slicing_cfg_create);

        rsm_supported_slicing_cfg_update = (E2SM_RSM_SupportedSlicingConfig_Item_t *)calloc(1, sizeof(*rsm_supported_slicing_cfg_update));
        rsm_supported_slicing_cfg_update->slicingConfigType = E2SM_RSM_E2SM_RSM_Command_sliceUpdate;
        ASN_SEQUENCE_ADD(&rsm_node_slicing_cap_item->supportedConfig.list, rsm_supported_slicing_cfg_update);

        rsm_supported_slicing_cfg_delete = (E2SM_RSM_SupportedSlicingConfig_Item_t *)calloc(1, sizeof(*rsm_supported_slicing_cfg_delete));
        rsm_supported_slicing_cfg_delete->slicingConfigType = E2SM_RSM_E2SM_RSM_Command_sliceDelete;
        ASN_SEQUENCE_ADD(&rsm_node_slicing_cap_item->supportedConfig.list, rsm_supported_slicing_cfg_delete);

        rsm_supported_slicing_cfg_ueAssoc = (E2SM_RSM_SupportedSlicingConfig_Item_t *)calloc(1, sizeof(*rsm_supported_slicing_cfg_ueAssoc));
        rsm_supported_slicing_cfg_ueAssoc->slicingConfigType = E2SM_RSM_E2SM_RSM_Command_ueAssociate;
        ASN_SEQUENCE_ADD(&rsm_node_slicing_cap_item->supportedConfig.list, rsm_supported_slicing_cfg_ueAssoc);
    }
    else if (e2node_type == E2NODE_TYPE_ENB_CU)
    {
        rsm_supported_slicing_cfg_eventTrigger = (E2SM_RSM_SupportedSlicingConfig_Item_t *)calloc(1, sizeof(*rsm_supported_slicing_cfg_eventTrigger));
        rsm_supported_slicing_cfg_eventTrigger->slicingConfigType = E2SM_RSM_E2SM_RSM_Command_eventTriggers;
        ASN_SEQUENCE_ADD(&rsm_node_slicing_cap_item->supportedConfig.list, rsm_supported_slicing_cfg_eventTrigger);
    }
    else
    {
        RIC_AGENT_ERROR("INCORRECT NODE TYPE:%d\n",e2node_type);
        return -1;
    }
 
    ASN_SEQUENCE_ADD(&func_def->ric_Slicing_Node_Capability_List.list, rsm_node_slicing_cap_item);

    //xer_fprint(stderr, &asn_DEF_E2SM_RSM_E2SM_RSM_RANfunction_Description, func_def);

    RIC_AGENT_INFO("_______\n"); 
    func->enc_definition_len = e2ap_encode(&asn_DEF_E2SM_RSM_E2SM_RSM_RANfunction_Description,0, func_def,&func->enc_definition);
    RIC_AGENT_INFO("_______\n");

    RIC_AGENT_INFO("------ RAN SLICING FUNC DEF ENC Len:%lu-------\n", func->enc_definition_len);

    if (func->enc_definition_len < 0) {
        RIC_AGENT_ERROR("failed to encode RANfunction_List in E2SM RSM func description; aborting!");
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2SM_RSM_E2SM_RSM_RANfunction_Description, func_def);
        free(func_def);
        free(func);
        return -1;
    }

    func->enabled = 1;
    func->definition = func_def;

#if 0   
    /* Test code */
    E2SM_RSM_E2SM_RSM_RANfunction_Description_t *func_defi;
    asn_dec_rval_t decode_result;
    decode_result = aper_decode_complete(NULL, &asn_DEF_E2SM_RSM_E2SM_RSM_RANfunction_Description,
                                               (void **)&func_defi, func->enc_definition, func->enc_definition_len);
    DevAssert(decode_result.code == RC_OK);
#endif

    return ric_agent_register_ran_function(func);
}

static int e2sm_rsm_subscription_add(ric_agent_info_t *ric, ric_subscription_t *sub)
{
  /* XXX: process E2SM content. */
  if (LIST_EMPTY(&ric->subscription_list)) {
    LIST_INSERT_HEAD(&ric->subscription_list,sub,subscriptions);
  }
  else {
    LIST_INSERT_BEFORE(LIST_FIRST(&ric->subscription_list),sub,subscriptions);
  }
  rsm_emm_event_trigger = 1;
  ric->e2sm_rsm_function_id = sub->function_id;
  ric->e2sm_rsm_request_id = sub->request_id;
  ric->e2sm_rsm_instance_id = sub->instance_id;
  RIC_AGENT_INFO("RSM Subscription Added Successfully %d!\n",rsm_emm_event_trigger);
  return 0;
}

static int e2sm_rsm_subscription_del(ric_agent_info_t *ric, ric_subscription_t *sub, int force,long *cause,long *cause_detail)
{
    LIST_REMOVE(sub, subscriptions);
    ric_free_subscription(sub);
    rsm_emm_event_trigger = 0;
    return 0;
}

void encode_e2sm_rsm_indication_header(ranid_t ranid, E2SM_RSM_E2SM_RSM_IndicationHeader_t *ihead) 
{
    e2node_type_t node_type;
    ihead->present = E2SM_RSM_E2SM_RSM_IndicationHeader_PR_indicationHeader_Format1;
    
    E2SM_RSM_E2SM_RSM_IndicationHeader_Format1_t* ind_header = &ihead->choice.indicationHeader_Format1;
    
    node_type = e2_conf[ranid]->e2node_type;
    
    //printf("node_type:%d\n",node_type);    
    if (node_type == E2NODE_TYPE_ENB_CU)
    {
        MCC_MNC_TO_PLMNID(
                e2_conf[ranid]->mcc,
                e2_conf[ranid]->mnc,
                e2_conf[ranid]->mnc_digit_length,
                &ind_header->cgi.choice.eUTRA_CGI.pLMNIdentity);
    //printf("mcc:%d mnc:%d len:%d plmnId:%s\n", e2_conf[ranid]->mcc, e2_conf[ranid]->mnc, e2_conf[ranid]->mnc_digit_length,ind_header->cgi.choice.eUTRA_CGI.pLMNIdentity.buf);
        ind_header->cgi.present = E2SM_RSM_CGI_PR_eUTRA_CGI;
            
        //MACRO_ENB_ID_TO_BIT_STRING(
        MACRO_ENB_ID_TO_CELL_IDENTITY(
                e2_conf[ranid]->cell_identity,0,
                &ind_header->cgi.choice.eUTRA_CGI.eUTRACellIdentity);
    //printf("cellid:%d eutraCellId:%s\n", e2_conf[ranid]->cell_identity, ind_header->cgi.choice.eUTRA_CGI.eUTRACellIdentity.buf);
    }

    /* Collect Start Time Stamp */
    /* Encoded in the same format as the first four octets of the 64-bit timestamp format as defined in section 6 of IETF RFC 5905 */
    //ind_header->colletStartTime.buf = (uint8_t *)calloc(1, 4);
    //ind_header->colletStartTime.size = 4;
    //*((uint32_t *)(ind_header->colletStartTime.buf)) = htonl((uint32_t)time(NULL));
    xer_fprint(stderr, &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationHeader, ihead);
}


static E2SM_RSM_E2SM_RSM_IndicationMessage_t*
encode_rsm_Indication_Msg(ric_agent_info_t* ric, ric_subscription_t *rs, ueStatusInd *emmTriggerBuff, uint16_t trigger_type)
{
    int ret;
    E2SM_RSM_UE_Identity_t* rsm_ue_id[5];

    E2SM_RSM_E2SM_RSM_IndicationMessage_Format2_t* format = 
                        (E2SM_RSM_E2SM_RSM_IndicationMessage_Format2_t*)calloc(1, sizeof(E2SM_RSM_E2SM_RSM_IndicationMessage_Format2_t));
    ASN_STRUCT_RESET(asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage_Format2, format);

    if (trigger_type == UE_ATTACH_EVENT_TRIGGER)
    { 
        format->emmType = E2SM_RSM_RSM_EMM_Trigger_Type_ueAttach;
        format->prefferedUeIDType = E2SM_RSM_UE_ID_Type_duUeF1ApID;

        rsm_ue_id[0] = (E2SM_RSM_UE_Identity_t *)calloc(1, sizeof(E2SM_RSM_UE_Identity_t));
        rsm_ue_id[0]->present = E2SM_RSM_UE_Identity_PR_cuUeF1ApID;
        rsm_ue_id[0]->choice.cuUeF1ApID = f1ap_get_cu_ue_f1ap_id(&f1ap_cu_inst[0],//emmTriggerBuff->cu_ue_f1ap_id;
                                                                 emmTriggerBuff->rnti);
    //RIC_AGENT_INFO("cuUeF1ApID:%lu %lu %u\n",rsm_ue_id[0]->choice.cuUeF1ApID, rs->instance_id, emmTriggerBuff->rnti);
        ret = ASN_SEQUENCE_ADD(&format->ueIDlist.list, rsm_ue_id[0]);
        DevAssert(ret == 0);

        rsm_ue_id[1] = (E2SM_RSM_UE_Identity_t *)calloc(1, sizeof(E2SM_RSM_UE_Identity_t));
        rsm_ue_id[1]->present = E2SM_RSM_UE_Identity_PR_duUeF1ApID;
        rsm_ue_id[1]->choice.duUeF1ApID = f1ap_get_du_ue_f1ap_id(&f1ap_cu_inst[0],//emmTriggerBuff->du_ue_f1ap_id;
                                                                 emmTriggerBuff->rnti);
    //RIC_AGENT_INFO("duUeF1ApID:%lu %lu %u\n",rsm_ue_id[1]->choice.duUeF1ApID, rs->instance_id, emmTriggerBuff->rnti);
        ret = ASN_SEQUENCE_ADD(&format->ueIDlist.list, rsm_ue_id[1]);
        DevAssert(ret == 0);

        rsm_ue_id[2] = (E2SM_RSM_UE_Identity_t *)calloc(1, sizeof(E2SM_RSM_UE_Identity_t));
        rsm_ue_id[2]->present = E2SM_RSM_UE_Identity_PR_enbUeS1ApID;
        rsm_ue_id[2]->choice.enbUeS1ApID = emmTriggerBuff->eNB_ue_s1ap_id;
    
    //RIC_AGENT_INFO("enbUeS1ApID:%lu\n",rsm_ue_id[2]->choice.enbUeS1ApID);
        ret = ASN_SEQUENCE_ADD(&format->ueIDlist.list, rsm_ue_id[2]);
        DevAssert(ret == 0);

        /* E2SM_RSM_UE_Identity_t struct need to be enhanced to include mme_ue_s1ap_id */

        E2SM_RSM_Bearer_ID_t* rsm_bearer_info = (E2SM_RSM_Bearer_ID_t *)calloc(1, sizeof(E2SM_RSM_Bearer_ID_t));
        rsm_bearer_info->present = E2SM_RSM_Bearer_ID_PR_drbID;
        rsm_bearer_info->choice.drbID.present = E2SM_RSM_Drb_ID_PR_fourGDrbID;
        rsm_bearer_info->choice.drbID.choice.fourGDrbID.value = emmTriggerBuff->e_rab_id; 
        rsm_bearer_info->choice.drbID.choice.fourGDrbID.qci = emmTriggerBuff->qci;

        format->bearerID = (struct E2SM_RSM_E2SM_RSM_IndicationMessage_Format2__bearerID *)calloc(1, 
                                            sizeof(struct E2SM_RSM_E2SM_RSM_IndicationMessage_Format2__bearerID));
        ret = ASN_SEQUENCE_ADD(&format->bearerID->list, rsm_bearer_info);
        DevAssert(ret == 0);
    }
    else if (trigger_type == UE_DETACH_EVENT_TRIGGER)
    {
        format->emmType = E2SM_RSM_RSM_EMM_Trigger_Type_ueDetach;
        format->prefferedUeIDType = E2SM_RSM_UE_ID_Type_enbUeS1ApID;

        rsm_ue_id[0] = (E2SM_RSM_UE_Identity_t *)calloc(1, sizeof(E2SM_RSM_UE_Identity_t));
        rsm_ue_id[0]->present = E2SM_RSM_UE_Identity_PR_enbUeS1ApID;
        rsm_ue_id[0]->choice.enbUeS1ApID = emmTriggerBuff->eNB_ue_s1ap_id;
    //RIC_AGENT_INFO("enbUeS1ApID:%lu\n", rsm_ue_id[0]->choice.enbUeS1ApID);
        ret = ASN_SEQUENCE_ADD(&format->ueIDlist.list, rsm_ue_id[0]);
        DevAssert(ret == 0);

        E2SM_RSM_Bearer_ID_t* rsm_bearer_info = (E2SM_RSM_Bearer_ID_t *)calloc(1, sizeof(E2SM_RSM_Bearer_ID_t));
        //rsm_bearer_info->present = E2SM_RSM_Bearer_ID_PR_NOTHING;
        rsm_bearer_info->present = E2SM_RSM_Bearer_ID_PR_drbID;
        rsm_bearer_info->choice.drbID.present = E2SM_RSM_Drb_ID_PR_fourGDrbID;
        rsm_bearer_info->choice.drbID.choice.fourGDrbID.value = emmTriggerBuff->e_rab_id;
        rsm_bearer_info->choice.drbID.choice.fourGDrbID.qci = emmTriggerBuff->qci;

        format->bearerID = (struct E2SM_RSM_E2SM_RSM_IndicationMessage_Format2__bearerID *)calloc(1,
                                            sizeof(struct E2SM_RSM_E2SM_RSM_IndicationMessage_Format2__bearerID));
        ret = ASN_SEQUENCE_ADD(&format->bearerID->list, rsm_bearer_info);
        DevAssert(ret == 0);
    }

    /*
     * IndicationMessage -> IndicationMessage_Format1
     */
    E2SM_RSM_E2SM_RSM_IndicationMessage_t* indicationmessage = 
                                (E2SM_RSM_E2SM_RSM_IndicationMessage_t*)calloc(1, sizeof(E2SM_RSM_E2SM_RSM_IndicationMessage_t));
    indicationmessage->present = E2SM_RSM_E2SM_RSM_IndicationMessage_PR_indicationMessage_Format2;
    indicationmessage->choice.indicationMessage_Format2 = *format;
    
    return indicationmessage;
}

int e2sm_rsm_ricInd(
        ric_agent_info_t *ric,
        ric_ran_function_id_t function_id,
        long request_id,
        long instance_id,
        uint16_t trigger_type,
        ueStatusInd *emmTriggerBuff,
        uint8_t **outbuf,
        uint32_t *outlen)
{

    E2SM_RSM_E2SM_RSM_IndicationMessage_t* indicationmessage;
    ric_subscription_t *rs;

    RIC_AGENT_INFO("----  Reporting Event[%d] RSM RIC Ind, function_id %ld ranId:%d---------\n", 
                   trigger_type, function_id, ric->ranid);

    /* Fetch the RIC Subscription */
    rs = ric_agent_lookup_subscription(ric,request_id,instance_id,function_id);
    if (!rs) {
        RIC_AGENT_ERROR("failed to find subscription %ld/%ld/%ld\n", request_id,instance_id,function_id);
    }

    indicationmessage = encode_rsm_Indication_Msg(ric, rs, emmTriggerBuff, trigger_type);

    {
        char *error_buf = (char*)calloc(300, sizeof(char));
        size_t errlen;
        asn_check_constraints(&asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage, indicationmessage, error_buf, &errlen);
        fprintf(stderr,"RSM IND error length %zu\n", errlen);
        fprintf(stderr,"RSM IND error buf %s\n", error_buf);
        free(error_buf);
        //xer_fprint(stderr, &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage, indicationmessage);
    }

    xer_fprint(stderr, &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage, indicationmessage);
    uint8_t e2smbuffer[8192];
    size_t e2smbuffer_size = 8192;

    asn_enc_rval_t er = asn_encode_to_buffer(NULL,
            ATS_ALIGNED_BASIC_PER,
            &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage,
            indicationmessage, e2smbuffer, e2smbuffer_size);

    fprintf(stderr, "er encded is %zu %ld\n", er.encoded, er.encoded);
    fprintf(stderr, "after encoding RSM IND message\n");

#if 0
    /* Test code */
    printf("-----test code-----\n");
    E2SM_RSM_E2SM_RSM_IndicationMessage_t *func_defi;
    asn_dec_rval_t decode_result;
    uint8_t enc_def[11] = {0x41,0x00,0x01,0x10,0x01,0x40,0x01,0x10,0x01,0x00,0x09};
    decode_result = aper_decode_complete(NULL, &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage,
                                               (void **)&func_defi, enc_def, 11);
    DevAssert(decode_result.code == RC_OK);
    xer_fprint(stderr, &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationMessage, func_defi);
    printf("-----test code-----\n");
#endif

    E2AP_E2AP_PDU_t *e2ap_pdu = (E2AP_E2AP_PDU_t*)calloc(1, sizeof(E2AP_E2AP_PDU_t));

    E2SM_RSM_E2SM_RSM_IndicationHeader_t* ind_header_style1 =
        (E2SM_RSM_E2SM_RSM_IndicationHeader_t*)calloc(1,sizeof(E2SM_RSM_E2SM_RSM_IndicationHeader_t));

    encode_e2sm_rsm_indication_header(ric->ranid, ind_header_style1);

    uint8_t e2sm_header_buf_style1[8192];
    size_t e2sm_header_buf_size_style1 = 8192;
    asn_enc_rval_t er_header_style1 = asn_encode_to_buffer(
            NULL,
            ATS_ALIGNED_BASIC_PER,
            &asn_DEF_E2SM_RSM_E2SM_RSM_IndicationHeader,
            ind_header_style1,
            e2sm_header_buf_style1,
            e2sm_header_buf_size_style1);

    if (er_header_style1.encoded < 0) {
        fprintf(stderr, "ERROR encoding indication header, name=%s, tag=%s", er_header_style1.failed_type->name, er_header_style1.failed_type->xml_tag);
    }

    DevAssert(er_header_style1.encoded >= 0);

    // TODO - remove hardcoded values
    generate_e2apv1_indication_request_parameterized(
            e2ap_pdu, request_id, instance_id, function_id, 0,
            0, e2sm_header_buf_style1, er_header_style1.encoded,
            e2smbuffer, er.encoded);

    *outlen = e2ap_asn1c_encode_pdu(e2ap_pdu, outbuf);

    return 0;
}





























