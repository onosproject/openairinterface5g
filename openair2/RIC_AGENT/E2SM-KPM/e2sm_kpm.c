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
#include "e2sm_kpm.h"
#include "e2ap_generate_messages.h"

#include "E2AP_Cause.h"
#include "E2SM_KPM_E2SM-KPMv2-RANfunction-Description.h"
#include "E2SM_KPM_RIC-KPMNode-Item-KPMv2.h"
#include "E2SM_KPM_Cell-Measurement-Object-Item-KPMv2.h"
#include "E2SM_KPM_RIC-EventTriggerStyle-Item-KPMv2.h"
#include "E2SM_KPM_RIC-ReportStyle-Item-KPMv2.h"
#include "E2SM_KPM_MeasurementInfo-Action-Item-KPMv2.h"
#include "E2SM_KPM_E2SM-KPMv2-ActionDefinition-Format1.h"
#include "E2SM_KPM_MeasurementInfoItem-KPMv2.h"
#include "E2SM_KPM_E2SM-KPMv2-ActionDefinition.h"
#include "E2SM_KPM_MeasurementRecord-KPMv2.h"
#include "E2SM_KPM_MeasurementRecordItem-KPMv2.h"
#include "E2SM_KPM_E2SM-KPMv2-IndicationMessage.h"
#include "E2SM_KPM_MeasurementDataItem-KPMv2.h"
#include "E2AP_ProtocolIE-Field.h"
#include "E2SM_KPM_E2SM-KPMv2-IndicationHeader.h"
#include "E2SM_KPM_SNSSAI-KPMv2.h"
#include "E2SM_KPM_GlobalKPMnode-ID-KPMv2.h"
#include "E2SM_KPM_GNB-ID-Choice-KPMv2.h"
#include "E2SM_KPM_NRCGI-KPMv2.h"

extern f1ap_cudu_inst_t f1ap_cu_inst[MAX_eNB];
extern int global_e2_node_id(ranid_t ranid, E2AP_GlobalE2node_ID_t* node_id);
extern RAN_CONTEXT_t RC;
extern eNB_RRC_KPI_STATS    rrc_kpi_stats;

/**
 ** The main thing with this abstraction is that we need per-SM modules
 ** to handle the details of the function, event trigger, action, etc
 ** definitions... and to actually do all the logic and implement the
 ** inner parts of the message encoding.  generic e2ap handles the rest.
 **/

static int e2sm_kpm_subscription_add(ric_agent_info_t *ric, ric_subscription_t *sub);
static int e2sm_kpm_subscription_del(ric_agent_info_t *ric, ric_subscription_t *sub, int force,long *cause,long *cause_detail);
static int e2sm_kpm_control(ric_agent_info_t *ric,ric_control_t *control);
static char *time_stamp(void);
static int e2sm_kpm_ricInd_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        ric_ran_function_id_t function_id,
        long request_id,
        long instance_id,
        long action_id,
        uint8_t **outbuf,
        uint32_t *outlen);
static int e2sm_kpm_gp_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        ric_ran_function_id_t function_id,
        long request_id,
        long instance_id,
        long action_id,
        uint8_t **outbuf,
        uint32_t *outlen);
static E2SM_KPM_E2SM_KPMv2_IndicationMessage_t* encode_kpm_Indication_Msg(ric_agent_info_t* ric, ric_subscription_t *rs);
//static void generate_e2apv1_indication_request_parameterized(E2AP_E2AP_PDU_t *e2ap_pdu, long requestorId, long instanceId, long ranFunctionId, long actionId, long seqNum, uint8_t *ind_header_buf, int header_length, uint8_t *ind_message_buf, int message_length);
static void encode_e2sm_kpm_indication_header(ranid_t ranid, E2SM_KPM_E2SM_KPMv2_IndicationHeader_t *ihead);

//static int e2ap_asn1c_encode_pdu(E2AP_E2AP_PDU_t* pdu, unsigned char **buffer);

#define MAX_KPM_MEAS    5
#define MAX_GRANULARITY_INDEX   50
uint8_t g_indMsgMeasInfoCnt = 0;
uint8_t g_granularityIndx = 0;
bool action_def_missing = FALSE;
E2SM_KPM_MeasurementInfoItem_KPMv2_t *g_indMsgMeasInfoItemArr[MAX_KPM_MEAS];
E2SM_KPM_MeasurementRecordItem_KPMv2_t *g_indMsgMeasRecItemArr[MAX_GRANULARITY_INDEX][MAX_KPM_MEAS];
E2SM_KPM_GranularityPeriod_KPMv2_t     *g_granulPeriod;
E2SM_KPM_SubscriptionID_KPMv2_t    g_subscriptionID;

kmp_meas_info_t e2sm_kpm_meas_info[MAX_KPM_MEAS] = {
                                            {1, "RRC.ConnEstabAtt.sum", 0, FALSE},
                                            {2, "RRC.ConnEstabSucc.sum", 0, FALSE},
                                            {3, "RRC.ConnReEstabAtt.sum", 0, FALSE},
                                            {4, "RRC.ConnMean", 0, FALSE},
                                            {5, "RRC.ConnMax", 0, FALSE}
                                        };

static ric_service_model_t e2sm_kpm_model = {
    .name = "e2sm_kpm-v2beta1",
    /* iso(1) identified-organization(3) dod(6) internet(1) private(4) enterprise(1) oran(53148) e2(1) version2(2) e2sm(2) e2sm-KPMMON-IEs (2) */
    .oid = "1.3.6.1.4.1.53148.1.2.2.2",
    .handle_subscription_add = e2sm_kpm_subscription_add,
    .handle_subscription_del = e2sm_kpm_subscription_del,
    .handle_control = e2sm_kpm_control,
    .handle_ricInd_timer_expiry = e2sm_kpm_ricInd_timer_expiry,
    .handle_gp_timer_expiry = e2sm_kpm_gp_timer_expiry
};

/**
 * Initializes KPM state and registers KPM e2ap_ran_function_id_t number(s).
 */
int e2sm_kpm_init(void)
{
    uint16_t i;
    ric_ran_function_t *func;
    E2SM_KPM_E2SM_KPMv2_RANfunction_Description_t *func_def;
    E2SM_KPM_RIC_ReportStyle_Item_KPMv2_t *ric_report_style_item;
    E2SM_KPM_RIC_EventTriggerStyle_Item_KPMv2_t *ric_event_trigger_style_item;
    E2SM_KPM_RIC_KPMNode_Item_KPMv2_t *ric_kpm_node_item;
    E2SM_KPM_Cell_Measurement_Object_Item_KPMv2_t *cell_meas_object_item;
    E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *meas_action_item1;
    E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *meas_action_item2;
    E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *meas_action_item3;
    E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *meas_action_item4;
    E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *meas_action_item5;

    func = (ric_ran_function_t *)calloc(1, sizeof(*func));
    func->model = &e2sm_kpm_model;
    func->revision = 1;
    func->name = "ORAN-E2SM-KPM";
    func->description = "KPM monitor";


    func_def = (E2SM_KPM_E2SM_KPMv2_RANfunction_Description_t *)calloc(1, sizeof(*func_def));

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

    /* KPM Node List */
    func_def->ric_KPM_Node_List = (struct E2SM_KPM_E2SM_KPMv2_RANfunction_Description__ric_KPM_Node_List *)calloc(1, sizeof(*func_def->ric_KPM_Node_List));
    ric_kpm_node_item = (E2SM_KPM_RIC_KPMNode_Item_KPMv2_t *)calloc(1, sizeof(*ric_kpm_node_item));
    ric_kpm_node_item->ric_KPMNode_Type.present = E2SM_KPM_GlobalKPMnode_ID_KPMv2_PR_eNB; 
    
    /* Fetching PLMN ID*/
    for (i = 0; i < RC.nb_inst; ++i) { //is there a better way to fetch RANID,otherwise PLMNID of first intance will get populated ?
        if ( (e2_conf[i]->enabled) && 
             ((e2_conf[i]->e2node_type == E2NODE_TYPE_ENB_CU) || (e2_conf[i]->e2node_type == E2NODE_TYPE_NG_ENB_CU)) 
           ){
            MCC_MNC_TO_PLMNID(
                e2_conf[i]->mcc,
                e2_conf[i]->mnc,
                e2_conf[i]->mnc_digit_length,
                &ric_kpm_node_item->ric_KPMNode_Type.choice.eNB.global_eNB_ID.pLMN_Identity);
            break;
        }
    }

    /* eNB_ID */
    ric_kpm_node_item->ric_KPMNode_Type.choice.eNB.global_eNB_ID.eNB_ID.present = E2AP_ENB_ID_PR_macro_eNB_ID;
    MACRO_ENB_ID_TO_BIT_STRING(e2_conf[i]->cell_identity,
                               &ric_kpm_node_item->ric_KPMNode_Type.choice.eNB.global_eNB_ID.eNB_ID.choice.macro_eNB_ID);

    ric_kpm_node_item->cell_Measurement_Object_List = 
            (struct E2SM_KPM_RIC_KPMNode_Item_KPMv2__cell_Measurement_Object_List *)calloc(1, sizeof(*ric_kpm_node_item->cell_Measurement_Object_List));
    
    cell_meas_object_item = (E2SM_KPM_Cell_Measurement_Object_Item_KPMv2_t *)calloc(1, sizeof(*cell_meas_object_item));
    cell_meas_object_item->cell_object_ID.buf = (uint8_t *)strdup("1"); //if cell is TDD then EUtranCellTDD
    cell_meas_object_item->cell_object_ID.size = strlen("1");
    cell_meas_object_item->cell_global_ID.present = E2SM_KPM_CellGlobalID_KPMv2_PR_eUTRA_CGI;

    MCC_MNC_TO_PLMNID(e2_conf[i]->mcc,
                      e2_conf[i]->mnc,
                      e2_conf[i]->mnc_digit_length,
                      &cell_meas_object_item->cell_global_ID.choice.eUTRA_CGI.pLMN_Identity);
 
    //MACRO_ENB_ID_TO_BIT_STRING(e2_conf[i]->cell_identity,
    MACRO_ENB_ID_TO_CELL_IDENTITY(e2_conf[i]->cell_identity,0,
                               &cell_meas_object_item->cell_global_ID.choice.eUTRA_CGI.eUTRACellIdentity);

    ASN_SEQUENCE_ADD(&ric_kpm_node_item->cell_Measurement_Object_List->list, cell_meas_object_item);

    ASN_SEQUENCE_ADD(&func_def->ric_KPM_Node_List->list, ric_kpm_node_item);

    /* Sequence of Event trigger styles */
    func_def->ric_EventTriggerStyle_List = (struct E2SM_KPM_E2SM_KPMv2_RANfunction_Description__ric_EventTriggerStyle_List *)calloc(1, sizeof(*func_def->ric_EventTriggerStyle_List));
    ric_event_trigger_style_item = (E2SM_KPM_RIC_EventTriggerStyle_Item_KPMv2_t *)calloc(1, sizeof(*ric_event_trigger_style_item));
    ric_event_trigger_style_item->ric_EventTriggerStyle_Type = 1;
    ric_event_trigger_style_item->ric_EventTriggerStyle_Name.buf = (uint8_t *)strdup("Trigger1");
    ric_event_trigger_style_item->ric_EventTriggerStyle_Name.size = strlen("Trigger1");
    ric_event_trigger_style_item->ric_EventTriggerFormat_Type = 1;
    ASN_SEQUENCE_ADD(&func_def->ric_EventTriggerStyle_List->list, ric_event_trigger_style_item);

    /* Sequence of Report styles */
    func_def->ric_ReportStyle_List = (struct E2SM_KPM_E2SM_KPMv2_RANfunction_Description__ric_ReportStyle_List *)calloc(1, sizeof(*func_def->ric_ReportStyle_List));
    ric_report_style_item = (E2SM_KPM_RIC_ReportStyle_Item_KPMv2_t *)calloc(1, sizeof(*ric_report_style_item));
    ric_report_style_item->ric_ReportStyle_Type = 6;
    ric_report_style_item->ric_ReportStyle_Name.buf = (uint8_t *)strdup("O-CU-UP Measurement Container for the EPC connected deployment");
    ric_report_style_item->ric_ReportStyle_Name.size = strlen("O-CU-UP Measurement Container for the EPC connected deployment");
    ric_report_style_item->ric_ActionFormat_Type = 6; //pending 
    
      meas_action_item1 = (E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *)calloc(1, sizeof(*meas_action_item1));
    meas_action_item1->measName.buf = (uint8_t *)strdup(e2sm_kpm_meas_info[0].meas_type_name);
    meas_action_item1->measName.size = strlen(e2sm_kpm_meas_info[0].meas_type_name);

    E2SM_KPM_MeasurementTypeID_KPMv2_t *measID1;
    measID1 = (E2SM_KPM_MeasurementTypeID_KPMv2_t *)calloc(1, sizeof(*measID1));
    *measID1 = e2sm_kpm_meas_info[0].meas_type_id;

    meas_action_item1->measID = measID1;
    ASN_SEQUENCE_ADD(&ric_report_style_item->measInfo_Action_List.list, meas_action_item1);
    
    meas_action_item2 = (E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *)calloc(1, sizeof(*meas_action_item2));
    meas_action_item2->measName.buf = (uint8_t *)strdup(e2sm_kpm_meas_info[1].meas_type_name); //(uint8_t *)strdup("RRC.ConnEstabSucc.sum");
    meas_action_item2->measName.size = strlen(e2sm_kpm_meas_info[1].meas_type_name);

    E2SM_KPM_MeasurementTypeID_KPMv2_t *measID2;
    measID2 = (E2SM_KPM_MeasurementTypeID_KPMv2_t *)calloc(1, sizeof(*measID2));
    *measID2 = e2sm_kpm_meas_info[1].meas_type_id;

    meas_action_item2->measID = measID2;
    ASN_SEQUENCE_ADD(&ric_report_style_item->measInfo_Action_List.list, meas_action_item2);
    
    meas_action_item3 = (E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *)calloc(1, sizeof(*meas_action_item3));
    meas_action_item3->measName.buf = (uint8_t *)strdup(e2sm_kpm_meas_info[2].meas_type_name);
    meas_action_item3->measName.size = strlen(e2sm_kpm_meas_info[2].meas_type_name);

    E2SM_KPM_MeasurementTypeID_KPMv2_t *measID3;
    measID3 = (E2SM_KPM_MeasurementTypeID_KPMv2_t *)calloc(1, sizeof(*measID3));
    *measID3 = e2sm_kpm_meas_info[2].meas_type_id;

    meas_action_item3->measID = measID3;
    ASN_SEQUENCE_ADD(&ric_report_style_item->measInfo_Action_List.list, meas_action_item3);

    meas_action_item4 = (E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *)calloc(1, sizeof(*meas_action_item4));
    meas_action_item4->measName.buf = (uint8_t *)strdup(e2sm_kpm_meas_info[3].meas_type_name);
    meas_action_item4->measName.size = strlen(e2sm_kpm_meas_info[3].meas_type_name);

    E2SM_KPM_MeasurementTypeID_KPMv2_t *measID4;
    measID4 = (E2SM_KPM_MeasurementTypeID_KPMv2_t *)calloc(1, sizeof(*measID4));
    *measID4 = e2sm_kpm_meas_info[3].meas_type_id;

    meas_action_item4->measID = measID4;
    ASN_SEQUENCE_ADD(&ric_report_style_item->measInfo_Action_List.list, meas_action_item4);

    meas_action_item5 = (E2SM_KPM_MeasurementInfo_Action_Item_KPMv2_t *)calloc(1, sizeof(*meas_action_item5));
    meas_action_item5->measName.buf = (uint8_t *)strdup(e2sm_kpm_meas_info[4].meas_type_name);
    meas_action_item5->measName.size = strlen(e2sm_kpm_meas_info[4].meas_type_name);

    E2SM_KPM_MeasurementTypeID_KPMv2_t *measID5;
    measID5 = (E2SM_KPM_MeasurementTypeID_KPMv2_t *)calloc(1, sizeof(*measID5));
    *measID5 = e2sm_kpm_meas_info[4].meas_type_id;

    meas_action_item5->measID = measID5;
    ASN_SEQUENCE_ADD(&ric_report_style_item->measInfo_Action_List.list, meas_action_item5);

    ric_report_style_item->ric_IndicationHeaderFormat_Type = 1;
    ric_report_style_item->ric_IndicationMessageFormat_Type = 1;
    ASN_SEQUENCE_ADD(&func_def->ric_ReportStyle_List->list, ric_report_style_item);

    //xer_fprint(stderr, &asn_DEF_E2SM_KPM_E2SM_KPMv2_RANfunction_Description, func_def);

    RIC_AGENT_INFO("_______\n"); 
    func->enc_definition_len = e2ap_encode(&asn_DEF_E2SM_KPM_E2SM_KPMv2_RANfunction_Description,0, func_def,&func->enc_definition);
    RIC_AGENT_INFO("_______\n");

    RIC_AGENT_INFO("------ RAN FUNC DEF ENC Len:%lu-------\n", func->enc_definition_len);


    if (func->enc_definition_len < 0) {
        RIC_AGENT_ERROR("failed to encode RANfunction_List in E2SM KPM func description; aborting!");
        ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2SM_KPM_E2SM_KPMv2_RANfunction_Description, func_def);
        free(func_def);
        free(func);
        return -1;
    }

    func->enabled = 1;
    func->definition = func_def;

#if 0   
    /* Test code */
    E2SM_KPM_E2SM_KPMv2_RANfunction_Description_t *func_defi;
    asn_dec_rval_t decode_result;
    decode_result = aper_decode_complete(NULL, &asn_DEF_E2SM_KPM_E2SM_KPMv2_RANfunction_Description,
                                               (void **)&func_defi, func->enc_definition, func->enc_definition_len);
    DevAssert(decode_result.code == RC_OK);
#endif
    return ric_agent_register_ran_function(func);
}

static int e2sm_kpm_subscription_add(ric_agent_info_t *ric, ric_subscription_t *sub)
{
  /* XXX: process E2SM content. */
  if (LIST_EMPTY(&ric->subscription_list)) {
    LIST_INSERT_HEAD(&ric->subscription_list,sub,subscriptions);
  }
  else {
    LIST_INSERT_BEFORE(LIST_FIRST(&ric->subscription_list),sub,subscriptions);
  }
  return 0;
}

static int e2sm_kpm_subscription_del(ric_agent_info_t *ric, ric_subscription_t *sub, int force,long *cause,long *cause_detail)
{
    timer_remove(ric->e2sm_kpm_timer_id);
    LIST_REMOVE(sub, subscriptions);
    ric_free_subscription(sub);
    return 0;
}

static int e2sm_kpm_control(ric_agent_info_t *ric,ric_control_t *control)
{
    return 0;
}

static char *time_stamp(void)
{
    char *timestamp = (char *)malloc(sizeof(char) * 128);
    time_t ltime;
    ltime=time(NULL);
    struct tm *tm;
    tm=localtime(&ltime);

    sprintf(timestamp,"%d/%d/%d | %d:%d:%d", tm->tm_year+1900, tm->tm_mon,
            tm->tm_mday, tm->tm_hour, tm->tm_min, tm->tm_sec);
    return timestamp;
}

static int e2sm_kpm_ricInd_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        ric_ran_function_id_t function_id,
        long request_id,
        long instance_id,
        long action_id,
        uint8_t **outbuf,
        uint32_t *outlen)
{

    E2SM_KPM_E2SM_KPMv2_IndicationMessage_t* indicationmessage;
    ric_subscription_t *rs;

    DevAssert(timer_id == ric->e2sm_kpm_timer_id);

    char *time = time_stamp();
    RIC_AGENT_INFO("[%s] ----  Sending KPM RIC Indication, timer_id %ld function_id %ld---------\n", 
                   time, timer_id, function_id);
    free(time);

    /* Fetch the RIC Subscription */
    rs = ric_agent_lookup_subscription(ric,request_id,instance_id,function_id);
    if (!rs) {
        RIC_AGENT_ERROR("failed to find subscription %ld/%ld/%ld\n", request_id,instance_id,function_id);
    }

    indicationmessage = encode_kpm_Indication_Msg(ric, rs);

    {
        char *error_buf = (char*)calloc(300, sizeof(char));
        size_t errlen;
        asn_check_constraints(&asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage, indicationmessage, error_buf, &errlen);
        //fprintf(stderr,"KPM IND error length %zu\n", errlen);
        //fprintf(stderr,"KPM IND error buf %s\n", error_buf);
        free(error_buf);
        //xer_fprint(stderr, &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage, indicationmessage);
    }
    g_granularityIndx = 0; // Resetting

    //xer_fprint(stderr, &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage, indicationmessage);
    uint8_t e2smbuffer[8192];
    size_t e2smbuffer_size = 8192;

    asn_enc_rval_t er = asn_encode_to_buffer(NULL,
            ATS_ALIGNED_BASIC_PER,
            &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage,
            indicationmessage, e2smbuffer, e2smbuffer_size);

    //fprintf(stderr, "er encded is %zu\n", er.encoded);
    //fprintf(stderr, "after encoding KPM IND message\n");

    E2AP_E2AP_PDU_t *e2ap_pdu = (E2AP_E2AP_PDU_t*)calloc(1, sizeof(E2AP_E2AP_PDU_t));

    E2SM_KPM_E2SM_KPMv2_IndicationHeader_t* ind_header_style1 =
        (E2SM_KPM_E2SM_KPMv2_IndicationHeader_t*)calloc(1,sizeof(E2SM_KPM_E2SM_KPMv2_IndicationHeader_t));

    encode_e2sm_kpm_indication_header(ric->ranid, ind_header_style1);

    uint8_t e2sm_header_buf_style1[8192];
    size_t e2sm_header_buf_size_style1 = 8192;
    asn_enc_rval_t er_header_style1 = asn_encode_to_buffer(
            NULL,
            ATS_ALIGNED_BASIC_PER,
            &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationHeader,
            ind_header_style1,
            e2sm_header_buf_style1,
            e2sm_header_buf_size_style1);

    if (er_header_style1.encoded < 0) {
        fprintf(stderr, "ERROR encoding indication header, name=%s, tag=%s", er_header_style1.failed_type->name, er_header_style1.failed_type->xml_tag);
    }

    DevAssert(er_header_style1.encoded >= 0);

    // TODO - remove hardcoded values
    generate_e2apv1_indication_request_parameterized(
            e2ap_pdu, request_id, instance_id, function_id, action_id,
            0, e2sm_header_buf_style1, er_header_style1.encoded,
            e2smbuffer, er.encoded);

    *outlen = e2ap_asn1c_encode_pdu(e2ap_pdu, outbuf);

    return 0;
}

struct timeval g_captureStartTime;

static int e2sm_kpm_gp_timer_expiry(
        ric_agent_info_t *ric,
        long timer_id,
        ric_ran_function_id_t function_id,
        long request_id,
        long instance_id,
        long action_id,
        uint8_t **outbuf,
        uint32_t *outlen)

{
    int i,j=0;

    DevAssert(timer_id == ric->gran_prd_timer_id);

	if (g_granularityIndx == 0) /*First Granularity Period Expiry */
	{
		gettimeofday(&g_captureStartTime, NULL);
	}

    //char *time = time_stamp();
    //RIC_AGENT_INFO("[%s] +++  Granularity Period expired, timer_id %ld function_id %ld +++ \n",
    //               time, timer_id, function_id);
    //free(time);

    for (i = 0; i < MAX_KPM_MEAS; i++)
    {
        if (e2sm_kpm_meas_info[i].subscription_status == TRUE)
        {
            g_indMsgMeasRecItemArr[g_granularityIndx][j] = 
                            (E2SM_KPM_MeasurementRecordItem_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_MeasurementRecordItem_KPMv2_t));
            g_indMsgMeasRecItemArr[g_granularityIndx][j]->present = E2SM_KPM_MeasurementRecordItem_KPMv2_PR_integer;

            switch(e2sm_kpm_meas_info[i].meas_type_id)
            {
                case 1:/*RRC.ConnEstabAtt.sum*/
                    g_indMsgMeasRecItemArr[g_granularityIndx][j]->choice.integer = 
                                                                        rrc_kpi_stats.rrc_conn_estab_att_sum;
                    break;
                case 2:/*RRC.ConnEstabSucc.sum*/
                    g_indMsgMeasRecItemArr[g_granularityIndx][j]->choice.integer = 
                                                                        rrc_kpi_stats.rrc_conn_estab_succ_sum;
                    break;
                case 3:/*RRC.ConnReEstabAtt.sum*/
                    g_indMsgMeasRecItemArr[g_granularityIndx][j]->choice.integer = 
                                                                        rrc_kpi_stats.rrc_conn_reestab_att_sum;
                    break;
                case 4:/*RRC.ConnMean*/
                    g_indMsgMeasRecItemArr[g_granularityIndx][j]->choice.integer = 
                                                                            f1ap_cu_inst[ric->ranid].num_ues;
                    break;
                case 5:/*RRC.ConnMax*/
                    g_indMsgMeasRecItemArr[g_granularityIndx][j]->choice.integer = rrc_kpi_stats.rrc_conn_max;
                    break;
                default:
                    break;
            }
            j++;
        }
    }

    g_granularityIndx++;

    *outbuf = NULL;
    *outlen = 0;
    return 0;
}

uint8_t 
getMeasIdFromMeasName(uint8_t *measName)
{
    uint8_t i =0;
    int ret;

    for(i=0; i < MAX_KPM_MEAS; i++)
    {
        ret = strcmp(e2sm_kpm_meas_info[i].meas_type_name, (char *)measName);
        if (ret ==0)
        {
            RIC_AGENT_INFO("[%s] found, MeasId:%d\n",measName, (i+1));
            return (i+1);
        }
    }

    return 0xFF;
}

int 
e2sm_kpm_decode_and_handle_action_def(uint8_t *def_buf, 
                                          size_t def_size, 
                                          ric_ran_function_t *func,
                                          uint32_t      interval_ms,
                                          ric_subscription_t* rs,
                                          ric_agent_info_t *ric)
{
    E2SM_KPM_E2SM_KPMv2_ActionDefinition_t *actionDef = NULL;
    // or uncomment below:
    //actionDef = calloc(1, sizeof(E2SM_KPM_E2SM_KPMv2_ActionDefinition_t));
    E2SM_KPM_E2SM_KPMv2_ActionDefinition_Format1_t *actionDefFormat1;
    E2SM_KPM_MeasurementInfoItem_KPMv2_t *actionDefMeasInfoItem;
    E2SM_KPM_MeasurementTypeID_KPMv2_t localMeasID;
    asn_dec_rval_t decode_result;
    uint32_t      gp_interval_sec = 0;
    uint32_t      gp_interval_us = 0;
    uint32_t      gp_interval_ms = 0;
    uint8_t i,ret;
    uint16_t subsId = 10;//hack

    g_granulPeriod = (E2SM_KPM_GranularityPeriod_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_GranularityPeriod_KPMv2_t));

    /*Reset Subscriptions */
    for (i = 0; i < MAX_KPM_MEAS; i++)
    {
        e2sm_kpm_meas_info[i].subscription_status = FALSE;
    }
    g_indMsgMeasInfoCnt = 0; // resetting
   
    RIC_AGENT_INFO("ACTION Def size:%lu\n", def_size);
    if (def_size == 0)
    {
        /* In case of missing action list, all Meas Info should be reported to RIC */
        RIC_AGENT_INFO("ACTION Def missing, populating all KPM Data\n");

        for (i = 0; i < MAX_KPM_MEAS; i++)
        {
            g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt] =
                                     (E2SM_KPM_MeasurementInfoItem_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_MeasurementInfoItem_KPMv2_t));
            g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.present = E2SM_KPM_MeasurementType_KPMv2_PR_measName;
            g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.choice.measName.buf =
                                                     (uint8_t *)strdup(e2sm_kpm_meas_info[i].meas_type_name);
            g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.choice.measName.size =
                                                                strlen(e2sm_kpm_meas_info[i].meas_type_name);
            e2sm_kpm_meas_info[i].subscription_status = TRUE;
            g_indMsgMeasInfoCnt++;
        }
        *g_granulPeriod = 10; //Hack

        /* Hack - Subscription ID */
        //g_subscriptionID.size = sizeof(subsId);
        //g_subscriptionID.buf = (uint8_t *)calloc(1,sizeof(subsId));
        //*g_subscriptionID.buf = subsId;
		g_subscriptionID = subsId; 

        action_def_missing = TRUE; /* Granularity Timer will not start */
        return 0;
    }
 
    decode_result = aper_decode_complete(NULL, &asn_DEF_E2SM_KPM_E2SM_KPMv2_ActionDefinition,
                                         (void **)&actionDef, def_buf, def_size);
    DevAssert(decode_result.code == RC_OK);
    xer_fprint(stdout, &asn_DEF_E2SM_KPM_E2SM_KPMv2_ActionDefinition, actionDef);

    if (actionDef->actionDefinition_formats.present == /*E2SM-KPM Action Definition Format 1*/
                            E2SM_KPM_E2SM_KPMv2_ActionDefinition__actionDefinition_formats_PR_actionDefinition_Format1)
    {
        actionDefFormat1 = &actionDef->actionDefinition_formats.choice.actionDefinition_Format1;

        if (actionDefFormat1->granulPeriod > interval_ms)
        {
            RIC_AGENT_ERROR("Subscription Failure: Granularity Period:%lu ms Reporting Interval:%u ms\n",
                            actionDefFormat1->granulPeriod, interval_ms);
            return -1;
        }

        *g_granulPeriod = actionDefFormat1->granulPeriod;

#if 0        
        if (actionDefFormat1->subscriptID.size)
        {
            g_subscriptionID.size = actionDefFormat1->subscriptID.size;
            g_subscriptionID.buf = (uint8_t *)calloc(1,actionDefFormat1->subscriptID.size);
            memcpy(g_subscriptionID.buf,
                   actionDefFormat1->subscriptID.buf,
                   actionDefFormat1->subscriptID.size);
        }       
#endif
		g_subscriptionID = actionDefFormat1->subscriptID;
 
        /* Fetch KPM subscription details */
        for (i=0; i < actionDefFormat1->measInfoList.list.count; i++)
        {
            actionDefMeasInfoItem = (E2SM_KPM_MeasurementInfoItem_KPMv2_t *)(actionDefFormat1->measInfoList.list.array[i]);

            //if (actionDefMeasInfoItem->measType.present == E2SM_KPM_MeasurementType_PR_measID)
            if (actionDefMeasInfoItem->measType.present == E2SM_KPM_MeasurementType_KPMv2_PR_measName)
            {
                //localMeasID = actionDefMeasInfoItem->measType.choice.measID;
                localMeasID = getMeasIdFromMeasName(actionDefMeasInfoItem->measType.choice.measName.buf);

                if ( ( (localMeasID > 0) &&
                       (localMeasID < (MAX_KPM_MEAS+1) ) ) &&  /*Expecting KPM MeasID to be within limits */
                     (e2sm_kpm_meas_info[localMeasID-1].subscription_status == FALSE) ) /*Avoid subscribing duplicate */
                {
                    /* Set the Subscription Status */
                    g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt] =
                                                 (E2SM_KPM_MeasurementInfoItem_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_MeasurementInfoItem_KPMv2_t));
                    g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.present = E2SM_KPM_MeasurementType_KPMv2_PR_measName;
                    g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.choice.measName.buf =
                                                     (uint8_t *)strdup(e2sm_kpm_meas_info[localMeasID-1].meas_type_name);
                    g_indMsgMeasInfoItemArr[g_indMsgMeasInfoCnt]->measType.choice.measName.size =
                                                                strlen(e2sm_kpm_meas_info[localMeasID-1].meas_type_name);
                    e2sm_kpm_meas_info[localMeasID-1].subscription_status = TRUE;
                    g_indMsgMeasInfoCnt++;
                }
                else
                {
                    RIC_AGENT_ERROR("Act Def Err i=%d MeasId:%ld indMsgMeasInfoCnt:%d\n",
                                    i, localMeasID, g_indMsgMeasInfoCnt);
                    return -1;
                }
            }
            else
            {
                RIC_AGENT_ERROR("Meas Name not found in Action Def\n");
                return -1;
            }
        }
        gp_interval_ms = actionDefFormat1->granulPeriod;
        gp_interval_us = (gp_interval_ms%1000)*1000;
        gp_interval_sec = (gp_interval_ms/1000);

        ric_ran_function_requestor_info_t* arg_gp
                    = (ric_ran_function_requestor_info_t*)calloc(1, sizeof(ric_ran_function_requestor_info_t));
        arg_gp->function_id = func->function_id;
        arg_gp->request_id = rs->request_id;
        arg_gp->instance_id = rs->instance_id;
        arg_gp->action_id = (LIST_FIRST(&rs->action_list))->id;
        /*Start Timer for Granularity Period */
        ret = timer_setup(gp_interval_sec, gp_interval_us,
                          TASK_RIC_AGENT,
                          ric->ranid,
                          TIMER_PERIODIC,
                          (void *)arg_gp,
                          &ric->gran_prd_timer_id);
        if (ret < 0) {
            RIC_AGENT_ERROR("failed to start Granularity Period timer\n");
            return -1;
        }
    }
    else
    {
        RIC_AGENT_ERROR("Subscription Failure: Invalid Action Def Format:%d\n",
                        actionDef->actionDefinition_formats.present);
        return -1;
    }

    return 0;
}

static E2SM_KPM_E2SM_KPMv2_IndicationMessage_t*
encode_kpm_Indication_Msg(ric_agent_info_t* ric, ric_subscription_t *rs)
{
    int ret;
    uint8_t i,k;
    E2SM_KPM_MeasurementDataItem_KPMv2_t* meas_data_item[MAX_GRANULARITY_INDEX];
    E2SM_KPM_MeasurementRecord_KPMv2_t* meas_rec[MAX_GRANULARITY_INDEX];
    E2SM_KPM_MeasurementData_KPMv2_t* meas_data;

    if (action_def_missing == TRUE)
    { 
        for (i = 0; i < MAX_KPM_MEAS; i++)
        {
            g_indMsgMeasRecItemArr[0][i] = (E2SM_KPM_MeasurementRecordItem_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_MeasurementRecordItem_KPMv2_t));
            g_indMsgMeasRecItemArr[0][i]->present = E2SM_KPM_MeasurementRecordItem_KPMv2_PR_integer;

            switch(i)
            {
                case 0:/*RRC.ConnEstabAtt.sum*/
                    g_indMsgMeasRecItemArr[0][i]->choice.integer = rrc_kpi_stats.rrc_conn_estab_att_sum;
                    break;
                case 1:/*RRC.ConnEstabSucc.sum*/
                    g_indMsgMeasRecItemArr[0][i]->choice.integer = rrc_kpi_stats.rrc_conn_estab_succ_sum; 
                    break;
                case 2:/*RRC.ConnReEstabAtt.sum*/
                    g_indMsgMeasRecItemArr[0][i]->choice.integer = rrc_kpi_stats.rrc_conn_reestab_att_sum;
                    break;
                case 3:/*RRC.ConnMean*/
                    g_indMsgMeasRecItemArr[0][i]->choice.integer = f1ap_cu_inst[ric->ranid].num_ues;
                    break;
                case 4:/*RRC.ConnMax*/
                    g_indMsgMeasRecItemArr[0][i]->choice.integer = rrc_kpi_stats.rrc_conn_max;
                    break;

                default:
                    break;
            }
        }
        g_granularityIndx = 1;
    } 

    //RIC_AGENT_INFO("Granularity Idx=:%d\n",g_granularityIndx);

    /*
     * measData->measurementRecord (List)
     */
    meas_data = (E2SM_KPM_MeasurementData_KPMv2_t*)calloc(1, sizeof(E2SM_KPM_MeasurementData_KPMv2_t));
    DevAssert(meas_data!=NULL);
    
    for (k=0; k < g_granularityIndx; k++)
    {
        /*
         * Measurement Record->MeasurementRecordItem (List)
         */
        meas_rec[k] = (E2SM_KPM_MeasurementRecord_KPMv2_t *)calloc(1, sizeof(E2SM_KPM_MeasurementRecord_KPMv2_t));
        for(i=0; i < g_indMsgMeasInfoCnt; i++)
        { 
            /* Meas Records meas_rec[]  have to be prepared for each Meas data item */
            ret = ASN_SEQUENCE_ADD(&meas_rec[k]->list, g_indMsgMeasRecItemArr[k][i]);
            DevAssert(ret == 0);
        }

        /* MeasDataItem*/
        meas_data_item[k] = (E2SM_KPM_MeasurementDataItem_KPMv2_t*)calloc(1, sizeof(E2SM_KPM_MeasurementDataItem_KPMv2_t));
        meas_data_item[k]->measRecord = *meas_rec[k];

        /* Enqueue Meas data items */
        ret = ASN_SEQUENCE_ADD(&meas_data->list, meas_data_item[k]);
        DevAssert(ret == 0);
    }

   /*
    * measInfoList
    */
    E2SM_KPM_MeasurementInfoList_KPMv2_t* meas_info_list = (E2SM_KPM_MeasurementInfoList_KPMv2_t*)calloc(1, sizeof(E2SM_KPM_MeasurementInfoList_KPMv2_t));
    for(i=0; i < g_indMsgMeasInfoCnt; i++)
    {
        ret = ASN_SEQUENCE_ADD(&meas_info_list->list, g_indMsgMeasInfoItemArr[i]);
        DevAssert(ret == 0);
    }

    /*
     * IndicationMessage_Format1 -> measInfoList
     * IndicationMessage_Format1 -> measData
     */
    E2SM_KPM_E2SM_KPMv2_IndicationMessage_Format1_t* format = 
                        (E2SM_KPM_E2SM_KPMv2_IndicationMessage_Format1_t*)calloc(1, sizeof(E2SM_KPM_E2SM_KPMv2_IndicationMessage_Format1_t));
    ASN_STRUCT_RESET(asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationMessage_Format1, format);
    //format->subscriptID.size = g_subscriptionID.size;
    //format->subscriptID.buf = g_subscriptionID.buf;
    format->subscriptID = g_subscriptionID;
	format->measInfoList = meas_info_list;
    format->measData = *meas_data;
    format->granulPeriod = g_granulPeriod;

    /*
     * IndicationMessage -> IndicationMessage_Format1
     */
    E2SM_KPM_E2SM_KPMv2_IndicationMessage_t* indicationmessage = 
                                (E2SM_KPM_E2SM_KPMv2_IndicationMessage_t*)calloc(1, sizeof(E2SM_KPM_E2SM_KPMv2_IndicationMessage_t));
    indicationmessage->indicationMessage_formats.present = 
                                    E2SM_KPM_E2SM_KPMv2_IndicationMessage__indicationMessage_formats_PR_indicationMessage_Format1;
    indicationmessage->indicationMessage_formats.choice.indicationMessage_Format1 = *format;
    
    return indicationmessage;
}

const unsigned long long EPOCH = 2208988800ULL;
const unsigned long long NTP_SCALE_FRAC = 4294967296ULL;

unsigned int tv_to_ntp(struct timeval tv)
{
    unsigned long long tv_ntp, tv_usecs;

    tv_ntp = tv.tv_sec + EPOCH;
    tv_usecs = (NTP_SCALE_FRAC * tv.tv_usec) / 1000000UL;

    return (((tv_ntp << 32) | tv_usecs) & 0xFFFFFFFF);//just returning 32bits
}

void encode_e2sm_kpm_indication_header(ranid_t ranid, E2SM_KPM_E2SM_KPMv2_IndicationHeader_t *ihead) 
{
    e2node_type_t node_type;
    ihead->indicationHeader_formats.present = E2SM_KPM_E2SM_KPMv2_IndicationHeader__indicationHeader_formats_PR_indicationHeader_Format1;

    E2SM_KPM_E2SM_KPMv2_IndicationHeader_Format1_t* ind_header = &ihead->indicationHeader_formats.choice.indicationHeader_Format1;

    /* KPM Node ID */
    ind_header->kpmNodeID = (E2SM_KPM_GlobalKPMnode_ID_KPMv2_t *)calloc(1,sizeof(E2SM_KPM_GlobalKPMnode_ID_KPMv2_t));
    ind_header->kpmNodeID->present = E2SM_KPM_GlobalKPMnode_ID_KPMv2_PR_eNB;

    node_type = e2_conf[ranid]->e2node_type;
    
    if (node_type == E2NODE_TYPE_ENB_CU) 
    {
        MCC_MNC_TO_PLMNID(
                e2_conf[ranid]->mcc,
                e2_conf[ranid]->mnc,
                e2_conf[ranid]->mnc_digit_length,
                &ind_header->kpmNodeID->choice.eNB.global_eNB_ID.pLMN_Identity);
    
        ind_header->kpmNodeID->choice.eNB.global_eNB_ID.eNB_ID.present = E2SM_KPM_ENB_ID_KPMv2_PR_macro_eNB_ID; 
    
        MACRO_ENB_ID_TO_BIT_STRING(
                e2_conf[ranid]->cell_identity,
                &ind_header->kpmNodeID->choice.eNB.global_eNB_ID.eNB_ID.choice.macro_eNB_ID);
    }

    /* Collect Start Time Stamp */
    /* Encoded in the same format as the first four octets of the 64-bit timestamp format as defined in section 6 of IETF RFC 5905 */
    ind_header->colletStartTime.buf = (uint8_t *)calloc(1, 4);
    ind_header->colletStartTime.size = 4;
    *((uint32_t *)(ind_header->colletStartTime.buf)) = htonl((uint32_t)time(NULL));
    //xer_fprint(stderr, &asn_DEF_E2SM_KPM_E2SM_KPMv2_IndicationHeader, ihead);
}
