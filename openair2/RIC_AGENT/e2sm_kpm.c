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
#include <stdlib.h>

#include "ric_agent_defs.h"
#include "ric_agent_common.h"
#include "e2ap_common.h"
#include "e2ap_encoder.h"
#include "e2sm_common.h"

#include "E2AP_Cause.h"
#include "E2SM_KPM_E2SM-KPM-RANfunction-Description.h"
#include "E2SM_KPM_RIC-ReportStyle-List.h"
#include "E2SM_KPM_RIC-EventTriggerStyle-List.h"

/**
 ** The main thing with this abstraction is that we need per-SM modules
 ** to handle the details of the function, event trigger, action, etc
 ** definitions... and to actually do all the logic and implement the
 ** inner parts of the message encoding.  generic e2ap handles the rest.
 **/

static int e2sm_kpm_subscription_add(ric_agent_info_t *ric,
				     ric_subscription_t *sub);
static int e2sm_kpm_subscription_del(ric_agent_info_t *ric,
				     ric_subscription_t *sub,
				     int force,long *cause,long *cause_detail);
static int e2sm_kpm_control(ric_agent_info_t *ric,ric_control_t *control);

static ric_service_model_t e2sm_kpm_model = {
  .name = "ORAN-E2SM-KPM",
  .oid = "1.3.6.1.4.1.1.1.2.2",
  .handle_subscription_add = e2sm_kpm_subscription_add,
  .handle_subscription_del = e2sm_kpm_subscription_del,
  .handle_control = e2sm_kpm_control
};

/**
 * Initializes KPM state and registers KPM e2ap_ran_function_id_t number(s).
 */
int e2sm_kpm_init(void)
{
  ric_ran_function_t *func;
  E2SM_KPM_E2SM_KPM_RANfunction_Description_t *func_def;
  E2SM_KPM_RIC_ReportStyle_List_t *ric_report_style_item;
  E2SM_KPM_RIC_EventTriggerStyle_List_t *ric_event_trigger_style_item;

  func = (ric_ran_function_t *)calloc(1,sizeof(*func));
  func->model = &e2sm_kpm_model;
  func->revision = 0;
  func->name = "ORAN-E2SM-KPM";
  func->description = "KPM monitor";

  func_def = (E2SM_KPM_E2SM_KPM_RANfunction_Description_t *) \
    calloc(1,sizeof(*func_def));

  func_def->ranFunction_Name.ranFunction_ShortName.buf = \
    (uint8_t *)strdup(func->name);
  func_def->ranFunction_Name.ranFunction_ShortName.size = \
    strlen(func->name);
  func_def->ranFunction_Name.ranFunction_E2SM_OID.buf = \
    (uint8_t *)strdup(func->model->oid);
  func_def->ranFunction_Name.ranFunction_E2SM_OID.size = \
    strlen(func->model->oid);
  func_def->ranFunction_Name.ranFunction_Description.buf = \
    (uint8_t *)strdup(func->description);
  func_def->ranFunction_Name.ranFunction_Description.size = \
    strlen(func->description);
  func_def->e2SM_KPM_RANfunction_Item.ric_EventTriggerStyle_List = \
    (struct E2SM_KPM_E2SM_KPM_RANfunction_Description__e2SM_KPM_RANfunction_Item__ric_EventTriggerStyle_List *)calloc(1,sizeof(*func_def->e2SM_KPM_RANfunction_Item.ric_EventTriggerStyle_List));
  ric_event_trigger_style_item = (E2SM_KPM_RIC_EventTriggerStyle_List_t *)calloc(1,sizeof(*ric_event_trigger_style_item));
  ric_event_trigger_style_item->ric_EventTriggerStyle_Type = 1;
  ric_event_trigger_style_item->ric_EventTriggerStyle_Name.buf = (uint8_t *)strdup("Trigger1");
  ric_event_trigger_style_item->ric_EventTriggerStyle_Name.size = strlen("Trigger1");
  ric_event_trigger_style_item->ric_EventTriggerFormat_Type = 1;
  ASN_SEQUENCE_ADD(
    &func_def->e2SM_KPM_RANfunction_Item.ric_EventTriggerStyle_List->list,
    ric_event_trigger_style_item);

  func_def->e2SM_KPM_RANfunction_Item.ric_ReportStyle_List = \
    (struct E2SM_KPM_E2SM_KPM_RANfunction_Description__e2SM_KPM_RANfunction_Item__ric_ReportStyle_List *)calloc(1,sizeof(*func_def->e2SM_KPM_RANfunction_Item.ric_ReportStyle_List));
  ric_report_style_item = (E2SM_KPM_RIC_ReportStyle_List_t *)calloc(1,sizeof(*ric_report_style_item));
  ric_report_style_item->ric_ReportStyle_Type = 6;
  ric_report_style_item->ric_ReportStyle_Name.buf = (uint8_t *)strdup("O-CU-UP Measurement Container for the EPC connected deployment");
  ric_report_style_item->ric_ReportStyle_Name.size = strlen("O-CU-UP Measurement Container for the EPC connected deployment");
  ric_report_style_item->ric_IndicationHeaderFormat_Type = 1;
  ric_report_style_item->ric_IndicationMessageFormat_Type = 1;
  ASN_SEQUENCE_ADD(&func_def->e2SM_KPM_RANfunction_Item.ric_ReportStyle_List->list,
  		   ric_report_style_item);

  func->enc_definition_len = e2ap_encode(
    &asn_DEF_E2SM_KPM_E2SM_KPM_RANfunction_Description,0,
    func_def,&func->enc_definition);
  if (func->enc_definition_len < 0) {
    E2AP_ERROR("failed to encode RANfunction_List in E2SM KPM func description; aborting!");
    ASN_STRUCT_FREE_CONTENTS_ONLY(asn_DEF_E2SM_KPM_E2SM_KPM_RANfunction_Description,
				  func_def);
    free(func_def);
    free(func);

    return -1;
  }

  func->enabled = 1;
  func->definition = func_def;

  return ric_agent_register_ran_function(func);
}

static int e2sm_kpm_subscription_add(ric_agent_info_t *ric,
				     ric_subscription_t *sub)
{
  /* XXX: process E2SM content. */
  if (LIST_EMPTY(&ric->subscription_list) == 0) {
    LIST_INSERT_HEAD(&ric->subscription_list,sub,subscriptions);
  }
  else {
    LIST_INSERT_BEFORE(LIST_FIRST(&ric->subscription_list),sub,subscriptions);
  }
  return 0;
}

static int e2sm_kpm_subscription_del(ric_agent_info_t *ric,
				     ric_subscription_t *sub,
				     int force,long *cause,long *cause_detail)
{
  LIST_REMOVE(sub,subscriptions);
  ric_free_subscription(sub);
  return 0;
}

static int e2sm_kpm_control(ric_agent_info_t *ric,ric_control_t *control)
{
    return 0;
}
