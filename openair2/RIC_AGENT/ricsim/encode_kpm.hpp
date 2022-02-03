/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ENCODE_KPM_HPP
#define ENCODE_KPM_HPP

extern "C" {
  #include "OCUCP-PF-Container.h"
  #include "OCTET_STRING.h"
  #include "asn_application.h"
  #include "E2SM-KPM-IndicationMessage.h"
  #include "FQIPERSlicesPerPlmnListItem.h"
  #include "E2SM-KPM-RANfunction-Description.h"
  #include "Timestamp.h"
}

void encode_kpm(E2SM_KPM_IndicationMessage_t* indicationmessage);

void encode_kpm_bak(E2SM_KPM_IndicationMessage_t* indicationmessage);

void encode_kpm_function_description(E2SM_KPM_RANfunction_Description_t* ranfunc_desc);

void encode_kpm_report_style5(E2SM_KPM_IndicationMessage_t* indicationmessage);

void encode_kpm_odu_user_level(RAN_Container_t *ranco);

void encode_kpm_ocucp_user_level(RAN_Container_t *ranco);

void encode_kpm_ocuup_user_level(RAN_Container_t *ranco);

void encode_kpm_report_rancontainer_du(E2SM_KPM_IndicationMessage_t *indMsg);

void encode_kpm_report_rancontainer_cucp(E2SM_KPM_IndicationMessage_t *indMsg);

void encode_kpm_report_rancontainer_cuup(E2SM_KPM_IndicationMessage_t *indMsg);

void encode_kpm_report_style1(E2SM_KPM_IndicationMessage_t* indicationmessage);

void encode_kpm_report_rancontainer_cucp_parameterized(E2SM_KPM_IndicationMessage_t* indicationmessage,uint8_t *plmnid_buf,uint8_t *nrcellid_buf,uint8_t *crnti_buf,const uint8_t *serving_buf, const uint8_t *neighbor_buf);

void encode_kpm_report_rancontainer_cuup_parameterized(E2SM_KPM_IndicationMessage_t* indicationmessage, uint8_t *plmnid_buf, uint8_t *nrcellid_buf, uint8_t *crnti_buf,int pdcp_bytesdl, int pdcp_bytesul);

void encode_kpm_report_style1_parameterized(E2SM_KPM_IndicationMessage_t* indicationmessage, long fiveqi, long dl_prb_usage, long ul_prb_usage, uint8_t* sd_buf, uint8_t* sst_buf,uint8_t* plmnid_buf, uint8_t* nrcellid_buf, long *dl_prbs, long *ul_prbs);

void encode_kpm_report_style5_parameterized(E2SM_KPM_IndicationMessage_t* indicationmessage, uint8_t *gnbcuupname_buf, int bytes_dl,int bytes_ul, uint8_t *sst_buf, uint8_t *sd_buf, uint8_t *plmnid_buf);


#endif
