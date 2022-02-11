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

#include "ric_agent.h"
#include "e2ap_decoder.h"

#include "E2AP_E2AP-PDU.h"
#include "E2AP_ProcedureCode.h"
#include "per_decoder.h"

#define CASE_E2AP_I(id,name)					\
    case id:							\
    RIC_AGENT_INFO("decoded initiating " #name " (%ld)\n",id);	\
    break

#define CASE_E2AP_S(id,name)						\
    case id:								\
    RIC_AGENT_INFO("decoded successful outcome " #name " (%ld)\n",id);	\
    break

#define CASE_E2AP_U(id,name)						\
    case id:								\
    RIC_AGENT_INFO("decoded unsuccessful outcome " #name " (%ld)\n",id);	\
    break

int e2ap_decode_pdu(E2AP_E2AP_PDU_t *pdu,
		    const uint8_t * const buf,const uint32_t len)
{
  asn_dec_rval_t dres;

  DevAssert(pdu != NULL);
  DevAssert(buf != NULL);

  dres = aper_decode(NULL,&asn_DEF_E2AP_E2AP_PDU,(void **)&pdu,buf,len,0,0);
  if (dres.code != RC_OK) {
    RIC_AGENT_ERROR("failed to decode PDU (%d)\n",dres.code);
    return -1;
  }

  xer_fprint(stdout, &asn_DEF_E2AP_E2AP_PDU, pdu);

  switch (pdu->present) {
  case E2AP_E2AP_PDU_PR_initiatingMessage:
    switch (pdu->choice.initiatingMessage.procedureCode) 
    {
      CASE_E2AP_I(E2AP_ProcedureCode_id_Reset,Reset);
      CASE_E2AP_I(E2AP_ProcedureCode_id_RICsubscription,
		          RICsubscription);
      CASE_E2AP_I(E2AP_ProcedureCode_id_RICsubscriptionDelete,
		          RICsubscriptionDelete);
      CASE_E2AP_I(E2AP_ProcedureCode_id_RICcontrol,RICcontrol);
      CASE_E2AP_I(E2AP_ProcedureCode_id_RICserviceQuery,RICserviceQuery);
      CASE_E2AP_I(E2AP_ProcedureCode_id_ErrorIndication,ErrorIndication);
      CASE_E2AP_I(E2AP_ProcedureCode_id_E2connectionUpdate,E2ConnectionUpdate);
    
      default:
        RIC_AGENT_ERROR("unknown procedure ID (%d) for initiating message\n",
		                (int)pdu->choice.initiatingMessage.procedureCode);
      return -1;
    }
    break;
  case E2AP_E2AP_PDU_PR_successfulOutcome:
    switch (pdu->choice.successfulOutcome.procedureCode) {
    CASE_E2AP_S(E2AP_ProcedureCode_id_E2setup,E2SetupResponse);
    CASE_E2AP_S(E2AP_ProcedureCode_id_Reset,Reset);
    CASE_E2AP_S(E2AP_ProcedureCode_id_RICserviceUpdate,RICserviceUpdate);
    CASE_E2AP_S(E2AP_ProcedureCode_id_E2nodeConfigurationUpdate,E2nodeConfigurationUpdateAcknowledge);
    default:
      RIC_AGENT_ERROR("unknown procedure ID (%d) for successful outcome\n",
		 (int)pdu->choice.successfulOutcome.procedureCode);
      return -1;
    }
    break;
  case E2AP_E2AP_PDU_PR_unsuccessfulOutcome:
    switch (pdu->choice.unsuccessfulOutcome.procedureCode) {
        CASE_E2AP_U(E2AP_ProcedureCode_id_E2setup,E2setupFailure);
        CASE_E2AP_U(E2AP_ProcedureCode_id_RICserviceUpdate,RICserviceUpdate);
        default:
        RIC_AGENT_ERROR("unknown procedure ID (%d) for unsuccessful outcome\n",
		   (int)pdu->choice.unsuccessfulOutcome.procedureCode);
	return -1;
    }
    break;
  default:
    RIC_AGENT_ERROR("unknown presence (%d)\n",(int)pdu->present);
    return -1;
  }

  return 0;
}
