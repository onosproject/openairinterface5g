/*
 * RemoteUEReportResponse.h
 *
 *  Created on: Jun 7, 2019
 *      Author: Mohit Vyas
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "ProtocolDiscriminator.h"
#include "EpsBearerIdentity.h"
#include "ProcedureTransactionIdentity.h"

#ifndef OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORTRESPONSE_H_
#define OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORTRESPONSE_H_

/* Minimum length macro. Formed by minimum length of each mandatory field */
#define REMOTE_UE_REPORT_RESPONSE_MINIMUM_LENGTH (0)

/* Maximum length macro. Formed by minimum length of each mandatory field */
#define REMOTE_UE_REPORT_RESPONSE_MAXIMUM_LENGTH (0)

typedef struct remote_ue_report_response_msg_tag {
  /* Mandatory fields */
  ProtocolDiscriminator               protocoldiscriminator:4;
  EpsBearerIdentity                   epsbeareridentity:4;
  ProcedureTransactionIdentity        proceduretransactionidentity;
} remote_ue_report_response_msg;

int decode_remote_ue_report_response(remote_ue_report_response_msg *remoteuereportresponse, uint8_t *buffer, uint32_t len);

int encode_remote_ue_report_response(remote_ue_report_response_msg *remoteuereportresponse, uint8_t *buffer, uint32_t len);


#endif /* OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORTRESPONSE_H_ */
