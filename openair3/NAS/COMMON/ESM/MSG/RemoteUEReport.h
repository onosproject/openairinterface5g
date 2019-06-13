/*
 * RemoteUEReport.h
 *
 *  Created on: Jun 5, 2019
 *      Author: nepes
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "ProtocolDiscriminator.h"
#include "EpsBearerIdentity.h"
#include "ProcedureTransactionIdentity.h"
#include "PKMFAddress.h"

#ifndef OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_
#define OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_

/* Minimum length macro. Formed by minimum length of each mandatory field */
#define REMOTE_UE_REPORT_RESPONSE_MINIMUM_LENGTH (0)

typedef struct remote_ue_report_msg_tag {
  /* Mandatory fields */
  ProtocolDiscriminator               protocoldiscriminator:4;
  EpsBearerIdentity                   epsbeareridentity:4;
  ProcedureTransactionIdentity        proceduretransactionidentity;
  /* Optional fields */
  pkmf_address_t                   		pkmfaddress;
  //RemoteUEContext        				remoteuecontext;
} remote_ue_report_msg;

int decode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len);

int encode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len);


#endif /* OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_ */
