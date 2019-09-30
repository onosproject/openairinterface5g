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
#include "MessageType.h"
#include "RemoteUEContext.h"

#ifndef OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_
#define OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_

/* Minimum length macro. Formed by minimum length of each mandatory field */
#define REMOTE_UE_REPORT_MINIMUM_LENGTH (0)
#define REMOTE_UE_REPORT_MAXIMUM_LENGTH (20)

# define REMOTE_UE_CONTEXT_PRESENT (1<<0)
# define REMOTE_UE_REPORT_PKMF_ADDRESS_PRESENT (1<<1)


typedef enum remote_ue_report_iei_tag {
	REMOTE_UE_REPORT_REMOTE_UE_CONTEXT_IEI = 0x79,
	REMOTE_UE_REPORT_PKMF_ADDRESS_IEI = 0x6f,
	} remote_ue_report_iei;

typedef struct remote_ue_report_msg_tag {
  /* Mandatory fields */
  ProtocolDiscriminator                 protocoldiscriminator:4;
  EpsBearerIdentity                     epsbeareridentity:4;
  ProcedureTransactionIdentity          proceduretransactionidentity;
  MessageType                           messagetype;
  /* Optional fields */
  uint32_t                              presencemask;
  remote_ue_context_t                   remoteuecontext;
  pkmf_address_t                   		pkmfaddress;
  } remote_ue_report_msg;

int decode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len);

int encode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len);


#endif /* OPENAIR3_NAS_COMMON_ESM_MSG_REMOTEUEREPORT_H_ */
