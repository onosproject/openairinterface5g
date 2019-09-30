/*
 * RemoteUEReport.c
 *
 *  Created on: Jun 5, 2019
 *      Author: nepes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#include "TLVEncoder.h"
#include "TLVDecoder.h"
#include "RemoteUEReport.h"
#include "PKMFAddress.h"
#include "RemoteUEContext.h"

int decode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len)
{
uint32_t decoded = 0;
int decoded_result = 0;
//remoteuereport->pkmfaddress.pkmfipv4address = 1 ;
// Check if we got a NULL pointer and if buffer length is >= minimum length expected for the message.
CHECK_PDU_POINTER_AND_LENGTH_DECODER(buffer, REMOTE_UE_REPORT_MINIMUM_LENGTH, len);

/* Decoding mandatory fields */
  /* Decoding optional fields */
  while(len - decoded > 0) {
    uint8_t ieiDecoded = *(buffer + decoded);

    /* Type | value iei are below 0x80 so just return the first 4 bits */
    if (ieiDecoded >= 0x80)
      ieiDecoded = ieiDecoded & 0xf0;

    switch(ieiDecoded) {
        case REMOTE_UE_REPORT_PKMF_ADDRESS_IEI:
        if ((decoded_result = decode_pkmf_address(&remoteuereport->pkmfaddress,
		REMOTE_UE_REPORT_PKMF_ADDRESS_IEI,
		buffer + decoded,
		len - decoded)) < 0)
    return decoded_result;
  else
    decoded += decoded_result;
/* Set corresponding mask to 1 in presencemask */
        remoteuereport->presencemask |= REMOTE_UE_REPORT_PKMF_ADDRESS_PRESENT;
      break;
    }
  }
return decoded;
}

int encode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len)
{
int encoded = 0;
int encode_result = 0;
//uint32_t testip [4] = {0,1,2,3};
//remoteuereport->pkmfaddress.pkmfipv4address = testip ;

/* Checking IEI and pointer */
CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, REMOTE_UE_REPORT_MINIMUM_LENGTH, len);

if ((remoteuereport->presencemask & REMOTE_UE_CONTEXT_PRESENT) == REMOTE_UE_CONTEXT_PRESENT)
{
if ((encode_result = encode_remote_ue_context(&remoteuereport->remoteuecontext,
		REMOTE_UE_REPORT_REMOTE_UE_CONTEXT_IEI,
		buffer + encoded,
		len - encoded)) < 0)
	return encode_result;

	else

	   encoded += encode_result;
}

if ((remoteuereport->presencemask & REMOTE_UE_REPORT_PKMF_ADDRESS_PRESENT)
      == REMOTE_UE_REPORT_PKMF_ADDRESS_PRESENT)
{

if ((encode_result = encode_pkmf_address(&remoteuereport->pkmfaddress,
		REMOTE_UE_REPORT_PKMF_ADDRESS_IEI ,
		buffer + encoded,
		len - encoded)) < 0)//Return in case of error

		return encode_result;

else

   encoded += encode_result;

}

LOG_TRACE(INFO, "ESM-SAP   - Remote UE Report message is out %d", len);

return encoded;
  }

