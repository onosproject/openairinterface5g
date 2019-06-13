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

int decode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len)
{
uint32_t decoded = 0;
int decoded_result = 0;

// Check if we got a NULL pointer and if buffer length is >= minimum length expected for the message.
CHECK_PDU_POINTER_AND_LENGTH_DECODER(buffer, REMOTE_UE_REPORT_RESPONSE_MINIMUM_LENGTH, len);

if ((decoded_result = decode_pkmf_address(&remoteuereport->pkmfaddress, 0, buffer + decoded, len - decoded)) < 0)
    return decoded_result;
  else
    decoded += decoded_result;
return decoded;
}


int encode_remote_ue_report(remote_ue_report_msg *remoteuereport, uint8_t *buffer, uint32_t len)
{
int encoded = 0;
int encode_result = 0;
/* Checking IEI and pointer */
CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, REMOTE_UE_REPORT_RESPONSE_MINIMUM_LENGTH, len);

remoteuereport->pkmfaddress.pkmfipv4address = 0;
if ((encode_result = encode_pkmf_address(&remoteuereport->pkmfaddress, 0, buffer + encoded, len - encoded)) < 0)//Return in case of error
   return encode_result;
else
   encoded += encode_result;
return encoded;
}
