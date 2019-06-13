/*
 * RemoteUEContext.c
 *
 *  Created on: Jun 11, 2019
 *      Author: nepes
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "RemoteUEContext.h"
#include "TLVEncoder.h"
#include "TLVDecoder.h"
#include "nas_log.h"
//#include "RemoteUserID.h"



static int nas_decode_imsi (imsi_identity_t * imsi, uint8_t *  buffer, const uint8_t ie_len);

static int nas_encode_imsi (imsi_identity_t * imsi, uint8_t * buffer);


int decode_remote_ue_context(
		remote_ue_context_t *remoteuecontext,
		uint8_t iei,
		uint8_t *buffer,
		uint32_t len)
{

int                                     decoded = 0;
  uint8_t                                 ielen = 0;

  	  	  if (iei > 0)
  {
	  	  CHECK_IEI_DECODER (iei, *buffer);
	      decoded++;
  }

  DECODE_U8 (buffer + decoded, ielen, decoded);
    	  memset (remoteuecontext, 0, sizeof (remote_ue_context_t));
    	  //OAILOG_TRACE (LOG_NAS_EMM, "decode_remote_ue_context = %d\n", ielen);
    	  CHECK_LENGTH_DECODER (len - decoded, ielen);
    	  remoteuecontext->numberofuseridentity = *(buffer + decoded ) & 0x1;;
    	  //OAILOG_TRACE (LOG_NAS_EMM, "remoteuecontext decoded number of identities\n");


    	  if (iei > 1){
    	  remoteuecontext->imsi_identity->num_digits = *(buffer + decoded ) & 0xf;
    	  decoded++;
    	      	  }

    	  if (iei > 2){
          decoded += nas_decode_imsi(&remoteuecontext->imsi_identity, buffer + decoded, len - decoded);
    	  decoded++;

    	  }

    	  //OAILOG_TRACE (LOG_NAS_EMM, "remoteuserid decoded=%u\n", decoded);

    	  if ((ielen + 2) != decoded) {
    	  decoded = ielen + 1 + (iei > 0 ? 1 : 0) /* Size of header for this IE */ ;
    	            //OAILOG_TRACE (LOG_NAS_EMM, "remoteuecontext then decoded=%u\n", decoded);
    	   }
    	   return decoded;
           }



int encode_remote_ue_context(
		remote_ue_context_t *remoteuecontext,
		uint8_t iei,
		uint8_t *buffer,
		uint32_t len)
{
	uint8_t                                *lenPtr;
	uint32_t                                encoded = 0;

	if (iei > 0)
	{
		*buffer = iei;
		encoded++;
    }
	lenPtr = (buffer + encoded);
	encoded++;
	*(buffer + encoded) = remoteuecontext->numberofuseridentity;
	encoded++;
	*(buffer + encoded) = remoteuecontext->imsi_identity->num_digits;
	encoded++;
	encoded += nas_encode_imsi(&remoteuecontext->imsi_identity, buffer + encoded);

	*lenPtr = encoded - 1 - ((iei > 0) ? 1 : 0);
	return encoded;
}

//-----------------------------------------------------
 int nas_decode_imsi (imsi_identity_t * imsi, uint8_t *  buffer, const uint8_t ie_len)

{

	//OAILOG_FUNC_IN (LOG_NAS_EMM);
	  int                                     decoded = 0;
	  imsi->typeofidentity = *(buffer + decoded) & 0x7;
	//  if (imsi->typeofidentity != EPS_MOBILE_IDENTITY_IMSI)
	  //{
	    //return (TLV_VALUE_DOESNT_MATCH);
//}
	  imsi->oddeven = (*(buffer + decoded) >> 3) & 0x1;
	    imsi->identity_digit1 = (*(buffer + decoded) >> 4) & 0xf;
	    imsi->num_digits = 1;
	    decoded++;
	    if (decoded < ie_len)
	    {
	        imsi->identity_digit2 = *(buffer + decoded) & 0xf;
	        imsi->identity_digit3 = (*(buffer + decoded) >> 4) & 0xf;
	        decoded++;
	        imsi->num_digits += 2;
	        if (decoded < ie_len)
	        {
	          imsi->identity_digit4 = *(buffer + decoded) & 0xf;
	          imsi->num_digits++;
	          imsi->identity_digit5 = (*(buffer + decoded) >> 4) & 0xf;
	          if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit5 != 0x0f))
	          {
	          return (TLV_DECODE_VALUE_DOESNT_MATCH);
	          }
	          else
	          {
	           imsi->num_digits++;
	          }
	          decoded++;
	          if (decoded < ie_len)
	          {
	            imsi->identity_digit6 = *(buffer + decoded) & 0xf;
	            imsi->num_digits++;
	            imsi->identity_digit7 = (*(buffer + decoded) >> 4) & 0xf;
	          if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit7 != 0x0f))
	          {
	            return (TLV_DECODE_VALUE_DOESNT_MATCH);
	          }
	          else
{
	              imsi->num_digits++;
	          }
	            decoded++;
	            if (decoded < ie_len)
	            {
	              imsi->identity_digit8 = *(buffer + decoded) & 0xf;
	              imsi->num_digits++;
	              imsi->identity_digit9 = (*(buffer + decoded) >> 4) & 0xf;
	          if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit9 != 0x0f))
	          {
	             return (TLV_DECODE_VALUE_DOESNT_MATCH);
	           }
	           else
	          {
	                imsi->num_digits++;
	          }
	              decoded++;
	              if (decoded < ie_len)
	              {
	                imsi->identity_digit10 = *(buffer + decoded) & 0xf;
	                imsi->num_digits++;
	                imsi->identity_digit11 = (*(buffer + decoded) >> 4) & 0xf;
	           if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit11 != 0x0f)) {
	          return (TLV_DECODE_VALUE_DOESNT_MATCH);
	             }
	           else
	            {
	                  imsi->num_digits++;
	           }
	                decoded++;
	                if (decoded < ie_len)
	                {
	                  imsi->identity_digit12 = *(buffer + decoded) & 0xf;
	                  imsi->num_digits++;
	                  imsi->identity_digit13 = (*(buffer + decoded) >> 4) & 0xf;
	            if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit13 != 0x0f))
	            {
	              return (TLV_DECODE_VALUE_DOESNT_MATCH);
	             }
	              else
	             {
	                   imsi->num_digits++;
	            }
	                decoded++;
	                  if (decoded < ie_len)
	                  {
	                    imsi->identity_digit14 = *(buffer + decoded) & 0xf;
	                    imsi->num_digits++;
	                    imsi->identity_digit15 = (*(buffer + decoded) >> 4) & 0xf;
	            if ((IMSI_EVEN == imsi->oddeven)  && (imsi->identity_digit15 != 0x0f))
	             {
	                      return (TLV_DECODE_VALUE_DOESNT_MATCH);
	                    }
	                 else
	                    {
	                      imsi->num_digits++;
	                 }
	                    decoded++;
	                  }
	                }
	              }
	            }
	          }
	        }
	      }
return decoded;
	      //OAILOG_FUNC_RETURN (LOG_NAS_EMM, decoded);
	    }
//---------------------------------------------------------------------
int nas_encode_imsi (imsi_identity_t * imsi, uint8_t * buffer)
{
  uint32_t                                encoded = 0;

  *(buffer + encoded) = 0x00 | (imsi->identity_digit1 << 4) | (imsi->oddeven << 3) | (imsi->typeofidentity);
  encoded++;
  *(buffer + encoded) = 0x00 | (imsi->identity_digit3 << 4) | imsi->identity_digit2;
  encoded++;
  // Quick fix, should do a loop, but try without modifying struct!
  if (imsi->num_digits > 3) {
    if (imsi->oddeven != IMSI_EVEN) {
      *(buffer + encoded) = 0x00 | (imsi->identity_digit5 << 4) | imsi->identity_digit4;
    } else {
      *(buffer + encoded) = 0xf0 | imsi->identity_digit4;
    }
    encoded++;
    if (imsi->num_digits > 5) {
      if (imsi->oddeven != IMSI_EVEN) {
        *(buffer + encoded) = 0x00 | (imsi->identity_digit7 << 4) | imsi->identity_digit6;
      } else {
        *(buffer + encoded) = 0xf0 | imsi->identity_digit6;
     }
      encoded++;
      if (imsi->num_digits > 7) {
        if (imsi->oddeven != IMSI_EVEN) {
          *(buffer + encoded) = 0x00 | (imsi->identity_digit9 << 4) | imsi->identity_digit8;
        } else {
          *(buffer + encoded) = 0xf0 | imsi->identity_digit8;
        }
        encoded++;
        if (imsi->num_digits > 9) {
          if (imsi->oddeven != IMSI_EVEN) {
            *(buffer + encoded) = 0x00 | (imsi->identity_digit11 << 4) | imsi->identity_digit10;
          } else {
            *(buffer + encoded) = 0xf0 | imsi->identity_digit10;
          }
          encoded++;
          if (imsi->num_digits > 11) {
            if (imsi->oddeven != IMSI_EVEN) {
              *(buffer + encoded) = 0x00 | (imsi->identity_digit13 << 4) | imsi->identity_digit12;
           } else {
              *(buffer + encoded) = 0xf0 | imsi->identity_digit12;
            }
            encoded++;
            if (imsi->num_digits > 13) {
              if (imsi->oddeven != IMSI_EVEN) {
                *(buffer + encoded) = 0x00 | (imsi->identity_digit15 << 4) | imsi->identity_digit14;
              } else {
                *(buffer + encoded) = 0xf0 | imsi->identity_digit14;
              }
              encoded++;
            }
          }
        }
      }
    }
  }

  return encoded;
}



