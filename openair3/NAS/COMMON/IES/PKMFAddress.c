/*
 * PKMFAddress.c
 *
 *  Created on: Jun 11, 2019
 *      Author: nepes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include "PKMFAddress.h"
#include "TLVDecoder.h"

//static int encode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);

//static int decode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);


int decode_pkmf_address(
		pkmf_address_t *pkmfaddress,
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
  	  	  	  memset (pkmfaddress, 0, sizeof (pkmf_address_t));
  	  	  	  //OAILOG_TRACE (LOG_NAS_EMM, "decode_pkmf_address = %d\n", ielen);
  	  	  	  CHECK_LENGTH_DECODER (len - decoded, ielen);
  	  	  	  pkmfaddress->spare = (*(buffer + decoded) >> 3) & 0x1;
  	  	  	  pkmfaddress->addresstype = *(buffer + decoded) & 0x1;
  	  	      decoded++;
  	  	  if (iei > 1)
  	  	  {
  	  		pkmfaddress->pkmfipv4address = *(buffer + decoded ) & 0xf;
  	  		decoded++;
  	  	  }

	//OAILOG_TRACE (LOG_NAS_EMM, "PKMFAddress decoded=%u\n", decoded);

        if ((ielen + 2) != decoded) {
          decoded = ielen + 1 + (iei > 0 ? 1 : 0) /* Size of header for this IE */ ;
          //OAILOG_TRACE (LOG_NAS_EMM, "PKMFAddress then decoded=%u\n", decoded);
        }
        return decoded;
        }

int encode_pkmf_address(
		pkmf_address_t *pkmfaddress,
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
	    lenPtr = (buffer + encoded);
	    encoded++;
	 	*(buffer + encoded) = 0x00 | ((pkmfaddress->spare & 0x1) << 3) | (pkmfaddress->addresstype & 0x1);
	    encoded++;
	    lenPtr = (buffer + encoded);
	    encoded++;
	    *(buffer + encoded) = pkmfaddress->pkmfipv4address;
	    encoded++;
	    *lenPtr = encoded - 1 - ((iei > 0) ? 1 : 0);
	}
	    	return encoded;


	}

