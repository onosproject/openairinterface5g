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
#include "TLVEncoder.h"


//static int encode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);

//static int decode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);

//static int encode_pkmf_addressfield(pkmfaddress_t *pkmf, uint8_t *buffer);

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
  	  	  	  pkmfaddress->spare = (*(buffer + decoded) >> 3) & 0xf;
  	  	  	  pkmfaddress->addresstype = *(buffer + decoded) & 0xf;
  	  	      decoded++;
  	  	  if (iei > 1)
  	  	  {
  	  		//pkmfaddress->pkmfipaddress = *(buffer + decoded ) & 0xf;
  	  		decoded++;
  	  	  }

	//OAILOG_TRACE (LOG_NAS_EMM, "PKMFAddress decoded=%u\n", decoded);

        if ((ielen + 2) != decoded) {
          decoded = ielen + 1 + (iei > 0 ? 1 : 0) /* Size of header for this IE */ ;

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
	int encoded_rc = TLV_ENCODE_VALUE_DOESNT_MATCH;
	LOG_TRACE(INFO, "Send PKMF address %d", len);

	/* Checking IEI and pointer */
	CHECK_PDU_POINTER_AND_LENGTH_ENCODER(buffer, PKMF_ADDRESS_MINIMUM_LENGTH, len);
#if defined (NAS_DEBUG)
	dump_pkmf_address_xml(pkmfaddress, iei);
#endif

	if (iei > 0)
	{
		*buffer = iei;
	 	 encoded++;
	}

	   lenPtr = (buffer + encoded);
	   encoded++;
	   //pkmfaddress->addresstype = ADDRESS_TYPE_IPV4;
	   *(buffer + encoded) =  0x00 | (((pkmfaddress->spare) & 0x1f) << 3)|
	   ((pkmfaddress->addresstype = ADDRESS_TYPE_IPV4)& 0x7);
	   encoded++;

	   //if (pkmfaddress->addresstype = ADDRESS_TYPE_IPV4){
	   *(buffer + encoded) =  (&pkmfaddress->pkmfipaddress);
	   memcpy((void*)(buffer + encoded), (const void*)(&pkmfaddress->pkmfipaddress.ipv4), 4);
	   encoded += 4;
	   // }
	   //else if(pkmfaddress->addresstype = ADDRESS_TYPE_IPV6)
	   // {
	   //*(buffer + encoded) = (&pkmfaddress->pkmfipaddress);
	   // memcpy((void*)(buffer + encoded), (const void*)(&pkmfaddress->pkmfipaddress.ipv6),6);
	   // encoded +=6;
	   //}

        //encoded++;

	   *lenPtr = encoded - 1 - ((iei > 0) ? 1 : 0);
  		LOG_TRACE(INFO, "Send PKMF address %d", len);

	    return encoded;
	}

void dump_pkmf_address_xml(pkmf_address_t *pkmfaddress, uint8_t iei)
{
  printf("<PKMF Address>\n");

  if (iei > 0)
    /* Don't display IEI if = 0 */
    printf("    <IEI>0x%X</IEI>\n", iei);

  printf("%s</Access Point Name>\n",
         dump_octet_string_xml(&pkmfaddress->pkmfipaddress));
}

