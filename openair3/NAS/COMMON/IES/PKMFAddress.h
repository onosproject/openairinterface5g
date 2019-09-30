/*
 * PKMFAddress.h
 *
 *  Created on: Jun 11, 2019
 *      Author: nepes
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <netinet/in.h>


#ifndef OPENAIR3_NAS_COMMON_IES_PKMFADDRESS_H_
#define OPENAIR3_NAS_COMMON_IES_PKMFADDRESS_H_

#define PKMF_ADDRESS_MINIMUM_LENGTH 2
#define PKMF_ADDRESS_MAXIMUM_LENGTH 5

//typedef struct pkmf_s{
//uint8_t digit1:8;
//uint8_t digit2:8;
//uint8_t digit3:8;
//uint8_t digit4:8;
//}pkmf_t;

//typedef pkmf_t pkmfaddress_t;


typedef struct pkmf_address_s {
uint8_t  spare:5;
#define ADDRESS_TYPE_IPV4 0b001
#define ADDRESS_TYPE_IPV6 0b010
uint8_t  addresstype:3;
#define PKMF_IP_ADDRESS
union {
struct in_addr ipv4;  // char ipv4[4]; 4 bytes
struct in6_addr ipv6; // char ipv6[16] // 16 bytes
}pkmfipaddress;
//pkmfaddress_t  pkmfipv4address;
}pkmf_address_t;


int encode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);

int decode_pkmf_address(pkmf_address_t *pkmfaddress, uint8_t iei, uint8_t *buffer, uint32_t len);

void dump_pkmf_address_xml(pkmf_address_t *pkmfaddress, uint8_t iei);

#endif /* OPENAIR3_NAS_COMMON_IES_PKMFADDRESS_H_ */
