/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_CODING/ccoding_byte_NB_IoT.c
* \Fucntions for CRC attachment and tail-biting convolutional coding for NPBCH channel,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

#include "PHY/CODING/defs_NB_IoT.h"

unsigned char  ccodelte_table_NB_IoT[128];      // for transmitter
unsigned short glte_NB_IoT[] = { 0133, 0171, 0165 }; // {A,B} //renaimed but is exactly the same as the one in the old implementation

/*************************************************************************
  Encodes for an arbitrary convolutional code of rate 1/3
  with a constraint length of 7 bits.
  The inputs are bit packed in octets (from MSB to LSB).
  An optional 8-bit CRC (3GPP) can be added.
  Trellis tail-biting is included here
*************************************************************************/
void ccode_encode_NB_IoT (int32_t  numbits,
						              uint8_t  add_crc,
						              uint8_t  *inPtr,
						              uint8_t  *outPtr,
						              uint16_t rnti)
{
  uint32_t state;
  uint8_t  c, out, first_bit;
  int8_t   shiftbit=0;
  uint16_t c16;
  uint16_t next_last_byte=0;
  uint32_t crc=0;

  /* The input bit is shifted in position 8 of the state.
     Shiftbit will take values between 1 and 8 */
  state = 0;

  if (add_crc == 2) {

      crc = crc16_NB_IoT(inPtr,numbits);     // crc is 2 bytes
      // scramble with RNTI
      crc ^= (((uint32_t)rnti)<<16);  // XOR with crc
      first_bit = 2;
      c = (uint8_t)((crc>>16)&0xff);

  } else {

      next_last_byte = numbits>>3;
      first_bit      = (numbits-6)&7;
      c = inPtr[next_last_byte-1];
  }

  // Perform Tail-biting
  // get bits from last byte of input (or crc)
  
  for (shiftbit = 0 ; shiftbit <(8-first_bit) ; shiftbit++) {

      if ((c&(1<<(7-first_bit-shiftbit))) != 0)
          state |= (1<<shiftbit);
  }

  state = state & 0x3f;   // true initial state of Tail-biting CCode
  state<<=1;              // because of loop structure in CCode

  while (numbits > 0) {											// Tail-biting is applied to input bits , input 34 bits , output 102 bits

    c = *inPtr++;

    for (shiftbit = 7; (shiftbit>=0) && (numbits>0); shiftbit--,numbits--) {

        state >>= 1;

        if ((c&(1<<shiftbit)) != 0) {
            state |= 64;
        }

        out = ccodelte_table_NB_IoT[state];

        *outPtr++ = out  & 1;
        *outPtr++ = (out>>1)&1;
        *outPtr++ = (out>>2)&1;
    }
  }
  
  // now code 16-bit CRC for DCI 						// Tail-biting is applied to CRC bits , input 16 bits , output 48 bits
  if (add_crc == 2) {

    c16 = (uint16_t)(crc>>16);
	
    for (shiftbit = 15; (shiftbit>=0); shiftbit--) {

        state >>= 1;

        if ((c16&(1<<shiftbit)) != 0) {
            state |= 64;
        }

        out = ccodelte_table_NB_IoT[state];

        *outPtr++ = out  & 1;
        *outPtr++ = (out>>1)&1;
        *outPtr++ = (out>>2)&1;  
    }
  }
}
/*************************************************************************

  Functions to initialize the code tables

*************************************************************************/
/* Basic code table initialization for constraint length 7 */

/* Input in MSB, followed by state in 6 LSBs */
void ccodelte_init_NB_IoT(void)
{
  unsigned int  i, j, k, sum;

  for (i = 0; i < 128; i++) {

    ccodelte_table_NB_IoT[i] = 0;

    /* Compute 3 output bits */
    for (j = 0; j < 3; j++) {
      sum = 0;

      for (k = 0; k < 7; k++)
        if ((i & glte_NB_IoT[j]) & (1 << k))
          sum++;

      /* Write the sum modulo 2 in bit j */
      ccodelte_table_NB_IoT[i] |= (sum & 1) << j;
    }
  }
}

