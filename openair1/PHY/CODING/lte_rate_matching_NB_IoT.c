/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_CODING/lte_rate_matching_NB_IoT.c
* \Procedures for rate matching/interleaving for NB-IoT (turbo-coded transport channels) (TX/RX),	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

#ifdef MAIN
#include <stdio.h>
#include <stdlib.h>
#endif
#include "PHY/defs.h"
#include "assertions.h"

#include "PHY/defs_NB_IoT.h"

static uint32_t bitrev_cc[32] = {1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30};

uint32_t sub_block_interleaving_cc_NB_IoT(uint32_t D, uint8_t *d,uint8_t *w)
{
  uint32_t RCC = (D>>5), ND, ND3;   // D = 50 ,  
  uint32_t row,col,Kpi,index;
  uint32_t index3,k;

  if ((D&0x1f) > 0)
    RCC++;

  Kpi = (RCC<<5); 					// Kpi = 32
  ND = Kpi - D;
  ND3 = ND*3;   					// ND3 = ND*3 = 18 *3 = 54
  k=0;

  for (col=0; col<32; col++) {

    index = bitrev_cc[col];
    index3 = 3*index;

    for (row=0; row<RCC; row++) {
      w[k]          =   d[(int32_t)index3-(int32_t)ND3];
      w[Kpi+k]      =   d[(int32_t)index3-(int32_t)ND3+1];
      w[(Kpi<<1)+k] =   d[(int32_t)index3-(int32_t)ND3+2];

      index3+=96;
      index+=32;
      k++;
    }
  }
  return(RCC);
}


uint32_t lte_rate_matching_cc_NB_IoT(uint32_t RCC,      // RRC = 2
				     uint16_t E,        // E = 1600
				     uint8_t *w,	// length
				     uint8_t *e)	// length 1600
{
  uint32_t ind=0,k;

  uint16_t Kw = 3*(RCC<<5);   				  // 3*64 = 192

  for (k=0; k<E; k++) {

    while(w[ind] == LTE_NULL) {

      ind++;

      if (ind==Kw)
        ind=0;
    }

    e[k] = w[ind];
    ind++;

    if (ind==Kw)
      ind=0;
  }

  return(E);
}
