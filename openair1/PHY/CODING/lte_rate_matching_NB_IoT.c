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

/* // check if this ifdef MAIN is required for NB-IoT
#ifdef MAIN
#include <stdio.h>
#include <stdlib.h>
#endif
*/

//#include "PHY/CODING/defs_NB_IoT.h"
#include "PHY/defs_L1_NB_IoT.h"
//#include "assertions.h"

//#include "PHY/LTE_REFSIG/defs_NB_IoT.h"   // does this file is needed ?

static uint32_t bitrev_cc_NB_IoT[32] = {1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31,0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30};

uint32_t sub_block_interleaving_cc_NB_IoT(uint32_t D, uint8_t *d,uint8_t *w)
{
    uint32_t RCC = (D>>5), ND, ND3;   // D = 50 ,  
    uint32_t row,col,Kpi,index;
    uint32_t index3,k;

    if ((D&0x1f) > 0)
        RCC++;

    Kpi = (RCC<<5); 					// Kpi = 32
    ND = Kpi - D;
    ND3 = ND*3;   					  // ND3 = ND*3 = 18 *3 = 54
    k=0;

    for (col=0; col<32; col++) {

      index = bitrev_cc_NB_IoT[col];
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


uint32_t lte_rate_matching_cc_NB_IoT(uint32_t  RCC,      // RRC = 2
				                             uint16_t  E,        // E = 1600
				                             uint8_t   *w,	     // length
				                             uint8_t   *e)	     // length 1600
{
  uint32_t ind=0,k;
  uint16_t Kw = 3*(RCC<<5);   				  // 3*64 = 192

  for (k=0; k<E; k++) {

    while(w[ind] == LTE_NULL_NB_IoT) {

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

//******************* below functions related to uplink transmission , to be reviwed *********
// this function should be adapted to NB-IoT , this deinterleaving is for LTE
void sub_block_deinterleaving_cc_NB_IoT(uint32_t D,int8_t *d,int8_t *w)
{

  //WANG_Hao uint32_t RCC = (D>>5), ND, ND3;
  uint32_t RCC = (D>>5);
  ptrdiff_t   ND, ND3;
  uint32_t row,col,Kpi,index;
  //WANG_Hao uint32_t index3,k;
  ptrdiff_t index3;
  uint32_t k;

  if ((D&0x1f) > 0)
    RCC++;

  Kpi = (RCC<<5);
  //  Kpi3 = Kpi*3;
  ND = Kpi - D;

  ND3 = ND*3;

  k=0;

  for (col=0; col<32; col++) {

    index = bitrev_cc_NB_IoT[col];
    index3 = 3*index;

    for (row=0; row<RCC; row++) {

      d[index3-ND3]   = w[k];
      d[index3-ND3+1] = w[Kpi+k];
      d[index3-ND3+2] = w[(Kpi<<1)+k];

      index3+=96;
      index+=32;
      k++;
    }
  }

}



void lte_rate_matching_cc_rx_NB_IoT(uint32_t RCC,
                                    uint16_t E,
                                    int8_t *w,
                                    uint8_t *dummy_w,
                                    int8_t *soft_input)
{



  uint32_t ind=0,k;
  uint16_t Kw = 3*(RCC<<5);
  uint32_t acc=1;
  int16_t w16[Kw];

  memset(w,0,Kw);
  memset(w16,0,Kw*sizeof(int16_t));

  for (k=0; k<E; k++) {


    while(dummy_w[ind] == LTE_NULL_NB_IoT) {

      ind++;

      if (ind==Kw)
        ind=0;
    }

 
    w16[ind] += soft_input[k];

    ind++;

    if (ind==Kw) {
      ind=0;
      acc++;
    }
  }

  // rescale
  for (ind=0; ind<Kw; ind++) {
    //    w16[ind]=(w16[ind]/acc);
    if (w16[ind]>7)
      w[ind]=7;
    else if (w16[ind]<-8)
      w[ind]=-8;
    else
      w[ind]=(int8_t)w16[ind];
  }

}


uint32_t generate_dummy_w_cc_NB_IoT(uint32_t D, uint8_t *w)
{

  uint32_t RCC = (D>>5), ND;
  uint32_t col,Kpi,index;
  int32_t k;

  if ((D&0x1f) > 0)
    RCC++;

  Kpi = (RCC<<5);
  //  Kpi3 = Kpi*3;
  ND = Kpi - D;

  // copy d02 to dD2 (for mod Kpi operation from clause (4), p.16 of 36.212
  k=0;

  for (col=0; col<32; col++) {

    index = bitrev_cc_NB_IoT[col];

    if (index<ND) {
      w[k]          = LTE_NULL_NB_IoT;
      w[Kpi+k]      = LTE_NULL_NB_IoT;
      w[(Kpi<<1)+k] = LTE_NULL_NB_IoT;

    }


    k+=RCC;
  }

  return(RCC);
}
