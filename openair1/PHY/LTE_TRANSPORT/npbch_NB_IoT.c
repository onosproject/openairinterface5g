/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_TRANSPORT/npbch_NB_IoT.c
* \Fucntions for the generation of broadcast channel (NPBCH) for NB_IoT,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

//#include "PHY/defs.h"
//#include "PHY/defs_NB_IoT.h"
//#include "PHY/CODING/extern.h"
//#include "PHY/CODING/lte_interleaver_inline.h"
//#include "extern_NB_IoT.h"
//#include "PHY/extern_NB_IoT.h"
//#include "PHY/sse_intrin.h"
#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "PHY/CODING/defs_NB_IoT.h"
#include "PHY/LTE_REFSIG/defs_NB_IoT.h"
#include "PHY/impl_defs_lte_NB_IoT.h"
#include "PHY/impl_defs_top_NB_IoT.h"

//#ifdef PHY_ABSTRACTION
//#include "SIMULATION/TOOLS/defs.h"
//#endif

//#ifdef OPENAIR2
//#include "PHY_INTERFACE/defs.h"
//#endif

#define NPBCH_A 34                             // 34 for NB-IoT and 24 for LTE

int allocate_npbch_REs_in_RB(NB_IoT_DL_FRAME_PARMS 	*frame_parms,
                             int32_t 				**txdataF,
                             uint32_t 				*jj,
                             uint32_t 				symbol_offset,
                             uint8_t 				*x0,
                             uint8_t 				pilots,
                             int16_t 				amp,
			     unsigned short 		id_offset,
                             uint32_t 				*re_allocated)  //  not used variable ??!!
{
  MIMO_mode_NB_IoT_t mimo_mode = (frame_parms->mode1_flag==1)?SISO_NB_IoT:ALAMOUTI_NB_IoT;

  uint32_t  tti_offset,aa;
  uint8_t   re;
  int16_t   gain_lin_QPSK;
  uint8_t   first_re,last_re;
  int32_t   tmp_sample1,tmp_sample2;

  gain_lin_QPSK = (int16_t)((amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
  first_re      = 0;
  last_re       = 12;

  for (re=first_re; re<last_re; re++) {      		// re varies between 0 and 12 sub-carriers

    tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re
	
	if (pilots != 1 || re%3 != id_offset)  			// if re is not a pilot
	{
													//	diff_re = re%3 - id_offset;  
      if (mimo_mode == SISO_NB_IoT) {  								//SISO mapping
        *re_allocated = *re_allocated + 1;						// variable incremented but never used
			for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
					((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
			}
			*jj = *jj + 1;
			for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
				((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
			}
			*jj = *jj + 1;	
      } else if (mimo_mode == ALAMOUTI_NB_IoT) {
			*re_allocated = *re_allocated + 1;

			((int16_t*)&tmp_sample1)[0] = (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
			*jj=*jj+1;
			((int16_t*)&tmp_sample1)[1] = (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
			*jj=*jj+1;

			// second antenna position n -> -x1*

			((int16_t*)&tmp_sample2)[0] = (x0[*jj]==1) ? (gain_lin_QPSK) : -gain_lin_QPSK;
			*jj=*jj+1;
			((int16_t*)&tmp_sample2)[1] = (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
			*jj=*jj+1;

			// normalization for 2 tx antennas
			((int16_t*)&txdataF[0][tti_offset])[0] += (int16_t)((((int16_t*)&tmp_sample1)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
			((int16_t*)&txdataF[0][tti_offset])[1] += (int16_t)((((int16_t*)&tmp_sample1)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
			((int16_t*)&txdataF[1][tti_offset])[0] += (int16_t)((((int16_t*)&tmp_sample2)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
			((int16_t*)&txdataF[1][tti_offset])[1] += (int16_t)((((int16_t*)&tmp_sample2)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);

			// fill in the rest of the ALAMOUTI precoding
			if ( pilots != 1 || (re+1)%3 != id_offset) {
				((int16_t *)&txdataF[0][tti_offset+1])[0] += -((int16_t *)&txdataF[1][tti_offset])[0]; //x1
				((int16_t *)&txdataF[0][tti_offset+1])[1] += ((int16_t *)&txdataF[1][tti_offset])[1];
				((int16_t *)&txdataF[1][tti_offset+1])[0] += ((int16_t *)&txdataF[0][tti_offset])[0];  //x0*
				((int16_t *)&txdataF[1][tti_offset+1])[1] += -((int16_t *)&txdataF[0][tti_offset])[1];
			} else {
				((int16_t *)&txdataF[0][tti_offset+2])[0] += -((int16_t *)&txdataF[1][tti_offset])[0]; //x1
				((int16_t *)&txdataF[0][tti_offset+2])[1] += ((int16_t *)&txdataF[1][tti_offset])[1];
				((int16_t *)&txdataF[1][tti_offset+2])[0] += ((int16_t *)&txdataF[0][tti_offset])[0];  //x0*
				((int16_t *)&txdataF[1][tti_offset+2])[1] += -((int16_t *)&txdataF[0][tti_offset])[1];
		
				re++;														// skip pilots
				*re_allocated = *re_allocated + 1;
			}
			re++;  															// adjacent carriers are taken care of by precoding
			*re_allocated = *re_allocated + 1;   							// incremented variable but never used
		}
    }
  }

  return(0);
}
/**********************************************************
**********************************************************/
int generate_npbch(NB_IoT_eNB_NPBCH_t 		*eNB_npbch,
                   int32_t 					**txdataF,
                   int 						amp,
                   NB_IoT_DL_FRAME_PARMS 	*frame_parms,
                   uint8_t 					*npbch_pdu,
                   uint8_t 					frame_mod64,
				   unsigned short 			NB_IoT_RB_ID)
{
  int 			  i, l;
  int 			  id_offset;
  uint32_t 		  npbch_D,npbch_E;
  uint8_t 		  npbch_a[5];   							// 34/8 =4.25 => 4 bytes and 2 bits
  uint8_t 		  RCC;
  unsigned short  bandwidth_even_odd;
  unsigned short  NB_IoT_start, RB_IoT_ID;
  uint32_t 		  pilots;
  uint32_t 		  jj=0;
  uint32_t 		  re_allocated=0;
  uint32_t 		  symbol_offset;
  uint16_t 		  amask=0;

  npbch_D  = 16 + NPBCH_A;
  npbch_E  = 1600; 									
									
	if (frame_mod64==0) {
		bzero(npbch_a,5);      									// initializing input data stream , filling with zeros
		bzero(eNB_npbch->npbch_e,npbch_E);						// filling with "0" the table pbch_e[1600]
		memset(eNB_npbch->npbch_d,LTE_NULL_NB_IoT,96);					// filling with "2" the first 96 elements of table pbch_d[216]
		
		for (i=0; i<5; i++) 									// set input bits stream
		{	
			if (i != 4)
			{
				npbch_a[5-i-1] = npbch_pdu[i];            		// ????????/*****?? in LTE 24 bits with 3 bytes, but in NB_IoT 34 bits will require 4 bytes+2 bits !! to verify
			} else {
				npbch_a[5-i-1]= npbch_pdu[i] & 0x03;
			}
		}
	
		if (frame_parms->mode1_flag == 1)						// setting CRC mask depending on the number of used eNB antennas 
			amask = 0x0000;
		else {
			switch (frame_parms->nb_antennas_tx) {			// *****???? better replacing nb_antennas_tx_eNB by nb_antennas_tx_eNB_NB_IoT
				case 1:
					amask = 0x0000;
				break;
				case 2:
					amask = 0xffff;
				break;
			}
		}
		
		ccode_encode_NB_IoT(NPBCH_A,2,npbch_a,eNB_npbch->npbch_d+96,amask);						// step 1 CRC Attachment
		RCC = sub_block_interleaving_cc_NB_IoT(npbch_D,eNB_npbch->npbch_d+96,eNB_npbch->npbch_w);   	// step 2 Channel Coding
		lte_rate_matching_cc_NB_IoT(RCC,npbch_E,eNB_npbch->npbch_w,eNB_npbch->npbch_e);				// step 3 Rate Matching
		npbch_scrambling(frame_parms,															// step 4 Scrambling
						 eNB_npbch->npbch_e,
						 npbch_E);

	}
	// testing if the total number of RBs is even or odd 
	bandwidth_even_odd = frame_parms->N_RB_DL % 2; 	 	// 0 even, 1 odd
	RB_IoT_ID = NB_IoT_RB_ID;
	// step  5, 6, 7   									 	// modulation and mapping (slot 1, symbols 0..3)
	for (l=3; l<14; l++) { 								 	// loop on OFDM symbols
			
		if((l>=4 && l<=8) || (l>=11 && l<=13))
		{
			pilots =1;
		} else {
			pilots=0;
		}
		
		id_offset = frame_parms->Nid_cell % 3;    		// Cell_ID_NB_IoT % 3
		
		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID%(int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		
		symbol_offset = frame_parms->ofdm_symbol_size*l + NB_IoT_start;  						// symbol_offset = 512 * L + NB_IOT_RB start
		
		allocate_npbch_REs_in_RB(frame_parms,
								 txdataF,
								 &jj,
								 symbol_offset,
								 &eNB_npbch->npbch_e[(frame_mod64/8)*(npbch_E>>3)],
								 pilots,
								 amp,
								 id_offset,
								 &re_allocated);
			
	
	}
return(0);
}
/**********************************************************
**********************************************************/
void npbch_scrambling(NB_IoT_DL_FRAME_PARMS 	*frame_parms,
                      uint8_t 					*npbch_e,
                      uint32_t 					length)  // 1600
{
  int 		i;
  uint8_t 	reset;
  uint32_t 	x1, x2, s=0;

  reset = 1;
  x2 	= frame_parms->Nid_cell;

  for (i=0; i<length; i++) {
    if ((i&0x1f)==0) {

      s 	= lte_gold_generic_NB_IoT(&x1, &x2, reset);
      reset = 0;
    }
    npbch_e[i] = (npbch_e[i]&1) ^ ((s>>(i&0x1f))&1);
  }
}
