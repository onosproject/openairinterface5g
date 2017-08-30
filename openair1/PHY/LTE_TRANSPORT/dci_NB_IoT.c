/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file PHY/LTE_TRANSPORT/dci.c
* \brief Implements PDCCH physical channel TX/RX procedures (36.211) and DCI encoding/decoding (36.212/36.213). Current LTE compliance V8.6 2009-03.
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif
//#include "PHY/defs.h"
//#include "PHY/LTE_TRANSPORT/dci_NB_IoT.h"
//#include "PHY/CODING/defs_NB_IoT.h"
#include "PHY/defs_NB_IoT.h"  // /LTE_TRANSPORT/defs_NB_IoT.h
//#include "PHY/LTE_REFSIG/defs_NB_IoT.h"
//#include "PHY/extern.h"
//////////#include "PHY/extern_NB_IoT.h"
//#include "SCHED/defs.h"
/////////////////////////////#include "SCHED/defs_nb_iot.h"
//#include "SIMULATION/TOOLS/defs.h" // for taus 
//#include "PHY/sse_intrin.h"

//#include "assertions.h" 
//#include "T.h"


//------------------------------------------------
// BCOM code functions npdcch start
// (TODO solve some error in compilation)
//------------------------------------------------
static uint8_t d[2][3*(MAX_DCI_SIZE_BITS_NB_IoT + 16) + 96];
static uint8_t w[2][3*3*(MAX_DCI_SIZE_BITS_NB_IoT+16)];



void dci_encoding_NB_IoT(uint8_t    *a[2],				// Array of two DCI pdus, even if one DCI is to transmit , the number of DCI is indicated in dci_number
						 uint8_t    A,					// Length of array a (in number of bytes)(es 4 bytes = 32 bits) is a parameter fixed
						 uint16_t   E,					// E should equals to G (number of available bits in one RB)
						 uint8_t    *e[2],				// *e should be e[2][G]
						 uint16_t   rnti[2],			// RNTI for UE specific or common search space
						 uint8_t    dci_number,			// This variable should takes the 1 or 2 (1 for in case of one DCI, 2 in case of two DCI)
						 uint8_t    agr_level)			// Aggregation level
{
	uint8_t  D = (A + 16);
	uint32_t RCC;
	uint8_t  occupation_size=1;
	// encode dci
	if(dci_number == 1)
	{
		if(agr_level == 2)
		{
			occupation_size=1;
		}else{
			occupation_size=2;
		}
		memset((void *)d[0],LTE_NULL_NB_IoT,96);

		ccode_encode_NB_IoT(A,2,a[0],d[0]+96,rnti[0]);    					// CRC attachement & Tail-biting convolutional coding
		RCC = sub_block_interleaving_cc_NB_IoT(D,d[0]+96,w[0]);				// Interleaving
		lte_rate_matching_cc_NB_IoT(RCC,(E/occupation_size),w[0],e[0]);		// Rate Matching

	}else if (dci_number == 2) {

		memset((void *)d[0],LTE_NULL_NB_IoT,96);
		memset((void *)d[1],LTE_NULL_NB_IoT,96);
		// first DCI encoding
		ccode_encode_NB_IoT(A,2,a[0],d[0]+96,rnti[0]);    					// CRC attachement & Tail-biting convolutional coding
		RCC = sub_block_interleaving_cc_NB_IoT(D,d[0]+96,w[0]);				// interleaving
		lte_rate_matching_cc_NB_IoT(RCC,E/2,w[0],e[0]);						// Rate Matching , E/2 ,  NCCE0
		// second DCI encoding
		ccode_encode_NB_IoT(A,2,a[1],d[1]+96,rnti[1]);    					// CRC attachement & Tail-biting convolutional coding
		RCC = sub_block_interleaving_cc_NB_IoT(D,d[1]+96,w[1]);				// Interleaving
		lte_rate_matching_cc_NB_IoT(RCC,E/2,w[1],e[1]);						// Rate Matching, E/2 , NCCE1

	}
}

///The scrambling sequence shall be initialised at the start of the search space and after every 4th NPDCCH subframes.
///
///
void npdcch_scrambling_NB_IoT(NB_IoT_DL_FRAME_PARMS    *frame_parms,
							  uint8_t                  *e[2],			// Input data
							  int                      length,        	// Total number of bits to transmit in one subframe(case of DCI = G)
							  uint8_t 				   Ns,				//XXX we pass the subframe	// Slot number (0..19)
							  uint8_t 				   dci_number,		// This variable should takes the 1 or 2 (1 for in case of one DCI, 2 in case of two DCI)
							  uint8_t 				   agr_level)		// Aggregation level
{
	int 	  i,k=0;
	uint32_t  x1, x2, s=0;
	uint8_t   reset;
	uint8_t   occupation_size=1;

	reset = 1;

	if(agr_level == 2)
	{
		occupation_size=1;
	}else{
		occupation_size=2;
	}

	if(dci_number == 1)														// Case of one DCI
	{
		x2 = ((Ns>>1)<<9) + frame_parms->Nid_cell;                          // This is c_init in 36.211 Sec 10.2.3.1

		for (i=0; i<length/occupation_size; i++) {
			if ((i&0x1f)==0) {
				s = lte_gold_generic_NB_IoT(&x1, &x2, reset);
				reset = 0;
			}
			e[0][k] = (e[0][k]&1) ^ ((s>>(i&0x1f))&1);
		}

	}else if(dci_number == 2 && occupation_size == 2) {						// Case of two DCI

		// Scrambling the first DCI
		//
		x2 = ((Ns>>1)<<9) + frame_parms->Nid_cell;                          // This is c_init in 36.211 Sec 10.2.3.1

		for (i=0; i<length/occupation_size; i++) {
			if ((i&0x1f)==0) {
				s = lte_gold_generic_NB_IoT(&x1, &x2, reset);
				reset = 0;
			}
			e[0][k] = (e[0][k]&1) ^ ((s>>(i&0x1f))&1);
		}
		// reset of the scrambling function
		reset = 1;
		// Scrambling the second DCI
		//
		x2 = ((Ns>>1)<<9) + frame_parms->Nid_cell;                          //this is c_init in 36.211 Sec 10.2.3.1

		for (i=0; i<length/occupation_size; i++) {
			if ((i&0x1f)==0) {
				s = lte_gold_generic_NB_IoT(&x1, &x2, reset);
				reset = 0;
			}
			e[1][k] = (e[1][k]&1) ^ ((s>>(i&0x1f))&1);
		}
	}
}


int dci_allocate_REs_in_RB_NB_IoT(NB_IoT_DL_FRAME_PARMS 	*frame_parms,
                                  int32_t 					**txdataF,
                                  uint32_t 					*jj,
                                  uint32_t 					symbol_offset,
                                  uint8_t 					*x0[2],
                                  uint8_t 					pilots,
                                  int16_t 					amp,
						  	      unsigned short 			id_offset,
                                  uint32_t 					*re_allocated, 	//  not used variable ??!!
								  uint8_t 					dci_number,		// This variable should takes the 1 or 2 (1 for in case of one DCI, 2 in case of two DCI)
								  uint8_t 					agr_level)
{
	MIMO_mode_NB_IoT_t mimo_mode = (frame_parms->mode1_flag==1)?SISO_NB_IoT:ALAMOUTI_NB_IoT;

	uint32_t  tti_offset,aa;
	uint8_t   re;
	int16_t   gain_lin_QPSK;
	uint8_t   first_re,last_re;
	int32_t   tmp_sample1,tmp_sample2,tmp_sample3,tmp_sample4;

	gain_lin_QPSK = (int16_t)((amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
	first_re=0;
	last_re=12;

	if(agr_level == 2 && dci_number == 1)
	{
		for (re=first_re; re<last_re; re++) {      	// re varies between 0 and 12 sub-carriers

			tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re

			if (pilots != 1 || re%3 != id_offset)  			// if re is not a pilot
			{
															//	diff_re = re%3 - id_offset;
				if (mimo_mode == SISO_NB_IoT) {  					//SISO mapping
					*re_allocated = *re_allocated + 1;				// variable incremented but never used

					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
					}
					*jj = *jj + 1;
					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
					}
					*jj = *jj + 1;

				} else if (mimo_mode == ALAMOUTI_NB_IoT) {

					*re_allocated = *re_allocated + 1;

					((int16_t*)&tmp_sample1)[0] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample1)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;

					// second antenna position n -> -x1*

					((int16_t*)&tmp_sample2)[0] = (x0[0][*jj]==1) ? (gain_lin_QPSK) : -gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample2)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
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
  	}else if(agr_level == 1 && dci_number == 1){

		for (re=first_re; re<6; re++) {      		// re varies between 0 and 6 sub-carriers

    		tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re

			if (pilots != 1 || re%3 != id_offset)  			// if re is not a pilot
			{
													//	diff_re = re%3 - id_offset;
				if (mimo_mode == SISO_NB_IoT) {  								//SISO mapping
					*re_allocated = *re_allocated + 1;						// variable incremented but never used

					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
					}
					*jj = *jj + 1;
					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
					}
					*jj = *jj + 1;

				} else if (mimo_mode == ALAMOUTI_NB_IoT) {

					*re_allocated = *re_allocated + 1;

					((int16_t*)&tmp_sample1)[0] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample1)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;

					// second antenna position n -> -x1*

					((int16_t*)&tmp_sample2)[0] = (x0[0][*jj]==1) ? (gain_lin_QPSK) : -gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample2)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
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
 	} else {

		// allocate first DCI
		for (re=first_re; re<6; re++) {      		// re varies between 0 and 12 sub-carriers

    		tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re

			if (pilots != 1 || re%3 != id_offset) { 			// if re is not a pilot
			
													//	diff_re = re%3 - id_offset;
     			if (mimo_mode == SISO_NB_IoT) {  								//SISO mapping
					*re_allocated = *re_allocated + 1;						// variable incremented but never used

					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
						((int16_t*)&txdataF[aa][tti_offset+6])[0] += (x0[1][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
					}
					*jj = *jj + 1;
					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
						((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
						((int16_t*)&txdataF[aa][tti_offset+6])[1] += (x0[1][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
					}
					*jj = *jj + 1;

     			} else if (mimo_mode == ALAMOUTI_NB_IoT) {

					*re_allocated = *re_allocated + 1;

					((int16_t*)&tmp_sample1)[0] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					((int16_t*)&tmp_sample3)[0] = (x0[1][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample1)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					((int16_t*)&tmp_sample3)[1] = (x0[1][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;

					// second antenna position n -> -x1*

					((int16_t*)&tmp_sample2)[0] = (x0[0][*jj]==1) ? (gain_lin_QPSK) : -gain_lin_QPSK;
					((int16_t*)&tmp_sample4)[0] = (x0[1][*jj]==1) ? (gain_lin_QPSK) : -gain_lin_QPSK;
					*jj=*jj+1;
					((int16_t*)&tmp_sample2)[1] = (x0[0][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					((int16_t*)&tmp_sample4)[1] = (x0[1][*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK;
					*jj=*jj+1;

					// normalization for 2 tx antennas
					((int16_t*)&txdataF[0][tti_offset])[0] += (int16_t)((((int16_t*)&tmp_sample1)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[0][tti_offset])[1] += (int16_t)((((int16_t*)&tmp_sample1)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[1][tti_offset])[0] += (int16_t)((((int16_t*)&tmp_sample2)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[1][tti_offset])[1] += (int16_t)((((int16_t*)&tmp_sample2)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);

					((int16_t*)&txdataF[0][tti_offset+6])[0] += (int16_t)((((int16_t*)&tmp_sample3)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[0][tti_offset+6])[1] += (int16_t)((((int16_t*)&tmp_sample3)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[1][tti_offset+6])[0] += (int16_t)((((int16_t*)&tmp_sample4)[0]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
					((int16_t*)&txdataF[1][tti_offset+6])[1] += (int16_t)((((int16_t*)&tmp_sample4)[1]*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);

					// fill in the rest of the ALAMOUTI precoding
					if ( pilots != 1 || (re+1)%3 != id_offset) {
						((int16_t *)&txdataF[0][tti_offset+1])[0] += -((int16_t *)&txdataF[1][tti_offset])[0]; //x1
						((int16_t *)&txdataF[0][tti_offset+1])[1] += ((int16_t *)&txdataF[1][tti_offset])[1];
						((int16_t *)&txdataF[1][tti_offset+1])[0] += ((int16_t *)&txdataF[0][tti_offset])[0];  //x0*
						((int16_t *)&txdataF[1][tti_offset+1])[1] += -((int16_t *)&txdataF[0][tti_offset])[1];

						((int16_t *)&txdataF[0][tti_offset+6+1])[0] += -((int16_t *)&txdataF[1][tti_offset+6])[0]; //x1
						((int16_t *)&txdataF[0][tti_offset+6+1])[1] += ((int16_t *)&txdataF[1][tti_offset+6])[1];
						((int16_t *)&txdataF[1][tti_offset+6+1])[0] += ((int16_t *)&txdataF[0][tti_offset+6])[0];  //x0*
						((int16_t *)&txdataF[1][tti_offset+6+1])[1] += -((int16_t *)&txdataF[0][tti_offset+6])[1];
					} else {
						((int16_t *)&txdataF[0][tti_offset+2])[0] += -((int16_t *)&txdataF[1][tti_offset])[0]; //x1
						((int16_t *)&txdataF[0][tti_offset+2])[1] += ((int16_t *)&txdataF[1][tti_offset])[1];
						((int16_t *)&txdataF[1][tti_offset+2])[0] += ((int16_t *)&txdataF[0][tti_offset])[0];  //x0*
						((int16_t *)&txdataF[1][tti_offset+2])[1] += -((int16_t *)&txdataF[0][tti_offset])[1];

						((int16_t *)&txdataF[0][tti_offset+6+2])[0] += -((int16_t *)&txdataF[1][tti_offset+6])[0]; //x1
						((int16_t *)&txdataF[0][tti_offset+6+2])[1] += ((int16_t *)&txdataF[1][tti_offset+6])[1];
						((int16_t *)&txdataF[1][tti_offset+6+2])[0] += ((int16_t *)&txdataF[0][tti_offset+6])[0];  //x0*
						((int16_t *)&txdataF[1][tti_offset+6+2])[1] += -((int16_t *)&txdataF[0][tti_offset+6])[1];

						re++;														// skip pilots
						*re_allocated = *re_allocated + 1;
					}
					re++;  															// adjacent carriers are taken care of by precoding
					*re_allocated = *re_allocated + 1;   							// incremented variable but never used
				}
      		}
    	}
  	}
  	return(0);
}


int dci_modulation_NB_IoT(int32_t 					**txdataF,
						  int16_t 					amp,
						  NB_IoT_DL_FRAME_PARMS 	*frame_parms,
						  uint8_t 					control_region_size,    //XXX we pass the npdcch_start_symbol // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
						  uint8_t 					*e[2],					// Input data
						  int 						G,						// number of bits per subframe
						  uint8_t 					dci_number,				// This variable should takes the 1 or 2 (1 for in case of one DCI, 2 in case of two DCI)
						  uint8_t 					agr_level)				// Aggregation level
{
    uint32_t 		jj=0;
	uint32_t 		re_allocated,symbol_offset;
    uint16_t 		l;
    uint8_t 		id_offset,pilots=0;
	unsigned short  bandwidth_even_odd;
    unsigned short  NB_IoT_start, RB_IoT_ID;

    re_allocated=0;
	id_offset=0;
	// testing if the total number of RBs is even or odd
	bandwidth_even_odd = frame_parms->N_RB_DL % 2; 	 		// 0 even, 1 odd
	RB_IoT_ID = frame_parms->NB_IoT_RB_ID;
	// step  5, 6, 7   									 	// modulation and mapping (slot 1, symbols 0..3)
	for (l=control_region_size; l<14; l++) { 				// loop on OFDM symbols
		if((l>=4 && l<=8) || (l>=11 && l<=13))
		{
			pilots =1;
		} else {
			pilots=0;
		}
		id_offset = frame_parms->Nid_cell % 3;    			// Cell_ID_NB_IoT % 3
		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID% (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID % (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		symbol_offset = frame_parms->ofdm_symbol_size*l + NB_IoT_start;  						// symbol_offset = 512 * L + NB_IOT_RB start
		dci_allocate_REs_in_RB_NB_IoT(frame_parms,
								      txdataF,
							    	  &jj,
									  symbol_offset,
									  e,
									  pilots,
									  amp,
									  id_offset,
									  &re_allocated,
									  dci_number,
									  agr_level);
	}

    // VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_ENB_DLSCH_MODULATION, VCD_FUNCTION_OUT);
  return (re_allocated);
}

//------------------------------------------------
// BCOM code functions npdcch end
//------------------------------------------------




uint8_t generate_dci_top_NB_IoT(NB_IoT_eNB_NPDCCH_t		*npdcch,
						 		uint8_t 				Num_dci,
                         		DCI_ALLOC_NB_IoT_t 		*dci_alloc,
                         		int16_t 				amp,
                         		NB_IoT_DL_FRAME_PARMS 	*fp,
                         		int32_t 				**txdataF,
                         		uint32_t 				subframe,
						 		uint8_t 				npdcch_start_symbol)
{


  int      i, G;
  //temporary variable
  uint16_t rnti[2];
  uint8_t  L = 0;


  /* PARAMETERS may not needed
  **e_ptr : store the encoding result, and as a input to modulation
  *num_pdcch_symbols : to calculate the resource allocation for pdcch
  *L = aggregation level (there is 2 (at most) in NB-IoT) (Note this is not the real value but the index)
  *lprime,kprime,kprime_mod12,mprime,nsymb,symbol_offset,tti_offset,re_offset : used in the REG allocation
  *gain_lin_QPSK,yseq0[Msymb],yseq1[Msymb],*y[2] : used in the modulation
  *mi = used in interleaving
  *e = used to store the taus sequence (taus sequence is used to generate the first sequence for DCI) Turbo coding
  *wbar used in the interleaving and also REG allocation
  */

  // MAC is assumed to have ordered the UE spec DCI according to the RNTI-based randomization???

  // Value of aggregation level (FAPI/NFAPI specs v.9.0 pag 221 value 1,2)
  /*
   * in NB-IoT we can have at most 2 aggregation level since we have only 2 NCCE (Narrowband control channel element)
   * if only 1 DCI transmitted:
   * 	- Aggregation level could be 1 or 2
   * if 2 DCI transmitted:
   * 	- Aggregation level should be 1
   *
   */

  //First take all the DCI pdu and their corrispondent rnti
  for(i = 0; i<Num_dci;i++)
  {
	  npdcch->a[i]=dci_alloc[i].dci_pdu;
	  rnti[i]=dci_alloc[i].rnti;
	  L = dci_alloc[i].L;

  }

  if(Num_dci == 2 && L == 1)
	  LOG_E(PHY,"generate_dci_top_NB_IoT: Aggregation level not compatible with Num_dci\n" );


  //Second, evaluate the G variable based of the npdcch_start_sysmbol

  /*
   * TS 36.213 ch 16.6.1
   * npdcch_start_symbol  indicate the starting OFDM symbol for NPDCCH in the first slot of a subframe k ad is determined as follow:
   * - if eutracontrolregionsize is present (defined for in-band operating mode (mode 0,1 for FAPI specs))
   * 	npdcch_start_symbol = eutracontrolregionsize (value 1,2,3) [units in number of OFDM symbol]
   * -otherwise
   * 	npdcch_start_symbol = 0
   *
   *XXX npdcch_start symbol should be the same for every DCI once is decided since depends on the parameters
   *(the setting of this npdcch_start_symbol parameter should be done in the MAC)
   *Depending on npdcch_start_symbol then we define different values for G
   *
   */


  switch(npdcch_start_symbol) //mail Bcom matthieu
  {
	  	  case 0:
	  		  G = 304;
		 	break;
	  	  case 1:
	  		  G = 240;
	  		  break;
	  	  case 2:
	  		  G = 224;
	  		  break;
	  	  case 3:
	  		  G =200;
	  		  break;
	  	  default:
	  		  LOG_E (PHY,"npdcch_start_symbol has unwanted value\n");
	  		  break;
  }

   //NB-IoT encoding
//  dci_encoding_NB_IoT(
//		  	  	  	  a,
//					  4, // total length (in byte) of a [assume max 2 pdus of  ??]
//					  G,
//					  npdcch->e,
//					  rnti,
//					  Num_dci,
//					  L
//		  	  	  	  );



  //NB-IoT scrambling
//  npdcch_scrambling_NB_IoT(
//		  	  	  	  	  fp,
//						  npdcch->e,
//						  G,
//						  subframe,
//						  Num_dci,
//						  L
//		  	  	  	  	   );


  //NB-IoT Modulation
//  dci_modulation_NB_IoT(
//		  	  	  	  txdataF,
//					  amp,
//					  fp,
//					  npdcch_start_symbol,
//					  npdcch->e,
//					  G,
//					  Num_dci,
//					  L
//		  	  	  	  );



  return 0;
}
