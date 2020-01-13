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

/*! \file PHY/LTE_TRANSPORT/dci_NB_IoT.c
* \brief Implements PDCCH physical channel TX/RX procedures (36.211) and DCI encoding/decoding (36.212/36.213). Current LTE compliance V8.6 2009-03.
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom, B-COM
* \email: knopp@eurecom.fr
* \note
* \warning
*/

#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "PHY/defs_NB_IoT.h"  // /LTE_TRANSPORT/defs_NB_IoT.h


//static uint8_t d[2][3*(MAX_DCI_SIZE_BITS_NB_IoT + 16) + 96];
//static uint8_t w[2][3*3*(MAX_DCI_SIZE_BITS_NB_IoT+16)];

void dci_encoding_NB_IoT(uint8_t                  *a,				         // Array of two DCI pdus, even if one DCI is to transmit , the number of DCI is indicated in dci_number
						 NB_IoT_eNB_NPDCCH_t      *dlcch,                  ////uint8_t    *e[2],				// *e should be e[2][G]
						 uint8_t                  A,
						 uint16_t                 G,
						 uint8_t 			      ncce_index,
						 uint8_t                  agr_level)			         // Aggregation level
{
	uint8_t  D = (A + 16);
	uint32_t RCC;
	uint8_t  occupation_size=1;
	
		if(agr_level == 2 && ncce_index == 0)
		{
			occupation_size=1;
		}else{
			occupation_size=2;
		}
		memset((void *)dlcch->npdcch_d[ncce_index],LTE_NULL_NB_IoT,96);

		ccode_encode_NB_IoT(A,2,a,dlcch->npdcch_d[ncce_index]+96,dlcch->rnti[ncce_index]);    					// CRC attachement & Tail-biting convolutional coding
		RCC = sub_block_interleaving_cc_NB_IoT(D,dlcch->npdcch_d[ncce_index]+96,dlcch->npdcch_w[ncce_index]);				// Interleaving
		lte_rate_matching_cc_NB_IoT(RCC,(G/occupation_size),dlcch->npdcch_w[ncce_index],dlcch->npdcch_e[ncce_index]);		// Rate Matching

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///The scrambling sequence shall be initialised at the start of the search space and after every 4th NPDCCH subframes.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void npdcch_scrambling_NB_IoT(LTE_DL_FRAME_PARMS     *frame_parms,
							  NB_IoT_eNB_NPDCCH_t     *dlcch,			// Input data
							  int                     G,        	// Total number of bits to transmit in one subframe(case of DCI = G)
							  uint8_t 				  Ns,				//XXX we pass the subframe	// Slot number (0..19)
							  uint8_t 			      ncce_index,
							  uint8_t 				  agr_level)		// Aggregation level
{
	int 	  i;
	uint32_t  x1, x2, s=0;
	uint8_t   reset;
	uint8_t   occupation_size=1;

	reset = 1;

	if(agr_level == 2 && ncce_index == 0)
	{
		occupation_size=1;
	}else{
		occupation_size=2;
	}

		x2 = ((Ns>>1)<<9) + frame_parms->Nid_cell;                          // This is c_init in 36.211 Sec 10.2.3.1

		for (i=0; i<G/occupation_size; i++) {
			if ((i&0x1f)==0) {
				s = lte_gold_generic_NB_IoT(&x1, &x2, reset);
				reset = 0;
			}
			dlcch->npdcch_e[ncce_index][i] = (dlcch->npdcch_e[ncce_index][i]&1) ^ ((s>>(i&0x1f))&1);
		}
	
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int dci_allocate_REs_in_RB_NB_IoT(LTE_DL_FRAME_PARMS 	*frame_parms,
                                  int32_t 					**txdataF,
                                  uint32_t 					*jj,
                                  uint32_t 					symbol_offset,
                                  uint8_t 					*x0,
                                  uint8_t 					pilots,
                                  uint8_t                   pilot_shift,
                                  int16_t 					amp,
						  	      unsigned short 			id_offset,
								  uint8_t 					ncce_index,
								  uint8_t 					agr_level,
								  uint32_t 					*re_allocated)
{
  MIMO_mode_t 	mimo_mode = (frame_parms->mode1_flag==1)? SISO:ALAMOUTI;

  uint32_t  tti_offset,aa;
  uint8_t   re;
  int16_t   gain_lin_QPSK;
  uint8_t   first_re,last_re;
  int32_t   tmp_sample1,tmp_sample2;

  gain_lin_QPSK = (int16_t)((amp*ONE_OVER_SQRT2_Q15_NB_IoT)>>15);
  first_re      = 0;
  last_re       = 12;


   if(ncce_index == 0 && agr_level == 2)
   {

		  for (re=first_re; re<last_re; re++) {      		// re varies between 0 and 12 sub-carriers

		    tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re
			
			if (pilots != 1 || (re%6 != ((id_offset + 3*pilot_shift) % 6) ) )  			// if re is not a pilot
			{
															 
		      if (mimo_mode == SISO) {  								//SISO mapping
					*re_allocated = *re_allocated + 1;						// variable incremented but never used
					
					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
							((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
					}
					*jj = *jj + 1;
					for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
							((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
					}
					*jj = *jj + 1;	
					
		      } else if (mimo_mode == ALAMOUTI) {
			  
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

   } else if(agr_level == 1) {

	   		for (re=(first_re + ncce_index*6); re<(6 + ncce_index*6); re++) {      		// re varies between 0 and 6 or 6 and 12 sub-carriers

			    tti_offset = symbol_offset + re;				// symbol_offset = 512 * L ,  re_offset = 512 - 3*12  , re
				
				if (pilots != 1 || (re%6 != ((id_offset + 3*pilot_shift) % 6) ) )  			// if re is not a pilot
				{
																 
			      if (mimo_mode == SISO) {  								//SISO mapping
						*re_allocated = *re_allocated + 1;						// variable incremented but never used
						
						for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
								((int16_t*)&txdataF[aa][tti_offset])[0] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //I //b_i
						}
						*jj = *jj + 1;
						for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
								((int16_t*)&txdataF[aa][tti_offset])[1] += (x0[*jj]==1) ? (-gain_lin_QPSK) : gain_lin_QPSK; //Q //b_{i+1}
						}
						*jj = *jj + 1;	
						
			      } else if (mimo_mode == ALAMOUTI) {
				  
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

   } else {

   		printf("Error in allocate RE of DCI NB_IoT");

   }

  return(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int dci_modulation_NB_IoT(int32_t 				  **txdataF,
						  int16_t 				  amp,
						  LTE_DL_FRAME_PARMS 	  *frame_parms,
						  uint8_t 				  control_region_size,      // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
						  NB_IoT_eNB_NPDCCH_t    *dlcch,
						  unsigned int		      npdsch_data_subframe,     // subframe index of the data table of npdsch channel (G*Nsf)  , values are between 0..Nsf  			
						  uint8_t 				  agr_level,
						  uint8_t 				  ncce_index,
						  unsigned int		      subframe,
						  unsigned short 		  NB_IoT_RB_ID)            /// NB_IoT_RB_ID should be initialized in the LTE_DL_FRAME_PARMS
{
    
    uint32_t 		jj = 0;
	uint32_t 		re_allocated,symbol_offset;
    uint16_t 		l;
    uint8_t 		id_offset,pilot_shift,pilots = 0; 
	unsigned short 	bandwidth_even_odd;
    unsigned short 	NB_IoT_start, RB_IoT_ID;

    re_allocated = 0;
	id_offset    = 0;
	pilot_shift  = 0;
	// testing if the total number of RBs is even or odd 
	bandwidth_even_odd  =  frame_parms->N_RB_DL % 2; 	 	// 0 even, 1 odd
	RB_IoT_ID 			=  NB_IoT_RB_ID;
	// step  5, 6, 7   									 	// modulation and mapping (slot 1, symbols 0..3)
	for (l=control_region_size; l<14; l++) {                  // loop on OFDM symbols	

		if((l>=4 && l<=7) || (l>=11 && l<=13))
		{
			pilots = 1;
			if(l==4 || l==6 || l==11 || l==13)
			{
				pilot_shift  = 1;
			}
		} else {
			pilots = 0;
		}
		id_offset = frame_parms->Nid_cell % 6;    			// Cell_ID_NB_IoT % 6
		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID % (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = 1 + (bandwidth_even_odd*6) + 12*(RB_IoT_ID % (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		symbol_offset = (14*subframe*frame_parms->ofdm_symbol_size) + frame_parms->ofdm_symbol_size*l + NB_IoT_start;  						// symbol_offset = 512 * L + NB_IOT_RB start


		if(agr_level == 2)
		{
				dci_allocate_REs_in_RB_NB_IoT(frame_parms,
											  txdataF,
											  &jj,
											  symbol_offset,
											  &dlcch->npdcch_e[0],
											  pilots,
											  pilot_shift,
											  amp,
											  id_offset,
											  ncce_index,
								              agr_level,
											  &re_allocated);
	    } else {
	    		dci_allocate_REs_in_RB_NB_IoT(frame_parms,
											  txdataF,
											  &jj,
											  symbol_offset,
											  &dlcch->npdcch_e[ncce_index],
											  pilots,
											  pilot_shift,
											  amp,
											  id_offset,
											  ncce_index,
								              agr_level,
											  &re_allocated);
	    }

	}

  return (re_allocated);
}
