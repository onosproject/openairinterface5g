/***********************************************************************

**********************************************************************/
/*! \file PHY/LTE_TRANSPORT/dlsch_modulation_NB_IoT.c
* \brief Top-level routines for generating the NPDSCH physical channel for NB_IoT,
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/
#include <math.h>
//#include "PHY/defs.h"
//#include "PHY/defs_NB_IoT.h"
//#include "PHY/extern_NB_IoT.h"
//#include "PHY/CODING/defs_nb_iot.h"
//#include "PHY/CODING/extern.h"
//#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "PHY/impl_defs_lte_NB_IoT.h"
#include "PHY/impl_defs_top_NB_IoT.h"
//#include "defs.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"

int allocate_REs_in_RB_NB_IoT(NB_IoT_DL_FRAME_PARMS 	*frame_parms,
                              int32_t 					**txdataF,
                              uint32_t 					*jj,
                              uint32_t 					symbol_offset,
                              uint8_t 					*x0,
                              uint8_t 					pilots,
                              int16_t 					amp,
						  	  unsigned short 			id_offset,
                              uint32_t 					*re_allocated)  //  not used variable ??!!
{
  MIMO_mode_NB_IoT_t 	mimo_mode = (frame_parms->mode1_flag==1)? SISO_NB_IoT:ALAMOUTI_NB_IoT;

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


int dlsch_modulation_NB_IoT(int32_t 				**txdataF,
							int16_t 				amp,
							NB_IoT_DL_FRAME_PARMS 	*frame_parms,
							uint8_t 				control_region_size,      // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
							NB_IoT_eNB_DLSCH_t 		*dlsch0,
							int 					G,						  // number of bits per subframe
							unsigned 				npdsch_data_subframe,     // subframe index of the data table of npdsch channel (G*Nsf)  , values are between 0..Nsf  			
							unsigned short 			NB_IoT_RB_ID)
{
    //uint8_t harq_pid = dlsch0->current_harq_pid;
    //NB_IoT_DL_eNB_HARQ_t *dlsch0_harq = dlsch0->harq_processes[harq_pid];
    uint32_t 		jj = 0;
	uint32_t 		re_allocated,symbol_offset;
    uint16_t 		l;
    uint8_t 		id_offset,pilots = 0; 
	unsigned short 	bandwidth_even_odd;
    unsigned short 	NB_IoT_start, RB_IoT_ID;

    re_allocated = 0;
	id_offset    = 0;
	// testing if the total number of RBs is even or odd 
	bandwidth_even_odd  =  frame_parms->N_RB_DL % 2; 	 	// 0 even, 1 odd
	RB_IoT_ID 			=  NB_IoT_RB_ID;
	// step  5, 6, 7   									 	// modulation and mapping (slot 1, symbols 0..3)
	for (l=control_region_size; l<14; l++) { 								 	// loop on OFDM symbols	
		if((l>=4 && l<=8) || (l>=11 && l<=13))
		{
			pilots = 1;
		} else {
			pilots = 0;
		}
		id_offset = frame_parms->Nid_cell % 3;    			// Cell_ID_NB_IoT % 3
		if(RB_IoT_ID < (frame_parms->N_RB_DL/2))
		{
			NB_IoT_start = frame_parms->ofdm_symbol_size - 12*(frame_parms->N_RB_DL/2) - (bandwidth_even_odd*6) + 12*(RB_IoT_ID % (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		} else {
			NB_IoT_start = (bandwidth_even_odd*6) + 12*(RB_IoT_ID % (int)(ceil(frame_parms->N_RB_DL/(float)2)));
		}
		symbol_offset = frame_parms->ofdm_symbol_size*l + NB_IoT_start;  						// symbol_offset = 512 * L + NB_IOT_RB start

		allocate_REs_in_RB_NB_IoT(frame_parms,
								  txdataF,
								  &jj,
								  symbol_offset,
								  &dlsch0->harq_process.s_e[G*npdsch_data_subframe],
								  pilots,
								  amp,
								  id_offset,
								  &re_allocated);
	}
	
 // VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_ENB_DLSCH_MODULATION, VCD_FUNCTION_OUT);
  return (re_allocated);
}
