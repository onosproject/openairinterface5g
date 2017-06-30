/*******************************************************************************

*******************************************************************************/
/*! \file PHY/LTE_TRANSPORT/dlsch_coding_NB_IoT.c
* \brief Top-level routines for implementing Tail-biting convolutional coding for transport channels (NPDSCH) for NB_IoT,	 TS 36-212, V13.4.0 2017-02
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/

#include "PHY/defs.h"
#include "PHY/extern.h"
#include "PHY/CODING/defs.h"
#include "PHY/CODING/extern.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/LTE_TRANSPORT/defs.h"
#include "PHY/LTE_TRANSPORT/proto.h"
#include "SCHED/defs.h"
#include "defs.h"
#include "UTIL/LOG/vcd_signal_dumper.h"

#define is_not_pilot(pilots,first_pilot,re) (1)

#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h" // newly added for NB_IoT

void ccode_encode_npdsch_NB_IoT (int32_t numbits,
								 uint8_t *inPtr,
								 uint8_t *outPtr,
								 uint32_t crc)
{
	uint32_t  state;
	uint8_t  c, out, first_bit;
	int8_t shiftbit=0;
  /* The input bit is shifted in position 8 of the state.
	Shiftbit will take values between 1 and 8 */
	state = 0;
	first_bit = 2;
	c = ((uint8_t*)&crc)[0];
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
}


int dlsch_encoding_NB_IoT(unsigned char *a,
						  NB_IoT_eNB_DLSCH_t *dlsch,
						  uint8_t Nsf,					// number of subframes required for npdsch pdu transmission calculated from Isf (3GPP spec table)
						  unsigned int G; 				// G (number of available RE) is implicitly multiplied by 2 (since only QPSK modulation)
						  time_stats_t *rm_stats,
						  time_stats_t *te_stats,
						  time_stats_t *i_stats)
{
	unsigned int crc=1;
	//unsigned char harq_pid = dlsch->current_harq_pid;  			// to check during implementation if harq_pid is required in the NB_IoT_eNB_DLSCH_t structure  in defs_NB_IoT.h
	unsigned int A;
	uint8_t RCC;
	A = dlsch->harq_processe->TBS;  				// 680
	dlsch->harq_processe->length_e = G*Nsf			// G*Nsf (number_of_subframes) = total number of bits to transmit 
	int32_t numbits = A+24;
	
	if (dlsch->harq_processe->round == 0) { 	    // This is a new packet

		crc = crc24a(a,A)>>8;						// CRC calculation (24 bits CRC)
												// CRC attachment to payload
		a[A>>3] = ((uint8_t*)&crc)[2];
		a[1+(A>>3)] = ((uint8_t*)&crc)[1];
		a[2+(A>>3)] = ((uint8_t*)&crc)[0];
		
		dlsch->harq_processe->B = numbits;			// The length of table b in bits
		
		memcpy(dlsch->harq_processe->b,a,numbits/8); 
		memset(dlsch->harq_processe->d,LTE_NULL,96);
		
		start_meas(te_stats);
		ccode_encode_npdsch_NB_IoT(numbits, dlsch->harq_processe->b, dlsch->harq_processe->d+96, crc);  					//   step 1 Tail-biting convolutional coding
		stop_meas(te_stats);
		
		start_meas(i_stats);
		RCC = sub_block_interleaving_cc_NB_IoT(numbits,dlsch->harq_processe->d+96,dlsch->harq_processe->w);					//   step 2 interleaving
		stop_meas(i_stats);
		
		start_meas(rm_stats);
		lte_rate_matching_cc_NB_IoT(RCC,dlsch->harq_processe->length_e,dlsch->harq_processe->w,dlsch->harq_processe->e);    // step 3 Rate Matching
		stop_meas(rm_stats);		
    }
  return(0);
}
