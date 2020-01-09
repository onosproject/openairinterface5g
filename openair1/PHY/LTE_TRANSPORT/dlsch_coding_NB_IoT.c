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
#include <string.h>
//#include "PHY/impl_defs_lte.h"
//#include "openair2/COMMON/openair_defs.h"
#include "PHY/defs.h"
//#include "PHY/extern_NB_IoT.h"
#include "PHY/CODING/defs_NB_IoT.h"
//#include "PHY/CODING/extern.h"
//#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "PHY/LTE_TRANSPORT/proto.h"
//#include "SCHED/defs_NB_IoT.h"
//#include "defs_nb_iot.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"
#include "PHY/TOOLS/time_meas_NB_IoT.h" 

unsigned char  ccodelte_table2_NB_IoT[128];
unsigned short glte2_NB_IoT[] = { 0133, 0171, 0165 }; 


void free_eNB_dlsch_NB_IoT(NB_IoT_eNB_NDLSCH_t *dlsch)
{
  
  if (dlsch) {
/*
#ifdef DEBUG_DLSCH_FREE
    printf("Freeing dlsch %p\n",dlsch);
#endif*/


      if (dlsch->harq_process) {


        if (dlsch->harq_process->b) {
          free16(dlsch->harq_process->b,300);
          dlsch->harq_process->b = NULL;

        }

          if (dlsch->harq_process->d) {
            free16(dlsch->harq_process->d,96+(3*(24+MAX_TBS_DL_SIZE_BITS_NB_IoT)));
           // dlsch->harq_process->d = NULL;
          }

	
	     free16(dlsch->harq_process,sizeof(NB_IoT_DL_eNB_HARQ_t));
	     dlsch->harq_process = NULL;
      }
    

    free16(dlsch,sizeof(NB_IoT_eNB_NDLSCH_t));
    dlsch = NULL;
    }

}

void free_eNB_dlcch_NB_IoT(NB_IoT_eNB_NPDCCH_t *dlcch)
{
  
  if (dlcch) {



    free16(dlcch,sizeof(NB_IoT_eNB_NPDCCH_t));
    dlcch = NULL;
    }

}


void ccode_encode_npdsch_NB_IoT (int32_t   numbits,
								 uint8_t   *inPtr,
								 uint8_t   *outPtr,
								 uint32_t  crc)
{
	uint32_t  	state;
	uint8_t  	c, out, first_bit;
	int8_t 		shiftbit = 0;
    /* The input bit is shifted in position 8 of the state.
	Shiftbit will take values between 1 and 8 */
	state 		= 0;
	first_bit   = 2;
	c 			= ((uint8_t*)&crc)[0];
	// Perform Tail-biting
	// get bits from last byte of input (or crc)
	for (shiftbit = 0 ; shiftbit <(8-first_bit) ; shiftbit++) {
		if ((c&(1<<(7-first_bit-shiftbit))) != 0)
			state |= (1<<shiftbit);
	}
	state = state & 0x3f;   			  // true initial state of Tail-biting CCode
	state<<=1;            				  // because of loop structure in CCode
		while (numbits > 0) {											// Tail-biting is applied to input bits , input 34 bits , output 102 bits
			c = *inPtr++;
			for (shiftbit = 7; (shiftbit>=0) && (numbits>0); shiftbit--,numbits--) {
				state >>= 1;
				if ((c&(1<<shiftbit)) != 0) {
					state |= 64;
				}
				out = ccodelte_table2_NB_IoT[state];

				*outPtr++ = out  & 1;
				*outPtr++ = (out>>1)&1;
				*outPtr++ = (out>>2)&1;
			}
		}
}

///////////////////////////////////////////////////////////////////////////////

int dlsch_encoding_NB_IoT(unsigned char      	  *a,
			              NB_IoT_eNB_NDLSCH_t 	  *dlsch,    //NB_IoT_eNB_NDLSCH_t
			              uint8_t 			 	  Nsf,       // number of subframes required for npdsch pdu transmission calculated from Isf (3GPP spec table)
			              unsigned int 		 	  G) 		    // G (number of available RE) is implicitly multiplied by 2 (since only QPSK modulation)
{
       //printf("Get into dlsch_encoding_NB_IoT() ***********************************\n");
	uint32_t  crc = 1;
	//unsigned char harq_pid = dlsch->current_harq_pid;  			// to check during implementation if harq_pid is required in the NB_IoT_eNB_DLSCH_t structure  in defs_NB_IoT.h
	//uint8_t 	  option1,option2,option3,option4;
	unsigned int  A=0;
	A 						= dlsch->harq_process->TBS / 8;

	uint8_t 	  RCC;

    uint8_t       npbch_a[A];
    uint8_t       npbch_a_crc[A+3];
	bzero(npbch_a,A); 
	bzero(npbch_a_crc,A+3);

	dlsch->harq_process->length_e = G*Nsf;									// G*Nsf (number_of_subframes) = total number of bits to transmit G=236

	
	for (int i=0; i<A; i++) 												
	{	
		npbch_a[i] = a[i];    
	}
    
    int32_t numbits = (A*8)+24;

		crc = crc24a_NB_IoT(npbch_a,A*8)>>8;
	

    	for (int j=0; j<A; j++) 												
		{	
			npbch_a_crc[j] = npbch_a[j];    
		}

	    npbch_a_crc[A] = ((uint8_t*)&crc)[2];
	    npbch_a_crc[A+1] = ((uint8_t*)&crc)[1];
		npbch_a_crc[A+2] = ((uint8_t*)&crc)[0];
		
			dlsch->harq_process->B = numbits;			// The length of table b in bits
			//memcpy(dlsch->b,a,numbits/8);        // comment if option 2 
			memset(dlsch->harq_process->d,LTE_NULL_NB_IoT,96);
			ccode_encode_npdsch_NB_IoT(numbits,npbch_a_crc,dlsch->harq_process->d+96,crc);
			RCC = sub_block_interleaving_cc_NB_IoT(numbits,dlsch->harq_process->d+96,dlsch->harq_process->w);		//   step 2 interleaving
			lte_rate_matching_cc_NB_IoT(RCC,dlsch->harq_process->length_e,dlsch->harq_process->w,dlsch->harq_process->e);  // step 3 Rate Matching


  return(0);
}

///////////////////////////////////////////////////////////////////////////
NB_IoT_eNB_NDLSCH_t *new_eNB_dlsch_NB_IoT(uint8_t type, LTE_DL_FRAME_PARMS* frame_parms)
{

  NB_IoT_eNB_NDLSCH_t *dlsch;
  unsigned char exit_flag = 0;

  dlsch = (NB_IoT_eNB_NDLSCH_t *)malloc16(sizeof(NB_IoT_eNB_NDLSCH_t));

  if (dlsch) {

       bzero(dlsch,sizeof(NB_IoT_eNB_NDLSCH_t));

       dlsch->harq_process = (NB_IoT_DL_eNB_HARQ_t *)malloc16(sizeof(NB_IoT_DL_eNB_HARQ_t));

	  if (dlsch->harq_process) {
		    bzero(dlsch->harq_process,sizeof(NB_IoT_DL_eNB_HARQ_t));
		    //    dlsch->harq_processes[i]->first_tx=1;
		    dlsch->harq_process->b = (unsigned char*)malloc(300);   // to set a new one that replace 300 , MAX_DLSCH_PAYLOAD_BYTES/bw_scaling

		    if (dlsch->harq_process->b) {
		      bzero(dlsch->harq_process->b,300);
		    } else {
		      printf("Can't get b\n");
		      exit_flag=1;
		    }



		        if (dlsch->harq_process->d) {
		          bzero((void *)dlsch->harq_process->d,96+(3*(24+MAX_TBS_DL_SIZE_BITS_NB_IoT)));

		        } else {
		          printf("Can't get d\n");
		          exit_flag=2;
		        }
		      
		  //  }

	  } else {
	   
	    exit_flag=3;
	  }
    

	    if (exit_flag==0) {
	      
	        dlsch->harq_process->round=0;

			//  for (r=0; r<(96+(3*(24+MAX_TBS_DL_SIZE_BITS_NB_IoT))); r++) {
			//  
			 //   if (dlsch->harq_process->d)
			 //     dlsch->harq_process->d[0]= LTE_NULL_NB_IoT;
			 // }

	      return(dlsch);
	    }
  }

 /// LOG_D(PHY,"new_eNB_dlsch exit flag %d, size of  %ld\n",
//	exit_flag, sizeof(NB_IoT_eNB_NDLSCH_t));
  free_eNB_dlsch_NB_IoT(dlsch);
  return(NULL);


}


///////////////////////////////////////////////////////////////////////////
NB_IoT_eNB_NPDCCH_t *new_eNB_dlcch_NB_IoT(LTE_DL_FRAME_PARMS* frame_parms)
{

  NB_IoT_eNB_NPDCCH_t *dlcch;

  dlcch = (NB_IoT_eNB_NPDCCH_t *)malloc16(sizeof(NB_IoT_eNB_NPDCCH_t));

  if (dlcch) {

       bzero(dlcch,sizeof(NB_IoT_eNB_NPDCCH_t));



	    return(dlcch);
	    
  }

 /// LOG_D(PHY,"new_eNB_dlsch exit flag %d, size of  %ld\n",
//	exit_flag, sizeof(NB_IoT_eNB_NDLSCH_t));
  free_eNB_dlcch_NB_IoT(dlcch);
  return(NULL);


}

/*************************************************************************

  Functions to initialize the code tables

*************************************************************************/
/* Basic code table initialization for constraint length 7 */

/* Input in MSB, followed by state in 6 LSBs */
void ccodelte_init2_NB_IoT(void)
{
  unsigned int  i, j, k, sum;

  for (i = 0; i < 128; i++) {

    ccodelte_table2_NB_IoT[i] = 0;

    /* Compute 3 output bits */
    for (j = 0; j < 3; j++) {
      sum = 0;

      for (k = 0; k < 7; k++)
        if ((i & glte2_NB_IoT[j]) & (1 << k))
          sum++;

      /* Write the sum modulo 2 in bit j */
      ccodelte_table2_NB_IoT[i] |= (sum & 1) << j;
    }
  }
}

