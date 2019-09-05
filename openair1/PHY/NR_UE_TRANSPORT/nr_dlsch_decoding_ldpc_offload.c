/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*! \file PHY/LTE_TRANSPORT/dlsch_decoding.c
* \brief Top-level routines for decoding  LDPC-coded (DLSCH) transport channels from 38-212
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/

#include "common/utils/LOG/vcd_signal_dumper.h"

//#include "defs.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "PHY/CODING/coding_extern.h"
#include "SCHED_NR_UE/defs.h"
#include "SIMULATION/TOOLS/sim.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_decoder.h"
//#include "PHY/CODING/nrLDPC_types.h"
//#define DEBUG_DLSCH_DECODING

#if LDPC_FPGA_OFFLOAD
static uint64_t nb_total_decod	= 0;
static uint64_t	nb_error_decod	= 0;

uint64_t	total_decod = 0;
uint64_t	total_decod_error = 0;
uint64_t	total_decod_error_crc = 0;
uint64_t	total_decod_it = 0;

typedef struct ldpc_stat_per_subframe{
	uint64_t  	ldpc_block_decod_count;
	uint64_t 	ldpc_block_decod_error;
	uint64_t 	ldpc_block_decod_error_crc;
	uint64_t 	ldpc_block_decod_it;
	uint64_t 	ldpc_block_decod_max_it;
	uint64_t 	ldpc_tb_error_block_max;
} ldpc_stat_per_subframe_t;


int 	ldpc_stat_flag = 0;
ldpc_stat_per_subframe_t	g_ldpc_stat_per_subframe_array[10];

//#define DLSIM_DECOD_CHECK
#ifdef DLSIM_DECOD_CHECK
	extern uint16_t dlsch_coding_c[16][22*384];
	extern uint8_t 	dlsch_coding_d[16][96+(22*384*3)+12];
	extern uint16_t dlsch_decoding_d[16][96+(22*384*3)+12];

	void debug_check_decoder_input_buff( uint32_t Kr, uint32_t Zc,  uint32_t r , LTE_DL_UE_HARQ_t *harq_process );
#endif


#if LDPC_FPGA_OFFLOAD
	#include <dlfcn.h>
	#include "ldpc_decoder_offload.h"

	
	typedef session_t (*threegpp_nr_ldpc_decode_start_t)(session_desc_t *session_desc); 
	extern threegpp_nr_ldpc_decode_start_t threegpp_nr_ldpc_decode_start_funct;

	typedef int32_t (*threegpp_nr_ldpc_decode_putq_t)(	session_t 	fd, 
														int16_t 	*y_16bits,
														uint8_t 	*decoded_bytes,
														uint8_t		r,
														uint16_t 	n,
														uint8_t		max_iterations,
														uint8_t		crc_type);
	extern threegpp_nr_ldpc_decode_putq_t threegpp_nr_ldpc_decode_putq_funct;


	typedef int32_t (*threegpp_nr_ldpc_decode_getq_t)(	session_t 	fd, 
														uint8_t 	*decoded_bytes,
														uint8_t		r,
														uint8_t		crc_type,
														uint8_t		*crc_status,
														uint8_t		*nb_iterations
													); 
	extern threegpp_nr_ldpc_decode_getq_t threegpp_nr_ldpc_decode_getq_funct;

	typedef int32_t (*threegpp_nr_ldpc_decode_stop_t)(	session_t 	fd);
	extern threegpp_nr_ldpc_decode_stop_t threegpp_nr_ldpc_decode_stop_funct;

	typedef int32_t (*threegpp_nr_ldpc_decode_run_t)(	session_t 	fd);
	extern threegpp_nr_ldpc_decode_run_t threegpp_nr_ldpc_decode_run_funct;



#endif


//#define DEBUG_LLR_STATS
#ifdef DEBUG_LLR_STATS

	static int16_t		llr_max	= 0;
	static int16_t		llr_min	= 0;
	static uint16_t		llr_cnt	= 0;
	static double		llr_p_mean	= 0.0;
	static double		llr_p_cnt	= 0.0;
	static double		llr_n_mean	= 0.0;
	static double		llr_n_cnt	= 0.0;

	static double		llr_p_mean1	= 0.0;
	static double		llr_p_cnt1	= 0.0;
	static double		llr_n_mean1	= 0.0;
	static double		llr_n_cnt1	= 0.0;

	static void debug_llr_stats(void);

#endif

extern double cpuf;




// static 	uint32_t dbg_counter = 0;
// static 	uint8_t  current_sf = 0;
uint32_t  nr_dlsch_decoding_ldpc_offload(PHY_VARS_NR_UE *phy_vars_ue,
		                         short *dlsch_llr,
		                         NR_DL_FRAME_PARMS *frame_parms,
		                         NR_UE_DLSCH_t *dlsch,
		                         NR_DL_UE_HARQ_t *harq_process,
		                         uint32_t frame,
		                         uint16_t nb_symb_sch,
		                         uint8_t nr_tti_rx,
		                         uint8_t harq_pid,
		                         uint8_t is_crnti,
		                         uint8_t llr8_flag)
{

#if UE_TIMING_TRACE
	time_stats_t *dlsch_rate_unmatching_stats=&phy_vars_ue->dlsch_rate_unmatching_stats;
	time_stats_t *dlsch_turbo_decoding_stats=&phy_vars_ue->dlsch_turbo_decoding_stats;
	time_stats_t *dlsch_deinterleaving_stats=&phy_vars_ue->dlsch_deinterleaving_stats;
#endif

	uint8_t no_iteration_ldpc;
	uint8_t crc_status_ldpc;

	
	uint32_t A,E;
	uint32_t G;
	uint32_t ret,offset;

	short dummy_w[MAX_NUM_DLSCH_SEGMENTS][3*(8448+64)];
	uint32_t r,r_offset=0,Kr,Kr_bytes,err_flag=0, K_bytes_F;
	uint8_t crc_type;

	t_nrLDPC_dec_params decParams;
	t_nrLDPC_dec_params* p_decParams = &decParams;

	uint8_t kb;
	uint8_t kc;

	uint8_t Ilbrm = 0;
	uint32_t Tbslbrm = 950984;
	uint16_t nb_rb = 30;
	double Coderate = 0.0;
	//nfapi_nr_config_request_t *cfg = &phy_vars_ue->nrUE_config;
	//uint8_t dmrs_type = cfg->pdsch_config.dmrs_type.value;
	uint8_t nb_re_dmrs = 6; //(dmrs_type==NFAPI_NR_DMRS_TYPE1)?6:4;
	uint16_t length_dmrs = 1; //cfg->pdsch_config.dmrs_max_length.value;

	uint8_t iteration_max;

	session_t 		ldpc_decoder_fd;
	session_desc_t	ldpc_decod_session_param;
	uint32_t		lpdc_decod_return;




#ifndef __AVX2__
	AssertFatal (0, "dlsch_decoding_ldpc_offload - platform not supported (no AVX2)\n");
#endif


	if (!dlsch_llr) {
		printf("dlsch_decoding.c: NULL dlsch_llr pointer\n");
		return(dlsch->max_ldpc_iterations);
	}

	if (!harq_process) {
		printf("dlsch_decoding.c: NULL harq_process pointer\n");
		return(dlsch->max_ldpc_iterations);
	}

	if (!frame_parms) {
		printf("dlsch_decoding.c: NULL frame_parms pointer\n");
		return(dlsch->max_ldpc_iterations);
	}

	/*if (nr_tti_rx> (10*frame_parms->ttis_per_subframe-1)) {
		printf("dlsch_decoding.c: Illegal subframe index %d\n",nr_tti_rx);
		return(dlsch->max_ldpc_iterations);
	}

	if (dlsch->harq_ack[nr_tti_rx].ack != 2) {
		LOG_D(PHY, "[UE %d] DLSCH @ SF%d : ACK bit is %d instead of DTX even before PDSCH is decoded!\n",
		phy_vars_ue->Mod_id, nr_tti_rx, dlsch->harq_ack[nr_tti_rx].ack);
	}

	if (llr8_flag != 0) {
		AssertFatal (harq_process->TBS >= 256 , "Mismatch flag nbRB=%d TBS=%d mcs=%d Qm=%d RIV=%d round=%d \n",
					harq_process->nb_rb, harq_process->TBS,harq_process->mcs,harq_process->Qm,harq_process->rvidx,harq_process->round);
	}*/



	err_flag		= 0;
	iteration_max 	= 0;
	r_offset = 0;
	nb_rb = harq_process->nb_rb;
	harq_process->trials[harq_process->round]++;

	harq_process->TBS = nr_compute_tbs(harq_process->mcs,nb_rb,nb_symb_sch,nb_re_dmrs,length_dmrs, harq_process->Nl);
	A = harq_process->TBS;
	ret = dlsch->max_ldpc_iterations;

	harq_process->G = nr_get_G(nb_rb, nb_symb_sch, nb_re_dmrs, length_dmrs, harq_process->Qm,harq_process->Nl);
	G = harq_process->G;

	#ifdef DEBUG_DLSCH_DECODING
  	printf("\nDLSCH Decoding, harq_pid %d DCINdi %d round %d\n",harq_pid,harq_process->DCINdi, harq_process->round );
	#endif

	if (harq_process->round == 0) {

		// printf("dlsch_decoding_ldpc_offload : Segmentation in (proc %d ) %d/%d \n",
		// 			phy_vars_ue->current_thread_id[nr_tti_rx],
		// 			frame,
		// 			nr_tti_rx);
		// fflush(stdout);

		// This is a new packet, so compute quantities regarding segmentation
		harq_process->B = A+24;
	    nr_segmentation(NULL,
	                    NULL,
	                    harq_process->B,
	                    &harq_process->C,
	                    &harq_process->K,
	                    &harq_process->Z, // [hna] Z is Zc
	                    &harq_process->F);
	}

	p_decParams->Z = harq_process->Z;

	Coderate = (float) A /(float) G;

	if ((A <=292) || ((A<=3824) && (Coderate <= 0.6667)) || Coderate <= 0.25)
	{
	    p_decParams->BG = 2;
	    if (Coderate < 0.3333){
	      p_decParams->R = 15;
	      kc = 52;
	    }
	    else if (Coderate <0.6667){
	      p_decParams->R = 13;
	      kc = 32;
	    }
	    else {
	      p_decParams->R = 23;
	      kc = 17;
	    }
	}
	else{
	    p_decParams->BG = 1;
	    if (Coderate < 0.6667){
	      p_decParams->R = 13;
	      kc = 68;
	    }
	    else if (Coderate <0.8889){
	      p_decParams->R = 23;
	      kc = 35;
	    }
	    else {
	      p_decParams->R = 89;
	      kc = 27;
	}
	  }

	// Select CRC type
	if (harq_process->C == 1)
		crc_type = CRC24_A;
	else
		crc_type = CRC24_B;

		p_decParams->numMaxIter = dlsch->max_ldpc_iterations;
		p_decParams->outMode= 0;

		#ifdef DEBUG_DLSCH_DECODING
		printf("p_decParams Kplus=%d Z=%d BG=%d, R=%d, MaxIt.=%d \n",harq_process->Kplus,harq_process->Z,p_decParams->BG,p_decParams->R,p_decParams->numMaxIter);
		#endif		        

		ldpc_decod_session_param.proc_nb 				= phy_vars_ue->current_thread_id[nr_tti_rx];
		ldpc_decod_session_param.frame 					= frame;
		ldpc_decod_session_param.nr_tti_rx				= nr_tti_rx;

		ldpc_decod_session_param.BG						= p_decParams->BG;
		ldpc_decod_session_param.R 						= p_decParams->R;
		ldpc_decod_session_param.Zc 					= harq_process->Z;
		ldpc_decod_session_param.max_decoding_iterations= p_decParams->numMaxIter;
		ldpc_decod_session_param.C						= harq_process->C;
		ldpc_decod_session_param.crc_type				= crc_type;
		ldpc_decod_session_param.mcs 					= harq_process->mcs;

		ldpc_decoder_fd = threegpp_nr_ldpc_decode_start_funct(&ldpc_decod_session_param);
		if( ldpc_decoder_fd == NULL )
		{


			printf("dlsch_decoding: Error getting FPGA decod session!!!\n");

			harq_process->harq_ack.ack = 0;
		    harq_process->harq_ack.harq_id = harq_pid;
		    harq_process->harq_ack.send_harq_status = 1;
		    harq_process->errors[harq_process->round]++;
		    harq_process->round++;

			// If we have reach harq max round or no mac (phytest mode) -> set harq process in idle mode
			if ((harq_process->round >= dlsch->Mdlharq) || (phy_vars_ue->mac_enabled==0) )
			{
				harq_process->status = SCH_IDLE;
				harq_process->round  = 0;
			}
			if(is_crnti)
			{
				LOG_D(PHY,"[UE %d] DLSCH: Setting NACK for nr_tti_rx %d (pid %d, pid status %d, round %d/Max %d, TBS %d)\n",
							phy_vars_ue->Mod_id,
							nr_tti_rx,
							harq_pid, harq_process->status, harq_process->round, dlsch->Mdlharq, harq_process->TBS);
			}

			return((1+dlsch->max_ldpc_iterations));
		}

	unsigned char bw_scaling =1;
	switch (frame_parms->N_RB_DL) {
    case 106:
      bw_scaling =2;
      break;

    default:
      bw_scaling =1;
      break;
	}

	if (harq_process->C > MAX_NUM_DLSCH_SEGMENTS/bw_scaling) {
		printf("Illegal harq_process->C %d > %d\n",harq_process->C,MAX_NUM_DLSCH_SEGMENTS/bw_scaling);
		return( (1+dlsch->max_ldpc_iterations) );
	}


	#ifdef DEBUG_DLSCH_DECODING
	  printf("Segmentation: C %d, K %d\n",harq_process->C,harq_process->K);
	#endif



	/* ---------------------------------------------------------------------- */
	/* ---------------------------------------------------------------------- */
	/*  LOOP OVER SEGMENTS                                                    */
	/* ---------------------------------------------------------------------- */
	/* ---------------------------------------------------------------------- */


	#if UE_TIMING_TRACE
	opp_enabled=1;		// enable openair performance monitor
	#endif

	Kr = harq_process->K;
	Kr_bytes = Kr>>3;

	K_bytes_F = Kr_bytes-(harq_process->F>>3);
	Tbslbrm = nr_compute_tbs(28,nb_rb,frame_parms->symbols_per_slot,0,0, harq_process->Nl);

	for (r=0; r<harq_process->C; r++) {

	    E = nr_get_E(G, harq_process->C, harq_process->Qm, harq_process->Nl, r);

	#if UE_TIMING_TRACE
	    start_meas(dlsch_deinterleaving_stats);
	#endif
	    nr_deinterleaving_ldpc(E,
	                           harq_process->Qm,
	                           harq_process->w[r], // [hna] w is e
	                           dlsch_llr+r_offset);

	    //for (int i =0; i<16; i++)
	    //          printf("rx output deinterleaving w[%d]= %d r_offset %d\n", i,harq_process->w[r][i], r_offset);

	#if UE_TIMING_TRACE
	    stop_meas(dlsch_deinterleaving_stats);
	#endif

	#if UE_TIMING_TRACE
	    start_meas(dlsch_rate_unmatching_stats);
	#endif

	#ifdef DEBUG_DLSCH_DECODING
	    LOG_D(PHY,"HARQ_PID %d Rate Matching Segment %d (coded bits %d,unpunctured/repeated bits %d, TBS %d, mod_order %d, nb_rb %d, Nl %d, rv %d, round %d)...\n",
	          harq_pid,r, G,
	          Kr*3,
	          harq_process->TBS,
	          harq_process->Qm,
	          harq_process->nb_rb,
	          harq_process->Nl,
	          harq_process->rvidx,
	          harq_process->round);
	#endif

	    if (nr_rate_matching_ldpc_rx(Ilbrm,
	                                 Tbslbrm,
	                                 p_decParams->BG,
	                                 p_decParams->Z,
	                                 harq_process->d[r],
	                                 harq_process->w[r],
	                                 harq_process->C,
	                                 harq_process->rvidx,
	                                 (harq_process->round==0)?1:0,
	                                 E)==-1) {
	#if UE_TIMING_TRACE
	      stop_meas(dlsch_rate_unmatching_stats);
	#endif
	      LOG_E(PHY,"dlsch_decoding.c: Problem in rate_matching\n");
	      return(dlsch->max_ldpc_iterations);
	    } else {
	#if UE_TIMING_TRACE
	      stop_meas(dlsch_rate_unmatching_stats);
	#endif
	    }

	    //for (int i =0; i<16; i++)
	    //      printf("rx output ratematching d[%d]= %d r_offset %d\n", i,harq_process->d[r][i], r_offset);

	    r_offset += E;

	#ifdef DEBUG_DLSCH_DECODING
	    if (r==0) {
	      write_output("decoder_llr.m","decllr",dlsch_llr,G,1,0);
	      write_output("decoder_in.m","dec",&harq_process->d[0][0],(3*8*Kr_bytes)+12,1,0);
	    }

	    printf("decoder input(segment %d) :",r);
	    int i;
	    for (i=0;i<(3*8*Kr_bytes)+12;i++)
	      printf("%d : %d\n",i,harq_process->d[r][i]);
	    printf("\n");
	#endif

	    //    printf("Clearing c, %p\n",harq_process->c[r]);
	    memset(harq_process->c[r],0,Kr_bytes);

		/*  CLEARING OUTPUT BUFFER                                                */
		/* ---------------------------------------------------------------------- */
		memset(harq_process->c[r],0,Kr_bytes);

		/* Compute LLRs 16 statistics -> usefull for 8bits conversion dynamic */
		#ifdef DEBUG_LLR_STATS
			debug_llr_stats();
		#endif


		/*  DECODING PROCESS                                                      */
		/* ---------------------------------------------------------------------- */
    	// LDPC Decoder

			#ifdef DLSIM_DECOD_CHECK
				debug_check_decoder_input_buff( Kr, harq_process->Z , r , harq_process );
			#endif

			lpdc_decod_return = threegpp_nr_ldpc_decode_putq_funct(	ldpc_decoder_fd,
														&harq_process->d[r],
														harq_process->c[r], 
														r,
														Kr, 
														dlsch->max_ldpc_iterations,
														crc_type
													); 
			if (lpdc_decod_return){
				printf("dlsch_decoding_ldpc_offload putq FAILED %s() %s:%d\n",
					__FUNCTION__,
					__FILE__,
					__LINE__);


			lpdc_decod_return = threegpp_nr_ldpc_decode_stop_funct(ldpc_decoder_fd);
			return((1+dlsch->max_ldpc_iterations));

			}
	}	// r Segments LOOP

#if 1
	lpdc_decod_return = threegpp_nr_ldpc_decode_run_funct(ldpc_decoder_fd);
	if (lpdc_decod_return)
	{
		printf("dlsch_decoding_ldpc_offload run FAILED %s() %s:%d !!!!!!!!!!\n",
			__FUNCTION__,
			__FILE__,
			__LINE__);

		lpdc_decod_return = threegpp_nr_ldpc_decode_stop_funct(ldpc_decoder_fd);
		return((1+dlsch->max_ldpc_iterations));

	}


	// Wait for decoding offload ending
	for (r=0; r<harq_process->C; r++)
	{
		nb_total_decod++;
		total_decod ++;
		g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_count ++;

		// If no errors try to get next segment
//		if ( ! err_flag )
//		{

			//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_UE_LDPC_DECOD_GETQ0+(phy_vars_ue->current_thread_id[nr_tti_rx]), 1);

			lpdc_decod_return = threegpp_nr_ldpc_decode_getq_funct(	ldpc_decoder_fd,
													harq_process->c[r], 
													r,
													crc_type,
													&crc_status_ldpc,
													&no_iteration_ldpc
												); 

			//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_UE_LDPC_DECOD_GETQ0+(phy_vars_ue->current_thread_id[nr_tti_rx]), 0);


			total_decod_it += no_iteration_ldpc;
			g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_it += no_iteration_ldpc;

			if ( lpdc_decod_return != LDPC_DECODER_OFFLOAD_OK )
			{
				nb_error_decod++;
				total_decod_error++;
				g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_error++;
				err_flag++;

				lpdc_decod_return = threegpp_nr_ldpc_decode_stop_funct(ldpc_decoder_fd);
				return((1+dlsch->max_ldpc_iterations));

			}	    	
	    	else if ( !crc_status_ldpc  ) 
	    	{  
	    		nb_error_decod++;
				total_decod_error++;
				total_decod_error_crc++;

				err_flag++;

				g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_error++;
				g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_error_crc++;

				// If CRC has failed it is likely that we will get max_iteration
				iteration_max = (1+dlsch->max_ldpc_iterations);

				LOG_W(PHY,"AbsSubframe %d.%d CRC failed, segment %d/%d \n",frame%1024,nr_tti_rx,r,harq_process->C-1);
			}

			else
			{
				if( iteration_max < no_iteration_ldpc )
					iteration_max = no_iteration_ldpc;
			}

		if(!(total_decod%100000))
		{
			ldpc_stat_flag = 1;
		}


	}	// r Segments LOOP


	if( !nr_tti_rx && !(frame%1000) && ldpc_stat_flag) {

		printf("\nLDPC BLER %f (%.6f %%) error %ld total %ld\n", (float)nb_error_decod/(float)nb_total_decod, (float)(nb_error_decod*100.0)/(float)nb_total_decod, nb_error_decod, nb_total_decod);
		nb_error_decod	= 0;
		nb_total_decod	= 0;

	 	printf("ldpc_decode_getq total=%ld err=%ld (%.6f %%) crc_err=%ld it=%ld mean it.=%ld \n",
	 		total_decod, total_decod_error, (float)total_decod_error/total_decod*100 ,total_decod_error_crc, total_decod_it, total_decod_it/total_decod );


		for ( int dbg_loop=0; dbg_loop<10; dbg_loop++)
		{
		 	printf(" ldpc_stat subfram=%d %.4ld/%ld (%.2f %%) crc_err=%4.4ld it=%2.2ld mean it.=%ld max it.=%ld max_iterationblock_err=%ld\n",
		 		dbg_loop,
		 		g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_error,
		 		g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_count,
		 		(float)g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_error/(g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_count+1)*100 ,
		 		g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_error_crc,
		 		g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_it,
		 		g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_it/(g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_count+1),
				g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_max_it,
				g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_tb_error_block_max);



			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_count = 0;
			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_error = 0;
			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_error_crc = 0;
			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_it = 0;
			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_block_decod_max_it = 0;
			g_ldpc_stat_per_subframe_array[dbg_loop].ldpc_tb_error_block_max = 0;
		}
		printf("\n");
	}



	if(g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_max_it < iteration_max)
		g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_block_decod_max_it = iteration_max;

	if( g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_tb_error_block_max < err_flag )
		g_ldpc_stat_per_subframe_array[nr_tti_rx].ldpc_tb_error_block_max = err_flag;


	lpdc_decod_return = threegpp_nr_ldpc_decode_stop_funct(ldpc_decoder_fd);
#endif

	// If we detect an error
	if (err_flag)
		ret = (1+dlsch->max_ldpc_iterations);
	else
		ret = iteration_max;


	if (err_flag || ret == (1+dlsch->max_ldpc_iterations) )
	{

		#if UE_DEBUG_TRACE
		LOG_I(PHY,"[UE %d] DLSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d, mcs %d) Kr %d r %d harq_process->round %d\n",
		    phy_vars_ue->Mod_id, frame, nr_tti_rx, harq_pid,harq_process->status, harq_process->round,harq_process->TBS,harq_process->mcs,Kr,r,harq_process->round);
		#endif
		harq_process->harq_ack.ack = 0;
	    harq_process->harq_ack.harq_id = harq_pid;
	    harq_process->harq_ack.send_harq_status = 1;
	    harq_process->errors[harq_process->round]++;
	    harq_process->round++;

		// If we have reach harq max round or no mac (phytest mode) -> set harq process in idle mode
		if ((harq_process->round >= dlsch->Mdlharq) || (phy_vars_ue->mac_enabled==0) )
		{
			harq_process->status = SCH_IDLE;
			harq_process->round  = 0;
		}
		if(is_crnti)
		{
			LOG_D(PHY,"[UE %d] DLSCH: Setting NACK for nr_tti_rx %d (pid %d, pid status %d, round %d/Max %d, TBS %d)\n",
							phy_vars_ue->Mod_id,
							nr_tti_rx,
							harq_pid, harq_process->status, harq_process->round, dlsch->Mdlharq, harq_process->TBS);
		}

		return((1+dlsch->max_ldpc_iterations));
  
	} 
	else 
	{
		#if UE_DEBUG_TRACE
		LOG_I(PHY,"[UE %d] DLSCH: Setting ACK for nr_tti_rx %d TBS %d mcs %d nb_rb %d harq_process->round %d\n",
				phy_vars_ue->Mod_id,nr_tti_rx,harq_process->TBS,harq_process->mcs,harq_process->nb_rb, harq_process->round);
		#endif

		harq_process->status = SCH_IDLE;
	    harq_process->round  = 0;
	    harq_process->harq_ack.ack = 1;
	    harq_process->harq_ack.harq_id = harq_pid;
	    harq_process->harq_ack.send_harq_status = 1;

		if(is_crnti)
		{
			LOG_D(PHY,"[UE %d] DLSCH: Setting ACK for nr_tti_rx %d (pid %d, round %d, TBS %d)\n",
						phy_vars_ue->Mod_id,
						nr_tti_rx,
						harq_pid, harq_process->round, harq_process->TBS);
		}
	}	


	// Reassembly of Transport block here
	offset = 0;
	Kr = harq_process->K;
	Kr_bytes = Kr>>3;

	for (r=0; r<harq_process->C; r++) {

	    memcpy(harq_process->b+offset,
	             harq_process->c[r],
	             Kr_bytes- - (harq_process->F>>3) -((harq_process->C>1)?3:0));
	    offset += (Kr_bytes - (harq_process->F>>3) - ((harq_process->C>1)?3:0));

	#ifdef DEBUG_DLSCH_DECODING
	    printf("Segment %d : Kr= %d bytes\n",r,Kr_bytes);
	    printf("copied %d bytes to b sequence (harq_pid %d)\n",
	              (Kr_bytes - (harq_process->F>>3)-((harq_process->C>1)?3:0)),harq_pid);
	              printf("b[0] = %x,c[%d] = %x\n",
	              harq_process->b[offset],
	              harq_process->F>>3,
	              harq_process->c[r]);
	#endif
	}






	dlsch->last_iteration_cnt = ret;

	return(ret);

}


#ifdef DEBUG_LLR_STATS
void debug_llr_stats(void)
{
	
	// first LLR seems to be buggous, wait a little before compute statistics
	if (llr_cnt > 1024)
	{
		llr_p_mean1	= 0.0;
		llr_p_cnt1	= 0.0;
		llr_n_mean1	= 0.0;
		llr_n_cnt1	= 0.0;
		for (int i = 0; i < Kr*3; i++)
		{
			if (harq_process->d[r][96+i] > llr_max)
				llr_max = harq_process->d[r][96+i];
			if (harq_process->d[r][96+i] < llr_min)
				llr_min = harq_process->d[r][96+i];
			if (harq_process->d[r][96+i] > 0)
			{
				llr_p_mean1	+= harq_process->d[r][96+i];
				llr_p_cnt1	+= 1.0;
			}
			if (harq_process->d[r][96+i] < 0)
			{
				llr_n_mean1	+= harq_process->d[r][96+i];
				llr_n_cnt1	+= 1.0;
			}
		}
		llr_n_mean	+= llr_n_mean1;
		llr_n_cnt	+= llr_n_cnt1;
		llr_p_mean	+= llr_p_mean1;
		llr_p_cnt	+= llr_p_mean1;
	}

	llr_cnt++;
	
	if ( !(llr_cnt%4096) )
	{
		printf("LDPC decoding LLRmax=%d LLRmin=%d pLLRmean=%lf (cnt:%lf) nLLRmean=%lf (cnt:%lf) pLLRmean1=%lf (cnt1:%lf) nLLRmean1=%lf (cnt1:%lf)\n",
			llr_max,
			llr_min,
			(llr_p_mean/llr_p_cnt),
			llr_p_cnt,
			(llr_n_mean/llr_n_cnt),
			llr_n_cnt,
			(llr_p_mean1/llr_p_cnt1),
			llr_p_cnt1,
			(llr_n_mean1/llr_n_cnt1),
			llr_n_cnt1);
	}
	return;
}
#endif



#ifdef DLSIM_DECOD_CHECK
static uint32_t failure_cnt = 0;
static uint32_t total_cnt = 0;
void debug_check_decoder_input_buff( uint32_t Kr, uint32_t Zc,  uint32_t r , LTE_DL_UE_HARQ_t *harq_process )	
{
	uint32_t 	failure = 0;
	uint32_t 	poncture = 0;
	uint32_t 	failure_systematic = 0;
	uint32_t 	poncture_systematic = 0;
	uint32_t 	failure_parity89 = 0;
	uint32_t 	poncture_parity89 = 0;
	uint32_t 	failure_parity13 = 0;
	uint32_t 	poncture_parity13 = 0;


	total_cnt++;


	// LLR coded data
	for (int i = 0; i < (Kr*3); i++)
	{
		uint8_t bit = ((uint16_t *)harq_process->d[r])[96+i]&0x8000 ? 0 : 1;

		// Check for Poncturing
		if (!((uint16_t *)harq_process->d[r])[96+i]){
			poncture++;
			if (i < Zc*20)
				poncture_systematic++;
			else if (i < Zc*25)
				poncture_parity89++;
			else{
				//poncture_parity13++;
			}


		}

		// Check for erronous position
		if ((bit-dlsch_coding_d[r][96+i]) || (!((uint16_t *)harq_process->d[r])[96+i]) ) {
			if (i< Zc*20){
				failure++;
				failure_systematic++;
			}
			else if (i< Zc*25){
				failure++;
				failure_parity89++;
			}
			else{
				//failure++;
				//failure_parity13++;
			}
		}

	}
	

	if (failure)
	{

		failure_cnt++;
		#if 1
			printf("!! SYRTEM failure_cnt = %d/%d - r=%d - failure=%d/%d sys=%d parity89=%d parity13=%d !!!\n", 
						failure_cnt, total_cnt,  r,
						failure, Kr*3, failure_systematic, failure_parity89, failure_parity13 );
			printf("!! SYRTEM 	%d - r=%d - poncture=%d sys=%d parity89=%d parity13=%d !!!\n", 
						Kr*3, r,
						poncture, poncture_systematic, poncture_parity89, poncture_parity13 );
		#endif
	}

	return;


}
#endif 
#endif	// LDPC_FPGA_OFFLOAD
