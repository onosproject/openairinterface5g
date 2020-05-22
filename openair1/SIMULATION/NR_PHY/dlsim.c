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

#define _GNU_SOURCE
#include <sched.h>
#include <time.h>
#include "PHY/phy_extern.h"
#include <fcntl.h>
#include <math.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common/ran_context.h"
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_defs.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_extern.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_vars_nr_ue.h"
#include "PHY/types.h"
#include "PHY/INIT/phy_init.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/NR_REFSIG/nr_mod_table.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_TRANSPORT/nr_transport.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "SCHED_NR/fapi_nr_l1.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR_UE/defs.h"
#include "SCHED_NR_UE/fapi_nr_ue_l1.h"
#include "NR_PHY_INTERFACE/NR_IF_Module.h"
#include "NR_UE_PHY_INTERFACE/NR_IF_Module.h"
#include "RRC/NR/MESSAGES/asn1_msg.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/SIMULATION/NR_PHY/nr_dummy_functions.c"
#define CPU_AFF

#include "PHY/CODING/nrLDPC_encoder/defs.h"

PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
int32_t uplink_frequency_offset[MAX_NUM_CCs][4];

double cpuf;
int nfapi_mode=0;
uint16_t NB_UE_INST = 1;

//Dummy Functions
lte_subframe_t subframe_select(LTE_DL_FRAME_PARMS *frame_parms, unsigned char subframe) {return(SF_DL);}
int rlc_module_init (int enb) {return(0);}
void pdcp_layer_init (void) {}
int rrc_init_nr_global_param (void) {return(0);}
void config_common(int Mod_idP,int CC_idP,int Nid_cell,int nr_bandP,uint64_t SSB_positions,uint16_t ssb_periodicity,uint64_t dl_CarrierFreqP,uint32_t dl_BandwidthP);
int8_t nr_mac_rrc_data_ind_ue(const module_id_t module_id, const int CC_id, const uint8_t gNB_index,
                              const int8_t channel, const uint8_t* pduP, const sdu_size_t pdu_len) {return(0);}
uint64_t get_softmodem_optmask(void) {return 0;}
mac_rlc_status_resp_t mac_rlc_status_ind(const module_id_t       module_idP,
                                         const rnti_t            rntiP,
                                         const eNB_index_t       eNB_index,
                                         const frame_t           frameP,
                                         const sub_frame_t       subframeP,
                                         const eNB_flag_t        enb_flagP,
                                         const MBMS_flag_t       MBMS_flagP,
                                         const logical_chan_id_t channel_idP,
                                         const tb_size_t         tb_sizeP,
                                         const uint32_t          sourceL2Id,
                                         const uint32_t          destinationL2Id) {mac_rlc_status_resp_t  mac_rlc_status_resp = {0}; return mac_rlc_status_resp;}
tbs_size_t mac_rlc_data_req(const module_id_t       module_idP,
                            const rnti_t            rntiP,
                            const eNB_index_t       eNB_index,
                            const frame_t           frameP,
                            const eNB_flag_t        enb_flagP,
                            const MBMS_flag_t       MBMS_flagP,
                            const logical_chan_id_t channel_idP,
                            const tb_size_t         tb_sizeP,
                            char                    *buffer_pP,
                            const uint32_t          sourceL2Id,
                            const uint32_t          destinationL2Id) {return 0;}
int generate_dlsch_header(unsigned char *mac_header,
                          unsigned char num_sdus,
                          unsigned short *sdu_lengths,
                          unsigned char *sdu_lcids,
                          unsigned char drx_cmd,
                          unsigned short timing_advance_cmd,
                          unsigned char *ue_cont_res_id,
                          unsigned char short_padding,
                          unsigned short post_padding){return 0;}
void nr_ip_over_LTE_DRB_preconfiguration(void){}
void mac_rlc_data_ind     (
  const module_id_t         module_idP,
  const rnti_t              rntiP,
  const eNB_index_t         eNB_index,
  const frame_t             frameP,
  const eNB_flag_t          enb_flagP,
  const MBMS_flag_t         MBMS_flagP,
  const logical_chan_id_t   channel_idP,
  char                     *buffer_pP,
  const tb_size_t           tb_sizeP,
  num_tb_t                  num_tbP,
  crc_t                    *crcs_pP)
{}

// needed for some functions
openair0_config_t openair0_cfg[MAX_CARDS];


uint32_t nr_get_G(uint16_t nb_rb, uint16_t nb_symb_sch, uint8_t nb_re_dmrs, uint16_t length_dmrs, uint8_t Qm, uint8_t Nl);
uint32_t nr_get_E(uint32_t G, uint8_t C, uint8_t Qm, uint8_t Nl, uint8_t r);
uint32_t nr_compute_tbslbrm(uint16_t table, uint16_t nb_rb, uint8_t Nl, uint8_t C);
void nr_interleaving_ldpc(uint32_t E, uint8_t Qm, uint8_t *e,uint8_t *f);
int nr_rate_matching_ldpc(uint8_t Ilbrm, uint32_t Tbslbrm, uint8_t BG, uint16_t Z, uint8_t *w, uint8_t *e, uint8_t C, uint8_t rvidx, uint32_t E);
int32_t nr_segmentation(unsigned char *input_buffer, unsigned char **output_buffers, unsigned int B, unsigned int *C, unsigned int *K, unsigned int *Zout, unsigned int *F, uint8_t BG);
int ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput);

/*! \file openair1/SIMULATION/NR_PHY/dlsim.c
 * \brief multi-parallelism threads for pdsch
 * \author Terngyin Hsu, Sendren Xu, Nungyi Kuo, Kuankai Hsiung, Kaimi Yang (OpInConnect_NCTU)
 * \email tyhsu@cs.nctu.edu.tw
 * \date 22-05-2020
 * \version 3.0
 * \note
 * \warning
 */

//[START]multi_genetate_pdsch_proc
struct timespec start_encoder_ts[thread_num_pdsch], end_encoder_ts[thread_num_pdsch], start_perenc_ts[thread_num_pdsch], end_perenc_ts[thread_num_pdsch], start_pressure_ts[thread_num_pressure], end_pressure_ts[thread_num_pressure], start_perpre_ts[thread_num_pressure], end_perpre_ts[thread_num_pressure];
struct timespec start_multi_enc_ts[2], end_multi_enc_ts[2], start_multi_scr_ts[2], end_multi_scr_ts[2], start_multi_mod_ts[2], end_multi_mod_ts[2];
int vcd = 0;	//default : 0
/*original genetate_pdsch for multi threads*/
static void *multi_genetate_pdsch_proc(void *ptr){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  multi_ldpc_encoder_gNB *test =(multi_ldpc_encoder_gNB*) ptr;
  printf("[READY] : %d\n", test->id);
  while(!oai_exit){
  	while(pthread_cond_wait(&(gNB->multi_encoder[test->id].cond),&(gNB->multi_encoder[test->id].mutex))!=0);
  	  if(oai_exit){	//If oai_exit, KILL this thread!
  	  	pthread_mutex_destroy(&gNB->multi_encoder[test->id].mutex);
  	  	pthread_mutex_destroy(&gNB->multi_encoder[test->id].mutex_scr_mod);
      	pthread_join(gNB->multi_encoder[test->id].pthread, NULL);
      	return 0;
  	  }
	  //Print params of multi_ldpc_enc
	  // printf("[b]ldpc_encoder_optim_8seg: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",
	  //       ldpc_enc[test->id].BG,
	  //       ldpc_enc[test->id].Zc,
	  //       ldpc_enc[test->id].Kb,
	  //       ldpc_enc[test->id].block_length,
	  //       ldpc_enc[test->id].n_segments);

	  int j_start, j_end; //Check if our situation((n_segments > 7)&(thread_num_pdsch == 2))
	  if((gNB->multi_encoder[test->id].n_segments > 7)&&(thread_num_pdsch == 2)){
	    j_start = test->id;
	    j_end = j_start + 1;
	  }else{
	    j_start = 0;
	    j_end = (gNB->multi_encoder[test->id].n_segments/8+1);
	  }
	  int offset = test->id>7?7:test->id;
	  //printf("[OFFSET] : %d %d\n", offset, test->id);
	  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	  if(check_time)
	  	clock_gettime(CLOCK_MONOTONIC, &start_encoder_ts[test->id]);  //timing	  
	  for(int j=j_start;j<j_end;j++){	  	
	  	if(check_time){
	  		printf("   Active      %d       %d\n", test->id, j);
	    	clock_gettime(CLOCK_MONOTONIC, &start_perenc_ts[test->id]);  //timing
	  	}	    
	    ldpc_encoder_optim_8seg_multi(gNB->multi_encoder[test->id].test_input,//gNB->multi_encoder[0].c_test,
	                                  gNB->multi_encoder[test->id].channel_input_optim,//gNB->multi_encoder[0].d_test,
	                                  gNB->multi_encoder[test->id].Zc,
	                                  gNB->multi_encoder[test->id].Kb,
	                                  gNB->multi_encoder[test->id].block_length,
	                                  gNB->multi_encoder[test->id].BG,
	                                  gNB->multi_encoder[test->id].n_segments,
	                                  j,
	                                  NULL, NULL, NULL, NULL);
	    if(check_time){
	    	clock_gettime(CLOCK_MONOTONIC, &end_perenc_ts[test->id]);  //timing
	    	printf("    Done       %d       %d      %.2f usec\n", test->id, j, (end_perenc_ts[test->id].tv_nsec - start_perenc_ts[test->id].tv_nsec) *1.0 / 1000);
	    }	    
	  }
	  if(check_time){
	  	clock_gettime(CLOCK_MONOTONIC, &end_encoder_ts[test->id]);  //timing
	  	printf("  All done     %d              %.2f usec\n", test->id, (end_encoder_ts[test->id].tv_nsec - start_encoder_ts[test->id].tv_nsec) *1.0 / 1000);
	  }	  
	  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	  
	  gNB->multi_encoder[test->id].complete = 1;
	  //==================================================
	while(pthread_cond_wait(&(gNB->multi_encoder[test->id].cond_scr_mod),&(gNB->multi_encoder[test->id].mutex_scr_mod))!=0);
	VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	if(check_time)
		clock_gettime(CLOCK_MONOTONIC, &start_encoder_ts[test->id]);  //timing	
	for (int q=0; q<1; q++){	//Need to change by codewords
	  if(check_time){
	  	printf("   Active      %d       %d\n", test->id, q);
	  	clock_gettime(CLOCK_MONOTONIC, &start_perenc_ts[test->id]);  //timing
	  }	  
	  nr_pdsch_codeword_scrambling(gNB->multi_encoder[test->id].f,
	                               gNB->multi_encoder[test->id].encoded_length,
	                               q,
	                               gNB->multi_encoder[test->id].Nid,
	                               gNB->multi_encoder[test->id].n_RNTI,
	                               gNB->multi_encoder[test->id].scrambled_output);	  
	  nr_modulation(gNB->multi_encoder[test->id].scrambled_output,
	                gNB->multi_encoder[test->id].encoded_length,
	                gNB->multi_encoder[test->id].Qm,
	                gNB->multi_encoder[test->id].mod_symbs);
	  if(check_time){
	  	clock_gettime(CLOCK_MONOTONIC, &end_perenc_ts[test->id]);  //timing
	  	printf("    Done       %d       %d      %.2f usec\n", test->id, q, (end_perenc_ts[test->id].tv_nsec - start_perenc_ts[test->id].tv_nsec) *1.0 / 1000);
	  }
	}
	if(check_time){
		clock_gettime(CLOCK_MONOTONIC, &end_encoder_ts[test->id]);  //timing
		printf("  All done     %d              %.2f usec\n", test->id, (end_encoder_ts[test->id].tv_nsec - start_encoder_ts[test->id].tv_nsec) *1.0 / 1000);
	}	
	VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	
	gNB->multi_encoder[test->id].complete_scr_mod = 1;
  }
  return 0;
}
/*pressure test*/
static void *multi_genetate_pdsch_pressure(void *ptr){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  multi_ldpc_encoder_gNB *test =(multi_ldpc_encoder_gNB*) ptr;
  printf("[READY] : %d(p)\n", test->id);
  while(!oai_exit){
  	while(pthread_cond_wait(&(gNB->pressure_test[test->id].cond),&(gNB->pressure_test[test->id].mutex))!=0);
  	  if(oai_exit){	//If oai_exit, KILL this thread!
  	  	pthread_mutex_destroy(&gNB->pressure_test[test->id].mutex);
  	  	pthread_mutex_destroy(&gNB->pressure_test[test->id].mutex_scr_mod);
      	pthread_join(gNB->pressure_test[test->id].pthread, NULL);
      	return 0;
  	  }
	  //Print params of multi_ldpc_enc
	  // printf("[b]ldpc_encoder_optim_8seg: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",
	  //       ldpc_enc[test->id].BG,
	  //       ldpc_enc[test->id].Zc,
	  //       ldpc_enc[test->id].Kb,
	  //       ldpc_enc[test->id].block_length,
	  //       ldpc_enc[test->id].n_segments);

	  int j_start, j_end; //Check if our situation((n_segments > 7)&(thread_num_pdsch == 2))
	  if((gNB->pressure_test[test->id].n_segments > 7)&&(thread_num_pdsch == 2)){
	    j_start = test->id;
	    j_end = j_start + 1;
	  }else{
	    j_start = 0;
	    j_end = (gNB->pressure_test[test->id].n_segments/8+1);
	  }
	  int offset = test->id+thread_num_pdsch;
	  if(offset>7){
	  	offset = 7;
	  }
	  //printf("[OFFSET] : %d %d\n", offset, test->id);
	  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	  if(check_time)
	  	clock_gettime(CLOCK_MONOTONIC, &start_pressure_ts[test->id]);  //timing	  
	  for(int j=j_start;j<j_end;j++){
	  	if(check_time){
	  		printf("   Active      %d(p)    %d\n", test->id, j);
	    	clock_gettime(CLOCK_MONOTONIC, &start_perpre_ts[test->id]);  //timing
	  	}
	    ldpc_encoder_optim_8seg_multi(gNB->pressure_test[test->id].c_test,
	                                  gNB->pressure_test[test->id].d_test,
	                                  gNB->pressure_test[test->id].Zc,
	                                  gNB->pressure_test[test->id].Kb,
	                                  gNB->pressure_test[test->id].block_length,
	                                  gNB->pressure_test[test->id].BG,
	                                  gNB->pressure_test[test->id].n_segments,
	                                  j,
	                                  NULL, NULL, NULL, NULL);
	    if(check_time){
	    	clock_gettime(CLOCK_MONOTONIC, &end_perpre_ts[test->id]);  //timing
	    	printf("    Done       %d(p)    %d      %.2f usec\n", test->id, j, (end_perpre_ts[test->id].tv_nsec - start_perpre_ts[test->id].tv_nsec) *1.0 / 1000);
	    }
	  }
	  if(check_time){
	  	clock_gettime(CLOCK_MONOTONIC, &end_pressure_ts[test->id]);  //timing
	  	printf("  All done     %d(p)           %.2f usec\n", test->id, (end_pressure_ts[test->id].tv_nsec - start_pressure_ts[test->id].tv_nsec) *1.0 / 1000);
	  }	  
	  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	  
	  gNB->pressure_test[test->id].complete = 1;
	  //==================================================
	while(pthread_cond_wait(&(gNB->pressure_test[test->id].cond_scr_mod),&(gNB->pressure_test[test->id].mutex_scr_mod))!=0);
	VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	if(check_time)
		clock_gettime(CLOCK_MONOTONIC, &start_pressure_ts[test->id]);  //timing	
	for (int q=0; q<1; q++){	//Need to change by codewords
	  if(check_time){
	  	printf("   Active      %d(p)    %d\n", test->id, q);
	  	clock_gettime(CLOCK_MONOTONIC, &start_perpre_ts[test->id]);  //timing
	  }	  
	  nr_pdsch_codeword_scrambling(gNB->pressure_test[test->id].f_test,
	                               gNB->pressure_test[test->id].encoded_length,
	                               q,
	                               gNB->pressure_test[test->id].Nid,
	                               gNB->pressure_test[test->id].n_RNTI,
	                               gNB->pressure_test[test->id].scrambled_output_test);	  
	  nr_modulation(gNB->pressure_test[test->id].scrambled_output_test,
	                gNB->pressure_test[test->id].encoded_length,
	                gNB->pressure_test[test->id].Qm,
	                gNB->pressure_test[test->id].mod_symbs_test);
	  if(check_time){
	  	clock_gettime(CLOCK_MONOTONIC, &end_perpre_ts[test->id]);  //timing
	  	printf("    Done       %d(p)    %d      %.2f usec\n", test->id, q, (end_perpre_ts[test->id].tv_nsec - start_perpre_ts[test->id].tv_nsec) *1.0 / 1000);
	  }
	}
	if(check_time){
		clock_gettime(CLOCK_MONOTONIC, &end_pressure_ts[test->id]);  //timing
		printf("  All done     %d(p)           %.2f usec\n", test->id, (end_pressure_ts[test->id].tv_nsec - start_pressure_ts[test->id].tv_nsec) *1.0 / 1000);
	}	
	VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	
	gNB->pressure_test[test->id].complete_scr_mod = 1;
  }
  return 0;
}
/*ldpc_encoder*/
static void *multi_ldpc_encoder_proc(int id){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  printf("[READY] : %d(e)\n", id);
  while(!oai_exit){
  	while(pthread_cond_wait(&(gNB->multi_pdsch.cond_enc[id]),&(gNB->multi_pdsch.mutex_enc[id]))!=0);
  	  if(oai_exit){	//If oai_exit, KILL this thread!
  	  	pthread_mutex_destroy(&gNB->multi_pdsch.mutex_enc[id]);
      	pthread_join(gNB->multi_pdsch.pthread_enc[id], NULL);
      	return 0;
  	  }

	  // int offset = id+thread_num_pdsch;
	  // if(offset>7){
	  // 	offset = 7;
	  // }
	  //printf("[OFFSET] : %d %d\n", offset, id);

	  int j_start, j_end;	// ==Wait for n_segments increasing ==***
	  j_start = 0;
	  j_end = 1;
	  //VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	  if(check_time)
	  	clock_gettime(CLOCK_MONOTONIC, &start_multi_enc_ts[id]);  //timing	  
	  for(int j=j_start;j<j_end;j++){
	  	if(check_time){
	  		printf("   Active      %d(e)    %d\n", id, j);
	    	//clock_gettime(CLOCK_MONOTONIC, &start_perpre_ts[id]);  //timing
	  	}
		ldpc_encoder_optim_8seg_multi(gNB->multi_pdsch.c[id],
		                              gNB->multi_pdsch.d[id],
		                              gNB->multi_pdsch.Zc[id],
		                              gNB->multi_pdsch.Kb[id],
		                              gNB->multi_pdsch.block_length[id],
		                              gNB->multi_pdsch.BG[id],
		                              gNB->multi_pdsch.n_segments[id],
		                              j,
		                              NULL, NULL, NULL, NULL);   
	    if(check_time){
	    	// clock_gettime(CLOCK_MONOTONIC, &end_perpre_ts[id]);  //timing
	    	// printf("    Done       %d(e)    %d      %.2f usec\n", id, j, (end_perpre_ts[id].tv_nsec - start_perpre_ts[id].tv_nsec) *1.0 / 1000);
	    }
	  }
	  if(check_time){
	  	clock_gettime(CLOCK_MONOTONIC, &end_multi_enc_ts[id]);  //timing
	  	printf("  All done     %d(e)           %.2f usec\n", id, (end_multi_enc_ts[id].tv_nsec - start_multi_enc_ts[id].tv_nsec) *1.0 / 1000);
	  }	  
	  //VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	  
	  gNB->multi_pdsch.complete_enc[id] = 1;
  }
  return 0;
}
/*scrambling*/
static void *multi_scrambling_proc(int id){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  printf("[READY] : %d(s)\n", id);
  while(!oai_exit){
	while(pthread_cond_wait(&(gNB->multi_pdsch.cond_scr[id]),&(gNB->multi_pdsch.mutex_scr[id]))!=0);
	if(oai_exit){	//If oai_exit, KILL this thread!
  	  	pthread_mutex_destroy(&gNB->multi_pdsch.mutex_scr[id]);
      	pthread_join(gNB->multi_pdsch.pthread_scr[id], NULL);
      	return 0;
  	  }
	//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	if(check_time)
		clock_gettime(CLOCK_MONOTONIC, &start_multi_scr_ts[id]);  //timing	
	for (int q=0; q<1; q++){	//Need to change by codewords
	  if(check_time){
	  	printf("   Active      %d(s)    %d\n", id, q);
	  	// clock_gettime(CLOCK_MONOTONIC, &start_perpre_ts[id]);  //timing
	  }
	  //if(0){
	  nr_pdsch_codeword_scrambling(gNB->multi_pdsch.f[id],
		                           gNB->multi_pdsch.encoded_length_scr[id],
		                           q,
		                           gNB->multi_pdsch.Nid[id],
		                           gNB->multi_pdsch.n_RNTI[id],
		                           gNB->multi_pdsch.scrambled_output_scr[id]);
	  if(check_time){
	  	// clock_gettime(CLOCK_MONOTONIC, &end_perpre_ts[id]);  //timing
	  	// printf("    Done       %d(p)    %d      %.2f usec\n", id, q, (end_perpre_ts[id].tv_nsec - start_perpre_ts[id].tv_nsec) *1.0 / 1000);
	  }
	}
	if(check_time){
		clock_gettime(CLOCK_MONOTONIC, &end_multi_scr_ts[id]);  //timing
		printf("  All done     %d(s)           %.2f usec\n", id, (end_multi_scr_ts[id].tv_nsec - start_multi_scr_ts[id].tv_nsec) *1.0 / 1000);
	}	
	//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	
	gNB->multi_pdsch.complete_scr[id] = 1;
  }
  return 0;
}
/*modulation*/
static void *multi_modulation_proc(int id){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  printf("[READY] : %d(m)\n", id);
  while(!oai_exit){
	while(pthread_cond_wait(&(gNB->multi_pdsch.cond_mod[id]),&(gNB->multi_pdsch.mutex_mod[id]))!=0);
	if(oai_exit){	//If oai_exit, KILL this thread!
  	  	pthread_mutex_destroy(&gNB->multi_pdsch.mutex_mod[id]);
      	pthread_join(gNB->multi_pdsch.pthread_mod[id], NULL);
      	return 0;
  	  }
	//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
	if(check_time)
		clock_gettime(CLOCK_MONOTONIC, &start_multi_mod_ts[id]);  //timing
	//printf("%d, %p\n", id, gNB->multi_pdsch.mod_symbs[id]);
	//printf("%d, %p\n", id, *gNB->multi_pdsch.mod_symbs[id]);
	for (int q=0; q<1; q++){	//Need to change by codewords
	  if(check_time){
	  	printf("   Active      %d(m)    %d\n", id, q);
	  	// clock_gettime(CLOCK_MONOTONIC, &start_perpre_ts[id]);  //timing
	  }
	  nr_modulation(gNB->multi_pdsch.scrambled_output_mod[id],
		            gNB->multi_pdsch.encoded_length_mod[id],
		            gNB->multi_pdsch.Qm[id],
		            *gNB->multi_pdsch.mod_symbs[id]);//gNB->pressure_test[id].mod_symbs_test);
	  if(check_time){
	  	// clock_gettime(CLOCK_MONOTONIC, &end_perpre_ts[id]);  //timing
	  	// printf("    Done       %d(p)    %d      %.2f usec\n", id, q, (end_perpre_ts[id].tv_nsec - start_perpre_ts[id].tv_nsec) *1.0 / 1000);
	  }
	}
	if(check_time){
		clock_gettime(CLOCK_MONOTONIC, &end_multi_mod_ts[id]);  //timing
		printf("  All done     %d(m)           %.2f usec\n", id, (end_multi_mod_ts[id].tv_nsec - start_multi_mod_ts[id].tv_nsec) *1.0 / 1000);
	}	
	//VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);	
	gNB->multi_pdsch.complete_mod[id] = 1;
  }
  return 0;
}
//[END]multi_genetate_pdsch_proc

int main(int argc, char **argv)
{
  char c;
  int i,aa;//,l;
  double sigma2, sigma2_dB=10, SNR, snr0=-2.0, snr1=2.0;
  uint8_t snr1set=0;
  int **txdata;
  double **s_re,**s_im,**r_re,**r_im;
  //double iqim = 0.0;
  //unsigned char pbch_pdu[6];
  //  int sync_pos, sync_pos_slot;
  //  FILE *rx_frame_file;
  FILE *output_fd = NULL;
  //uint8_t write_output_file=0;
  //int result;
  //int freq_offset;
  //  int subframe_offset;
  //  char fname[40], vname[40];
  int trial, n_trials = 1, n_errors = 0, n_false_positive = 0;
  //int n_errors2, n_alamouti;
  uint8_t transmission_mode = 1,n_tx=1,n_rx=1;
  uint16_t Nid_cell=0;
  uint64_t SSB_positions=0x01;

  channel_desc_t *gNB2UE;
  //uint32_t nsymb,tx_lev,tx_lev1 = 0,tx_lev2 = 0;
  //uint8_t extended_prefix_flag=0;
  //int8_t interf1=-21,interf2=-21;

  FILE *input_fd=NULL,*pbch_file_fd=NULL;
  //char input_val_str[50],input_val_str2[50];

  //uint8_t frame_mod4,num_pdcch_symbols = 0;

  SCM_t channel_model=AWGN;//Rayleigh1_anticorr;



  //double pbch_sinr;
  //int pbch_tx_ant;
  int N_RB_DL=106,mu=1;
  nfapi_nr_dl_config_dlsch_pdu_rel15_t dlsch_config;
  dlsch_config.start_prb = 0;
  dlsch_config.n_prb = 50;
  dlsch_config.start_symbol = 2;
  dlsch_config.nb_symbols = 9;
  dlsch_config.mcs_idx = 9;

  uint16_t ssb_periodicity = 10;

  //unsigned char frame_type = 0;
  unsigned char pbch_phase = 0;

  int frame=0,slot=1;
  int frame_length_complex_samples;
  int frame_length_complex_samples_no_prefix;
  int slot_length_complex_samples_no_prefix;
  NR_DL_FRAME_PARMS *frame_parms;
  nfapi_nr_config_request_t *gNB_config;
  UE_nr_rxtx_proc_t UE_proc;
  NR_Sched_Rsp_t Sched_INFO;
  gNB_MAC_INST *gNB_mac;
  NR_UE_MAC_INST_t *UE_mac;

  int ret;
  int run_initial_sync=0;
  int do_pdcch_flag=1;

  uint16_t cset_offset = 0;
  int loglvl=OAILOG_WARNING;

  float target_error_rate = 0.01;

  cpuf = get_cpu_freq_GHz();
  
  if ( load_configmodule(argc,argv,CONFIG_ENABLECMDLINEONLY) == 0) {
    exit_fun("[NR_DLSIM] Error, configuration module init failed\n");
  }

  randominit(0);

  while ((c = getopt (argc, argv, "f:hA:pf:g:i:j:n:s:S:t:x:y:z:M:N:F:GR:dP:IL:o:a:b:c:j:e:V:")) != -1) {
    switch (c) {
    /*case 'f':
      write_output_file=1;
      output_fd = fopen(optarg,"w");

      if (output_fd==NULL) {
        printf("Error opening %s\n",optarg);
        exit(-1);
      }
      break;*/

    /*case 'd':
      frame_type = 1;
      break;*/

    case 'g':
      switch((char)*optarg) {
      case 'A':
        channel_model=SCM_A;
        break;

      case 'B':
        channel_model=SCM_B;
        break;

      case 'C':
        channel_model=SCM_C;
        break;

      case 'D':
        channel_model=SCM_D;
        break;

      case 'E':
        channel_model=EPA;
        break;

      case 'F':
        channel_model=EVA;
        break;

      case 'G':
        channel_model=ETU;
        break;

      default:
        printf("Unsupported channel model!\n");
        exit(-1);
      }

      break;

    /*case 'i':
      interf1=atoi(optarg);
      break;

    case 'j':
      interf2=atoi(optarg);
      break;*/

    case 'n':
      n_trials = atoi(optarg);
      break;

    case 's':
      snr0 = atof(optarg);
      printf("Setting SNR0 to %f\n",snr0);
      break;

    case 'S':
      snr1 = atof(optarg);
      snr1set=1;
      printf("Setting SNR1 to %f\n",snr1);
      break;

      /*
      case 't':
      Td= atof(optarg);
      break;
      */
    /*case 'p':
      extended_prefix_flag=1;
      break;*/

      /*
      case 'r':
      ricean_factor = pow(10,-.1*atof(optarg));
      if (ricean_factor>1) {
        printf("Ricean factor must be between 0 and 1\n");
        exit(-1);
      }
      break;
      */
    case 'x':
      transmission_mode=atoi(optarg);

      if ((transmission_mode!=1) &&
          (transmission_mode!=2) &&
          (transmission_mode!=6)) {
        printf("Unsupported transmission mode %d\n",transmission_mode);
        exit(-1);
      }

      break;

    case 'y':
      n_tx=atoi(optarg);

      if ((n_tx==0) || (n_tx>2)) {
        printf("Unsupported number of tx antennas %d\n",n_tx);
        exit(-1);
      }

      break;

    case 'z':
      n_rx=atoi(optarg);

      if ((n_rx==0) || (n_rx>2)) {
        printf("Unsupported number of rx antennas %d\n",n_rx);
        exit(-1);
      }

      break;

    case 'M':
      SSB_positions = atoi(optarg);
      break;

    case 'N':
      Nid_cell = atoi(optarg);
      break;

    case 'R':
      N_RB_DL = atoi(optarg);
      break;

    case 'F':
      input_fd = fopen(optarg,"r");

      if (input_fd==NULL) {
        printf("Problem with filename %s\n",optarg);
        exit(-1);
      }

      break;

    case 'P':
      pbch_phase = atoi(optarg);

      if (pbch_phase>3)
        printf("Illegal PBCH phase (0-3) got %d\n",pbch_phase);

      break;
      
    case 'I':
      run_initial_sync=1;
      target_error_rate=0.1;
      break;

    case 'L':
      loglvl = atoi(optarg);
      break;

    case 'o':
      cset_offset = atoi(optarg);
      break;

    case 'a':
      dlsch_config.start_prb = atoi(optarg);
      break;

    case 'b':
      dlsch_config.n_prb = atoi(optarg);
      break;

    case 'c':
      dlsch_config.start_symbol = atoi(optarg);
      break;

    case 'j':
      dlsch_config.nb_symbols = atoi(optarg);
      break;

    case 'e':
      dlsch_config.mcs_idx = atoi(optarg);
      break;

    case 'V':
      vcd = atoi(optarg);
      if(vcd)
      	T_init(2021, 1, 0);	//VCD init
      break;

    default:
    case 'h':
      printf("%s -h(elp) -p(extended_prefix) -N cell_id -f output_filename -F input_filename -g channel_model -n n_frames -t Delayspread -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant -i Intefrence0 -j Interference1 -A interpolation_file -C(alibration offset dB) -N CellId\n",
             argv[0]);
      printf("-h This message\n");
      //printf("-p Use extended prefix mode\n");
      //printf("-d Use TDD\n");
      printf("-n Number of frames to simulate\n");
      printf("-s Starting SNR, runs from SNR0 to SNR0 + 5 dB.  If n_frames is 1 then just SNR is simulated\n");
      printf("-S Ending SNR, runs from SNR0 to SNR1\n");
      printf("-t Delay spread for multipath channel\n");
      printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
      printf("-x Transmission mode (1,2,6 for the moment)\n");
      printf("-y Number of TX antennas used in eNB\n");
      printf("-z Number of RX antennas used in UE\n");
      //printf("-i Relative strength of first intefering eNB (in dB) - cell_id mod 3 = 1\n");
      //printf("-j Relative strength of second intefering eNB (in dB) - cell_id mod 3 = 2\n");
      printf("-M Multiple SSB positions in burst\n");
      printf("-N Nid_cell\n");
      printf("-R N_RB_DL\n");
      printf("-O oversampling factor (1,2,4,8,16)\n");
      printf("-A Interpolation_filname Run with Abstraction to generate Scatter plot using interpolation polynomial in file\n");
      //printf("-C Generate Calibration information for Abstraction (effective SNR adjustment to remove Pe bias w.r.t. AWGN)\n");
      //printf("-f Output filename (.txt format) for Pe/SNR results\n");
      printf("-F Input filename (.txt format) for RX conformance testing\n");
      printf("-o CORESET offset\n");
      printf("-a Start PRB for PDSCH\n");
      printf("-b Number of PRB for PDSCH, default : 50\n");
      printf("-c Start symbol for PDSCH (fixed for now)\n");
      printf("-j Number of symbols for PDSCH (fixed for now)\n");
      printf("-e MSC index\n");
      exit (-1);
      break;
    }
  }  

  logInit();
  set_glog(loglvl);
  T_stdout = 1;

  if (snr1set==0)
    snr1 = snr0+10;

  printf("Initializing gNodeB for mu %d, N_RB_DL %d\n",mu,N_RB_DL);

  RC.gNB = (PHY_VARS_gNB***) malloc(sizeof(PHY_VARS_gNB **));
  RC.gNB[0] = (PHY_VARS_gNB**) malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0][0] = malloc(sizeof(PHY_VARS_gNB));
  memset(RC.gNB[0][0],0,sizeof(PHY_VARS_gNB));

  gNB = RC.gNB[0][0];
  gNB_config = &gNB->gNB_config;
  frame_parms = &gNB->frame_parms; //to be initialized I suppose (maybe not necessary for PBCH)
  frame_parms->nb_antennas_tx = n_tx;
  frame_parms->nb_antennas_rx = n_rx;
  frame_parms->N_RB_DL = N_RB_DL;
  frame_parms->N_RB_UL = N_RB_DL;

  // stub to configure frame_parms
  nr_phy_config_request_sim(gNB,N_RB_DL,N_RB_DL,mu,Nid_cell,SSB_positions);
  // call MAC to configure common parameters

  phy_init_nr_gNB(gNB,0,0);
  mac_top_init_gNB();

  double fs,bw;

  if (mu == 1 && N_RB_DL == 217) { 
    fs = 122.88e6;
    bw = 80e6;
  }					       
  else if (mu == 1 && N_RB_DL == 245) {
    fs = 122.88e6;
    bw = 90e6;
  }
  else if (mu == 1 && N_RB_DL == 273) {
    fs = 122.88e6;
    bw = 100e6;
  }
  else if (mu == 1 && N_RB_DL == 106) { 
    fs = 61.44e6;
    bw = 40e6;
  }
  else AssertFatal(1==0,"Unsupported numerology for mu %d, N_RB %d\n",mu, N_RB_DL);

  gNB2UE = new_channel_desc_scm(n_tx,
                                n_rx,
                                channel_model,
 				fs,
				bw,
                                0,
                                0,
                                0);

  if (gNB2UE==NULL) {
    printf("Problem generating channel model. Exiting.\n");
    exit(-1);
  }

  frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  slot_length_complex_samples_no_prefix = frame_parms->samples_per_slot_wCP;

  s_re = malloc(2*sizeof(double*));
  s_im = malloc(2*sizeof(double*));
  r_re = malloc(2*sizeof(double*));
  r_im = malloc(2*sizeof(double*));
  txdata = malloc(2*sizeof(int*));

  for (i=0; i<2; i++) {

    s_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_re[i],frame_length_complex_samples*sizeof(double));
    s_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(s_im[i],frame_length_complex_samples*sizeof(double));

    r_re[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_re[i],frame_length_complex_samples*sizeof(double));
    r_im[i] = malloc(frame_length_complex_samples*sizeof(double));
    bzero(r_im[i],frame_length_complex_samples*sizeof(double));

    printf("Allocating %d samples for txdata\n",frame_length_complex_samples);
    txdata[i] = malloc(frame_length_complex_samples*sizeof(int));
    bzero(txdata[i],frame_length_complex_samples*sizeof(int));
  
  }

  if (pbch_file_fd!=NULL) {
    load_pbch_desc(pbch_file_fd);
  }


  //configure UE
  UE = malloc(sizeof(PHY_VARS_NR_UE));
  memset((void*)UE,0,sizeof(PHY_VARS_NR_UE));
  PHY_vars_UE_g = malloc(sizeof(PHY_VARS_NR_UE**));
  PHY_vars_UE_g[0] = malloc(sizeof(PHY_VARS_NR_UE*));
  PHY_vars_UE_g[0][0] = UE;
  memcpy(&UE->frame_parms,frame_parms,sizeof(NR_DL_FRAME_PARMS));

  if (run_initial_sync==1)  UE->is_synchronized = 0;
  else                      {UE->is_synchronized = 1; UE->UE_mode[0]=PUSCH;}
                      
  UE->perfect_ce = 0;
  for (i=0;i<10;i++) UE->current_thread_id[i] = 0;

  if (init_nr_ue_signal(UE, 1, 0) != 0)
  {
    printf("Error at UE NR initialisation\n");
    exit(-1);
  }
  
 //  for (int t = 0; t < 4; t++){	//create threads
	// gNB->thread_encode[t].id = t;
	// gNB->thread_encode[t].flag_wait = 1;
	// pthread_mutex_init( &(gNB->thread_encode[t].mutex_encode), NULL);
	// pthread_cond_init( &(gNB->thread_encode[t].cond_encode), NULL);
	// pthread_create(&(gNB->thread_encode[t].pthread_encode),&gNB->thread_encode[t].attr_encode,dlsch_encoding_proc,&gNB->thread_encode[t]);
	// printf("[CREATE] DLSCH ENCODER Thread %d\n", gNB->thread_encode[t].id);
 //  }

 //  for(int aa = 0;aa<1;aa++){
	// pthread_attr_init( &gNB->thread_scrambling[aa].attr_scrambling);
	// pthread_mutex_init(&(gNB->thread_scrambling[aa].mutex_tx), NULL);
	// pthread_cond_init(&(gNB->thread_scrambling[aa].cond_tx), NULL);
	// gNB->thread_scrambling[aa].q_id = aa;
	// pthread_create(&(gNB->thread_scrambling[aa].pthread_scrambling),&gNB->thread_scrambling[aa].attr_scrambling,scrambling_proc,&gNB->thread_scrambling[aa]);
	// printf("[CREATE] scrambling_channel thread[%d] \n",aa);
 //  }
	
  // pthread_mutex_init( &(gNB->thread_modulation.mutex_tx), NULL);
  // pthread_cond_init( &(gNB->thread_modulation.cond_tx), NULL);
  // pthread_create(&(gNB->thread_modulation.pthread_modulation),&gNB->thread_modulation.attr_modulation,modulation_proc,&gNB->thread_modulation);
  // printf("[CREATE] modulation_channel thread \n");

  //[START]multi_genetate_pdsch_proc:create thread
  for(int th=0;th<thread_num_pdsch;th++){
    pthread_attr_init(&(gNB->multi_encoder[th].attr));
    pthread_mutex_init(&(gNB->multi_encoder[th].mutex), NULL);
    pthread_mutex_init(&(gNB->multi_encoder[th].mutex_scr_mod), NULL);
    pthread_cond_init(&(gNB->multi_encoder[th].cond), NULL);
    pthread_cond_init(&(gNB->multi_encoder[th].cond_scr_mod), NULL);
    gNB->multi_encoder[th].id = th;
    gNB->multi_encoder[th].flag_wait = 1;
    gNB->multi_encoder[th].complete = 0;
    gNB->multi_encoder[th].complete_scr_mod = 0;
    pthread_create(&(gNB->multi_encoder[th].pthread), &(gNB->multi_encoder[th].attr), multi_genetate_pdsch_proc, &(gNB->multi_encoder[th]));    
    printf("[CREATE] LDPC encoder thread %d \n",gNB->multi_encoder[th].id);
  }
  for(int th=0;th<thread_num_pressure;th++){
    pthread_attr_init(&(gNB->pressure_test[th].attr));
    pthread_mutex_init(&(gNB->pressure_test[th].mutex), NULL);
    pthread_mutex_init(&(gNB->pressure_test[th].mutex_scr_mod), NULL);
    pthread_cond_init(&(gNB->pressure_test[th].cond), NULL);
    pthread_cond_init(&(gNB->pressure_test[th].cond_scr_mod), NULL);
    gNB->pressure_test[th].id = th;
    gNB->pressure_test[th].flag_wait = 1;
    gNB->pressure_test[th].complete = 0;
    gNB->pressure_test[th].complete_scr_mod = 0;
    pthread_create(&(gNB->pressure_test[th].pthread), &(gNB->pressure_test[th].attr), multi_genetate_pdsch_pressure, &(gNB->pressure_test[th]));    
    printf("[CREATE] LDPC encoder thread %d(p) \n",gNB->pressure_test[th].id);
  }
  /*multi pdsch*/
  for(int th=0;th<2;th++){
    pthread_attr_init(&(gNB->multi_pdsch.attr_enc[th]));
    pthread_mutex_init(&(gNB->multi_pdsch.mutex_enc[th]), NULL);
    pthread_cond_init(&(gNB->multi_pdsch.cond_enc[th]), NULL);
    gNB->multi_pdsch.id_enc[th] = th;
    gNB->multi_pdsch.complete_enc[th] = 0;
    pthread_create(&(gNB->multi_pdsch.pthread_enc[th]), &(gNB->multi_pdsch.attr_enc[th]), multi_ldpc_encoder_proc, gNB->multi_pdsch.id_enc[th]);    
    printf("[CREATE] multi_ldpc_encoder_proc %d \n",gNB->multi_pdsch.id_enc[th]);
  }
  for(int th=0;th<2;th++){
    pthread_attr_init(&(gNB->multi_pdsch.attr_scr[th]));
    pthread_mutex_init(&(gNB->multi_pdsch.mutex_scr[th]), NULL);
    pthread_cond_init(&(gNB->multi_pdsch.cond_scr[th]), NULL);
    gNB->multi_pdsch.id_scr[th] = th;
    gNB->multi_pdsch.complete_scr[th] = 0;
    pthread_create(&(gNB->multi_pdsch.pthread_scr[th]), &(gNB->multi_pdsch.attr_scr[th]), multi_scrambling_proc, gNB->multi_pdsch.id_scr[th]);    
    printf("[CREATE] multi_scrambling_proc %d \n",gNB->multi_pdsch.id_scr[th]);
  }
  //for(int th=0;th<0;th++){
  for(int th=0;th<2;th++){
    pthread_attr_init(&(gNB->multi_pdsch.attr_mod[th]));
    pthread_mutex_init(&(gNB->multi_pdsch.mutex_mod[th]), NULL);
    pthread_cond_init(&(gNB->multi_pdsch.cond_mod[th]), NULL);
    gNB->multi_pdsch.id_mod[th] = th;
    gNB->multi_pdsch.complete_mod[th] = 0;
    pthread_create(&(gNB->multi_pdsch.pthread_mod[th]), &(gNB->multi_pdsch.attr_mod[th]), multi_modulation_proc, gNB->multi_pdsch.id_mod[th]);    
    printf("[CREATE] multi_modulation_proc %d \n",gNB->multi_pdsch.id_mod[th]);
  }
  //[END]multi_genetate_pdsch_proc:create thread

  init_nr_ue_transport(UE,0);
  usleep(1000000);
  nr_gold_pbch(UE);
  nr_gold_pdcch(UE,0,2);

  RC.nb_nr_macrlc_inst = 1;
  mac_top_init_gNB();
  gNB_mac = RC.nrmac[0];

  config_common(0,0,Nid_cell,78,SSB_positions,ssb_periodicity,(uint64_t)3640000000L,N_RB_DL);
  config_nr_mib(0,0,1,kHz30,0,0,0,0,0);

  nr_l2_init_ue();
  UE_mac = get_mac_inst(0);
  
  UE->pdcch_vars[0][0]->crnti = 0x1234;

  UE->if_inst = nr_ue_if_module_init(0);
  UE->if_inst->scheduled_response = nr_ue_scheduled_response;
  UE->if_inst->phy_config_request = nr_ue_phy_config_request;
  UE->if_inst->dl_indication = nr_ue_dl_indication;
  UE->if_inst->ul_indication = dummy_nr_ue_ul_indication;
  

  UE_mac->if_module = nr_ue_if_module_init(0);
  
  unsigned int available_bits;
  unsigned char *estimated_output_bit;
  unsigned char *test_input_bit;
  unsigned int errors_bit    = 0;
  uint32_t errors_scrambling = 0;

  test_input_bit       = (unsigned char *) malloc16(sizeof(unsigned char) * 16 * 68 * 384);
  estimated_output_bit = (unsigned char *) malloc16(sizeof(unsigned char) * 16 * 68 * 384);
  
  uint16_t rb_offset_count = cset_offset/6;
  set_cset_offset(rb_offset_count);
  // generate signal
  if (input_fd==NULL) {
    gNB->pbch_configured = 1;
    for (int i=0;i<4;i++) gNB->pbch_pdu[i]=i+1;

    nr_schedule_uss_dlsch_phytest(0,frame,slot,&dlsch_config);
    Sched_INFO.module_id = 0;
    Sched_INFO.CC_id     = 0;
    Sched_INFO.frame     = frame;
    Sched_INFO.slot      = slot;
    Sched_INFO.DL_req    = &gNB_mac->DL_req[0];
    Sched_INFO.UL_tti_req    = &gNB_mac->UL_tti_req[0];
    Sched_INFO.HI_DCI0_req  = NULL;
    Sched_INFO.TX_req    = &gNB_mac->TX_req[0];
    nr_schedule_response(&Sched_INFO);

    phy_procedures_gNB_TX(gNB,frame,slot,0);	// ==the way to downlink ==
    
    //nr_common_signal_procedures (gNB,frame,subframe);
    int txdataF_offset = (slot%2) * frame_parms->samples_per_slot_wCP;

    LOG_M("txsigF0.m","txsF0", gNB->common_vars.txdataF[0],frame_length_complex_samples_no_prefix,1,1);
    if (gNB->frame_parms.nb_antennas_tx>1)
      LOG_M("txsigF1.m","txsF1", gNB->common_vars.txdataF[1],frame_length_complex_samples_no_prefix,1,1);

    int tx_offset = slot*frame_parms->samples_per_slot;
  //  printf("samples_per_slot_wCP = %d\n", frame_parms->samples_per_slot_wCP);

    //TODO: loop over slots
    for (aa=0; aa<gNB->frame_parms.nb_antennas_tx; aa++) {
      if (gNB_config->subframe_config.dl_cyclic_prefix_type.value == 1) {
	PHY_ofdm_mod(&gNB->common_vars.txdataF[aa][txdataF_offset],
		     &txdata[aa][tx_offset],
		     frame_parms->ofdm_symbol_size,
		     12,
		     frame_parms->nb_prefix_samples,
		     CYCLIC_PREFIX);
      } else {
	nr_normal_prefix_mod(&gNB->common_vars.txdataF[aa][txdataF_offset],
			     &txdata[aa][tx_offset],
			     14,
			     frame_parms);
      }
    }
  } else {
    printf("Reading %d samples from file to antenna buffer %d\n",frame_length_complex_samples,0);
    
    if (fread(txdata[0],
	      sizeof(int32_t),
	      frame_length_complex_samples,
	      input_fd) != frame_length_complex_samples) {
      printf("error reading from file\n");
      //exit(-1);
    }
  }

  LOG_M("txsig0.m","txs0", txdata[0],frame_length_complex_samples,1,1);
  if (gNB->frame_parms.nb_antennas_tx>1)
    LOG_M("txsig1.m","txs1", txdata[1],frame_length_complex_samples,1,1);

  if (output_fd) 
    fwrite(txdata[0],sizeof(int32_t),frame_length_complex_samples,output_fd);

  int txlev = signal_energy(&txdata[0][5*frame_parms->ofdm_symbol_size + 4*frame_parms->nb_prefix_samples + frame_parms->nb_prefix_samples0],
			    frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples);

  //  printf("txlev %d (%f)\n",txlev,10*log10(txlev));

  for (i=0; i<frame_length_complex_samples; i++) {
    for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
      r_re[aa][i] = ((double)(((short *)txdata[aa]))[(i<<1)]);
      r_im[aa][i] = ((double)(((short *)txdata[aa]))[(i<<1)+1]);
    }
  }


  //Configure UE
  rrc_gNB_carrier_data_t carrier;
  uint32_t pdcch_ConfigSIB1     = 0;
  uint32_t ssb_SubcarrierOffset = 0;
  carrier.MIB = (uint8_t*) malloc(4);
  carrier.sizeof_MIB = do_MIB_NR(&carrier,0,ssb_SubcarrierOffset,pdcch_ConfigSIB1,30,2);

  nr_rrc_mac_config_req_ue(0,0,0,carrier.mib.message.choice.mib,NULL,NULL,NULL);

  // Initial bandwidth part configuration -- full carrier bandwidth
  UE_mac->initial_bwp_dl.bwp_id = 0;
  UE_mac->initial_bwp_dl.location = 0;
  UE_mac->initial_bwp_dl.scs = UE->frame_parms.subcarrier_spacing;
  UE_mac->initial_bwp_dl.N_RB = UE->frame_parms.N_RB_DL;
  UE_mac->initial_bwp_dl.cyclic_prefix = UE->frame_parms.Ncp;
  
  fapi_nr_dl_config_request_t *dl_config = &UE_mac->dl_config_request; 
  //  Type0 PDCCH search space
  dl_config->number_pdus =  1;
  dl_config->dl_config_list[0].pdu_type = FAPI_NR_DL_CONFIG_TYPE_DCI;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.rnti = 0x1234;	
  
  uint64_t mask = 0x0;
  uint16_t num_rbs=24;
  uint16_t rb_offset=gNB->pdcch_vars.dci_alloc[0].pdcch_params.rb_offset;
  uint16_t cell_id=0;
  uint16_t num_symbols=2;
  for(i=0; i<(num_rbs/6); ++i){   //  38.331 Each bit corresponds a group of 6 RBs
    mask = mask >> 1;
    mask = mask | 0x100000000000;
  }
  uint16_t UE_rb_offset_count = rb_offset/6;
  mask = mask >> UE_rb_offset_count;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.frequency_domain_resource = mask;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.rb_offset = rb_offset;  //  additional parameter other than coreset
  
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.duration = num_symbols;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.cce_reg_mapping_type =CCE_REG_MAPPING_TYPE_NON_INTERLEAVED;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.cce_reg_interleaved_reg_bundle_size = 0;   //  L 38.211 7.3.2.2
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.cce_reg_interleaved_interleaver_size = 0;  //  R 38.211 7.3.2.2
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.cce_reg_interleaved_shift_index = cell_id;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.precoder_granularity = PRECODER_GRANULARITY_SAME_AS_REG_BUNDLE;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.coreset.pdcch_dmrs_scrambling_id = cell_id;
  
  uint32_t number_of_search_space_per_slot=1;
  uint32_t first_symbol_index=0;
  uint32_t search_space_duration=0;  //  element of search space
  uint32_t coreset_duration;  //  element of coreset
  
  coreset_duration = num_symbols * number_of_search_space_per_slot;
  
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.number_of_candidates[0] = table_38213_10_1_1_c2[0];
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.number_of_candidates[1] = table_38213_10_1_1_c2[1];
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.number_of_candidates[2] = table_38213_10_1_1_c2[2];   //  CCE aggregation level = 4
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.number_of_candidates[3] = table_38213_10_1_1_c2[3];   //  CCE aggregation level = 8
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.number_of_candidates[4] = table_38213_10_1_1_c2[4];   //  CCE aggregation level = 16
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.duration = search_space_duration;
  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.monitoring_symbols_within_slot = (0x3fff << first_symbol_index) & (0x3fff >> (14-coreset_duration-first_symbol_index)) & 0x3fff;

  dl_config->dl_config_list[0].dci_config_pdu.dci_config_rel15.N_RB_BWP = N_RB_DL;

  UE_mac->scheduled_response.dl_config = dl_config;
  UE_mac->scheduled_response.ul_config = NULL;
  UE_mac->scheduled_response.tx_request = NULL;
  UE_mac->scheduled_response.module_id = 0;
  UE_mac->scheduled_response.CC_id = 0;
  UE_mac->scheduled_response.frame = frame;
  UE_mac->scheduled_response.slot = slot;

  UE_mac->phy_config.config_req.pbch_config.system_frame_number = frame;
  UE_mac->phy_config.config_req.pbch_config.ssb_index = 0;
  UE_mac->phy_config.config_req.pbch_config.half_frame_bit = 0;

  nr_ue_phy_config_request(&UE_mac->phy_config);

  for (SNR = snr0; SNR < snr1; SNR += .2) {

    n_errors = 0;
    //n_errors2 = 0;
    //n_alamouti = 0;

    n_false_positive = 0;
    for (trial = 0; trial < n_trials; trial++) {

      errors_bit = 0;
      //multipath channel
      //multipath_channel(gNB2UE,s_re,s_im,r_re,r_im,frame_length_complex_samples,0);
      
      //AWGN
      sigma2_dB = 10 * log10((double)txlev) - SNR;
      sigma2    = pow(10, sigma2_dB/10);
      // printf("sigma2 %f (%f dB)\n",sigma2,sigma2_dB);

      for (i=0; i<frame_length_complex_samples; i++) {
	for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {
	  ((short*) UE->common_vars.rxdata[aa])[2*i]   = (short) ((r_re[aa][i] + sqrt(sigma2/2)*gaussdouble(0.0,1.0)));
	  ((short*) UE->common_vars.rxdata[aa])[2*i+1] = (short) ((r_im[aa][i] + sqrt(sigma2/2)*gaussdouble(0.0,1.0)));
	}
      }

      if (n_trials == 1) {

        LOG_M("rxsig0.m","rxs0", UE->common_vars.rxdata[0], frame_length_complex_samples, 1, 1);
        if (UE->frame_parms.nb_antennas_rx>1)
          LOG_M("rxsig1.m","rxs1", UE->common_vars.rxdata[1], frame_length_complex_samples, 1, 1);

      }

      if (UE->is_synchronized == 0) {

        UE_nr_rxtx_proc_t proc = {0};
        ret = nr_initial_sync(&proc, UE, normal_txrx, 1);
        printf("nr_initial_sync1 returns %d\n", ret);

        if (ret < 0) 
          n_errors++;

      } else { // UE->is_synchronized != 0

        UE->rx_offset       = 0;
        UE_proc.frame_rx    = frame;
        UE_proc.nr_tti_rx   = slot;
        UE_proc.subframe_rx = slot;

        nr_ue_scheduled_response(&UE_mac->scheduled_response);

        printf("Running phy procedures UE RX %d.%d\n", frame, slot);

        phy_procedures_nrUE_RX(UE,
                               &UE_proc,
                               0,
                               do_pdcch_flag,
                               normal_txrx);

        if (n_trials == 1) {

          LOG_M("rxsigF0.m","rxsF0", UE->common_vars.common_vars_rx_data_per_thread[0].rxdataF[0], slot_length_complex_samples_no_prefix, 1, 1);
          if (UE->frame_parms.nb_antennas_rx > 1)
            LOG_M("rxsigF1.m","rxsF1", UE->common_vars.common_vars_rx_data_per_thread[0].rxdataF[1], slot_length_complex_samples_no_prefix, 1, 1);

        }

        if (UE->dlsch[UE->current_thread_id[slot]][0][0]->last_iteration_cnt >= 
	    UE->dlsch[UE->current_thread_id[slot]][0][0]->max_ldpc_iterations+1)
          n_errors++;

        //----------------------------------------------------------
        //---------------------- count errors ----------------------
        //----------------------------------------------------------
        
        NR_gNB_DLSCH_t *gNB_dlsch = gNB->dlsch[0][0];

        NR_UE_DLSCH_t *dlsch0 = UE->dlsch[UE->current_thread_id[UE_proc.nr_tti_rx]][0][0];
        int harq_pid = dlsch0->current_harq_pid;
        NR_DL_UE_HARQ_t *UE_harq_process = dlsch0->harq_processes[harq_pid];
        
        NR_UE_PDSCH **pdsch_vars = UE->pdsch_vars[UE->current_thread_id[UE_proc.nr_tti_rx]];
        int16_t *UE_llr = pdsch_vars[0]->llr[0];

        nfapi_nr_dl_config_dlsch_pdu_rel15_t rel15 = gNB_dlsch->harq_processes[harq_pid]->dlsch_pdu.dlsch_pdu_rel15;
        uint32_t TBS         = rel15.transport_block_size;
        uint16_t length_dmrs = 1;
        uint16_t nb_rb       = rel15.n_prb;
        uint8_t  nb_re_dmrs  = rel15.nb_re_dmrs;
        uint8_t  mod_order   = rel15.modulation_order;
        uint8_t  nb_symb_sch = rel15.nb_symbols;
        
        available_bits = nr_get_G(nb_rb, nb_symb_sch, nb_re_dmrs, length_dmrs, mod_order, rel15.nb_layers);
        
        printf("\n");
        printf("available_bits = %u\n", available_bits);
  
        for (i = 0; i < available_bits; i++) {
          
          if(((gNB_dlsch->harq_processes[harq_pid]->f[i] == 0) && (UE_llr[i] <= 0)) || 
             ((gNB_dlsch->harq_processes[harq_pid]->f[i] == 1) && (UE_llr[i] >= 0)))
          {
            if(errors_scrambling == 0) {
              printf("\n");
              printf("First bit in error in unscrambling = %d\n",i);
            }
            errors_scrambling++;
          }
  
        }
  
        for (i = 0; i < TBS; i++) {
  
          estimated_output_bit[i] = (UE_harq_process->b[i/8] & (1 << (i & 7))) >> (i & 7);
          test_input_bit[i]       = (gNB_dlsch->harq_processes[harq_pid]->b[i / 8] & (1 << (i & 7))) >> (i & 7); // Further correct for multiple segments
  
          if (estimated_output_bit[i] != test_input_bit[i]) {
            if(errors_bit == 0)
              printf("First bit in error in decoding = %d\n",i);
            errors_bit++;
          }
          
        }
  
        ////////////////////////////////////////////////////////////
  
        if (errors_scrambling > 0) {
          if (n_trials == 1)
            printf("errors_scrambling = %u (trial %d)\n", errors_scrambling, trial);
        }
  
        if (errors_bit > 0) {
          n_false_positive++;
          if (n_trials == 1)
            printf("errors_bit = %u (trial %d)\n", errors_bit, trial);
        }

        printf("\n");

      } // if (UE->is_synchronized == 0)

    } // noise trials

    printf("*****************************************\n");
    printf("SNR %f, (false positive %f)\n", SNR,
           (float) n_errors / (float) n_trials);
    printf("*****************************************\n");
    printf("\n");

    printf("SNR %f : n_errors (negative CRC) = %d/%d\n", SNR, n_errors, n_trials);
    printf("\n");

    if ((float)n_errors/(float)n_trials <= target_error_rate) {
      printf("PDSCH test OK\n");
      break;
    }

  } // NSR

//[START]Send Kill massage
  oai_exit = 1;	//set oai status oai_exit = 1
  printf("Kill them all!\n");
  for(int th=0; th<thread_num_pdsch; th++){
  	pthread_cond_signal(&(gNB->multi_encoder[th].cond));
  }
  for(int th=0; th<thread_num_pressure; th++){
  	pthread_cond_signal(&(gNB->pressure_test[th].cond));
  }
//free pressure memorys
	unsigned char bw_scaling =2; // ==Need to change ==***
  for(int th=0;th<thread_num_pressure;th++){
  	for(int j=0;j<MAX_NUM_NR_DLSCH_SEGMENTS/bw_scaling;j++){
  	  free(gNB->pressure_test[th].c_test[j]);
  	  free(gNB->pressure_test[th].d_test[j]);
    }
    // ==We have some problom here ==???
    // for (int q=0; q<NR_MAX_NB_CODEWORDS; q++){
    //   free(*gNB->pressure_test[th].mod_symbs_test[q]);
    // }
  }
//free multi_pdsch memorys
  for(int th=0;th<2;th++){
  	for(int j=0;j<MAX_NUM_NR_DLSCH_SEGMENTS/bw_scaling;j++){
	  	free(gNB->multi_pdsch.c[th][j]);
	  	free(gNB->multi_pdsch.d[th][j]);
	}
  }
  for(int th=0;th<2;th++){
  	for (int q=0; q<NR_MAX_NB_CODEWORDS; q++){
	    free(gNB->multi_pdsch.mod_symbs[th][q]);
	}
  }
//[END]Send Kill massage

  for (i = 0; i < 2; i++) {
    free(s_re[i]);
    free(s_im[i]);
    free(r_re[i]);
    free(r_im[i]);
    free(txdata[i]);
  }

  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);
  free(txdata);
  free(test_input_bit);
  free(estimated_output_bit);
  
  if (output_fd)
    fclose(output_fd);

  if (input_fd)
    fclose(input_fd);

  return(n_errors);

}
