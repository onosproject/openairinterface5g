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

/*!\file ldpctest.c
 * \brief Implementation of multi_ldpc_encoder
 * \author Terngyin, NY, GK, KM (ISIP)
 * \email tyhsu@cs.nctu.edu.tw
 * \date 07-04-2020
 * \version 1.0
 * \note
 * \warning
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "assertions.h"
#include "SIMULATION/TOOLS/sim.h"
#include "PHY/CODING/nrLDPC_encoder/defs.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_decoder.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include <time.h>

//#include "T.h"
//#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"				 

#define MAX_NUM_DLSCH_SEGMENTS 16
#define MAX_BLOCK_LENGTH 8448

#ifndef malloc16
#  ifdef __AVX2__
#    define malloc16(x) memalign(32,x)
#  else
#    define malloc16(x) memalign(16,x)
#  endif
#endif

#define NR_LDPC_PROFILER_DETAIL
#define NR_LDPC_ENABLE_PARITY_CHECK

// 4-bit quantizer
char quantize4bit(double D,double x)
{
  double qxd;
  qxd = floor(x/D);
  //  printf("x=%f,qxd=%f\n",x,qxd);

  if (qxd <= -8)
    qxd = -8;
  else if (qxd > 7)
    qxd = 7;

  return((char)qxd);
}

char quantize8bit(double D,double x)
{
  double qxd;
  //char maxlev;
  qxd = floor(x/D);

  //maxlev = 1<<(B-1);

  //printf("x=%f,qxd=%f,maxlev=%d\n",x,qxd, maxlev);

  if (qxd <= -128)
    qxd = -128;
  else if (qxd >= 128)
    qxd = 127;

  return((char)qxd);
}

typedef struct {
  double n_iter_mean;
  double n_iter_std;
  int n_iter_max;
} n_iter_stats_t;

RAN_CONTEXT_t RC;
PHY_VARS_UE ***PHY_vars_UE_g;
uint16_t NB_UE_INST = 1;

short lift_size[51]= {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};

// ==[START]multi_lpdc_encoder
struct timespec start_ts, end_ts, start_per_ts, end_per_ts, start_enc_ts[thread_num_max], end_enc_ts[thread_num_max], start_perenc_ts[thread_num_max], end_perenc_ts[thread_num_max];
multi_ldpc_encoder ldpc_enc[thread_num_max];
int thread_num = 2; //Craete 2 threads
int ifRand = 0;
int vcd = 0;
// unsigned char *test_input0[MAX_NUM_DLSCH_SEGMENTS]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};;
// unsigned char *test_input1[MAX_NUM_DLSCH_SEGMENTS]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};;
// unsigned char *channel_input_optim0[MAX_NUM_DLSCH_SEGMENTS];
// unsigned char *channel_input_optim1[MAX_NUM_DLSCH_SEGMENTS];

static void *multi_ldpc_encoder_proc(void *ptr){
  //ldpc_encoder_optim_8seg_multi(dlsch->harq_processes[harq_pid]->c,dlsch->harq_processes[harq_pid]->d,*Zc,Kb,Kr,BG,dlsch->harq_processes[harq_pid]->C,j,NULL,NULL,NULL,NULL);
  //ldpc_encoder_optim_8seg_multi(unsigned char **test_input,unsigned char **channel_input,int Zc,int Kb,short block_length, short BG, int n_segments,unsigned int macro_num, time_stats_t *tinput,time_stats_t *tprep,time_stats_t *tparity,time_stats_t *toutput)
  multi_ldpc_encoder *test =(multi_ldpc_encoder*) ptr;
  printf("[READY] : %d\n", test->id);  
  while(pthread_cond_wait(&ldpc_enc[test->id].cond,&ldpc_enc[test->id].mutex)!=0);  

  //Print params of multi_ldpc_enc
  // printf("[b]ldpc_encoder_optim_8seg: BG %d, Zc %d, Kb %d, block_length %d, segments %d\n",
  //       ldpc_enc[test->id].BG,
  //       ldpc_enc[test->id].Zc,
  //       ldpc_enc[test->id].Kb,
  //       ldpc_enc[test->id].block_length,
  //       ldpc_enc[test->id].n_segments);

  int j_start, j_end; //Check if our situation((n_segments == 15)&(thread_num == 2))
  if((ldpc_enc[test->id].n_segments == 15)&&(thread_num == 2)){
    j_start = test->id;
    j_end = j_start + 1;
  }else{
    j_start = 0;
    j_end = (ldpc_enc[test->id].n_segments/8+1);
  }
  int offset = test->id>7?7:test->id;
  //printf("[OFFSET] : %d %d\n", offset, test->id); 
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,1);
  clock_gettime(CLOCK_MONOTONIC, &start_enc_ts[test->id]);  //timing
  for(int j=j_start;j<j_end;j++){
    //printf("  Movement    No.    Round    Cost time  \n");
    printf("   Active      %d       %d\n", test->id, j);
    clock_gettime(CLOCK_MONOTONIC, &start_perenc_ts[test->id]);  //timing
    ldpc_encoder_optim_8seg_multi(ldpc_enc[test->id].test_input,
                                  ldpc_enc[test->id].channel_input_optim,
                                  ldpc_enc[test->id].Zc,
                                  ldpc_enc[test->id].Kb,
                                  ldpc_enc[test->id].block_length,
                                  ldpc_enc[test->id].BG,
                                  ldpc_enc[test->id].n_segments,
                                  j,
                                  NULL, NULL, NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &end_perenc_ts[test->id]);  //timing
    //printf("  Movement    No.    Round    Cost time  \n");
    printf("    Done       %d       %d      %.2f usec\n", test->id, j, (end_perenc_ts[test->id].tv_nsec - start_perenc_ts[test->id].tv_nsec) *1.0 / 1000);
  }
  clock_gettime(CLOCK_MONOTONIC, &end_enc_ts[test->id]);  //timing
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC_0 + offset,0);
  //printf("[OK] : %d : %.2f usec\n", test->id, (end_perenc_ts[test->id].tv_nsec - start_perenc_ts[test->id].tv_nsec) *1.0 / 1000);
  ldpc_enc[test->id].complete = 1;
  return 0;
}
// ==[END]multi_lpdc_encoder

int test_ldpc(short No_iteration,
              int nom_rate,
              int denom_rate,
              double SNR,
              unsigned char qbits,
              short block_length,
              unsigned int ntrials,
              int n_segments,
              unsigned int *errors,
              unsigned int *errors_bit,
              double *errors_bit_uncoded,
              unsigned int *crc_misses,
              time_stats_t *time_optim,
              time_stats_t *time_decoder,
              n_iter_stats_t *dec_iter)
{
  //clock initiate
  //time_stats_t time,time_optim,tinput,tprep,tparity,toutput, time_decoder;
  time_stats_t time, tinput,tprep,tparity,toutput;
  double n_iter_mean = 0;
  double n_iter_std = 0;
  int n_iter_max = 0;
  unsigned int segment_bler = 0;

  double sigma;
  sigma = 1.0/sqrt(2*SNR);

  opp_enabled=1;
  cpu_freq_GHz = get_cpu_freq_GHz();
  //short test_input[block_length];
  unsigned char *test_input[MAX_NUM_DLSCH_SEGMENTS]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};;
  unsigned char *test_input_multi[thread_num_max][MAX_NUM_DLSCH_SEGMENTS];// ==For multi_ldpc_enc ==																								
  //short *c; //padded codeword
  unsigned char *estimated_output[MAX_NUM_DLSCH_SEGMENTS];
  unsigned char *estimated_output_bit[MAX_NUM_DLSCH_SEGMENTS];
  unsigned char *test_input_bit;
  unsigned char *channel_input[MAX_NUM_DLSCH_SEGMENTS];
  unsigned char *channel_output_uncoded[MAX_NUM_DLSCH_SEGMENTS];
  unsigned char *channel_input_optim[MAX_NUM_DLSCH_SEGMENTS];
  unsigned char *channel_input_optim_multi[thread_num_max][MAX_NUM_DLSCH_SEGMENTS];																				   
  double *channel_output;
  double *modulated_input[MAX_NUM_DLSCH_SEGMENTS];
  char *channel_output_fixed[MAX_NUM_DLSCH_SEGMENTS];
  unsigned int i,j,trial=0;
  short BG=0,nrows=0;//,ncols;
  int no_punctured_columns,removed_bit;
  int i1,Zc,Kb=0;
  int R_ind = 0;
  //Table of possible lifting sizes
  //short lift_size[51]= {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};
  //int n_segments=1;
  int code_rate_vec[8] = {15, 13, 25, 12, 23, 34, 56, 89};
  //double code_rate_actual_vec[8] = {0.2, 0.33333, 0.4, 0.5, 0.66667, 0.73333, 0.81481, 0.88};

  t_nrLDPC_dec_params decParams;
  t_nrLDPC_procBuf nrLDPC_procBuf;
  t_nrLDPC_procBuf* p_nrLDPC_procBuf = &nrLDPC_procBuf;
    
  t_nrLDPC_time_stats decoder_profiler;
  t_nrLDPC_time_stats* p_decoder_profiler =&decoder_profiler ;

  int32_t n_iter = 0;

  *errors=0;
  *errors_bit=0;
  *errors_bit_uncoded=0;
  *crc_misses=0;

  // generate input block
  for(j=0;j<MAX_NUM_DLSCH_SEGMENTS;j++) {
    test_input[j]=(unsigned char *)malloc16(sizeof(unsigned char) * block_length/8);
    channel_input[j] = (unsigned char *)malloc16(sizeof(unsigned char) * 68*384);
    channel_input_optim[j] = (unsigned char *)malloc16(sizeof(unsigned char) * 68*384);
	for(int th=0;th<thread_num;th++){// ==For multi_ldpc_enc ==
      test_input_multi[th][j]=(unsigned char *)malloc16(sizeof(unsigned char) * block_length/8);
      channel_input_optim_multi[th][j] = (unsigned char *)malloc16(sizeof(unsigned char) * 68*384);
    }
    channel_output_uncoded[j] = (unsigned char *)malloc16(sizeof(unsigned char) * 68*384);
    estimated_output[j] = (unsigned char*) malloc16(sizeof(unsigned char) * block_length);
    estimated_output_bit[j] = (unsigned char*) malloc16(sizeof(unsigned char) * block_length);
    modulated_input[j] = (double *)malloc16(sizeof(double) * 68*384);
    channel_output_fixed[j]  =  (char *)malloc16(sizeof( char) * 68*384);
  }
  //modulated_input = (double *)malloc(sizeof(double) * 68*384);
  //channel_output  = (double *)malloc(sizeof(double) * 68*384);
  //channel_output_fixed  = (char *)malloc16(sizeof(char) * 68*384);
  //modulated_input = (double *)calloc(68*384, sizeof(double));
  channel_output  =  (double *)calloc(68*384, sizeof(double));
  //channel_output_fixed  =  (double *)calloc(68*384, sizeof(double));
  //channel_output_fixed  =  (unsigned char*)calloc(68*384, sizeof(unsigned char*));
  //estimated_output = (unsigned char*) malloc16(sizeof(unsigned char) * block_length);///8);
  //estimated_output_bit = (unsigned char*) malloc16(sizeof(unsigned char) * block_length);
  test_input_bit = (unsigned char*) malloc16(sizeof(unsigned char) * block_length);

  reset_meas(&time);
  reset_meas(time_optim);
  reset_meas(time_decoder);
  reset_meas(&tinput);
  reset_meas(&tprep);
  reset_meas(&tparity);
  reset_meas(&toutput);
  //Reset Decoder profiles
  reset_meas(&decoder_profiler.llr2llrProcBuf);
  reset_meas(&decoder_profiler.llr2CnProcBuf);
  reset_meas(&decoder_profiler.cnProc);
  reset_meas(&decoder_profiler.cnProcPc);
  reset_meas(&decoder_profiler.bnProc);
  reset_meas(&decoder_profiler.bnProcPc);
  reset_meas(&decoder_profiler.cn2bnProcBuf);
  reset_meas(&decoder_profiler.bn2cnProcBuf);
  reset_meas(&decoder_profiler.llrRes2llrOut);
  reset_meas(&decoder_profiler.llr2bit);
  //reset_meas(&decoder_profiler.total);

  // Allocate LDPC decoder buffers
  p_nrLDPC_procBuf = nrLDPC_init_mem();

  for (j=0;j<MAX_NUM_DLSCH_SEGMENTS;j++) {
    for (i=0; i<block_length/8; i++) {
      test_input[j][i]=(unsigned char) rand();
	  for(int th = 0;th<thread_num;th++){
        if(ifRand){
          test_input_multi[th][j][i] = (unsigned char) rand();  // use rand value
        }else{
          test_input_multi[th][j][i] = test_input[j][i];  // use the same value
        }        
      }
    }
  }
  //Print out test_input
  printf("===============[test_input]================\n");
  for (j=0;j<1;j++) {
    for (i=0; i<10; i++) {      
      printf("%u", test_input[j][i]);
      for(int th = 0;th<thread_num;th++){
        printf(", %u", test_input_multi[th][j][i]);
      }
      printf("\n");
    }
  }


  //determine number of bits in codeword
  if (block_length>3840)
  {
    BG = 1;
    Kb = 22;
    nrows = 46; //parity check bits
    //ncols=22; //info bits
  }
  else if (block_length<=3840)
  {
    BG = 2;
    nrows = 42; //parity check bits
    //ncols=10; // info bits

    if (block_length>640)
      Kb = 10;
    else if (block_length>560)
      Kb = 9;
    else if (block_length>192)
      Kb = 8;
    else
      Kb = 6;
  }

  if (nom_rate == 1)
	  if (denom_rate == 5)
		  if (BG == 2)
			  R_ind = 0;
		  else
			  printf("Not supported");
	  else if (denom_rate == 3)
		  R_ind = 1;
	  else if (denom_rate == 2)
		  //R_ind = 3;
  	  	  printf("Not supported");
	  else
		  printf("Not supported");

  else if (nom_rate == 2)
	  if (denom_rate == 5)
		  //R_ind = 2;
  	  	  printf("Not supported");
	  else if (denom_rate == 3)
		  R_ind = 4;
	  else
		  printf("Not supported");

  else if ((nom_rate == 22) && (denom_rate == 30))
		  //R_ind = 5;
  	  	  printf("Not supported");
  else if ((nom_rate == 22) && (denom_rate == 27))
		  //R_ind = 6;
  	  	  printf("Not supported");
  else if ((nom_rate == 22) && (denom_rate == 25))
	  if (BG == 1)
		  R_ind = 7;
	  else
		  printf("Not supported");
  else
	  printf("Not supported");

  //find minimum value in all sets of lifting size
  Zc=0;

  for (i1=0; i1 < 51; i1++)
  {
    if (lift_size[i1] >= (double) block_length/Kb)
    {
      Zc = lift_size[i1];
      //printf("%d\n",Zc);
      break;
    }
  }
  printf("============== Check params ===============\n");
  printf("ldpc_test: codeword_length %d, n_segments %d, block_length %d, BG %d, Zc %d, Kb %d\n",n_segments *block_length, n_segments, block_length, BG, Zc, Kb);
  no_punctured_columns=(int)((nrows-2)*Zc+block_length-block_length*(1/((float)nom_rate/(float)denom_rate)))/Zc;
  //  printf("puncture:%d\n",no_punctured_columns);
  removed_bit=(nrows-no_punctured_columns-2) * Zc+block_length-(int)(block_length/((float)nom_rate/(float)denom_rate));
  if (ntrials==0)
    ldpc_encoder_orig(test_input[0],channel_input[0], Zc, BG, block_length, BG, 1);

  for (trial=0; trial < ntrials; trial++)
  {
	segment_bler = 0;
    //// encoder
    start_meas(&time);
    for(j=0;j<n_segments;j++) {
      ldpc_encoder_orig(test_input[j], channel_input[j],Zc,Kb,block_length,BG,0);
    }
    stop_meas(&time);

/*    start_meas(time_optim);
    ldpc_encoder_optim_8seg(test_input,channel_input_optim,Zc,Kb,block_length,BG,n_segments,&tinput,&tprep,&tparity,&toutput);
    for(j=0;j<n_segments;j++) {
      ldpc_encoder_optim(test_input[j],channel_input_optim[j],Zc,Kb,block_length,BG,&tinput,&tprep,&tparity,&toutput);
      }
    stop_meas(time_optim);*/
	// unsigned char c_extension_try[2*22*384*32] __attribute__((aligned(32)));      //double size matrix of c

    // ==[START]multi_ldpc_enc
    printf("============== Multi threads ==============\n");
    printf(" [Movement]  [No.]  [Part]   [Cost time] \n");
    for(int th = 0;th<thread_num;th++){      
      ldpc_enc[th].Zc = Zc;
      ldpc_enc[th].Kb = Kb;
      ldpc_enc[th].block_length = block_length;
      ldpc_enc[th].BG = BG;
      ldpc_enc[th].n_segments = n_segments; // ==Do all n_segments ==
      //ldpc_enc[th].n_segments = th+1; // ==Do n_segments increasing ==
      //ldpc_enc[th].n_segments = 1; // ==Just do first segment ==
      ldpc_enc[th].test_input = test_input_multi[th]; // It's a ptr @ the head of "th"th line
      ldpc_enc[th].channel_input_optim = channel_input_optim_multi[th]; // It's a ptr @ the head of "th"th line
    }    
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC,1);
    clock_gettime(CLOCK_MONOTONIC, &start_ts);//timing  
    for(int th = 0;th<thread_num;th++){
      pthread_cond_signal(&ldpc_enc[th].cond);
    }
    for(int th = 0;th<thread_num;th++){
      while(ldpc_enc[th].complete!=1);  // ==check if multi_ldpc_enc done ==
      VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_MULTI_ENC_FINISH,th);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_ts);//timing
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MULTI_ENC,0);
    for(int th = 0;th<thread_num;th++){
      pthread_mutex_destroy(&ldpc_enc[th].mutex);
      pthread_join(ldpc_enc[th].pthread, NULL);
    }    
    printf("========== Time of Multi threads ==========\n");
    printf(" [No.]  [Total time] \n");
    for(int th = 0;th<thread_num;th++){
    printf("   %d     %.2f usec\n", th, (end_enc_ts[th].tv_nsec - start_enc_ts[th].tv_nsec) *1.0 / 1000);
    }
    // printf("   t      %.2f usec\n", (end_enc_ts[0].tv_nsec - start_enc_ts[0].tv_nsec) *1.0 / 1000);
    // printf("   t      %.2f usec\n", (end_enc_ts[1].tv_nsec - start_enc_ts[0].tv_nsec) *1.0 / 1000);
    // printf("   t      %.2f usec\n", (end_enc_ts[0].tv_nsec - start_enc_ts[1].tv_nsec) *1.0 / 1000);
    // printf("   t      %.2f usec\n", (end_enc_ts[1].tv_nsec - start_enc_ts[1].tv_nsec) *1.0 / 1000);
    printf(" Total   %.2f usec\n", (end_ts.tv_nsec - start_ts.tv_nsec) *1.0 / 1000);    
    // ==[END]multi_ldpc_enc
    printf("============== Single thread ==============\n");
    printf(" [Movement]  [No.]  [Part]   [Cost time] \n");
    clock_gettime(CLOCK_MONOTONIC, &start_ts);//timing																										  

    for(j=0;j<(n_segments/8+1);j++) {
		printf("   Active    Single    %d\n", j);									   
    	start_meas(time_optim);
		clock_gettime(CLOCK_MONOTONIC, &start_per_ts);//timing													
    	ldpc_encoder_optim_8seg_multi(test_input,channel_input_optim,Zc,Kb,block_length, BG, n_segments,j,&tinput,&tprep,&tparity,&toutput);
		clock_gettime(CLOCK_MONOTONIC, &end_per_ts);//timing												  
    	stop_meas(time_optim);
		printf("    Done     Single    %d      %.2f usec\n", j, (end_per_ts.tv_nsec - start_per_ts.tv_nsec) *1.0 / 1000);      																													 
    }
    clock_gettime(CLOCK_MONOTONIC, &end_ts);//timing
    printf("========== Time of Single thread ==========\n");
    printf(" [No.]  [Total time] \n");
    printf(" Total   %.2f usec\n", (end_ts.tv_nsec - start_ts.tv_nsec) *1.0 / 1000);
    if((n_segments != 15)||(thread_num != 2))  //if not our situation
      printf("Total*%d  %.2f usec\n", thread_num, (end_ts.tv_nsec - start_ts.tv_nsec) *1.0*thread_num / 1000);

    //Print out channel_input
    // printf("[channel_input]:\n");
    // for (j=0;j<n_segments;j++){
    //   for (i = 0; i < 10; i++){
    //     printf("%u,%u\n", channel_input[j][i], channel_input_optim_multi[0][j][i]);
    //   }
    // }

    // ==dec here ==    
    if (ntrials==1)    
      for (j=0;j<n_segments;j++)
        for (i = 0; i < block_length+(nrows-no_punctured_columns) * Zc - removed_bit; i++)
          if (channel_input[j][i]!=channel_input_optim[j][i]) {
            printf("[ERROR][Single]differ in seg %u pos %u (%u,%u)\n", j, i, channel_input[j][i], channel_input_optim[j][i]);
            free(channel_output);
            return (-1);
          }
	//[START]Check the value of channel_input and channel_input_multi
    if((n_segments != 15)||(thread_num != 2)){  //if not our situation
      if (ntrials==1)
        for(int th=0;th<thread_num;th++)    
          for (j=0;j<n_segments;j++)
            for (i = 0; i < block_length+(nrows-no_punctured_columns) * Zc - removed_bit; i++)
              if (channel_input[j][i]!=channel_input_optim_multi[0][j][i]) {
                printf("[ERROR][Multi(%d)]differ in seg %u pos %u (%u,%u)\n", th, j, i, channel_input[j][i], channel_input_optim_multi[th][j][i]);
                free(channel_output);
                return (-1);
              }
      printf("============ [OK]Result compare ===========\n");
    }else{
      printf("=================== [OK] ==================\n");
    }
    
    
    //[END]Check the value of channel_input and channel_input_multi																 
      //else{
           // printf("NOT differ in seg %d pos %d (%d,%d)\n",j,i,channel_input[j][i],channel_input_optim[j][i]);
     // }
    if (trial== 0) {
      printf("nrows: %d\n", nrows);
      printf("no_punctured_columns: %d\n", no_punctured_columns);
      printf("removed_bit: %d\n", removed_bit);
      printf("To: %d\n", (Kb+nrows-no_punctured_columns) * Zc-removed_bit);
      printf("number of undecoded bits: %d\n", (Kb+nrows-no_punctured_columns-2) * Zc-removed_bit);
    }

    //print_meas_now(&time, "", stdout);

    // for (i=0;i<6400;i++)
    //printf("channel_input[%d]=%d\n",i,channel_input[i]);
    //printf("%d ",channel_input[i]);

    //if ((BG==2) && (Zc==128||Zc==256))
    if (1) { // Transmitting one segment 
      for(j=0;j<n_segments;j++) {
	for (i = 2*Zc; i < (Kb+nrows-no_punctured_columns) * Zc-removed_bit; i++) {
#ifdef DEBUG_CODER
        if ((i&0xf)==0)
          printf("\ne %u..%u:    ",i,i+15);
#endif
		// ==dec here ==
        if (channel_input_optim[j][i-2*Zc]==0)
          modulated_input[j][i]=1.0;///sqrt(2);  //QPSK
        else
          modulated_input[j][i]=-1.0;///sqrt(2);

		// if (channel_input_optim_multi[0][i-2*Zc]==0)
        //   modulated_input[j][i]=1.0;///sqrt(2);  //QPSK
        // else
        //   modulated_input[j][i]=-1.0;///sqrt(2);
											   
        ///channel_output[i] = modulated_input[i] + gaussdouble(0.0,1.0) * 1/sqrt(2*SNR);
        //channel_output_fixed[i] = (char) ((channel_output[i]*128)<0?(channel_output[i]*128-0.5):(channel_output[i]*128+0.5)); //fixed point 9-7
        //printf("llr[%d]=%d\n",i,channel_output_fixed[i]);

        //channel_output_fixed[i] = (char)quantize(sigma/4.0,(2.0*modulated_input[i]) - 1.0 + sigma*gaussdouble(0.0,1.0),qbits);
        channel_output_fixed[j][i] = (char)quantize(sigma/4.0/4.0,modulated_input[j][i] + sigma*gaussdouble(0.0,1.0),qbits);
        //channel_output_fixed[i] = (char)quantize8bit(sigma/4.0,(2.0*modulated_input[i]) - 1.0 + sigma*gaussdouble(0.0,1.0));
        //printf("llr[%d]=%d\n",i,channel_output_fixed[i]);
        //printf("channel_output_fixed[%d]: %d\n",i,channel_output_fixed[i]);


        //channel_output_fixed[i] = (char)quantize(1,channel_output_fixed[i],qbits);

        //Uncoded BER
        if (channel_output_fixed[j][i]<0)
            channel_output_uncoded[j][i]=1;  //QPSK demod
        else
            channel_output_uncoded[j][i]=0;
// ==dec here ==
        if (channel_output_uncoded[j][i] != channel_input_optim[j][i-2*Zc])
	  *errors_bit_uncoded = (*errors_bit_uncoded) + 1;
	//     if (channel_output_uncoded[j][i] != channel_input_optim_multi[0][j][i-2*Zc])
    // *errors_bit_uncoded = (*errors_bit_uncoded) + 1;
	}
      } // End segments

      //for (i=(Kb+nrows) * Zc-5;i<(Kb+nrows) * Zc;i++)
      //{
      //  printf("channel_input[%d]=%d\n",i,channel_input[i]);
      //printf("%lf %d\n",channel_output[i], channel_output_fixed[i]);
      //printf("v[%d]=%lf\n",i,modulated_input[i]);}
#ifdef DEBUG_CODER
      printf("\n");
      exit(-1);
#endif

      decParams.BG=BG;
      decParams.Z=Zc;
      decParams.R=code_rate_vec[R_ind];//13;
      decParams.numMaxIter=No_iteration;
      decParams.outMode = nrLDPC_outMode_BIT;
      //decParams.outMode =nrLDPC_outMode_LLRINT8;


      for(j=0;j<n_segments;j++) {
    	  start_meas(time_decoder);
      // decode the sequence
      // decoder supports BG2, Z=128 & 256
      //esimated_output=ldpc_decoder(channel_output_fixed, block_length, No_iteration, (double)((float)nom_rate/(float)denom_rate));
      ///nrLDPC_decoder(&decParams, channel_output_fixed, estimated_output, NULL);
          n_iter = nrLDPC_decoder(&decParams, (int8_t*)channel_output_fixed[j], (int8_t*)estimated_output[j], p_nrLDPC_procBuf, p_decoder_profiler);
      
	stop_meas(time_decoder);
      }

      //for (i=(Kb+nrows) * Zc-5;i<(Kb+nrows) * Zc;i++)
      //  printf("esimated_output[%d]=%d\n",i,esimated_output[i]);

      //count errors
      for(j=0;j<n_segments;j++) {
      for (i=0; i<block_length>>3; i++)
      {
          //printf("block_length>>3: %d \n",block_length>>3);
         /// printf("i: %d \n",i);
          ///printf("estimated_output[%d]: %d \n",i,estimated_output[i]);
          ///printf("test_input[0][%d]: %d \n",i,test_input[0][i]);
	// ==dec here ==				  
        if (estimated_output[j][i] != test_input[j][i])
        {
      //////printf("error pos %d (%d, %d)\n\n",i,estimated_output[i],test_input[0][i]);
          segment_bler = segment_bler + 1;
          break;
        }
	  //   if (estimated_output[j][i] != test_input_multi[0][j][i])
      //   {
      // //////printf("error pos %d (%d, %d)\n\n",i,estimated_output[i],test_input[0][i]);
      //     segment_bler = segment_bler + 1;
      //     break;
      //   }
      }

      for (i=0; i<block_length; i++)
        {
          estimated_output_bit[j][i] = (estimated_output[j][i/8]&(1<<(i&7)))>>(i&7);
		  // ==dec here ==			  
          test_input_bit[i] = (test_input[j][i/8]&(1<<(i&7)))>>(i&7); // Further correct for multiple segments
		  // test_input_bit[i] = (test_input_multi[0][j][i/8]&(1<<(i&7)))>>(i&7); // Further correct for multiple segments																											  
          if (estimated_output_bit[j][i] != test_input_bit[i])
          {
            *errors_bit = (*errors_bit) + 1;
          }
        }

      //if (*errors == 1000)
    	  //break;

      n_iter_mean =  n_iter_mean + n_iter;
      n_iter_std = n_iter_std + pow(n_iter-1,2);

      if ( n_iter > n_iter_max )
        n_iter_max = n_iter;

    } // end segments

      if (segment_bler != 0)
		*errors = (*errors) + 1;

    }
    /*else if (trial==0)
      printf("decoder is not supported\n");*/
  }


  dec_iter->n_iter_mean = n_iter_mean/(double)ntrials/(double)n_segments - 1;
  dec_iter->n_iter_std = sqrt(n_iter_std/(double)ntrials/(double)n_segments - pow(n_iter_mean/(double)ntrials/(double)n_segments - 1,2));
  dec_iter->n_iter_max = n_iter_max -1;

  *errors_bit_uncoded = *errors_bit_uncoded / (double)((Kb+nrows-no_punctured_columns-2) * Zc-removed_bit);

  for(j=0;j<MAX_NUM_DLSCH_SEGMENTS;j++) {
    free(test_input[j]);
    free(channel_input[j]);
    free(channel_output_uncoded[j]);
    free(channel_input_optim[j]);
	for(int th = 0;th<thread_num;th++){// ==For multi_ldpc_enc ==
      free(test_input_multi[th][j]);
      free(channel_input_optim_multi[th][j]);
    }															 
    free(modulated_input[j]);
    free(channel_output_fixed[j]);
    free(estimated_output[j]);
    free(estimated_output_bit[j]);
  }
  //free(modulated_input);
  free(channel_output);
  //free(channel_output_fixed);
  //free(estimated_output);

  nrLDPC_free_mem(p_nrLDPC_procBuf);

  print_meas(&time,"ldpc_encoder",NULL,NULL);
  print_meas(time_optim,"ldpc_encoder_optim",NULL,NULL);
  print_meas(&tinput,"ldpc_encoder_optim(input)",NULL,NULL);
  print_meas(&tprep,"ldpc_encoder_optim(prep)",NULL,NULL);
  print_meas(&tparity,"ldpc_encoder_optim(parity)",NULL,NULL);
  print_meas(&toutput,"ldpc_encoder_optim(output)",NULL,NULL);
  printf("\n");
  print_meas(time_decoder,"ldpc_decoder",NULL,NULL);
  print_meas(&decoder_profiler.llr2llrProcBuf,"llr2llrProcBuf",NULL,NULL);
  print_meas(&decoder_profiler.llr2CnProcBuf,"llr2CnProcBuf",NULL,NULL);
  print_meas(&decoder_profiler.cnProc,"cnProc (per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.cnProcPc,"cnProcPc (per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.bnProc,"bnProc (per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.bnProcPc,"bnProcPc(per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.cn2bnProcBuf,"cn2bnProcBuf (per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.bn2cnProcBuf,"bn2cnProcBuf (per iteration)",NULL,NULL);
  print_meas(&decoder_profiler.llrRes2llrOut,"llrRes2llrOut",NULL,NULL);
  print_meas(&decoder_profiler.llr2bit,"llr2bit",NULL,NULL);
  printf("\n");

  return *errors;
}

int main(int argc, char *argv[])
{
  unsigned int errors, errors_bit, crc_misses;
  double errors_bit_uncoded;
  short block_length=8448; // decoder supports length: 1201 -> 1280, 2401 -> 2560

  short No_iteration=5;
  //int n_segments=1;
  int n_segments=15;  //Set n_segments for test
  //double rate=0.333;
  int nom_rate=1;
  int denom_rate=3;
  double SNR0=-2.0,SNR,SNR_lin;
  unsigned char qbits=8;
  unsigned int decoded_errors[10000]; // initiate the size of matrix equivalent to size of SNR
  int c,i=0, i1 = 0;

  int n_trials = 1;
  double SNR_step = 0.1;

  randominit(0);
  int test_uncoded= 0;

  time_stats_t time_optim[10], time_decoder[10];
  n_iter_stats_t dec_iter[3];

  short BG=0,Zc,Kb;	

  while ((c = getopt (argc, argv, "q:r:R:s:S:l:n:d:i:t:T:V:u:h")) != -1)
    switch (c)
    {
      case 'q':
        qbits = atoi(optarg);
        break;

      case 'r':
        nom_rate = atoi(optarg);
        break;

	  case 'R':
        ifRand = atoi(optarg);
        if(ifRand)
          printf("[WARNING]It will cause ERROR when comparing results\n");
        break;
	   
      case 'd':
        denom_rate = atoi(optarg);
        break;

      case 'l':
        block_length = atoi(optarg);
        break;

      case 'n':
        n_trials = atoi(optarg);
        break;

      case 's':
        SNR0 = atof(optarg);
        break;

      case 'S':
        n_segments = atof(optarg);
        break;

      case 't':
        SNR_step = atof(optarg);
        break;

	  case 'T':
        thread_num = atoi(optarg);
        if(thread_num > 8){
          thread_num = 8;
          printf("[WARNING]Too many threads !!! Set thread_num 8\n");
        }else if(thread_num < 0){
          thread_num = 0;
          printf("[WARNING]Too less threads !!! Set thread_num 0\n");
        }
        break;

      case 'V':
        vcd = atoi(optarg);
        break;
	   
      case 'i':
        No_iteration = atoi(optarg);
        break;

      case 'u':
        test_uncoded = atoi(optarg);
        break;

      case 'h':
            default:
              printf("CURRENTLY SUPPORTED CODE RATES: \n");
              printf("BG1 (blocklength > 3840): 1/3, 2/3, 22/25 (8/9) \n");
              printf("BG2 (blocklength <= 3840): 1/5, 1/3, 2/3 \n\n");
              printf("-h This message\n");
              printf("-q Quantization bits, Default: 8\n");
              printf("-r Nominator rate, (1, 2, 22), Default: 1\n");
			        printf("-R Will test_input be different for each thread, Default: 0\n");																		  
              printf("-d Denominator rate, (3, 5, 25), Default: 1\n");
              printf("-l Block length (l > 3840 -> BG1, rest BG2 ), Default: 8448\n");
              printf("-n Number of simulation trials, Default: 1\n");
              //printf("-M MCS2 for TB 2\n");
              printf("-s SNR per information bit (EbNo) in dB, Default: -2\n");
              printf("-S Number of segments (Maximum: 8), Default: 1\n");
              printf("-t SNR simulation step, Default: 0.1\n");
			        printf("-T Number of threads, Default: 2\n");
              printf("-V VCD ON(1), OFF(0), Default: 0\n");										   
              printf("-i Max decoder iterations, Default: 5\n");
              printf("-u Set SNR per coded bit, Default: 0\n");
              exit(1);
              break;
    }

	// ==[START]VCD ==
  if(vcd == 1){
    T_init(2021, 1, 0);
  }
  // ==[END]VCD ==

  // ==[START]Create multi_ldpc_encoder ==    
  for(int th=0;th<thread_num;th++){
    pthread_attr_init(&ldpc_enc[th].attr);
    pthread_mutex_init(&ldpc_enc[th].mutex, NULL);
    pthread_cond_init(&ldpc_enc[th].cond, NULL);
    ldpc_enc[th].id = th;
    ldpc_enc[th].flag_wait = 1;
    ldpc_enc[th].complete = 0;
    pthread_create(&(ldpc_enc[th].pthread), &ldpc_enc[th].attr, multi_ldpc_encoder_proc, &ldpc_enc[th]);    
    printf("[CREATE] LDPC encoder thread %d \n",ldpc_enc[th].id);
  }
  // ==[END]Create multi_ldpc_encoder ==					   
  cpu_freq_GHz = get_cpu_freq_GHz();
  //printf("the decoder supports BG2, Kb=10, Z=128 & 256\n");
  //printf(" range of blocklength: 1201 -> 1280, 2401 -> 2560\n");
  printf("block length %d: \n", block_length);
  printf("n_trials %d: \n", n_trials);
  printf("SNR0 %f: \n", SNR0);




  //for (block_length=8;block_length<=MAX_BLOCK_LENGTH;block_length+=8)


  //determine number of bits in codeword
  if (block_length>3840)
  {
    BG = 1;
    Kb = 22;
    //nrows=46; //parity check bits
    //ncols=22; //info bits
  }
  else if (block_length<=3840)
  {
    BG = 2;
    //nrows=42; //parity check bits
    //ncols=10; // info bits

    if (block_length>640)
      Kb = 10;
    else if (block_length>560)
      Kb = 9;
    else if (block_length>192)
      Kb = 8;
    else
      Kb = 6;
      }

  //find minimum value in all sets of lifting size
  Zc=0;

  for (i1=0; i1 < 51; i1++)
  {
    if (lift_size[i1] >= (double) block_length/Kb)
    {
      Zc = lift_size[i1];
      //printf("%d\n",Zc);
      break;
    }
  }

  char fname[200];
  sprintf(fname,"ldpctest_BG_%d_Zc_%d_rate_%d-%d_block_length_%d_maxit_%d.txt",BG,Zc,nom_rate,denom_rate,block_length, No_iteration);
  FILE *fd=fopen(fname,"w");
  AssertFatal(fd!=NULL,"cannot open %s\n",fname);

  fprintf(fd,"SNR BLER BER UNCODED_BER ENCODER_MEAN ENCODER_STD ENCODER_MAX DECODER_TIME_MEAN DECODER_TIME_STD DECODER_TIME_MAX DECODER_ITER_MEAN DECODER_ITER_STD DECODER_ITER_MAX\n");

  for (SNR=SNR0;SNR<SNR0+20.0;SNR+=SNR_step) {
	  //reset_meas(&time_optim);
	  //reset_meas(&time_decoder);
	  //n_iter_stats_t dec_iter = {0, 0, 0};
    if (test_uncoded == 1)
    	SNR_lin = pow(10,SNR/10.0);
    else
    	SNR_lin = pow(10,SNR/10.0)*nom_rate/denom_rate;
    printf("Linear SNR: %f\n", SNR_lin);
    decoded_errors[i]=test_ldpc(No_iteration,
                                nom_rate,
                                denom_rate,
                                SNR_lin,   // noise standard deviation
                                qbits,
                                block_length,   // block length bytes
                                n_trials,
                                n_segments,
                                &errors,
                                &errors_bit,
                                &errors_bit_uncoded,
                                &crc_misses,
                                time_optim,
                                time_decoder,
                                dec_iter);

    printf("SNR %f, BLER %f (%u/%d)\n", SNR, (float)decoded_errors[i]/(float)n_trials, decoded_errors[i], n_trials);
    printf("SNR %f, BER %f (%u/%d)\n", SNR, (float)errors_bit/(float)n_trials/(float)block_length/(double)n_segments, decoded_errors[i], n_trials);
    printf("SNR %f, Uncoded BER %f (%u/%d)\n",SNR, errors_bit_uncoded/(float)n_trials/(double)n_segments, decoded_errors[i], n_trials);
    printf("SNR %f, Mean iterations: %f\n",SNR, dec_iter->n_iter_mean);
    printf("SNR %f, Std iterations: %f\n",SNR, dec_iter->n_iter_std);
    printf("SNR %f, Max iterations: %d\n",SNR, dec_iter->n_iter_max);
    printf("\n");
    printf("Encoding time mean: %15.3f us\n",(double)time_optim->diff/time_optim->trials/1000.0/cpu_freq_GHz);
    printf("Encoding time std: %15.3f us\n",sqrt((double)time_optim->diff_square/time_optim->trials/pow(1000,2)/pow(cpu_freq_GHz,2)-pow((double)time_optim->diff/time_optim->trials/1000.0/cpu_freq_GHz,2)));
    printf("Encoding time max: %15.3f us\n",(double)time_optim->max/1000.0/cpu_freq_GHz);
    printf("\n");
    printf("Decoding time mean: %15.3f us\n",(double)time_decoder->diff/time_decoder->trials/1000.0/cpu_freq_GHz);
    printf("Decoding time std: %15.3f us\n",sqrt((double)time_decoder->diff_square/time_decoder->trials/pow(1000,2)/pow(cpu_freq_GHz,2)-pow((double)time_decoder->diff/time_decoder->trials/1000.0/cpu_freq_GHz,2)));
    printf("Decoding time max: %15.3f us\n",(double)time_decoder->max/1000.0/cpu_freq_GHz);

    fprintf(fd,"%f %f %f %f %f %f %f %f %f %f %f %f %d \n",
    		SNR,
    		(double)decoded_errors[i]/(double)n_trials ,
    		(double)errors_bit/(double)n_trials/(double)block_length/(double)n_segments ,
    		errors_bit_uncoded/(double)n_trials/(double)n_segments ,
    		(double)time_optim->diff/time_optim->trials/1000.0/cpu_freq_GHz,
    		sqrt((double)time_optim->diff_square/time_optim->trials/pow(1000,2)/pow(cpu_freq_GHz,2)-pow((double)time_optim->diff/time_optim->trials/1000.0/cpu_freq_GHz,2)),
    		(double)time_optim->max/1000.0/cpu_freq_GHz,
    		(double)time_decoder->diff/time_decoder->trials/1000.0/cpu_freq_GHz,
    		sqrt((double)time_decoder->diff_square/time_decoder->trials/pow(1000,2)/pow(cpu_freq_GHz,2)-pow((double)time_decoder->diff/time_decoder->trials/1000.0/cpu_freq_GHz,2)),
    		(double)time_decoder->max/1000.0/cpu_freq_GHz,
    		dec_iter->n_iter_mean,
    		dec_iter->n_iter_std,
    		dec_iter->n_iter_max
    		);

    if (decoded_errors[i] == 0) break;

    i=i+1;
  }
  fclose(fd);

  return(0);
}
