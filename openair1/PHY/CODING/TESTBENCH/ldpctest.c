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

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "SIMULATION/TOOLS/defs.h"

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

char quantize(double D,double x,unsigned char B)
{

  double qxd;
  char maxlev;

  qxd = floor(x/D);
  //    printf("x=%f,qxd=%f\n",x,qxd);

  maxlev = 1<<(B-1);

  if (qxd <= -maxlev)
    qxd = -maxlev;
  else if (qxd >= maxlev)
    qxd = maxlev-1;

  return((char)qxd);
}

#define MAX_BLOCK_LENGTH 6000

int test_ldpc(unsigned int coded_bits,
	      double sigma,
	      unsigned char qbits,
	      unsigned int block_length,
	      unsigned int ntrials,
	      unsigned int *errors,
	      unsigned int *trials,
	      unsigned int *uerrors,
	      unsigned int *crc_misses,
	      unsigned int *iterations)
{

  unsigned char test_input[block_length+1];
  unsigned char decoded_output[block_length];
  short *channel_input, *channel_output;
  unsigned int i,trial=0;
  unsigned int crc=0;
  unsigned char ret;
  unsigned char uerr;
  unsigned char crc_type;

  channel_input  = (short *)malloc(coded_bits*sizeof(short));
  channel_output = (short *)malloc(coded_bits*sizeof(short));

  *iterations=0;
  *errors=0;
  *crc_misses=0;
  *uerrors=0;


  while (trial++ < ntrials) {

    //    printf("encoding\n");
    //    test_input[0] = 0x80;
    for (i=0; i<block_length; i++) {

      test_input[i] = (unsigned char)(taus()&0xff);
    }

    //replace with ldpc encoder, write to channel_input
    /*
    dlsch_encoding(test_input,
                   &PHY_vars_eNB->lte_frame_parms,
                   num_pdcch_symbols,
                   PHY_vars_eNB->dlsch_eNB[0][0],
                   0,
                   subframe,
                   &PHY_vars_eNB->dlsch_rate_matching_stats,
                   &PHY_vars_eNB->dlsch_turbo_encoding_stats,
                   &PHY_vars_eNB->dlsch_interleaving_stats);
    */
    uerr=0;


    for (i = 0; i < coded_bits; i++) {
#ifdef DEBUG_CODER
      if ((i&0xf)==0)
        printf("\ne %d..%d:    ",i,i+15);
#endif
      channel_output[i] = (short)quantize(sigma/4.0,(2.0*channel_input[i]) - 1.0 + sigma*gaussdouble(0.0,1.0),qbits);
    }

#ifdef DEBUG_CODER
    printf("\n");
    exit(-1);
#endif

    // replace this with ldpc decoder
    /*
    ret = dlsch_decoding(PHY_vars_UE,
                         channel_output,
                         &PHY_vars_UE->lte_frame_parms,
                         PHY_vars_UE->dlsch_ue[0][0],
                         PHY_vars_UE->dlsch_ue[0][0]->harq_processes[PHY_vars_UE->dlsch_ue[0][0]->current_harq_pid],
                         frame,
                         subframe,
                         PHY_vars_UE->dlsch_ue[0][0]->current_harq_pid,
                         num_pdcch_symbols,1);
    */

    /*
    if (ret < dlsch_ue->max_turbo_iterations+1) {
      *iterations = (*iterations) + ret;
      //      if (ret>1)
      //  printf("ret %d\n",ret);
    } else
      *iterations = (*iterations) + (ret-1);

    if (uerr==1)
      *uerrors = (*uerrors) + 1;
      */

    for (i=0; i<block_length; i++) {

      if (decoded_output[i] != test_input[i]) {
        *errors = (*errors) + 1;
        //  printf("*%d, ret %d\n",*errors,ret);

	/*
        if (ret < dlsch_ue->max_turbo_iterations+1)
          *crc_misses = (*crc_misses)+1;
	  */

        break;
      }

    }

    /*
    if (ret == dlsch_ue->max_turbo_iterations+1) {
      //      exit(-1);
    }
    */

    if (*errors == 100) {
      printf("trials %d\n",trial);
      break;
    }
  }

  *trials = trial;
  //  printf("lte: trials %d, errors %d\n",trial,*errors);
  return(0);
}

#define NTRIALS 10000

int main(int argc, char *argv[])
{

  int ret,ret2;
  unsigned int errors,uerrors,errors2,crc_misses,iterations,trials,trials2,block_length,errors3,trials3;
  double SNR,sigma,rate=.5;
  unsigned char qbits,mcs;

  char done0=0;
  char done1=1;
  char done2=1;

  unsigned int coded_bits;
  unsigned char NB_RB=25;

  int num_pdcch_symbols = 1;
  int subframe = 6;

  randominit(0);
  //logInit();

  if (argc>1)
    qbits = atoi(argv[1]);
  else
    qbits = 4;

  printf("Quantization bits %d\n",qbits);

  printf("Coded_bits (G) = %d\n",coded_bits);

  for (SNR=-5; SNR<15; SNR+=.1) {

    sigma = pow(10.0,-.05*SNR);
    printf("\n\nSNR %f dB => sigma %f\n",SNR,sigma);

    errors=0;
    crc_misses=0;
    errors2=0;
    errors3=0;

    iterations=0;

    if (done0 == 0) {



      ret = test_ldpc(coded_bits,
		      sigma,   // noise standard deviation
		      qbits,
		      block_length,   // block length bytes
		      NTRIALS,
		      &errors,
		      &trials,
		      &uerrors,
		      &crc_misses,
		      &iterations);

      if (ret>=0)
        printf("%f,%f,%f,%f\n",SNR,(double)errors/trials,(double)crc_misses/trials,(double)iterations/trials);

      if (((double)errors/trials) < 1e-2)
        done0=1;
    }

    if ((done0==1) && (done1==1) && (done2==1)) {
      printf("done\n");
      break;
    }
  }

  return(0);
}


