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

#include "Gen_shift_value.h"
#include "choose_generator_matrix.h"
#include "ldpc_encoder_header.h"

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

#define MAX_BLOCK_LENGTH 8448

int test_ldpc(short No_iteration,
	      double rate,	     
	      double SNR,
	      unsigned char qbits,
	      short block_length,
	      unsigned int ntrials,
	      unsigned int *errors, 	      
	      unsigned int *crc_misses)
{

//clock initiate
 time_stats_t time;
 opp_enabled=1;
 cpu_freq_GHz = get_cpu_freq_GHz();
 
  //short test_input[block_length];
  short *test_input;
  //short *c; //padded codeword  
  short *esimated_output;
  short *channel_input;
  double *channel_output;
  double *modulated_input;
  short *channel_output_fixed;
  unsigned int i,trial=0;   

  /*
short BG,Zc,Kb,nrows,ncols,channel_temp;
int no_punctured_columns;  //new

//Table of possible lifting sizes
  short lift_size[51]={2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};
  */
  
  *errors=0;
  *crc_misses=0;

  while (trial++ < ntrials) 
   {        
     // generate input block
	 test_input=(short*)malloc(sizeof(short) * block_length);
     for (i=0; i<block_length; i++) 
       {    
          test_input[i]=rand()%2;
          //test_input[i]=i%2;
       }
        
/*
     //determine number of bits in codeword
     if (block_length>3840)
       {
          BG=1;
	  Kb = 22;
          nrows=46;	//parity check bits
          ncols=22;	//info bits
       }
     else if (block_length<=3840)
       {
          BG=2;
	  nrows=42;	//parity check bits
          ncols=10;	// info bits
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
     for (i1=0; i1 < 51; i1++)
       {
          if (lift_size[i1] >= (double) block_length/Kb)
            {
              Zc = lift_size[i1];
              // printf("%d\n",Zc);
              break;
            }
       }

     no_punctured_columns=(int)((nrows+Kb-2)*Zc-block_length/rate)/Zc; 
//printf("%d\n",no_punctured_columns);
	*/
    start_meas(&time);
    //// encoder
   
    channel_input=ldpc_encoder_header(test_input, block_length,rate);
    stop_meas(&time);
    print_meas_now(&time, "", stdout);

 //for (i=0;i<10;i++)
   //printf("channel_input[%d]=%d test_input[%d]=%d\n",i,channel_input[i],i,test_input[i]);
 /*
     //channel 
     modulated_input  = (double *)malloc( (Kb+nrows) * Zc*sizeof(double));   
     channel_output  = (double *)malloc( (Kb+nrows) * Zc*sizeof(double));
     channel_output_fixed  = (short *)malloc( (Kb+nrows) * Zc*sizeof(short));
     memset(channel_output_fixed,0,(Kb+nrows) * Zc*sizeof(short));

     for (i = 2*Zc; i < (Kb+nrows) * Zc; i++) 
      {
          #ifdef DEBUG_CODER
          if ((i&0xf)==0)
          printf("\ne %d..%d:    ",i,i+15);
          #endif

	  if (channel_input[i]==0)
	     modulated_input[i]=1/sqrt(2);	//QPSK
	  else
	     modulated_input[i]=-1/sqrt(2);
	  channel_output[i] = modulated_input[i] + gaussdouble(0.0,1.0) * 1/sqrt(2*SNR);
          channel_output_fixed[i] = (short) ((channel_output[i]*128)<0?(channel_output[i]*128-0.5):(channel_output[i]*128+0.5)); //fixed point 9-7
       }

//for (i=(Kb+nrows) * Zc-5;i<(Kb+nrows) * Zc;i++)
//{
 //  printf("channel_input[%d]=%d\n",i,channel_input[i]);
//printf("%lf %d\n",channel_output[i], channel_output_fixed[i]);
//printf("v[%d]=%lf\n",i,modulated_input[i]);}

#ifdef DEBUG_CODER
    printf("\n");
    exit(-1);
#endif
/*
     // decode the sequence
     esimated_output=ldpc_decoder(channel_output_fixed, block_length, No_iteration, rate);

//for (i=(Kb+nrows) * Zc-5;i<(Kb+nrows) * Zc;i++)
 //  printf("esimated_output[%d]=%d\n",i,esimated_output[i]);  
  
     //count errors
     for (i=0;i<(Kb+nrows) * Zc;i++) 
       {
          if (esimated_output[i] != channel_input[i]) 
            {
               *errors = (*errors) + 1;
               break;
            }

       }  
    
     free(channel_input);
     free(modulated_input);  
     free(channel_output); 
     free(channel_output_fixed); 
*/
   }
  
 return *errors;

}

//#define NTRIALS 10000
#define NTRIALS 30

int main(int argc, char *argv[])
{
  
  unsigned int errors,crc_misses;
  short block_length=22*384;
  short No_iteration=25;
  double rate=0.667;
  double SNR,SNR_lin;
  unsigned char qbits;  
  
  int i=0;

  randominit(0);  

  if (argc>1)
    qbits = atoi(argv[1]);
  else
    qbits = 4;  

  unsigned int decoded_errors[100]; // initiate the size of matrix equivalent to size of SNR
  for (SNR=-2.1; SNR<-2; SNR+=.1) 
   {

       SNR_lin = pow(10,SNR/10);
  
       decoded_errors[i]=test_ldpc(No_iteration,
		      rate,		     
		      SNR_lin,   // noise standard deviation
		      qbits,
		      block_length,   // block length bytes
		      NTRIALS,
		      &errors,
		      &crc_misses);

       printf("SNR %f, BLER %f (%d/%d)\n",SNR,(float)decoded_errors[i]/(float)NTRIALS,decoded_errors[i],NTRIALS);

       i=i+1;   
   }

  return(0);

}


