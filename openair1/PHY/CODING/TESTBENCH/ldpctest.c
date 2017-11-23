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
	      unsigned int block_length,
	      unsigned int ntrials,
	      unsigned int *errors, 	      
	      unsigned int *crc_misses)
{

//clock initiate
 time_stats_t time;
 opp_enabled=1;
 cpu_freq_GHz = get_cpu_freq_GHz();
 
  short test_input[block_length];
  short *c; //padded codeword  
  short *esimated_output;
  short *channel_input;
  double *channel_output;
  double *modulated_input;
  short *channel_output_fixed;
  unsigned int i,trial=0;   

short *Gen_shift_values, *no_shift_values, *pointer_shift_values;
short BG,Zc,Kb,nrows,ncols;
int i1,i2,i3,i4,i5,i6,t,temp,k;

//Table of possible lifting sizes
  short lift_size[51]={2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,144,160,176,192,208,224,240,256,288,320,352,384};
  
  *errors=0;
  *crc_misses=0;

  while (trial++ < ntrials) {
        
// generate input block
for (i=0; i<block_length; i++) {

      //test_input[i] = (unsigned char)(taus()&0xff);
      test_input[i]=rand()%2;
//test_input[i]=i%2;
    }
  start_meas(&time);   

//determine number of bits in codeword
    if (block_length>3840)
     {
         BG=1;
	 Kb = 22;
         nrows=46;
         ncols=22;
     }
     else if (block_length<=3840)
     {
         BG=2;
	 nrows=42;
         ncols=10;
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

// load base graph of generator matrix
   if (BG==1)
   {
       no_shift_values=(short*) no_shift_values_BG1;
       pointer_shift_values=(short*) pointer_shift_values_BG1;
       if (Zc==2||Zc==4||Zc==8||Zc==16||Zc==32||Zc==64||Zc==128||Zc==256)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_2;
      // else if (Zc==3||Zc==6||Zc==12||Zc==24||Zc==48||Zc==96||Zc==192||Zc==384)
            //Gen_shift_values=(short*) Gen_shift_values_BG1_a_3;
       else if (Zc==384)
            Gen_shift_values=(short*) Gen_shift_values_BG1_Z_384;
       else if (Zc==5||Zc==10||Zc==20||Zc==40||Zc==80||Zc==160||Zc==320)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_5;
       else if (Zc==7||Zc==14||Zc==28||Zc==56||Zc==112||Zc==224)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_7;
       else if (Zc==9||Zc==18||Zc==36||Zc==72||Zc==144||Zc==288)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_9;
       else if (Zc==11||Zc==22||Zc==44||Zc==88||Zc==176||Zc==352)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_11;
       else if (Zc==13||Zc==26||Zc==52||Zc==104||Zc==208)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_13;
       else if (Zc==15||Zc==30||Zc==60||Zc==120||Zc==240)
            Gen_shift_values=(short*) Gen_shift_values_BG1_a_15;
   }

   else if (BG==2)
   {       
       no_shift_values=(short*) no_shift_values_BG2;
       pointer_shift_values=(short*) pointer_shift_values_BG2;
       //if (Zc==2||Zc==4||Zc==8||Zc==16||Zc==32||Zc==64||Zc==128||Zc==256)
           // Gen_shift_values=(short*) Gen_shift_values_BG2_a_2;
	if (Zc==128)
            Gen_shift_values=(short*) Gen_shift_values_BG2_Z_128;
       else if (Zc==3||Zc==6||Zc==12||Zc==24||Zc==48||Zc==96||Zc==192||Zc==384)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_3;
       else if (Zc==5||Zc==10||Zc==20||Zc==40||Zc==80||Zc==160||Zc==320)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_5;
       else if (Zc==7||Zc==14||Zc==28||Zc==56||Zc==112||Zc==224)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_7;
       else if (Zc==9||Zc==18||Zc==36||Zc==72||Zc==144||Zc==288)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_9;
       else if (Zc==11||Zc==22||Zc==44||Zc==88||Zc==176||Zc==352)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_11;
       else if (Zc==13||Zc==26||Zc==52||Zc==104||Zc==208)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_13;
       else if (Zc==15||Zc==30||Zc==60||Zc==120||Zc==240)
            Gen_shift_values=(short*) Gen_shift_values_BG2_a_15;
   }
  c=(short *)malloc(sizeof(short) * Kb * Zc);
  channel_input  = (short *)malloc( (Kb+nrows) * Zc *sizeof(short)); 

//padded input sequence   
   memset(c,0,sizeof(short) * Kb * Zc);
   memcpy(c,test_input,block_length * sizeof(short));
//start_meas(&time);
//encode the input sequence
   memset(channel_input,0,(Kb+nrows) * Zc*sizeof(short));

     // parity check part
   
	 for (i2=0; i2 < Zc; i2++)
	 {
		 t=Kb*Zc+i2;
		 //rotate matrix here
		 for (i5=0; i5 < Kb; i5++)
		 {
			 temp = c[i5*Zc];
			memmove(&c[i5*Zc], &c[i5*Zc+1], (Zc-1)*sizeof(short));
			c[i5*Zc+Zc-1] = temp;
			 //for (i6 = 0; i6 < Zc-1; i6++)
			//for (i6 = i5*Zc; i6 < i5*Zc + Zc-1; i6++)
             //{
				//c[i5*Zc+i6] = c[i5*Zc+i6+1];
				//c[i6] = c[i6+1];
             
             //}
			 //c[i5*Zc+i6] = temp;
			//c[i6] = temp;             
		 }            
        
		for (i1=0; i1 < nrows; i1++)
		{
		for (i3=0; i3 < Kb; i3++)
            	{
                for (i4=0; i4 < no_shift_values[i1 * ncols + i3]; i4++)
                {
		channel_input[t+i1*Zc] = channel_input[t+i1*Zc] + c[ i3*Zc + Gen_shift_values[ pointer_shift_values[i1 * ncols + i3]+i4 ] ];
		}
		}
			channel_input[t+i1*Zc]=channel_input[t+i1*Zc]&1;
		
		}
          
/*		
for (i1=0; i1 < nrows; i1++)
		{
		k=i1*Zc;
		for (i3=0; i3 < Kb; i3++)
            	{
                for (i4=0; i4 < no_shift_values[i1 * ncols + i3]; i4++)
                {
		channel_input[t+k] = channel_input[t+k] + c[ i3*Zc + Gen_shift_values[ i4] ];
		}
		}
			channel_input[t+k]=channel_input[t+k]&1;
		
		}
*/
	 }
//stop_meas(&time);
     // information part
     memcpy(channel_input,c,Kb*Zc*sizeof(short));
 stop_meas(&time);

   print_meas_now(&time, "", stdout);
 //for (i=(Kb+nrows) * Zc-10;i<(Kb+nrows) * Zc;i++)
  // printf("channel_input[%d]=%d\n",i,channel_input[i]);
 
  //channel 
modulated_input  = (double *)malloc( (Kb+nrows) * Zc*sizeof(double)); 
  //channel_output = (short *)malloc( (Kb+nrows) * Zc*sizeof(short));
channel_output  = (double *)malloc( (Kb+nrows) * Zc*sizeof(double));
channel_output_fixed  = (short *)malloc( (Kb+nrows) * Zc*sizeof(short));
memset(channel_output_fixed,0,(Kb+nrows) * Zc*sizeof(short));
for (i = 2*Zc; i < (Kb+nrows) * Zc; i++) {
#ifdef DEBUG_CODER
      if ((i&0xf)==0)
        printf("\ne %d..%d:    ",i,i+15);
#endif
      //channel_output[i] = (short)quantize(sigma/4.0,(2.0*channel_input[i]) - 1.0 + sigma*gaussdouble(0.0,1.0),qbits);
	if (channel_input[i]==0)
		modulated_input[i]=1/sqrt(2);	//QPSK
	else
		modulated_input[i]=-1/sqrt(2);
	channel_output[i] = modulated_input[i] + gaussdouble(0.0,1.0) * 1/sqrt(2*SNR);
channel_output_fixed[i] = (short) ((channel_output[i]*128)<0?(channel_output[i]*128-0.5):(channel_output[i]*128+0.5)); //fixed point 9-7
//printf("%lf %d\n",channel_output[i], channel_output_fixed[i]);
//printf("v[%d]=%lf\n",i,modulated_input[i]);
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
esimated_output=ldpc_decoder(channel_output_fixed, block_length, No_iteration, rate);
//for (i=(Kb+nrows) * Zc-5;i<(Kb+nrows) * Zc;i++)
 //  printf("esimated_output[%d]=%d\n",i,esimated_output[i]);    

    for (i=0;i<(Kb+nrows) * Zc;i++) {

      if (esimated_output[i] != channel_input[i]) {
        *errors = (*errors) + 1;
        break;
      }

    }  
*/
 free(c);
free(channel_input);
free(modulated_input);  
free(channel_output); 
free(channel_output_fixed);  
  }
//printf("%d\n",*errors);
  
 return *errors;

}

//#define NTRIALS 10000
#define NTRIALS 30

int main(int argc, char *argv[])
{
  
  unsigned int errors,crc_misses;
  unsigned int block_length=22*384;
  short No_iteration=25;
  double rate=0.2;
  double SNR,SNR_lin;
  unsigned char qbits;
  //time_stats_t time;
  
  int i=0;

 //opp_enabled=1;
 // cpu_freq_GHz = get_cpu_freq_GHz();

  randominit(0);
  //logInit();

  if (argc>1)
    qbits = atoi(argv[1]);
  else
    qbits = 4;

  //printf("Quantization bits %d\n",qbits);

  unsigned int decoded_errors[100]; // initiate the size of matrix equivalent to
  // size of SNR
  for (SNR=-2.1; SNR<-2; SNR+=.1) {

    SNR_lin = pow(10,SNR/10);
    
/*
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
*/

  //  start_meas(&time);
    decoded_errors[i]=test_ldpc(No_iteration,
		      rate,		     
		      SNR_lin,   // noise standard deviation
		      qbits,
		      block_length,   // block length bytes
		      NTRIALS,
		      &errors,
		      &crc_misses);
   // stop_meas(&time);

    //print_meas_now(&time, "", stdout);

    printf("SNR %f, BLER %f (%d/%d)\n",SNR,(float)decoded_errors[i]/(float)NTRIALS,decoded_errors[i],NTRIALS);

    i=i+1;
   
  }

  return(0);
}


