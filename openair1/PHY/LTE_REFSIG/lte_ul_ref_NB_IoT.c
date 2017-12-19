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
/*! \file PHY/LTE_REFSIG/lte_ul_ref_NB_IoT.c
* \function called by lte_dl_cell_spec_NB_IoT.c ,  TS 36-211, V13.4.0 2017-02
* \author: Vincent Savaux
* \date 2017
* \version 0.0
* \company b<>com
* \email: vincent.savaux@b-com.com
* \note
* \warning
*/

#ifdef MAIN
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#endif
// #include "defs.h"
#include "PHY/LTE_REFSIG/defs_NB_IoT.h"
#include "PHY/defs_NB_IoT.h"

// uint16_t dftsizes[33] = {12,24,36,48,60,72,96,108,120,144,180,192,216,240,288,300,324,360,384,432,480,540,576,600,648,720,864,900,960,972,1080,1152,1200};

// uint16_t ref_primes[33] = {11,23,31,47,59,71,89,107,113,139,179,191,211,239,283,293,317,359,383,431,479,523,571,599,647,719,863,887,953,971,1069,1151,1193};

uint16_t sequence_length[4] = {32,3,6,12}; //the "32" value corresponds to the max gold sequence length 

// int16_t *ul_ref_sigs[30][33];
int16_t *ul_ref_sigs_rx_NB_IoT[30][4]; //these contain the sequences in repeated format and quantized to QPSK ifdef IFFT_FPGA
uint16_t u_max[4] = {16,12,14,30}; // maximum u value, see 36.211, Section 10.1.4

/* 36.211 table 5.5.1.2-1 */
char ref12_NB_IoT[360] = {-1,1,3,-3,3,3,1,1,3,1,-3,3,1,1,3,3,3,-1,1,-3,-3,1,-3,3,1,1,-3,-3,-3,-1,-3,-3,1,-3,1,-1,-1,1,1,1,1,-1,-3,-3,1,-3,3,-1,-1,3,1,-1,1,-1,-3,-1,1,-1,1,3,1,-3,3,-1,-1,1,1,-1,-1,3,-3,1,-1,3,-3,-3,-3,3,1,-1,3,3,-3,1,-3,-1,-1,-1,1,-3,3,-1,1,-3,3,1,1,-3,3,1,-1,-1,-1,1,1,3,-1,1,1,-3,-1,3,3,-1,-3,1,1,1,1,1,-1,3,-1,1,1,-3,-3,-1,-3,-3,3,-1,3,1,-1,-1,3,3,-3,1,3,1,3,3,1,-3,1,1,-3,1,1,1,-3,-3,-3,1,3,3,-3,3,-3,1,1,3,-1,-3,3,3,-3,1,-1,-3,-1,3,1,3,3,3,-1,1,3,-1,1,-3,-1,-1,1,1,3,1,-1,-3,1,3,1,-1,1,3,3,3,-1,-1,3,-1,-3,1,1,3,-3,3,-3,-3,3,1,3,-1,-3,3,1,1,-3,1,-3,-3,-1,-1,1,-3,-1,3,1,3,1,-1,-1,3,-3,-1,-3,-1,-1,-3,1,1,1,1,3,1,-1,1,-3,-1,-1,3,-1,1,-3,-3,-3,-3,-3,1,-1,-3,1,1,-3,-3,-3,-3,-1,3,-3,1,-3,3,1,1,-1,-3,-1,-3,1,-1,1,3,-1,1,1,1,3,1,3,3,-1,1,-1,-3,-3,1,1,-3,3,3,1,3,3,1,-3,-1,-1,3,1,3,-3,-3,3,-3,1,-1,-1,3,-1,-3,-3,-1,-3,-1,-3,3,1,-1,1,3,-3,-3,-1,3,-3,3,-1,3,3,-3,3,3,-1,-1,3,-3,-3,-1,-1,-3,-1,3,-3,3,1,-1};

/* 36.211 table 5.5.1.2-2 */
// char ref24[720] = {
//   -1,3,1,-3,3,-1,1,3,-3,3,1,3,-3,3,1,1,-1,1,3,-3,3,-3,-1,-3,-3,3,-3,-3,-3,1,-3,-3,3,-1,1,1,1,3,1,-1,3,-3,-3,1,3,1,1,-3,3,-1,3,3,1,1,-3,3,3,3,3,1,-1,3,-1,1,1,-1,-3,-1,-1,1,3,3,-1,-3,1,1,3,-3,1,1,-3,-1,-1,1,3,1,3,1,-1,3,1,1,-3,-1,-3,-1,-1,-1,-1,-3,-3,-1,1,1,3,3,-1,3,-1,1,-1,-3,1,-1,-3,-3,1,-3,-1,-1,-3,1,1,3,-1,1,3,1,-3,1,-3,1,1,-1,-1,3,-1,-3,3,-3,-3,-3,1,1,1,1,-1,-1,3,-3,-3,3,-3,1,-1,-1,1,-1,1,1,-1,-3,-1,1,-1,3,-1,-3,-3,3,3,-1,-1,-3,-1,3,1,3,1,3,1,1,-1,3,1,-1,1,3,-3,-1,-1,1,-3,1,3,-3,1,-1,-3,3,-3,3,-1,-1,-1,-1,1,-3,-3,-3,1,-3,-3,-3,1,-3,1,1,-3,3,3,-1,-3,-1,3,-3,3,3,3,-1,1,1,-3,1,-1,1,1,-3,1,1,-1,1,-3,-3,3,-1,3,-1,-1,-3,-3,-3,-1,-3,-3,1,-1,1,3,3,-1,1,-1,3,1,3,3,-3,-3,1,3,1,-1,-3,-3,-3,3,3,-3,3,3,-1,-3,3,-1,1,-3,1,1,3,3,1,1,1,-1,-1,1,-3,3,-1,1,1,-3,3,3,-1,-3,3,-3,-1,-3,-1,3,-1,-1,-1,-1,-3,-1,3,3,1,-1,1,3,3,3,-1,1,1,-3,1,3,-1,-3,3,-3,-3,3,1,3,1,-3,3,1,3,1,1,3,3,-1,-1,-3,1,-3,-1,3,1,1,3,-1,-1,1,-3,1,3,-3,1,-1,-3,-1,3,1,3,1,-1,-3,-3,-1,-1,-3,-3,-3,-1,-1,-3,3,-1,-1,-1,-1,1,1,-3,3,1,3,3,1,-1,1,-3,1,-3,1,1,-3,-1,1,3,-1,3,3,-1,-3,1,-1,-3,3,3,3,-1,1,1,3,-1,-3,-1,3,-1,-1,-1,1,1,1,1,1,-1,3,-1,-3,1,1,3,-3,1,-3,-1,1,1,-3,-3,3,1,1,-3,1,3,3,1,-1,-3,3,-1,3,3,3,-3,1,-1,1,-1,-3,-1,1,3,-1,3,-3,-3,-1,-3,3,-3,-3,-3,-1,-1,-3,-1,-3,3,1,3,-3,-1,3,-1,1,-1,3,-3,1,-1,-3,-3,1,1,-1,1,-1,1,-1,3,1,-3,-1,1,-1,1,-1,-1,3,3,-3,-1,1,-3,-3,-1,-3,3,1,-1,-3,-1,-3,-3,3,-3,3,-3,-1,1,3,1,-3,1,3,3,-1,-3,-1,-1,-1,-1,3,3,3,1,3,3,-3,1,3,-1,3,-1,3,3,-3,3,1,-1,3,3,1,-1,3,3,-1,-3,3,-3,-1,-1,3,-1,3,-1,-1,1,1,1,1,-1,-1,-3,-1,3,1,-1,1,-1,3,-1,3,1,1,-1,-1,-3,1,1,-3,1,3,-3,1,1,-3,-3,-1,-1,-3,-1,1,3,1,1,-3,-1,-1,-3,3,-3,3,1,-3,3,-3,1,-1,1,-3,1,1,1,-1,-3,3,3,1,1,3,-1,-3,-1,-1,-1,3,1,-3,-3,-1,3,-3,-1,-3,-1,-3,-1,-1,-3,-1,-1,1,-3,-1,-1,1,-1,-3,1,1,-3,1,-3,-3,3,1,1,-1,3,-1,-1,1,1,-1,-1,-3,-1,3,-1,3,-1,1,3,1,-1,3,1,3,-3,-3,1,-1,-1,1,3
// }; 

// 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-1 
char ref3[36] = {1,  -3,  -3, 1, -3,  -1, 1, -3,  3, 1, -1,  -1, 1, -1,  1, 1, -1,  3, 1, 1, -3, 1, 1, -1, 1, 1, 3, 1, 3, -1, 1, 3, 1, 1, 3, 3 }; 

// 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-2 
char ref6[84] = {1,  1, 1, 1, 3, -3, 1, 1, 3, 1, -3,  3, 1, -1,  -1,  -1,  1, -3, 1, -1,  3, -3,  -1,  -1, 1, 3, 1, -1,  -1,  3, 1, -3,  -3,  1, 3, 1, -1,  -1,  1, -3,  -3,  -1, 
                -1,  -1,  -1,  3, -3,  -1, 3, -1,  1, -3,  -3,  3, 3, -1,  3, -3,  -1,  1, 3, -3,  3, -1,  3, 3, -3,  1, 3, 1, -3,  -1, -3,  1, -3,  3, -3,  -1, -3,  3, -3,  1, 1, -3};

// NB-IoT: 36.211, Section 10.1.4.1.1, Table 10.1.4.1.1-1
int16_t w_n[256] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1,
  1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1,
  1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1,
  1, 1, 1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1,
  1,-1, 1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1,
  1, 1,-1,-1,-1,-1, 1, 1, 1, 1,-1,-1,-1,-1, 1, 1,
  1,-1,-1, 1,-1, 1, 1,-1, 1,-1,-1, 1,-1, 1, 1,-1,
  1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,
  1,-1, 1,-1, 1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1,
  1, 1,-1,-1, 1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1,
  1,-1,-1, 1, 1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,
  1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1, 1,
  1,-1, 1,-1,-1, 1,-1, 1,-1, 1,-1, 1, 1,-1, 1,-1,
  1, 1,-1,-1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1,-1,
  1,-1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1,-1, 1
}; 



// void generate_ul_ref_sigs(void)
// {
//   double qbar,phase;
//   unsigned int u,v,Msc_RS,q,m,n;

//   // These are the Zadoff-Chu sequences (for RB 3-100)
//   for (Msc_RS=2; Msc_RS<33; Msc_RS++) {
//     for (u=0; u<30; u++) {
//       for (v=0; v<2; v++) {
//         qbar = ref_primes[Msc_RS] * (u+1)/(double)31;
//         ul_ref_sigs[u][v][Msc_RS] = (int16_t*)malloc16(2*sizeof(int16_t)*dftsizes[Msc_RS]);

//         if ((((int)floor(2*qbar))&1) == 0)
//           q = (int)(floor(qbar+.5)) - v;
//         else
//           q = (int)(floor(qbar+.5)) + v;

// #ifdef MAIN
//         printf("Msc_RS %d (%d), u %d, v %d -> q %d (qbar %f)\n",Msc_RS,dftsizes[Msc_RS],u,v,q,qbar);
// #endif

//         for (n=0; n<dftsizes[Msc_RS]; n++) {
//           m=n%ref_primes[Msc_RS];
//           phase = (double)q*m*(m+1)/ref_primes[Msc_RS];
//           ul_ref_sigs[u][v][Msc_RS][n<<1]     =(int16_t)(floor(32767*cos(M_PI*phase)));
//           ul_ref_sigs[u][v][Msc_RS][1+(n<<1)] =-(int16_t)(floor(32767*sin(M_PI*phase)));
// #ifdef MAIN

//           if (Msc_RS<5)
//             printf("(%d,%d) ",ul_ref_sigs[u][v][Msc_RS][n<<1],ul_ref_sigs[u][v][Msc_RS][1+(n<<1)]);

// #endif
//         }

// #ifdef MAIN

//         if (Msc_RS<5)
//           printf("\n");

// #endif
//       }
//     }
//   }

//   // These are the sequences for RB 1
//   for (u=0; u<30; u++) {
//     ul_ref_sigs[u][0][0] = (int16_t*)malloc16(2*sizeof(int16_t)*dftsizes[0]);

//     for (n=0; n<dftsizes[0]; n++) {
//       ul_ref_sigs[u][0][0][n<<1]    =(int16_t)(floor(32767*cos(M_PI*ref12[(u*12) + n]/4)));
//       ul_ref_sigs[u][0][0][1+(n<<1)]=(int16_t)(floor(32767*sin(M_PI*ref12[(u*12) + n]/4)));
//     }

//   }

//   // These are the sequences for RB 2
//   for (u=0; u<30; u++) {
//     ul_ref_sigs[u][0][1] = (int16_t*)malloc16(2*sizeof(int16_t)*dftsizes[1]);

//     for (n=0; n<dftsizes[1]; n++) {
//       ul_ref_sigs[u][0][1][n<<1]    =(int16_t)(floor(32767*cos(M_PI*ref24[(u*24) + n]/4)));
//       ul_ref_sigs[u][0][1][1+(n<<1)]=(int16_t)(floor(32767*sin(M_PI*ref24[(u*24) + n]/4)));
//     }
//   }
// }

void generate_ul_ref_sigs_rx_NB_IoT(void)
{

  unsigned int u,index_Nsc_RU,n; // Vincent: index_Nsc_RU 0,1,2,3 ---> number of sc 1,3,6,12 
  uint8_t npusch_format = 1; // NB-IoT: format 1 (data), or 2: ack. Should be defined in higher layer 
  int16_t  a;
  int16_t   qpsk[2]; 
  unsigned int x1, x2=35; // NB-IoT: defined in 36.211, Section 10.1.4.1.1
  int16_t ref_sigs_sc1[2*sequence_length[0]]; 
  uint32_t s; 

  a = (ONE_OVER_SQRT2_Q15_NB_IoT)>>15;
  qpsk[0] = a; 
  qpsk[1] = -a;
  s = lte_gold_generic_NB_IoT(&x1, &x2, 1);

  for (index_Nsc_RU=0; index_Nsc_RU<4; index_Nsc_RU++) {
    for (u=0; u<u_max[index_Nsc_RU]; u++) { 
      // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU] = (int16_t*)malloc16(2*sizeof(int16_t)*sequence_length[index_Nsc_RU]); 
      switch (index_Nsc_RU){
        case 0: // 36.211, Section 10.1.4.1.1
          ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU] = (int16_t*)malloc16(sizeof(int16_t)*(2*sequence_length[index_Nsc_RU]*12+24)); // *12 is mandatory to fit channel estimation functions
          // NB-IoT: for same reason, +24 is added in order to fit the possible subcarrier start shift when index_Nsc_RU = 0, 1, 2 --> see ul_sc_start in channel estimation function
          for (n=0; n<sequence_length[index_Nsc_RU]; n++) {
            ref_sigs_sc1[n<<1]    = qpsk[(s>>n)&1]*w_n[16*u+n%16]; 
            // ref_sigs_sc1[1+(n<<1)] = qpsk[(s>>n)&1]*w_n[16*u+n%16];
            ref_sigs_sc1[1+(n<<1)] = ref_sigs_sc1[n<<1];  
          }
          if (npusch_format==1){
            for (n=0; n<sequence_length[index_Nsc_RU]; n++) {
              // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][n<<1]    = ref_sigs_sc1[n<<1];
              // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)]= ref_sigs_sc1[1+(n<<1)];
              ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][12*(n<<1)+24]    = ref_sigs_sc1[n<<1]; // ul_ref_sigs_rx_NB_IoT is filled every 12 RE, real part
              ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+12*(n<<1)+24]= ref_sigs_sc1[1+(n<<1)]; // ul_ref_sigs_rx_NB_IoT is filled every 12 RE, imaginary part    
            }
          }
          if (npusch_format==2){// NB-IoT: to be implemented
            printf("Not coded yet\n"); 
          }
          break; 
        case 1: // 36.211, Section 10.1.4.1.2
          ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU] = (int16_t*)malloc16(sizeof(int16_t)*(2*12+24)); // *12 is mandatory to fit channel estimation functions
          for (n=0; n<sequence_length[index_Nsc_RU]; n++) {
            // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref3[(u*3) + n]/4 + alpha3[threetnecyclicshift])));
            // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref3[(u*3) + n]/4 + alpha3[threetnecyclicshift]))); 
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][(n<<1)+24]    = (int16_t)(floor(32767*cos(M_PI*ref3[(u*3) + n]/4 )));
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)+24]= (int16_t)(floor(32767*sin(M_PI*ref3[(u*3) + n]/4 )));
          }
          break; 
        case 2:
          ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU] = (int16_t*)malloc16(sizeof(int16_t)*(2*12+24)); // *12 is mandatory to fit channel estimation functions
          for (n=0; n<sequence_length[index_Nsc_RU]; n++) {
            // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref6[(u*6) + n]/4 + alpha6[sixtonecyclichift])));
            // ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref6[(u*6) + n]/4 + alpha6[sixtonecyclichift])));
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][(n<<1)+24]    = (int16_t)(floor(32767*cos(M_PI*ref6[(u*6) + n]/4 )));
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)+24]= (int16_t)(floor(32767*sin(M_PI*ref6[(u*6) + n]/4 )));
          }
          break; 
        case 3:
          ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU] = (int16_t*)malloc16(sizeof(int16_t)*(2*12+24)); // *12 is mandatory to fit channel estimation functions
          for (n=0; n<sequence_length[index_Nsc_RU]; n++) {
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref12_NB_IoT[(u*12) + n]/4)));
            ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref12_NB_IoT[(u*12) + n]/4)));
          }
          break; 
      }
    }
  }


}

void free_ul_ref_sigs_NB_IoT(void)
{

  unsigned int u,index_Nsc_RU;

  for (index_Nsc_RU=0; index_Nsc_RU<4; index_Nsc_RU++) {
    for (u=0; u<30; u++) {
        if (ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU])
          free16(ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU],4*sizeof(int16_t)*sequence_length[index_Nsc_RU]);
    }
  }
}

// void free_ul_ref_sigs(void)
// {

//   unsigned int u,index_Nsc_RU;

//   for (index_Nsc_RU=0; index_Nsc_RU<4; index_Nsc_RU++) {
//     for (u=0; u<30; u++) {
//         if (ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU])
//           free16(ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU],4*sizeof(int16_t)*sequence_length[index_Nsc_RU]);
//     }
//   }
// }

#ifdef MAIN
main()
{

  //generate_ul_ref_sigs();
  generate_ul_ref_sigs_rx();
  free_ul_ref_sigs();
}
#endif
