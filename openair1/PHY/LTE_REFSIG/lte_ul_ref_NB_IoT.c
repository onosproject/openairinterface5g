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

#ifdef MAIN
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#endif
#include "defs.h"

uint16_t dftsizes[33] = {12,24,36,48,60,72,96,108,120,144,180,192,216,240,288,300,324,360,384,432,480,540,576,600,648,720,864,900,960,972,1080,1152,1200};

uint16_t ref_primes[33] = {11,23,31,47,59,71,89,107,113,139,179,191,211,239,283,293,317,359,383,431,479,523,571,599,647,719,863,887,953,971,1069,1151,1193};

uint16_t sequence_length[4] = {1,3,6,12}; //the "1" value should be 

// int16_t *ul_ref_sigs[30][33];
int16_t *ul_ref_sigs_rx[30][4]; //these contain the sequences in repeated format and quantized to QPSK ifdef IFFT_FPGA

/* 36.211 table 5.5.1.2-1 */
char ref12[360] = {-1,1,3,-3,3,3,1,1,3,1,-3,3,1,1,3,3,3,-1,1,-3,-3,1,-3,3,1,1,-3,-3,-3,-1,-3,-3,1,-3,1,-1,-1,1,1,1,1,-1,-3,-3,1,-3,3,-1,-1,3,1,-1,1,-1,-3,-1,1,-1,1,3,1,-3,3,-1,-1,1,1,-1,-1,3,-3,1,-1,3,-3,-3,-3,3,1,-1,3,3,-3,1,-3,-1,-1,-1,1,-3,3,-1,1,-3,3,1,1,-3,3,1,-1,-1,-1,1,1,3,-1,1,1,-3,-1,3,3,-1,-3,1,1,1,1,1,-1,3,-1,1,1,-3,-3,-1,-3,-3,3,-1,3,1,-1,-1,3,3,-3,1,3,1,3,3,1,-3,1,1,-3,1,1,1,-3,-3,-3,1,3,3,-3,3,-3,1,1,3,-1,-3,3,3,-3,1,-1,-3,-1,3,1,3,3,3,-1,1,3,-1,1,-3,-1,-1,1,1,3,1,-1,-3,1,3,1,-1,1,3,3,3,-1,-1,3,-1,-3,1,1,3,-3,3,-3,-3,3,1,3,-1,-3,3,1,1,-3,1,-3,-3,-1,-1,1,-3,-1,3,1,3,1,-1,-1,3,-3,-1,-3,-1,-1,-3,1,1,1,1,3,1,-1,1,-3,-1,-1,3,-1,1,-3,-3,-3,-3,-3,1,-1,-3,1,1,-3,-3,-3,-3,-1,3,-3,1,-3,3,1,1,-1,-3,-1,-3,1,-1,1,3,-1,1,1,1,3,1,3,3,-1,1,-1,-3,-3,1,1,-3,3,3,1,3,3,1,-3,-1,-1,3,1,3,-3,-3,3,-3,1,-1,-1,3,-1,-3,-3,-1,-3,-1,-3,3,1,-1,1,3,-3,-3,-1,3,-3,3,-1,3,3,-3,3,3,-1,-1,3,-3,-3,-1,-1,-3,-1,3,-3,3,1,-1};

/* 36.211 table 5.5.1.2-2 */
// char ref24[720] = {
//   -1,3,1,-3,3,-1,1,3,-3,3,1,3,-3,3,1,1,-1,1,3,-3,3,-3,-1,-3,-3,3,-3,-3,-3,1,-3,-3,3,-1,1,1,1,3,1,-1,3,-3,-3,1,3,1,1,-3,3,-1,3,3,1,1,-3,3,3,3,3,1,-1,3,-1,1,1,-1,-3,-1,-1,1,3,3,-1,-3,1,1,3,-3,1,1,-3,-1,-1,1,3,1,3,1,-1,3,1,1,-3,-1,-3,-1,-1,-1,-1,-3,-3,-1,1,1,3,3,-1,3,-1,1,-1,-3,1,-1,-3,-3,1,-3,-1,-1,-3,1,1,3,-1,1,3,1,-3,1,-3,1,1,-1,-1,3,-1,-3,3,-3,-3,-3,1,1,1,1,-1,-1,3,-3,-3,3,-3,1,-1,-1,1,-1,1,1,-1,-3,-1,1,-1,3,-1,-3,-3,3,3,-1,-1,-3,-1,3,1,3,1,3,1,1,-1,3,1,-1,1,3,-3,-1,-1,1,-3,1,3,-3,1,-1,-3,3,-3,3,-1,-1,-1,-1,1,-3,-3,-3,1,-3,-3,-3,1,-3,1,1,-3,3,3,-1,-3,-1,3,-3,3,3,3,-1,1,1,-3,1,-1,1,1,-3,1,1,-1,1,-3,-3,3,-1,3,-1,-1,-3,-3,-3,-1,-3,-3,1,-1,1,3,3,-1,1,-1,3,1,3,3,-3,-3,1,3,1,-1,-3,-3,-3,3,3,-3,3,3,-1,-3,3,-1,1,-3,1,1,3,3,1,1,1,-1,-1,1,-3,3,-1,1,1,-3,3,3,-1,-3,3,-3,-1,-3,-1,3,-1,-1,-1,-1,-3,-1,3,3,1,-1,1,3,3,3,-1,1,1,-3,1,3,-1,-3,3,-3,-3,3,1,3,1,-3,3,1,3,1,1,3,3,-1,-1,-3,1,-3,-1,3,1,1,3,-1,-1,1,-3,1,3,-3,1,-1,-3,-1,3,1,3,1,-1,-3,-3,-1,-1,-3,-3,-3,-1,-1,-3,3,-1,-1,-1,-1,1,1,-3,3,1,3,3,1,-1,1,-3,1,-3,1,1,-3,-1,1,3,-1,3,3,-1,-3,1,-1,-3,3,3,3,-1,1,1,3,-1,-3,-1,3,-1,-1,-1,1,1,1,1,1,-1,3,-1,-3,1,1,3,-3,1,-3,-1,1,1,-3,-3,3,1,1,-3,1,3,3,1,-1,-3,3,-1,3,3,3,-3,1,-1,1,-1,-3,-1,1,3,-1,3,-3,-3,-1,-3,3,-3,-3,-3,-1,-1,-3,-1,-3,3,1,3,-3,-1,3,-1,1,-1,3,-3,1,-1,-3,-3,1,1,-1,1,-1,1,-1,3,1,-3,-1,1,-1,1,-1,-1,3,3,-3,-1,1,-3,-3,-1,-3,3,1,-1,-3,-1,-3,-3,3,-3,3,-3,-1,1,3,1,-3,1,3,3,-1,-3,-1,-1,-1,-1,3,3,3,1,3,3,-3,1,3,-1,3,-1,3,3,-3,3,1,-1,3,3,1,-1,3,3,-1,-3,3,-3,-1,-1,3,-1,3,-1,-1,1,1,1,1,-1,-1,-3,-1,3,1,-1,1,-1,3,-1,3,1,1,-1,-1,-3,1,1,-3,1,3,-3,1,1,-3,-3,-1,-1,-3,-1,1,3,1,1,-3,-1,-1,-3,3,-3,3,1,-3,3,-3,1,-1,1,-3,1,1,1,-1,-3,3,3,1,1,3,-1,-3,-1,-1,-1,3,1,-3,-3,-1,3,-3,-1,-3,-1,-3,-1,-1,-3,-1,-1,1,-3,-1,-1,1,-1,-3,1,1,-3,1,-3,-3,3,1,1,-1,3,-1,-1,1,1,-1,-1,-3,-1,3,-1,3,-1,1,3,1,-1,3,1,3,-3,-3,1,-1,-1,1,3
// }; 

// 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-1 
char ref3[36] = {1,  -3,  -3, 1, -3,  -1, 1, -3,  3, 1, -1,  -1, 1, -1,  1, 1, -1,  3, 1, 1, -3, 1, 1, -1, 1, 1, 3, 1, 3, -1, 1, 3, 1, 1, 3, 3 }; 

// 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-2 
char ref6[84] = {1,  1, 1, 1, 3, -3, 1, 1, 3, 1, -3,  3, 1, -1,  -1,  -1,  1, -3, 1, -1,  3, -3,  -1,  -1, 1, 3, 1, -1,  -1,  3, 1, -3,  -3,  1, 3, 1, -1,  -1,  1, -3,  -3,  -1, 
                -1,  -1,  -1,  3, -3,  -1, 3, -1,  1, -3,  -3,  3, 3, -1,  3, -3,  -1,  1, 3, -3,  3, -1,  3, 3, -3,  1, 3, 1, -3,  -1, -3,  1, -3,  3, -3,  -1, -3,  3, -3,  1, 1, -3};

// 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-3 
double alpha3[3] = {0 , M_PI*2/3, M_PI*4/3}; 
double alpha6[4] = {0 , M_PI*2/6, M_PI*4/6, M_PI*8/6}; 

// NB-IoT: 36.211, Section 10.1.4.1.1, Table 10.1.4.1.1-1
char w_n[256] = {
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

  unsigned int u,Msc_RS,n; 
  uint8_t threetnecyclicshift=0, sixtonecyclichift=0;// NB-IoT: to be defined from higher layer, see 36.211 Section 10.1.4.1.2

  for (Msc_RS=0; Msc_RS<4; Msc_RS++) {
    for (u=0; u<30; u++) { 
      ul_ref_sigs_rx[u][Msc_RS] = (int16_t*)malloc16(2*sizeof(int16_t)*sequence_length[Msc_RS]); 
      switch (Msc_RS){
        case 0: 
          printf("Not coded yet\n");
          break; 
        case 1:
          for (n=0; n<sequence_length[Msc_RS]; n++) {
            ul_ref_sigs_rx[u][Msc_RS][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref3[(u*3) + n]/4 + alpha3[threetnecyclicshift])));
            ul_ref_sigs_rx[u][Msc_RS][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref3[(u*3) + n]/4 + alpha3[threetnecyclicshift])));
          }
          break; 
        case 2:
          for (n=0; n<sequence_length[Msc_RS]; n++) {
            ul_ref_sigs_rx[u][Msc_RS][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref6[(u*6) + n]/4 + alpha6[sixtonecyclichift])));
            ul_ref_sigs_rx[u][Msc_RS][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref6[(u*6) + n]/4 + alpha6[sixtonecyclichift])));
          }
          break; 
        case 3:
          for (n=0; n<sequence_length[Msc_RS]; n++) {
            ul_ref_sigs_rx[u][Msc_RS][n<<1]    = (int16_t)(floor(32767*cos(M_PI*ref12[(u*12) + n]/4)));
            ul_ref_sigs_rx[u][Msc_RS][1+(n<<1)]= (int16_t)(floor(32767*sin(M_PI*ref12[(u*12) + n]/4)));
          }
          break; 
      }
    }
  }


}


void free_ul_ref_sigs(void)
{

  unsigned int u,Msc_RS;

  for (Msc_RS=0; Msc_RS<4; Msc_RS++) {
    for (u=0; u<30; u++) {
        if (ul_ref_sigs_rx[u][Msc_RS])
          free16(ul_ref_sigs_rx[u][Msc_RS],4*sizeof(int16_t)*sequence_length[Msc_RS]);
    }
  }
}

#ifdef MAIN
main()
{

  //generate_ul_ref_sigs();
  generate_ul_ref_sigs_rx();
  free_ul_ref_sigs();
}
#endif
