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

/*! \file PHY/LTE_TRANSPORT/group_hopping.c
* \brief Top-level routines for group hopping for DMRS from 36.211 R14, Section 10.1.4.1.3
* \author Vincent Savaux
* \date 2017
* \version 0.1
* \company b<>com
* \email: vincent.savaux@b<>com.com
* \note
* \warning
*/

#include "PHY/defs_NB_IoT.h"
#include <math.h> 

//#define DEBUG_GROUPHOP 1

void generate_grouphop_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms)
{

  uint8_t ns; 
  uint8_t reset=1; 
  uint32_t x1, x2, s=0;
  uint16_t Nseq_RU[4] = {16,12,14,30}; // Table 10.1.4.1.3-1
  // This is from Section 10.1.4.1.3
  uint32_t fss_npusch_nn = frame_parms->Nid_cell + frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH;
  uint32_t fss_npusch; 
  int k; 

  if (frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled){

    for (k=0;k<4;k++){ // all four possible sucarrier configurations
      fss_npusch = fss_npusch_nn % Nseq_RU[k]; 
      x2 = (uint32_t) floor(frame_parms->Nid_cell/Nseq_RU[k]); 

      // x2 = frame_parms->Nid_cell/30;
    #ifdef DEBUG_GROUPHOP
      printf("[PHY] GroupHop:");
    #endif

      for (ns=0; ns<20; ns++) { // slot number
        // if (k==0){ // 1 subcarrier --> initialize at the beginning of the sequence 
        if ((ns&3) == 0) {
          s = lte_gold_generic_NB_IoT(&x1,&x2,reset);
          reset = 0;
        }
        // }else{ // 3, 6, 12 subcarriers --> initialize at every even slot number
        //   if ((ns%2) == 0) {
        //     s = lte_gold_generic_NB_IoT(&x1,&x2,reset);
        //     reset = 1;
        //   }         
        // }

        frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[ns][k] = (((uint8_t*)&s)[ns&3]%Nseq_RU[k]+fss_npusch)%Nseq_RU[k]; 

    #ifdef DEBUG_GROUPHOP
        printf("%d.",frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[ns]);
    #endif
      }
    }
  }else{
    for (ns=0; ns<20; ns++) {
      for (k=0;k<4;k++){
        frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[ns][k] = frame_parms->Nid_cell % Nseq_RU[k]; 
      }
    }
  }

#ifdef DEBUG_GROUPHOP
  printf("\n");
#endif
}

// void generate_seqhop(LTE_DL_FRAME_PARMS *frame_parms)
// {

//   uint8_t ns,reset=1;
//   uint32_t x1, x2, s=0;
//   // This is from Section 5.5.1.3
//   uint32_t fss_pusch = frame_parms->Nid_cell + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.groupAssignmentPUSCH;

//   x2 = (32*(frame_parms->Nid_cell/30) + fss_pusch)%30;

//   s = lte_gold_generic(&x1,&x2,reset);
// #ifdef DEBUG_GROUPHOP
//   printf("[PHY] SeqHop:");
// #endif

//   for (ns=0; ns<20; ns++) {
//     if ((frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.groupHoppingEnabled == 0) &&
//         (frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.sequenceHoppingEnabled == 1))
//       frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[ns] = (s>>(ns&0x1f))&1;
//     else
//       frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[ns] = 0;

// #ifdef DEBUG_GROUPHOP
//     printf("%d.",frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.seqhop[ns]);
// #endif
//   }

// #ifdef DEBUG_GROUPHOP
//   printf("\n");
// #endif
// }

// void generate_nPRS(LTE_DL_FRAME_PARMS *frame_parms)
// {

//   uint16_t n=0;
//   uint8_t reset=1;
//   uint32_t x1, x2, s=0;
//   // This is from Section 5.5.1.3
//   uint8_t Nsymb_UL = (frame_parms->Ncp_UL == NORMAL) ? 7 : 6;
//   uint16_t next = 0;
//   uint8_t ns=0;

//   uint32_t fss_pucch = (frame_parms->Nid_cell) % 30;
//   uint32_t fss_pusch = (fss_pucch + frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.groupAssignmentPUSCH) % 30;

//   x2 = (32*(uint32_t)(frame_parms->Nid_cell/30)) + fss_pusch;
// #ifdef DEBUG_GROUPHOP
//   printf("[PHY] nPRS:");
// #endif

//   for (n=0; n<(20*Nsymb_UL); n++) { //loop over total number of bytes to generate
//     if ((n&3) == 0) {
//       s = lte_gold_generic(&x1,&x2,reset);
//       reset = 0;
//       //      printf("n %d : s (%d,%d,%d,%d)\n",n,((uint8_t*)&s)[0],((uint8_t*)&s)[1],((uint8_t*)&s)[2],((uint8_t*)&s)[3]);
//     }

//     if (n == next) {
//       frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.nPRS[ns] = ((uint8_t*)&s)[next&3];
// #ifdef DEBUG_GROUPHOP
//       printf("%d.",frame_parms->pusch_config_common.ul_ReferenceSignalsPUSCH.nPRS[ns]);
// #endif
//       ns++;
//       next+=Nsymb_UL;
//     }
//   }

// #ifdef DEBUG_GROUPHOP
//   printf("\n");
// #endif
// }

void init_ul_hopping_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms)
{

  ////////////////////////////////////////generate_grouphop(frame_parms);
  // generate_seqhop(frame_parms);
  // generate_nPRS(frame_parms);
}
