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
/*! \file extern_NB_IoT.h
 * \brief MAC extern
 * \author  NTUST BMW Lab./Nick HO, Xavier LIU, Calvin HSU
 * \date 2017 - 2018
 * \email: nick133371@gmail.com, sephiroth7277@gmail.com , kai-hsiang.hsu@eurecom.fr
 * \version 1.0
 *
 */

#ifndef __MAC_EXTERN_NB_IOT_H__
#define __MAC_EXTERN_NB_IOT_H__

#include "openair2/PHY_INTERFACE/defs_NB_IoT.h" 



extern uint8_t Is_rrc_registered;

#ifndef PHY_EMUL
#ifndef PHYSIM
#define NB_INST 1
#else
extern unsigned char NB_INST;
#endif
extern unsigned char NB_eNB_INST;
extern unsigned char NB_UE_INST;
extern unsigned char NB_RN_INST;
extern unsigned short NODE_ID[1];
extern void* bigphys_malloc(int);
#else
extern EMULATION_VARS *Emul_vars;
#endif //PHY_EMUL




//NB-IoT---------------------------------

extern eNB_MAC_INST_NB_IoT *mac_inst;
extern uint8_t Is_rrc_registered_NB_IoT;

extern available_resource_DL_t *available_resource_DL;
extern available_resource_tones_UL_t *available_resource_UL;
extern available_resource_DL_t *available_resource_DL_last;
extern schedule_result_t *schedule_result_list_UL;
extern schedule_result_t *schedule_result_list_DL;

// array will be active when they are used

// 10 -> single-tone / 12 -> multi-tone
extern const uint32_t max_mcs[2];

// [CE level] [0 - 3] -> single-tone / [CE level] [4-7] -> multi-tone
extern const uint32_t mapped_mcs[3][8];

//TBS table for NPUSCH transmission TS 36.213 v14.2 table Table 16.5.1.2-2:
extern const int UL_TBS_Table[14][8];

const int UL_TBS_Table_msg3[8];

extern const int ULrep[8];
extern const int rachperiod[8];
extern const int rachstart[8];
extern const int rachrepeat[8];
extern const int rachscofst[7];
extern const int rachnumsc[4];
extern const int rmax[12];

extern const double gvalue[8];

extern const double pdcchoffset[4];

extern const uint32_t RU_table[8];
extern const uint32_t RU_table_msg3[8];

extern const uint32_t scheduling_delay[4];
extern const uint32_t msg3_scheduling_delay_table[4];

extern const uint32_t ack_nack_delay[4];
extern const uint32_t R_dl_table[16];

// TBS table for the case not containing SIB1-NB_IoT, Table 16.4.1.5.1-1 in TS 36.213 v14.2
extern const uint32_t MAC_TBStable_NB_IoT[14][8];

//TBS table for the case containing S1B1-NB_IoT, Table 16.4.1.5.2-1 in TS 36.213 v14.2 (Itbs = 12 ~ 15 is reserved field
//mapping ITBS to SIB1-NB_IoT
extern const unsigned int MAC_TBStable_NB_IoT_SIB1[16];

extern const int DV_table[16];
extern const int BSR_table[64];

extern const int dl_rep[3];
extern const uint32_t dci_rep[3];
extern const uint32_t harq_rep[3];

extern rach_state_t UE_state_machine;

//SIBs
extern int extend_space[2];
extern int extend_alpha_offset[2];

extern const int si_repetition_pattern[4];
extern int waiting_flag_from_RLC;
extern int block_RLC;
extern int Valid_msg3;
extern int RLC_RECEIVE_MSG5_FAILED;
#endif //DEF_H
