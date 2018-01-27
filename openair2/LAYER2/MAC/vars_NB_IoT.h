
/*! \file vars_NB_IoT.h
 * \brief declare the MAC global variables
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */


#ifndef __MAC_VARS_NB_IOT_H__
#define __MAC_VARS_NB_IOT_H__
#ifdef USER_MODE
//#include "stdio.h"
#endif //USER_MODE
//#include "PHY/defs.h"
//#include "defs.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
//#include "PHY_INTERFACE/defs.h"
//#include "COMMON/mac_rrc_primitives.h"


//NB-IoT--------------------------------------
eNB_MAC_INST_NB_IoT *mac_inst;

//BCCH_BCH_Message_NB_t               MIB;
//BCCH_DL_SCH_Message_NB_t            SIB;
//RRCConnectionSetup_NB_r13_IEs_t     DED_Config;
schedule_result_t *schedule_result_list_UL;
schedule_result_t *schedule_result_list_DL;
available_resource_DL_t *available_resource_DL;
available_resource_tones_UL_t *available_resource_UL;
available_resource_DL_t *available_resource_DL_last;
//should be utilized in: schedule_RA_NB_IoT,rx_sdu_NB_IoT, mac_top_init_NB_IoT,
uint8_t Is_rrc_registered_NB_IoT;

// array will be active when they are used

// 10 -> single-tone / 12 -> multi-tone
const uint32_t max_mcs[2] = {10, 12};

// [CE level] [0 - 3] -> single-tone / [CE level] [4-7] -> multi-tone
const uint32_t mapped_mcs[3][8]={{1,5,9,10,3,7,11,12},
                            {0,3,7,10,3,7,11,12},
                            {0,2,6,10,0,4,8,12}};

//TBS table for NPUSCH transmission TS 36.213 v14.2 table Table 16.5.1.2-2:
const int UL_TBS_Table[14][8]=
{
  {16,2,56,88,120,152,208,256},
  {24,56,88,144,176,208,256,344},
  {32,72,144,176,208,256,328,424},
  {40,104,176,208,256,328,440,568},
  {56,120,208,256,328,408,552,680},
  {72,144,224,328,424,504,680,872},
  {88,176,256,392,504,600,808,1000},
  {104,224,328,472,584,712,1000,1224},
  {120,256,392,536,680,808,1096,1384},
  {136,296,456,616,776,936,1256,1544},
  {144,328,504,680,872,1000,1384,1736},
  {176,376,584,776,1000,1192,1608,2024},
  {208,440,680,1000,1128,1352,1800,2280},
  {224,488,744,1128,1256,1544,2024,2536}
};

const int rachperiod[8]={40,80,160,240,320,640,1280,2560};
const int rachstart[8]={8,16,32,64,128,256,512,1024};
const int rachrepeat[8]={1,2,4,8,16,32,64,128};
const int rachscofst[7]={0,12,24,36,2,18,34};
const int rachnumsc[4]={12,24,36,48};

const uint32_t RU_table[8]={1,2,3,4,5,6,8,10};

const uint32_t scheduling_delay[4]={8,16,32,64};
const uint32_t msg3_scheduling_delay_table[4] = {12,16,32,64};

const uint32_t ack_nack_delay[4]={13,15,17,18};
const uint32_t R_dl_table[16]={1,2,4,8,16,32,64,128,192,256,384,512,768,1024,1536,2048};

// TBS table for the case not containing SIB1-NB_IoT, Table 16.4.1.5.1-1 in TS 36.213 v14.2
const uint32_t MAC_TBStable_NB_IoT[14][8] ={ //[ITBS][ISF]
  {16,32,56,88,120.152,208,256},
  {24,56,88,144,176,208,256,344},
  {32,72,144,176,208,256,328,424},
  {40,104,176,208,256,328,440,568},
  {56,120,208,256,328,408,552,680},
  {72,144,244,328,424,504,680,872},
  {88,176,256,392,504,600,808,1032},
  {104,224,328,472,584,680,968,1224},
  {120,256,392,536,680,808,1096,1352},
  {136,296,456,616,776,936,1256,1544},
  {144,328,504,680,872,1032,1384,1736},
  {176,376,584,776,1000,1192,1608,2024},
  {208,440,680,904,1128,1352,1800,2280},
  {224,488,744,1128,1256,1544,2024,2536}
};

//TBS table for the case containing S1B1-NB_IoT, Table 16.4.1.5.2-1 in TS 36.213 v14.2 (Itbs = 12 ~ 15 is reserved field
//mapping ITBS to SIB1-NB_IoT
const unsigned int MAC_TBStable_NB_IoT_SIB1[16] = {208,208,208,328,328,328,440,440,440,680,680,680};

const int DV_table[16]={0,10,14,19,26,36,49,67,91,125,171,234,321,768,1500,1500};

const int BSR_table[64]= {0,10,12,14,17,19,22,26,31,36,42,49,57,67,78,91,
                           105,125,146,171,200,234,274,321,376,440,515,603,706,826,967,1132,
                           1326,1552,1817,2127,2490,2915,3413,3995,4677,5467,6411,7505,8787,10287,12043,14099,
                           16507,19325,22624,26487,31009,36304,42502,49759,58255,68201,79846,93479,109439,128125,150000,300000
                           };

const int dl_rep[3] = {1, 2, 4};
const uint32_t dci_rep[3] = {1, 2, 4};
const uint32_t harq_rep[3] = {1, 2, 4};


#endif


