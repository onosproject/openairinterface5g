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

/*! \file vars_nb_iot.c
* \brief rrc variables for NB_IoT
* \author Raymond Knopp, Navid Nikaein and Michele Paffetti
* \date 2013 -2017
* \version 1.0
* \company Eurecom
* \email: navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
*/


//#ifndef __OPENAIR_RRC_VARS_NB_H__
//#define __OPENAIR_RRC_VARS_NB_H__
#include "defs_nb_iot.h"
#include "LAYER2/RLC/rlc.h"
#include "COMMON/mac_rrc_primitives.h"
#include "LAYER2/MAC/defs_nb_iot.h"
#include "LogicalChannelConfig-NB-r13.h"


//UE_RRC_INST *UE_rrc_inst; //MP: may not used for the moment


#ifndef USER_MODE

#ifndef NO_RRM
int S_rrc= RRC2RRM_FIFO;
#endif //NO_RRM

#else
#include "LAYER2/MAC/extern.h"
#ifndef NO_RRM
sock_rrm_t S_rrc;
#endif
#endif

#ifndef NO_RRM
#ifndef USER_MODE
char *Header_buf;
char *Data;
unsigned short Header_read_idx,Data_read_idx,Header_size;
#endif
unsigned short Data_to_read;
#endif //NO_RRM

#include "LAYER2/MAC/extern.h"
#define MAX_U32 0xFFFFFFFF

//LogicalChannelConfig__ext1 logicalChannelSR_Mask_r9_ext1--> not defined for NB-IoT

eNB_RRC_INST_NB *eNB_rrc_inst_NB;

uint8_t DRB2LCHAN_NB[2];//max can be 2 DRBs for NB_IoT --> it used for saving the LCID of DRBs

BOOLEAN_t logicalChannelSR_Prohibit = 1;
long priority =1;

// Default SRB configurations from 36.331 (9.2.1.1 pag 641  V14.2.1)
struct LogicalChannelConfig_NB_r13 SRB1bis_NB_logicalChannelConfig_defaultValue = {
		priority_r13: &priority, //priority
		logicalChannelSR_Prohibit_r13: &logicalChannelSR_Prohibit //set to TRUE
};
struct LogicalChannelConfig_NB_r13 SRB1_NB_logicalChannelConfig_defaultValue = {
		priority_r13: &priority, //priority
		logicalChannelSR_Prohibit_r13: &logicalChannelSR_Prohibit //set to TRUE
};



//CONSTANTS
rlc_info_t Rlc_info_um,Rlc_info_am_config, Rlc_info_am;
uint16_t RACH_FREQ_ALLOC;

//TODO: MP: to e changed --> mac_rrc_primitives
LCHAN_DESC BCCH_LCHAN_DESC,CCCH_LCHAN_DESC,DCCH_LCHAN_DESC,DTCH_DL_LCHAN_DESC,DTCH_UL_LCHAN_DESC;

//XXX //MP:May not needed
//MAC_MEAS_T BCCH_MEAS_TRIGGER,CCCH_MEAS_TRIGGER,DCCH_MEAS_TRIGGER,DTCH_MEAS_TRIGGER;
//MAC_AVG_T BCCH_MEAS_AVG, CCCH_MEAS_AVG,DCCH_MEAS_AVG, DTCH_MEAS_AVG;

// timers (TS 36.331 "UE-TimersAndConstants-NB" pag 622) (milliseconds) XXX MP:--> not sure they will be used
uint16_t T300[8] = {2500,4000,6000,10000, 15000,25000,40000,60000};
uint16_t T301[8] = {2500,4000,6000,10000, 15000,25000,40000,60000}; //MP: this start at RRCconnectionReestablishmentReq (not implemented in OAI)
uint16_t T310[8] = {0,200,500,1000,2000,4000,8000};
uint16_t T311[8] = {1000, 3000, 5000, 10000, 15000, 20000, 30000}; //MP: may not used
uint16_t N310[8] = {1,2,3,4,6,8,10,20};
uint16_t N311[8] = {1,2,3,4,5,6,8,10};


// MP: TimeToTrigger not used in NB-IoT
/* MP: 36.133 Section 9.1.4 RSRP Measurement Report Mapping and RSRQ Mapping, Table: 9.1.4-1 --> not for NB-IoT*/


//#endif
