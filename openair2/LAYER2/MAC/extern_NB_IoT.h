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

/*! \file extern.h
* \brief mac externs
* \author  Navid Nikaein and Raymond Knopp
* \date 2010 - 2014
* \version 1.0
* \email navid.nikaein@eurecom.fr
* @ingroup _mac

*/

#ifndef __MAC_EXTERN_NB_IOT_H__
#define __MAC_EXTERN_NB_IOT_H__


// #ifdef USER_MODE
// //#include "stdio.h"
// #endif //USER_MODE
// #include "PHY/defs.h"
// #include "defs.h"
// #include "COMMON/mac_rrc_primitives.h"
// #ifdef PHY_EMUL
// //#include "SIMULATION/simulation_defs.h"
// #endif //PHY_EMUL
#include "openair2/PHY_INTERFACE/defs_NB_IoT.h" 
//#include "RRC/LITE/defs_NB_IoT.h"

#include "LAYER2/MAC/defs_NB_IoT.h"

//NB-IoT
extern IF_Module_t *if_inst;
extern eNB_MAC_INST_NB_IoT *eNB_mac_inst_NB_IoT;

// //extern uint32_t EBSR_Level[63];
// extern const uint32_t Extended_BSR_TABLE[BSR_TABLE_SIZE];
// //extern uint32_t Extended_BSR_TABLE[63];  ----currently not used 

// extern const uint8_t cqi2fmt0_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern const uint8_t cqi2fmt1x_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern const uint8_t cqi2fmt2x_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern UE_MAC_INST *UE_mac_inst;
// extern eNB_MAC_INST *eNB_mac_inst;
// extern eNB_RRC_INST *eNB_rrc_inst;
//extern UE_RRC_INST_NB_IoT *UE_rrc_inst_NB_IoT;
// extern UE_MAC_INST *ue_mac_inst;
// extern MAC_RLC_XFACE *Mac_rlc_xface;
// extern uint8_t Is_rrc_registered;


//#ifndef USER_MODE

// extern RRC_XFACE *Rrc_xface;          //// to uncomment when it is used

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

extern eNB_MAC_INST_NB_IoT *eNB_mac_inst_NB_IoT;
extern uint8_t Is_rrc_registered_NB_IoT;



#endif //DEF_H


