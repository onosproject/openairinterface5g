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

/*! \file vars.h
* \brief mac vars
* \author  Navid Nikaein and Raymond Knopp
* \date 2010 - 2014
* \version 1.0
* \email navid.nikaein@eurecom.fr
* @ingroup _mac

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


IF_Module_t *if_inst;

//NB-IoT--------------------------------------
eNB_MAC_INST_NB_IoT *mac_inst;

BCCH_BCH_Message_NB_t               MIB;
BCCH_DL_SCH_Message_NB_t            SIB;
RRCConnectionSetup_NB_r13_IEs_t     DED_Config;
schedule_result_t *schedule_result_list_UL;
schedule_result_t *schedule_result_list_DL;
available_resource_DL_t *available_resource_DL;
available_resource_tones_UL_t *available_resource_UL;
//should be utilized in: schedule_RA_NB_IoT,rx_sdu_NB_IoT, mac_top_init_NB_IoT,
uint8_t Is_rrc_registered_NB_IoT;


#endif


