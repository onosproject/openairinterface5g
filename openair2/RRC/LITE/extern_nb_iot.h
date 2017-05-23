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
* \brief rrc external vars
* \author Navid Nikaein and Raymond Knopp, Michele Paffetti
* \date 2011-2017
* \version 1.0
* \company Eurecom
* \email: navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
*/

#ifndef __OPENAIR_RRC_EXTERN_NB_IOT_H__
#define __OPENAIR_RRC_EXTERN_NB_IOT_H__
#include "defs_nb_iot.h"
#include "COMMON/mac_rrc_primitives.h"
#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/RLC/rlc.h"
#include "LogicalChannelConfig-NB-r13.h"

//MP: NOTE:XXX some of the parameters defined in vars_nb_iot are called by the extern.h file so not replicated here


extern eNB_RRC_INST_NB *eNB_rrc_inst_NB;

extern rlc_info_t Rlc_info_um,Rlc_info_am_config,Rlc_info_am;
extern uint8_t DRB2LCHAN_NB[2];
extern LogicalChannelConfig_NB_r13_t SRB1bis_NB_logicalChannelConfig_defaultValue;
extern LogicalChannelConfig_NB_r13_t SRB1_NB_logicalChannelConfig_defaultValue;

#endif


