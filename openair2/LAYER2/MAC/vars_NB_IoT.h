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

// #ifdef NB_IOT
// //NB-IoT
// eNB_MAC_INST_NB_IoT *eNB_mac_inst_NB_IoT;
// IF_Module_t *if_inst;
// #endif

const uint32_t BSR_TABLE_NB_IoT[BSR_TABLE_SIZE_NB_IoT]= {0,10,12,14,17,19,22,26,31,36,42,49,57,67,78,91,
                                           				105,125,146,171,200,234,274,321,376,440,515,603,706,826,967,1132,
                                           				1326,1552,1817,2127,2490,2915,3413,3995,4677,5467,6411,7505,8787,10287,12043,14099,
                                           				16507,19325,22624,26487,31009,36304,42502,49759,58255,68201,79846,93479,109439, 128125,150000, 300000
                                          				};


#endif


