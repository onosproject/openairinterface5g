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

//#include "dlsch_tbs.h"
//#include "dlsch_tbs_full.h"
//#include "sss.h"
#ifndef __PHY_LTE_TRANSPORT_VARS_NB_IOT__H__
#define __PHY_LTE_TRANSPORT_VARS_NB_IOT__H__

unsigned char cs_ri_normal_NB_IoT[4]    = {1,4,7,10};
unsigned char cs_ri_extended_NB_IoT[4]  = {0,3,5,8};
unsigned char cs_ack_normal_NB_IoT[4]   = {2,3,8,9};
unsigned char cs_ack_extended_NB_IoT[4] = {1,2,6,7};


int8_t wACK_RX_NB_IoT[5][4] = {{-1,-1,-1,-1},{-1,1,-1,1},{-1,-1,1,1},{-1,1,1,-1},{1,1,1,1}};

#endif
