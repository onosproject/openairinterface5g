
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


#ifndef __DLSCH_TBS_FULL_NB_IOT_H__
#define __DLSCH_TBS_FULL_NB_IOT_H__

/** \brief "Transport block size table"
 *  (Table 7.1.7.2.1-1 in 3GPP TS 36.213 V8.6.0)
 */


// NB-IoT------------------

// TBS table for the case not containing SIB1-NB, Table 16.4.1.5.1-1 in TS 36.213 v14.2 
unsigned int TBStable_NB_IoT[14][8] ={ //[ITBS][ISF]
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

//TBS table for the case containing S1B1-NB, Table 16.4.1.5.2-1 in TS 36.213 v14.2 (Itbs = 12 ~ 15 is reserved field
//mapping ITBS to SIB1-NB
unsigned int TBStable_NB_IoT_SIB1[16] = {208,208,208,328,328,328,440,440,440,680,680,680};

#endif