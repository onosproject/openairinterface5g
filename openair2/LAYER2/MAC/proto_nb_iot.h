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

/*! \file LAYER2/MAC/proto.h
 * \brief MAC functions prototypes for eNB and UE
 * \author Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email navid.nikaein@eurecom.fr
 * \version 1.0
 */

 


/** \addtogroup _mac
 *  @{
 */

/*for NB-IoT*/

/* \brief Function to indicate a received SDU on ULSCH for NB-IoT.
*/
void NB_rx_sdu(const module_id_t module_idP, const int CC_id,const frame_t frameP, const sub_frame_t subframeP, const rnti_t rnti, uint8_t *sdu, const uint16_t sdu_len, const int harq_pid);

/* \brief Function to retrieve result of scheduling (DCI) in current subframe.  Can be called an arbitrary numeber of times after eNB_dlsch_ulsch_scheduler
in a given subframe. 
*/
DCI_PDU_NB *NB_get_dci_sdu(module_id_t module_idP,int CC_id,frame_t frameP,sub_frame_t subframe);


/* \brief Function to trigger the eNB scheduling procedure.  It is called by PHY at the beginning of each subframe, \f$n$\f
   and generates all DLSCH allocations for subframe \f$n\f$ and ULSCH allocations for subframe \f$n+k$\f. The resultant DCI_PDU is
   ready after returning from this call.

*/
void NB_eNB_dlsch_ulsch_scheduler(module_id_t module_idP, uint8_t cooperation_flag, frame_t frameP, sub_frame_t subframeP);

/* \brief Function to indicate a received preamble on PRACH.  It initiates the RA procedure.
    In NB-IoT, it indicate preamble using the frequency to indicate the preamble.
*/
void NB_schedule_RA(module_id_t module_idP,frame_t frameP, sub_frame_t subframeP);

void NB_initiate_ra_proc(module_id_t module_idP, int CC_id,frame_t frameP, uint16_t preamble_index,int16_t timing_offset,sub_frame_t subframeP);

uint8_t *NB_get_dlsch_sdu(module_id_t module_idP,int CC_id,frame_t frameP,rnti_t rnti,uint8_t TBindex);


int NB_rrc_mac_remove_ue(module_id_t Mod_id, rnti_t rntiP);

int NB_l2_init_eNB(void);
int mac_init_global_param_NB(void);



