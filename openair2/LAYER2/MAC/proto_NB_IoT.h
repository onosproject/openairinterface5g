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

/*! \file LAYER2/MAC/proto_NB_IoT.h
 * \brief MAC functions prototypes for eNB and UE
 * \author Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email navid.nikaein@eurecom.fr
 * \version 1.0
 */

#ifndef __LAYER2_MAC_PROTO_NB_IoT_H__
#define __LAYER2_MAC_PROTO_NB_IoT_H__

#include "openair1/PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "COMMON/platform_types.h"
/** \addtogroup _mac
 *  @{
 */

/*for NB-IoT*/

/* \brief Function to indicate a received SDU on ULSCH for NB-IoT.
*/
void rx_sdu_NB_IoT(const module_id_t module_idP, const int CC_id,const frame_t frameP, const sub_frame_t subframeP, const rnti_t rnti, uint8_t *sdu, const uint16_t sdu_len, const int harq_pid);

/* \brief Function to retrieve result of scheduling (DCI) in current subframe.  Can be called an arbitrary numeber of times after eNB_dlsch_ulsch_scheduler
in a given subframe. 
*/
DCI_PDU_NB_IoT *get_dci_sdu_NB_IoT(module_id_t module_idP,int CC_id,frame_t frameP,sub_frame_t subframe);


/* \brief Function to trigger the eNB scheduling procedure.  It is called by PHY at the beginning of each subframe, \f$n$\f
   and generates all DLSCH allocations for subframe \f$n\f$ and ULSCH allocations for subframe \f$n+k$\f. The resultant DCI_PDU is
   ready after returning from this call.

*/
void eNB_dlsch_ulsch_scheduler_NB_IoT(module_id_t module_idP, uint8_t cooperation_flag, frame_t frameP, sub_frame_t subframeP);

/* \brief Function to indicate a received preamble on PRACH.  It initiates the RA procedure.
    In NB-IoT, it indicate preamble using the frequency to indicate the preamble.
*/
void schedule_RA_NB_IoT(module_id_t module_idP,frame_t frameP, sub_frame_t subframeP);

void initiate_ra_proc_NB_IoT(module_id_t module_idP, int CC_id,frame_t frameP, uint16_t preamble_index,int16_t timing_offset,sub_frame_t subframeP);

uint8_t *get_dlsch_sdu_NB_IoT(module_id_t module_idP,int CC_id,frame_t frameP,rnti_t rnti,uint8_t TBindex);


int rrc_mac_remove_ue_NB_IoT(module_id_t Mod_id, rnti_t rntiP);

int l2_init_eNB_NB_IoT(void);
int mac_init_global_param_NB_IoT(void);
int mac_top_init_NB_IoT(void);

int find_UE_id_NB_IoT (module_id_t module_idP, rnti_t rnti) ;
int UE_PCCID_NB_IoT (module_id_t module_idP, int UE_id);
rnti_t  UE_RNTI_NB_IoT (module_id_t module_idP, int UE_id);

/*! \fn  UE_L2_state_t ue_scheduler(const module_id_t module_idP,const frame_t frameP, const sub_frame_t subframe, const lte_subframe_t direction,const uint8_t eNB_index)
   \brief UE scheduler where all the ue background tasks are done.  This function performs the following:  1) Trigger PDCP every 5ms 2) Call RRC for link status return to PHY3) Perform SR/BSR procedures for scheduling feedback 4) Perform PHR procedures.
\param[in] module_idP instance of the UE
\param[in] rxFrame the RX frame number
\param[in] rxSubframe the RX subframe number
\param[in] txFrame the TX frame number
\param[in] txSubframe the TX subframe number
\param[in] direction  subframe direction
\param[in] eNB_index  instance of eNB
@returns L2 state (CONNETION_OK or CONNECTION_LOST or PHY_RESYNCH)
*/
UE_L2_STATE_NB_IoT_t ue_scheduler_NB_IoT(
  const module_id_t module_idP,
  const frame_t rxFrameP,
  const sub_frame_t rxSubframe,
  const frame_t txFrameP,
  const sub_frame_t txSubframe,
  const NB_IoT_subframe_t direction,
  const uint8_t eNB_index,
  const int CC_id);

/* \brief Function used by PHY to inform MAC that an uplink is scheduled
          for Msg3 in given subframe. This is used so that the MAC
          scheduler marks as busy the RBs used by the Msg3.
@param Mod_id        Instance ID of eNB
@param CC_id         CC ID of eNB
@param frame         current frame
@param subframe      current subframe
@param rnti          UE rnti concerned
@param Msg3_frame    frame where scheduling takes place
@param Msg3_subframe subframe where scheduling takes place
*/
void set_msg3_subframe_NB_IoT(module_id_t Mod_id,
                      		  int CC_id,
                       		  int frame,
                       		  int subframe,
                       		  int rnti,
                       		  int Msg3_frame,
                       		  int Msg3_subframe);

/* \brief Parse header for UL-SCH.  This function parses the received UL-SCH header as described
in 36-321 MAC layer specifications.  It returns the number of bytes used for the header to be used as an offset for the payload
in the ULSCH buffer.
@param mac_header Pointer to the first byte of the MAC header (UL-SCH buffer)
@param num_ces Number of SDUs in the payload
@param num_sdu Number of SDUs in the payload
@param rx_ces Pointer to received CEs in the header
@param rx_lcids Pointer to array of LCIDs (the order must be the same as the SDU length array)
@param rx_lengths Pointer to array of SDU lengths
@returns Pointer to payload following header
*/
uint8_t *parse_ulsch_header_NB_IoT(uint8_t *mac_header,
                                   uint8_t *num_ce,
                                   uint8_t *num_sdu,
                                   uint8_t *rx_ces,
                                   uint8_t *rx_lcids,
                                   uint16_t *rx_lengths,
                                   uint16_t tx_lenght);

int add_new_ue_NB_IoT(module_id_t Mod_id, int CC_id, rnti_t rnti,int harq_pid);
//void dump_ue_list_NB_IoT(UE_list_t *listP, int ul_flag);

#endif
