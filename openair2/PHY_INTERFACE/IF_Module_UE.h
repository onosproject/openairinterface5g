
/*This is the interface module between PHY
*Provided the FAPI style interface structures for P7.
*
*
*
*//*
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

/*! \file openair2/PHY_INTERFACE/IF_Module.h
* \brief data structures for PHY/MAC interface modules
* \author EURECOM/NTUST
* \date 2017
* \version 0.1
* \company Eurecom
* \email: raymond.knopp@eurecom.fr
* \note
* \warning
*/
#ifndef __IF_MODULE_UE__H__
#define __IF_MODULE_UE__H__


#include <stdint.h>
#include "openair1/PHY/LTE_TRANSPORT/defs.h"
#include "UE_MAC_interface.h"

#define MAX_NUM_DL_PDU 100
#define MAX_NUM_UL_PDU 100
#define MAX_NUM_HI_DCI0_PDU 100
#define MAX_NUM_TX_REQUEST_PDU 100

#define MAX_NUM_HARQ_IND 100
#define MAX_NUM_CRC_IND 100
#define MAX_NUM_SR_IND 100
#define MAX_NUM_CQI_IND 100
#define MAX_NUM_RACH_IND 100
#define MAX_NUM_SRS_IND 100

typedef struct{
  /// Module ID
  module_id_t module_id;
  /// CC ID
  int CC_id;
  // / frame
  frame_t frame;
  /// subframe
  sub_frame_t subframe;

  /// harq ACKs indication list
  //UE_MAC_hi_indication_body_t UE_hi_ind;

  /// crc indication list
  //UE_MAC_crc_indication_body_t UE_crc_ind;

  /// RX BCH indication
  UE_MAC_BCH_indication_body_t UE_BCH_ind;

  /// RX DLSCH indication
  UE_MAC_DLSCH_indication_body_t UE_DLSCH_ind;

} UE_DL_IND_t;


typedef struct{
	/// Module ID
	module_id_t module_id;
	/// CC ID
	int CC_id;
	/// frame
	frame_t frame;
	/// subframe
	sub_frame_t subframe;
	/// Txon Indication type (Msg1 or Msg3)
	uint8_t ind_type;
}UE_Tx_IND_t;


typedef struct{
	/// Module ID
	module_id_t module_id;
	/// CC ID
	int CC_id;
	/// frame
	frame_t frame;
	/// subframe
	sub_frame_t subframe;
	/// Sidelink Control Information indication
	ue_sci_indication_body_t UE_SCI_ind;
    /// RX SLSCH indication
	ue_SLSCH_indication_body_t UE_SLSCH_ind;
	/// RX SLDCH indication
	ue_SLDCH_indication_body_t UE_SLDCH_ind;
	/// RX SLBCH indication
	ue_SLBCH_indication_body_t UE_SLBCH_ind;

} UE_SL_IND_t;

// Downlink subframe P7


typedef struct{
  /// Module ID
  module_id_t module_id; 
  /// CC ID
  uint8_t CC_id;
  /// frame
  frame_t frame;
  /// subframe
  sub_frame_t subframe;
  /// UE_Mode to be filled only after
  UE_MODE_t UE_mode[NUMBER_OF_CONNECTED_eNB_MAX];
  /// MAC IFace UL Config Request
  UE_MAC_ul_config_request_t *UE_UL_req;
  /// MAC IFace SL Transmission Config Request
  UE_MAC_sl_config_request_Tx_t *SL_Tx_req;
  /// MAC IFace SL Reception Config Request
  UE_MAC_sl_config_request_Rx_t *SL_Rx_req;
  /// Pointers to UL SDUs
  UE_MAC_tx_request_t *UE_TX_req;
  /// Pointers to SL SDUs
  UE_MAC_sl_tx_request_t *TX_SL_req;
}UE_Sched_Rsp_t;

typedef struct {
    uint8_t Mod_id;
    int CC_id;
    UE_PHY_config_common_request_t *cfg_common;
    UE_PHY_config_dedicated_request_t *cfg_dedicated;
}UE_PHY_Config_t;

typedef struct IF_Module_UE_s{
//define the function pointer
  void (*UE_DL_indication)(UE_DL_IND_t *UE_DL_INFO);
  void (*UE_SL_indication)(UE_SL_IND_t *UE_SL_INFO);
  void (*UE_Tx_indication)(UE_Tx_IND_t *UE_Tx_INFO);
  void (*UE_sched_response)(UE_Sched_Rsp_t *UE_Sched_INFO);
  void (*UE_config_req)(UE_PHY_Config_t* UE_config_INFO);
//P: Perhaps an additional separate function for dedicated PHY configuration is needed.
  //uint32_t CC_mask_ue;
  uint16_t current_frame;
  uint8_t current_subframe;
  pthread_mutex_t if_mutex;
}IF_Module_UE_t;


IF_Module_UE_t *IF_Module_UE_init(int Mod_id);
void IF_Module_UE_kill(int Mod_id);


void UE_DL_indication(UE_DL_IND_t *UE_DL_INFO);


void UE_Tx_indication(UE_Tx_IND_t *UE_Tx_INFO);


/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void UE_schedule_response(UE_Sched_Rsp_t *UE_Sched_INFO);

#endif

