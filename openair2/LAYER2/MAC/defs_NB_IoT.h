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
/*! \file LAYER2/MAC/defs.h
* \brief MAC data structures, constant, and function prototype
* \author Navid Nikaein and Raymond Knopp
* \date 2011
* \version 0.5
* \email navid.nikaein@eurecom.fr
*/
/** @defgroup _oai2  openair2 Reference Implementation
 * @ingroup _ref_implementation_
 * @{
 */
/*@}*/
#ifndef __LAYER2_MAC_DEFS_NB_IOT_H__
#define __LAYER2_MAC_DEFS_NB_IOT_H__
#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif
//#include "COMMON/openair_defs.h"
#include "COMMON/platform_constants.h"
#include "COMMON/mac_rrc_primitives.h"
//#include "PHY/defs.h"
#include "PHY/defs_NB_IoT.h"
#include "RadioResourceConfigCommonSIB-NB-r13.h"
#include "RadioResourceConfigDedicated-NB-r13.h"
#include "RACH-ConfigCommon-NB-r13.h"
#include "MasterInformationBlock-NB.h"
#include "BCCH-BCH-Message-NB.h"
#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"
//#include "defs.h"
//#ifdef PHY_EMUL
//#include "SIMULATION/PHY_EMULATION/impl_defs.h"
//#endif
/** @defgroup _mac  MAC
 * @ingroup _oai2
 * @{
 */
#define SCH_PAYLOAD_SIZE_MAX_NB_IoT 4096

#define CCCH_PAYLOAD_SIZE_MAX_NB_IoT 128
/*!\brief Maximum number of random access process */
#define RA_PROC_MAX_NB_IoT 4
/*!\brief Maximum number of logical channl group IDs */
#define MAX_NUM_LCGID_NB_IoT 4
/*!\brief Maximum number of logical chanels */
#define MAX_NUM_LCID_NB_IoT 11

/*!\brief  UE ULSCH scheduling states*/
typedef enum {
  S_UL_NONE_NB_IoT =0,
  S_UL_WAITING_NB_IoT,
  S_UL_SCHEDULED_NB_IoT,
  S_UL_BUFFERED_NB_IoT,
  S_UL_NUM_STATUS_NB_IoT
} UE_ULSCH_STATUS_NB_IoT;

/*!\brief  UE DLSCH scheduling states*/
typedef enum {
  S_DL_NONE_NB_IoT =0,
  S_DL_WAITING_NB_IoT,
  S_DL_SCHEDULED_NB_IoT,
  S_DL_BUFFERED_NB_IoT,
  S_DL_NUM_STATUS_NB_IoT
} UE_DLSCH_STATUS_NB_IoT;

/*! \brief temporary struct for ULSCH sched */
typedef struct {
  rnti_t rnti;
  uint16_t subframe;
  uint16_t serving_num;
  UE_ULSCH_STATUS_NB_IoT status;
} eNB_ULSCH_INFO_NB_IoT;
/*! \brief temp struct for DLSCH sched */
typedef struct {
  rnti_t rnti;
  uint16_t weight;
  uint16_t subframe;
  uint16_t serving_num;
  UE_DLSCH_STATUS_NB_IoT status;
} eNB_DLSCH_INFO_NB_IoT;

/*! \brief Downlink SCH PDU Structure */
typedef struct {
  int8_t payload[8][SCH_PAYLOAD_SIZE_MAX_NB_IoT];
  uint16_t Pdu_size[8];
} __attribute__ ((__packed__)) DLSCH_PDU_NB_IoT;
/*! \brief eNB template for UE context information  */
typedef struct {
  /// C-RNTI of UE
  rnti_t rnti;
  /// NDI from last scheduling
  uint8_t oldNDI[8];
  /// NDI from last UL scheduling
  uint8_t oldNDI_UL[8];
  /// Flag to indicate UL has been scheduled at least once
  boolean_t ul_active;
  /// Flag to indicate UE has been configured (ACK from RRCConnectionSetup received)
  boolean_t configured;
  /// MCS from last scheduling
  uint8_t mcs[8];
  // PHY interface infoerror
  /// DCI format for DLSCH
  uint16_t DLSCH_dci_fmt;
  /// Current Aggregation Level for DCI
  uint8_t DCI_aggregation_min;
  /// size of DLSCH size in bit
  uint8_t DLSCH_dci_size_bits;
  /// DCI buffer for DLSCH
  /* rounded to 32 bits unit (actual value should be 8 due to the logic
   * of the function generate_dci0) */
  // need to modify
  uint8_t DLSCH_DCI[8][(((MAX_DCI_SIZE_BITS_NB_IoT)+31)>>5)*4];
  /// pre-assigned MCS by the ulsch preprocessorerror
  uint8_t pre_assigned_mcs_ul;
  /// assigned MCS by the ulsch scheduler
  uint8_t assigned_mcs_ul;
  /// DCI buffer for ULSCH
  /* rounded to 32 bits unit (actual value should be 8 due to the logic
   * of the function generate_dci0) */
  // need to modify
  uint8_t ULSCH_DCI[8][(((MAX_DCI_SIZE_BITS_NB_IoT)+31)>>5)*4];
  // Logical channel info for link with RLC
  /// Last received UE BSR info for each logical channel group id
  uint8_t bsr_info[MAX_NUM_LCGID_NB_IoT];
  /// phr information, received from DPR MAC control element
  int8_t phr_info;
  /// phr information, received from DPR MAC control element
  int8_t phr_info_configured;
  ///dl buffer info
  uint32_t dl_buffer_info[MAX_NUM_LCID_NB_IoT];
  /// total downlink buffer info
  uint32_t dl_buffer_total;
  /// total downlink pdus
  uint32_t dl_pdus_total;
  /// downlink pdus for each LCID
  uint32_t dl_pdus_in_buffer[MAX_NUM_LCID_NB_IoT];
  /// creation time of the downlink buffer head for each LCID
  uint32_t dl_buffer_head_sdu_creation_time[MAX_NUM_LCID_NB_IoT];
  /// maximum creation time of the downlink buffer head across all LCID
  uint32_t  dl_buffer_head_sdu_creation_time_max;
  /// a flag indicating that the downlink head SDU is segmented
  uint8_t    dl_buffer_head_sdu_is_segmented[MAX_NUM_LCID_NB_IoT];
  /// size of remaining size to send for the downlink head SDU
  uint32_t dl_buffer_head_sdu_remaining_size_to_send[MAX_NUM_LCID_NB_IoT];
  /// total uplink buffer size
  uint32_t ul_total_buffer;
  /// uplink buffer creation time for each LCID
  uint32_t ul_buffer_creation_time[MAX_NUM_LCGID_NB_IoT];
  /// maximum uplink buffer creation time across all the LCIDs
  uint32_t ul_buffer_creation_time_max;
  /// uplink buffer size per LCID
  uint32_t ul_buffer_info[MAX_NUM_LCGID_NB_IoT];
  /// UE tx power
  int32_t ue_tx_power;
} UE_TEMPLATE_NB_IoT;
/*! \brief eNB statistics for the connected UEs*/
typedef struct {
  /// CRNTI of UE
  rnti_t crnti; ///user id (rnti) of connected UEs
  // rrc status
  uint8_t rrc_status;
  /// harq pid
  uint8_t harq_pid;
  /// harq rounf
  uint8_t harq_round;
  /// DL Wideband CQI index (2 TBs)
  uint8_t dl_cqi;
  /// total available number of PRBs for a new transmission
  uint16_t rbs_used;
  /// total available number of PRBs for a retransmission
  uint16_t rbs_used_retx;
  /// total nccc used for a new transmission: num control channel element
  uint16_t ncce_used;
  /// total avilable nccc for a retransmission: num control channel element
  uint16_t ncce_used_retx;
  // mcs1 before the rate adaptaion
  uint8_t dlsch_mcs1;
  /// Target mcs2 after rate-adaptation
  uint8_t dlsch_mcs2;
  //  current TBS with mcs2
  uint32_t TBS;
  //  total TBS with mcs2
  //  uint32_t total_TBS;
  //  total rb used for a new transmission
  uint32_t total_rbs_used;
  //  total rb used for retransmission
  uint32_t total_rbs_used_retx;
   /// TX
  /// Num pkt
  uint32_t num_pdu_tx[NB_RB_MAX];
  /// num bytes
  uint32_t num_bytes_tx[NB_RB_MAX];
  /// num retransmission / harq
  uint32_t num_retransmission;
  /// instantaneous tx throughput for each TTI
  //  uint32_t tti_throughput[NB_RB_MAX];
  /// overall
  //
  uint32_t  dlsch_bitrate;
  //total
  uint32_t  total_dlsch_bitrate;
  /// headers+ CE +  padding bytes for a MAC PDU
  uint64_t overhead_bytes;
  /// headers+ CE +  padding bytes for a MAC PDU
  uint64_t total_overhead_bytes;
  /// headers+ CE +  padding bytes for a MAC PDU
  uint64_t avg_overhead_bytes;
  // MAC multiplexed payload
  uint64_t total_sdu_bytes;
  // total MAC pdu bytes
  uint64_t total_pdu_bytes;
  // total num pdu
  uint32_t total_num_pdus;
  //
  //  uint32_t avg_pdu_size;
  /// RX
  /// preassigned mcs after rate adaptation
  uint8_t ulsch_mcs1;
  /// adjusted mcs
  uint8_t ulsch_mcs2;
  /// estimated average pdu inter-departure time
  uint32_t avg_pdu_idt;
  /// estimated average pdu size
  uint32_t avg_pdu_ps;
  ///
  uint32_t aggregated_pdu_size;
  uint32_t aggregated_pdu_arrival;
  ///  uplink transport block size
  uint32_t ulsch_TBS;
  ///  total rb used for a new uplink transmission
  uint32_t num_retransmission_rx;
  ///  total rb used for a new uplink transmission
  uint32_t rbs_used_rx;
   ///  total rb used for a new uplink retransmission
  uint32_t rbs_used_retx_rx;
  ///  total rb used for a new uplink transmission
  uint32_t total_rbs_used_rx;
  /// normalized rx power
  int32_t      normalized_rx_power;
   /// target rx power
  int32_t    target_rx_power;
  /// num rx pdu
  uint32_t num_pdu_rx[NB_RB_MAX];
  /// num bytes rx
  uint32_t num_bytes_rx[NB_RB_MAX];
  /// instantaneous rx throughput for each TTI
  //  uint32_t tti_goodput[NB_RB_MAX];
  /// errors
  uint32_t num_errors_rx;
  uint64_t overhead_bytes_rx;
  /// headers+ CE +  padding bytes for a MAC PDU
  uint64_t total_overhead_bytes_rx;
  /// headers+ CE +  padding bytes for a MAC PDU
  uint64_t avg_overhead_bytes_rx;
 //
  uint32_t  ulsch_bitrate;
  //total
  uint32_t  total_ulsch_bitrate;
  /// overall
  ///  MAC pdu bytes
  uint64_t pdu_bytes_rx;
  /// total MAC pdu bytes
  uint64_t total_pdu_bytes_rx;
  /// total num pdu
  uint32_t total_num_pdus_rx;
  /// num of error pdus
  uint32_t total_num_errors_rx;
} eNB_UE_STATS_NB_IoT;
/*! \brief scheduling control information set through an API (not used)*/
typedef struct {
  ///UL transmission bandwidth in RBs
  uint8_t ul_bandwidth[MAX_NUM_LCID_NB_IoT];
  ///DL transmission bandwidth in RBs
  uint8_t dl_bandwidth[MAX_NUM_LCID_NB_IoT];
  //To do GBR bearer
  uint8_t min_ul_bandwidth[MAX_NUM_LCID_NB_IoT];
  uint8_t min_dl_bandwidth[MAX_NUM_LCID_NB_IoT];
  ///aggregated bit rate of non-gbr bearer per UE
  uint64_t  ue_AggregatedMaximumBitrateDL;
  ///aggregated bit rate of non-gbr bearer per UE
  uint64_t  ue_AggregatedMaximumBitrateUL;
  ///CQI scheduling interval in subframes.
  //Delete uint16_t cqiSchedInterval;
  ///Contention resolution timer used during random access
  uint8_t mac_ContentionResolutionTimer;
  //Delete uint16_t max_allowed_rbs[MAX_NUM_LCID];
  uint8_t max_mcs[MAX_NUM_LCID_NB_IoT];
  uint16_t priority[MAX_NUM_LCID_NB_IoT];
  // resource scheduling information
  uint8_t       harq_pid[MAX_NUM_CCs];
  uint8_t       round[MAX_NUM_CCs];
  uint8_t       dl_pow_off[MAX_NUM_CCs];
  //Delete uint16_t      pre_nb_available_rbs[MAX_NUM_CCs];
  //Delete unsigned char rballoc_sub_UE[MAX_NUM_CCs][N_RBG_MAX];
  uint16_t      ta_timer;
  int16_t       ta_update;
  int32_t       context_active_timer;
  //Delete int32_t       cqi_req_timer;
  int32_t       ul_inactivity_timer;
  int32_t       ul_failure_timer;
  int32_t       ul_scheduled;
  int32_t       ra_pdcch_order_sent;
  int32_t       ul_out_of_sync;
  int32_t       phr_received;// received from Msg3 MAC Control Element
} UE_sched_ctrl_NB_IoT;

/*! \brief UE list used by eNB to order UEs/CC for scheduling*/
typedef struct {
  /// DLSCH pdu
  DLSCH_PDU_NB_IoT DLSCH_pdu[MAX_NUM_CCs][2][NUMBER_OF_UE_MAX_NB_IoT];
  /// DCI template and MAC connection parameters for UEs
  UE_TEMPLATE_NB_IoT UE_template[MAX_NUM_CCs][NUMBER_OF_UE_MAX_NB_IoT];
  /// DCI template and MAC connection for RA processes
  int pCC_id[NUMBER_OF_UE_MAX_NB_IoT];
  /// eNB to UE statistics
  eNB_UE_STATS_NB_IoT eNB_UE_stats[MAX_NUM_CCs][NUMBER_OF_UE_MAX_NB_IoT];
  /// scheduling control info
  UE_sched_ctrl_NB_IoT UE_sched_ctrl[NUMBER_OF_UE_MAX_NB_IoT];

  /// sorted downlink component carrier for the scheduler 
  int ordered_CCids[MAX_NUM_CCs][NUMBER_OF_UE_MAX_NB_IoT];
  /// number of downlink active component carrier 
  int numactiveCCs[NUMBER_OF_UE_MAX_NB_IoT];
  /// sorted uplink component carrier for the scheduler 
  int ordered_ULCCids[MAX_NUM_CCs][NUMBER_OF_UE_MAX_NB_IoT];
  /// number of uplink active component carrier 
  int numactiveULCCs[NUMBER_OF_UE_MAX_NB_IoT];

  int next[NUMBER_OF_UE_MAX_NB_IoT];
  int head;
  int next_ul[NUMBER_OF_UE_MAX_NB_IoT];
  int head_ul;
  int avail;
  int num_UEs;
  boolean_t active[NUMBER_OF_UE_MAX_NB_IoT];
} UE_list_NB_IoT_t;

/*!\brief MCCH logical channel */
#define MCCH_NB_IoT 4 
/*!\brief The power headroom reporting range is from -23 ...+40 dB and beyond, with step 1 */
#define PHR_MAPPING_OFFSET_NB_IoT 23  // if ( x>= -23 ) val = floor (x + 23) 
/*!\brief Values of BCCH logical channel */
#define BCCH_NB_IoT 3  // SI 
/*!\brief Value of CCCH / SRB0 logical channel */
#define CCCH_NB_IoT 0  // srb0
/*!\brief DCCH / SRB1 logical channel */
#define DCCH_NB_IoT 1  // srb1
/*!\brief Values of BCCH0 logical channel for MIB*/
#define BCCH0_NB_IoT 11 // MIB-NB
/*!\brief Values of BCCH1 logical channel for SIBs */
#define BCCH1_NB_IoT 12 // SI-SIB-NBs
/*!\brief Values of PCCH logical channel */
#define PCCH_NB_IoT 13  // Paging XXX not used for the moment
/*!\brief Value of CCCH / SRB0 logical channel */
#define CCCH_NB_IoT 0  // srb0 ---> XXX exactly the same as in LTE (commented for compilation purposes)
/*!\brief DCCH0 / SRB1bis logical channel */
#define DCCH0_NB_IoT 3  // srb1bis
/*!\brief DCCH1 / SRB1  logical channel */
#define DCCH1_NB_IoT 1 // srb1 //XXX we redefine it for the SRB1
/*!\brief DTCH0 DRB0  logical channel */
#define DTCH0_NB_IoT 4 // DRB0
/*!\brief DTCH1 DRB1  logical channel */
#define DTCH1_NB_IoT 5 // DRB1
/*!\brief size of buffer status report table */
#define BSR_TABLE_SIZE_NB_IoT 64
/*!\brief LCID of short BSR for ULSCH */
#define SHORT_BSR_NB_IoT 29
/*!\brief LCID of CRNTI for ULSCH */
#define CRNTI_NB_IoT 27
/*!\brief Maximum number od control elemenets */
#define MAX_NUM_CE_NB_IoT 5
/*!\brief LCID of power headroom for ULSCH */
#define POWER_HEADROOM_NB_IoT 26
/*!\brief LCID of extended power headroom for ULSCH */
#define EXTENDED_POWER_HEADROOM_NB_IoT 25
/*!\brief LCID of padding LCID for DLSCH */
#define SHORT_PADDING_NB_IoT 31
/*!\brief LCID of long BSR for ULSCH */
#define LONG_BSR_NB_IoT 30
/*!\brief LCID of truncated BSR for ULSCH */
#define TRUNCATED_BSR_NB_IoT 28

// DLSCH LCHAN ID all the same as NB-IoT
/*!\brief  DCI PDU filled by MAC for the PHY  */
/* 
 * eNB part 
 */ 
/*!\brief UE layer 2 status */
typedef enum {
  CONNECTION_OK_NB_IoT=0,
  CONNECTION_LOST_NB_IoT,
  PHY_RESYNCH_NB_IoT,
  PHY_HO_PRACH_NB_IoT
} UE_L2_STATE_NB_IoT_t;

/* 
 * UE/ENB common part 
 */ 
/*!\brief MAC header of Random Access Response for Random access preamble identifier (RAPID) for NB-IoT */
typedef struct {
  uint8_t RAPID:6;
  uint8_t T:1;
  uint8_t E:1;
} __attribute__((__packed__))RA_HEADER_RAPID_NB_IoT;

/*!\brief  MAC header of Random Access Response for backoff indicator (BI) for NB-IoT*/
typedef struct {
  uint8_t BI:4;
  uint8_t R:2;
  uint8_t T:1;
  uint8_t E:1;
} __attribute__((__packed__))RA_HEADER_BI_NB_IoT;

/*Seems not to do the packed of RAR pdu*/

/*!\brief  MAC subheader short with 7bit Length field */
typedef struct {
  uint8_t LCID:5;  // octet 1 LSB
  uint8_t E:1;
  uint8_t R:2;     // octet 1 MSB
  uint8_t L:7;     // octet 2 LSB
  uint8_t F:1;     // octet 2 MSB
} __attribute__((__packed__))SCH_SUBHEADER_SHORT_NB_IoT;
/*!\brief  MAC subheader long  with 15bit Length field */
typedef struct {
  uint8_t LCID:5;   // octet 1 LSB
  uint8_t E:1;
  uint8_t R:2;      // octet 1 MSB
  uint8_t L_MSB:7;
  uint8_t F:1;      // octet 2 MSB
  uint8_t L_LSB:8;
  uint8_t padding;
} __attribute__((__packed__))SCH_SUBHEADER_LONG_NB_IoT;
/*!\brief MAC subheader short without length field */
typedef struct {
  uint8_t LCID:5;
  uint8_t E:1;
  uint8_t R:2;
} __attribute__((__packed__))SCH_SUBHEADER_FIXED_NB_IoT;

/*!\brief  mac control element: short buffer status report for a specific logical channel group ID*/
typedef struct {
  uint8_t Buffer_size:6;  // octet 1 LSB
  uint8_t LCGID:2;        // octet 1 MSB
} __attribute__((__packed__))BSR_SHORT_NB_IoT;

/*!\TRUNCATED BSR and Long BSR is not supported in NB-IoT*/

/*!\brief  mac control element: timing advance  */
typedef struct {
  uint8_t TA:6;
  uint8_t R:2;
} __attribute__((__packed__))TIMING_ADVANCE_CMD_NB_IoT;
/*!\brief  mac control element: power headroom report  */
typedef struct {
  uint8_t PH:6;
  uint8_t R:2;
} __attribute__((__packed__))POWER_HEADROOM_CMD_NB_IoT;

typedef struct {
  uint8_t payload[BCCH_PAYLOAD_SIZE_MAX_NB_IoT] ;
} __attribute__((__packed__))BCCH_PDU_NB_IoT;
/*! \brief CCCH payload */
typedef struct {
  uint8_t payload[CCCH_PAYLOAD_SIZE_MAX_NB_IoT] ;
} __attribute__((__packed__))CCCH_PDU_NB_IoT;


/*! \brief eNB template for the Random access information */
typedef struct {
  /// Flag to indicate this process is active
  boolean_t RA_active;
  /// Size of DCI for RA-Response (bytes)
  uint8_t RA_dci_size_bytes1;
  /// Size of DCI for RA-Response (bits)
  uint8_t RA_dci_size_bits1;
  /// Actual DCI to transmit for RA-Response
  uint8_t RA_alloc_pdu1[(MAX_DCI_SIZE_BITS_NB_IoT>>3)+1];
  /// DCI format for RA-Response (should be N1 RAR)
  uint8_t RA_dci_fmt1;
  /// Size of DCI for Msg4/ContRes (bytes)
  uint8_t RA_dci_size_bytes2;
  /// Size of DCI for Msg4/ContRes (bits)
  uint8_t RA_dci_size_bits2;
  /// Actual DCI to transmit for Msg4/ContRes
  uint8_t RA_alloc_pdu2[(MAX_DCI_SIZE_BITS_NB_IoT>>3)+1];
  /// DCI format for Msg4/ContRes (should be 1A)
  uint8_t RA_dci_fmt2;
  /// Flag to indicate the eNB should generate RAR.  This is triggered by detection of PRACH
  uint8_t generate_rar;
  /// Subframe where preamble was received, Delete?
  uint8_t preamble_subframe;
  /// Subframe where Msg3 is to be sent
  uint8_t Msg3_subframe;
  /// Flag to indicate the eNB should generate Msg4 upon reception of SDU from RRC.  This is triggered by first ULSCH reception at eNB for new user.
  uint8_t generate_Msg4;
  /// Flag to indicate that eNB is waiting for ACK that UE has received Msg3.
  uint8_t wait_ack_Msg4;
  /// UE RNTI allocated during RAR
  rnti_t rnti;
  /// RA RNTI allocated from received PRACH
  uint16_t RA_rnti;
  /// Re-use preamble_index, but it would be subcarrier index (0-47)
  uint8_t preamble_index;
  /// Received UE Contention Resolution Identifier
  uint8_t cont_res_id[6];
  /// Timing offset indicated by PHY
  int16_t timing_offset;
  /// Timeout for RRC connection
  int16_t RRC_timer;
} RA_TEMPLATE_NB_IoT;
/*! \brief eNB common channels */
typedef struct {
  int                              physCellId;
  int                              p_eNB; //number of tx antenna port
  int							   p_rx_eNB; //number of Rx antenna port
  int                              Ncp;
  int							   Ncp_UL;
  int                              eutra_band;
  uint32_t                         dl_CarrierFreq;
  BCCH_BCH_Message_NB_t               *mib_NB_IoT;
  RadioResourceConfigCommonSIB_NB_r13_t   *radioResourceConfigCommon;
  ARFCN_ValueEUTRA_r9_t               ul_CarrierFreq;
  struct MasterInformationBlock_NB__operationModeInfo_r13 operationModeInfo;
  /// Outgoing DCI for PHY generated by eNB scheduler
  DCI_PDU_NB_IoT DCI_pdu;
  /// Outgoing BCCH pdu for PHY
  BCCH_PDU_NB_IoT BCCH_pdu;
  /// Outgoing BCCH DCI allocation
  uint32_t BCCH_alloc_pdu;
  /// Outgoing CCCH pdu for PHY
  CCCH_PDU_NB_IoT CCCH_pdu;
  RA_TEMPLATE_NB_IoT RA_template[RA_PROC_MAX_NB_IoT];
  /// Delete VRB map for common channels
  /// Delete MBSFN SubframeConfig
  /// Delete number of subframe allocation pattern available for MBSFN sync area
// #if defined(Rel10) || defined(Rel14)
  /// Delete MBMS Flag
  /// Delete Outgoing MCCH pdu for PHY
  /// Delete MCCH active flag
  /// Delete MCCH active flag
  /// Delete MTCH active flag
  /// Delete number of active MBSFN area
  /// Delete MBSFN Area Info
  /// Delete PMCH Config
  /// Delete MBMS session info list
  /// Delete Outgoing MCH pdu for PHY
// #endif
// #ifdef CBA
  /// Delete number of CBA groups
  /// Delete RNTI for each CBA group
  /// Delete MCS for each CBA group
// #endif
}COMMON_channels_NB_IoT_t;
/*! \brief eNB overall statistics */
typedef struct {
  /// num BCCH PDU per CC
  uint32_t total_num_bcch_pdu;
  /// BCCH buffer size
  uint32_t bcch_buffer;
  /// total BCCH buffer size
  uint32_t total_bcch_buffer;
  /// BCCH MCS
  uint32_t bcch_mcs;
  /// num CCCH PDU per CC
  uint32_t total_num_ccch_pdu;
  /// BCCH buffer size
  uint32_t ccch_buffer;
  /// total BCCH buffer size
  uint32_t total_ccch_buffertotal_ccch_buffer;
  /// BCCH MCS
  uint32_t ccch_mcs;
/// num active users
  uint16_t num_dlactive_UEs;
  ///  available number of PRBs for a give SF fixed in 1 in NB-IoT
  uint16_t available_prbs;
  /// total number of PRB available for the user plane fixed in 1 in NB-IoT
  uint32_t total_available_prbs;
  /// aggregation
  /// total avilable nccc : num control channel element
  uint16_t available_ncces;
  // only for a new transmission, should be extended for retransmission
  // current dlsch  bit rate for all transport channels
  uint32_t dlsch_bitrate;
  //
  uint32_t dlsch_bytes_tx;
  //
  uint32_t dlsch_pdus_tx;
  //
  uint32_t total_dlsch_bitrate;
  //
  uint32_t total_dlsch_bytes_tx;
  //
  uint32_t total_dlsch_pdus_tx;
  // here for RX
  //
  uint32_t ulsch_bitrate;
  //
  uint32_t ulsch_bytes_rx;
  //
  uint64_t ulsch_pdus_rx;
  uint32_t total_ulsch_bitrate;
  //
  uint32_t total_ulsch_bytes_rx;
  //
  uint32_t total_ulsch_pdus_rx;
  /// MAC agent-related stats
  /// total number of scheduling decisions
  int sched_decisions;
  /// missed deadlines
  int missed_deadlines;
} eNB_STATS_NB_IoT;
/*! \brief top level eNB MAC structure */
typedef struct {

  ///
  uint16_t Node_id;
  /// frame counter
  frame_t frame;
  /// subframe counter
  sub_frame_t subframe;
  /// Common cell resources
  COMMON_channels_NB_IoT_t common_channels[MAX_NUM_CCs];
  UE_list_NB_IoT_t UE_list;
  ///Delete subband bitmap configuration, no related CQI report
  // / Modify CCE table used to build DCI scheduling information
  int CCE_table[MAX_NUM_CCs][12];//180 khz for Anchor carrier
  ///  active flag for Other lcid
  uint8_t lcid_active[NB_RB_MAX];
  /// eNB stats
  eNB_STATS_NB_IoT eNB_stats[MAX_NUM_CCs];
  // MAC function execution peformance profiler
  /// processing time of eNB scheduler
  time_stats_t eNB_scheduler;
  /// processing time of eNB scheduler for SI
  time_stats_t schedule_si;
  /// processing time of eNB scheduler for Random access
  time_stats_t schedule_ra;
  /// processing time of eNB ULSCH scheduler
  time_stats_t schedule_ulsch;
  /// processing time of eNB DCI generation
  time_stats_t fill_DLSCH_dci;
  /// processing time of eNB MAC preprocessor
  time_stats_t schedule_dlsch_preprocessor;
  /// processing time of eNB DLSCH scheduler
  time_stats_t schedule_dlsch; // include rlc_data_req + MAC header + preprocessor
  /// Delete processing time of eNB MCH scheduler
  /// processing time of eNB ULSCH reception
  time_stats_t rx_ulsch_sdu; // include rlc_data_ind
} eNB_MAC_INST_NB_IoT;

/*!\brief Top level UE MAC structure */
typedef struct {
  uint16_t Node_id;
  /// RX frame counter
  frame_t     rxFrame;
  /// RX subframe counter
  sub_frame_t rxSubframe;
  /// TX frame counter
  frame_t     txFrame;
  /// TX subframe counter
  sub_frame_t txSubframe;
  /// C-RNTI of UE
  uint16_t crnti;
  /// C-RNTI of UE before HO
  rnti_t crnti_before_ho; ///user id (rnti) of connected UEs
  /// uplink active flag
  uint8_t ul_active;
  /// pointer to RRC PHY configuration
  RadioResourceConfigCommonSIB_t *radioResourceConfigCommon;
  /// pointer to RACH_ConfigDedicated (NULL when not active, i.e. upon HO completion or T304 expiry)
  struct RACH_ConfigDedicated *rach_ConfigDedicated;
  /// pointer to RRC PHY configuration
  struct PhysicalConfigDedicated *physicalConfigDedicated;
#if defined(Rel10) || defined(Rel14)
  /// pointer to RRC PHY configuration SCEll
  struct PhysicalConfigDedicatedSCell_r10 *physicalConfigDedicatedSCell_r10;
#endif
  /// pointer to TDD Configuration (NULL for FDD)
  TDD_Config_t *tdd_Config;
  /// Number of adjacent cells to measure
  uint8_t  n_adj_cells;
  /// Array of adjacent physical cell ids
  uint32_t adj_cell_id[6];
  /// Pointer to RRC MAC configuration
  MAC_MainConfig_t *macConfig;
  /// Pointer to RRC Measurement gap configuration
  MeasGapConfig_t  *measGapConfig;
  /// Pointers to LogicalChannelConfig indexed by LogicalChannelIdentity. Note NULL means LCHAN is inactive.
  //////////////////////////////////////////////////////LogicalChannelConfig_t *logicalChannelConfig[MAX_NUM_LCID];
  /// Scheduling Information
  /////////////////////////////////////////////UE_SCHEDULING_INFO_NB_IoT scheduling_info;
  /// Outgoing CCCH pdu for PHY
  CCCH_PDU_NB_IoT CCCH_pdu;
  /// Incoming DLSCH pdu for PHY
  //DLSCH_PDU DLSCH_pdu[NUMBER_OF_UE_MAX][2];
  /// number of attempt for rach
  uint8_t RA_attempt_number;
  /// Random-access procedure flag
  uint8_t RA_active;
  /// Random-access window counter
  int8_t RA_window_cnt;
  /// Random-access Msg3 size in bytes
  uint8_t RA_Msg3_size;
  /// Random-access prachMaskIndex
  uint8_t RA_prachMaskIndex;
  /// Flag indicating Preamble set (A,B) used for first Msg3 transmission
  uint8_t RA_usedGroupA;
  /// Random-access Resources
  /////////////////////////////////////////////////////////////////////PRACH_RESOURCES_NB_IoT_t RA_prach_resources;
  /// Random-access PREAMBLE_TRANSMISSION_COUNTER
  uint8_t RA_PREAMBLE_TRANSMISSION_COUNTER;
  /// Random-access backoff counter
  int16_t RA_backoff_cnt;
  /// Random-access variable for window calculation (frame of last change in window counter)
  uint32_t RA_tx_frame;
  /// Random-access variable for window calculation (subframe of last change in window counter)
  uint8_t RA_tx_subframe;
  /// Random-access Group B maximum path-loss
  /// Random-access variable for backoff (frame of last change in backoff counter)
  uint32_t RA_backoff_frame;
  /// Random-access variable for backoff (subframe of last change in backoff counter)
  uint8_t RA_backoff_subframe;
  /// Random-access Group B maximum path-loss
  uint16_t RA_maxPL;
  /// Random-access Contention Resolution Timer active flag
  uint8_t RA_contention_resolution_timer_active;
  /// Random-access Contention Resolution Timer count value
  uint8_t RA_contention_resolution_cnt;
  /// power headroom reporitng reconfigured
  uint8_t PHR_reconfigured;
  /// power headroom state as configured by the higher layers
  uint8_t PHR_state;
  /// power backoff due to power management (as allowed by P-MPRc) for this cell
  uint8_t PHR_reporting_active;
  /// power backoff due to power management (as allowed by P-MPRc) for this cell
  uint8_t power_backoff_db[NUMBER_OF_eNB_MAX];
  /// BSR report falg management
  uint8_t BSR_reporting_active;
  /// retxBSR-Timer expires flag
  uint8_t retxBSRTimer_expires_flag;
  /// periodBSR-Timer expires flag
  uint8_t periodBSRTimer_expires_flag;

  /// MBSFN_Subframe Configuration
  struct MBSFN_SubframeConfig *mbsfn_SubframeConfig[8]; // FIXME replace 8 by MAX_MBSFN_AREA?
  /// number of subframe allocation pattern available for MBSFN sync area
  uint8_t num_sf_allocation_pattern;
// #if defined(Rel10) || defined(Rel14)
//   /// number of active MBSFN area
//   uint8_t num_active_mbsfn_area;
//   /// MBSFN Area Info
//   struct  MBSFN_AreaInfo_r9 *mbsfn_AreaInfo[MAX_MBSFN_AREA];
//   /// PMCH Config
//   struct PMCH_Config_r9 *pmch_Config[MAX_PMCH_perMBSFN];
//   /// MCCH status
//   uint8_t mcch_status;
//   /// MSI status
//   uint8_t msi_status;// could be an array if there are >1 MCH in one MBSFN area
// #endif
  //#ifdef CBA
  /// CBA RNTI for each group 
  uint16_t cba_rnti[NUM_MAX_CBA_GROUP];
  /// last SFN for CBA channel access 
  uint8_t cba_last_access[NUM_MAX_CBA_GROUP];
  //#endif
  /// total UE scheduler processing time 
  time_stats_t ue_scheduler; // total
  /// UE ULSCH tx  processing time inlcuding RLC interface (rlc_data_req) and mac header generation 
  time_stats_t tx_ulsch_sdu;  
  /// UE DLSCH rx  processing time inlcuding RLC interface (mac_rrc_data_ind or mac_rlc_status_ind+mac_rlc_data_ind) and mac header parser
  time_stats_t rx_dlsch_sdu ; 
  /// UE query for MCH subframe processing time 
  time_stats_t ue_query_mch;
  /// UE MCH rx processing time 
  time_stats_t rx_mch_sdu;
  /// UE BCCH rx processing time including RLC interface (mac_rrc_data_ind) 
  time_stats_t rx_si; 
  /// UE PCCH rx processing time including RLC interface (mac_rrc_data_ind) 
  time_stats_t rx_p; 
} UE_MAC_INST_NB_IoT;


#endif /*__LAYER2_MAC_DEFS_NB_IoT_H__ */
