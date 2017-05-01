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

#ifndef __LAYER2_MAC_DEFS_H__
#define __LAYER2_MAC_DEFS_H__



#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

//#include "COMMON/openair_defs.h"

#include "COMMON/platform_constants.h"
#include "COMMON/mac_rrc_primitives.h"
#include "PHY/defs.h"
#include "RadioResourceConfigCommon.h"
#include "RadioResourceConfigDedicated.h"
#include "MeasGapConfig.h"
#include "TDD-Config.h"
#include "RACH-ConfigCommon.h"
#include "MeasObjectToAddModList.h"
#include "MobilityControlInfo.h"
#if defined(Rel10) || defined(Rel14)
#include "MBSFN-AreaInfoList-r9.h"
#include "MBSFN-SubframeConfigList.h"
#include "PMCH-InfoList-r9.h"
#include "SCellToAddMod-r10.h"
#endif

//#ifdef PHY_EMUL
//#include "SIMULATION/PHY_EMULATION/impl_defs.h"
//#endif

/** @defgroup _mac  MAC
 * @ingroup _oai2
 * @{
 */

#define BCCH_PAYLOAD_SIZE_MAX 128
#define CCCH_PAYLOAD_SIZE_MAX 128
#define PCCH_PAYLOAD_SIZE_MAX 128

#define SCH_PAYLOAD_SIZE_MAX 4096
/// Logical channel ids from 36-311 (Note BCCH is not specified in 36-311, uses the same as first DRB)

/*!MBMS is not supported in NB-IoT */

#ifdef USER_MODE
#define printk printf
#endif //USER_MODE

/*In NB-IoT, 36.321 6.1.3.1 Logical channel group ID is set to #0 */
/*!\brief Maximum number of logical channel group IDs */
#define MAX_NUM_LCGID 4
/*!\brief logical channl group ID 0 */
#define LCGID0 0

/*!\brief Maximum number of logical chanels 0-10*/
#define MAX_NUM_LCID 11
/*!\brief Maximum number od control elemenets  */
#define MAX_NUM_CE 5
/*!\brief Maximum number of random access process */
#define NB_RA_PROC_MAX 4
/*!\brief size of buffer status report table */
#define BSR_TABLE_SIZE 64
/*!\brief The power headroom reporting range is from -23 ...+40 dB and beyond, with step 1 */
#define PHR_MAPPING_OFFSET 23  // if ( x>= -23 ) val = floor (x + 23)
	
/*There is no CQI & RB concept in NB-IoT*/	
/*!\brief maximum number of resource block groups, not used in NB-IoT */
/*!\brief minimum value for channel quality indicator, not used in NB-IoT */
/*!\brief maximum value for channel quality indicator, not used in NB-IoT */

/*!\brief maximum number of supported bandwidth (1.4, 5, 10, 20 MHz) */
#define MAX_SUPPORTED_BW  4  
/*!\brief CQI values range from 1 to 15 (4 bits), not used in NB-IoT */

/*!\brief value for indicating BSR Timer is not running */
#define MAC_UE_BSR_TIMER_NOT_RUNNING   (0xFFFF)

#define LCID_EMPTY 0
#define LCID_NOT_EMPTY 1

/*!\brief minimum RLC PDU size to be transmitted = min RLC Status PDU or RLC UM PDU SN 5 bits */
#define MIN_RLC_PDU_SIZE    (2)

/*!\brief minimum MAC data needed for transmitting 1 min RLC PDU size + 1 byte MAC subHeader */
#define MIN_MAC_HDR_RLC_SIZE    (1 + MIN_RLC_PDU_SIZE)

/*!\brief maximum number of slices / groups */
#define MAX_NUM_SLICES 4 

/* 
 * eNB part 
 */ 


/* 
 * UE/ENB common part 
 */ 
/*!\brief MAC header of Random Access Response for Random access preamble identifier (RAPID) for NB-IoT */
typedef struct {
  uint8_t RAPID:6;
  uint8_t T:1;
  uint8_t E:1;
} __attribute__((__packed__))RA_HEADER_RAPID_NB;

/*!\brief  MAC header of Random Access Response for backoff indicator (BI) for NB-IoT*/
typedef struct {
  uint8_t BI:4;
  uint8_t R:2;
  uint8_t T:1;
  uint8_t E:1;
} __attribute__((__packed__))RA_HEADER_BI_NB;

/*Seems not to do the packed of RAR pdu*/

/*!\brief  MAC subheader short with 7bit Length field */
typedef struct {
  uint8_t LCID:5;  // octet 1 LSB
  uint8_t E:1;
  uint8_t R:2;     // octet 1 MSB
  uint8_t L:7;     // octet 2 LSB
  uint8_t F:1;     // octet 2 MSB
} __attribute__((__packed__))SCH_SUBHEADER_SHORT_NB;
/*!\brief  MAC subheader long  with 15bit Length field */
typedef struct {
  uint8_t LCID:5;   // octet 1 LSB
  uint8_t E:1;
  uint8_t R:2;      // octet 1 MSB
  uint8_t L_MSB:7;
  uint8_t F:1;      // octet 2 MSB
  uint8_t L_LSB:8;
  uint8_t padding;
} __attribute__((__packed__))SCH_SUBHEADER_LONG_NB;
/*!\brief MAC subheader short without length field */
typedef struct {
  uint8_t LCID:5;
  uint8_t E:1;
  uint8_t R:2;
} __attribute__((__packed__))SCH_SUBHEADER_FIXED_NB;

/*!\brief  mac control element: short buffer status report for a specific logical channel group ID*/
typedef struct {
  uint8_t Buffer_size:6;  // octet 1 LSB
  uint8_t LCGID:2;        // octet 1 MSB
} __attribute__((__packed__))BSR_SHORT_NB;

/*!\TRUNCATED BSR and Long BSR is not supported in NB-IoT*/

/*!\brief  mac control element: timing advance  */
typedef struct {
  uint8_t TA:6;
  uint8_t R:2;
} __attribute__((__packed__))TIMING_ADVANCE_CMD_NB;
/*!\brief  mac control element: power headroom report  */
typedef struct {
  uint8_t PH:6;
  uint8_t R:2;
} __attribute__((__packed__))POWER_HEADROOM_CMD_NB;

/*!\brief  DCI PDU filled by MAC for the PHY  */
typedef struct {
  uint8_t Num_ue_spec_dci ;
  uint8_t Num_common_dci  ;
  //  uint32_t nCCE;
  uint32_t num_pdcch_symbols;
  DCI_ALLOC_t dci_alloc[NUM_DCI_MAX] ;
} DCI_PDU_NB;
/*! \brief CCCH payload */
typedef struct {
  uint8_t payload[CCCH_PAYLOAD_SIZE_MAX] ;
} __attribute__((__packed__))CCCH_PDU_NB;
/*! \brief BCCH payload */
typedef struct {
  uint8_t payload[BCCH_PAYLOAD_SIZE_MAX] ;
} __attribute__((__packed__))BCCH_PDU_NB;
/*! \brief PCCH payload */
typedef struct {
  uint8_t payload[PCCH_PAYLOAD_SIZE_MAX] ;
} __attribute__((__packed__))PCCH_PDU_NB;


/*MCCH & CC is not used in NB-IoT*/


/*! \brief Values of CCCH LCID for DLSCH */ 
//#define CCCH_LCHANID 0 not used in OAI 
/*!\brief Values of BCCH0 logical channel for MIB*/
#define BCCH0 11 // MIB-NB 
/*!\brief Values of BCCH1 logical channel for SIBs */
#define BCCH1 12 // SI-SIB-NBs 
/*!\brief Values of PCCH logical channel */
#define PCCH 13  // Paging 
/*!\brief Value of CCCH / SRB0 logical channel */
#define CCCH 0  // srb0
/*!\brief DCCH0 / SRB1bis logical channel */
#define DCCH0 3  // srb1bis
/*!\brief DCCH1 / SRB1  logical channel */
#define DCCH1 1 // srb1
/*!\brief DTCH0 DRB0  logical channel */
#define DTCH0 4 // DRB0
/*!\brief DTCH1 DRB1  logical channel */
#define DTCH1 5 // DRB1

// DLSCH LCHAN ID all the same as NB-IoT
/*!\brief LCID of UE contention resolution identity for DLSCH*/
#define UE_CONT_RES 28
/*!\brief LCID of timing advance for DLSCH */
#define TIMING_ADV_CMD 29
/*!\brief LCID of discontinous reception mode for DLSCH */
#define DRX_CMD 30
/*!\brief LCID of padding LCID for DLSCH */
#define SHORT_PADDING 31

//MCH/CC not defined in NB-IoT

// ULSCH LCHAN IDs the EXTENDED_POWER_HEADROOM POWER_HEADROOM TRUNCATED_BSR LONG_BSR is not used in NB-IoT
/*!\brief LCID of CRNTI for ULSCH */
#define CRNTI 27
/*!\brief LCID of short BSR for ULSCH */
#define SHORT_BSR 29

/*!\bitmaps for BSR Triggers */
#define	BSR_TRIGGER_NONE		(0)			/* No BSR Trigger */
#define	BSR_TRIGGER_REGULAR		(1)			/* For Regular and ReTxBSR Expiry Triggers */
#define	BSR_TRIGGER_PERIODIC	(2)			/* For BSR Periodic Timer Expiry Trigger */
#define	BSR_TRIGGER_PADDING		(4)			/* For Padding BSR Trigger */


/*! \brief Downlink SCH PDU Structure */
typedef struct {
  int8_t payload[8][SCH_PAYLOAD_SIZE_MAX];
  uint16_t Pdu_size[8];
} __attribute__ ((__packed__)) DLSCH_PDU_NB;

/*MCH is not defined in NB-IoT*/

/*! \brief Uplink SCH PDU Structure */
typedef struct {
  int8_t payload[SCH_PAYLOAD_SIZE_MAX];         /*!< \brief SACH payload */
  uint16_t Pdu_size;
} __attribute__ ((__packed__)) ULSCH_PDU_NB;

#include "PHY/impl_defs_top.h"

/*!\brief  UE ULSCH scheduling states*/
typedef enum {
  S_UL_NONE =0,// used in rrc_mac_remove_ue
  S_UL_WAITING,// used in add_new_ue
  S_UL_SCHEDULED,// used in scheudle_ulsch_rnti
  S_UL_BUFFERED,// not used
  S_UL_NUM_STATUS// not used
} UE_ULSCH_STATUS_NB;

/*!\brief  UE DLSCH scheduling states*/
typedef enum {
  S_DL_NONE =0,//used in rrc_mac_remove_ue, scheudle_ue_spec,init_ue_sched_info 
  S_DL_WAITING,//used in schedule_next_dlue,fill_DLSCH_dci,add_new_ue
  S_DL_SCHEDULED,//used in scheudle_ue_spec,fill_DLSCH_dci
  S_DL_BUFFERED,//used in schedule_next_dlue
  S_DL_NUM_STATUS// not used
} UE_DLSCH_STATUS_NB;

/*!\brief  scheduling policy for the contention-based access */
/*CBA is not defined in NB-IoT*/


/*! \brief temporary struct for ULSCH sched */
typedef struct {
  rnti_t rnti;
  uint16_t subframe;
  uint16_t serving_num;
  UE_ULSCH_STATUS status;
} eNB_ULSCH_INFO_NB;
/*! \brief temp struct for DLSCH sched */
typedef struct {
  rnti_t rnti;
  uint16_t weight;
  uint16_t subframe;
  uint16_t serving_num;
  UE_DLSCH_STATUS status;
} eNB_DLSCH_INFO_NB;
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

} eNB_STATS_NB;
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

} eNB_UE_STATS_NB;
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
  //Modify uint8_t mcs[8];

  /// TPC from last scheduling
  //Delete uint8_t oldTPC[8];

  // PHY interface info

  /// DCI format for DLSCH
  uint16_t DLSCH_dci_fmt;

  /// Current Aggregation Level for DCI
  uint8_t DCI_aggregation_min;

  /// size of DLSCH size in bit
  uint8_t DLSCH_dci_size_bits;

  /// DCI buffer for DLSCH
  /* rounded to 32 bits unit (actual value should be 8 due to the logic
   * of the function generate_dci0) */
  //Modifyuint8_t DLSCH_DCI[8][(((MAX_DCI_SIZE_BITS)+31)>>5)*4];

  /// Number of Allocated RBs for DL after scheduling (prior to frequency allocation)
  //Delete uint16_t nb_rb[8]; // num_max_harq

  /// Number of Allocated RBs for UL after scheduling (prior to frequency allocation)
  //Delete uint16_t nb_rb_ul[8]; // num_max_harq

  /// Number of Allocated RBs by the ulsch preprocessor
  //Delete uint8_t pre_allocated_nb_rb_ul;

  /// index of Allocated RBs by the ulsch preprocessor
  //Delete int8_t pre_allocated_rb_table_index_ul;

  /// total allocated RBs
  //Delete int8_t total_allocated_rbs;

  /// pre-assigned MCS by the ulsch preprocessor
  uint8_t pre_assigned_mcs_ul;

  /// assigned MCS by the ulsch scheduler
  uint8_t assigned_mcs_ul;

  /// DCI buffer for ULSCH
  /* rounded to 32 bits unit (actual value should be 8 due to the logic
   * of the function generate_dci0) */
  //Modify uint8_t ULSCH_DCI[8][(((MAX_DCI_SIZE_BITS)+31)>>5)*4];

  /// DL DAI
  //Delete uint8_t DAI;

  /// UL DAI
  //Delete uint8_t DAI_ul[10];

  /// UL Scheduling Request Received
  //Delete uint8_t ul_SR;

  /// Resource Block indication for each sub-band in MU-MIMO
  //Delete uint8_t rballoc_subband[8][50];

  // Logical channel info for link with RLC

  /// Last received UE BSR info for each logical channel group id
  uint8_t bsr_info[MAX_NUM_LCGID];

  /// LCGID mapping
  //Delete long lcgidmap[11];

  /// phr information, received from DPR MAC control element
  int8_t phr_info_DPR;

  /// phr information, received from DPR MAC control element
  int8_t phr_info_configured_DPR;

  ///dl buffer info
  uint32_t dl_buffer_info[MAX_NUM_LCID];
  /// total downlink buffer info
  uint32_t dl_buffer_total;
  /// total downlink pdus
  uint32_t dl_pdus_total;
  /// downlink pdus for each LCID
  uint32_t dl_pdus_in_buffer[MAX_NUM_LCID];
  /// creation time of the downlink buffer head for each LCID
  uint32_t dl_buffer_head_sdu_creation_time[MAX_NUM_LCID];
  /// maximum creation time of the downlink buffer head across all LCID
  uint32_t  dl_buffer_head_sdu_creation_time_max;
  /// a flag indicating that the downlink head SDU is segmented
  uint8_t    dl_buffer_head_sdu_is_segmented[MAX_NUM_LCID];
  /// size of remaining size to send for the downlink head SDU
  uint32_t dl_buffer_head_sdu_remaining_size_to_send[MAX_NUM_LCID];

  /// total uplink buffer size
  uint32_t ul_total_buffer;
  /// uplink buffer creation time for each LCID
  uint32_t ul_buffer_creation_time[MAX_NUM_LCGID];
  /// maximum uplink buffer creation time across all the LCIDs
  uint32_t ul_buffer_creation_time_max;
  /// uplink buffer size per LCID
  uint32_t ul_buffer_info[MAX_NUM_LCGID];

  /// UE tx power
  int32_t ue_tx_power;

  /// stores the frame where the last TPC was transmitted
  //Delete uint32_t pusch_tpc_tx_frame;
  //Delete uint32_t pusch_tpc_tx_subframe;
  //Delete uint32_t pucch_tpc_tx_frame;
  //Delete uint32_t pucch_tpc_tx_subframe;

//Delete eNB_UE_estimated_distances distance;

} UE_TEMPLATE_NB;

/*! \brief scheduling control information set through an API (not used)*/
typedef struct {
  ///UL transmission bandwidth in RBs
  uint8_t ul_bandwidth[MAX_NUM_LCID];
  ///DL transmission bandwidth in RBs
  uint8_t dl_bandwidth[MAX_NUM_LCID];

  //To do GBR bearer
  uint8_t min_ul_bandwidth[MAX_NUM_LCID];

  uint8_t min_dl_bandwidth[MAX_NUM_LCID];

  ///aggregated bit rate of non-gbr bearer per UE
  uint64_t  ue_AggregatedMaximumBitrateDL;
  ///aggregated bit rate of non-gbr bearer per UE
  uint64_t  ue_AggregatedMaximumBitrateUL;
  ///CQI scheduling interval in subframes.
  //Delete uint16_t cqiSchedInterval;
  ///Contention resolution timer used during random access
  uint8_t mac_ContentionResolutionTimer;

  //Delete uint16_t max_allowed_rbs[MAX_NUM_LCID];

  uint8_t max_mcs[MAX_NUM_LCID];

  uint16_t priority[MAX_NUM_LCID];

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
} UE_sched_ctrl_NB;
/*! \brief eNB template for the Random access information */
typedef struct {
  /// Flag to indicate this process is active
  boolean_t RA_active;
  /// Size of DCI for RA-Response (bytes)
  uint8_t RA_dci_size_bytes1;
  /// Size of DCI for RA-Response (bits)
  uint8_t RA_dci_size_bits1;
  /// Actual DCI to transmit for RA-Response
  uint8_t RA_alloc_pdu1[(MAX_DCI_SIZE_BITS>>3)+1];
  /// DCI format for RA-Response (should be 1A)
  uint8_t RA_dci_fmt1;
  /// Size of DCI for Msg4/ContRes (bytes)
  uint8_t RA_dci_size_bytes2;
  /// Size of DCI for Msg4/ContRes (bits)
  uint8_t RA_dci_size_bits2;
  /// Actual DCI to transmit for Msg4/ContRes
  uint8_t RA_alloc_pdu2[(MAX_DCI_SIZE_BITS>>3)+1];
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
  /// Delete Received preamble_index, use subcarrier index?

  /// Received UE Contention Resolution Identifier
  uint8_t cont_res_id[6];
  /// Timing offset indicated by PHY
  int16_t timing_offset;
  /// Timeout for RRC connection
  int16_t RRC_timer;
} RA_TEMPLATE_NB;


/*! \Delete struct SBMAP_CONF, brief subband bitmap confguration (for ALU icic algo purpose), in test phase */

/*! \brief UE list used by eNB to order UEs/CC for scheduling*/
typedef struct {
  /// DLSCH pdu
  DLSCH_PDU_NB DLSCH_pdu[MAX_NUM_CCs][2][NUMBER_OF_UE_MAX];
  /// DCI template and MAC connection parameters for UEs
  UE_TEMPLATE_NB UE_template[MAX_NUM_CCs][NUMBER_OF_UE_MAX];
  /// DCI template and MAC connection for RA processes
  int pCC_id[NUMBER_OF_UE_MAX];
  /// Delete sorted downlink component carrier for the scheduler

  /// Delete number of downlink active component carrier

  /// Delete sorted uplink component carrier for the scheduler

  /// Delete number of uplink active component carrier

  /// Delete number of downlink active component carrier

  /// eNB to UE statistics
  eNB_UE_STATS eNB_UE_stats[MAX_NUM_CCs][NUMBER_OF_UE_MAX];
  /// scheduling control info
  UE_sched_ctrl UE_sched_ctrl[NUMBER_OF_UE_MAX];

  int next[NUMBER_OF_UE_MAX];
  int head;
  int next_ul[NUMBER_OF_UE_MAX];
  int head_ul;
  int avail;
  int num_UEs;
  boolean_t active[NUMBER_OF_UE_MAX];
} UE_list_NB_t;

/*! \brief eNB common channels */
typedef struct {
  int                              physCellId;
  int                              p_eNB;
  int                              Ncp;
  int                              eutra_band;
  uint32_t                         dl_CarrierFreq;
  BCCH_BCH_Message_NB_t               *mib;
  RadioResourceConfigCommonSIB_NB_r13   *radioResourceConfigCommon;  
  ARFCN_ValueEUTRA_r9_t               ul_CarrierFreq;
  struct MasterInformationBlock_NB__operationModeInfo_r13_u operationModeInfo;
  
  
  /// Outgoing DCI for PHY generated by eNB scheduler
  DCI_PDU_NB DCI_pdu;
  /// Outgoing BCCH pdu for PHY
  BCCH_PDU_NB BCCH_pdu;
  /// Outgoing BCCH DCI allocation
  uint32_t BCCH_alloc_pdu;
  /// Outgoing CCCH pdu for PHY
  CCCH_PDU CCCH_pdu;
  RA_TEMPLATE_NB RA_template[NB_RA_PROC_MAX];
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
} COMMON_channels_NB_t;
/*! \brief top level eNB MAC structure */
typedef struct {
  ///
  uint16_t Node_id;
  /// frame counter
  frame_t frame;
  /// subframe counter
  sub_frame_t subframe;
  /// Common cell resources
  COMMON_channels_NB_t common_channels[MAX_NUM_CCs];
  UE_list_NB_t UE_list;

  ///Delete subband bitmap configuration, no related CQI report

  // / Modify CCE table used to build DCI scheduling information
  int CCE_table[MAX_NUM_CCs][12];//180 khz for Anchor carrier

  ///  active flag for Other lcid
  uint8_t lcid_active[NB_RB_MAX];
  /// eNB stats
  eNB_STATS eNB_stats[MAX_NUM_CCs];
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

} eNB_MAC_INST_NB;

/*
 * UE part
 */

/*!\brief UE layer 2 status */
typedef enum {
  CONNECTION_OK=0,
  CONNECTION_LOST,
  PHY_RESYNCH,
  PHY_HO_PRACH
} UE_L2_STATE_t_NB;

/*!\brief UE scheduling info */
typedef struct {
  /// buffer status for each lcgid
  uint8_t  BSR[MAX_NUM_LCGID]; // should be more for mesh topology
  /// keep the number of bytes in rlc buffer for each lcgid
  int32_t  BSR_bytes[MAX_NUM_LCGID];
  /// after multiplexing buffer remain for each lcid
  int32_t  LCID_buffer_remain[MAX_NUM_LCID];
  /// sum of all lcid buffer size
  uint16_t  All_lcid_buffer_size_lastTTI;
  /// buffer status for each lcid
  uint8_t  LCID_status[MAX_NUM_LCID];
  /// Delete SR pending as defined in 36.321

  /// Delete SR_COUNTER as defined in 36.321

  /// logical channel group ide for each LCID
  uint8_t  LCGID[MAX_NUM_LCID];
  /// retxBSR-Timer, default value is sf2560
  uint16_t retxBSR_Timer;
  /// retxBSR_SF, number of subframe before triggering a regular BSR
  uint16_t retxBSR_SF;
  /// periodicBSR-Timer, default to infinity
  uint16_t periodicBSR_Timer;
  /// periodicBSR_SF, number of subframe before triggering a periodic BSR
  uint16_t periodicBSR_SF;
  /// Delete sr_ProhibitTimer in MAC_MainConfig, default value is 0: not configured

  /// Delete sr ProhibitTime running

  ///  Delete maxHARQ_Tx in MAC_MainConfig, default value to n5

  /// delete ttiBundling in MAC_MainConfig, default value is false

  /// default value is release
  struct DRX_Config *drx_config;
  /// Delete phr_config in MAC_MainConfig, default value is release

  ///Delete timer before triggering a periodic PHR

  ///Delete timer before triggering a prohibit PHR

  ///DL Pathloss change value
  uint16_t PathlossChange;
  ///Delete number of subframe before triggering a periodic PHR

  ///Delete number of subframe before triggering a prohibit PHR

  ///DL Pathloss Change in db
  uint16_t PathlossChange_db;

  /// Delete extendedBSR_Sizes_r10, default value is false, only support short BSR

  /// Delete extendedPHR_r10, default value is false


  //For NB-IoT in TS 36.321, prioritisedBitRate, bucketSizeDuration and the corresponding steps of the Logical Channel Prioritisation procedure (i.e., Step 1 and Step 2 below) are not applicable.
  //Delete Bj bucket usage per  lcid,

  //Delete Bucket size per lcid

} UE_SCHEDULING_INFO_NB;
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
  /// Delete C-RNTI of UE before HO

  /// uplink active flag
  uint8_t ul_active;
  /// pointer to RRC PHY configuration
  RadioResourceConfigCommonSIB_NB_r13 *radioResourceConfigCommon;
  /// pointer to RACH_ConfigDedicated (NULL when not active, i.e. upon HO completion or T304 expiry)
  struct RACH_ConfigDedicated *rach_ConfigDedicated;
  /// pointer to RRC PHY configuration
  struct PhysicalConfigDedicated_NB_r13 *physicalConfigDedicated;
#if defined(Rel10) || defined(Rel14)
  /// Delete pointer to RRC PHY configuration SCEll

#endif
  /// Delete pointer to TDD Configuration (NULL for FDD)

  /// Delete Number of adjacent cells to measure

  /// Delete Array of adjacent physical cell ids

  /// Pointer to RRC MAC configuration
  MAC_MainConfig_NB_r13 *macConfig;
  /// Delete Pointer to RRC Measurement gap configuration

  /// Pointers to LogicalChannelConfig indexed by LogicalChannelIdentity. Note NULL means LCHAN is inactive.
  LogicalChannelConfig_NB_r13 *logicalChannelConfig[MAX_NUM_LCID];
  /// Scheduling Information
  UE_SCHEDULING_INFO_NB scheduling_info;
  /// Outgoing CCCH pdu for PHY
  CCCH_PDU_NB CCCH_pdu;
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
  /// delete Random-access prachMaskIndex, NB use subcarrier index

  /// Flag indicating Preamble set (A,B) used for first Msg3 transmission
  uint8_t RA_usedGroupA;
  /// Delete Random-access Resources, cause it use for ra_PreambleIndex and ra_RACH_MaskIndex.

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

  /// Delete MBSFN_Subframe Configuration

  /// Delete number of subframe allocation pattern available for MBSFN sync area


// #if defined(Rel10) || defined(Rel14)
  /// Delete number of active MBSFN area

  /// Delete MBSFN Area Info

  /// Delete PMCH Config

  /// Delete MCCH status

  /// Delete MSI status

// #endif
  /// Delete UE query for MCH subframe processing time
  //#ifdef CBA
  /// CBA RNTI for each group

  /// Delete last SFN for CBA channel access

  //#endif
  /// total UE scheduler processing time
  time_stats_t ue_scheduler; // total
  /// UE ULSCH tx  processing time inlcuding RLC interface (rlc_data_req) and mac header generation
  time_stats_t tx_ulsch_sdu;
  /// UE DLSCH rx  processing time inlcuding RLC interface (mac_rrc_data_ind or mac_rlc_status_ind+mac_rlc_data_ind) and mac header parser
  time_stats_t rx_dlsch_sdu;
  /// Delete UE query for MCH subframe processing time

  /* Delete UE MCH rx processing time , no support in NB-IoT*/

  /// UE BCCH rx processing time including RLC interface (mac_rrc_data_ind)
  time_stats_t rx_si;
  /// UE PCCH rx processing time including RLC interface (mac_rrc_data_ind)
  time_stats_t rx_p;
} UE_MAC_INST_NB;

/* Delete struct neigh_cell_id_t, no support in NB-IoT*/

#include "proto.h"
/*@}*/
#endif /*__LAYER2_MAC_DEFS_H__ */