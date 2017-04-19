/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file RRC/LITE/defs_nb_iot.h
* \brief NB-IoT RRC struct definitions and function prototypes
* \author Navid Nikaein, Raymond Knopp and Michele Paffetti
* \date 2010 - 2014, 2017
* \version 1.0
* \company Eurecom
* \email: navid.nikaein@eurecom.fr, raymond.knopp@eurecom.fr, michele.paffetti@studio.unibo.it
*/

#ifndef __OPENAIR_RRC_DEFS_NB_IOT_H__
#define __OPENAIR_RRC_DEFS_NB_IOT_H__


#ifdef USER_MODE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#endif

#include "collection/tree.h"
#include "rrc_types.h"
#include "PHY/defs.h"
#include "COMMON/platform_constants.h"
#include "COMMON/platform_types.h"

#include "COMMON/mac_rrc_primitives.h"
#include "LAYER2/MAC/defs.h"

//#include "COMMON/openair_defs.h"
#ifndef USER_MODE
#include <rtai.h>
#endif

//----Not needed for NB-IoT??-------- (but have to left for UE?)
//#include "SystemInformationBlockType1.h"
//#include "SystemInformation.h"
//#include "RRCConnectionReconfiguration.h"
//#include "RRCConnectionReconfigurationComplete.h"
//#include "RRCConnectionSetup.h"
//#include "RRCConnectionSetupComplete.h"
//#include "RRCConnectionRequest.h"
//#include "RRCConnectionReestablishmentRequest.h"
//#include "BCCH-DL-SCH-Message.h"
//#include "BCCH-BCH-Message.h"
//
//#if defined(Rel10) || defined(Rel14)
//#include "MCCH-Message.h"
//#include "MBSFNAreaConfiguration-r9.h"
//#include "SCellToAddMod-r10.h"
//#endif
//
//#include "AS-Config.h"
//#include "AS-Context.h"
//#include "UE-EUTRA-Capability.h"
//#include "MeasResults.h"
//--------------------------------------


//-----NB-IoT #include files-------

#include "SystemInformationBlockType1-NB.h"
#include "SystemInformation-NB.h"
#include "RRCConnectionReconfiguration-NB.h"
#include "RRCConnectionReconfigurationComplete-NB.h"
#include "RRCConnectionSetup-NB.h"
#include "RRCConnectionSetupComplete-NB.h"
#include "RRCConnectionRequest-NB.h"
#include "RRCConnectionReestablishmentRequest-NB.h"
#include "BCCH-DL-SCH-Message-NB.h"
#include "BCCH-BCH-Message-NB.h"
#include "AS-Config-NB.h"
#include "AS-Context-NB.h"
#include "UE-Capability-NB-r13.h" //equivalent of UE-EUTRA-Capability.h
//-------------------


/* correct Rel(8|10)/Rel14 differences
 * the code is in favor of Rel14, those defines do the translation
 */
#if !defined(Rel14)
#  define CipheringAlgorithm_r12_t e_SecurityAlgorithmConfig__cipheringAlgorithm
#  define CipheringAlgorithm_r12_t e_CipheringAlgorithm_r12 //maybe this solve the problem of the previous line
#  define CipheringAlgorithm_r12_eea0 SecurityAlgorithmConfig__cipheringAlgorithm_eea0
#  define CipheringAlgorithm_r12_eea1 SecurityAlgorithmConfig__cipheringAlgorithm_eea1
#  define CipheringAlgorithm_r12_eea2 SecurityAlgorithmConfig__cipheringAlgorithm_eea2
#  define CipheringAlgorithm_r12_spare1 SecurityAlgorithmConfig__cipheringAlgorithm_spare1
#  define Alpha_r12_al0 UplinkPowerControlCommon__alpha_al0
#  define Alpha_r12_al04 UplinkPowerControlCommon__alpha_al04
#  define Alpha_r12_al05 UplinkPowerControlCommon__alpha_al05
#  define Alpha_r12_al06 UplinkPowerControlCommon__alpha_al06
#  define Alpha_r12_al07 UplinkPowerControlCommon__alpha_al07
#  define Alpha_r12_al08 UplinkPowerControlCommon__alpha_al08
#  define Alpha_r12_al09 UplinkPowerControlCommon__alpha_al09
#  define Alpha_r12_al1 UplinkPowerControlCommon__alpha_al1
#  define PreambleTransMax_t e_PreambleTransMax //maybe this solve problem (asn1_msg_nb_iot.c line 726)
#  define PreambleTransMax_n3 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n3
#  define PreambleTransMax_n4 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n4
#  define PreambleTransMax_n5 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n5
#  define PreambleTransMax_n6 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n6
#  define PreambleTransMax_n7 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n7
#  define PreambleTransMax_n8 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n8
#  define PreambleTransMax_n10 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n10
#  define PreambleTransMax_n20 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n20
#  define PreambleTransMax_n50 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n50
#  define PreambleTransMax_n100 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n100
#  define PreambleTransMax_n200 RACH_ConfigCommon__ra_SupervisionInfo__preambleTransMax_n200
#  define PeriodicBSR_Timer_r12_sf5 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf5
#  define PeriodicBSR_Timer_r12_sf10 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf10
#  define PeriodicBSR_Timer_r12_sf16 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf16
#  define PeriodicBSR_Timer_r12_sf20 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf20
#  define PeriodicBSR_Timer_r12_sf32 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf32
#  define PeriodicBSR_Timer_r12_sf40 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf40
#  define PeriodicBSR_Timer_r12_sf64 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf64
#  define PeriodicBSR_Timer_r12_sf80 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf80
#  define PeriodicBSR_Timer_r12_sf128 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf128
#  define PeriodicBSR_Timer_r12_sf160 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf160
#  define PeriodicBSR_Timer_r12_sf320 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf320
#  define PeriodicBSR_Timer_r12_sf640 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf640
#  define PeriodicBSR_Timer_r12_sf1280 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf1280
#  define PeriodicBSR_Timer_r12_sf2560 MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_sf2560
#  define PeriodicBSR_Timer_r12_infinity MAC_MainConfig__ul_SCH_Config__periodicBSR_Timer_infinity
#  define RetxBSR_Timer_r12_sf320 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf320
#  define RetxBSR_Timer_r12_sf640 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf640
#  define RetxBSR_Timer_r12_sf1280 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf1280
#  define RetxBSR_Timer_r12_sf2560 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf2560
#  define RetxBSR_Timer_r12_sf5120 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf5120
#  define RetxBSR_Timer_r12_sf10240 MAC_MainConfig__ul_SCH_Config__retxBSR_Timer_sf10240
#endif

// This corrects something generated by asn1c which is different between Rel8 and Rel10
#if !defined(Rel10) && !defined(Rel14)
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member SystemInformation_r8_IEs_sib_TypeAndInfo_Member
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib2 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib2
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib3 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib3
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib4 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib4
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib5 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib5
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib6 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib6
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib7 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib7
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib8 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib8
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib9 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib9
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib10 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib10
#define SystemInformation_r8_IEs__sib_TypeAndInfo__Member_PR_sib11 SystemInformation_r8_IEs_sib_TypeAndInfo_Member_PR_sib11
#endif


#ifndef NO_RRM
#include "L3_rrc_interface.h"
#include "rrc_rrm_msg.h"
#include "rrc_rrm_interface.h"
#endif

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

/* TODO: be sure this include is correct.
 * It solves a problem of compilation of the RRH GW,
 * issue #186.
 */
#if !defined(ENABLE_ITTI)
# include "as_message.h"
#endif

#if defined(ENABLE_USE_MME)
# include "commonDef.h"
#endif

#if ENABLE_RAL
# include "collection/hashtable/obj_hashtable.h"
#endif



/*I will change the name of the structure for compile purposes--> hope not to undo this process*/

typedef unsigned int uid_NB_t;
#define UID_LINEAR_ALLOCATOR_BITMAP_SIZE (((NUMBER_OF_UE_MAX/8)/sizeof(unsigned int)) + 1)

typedef struct uid_linear_allocator_NB_s {
  unsigned int   bitmap[UID_LINEAR_ALLOCATOR_BITMAP_SIZE];
} uid_allocator_NB_t;


#define PROTOCOL_RRC_CTXT_UE_FMT           PROTOCOL_CTXT_FMT
#define PROTOCOL_RRC_CTXT_UE_ARGS(CTXT_Pp) PROTOCOL_CTXT_ARGS(CTXT_Pp)

#define PROTOCOL_RRC_CTXT_FMT           PROTOCOL_CTXT_FMT
#define PROTOCOL_RRC_CTXT_ARGS(CTXT_Pp) PROTOCOL_CTXT_ARGS(CTXT_Pp)

/** @defgroup _rrc RRC
 * @ingroup _oai2
 * @{
 */


//#define NUM_PRECONFIGURED_LCHAN (NB_CH_CX*2)  //BCCH, CCCH

#define UE_MODULE_INVALID ((module_id_t) ~0) // FIXME attention! depends on type uint8_t!!!
#define UE_INDEX_INVALID  ((module_id_t) ~0) // FIXME attention! depends on type uint8_t!!! used to be -1


//left as they are --> used in LAYER2/epenair2_proc.c and UE side
//typedef enum UE_STATE_NB_e {
//  RRC_INACTIVE=0,
//  RRC_IDLE,
//  RRC_SI_RECEIVED,
//  RRC_CONNECTED,
//  RRC_RECONFIGURED,
//  RRC_HO_EXECUTION //maybe not needed?
//} UE_STATE_NB_t;


// HO_STATE is not supported by NB-IoT

//#define NUMBER_OF_UE_MAX MAX_MOBILES_PER_RG
#define RRM_FREE(p)       if ( (p) != NULL) { free(p) ; p=NULL ; }
#define RRM_MALLOC(t,n)   (t *) malloc16( sizeof(t) * n )
#define RRM_CALLOC(t,n)   (t *) malloc16( sizeof(t) * n)
#define RRM_CALLOC2(t,s)  (t *) malloc16( s )

//Measurement Report not supported in NB-IoT

#define PAYLOAD_SIZE_MAX 1024
#define RRC_BUF_SIZE 255
#define UNDEF_SECURITY_MODE 0xff
#define NO_SECURITY_MODE 0x20

/* TS 36.331: RRC-TransactionIdentifier ::= INTEGER (0..3) */
#define RRC_TRANSACTION_IDENTIFIER_NUMBER  3

//NB-IoT version (not used for the moment)
//typedef struct UE_RRC_INFO_s { //is used in UE_RRC_INST
//  UE_STATE_NB_t State;
//  uint8_t SIB1systemInfoValueTag;
//  uint32_t SIStatus;
//  uint32_t SIcnt;
//#if defined(Rel10) || defined(Rel14)
//  uint8_t MCCHStatus[8]; // MAX_MBSFN_AREA
//#endif
//  uint8_t SIwindowsize; //!< Corresponds to the SIB1 si-WindowLength parameter. The unit is ms. Possible values are (final): 1,2,5,10,15,20,40
//  uint8_t handoverTarget;
//  HO_STATE_t ho_state;
//  uint16_t SIperiod; //!< Corresponds to the SIB1 si-Periodicity parameter (multiplied by 10). Possible values are (final): 80,160,320,640,1280,2560,5120
//  unsigned short UE_index;
//  uint32_t T300_active;
//  uint32_t T300_cnt;
//  uint32_t T304_active;
//  uint32_t T304_cnt;
//  uint32_t T310_active;
//  uint32_t T310_cnt;
//  uint32_t N310_cnt;
//  uint32_t N311_cnt;
//  rnti_t   rnti;
//} __attribute__ ((__packed__)) UE_RRC_INFO;



//left as it is
typedef struct UE_S_TMSI_NB_s {
  boolean_t  presence;
  mme_code_t mme_code;
  m_tmsi_t   m_tmsi;
} __attribute__ ((__packed__)) UE_S_TMSI_NB;


typedef enum e_rab_satus_NB_e {
  E_RAB_NB_STATUS_NEW,
  E_RAB_NB_STATUS_DONE, // from the eNB perspective
  E_RAB_NB_STATUS_ESTABLISHED, // get the reconfigurationcomplete form UE
  E_RAB_NB_STATUS_FAILED,
} e_rab_status_NB_t;

typedef struct e_rab_param_NB_s {
  e_rab_t param;
  uint8_t status;
  uint8_t xid; // transaction_id
} __attribute__ ((__packed__)) e_rab_param_NB_t;


//HANDOVER_INFO not implemented in NB-IoT delete


#define RRC_HEADER_SIZE_MAX 64
#define RRC_BUFFER_SIZE_MAX 1024

typedef struct {
  char Payload[RRC_BUFFER_SIZE_MAX];
  char Header[RRC_HEADER_SIZE_MAX];
  char payload_size;
} RRC_BUFFER_NB;

#define RRC_BUFFER_SIZE_NB sizeof(RRC_BUFFER_NB)


typedef struct RB_INFO_NB_s {
  uint16_t Rb_id;  //=Lchan_id
  LCHAN_DESC Lchan_desc[2]; //LCHAN_DESC should be changed for NB-IoT
  MAC_MEAS_REQ_ENTRY *Meas_entry; //may not needed for NB-IoT
} RB_INFO_NB;

typedef struct SRB_INFO_NB_s {
  uint16_t Srb_id;  //=Lchan_id---> useful for distinguish between SRB1 and SRB1bis?
  RRC_BUFFER_NB Rx_buffer;
  RRC_BUFFER_NB Tx_buffer;
  LCHAN_DESC Lchan_desc[2];//LCHAN_DESC should be changed for NB-IoT
  unsigned int Trans_id;
  uint8_t Active;
} SRB_INFO_NB;


typedef struct RB_INFO_TABLE_ENTRY_NB_s {
  RB_INFO_NB Rb_info;
  uint8_t Active;
  uint32_t Next_check_frame;
  uint8_t Status;
} RB_INFO_TABLE_ENTRY_NB;

typedef struct SRB_INFO_TABLE_ENTRY_NB_s {
  SRB_INFO_NB Srb_info;
  uint8_t Active;
  uint8_t Status;
  uint32_t Next_check_frame;
} SRB_INFO_TABLE_ENTRY_NB;

//MEAS_REPORT_LIST_s not implemented in NB-IoT but is used at UE side
//HANDOVER_INFO_UE not implemented in NB-IoT


//NB-IoT eNB_RRC_UE_NB_s--(used as a context in eNB --> ue_context in rrc_eNB_ue_context)------
typedef struct eNB_RRC_UE_NB_s {
  uint8_t                            primaryCC_id;

  //Radio Bearers carried by RadioResourceConfigDedicated-NB-r13.h
  //used in generate_default/dedicatedRRCConnectionReconfiguration (rrc_eNB.c)
  //in NB-IoT only SRB0, SRB1 and SRB1bis (until AS security activation) exist

  SRB_ToAddModList_NB_r13_t*                SRB_configList;//for SRB1 and SRB1bis
  SRB_ToAddModList_NB_r13_t*                SRB1_configList[RRC_TRANSACTION_IDENTIFIER_NUMBER]; //only for SRB1
  DRB_ToAddModList_NB_r13_t*                DRB_configList; //for all the DRBs
  DRB_ToAddModList_NB_r13_t*                DRB_configList2[RRC_TRANSACTION_IDENTIFIER_NUMBER]; //for the configured DRBs of a xid
  uint8_t                            		DRB_active[2];//in LTE was 8 --> at most 2 for NB-IoT

  struct PhysicalConfigDedicated_NB_r13*    physicalConfigDedicated_NB;
  MAC_MainConfig_NB_r13_t*           mac_MainConfig_NB;

  //No SPS(semi-persistent scheduling) in NB-IoT
  //No Measurement report in NB-IoT

  SRB_INFO_NB                           SI;
  SRB_INFO_NB                           Srb0;
  SRB_INFO_TABLE_ENTRY_NB               Srb1;
  SRB_INFO_TABLE_ENTRY_NB               Srb1bis;

#if defined(ENABLE_SECURITY)
  /* KeNB as derived from KASME received from EPC */
  uint8_t kenb[32];
#endif

  /* Used integrity/ciphering algorithms--> maintained the same for NB-IoT */
  e_CipheringAlgorithm_r12     ciphering_algorithm; //Specs. TS 36.331 V14.1.0 pag 432 Change position of chipering enumerative w.r.t previous version
  e_SecurityAlgorithmConfig__integrityProtAlgorithm integrity_algorithm;

  uint8_t                            Status;
  rnti_t                             rnti;
  uint64_t                           random_ue_identity;



  /* Information from UE RRC ConnectionRequest-NB-r13_IE--> NB-IoT */
  UE_S_TMSI_NB                          Initialue_identity_s_TMSI;
  EstablishmentCause_NB_r13_t               establishment_cause_NB; //different set for NB-IoT

  /* Information from UE RRC ConnectionReestablishmentRequest-NB--> NB-IoT */
  ReestablishmentCause_NB_r13_t             reestablishment_cause_NB; //different set for NB_IoT

///nothing to be changed for NB-IoT?
  /* UE id for initial connection to S1AP */
  uint16_t                           ue_initial_id;

  /* Information from S1AP initial_context_setup_req */
  uint32_t                           eNB_ue_s1ap_id :24;

  security_capabilities_t            security_capabilities;

  /* Total number of e_rab already setup in the list */ //NAS list?
  uint8_t                           setup_e_rabs;
  /* Number of e_rab to be setup in the list */ //NAS list?
  uint8_t                            nb_of_e_rabs;
  /* list of e_rab to be setup by RRC layers */
  e_rab_param_NB_t                      e_rab[NB_RB_MAX];//[S1AP_MAX_E_RAB];

  // LG: For GTPV1 TUNNELS
  uint32_t                           enb_gtp_teid[S1AP_MAX_E_RAB];
  transport_layer_addr_t             enb_gtp_addrs[S1AP_MAX_E_RAB];
  rb_id_t                            enb_gtp_ebi[S1AP_MAX_E_RAB];


  uint32_t                           ul_failure_timer;
  uint32_t                           ue_release_timer;
  uint32_t                           ue_release_timer_thres;
} eNB_RRC_UE_NB_t;
//--------------------------------------------------------------------------------

typedef uid_NB_t ue_uid_t;


//Not touched - generally variable called: ue_context_pP
typedef struct rrc_eNB_ue_context_NB_s {

  /* Tree related data */
  RB_ENTRY(rrc_eNB_ue_context_NB_s) entries;

  /* Uniquely identifies the UE between MME and eNB within the eNB.
   * This id is encoded on 24bits.
   */
  rnti_t         ue_id_rnti;

  // another key for protocol layers but should not be used as a key for RB tree
  ue_uid_t       local_uid;

  /* UE id for initial connection to S1AP */
  struct eNB_RRC_UE_NB_s   ue_context; //context of ue in the e-nB

} rrc_eNB_ue_context_NB_t;



//---NB-IoT (completely changed)-------------------------------
//called "carrier"--> data from PHY layer
typedef struct {

  // buffer that contains the encoded messages
  uint8_t							*MIB_NB;
  uint8_t							sizeof_MIB_NB;
  uint8_t                           *SIB1_NB;
  uint8_t                           sizeof_SIB1_NB;
  uint8_t                         	*SIB23_NB;
  uint8_t                        	sizeof_SIB23_NB;

  //uint8_t                           *SIB2_NB;
  //uint8_t                           sizeof_SIB2_NB;
  //uint8_t                           *SIB3_NB;
  //uint8_t                           sizeof_SIB3_NB;

  //not actually implemented in OAI
  uint8_t                           *SIB4_NB;
  uint8_t                           sizeof_SIB4_NB;
  uint8_t                           *SIB5_NB;
  uint8_t                           sizeof_SIB5_NB;
  uint8_t                           *SIB14_NB;
  uint8_t                           sizeof_SIB14_NB;
  uint8_t                           *SIB16_NB;
  uint8_t                           sizeof_SIB16_NB;

  //TS 36.331 V14.2.1
  uint8_t                           *SIB15_NB;
  uint8_t                           sizeof_SIB15_NB;
  uint8_t                           *SIB20_NB;
  uint8_t                           sizeof_SIB20_NB;
  uint8_t                           *SIB22_NB;
  uint8_t                           sizeof_SIB22_NB;

  int                               Ncp; //extended cyclic prefix (needed?)
  int                               p_eNB; //number of Transmit antenna (may not needed)
  uint32_t                          dl_CarrierFreq;
  uint32_t                          ul_CarrierFreq;
  uint16_t                          physCellId; //SIB5-NB //look also at the difference in In-Band implementation

  //are the only static one (memory has been already allocated)
  BCCH_BCH_Message_NB_t                mib_NB;
  BCCH_DL_SCH_Message_NB_t             siblock1_NB; //SIB1-NB
  BCCH_DL_SCH_Message_NB_t             systemInformation_NB; //SI

  //memory should be allocated--> in principle are only for commodity?
  SystemInformationBlockType1_NB_t     		*sib1_NB;
  SystemInformationBlockType2_NB_r13_t   	*sib2_NB;
  SystemInformationBlockType3_NB_r13_t   	*sib3_NB;
  //not implemented yet
  SystemInformationBlockType4_NB_r13_t    	*sib4_NB;
  SystemInformationBlockType5_NB_r13_t     	*sib5_NB;
  SystemInformationBlockType14_NB_r13_t     *sib14_NB;
  SystemInformationBlockType16_NB_r13_t     *sib16_NB;

  /*future implementation TS 36.331 V14.2.1
  SystemInformationBlockType15_NB_r14_t     *sib15;
  SystemInformationBlockType20_NB_r14_t     *sib20;
  SystemInformationBlockType22_NB_r14_t     *sib22;

  uint8_t							SCPTM_flag;
  uint8_t							sizeof_SC_MCHH_MESS[];
  SC_MCCH_Message_NB_t				scptm;*/

  SRB_INFO_NB                          SI;
  SRB_INFO_NB                          Srb0;
} rrc_eNB_carrier_data_NB_t;
//---------------------------------------------------



//---NB-IoT---(completely change)---------------------
typedef struct eNB_RRC_INST_NB_s {

  rrc_eNB_carrier_data_NB_t          carrier[MAX_NUM_CCs];

  uid_allocator_NB_t                    uid_allocator; // for rrc_ue_head
  RB_HEAD(rrc_ue_tree_NB_s, rrc_eNB_ue_context_NB_s)     rrc_ue_head; // ue_context tree key search by rnti

  uint8_t                           Nb_ue;

  hash_table_t                      *initial_id2_s1ap_ids; // key is    content is rrc_ue_s1ap_ids_t
  hash_table_t                      *s1ap_id2_s1ap_ids   ; // key is    content is rrc_ue_s1ap_ids_t

  //RRC configuration
  RrcConfigurationReq configuration; //should be changed but need PHY specs also

  //new--> to be check
  // other PLMN parameters
  /// Mobile country code
  int mcc;
  /// Mobile network code
  int mnc;
  /// number of mnc digits
  int mnc_digit_length;

  // other RAN parameters
  int srb1_timer_poll_retransmit;
  int srb1_poll_pdu;
  int srb1_poll_byte;
  int srb1_max_retx_threshold;
  int srb1_timer_reordering;
  int srb1_timer_status_prohibit;
  int srs_enable[MAX_NUM_CCs];


} eNB_RRC_INST_NB;


#define MAX_UE_CAPABILITY_SIZE 255
//not needed for the moment
//typedef struct OAI_UECapability_s {
//  uint8_t sdu[MAX_UE_CAPABILITY_SIZE];
//  uint8_t sdu_size;
////NB-IoT------
//  UE_Capability_NB_r13_t	UE_Capability_NB; //replace the UE_EUTRA_Capability of LTE
//} OAI_UECapability_t;


//Old implementation --> see NB_iot version
//typedef struct UE_RRC_INST_s {
//  Rrc_State_t     RrcState;
//  Rrc_Sub_State_t RrcSubState;
//# if defined(ENABLE_USE_MME)
//  plmn_t          plmnID;
//  Byte_t          rat;
//  as_nas_info_t   initialNasMsg;
//# endif
//  OAI_UECapability_t *UECap;
//  uint8_t *UECapability;
//  uint8_t UECapability_size;
//  UE_RRC_INFO Info[NB_SIG_CNX_UE];
//  SRB_INFO Srb0[NB_SIG_CNX_UE];
//  SRB_INFO_TABLE_ENTRY Srb1[NB_CNX_UE];
//  SRB_INFO_TABLE_ENTRY Srb2[NB_CNX_UE];
//  HANDOVER_INFO_UE HandoverInfoUe;
//  uint8_t *SIB1[NB_CNX_UE];
//  uint8_t sizeof_SIB1[NB_CNX_UE];
//  uint8_t *SI[NB_CNX_UE];
//  uint8_t sizeof_SI[NB_CNX_UE];
//  uint8_t SIB1Status[NB_CNX_UE];
//  uint8_t SIStatus[NB_CNX_UE];
//  SystemInformationBlockType1_t *sib1[NB_CNX_UE];
//  SystemInformation_t *si[NB_CNX_UE]; //!< Temporary storage for an SI message. Decoding happens in decode_SI().
//  SystemInformationBlockType2_t *sib2[NB_CNX_UE];
//  SystemInformationBlockType3_t *sib3[NB_CNX_UE];
//  SystemInformationBlockType4_t *sib4[NB_CNX_UE];
//  SystemInformationBlockType5_t *sib5[NB_CNX_UE];
//  SystemInformationBlockType6_t *sib6[NB_CNX_UE];
//  SystemInformationBlockType7_t *sib7[NB_CNX_UE];
//  SystemInformationBlockType8_t *sib8[NB_CNX_UE];
//  SystemInformationBlockType9_t *sib9[NB_CNX_UE];
//  SystemInformationBlockType10_t *sib10[NB_CNX_UE];
//  SystemInformationBlockType11_t *sib11[NB_CNX_UE];
//
//#if defined(Rel10) || defined(Rel14)
//  uint8_t                           MBMS_flag;
//  uint8_t *MCCH_MESSAGE[NB_CNX_UE];
//  uint8_t sizeof_MCCH_MESSAGE[NB_CNX_UE];
//  uint8_t MCCH_MESSAGEStatus[NB_CNX_UE];
//  MBSFNAreaConfiguration_r9_t       *mcch_message[NB_CNX_UE];
//  SystemInformationBlockType12_r9_t *sib12[NB_CNX_UE];
//  SystemInformationBlockType13_r9_t *sib13[NB_CNX_UE];
//#endif
//#ifdef CBA
//  uint8_t                         num_active_cba_groups;
//  uint16_t                        cba_rnti[NUM_MAX_CBA_GROUP];
//#endif
//  uint8_t                         num_srb;
//  struct SRB_ToAddMod             *SRB1_config[NB_CNX_UE];
//  struct SRB_ToAddMod             *SRB2_config[NB_CNX_UE];
//  struct DRB_ToAddMod             *DRB_config[NB_CNX_UE][8];
//  rb_id_t                         *defaultDRB; // remember the ID of the default DRB
//  MeasObjectToAddMod_t            *MeasObj[NB_CNX_UE][MAX_MEAS_OBJ];
//  struct ReportConfigToAddMod     *ReportConfig[NB_CNX_UE][MAX_MEAS_CONFIG];
//  struct QuantityConfig           *QuantityConfig[NB_CNX_UE];
//  struct MeasIdToAddMod           *MeasId[NB_CNX_UE][MAX_MEAS_ID];
//  MEAS_REPORT_LIST      *measReportList[NB_CNX_UE][MAX_MEAS_ID];
//  uint32_t           measTimer[NB_CNX_UE][MAX_MEAS_ID][6]; // 6 neighboring cells
//  RSRP_Range_t                    s_measure;
//  struct MeasConfig__speedStatePars *speedStatePars;
//  struct PhysicalConfigDedicated  *physicalConfigDedicated[NB_CNX_UE];
//  struct SPS_Config               *sps_Config[NB_CNX_UE];
//  MAC_MainConfig_t                *mac_MainConfig[NB_CNX_UE];
//  MeasGapConfig_t                 *measGapConfig[NB_CNX_UE];
//  double                          filter_coeff_rsrp; // [7] ???
//  double                          filter_coeff_rsrq; // [7] ???
//  float                           rsrp_db[7];
//  float                           rsrq_db[7];
//  float                           rsrp_db_filtered[7];
//  float                           rsrq_db_filtered[7];
//#if ENABLE_RAL
//  obj_hash_table_t               *ral_meas_thresholds;
//  ral_transaction_id_t            scan_transaction_id;
//#endif
//#if defined(ENABLE_SECURITY)
//  /* KeNB as computed from parameters within USIM card */
//  uint8_t kenb[32];
//#endif
//
//  /* Used integrity/ciphering algorithms */
//  e_CipheringAlgorithm_r12                          ciphering_algorithm;
//  e_SecurityAlgorithmConfig__integrityProtAlgorithm integrity_algorithm;
//
//} UE_RRC_INST;



//NB-IoT version (for now not consider UE-side)--------------

//typedef struct UE_RRC_INST_s {
//  Rrc_State_t     RrcState; //are the same?
//  Rrc_Sub_State_t RrcSubState;
//
//# if defined(ENABLE_USE_MME)
//  plmn_t          plmnID;
//  Byte_t          rat;
//  as_nas_info_t   initialNasMsg;
//# endif
//
//  OAI_UECapability_t *UECap;
//  uint8_t *UECapability;
//  uint8_t UECapability_size;
//  UE_RRC_INFO Info[NB_SIG_CNX_UE];
//  SRB_INFO Srb0[NB_SIG_CNX_UE];
//  SRB_INFO_TABLE_ENTRY Srb1[NB_CNX_UE];
//  SRB_INFO_TABLE_ENTRY Srb1bis[NB_CNX_UE]; //not exist srb2
//
//  //needed for NB-IoT?
//  HANDOVER_INFO_UE HandoverInfoUe;
//
//  uint8_t *SIB1_NB[NB_CNX_UE];
//  uint8_t sizeof_SIB1_NB[N_NBB_CNX_UE];
//  uint8_t *SI_NB[NB_CNX_UE];
//  uint8_t sizeof_SI_NB[NB_CNX_UE];
//  uint8_t SIB1_NB_Status[NB_CNX_UE];
//  uint8_t SI_NB_Status[NB_CNX_UE];
//
//  //NB-IoT SIBs
//  SystemInformationBlockType1_NB_t *sib1_NB[NB_CNX_UE];
//  SystemInformation_NB_t *si_NB[NB_CNX_UE]; //!< Temporary storage for an SI-NB message. Decoding happens in decode_SI()-->UE side (to be changed).
//
//  SystemInformationBlockType2_NB_r13_t *sib2_NB[NB_CNX_UE];
//  SystemInformationBlockType3_NB_r13_t *sib3_NB[NB_CNX_UE];
//  SystemInformationBlockType4_NB_r13_t *sib4_NB[NB_CNX_UE];
//  SystemInformationBlockType5_NB_r13_t *sib5_NB[NB_CNX_UE];
//  SystemInformationBlockType14_NB_r13_t *sib14_NB[NB_CNX_UE];
//  SystemInformationBlockType16_NB_r13_t *sib16_NB[NB_CNX_UE];
//
//  future implementation TS 36.331 V14.2.1--(preliminary)
//  SystemInformationBlockType15_NB_r14_t *sib15_NB[NB_CNX_UE];
//  SystemInformationBlockType20_NB_r14_t *sib20_NB[NB_CNX_UE];
//  SystemInformationBlockType22_NB_r14_t *sib22_NB[NB_CNX_UE];
//
//  uint8_t							SCPTM_flag;
//  uint8_t							sizeof_SC_MCHH_MESS[];
//  SC_MCCH_Message_NB_t				scptm;
//
//
//  uint8_t                         num_srb;
//  struct SRB_ToAddMod_NB_r13_t             *SRB1_config_NB[NB_CNX_UE];
//  struct SRB_ToAddMod_NB_r13_t             *SRB1bis_config_NB[NB_CNX_UE];
//  struct DRB_ToAddMod_NB_r13_t             *DRB_config_NB[NB_CNX_UE][8];
//  rb_id_t                         *defaultDRB; // remember the ID of the default DRB
//
//  //no measurement Reports in NB-IoT
//
//  struct PhysicalConfigDedicated_NB_r13  *physicalConfigDedicated_NB[NB_CNX_UE];
//  MAC_MainConfig_NB_r13_t                *mac_MainConfig_NB[NB_CNX_UE];
//
//  //no RAL
//
//#if defined(ENABLE_SECURITY)
//  /* KeNB as computed from parameters within USIM card */
//  uint8_t kenb[32];
//#endif
//  /* KeNB as computed from parameters within USIM card */
//  uint8_t kenb[32];
//#endif
//
//  // Used integrity/ciphering algorithms
//  e_CipheringAlgorithm_r12                          ciphering_algorithm;
//  e_SecurityAlgorithmConfig__integrityProtAlgorithm integrity_algorithm;
//
//} UE_RRC_INST;

#include "proto_nb_iot.h" //should be put here otherwise compilation error

#endif
/** @} */
