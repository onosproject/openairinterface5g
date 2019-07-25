/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*
                                enb_config.h
                             -------------------
  AUTHOR  : Lionel GAUTHIER, Navid Nikaein, Laurent Winckel
  COMPANY : EURECOM
  EMAIL   : Lionel.Gauthier@eurecom.fr, navid.nikaein@eurecom.fr
*/

#ifndef ENB_CONFIG_H_
#define ENB_CONFIG_H_
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <libconfig.h>

#include "commonDef.h"
#include "platform_types.h"
#include "platform_constants.h"
#include "PHY/impl_defs_lte.h"
#include "PHY/defs_eNB.h"
#include "s1ap_messages_types.h"
#include "f1ap_messages_types.h"
#include "LTE_SystemInformationBlockType2.h"
#include "rrc_messages_types.h"
#include "RRC/LTE/rrc_defs.h"
#include <intertask_interface.h>
#include "enb_paramdef.h"

#define IPV4_STR_ADDR_TO_INT_NWBO(AdDr_StR,NwBo,MeSsAgE ) do {\
            struct in_addr inp;\
            if ( inet_aton(AdDr_StR, &inp ) < 0 ) {\
                AssertFatal (0, MeSsAgE);\
            } else {\
                NwBo = inp.s_addr;\
            }\
        } while (0);

/** @defgroup _enb_app ENB APP 
 * @ingroup _oai2
 * @{
 */

// Hard to find a defined value for max enb...
#define MAX_ENB 16

#define MAX_DU	4
#define CU_BALANCING_ALL		127
#define CU_BALANCING_ROUND_ROBIN	126

typedef struct mme_ip_address_s {
  unsigned  ipv4:1;
  unsigned  ipv6:1;
  unsigned  active:1;
  char     *ipv4_address;
  char     *ipv6_address;
} mme_ip_address_t;

typedef struct cu_params {
  const char    *local_ipv4_address;
  const uint16_t local_port;
  const char    *remote_ipv4_address;
  const int16_t  remote_port;
} cudu_params_t;

typedef struct ru_config_s {
  // indicates if local or remote rf is used (1 == LOCAL)
  unsigned  local_rf:1;
  // indicates if UDP socket is used
  unsigned  udp:1;
  // indicates if RAW socket is used
  unsigned  raw:1;
  char      *ru_if_name;
  char     *local_address;
  char     *remote_address;
  uint16_t  local_port;
  uint16_t  remote_port;
  uint8_t   udpif4p5;
  uint8_t   rawif4p5;
  uint8_t   rawif5_mobipass;
  uint8_t   if_compress;
} ru_config_t;

extern void RCconfig_RU(void);
extern void RCconfig_flexran(void);
extern void RCconfig_L1(void);
extern void RCconfig_macrlc(int macrlc_has_f1[MAX_MAC_INST]);
extern void UE_config_stub_pnf(void);
extern int  RCconfig_gtpu(void );
extern void RCConfig(void);

  /* Physical parameters */
  int16_t                 nb_cc;
#ifndef OCP_FRAMEWORK
  eNB_func_t              cc_node_function[1+MAX_NUM_CCs];
  eNB_timing_t            cc_node_timing[1+MAX_NUM_CCs];
  int16_t                 cc_node_synch_ref[1+MAX_NUM_CCs];
  lte_frame_type_t        frame_type[1+MAX_NUM_CCs];
  uint8_t                 tdd_config[1+MAX_NUM_CCs];
  uint8_t                 tdd_config_s[1+MAX_NUM_CCs];
  lte_prefix_type_t       prefix_type[1+MAX_NUM_CCs];
  int16_t                 eutra_band[1+MAX_NUM_CCs];
  uint64_t                downlink_frequency[1+MAX_NUM_CCs];
  int32_t                 uplink_frequency_offset[1+MAX_NUM_CCs];

  int16_t                 Nid_cell[1+MAX_NUM_CCs];// for testing, change later
  int16_t                 N_RB_DL[1+MAX_NUM_CCs];// for testing, change later
  int                     nb_antenna_ports[1+MAX_NUM_CCs];
  int                     nb_antennas_tx[1+MAX_NUM_CCs];
  int                     nb_antennas_rx[1+MAX_NUM_CCs];
  int                     tx_gain[1+MAX_NUM_CCs];
  int                     rx_gain[1+MAX_NUM_CCs];
  long                    prach_root[1+MAX_NUM_CCs];
  long                    prach_config_index[1+MAX_NUM_CCs];
  BOOLEAN_t               prach_high_speed[1+MAX_NUM_CCs];
  long                    prach_zero_correlation[1+MAX_NUM_CCs];
  long                    prach_freq_offset[1+MAX_NUM_CCs];
  long                    pucch_delta_shift[1+MAX_NUM_CCs];
  long                    pucch_nRB_CQI[1+MAX_NUM_CCs];
  long                    pucch_nCS_AN[1+MAX_NUM_CCs];
#if !defined(Rel10) && !defined(Rel14)
  long                    pucch_n1_AN[1+MAX_NUM_CCs];
#endif
  long                    pdsch_referenceSignalPower[1+MAX_NUM_CCs];
  long                    pdsch_p_b[1+MAX_NUM_CCs];
  long                    pusch_n_SB[1+MAX_NUM_CCs];
  long                    pusch_hoppingMode[1+MAX_NUM_CCs];
  long                    pusch_hoppingOffset[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_enable64QAM[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_groupHoppingEnabled[1+MAX_NUM_CCs];
  long                    pusch_groupAssignment[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_sequenceHoppingEnabled[1+MAX_NUM_CCs];
  long                    pusch_nDMRS1[1+MAX_NUM_CCs];
  long                    phich_duration[1+MAX_NUM_CCs];
  long                    phich_resource[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_enable[1+MAX_NUM_CCs];
  long                    srs_BandwidthConfig[1+MAX_NUM_CCs];
  long                    srs_SubframeConfig[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_ackNackST[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_MaxUpPts[1+MAX_NUM_CCs];
  long                    pusch_p0_Nominal[1+MAX_NUM_CCs];
  long                    pusch_alpha[1+MAX_NUM_CCs];
  long                    pucch_p0_Nominal[1+MAX_NUM_CCs];
  long                    msg3_delta_Preamble[1+MAX_NUM_CCs];
  long                    ul_CyclicPrefixLength[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format1                    pucch_deltaF_Format1[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format1b                   pucch_deltaF_Format1b[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2                    pucch_deltaF_Format2[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2a                   pucch_deltaF_Format2a[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2b                   pucch_deltaF_Format2b[1+MAX_NUM_CCs];
  long                    rach_numberOfRA_Preambles[1+MAX_NUM_CCs];
  BOOLEAN_t               rach_preamblesGroupAConfig[1+MAX_NUM_CCs];
  long                    rach_sizeOfRA_PreamblesGroupA[1+MAX_NUM_CCs];
  long                    rach_messageSizeGroupA[1+MAX_NUM_CCs];
  e_RACH_ConfigCommon__preambleInfo__preamblesGroupAConfig__messagePowerOffsetGroupB                    rach_messagePowerOffsetGroupB[1+MAX_NUM_CCs];
  long                    rach_powerRampingStep[1+MAX_NUM_CCs];
  long                    rach_preambleInitialReceivedTargetPower[1+MAX_NUM_CCs];
  long                    rach_preambleTransMax[1+MAX_NUM_CCs];
  long                    rach_raResponseWindowSize[1+MAX_NUM_CCs];
  long                    rach_macContentionResolutionTimer[1+MAX_NUM_CCs];
  long                    rach_maxHARQ_Msg3Tx[1+MAX_NUM_CCs];
  long                    bcch_modificationPeriodCoeff[1+MAX_NUM_CCs];
  long                    pcch_defaultPagingCycle[1+MAX_NUM_CCs];
  long                    pcch_nB[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t300[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t301[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t310[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t311[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_n310[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_n311[1+MAX_NUM_CCs];
#else
   RrcConfigurationReq    RrcReq;
#endif
  long                    ue_TransmissionMode[1+MAX_NUM_CCs];
  //
  /////////////////////////////////////////// NB-IoT ///////////////////////////////////////////////////
  //
  long              rach_raResponseWindowSize_NB[1+MAX_NUM_CCs];
  long              rach_macContentionResolutionTimer_NB[1+MAX_NUM_CCs];
  long              rach_powerRampingStep_NB[1+MAX_NUM_CCs];
  long              rach_preambleInitialReceivedTargetPower_NB[1+MAX_NUM_CCs];
  long              rach_preambleTransMax_CE_NB[1+MAX_NUM_CCs];
  long              bcch_modificationPeriodCoeff_NB[1+MAX_NUM_CCs];
  long              pcch_defaultPagingCycle_NB[1+MAX_NUM_CCs];
  long              nprach_CP_Length[1+MAX_NUM_CCs];
  long              nprach_rsrp_range[1+MAX_NUM_CCs];
  long             nprach_SubcarrierMSG3_RangeStart[1+MAX_NUM_CCs];
  long              maxNumPreambleAttemptCE_NB[1+MAX_NUM_CCs];
  long              npdsch_nrs_Power[1+MAX_NUM_CCs];
  long              npusch_ack_nack_numRepetitions_NB[1+MAX_NUM_CCs];
  long              npusch_srs_SubframeConfig_NB[1+MAX_NUM_CCs];
  long              npusch_threeTone_CyclicShift_r13[1+MAX_NUM_CCs];
  long              npusch_sixTone_CyclicShift_r13[1+MAX_NUM_CCs];
  BOOLEAN_t         npusch_groupHoppingEnabled[1+MAX_NUM_CCs];
  long              npusch_groupAssignmentNPUSCH_r13[1+MAX_NUM_CCs];
  long              dl_GapThreshold_NB[1+MAX_NUM_CCs];
  long              dl_GapPeriodicity_NB[1+MAX_NUM_CCs];
  long       dl_GapDurationCoeff_NB[1+MAX_NUM_CCs];
  long              npusch_p0_NominalNPUSCH[1+MAX_NUM_CCs];
  long              npusch_alpha[1+MAX_NUM_CCs];
  long              deltaPreambleMsg3[1+MAX_NUM_CCs];

  long              ue_TimersAndConstants_t300_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t301_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t310_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t311_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_n310_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_n311_NB[1+MAX_NUM_CCs];

  long              nprach_Periodicity[1+MAX_NUM_CCs];
  long              nprach_StartTime[1+MAX_NUM_CCs];
  long              nprach_SubcarrierOffset[1+MAX_NUM_CCs];
  long              nprach_NumSubcarriers[1+MAX_NUM_CCs];
  long              numRepetitionsPerPreambleAttempt[1+MAX_NUM_CCs];
  long              npdcch_NumRepetitions_RA[1+MAX_NUM_CCs];
  long              npdcch_StartSF_CSS_RA[1+MAX_NUM_CCs];
  long             npdcch_Offset_RA[1+MAX_NUM_CCs];

  /////////////////////////////////////////////END/////////////////////////////////////////////
  long                    srb1_timer_poll_retransmit;
  long                    srb1_timer_reordering;
  long                    srb1_timer_status_prohibit;
  long                    srb1_poll_pdu;
  long                    srb1_poll_byte;
  long                    srb1_max_retx_threshold;
  /* Nb of MME to connect to */
  uint8_t             nb_mme;
  /* List of MME to connect to */
  mme_ip_address_t    mme_ip_address[S1AP_MAX_NB_MME_IP_ADDRESS];

  int                 sctp_in_streams;
  int                 sctp_out_streams;

  char               *enb_interface_name_for_S1U;
  in_addr_t           enb_ipv4_address_for_S1U;
  tcp_udp_port_t      enb_port_for_S1U;

  char               *enb_interface_name_for_S1_MME;
  in_addr_t           enb_ipv4_address_for_S1_MME;

  char               *flexran_agent_interface_name;
  in_addr_t           flexran_agent_ipv4_address;
  tcp_udp_port_t      flexran_agent_port;
  char               *flexran_agent_cache;

  /* Nb of RRH to connect to */
  uint8_t             nb_rrh_gw;
  char               *rrh_gw_if_name;
  /* List of MME to connect to */
  rrh_gw_config_t       rrh_gw_config[4];
void                          enb_config_display(void);
void                          ru_config_display(void);

int RCconfig_RRC(uint32_t i, eNB_RRC_INST *rrc, int macrlc_has_f1);
int RCconfig_S1(MessageDef *msg_p, uint32_t i);

void read_config_and_init(void);
int RCconfig_X2(MessageDef *msg_p, uint32_t i);

void fill_SL_configuration(MessageDef *msg_p,  ccparams_sidelink_t *SLconfig,int cell_idx,int cc_idx,char *config_fname);
void fill_eMTC_configuration(MessageDef *msg_p,  ccparams_eMTC_t *eMTCconfig, int cell_idx,int cc_idx,char *config_fname,char *brparamspath);

int RCconfig_DU_F1(MessageDef *msg_p, uint32_t i);
void handle_f1ap_setup_resp(f1ap_setup_resp_t *resp);


// ******************************************************** NB-IoT ********************************************************** //

//typedef struct Enb_properties_NB_IoT_s {
  /* Unique eNB_id to identify the eNB within EPC.
   * For macro eNB ids this field should be 20 bits long.
   * For home eNB ids this field should be 28 bits long.
   */
//  uint32_t            eNB_id;

  /* The type of the cell */
 // enum cell_type_e    cell_type;

  /* Optional name for the cell
   * NOTE: the name can be NULL (i.e no name) and will be cropped to 150
   * characters.
   */
 // char               *eNB_name;


  /* Tracking area code */
//  uint16_t            tac;

  /* Mobile Country Code
   * Mobile Network Code
   */
 /* uint16_t            mcc;
  uint16_t            mnc;
  uint8_t             mnc_digit_length;
*/

/*
  // Physical parameters //
  int16_t                 nb_cc;
#ifndef OCP_FRAMEWORK
  eNB_func_t              cc_node_function[1+MAX_NUM_CCs];
  eNB_timing_t            cc_node_timing[1+MAX_NUM_CCs];
  int16_t                 cc_node_synch_ref[1+MAX_NUM_CCs];
  lte_frame_type_t        frame_type[1+MAX_NUM_CCs];
  uint8_t                 tdd_config[1+MAX_NUM_CCs];
  uint8_t                 tdd_config_s[1+MAX_NUM_CCs];
  lte_prefix_type_t       prefix_type[1+MAX_NUM_CCs];
  int16_t                 eutra_band[1+MAX_NUM_CCs];
  uint64_t                downlink_frequency[1+MAX_NUM_CCs];
  int32_t                 uplink_frequency_offset[1+MAX_NUM_CCs];

  int16_t                 Nid_cell[1+MAX_NUM_CCs];// for testing, change later
  int16_t                 N_RB_DL[1+MAX_NUM_CCs];// for testing, change later
  int                     nb_antenna_ports[1+MAX_NUM_CCs];
  int                     nb_antennas_tx[1+MAX_NUM_CCs];
  int                     nb_antennas_rx[1+MAX_NUM_CCs];
  int                     tx_gain[1+MAX_NUM_CCs];
  int                     rx_gain[1+MAX_NUM_CCs];
  long                    prach_root[1+MAX_NUM_CCs];
  long                    prach_config_index[1+MAX_NUM_CCs];
  BOOLEAN_t               prach_high_speed[1+MAX_NUM_CCs];
  long                    prach_zero_correlation[1+MAX_NUM_CCs];
  long                    prach_freq_offset[1+MAX_NUM_CCs];
  long                    pucch_delta_shift[1+MAX_NUM_CCs];
  long                    pucch_nRB_CQI[1+MAX_NUM_CCs];
  long                    pucch_nCS_AN[1+MAX_NUM_CCs];
#if !defined(Rel10) && !defined(Rel14)
  long                    pucch_n1_AN[1+MAX_NUM_CCs];
#endif
  long                    pdsch_referenceSignalPower[1+MAX_NUM_CCs];
  long                    pdsch_p_b[1+MAX_NUM_CCs];
  long                    pusch_n_SB[1+MAX_NUM_CCs];
  long                    pusch_hoppingMode[1+MAX_NUM_CCs];
  long                    pusch_hoppingOffset[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_enable64QAM[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_groupHoppingEnabled[1+MAX_NUM_CCs];
  long                    pusch_groupAssignment[1+MAX_NUM_CCs];
  BOOLEAN_t               pusch_sequenceHoppingEnabled[1+MAX_NUM_CCs];
  long                    pusch_nDMRS1[1+MAX_NUM_CCs];
  long                    phich_duration[1+MAX_NUM_CCs];
  long                    phich_resource[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_enable[1+MAX_NUM_CCs];
  long                    srs_BandwidthConfig[1+MAX_NUM_CCs];
  long                    srs_SubframeConfig[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_ackNackST[1+MAX_NUM_CCs];
  BOOLEAN_t               srs_MaxUpPts[1+MAX_NUM_CCs];
  long                    pusch_p0_Nominal[1+MAX_NUM_CCs];
  long                    pusch_alpha[1+MAX_NUM_CCs];
  long                    pucch_p0_Nominal[1+MAX_NUM_CCs];
  long                    msg3_delta_Preamble[1+MAX_NUM_CCs];
  long                    ul_CyclicPrefixLength[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format1                    pucch_deltaF_Format1[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format1b                   pucch_deltaF_Format1b[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2                    pucch_deltaF_Format2[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2a                   pucch_deltaF_Format2a[1+MAX_NUM_CCs];
  e_DeltaFList_PUCCH__deltaF_PUCCH_Format2b                   pucch_deltaF_Format2b[1+MAX_NUM_CCs];
  long                    rach_numberOfRA_Preambles[1+MAX_NUM_CCs];
  BOOLEAN_t               rach_preamblesGroupAConfig[1+MAX_NUM_CCs];
  long                    rach_sizeOfRA_PreamblesGroupA[1+MAX_NUM_CCs];
  long                    rach_messageSizeGroupA[1+MAX_NUM_CCs];
  e_RACH_ConfigCommon__preambleInfo__preamblesGroupAConfig__messagePowerOffsetGroupB                    rach_messagePowerOffsetGroupB[1+MAX_NUM_CCs];
  long                    rach_powerRampingStep[1+MAX_NUM_CCs];
  long                    rach_preambleInitialReceivedTargetPower[1+MAX_NUM_CCs];
  long                    rach_preambleTransMax[1+MAX_NUM_CCs];
  long                    rach_raResponseWindowSize[1+MAX_NUM_CCs];
  long                    rach_macContentionResolutionTimer[1+MAX_NUM_CCs];
  long                    rach_maxHARQ_Msg3Tx[1+MAX_NUM_CCs];
  long                    bcch_modificationPeriodCoeff[1+MAX_NUM_CCs];
  long                    pcch_defaultPagingCycle[1+MAX_NUM_CCs];
  long                    pcch_nB[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t300[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t301[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t310[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_t311[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_n310[1+MAX_NUM_CCs];
  long                    ue_TimersAndConstants_n311[1+MAX_NUM_CCs];
#else
   RrcConfigurationReq    RrcReq;
#endif
  long                    ue_TransmissionMode[1+MAX_NUM_CCs];
  //
  /////////////////////////////////// NB-IoT /////////////////////////////////////////////////////////
  //
  long              rach_raResponseWindowSize_NB[1+MAX_NUM_CCs];
  long              rach_macContentionResolutionTimer_NB[1+MAX_NUM_CCs];
  long              rach_powerRampingStep_NB[1+MAX_NUM_CCs];
  long              rach_preambleInitialReceivedTargetPower_NB[1+MAX_NUM_CCs];
  long              rach_preambleTransMax_CE_NB[1+MAX_NUM_CCs];
  long              bcch_modificationPeriodCoeff_NB[1+MAX_NUM_CCs];
  long              pcch_defaultPagingCycle_NB[1+MAX_NUM_CCs];
  long              nprach_CP_Length[1+MAX_NUM_CCs];
  long              nprach_rsrp_range[1+MAX_NUM_CCs];
  char*             nprach_SubcarrierMSG3_RangeStart;
  long              maxNumPreambleAttemptCE_NB[1+MAX_NUM_CCs];
  long              npdsch_nrs_Power[1+MAX_NUM_CCs];
  long              npusch_ack_nack_numRepetitions_NB[1+MAX_NUM_CCs];
  long              npusch_srs_SubframeConfig_NB[1+MAX_NUM_CCs];
  long              npusch_threeTone_CyclicShift_r13[1+MAX_NUM_CCs];
  long              npusch_sixTone_CyclicShift_r13[1+MAX_NUM_CCs];
  BOOLEAN_t         npusch_groupHoppingEnabled[1+MAX_NUM_CCs];
  long              npusch_groupAssignmentNPUSCH_r13[1+MAX_NUM_CCs];
  long              dl_GapThreshold_NB[1+MAX_NUM_CCs];
  long              dl_GapPeriodicity_NB[1+MAX_NUM_CCs];
  char*       dl_GapDurationCoeff_NB;
  long              npusch_p0_NominalNPUSCH[1+MAX_NUM_CCs];
  long              npusch_alpha[1+MAX_NUM_CCs];
  long              deltaPreambleMsg3[1+MAX_NUM_CCs];

  long              ue_TimersAndConstants_t300_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t301_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t310_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_t311_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_n310_NB[1+MAX_NUM_CCs];
  long              ue_TimersAndConstants_n311_NB[1+MAX_NUM_CCs];

  long              nprach_Periodicity[1+MAX_NUM_CCs];
  long              nprach_StartTime[1+MAX_NUM_CCs];
  long              nprach_SubcarrierOffset[1+MAX_NUM_CCs];
  long              nprach_NumSubcarriers[1+MAX_NUM_CCs];
  long              numRepetitionsPerPreambleAttempt[1+MAX_NUM_CCs];
  long              npdcch_NumRepetitions_RA[1+MAX_NUM_CCs];
  long              npdcch_StartSF_CSS_RA[1+MAX_NUM_CCs];
  char*             npdcch_Offset_RA;

  /////////////////////////////////////////////END///////////////////////////////////////////////////////
  long                    srb1_timer_poll_retransmit;
  long                    srb1_timer_reordering;
  long                    srb1_timer_status_prohibit;
  long                    srb1_poll_pdu;
  long                    srb1_poll_byte;
  long                    srb1_max_retx_threshold;
  // Nb of MME to connect to /
  uint8_t             nb_mme;
  // List of MME to connect to /
  mme_ip_address_t    mme_ip_address[S1AP_MAX_NB_MME_IP_ADDRESS];

  int                 sctp_in_streams;
  int                 sctp_out_streams;

  char               *enb_interface_name_for_S1U;
  in_addr_t           enb_ipv4_address_for_S1U;
  tcp_udp_port_t      enb_port_for_S1U;

  char               *enb_interface_name_for_S1_MME;
  in_addr_t           enb_ipv4_address_for_S1_MME;

  char               *flexran_agent_interface_name;
  in_addr_t           flexran_agent_ipv4_address;
  tcp_udp_port_t      flexran_agent_port;
  char               *flexran_agent_cache;

  // Nb of RRH to connect to /
  uint8_t             nb_rrh_gw;
  char               *rrh_gw_if_name;
  // List of MME to connect to /
  rrh_gw_config_t       rrh_gw_config[4];

#ifndef OCP_FRAMEWORK
  // otg config
  // Nb of OTG elements 
  uint8_t            num_otg_elements;
  // element config
  uint16_t          otg_ue_id[NB_MODULES_MAX+1];
  uint8_t          otg_app_type[NB_MODULES_MAX+1];
  uint8_t            otg_bg_traffic[NB_MODULES_MAX+1];
  // log config
  int16_t           glog_level;
  int16_t           glog_verbosity;
  int16_t           hw_log_level;
  int16_t           hw_log_verbosity;
  int16_t           phy_log_level;
  int16_t           phy_log_verbosity;
  int16_t           mac_log_level;
  int16_t           mac_log_verbosity;
  int16_t           rlc_log_level;
  int16_t           rlc_log_verbosity;
  int16_t           pdcp_log_level;
  int16_t           pdcp_log_verbosity;
  int16_t           rrc_log_level;
  int16_t           rrc_log_verbosity;
  int16_t           gtpu_log_level;
  int16_t           gtpu_log_verbosity;
  int16_t           udp_log_level;
  int16_t           udp_log_verbosity;
  int16_t           osa_log_level;
  int16_t           osa_log_verbosity;
#endif
} Enb_properties_NB_IoT_t;
*/
/*
typedef struct Enb_properties_array_NB_IoT_s {
  int                  number;
  Enb_properties_NB_IoT_t    *properties[MAX_ENB];
} Enb_properties_array_NB_IoT_t;

void                          enb_config_display_NB_IoT(void);

const Enb_properties_array_NB_IoT_t *enb_config_init_NB_IoT(char* lib_config_file_name_pP);

const Enb_properties_array_NB_IoT_t *enb_config_get_NB_IoT(void);
*/
#endif 

/* ENB_CONFIG_H_ */
/** @} */
