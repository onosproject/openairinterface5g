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

/*
                                enb_config.c
                             -------------------
  AUTHOR  : Lionel GAUTHIER, navid nikaein, Laurent Winckel
  COMPANY : EURECOM
  EMAIL   : Lionel.Gauthier@eurecom.fr, navid.nikaein@eurecom.fr
 */

#include <string.h>
#include <libconfig.h>
#include <inttypes.h>

#include "common/utils/LOG/log.h"
#include "common/utils/LOG/log_extern.h"
#include "assertions.h"
#include "enb_config.h"
#include "intertask_interface.h"

#include "LTE_SystemInformationBlockType2.h"
#include "LAYER2/MAC/mac_extern.h"
#include "PHY/phy_extern.h"

#include "LTE_SystemInformationBlockType2-NB-r13.h"


void fill_NB_IoT_configuration(MessageDef *msg_p, ccparams_NB_IoT_t *NBconfig, int cell_idx, int cc_idx, char *config_fname, char *NBparamspath)
{

  printf("Found parameters for NB-IoT from %s : %s\n",config_fname,NBparamspath);
  
        //************************************ NB-IoT part ***************************************************************

    switch (NBconfig->rach_raResponseWindowSize_NB) {
              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp2;
                break;

              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp4;
                break;

              case 5:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp5;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp6;
                break;

              case 7:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp7;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp8;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_raResponseWindowSize_NB[cc_idx] = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp10;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for rach_raResponseWindowSize_NB choice: 2,3,4,5,6,7,8,10 ",
                             config_fname, cell_idx, rach_raResponseWindowSize_NB);
                break;

              }

      switch (NBconfig->rach_macContentionResolutionTimer_NB) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp2;
                break;

              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp4;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_macContentionResolutionTimer_NB[cc_idx] = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp64;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for rach_macContentionResolutionTimer_NB choice: 1,2,3,4,8,16,32,64 ",
                             config_fname, cell_idx, rach_macContentionResolutionTimer_NB);
                break;

              }

      switch (NBconfig->rach_powerRampingStep_NB) {
              case 0:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_powerRampingStep_NB[cc_idx] = PowerRampingParameters__powerRampingStep_dB0;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_powerRampingStep_NB[cc_idx] = PowerRampingParameters__powerRampingStep_dB2;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_powerRampingStep_NB[cc_idx] = PowerRampingParameters__powerRampingStep_dB4;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_powerRampingStep_NB[cc_idx] = PowerRampingParameters__powerRampingStep_dB6;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for rach_powerRampingStep_NB choice: 0,2,4,6 ",
                             config_fname, cell_idx, rach_powerRampingStep_NB);
                break;

              }

      switch (NBconfig->rach_preambleInitialReceivedTargetPower_NB) {
              case -120:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_120;
                break;

              case -118:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_118;
                break;

              case -116:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_116;
                break;

              case -114:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_114;
                break;

              case -112:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_112;
                break;

              case -110:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_110;
                break;

              case -108:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_108;
                break;

              case -106:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_106;
                break;

              case -104:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_104;
                break;

              case -102:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_102;
                break;

              case -100:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_100;
                break;

              case -98:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_98;
                break;

              case -96:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_96;
                break;

              case -94:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_94;
                break;

              case -92:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_92;
                break;

              case -90:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleInitialReceivedTargetPower_NB[cc_idx] = PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_90;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for rach_preambleInitialReceivedTargetPower_NB choice: -120,-118,-116,-114,-112,-110,-108,-106,-104,-102,-100,-98,-96,-94,-92,-90 ",
                             config_fname, cell_idx, rach_preambleInitialReceivedTargetPower_NB);
                break;

              }

      switch (NBconfig->rach_preambleTransMax_CE_NB) {
              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n4;
                break;

              case 5:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n5;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n6;
                break;

              case 7:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n7;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n8;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n10;
                break;

              case 20:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n20;
                break;

              case 50:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n50;
                break;

              case 100:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n100;
                break;

              case 200:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).rach_preambleTransMax_CE_NB[cc_idx] = PreambleTransMax_n200;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for rach_preambleTransMax_CE_NB choice: 3,4,5,6,7,8,10,20,50,100,200 ",
                             config_fname, cell_idx, rach_preambleTransMax_CE_NB);
                break;

              }

      switch (NBconfig->bcch_modificationPeriodCoeff_NB) {
              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).bcch_modificationPeriodCoeff_NB[cc_idx] = BCCH_Config_NB_r13__modificationPeriodCoeff_r13_n16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).bcch_modificationPeriodCoeff_NB[cc_idx] = BCCH_Config_NB_r13__modificationPeriodCoeff_r13_n16;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).bcch_modificationPeriodCoeff_NB[cc_idx] = BCCH_Config_NB_r13__modificationPeriodCoeff_r13_n16;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).bcch_modificationPeriodCoeff_NB[cc_idx] = BCCH_Config_NB_r13__modificationPeriodCoeff_r13_n16;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for bcch_modificationPeriodCoeff_NB choice: 16,32,64,128 ",
                             config_fname, cell_idx, bcch_modificationPeriodCoeff_NB);
                break;

              }

      switch (NBconfig->pcch_defaultPagingCycle_NB) {
              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).pcch_defaultPagingCycle_NB[cc_idx] = PCCH_Config_NB_r13__defaultPagingCycle_r13_rf128;
                break;

              case 256:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).pcch_defaultPagingCycle_NB[cc_idx] = PCCH_Config_NB_r13__defaultPagingCycle_r13_rf256;
                break;

              case 512:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).pcch_defaultPagingCycle_NB[cc_idx] = PCCH_Config_NB_r13__defaultPagingCycle_r13_rf512;
                break;

              case 1024:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).pcch_defaultPagingCycle_NB[cc_idx] = PCCH_Config_NB_r13__defaultPagingCycle_r13_rf1024;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for pcch_defaultPagingCycle_NB choice: 128,256,512,1024 ",
                             config_fname, cell_idx, pcch_defaultPagingCycle_NB);
                break;

              }

      switch (NBconfig->nprach_CP_Length) {
              case 0:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_CP_Length[cc_idx] = NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us66dot7;
                break;

              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_CP_Length[cc_idx] = NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us266dot7;
                break;


              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_CP_Length choice: 0,1 ",
                             config_fname, cell_idx, nprach_CP_Length);
                break;

              }

      NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_rsrp_range[cc_idx] =  nprach_rsrp_range;
 
              if ((nprach_rsrp_range<0)||(nprach_rsrp_range>96))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_rsrp_range choice: 0..96 !\n",
                             config_fname, cell_idx, nprach_rsrp_range);

      NBIOTRRC_CONFIGURATION_REQ(msg_p).npdsch_nrs_Power[cc_idx] =  npdsch_nrs_Power;
 
              if ((npdsch_nrs_Power<-60)||(npdsch_nrs_Power>50))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npdsch_nrs_Power choice: -60..50 !\n",
                             config_fname, cell_idx, npdsch_nrs_Power);

      
      switch (NBconfig->npusch_ack_nack_numRepetitions_NB) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r2;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r4;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r64;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_ack_nack_numRepetitions_NB[cc_idx] = ACK_NACK_NumRepetitions_NB_r13_r128;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_ack_nack_numRepetitions_NB choice: 1,2,4,8,16,32,64,128 ",
                             config_fname, cell_idx, npusch_ack_nack_numRepetitions_NB);
                break;

              }

      switch (NBconfig->npusch_srs_SubframeConfig_NB) {
              case 0:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc0;
                break;

              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc2;
                break;

              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc4;
                break;

              case 5:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc5;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc6;
                break;

              case 7:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc7;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc8;
                break;

              case 9:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc9;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc10;
                break;

              case 11:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc11;
                break;

              case 12:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc12;
                break;

              case 13:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc13;
                break;

              case 14:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc14;
                break;

              case 15:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_srs_SubframeConfig_NB[cc_idx] = NPUSCH_ConfigCommon_NB_r13__srs_SubframeConfig_r13_sc15;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_srs_SubframeConfig_NB choice: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 ",
                             config_fname, cell_idx, npusch_srs_SubframeConfig_NB);
                break;

              }

      NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_threeTone_CyclicShift_r13[cc_idx] =  npusch_threeTone_CyclicShift_r13;
 
              if ((npusch_threeTone_CyclicShift_r13<0)||(npusch_threeTone_CyclicShift_r13>2))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_threeTone_CyclicShift_r13 choice: 0..2 !\n",
                             config_fname, cell_idx, npusch_threeTone_CyclicShift_r13);

      NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_sixTone_CyclicShift_r13[cc_idx] =  npusch_sixTone_CyclicShift_r13;
 
              if ((npusch_sixTone_CyclicShift_r13<0)||(npusch_sixTone_CyclicShift_r13>3))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_sixTone_CyclicShift_r13 choice: 0..3 !\n",
                             config_fname, cell_idx, npusch_sixTone_CyclicShift_r13);

      if (!npusch_groupHoppingEnabled)
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d define %s: ENABLE,DISABLE!\n",
                             config_fname, cell_idx, ENB_CONFIG_STRING_NPUSCH_GROUP_HOPPING_EN_NB_IOT);
              else if (strcmp(npusch_groupHoppingEnabled, "ENABLE") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_groupHoppingEnabled[cc_idx] = TRUE;
              }  else if (strcmp(npusch_groupHoppingEnabled, "DISABLE") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_groupHoppingEnabled[cc_idx] = FALSE;
              } else
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%s\" for npusch_groupHoppingEnabled choice: ENABLE,DISABLE!\n",
                             config_fname, cell_idx, npusch_groupHoppingEnabled);


      NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_groupAssignmentNPUSCH_r13[cc_idx] = npusch_groupAssignmentNPUSCH_r13;

              if ((npusch_groupAssignmentNPUSCH_r13<0)||(npusch_groupAssignmentNPUSCH_r13>29))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_groupAssignmentNPUSCH_r13 choice: 0..29!\n",
                             config_fname, cell_idx, npusch_groupAssignmentNPUSCH_r13);

      switch (NBconfig->dl_GapThreshold_NB) {
              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapThreshold_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapThreshold_r13_n32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapThreshold_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapThreshold_r13_n64;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapThreshold_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapThreshold_r13_n128;
                break;

              case 256:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapThreshold_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapThreshold_r13_n256;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for dl_GapThreshold_NB choice: 32,64,128,256 ",
                             config_fname, cell_idx, dl_GapThreshold_NB);
                break;

              }

      switch (NBconfig->dl_GapPeriodicity_NB) {
              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapPeriodicity_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapPeriodicity_r13_sf64;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapPeriodicity_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapPeriodicity_r13_sf128;
                break;

              case 256:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapPeriodicity_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapPeriodicity_r13_sf256;
                break;

              case 512:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).dl_GapPeriodicity_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapPeriodicity_r13_sf512;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for dl_GapPeriodicity_NB choice: 64,128,256,512 ",
                             config_fname, cell_idx, dl_GapPeriodicity_NB);
                break;

              }

      if (strcmp(NBconfig->dl_GapDurationCoeff_NB, "oneEighth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->dl_GapDurationCoeff_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapDurationCoeff_r13_oneEighth;
              } else if (strcmp(NBconfig->dl_GapDurationCoeff_NB, "oneFourth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->dl_GapDurationCoeff_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapDurationCoeff_r13_oneFourth;
              } else if (strcmp(NBconfig->dl_GapDurationCoeff_NB, "threeEighth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->dl_GapDurationCoeff_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapDurationCoeff_r13_threeEighth;
              } else if (strcmp(NBconfig->dl_GapDurationCoeff_NB, "oneHalf") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->dl_GapDurationCoeff_NB[cc_idx] = DL_GapConfig_NB_r13__dl_GapDurationCoeff_r13_oneHalf;
              } else
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%s\" for NBconfig->dl_GapDurationCoeff_NB choice: oneEighth,oneFourth,threeEighth,oneHalf !\n",
                             config_fname, cell_idx, NBconfig->dl_GapDurationCoeff_NB);

      NBIOTRRC_CONFIGURATION_REQ(msg_p).npusch_p0_NominalNPUSCH[cc_idx] =  npusch_p0_NominalNPUSCH;
 
              if ((npusch_p0_NominalNPUSCH < -126)||(npusch_p0_NominalNPUSCH> 24))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npusch_p0_NominalNPUSCH choice: -126..24 !\n",
                             config_fname, cell_idx, npusch_p0_NominalNPUSCH);

     if (strcmp(NBconfig->npusch_alpha,"AL0")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al0;
              } else if (strcmp(NBconfig->npusch_alpha,"AL04")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al04;
              } else if (strcmp(NBconfig->npusch_alpha,"AL05")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al05;
              } else if (strcmp(NBconfig->npusch_alpha,"AL06")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al06;
              } else if (strcmp(NBconfig->npusch_alpha,"AL07")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al07;
              } else if (strcmp(NBconfig->npusch_alpha,"AL08")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al08;
              } else if (strcmp(NBconfig->npusch_alpha,"AL09")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al09;
              } else if (strcmp(NBconfig->npusch_alpha,"AL1")==0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).NBconfig->npusch_alpha[cc_idx] = UplinkPowerControlCommon_NB_r13__alpha_r13_al1;
              } else
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%s\" for NBconfig->npusch_alpha choice: AL0,AL04,AL05,AL06,AL07,AL08,AL09,AL1!\n",
                             config_fname, cell_idx, NBconfig->npusch_alpha);

        NBIOTRRC_CONFIGURATION_REQ(msg_p).deltaPreambleMsg3[cc_idx] =  deltaPreambleMsg3;
 
              if ((deltaPreambleMsg3 < -1)||(deltaPreambleMsg3> 6))
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for deltaPreambleMsg3 choice: -1..6 !\n",
                             config_fname, cell_idx, deltaPreambleMsg3);
      

        //************************************************************************* NB-IoT Timer ************************************************************
        switch (NBconfig->ue_TimersAndConstants_t300_NB) {
              case 2500:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms2500;
                break;

              case 4000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms4000;
                break;

              case 6000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms6000;
                break;

              case 10000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms10000;
                break;

              case 15000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms15000;
                break;

              case 25000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms25000;
                break;

              case 40000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms40000;
                break;

              case 60000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t300_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t300_r13_ms60000;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_t300_NB choice: 2500,4000,6000,10000,15000,25000,40000,60000 ",
                             config_fname, cell_idx, ue_TimersAndConstants_t300_NB);
                break;

              }
        switch (NBconfig->ue_TimersAndConstants_t301_NB) {
              case 2500:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms2500;
                break;

              case 4000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms4000;
                break;

              case 6000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms6000;
                break;

              case 10000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms10000;
                break;

              case 15000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms15000;
                break;

              case 25000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms25000;
                break;

              case 40000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms40000;
                break;

              case 60000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t301_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t301_r13_ms60000;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_t301_NB choice: 2500,4000,6000,10000,15000,25000,40000,60000 ",
                             config_fname, cell_idx, ue_TimersAndConstants_t301_NB);
                break;

              }

        switch (NBconfig->ue_TimersAndConstants_t310_NB) {
              case 0:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms0;
                break;

              case 200:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms200;
                break;

              case 500:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms500;
                break;

              case 1000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms1000;
                break;

              case 2000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms2000;
                break;

              case 4000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms4000;
                break;

              case 8000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t310_r13_ms8000;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_t310_NB choice: 0,200,500,1000,2000,4000,8000 ",
                             config_fname, cell_idx, ue_TimersAndConstants_t310_NB);
                break;

              }

        switch (NBconfig->ue_TimersAndConstants_t311_NB) {
              case 1000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms1000;
                break;

              case 3000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms3000;
                break;

              case 5000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms5000;
                break;

              case 10000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms10000;
                break;

              case 15000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms15000;
                break;

              case 20000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms20000;
                break;

              case 30000:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_t311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__t311_r13_ms30000;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_t311_NB choice: 1000,3000,5000,10000,150000,20000,30000",
                             config_fname, cell_idx, ue_TimersAndConstants_t311_NB);
                break;

              }

        switch (NBconfig->ue_TimersAndConstants_n310_NB) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n2;
                break;

              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n4;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n6;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n8;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n10;
                break;

              case 20:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n310_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n310_r13_n20;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_n310_NB choice: 1,2,3,4,6,8,10,20",
                             config_fname, cell_idx, ue_TimersAndConstants_n310_NB);
                break;

              }

        switch (NBconfig->ue_TimersAndConstants_n311_NB) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n2;
                break;

              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n4;
                break;

              case 5:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n5;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n6;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n8;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).ue_TimersAndConstants_n311_NB[cc_idx] = UE_TimersAndConstants_NB_r13__n311_r13_n10;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for ue_TimersAndConstants_n311_NB choice: 1,2,3,4,5,6,8,10",
                             config_fname, cell_idx, ue_TimersAndConstants_n311_NB);
                break;

              }

        //************************************************************************** NBPRACH NB-IoT *****************************************************
        switch (NBconfig->nprach_Periodicity) {
              case 40:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms40;
                break;

              case 80:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms80;
                break;

              case 160:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms160;
                break;

              case 240:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms240;
                break;

                case 320:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms320;
                break;

              case 640:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms640;
                break;

              case 1280:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms1280;
                break;

              case 2560:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_Periodicity[cc_idx] = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms2560;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_Periodicity choice: 40,80,160,240,320,640,1280,2560",
                             config_fname, cell_idx, nprach_Periodicity);

                break;
              }

      switch (NBconfig->nprach_StartTime) {
              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms64;
                break;

                case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms128;
                break;

              case 256:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms256;
                break;

              case 512:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms512;
                break;

              case 1024:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_StartTime[cc_idx] = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms1024;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_StartTime choice: 8,16,32,64,128,256,512,1024",
                             config_fname, cell_idx, nprach_StartTime);

                break;
              }

        switch (NBconfig->nprach_SubcarrierOffset) {
              case 40:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n0;
                break;

              case 80:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n12;
                break;

              case 160:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n24;
                break;

              case 240:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n36;
                break;

                case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n2;
                break;

              case 18:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n18;
                break;

              case 34:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n34;
                break;

              //case 1:
               // NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierOffset[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_spare1;
               // break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_SubcarrierOffset choice: 0,12,24,36,2,18,34",
                             config_fname, cell_idx, nprach_SubcarrierOffset);

                break;
              }

        switch (NBconfig->nprach_NumSubcarriers) {
              case 12:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_NumSubcarriers[cc_idx] = NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n12;
                break;

              case 24:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_NumSubcarriers[cc_idx] = NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n24;
                break;

              case 36:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_NumSubcarriers[cc_idx] = NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n36;
                break;

              case 48:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_NumSubcarriers[cc_idx] = NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n48;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for nprach_NumSubcarriers choice: 12,24,36,48",
                             config_fname, cell_idx, nprach_NumSubcarriers);

                break;
              }
                
        if (strcmp(NBconfig->nprach_SubcarrierMSG3_RangeStart, "zero") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierMSG3_RangeStart[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierMSG3_RangeStart_r13_zero;
              } else if (strcmp(NBconfig->nprach_SubcarrierMSG3_RangeStart, "oneThird") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierMSG3_RangeStart[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierMSG3_RangeStart_r13_oneThird;
              } else if (strcmp(NBconfig->nprach_SubcarrierMSG3_RangeStart, "twoThird") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierMSG3_RangeStart[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierMSG3_RangeStart_r13_twoThird;
              } else if (strcmp(NBconfig->nprach_SubcarrierMSG3_RangeStart, "one") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).nprach_SubcarrierMSG3_RangeStart[cc_idx] = NPRACH_Parameters_NB_r13__nprach_SubcarrierMSG3_RangeStart_r13_one;
              } else
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%s\" for nprach_SubcarrierMSG3_RangeStart choice: zero,oneThird,twoThird,one !\n",
                             config_fname, cell_idx, nprach_SubcarrierMSG3_RangeStart);

        switch (NBconfig->maxNumPreambleAttemptCE_NB) {
              case 3:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n3;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n4;
                break;

              case 5:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n5;
                break;

              case 6:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n6;
                break;

              case 7:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n7;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n8;
                break;

              case 10:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).maxNumPreambleAttemptCE_NB[cc_idx] = NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n10;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for maxNumPreambleAttemptCE_NB choice: 3,4,5,6,7,8,10",
                             config_fname, cell_idx, maxNumPreambleAttemptCE_NB);

                break;
              }

        switch (NBconfig->numRepetitionsPerPreambleAttempt) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n2;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n4;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n64;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).numRepetitionsPerPreambleAttempt[cc_idx] = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n128;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for numRepetitionsPerPreambleAttempt choice: 1,2,4,8,16,32,64,128",
                             config_fname, cell_idx, numRepetitionsPerPreambleAttempt);

                break;
              }

        switch (NBconfig->npdcch_NumRepetitions_RA) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r1;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r2;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r4;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r64;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r128;
                break;

              case 256:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r256;
                break;

              case 512:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r512;
                break;

              case 1024:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r1024;
                break;

              case 2048:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_NumRepetitions_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r2048;
                break; 

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npdcch_NumRepetitions_RA choice: 1,2,4,8,16,32,64,128,512,1024,2048",
                             config_fname, cell_idx, npdcch_NumRepetitions_RA);

                break;
              }

      switch (NBconfig->npdcch_StartSF_CSS_RA) {
              case 1:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v1dot5;
                break;

              case 2:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v2;
                break;

              case 4:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v4;
                break;

              case 8:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v8;
                break;

              case 16:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v16;
                break;

              case 32:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v32;
                break;

              case 64:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v48;
                break;

              case 128:
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_StartSF_CSS_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v64;
                break;

              default:
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%d\" for npdcch_StartSF_CSS_RA choice: 1.5,2,4,8,16,32,48,64",
                             config_fname, cell_idx, npdcch_StartSF_CSS_RA);

                break;
              }

        if (strcmp(NBconfig->npdcch_Offset_RA, "zero") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_Offset_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_zero;
              } else if (strcmp(NBconfig->npdcch_Offset_RA, "oneEighth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_Offset_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_oneEighth;
              } else if (strcmp(NBconfig->npdcch_Offset_RA, "oneFourth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_Offset_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_oneFourth;
              } else if (strcmp(NBconfig->npdcch_Offset_RA, "threeEighth") == 0) {
                NBIOTRRC_CONFIGURATION_REQ(msg_p).npdcch_Offset_RA[cc_idx] = NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_threeEighth;
              } else
                AssertFatal (0,
                             "Failed to parse eNB configuration file %s, enb %d unknown value \"%s\" for npdcch_Offset_RA choice: zero,oneEighth,oneFourth,threeEighth !\n",
                             config_fname, cell_idx, npdcch_Offset_RA);


        //****************************************************************************************************************
        //****************************************************************************************************************
        //****************************************************************************************************************

}


