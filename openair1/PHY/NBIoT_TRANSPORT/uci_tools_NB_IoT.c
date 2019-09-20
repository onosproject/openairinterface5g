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

/*! \file PHY/LTE_TRANSPORT/phich.c
* \brief Routines for generation of and computations regarding the uplink control information (UCI) for PUSCH. V8.6 2009-03
* \author R. Knopp, F. Kaltenberger, A. Bhamri
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr, florian.kaltenberger@eurecom.fr, ankit.bhamri@eurecom.fr
* \note
* \warning
*/
#include "PHY/defs_L1_NB_IoT.h"
#include "PHY/extern_NB_IoT.h"
#ifdef DEBUG_UCI_TOOLS
#include "PHY/vars_NB_IoT.h"
#endif


void do_diff_cqi_NB_IoT(uint8_t N_RB_DL,
                        uint8_t *DL_subband_cqi,
                        uint8_t DL_cqi,
                        uint32_t diffcqi1)
{

  uint8_t nb_sb,i,offset;

  // This is table 7.2.1-3 from 36.213 (with k replaced by the number of subbands, nb_sb)
  switch (N_RB_DL) {
  case 6:
    nb_sb=1;
    break;

  case 15:
    nb_sb = 4;
    break;

  case 25:
    nb_sb = 7;
    break;

  case 50:
    nb_sb = 9;
    break;

  case 75:
    nb_sb = 10;
    break;

  case 100:
    nb_sb = 13;
    break;

  default:
    nb_sb=0;
    break;
  }

  memset(DL_subband_cqi,0,13);

  for (i=0; i<nb_sb; i++) {
    offset = (diffcqi1>>(2*i))&3;

    if (offset == 3)
      DL_subband_cqi[i] = DL_cqi - 1;
    else
      DL_subband_cqi[i] = DL_cqi + offset;
  }
}

void extract_CQI_NB_IoT(void *o,UCI_format_NB_IoT_t uci_format,NB_IoT_eNB_UE_stats *stats, uint8_t N_RB_DL, uint16_t * crnti, uint8_t * access_mode)
{

  
  uint8_t i;
  LOG_D(PHY,"[eNB][UCI] N_RB_DL %d uci format %d\n", N_RB_DL,uci_format);

  switch(N_RB_DL) {
  case 6:
    switch(uci_format) {
    case wideband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank1_2A_1_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_pmi_single = ((wideband_cqi_rank1_2A_1_5MHz_NB_IoT *)o)->pmi;
      break;

    case wideband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((wideband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      stats->DL_pmi_dual   = ((wideband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_nopmi_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_nopmi_1_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],((HLC_subband_cqi_nopmi_1_5MHz_NB_IoT *)o)->diffcqi1);
      break;

    case HLC_subband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank1_2A_1_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank1_2A_1_5MHz_NB_IoT *)o)->diffcqi1));
      stats->DL_pmi_single = ((HLC_subband_cqi_rank1_2A_1_5MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((HLC_subband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->diffcqi1));
      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[1],stats->DL_cqi[1],(((HLC_subband_cqi_rank2_2A_1_5MHz *)o)->diffcqi2));
      stats->DL_pmi_dual   = ((HLC_subband_cqi_rank2_2A_1_5MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_mcs_CBA_NB_IoT:
      if ((*crnti == ((HLC_subband_cqi_mcs_CBA_1_5MHz_NB_IoT *)o)->crnti) && (*crnti !=0)) {
        *access_mode=CBA_ACCESS;
        LOG_N(PHY,"[eNB] UCI for CBA : mcs %d  crnti %x\n",
              ((HLC_subband_cqi_mcs_CBA_1_5MHz_NB_IoT *)o)->mcs, ((HLC_subband_cqi_mcs_CBA_1_5MHz_NB_IoT *)o)->crnti);
      } else {
        LOG_D(PHY,"[eNB] UCI for CBA : rnti (enb context %x, rx uci %x) invalid, unknown access\n",
              *crnti, ((HLC_subband_cqi_mcs_CBA_1_5MHz_NB_IoT *)o)->crnti);
      }

      break;

    case unknown_cqi:
    default:
      LOG_N(PHY,"[eNB][UCI] received unknown uci (rb %d)\n",N_RB_DL);
      break;
    }

    break;

  case 25:

    switch(uci_format) {
    case wideband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank1_2A_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_pmi_single = ((wideband_cqi_rank1_2A_5MHz_NB_IoT *)o)->pmi;
      break;

    case wideband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank2_2A_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((wideband_cqi_rank2_2A_5MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      stats->DL_pmi_dual   = ((wideband_cqi_rank2_2A_5MHz_NB_IoT *)o)->pmi;
      //this translates the 2-layer PMI into a single layer PMI for the first codeword
      //the PMI for the second codeword will be stats->DL_pmi_single^0x1555
      stats->DL_pmi_single = 0;
      for (i=0;i<7;i++)
	stats->DL_pmi_single = stats->DL_pmi_single | (((stats->DL_pmi_dual&(1<i))>>i)*2)<<2*i;  
      break;

    case HLC_subband_cqi_nopmi_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_nopmi_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],((HLC_subband_cqi_nopmi_5MHz_NB_IoT *)o)->diffcqi1);
      break;

    case HLC_subband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank1_2A_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank1_2A_5MHz_NB_IoT *)o)->diffcqi1));
      stats->DL_pmi_single = ((HLC_subband_cqi_rank1_2A_5MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank2_2A_5MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((HLC_subband_cqi_rank2_2A_5MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank2_2A_5MHz_NB_IoT *)o)->diffcqi1));
      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[1],stats->DL_cqi[1],(((HLC_subband_cqi_rank2_2A_5MHz_NB_IoT *)o)->diffcqi2));
      stats->DL_pmi_dual   = ((HLC_subband_cqi_rank2_2A_5MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_mcs_CBA_NB_IoT:
      if ((*crnti == ((HLC_subband_cqi_mcs_CBA_5MHz_NB_IoT *)o)->crnti) && (*crnti !=0)) {
        *access_mode=CBA_ACCESS;
        LOG_N(PHY,"[eNB] UCI for CBA : mcs %d  crnti %x\n",
              ((HLC_subband_cqi_mcs_CBA_5MHz_NB_IoT *)o)->mcs, ((HLC_subband_cqi_mcs_CBA_5MHz_NB_IoT *)o)->crnti);
      } else {
        LOG_D(PHY,"[eNB] UCI for CBA : rnti (enb context %x, rx uci %x) invalid, unknown access\n",
              *crnti, ((HLC_subband_cqi_mcs_CBA_5MHz_NB_IoT *)o)->crnti);
      }

      break;

    case unknown_cqi_NB_IoT:
    default:
      LOG_N(PHY,"[eNB][UCI] received unknown uci (rb %d)\n",N_RB_DL);
      break;
    }

    break;

  case 50:
    switch(uci_format) {
    case wideband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank1_2A_10MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_pmi_single = ((wideband_cqi_rank1_2A_10MHz_NB_IoT *)o)->pmi;
      break;

    case wideband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank2_2A_10MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((wideband_cqi_rank2_2A_10MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      stats->DL_pmi_dual   = ((wideband_cqi_rank2_2A_10MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_nopmi_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_nopmi_10MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],((HLC_subband_cqi_nopmi_10MHz_NB_IoT *)o)->diffcqi1);
      break;

    case HLC_subband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank1_2A_10MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank1_2A_10MHz_NB_IoT *)o)->diffcqi1));
      stats->DL_pmi_single = ((HLC_subband_cqi_rank1_2A_10MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank2_2A_10MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((HLC_subband_cqi_rank2_2A_10MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank2_2A_10MHz_NB_IoT *)o)->diffcqi1));
      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[1],stats->DL_cqi[1],(((HLC_subband_cqi_rank2_2A_10MHz_NB_IoT *)o)->diffcqi2));
      stats->DL_pmi_dual   = ((HLC_subband_cqi_rank2_2A_10MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_mcs_CBA_NB_IoT:
      if ((*crnti == ((HLC_subband_cqi_mcs_CBA_10MHz_NB_IoT *)o)->crnti) && (*crnti !=0)) {
        *access_mode=CBA_ACCESS;
        LOG_N(PHY,"[eNB] UCI for CBA : mcs %d  crnti %x\n",
              ((HLC_subband_cqi_mcs_CBA_10MHz_NB_IoT *)o)->mcs, ((HLC_subband_cqi_mcs_CBA_10MHz_NB_IoT *)o)->crnti);
      } else {
        LOG_D(PHY,"[eNB] UCI for CBA : rnti (enb context %x, rx uci %x) invalid, unknown access\n",
              *crnti, ((HLC_subband_cqi_mcs_CBA_10MHz_NB_IoT *)o)->crnti);
      }

      break;

    case unknown_cqi_NB_IoT:
    default:
      LOG_N(PHY,"[eNB][UCI] received unknown uci (RB %d)\n",N_RB_DL);
      break;
    }

    break;

  case 100:
    switch(uci_format) {
    case wideband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank1_2A_20MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_pmi_single = ((wideband_cqi_rank1_2A_20MHz_NB_IoT *)o)->pmi;
      break;

    case wideband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((wideband_cqi_rank2_2A_20MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((wideband_cqi_rank2_2A_20MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      stats->DL_pmi_dual   = ((wideband_cqi_rank2_2A_20MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_nopmi_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_nopmi_20MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],((HLC_subband_cqi_nopmi_20MHz_NB_IoT *)o)->diffcqi1);
      break;

    case HLC_subband_cqi_rank1_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank1_2A_20MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank1_2A_20MHz_NB_IoT *)o)->diffcqi1));
      stats->DL_pmi_single = ((HLC_subband_cqi_rank1_2A_20MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_rank2_2A_NB_IoT:
      stats->DL_cqi[0]     = (((HLC_subband_cqi_rank2_2A_20MHz_NB_IoT *)o)->cqi1);

      if (stats->DL_cqi[0] > 24)
        stats->DL_cqi[0] = 24;

      stats->DL_cqi[1]     = (((HLC_subband_cqi_rank2_2A_20MHz_NB_IoT *)o)->cqi2);

      if (stats->DL_cqi[1] > 24)
        stats->DL_cqi[1] = 24;

      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[0],stats->DL_cqi[0],(((HLC_subband_cqi_rank2_2A_20MHz_NB_IoT *)o)->diffcqi1));
      do_diff_cqi_NB_IoT(N_RB_DL,stats->DL_subband_cqi[1],stats->DL_cqi[1],(((HLC_subband_cqi_rank2_2A_20MHz_NB_IoT *)o)->diffcqi2));
      stats->DL_pmi_dual   = ((HLC_subband_cqi_rank2_2A_20MHz_NB_IoT *)o)->pmi;
      break;

    case HLC_subband_cqi_mcs_CBA_NB_IoT:
      if ((*crnti == ((HLC_subband_cqi_mcs_CBA_20MHz_NB_IoT *)o)->crnti) && (*crnti !=0)) {
        *access_mode=CBA_ACCESS;
        LOG_N(PHY,"[eNB] UCI for CBA : mcs %d  crnti %x\n",
              ((HLC_subband_cqi_mcs_CBA_20MHz_NB_IoT *)o)->mcs, ((HLC_subband_cqi_mcs_CBA_20MHz_NB_IoT *)o)->crnti);
      } else {
        LOG_D(PHY,"[eNB] UCI for CBA : rnti (enb context %x, rx uci %x) invalid, unknown access\n",
              *crnti, ((HLC_subband_cqi_mcs_CBA_20MHz_NB_IoT *)o)->crnti);
      }

      break;

    case unknown_cqi_NB_IoT:
    default:
      LOG_N(PHY,"[eNB][UCI] received unknown uci (RB %d)\n",N_RB_DL);
      break;
    }

    break;

  default:
    LOG_N(PHY,"[eNB][UCI] unknown RB %d\n",N_RB_DL);
    break;
  }


}

