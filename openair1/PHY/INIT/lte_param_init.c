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

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <execinfo.h>
#include <signal.h>

#include "SIMULATION/TOOLS/sim.h"
#include "PHY/types.h"
#include "PHY/defs_eNB.h"
#include "PHY/defs_UE.h"
#include "PHY/phy_extern.h"
#include "phy_init.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "PHY/LTE_TRANSPORT/transport_common_proto.h"
#include "targets/RT/USER/lte-softmodem.h"
extern PHY_VARS_eNB *eNB;
extern PHY_VARS_UE *UE;
extern RU_t *ru;
extern void  phy_init_RU(RU_t *);

void phy_measurements_eNB_malloc(PHY_MEASUREMENTS_eNB *meas)
{
  int i, j;
  meas->rx_spatial_power = (unsigned int ***)malloc(sizeof(unsigned int **)*NUMBER_OF_UE_MAX);
  meas->rx_spatial_power_dB = (unsigned short ***)malloc(sizeof(unsigned short **)*NUMBER_OF_UE_MAX);
  meas->rx_rssi_dBm = (short *)malloc(sizeof(short)*NUMBER_OF_UE_MAX);
  meas->rx_correlation = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  meas->rx_correlation_dB = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  meas->wideband_cqi = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  meas->wideband_cqi_dB = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  meas->wideband_cqi_tot = (char *)malloc(sizeof(char)*NUMBER_OF_UE_MAX);
  meas->subband_cqi = (int ***)malloc(sizeof(int **)*NUMBER_OF_UE_MAX);
  meas->subband_cqi_tot = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  meas->subband_cqi_dB = (int ***)malloc(sizeof(int **)*NUMBER_OF_UE_MAX);
  meas->subband_cqi_tot_dB = (int **)malloc(sizeof(int *)*NUMBER_OF_UE_MAX);
  for (i = 0; i < NUMBER_OF_UE_MAX; i++) {
	meas->rx_spatial_power[i] = (unsigned int **)malloc(sizeof(unsigned int *)*2);
	meas->rx_spatial_power_dB[i] = (unsigned short **)malloc(sizeof(unsigned short *)*2);
    for (j = 0; j < 2; j++) {
      meas->rx_spatial_power[i][j] = (unsigned int *)malloc(sizeof(unsigned int)*2);
      meas->rx_spatial_power_dB[i][j] = (unsigned short *)malloc(sizeof(unsigned short)*2);
    }
    meas->rx_correlation[i] = (int *)malloc(sizeof(int)*2);
    meas->rx_correlation_dB[i] = (int *)malloc(sizeof(int)*2);
    meas->wideband_cqi[i] = (int *)malloc(sizeof(int)*MAX_NUM_RU_PER_eNB);
    meas->wideband_cqi_dB[i] = (int *)malloc(sizeof(int)*MAX_NUM_RU_PER_eNB);
    meas->subband_cqi[i] = (int **)malloc(sizeof(int *)*MAX_NUM_RU_PER_eNB);
    meas->subband_cqi_dB[i] = (int **)malloc(sizeof(int *)*MAX_NUM_RU_PER_eNB);
    for (j = 0; j < MAX_NUM_RU_PER_eNB; j++) {
      meas->subband_cqi[i][j] = (int *)malloc(sizeof(int)*100);
      meas->subband_cqi_dB[i][j] = (int *)malloc(sizeof(int)*100);
    }
    meas->subband_cqi_tot[i] = (int *)malloc(sizeof(int)*100);
    meas->subband_cqi_tot_dB[i] = (int *)malloc(sizeof(int)*100);
  }
}
void phy_vars_eNB_malloc(PHY_VARS_eNB *eNB)
{
  int i, j;
  eNB->uci_vars = (LTE_eNB_UCI *)malloc(sizeof(LTE_eNB_UCI)*NUMBER_OF_UCI_VARS_MAX);
  eNB->srs_vars = (LTE_eNB_SRS *)malloc(sizeof(LTE_eNB_SRS)*NUMBER_OF_UE_MAX);
  eNB->pusch_vars = (LTE_eNB_PUSCH **)malloc(sizeof(LTE_eNB_PUSCH *)*NUMBER_OF_UE_MAX);
  eNB->dlsch = (LTE_eNB_DLSCH_t ***)malloc(sizeof(LTE_eNB_DLSCH_t **)*NUMBER_OF_UE_MAX);
  eNB->ulsch = (LTE_eNB_ULSCH_t **)malloc(sizeof(LTE_eNB_ULSCH_t *)*(NUMBER_OF_UE_MAX+1));
  eNB->UE_stats = (LTE_eNB_UE_stats *)malloc(sizeof(LTE_eNB_UE_stats)*NUMBER_OF_UE_MAX);
  eNB->UE_stats_ptr = (LTE_eNB_UE_stats **)malloc(sizeof(LTE_eNB_UE_stats *)*(NUMBER_OF_UE_MAX+1));
  for (i = 0; i < NUMBER_OF_UE_MAX; i++) {
    eNB->dlsch[i] = (LTE_eNB_DLSCH_t **)malloc(sizeof(LTE_eNB_DLSCH_t *)*2);
  }
  eNB->lte_gold_uespec_port5_table = (uint32_t ***)malloc(sizeof(uint32_t **)*NUMBER_OF_UE_MAX);
  for (i = 0; i < NUMBER_OF_UE_MAX; i++) {
    eNB->lte_gold_uespec_port5_table[i] = (uint32_t **)malloc(sizeof(uint32_t *)*20);
    for  (j = 0; j < 20 ; j++) {
      eNB->lte_gold_uespec_port5_table[i][j] = (uint32_t *)malloc(sizeof(uint32_t)*38);
    }
  }
  eNB->first_sr = (uint8_t *)malloc(sizeof(uint8_t)*NUMBER_OF_UE_MAX);
  eNB->first_run_timing_advance = (unsigned char *)malloc(sizeof(unsigned char)*NUMBER_OF_UE_MAX);
  eNB->pdsch_config_dedicated = (PDSCH_CONFIG_DEDICATED *)malloc(sizeof(PDSCH_CONFIG_DEDICATED)*NUMBER_OF_UE_MAX);
  eNB->pusch_config_dedicated = (PUSCH_CONFIG_DEDICATED *)malloc(sizeof(PUSCH_CONFIG_DEDICATED)*NUMBER_OF_UE_MAX);
  eNB->pucch_config_dedicated = (PUCCH_CONFIG_DEDICATED *)malloc(sizeof(PUCCH_CONFIG_DEDICATED)*NUMBER_OF_UE_MAX);
  eNB->ul_power_control_dedicated = (UL_POWER_CONTROL_DEDICATED *)malloc(sizeof(UL_POWER_CONTROL_DEDICATED)*NUMBER_OF_UE_MAX);
  eNB->tpc_pdcch_config_pucch = (TPC_PDCCH_CONFIG *)malloc(sizeof(TPC_PDCCH_CONFIG)*NUMBER_OF_UE_MAX);
  eNB->tpc_pdcch_config_pusch = (TPC_PDCCH_CONFIG *)malloc(sizeof(TPC_PDCCH_CONFIG)*NUMBER_OF_UE_MAX);
  eNB->cqi_report_config = (CQI_REPORT_CONFIG *)malloc(sizeof(CQI_REPORT_CONFIG)*NUMBER_OF_UE_MAX);
  eNB->soundingrs_ul_config_dedicated = (SOUNDINGRS_UL_CONFIG_DEDICATED *)malloc(sizeof(SOUNDINGRS_UL_CONFIG_DEDICATED)*NUMBER_OF_UE_MAX);
  eNB->scheduling_request_config = (SCHEDULING_REQUEST_CONFIG *)malloc(sizeof(SCHEDULING_REQUEST_CONFIG)*NUMBER_OF_UE_MAX);
  eNB->transmission_mode = (uint8_t *)malloc(sizeof(uint8_t)*NUMBER_OF_UE_MAX);
  eNB->physicalConfigDedicated = (struct LTE_PhysicalConfigDedicated **)malloc(sizeof(struct LTE_PhysicalConfigDedicated *)*NUMBER_OF_UE_MAX);
  eNB->mu_mimo_mode = (MU_MIMO_mode *)malloc(sizeof(MU_MIMO_mode)*NUMBER_OF_UE_MAX);
  eNB->pucch1_stats_cnt = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pucch1_stats = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pucch1_stats_thres = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pucch1ab_stats_cnt = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pucch1ab_stats = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pusch_stats_rb = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pusch_stats_round = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pusch_stats_mcs = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pusch_stats_bsr = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  eNB->pusch_stats_BO = (int32_t **)malloc(sizeof(int32_t *)*NUMBER_OF_UE_MAX);
  for (i = 0; i < NUMBER_OF_UE_MAX; i++) {
    eNB->pucch1_stats_cnt[i] = (int32_t *)malloc(sizeof(uint32_t)*10);
    eNB->pucch1_stats[i] = (int32_t *)malloc(sizeof(uint32_t)*10*1024);
    eNB->pucch1_stats_thres[i] = (int32_t *)malloc(sizeof(uint32_t)*10*1024);
    eNB->pucch1ab_stats_cnt[i] = (int32_t *)malloc(sizeof(uint32_t)*10);
    eNB->pucch1ab_stats[i] = (int32_t *)malloc(sizeof(uint32_t)*2*10*1024);
    eNB->pusch_stats_rb[i] = (int32_t *)malloc(sizeof(uint32_t)*10240);
    eNB->pusch_stats_round[i] = (int32_t *)malloc(sizeof(uint32_t)*10240);
    eNB->pusch_stats_mcs[i] = (int32_t *)malloc(sizeof(uint32_t)*10240);
    eNB->pusch_stats_bsr[i] = (int32_t *)malloc(sizeof(uint32_t)*10240);
    eNB->pusch_stats_BO[i] = (int32_t *)malloc(sizeof(uint32_t)*10240);
  }
  phy_measurements_eNB_malloc(&(eNB->measurements));
}
void lte_param_init(PHY_VARS_eNB **eNBp,
                    PHY_VARS_UE **UEp,
                    RU_t **rup,
                    unsigned char N_tx_port_eNB,
                    unsigned char N_tx_phy,
                    unsigned char N_rx_ru,
                    unsigned char N_rx_ue,
                    unsigned char transmission_mode,
                    uint8_t extended_prefix_flag,
                    frame_t frame_type,
                    uint16_t Nid_cell,
                    uint8_t tdd_config,
                    uint8_t N_RB_DL,
                    uint8_t pa,
                    uint8_t threequarter_fs,
                    uint8_t osf,
                    uint32_t perfect_ce) {
  LTE_DL_FRAME_PARMS *frame_parms;
  int i;
  PHY_VARS_eNB *eNB;
  PHY_VARS_UE  *UE;
  RU_t         *ru;
  printf("Start lte_param_init\n");
  // dlsim does not read the config file to get number_of_ue_max, so we set NUMBER_OF_UE_MAX to 16
  NUMBER_OF_UE_MAX = 16;
  NUMBER_OF_UCI_VARS_MAX = 4*10 + NUMBER_OF_UE_MAX;
  *eNBp = malloc(sizeof(PHY_VARS_eNB));
  *UEp = malloc(sizeof(PHY_VARS_UE));
  *rup = malloc(sizeof(RU_t));
  eNB = *eNBp;
  UE  = *UEp;
  ru  = *rup;
  printf("eNB %p, UE %p, ru %p\n",eNB,UE,ru);
  memset((void *)eNB,0,sizeof(PHY_VARS_eNB));
  memset((void *)UE,0,sizeof(PHY_VARS_UE));
  memset((void *)ru,0,sizeof(RU_t));
  phy_vars_eNB_malloc(eNB);
  ru->eNB_list[0] = eNB;
  eNB->RU_list[0] = ru;
  ru->num_eNB=1;
  srand(0);
  randominit(0);
  set_taus_seed(0);
  frame_parms = &(eNB->frame_parms);
  frame_parms->N_RB_DL            = N_RB_DL;   //50 for 10MHz and 25 for 5 MHz
  frame_parms->N_RB_UL            = N_RB_DL;
  frame_parms->threequarter_fs    = threequarter_fs;
  frame_parms->Ncp                = extended_prefix_flag;
  frame_parms->Ncp_UL             = extended_prefix_flag;
  frame_parms->Nid_cell           = Nid_cell;
  frame_parms->nushift            = Nid_cell%6;
  frame_parms->nb_antennas_tx     = N_tx_phy;
  frame_parms->nb_antennas_rx     = N_rx_ru;
  frame_parms->nb_antenna_ports_eNB = N_tx_port_eNB;
  frame_parms->phich_config_common.phich_resource         = oneSixth;
  frame_parms->phich_config_common.phich_duration         = normal;
  frame_parms->tdd_config         = tdd_config;
  frame_parms->frame_type         = frame_type;
  //  frame_parms->Csrs = 2;
  //  frame_parms->Bsrs = 0;
  //  frame_parms->kTC = 0;44
  //  frame_parms->n_RRC = 0;
  init_frame_parms(frame_parms,osf);
  //copy_lte_parms_to_phy_framing(frame_parms, &(PHY_config->PHY_framing));
  //  phy_init_top(frame_parms); //allocation
  UE->is_secondary_ue = 0;
  UE->frame_parms = *frame_parms;
  UE->frame_parms.nb_antennas_rx=N_rx_ue;
  //  eNB->frame_parms = *frame_parms;
  ru->frame_parms = *frame_parms;
  ru->nb_tx = N_tx_phy;
  ru->nb_rx = N_rx_ru;
  ru->if_south = LOCAL_RF;
  eNB->configured=1;
  eNB->transmission_mode[0] = transmission_mode;
  UE->transmission_mode[0] = transmission_mode;
  dump_frame_parms(frame_parms);
  UE->measurements.n_adj_cells=0;
  UE->measurements.adj_cell_id[0] = Nid_cell+1;
  UE->measurements.adj_cell_id[1] = Nid_cell+2;

  for (i=0; i<3; i++)
    lte_gold(frame_parms,UE->lte_gold_table[i],Nid_cell+i);

  printf("Calling init_lte_ue_signal\n");
  init_lte_ue_signal(UE,1,0);
  printf("Calling phy_init_lte_eNB\n");
  phy_init_lte_eNB(eNB,0,0);
  printf("Calling phy_init_RU (%p)\n",ru);
  phy_init_RU(ru);
  generate_pcfich_reg_mapping(&UE->frame_parms);
  generate_phich_reg_mapping(&UE->frame_parms);
  // DL power control init
  //if (transmission_mode == 1) {
  UE->pdsch_config_dedicated->p_a  = pa;

  if (transmission_mode == 1 || transmission_mode ==7) {
    ((eNB->frame_parms).pdsch_config_common).p_b = 0;
    ((UE->frame_parms).pdsch_config_common).p_b = 0;
  } else { // rho_a = rhob
    ((eNB->frame_parms).pdsch_config_common).p_b = 1;
    ((UE->frame_parms).pdsch_config_common).p_b = 1;
  }

  UE->perfect_ce = perfect_ce;

  /* the UE code is multi-thread "aware", we need to setup this array */
  for (i = 0; i < 10; i++) UE->current_thread_id[i] = i % 2;

  if (eNB->frame_parms.frame_type == TDD) {
    if      (eNB->frame_parms.N_RB_DL == 100) ru->N_TA_offset = 624;
    else if (eNB->frame_parms.N_RB_DL == 50)  ru->N_TA_offset = 624/2;
    else if (eNB->frame_parms.N_RB_DL == 25)  ru->N_TA_offset = 624/4;
  } else ru->N_TA_offset=0;

  if (IS_SOFTMODEM_BASICSIM)
    /* this is required for the basic simulator in TDD mode
     * TODO: find a proper cleaner solution
     */
    UE->N_TA_offset = 0;

  printf("Done lte_param_init\n");
}
