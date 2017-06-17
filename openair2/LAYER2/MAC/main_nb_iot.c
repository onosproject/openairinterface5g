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

/*! \file main.c
 * \brief top init of Layer 2
 * \author  Navid Nikaein and Raymond Knopp, Michele Paffetti
 * \date 2010 - 2014
 * \version 1.0
 * \email: navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
 * @ingroup _mac

 */


#include "asn1_constants.h"


int mac_init_global_param_NB(void)
{

//XXX commented parts are called in the parallel path of OAI
//  Mac_rlc_xface = NULL;
//  LOG_I(MAC,"[MAIN] CALLING RLC_MODULE_INIT...\n");
//
//  if (rlc_module_init()!=0) {
//    return(-1);
//  }
//
//  LOG_I(MAC,"[MAIN] RLC_MODULE_INIT OK, malloc16 for mac_rlc_xface...\n");
//
//  Mac_rlc_xface = (MAC_RLC_XFACE*)malloc16(sizeof(MAC_RLC_XFACE));
//  bzero(Mac_rlc_xface,sizeof(MAC_RLC_XFACE));
//
//  if(Mac_rlc_xface == NULL) {
//    LOG_E(MAC,"[MAIN] FATAL EROOR: Could not allocate memory for Mac_rlc_xface !!!\n");
//    return (-1);
//
//  }
//
//  LOG_I(MAC,"[MAIN] malloc16 OK, mac_rlc_xface @ %p\n",(void *)Mac_rlc_xface);
//
//  mac_xface->mrbch_phy_sync_failure=mrbch_phy_sync_failure;
//  mac_xface->dl_phy_sync_success=dl_phy_sync_success;
//  mac_xface->out_of_sync_ind=rrc_out_of_sync_ind;
//
//  LOG_I(MAC,"[MAIN] RLC interface (mac_rlc_xface) setup and init (maybe no mre used??)\n");

  LOG_I(MAC,"[MAIN] RRC NB-IoT initialization of global params\n");
  rrc_init_global_param_NB();


//  LOG_I(MAC,"[MAIN] PDCP layer init\n");
//#ifdef USER_MODE
//  pdcp_layer_init ();
//#else
//  pdcp_module_init ();
//#endif
//
//  LOG_I(MAC,"[MAIN] Init Global Param Done\n");

  return 0;
}


int l2_init_eNB_NB()
{


  LOG_I(MAC,"Mapping L2 IF-Module functions\n");
  IF_Module_init_L2();

  LOG_I(MAC,"[MAIN] MAC_INIT_GLOBAL_PARAM NB-IoT IN...\n");

  Is_rrc_nb_iot_registered=0;
  NB_mac_init_global_param();
  Is_rrc_nb_iot_registered=1;

//XXX called in the parallel path
//    mac_xface->macphy_init = mac_top_init;
//  #ifndef USER_MODE
//    mac_xface->macphy_exit = openair_sched_exit;
//  #else
//    mac_xface->macphy_exit=(void (*)(const char*)) exit;
//  #endif
//    LOG_I(MAC,"[MAIN] init eNB MAC functions  \n");
//    mac_xface->eNB_dlsch_ulsch_scheduler = eNB_dlsch_ulsch_scheduler;
//    mac_xface->get_dci_sdu               = get_dci_sdu;
//    mac_xface->fill_rar                  = fill_rar;
//    mac_xface->initiate_ra_proc          = initiate_ra_proc;
//    mac_xface->cancel_ra_proc            = cancel_ra_proc;
//    mac_xface->set_msg3_subframe         = set_msg3_subframe;
//    mac_xface->SR_indication             = SR_indication;
//    mac_xface->UL_failure_indication     = UL_failure_indication;
//    mac_xface->rx_sdu                    = rx_sdu;
//    mac_xface->get_dlsch_sdu             = get_dlsch_sdu;
//    mac_xface->get_eNB_UE_stats          = get_UE_stats;
//    mac_xface->get_transmission_mode     = get_transmission_mode;
//    mac_xface->get_rballoc               = get_rballoc;
//    mac_xface->get_nb_rb                 = conv_nprb;
//    mac_xface->get_prb                   = get_prb;
//    //  mac_xface->get_SB_size               = Get_SB_size;
//    mac_xface->get_subframe_direction    = get_subframe_direction;
//    mac_xface->Msg3_transmitted          = Msg3_tx;
//    mac_xface->Msg1_transmitted          = Msg1_tx;
//    mac_xface->ra_failed                 = ra_failed;
//    mac_xface->ra_succeeded              = ra_succeeded;
//    mac_xface->mac_phy_remove_ue         = mac_phy_remove_ue;
//
//    LOG_I(MAC,"[MAIN] init UE MAC functions \n");
//    mac_xface->ue_decode_si              = ue_decode_si;
//    mac_xface->ue_decode_p               = ue_decode_p;
//    mac_xface->ue_send_sdu               = ue_send_sdu;
//  #if defined(Rel10) || defined(Rel14)
//    mac_xface->ue_send_mch_sdu           = ue_send_mch_sdu;
//    mac_xface->ue_query_mch              = ue_query_mch;
//  #endif
//    mac_xface->ue_get_SR                 = ue_get_SR;
//    mac_xface->ue_get_sdu                = ue_get_sdu;
//    mac_xface->ue_get_rach               = ue_get_rach;
//    mac_xface->ue_process_rar            = ue_process_rar;
//    mac_xface->ue_scheduler              = ue_scheduler;
//    mac_xface->process_timing_advance    = process_timing_advance;
//
//
//    LOG_I(MAC,"[MAIN] PHY Frame configuration \n");
//    mac_xface->frame_parms = frame_parms;
//
//    mac_xface->get_ue_active_harq_pid = get_ue_active_harq_pid;
//    mac_xface->get_PL                 = get_PL;
//    mac_xface->get_RSRP               = get_RSRP;
//    mac_xface->get_RSRQ               = get_RSRQ;
//    mac_xface->get_RSSI               = get_RSSI;
//    mac_xface->get_n_adj_cells        = get_n_adj_cells;
//    mac_xface->get_rx_total_gain_dB   = get_rx_total_gain_dB;
//    mac_xface->get_Po_NOMINAL_PUSCH   = get_Po_NOMINAL_PUSCH;
//    mac_xface->get_num_prach_tdd      = get_num_prach_tdd;
//    mac_xface->get_fid_prach_tdd      = get_fid_prach_tdd;
//    mac_xface->get_deltaP_rampup      = get_deltaP_rampup;
//    mac_xface->computeRIV             = computeRIV;
//    mac_xface->get_TBS_DL             = get_TBS_DL;
//    mac_xface->get_TBS_UL             = get_TBS_UL;
//    mac_xface->get_nCCE_max           = get_nCCE_mac;
//    mac_xface->get_nCCE_offset        = get_nCCE_offset;
//    mac_xface->get_ue_mode            = get_ue_mode;
//    mac_xface->phy_config_sib1_eNB    = phy_config_sib1_eNB;
//    mac_xface->phy_config_sib1_ue     = phy_config_sib1_ue;
//
//    mac_xface->phy_config_sib2_eNB        = phy_config_sib2_eNB;
//    mac_xface->phy_config_sib2_ue         = phy_config_sib2_ue;
//    mac_xface->phy_config_afterHO_ue      = phy_config_afterHO_ue;
//  #if defined(Rel10) || defined(Rel14)
//    mac_xface->phy_config_sib13_eNB        = phy_config_sib13_eNB;
//    mac_xface->phy_config_sib13_ue         = phy_config_sib13_ue;
//  #endif
//  #ifdef CBA
//    mac_xface->phy_config_cba_rnti         = phy_config_cba_rnti ;
//  #endif
//    mac_xface->estimate_ue_tx_power        = estimate_ue_tx_power;
//    mac_xface->phy_config_meas_ue          = phy_config_meas_ue;
//    mac_xface->phy_reset_ue    = phy_reset_ue;
//
//    mac_xface->phy_config_dedicated_eNB    = phy_config_dedicated_eNB;
//    mac_xface->phy_config_dedicated_ue     = phy_config_dedicated_ue;
//    mac_xface->phy_config_harq_ue          = phy_config_harq_ue;
//
//    mac_xface->get_lte_frame_parms        = get_lte_frame_parms;
//    mac_xface->get_mu_mimo_mode           = get_mu_mimo_mode;
//
//    mac_xface->get_hundred_times_delta_TF = get_hundred_times_delta_IF_mac;
//    mac_xface->get_target_pusch_rx_power     = get_target_pusch_rx_power;
//    mac_xface->get_target_pucch_rx_power     = get_target_pucch_rx_power;
//
//    mac_xface->get_prach_prb_offset       = get_prach_prb_offset;
//    mac_xface->is_prach_subframe          = is_prach_subframe;
//
//  #if defined(Rel10) || defined(Rel14)
//    mac_xface->get_mch_sdu                 = get_mch_sdu;
//    mac_xface->phy_config_dedicated_scell_eNB= phy_config_dedicated_scell_eNB;
//    mac_xface->phy_config_dedicated_scell_ue= phy_config_dedicated_scell_ue;
//
//  #endif
//
//    mac_xface->get_PHR = get_PHR;
//    LOG_D(MAC,"[MAIN] ALL INIT OK\n");
//
//    mac_xface->macphy_init(eMBMS_active,uecap_xer,cba_group_active,HO_active);


//XXX call mac_top_init_NB!!!

  return(1);
}

