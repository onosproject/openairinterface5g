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

//#include "defs.h"
#include "SCHED/defs_NB_IoT.h"
//#include "PHY/extern.h"
#include "PHY/extern_NB_IoT.h"   	// PHY/defs_NB_IoT.h is called here , log.h & LTE_TRANSPORT/defs_NB_IoT.h are included through PHY/defs_NB_IoT.h
#include "openair2/LAYER2/MAC/proto_NB_IoT.h"	// for functions: from_earfcn_NB_IoT, get_uldl_offset_NB_IoT
//#include "SIMULATION/TOOLS/defs.h"
//#include "RadioResourceConfigCommonSIB.h"
//#include "RadioResourceConfigDedicated.h"
//#include "TDD-Config.h"
//#include "LAYER2/MAC/extern.h"
//#include "MBSFN-SubframeConfigList.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"
//#define DEBUG_PHY
#include "assertions.h"
//#include <math.h>

//NB-IoT
#include "PHY/INIT/defs_NB_IoT.h"   // nfapi_interface.h & IF_Module_NB_IoT.h are included here
//#include "RadioResourceConfigCommonSIB-NB-r13.h"
//#include "RadioResourceConfigDedicated-NB-r13.h"
//#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"
//#include "openair2/RRC/LITE/proto_NB_IoT.h"

//extern uint16_t prach_root_sequence_map0_3[838];
//extern uint16_t prach_root_sequence_map4[138];
//uint8_t dmrs1_tab[8] = {0,2,3,4,6,8,9,10};


void phy_config_mib_eNB_NB_IoT(int  			Mod_id,
							   int              CC_id,
							   int              eutra_band,
							   int              Nid_cell,
							   int              Ncp,
							   int				Ncp_UL,
							   int              p_eNB,
							   uint16_t			EARFCN,
							   uint16_t			prb_index, // NB_IoT_RB_ID,
							   uint16_t 		operating_mode,
							   uint16_t			control_region_size,
							   uint16_t			eutra_NumCRS_ports)
{


  AssertFatal(PHY_vars_eNB_NB_IoT_g != NULL, "PHY_vars_eNB_NB_IoT_g instance pointer doesn't exist\n");
  AssertFatal(PHY_vars_eNB_NB_IoT_g[Mod_id] != NULL, "PHY_vars_eNB_NB_IoT_g instance %d doesn't exist\n",Mod_id);
  AssertFatal(PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id] != NULL, "PHY_vars_eNB_NB_IoT_g instance %d, CCid %d doesn't exist\n",Mod_id,CC_id);

  NB_IoT_DL_FRAME_PARMS *fp = &PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms_NB_IoT;

  	fp = (NB_IoT_DL_FRAME_PARMS*) malloc (sizeof(NB_IoT_DL_FRAME_PARMS));

   LOG_I(PHY,"Configuring MIB-NB for instance %d, CCid %d : (band %d,Nid_cell %d,p %d,EARFCN %u)\n",Mod_id, CC_id, eutra_band, Nid_cell, p_eNB,EARFCN);
  

//  fp->N_RB_DL
//  fp->N_RB_UL						also this two values need to be known when we are dealing with in-band and guard-band operating mode
  fp->Nid_cell                           = Nid_cell;
  fp->nushift                            = Nid_cell%6;
  fp->eutra_band                         = eutra_band;
  fp->Ncp                             	 = Ncp;
  fp->Ncp_UL							 = Ncp_UL;
  fp->nb_antenna_ports_eNB               = p_eNB; //tx antenna port
  fp->dl_CarrierFreq                     = from_earfcn_NB_IoT(eutra_band,EARFCN,0);
  fp->ul_CarrierFreq                     = fp->dl_CarrierFreq - get_uldl_offset_NB_IoT(eutra_band);

  fp->operating_mode					 = operating_mode; //see how are defined by FAPI structure
  fp->NB_IoT_RB_ID						 = prb_index; //XXX to be better understand how should be managed
  //fp->nb_rx_antenna_ports_eNB
  fp->control_region_size			 	 = control_region_size; //(assume that this value is negative if not used)
  fp->eutra_NumCRS_ports				 = eutra_NumCRS_ports; //(valid only for in-band operating mode with different PCI)
  
  LOG_I(PHY,"Configure-MIB complete\n");


  
  //TODO  (new Raymond implementation) in the classic implementation seems to be used only by oaisim
  //init_frame_parms(fp,1);
  //init_lte_top(fp);

}

//Before FAPI implementation
//void NB_phy_config_sib2_eNB(uint8_t Mod_id,
//                         int CC_id,
//                         RadioResourceConfigCommonSIB_NB_r13_t *radioResourceConfigCommon
//                         )
//{
//	NB_IoT_DL_FRAME_PARMS *fp = &PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms;
//	  //LTE_eNB_UE_stats *eNB_UE_stats		= PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->eNB_UE_stats;
//	  //int32_t rx_total_gain_eNB_dB		= PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->rx_total_gain_eNB_dB;
//	  uint8_t MAX_NPRACH = 4;
//          NPRACH_Parameters_NB_r13_t *np;
//
//	  LOG_D(PHY,"[eNB%d] CCid %d: Applying radioResourceConfigCommon_NB\n",Mod_id,CC_id);
//
//      /*NPRACH configCommon*/
//	  fp->nprach_config_common.nprach_CP_Length                                     =radioResourceConfigCommon->nprach_Config_r13.nprach_CP_Length_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_CP_Length = %d\n",fp->nprach_config_common.nprach_CP_Length);
//	  //fp->nprach_config_common.rsrp_ThresholdsPrachInfoList.list                    =radioResourceConfigCommon->nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13.list;
//	  //LOG_D(PHY,"nprach_config_common.rsrp_ThresholdsPrachInfoList = %d\n",fp->nprach_config_common.rsrp_ThresholdsPrachInfoList);
//
//	  /*Loop over the configuration according to the maxNPRACH_Resources*/
//      for (fp->CE=1; fp->CE <= MAX_NPRACH;fp->CE++){
//      np = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[fp->CE];
//      /*fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->maxNumPreambleAttemptCE           =np->maxNumPreambleAttemptCE_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.maxNumPreambleAttemptCE = %d\n",fp->nprach_config_common.nprach_ParametersList.list.maxNumPreambleAttemptCE);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_NumRepetitions_RA          =np->npdcch_NumRepetitions_RA_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_NumRepetitions_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.npdcch_NumRepetitions_RA);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_Periodicity                =np->nprach_Periodicity_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_Periodicity = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_Periodicity);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_StartTime                  =np->nprach_StartTime_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_StartTime = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_StartTime);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_SubcarrierOffset           =np->nprach_SubcarrierOffset_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierOffset = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierOffset);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_SubcarrierMSG3_RangeStart  =np->nprach_SubcarrierMSG3_RangeStart_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierMSG3_RangeStart = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierMSG3_RangeStart);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_Offset_RA                  =np->npdcch_Offset_RA_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_Offset_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.npdcch_Offset_RA);
//      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_StartSF_CSS_RA             =np->npdcch_StartSF_CSS_RA_r13;
//	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_StartSF_CSS_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_StartSF_CSS_RA);
//      */
//      }
//
//	  /*Should modify to compute_nprach_seq*/
//	  //compute_prach_seq(&fp->prach_config_common,fp->frame_type,PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->X_u);
//
//	  /*NPDSCH ConfigCommon*/
//	  fp->npdsch_config_common.nrs_Power           = radioResourceConfigCommon->npdsch_ConfigCommon_r13.nrs_Power_r13;
//
//	  /*NPUSCH ConfigCommon*/
//	  /*A list (1-3) should be loop for ack_NACK_NumRepetitions_Msg4*/
//	  for (fp->CE=1; fp->CE <= MAX_NPRACH;fp->CE++){
//	  fp->npusch_config_common.ack_NACK_NumRepetitions_Msg4[fp->CE]        = radioResourceConfigCommon->npusch_ConfigCommon_r13.ack_NACK_NumRepetitions_Msg4_r13.list.array[fp->CE];
//          //LOG_D(PHY,"npusch_config_common.ack_NACK_NumRepetitions_Msg4 = %d]n",fp->npusch_config_common.ack_NACK_NumRepetitions_Msg4);
//	  }
//	  fp->npusch_config_common.srs_SubframeConfig                          = radioResourceConfigCommon->npusch_ConfigCommon_r13.srs_SubframeConfig_r13;
//	  LOG_D(PHY,"npusch_config_common.srs_SubframeConfig = %d]n",fp->npusch_config_common.srs_SubframeConfig);
//	  fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence          = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_BaseSequence_r13;
//	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence);
//	  fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence            = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_BaseSequence_r13;
//	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence);
//	  fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift           = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13;
//	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_CyclicShift = %d]n",fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift);
//	  fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift             = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13;
//	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_CyclicShift = %d]n",fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift);
//	  fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence         = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->twelveTone_BaseSequence_r13;
//	  LOG_D(PHY,"npusch_config_common.dmrs_Config.twelveTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence);
//
//	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH  = radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13;
//	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH = %d]n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH);
//	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled    = radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13;
//	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled = %d]n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled);
//
//      /*should change the part that implement the ul hopping in NB-IoT*/
//	  //init_ul_hopping(fp);
//
//	  /*UL Power Control Config Common*/
//
//	  fp->ul_power_control_config_common.p0_NominalNPUSCH      = radioResourceConfigCommon->uplinkPowerControlCommon_r13.p0_NominalNPUSCH_r13;
//	  fp->ul_power_control_config_common.alpha				   = radioResourceConfigCommon->uplinkPowerControlCommon_r13.alpha_r13;
//      fp->ul_power_control_config_common.deltaPreambleMsg3     = radioResourceConfigCommon->uplinkPowerControlCommon_r13.deltaPreambleMsg3_r13;
//
//	  /*DL gap*/
//
//      fp->DL_gap_config.dl_GapDurationCoeff                        = radioResourceConfigCommon->dl_Gap_r13->dl_GapDurationCoeff_r13;
//	  fp->DL_gap_config.dl_GapPeriodicity                      = radioResourceConfigCommon->dl_Gap_r13->dl_GapPeriodicity_r13;
//	  fp->DL_gap_config.dl_GapThreshold                        = radioResourceConfigCommon->dl_Gap_r13->dl_GapThreshold_r13;
//
//	  /*PUCCH stuff in LTE*/
//	  //init_ncs_cell(fp,PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->ncs_cell);
//
//	  //init_ul_hopping(fp);
//
//
//
//}

void phy_config_sib2_eNB_NB_IoT(uint8_t 								  Mod_id,
                         		int 									  CC_id,
                         		nfapi_config_NB_IoT_t 					  *config,
						 		nfapi_rf_config_t 						  *rf_config,
						 		nfapi_uplink_reference_signal_config_t	  *ul_nrs_config,
						 		extra_phyConfig_t						  *extra_phy_parms)
{

  NB_IoT_DL_FRAME_PARMS *fp = &PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms_NB_IoT;

	LOG_I(PHY,"[eNB%d] CCid %d: Applying config_NB_IoT from sib2_NB\n",Mod_id,CC_id);

	  	fp = (NB_IoT_DL_FRAME_PARMS*) malloc (sizeof(NB_IoT_DL_FRAME_PARMS));

	  
	/*NPRACH_ConfigSIB_NB_r13----------------------------------------------------------*/

	//MP: FAPI style approach: instead of a list they consider the 3 possible configuration separately

	  if(config->nprach_config_0_enabled.value == 1){
		  LOG_I(PHY, "NPRACH Config #0 enabled\n");

		  fp->nprach_config_common.nprach_CP_Length = config->nprach_config_0_cp_length.value; //NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us66dot7
		  LOG_D(PHY," config#0: nprach_CP_Length = %d\n",fp->nprach_config_common.nprach_CP_Length);
		  //FIXME: MP: memory for the list should be allocated? initialization??
		  fp->nprach_config_common.nprach_ParametersList.list[0].nprach_Periodicity = config->nprach_config_0_sf_periodicity.value;
		  LOG_D(PHY,"config#0: nprach_Periodicity = %d\n", fp->nprach_config_common.nprach_ParametersList.list[0].nprach_Periodicity);
		  fp->nprach_config_common.nprach_ParametersList.list[0].nprach_StartTime = config->nprach_config_0_start_time.value;
		  LOG_D(PHY,"config#0: nprach_StartTime = %d\n",fp->nprach_config_common.nprach_ParametersList.list[0].nprach_StartTime);
		  fp->nprach_config_common.nprach_ParametersList.list[0].nprach_SubcarrierOffset = config->nprach_config_0_subcarrier_offset.value;
		  LOG_D(PHY,"config#0: nprach_SubcarrierOffset= %d\n", fp->nprach_config_common.nprach_ParametersList.list[0].nprach_SubcarrierOffset);
		  fp->nprach_config_common.nprach_ParametersList.list[0].nprach_NumSubcarriers = config->nprach_config_0_number_of_subcarriers.value;
		  LOG_D(PHY,"config#0: nprach_NumSubcarriers= %d\n",fp->nprach_config_common.nprach_ParametersList.list[0].nprach_NumSubcarriers);
		  fp->nprach_config_common.nprach_ParametersList.list[0].numRepetitionsPerPreambleAttempt = config->nprach_config_0_number_of_repetitions_per_attempt.value;
		  LOG_D(PHY,"config#0: numRepetitionsPerPreambleAttempt= %d\n",fp->nprach_config_common.nprach_ParametersList.list[0].numRepetitionsPerPreambleAttempt);

		  //missed configuration in FAPI config_request (TS 36.331 pag 610) (may not needed)
		  /*fp->nprach_config_common.nprach_ParametersList.list.array[0]->nprach_SubcarrierMSG3_RangeStart = extra_phy_parms->nprach_config_0_subcarrier_MSG3_range_start;
		  fp->nprach_config_common.nprach_ParametersList.list.array[0]->npdcch_StartSF_CSS_RA = extra_phy_parms->nprach_config_0_npdcch_startSF_CSS_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[0]->npdcch_NumRepetitions_RA = extra_phy_parms->nprach_config_0_npdcch_num_repetitions_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[0]->npdcch_Offset_RA = extra_phy_parms->nprach_config_0_npdcch_offset_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[0]->maxNumPreambleAttemptCE = extra_phy_parms->nprach_config_0_max_num_preamble_attempt_CE;
		  */
		  //fp->nprach_config_common.rsrp_ThresholdsPrachInfoList.list /*OPTIONAL*/

	  }

	  if(config->nprach_config_1_enabled.value == 1){
		  LOG_I(PHY, "NPRACH Config #1 enabled\n");

		  fp->nprach_config_common.nprach_CP_Length = config->nprach_config_1_cp_length.value; //NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us66dot7
		  LOG_D(PHY," config#1: nprach_CP_Length = %d\n",fp->nprach_config_common.nprach_CP_Length);

		  //FIXME: MP: memory for the list should be allocated? initialization??
		  fp->nprach_config_common.nprach_ParametersList.list[1].nprach_Periodicity = config->nprach_config_1_sf_periodicity.value;
		  LOG_D(PHY,"config#1: nprach_Periodicity = %d\n", fp->nprach_config_common.nprach_ParametersList.list[1].nprach_Periodicity);
		  fp->nprach_config_common.nprach_ParametersList.list[1].nprach_StartTime = config->nprach_config_1_start_time.value;
		  LOG_D(PHY,"config#1: nprach_StartTime = %d\n",fp->nprach_config_common.nprach_ParametersList.list[1].nprach_StartTime);
		  fp->nprach_config_common.nprach_ParametersList.list[1].nprach_SubcarrierOffset = config->nprach_config_1_subcarrier_offset.value;
		  LOG_D(PHY,"config#1: nprach_SubcarrierOffset= %d\n", fp->nprach_config_common.nprach_ParametersList.list[1].nprach_SubcarrierOffset);
		  fp->nprach_config_common.nprach_ParametersList.list[1].nprach_NumSubcarriers = config->nprach_config_1_number_of_subcarriers.value;
		  LOG_D(PHY,"config#1: nprach_NumSubcarriers= %d\n",fp->nprach_config_common.nprach_ParametersList.list[1].nprach_NumSubcarriers);
		  fp->nprach_config_common.nprach_ParametersList.list[1].numRepetitionsPerPreambleAttempt = config->nprach_config_1_number_of_repetitions_per_attempt.value;
		  LOG_D(PHY,"config#1: numRepetitionsPerPreambleAttempt= %d\n",fp->nprach_config_common.nprach_ParametersList.list[1].numRepetitionsPerPreambleAttempt);

		  //missed configuration in FAPI config_request (TS 36.331 pag 610) (may not needed)
		  /*fp->nprach_config_common.nprach_ParametersList.list.array[1]->nprach_SubcarrierMSG3_RangeStart = extra_phy_parms->nprach_config_1_subcarrier_MSG3_range_start;
		  fp->nprach_config_common.nprach_ParametersList.list.array[1]->npdcch_StartSF_CSS_RA = extra_phy_parms->nprach_config_1_npdcch_startSF_CSS_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[1]->npdcch_NumRepetitions_RA = extra_phy_parms->nprach_config_1_npdcch_num_repetitions_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[1]->npdcch_Offset_RA = extra_phy_parms->nprach_config_1_npdcch_offset_RA;
		  fp->nprach_config_common.nprach_ParametersList.list.array[1]->maxNumPreambleAttemptCE = extra_phy_parms->nprach_config_1_max_num_preamble_attempt_CE;
		  */
		  //fp->nprach_config_common.rsrp_ThresholdsPrachInfoList.list /*OPTIONAL*/

	  }

	  if(config->nprach_config_2_enabled.value == 1){
		  LOG_I(PHY, "NPRACH Config #2 enabled\n");

		  fp->nprach_config_common.nprach_CP_Length = config->nprach_config_2_cp_length.value; //NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us66dot7
		  LOG_D(PHY," config#2: nprach_CP_Length = %d\n",fp->nprach_config_common.nprach_CP_Length);
		  //FIXME: MP: memory for the list should be allocated? initialization?? where??
		  fp->nprach_config_common.nprach_ParametersList.list[2].nprach_Periodicity = config->nprach_config_2_sf_periodicity.value;
		  LOG_D(PHY,"config#2: nprach_Periodicity = %d\n", fp->nprach_config_common.nprach_ParametersList.list[2].nprach_Periodicity);
		  fp->nprach_config_common.nprach_ParametersList.list[2].nprach_StartTime = config->nprach_config_2_start_time.value;
		  LOG_D(PHY,"config#2: nprach_StartTime = %d\n",fp->nprach_config_common.nprach_ParametersList.list[2].nprach_StartTime);
		  fp->nprach_config_common.nprach_ParametersList.list[2].nprach_SubcarrierOffset = config->nprach_config_2_subcarrier_offset.value;
		  LOG_D(PHY,"config#2: nprach_SubcarrierOffset= %d\n", fp->nprach_config_common.nprach_ParametersList.list[2].nprach_SubcarrierOffset);
		  fp->nprach_config_common.nprach_ParametersList.list[2].nprach_NumSubcarriers = config->nprach_config_2_number_of_subcarriers.value;
		  LOG_D(PHY,"config#2: nprach_NumSubcarriers= %d\n",fp->nprach_config_common.nprach_ParametersList.list[2].nprach_NumSubcarriers);
		  fp->nprach_config_common.nprach_ParametersList.list[2].numRepetitionsPerPreambleAttempt = config->nprach_config_2_number_of_repetitions_per_attempt.value;
		  LOG_D(PHY,"config#2: numRepetitionsPerPreambleAttempt= %d\n",fp->nprach_config_common.nprach_ParametersList.list[2].numRepetitionsPerPreambleAttempt);

		  //missed configuration in FAPI config_request (TS 36.331 pag 610) (may not needed)
		  /*fp->nprach_config_common.nprach_ParametersList.list.array[2]->nprach_SubcarrierMSG3_RangeStart = extra_phy_parms->nprach_config_2_subcarrier_MSG3_range_start;
		  LOG_D(PHY,"config#2: nprach_SubcarrierMSG3_RangeStart= %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[2]->nprach_SubcarrierMSG3_RangeStart);
		  fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_StartSF_CSS_RA = extra_phy_parms->nprach_config_2_npdcch_startSF_CSS_RA;
		  LOG_D(PHY,"config#2: npdcch_StartSF_CSS_RA= %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_StartSF_CSS_RA);
		  fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_NumRepetitions_RA = extra_phy_parms->nprach_config_2_npdcch_num_repetitions_RA;
		  LOG_D(PHY,"config#2: npdcch_NumRepetitions_RA= %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_NumRepetitions_RA);
		  fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_Offset_RA = extra_phy_parms->nprach_config_2_npdcch_offset_RA;
		  LOG_D(PHY,"config#2: npdcch_Offset_RA= %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[2]->npdcch_Offset_RA);
		  fp->nprach_config_common.nprach_ParametersList.list.array[2]->maxNumPreambleAttemptCE = extra_phy_parms->nprach_config_2_max_num_preamble_attempt_CE;
		  LOG_D(PHY,"config#2: maxNumPreambleAttemptCE= %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[2]->maxNumPreambleAttemptCE);
		  */
		  //fp->nprach_config_common.rsrp_ThresholdsPrachInfoList.list /*OPTIONAL*/

	  }

	  //TODO: Should modify to compute_nprach_seq --> nprach.
	  //compute_prach_seq(&fp->prach_config_common,fp->frame_type,PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->X_u);
	  

	  /*NPDSCH ConfigCommon-------------------------------------------------------------------*/
	  //NPDSCH_ConfigCommon_NB_r13_t b;

	  //FIXME: the FAPI specs pag 140 fix a range of value (0->255) but i don't find any similar correspondence in the 3GPP specs (TS 36.331 pag 608 and TS 36.213 ch 16.2.2)
	  fp->npdsch_config_common.nrs_Power = rf_config->reference_signal_power.value;
	
	  /*NPUSCH ConfigCommon-------------------------------------------------------------------*/
	  //NPUSCH_ConfigCommon_NB_r13_t c;

	  fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence = config->three_tone_base_sequence.value;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_BaseSequence = %d\n",fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence);
	  fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence = config->six_tone_base_sequence.value;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_BaseSequence = %d\n",fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence);
	  fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift = config->three_tone_cyclic_shift.value;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_CyclicShift = %d\n",fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift);
	  fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift = config->six_tone_cyclic_shift.value;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_CyclicShift = %d\n",fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift);
	  fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence= config->twelve_tone_base_sequence.value;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.twelveTone_BaseSequence = %d\n",fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence);


	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled = ul_nrs_config->uplink_rs_hopping.value;
	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled = %d\n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled);
	  LOG_D(PHY,"**%s**\n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled == 1 ? "RS_GROUP_HOPPING" : "RS_NO_HOPPING");
	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH = ul_nrs_config->group_assignment.value;
	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH = %d]n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH);



	  //MP: FAPI missed parameters (may not needed at eNB side and some of them are optional by the 3GPP specs)
	  //fp->npusch_config_common.ack_NACK_NumRepetitions_Msg4 --> list of size maxNPRACH_Resources_NB_r13 (3 elements)
	  //fp->npusch_config_common.srs_SubframeConfig /*OPTIONAL*/


	  //No Frequency hopping in NULSCH for NB-IoT and not init_ncs_cell used for PUCCH


	  /*UL Power Control Config Common---------------------------------------------------------*/
	  //F nothing has been defined in FAPI specs for this (may because are only UE stuffs)
	  /*fp->ul_power_control_config_common.p0_NominalNPUSCH = extra_phy_parms->p0_nominal_npusch;
	  fp->ul_power_control_config_common.alpha = extra_phy_parms->alpha;
	  fp->ul_power_control_config_common.deltaPreambleMsg3 = extra_phy_parms->delta_preamle_MSG*/

	  /*DL gap Config - OPTIONAL----------------------------------------------------------------*/
	  //DL_GapConfig_NB_r13_t a;
	  if(config->dl_gap_config_enable.value == 1){
		  fp->DL_gap_config.dl_GapDurationCoeff= config->dl_gap_periodicity.value;
		  fp->DL_gap_config.dl_GapPeriodicity = config->dl_gap_periodicity.value;
		  fp->DL_gap_config.dl_GapThreshold = config->dl_gap_threshold.value;
	  }

	  LOG_I(PHY,"SIB-2 configure complete\n");
}



void phy_config_dedicated_eNB_NB_IoT(uint8_t 			Mod_id,
                              		 int 				CC_id,
                             		 uint16_t 			rnti,
							 		 extra_phyConfig_t  *extra_parms)
{
	PHY_VARS_eNB_NB_IoT *eNB = PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id];
	NB_IoT_eNB_NPDCCH_t *npdcch;
	uint8_t UE_id = find_ue_NB_IoT(rnti,eNB);
	
	if (UE_id == -1) {

		LOG_E( PHY, "[eNB %"PRIu8"] find_ue() returns -1\n", Mod_id);
		return;
	}
	
	//configure UE specific parameters for NPDCCH Search Space

	if (eNB->npdcch[UE_id]) {
		npdcch = eNB->npdcch[UE_id];
		npdcch->rnti = rnti;
		npdcch->npdcch_NumRepetitions = extra_parms->npdcch_NumRepetitions; //Rmax maybe is the only one needed
		//npdcch->npdcch_Offset_USS = extra_parms->npdcch_Offset_USS;
		//npdcch->npdcch_StartSF_USS = extra_parms->npdcch_StartSF_USS;

		LOG_I(PHY,"phy_config_dedicated_eNB_NB_IoT: npdcch_NumRepetitions = %d\n",npdcch->npdcch_NumRepetitions);
	
	} else {
		LOG_E(PHY,"[eNB %d] Received NULL radioResourceConfigDedicated from eNB %d\n",Mod_id, UE_id);
		return;
	}
	
}

// void phy_init_lte_top_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms)
// {

// //   crcTableInit();

// //   ccodedot11_init();
// //   ccodedot11_init_inv();

// //   ccodelte_init();
// //   ccodelte_init_inv();

// //   treillis_table_init();

// //   phy_generate_viterbi_tables();
// //   phy_generate_viterbi_tables_lte();

// //   init_td8();
// //   init_td16();
// // #ifdef __AVX2__
// //   init_td16avx2();
// // #endif

// //   lte_sync_time_init(frame_parms);

// //   generate_ul_ref_sigs();
//   generate_ul_ref_sigs_rx_NB_IoT();

//   // generate_64qam_table();
//   // generate_16qam_table();
//   // generate_RIV_tables();

//   // init_unscrambling_lut();
//   // init_scrambling_lut();
//   // //set_taus_seed(1328);

// }


