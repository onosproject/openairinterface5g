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

#include "defs.h"
#include "SCHED/defs.h"
#include "PHY/extern.h"
#include "SIMULATION/TOOLS/defs.h"
#include "RadioResourceConfigCommonSIB.h"
#include "RadioResourceConfigDedicated.h"
#include "TDD-Config.h"
#include "LAYER2/MAC/extern.h"
#include "MBSFN-SubframeConfigList.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#define DEBUG_PHY
#include "assertions.h"
#include <math.h>

//NB-IoT
#include "defs_nb_iot.h"
#include "RadioResourceConfigCommonSIB-NB-r13.h"
#include "PHY/impl_defs_lte_nb_iot.h"
#include "RadioResourceConfigDedicated-NB-r13.h"

extern uint16_t prach_root_sequence_map0_3[838];
extern uint16_t prach_root_sequence_map4[138];
uint8_t dmrs1_tab[8] = {0,2,3,4,6,8,9,10};


void NB_phy_config_mib_eNB(int                 Mod_id,
			int                 CC_id,
			int                 eutra_band,
			int                 Nid_cell,
			int                 Ncp,
			int                 p_eNB,
			uint32_t            dl_CarrierFreq,
			uint32_t            ul_CarrierFreq) {

  /*Not sure if phy parameters should be initial here or not*/
  /*the phy_config_mib_eNB as the entry point to allocate the context for L1.  The RC contains the context for L1,L2. If RC.eNB is NULL, it hasn't been allocated earlier so we allocate it there.*/
  /*if (RC.eNB == NULL) {
    RC.eNB                               = (PHY_VARS_eNB ***)malloc((1+NUMBER_OF_eNB_MAX)*sizeof(PHY_VARS_eNB***));
    LOG_I(PHY,"RC.eNB = %p\n",RC.eNB);
    memset(RC.eNB,0,(1+NUMBER_OF_eNB_MAX)*sizeof(PHY_VARS_eNB***));
  }
  if (RC.eNB[Mod_id] == NULL) {
    RC.eNB[Mod_id]                       = (PHY_VARS_eNB **)malloc((1+MAX_NUM_CCs)*sizeof(PHY_VARS_eNB**));
    LOG_I(PHY,"RC.eNB[%d] = %p\n",Mod_id,RC.eNB[Mod_id]);
    memset(RC.eNB[Mod_id],0,(1+MAX_NUM_CCs)*sizeof(PHY_VARS_eNB***));
  }
  if (RC.eNB[Mod_id][CC_id] == NULL) {
    RC.eNB[Mod_id][CC_id] = (PHY_VARS_eNB *)malloc(sizeof(PHY_VARS_eNB));
    LOG_I(PHY,"RC.eNB[%d][%d] = %p\n",Mod_id,CC_id,RC.eNB[Mod_id][CC_id]);
    RC.eNB[Mod_id][CC_id]->Mod_id        = Mod_id;
    RC.eNB[Mod_id][CC_id]->CC_id         = CC_id;
  }

  RC.eNB[Mod_id][CC_id]->mac_enabled     = 1;

  fp = &RC.eNB[Mod_id][CC_id]->frame_parms; */

  NB_DL_FRAME_PARMS *fp = &PHY_vars_eNB_g[Mod_id][CC_id]->frame_parms;

   //LOG_I(PHY,"Configuring MIB-NB for instance %d, CCid %d : (band %d,N_RB_DL %d,Nid_cell %d,p %d,DL freq %u)\n",
	//Mod_id, CC_id, eutra_band, N_RB_DL_array[dl_Bandwidth], Nid_cell, p_eNB,dl_CarrierFreq);

  fp->Nid_cell                           = Nid_cell;
  fp->nushift                            = Nid_cell%6;
  fp->eutra_band                         = eutra_band;
  fp->Ncp                                = Ncp;
  fp->nb_antenna_ports_eNB               = p_eNB;
  fp->dl_CarrierFreq                     = dl_CarrierFreq;
  fp->ul_CarrierFreq                     = ul_CarrierFreq;
  

  //init_frame_parms(fp,1);
  //init_lte_top(fp);

}

void NB_phy_config_sib2_eNB(uint8_t Mod_id,
                         int CC_id,
                         RadioResourceConfigCommonSIB_NB_r13_t *radioResourceConfigCommon,
                         ARFCN_ValueEUTRA_r9_t *ul_CArrierFreq
                         )
{
	NB_DL_FRAME_PARMS *fp = &PHY_vars_eNB_g[Mod_id][CC_id]->frame_parms;
	  //LTE_eNB_UE_stats *eNB_UE_stats		= PHY_vars_eNB_g[Mod_id][CC_id]->eNB_UE_stats;
	  //int32_t rx_total_gain_eNB_dB		= PHY_vars_eNB_g[Mod_id][CC_id]->rx_total_gain_eNB_dB;
	  uint8_t MAX_NPRACH = 4;
          NPRACH_Parameters_NB_r13_t *np;
	  
	  LOG_D(PHY,"[eNB%d] CCid %d: Applying radioResourceConfigCommon_NB\n",Mod_id,CC_id);
	  
      /*NPRACH configCommon*/
	  fp->nprach_config_common.nprach_CP_Length                                     =radioResourceConfigCommon->nprach_Config_r13.nprach_CP_Length_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_CP_Length = %d\n",fp->nprach_config_common.nprach_CP_Length);
	  //fp->nprach_config_common.rsrp_ThresholdsPrachInfoList.list                    =radioResourceConfigCommon->nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13.list;
	  //LOG_D(PHY,"nprach_config_common.rsrp_ThresholdsPrachInfoList = %d\n",fp->nprach_config_common.rsrp_ThresholdsPrachInfoList);

	  /*Loop over the configuration according to the maxNPRACH_Resources*/
      for (fp->CE=1; fp->CE <= MAX_NPRACH;fp->CE++){
      np = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[fp->CE];	  	
      /*fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->maxNumPreambleAttemptCE           =np->maxNumPreambleAttemptCE_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.maxNumPreambleAttemptCE = %d\n",fp->nprach_config_common.nprach_ParametersList.list.maxNumPreambleAttemptCE);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_NumRepetitions_RA          =np->npdcch_NumRepetitions_RA_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_NumRepetitions_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.npdcch_NumRepetitions_RA);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_Periodicity                =np->nprach_Periodicity_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_Periodicity = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_Periodicity);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_StartTime                  =np->nprach_StartTime_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_StartTime = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_StartTime);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_SubcarrierOffset           =np->nprach_SubcarrierOffset_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierOffset = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierOffset);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->nprach_SubcarrierMSG3_RangeStart  =np->nprach_SubcarrierMSG3_RangeStart_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierMSG3_RangeStart = %d\n",fp->nprach_config_common.nprach_ParametersList.list.nprach_SubcarrierMSG3_RangeStart);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_Offset_RA                  =np->npdcch_Offset_RA_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_Offset_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.npdcch_Offset_RA);
      fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_StartSF_CSS_RA             =np->npdcch_StartSF_CSS_RA_r13;
	  //LOG_D(PHY,"nprach_config_common.nprach_ParametersList.list.npdcch_StartSF_CSS_RA = %d\n",fp->nprach_config_common.nprach_ParametersList.list.array[fp->CE]->npdcch_StartSF_CSS_RA);
      */	  
      }

	  /*Should modify to compute_nprach_seq*/
	  //compute_prach_seq(&fp->prach_config_common,fp->frame_type,PHY_vars_eNB_g[Mod_id][CC_id]->X_u);
	  
	  /*NPDSCH ConfigCommon*/
	  fp->npdsch_config_common.nrs_Power           = radioResourceConfigCommon->npdsch_ConfigCommon_r13.nrs_Power_r13;
	
	  /*NPUSCH ConfigCommon*/
	  /*A list (1-3) should be loop for ack_NACK_NumRepetitions_Msg4*/
	  for (fp->CE=1; fp->CE <= MAX_NPRACH;fp->CE++){
	  fp->npusch_config_common.ack_NACK_NumRepetitions_Msg4[fp->CE]        = radioResourceConfigCommon->npusch_ConfigCommon_r13.ack_NACK_NumRepetitions_Msg4_r13.list.array[fp->CE];
          //LOG_D(PHY,"npusch_config_common.ack_NACK_NumRepetitions_Msg4 = %d]n",fp->npusch_config_common.ack_NACK_NumRepetitions_Msg4);
	  }
	  fp->npusch_config_common.srs_SubframeConfig                          = radioResourceConfigCommon->npusch_ConfigCommon_r13.srs_SubframeConfig_r13;
	  LOG_D(PHY,"npusch_config_common.srs_SubframeConfig = %d]n",fp->npusch_config_common.srs_SubframeConfig);
	  fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence          = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_BaseSequence_r13;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.threeTone_BaseSequence);
	  fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence            = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_BaseSequence_r13;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.sixTone_BaseSequence);
	  fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift           = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.threeTone_CyclicShift = %d]n",fp->npusch_config_common.dmrs_Config.threeTone_CyclicShift);
	  fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift             = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.sixTone_CyclicShift = %d]n",fp->npusch_config_common.dmrs_Config.sixTone_CyclicShift);
	  fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence         = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->twelveTone_BaseSequence_r13;
	  LOG_D(PHY,"npusch_config_common.dmrs_Config.twelveTone_BaseSequence = %d]n",fp->npusch_config_common.dmrs_Config.twelveTone_BaseSequence);

	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH  = radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13;
	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH = %d]n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupAssignmentNPUSCH);
	  fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled    = radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13;
	  LOG_D(PHY,"npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled = %d]n",fp->npusch_config_common.ul_ReferenceSignalsNPUSCH.groupHoppingEnabled);

      /*should change the part that implement the ul hopping in NB-IoT*/
	  //init_ul_hopping(fp);

	  /*UL Power Control Config Common*/
	
	  fp->ul_power_control_config_common.p0_NominalNPUSCH      = radioResourceConfigCommon->uplinkPowerControlCommon_r13.p0_NominalNPUSCH_r13;
	  fp->ul_power_control_config_common.alpha				   = radioResourceConfigCommon->uplinkPowerControlCommon_r13.alpha_r13;
      fp->ul_power_control_config_common.deltaPreambleMsg3     = radioResourceConfigCommon->uplinkPowerControlCommon_r13.deltaPreambleMsg3_r13;

	  /*DL gap*/
                                                                                                    
      fp->DL_gap_config.dl_GapDurationCoeff                        = radioResourceConfigCommon->dl_Gap_r13->dl_GapDurationCoeff_r13;
	  fp->DL_gap_config.dl_GapPeriodicity                      = radioResourceConfigCommon->dl_Gap_r13->dl_GapPeriodicity_r13;
	  fp->DL_gap_config.dl_GapThreshold                        = radioResourceConfigCommon->dl_Gap_r13->dl_GapThreshold_r13;
	  
	  /*PUCCH stuff in LTE*/
	  //init_ncs_cell(fp,PHY_vars_eNB_g[Mod_id][CC_id]->ncs_cell);
	
	  //init_ul_hopping(fp);
	
	
	  
}

void NB_phy_config_dedicated_eNB(uint8_t Mod_id,
                              int CC_id,
                              uint16_t rnti,
                              struct PhysicalConfigDedicated_NB_r13 *physicalConfigDedicated)
{
	  PHY_VARS_eNB *eNB = PHY_vars_eNB_g[Mod_id][CC_id];
	  int8_t UE_id = find_ue(rnti,eNB);
	
	  if (UE_id == -1) {
		LOG_E( PHY, "[eNB %"PRIu8"] find_ue() returns -1\n", Mod_id);
		return;
	  }
	
	/*physicalconfigDedicated is defined in PHY_VARS_eNB in defs.h in PHY layer*/
	  if (physicalConfigDedicated) {
		eNB->physicalConfigDedicated[UE_id] = physicalConfigDedicated;
		LOG_I(PHY,"phy_config_dedicated_eNB: physicalConfigDedicated=%p\n",physicalConfigDedicated);
	
	  } else {
		LOG_E(PHY,"[eNB %d] Received NULL radioResourceConfigDedicated from eNB %d\n",Mod_id, UE_id);
		return;
	  }
	
}


