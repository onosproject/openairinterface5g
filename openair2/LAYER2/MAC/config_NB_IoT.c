
/*! \file config_NB_IoT.c
 * \brief configuration primitives between RRC and MAC
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "BCCH-DL-SCH-Message-NB.h"
#include "RRCConnectionSetup-NB.h"
#include "BCCH-BCH-Message-NB.h"
//#include "SIB-Type-NB-r13.h"

typedef struct eutra_bandentry_NB_s {
  //this should be the colum order of the table below
    int16_t   band;
    uint32_t  ul_min;
    uint32_t  ul_max;
    uint32_t  dl_min;
    uint32_t  dl_max;
    uint32_t  N_OFFs_DL;
} eutra_bandentry_NB_IoT_t;

typedef struct band_info_s {

    int                        nbands;
    eutra_bandentry_NB_IoT_t   band_info[100];

} band_info_t;

static const eutra_bandentry_NB_IoT_t eutra_bandtable[] = {
//[BAND] [FUL_low] [FUL_hi] [FDL_low] [FDL_hig] [NOFF_DL]
  { 1, 19200, 19800, 21100, 21700, 0},
  { 2, 18500, 19100, 19300, 19900, 6000},
  { 3, 17100, 17850, 18050, 18800, 12000},
  { 5,  8240,  8490,  8690,  8940, 24000},
  { 8,  8800, 9150,  9250,  9600, 34500},
  {11, 14279, 14529, 14759, 15009, 47500},
  {12,  6980,  7160,  7280,  7460, 50100},
  {13,  7770,  7870,  7460,  7560, 51800},
  {17,  7040,  7160,  7340,  7460, 57300},
  {18,  8150,  9650,  8600, 10100, 58500},
  {19,  8300,  8450,  8750,  8900, 60000},
  {20,  8320,  8620,  7910,  8210, 61500},
  {25, 18500, 19150, 19300, 19950, 80400},
  {26, 8140,  8490,  8590,  8940, 86900},
  {28, 7030,  7580,  7580,  8130, 92100},
  {31, 45250, 34900, 46250, 35900, 98700},
  {66, 17100, 18000, 21100, 22000, 664360},
  {70, 16950 , 17100,  19950,  20200, 683360}}; //may not used for Rel.13 equipment

//this function is used in phy_config_mib_NB for getting the DL Carrier Frequency from the EARFCN
uint32_t from_earfcn_NB_IoT(int eutra_bandP,uint32_t dl_earfcn, float m_dl) {

  int i;

 // float m_dl = 0; //for the moment we fix but maybe should be dynamic (anyway the 0 works for any case)

  AssertFatal(eutra_bandP <= 70,"eutra_band %d > 70\n",eutra_bandP);
  for (i=0;i<= 70 && eutra_bandtable[i].band!=eutra_bandP;i++);

  return(eutra_bandtable[i].dl_min + 0.0025*(2*m_dl+1)+(dl_earfcn-(eutra_bandtable[i].N_OFFs_DL/10)))*100000;
}


int32_t get_uldl_offset_NB_IoT(int eutra_band) {
  return(-eutra_bandtable[eutra_band].dl_min + eutra_bandtable[eutra_band].ul_min);
}

uint32_t to_earfcn_NB_IoT(int eutra_bandP,uint32_t dl_CarrierFreq, float m_dl) {

  uint32_t dl_CarrierFreq_by_100k = dl_CarrierFreq/100000;

  int i;

  AssertFatal(eutra_bandP < 70,"eutra_band %d > 70\n",eutra_bandP);
  for (i=0;i<69 && eutra_bandtable[i].band!=eutra_bandP;i++);

  AssertFatal(dl_CarrierFreq_by_100k>=eutra_bandtable[i].dl_min,
        "Band %d : DL carrier frequency %u Hz < %u\n",
        eutra_bandP,dl_CarrierFreq,eutra_bandtable[i].dl_min);

  //I would say that for sure the EUTRA band is larger that 1 PRB for NB-IoT so this check may is unuseful
//  AssertFatal(dl_CarrierFreq_by_100k<=(eutra_bandtable[i].dl_max-bw_by_100),
//        "Band %d, bw %u: DL carrier frequency %u Hz > %d\n",
//        eutra_bandP,bw,dl_CarrierFreq,eutra_bandtable[i].dl_max-bw_by_100);


  //based on formula TS 36.101 5.7.3F
  return(dl_CarrierFreq_by_100k - eutra_bandtable[i].dl_min - 0.0025*(2*m_dl+ 1)+ (eutra_bandtable[i].N_OFFs_DL/10));
}



void config_mib_fapi_NB_IoT(int                     physCellId,
        uint8_t                 eutra_band,
        int                     Ncp,
        int                     Ncp_UL,
        int                     p_eNB,
        int                     p_rx_eNB,
        int                     dl_CarrierFreq,
        int                     ul_CarrierFreq,
        long                    *eutraControlRegionSize,
        BCCH_BCH_Message_NB_t   *mib_NB_IoT
        )

{

  nfapi_config_request_t *cfg = &mac_inst->config;

  cfg->sch_config.physical_cell_id.value = physCellId;
  cfg->nfapi_config.rf_bands.rf_band[0] = eutra_band;
  cfg->subframe_config.dl_cyclic_prefix_type.value = Ncp;
  cfg->subframe_config.ul_cyclic_prefix_type.value = Ncp_UL;
  cfg->rf_config.tx_antenna_ports.value = p_eNB;
  cfg->nfapi_config.earfcn.value = 9370; // value from taiwan commercial base station, just setting for now, will use formula to calculate it layer
  //cfg->nb_iot_config.prb_index.value = // need to set in thread part
  
  switch (mib_NB_IoT->message.operationModeInfo_r13.present)
    {
      //FAPI specs pag 135
    case MasterInformationBlock_NB__operationModeInfo_r13_PR_inband_SamePCI_r13:
      
      cfg->nb_iot_config.operating_mode.value  = 0;
      cfg->nb_iot_config.prb_index.value       = mib_NB_IoT->message.operationModeInfo_r13.choice.inband_SamePCI_r13.eutra_CRS_SequenceInfo_r13; //see TS 36.213 ch 16.0
      cfg->nb_iot_config.assumed_crs_aps.value = -1; //is not defined so we put a negative value
      
      if(eutraControlRegionSize == NULL)
  LOG_E(RRC, "rrc_mac_config_req_eNB_NB_IoT: operation mode is in-band but eutraControlRegionSize is not defined\n");
      else
  cfg->nb_iot_config.control_region_size.value = *eutraControlRegionSize;
      
      
      //m_dl = NB_Category_Offset_anchor[rand()%4];
      
      
      break;
      
    case MasterInformationBlock_NB__operationModeInfo_r13_PR_inband_DifferentPCI_r13:
      
      cfg->nb_iot_config.operating_mode.value = 1;
      //XXX problem: fapi think to define also eutra_CRS_sequenceInfo also for in band with different PCI but the problem is that we don-t have i
      //XXX should pass the prb_index may defined by configuration file depending on the LTE band we are considering (see Rhode&Shwartz whitepaper pag9)
      //nb_iot_config.prb_index.value =
      cfg->nb_iot_config.assumed_crs_aps.value = mib_NB_IoT->message.operationModeInfo_r13.choice.inband_DifferentPCI_r13.eutra_NumCRS_Ports_r13;
      
      if(eutraControlRegionSize == NULL)
  LOG_E(RRC, "rrc_mac_config_req_eNB_NB_IoT: operation mode is in-band but eutraControlRegionSize is not defined\n");
      else
  cfg->nb_iot_config.control_region_size.value = *eutraControlRegionSize;
      
      break;
      
    case MasterInformationBlock_NB__operationModeInfo_r13_PR_guardband_r13:
      
      cfg->nb_iot_config.operating_mode.value      = 2;
      //XXX should pass the prb_index may defined by configuration file depending on the LTE band we are considering (see Rhode&Shwartz whitepaper pag9)
      //nb_iot_config.prb_index.value =
      cfg->nb_iot_config.control_region_size.value = -1; //should not being defined so we put a negative value
      cfg->nb_iot_config.assumed_crs_aps.value     = -1; //is not defined so we put a negative value
      
      break;
      
    case MasterInformationBlock_NB__operationModeInfo_r13_PR_standalone_r13:
      
      cfg->nb_iot_config.operating_mode.value       = 3;
      cfg->nb_iot_config.prb_index.value            = -1;   // is not defined for this case (put a negative random value--> will be not considered for encoding, scrambling procedures)
      cfg->nb_iot_config.control_region_size.value  = -1;   //is not defined so we put a negative value
      cfg->nb_iot_config.assumed_crs_aps.value      = -1;   //is not defined so we put a negative value
      
      break;
    default:
      LOG_E(RRC, "rrc_mac_config_req_eNB_NB_IoT: NB-IoT operating Mode (MIB-NB) not set\n");
      break;
    }
  
}

void config_sib2_fapi_NB_IoT(
                        int physCellId,
                        RadioResourceConfigCommonSIB_NB_r13_t   *radioResourceConfigCommon
                        )
{

  nfapi_config_request_t *cfg = &mac_inst->config;
    /*
     * Following the FAPI like approach:
     * 1)fill the PHY_Config_t structure (PHY_INTERFACE/IF_Module_NB_IoT.h)
     * 1.1) check for how many NPRACH resources has been set and enable the corresponding parameter
     * 1.2)fill the structure PHY_Config_t (shared structure of the IF_Module
     * 2)Call the PHY_config_req for trigger the NB_phy_config_sib2_eNB()
     */

    /*NPRACH Resources*/

    NPRACH_Parameters_NB_r13_t* nprach_parameter;

      cfg->nb_iot_config.nprach_config_0_enabled.value = 0;
      cfg->nb_iot_config.nprach_config_1_enabled.value = 0;
      cfg->nb_iot_config.nprach_config_2_enabled.value = 0;

    if(radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[0]!=NULL&&radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[1]!=NULL&&radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[2]!=NULL)
    {
      nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[0];
      cfg->nb_iot_config.nprach_config_0_enabled.value = 1;
      cfg->nb_iot_config.nprach_config_0_cp_length.value = radioResourceConfigCommon->nprach_Config_r13.nprach_CP_Length_r13;
      cfg->nb_iot_config.nprach_config_0_sf_periodicity.value = nprach_parameter->nprach_Periodicity_r13;
      cfg->nb_iot_config.nprach_config_0_start_time.value = nprach_parameter->nprach_StartTime_r13;
      cfg->nb_iot_config.nprach_config_0_subcarrier_offset.value = nprach_parameter->nprach_SubcarrierOffset_r13;
      cfg->nb_iot_config.nprach_config_0_number_of_subcarriers.value = nprach_parameter->nprach_NumSubcarriers_r13;
      cfg->nb_iot_config.nprach_config_0_number_of_repetitions_per_attempt.value = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;
      /*
      //MP: missed configuration for FAPI-style structure (I have added on my own byt maybe are not needed)
      extra_phy_parms.nprach_config_0_subcarrier_MSG3_range_start = nprach_parameter->nprach_SubcarrierMSG3_RangeStart_r13;
      extra_phy_parms.nprach_config_0_max_num_preamble_attempt_CE = nprach_parameter->maxNumPreambleAttemptCE_r13;
      extra_phy_parms.nprach_config_0_npdcch_num_repetitions_RA = nprach_parameter->npdcch_NumRepetitions_RA_r13;
      extra_phy_parms.nprach_config_0_npdcch_startSF_CSS_RA = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
      extra_phy_parms.nprach_config_0_npdcch_offset_RA = nprach_parameter->npdcch_Offset_RA_r13;
      */
      //rsrp_ThresholdsPrachInfoList_r13 /*OPTIONAL*/


      nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[1];
      cfg->nb_iot_config.nprach_config_1_enabled.value = 1;
      cfg->nb_iot_config.nprach_config_1_cp_length.value = radioResourceConfigCommon->nprach_Config_r13.nprach_CP_Length_r13;
      cfg->nb_iot_config.nprach_config_1_sf_periodicity.value = nprach_parameter->nprach_Periodicity_r13;
      cfg->nb_iot_config.nprach_config_1_start_time.value = nprach_parameter->nprach_StartTime_r13;
      cfg->nb_iot_config.nprach_config_1_subcarrier_offset.value = nprach_parameter->nprach_SubcarrierOffset_r13;
      cfg->nb_iot_config.nprach_config_1_number_of_subcarriers.value = nprach_parameter->nprach_NumSubcarriers_r13;
      cfg->nb_iot_config.nprach_config_1_number_of_repetitions_per_attempt.value = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;

      /*
      //MP: missed configuration for FAPI-style structure (I have added on my own byt maybe are not needed)
      extra_phy_parms.nprach_config_1_subcarrier_MSG3_range_start = nprach_parameter->nprach_SubcarrierMSG3_RangeStart_r13;
      extra_phy_parms.nprach_config_1_max_num_preamble_attempt_CE = nprach_parameter->maxNumPreambleAttemptCE_r13;
      extra_phy_parms.nprach_config_1_npdcch_num_repetitions_RA = nprach_parameter->npdcch_NumRepetitions_RA_r13;
      extra_phy_parms.nprach_config_1_npdcch_startSF_CSS_RA = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
      extra_phy_parms.nprach_config_1_npdcch_offset_RA = nprach_parameter->npdcch_Offset_RA_r13;
      */
      //rsrp_ThresholdsPrachInfoList_r13 /*OPTIONAL*/


      nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[2];
      cfg->nb_iot_config.nprach_config_2_enabled.value = 1;
      cfg->nb_iot_config.nprach_config_2_cp_length.value = radioResourceConfigCommon->nprach_Config_r13.nprach_CP_Length_r13;
      cfg->nb_iot_config.nprach_config_2_sf_periodicity.value = nprach_parameter->nprach_Periodicity_r13;
      cfg->nb_iot_config.nprach_config_2_start_time.value = nprach_parameter->nprach_StartTime_r13;
      cfg->nb_iot_config.nprach_config_2_subcarrier_offset.value = nprach_parameter->nprach_SubcarrierOffset_r13;
      cfg->nb_iot_config.nprach_config_2_number_of_subcarriers.value = nprach_parameter->nprach_NumSubcarriers_r13;
      cfg->nb_iot_config.nprach_config_2_number_of_repetitions_per_attempt.value = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;

      /*
      //MP: missed configuration for FAPI-style structure (I have added on my own byt maybe are not needed)
      extra_phy_parms.nprach_config_2_subcarrier_MSG3_range_start = nprach_parameter->nprach_SubcarrierMSG3_RangeStart_r13;
      extra_phy_parms.nprach_config_2_max_num_preamble_attempt_CE = nprach_parameter->maxNumPreambleAttemptCE_r13;
      extra_phy_parms.nprach_config_2_npdcch_num_repetitions_RA = nprach_parameter->npdcch_NumRepetitions_RA_r13;
      extra_phy_parms.nprach_config_2_npdcch_startSF_CSS_RA = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
      extra_phy_parms.nprach_config_2_npdcch_offset_RA = nprach_parameter->npdcch_Offset_RA_r13;
      */
      //rsrp_ThresholdsPrachInfoList_r13 /*OPTIONAL*/
    }else
    {
      LOG_E(MAC,"NPRACH Configuration isn't set properly\n");
    }

    LOG_I(MAC,"Fill parameters of FAPI NPRACH done\n");
    /*NPDSCH ConfigCommon*/

    //FIXME: MP: FAPI specs define a range of value [0-255]==[0db - 63.75db] with 0.25db step -- corrispondence in 3GPP specs???
    cfg->rf_config.reference_signal_power.value = radioResourceConfigCommon->npdsch_ConfigCommon_r13.nrs_Power_r13;

    /*NPUSCH ConfigCommon*/

    //a pointer to the first element of the list
    //    extra_phy_parms.ack_nack_numRepetitions_MSG4 = radioResourceConfigCommon->npusch_ConfigCommon_r13.ack_NACK_NumRepetitions_Msg4_r13.list.array[0];


    if(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13 != NULL)/* OPTIONAL */
    {
      /* OPTIONAL */
      if(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_BaseSequence_r13!= NULL)
          cfg->nb_iot_config.three_tone_base_sequence.value  = *(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_BaseSequence_r13);
      else
        cfg->nb_iot_config.three_tone_base_sequence.value = physCellId%12; //see spec TS 36.331 NPUSCH-Config-NB

      /* OPTIONAL */
      if(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_BaseSequence_r13!= NULL)
          cfg->nb_iot_config.six_tone_base_sequence.value = *(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_BaseSequence_r13);
      else
        cfg->nb_iot_config.six_tone_base_sequence.value = physCellId%14; //see spec TS 36.331 NPUSCH-Config-NB

      /* OPTIONAL */
      if(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->twelveTone_BaseSequence_r13!= NULL)
        cfg->nb_iot_config.twelve_tone_base_sequence.value = *(radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->twelveTone_BaseSequence_r13);
      else
        cfg->nb_iot_config.twelve_tone_base_sequence.value = physCellId%30; //see spec TS 36.331 NPUSCH-Config-NB

        cfg->nb_iot_config.three_tone_cyclic_shift.value = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13;
        cfg->nb_iot_config.six_tone_cyclic_shift.value = radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13;
    }


    //NOTE: MP: FAPI specs for UL RS Configurations seems to be targeted for LTE and not for NB-IoT
    if(radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13 == TRUE)
      cfg->uplink_reference_signal_config.uplink_rs_hopping.value = 1; //RS_GROUP_HOPPING (FAPI specs pag 127)
    else
      cfg->uplink_reference_signal_config.uplink_rs_hopping.value = 0;//RS_NO_HOPPING

    cfg->uplink_reference_signal_config.group_assignment.value = radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13;

    //Some missed parameters are in UL_CONFIG.request message (P7) in FAPI specs. and not configured through P5 procedure
       //ack_NACK_NumRepetitions_Msg4_r13
       //srs_SubframeConfig_r13 /* OPTIONAL */

      /*DL GAP config */
    if(radioResourceConfigCommon->dl_Gap_r13 !=NULL)/* OPTIONAL */
    {
      cfg->nb_iot_config.dl_gap_config_enable.value        = 1;
      cfg->nb_iot_config.dl_gap_threshold.value            = radioResourceConfigCommon->dl_Gap_r13->dl_GapThreshold_r13;
          cfg->nb_iot_config.dl_gap_duration_coefficient.value = radioResourceConfigCommon->dl_Gap_r13->dl_GapDurationCoeff_r13;
          cfg->nb_iot_config.dl_gap_periodicity.value          = radioResourceConfigCommon->dl_Gap_r13->dl_GapPeriodicity_r13;
    }
    else
      cfg->nb_iot_config.dl_gap_config_enable.value        = 0;

      /*UL Power Control ConfigCommon*/
    /*
    //nothing defined in FAPI specs
    extra_phy_parms.p0_nominal_npusch  = radioResourceConfigCommon->uplinkPowerControlCommon_r13.p0_NominalNPUSCH_r13;
    extra_phy_parms.alpha              = radioResourceConfigCommon->uplinkPowerControlCommon_r13.alpha_r13;
    extra_phy_parms.delta_preamle_MSG3 = radioResourceConfigCommon->uplinkPowerControlCommon_r13.deltaPreambleMsg3_r13;
    */

      /*RACH Config Common*/
    //nothing defined in FAPI specs

}

///-------------------------------------------Function---------------------------------------------///

void rrc_mac_config_req_NB_IoT(
    module_id_t                             Mod_idP,
    int                                     CC_idP,
    int                                     rntiP,
    rrc_eNB_carrier_data_NB_IoT_t           *carrier,
    SystemInformationBlockType1_NB_t        *sib1_NB_IoT,
    RadioResourceConfigCommonSIB_NB_r13_t   *radioResourceConfigCommon,
    PhysicalConfigDedicated_NB_r13_t        *physicalConfigDedicated,
    LogicalChannelConfig_NB_r13_t           *logicalChannelConfig,            //FIXME: decide how to use it
    uint8_t                                 ded_flag,
    uint8_t                                 ue_list_ded_num)
{

    int UE_id = -1;

    rrc_config_NB_IoT_t                     *mac_config=NULL;

    mac_top_init_eNB_NB_IoT();

    mac_config = &mac_inst->rrc_config;

    if(ded_flag==0)
    {
    }else
    {

      // now we only have 3 UE list USS for three CE levels
    // we fix value for RMAX to 8 / 16 / 32
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].R_max         = 8 + 8*ue_list_ded_num;
    // fix value for G to 8 / 4 / 2
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].G             = 2 + (2-ue_list_ded_num)*(3-ue_list_ded_num);
    // fix a_offest to 0 / 0 / 0 
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].a_offset      = 0;
   
    return;

    }




    if (&carrier->mib_NB_IoT != NULL){
           /*printf(MAC,
            "Configuring MIB for instance %d, CCid %d : (band %ld,Nid_cell %d,TX antenna port (p) %d,DL freq %u\n",
            Mod_idP,
            CC_idP,
            sib1_NB_IoT->freqBandIndicator_r13,
            carrier->physCellId,
            carrier->p_eNB,
            carrier->dl_CarrierFreq
           );*/

    /*
     * Following the FAPI like approach:
     * 1)fill the PHY_Config_t structure (PHY_INTERFACE/IF_Module_NB_IoT.h)
     * 2)Call the PHY_config_req for trigger the NB_phy_config_mib_eNB() at the end
     */

    //Mapping OAI params into FAPI params
      config_mib_fapi_NB_IoT(carrier->physCellId,
            sib1_NB_IoT->freqBandIndicator_r13,
            carrier->Ncp,
            carrier->Ncp_UL,
            carrier->p_eNB,
            carrier->p_rx_eNB,
            carrier->dl_CarrierFreq,
            carrier->ul_CarrierFreq,
            sib1_NB_IoT->eutraControlRegionSize_r13,
            &carrier->mib_NB_IoT
            );
      

    }else{
         LOG_E(MAC,"carrier->mib_NB_IoT is NULL\n"); 
   return;
    }

    if(sib1_NB_IoT != NULL)
    {
        mac_config->sib1_NB_IoT_sched_config.repetitions = 4;

        //printf("[ASN Debug] SI P: %ld\n",sib1_NB_IoT->schedulingInfoList_r13.list.array[0]->si_Periodicity_r13);



        mac_config->sib1_NB_IoT_sched_config.starting_rf = (intptr_t)(sib1_NB_IoT->si_RadioFrameOffset_r13);
        mac_config->si_window_length = sib1_NB_IoT->si_WindowLength_r13;

        SchedulingInfo_NB_r13_t *scheduling_info_list;


        ///OAI only supports SIB2/3-NB for the sibs
        if ( sib1_NB_IoT->schedulingInfoList_r13.list.array[0] != NULL){ 
            scheduling_info_list = sib1_NB_IoT->schedulingInfoList_r13.list.array[0];

            mac_config->sibs_NB_IoT_sched[0].si_periodicity =   scheduling_info_list->si_Periodicity_r13 ;
            //printf("Pass first SIBs Asn, SI P:%d\n",mac_config->sibs_NB_IoT_sched[0].si_periodicity);
            mac_config->sibs_NB_IoT_sched[0].si_repetition_pattern =  scheduling_info_list->si_RepetitionPattern_r13 ;
            mac_config->sibs_NB_IoT_sched[0].sib_mapping_info =   scheduling_info_list->sib_MappingInfo_r13.list.array[0][0];
            mac_config->sibs_NB_IoT_sched[0].si_tb =      scheduling_info_list->si_TB_r13  ;
          
        } else { //set this value for now to be test further

            mac_config->sibs_NB_IoT_sched[0].si_periodicity =   si_Periodicity_rf4096 ;
            mac_config->sibs_NB_IoT_sched[0].si_repetition_pattern =  si_RepetitionPattern_every2ndRF;
     
            mac_config->sibs_NB_IoT_sched[0].sib_mapping_info =   sib3_v;
            mac_config->sibs_NB_IoT_sched[0].si_tb =      si_TB_680;
        }

        /// Thiese value is setting for different SIB set
       if ( sib1_NB_IoT->schedulingInfoList_r13.list.array[1] != NULL) {
            scheduling_info_list = sib1_NB_IoT->schedulingInfoList_r13.list.array[1];
            mac_config->sibs_NB_IoT_sched[1].si_periodicity =   scheduling_info_list->si_Periodicity_r13 ;
            mac_config->sibs_NB_IoT_sched[1].si_repetition_pattern =  scheduling_info_list->si_RepetitionPattern_r13 ;
            mac_config->sibs_NB_IoT_sched[1].sib_mapping_info =   scheduling_info_list->sib_MappingInfo_r13.list.array[0][0] | scheduling_info_list->sib_MappingInfo_r13.list.array[1][0];
            mac_config->sibs_NB_IoT_sched[1].si_tb =      scheduling_info_list->si_TB_r13  ;
       } else { //set this value for now to be test further     
            mac_config->sibs_NB_IoT_sched[1].si_periodicity =   si_Periodicity_rf4096 ;
            mac_config->sibs_NB_IoT_sched[1].si_repetition_pattern =  si_RepetitionPattern_every2ndRF;     
            mac_config->sibs_NB_IoT_sched[1].sib_mapping_info =   sib3_v;
            mac_config->sibs_NB_IoT_sched[1].si_tb =      si_TB_680;
        }
 
      if ( sib1_NB_IoT->schedulingInfoList_r13.list.array[2] != NULL) {
            scheduling_info_list = sib1_NB_IoT->schedulingInfoList_r13.list.array[2];
            mac_config->sibs_NB_IoT_sched[2].si_periodicity =       scheduling_info_list->si_Periodicity_r13 ;
            mac_config->sibs_NB_IoT_sched[2].si_repetition_pattern =    scheduling_info_list->si_RepetitionPattern_r13 ;
            mac_config->sibs_NB_IoT_sched[2].sib_mapping_info =       scheduling_info_list->sib_MappingInfo_r13.list.array[0][0] | scheduling_info_list->sib_MappingInfo_r13.list.array[1][0];
            mac_config->sibs_NB_IoT_sched[2].si_tb =          scheduling_info_list->si_TB_r13  ;
       }  else { //set this value for now to be test further
            mac_config->sibs_NB_IoT_sched[2].si_periodicity =   si_Periodicity_rf4096 ;
            mac_config->sibs_NB_IoT_sched[2].si_repetition_pattern =  si_RepetitionPattern_every2ndRF;     
            mac_config->sibs_NB_IoT_sched[2].sib_mapping_info =   sib3_v;
            mac_config->sibs_NB_IoT_sched[2].si_tb =      si_TB_680;
        } 
        mac_config->sibs_NB_IoT_sched[3].sib_mapping_info = 0x0;
        mac_config->sibs_NB_IoT_sched[4].sib_mapping_info = 0x0;
        mac_config->sibs_NB_IoT_sched[5].sib_mapping_info = 0x0;

    }else{
         LOG_E(MAC,"sib1_NB_IoT is NULL\n"); 
    }



    if (radioResourceConfigCommon!=NULL) {

        //if(cfg == NULL) LOG_E(MAC, "rrc_mac_config_req_eNB_NB_IoT: trying to configure PHY but no config.request message in config_INFO is allocated\n");

        // need to fix these array setting issue

        //LOG_I(MAC,"[CONFIG]SIB2/3-NB radioResourceConfigCommon Contents (partial)\n");

        //LOG_I(MAC,"[CONFIG]npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13= %ld\n", radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13);
        //LOG_I(MAC,"[CONFIG]npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13= %ld\n", radioResourceConfigCommon->npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13);
        //LOG_I(MAC,"[CONFIG]npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13= %d\n", radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13);
        //LOG_I(MAC,"[CONFIG]npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13= %ld\n", radioResourceConfigCommon->npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13);

        NPRACH_Parameters_NB_r13_t* nprach_parameter;

        ///CE level 0
        if ( radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[0] != NULL) {
        
        nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[0];
        LOG_I(MAC,"NPRACH 0 setting: NumRepetiion: %ld Period: %ld size of list %d\n",nprach_parameter->numRepetitionsPerPreambleAttempt_r13,nprach_parameter->nprach_Periodicity_r13,radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.size);
        mac_config->mac_NPRACH_ConfigSIB[0].mac_numRepetitionsPerPreambleAttempt_NB_IoT = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;
        mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_NumRepetitions_RA_NB_IoT         = nprach_parameter->npdcch_NumRepetitions_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_StartSF_CSS_RA_NB_IoT            = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_Offset_RA_NB_IoT                 = nprach_parameter->npdcch_Offset_RA_r13;
        }
        ///CE level 1

        if ( radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[1] != NULL) {
        nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[1];
        LOG_I(MAC,"NPRACH 1 setting: NumRepetiion: %ld size of list %d\n",nprach_parameter->numRepetitionsPerPreambleAttempt_r13,radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.size);
        mac_config->mac_NPRACH_ConfigSIB[1].mac_numRepetitionsPerPreambleAttempt_NB_IoT = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;
        mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_NumRepetitions_RA_NB_IoT         = nprach_parameter->npdcch_NumRepetitions_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_StartSF_CSS_RA_NB_IoT            = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_Offset_RA_NB_IoT                 = nprach_parameter->npdcch_Offset_RA_r13;
        }
        ///CE level 2
        if ( radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[2] != NULL) {
        nprach_parameter = radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.array[2];
        LOG_I(MAC,"NPRACH 2 setting: NumRepetiion: %ld size of list %d\n",nprach_parameter->numRepetitionsPerPreambleAttempt_r13,radioResourceConfigCommon->nprach_Config_r13.nprach_ParametersList_r13.list.size);
        mac_config->mac_NPRACH_ConfigSIB[2].mac_numRepetitionsPerPreambleAttempt_NB_IoT = nprach_parameter->numRepetitionsPerPreambleAttempt_r13;
        mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_NumRepetitions_RA_NB_IoT         = nprach_parameter->npdcch_NumRepetitions_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_StartSF_CSS_RA_NB_IoT            = nprach_parameter->npdcch_StartSF_CSS_RA_r13;
        mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_Offset_RA_NB_IoT                 = nprach_parameter->npdcch_Offset_RA_r13;
        }


    config_sib2_fapi_NB_IoT(carrier->physCellId,radioResourceConfigCommon);

    }else{
         LOG_E(MAC,"radioResourceConfigCommon is NULL\n"); 
    }


    if (logicalChannelConfig!= NULL) {



        if (UE_id == -1)
        {
            LOG_E(MAC,"%s:%d:%s: ERROR, UE_id == -1\n", __FILE__, __LINE__, __FUNCTION__);
        }
        else
        {
        //logical channel group not defined for nb-iot --> no UL specific Parameter
        // or at least LCGID should be set to 0 for NB-IoT (See TS 36.321 ch 6.1.3.1) so no make sense to store this
        }
    }

    if (physicalConfigDedicated != NULL) {



        if (UE_id == -1)
            LOG_E(MAC,"%s:%d:%s: ERROR, UE_id == -1\n", __FILE__, __LINE__, __FUNCTION__);
        else
        {
    /*
    extra_phy_parms.npdcch_NumRepetitions = physicalConfigDedicated->npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13; //Rmax
    extra_phy_parms.npdcch_Offset_USS     = physicalConfigDedicated->npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    extra_phy_parms.npdcch_StartSF_USS    = physicalConfigDedicated->npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    */
        //extra_phy_parms.phy_config_dedicated = physicalConfigDedicated; //for the moment fapi not allow this so not used

        }
    }

    //LOG_I(MAC,"CP Length Checking %u\n",RC.nb_iot_mac[Mod_idP]->config.nb_iot_config.nprach_config_0_cp_length.value);
    
    if(mac_inst->if_inst_NB_IoT!=NULL)
    {
      if (radioResourceConfigCommon!=NULL) {
        AssertFatal( mac_inst->if_inst_NB_IoT->PHY_config_req != NULL, "rrc_mac_config_req_eNB_NB_IoT: PHY_config_req pointer function is NULL\n");
        PHY_Config_NB_IoT_t phycfg;
        phycfg.mod_id = Mod_idP;
        phycfg.cfg    = &mac_inst->config;
    
      if (mac_inst->if_inst_NB_IoT->PHY_config_req) mac_inst->if_inst_NB_IoT->PHY_config_req(&phycfg); 
    }
   }else{
    LOG_E(MAC,"NB-IoT IF INST is NULL, need to fix\n");
   }

    //return 0;

      init_mac_NB_IoT(mac_inst);

      LOG_I(MAC,"[NB-IoT] Init_MAC done\n");

      
      //for sacheduler testing
      /*for(int i =0;i<30;i++)
      {
        LOG_I(MAC,"[NB-IoT] scheduler testing start %d\n",i);

        eNB_dlsch_ulsch_scheduler_NB_IoT(RC.nb_iot_mac[Mod_idP], i);

        LOG_I(MAC,"[NB-IoT] scheduler testing done %d\n",i);
      }*/

//      RC.L1_NB_IoT[Mod_idP]->configured=1;


   /*if( ded_flag!=0 )
   {
    
    mac_config->npdcch_ConfigDedicated[0].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;
    mac_config->npdcch_ConfigDedicated[1].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;
    mac_config->npdcch_ConfigDedicated[2].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;

    mac_config->npdcch_ConfigDedicated[0].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    mac_config->npdcch_ConfigDedicated[1].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    mac_config->npdcch_ConfigDedicated[2].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;

    mac_config->npdcch_ConfigDedicated[0].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    mac_config->npdcch_ConfigDedicated[1].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    mac_config->npdcch_ConfigDedicated[2].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    
    // now we only have 3 UE list USS
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;

    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;

    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    }


    return 0;*/


}
