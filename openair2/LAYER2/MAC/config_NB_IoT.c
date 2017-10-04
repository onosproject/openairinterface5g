#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"

///-------------------------------------------Function---------------------------------------------///

void init_rrc_NB_IoT(void)
{
    int i;
    
    MIB.message.schedulingInfoSIB1_r13 = 0;

    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.si_RadioFrameOffset_r13 = (long*)malloc(sizeof(long));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.si_RadioFrameOffset_r13[0]= 0;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.si_WindowLength_r13 = SystemInformationBlockType1_NB__si_WindowLength_r13_ms480 ;

    //SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13 = (SchedulingInfoList_NB_r13_t*)malloc(sizeof(SchedulingInfoList_NB_r13_t));

    /// Allocate Three CE level to schedulingInfoList
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array = (SchedulingInfo_NB_r13_t**)malloc(3*sizeof(SchedulingInfo_NB_r13_t*));
    for(i=0;i<3;++i)
    {
        SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[i] = (SchedulingInfo_NB_r13_t*)malloc(sizeof(SchedulingInfo_NB_r13_t));
    }

    /// Allocate how many SIB mapping-----------------------------------------------------------------------------------------------------///
    /// CE0
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array    =(SIB_Type_NB_r13_t**)malloc(2*sizeof(SIB_Type_NB_r13_t*));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[0] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[1] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    /// CE1
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array    =(SIB_Type_NB_r13_t**)malloc(2*sizeof(SIB_Type_NB_r13_t*));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[0] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[1] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    /// CE2
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array    =(SIB_Type_NB_r13_t**)malloc(2*sizeof(SIB_Type_NB_r13_t*));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[0] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[1] =(SIB_Type_NB_r13_t*)malloc(sizeof(SIB_Type_NB_r13_t));
    /// End Allocate SIB mapping----------------------------------------------------------------------------------------------------------///


    /// Setting Scheduling Information SI for Three CE level------------------------------------------------------------------------------///
    /// CE0
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_Periodicity_r13                = SchedulingInfo_NB_r13__si_Periodicity_r13_rf4096;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_RepetitionPattern_r13          = SchedulingInfo_NB_r13__si_RepetitionPattern_r13_every2ndRF;
    //SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[0][0] = SIB_Type_NB_r13_sibType2_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[1][0] = SIB_Type_NB_r13_sibType3_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_TB_r13                         = SchedulingInfo_NB_r13__si_TB_r13_b680;
    /// CE1
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->si_Periodicity_r13                = SchedulingInfo_NB_r13__si_Periodicity_r13_rf4096;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->si_RepetitionPattern_r13          = SchedulingInfo_NB_r13__si_RepetitionPattern_r13_every2ndRF;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[0][0] = SIB_Type_NB_r13_sibType4_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[1][0] = SIB_Type_NB_r13_sibType5_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->si_TB_r13                         = SchedulingInfo_NB_r13__si_TB_r13_b680;
    /// CE2
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->si_Periodicity_r13                = SchedulingInfo_NB_r13__si_Periodicity_r13_rf4096;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->si_RepetitionPattern_r13          = SchedulingInfo_NB_r13__si_RepetitionPattern_r13_every2ndRF;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[0][0] = SIB_Type_NB_r13_sibType14_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[1][0] = SIB_Type_NB_r13_sibType16_NB_r13;
    SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->si_TB_r13                         = SchedulingInfo_NB_r13__si_TB_r13_b680;
    /// End Setting Scheduling Information SI---------------------------------------------------------------------------------------------///

    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array = (struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member **)malloc(sizeof(struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member*));
    /// Allocate RACH_ConfigCommon_NB_IoT for Three CE level
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list.array = (RACH_Info_NB_r13_t**)malloc(3*sizeof(RACH_Info_NB_r13_t*));
    for(i=0;i<3;i++){
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list.array[i] = (RACH_Info_NB_r13_t*)malloc(sizeof(RACH_Info_NB_r13_t));
    }

    /// Allocate NPRACH_ConfigSIB_NB_IoT for Three CE level
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array = (NPRACH_Parameters_NB_r13_t**)malloc(3*sizeof(NPRACH_Parameters_NB_r13_t*));
    for(i=0;i<3;i++){
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[i] = (NPRACH_Parameters_NB_r13_t*)malloc(sizeof(NPRACH_Parameters_NB_r13_t));
    }

    /// Setting RACH_ConfigCommon_NB_IoT ra_ResponseWindowSize for Three CE level
    /// CE0
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list.array[0].ra_ResponseWindowSize_r13
    /// CE1
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list.array[1].ra_ResponseWindowSize_r13
    /// CE2
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list.array[2].ra_ResponseWindowSize_r13

    /// Setting NPRACH_ConfigSIB_NB_IoT  for Three CE level
    /// CE0
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->nprach_Periodicity_r13               =NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms320;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->nprach_StartTime_r13                 =NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms8;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->nprach_SubcarrierOffset_r13          =NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n0;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->nprach_NumSubcarriers_r13            =NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n12;
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->nprach_SubcarrierMSG3_RangeStart_r13 =
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->maxNumPreambleAttemptCE_r13          =
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->numRepetitionsPerPreambleAttempt_r13 =NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n1;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_NumRepetitions_RA_r13         =NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r64;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_StartSF_CSS_RA_r13            =NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v4;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_Offset_RA_r13                 =NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_zero;
    /// CE1
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->nprach_Periodicity_r13               =NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms320;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->nprach_StartTime_r13                 =NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms8;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->nprach_SubcarrierOffset_r13          =NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n12;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->nprach_NumSubcarriers_r13            =NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n12;
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->nprach_SubcarrierMSG3_RangeStart_r13 =
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->maxNumPreambleAttemptCE_r13          =
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->numRepetitionsPerPreambleAttempt_r13 =NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n2;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_NumRepetitions_RA_r13         =NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r64;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_StartSF_CSS_RA_r13            =NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v4;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_Offset_RA_r13                 =NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_zero;
    /// CE2
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->nprach_Periodicity_r13               =NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms320;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->nprach_StartTime_r13                 =NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms8;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->nprach_SubcarrierOffset_r13          =NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n24;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->nprach_NumSubcarriers_r13            =NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n24;
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->nprach_SubcarrierMSG3_RangeStart_r13 =
    //SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->maxNumPreambleAttemptCE_r13          =
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->numRepetitionsPerPreambleAttempt_r13 =NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n4;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_NumRepetitions_RA_r13         =NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r64;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_StartSF_CSS_RA_r13            =NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v4;
    SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_Offset_RA_r13                 =NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_zero;

    /// Allocate RRCConnectionSetup_NB_IoT RadioResourceConfigDedicated for Three CE level
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13 = (PhysicalConfigDedicated_NB_r13_t*)malloc(3*sizeof(PhysicalConfigDedicated_NB_r13_t));
    for(i=0;i<3;i++){
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[i].npdcch_ConfigDedicated_r13 = (NPDCCH_ConfigDedicated_NB_r13_t*)malloc(sizeof(NPDCCH_ConfigDedicated_NB_r13_t));
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[i].npusch_ConfigDedicated_r13 = (NPUSCH_ConfigDedicated_NB_r13_t*)malloc(sizeof(NPUSCH_ConfigDedicated_NB_r13_t));
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[i].npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13 = (ACK_NACK_NumRepetitions_NB_r13_t*)malloc(sizeof(ACK_NACK_NumRepetitions_NB_r13_t));
    }

    /// Setting Dedicated Configuration for Three CE level
    /// CE0
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13      =NPDCCH_ConfigDedicated_NB_r13__npdcch_NumRepetitions_r13_r8;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13         =NPDCCH_ConfigDedicated_NB_r13__npdcch_StartSF_USS_r13_v8;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13          =NPDCCH_ConfigDedicated_NB_r13__npdcch_Offset_USS_r13_zero;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13[0] =ACK_NACK_NumRepetitions_NB_r13_r1;
    /// CE1
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13      =NPDCCH_ConfigDedicated_NB_r13__npdcch_NumRepetitions_r13_r16;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13         =NPDCCH_ConfigDedicated_NB_r13__npdcch_StartSF_USS_r13_v4;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13          =NPDCCH_ConfigDedicated_NB_r13__npdcch_Offset_USS_r13_zero;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13[0] =ACK_NACK_NumRepetitions_NB_r13_r2;
    /// CE2
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13      =NPDCCH_ConfigDedicated_NB_r13__npdcch_NumRepetitions_r13_r32;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13         =NPDCCH_ConfigDedicated_NB_r13__npdcch_StartSF_USS_r13_v2;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13          =NPDCCH_ConfigDedicated_NB_r13__npdcch_Offset_USS_r13_zero;
    DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13[0] =ACK_NACK_NumRepetitions_NB_r13_r4;
  //  printf("ack_NACK_NumRepetitions_r13 %d\n",DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13[0] );
  //  printf("RRC Initial Success ! :D \n");
}

void rrc_mac_config_req_NB_IoT(rrc_config_NB_IoT_t *mac_config,
							   uint8_t mib_flag,
							   uint8_t sib_flag,
							   uint8_t ded_flag,
							   uint8_t ue_list_ded_num)
{
    if (mib_flag != 0){

    }

    if (sib_flag != 0){
	mac_config->sib1_NB_IoT_sched_config.repetitions = 4;
    mac_config->sib1_NB_IoT_sched_config.starting_rf = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.si_RadioFrameOffset_r13[0];
    mac_config->si_window_length = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.si_WindowLength_r13;


	mac_config->sibs_NB_IoT_sched[0].si_periodicity = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_Periodicity_r13 ;
	mac_config->sibs_NB_IoT_sched[0].si_repetition_pattern = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_RepetitionPattern_r13 ;
	mac_config->sibs_NB_IoT_sched[0].sib_mapping_info = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[0][0] | SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->sib_MappingInfo_r13.list.array[1][0];
	mac_config->sibs_NB_IoT_sched[0].si_tb = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_TB_r13  ;
    //printf("si_tb %d\n",mac_config->sibs_NB_IoT_sched[0].si_tb);


	mac_config->sibs_NB_IoT_sched[1].si_periodicity = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->si_Periodicity_r13 ;
	mac_config->sibs_NB_IoT_sched[1].si_repetition_pattern = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->si_RepetitionPattern_r13;
	mac_config->sibs_NB_IoT_sched[1].sib_mapping_info = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[0][0] | SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[1]->sib_MappingInfo_r13.list.array[1][0];
	mac_config->sibs_NB_IoT_sched[1].si_tb = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_TB_r13;

	mac_config->sibs_NB_IoT_sched[2].si_periodicity = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->si_Periodicity_r13 ;
	mac_config->sibs_NB_IoT_sched[2].si_repetition_pattern = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->si_RepetitionPattern_r13;
	mac_config->sibs_NB_IoT_sched[2].sib_mapping_info = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[0][0] | SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[2]->sib_MappingInfo_r13.list.array[1][0];
	mac_config->sibs_NB_IoT_sched[2].si_tb = SIB.message.choice.c1.choice.systemInformationBlockType1_r13.schedulingInfoList_r13.list.array[0]->si_TB_r13;


	mac_config->sibs_NB_IoT_sched[3].sib_mapping_info = 0x0;
	mac_config->sibs_NB_IoT_sched[4].sib_mapping_info = 0x0;
	mac_config->sibs_NB_IoT_sched[5].sib_mapping_info = 0x0;

	/// testing
	mac_config->mac_NPRACH_ConfigSIB[0].mac_numRepetitionsPerPreambleAttempt_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->numRepetitionsPerPreambleAttempt_r13;
	mac_config->mac_NPRACH_ConfigSIB[1].mac_numRepetitionsPerPreambleAttempt_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->numRepetitionsPerPreambleAttempt_r13;
	mac_config->mac_NPRACH_ConfigSIB[2].mac_numRepetitionsPerPreambleAttempt_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->numRepetitionsPerPreambleAttempt_r13;

	mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_NumRepetitions_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_NumRepetitions_RA_r13 ;
	mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_NumRepetitions_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_NumRepetitions_RA_r13 ;
	mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_NumRepetitions_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_NumRepetitions_RA_r13 ;

	mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_StartSF_CSS_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_StartSF_CSS_RA_r13;
	mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_StartSF_CSS_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_StartSF_CSS_RA_r13;
	mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_StartSF_CSS_RA_NB_IoT = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_StartSF_CSS_RA_r13;

	mac_config->mac_NPRACH_ConfigSIB[0].mac_npdcch_Offset_RA_NB_IoT =  SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[0]->npdcch_Offset_RA_r13;
	mac_config->mac_NPRACH_ConfigSIB[1].mac_npdcch_Offset_RA_NB_IoT =  SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[1]->npdcch_Offset_RA_r13;
	mac_config->mac_NPRACH_ConfigSIB[2].mac_npdcch_Offset_RA_NB_IoT =  SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.array[0]->choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array[2]->npdcch_Offset_RA_r13;
   }

   if( ded_flag!=0 )
   {
    /*
    mac_config->npdcch_ConfigDedicated[0].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;
    mac_config->npdcch_ConfigDedicated[1].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;
    mac_config->npdcch_ConfigDedicated[2].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;

    mac_config->npdcch_ConfigDedicated[0].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    mac_config->npdcch_ConfigDedicated[1].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    mac_config->npdcch_ConfigDedicated[2].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;

    mac_config->npdcch_ConfigDedicated[0].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[0].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    mac_config->npdcch_ConfigDedicated[1].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[1].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    mac_config->npdcch_ConfigDedicated[2].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[2].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    */
    // now we only have 3 UE list USS
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].R_max         =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13;
    
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].G             =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13;
    
    mac_config->npdcch_ConfigDedicated[ue_list_ded_num].a_offset      =DED_Config.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13[ue_list_ded_num].npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13;
    }


	return ;


}