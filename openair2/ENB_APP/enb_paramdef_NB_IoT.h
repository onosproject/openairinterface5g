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

/*! \file openair2/ENB_APP/enb_paramdef_nbiot.h
 * \brief definition of configuration parameters for NB-IoT eNodeB modules
 * \author Raymond KNOPP
 * \date 2019
 * \version 0.1
 * \company EURECOM France
 * \email: raymond.knopp@eurecom.fr
 * \note
 * \warning
 */

#include "common/config/config_paramdesc.h"
#include "RRC_paramsvalues.h"
#include <libconfig.h>

#define ENB_CONFIG_STRING_NB_IoT_PARAMETERS                                "NBparameters"
//RRC parameters in the config file of merge branch
#define ENB_CONFIG_STRING_RACH_POWERRAMPINGSTEP_NB_IOT                     "rach_powerRampingStep_NB"
#define ENB_CONFIG_STRING_RACH_PREAMBLEINITIALRECEIVEDTARGETPOWER_NB_IOT   "rach_preambleInitialReceivedTargetPower_NB"
#define ENB_CONFIG_STRING_RACH_PREAMBLETRANSMAX_CE_NB_IOT                  "rach_preambleTransMax_CE_NB"
#define ENB_CONFIG_STRING_RACH_RARESPONSEWINDOWSIZE_NB_IOT                 "rach_raResponseWindowSize_NB"
#define ENB_CONFIG_STRING_RACH_MACCONTENTIONRESOLUTIONTIMER_NB_IOT         "rach_macContentionResolutionTimer_NB"

#define ENB_CONFIG_STRING_BCCH_MODIFICATIONPERIODCOEFF_NB_IOT              "bcch_modificationPeriodCoeff_NB"
#define ENB_CONFIG_STRING_PCCH_DEFAULT_PAGING_CYCLE_NB_IOT                 "pcch_defaultPagingCycle_NB"
#define ENB_CONFIG_STRING_NPRACH_CP_LENGTH_NB_IOT                          "nprach_CP_Length"
#define ENB_CONFIG_STRING_NPRACH_RSRP_RANGE_NB_IOT                         "nprach_rsrp_range"
#define ENB_CONFIG_STRING_NPDSCH_NRS_POWER_NB_IOT                          "npdsch_nrs_Power"
#define ENB_CONFIG_STRING_NPUSCH_ACK_NACK_NUMREPETITIONS_NB_IOT            "npusch_ack_nack_numRepetitions_NB"
#define ENB_CONFIG_STRING_NPUSCH_SRS_SUBFRAMECONFIG_NB_IOT                 "npusch_srs_SubframeConfig_NB"
#define ENB_CONFIG_STRING_NPUSCH_THREETONE_CYCLICSHIFT_R13_NB_IOT          "npusch_threeTone_CyclicShift_r13"
#define ENB_CONFIG_STRING_NPUSCH_SIXTONE_CYCLICSHIFT_R13_NB_IOT            "npusch_sixTone_CyclicShift_r13"
#define ENB_CONFIG_STRING_NPUSCH_GROUP_HOPPING_EN_NB_IOT                   "npusch_groupHoppingEnabled"
#define ENB_CONFIG_STRING_NPUSCH_GROUPASSIGNMENTNPUSH_R13_NB_IOT           "npusch_groupAssignmentNPUSCH_r13"
#define ENB_CONFIG_STRING_DL_GAPTHRESHOLD_NB_IOT                           "dl_GapThreshold_NB"
#define ENB_CONFIG_STRING_DL_GAPPERIODICITY_NB_IOT                         "dl_GapPeriodicity_NB"
#define ENB_CONFIG_STRING_DL_GAPDURATIONCOEFF_NB_IOT                       "dl_GapDurationCoeff_NB"
#define ENB_CONFIG_STRING_NPUSCH_P0NOMINALPUSH_NB_IOT                      "npusch_p0_NominalNPUSCH"
#define ENB_CONFIG_STRING_NPUSCH_ALPHA_NB_IOT                              "npusch_alpha"
#define ENB_CONFIG_STRING_DELTAPREAMBLEMSG3_NB_IOT                         "deltaPreambleMsg3"

#define ENB_CONFIG_STRING_UETIMERS_T300_NB_IOT                             "ue_TimersAndConstants_t300_NB"
#define ENB_CONFIG_STRING_UETIMERS_T301_NB_IOT                             "ue_TimersAndConstants_t301_NB"
#define ENB_CONFIG_STRING_UETIMERS_T310_NB_IOT                             "ue_TimersAndConstants_t310_NB"
#define ENB_CONFIG_STRING_UETIMERS_T311_NB_IOT                             "ue_TimersAndConstants_t311_NB"
#define ENB_CONFIG_STRING_UETIMERS_N310_NB_IOT                             "ue_TimersAndConstants_n310_NB"
#define ENB_CONFIG_STRING_UETIMERS_N311_NB_IOT                             "ue_TimersAndConstants_n311_NB"
// #define ENB_CONFIG_STRING_UE_TRANSMISSION_MODE_NB_IoT                   "ue_TransmissionMode_NB"

// NPRACH parameters 
#define ENB_CONFIG_STRING_NPRACH_PERIODICITY_NB_IOT                        "nprach_Periodicity"
#define ENB_CONFIG_STRING_NPRACH_STARTTIME_NB_IOT                          "nprach_StartTime"
#define ENB_CONFIG_STRING_NPRACH_SUBCARRIEROFFSET_NB_IOT                   "nprach_SubcarrierOffset"
#define ENB_CONFIG_STRING_NPRACH_NUMSUBCARRIERS_NB_IOT                     "nprach_NumSubcarriers"
#define ENB_CONFIG_STRING_NPRACH_SUBCARRIERMSG3_RANGESTART_NB_IOT          "nprach_SubcarrierMSG3_RangeStart"
#define ENB_CONFIG_STRING_MAXNUM_PREAMBLE_ATTEMPT_CE_NB_IOT                "maxNumPreambleAttemptCE_NB"
#define ENB_CONFIG_STRING_NUMREPETITIONSPERPREAMBLEATTEMPT_NB_IOT          "numRepetitionsPerPreambleAttempt"
#define ENB_CONFIG_STRING_NPDCCH_NUMREPETITIONS_RA_NB_IOT                  "npdcch_NumRepetitions_RA"
#define ENB_CONFIG_STRING_NPDCCH_STARTSF_CSS_RA_NB_IOT                     "npdcch_StartSF_CSS_RA"
#define ENB_CONFIG_STRING_NPDCCH_OFFSET_RA_NB_IOT                          "npdcch_Offset_RA"


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                     component carriers configuration parameters                                                                                                     */
/*   optname                                                   helpstr   paramflags    XXXptr                                        defXXXval                    type         numelt  */
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/* init for checkedparam_t structure */

typedef struct ccparams_NB_IoT_s {
	  int32_t 			NB_IoT_configured;
	  //RRC parameters in the config file of merge branch
	  libconfig_int     rach_raResponseWindowSize_NB;                 
  	libconfig_int     rach_macContentionResolutionTimer_NB;       
  	libconfig_int     rach_powerRampingStep_NB;                     
  	libconfig_int     rach_preambleInitialReceivedTargetPower_NB;   
  	libconfig_int     rach_preambleTransMax_CE_NB;                  
  	libconfig_int     bcch_modificationPeriodCoeff_NB;              
  	libconfig_int     pcch_defaultPagingCycle_NB;                   
  	libconfig_int     nprach_CP_Length;                             
  	libconfig_int     nprach_rsrp_range;                            
  	libconfig_int     npdsch_nrs_Power;                             
  	libconfig_int     npusch_ack_nack_numRepetitions_NB;            
  	libconfig_int     npusch_srs_SubframeConfig_NB;                 
  	libconfig_int     npusch_threeTone_CyclicShift_r13;             
  	libconfig_int     npusch_sixTone_CyclicShift_r13;            
  	const char*       npusch_groupHoppingEnabled;
  	libconfig_int     npusch_groupAssignmentNPUSCH_r13;             
  	libconfig_int     dl_GapThreshold_NB;                           
  	libconfig_int     dl_GapPeriodicity_NB;                         
  	const char*       dl_GapDurationCoeff_NB;
  	libconfig_int     npusch_p0_NominalNPUSCH;                      
  	const char*       npusch_alpha;
  	libconfig_int     deltaPreambleMsg3;                            

  	libconfig_int     ue_TimersAndConstants_t300_NB;     
  	libconfig_int     ue_TimersAndConstants_t301_NB;     
  	libconfig_int     ue_TimersAndConstants_t310_NB;      
  	libconfig_int     ue_TimersAndConstants_t311_NB;      
  	libconfig_int     ue_TimersAndConstants_n310_NB;      
  	libconfig_int     ue_TimersAndConstants_n311_NB;      

  	libconfig_int     nprach_Periodicity;                 
  	libconfig_int     nprach_StartTime;                   
  	libconfig_int     nprach_SubcarrierOffset;            
  	libconfig_int     nprach_NumSubcarriers;              
  	const char*       nprach_SubcarrierMSG3_RangeStart;
  	libconfig_int     maxNumPreambleAttemptCE_NB;         
  	libconfig_int     numRepetitionsPerPreambleAttempt;   
  	libconfig_int     npdcch_NumRepetitions_RA;           
  	libconfig_int     npdcch_StartSF_CSS_RA;              
  	const char*       npdcch_Offset_RA;
} ccparams_NB_IoT_t;


#define CCPARAMS_NB_IOT_DESC(NBconfig) {				\
{"NB_IoT_configured",                                            		  NULL, 0,      iptr:&NBconfig->NB_IoT_configured,                 			     defintval:0,				      TYPE_UINT,    0},  \
{ENB_CONFIG_STRING_RACH_POWERRAMPINGSTEP_NB_IOT,                 		  NULL, 0,      iptr:&NBconfig->rach_powerRampingStep_NB,          			     defintval:0,				      TYPE_UINT,    0},  \
{ENB_CONFIG_STRING_RACH_PREAMBLEINITIALRECEIVEDTARGETPOWER_NB_IOT	    NULL,	0,			iptr:&NBconfig->rach_preambleInitialReceivedTargetPower_NB 	 defintval:0,				      TYPE_UINT,		0},  \
{ENB_CONFIG_STRING_RACH_PREAMBLETRANSMAX_CE_NB_IOT						        NULL,	0,			iptr:&NBconfig->rach_preambleTransMax_CE_NB 				         defintval:0,				      TYPE_UINT,		0},  \
{ENB_CONFIG_STRING_RACH_RARESPONSEWINDOWSIZE_NB_IOT						        NULL,	0,			iptr:&NBconfig->rach_raResponseWindowSize_NB 				         defintval:0,				      TYPE_UINT,		0},  \
{ENB_CONFIG_STRING_RACH_MACCONTENTIONRESOLUTIONTIMER_NB_IOT				    NULL,	0,			iptr:&NBconfig->rach_macContentionResolutionTimer_NB 		     defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_BCCH_MODIFICATIONPERIODCOEFF_NB_IOT					      NULL,	0,			iptr:&NBconfig->bcch_modificationPeriodCoeff_NB 			       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_PCCH_DEFAULT_PAGING_CYCLE_NB_IOT						        NULL,	0,			iptr:&NBconfig->pcch_defaultPagingCycle_NB 					         defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_CP_LENGTH_NB_IOT    							          NULL,	0,			iptr:&NBconfig->nprach_CP_Length 							               defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_RSRP_RANGE_NB_IOT               				    NULL,	0,			iptr:&NBconfig->nprach_rsrp_range 							             defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPDSCH_NRS_POWER_NB_IOT                				    NULL,	0,			iptr:&NBconfig->npdsch_nrs_Power 							               defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_ACK_NACK_NUMREPETITIONS_NB_IOT  				    NULL,	0,			iptr:&NBconfig->npusch_ack_nack_numRepetitions_NB 			     defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_SRS_SUBFRAMECONFIG_NB_IOT       				    NULL,	0,			iptr:&NBconfig->npusch_srs_SubframeConfig_NB 				         defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_THREETONE_CYCLICSHIFT_R13_NB_IOT				    NULL,	0,			iptr:&NBconfig->npusch_threeTone_CyclicShift_r13 			       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_SIXTONE_CYCLICSHIFT_R13_NB_IOT  				    NULL,	0,			iptr:&NBconfig->npusch_sixTone_CyclicShift_r13 		 		       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_GROUP_HOPPING_EN_NB_IOT         				    NULL,	0,			strptr:&NBconfig->npusch_groupHoppingEnabled 				         defintval:"DISABLE",		  TYPE_STRING,	0},	 \
{ENB_CONFIG_STRING_NPUSCH_GROUPASSIGNMENTNPUSH_R13_NB_IOT 				    NULL,	0,			iptr:&NBconfig->npusch_groupAssignmentNPUSCH_r13 			       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_DL_GAPTHRESHOLD_NB_IOT                 				    NULL,	0,			iptr:&NBconfig->dl_GapThreshold_NB 							             defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_DL_GAPPERIODICITY_NB_IOT               				    NULL,	0,			iptr:&NBconfig->dl_GapPeriodicity_NB 						             defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_DL_GAPDURATIONCOEFF_NB_IOT             				    NULL,	0,			strptr:&NBconfig->dl_GapDurationCoeff_NB 					           defintval:"oneEighth",		TYPE_STRING,	0},	 \
{ENB_CONFIG_STRING_NPUSCH_P0NOMINALPUSH_NB_IOT            				    NULL,	0,			iptr:&NBconfig->npusch_p0_NominalNPUSCH 					           defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPUSCH_ALPHA_NB_IOT                    				    NULL,	0,			strptr:&NBconfig->npusch_alpha 								               defintval:"AL0",		      TYPE_STRING,	0},	 \
{ENB_CONFIG_STRING_DELTAPREAMBLEMSG3_NB_IOT               				    NULL,	0,			iptr:&NBconfig->deltaPreambleMsg3 							             defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_T300_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_t300_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_T301_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_t301_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_T310_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_t310_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_T311_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_t311_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_N310_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_n310_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_UETIMERS_N311_NB_IOT                   				    NULL,	0,			iptr:&NBconfig->ue_TimersAndConstants_n311_NB 				       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_PERIODICITY_NB_IOT              				    NULL,	0,			iptr:&NBconfig->nprach_Periodicity 							             defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_STARTTIME_NB_IOT                				    NULL,	0,			iptr:&NBconfig->nprach_StartTime 							               defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_SUBCARRIEROFFSET_NB_IOT         				    NULL,	0,			iptr:&NBconfig->nprach_SubcarrierOffset 					           defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_NUMSUBCARRIERS_NB_IOT           				    NULL,	0,			iptr:&NBconfig->nprach_NumSubcarriers 						           defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPRACH_SUBCARRIERMSG3_RANGESTART_NB_IOT				    NULL,	0,			strptr:&NBconfig->nprach_SubcarrierMSG3_RangeStart 			     defintval:"one",		      TYPE_STRING,	0},	 \
{ENB_CONFIG_STRING_MAXNUM_PREAMBLE_ATTEMPT_CE_NB_IOT      				    NULL,	0,			iptr:&NBconfig->maxNumPreambleAttemptCE_NB 					         defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NUMREPETITIONSPERPREAMBLEATTEMPT_NB_IOT				    NULL,	0,			iptr:&NBconfig->numRepetitionsPerPreambleAttempt 			       defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPDCCH_NUMREPETITIONS_RA_NB_IOT        				    NULL,	0,			iptr:&NBconfig->npdcch_NumRepetitions_RA 					           defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPDCCH_STARTSF_CSS_RA_NB_IOT           				    NULL,	0,			iptr:&NBconfig->npdcch_StartSF_CSS_RA 						           defintval:0,				      TYPE_UINT,		0},	 \
{ENB_CONFIG_STRING_NPDCCH_OFFSET_RA_NB_IOT                				    NULL,	0,			strptr:&NBconfig->npdcch_Offset_RA 							             defintval:"zero",	      TYPE_STRING,	0},	 \
}		         			 