/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file asn1_msg.c
* \brief primitives to build the asn1 messages
* \author Raymond Knopp, Navid Nikaein and Michele Paffetti
* \date 2011, 2017
* \version 1.0
* \company Eurecom
* \email: raymond.knopp@eurecom.fr, navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
*/

#ifdef USER_MODE
#include <stdio.h>
#include <sys/types.h>
#include <stdlib.h> /* for atoi(3) */
#include <unistd.h> /* for getopt(3) */
#include <string.h> /* for strerror(3) */
#include <sysexits.h> /* for EX_* exit codes */
#include <errno.h>  /* for errno */
#else
#include <linux/module.h>  /* Needed by all modules */
#endif
#ifdef USER_MODE
//#include "RRC/LITE/defs.h"
//#include "COMMON/mac_rrc_primitives.h"
#include "UTIL/LOG/log.h"
#endif
#include <asn_application.h>
#include <asn_internal.h> /* for _ASN_DEFAULT_STACK_MAX */
#include <per_encoder.h>

#include "assertions.h"
#include "RRCConnectionRequest.h"
#include "UL-CCCH-Message.h"
#include "UL-DCCH-Message.h"
#include "DL-CCCH-Message.h"
#include "DL-DCCH-Message.h"
#include "EstablishmentCause.h"
#include "RRCConnectionSetup.h"
#include "SRB-ToAddModList.h"
#include "DRB-ToAddModList.h"
#if defined(Rel10) || defined(Rel14)
#include "MCCH-Message.h"
//#define MRB1 1
#endif

#include "RRC/LITE/defs.h"
#include "RRCConnectionSetupComplete.h"
#include "RRCConnectionReconfigurationComplete.h"
#include "RRCConnectionReconfiguration.h"
#include "MasterInformationBlock.h"
#include "SystemInformation.h"

#include "SystemInformationBlockType1.h"

#include "SIB-Type.h"

#include "BCCH-DL-SCH-Message.h"

#include "PHY/defs.h"

#include "MeasObjectToAddModList.h"
#include "ReportConfigToAddModList.h"
#include "MeasIdToAddModList.h"
#include "enb_config.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

//#include "PHY/defs.h"
#ifndef USER_MODE
#define msg printk
#ifndef errno
int errno;
#endif
#else
# if !defined (msg)
#   define msg printf
# endif
#endif

//#include for NB-IoT-------------------
#include "RRCConnectionRequest-NB.h"
#include "BCCH-DL-SCH-Message-NB.h"
#include "UL-CCCH-Message-NB.h"
#include "UL-DCCH-Message-NB.h"
#include "DL-CCCH-Message-NB.h"
#include "DL-DCCH-Message-NB.h"
#include "EstablishmentCause-NB-r13.h"
#include "RRCConnectionSetup-NB.h"
#include "SRB-ToAddModList-NB-r13.h"
#include "DRB-ToAddModList-NB-r13.h"
#include "RRC/LITE/defs_nb_iot.h"
#include "RRCConnectionSetupComplete-NB.h"
#include "RRCConnectionReconfigurationComplete-NB.h"
#include "RRCConnectionReconfiguration-NB.h"
#include "MasterInformationBlock-NB.h"
#include "SystemInformation-NB.h"
#include "SystemInformationBlockType1.h"
#include "SIB-Type-NB-r13.h"
#include "RRCConnectionResume-NB.h"
#include "RRCConnectionReestablishment-NB.h"
//----------------------------------------

//Not touched
//#define XER_PRINT
extern Enb_properties_array_t enb_properties;
typedef struct xer_sprint_string_s {
  char *string;
  size_t string_size;
  size_t string_index;
} xer_sprint_string_t;

extern unsigned char NB_eNB_INST;
extern uint8_t usim_test;

uint16_t two_tier_hexagonal_cellIds[7] = {0,1,2,4,5,7,8};
uint16_t two_tier_hexagonal_adjacent_cellIds[7][6] = {{1,2,4,5,7,8},    // CellId 0
  {11,18,2,0,8,15}, // CellId 1
  {18,13,3,4,0,1},  // CellId 2
  {2,3,14,6,5,0},   // CellId 4
  {0,4,6,16,9,7},   // CellId 5
  {8,0,5,9,17,12},  // CellId 7
  {15,1,0,7,12,10}
};// CellId 8

/*
 * This is a helper function for xer_sprint, which directs all incoming data
 * into the provided string.
 */
static int xer__print2s (const void *buffer, size_t size, void *app_key)
{
  xer_sprint_string_t *string_buffer = (xer_sprint_string_t *) app_key;
  size_t string_remaining = string_buffer->string_size - string_buffer->string_index;

  if (string_remaining > 0) {
    if (size > string_remaining) {
      size = string_remaining;
    }

    memcpy(&string_buffer->string[string_buffer->string_index], buffer, size);
    string_buffer->string_index += size;
  }

  return 0;
}

int xer_sprint (char *string, size_t string_size, asn_TYPE_descriptor_t *td, void *sptr)
{
  asn_enc_rval_t er;
  xer_sprint_string_t string_buffer;

  string_buffer.string = string;
  string_buffer.string_size = string_size;
  string_buffer.string_index = 0;

  er = xer_encode(td, sptr, XER_F_BASIC, xer__print2s, &string_buffer);

  if (er.encoded < 0) {
    LOG_E(RRC, "xer_sprint encoding error (%d)!", er.encoded);
    er.encoded = string_buffer.string_size;
  } else {
    if (er.encoded > string_buffer.string_size) {
      LOG_E(RRC, "xer_sprint string buffer too small, got %d need %d!", string_buffer.string_size, er.encoded);
      er.encoded = string_buffer.string_size;
    }
  }

  return er.encoded;
}

uint16_t get_adjacent_cell_id(uint8_t Mod_id,uint8_t index)
{
  return(two_tier_hexagonal_adjacent_cellIds[Mod_id][index]);
}
/* This only works for the hexagonal topology...need a more general function for other topologies */

uint8_t get_adjacent_cell_mod_id(uint16_t phyCellId)
{
  uint8_t i;

  for(i=0; i<7; i++) {
    if(two_tier_hexagonal_cellIds[i] == phyCellId) {
      return i;
    }
  }

  LOG_E(RRC,"\nCannot get adjacent cell mod id! Fatal error!\n");
  return 0xFF; //error!
}

/*do_MIB_NB*/
uint8_t do_MIB_NB(
		rrc_eNB_carrier_data_t *carrier,
		uint32_t N_RB_DL,
		uint32_t frame)
{
  asn_enc_rval_t enc_rval;
  BCCH_BCH_Message_NB_t *mib_NB = &(carrier->mib_NB); //punta all'indirizzo di mib_NB

  //should be passed as a parameter? (how decide which value is included in carrier?
  uint8_t sfn_MSB = (uint8_t)((frame>>2)&0xff); //????? //4 bits
  uint8_t hsfn_LSB = (uint8_t)((frame>>2)&0xff); //?? //2 bits
  uint16_t spare=0; //11 bits --> use uint16

  //no DL_Bandwidth, no PCHIC

  mib_NB->message.systemFrameNumber_MSB_r13.buf = &sfn_MSB;
  mib_NB->message.systemFrameNumber_MSB_r13.size = 1; //if expressed in byte
  mib_NB->message.systemFrameNumber_MSB_r13.bits_unused = 4;

  mib_NB->message.hyperSFN_LSB_r13.buf= &hsfn_LSB;
  mib_NB->message.hyperSFN_LSB_r13.size= 1;
  mib_NB->message.hyperSFN_LSB_r13.bits_unused = 6;

  mib_NB->message.spare.buf = (uint8_t *)&spare; //left the pointer to type uint8?
  mib_NB->message.spare.size = 2;
  mib_NB->message.spare.bits_unused = 5;

  //decide how to set it
  mib_NB->message.schedulingInfoSIB1_r13 =0; //see TS 36.213-->tables 16.4.1.3-3 ecc...
  mib_NB->message.systemInfoValueTag_r13= 0;
  mib_NB->message.ab_Enabled_r13 = 0;

  //to be decided
  mib_NB->message.operationModeInfo_r13.present = MasterInformationBlock_NB__operationModeInfo_r13_PR_inband_SamePCI_r13;
  mib_NB->message.operationModeInfo_r13.choice.inband_SamePCI_r13.eutra_CRS_SequenceInfo_r13 = 0;


  printf("[MIB] something to write HERE ,sfn_MSB %x, hsfn_LSB %x\n",
         (uint32_t)sfn_MSB,
		 (uint32_t)hsfn_LSB);

  //only changes in "asn_DEF_BCCH_BCH_Message_NB"
  enc_rval = uper_encode_to_buffer(&asn_DEF_BCCH_BCH_Message_NB,
                                   (void*)mib_NB,
                                   carrier->MIB_NB, //non credo ci vada &(carrier->MIB_NB)
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

  if (enc_rval.encoded==-1) {
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
  /*
  printf("MIB: %x ((MIB>>10)&63)+(MIB&3<<6)=SFN %x, MIB>>2&3 = phich_resource %d, MIB>>4&1 = phich_duration %d, MIB>>5&7 = system_bandwidth %d)\n",*(uint32_t *)buffer,
   (((*(uint32_t *)buffer)>>10)&0x3f)+(((*(uint32_t *)buffer)&3)<<6),
   ((*(uint32_t *)buffer)>>2)&0x3,
   ((*(uint32_t *)buffer)>>4)&0x1,
   ((*(uint32_t *)buffer)>>5)&0x7
   );
  */
}

/*do_SIB1_NB*/
uint8_t do_SIB1_NB(uint8_t Mod_id, int CC_id,
				rrc_eNB_carrier_data_t *carrier,
                RrcConfigurationReq *configuration
               )
{
  BCCH_DL_SCH_Message_NB_t *bcch_message= &(carrier->siblock1_NB);
  SystemInformationBlockType1_NB_t *sib1_NB;

  asn_enc_rval_t enc_rval;

  PLMN_IdentityInfo_NB_r13_t PLMN_identity_info_NB;
  MCC_MNC_Digit_t dummy_mcc[3],dummy_mnc[3];
  SchedulingInfo_NB_r13_t schedulingInfo_NB;
  SIB_Type_NB_r13_t sib_type_NB;

  //New parameters
  //uint8_t hyperSFN_MSB_r13 ?? (BITSTRING)

  long* attachWithoutPDN_Connectivity = NULL;
  attachWithoutPDN_Connectivity = CALLOC(1,sizeof(long));
  long *nrs_CRS_PowerOffset=NULL;
  nrs_CRS_PowerOffset = CALLOC(1, sizeof(long));
  long *eutraControlRegionSize=NULL;
   eutraControlRegionSize = CALLOC(1,sizeof(long));
  long systemInfoValueTagSI = 0;

  memset(bcch_message,0,sizeof(BCCH_DL_SCH_Message_NB_t));
  bcch_message->message.present = BCCH_DL_SCH_MessageType_NB_PR_c1;
  bcch_message->message.choice.c1.present = BCCH_DL_SCH_MessageType_NB__c1_PR_systemInformationBlockType1_r13;

  //allocation
  carrier->sib1_NB = &bcch_message->message.choice.c1.choice.systemInformationBlockType1_r13;
  sib1_NB = carrier->sib1_NB;

  memset(&PLMN_identity_info_NB,0,sizeof(PLMN_IdentityInfo_NB_r13_t));
  memset(&schedulingInfo_NB,0,sizeof(SchedulingInfo_NB_r13_t));
  memset(&sib_type_NB,0,sizeof(SIB_Type_NB_r13_t));


  PLMN_identity_info_NB.plmn_Identity_r13.mcc = CALLOC(1,sizeof(*PLMN_identity_info_NB.plmn_Identity_r13.mcc));
  memset(PLMN_identity_info_NB.plmn_Identity_r13.mcc,0,sizeof(*PLMN_identity_info_NB.plmn_Identity_r13.mcc));

  asn_set_empty(&PLMN_identity_info_NB.plmn_Identity_r13.mcc->list);//.size=0;

  //left as it is???
#if defined(ENABLE_ITTI)
  dummy_mcc[0] = (configuration->mcc / 100) % 10;
  dummy_mcc[1] = (configuration->mcc / 10) % 10;
  dummy_mcc[2] = (configuration->mcc / 1) % 10;
#else
  dummy_mcc[0] = 0;
  dummy_mcc[1] = 0;
  dummy_mcc[2] = 1;
#endif
  ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mcc->list,&dummy_mcc[0]);
  ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mcc->list,&dummy_mcc[1]);
  ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mcc->list,&dummy_mcc[2]);

  PLMN_identity_info_NB.plmn_Identity_r13.mnc.list.size=0;
  PLMN_identity_info_NB.plmn_Identity_r13.mnc.list.count=0;

  //left as it is
#if defined(ENABLE_ITTI)

  if (configuration->mnc >= 100) {
    dummy_mnc[0] = (configuration->mnc / 100) % 10;
    dummy_mnc[1] = (configuration->mnc / 10) % 10;
    dummy_mnc[2] = (configuration->mnc / 1) % 10;
  } else {
    if (configuration->mnc_digit_length == 2) {
      dummy_mnc[0] = (configuration->mnc / 10) % 10;
      dummy_mnc[1] = (configuration->mnc / 1) % 10;
      dummy_mnc[2] = 0xf;
    } else {
      dummy_mnc[0] = (configuration->mnc / 100) % 100;
      dummy_mnc[1] = (configuration->mnc / 10) % 10;
      dummy_mnc[2] = (configuration->mnc / 1) % 10;
    }
  }

#else
  dummy_mnc[0] = 0;
  dummy_mnc[1] = 1;
  dummy_mnc[2] = 0xf;
#endif
  ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mnc.list,&dummy_mnc[0]);
  ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mnc.list,&dummy_mnc[1]);

  if (dummy_mnc[2] != 0xf) {
    ASN_SEQUENCE_ADD(&PLMN_identity_info_NB.plmn_Identity_r13.mnc.list,&dummy_mnc[2]);
  }

  //still set to "notReserved" as in the previous case
  PLMN_identity_info_NB.cellReservedForOperatorUse_r13=PLMN_IdentityInfo_NB_r13__cellReservedForOperatorUse_r13_notReserved;

  *attachWithoutPDN_Connectivity = 0;
  PLMN_identity_info_NB.attachWithoutPDN_Connectivity_r13 = attachWithoutPDN_Connectivity;

  ASN_SEQUENCE_ADD(&sib1_NB->cellAccessRelatedInfo_r13.plmn_IdentityList_r13.list,&PLMN_identity_info_NB);

  // 16 bits = 2 byte
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.buf = MALLOC(2); //MALLOC works in byte

  //lefts as it is?
#if defined(ENABLE_ITTI)
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.buf[0] = (configuration->tac >> 8) & 0xff;
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.buf[1] = (configuration->tac >> 0) & 0xff;
#else
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.buf[0] = 0x00;
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.buf[1] = 0x01;
#endif
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.size=2;
  sib1_NB->cellAccessRelatedInfo_r13.trackingAreaCode_r13.bits_unused=0;

  // 28 bits --> i have to use 32 bits = 4 byte
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf = MALLOC(8); // why allocate 8 byte?
#if defined(ENABLE_ITTI)
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[0] = (configuration->cell_identity >> 20) & 0xff;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[1] = (configuration->cell_identity >> 12) & 0xff;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[2] = (configuration->cell_identity >>  4) & 0xff;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[3] = (configuration->cell_identity <<  4) & 0xf0;
#else
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[0] = 0x00;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[1] = 0x00;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[2] = 0x00;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.buf[3] = 0x10;
#endif
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.size=4;
  sib1_NB->cellAccessRelatedInfo_r13.cellIdentity_r13.bits_unused=4;

  //Still set to "notBarred" as in the previous case
  sib1_NB->cellAccessRelatedInfo_r13.cellBarred_r13=SystemInformationBlockType1_NB__cellAccessRelatedInfo_r13__cellBarred_r13_notBarred;

  //Still Set to "notAllowed" like in the previous case
  sib1_NB->cellAccessRelatedInfo_r13.intraFreqReselection_r13=SystemInformationBlockType1_NB__cellAccessRelatedInfo_r13__intraFreqReselection_r13_notAllowed;

  //ClosedSubscriberGroup(CSG) is not supported by NB-IoT

  sib1_NB->cellSelectionInfo_r13.q_RxLevMin_r13=-65; //which value?? TS 36.331 V14.2.1 pag. 589
  sib1_NB->cellSelectionInfo_r13.q_QualMin_r13 = 0; // FIXME new parameter for SIB1-NB, not present in SIB1

  sib1_NB->p_Max_r13 = CALLOC(1, sizeof(P_Max_t));
  *(sib1_NB->p_Max_r13) = 23;

  sib1_NB->freqBandIndicator_r13 =
#if defined(ENABLE_ITTI)
    configuration->eutra_band[CC_id];
#else
    7; // UL:2500 MHz–2570 MHz	DL:2620 MHz	–2690 MHz	mode:FDD
       //FIXME For NB-IoT depends on the operation mode (in/out/guard band) and also, not all PRBs are allowed ?
#endif

    //OPTIONAL new parameters, to be used?
      /*
       * freqBandInfo_r13
       * multiBandInfoList_r13
       * nrs_CRS_PowerOffset_r13
       */

   sib1_NB->downlinkBitmap_r13.present= DL_Bitmap_NB_r13_PR_subframePattern10_r13;
   //sib1_NB->downlinkBitmap_r13.choice.subframePattern10_r13 =(is a BIT_STRING)


   *eutraControlRegionSize = 0;
   sib1_NB->eutraControlRegionSize_r13 = eutraControlRegionSize; //ok


   *nrs_CRS_PowerOffset= 0;
   sib1_NB->nrs_CRS_PowerOffset_r13 = nrs_CRS_PowerOffset;


  //FIXME which value to set?
  schedulingInfo_NB.si_Periodicity_r13=SchedulingInfo_NB_r13__si_Periodicity_r13_rf64;
  schedulingInfo_NB.si_RepetitionPattern_r13=SchedulingInfo_NB_r13__si_RepetitionPattern_r13_every2ndRF; //This Indicates the starting radio frames within the SI window used for SI message transmission.
  schedulingInfo_NB.si_TB_r13= SchedulingInfo_NB_r13__si_TB_r13_b56; //in 2 subframe = 2ms (pag 590 TS 36.331)

  // This is for SIB2/3
  /*SIB3 --> There is no mapping information of SIB2 since it is always present
    *  in the first SystemInformation message
    * listed in the schedulingInfoList list.
    * */
  //gli stiamo dicendo che ci sarà solo un SIB3 oltre che al due
  sib_type_NB=SIB_Type_NB_r13_sibType3_NB_r13;

  ASN_SEQUENCE_ADD(&schedulingInfo_NB.sib_MappingInfo_r13.list,&sib_type_NB);
  ASN_SEQUENCE_ADD(&sib1_NB->schedulingInfoList_r13.list,&schedulingInfo_NB);

#if defined(ENABLE_ITTI)

  if (configuration->frame_type[CC_id] == TDD)
#endif
  {
	//FIXME in NB-IoT mandatory to be FDD --> so must give an error
	  LOG_E(RRC,"[eNB %d] Frame Type is TDD --> not supported by NB-IoT, exiting\n", Mod_id); //correct?
	  exit(-1);
  }

  //FIXME which value chose for the following parameter?
  sib1_NB->si_WindowLength_r13=SystemInformationBlockType1_NB__si_WindowLength_r13_ms160;
  sib1_NB->si_RadioFrameOffset_r13= 0;

  /*In Nb-IoT change/update of specific SI message can additionally be indicated by a SI message specific value tag
   * systemInfoValueTagSI (there is no SystemInfoValueTag in SIB1-NB but only in MIB-NB)
   *contained in systemInfoValueTagList_r13
   **/
  asn_set_empty(&sib1_NB->systemInfoValueTagList_r13->list);   //FIXME good inizialization?
  ASN_SEQUENCE_ADD(&sib1_NB->systemInfoValueTagList_r13->list,&systemInfoValueTagSI);


  //only change "asn_DEF_BCCH_DL_SCH_Message" in "asn_DEF_BCCH_DL_SCH_Message_NB"
#ifdef XER_PRINT //generate xml files
  xer_fprint(stdout, &asn_DEF_BCCH_DL_SCH_Message_NB, (void*)bcch_message);
#endif


  enc_rval = uper_encode_to_buffer(&asn_DEF_BCCH_DL_SCH_Message_NB,
                                   (void*)bcch_message,
                                   carrier->SIB1_NB,
                                   100);

  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);


#ifdef USER_MODE
  LOG_D(RRC,"[eNB] SystemInformationBlockType1-NB Encoded %d bits (%d bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);
#endif

  if (enc_rval.encoded==-1) {
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}

/*SIB23_NB*/
//to be clarified is it is possible to carry SIB2 and SIB3  in the same SI message for NB-IoT?
uint8_t do_SIB23_NB(uint8_t Mod_id,
                 int CC_id,
                 rrc_eNB_carrier_data_t *carrier,
                 RrcConfigurationReq *configuration )
{
  struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member *sib2_NB_part;
  struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member *sib3_NB_part;

  BCCH_DL_SCH_Message_NB_t *bcch_message = &(carrier->systemInformation_NB); //is the systeminformation-->BCCH_DL_SCH_Message_NB
  SystemInformationBlockType2_NB_r13_t *sib2_NB;
  SystemInformationBlockType3_NB_r13_t *sib3_NB;

  asn_enc_rval_t enc_rval;
  RACH_Info_NB_r13_t rach_Info_NB;
  NPRACH_Parameters_NB_r13_t nprach_parameters;

  //new
  long *connEstFailOffset = NULL;
  connEstFailOffset = CALLOC(1, sizeof(long));

  RSRP_ThresholdsNPRACH_InfoList_NB_r13_t *rsrp_ThresholdsPrachInfoList;
  RSRP_Range_t rsrp_range;
  ACK_NACK_NumRepetitions_NB_r13_t ack_nack_repetition;

  long *srs_SubframeConfig;
  srs_SubframeConfig= CALLOC(1, sizeof(long));


  if (bcch_message) {
    memset(bcch_message,0,sizeof(BCCH_DL_SCH_Message_NB_t));
  } else {
    LOG_E(RRC,"[eNB %d] BCCH_MESSAGE_NB is null, exiting\n", Mod_id);
    exit(-1);
  }

  //signifa che prima deve essere stata allocata memoria per forza
  if (!carrier->sib2_NB) {
    LOG_E(RRC,"[eNB %d] sib2_NB is null, exiting\n", Mod_id);
    exit(-1);
  }

  if (!carrier->sib3_NB) {
    LOG_E(RRC,"[eNB %d] sib3_NB is null, exiting\n", Mod_id);
    exit(-1);
  }


  LOG_I(RRC,"[eNB %d] Configuration SIB2/3\n", Mod_id);

  sib2_NB_part = CALLOC(1,sizeof(struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member));
  sib3_NB_part = CALLOC(1,sizeof(struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member));
  memset(sib2_NB_part,0,sizeof(struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member));
  memset(sib3_NB_part,0,sizeof(struct SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member));

  sib2_NB_part->present = SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member_PR_sib2_r13;
  sib3_NB_part->present = SystemInformation_NB_r13_IEs__sib_TypeAndInfo_r13__Member_PR_sib3_r13;

  //attenzione potrebbe esserci un bug qui--> ricontrollare
  carrier->sib2_NB = &sib2_NB_part->choice.sib2_r13;
  carrier->sib3_NB = &sib3_NB_part->choice.sib3_r13;
  sib2_NB = carrier->sib2_NB;
  sib3_NB = carrier->sib3_NB;


/// SIB2-NB-----------------------------------------

  //Barring is manage by ab-Enabled in MIB-NB (but is not a struct as ac-BarringInfo)

  sib2_NB->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.preambleTransMax_CE_r13 =
   		  configuration->preambleTransMax_CE_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.powerRampingParameters_r13.powerRampingStep =
	configuration->rach_powerRampingStep_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.powerRampingParameters_r13.preambleInitialReceivedTargetPower =
    configuration->rach_preambleInitialReceivedTargetPower_NB[CC_id];

  rach_Info_NB.ra_ResponseWindowSize_r13 = configuration->rach_raResponseWindowSize_NB[CC_id];
  rach_Info_NB.mac_ContentionResolutionTimer_r13 = configuration-> rach_macContentionResolutionTimer_NB[CC_id];
  //initialize this list? how to use it? correct?
  ASN_SEQUENCE_ADD(&sib2_NB->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list,&rach_Info_NB);

  //new parameter
  *connEstFailOffset = 0;
   sib2_NB->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.connEstFailOffset_r13 = connEstFailOffset;


  // BCCH-Config-NB-IoT
  sib2_NB->radioResourceConfigCommon_r13.bcch_Config_r13.modificationPeriodCoeff_r13
    = configuration->bcch_modificationPeriodCoeff_NB[CC_id];

  // PCCH-Config-NB-IoT
  sib2_NB->radioResourceConfigCommon_r13.pcch_Config_r13.defaultPagingCycle_r13
    = configuration->pcch_defaultPagingCycle_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.pcch_Config_r13.nB_r13 = configuration->pcch_nB_NB[CC_id];
  //new
  sib2_NB->radioResourceConfigCommon_r13.pcch_Config_r13.npdcch_NumRepetitionPaging_r13 = configuration-> pcch_npdcch_NumRepetitionPaging_NB[CC_id];

  //NPRACH-Config-NB-IoT
  sib2_NB->radioResourceConfigCommon_r13.nprach_Config_r13.nprach_CP_Length_r13 = configuration->nprach_CP_Length[CC_id];

  //new(provo metodo short)
   sib2_NB->radioResourceConfigCommon_r13.nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13 =
		   CALLOC(1, sizeof(struct RSRP_ThresholdsNPRACH_InfoList_NB_r13)); //fatto uguale dopo
   rsrp_ThresholdsPrachInfoList = sib2_NB->radioResourceConfigCommon_r13.nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13;
   rsrp_range = configuration->nprach_rsrp_range_NB;
   ASN_SEQUENCE_ADD(&rsrp_ThresholdsPrachInfoList->list,rsrp_range);

  nprach_parameters->nprach_Periodicity_r13 = configuration->nprach_Periodicity[CC_id];
  nprach_parameters->nprach_StartTime_r13 = configuration->nprach_StartTime[CC_id];
  nprach_parameters->nprach_SubcarrierOffset_r13 = configuration->nprach_SubcarrierOffset[CC_id];
  nprach_parameters->nprach_NumSubcarriers_r13= configuration->nprach_NumSubcarriers[CC_id];
  nprach_parameters->nprach_SubcarrierMSG3_RangeStart_r13= configuration->nprach_SubcarrierMSG3_RangeStart[CC_id];
  nprach_parameters->maxNumPreambleAttemptCE_r13= configuration->maxNumPreambleAttemptCE_NB[CC_id];
  nprach_parameters->numRepetitionsPerPreambleAttempt_r13 = configuration->numRepetitionsPerPreambleAttempt_NB[CC_id];
  nprach_parameters->npdcch_NumRepetitions_RA_r13 = configuration->npdcch_NumRepetitions_RA[CC_id];
  nprach_parameters->npdcch_StartSF_CSS_RA_r13= configuration->npdcch_StartSF_CSS_RA[CC_id];
  nprach_parameters->npdcch_Offset_RA_r13= configuration->npdcch_Offset_RA[CC_id];
  //Correct?
  ASN_SEQUENCE_ADD(&sib2_NB->radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list,nprach_parameters);

  // NPDSCH-Config NB-IOT
  sib2_NB->radioResourceConfigCommon_r13.npdsch_ConfigCommon_r13.nrs_Power_r13= configuration->npdsch_nrs_Power[CC_id];


  //NPUSCH-Config NB-IoT
  //new
  ack_nack_repetition = configuration-> npusch_ack_nack_numRepetitions_NB[CC_id]; //is an enumerative
  ASN_SEQUENCE_ADD(&sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ack_NACK_NumRepetitions_Msg4_r13.list,ack_nack_repetition);

  *srs_SubframeConfig = configuration->npusch_srs_SubframeConfig_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.srs_SubframeConfig_r13= srs_SubframeConfig;


  //new (occhio che dmrs_config_r13 è un puntatore quando lo richiami)
  sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13 =configuration->npusch_threeTone_CyclicShift_r13[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13 = configuration->npusch_sixTone_CyclicShift_r13[CC_id];

  /*
    * threeTone_BaseSequence_r13
    * sixTone_BaseSequence_r13
    * twelveTone_BaseSequence_r13
    */

  sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13= configuration->npusch_groupHoppingEnabled[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13 =configuration->npusch_groupAssignmentNPUSCH_r13[CC_id];


  // PUCCH-Config NB-IoT
 /*
  * all data are sent over the NPUSCH. This includes also the UL control information (UCI),
  * which is transmitted using a different format. Consequently there is no equivalent to the PUCCH of LTE in NB-IoT.
  */

  //New: DL_GapConfig
  sib2_NB->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapDurationCoeff_r13= configuration-> dl_GapDurationCoeff_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapPeriodicity_r13= configuration->dl_GapPeriodicity_NB[CC_id];
  sib2_NB->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapThreshold_r13= configuration->dl_GapThreshold_NB[CC_id];

  // SRS Config --May not implemented in NB-IoT!

  // uplinkPowerControlCommon - NB-IoT
  sib2_NB->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.p0_NominalNPUSCH_r13 = configuration->npusch_p0_NominalNPUSCH_r13;
  sib2_NB->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.deltaPreambleMsg3_r13 = configuration->deltaPreambleMsg3_r13;
  sib2_NB->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.alpha_r13 = configuration->npusch_alpha_r13;
  //no deltaFlist_PUCCH and no UL cyclic prefix

  // UE Timers and Constants -NB-IoT
  sib2_NB->ue_TimersAndConstants_r13.t300_r13 = configuration-> ue_TimersAndConstants_t300_NB[CC_id];
  sib2_NB->ue_TimersAndConstants_r13.t301_r13 = configuration-> ue_TimersAndConstants_t301_NB[CC_id];
  sib2_NB->ue_TimersAndConstants_r13.t310_r13 = configuration-> ue_TimersAndConstants_t310_NB[CC_id];
  sib2_NB->ue_TimersAndConstants_r13.t311_r13 = configuration-> ue_TimersAndConstants_t311_NB[CC_id];
  sib2_NB->ue_TimersAndConstants_r13.n310_r13 = configuration-> ue_TimersAndConstants_n310_NB[CC_id];
  sib2_NB->ue_TimersAndConstants_r13.n311_r13 = configuration-> ue_TimersAndConstants_n311_NB[CC_id];

 /* static assignment will be not used

  //FIXME all values are almost set randomly

  	//RACH-Config-NB-IoT
  	//no numberOfRA_Preambles
    //no preamblesGroupAConfig..
    //no maxHARQ_Msg3Tx

    (*sib2_NB)->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.preambleTransMax_CE_r13 = PreambleTransMax_n10; //problem
    (*sib2_NB)->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.powerRampingParameters_r13.powerRampingStep =
    		PowerRampingParameters__powerRampingStep_dB2;
    (*sib2_NB)->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.powerRampingParameters_r13.preambleInitialReceivedTargetPower =
    		PowerRampingParameters__preambleInitialReceivedTargetPower_dBm_100;

    //valori a caso
    rach_Info_NB.ra_ResponseWindowSize_r13 = RACH_Info_NB_r13__ra_ResponseWindowSize_r13_pp2; //pp = PDCCH periods (pag 614)
    rach_Info_NB.mac_ContentionResolutionTimer_r13 = RACH_Info_NB_r13__mac_ContentionResolutionTimer_r13_pp1;
    ASN_SEQUENCE_ADD(&(*sib2_NB)->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.rach_InfoList_r13.list,rach_Info_NB);

    *connEstFailOffset= 0;
    (*sib2_NB)->radioResourceConfigCommon_r13.rach_ConfigCommon_r13.connEstFailOffset_r13 = connEstFailOffset;

    // BCCH-Config-NB-IoT
    (*sib2_NB)->radioResourceConfigCommon_r13.bcch_Config_r13.modificationPeriodCoeff_r13= BCCH_Config_NB_r13__modificationPeriodCoeff_r13_n16;

    // PCCH-Config-NB-IoT
    (*sib2_NB)->radioResourceConfigCommon_r13.pcch_Config_r13.defaultPagingCycle_r13 = PCCH_Config_NB_r13__defaultPagingCycle_r13_rf128;
    (*sib2_NB)->radioResourceConfigCommon_r13.pcch_Config_r13.nB_r13 = PCCH_Config_NB_r13__nB_r13_oneT;
    (*sib2_NB)->radioResourceConfigCommon_r13.pcch_Config_r13.npdcch_NumRepetitionPaging_r13 = PCCH_Config_NB_r13__npdcch_NumRepetitionPaging_r13_r1;

    //NPRACH-Config-NB-IoT
    (*sib2_NB)->radioResourceConfigCommon_r13.nprach_Config_r13.nprach_CP_Length_r13 = NPRACH_ConfigSIB_NB_r13__nprach_CP_Length_r13_us66dot7; //66.7 microsec

    (*sib2_NB)->radioResourceConfigCommon_r13.nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13 =
    		CALLOC(1, sizeof(struct RSRP_ThresholdsNPRACH_InfoList_NB_r13));
    rsrp_ThresholdsPrachInfoList = (*sib2_NB)->radioResourceConfigCommon_r13.nprach_Config_r13.rsrp_ThresholdsPrachInfoList_r13;
    rsrp_range = 0;
    ASN_SEQUENCE_ADD(&rsrp_ThresholdsPrachInfoList->list,rsrp_range);

    //totalmente a caso
      nprach_parameters->nprach_Periodicity_r13 = NPRACH_Parameters_NB_r13__nprach_Periodicity_r13_ms40;
      nprach_parameters->nprach_StartTime_r13 = NPRACH_Parameters_NB_r13__nprach_StartTime_r13_ms8;
      nprach_parameters->nprach_SubcarrierOffset_r13 = NPRACH_Parameters_NB_r13__nprach_SubcarrierOffset_r13_n0;
      nprach_parameters->nprach_NumSubcarriers_r13= NPRACH_Parameters_NB_r13__nprach_NumSubcarriers_r13_n12;
      nprach_parameters->nprach_SubcarrierMSG3_RangeStart_r13= NPRACH_Parameters_NB_r13__nprach_SubcarrierMSG3_RangeStart_r13_zero;
      nprach_parameters->maxNumPreambleAttemptCE_r13= NPRACH_Parameters_NB_r13__maxNumPreambleAttemptCE_r13_n3;
      nprach_parameters->numRepetitionsPerPreambleAttempt_r13 = NPRACH_Parameters_NB_r13__numRepetitionsPerPreambleAttempt_r13_n1;
      nprach_parameters->npdcch_NumRepetitions_RA_r13 = NPRACH_Parameters_NB_r13__npdcch_NumRepetitions_RA_r13_r1;
      nprach_parameters->npdcch_StartSF_CSS_RA_r13= NPRACH_Parameters_NB_r13__npdcch_StartSF_CSS_RA_r13_v1dot5;
      nprach_parameters->npdcch_Offset_RA_r13= NPRACH_Parameters_NB_r13__npdcch_Offset_RA_r13_zero;
      ASN_SEQUENCE_ADD(&(*sib2_NB)->radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list,nprach_parameters);

    // NPDSCH-Config NB-IOT
    (*sib2_NB)->radioResourceConfigCommon_r13.npdsch_ConfigCommon_r13.nrs_Power_r13= 0; //?? see TS 36.213 16.2 for the value

    //NPUSCH-Config NB-IoT
    ack_nack_repetition = ACK_NACK_NumRepetitions_NB_r13_r1; //is an enumerative
    ASN_SEQUENCE_ADD(&(*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ack_NACK_NumRepetitions_Msg4_r13.list,ack_nack_repetition);

    *srs_SubframeConfig = 0;
    (*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.srs_SubframeConfig_r13= srs_SubframeConfig;


       * threeTone_BaseSequence_r13
       * sixTone_BaseSequence_r13
       * twelveTone_BaseSequence_r13


      //new (occhio che dmrs_config_r13 è un puntatore quando lo richiami)
      (*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.dmrs_Config_r13->threeTone_CyclicShift_r13 =0;
      (*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.dmrs_Config_r13->sixTone_CyclicShift_r13 = 0;

      (*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupHoppingEnabled_r13= 1;
      (*sib2_NB)->radioResourceConfigCommon_r13.npusch_ConfigCommon_r13.ul_ReferenceSignalsNPUSCH_r13.groupAssignmentNPUSCH_r13 =0;


    // PUCCH-Config NB-IoT

    * all data are sent over the NPUSCH. This includes also the UL control information (UCI),
    * which is transmitted using a different format. Consequently there is no equivalent to the PUCCH of LTE in NB-IoT.

    // SRS Config --May not implemented in NB-IoT!

    // uplinkPowerControlCommon - NB-IoT
    (*sib2_NB)->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.p0_NominalNPUSCH_r13 = -108;
    (*sib2_NB)->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.deltaPreambleMsg3_r13 = -6;
    (*sib2_NB)->radioResourceConfigCommon_r13.uplinkPowerControlCommon_r13.alpha_r13 = UplinkPowerControlCommon_NB_r13__alpha_r13_al1;
    //no deltaFlist_PUCCH, no UL cyclic Prefix

    //New: DL_GapConfig
     (*sib2_NB)->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapDurationCoeff_r13= DL_GapConfig_NB_r13__dl_GapDurationCoeff_r13_oneEighth;
     (*sib2_NB)->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapPeriodicity_r13= DL_GapConfig_NB_r13__dl_GapPeriodicity_r13_sf64;
     (*sib2_NB)->radioResourceConfigCommon_r13.dl_Gap_r13->dl_GapThreshold_r13= DL_GapConfig_NB_r13__dl_GapThreshold_r13_n32;

    // UE Timers and Constants -NB-IoT
    (*sib2_NB)->ue_TimersAndConstants_r13.t300_r13 = UE_TimersAndConstants_NB_r13__t300_r13_ms2500;
    (*sib2_NB)->ue_TimersAndConstants_r13.t301_r13 = UE_TimersAndConstants_NB_r13__t301_r13_ms2500;
    (*sib2_NB)->ue_TimersAndConstants_r13.t310_r13 = UE_TimersAndConstants_NB_r13__t310_r13_ms1000;//(specs pag 643)
    (*sib2_NB)->ue_TimersAndConstants_r13.t311_r13 = UE_TimersAndConstants_NB_r13__t311_r13_ms1000;//(specs pag 643)
    (*sib2_NB)->ue_TimersAndConstants_r13.n310_r13 = UE_TimersAndConstants_NB_r13__n310_r13_n1;//(specs pag 643)
    (*sib2_NB)->ue_TimersAndConstants_r13.n311_r13 = UE_TimersAndConstants_NB_r13__n311_r13_n1;//(specs pag 643)

*/

  sib2_NB->freqInfo_r13.additionalSpectrumEmission_r13 = 1;
  sib2_NB->freqInfo_r13.ul_CarrierFreq_r13 = NULL;
  sib2_NB->timeAlignmentTimerCommon_r13=TimeAlignmentTimer_infinity;//TimeAlignmentTimer_sf5120;
  //new
  sib2_NB->multiBandInfoList_r13 = NULL; //-->Additional Spectrum Emision

/// SIB3-NB-------------------------------------------------------

  sib3_NB->cellReselectionInfoCommon_r13.q_Hyst_r13=SystemInformationBlockType3_NB_r13__cellReselectionInfoCommon_r13__q_Hyst_r13_dB4;
  sib3_NB->cellReselectionServingFreqInfo_r13.s_NonIntraSearch_r13=0; //or define in configuration?

  sib3_NB->intraFreqCellReselectionInfo_r13.q_RxLevMin_r13 = -70;
  //new
  sib3_NB->intraFreqCellReselectionInfo_r13.q_QualMin_r13 = CALLOC(1,sizeof(*sib3_NB->intraFreqCellReselectionInfo_r13.q_QualMin_r13));
  *(sib3_NB->intraFreqCellReselectionInfo_r13.q_QualMin_r13)= 10; //a caso

  sib3_NB->intraFreqCellReselectionInfo_r13.p_Max_r13 = NULL;
  sib3_NB->intraFreqCellReselectionInfo_r13.s_IntraSearchP_r13 = 31; // s_intraSearch --> s_intraSearchP!!! (they call in a different way)
  sib3_NB->intraFreqCellReselectionInfo_r13.t_Reselection_r13=1;

  //how to manage?
  sib3_NB->freqBandInfo_r13 = NULL;//??
  sib3_NB->multiBandInfoList_r13 = NULL;//??


///BCCH message (generate the SI message)--------------------------------
  bcch_message->message.present = BCCH_DL_SCH_MessageType_NB_PR_c1;
  bcch_message->message.choice.c1.present = BCCH_DL_SCH_MessageType_NB__c1_PR_systemInformation_r13;

  bcch_message->message.choice.c1.choice.systemInformation_r13.criticalExtensions.present = SystemInformation_NB__criticalExtensions_PR_systemInformation_r13;
  bcch_message->message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list.count=0;

  ASN_SEQUENCE_ADD(&bcch_message->message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list,
                   sib2_NB_part);
  ASN_SEQUENCE_ADD(&bcch_message->message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.list,
                   sib3_NB_part);

#ifdef XER_PRINT
  xer_fprint(stdout, &asn_DEF_BCCH_DL_SCH_Message_NB, (void*)bcch_message);
#endif
  enc_rval = uper_encode_to_buffer(&asn_DEF_BCCH_DL_SCH_Message_NB,
                                   (void*)bcch_message,
                                   carrier->SIB23_NB,
                                   900);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

//changed only "asn_DEF_BCCH_DL_SCH_Message_NB"
#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[15000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_BCCH_DL_SCH_Message_NB, (void *)bcch_message)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_BCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_bcch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_bcch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, Mod_id, msg_p);
    }
  }
# endif
#endif

#ifdef USER_MODE
  LOG_D(RRC,"[eNB] SystemInformation-NB Encoded %d bits (%d bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);
#endif

  if (enc_rval.encoded==-1) {
    msg("[RRC] ASN1 : SI-NB encoding failed for SIB23_NB\n");
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}

/*do_RRCConnectionRequest_NB*/
//from UE side (for now not needed)
//Establishment cause are separated
//uint8_t do_RRCConnectionRequest_NB(uint8_t Mod_id, uint8_t *buffer,uint8_t *rv){..}

/*do_do_RRCConnectionSetupComplete_NB*/
//from UE side (for now not needed)
//uint8_t do_RRCConnectionSetupComplete(uint8_t Mod_id, uint8_t *buffer, const uint8_t Transaction_id, const int dedicatedInfoNASLength, const char *dedicatedInfoNAS)

/*do_RRCConnectionSetup_NB--> the aim is to establish SRB1 and SRB1bis*/
uint8_t do_RRCConnectionSetup_NB(
  const protocol_ctxt_t*     const ctxt_pP,
  rrc_eNB_ue_context_t*      const ue_context_pP,
  int                              CC_id,
  uint8_t*                   const buffer,
  const uint8_t                    Transaction_id,
  const LTE_DL_FRAME_PARMS* const frame_parms, //to be changed ora maybe not used
  SRB_ToAddModList_NB_r13_t**             SRB_configList_NB,
  struct PhysicalConfigDedicated_NB_r13** physicalConfigDedicated_NB
)

{

 asn_enc_rval_t enc_rval;
 uint8_t ecause=0;

 //logical channel group not defined for Nb-IoT

 long* prioritySRB1 = NULL; //logical channel priority pag 605 (is 1 for SRB1 and for SRB1bis? is the same?)
 long* prioritySRB1bis = NULL;
 BOOLEAN_t* logicalChannelSR_Prohibit =NULL; //pag 605

 struct SRB_ToAddMod_NB_r13* SRB1_config_NB = NULL;
 struct SRB_ToAddMod_NB_r13__rlc_Config_r13* SRB1_rlc_config_NB = NULL;
 struct SRB_ToAddMod_NB_r13__logicalChannelConfig_r13* SRB1_lchan_config_NB = NULL;

 struct SRB_ToAddMod_NB_r13* SRB1bis_config_NB = NULL;
 struct SRB_ToAddMod_NB_r13__rlc_Config_r13* SRB1bis_rlc_config_NB = NULL;
 struct SRB_ToAddMod_NB_r13__logicalChannelConfig_r13* SRB1bis_lchan_config_NB = NULL;

 //No UL_specific parameters for NB-IoT in LogicalChanelConfig-NB

 PhysicalConfigDedicated_NB_r13_t* physicalConfigDedicated2_NB = NULL;
 DL_CCCH_Message_NB_t dl_ccch_msg_NB;
 RRCConnectionSetup_NB_t* rrcConnectionSetup_NB = NULL;

 memset((void *)&dl_ccch_msg_NB,0,sizeof(DL_CCCH_Message_NB_t));
 dl_ccch_msg_NB.message.present = DL_CCCH_MessageType_NB_PR_c1;
 dl_ccch_msg_NB.message.choice.c1.present = DL_CCCH_MessageType_NB__c1_PR_rrcConnectionSetup_r13;
 rrcConnectionSetup_NB = &dl_ccch_msg_NB.message.choice.c1.choice.rrcConnectionSetup_r13;


 if (*SRB_configList_NB) {
   free(*SRB_configList_NB);
 }
 *SRB_configList_NB = CALLOC(1,sizeof(SRB_ToAddModList_NB_r13_t));

/// SRB1

//logical channel identity = 1 for SRB1

 SRB1_config_NB = CALLOC(1,sizeof(*SRB1_config_NB));

 //no srb_Identity in SRB_ToAddMod_NB

 SRB1_rlc_config_NB = CALLOC(1,sizeof(*SRB1_rlc_config_NB));
 SRB1_config_NB->rlc_Config_r13   = SRB1_rlc_config_NB;

 SRB1_rlc_config_NB->present = SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_explicitValue;
 SRB1_rlc_config_NB->choice.explicitValue.present=RLC_Config_NB_r13_PR_am;//the only possible in NB_IoT


 SRB1_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = enb_properties.properties[ctxt_pP->module_id]->srb1_timer_poll_retransmit_r13;
 SRB1_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = enb_properties.properties[ctxt_pP->module_id]->srb1_max_retx_threshold_r13;
 //(musT be disabled--> SRB1 config pag 640 specs )
 SRB1_rlc_config_NB->choice.explicitValue.choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 =NULL;

 /*no static assignment
  * SRB1_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = T_PollRetransmit_NB_r13_ms25000;
 SRB1_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t8;
 //(musT be disabled--> SRB1 config pag 640 specs )
 SRB1_rlc_config_NB->choice.explicitValue.choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 = NULL;*/

 SRB1_lchan_config_NB = CALLOC(1,sizeof(*SRB1_lchan_config_NB));
 SRB1_config_NB->logicalChannelConfig_r13  = SRB1_lchan_config_NB;

 SRB1_lchan_config_NB->present = SRB_ToAddMod_NB_r13__logicalChannelConfig_r13_PR_explicitValue;


 prioritySRB1 = CALLOC(1, sizeof(long));
 *prioritySRB1 = 1;
 SRB1_lchan_config_NB->choice.explicitValue.priority_r13 = prioritySRB1;

 logicalChannelSR_Prohibit = CALLOC(1, sizeof(BOOLEAN_t));
 *logicalChannelSR_Prohibit = 1;
 //schould be set to TRUE (specs pag 641)
 SRB1_lchan_config_NB->choice.explicitValue->logicalChannelSR_Prohibit_r13 = logicalChannelSR_Prohibit;

 //ADD SRB1
 ASN_SEQUENCE_ADD(&(*SRB_configList_NB)->list,SRB1_config_NB);

 ///SRB1bis (The configuration for SRB1 and SRB1bis is the same)

 // the only difference is the logical channel identity = 3 but not setted here
 //they are assumng that 2 RLC-AM entities are used for SRB1 and SRB1bis--> what means?

		 SRB1bis_config_NB = CALLOC(1,sizeof(*SRB1bis_config_NB));

		 //no srb_Identity in SRB_ToAddMod_NB

		 SRB1bis_rlc_config_NB = CALLOC(1,sizeof(*SRB1bis_rlc_config_NB));
		 SRB1bis_config_NB->rlc_Config_r13   = SRB1bis_rlc_config_NB;

		 SRB1bis_rlc_config_NB->present = SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_explicitValue;
		 SRB1bis_rlc_config_NB->choice.explicitValue.present=RLC_Config_NB_r13_PR_am;//the only possible in NB_IoT

		 SRB1bis_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.t_PollRetransmit_r13 = enb_properties.properties[ctxt_pP->module_id]->srb1bis_timer_poll_retransmit_r13;
		 SRB1bis_rlc_config_NB->choice.explicitValue.choice.am.ul_AM_RLC_r13.maxRetxThreshold_r13 = enb_properties.properties[ctxt_pP->module_id]->srb1bis_max_retx_threshold_r13;
		 //(musT be disabled--> SRB1 config pag 640 specs )
		 SRB1_rlc_config_NB->choice.explicitValue.choice.am.dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 =NULL;

		 SRB1bis_lchan_config_NB = CALLOC(1,sizeof(*SRB1bis_lchan_config_NB));
		 SRB1bis_config_NB->logicalChannelConfig_r13  = SRB1bis_lchan_config_NB;

		 SRB1bis_lchan_config_NB->present = SRB_ToAddMod_NB_r13__logicalChannelConfig_r13_PR_explicitValue;

		 prioritySRB1bis = CALLOC(1, sizeof(long));
		 *prioritySRB1bis = 1; //same as SRB1?
		 SRB1bis_lchan_config_NB->choice.explicitValue.priority_r13 = prioritySRB1bis;

		 logicalChannelSR_Prohibit = CALLOC(1, sizeof(BOOLEAN_t));
		 *logicalChannelSR_Prohibit = 1;
		 //schould be set to TRUE (specs pag 641)
		 SRB1bis_lchan_config_NB->choice.explicitValue->logicalChannelSR_Prohibit_r13 = logicalChannelSR_Prohibit;

		 //ADD SRB1bis //FIXME: actually there is no way to distinguish SRB1 and SRB1bis, maybe MAC doesn't care
		 ASN_SEQUENCE_ADD(&(*SRB_configList_NB)->list,SRB1bis_config_NB);


 // PhysicalConfigDedicated (NPDCCH, NPUSCH, CarrierConfig, UplinkPowerControl)

 physicalConfigDedicated2_NB = CALLOC(1,sizeof(*physicalConfigDedicated2_NB));
 *physicalConfigDedicated_NB = physicalConfigDedicated2_NB;

 physicalConfigDedicated2_NB->carrierConfigDedicated_r13= CALLOC(1, sizeof(*physicalConfigDedicated2_NB->carrierConfigDedicated_r13));
 physicalConfigDedicated2_NB->npdcch_ConfigDedicated_r13 = CALLOC(1,sizeof(*physicalConfigDedicated2_NB->npdcch_ConfigDedicated_r13));
 physicalConfigDedicated2_NB->npusch_ConfigDedicated_r13 = CALLOC(1,sizeof(*physicalConfigDedicated2_NB->npusch_ConfigDedicated_r13));
 physicalConfigDedicated2_NB->uplinkPowerControlDedicated_r13 = CALLOC(1,sizeof(*physicalConfigDedicated2_NB->uplinkPowerControlDedicated_r13));

 //no tpc, no cqi and no pucch, no pdsch, no soundingRS, no AntennaInfo, no scheduling request config

 /*
  * NB-IoT supports the operation with either one or two antenna ports, AP0 and AP1.
  * For the latter case, Space Frequency Block Coding (SFBC) is applied.
  * Once selected, the same transmission scheme applies to NPBCH, NPDCCH, and NPDSCH.
  * */

 //CarrierConfigDedicated --> I don't know nothing --> settato valori a caso
  physicalConfigDedicated2_NB->carrierConfigDedicated_r13->dl_CarrierConfig_r13->downlinkBitmapNonAnchor_r13.present=
		  DL_CarrierConfigDedicated_NB_r13__downlinkBitmapNonAnchor_r13_PR_useNoBitmap_r13;
  physicalConfigDedicated2_NB->carrierConfigDedicated_r13->dl_CarrierConfig_r13->dl_GapNonAnchor_r13.present =
		  DL_CarrierConfigDedicated_NB_r13__dl_GapNonAnchor_r13_PR_useNoGap_r13;
  physicalConfigDedicated2_NB->carrierConfigDedicated_r13->dl_CarrierConfig_r13->inbandCarrierInfo_r13.eutraControlRegionSize_r13= 0;
  //physicalConfigDedicated2_NB->carrierConfigDedicated_r13->dl_CarrierConfig_r13->inbandCarrierInfo_r13->samePCI_Indicator_r13 (??)
  physicalConfigDedicated2_NB->carrierConfigDedicated_r13->ul_CarrierConfig_r13->ul_CarrierFreq_r13.carrierFreq_r13=0;
  physicalConfigDedicated2_NB->carrierConfigDedicated_r13->ul_CarrierConfig_r13->ul_CarrierFreq_r13->carrierFreqOffset_r13= NULL;

 // NPDCCH
 physicalConfigDedicated2_NB->npdcch_ConfigDedicated_r13->npdcch_NumRepetitions_r13 =0;
 physicalConfigDedicated2_NB->npdcch_ConfigDedicated_r13->npdcch_Offset_USS_r13 =0;
 physicalConfigDedicated2_NB->npdcch_ConfigDedicated_r13->npdcch_StartSF_USS_r13=0;

 // NPUSCH
 physicalConfigDedicated2_NB->npusch_ConfigDedicated_r13->ack_NACK_NumRepetitions_r13= NULL; //(specs pag 643)
 //physicalConfigDedicated2_NB->npusch_ConfigDedicated_r13->npusch_AllSymbols_r13= //(TRUE)
 physicalConfigDedicated2_NB->npusch_ConfigDedicated_r13->groupHoppingDisabled_r13=NULL; //(TRUE?? is a long*)

 // UplinkPowerControlDedicated
 physicalConfigDedicated2_NB->uplinkPowerControlDedicated_r13->p0_UE_NPUSCH_r13 = 0; // 0 dB (specs pag 643)


 //check if the one set to NULL are correct
 rrcConnectionSetup_NB->rrc_TransactionIdentifier = Transaction_id; //input value
 rrcConnectionSetup_NB->criticalExtensions.present = RRCConnectionSetup_NB__criticalExtensions_PR_c1;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.present =RRCConnectionSetup_NB__criticalExtensions__c1_PR_rrcConnectionSetup_r13 ;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.srb_ToAddModList_r13 = *SRB_configList_NB;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.drb_ToAddModList_r13 = NULL;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.drb_ToReleaseList_r13 = NULL;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.rlf_TimersAndConstants_r13 = NULL;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.physicalConfigDedicated_r13 = physicalConfigDedicated2_NB;
 rrcConnectionSetup_NB->criticalExtensions.choice.c1.choice.rrcConnectionSetup_r13.radioResourceConfigDedicated_r13.mac_MainConfig_r13 = NULL;

#ifdef XER_PRINT
 xer_fprint(stdout, &asn_DEF_DL_CCCH_Message, (void*)&dl_ccch_msg);
#endif
 enc_rval = uper_encode_to_buffer(&asn_DEF_DL_CCCH_Message_NB,
                                  (void*)&dl_ccch_msg_NB,
                                  buffer,
                                  100);
 AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
              enc_rval.failed_type->name, enc_rval.encoded);


#ifdef USER_MODE
 LOG_D(RRC,"RRCConnectionSetup-NB Encoded %d bits (%d bytes), ecause %d\n",
       enc_rval.encoded,(enc_rval.encoded+7)/8,ecause);
#endif

 //  FREEMEM(SRB_list);
 //  free(SRB1_config);
 //  free(SRB1_rlc_config);
 //  free(SRB1_lchan_config);
 //  free(SRB1_ul_SpecificParameters);

 return((enc_rval.encoded+7)/8);
}

/*do_SecurityModeCommand - exactly the same as previous implementation*/
uint8_t do_SecurityModeCommand_NB(
  const protocol_ctxt_t* const ctxt_pP,
  uint8_t* const buffer,
  const uint8_t Transaction_id,
  const uint8_t cipheringAlgorithm,
  const uint8_t integrityProtAlgorithm)
{
  DL_DCCH_Message_NB_t dl_dcch_msg_NB;
  asn_enc_rval_t enc_rval;

  memset(&dl_dcch_msg_NB,0,sizeof(DL_DCCH_Message_NB_t));

  dl_dcch_msg_NB.message.present = DL_DCCH_MessageType_NB_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.present = DL_DCCH_MessageType_NB__c1_PR_securityModeCommand_r13;

  dl_dcch_msg_NB.message.choice.c1.choice.securityModeCommand_r13.rrc_TransactionIdentifier = Transaction_id;
  dl_dcch_msg_NB.message.choice.c1.choice.securityModeCommand_r13.criticalExtensions.present = SecurityModeCommand__criticalExtensions_PR_c1;

  dl_dcch_msg_NB.message.choice.c1.choice.securityModeCommand_r13.criticalExtensions.choice.c1.present =
		  SecurityModeCommand__criticalExtensions__c1_PR_securityModeCommand_r8;

  // the two following information could be based on the mod_id
  dl_dcch_msg_NB.message.choice.c1.choice.securityModeCommand_r13.criticalExtensions.choice.c1.choice.securityModeCommand_r8.securityConfigSMC.securityAlgorithmConfig.cipheringAlgorithm
    = (CipheringAlgorithm_r12_t)cipheringAlgorithm; //bug solved

  dl_dcch_msg_NB.message.choice.c1.choice.securityModeCommand_r13.criticalExtensions.choice.c1.choice.securityModeCommand_r8.securityConfigSMC.securityAlgorithmConfig.integrityProtAlgorithm
    = (e_SecurityAlgorithmConfig__integrityProtAlgorithm)integrityProtAlgorithm;

//only changed "asn_DEF_DL_DCCH_Message_NB"
#ifdef XER_PRINT
  xer_fprint(stdout, &asn_DEF_DL_DCCH_Message_NB, (void*)&dl_dcch_msg);
#endif
  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_DCCH_Message_NB,
                                   (void*)&dl_dcch_msg_NB,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

//changed only "asn_DEF_DL_DCCH_Message_NB" //to be left?
#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[20000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_DCCH_Message_NB, (void *) &dl_dcch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_DCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_dcch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_dcch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);
    }
  }
# endif
#endif

#ifdef USER_MODE
  LOG_D(RRC,"[eNB %d] securityModeCommand-NB for UE %x Encoded %d bits (%d bytes)\n",
        ctxt_pP->module_id,
        ctxt_pP->rnti,
        enc_rval.encoded,
        (enc_rval.encoded+7)/8);
#endif

  if (enc_rval.encoded==-1) {
    LOG_E(RRC,"[eNB %d] ASN1 : securityModeCommand-NB encoding failed for UE %x\n",
          ctxt_pP->module_id,
          ctxt_pP->rnti);
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}

/*do_UECapabilityEnquiry_NB - very similar to legacy lte*/
uint8_t do_UECapabilityEnquiry_NB(
  const protocol_ctxt_t* const ctxt_pP,
  uint8_t*               const buffer,
  const uint8_t                Transaction_id
)

{

  DL_DCCH_Message_NB_t dl_dcch_msg_NB;
  //no RAT type in NB-IoT
  asn_enc_rval_t enc_rval;

  memset(&dl_dcch_msg_NB,0,sizeof(DL_DCCH_Message_NB_t));

  dl_dcch_msg_NB.message.present           = DL_DCCH_MessageType_NB_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.present = DL_DCCH_MessageType_NB__c1_PR_ueCapabilityEnquiry_r13;

  dl_dcch_msg_NB.message.choice.c1.choice.ueCapabilityEnquiry_r13.rrc_TransactionIdentifier = Transaction_id;

  dl_dcch_msg_NB.message.choice.c1.choice.ueCapabilityEnquiry_r13.criticalExtensions.present = UECapabilityEnquiry_NB__criticalExtensions_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.choice.ueCapabilityEnquiry_r13.criticalExtensions.choice.c1.present =
		  UECapabilityEnquiry_NB__criticalExtensions__c1_PR_ueCapabilityEnquiry_r13;

  //no ue_CapabilityRequest (list of RAT_Type)

//only changed "asn_DEF_DL_DCCH_Message_NB"
#ifdef XER_PRINT
  xer_fprint(stdout, &asn_DEF_DL_DCCH_Message_NB, (void*)&dl_dcch_msg);
#endif
  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_DCCH_Message_NB,
                                   (void*)&dl_dcch_msg_NB,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[20000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_DCCH_Message_NB, (void *) &dl_dcch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_CCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_ccch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_ccch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);
    }
  }
# endif
#endif

#ifdef USER_MODE
  LOG_D(RRC,"[eNB %d] UECapabilityEnquiry-NB for UE %x Encoded %d bits (%d bytes)\n",
        ctxt_pP->module_id,
        ctxt_pP->rnti,
        enc_rval.encoded,
        (enc_rval.encoded+7)/8);
#endif

  if (enc_rval.encoded==-1) {
    LOG_E(RRC,"[eNB %d] ASN1 : UECapabilityEnquiry-NB encoding failed for UE %x\n",
          ctxt_pP->module_id,
          ctxt_pP->rnti);
    return(-1);
  }

  return((enc_rval.encoded+7)/8);
}

/*do_RRCConnectionReconfiguration_NB-->may convey information for resource configuration
 * (including RBs, MAC main configuration and physical channel configuration)
 * including any associated dedicated NAS information.*/
uint16_t do_RRCConnectionReconfiguration_NB(
  const protocol_ctxt_t*        const ctxt_pP,
    uint8_t                            *buffer,
    uint8_t                             Transaction_id,
    SRB_ToAddModList_NB_r13_t          *SRB_list_NB, //SRB_ConfigList2 (default) //SRB_ConfigList2 (handover)
    DRB_ToAddModList_NB_r13_t          *DRB_list_NB, //DRB_ConfigList (default)  //DRB_ConfigList2 (handover)
    DRB_ToReleaseList_NB_r13_t         *DRB_list2_NB, //is NULL when passed
    struct PhysicalConfigDedicated_NB_r13     *physicalConfigDedicated_NB,
	MAC_MainConfig_NB_r13_t                   *mac_MainConfig_NB,
  struct RRCConnectionReconfiguration_NB_r13_IEs__dedicatedInfoNASList_r13* dedicatedInfoNASList_NB)

{

 //check on DRB_list if contains more than 2 DRB?

  asn_enc_rval_t enc_rval;
  DL_DCCH_Message_NB_t dl_dcch_msg_NB;
  RRCConnectionReconfiguration_NB_t *rrcConnectionReconfiguration_NB;


  memset(&dl_dcch_msg_NB,0,sizeof(DL_DCCH_Message_NB_t));

  dl_dcch_msg_NB.message.present           = DL_DCCH_MessageType_NB_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.present = DL_DCCH_MessageType_NB__c1_PR_rrcConnectionReconfiguration_r13;
  rrcConnectionReconfiguration_NB          = &dl_dcch_msg_NB.message.choice.c1.choice.rrcConnectionReconfiguration_r13;

  // RRCConnectionReconfiguration
  rrcConnectionReconfiguration_NB->rrc_TransactionIdentifier = Transaction_id;
  rrcConnectionReconfiguration_NB->criticalExtensions.present = RRCConnectionReconfiguration_NB__criticalExtensions_PR_c1;
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.present =RRCConnectionReconfiguration_NB__criticalExtensions__c1_PR_rrcConnectionReconfiguration_r13 ;

  //RAdioResourceconfigDedicated
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13 =
		  CALLOC(1,sizeof(*rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13));
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->srb_ToAddModList_r13 = SRB_list_NB;
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->drb_ToAddModList_r13 = DRB_list_NB;
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->drb_ToReleaseList_r13 = DRB_list2_NB;
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->physicalConfigDedicated_r13 = physicalConfigDedicated_NB;
  //used? pass as argument?
  //rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->rlf_TimersAndConstants_r13

  if (mac_MainConfig_NB!=NULL) {
    rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13 =
    		CALLOC(1, sizeof(*rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13));
    rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13->present
      =RadioResourceConfigDedicated_NB_r13__mac_MainConfig_r13_PR_explicitValue_r13;
   //why memcopy only this one?
    memcpy(&rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13->choice.explicitValue_r13,
           mac_MainConfig_NB, sizeof(*mac_MainConfig_NB));

  } else {
	  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13=NULL;
  }

  //no measConfig
  //no mobilityControlInfo

  //Other
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.dedicatedInfoNASList_r13 = dedicatedInfoNASList_NB;
  //(how to set?)
  rrcConnectionReconfiguration_NB->criticalExtensions.choice.c1.choice.rrcConnectionReconfiguration_r13.fullConfig_r13 = NULL;

  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_DCCH_Message_NB,
                                   (void*)&dl_dcch_msg_NB,
                                   buffer,
                                   RRC_BUF_SIZE);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %l)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

  //changed only asn_DEF_DL_DCCH_Message_NB
#ifdef XER_PRINT
  xer_fprint(stdout,&asn_DEF_DL_DCCH_Message_NB,(void*)&dl_dcch_msg);
#endif

#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[30000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_DCCH_Message_NB, (void *) &dl_dcch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_DCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_dcch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_dcch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);
    }
  }
# endif
#endif


  LOG_I(RRC,"RRCConnectionReconfiguration-NB Encoded %d bits (%d bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);

  return((enc_rval.encoded+7)/8);
}

/*do_RRCConnectionReestablishmentReject - exactly the same as legacy LTE*/
uint8_t do_RRCConnectionReestablishmentReject_NB(
    uint8_t                    Mod_id,
    uint8_t*                   const buffer)
{

  asn_enc_rval_t enc_rval;

  DL_CCCH_Message_NB_t dl_ccch_msg_NB;
  RRCConnectionReestablishmentReject_t *rrcConnectionReestablishmentReject;

  memset((void *)&dl_ccch_msg_NB,0,sizeof(DL_CCCH_Message_NB_t));
  dl_ccch_msg_NB.message.present = DL_CCCH_MessageType_NB_PR_c1;
  dl_ccch_msg_NB.message.choice.c1.present = DL_CCCH_MessageType_NB__c1_PR_rrcConnectionReestablishmentReject_r13;
  rrcConnectionReestablishmentReject    = &dl_ccch_msg_NB.message.choice.c1.choice.rrcConnectionReestablishmentReject_r13;

  // RRCConnectionReestablishmentReject //exactly the same as LTE
  rrcConnectionReestablishmentReject->criticalExtensions.present = RRCConnectionReestablishmentReject__criticalExtensions_PR_rrcConnectionReestablishmentReject_r8;

  //Only change in "asn_DEF_DL_CCCH_Message_NB"
#ifdef XER_PRINT
  xer_fprint(stdout, &asn_DEF_DL_CCCH_Message_NB, (void*)&dl_ccch_msg);
#endif
  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_CCCH_Message_NB,
                                   (void*)&dl_ccch_msg_NB,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %lu)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

  //Only change in "asn_DEF_DL_CCCH_Message_NB"
#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[20000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_CCCH_Message_NB, (void *) &dl_ccch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_CCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_ccch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_ccch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, Mod_id, msg_p);
    }
  }
# endif
#endif

#ifdef USER_MODE
  LOG_D(RRC,"RRCConnectionReestablishmentReject Encoded %d bits (%d bytes)\n",
        enc_rval.encoded,(enc_rval.encoded+7)/8);
#endif

  return((enc_rval.encoded+7)/8);
}

/*do_RRCConnectionReject_NB*/
uint8_t do_RRCConnectionReject_NB(
    uint8_t                    Mod_id,
    uint8_t*                   const buffer)

{

  asn_enc_rval_t enc_rval;

  DL_CCCH_Message_NB_t dl_ccch_msg_NB;
  RRCConnectionReject_NB_t *rrcConnectionReject_NB;

  memset((void *)&dl_ccch_msg_NB,0,sizeof(DL_CCCH_Message_NB_t));
  dl_ccch_msg_NB.message.present           = DL_CCCH_MessageType_NB_PR_c1;
  dl_ccch_msg_NB.message.choice.c1.present = DL_CCCH_MessageType_NB__c1_PR_rrcConnectionReject_r13;
  rrcConnectionReject_NB = &dl_ccch_msg_NB.message.choice.c1.choice.rrcConnectionReject_r13;

  // RRCConnectionReject-NB
  rrcConnectionReject_NB->criticalExtensions.present = RRCConnectionReject_NB__criticalExtensions_PR_c1;
  rrcConnectionReject_NB->criticalExtensions.choice.c1.present = RRCConnectionReject_NB__criticalExtensions__c1_PR_rrcConnectionReject_r13;
  /* let's put an extended wait time of 1s for the moment */
  rrcConnectionReject_NB->criticalExtensions.choice.c1.choice.rrcConnectionReject_r13.extendedWaitTime_r13 = 1;
  //new-use of suspend indication
  //If present, this field indicates that the UE should remain suspended and not release its stored context.
  rrcConnectionReject_NB->criticalExtensions.choice.c1.choice.rrcConnectionReject_r13->rrc_SuspendIndication_r13=
		  RRCConnectionReject_NB_r13_IEs__rrc_SuspendIndication_r13_true;

  //Only Modified "asn_DEF_DL_CCCH_Message_NB"
#ifdef XER_PRINT
  xer_fprint(stdout, &asn_DEF_DL_CCCH_Message_NB, (void*)&dl_ccch_msg);
#endif
  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_CCCH_Message_NB,
                                   (void*)&dl_ccch_msg_NB,
                                   buffer,
                                   100);
  AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %ld)!\n",
               enc_rval.failed_type->name, enc_rval.encoded);

#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[20000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_CCCH_Message_NB, (void *) &dl_ccch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_CCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_ccch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_ccch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, Mod_id, msg_p);
    }
  }
# endif
#endif

#ifdef USER_MODE
  LOG_D(RRC,"RRCConnectionReject-NB Encoded %d bits (%d bytes)\n",
        enc_rval.encoded,(enc_rval.encoded+7)/8);
#endif

  return((enc_rval.encoded+7)/8);
}

/*do_RRCConnectionRelease_NB*/
uint8_t do_RRCConnectionRelease_NB(uint8_t Mod_id, uint8_t *buffer,int Transaction_id)
{

  asn_enc_rval_t enc_rval;

  DL_DCCH_Message_NB_t dl_dcch_msg_NB;
  RRCConnectionRelease_NB_t *rrcConnectionRelease_NB;


  memset(&dl_dcch_msg_NB,0,sizeof(DL_DCCH_Message_NB_t));

  dl_dcch_msg_NB.message.present           = DL_DCCH_MessageType_NB_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.present = DL_DCCH_MessageType_NB__c1_PR_rrcConnectionRelease_r13;
  rrcConnectionRelease_NB = &dl_dcch_msg_NB.message.choice.c1.choice.rrcConnectionRelease_r13;

  // RRCConnectionRelease
  rrcConnectionRelease_NB->rrc_TransactionIdentifier = Transaction_id;
  rrcConnectionRelease_NB->criticalExtensions.present = RRCConnectionRelease_NB__criticalExtensions_PR_c1;
  rrcConnectionRelease_NB->criticalExtensions.choice.c1.present =RRCConnectionRelease_NB__criticalExtensions__c1_PR_rrcConnectionRelease_r13 ;

  rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.releaseCause_r13 = ReleaseCause_NB_r13_other;
  //which value set?
  rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.resumeIdentity_r13 = NULL;
  rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.extendedWaitTime_r13 = NULL;
  rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.redirectedCarrierInfo_r13 = NULL;

  //why in this case allocate memory for noncriticalExtension?
  rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.nonCriticalExtension=CALLOC(1,
      sizeof(*rrcConnectionRelease_NB->criticalExtensions.choice.c1.choice.rrcConnectionRelease_r13.nonCriticalExtension));

  enc_rval = uper_encode_to_buffer(&asn_DEF_DL_DCCH_Message_NB,
                                   (void*)&dl_dcch_msg_NB,
                                   buffer,
                                   RRC_BUF_SIZE);

  return((enc_rval.encoded+7)/8);
}

//no do_MBSFNAreaConfig(..) in NB-IoT
//no do_MeasurementReport(..) in NB-IoT

/*do_DLInformationTransfer_NB*/
uint8_t do_DLInformationTransfer_NB(
		uint8_t Mod_id,
		uint8_t **buffer,
		uint8_t transaction_id,
		uint32_t pdu_length,
		uint8_t *pdu_buffer)

{
  ssize_t encoded;

  DL_DCCH_Message_NB_t dl_dcch_msg_NB;

  memset(&dl_dcch_msg_NB, 0, sizeof(DL_DCCH_Message_NB_t));

  dl_dcch_msg_NB.message.present           = DL_DCCH_MessageType_NB_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.present = DL_DCCH_MessageType_NB__c1_PR_dlInformationTransfer_r13;
  dl_dcch_msg_NB.message.choice.c1.choice.dlInformationTransfer_r13.rrc_TransactionIdentifier = transaction_id;
  dl_dcch_msg_NB.message.choice.c1.choice.dlInformationTransfer_r13.criticalExtensions.present = DLInformationTransfer_NB__criticalExtensions_PR_c1;
  dl_dcch_msg_NB.message.choice.c1.choice.dlInformationTransfer_r13.criticalExtensions.choice.c1.present = DLInformationTransfer_NB__criticalExtensions__c1_PR_dlInformationTransfer_r13;
  dl_dcch_msg_NB.message.choice.c1.choice.dlInformationTransfer_r13.criticalExtensions.choice.c1.choice.dlInformationTransfer_r13.dedicatedInfoNAS_r13.size = pdu_length;
  dl_dcch_msg_NB.message.choice.c1.choice.dlInformationTransfer_r13.criticalExtensions.choice.c1.choice.dlInformationTransfer_r13.dedicatedInfoNAS_r13.buf = pdu_buffer;

  encoded = uper_encode_to_new_buffer (&asn_DEF_DL_DCCH_Message_NB, NULL, (void*) &dl_dcch_msg_NB, (void **) buffer);

  //only change in "asn_DEF_DL_DCCH_Message_NB"
#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[10000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_DCCH_Message_NB, (void *)&dl_dcch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_DCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_dcch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_dcch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, Mod_id, msg_p);
    }
  }
# endif
#endif

  return encoded;
}

/*do_ULInformationTransfer*/
//for the moment is not needed (UE-SIDE)

/*OAI_UECapability_t *fill_ue_capability*/

/*do_RRCConnectionReestablishment-->used to re-establish SRB1*/ //come fare??//quali parametri in ingresso?
uint8_t do_RRCConnectionReestablishment_NB(
		uint8_t Mod_id,
		uint8_t* const buffer,
		const uint8_t     Transaction_id,
		const LTE_DL_FRAME_PARMS* const frame_parms, //to be changed
		SRB_ToAddModList_NB_r13_t*      SRB_list_NB) //should contain SRB1 already configured?
{

	asn_enc_rval_t enc_rval;
	DL_CCCH_Message_NB_t dl_ccch_msg;
	RRCConnectionReestablishment_NB_t* rrcConnectionReestablishment_NB;

	memset(&dl_ccch_msg, 0, sizeof(DL_CCCH_Message_NB_t));

	dl_ccch_msg.message.present = DL_CCCH_MessageType_NB_PR_c1;
	dl_ccch_msg.message.choice.c1.present = DL_CCCH_MessageType_NB__c1_PR_rrcConnectionReestablishment_r13;
	rrcConnectionReestablishment_NB = &dl_ccch_msg.message.choice.c1.choice.rrcConnectionReestablishment_r13;

	//rrcConnectionReestablishment_NB
	rrcConnectionReestablishment_NB->rrc_TransactionIdentifier = Transaction_id;
	rrcConnectionReestablishment_NB->criticalExtensions.present = RRCConnectionReestablishment_NB__criticalExtensions_PR_c1;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.present = RRCConnectionReestablishment_NB__criticalExtensions__c1_PR_rrcConnectionReestablishment_r13;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13=
			CALLOC(1,sizeof(*rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13));

	//??
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->srb_ToAddModList_r13 = SRB_list_NB;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->drb_ToAddModList_r13 = NULL;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->drb_ToReleaseList_r13 = NULL;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->rlf_TimersAndConstants_r13= NULL;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->mac_MainConfig_r13= NULL;
	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.radioResourceConfigDedicated_r13->physicalConfigDedicated_r13 = NULL;

	rrcConnectionReestablishment_NB->criticalExtensions.choice.c1.choice.rrcConnectionReestablishment_r13.nextHopChainingCount_r13=0;//??

	enc_rval = uper_encode_to_buffer(&asn_DEF_DL_CCCH_Message_NB,
	                                   (void*)&dl_ccch_msg,
	                                   buffer,
	                                   RRC_BUF_SIZE);

	 AssertFatal (enc_rval.encoded > 0, "ASN1 message encoding failed (%s, %l)!\n",
	               enc_rval.failed_type->name, enc_rval.encoded);

#ifdef XER_PRINT
  xer_fprint(stdout,&asn_DEF_DL_CCCH_Message_NB,(void*)&dl_ccch_msg);
#endif

#if defined(ENABLE_ITTI)
# if !defined(DISABLE_XER_SPRINT)
  {
    char        message_string[30000];
    size_t      message_string_size;

    if ((message_string_size = xer_sprint(message_string, sizeof(message_string), &asn_DEF_DL_CCCH_Message_NB, (void *) &dl_ccch_msg)) > 0) {
      MessageDef *msg_p;

      msg_p = itti_alloc_new_message_sized (TASK_RRC_ENB, RRC_DL_CCCH, message_string_size + sizeof (IttiMsgText));
      msg_p->ittiMsg.rrc_dl_ccch.size = message_string_size;
      memcpy(&msg_p->ittiMsg.rrc_dl_ccch.text, message_string, message_string_size);

      itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);
    }
  }
# endif
#endif

  LOG_I(RRC,"RRCConnectionReestablishment-NB Encoded %d bits (%d bytes)\n",enc_rval.encoded,(enc_rval.encoded+7)/8);

}


// -----??????--------------------
#ifndef USER_MODE
int init_module(void)
{
  printk("Init asn1_msg_nb_iot module\n");

  // A non 0 return means init_module failed; module can't be loaded.
  return 0;
}


void cleanup_module(void)
{
  printk("Stopping asn1_msg_nb_iot module\n");
}

EXPORT_SYMBOL(do_SIB1_NB);
EXPORT_SYMBOL(do_SIB23_NB);
EXPORT_SYMBOL(do_RRCConnectionRequest_NB);
EXPORT_SYMBOL(do_RRCConnectionSetupComplete_NB);
EXPORT_SYMBOL(do_RRCConnectionReconfigurationComplete_NB);
EXPORT_SYMBOL(do_RRCConnectionSetup_NB);
EXPORT_SYMBOL(do_RRCConnectionReestablishmentReject);
EXPORT_SYMBOL(do_RRCConnectionReconfiguration_NB);
EXPORT_SYMBOL(asn_DEF_UL_DCCH_Message_NB);
EXPORT_SYMBOL(asn_DEF_UL_CCCH_Message_NB);
EXPORT_SYMBOL(asn_DEF_SystemInformation_NB);
EXPORT_SYMBOL(asn_DEF_DL_DCCH_Message_NB);
EXPORT_SYMBOL(asn_DEF_SystemInformationBlockType1_NB);
EXPORT_SYMBOL(asn_DEF_DL_CCCH_Message_NB);
EXPORT_SYMBOL(uper_decode_complete);
EXPORT_SYMBOL(uper_decode);
EXPORT_SYMBOL(transmission_mode_rrc);
#endif

//----------------------------------




