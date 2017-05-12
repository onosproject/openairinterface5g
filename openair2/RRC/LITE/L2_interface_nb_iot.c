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

/*! \file l2_interface_nb_iot.c
 * \brief layer 2 interface, used to support different RRC sublayer
 * \author Raymond Knopp and Navid Nikaein
 * \date 2010-2014
 * \version 1.0
 * \company Eurecom
 * \email: raymond.knopp@eurecom.fr
 */
#include "PHY/defs.h"
#include "SCHED/defs.h"
#include "platform_types.h"
#include "defs_nb_iot.h" //RRC
#include "LAYER2/MAC/defs_nb_iot.h" //MAC
#include "extern.h"
#include "LAYER2/MAC/extern.h"
#include "UTIL/LOG/log.h"
#include "UTIL/OCG/OCG_vars.h"
#include "rrc_eNB_UE_context.h"
#include "pdcp_primitives.h"
#include "pdcp.h"
#include "msc.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "gtpv1u.h"

#ifdef PHY_EMUL
#include "SIMULATION/simulation_defs.h"
extern EMULATION_VARS *Emul_vars;
extern eNB_MAC_INST *eNB_mac_inst;
extern UE_MAC_INST *UE_mac_inst;
#endif

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

//#define RRC_DATA_REQ_DEBUG
#define DEBUG_RRC 1

//to add into platform types
typedef boolean_t srb1bis_flag_t;
#define SRB1BIS_FLAG_NO	FALSE
#define SRB1BIS_FLAG_YES	TRUE //defined in interTask-interface = 1

mui_t mui=0;

//XXX MP: just temporary usage since i put in one single file all the primitives modified (but they should be relocated in the
//proper file where they are defined as indicated in the comments

//XXX access of protected variables in pdcp.h
extern unsigned int           pdcp_eNB_UE_instance_to_rnti_index;
extern rnti_t                 pdcp_eNB_UE_instance_to_rnti[NUMBER_OF_UE_MAX];
extern list_t                 pdcp_sdu_list;
//extern struct mac_data_req rlc_am_mac_data_request (const protocol_ctxt_t* const ctxtP,void * const rlc_pP);
//extern eNB_MAC_INST_NB *eNB;
extern void rlc_tm_init ( const protocol_ctxt_t* const  ctxt_pP, rlc_tm_entity_t * const rlcP);
extern void rlc_tm_set_debug_infos(
        const protocol_ctxt_t* const  ctxt_pP,
        rlc_tm_entity_t * const       rlcP,
        const srb_flag_t              srb_flagP,
        const rb_id_t                 rb_idP,
        const logical_chan_id_t chan_idP);
extern void rlc_tm_configure(
        const protocol_ctxt_t* const  ctxt_pP,
        rlc_tm_entity_t * const rlcP,
        const boolean_t is_uplink_downlinkP);
extern void     rlc_am_rx (const protocol_ctxt_t* const ctxtP,void * const rlc_pP, struct mac_data_ind);
extern tbs_size_t            mac_rlc_serialize_tb   (char*, list_t);
extern struct mac_data_ind mac_rlc_deserialize_tb (
		  char     *buffer_pP,
		  const tb_size_t tb_sizeP,
		  num_tb_t  num_tbP,
		  crc_t    *crcs_pP);


/*---------------------------------RRC-MAC-----------------------------------*/

//defined in MAC/config.c
//FIXME: this function has no implication in terms of logical channel configuration for MAC
int NB_rrc_mac_config_req_eNB(
			   module_id_t       				Mod_idP,
			   int                              CC_idP,
			   int								rntiP, //FIXME: Raymond bug?
			   int                              physCellId,
			   int                              p_eNB,
			   int                              Ncp,
			   long                             eutra_band,//FIXME: frequencyBandIndicator in sib1 (is a long not an int!!)
			   struct NS_PmaxList_NB_r13        *frequencyBandInfo, //optional SIB1
			   struct MultiBandInfoList_NB_r13  *multiBandInfoList, //optional SIB1
			   struct DL_Bitmap_NB_r13          *dl_bitmap, //optional SIB1
			   long*                            eutraControlRegionSize, //optional sib1
			   long*							nrs_CRS_PowerOffset, //optional
			   uint8_t                          *SIwindowsize, //maybe no more needed because TDD only
			   uint16_t                         *SIperiod, //maybe no more needed because TDD only
			   uint32_t                         dl_CarrierFreq,
			   uint32_t                         ul_CarrierFreq,
			   BCCH_BCH_Message_NB_t            *mib_NB,
			   RadioResourceConfigCommonSIB_NB_r13_t   *radioResourceConfigCommon,
			   struct PhysicalConfigDedicated_NB_r13  *physicalConfigDedicated,
			   MAC_MainConfig_NB_r13_t                *mac_MainConfig,
			   long                             logicalChannelIdentity,//FIXME: decide how to use it
			   LogicalChannelConfig_NB_r13_t          *logicalChannelConfig //FIXME: decide how to use it
			   )
{

  int i;

  int UE_id = -1;
  //eNB_MAC_INST_NB *eNB = &eNB_mac_inst_NB[Mod_idP];
  UE_list_NB_t *UE_list= &eNB_mac_inst_NB->UE_list;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_MAC_CONFIG, VCD_FUNCTION_IN);

  UE_id = find_UE_id(Mod_idP, rntiP);


  if (mib_NB!=NULL) {
    if (eNB_mac_inst == NULL) l2_init_eNB(); //XXX MP: to be included in the MAC/main.c

    mac_top_init_eNB(); //XXX MP:  to be included in the MAC/main.c

    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].mib_NB           = mib_NB;
    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].physCellId     = physCellId;
    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].p_eNB          = p_eNB;
    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].Ncp            = Ncp;
    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].eutra_band     = eutra_band;
    eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].dl_CarrierFreq = dl_CarrierFreq;

//    mac_xface->NB_phy_config_mib_eNB(Mod_idP,CC_idP, //XXX MP: defined by Nick in lte_init_nb_iot.c
//				  eutra_band,
//				  physCellId,
//				  Ncp,
//				  p_eNB,
//				  dl_CarrierFreq,
//				  ul_CarrierFreq,
//				  mib_NB->message.operationModeInfo_r13);

    mac_init_cell_params(Mod_idP,CC_idP);
  }

  //phy_config_sib1_eNB--> not implemented for NB_IoT

  if (radioResourceConfigCommon!=NULL) {
      LOG_I(MAC,"[CONFIG]SIB2/3-NB Contents (partial)\n");

      //TODO:show something on npusch parameters
      //LOG_I(MAC,"[CONFIG]npusch_config_common_r13.

      eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].radioResourceConfigCommon = radioResourceConfigCommon;

      if (ul_CarrierFreq>0) eNB_mac_inst_NB[Mod_idP].common_channels[CC_idP].ul_CarrierFreq   = ul_CarrierFreq;

      //no ul_Bandwidth

//      mac_xface->NB_phy_config_sib2_eNB( //XXX MP: defined by Nick in lte_init_nb_iot.c
//    		  Mod_idP,
//    		  CC_idP,
//			  radioResourceConfigCommon,
//			  NULL //ul_carrier_frequency
//			  );


  }

  if (logicalChannelConfig!= NULL) {
    if (UE_id == -1) {
      LOG_E(MAC,"%s:%d:%s: ERROR, UE_id == -1\n", __FILE__, __LINE__, __FUNCTION__);
    } else {
      //TODO logical channel configuration for UE_list->UE_template
    }
  }


  if (physicalConfigDedicated != NULL) {
    if (UE_id == -1)
      LOG_E(MAC,"%s:%d:%s: ERROR, UE_id == -1\n", __FILE__, __LINE__, __FUNCTION__);
    else
    {
//      mac_xface->NB_phy_config_dedicated_eNB( //XXX MP: defined by Nick in lte_init_nb_iot.c
//    		  Mod_idP,
//			  CC_idP,
//			  UE_RNTI(Mod_idP, UE_id),
//			  physicalConfigDedicated);
    }
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RRC_MAC_CONFIG, VCD_FUNCTION_OUT);
  return(0);

}

//defined in L2_interface
int8_t NB_mac_rrc_data_req_eNB(
  const module_id_t Mod_idP,
  const int         CC_id,
  const frame_t     frameP,
  const rb_id_t     Srb_id,
  uint8_t*    const buffer_pP,
  uint32_t				start_sfn_sib1_NB, //starting frame for sib1 transmission
  uint32_t				nb_sib1_NB_rep, //number of sib1 repetitions
  const frame_t		h_frameP, //HSFN
  long				si_periodicity, //SI-periodicity
  long				si_windowLength //Si-windwolength
)
{
  SRB_INFO *Srb_info;
  uint8_t Sdu_size=0;
  uint32_t y = 0;
  uint32_t sib1_NB_period = 256; //from specs TS 36.331

#ifdef DEBUG_RRC
  int i;
  LOG_T(RRC,"[eNB %d] NB_mac_rrc_data_req_eNB to SRB ID=%d\n",Mod_idP,Srb_id);
#endif

    if((Srb_id & RAB_OFFSET) == BCCH) {
      if(eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].SI.Active==0) {
        return 0;
      }

      //I have to check which SIB is transmitted

      if (eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB1_NB == 255) {
        LOG_E(RRC,"[eNB %d] MAC Request for SIB1-NB and SIB1-NB not initialized\n",Mod_idP);
        mac_xface->macphy_exit("mac_rrc_data_req:  MAC Request for SIB1-NB and SIB1-NB not initialized");
      }

      //the entire scheduling of SIB1-NB is based on the SchedulingInfoSIB1 of MIB-NB

      //SIB1-NB transmission occurs in subframe #4 of every other frame in 16 continuous frames
      //schedule with a periodicity of 2560 ms (256 Radio Frames) and repetitions (4, 8 or 16) are made, equally spaced,
      //within the 2560 ms period

      //Implementation: //FIXME should be checked if compatible if SIB23-NB scheduling
      //SchedulingInfoSIB_r13 = 0 --> n°repetitions= 4 + NcelliD mod 4 = 1 --> start_sfn_sib1_Nb = 16

      /*logic*/
      //1) check if the current frame is before/after the starting frame of sib1-NB transmission
      //2) check if the current frame is considered in the 16 continuous frame of sib1-NB transmission in the periodicity

      if(frameP < start_sfn_sib1_NB)
    	  return 0;

      //calculate offset between SIB1-NB repetitions
      int offset = (sib1_NB_period-(16*nb_sib1_NB_rep))/nb_sib1_NB_rep;

      for( int i = 0; i < nb_sib1_NB_rep; i++)
      {
    	  y = start_sfn_sib1_NB+ i*(16+offset);
    	  if(frameP>= y && frameP <= (y+15)) //SIB1_NB transmission
    	  {
    		  memcpy(&buffer_pP[0],
    				  eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].SIB1_NB,
    				  eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB1_NB);

				#ifdef DEBUG_RRC
    		          LOG_T(RRC,"[eNB %d] Frame %d : BCCH request => SIB1_NB\n",Mod_idP,frameP);

    		          for (i=0; i<eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB1_NB; i++) {
    		            LOG_T(RRC,"%x.",buffer_pP[i]);
    		          }

    		          LOG_T(RRC,"\n");
    		    #endif

    		  return (eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB1_NB);
    	  }
      }

///if we are here means that is a SIB23-NB transmission


      //**for NB_IoT Implementations
      //parameters that comes into play (defined in SIB1-NB):
      //1_ si_windowlenght (w) (same for all SI messages) = 160 ms
      //2_ si_radioFrameOffset (same for all SI messages) (not considered)
      //3_ si_periodicity (T) = 64 rf
      //4_ si_repetitionPattern = every2nd frame

      /*logic (see TS 36.331 5.2.3a)*/
      //since we have only 1 entry in the SchedulingInfoList (SIB23-NB) --> n=1
      //therefore  x = (n-1)*w = 0
	  //Starting RF --> (HSFN*1024 + SFN) mod T = FLOOR(x/10 = 0) + si_radioFrameOffset (suppose = 0)
      //Si-Window start radio frame = 0,64,...

      //Assume that SIB23_NB transmission occur in SF #4 (like SIB1-NB) --> when this function should be called

      for(int i = 0; i < (1024L/si_periodicity); i++) {

    	  if(frameP >= i*si_periodicity && frameP <= i*(si_periodicity)+ si_windowLength -1)
    	  {
    		  if((frameP%2)==1) //every2ndframe --> si_repetitionPattern
    		  {
    		        memcpy(&buffer_pP[0],
    		        		eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].SIB23_NB,
    						eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB23_NB);

    		#ifdef DEBUG_RRC
    		        LOG_T(RRC,"[eNB %d] Frame %d BCCH request => SIB 2-3\n",Mod_idP,frameP);

    		        for (i=0; i<eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB23_NB; i++) {
    		          LOG_T(RRC,"%x.",buffer_pP[i]);
    		        }

    		        LOG_T(RRC,"\n");
    		#endif
    		        return(eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].sizeof_SIB23_NB);
    		  }
    	  }

      }

        return(0);
    }

    if( (Srb_id & RAB_OFFSET ) == CCCH) {
      LOG_T(RRC,"[eNB %d] Frame %d CCCH request (Srb_id %d)\n",Mod_idP,frameP, Srb_id);

      if(eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].Srb0.Active==0) {
        LOG_E(RRC,"[eNB %d] CCCH Not active\n",Mod_idP);
        return -1;
      }

      Srb_info=&eNB_rrc_inst_NB[Mod_idP].carrier[CC_id].Srb0;

      // check if data is there for MAC
      if(Srb_info->Tx_buffer.payload_size>0) { //Fill buffer
        LOG_D(RRC,"[eNB %d] CCCH (%p) has %d bytes (dest: %p, src %p)\n",Mod_idP,Srb_info,Srb_info->Tx_buffer.payload_size,buffer_pP,Srb_info->Tx_buffer.Payload);

        memcpy(buffer_pP,Srb_info->Tx_buffer.Payload,Srb_info->Tx_buffer.payload_size);
        Sdu_size = Srb_info->Tx_buffer.payload_size;
        Srb_info->Tx_buffer.payload_size=0;
      }

      return (Sdu_size);
    }
  return(0);
}

//defined in L2_interface
//called by rx_sdu only in case of CCCH message (e.g RRCConnectionRequest-NB)
int8_t NB_mac_rrc_data_ind_eNB(
  const module_id_t     module_idP,
  const int             CC_id,
  const frame_t         frameP,
  const sub_frame_t     sub_frameP,
  const rnti_t          rntiP,
  const rb_id_t         srb_idP,//could be skipped since always go through the CCCH channel
  const uint8_t*        sduP,
  const sdu_size_t      sdu_lenP
)
{
  SRB_INFO *Srb_info;
  protocol_ctxt_t ctxt;
  sdu_size_t      sdu_size = 0;

  /* for no gcc warnings */
  (void)sdu_size;

  //MP: XXX eNB_index (last parameter) set = 0 (verify)
  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_idP, ENB_FLAG_YES, rntiP, frameP, sub_frameP,0);

    Srb_info = &eNB_rrc_inst_NB[module_idP].carrier[CC_id].Srb0;
    LOG_T(RRC,"[eNB %d] Received SDU for CCCH on SRB %d\n",module_idP,Srb_info->Srb_id);


    //this messages are used by the RRC if ITTI enabled
#if defined(ENABLE_ITTI)
    {
      MessageDef *message_p;
      int msg_sdu_size = sizeof(RRC_MAC_CCCH_DATA_IND (message_p).sdu);

      if (sdu_lenP > msg_sdu_size) {
        LOG_E(RRC, "SDU larger than CCCH SDU buffer size (%d, %d)", sdu_lenP, msg_sdu_size);
        sdu_size = msg_sdu_size;
      } else {
        sdu_size = sdu_lenP;
      }

      message_p = itti_alloc_new_message (TASK_MAC_ENB, RRC_MAC_CCCH_DATA_IND);
      RRC_MAC_CCCH_DATA_IND (message_p).frame     = frameP;
      RRC_MAC_CCCH_DATA_IND (message_p).sub_frame = sub_frameP;
      RRC_MAC_CCCH_DATA_IND (message_p).rnti      = rntiP;
      RRC_MAC_CCCH_DATA_IND (message_p).sdu_size  = sdu_size;
      RRC_MAC_CCCH_DATA_IND (message_p).CC_id = CC_id;
      memset (RRC_MAC_CCCH_DATA_IND (message_p).sdu, 0, CCCH_SDU_SIZE);
      memcpy (RRC_MAC_CCCH_DATA_IND (message_p).sdu, sduP, sdu_size);
      itti_send_msg_to_task (TASK_RRC_ENB, ctxt.instance, message_p);
    }
#else

    if (sdu_lenP > 0) {
      memcpy(Srb_info->Rx_buffer.Payload,sduP,sdu_lenP);
      Srb_info->Rx_buffer.payload_size = sdu_lenP;
      rrc_eNB_decode_ccch_NB(&ctxt, Srb_info, CC_id);
    }

#endif
  return(0);

}


//defined in L2_interface
void NB_mac_eNB_rrc_ul_failure(
		const module_id_t mod_idP,
	    const int CC_idP,
	    const frame_t frameP,
	    const sub_frame_t subframeP,
	    const rnti_t rntiP)
{
  struct rrc_eNB_ue_context_NB_s* ue_context_p = NULL;
  ue_context_p = rrc_eNB_get_ue_context_NB(
                   &eNB_rrc_inst_NB[mod_idP],
                   rntiP);

  if (ue_context_p != NULL) {
    LOG_I(RRC,"Frame %d, Subframe %d: UE %x UL failure, activating timer\n",frameP,subframeP,rntiP);
    ue_context_p->ue_context.ul_failure_timer=1;
  }
  else {
    LOG_W(RRC,"Frame %d, Subframe %d: UL failure: UE %x unknown \n",frameP,subframeP,rntiP);
  }
  NB_rrc_mac_remove_ue(mod_idP,rntiP);
}


//defined in eNB_scheduler_primitives.c
int NB_rrc_mac_remove_ue(
		module_id_t mod_idP,
		rnti_t rntiP)
{
  int i;
  UE_list_t *UE_list = &eNB_mac_inst_NB[mod_idP].UE_list;
  int UE_id = find_UE_id(mod_idP,rntiP); //may should be changed
  int pCC_id;

  if (UE_id == -1) {
printf("MAC: cannot remove UE rnti %x\n", rntiP);
    LOG_W(MAC,"NB_rrc_mac_remove_ue: UE %x not found\n", rntiP);
    mac_phy_remove_ue(mod_idP, rntiP); //PHY/defs.h
    return 0;
  }

  pCC_id = UE_PCCID(mod_idP,UE_id);

printf("MAC: remove UE %d rnti %x\n", UE_id, rntiP);
  LOG_I(MAC,"Removing UE %d from Primary CC_id %d (rnti %x)\n",UE_id,pCC_id, rntiP);
  dump_ue_list(UE_list,0); //may should be changed

  UE_list->active[UE_id] = FALSE;
  UE_list->num_UEs--;

  // clear all remaining pending transmissions
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID0]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID1]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID2]  = 0;
  UE_list->UE_template[pCC_id][UE_id].bsr_info[LCGID3]  = 0;

  UE_list->UE_template[pCC_id][UE_id].ul_SR             = 0;
  UE_list->UE_template[pCC_id][UE_id].rnti              = NOT_A_RNTI;
  UE_list->UE_template[pCC_id][UE_id].ul_active         = FALSE;
  eNB_ulsch_info[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_ulsch_info[mod_idP][pCC_id][UE_id].status                      = S_UL_NONE;
  eNB_dlsch_info[mod_idP][pCC_id][UE_id].rnti                        = NOT_A_RNTI;
  eNB_dlsch_info[mod_idP][pCC_id][UE_id].status                      = S_DL_NONE;

  NB_mac_phy_remove_ue(mod_idP,rntiP);

  // check if this has an RA process active
  RA_TEMPLATE_NB *RA_template;
  for (i=0;i<NB_RA_PROC_MAX;i++) {

    RA_template = (RA_TEMPLATE_NB *)&eNB_mac_inst[mod_idP].common_channels[pCC_id].RA_template[i];
    if (RA_template->rnti == rntiP){
      RA_template->RA_active=FALSE;
      RA_template->generate_rar=0;
      RA_template->generate_Msg4=0;
      RA_template->wait_ack_Msg4=0;
      RA_template->timing_offset=0;
      RA_template->RRC_timer=20;
      RA_template->rnti = 0;
      //break;
    }
  }

  return 0;
}

//defined in L2_interface
void NB_mac_eNB_rrc_ul_in_sync(
				const module_id_t mod_idP,
			    const int CC_idP,
			    const frame_t frameP,
			    const sub_frame_t subframeP,
			    const rnti_t rntiP)
{
  struct rrc_eNB_ue_context_NB_s* ue_context_p = NULL;
  ue_context_p = rrc_eNB_get_ue_context_NB(
                   &eNB_rrc_inst_NB[mod_idP],
                   rntiP);

  if (ue_context_p != NULL) {
    LOG_I(RRC,"Frame %d, Subframe %d: UE %x to UL in synch\n",
          frameP, subframeP, rntiP);
    ue_context_p->ue_context.ul_failure_timer = 0;
  } else {
    LOG_E(RRC,"Frame %d, Subframe %d: UE %x unknown \n",
          frameP, subframeP, rntiP);
  }
}

//defined in L2_interface
int NB_mac_eNB_get_rrc_status(
  const module_id_t Mod_idP,
  const rnti_t      rntiP
)
{
  struct rrc_eNB_ue_context_s* ue_context_p = NULL;
  ue_context_p = rrc_eNB_get_ue_context_NB(
                   &eNB_rrc_inst_NB[Mod_idP],
                   rntiP);

  if (ue_context_p != NULL) {
    return(ue_context_p->ue_context.Status);
  } else {
    return RRC_INACTIVE;
  }
}

/*----------------------------------RRC-PDCP--------------------------------------*/

//FIXME for the moment we not configure PDCP for SRB1bis (but used as it is SRB1)
//defined in pdcp.c
boolean_t NB_rrc_pdcp_config_asn1_req (
  const protocol_ctxt_t* const  ctxt_pP,
  SRB_ToAddModList_NB_r13_t  *const srb2add_list_pP,
  DRB_ToAddModList_NB_r13_t  *const drb2add_list_pP,
  DRB_ToReleaseList_NB_r13_t *const drb2release_list_pP,
  const uint8_t                   security_modeP,
  uint8_t                  *const kRRCenc_pP,
  uint8_t                  *const kRRCint_pP,
  uint8_t                  *const kUPenc_pP,
  rb_id_t                 *const defaultDRB,
  long						LCID
)
{
  long int        lc_id          = 0;
  DRB_Identity_t  srb_id         = 0;
  long int        mch_id         = 0;
  rlc_mode_t      rlc_type       = RLC_MODE_NONE;
  DRB_Identity_t  drb_id         = 0;
  DRB_Identity_t *pdrb_id_p      = NULL;
  uint8_t         drb_sn         = 7; //SN for NB-IoT DRB TS 36.323 Table 6.3.2.1 (user data plane)
  uint8_t         srb_sn         = 5; // fixed sn for SRBs (control plane)
  uint8_t         drb_report     = 0; //may not supported in NB-IoT
  long int        cnt            = 0;
  uint16_t        header_compression_profile = 0; //no compression
  config_action_t action         = CONFIG_ACTION_ADD;
  SRB_ToAddMod_NB_r13_t *srb_toaddmod_p = NULL;
  DRB_ToAddMod_NB_r13_t *drb_toaddmod_p = NULL;
  pdcp_t         *pdcp_p         = NULL;

  hash_key_t      key            = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t  h_rc;
  hash_key_t      key_defaultDRB = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t  h_defaultDRB_rc;

  LOG_T(PDCP, PROTOCOL_CTXT_FMT" %s() SRB2ADD %p DRB2ADD %p DRB2RELEASE %p\n",
        PROTOCOL_CTXT_ARGS(ctxt_pP),
        __FUNCTION__,
        srb2add_list_pP,
        drb2add_list_pP,
        drb2release_list_pP);

  // srb2add_list does not define pdcp config, we use rlc info to setup the pdcp dcch0 and dcch1 channels

  if (srb2add_list_pP != NULL) {

	  if(LCID == DCCH0) //SRB1bis
	  	{
		  LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" PDCP Configiration for SRB1bis not allowed\n");
		  return TRUE;
		}
	  else
	   {
		  //is SRB1
		  srb_id = DCCH1;
		  lc_id = srb_id;
	   }

    for (cnt=0; cnt<srb2add_list_pP->list.count; cnt++) {
      srb_toaddmod_p = srb2add_list_pP->list.array[cnt];
      rlc_type = RLC_MODE_AM; //only mode available in NB-IOT

      key = PDCP_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, srb_id, SRB_FLAG_YES);
      h_rc = hashtable_get(pdcp_coll_p, key, (void**)&pdcp_p);

      //for the moment left the same logic
      if (h_rc == HASH_TABLE_OK) {
        action = CONFIG_ACTION_MODIFY;
        LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_MODIFY key 0x%"PRIx64"\n",
              PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
              key);
      } else {

        action = CONFIG_ACTION_ADD;
        pdcp_p = calloc(1, sizeof(pdcp_t));
        h_rc = hashtable_insert(pdcp_coll_p, key, pdcp_p); //if all ok should get h_rc = HASH_TABLE_OK

        if (h_rc != HASH_TABLE_OK) {
          LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_ADD key 0x%"PRIx64" FAILED\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
                key);
          free(pdcp_p);
          return TRUE;

      } else {
          LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_ADD key 0x%"PRIx64"\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
                key);
        }
      }

      if (srb_toaddmod_p->rlc_Config_r13) {
        switch (srb_toaddmod_p->rlc_Config_r13->present) {
        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_NOTHING:
          break;

        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_explicitValue:
          switch (srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.present) {
          case RLC_Config_NB_r13_PR_NOTHING:
            break;

          default:
        	  //configure the pdcp depend on the action
        	  //is a switch case on the "action" parameter
            NB_pdcp_config_req_asn1 (
              ctxt_pP,
              pdcp_p,
              SRB_FLAG_YES,
              rlc_type,
              action,
              lc_id,
              srb_id,
              srb_sn, //rb_sn
              0, // drb_report
              0, // header compression
              security_modeP,
              kRRCenc_pP,
              kRRCint_pP,
              kUPenc_pP);
            break;
          }

          break;

        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_defaultValue:
        	NB_pdcp_config_req_asn1 (
        	              ctxt_pP,
        	              pdcp_p,
        	              SRB_FLAG_YES,
        	              rlc_type,
        	              action,
        	              lc_id,
        	              srb_id,
        	              srb_sn,
        	              0, // drb_report
        	              0, // header compression
        	              security_modeP,
        	              kRRCenc_pP,
        	              kRRCint_pP,
        	              kUPenc_pP);
          // already the default values
          break;

        default:
          DevParam(srb_toaddmod_p->rlc_Config_r13->present, ctxt_pP->module_id, ctxt_pP->rnti);
          break;
        }
      }
    }
  }

  // reset the action

  if (drb2add_list_pP != NULL) {
    for (cnt=0; cnt<drb2add_list_pP->list.count; cnt++) {

      drb_toaddmod_p = drb2add_list_pP->list.array[cnt];

      drb_id = drb_toaddmod_p->drb_Identity_r13;// + drb_id_offset;
      if (drb_toaddmod_p->logicalChannelIdentity_r13) {
        lc_id = *(drb_toaddmod_p->logicalChannelIdentity_r13);
      } else {
        LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" logicalChannelIdentity is missing in DRB-ToAddMod-NB information element!\n",
              PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p));
        continue;
      }

      if (lc_id == 1 || lc_id == 2 || lc_id == 3) {
        LOG_E(RLC, PROTOCOL_CTXT_FMT" logicalChannelIdentity = %ld is invalid in RRC message when adding DRB!\n", PROTOCOL_CTXT_ARGS(ctxt_pP), lc_id);
        continue;
      }

      DevCheck4(drb_id < maxDRB_NB_r13, drb_id, maxDRB_NB_r13, ctxt_pP->module_id, ctxt_pP->rnti);
      key = PDCP_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, drb_id, SRB_FLAG_NO);
      h_rc = hashtable_get(pdcp_coll_p, key, (void**)&pdcp_p);

      if (h_rc == HASH_TABLE_OK) {
        action = CONFIG_ACTION_MODIFY;
        LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_MODIFY key 0x%"PRIx64"\n",
              PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
              key);

      } else {
        action = CONFIG_ACTION_ADD;
        pdcp_p = calloc(1, sizeof(pdcp_t));
        h_rc = hashtable_insert(pdcp_coll_p, key, pdcp_p);

        // save the first configured DRB-ID as the default DRB-ID
        if ((defaultDRB != NULL) && (*defaultDRB == drb_id)) {
          key_defaultDRB = PDCP_COLL_KEY_DEFAULT_DRB_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag);
          h_defaultDRB_rc = hashtable_insert(pdcp_coll_p, key_defaultDRB, pdcp_p);
        } else {
          h_defaultDRB_rc = HASH_TABLE_OK; // do not trigger any error handling if this is not a default DRB
        }

        if (h_defaultDRB_rc != HASH_TABLE_OK) {
          LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_ADD ADD default DRB key 0x%"PRIx64" FAILED\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
                key_defaultDRB);
          free(pdcp_p);
          return TRUE;
        } else if (h_rc != HASH_TABLE_OK) {
          LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_ADD ADD key 0x%"PRIx64" FAILED\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
                key);
          free(pdcp_p);
          return TRUE;
        } else {
          LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_ADD ADD key 0x%"PRIx64"\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
                key);
         }
      }

      if (drb_toaddmod_p->pdcp_Config_r13) {
        if (drb_toaddmod_p->pdcp_Config_r13->discardTimer_r13) {
          // set the value of the timer
        }

        if (drb_toaddmod_p->pdcp_Config_r13) {
          //Status report operation is not defined for NB-IoT
          rlc_type = RLC_MODE_AM;
        }


        switch (drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.present) {
        case PDCP_Config_NB_r13__headerCompression_r13_PR_NOTHING:
        case PDCP_Config_NB_r13__headerCompression_r13_PR_notUsed:
          header_compression_profile=0x0;
          break;

        case PDCP_Config_NB_r13__headerCompression_r13_PR_rohc:

          // parse the struc and get the rohc profile
          //XXX MP: for NB-IoT only some profiles are defined in PDCP-Config-NB IE
          if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0002) {
            header_compression_profile=0x0002;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0003) {
            header_compression_profile=0x0003;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0004) {
            header_compression_profile=0x0004;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0006) {
            header_compression_profile=0x0006;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0102) {
            header_compression_profile=0x0102;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0103) {
            header_compression_profile=0x0103;
          } else if(drb_toaddmod_p->pdcp_Config_r13->headerCompression_r13.choice.rohc.profiles_r13.profile0x0104) {
            header_compression_profile=0x0104;
          } else {
            header_compression_profile=0x0;
            LOG_W(PDCP,"unknown header compression profile\n");
          }

          // set the applicable profile
          break;

        default:
          LOG_W(PDCP,PROTOCOL_PDCP_CTXT_FMT"[RB %ld] unknown drb_toaddmod->PDCP_Config_NB->headerCompression->present \n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_p), drb_id);
          break;
        }

        NB_pdcp_config_req_asn1 (
          ctxt_pP,
          pdcp_p,
          SRB_FLAG_NO,
          rlc_type,
          action,
          lc_id,
          drb_id,
          drb_sn,
          drb_report,
          header_compression_profile,
          security_modeP,
          kRRCenc_pP,
          kRRCint_pP,
          kUPenc_pP);
      }
    }
  }

  if (drb2release_list_pP != NULL) {
    for (cnt=0; cnt<drb2release_list_pP->list.count; cnt++) {
      pdrb_id_p = drb2release_list_pP->list.array[cnt];
      drb_id =  *pdrb_id_p;
      key = PDCP_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, srb_id, SRB_FLAG_NO);
      h_rc = hashtable_get(pdcp_coll_p, key, (void**)&pdcp_p);

      if (h_rc != HASH_TABLE_OK) {
        LOG_E(PDCP, PROTOCOL_CTXT_FMT" PDCP REMOVE FAILED drb_id %ld\n",
              PROTOCOL_CTXT_ARGS(ctxt_pP),
              drb_id);
        continue;
      }
      lc_id = pdcp_p->lcid;

      action = CONFIG_ACTION_REMOVE;
      NB_pdcp_config_req_asn1 (
        ctxt_pP,
        pdcp_p,
        SRB_FLAG_NO,
        rlc_type,
        action,
        lc_id,
        drb_id,
        0, //drb_sn
        0,
        0,
        security_modeP,
        kRRCenc_pP,
        kRRCint_pP,
        kUPenc_pP);
      h_rc = hashtable_remove(pdcp_coll_p, key);

      if ((defaultDRB != NULL) && (*defaultDRB == drb_id)) {
        // default DRB being removed. nevertheless this shouldn't happen as removing default DRB is not allowed in standard
        key_defaultDRB = PDCP_COLL_KEY_DEFAULT_DRB_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag);
        h_defaultDRB_rc = hashtable_get(pdcp_coll_p, key_defaultDRB, (void**)&pdcp_p);

        if (h_defaultDRB_rc == HASH_TABLE_OK) {
          h_defaultDRB_rc = hashtable_remove(pdcp_coll_p, key_defaultDRB);
        } else {
          LOG_E(PDCP, PROTOCOL_CTXT_FMT" PDCP REMOVE FAILED default DRB\n", PROTOCOL_CTXT_ARGS(ctxt_pP));
        }
      } else {
        key_defaultDRB = HASH_TABLE_OK; // do not trigger any error handling if this is not a default DRB
      }
    }
  }
  return 0;
}

//defined in pdcp.c --> should be called only by a SRB1 (is internal to PDCP so is not an interface)
//-----------------------------------------------------------------------------
boolean_t
NB_pdcp_config_req_asn1 (
  const protocol_ctxt_t* const  ctxt_pP,
  pdcp_t         * const        pdcp_pP,
  const srb_flag_t              srb_flagP,
  const rlc_mode_t              rlc_modeP, //rlc_type
  const config_action_t         actionP,
  const uint16_t                lc_idP, // 1 = SRB1 // 3 = SRB1bis // >= 4 for DRBs
  const rb_id_t                 rb_idP,
  const uint8_t                 rb_snP, //5 if srb_sn // 7 is drb_sn // 0 if drb_sn to be removed
  const uint8_t                 rb_reportP, //not for SRBand not for NB-IOT
  const uint16_t                header_compression_profileP, //not for SRB only DRB
  const uint8_t                 security_modeP,
  uint8_t         *const        kRRCenc_pP,
  uint8_t         *const        kRRCint_pP,
  uint8_t         *const        kUPenc_pP)
//-----------------------------------------------------------------------------
{

  switch (actionP) {
  case CONFIG_ACTION_ADD:

      pdcp_pP->is_ue = FALSE; //is an eNB PDCP
      pdcp_eNB_UE_instance_to_rnti[pdcp_eNB_UE_instance_to_rnti_index] = ctxt_pP->rnti;

    pdcp_pP->is_srb                     = (srb_flagP == SRB_FLAG_YES) ? TRUE : FALSE;
    pdcp_pP->lcid                       = lc_idP;
    pdcp_pP->rb_id                      = rb_idP;
    pdcp_pP->header_compression_profile = header_compression_profileP;
    pdcp_pP->status_report              = rb_reportP;


    if (rb_snP == 7) { //the only applicable for NB-IoT RLC-AM DRBs
      pdcp_pP->seq_num_size = PDCP_SN_7BIT;
    } else { //control plane (SRBs)
      pdcp_pP->seq_num_size = PDCP_SN_5BIT;
    }

    //check on RLC mode (in principle should not exist)
    if(rlc_modeP == RLC_MODE_UM)
    {
    	LOG_E(PDCP,"Error: NB-IoT cannot work in RLC-UM mode\n" );
    	return (0);
    }

    pdcp_pP->rlc_mode                         = rlc_modeP;
    pdcp_pP->next_pdcp_tx_sn                  = 0;
    pdcp_pP->next_pdcp_rx_sn                  = 0;
    pdcp_pP->next_pdcp_rx_sn_before_integrity = 0;
    pdcp_pP->tx_hfn                           = 0;
    pdcp_pP->rx_hfn                           = 0;
    pdcp_pP->last_submitted_pdcp_rx_sn        = 4095; //MP: ??
    pdcp_pP->first_missing_pdu                = -1;
    pdcp_pP->rx_hfn_offset                    = 0;

    LOG_N(PDCP, PROTOCOL_PDCP_CTXT_FMT" Action ADD  LCID %d (%s id %d) "
            "configured with SN size %d bits and RLC %s\n",
          PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_pP),
          lc_idP,
	  (srb_flagP == SRB_FLAG_YES) ? "SRB" : "DRB",
          rb_idP,
          pdcp_pP->seq_num_size,
	  (rlc_modeP == RLC_MODE_AM ) ? "AM" : "TM");
    /* Setup security */
    if (security_modeP != 0xff) {
    	//TODO: check this function
      pdcp_config_set_security(
        ctxt_pP,
        pdcp_pP,
        rb_idP,
        lc_idP,
        security_modeP,
        kRRCenc_pP,
        kRRCint_pP,
        kUPenc_pP);
    }
    break;

  case CONFIG_ACTION_MODIFY:
    DevAssert(pdcp_pP != NULL);
    pdcp_pP->header_compression_profile=header_compression_profileP;
    pdcp_pP->status_report = rb_reportP;
    pdcp_pP->rlc_mode = rlc_modeP;

    /* Setup security */
    if (security_modeP != 0xff) {
      pdcp_config_set_security(
        ctxt_pP,
        pdcp_pP,
        rb_idP,
        lc_idP,
        security_modeP,
        kRRCenc_pP,
        kRRCint_pP,
        kUPenc_pP);
    }

    if (rb_snP == 7) {
      pdcp_pP->seq_num_size = 7;
    } else {
      pdcp_pP->seq_num_size=5;
    }

    LOG_N(PDCP,PROTOCOL_PDCP_CTXT_FMT" Action MODIFY LCID %d "
            "RB id %d reconfigured with SN size %d and RLC %s \n",
          PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_pP),
          lc_idP,
          rb_idP,
          rb_snP,
            (rlc_modeP == RLC_MODE_AM) ? "AM" : "TM");
    break;

  case CONFIG_ACTION_REMOVE:
    DevAssert(pdcp_pP != NULL);
//#warning "TODO pdcp_module_id_to_rnti"
    //pdcp_module_id_to_rnti[ctxt_pP.module_id ][dst_id] = NOT_A_RNTI;
    LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT" CONFIG_ACTION_REMOVE LCID %d RBID %d configured\n",
          PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_pP),
          lc_idP,
          rb_idP);

    /* Security keys */
    if (pdcp_pP->kUPenc != NULL) {
      free(pdcp_pP->kUPenc);
    }

    if (pdcp_pP->kRRCint != NULL) {
      free(pdcp_pP->kRRCint);
    }

    if (pdcp_pP->kRRCenc != NULL) {
      free(pdcp_pP->kRRCenc);
    }

    memset(pdcp_pP, 0, sizeof(pdcp_t));
    break;

  case CONFIG_ACTION_SET_SECURITY_MODE:
    pdcp_config_set_security(
      ctxt_pP,
      pdcp_pP,
      rb_idP,
      lc_idP,
      security_modeP,
      kRRCenc_pP,
      kRRCint_pP,
      kUPenc_pP);
    break;

  default:
    DevParam(actionP, ctxt_pP->module_id, ctxt_pP->rnti);
    break;
  }

  return 0;
}

//defined in L2_interface/pdcp.c
//FIXME SRB1bis should bypass the pdcp
//Distinction between different SRBs will be done by means of rd_id
uint8_t NB_rrc_data_req(
  const protocol_ctxt_t*   const ctxt_pP,
  const rb_id_t                  rb_idP,
  const mui_t                    muiP,
  const confirm_t                confirmP,
  const sdu_size_t               sdu_sizeP,
  uint8_t*                 const buffer_pP,
  const pdcp_transmission_mode_t modeP //when go through SRB1bis should be set as Transparent mode
)
{
  MSC_LOG_TX_MESSAGE(
    ctxt_pP->enb_flag ? MSC_RRC_ENB : MSC_RRC_UE,
    ctxt_pP->enb_flag ? MSC_PDCP_ENB : MSC_PDCP_UE,
    buffer_pP,
    sdu_sizeP,
    MSC_AS_TIME_FMT"RRC_DCCH_DATA_REQ UE %x MUI %d size %u",
    MSC_AS_TIME_ARGS(ctxt_pP),
    ctxt_pP->rnti,
    muiP,
    sdu_sizeP);

  //check srb1bis (preliminar)
  if(rb_idP == 3 && modeP != PDCP_TRANSMISSION_MODE_TRANSPARENT)
	  LOG_E(PDCP,"ERROR: SRB1bis should go through PDCP transparently");

  //FIXME: the following type of messages are used by the pdcp_run function in pdcp.c --> should be left?(put out from ITTI)
#if defined(ENABLE_ITTI)
  {
    MessageDef *message_p;
    // Uses a new buffer to avoid issue with PDCP buffer content that could be changed by PDCP (asynchronous message handling).
    uint8_t *message_buffer;

    message_buffer = itti_malloc (
                       ctxt_pP->enb_flag ? TASK_RRC_ENB : TASK_RRC_UE,
                       ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE,
                       sdu_sizeP);

    memcpy (message_buffer, buffer_pP, sdu_sizeP);

    message_p = itti_alloc_new_message (ctxt_pP->enb_flag ? TASK_RRC_ENB : TASK_RRC_UE, RRC_DCCH_DATA_REQ);
    RRC_DCCH_DATA_REQ (message_p).frame     = ctxt_pP->frame;
    RRC_DCCH_DATA_REQ (message_p).enb_flag  = ctxt_pP->enb_flag;
    RRC_DCCH_DATA_REQ (message_p).rb_id     = rb_idP;
    RRC_DCCH_DATA_REQ (message_p).muip      = muiP;
    RRC_DCCH_DATA_REQ (message_p).confirmp  = confirmP;
    RRC_DCCH_DATA_REQ (message_p).sdu_size  = sdu_sizeP;
    RRC_DCCH_DATA_REQ (message_p).sdu_p     = message_buffer;
    RRC_DCCH_DATA_REQ (message_p).mode      = modeP;
    RRC_DCCH_DATA_REQ (message_p).module_id = ctxt_pP->module_id;
    RRC_DCCH_DATA_REQ (message_p).rnti      = ctxt_pP->rnti;
    RRC_DCCH_DATA_REQ (message_p).eNB_index = ctxt_pP->eNB_index;

    itti_send_msg_to_task (
      ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE,
      ctxt_pP->instance,
      message_p);
    return TRUE; // TODO should be changed to a CNF message later, currently RRC lite does not used the returned value anyway.

  }
#else

  //MP:in this case since is called by RRC for sure we have "SRB_FLAG_YES"
  return NB_pdcp_data_req(
           ctxt_pP,
           SRB_FLAG_YES,
           rb_idP,
           muiP,
           confirmP,
           sdu_sizeP,
           buffer_pP,
           modeP);
#endif
}


//we distinguish the SRBs based on the logical channel id and the transmission mode
boolean_t NB_pdcp_data_req(
  protocol_ctxt_t*  ctxt_pP,
  const srb_flag_t     srb_flagP, //SRB_FLAG_YES if called by RRC
  const rb_id_t        rb_idP,
  const mui_t          muiP,
  const confirm_t      confirmP,
  const sdu_size_t     sdu_buffer_sizeP, //the size of message that i should transmit
  unsigned char *const sdu_buffer_pP,
  const pdcp_transmission_mode_t modeP
)
{

  pdcp_t            *pdcp_p          = NULL;
  uint8_t            i               = 0;
  uint8_t            pdcp_header_len = 0;
  uint8_t            pdcp_tailer_len = 0;
  uint16_t           pdcp_pdu_size   = 0;
  uint16_t           current_sn      = 0;
  mem_block_t       *pdcp_pdu_p      = NULL;
  rlc_op_status_t    rlc_status;
  boolean_t          ret             = TRUE;

  hash_key_t         key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t     h_rc;
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_IN);
  CHECK_CTXT_ARGS(ctxt_pP);

#if T_TRACER
  if (ctxt_pP->enb_flag != ENB_FLAG_NO)
    T(T_ENB_PDCP_DL, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->rnti), T_INT(rb_idP), T_INT(sdu_buffer_sizeP));
#endif

  if (sdu_buffer_sizeP == 0) {
    LOG_W(PDCP, "Handed SDU is of size 0! Ignoring...\n");
    return FALSE;
  }

  /*
   * XXX MAX_IP_PACKET_SIZE is 4096, shouldn't this be MAX SDU size, which is 8188 bytes?
   *
   * MP: (LTE) The maximum supported size of a PDCP SDU (both data or control) is 8188 octets
   * 	 (NB-IoT) The maximum supported size of a PDCP SDU (both data or control) is 1600 octets
   * 	 MAX_IP_PACKET_SIZE = 1500 (typical MTU for IPv4 packet)
   * 	 MAX_IP_PACKET_SIZE = 1514 for Ethernet (1500 bytes IP packet + 14 bytes ethernet header)
   *	---> i think that the check for the MAX_IP_PACKET_SIZE should be left as it is
   */

  if (sdu_buffer_sizeP > MAX_IP_PACKET_SIZE) {
    LOG_E(PDCP, "Requested SDU size (%d) is bigger than that can be handled by PDCP (%u)!\n",
          sdu_buffer_sizeP, MAX_IP_PACKET_SIZE);
    // XXX What does following call do?
    mac_xface->macphy_exit("PDCP sdu buffer size > MAX_IP_PACKET_SIZE");
  }

  //check for MBMS not needed for NB-IoT*/
  //if (modeP == PDCP_TRANSMISSION_MODE_TRANSPARENT) {
    //AssertError (rb_idP < NB_RB_MBMS_MAX, return FALSE, "RB id is too high (%u/%d) %u %u!\n", rb_idP, NB_RB_MBMS_MAX, ctxt_pP->module_id, ctxt_pP->rnti);
  //} else {

    if (srb_flagP) {
      AssertError (rb_idP < 4 , return FALSE, "RB id is too high (%u/%d) %u %u!\n", rb_idP, 3, ctxt_pP->module_id, ctxt_pP->rnti);
    } else {//is a DRB
    	//FIXME: check if correct
      AssertError (rb_idP < 3+ maxDRB_NB_r13, return FALSE, "RB id is too high (%u/%d) %u %u!\n", rb_idP, maxDRB_NB_r13, ctxt_pP->module_id, ctxt_pP->rnti);
    }


  key = PDCP_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  h_rc = hashtable_get(pdcp_coll_p, key, (void**)&pdcp_p);

  if (h_rc != HASH_TABLE_OK) {
	  //if not included in the hashtable means that should be transparent mode
    if (modeP != PDCP_TRANSMISSION_MODE_TRANSPARENT) {
      LOG_W(PDCP, PROTOCOL_CTXT_FMT" Instance is not configured for rb_id %d Ignoring SDU...\n",
	    PROTOCOL_CTXT_ARGS(ctxt_pP),
	    rb_idP);
      ctxt_pP->configured=FALSE;
      return FALSE;
    }
  }else{
    // instance for a given RB is configured
    ctxt_pP->configured=TRUE;
  }

  //XXX Start/stop meas are used for measuring the CPU time

  if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
    start_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
  } else {
    start_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
  }

  //PDCP transparent mode for SRB1bis

  //MP: Check may not useful
  if((modeP == PDCP_TRANSMISSION_MODE_TRANSPARENT && rb_idP != 3) || (modeP != PDCP_TRANSMISSION_MODE_TRANSPARENT && rb_idP == 3 ))
	  LOG_E(PDCP, "PDCP_TRANSMISSION_MODE_TRANSPARENT set not for SRB1bis");

  if(modeP == PDCP_TRANSMISSION_MODE_TRANSPARENT && rb_idP == 3 && srb_flagP == SRB_FLAG_YES)
  {
	  LOG_D(PDCP, " [SRB1bis TM] Asking for a new mem_block of size %d\n",sdu_buffer_sizeP);
	      pdcp_pdu_p = get_free_mem_block(sdu_buffer_sizeP, __func__);

	   if (pdcp_pdu_p != NULL) {

	            memcpy(&pdcp_pdu_p->data[0], sdu_buffer_pP, sdu_buffer_sizeP);

	      #if defined(DEBUG_PDCP_PAYLOAD)
	            rlc_util_print_hex_octets(PDCP,
	                                      (unsigned char*)&pdcp_pdu_p->data[0],
	                                      sdu_buffer_sizeP);
	      #endif

	      rlc_status = NB_rlc_data_req(ctxt_pP, srb_flagP, rb_idP, muiP, confirmP, sdu_buffer_sizeP, pdcp_pdu_p);
	      //MP: if all ok rlc_status = RLC_OP_STATUS_OK

	   } else {
	         rlc_status = RLC_OP_STATUS_OUT_OF_RESSOURCES;
	         LOG_W(PDCP,PROTOCOL_CTXT_FMT" PDCP_DATA_REQ SDU for SRB1bis DROPPED, OUT OF MEMORY \n",
	               PROTOCOL_CTXT_ARGS(ctxt_pP));
	   #if defined(STOP_ON_IP_TRAFFIC_OVERLOAD)
	         AssertFatal(0, PROTOCOL_CTXT_FMT"[RB %u] PDCP_DATA_REQ for SRB1bis SDU DROPPED, OUT OF MEMORY \n",
	                     PROTOCOL_CTXT_ARGS(ctxt_pP),
	                     rb_idP);
	   #endif
	   }
  }
  else { //not transparent mode (SRB1 or DRBs)

    // calculate the pdcp header and trailer size

    if (srb_flagP) { //control plane SDU (SRB)
      pdcp_header_len = PDCP_CONTROL_PLANE_DATA_PDU_SN_SIZE;
      pdcp_tailer_len = PDCP_CONTROL_PLANE_DATA_PDU_MAC_I_SIZE;
    } else { // data PDU (DRBs)
      pdcp_header_len = PDCP_USER_PLANE_DATA_PDU_SHORT_SN_HEADER_SIZE; //Only 7bit SN allowed for NB-IoT
      pdcp_tailer_len = 0;
    }

    pdcp_pdu_size = sdu_buffer_sizeP + pdcp_header_len + pdcp_tailer_len;

    LOG_D(PDCP, PROTOCOL_PDCP_CTXT_FMT"Data request notification  pdu size %d (header%d, trailer%d)\n",
          PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_p),
          pdcp_pdu_size,
          pdcp_header_len,
          pdcp_tailer_len);

    /*
     * Allocate a new block for the new PDU (i.e. PDU header and SDU payload)
     */
    pdcp_pdu_p = get_free_mem_block(pdcp_pdu_size, __func__);

    if (pdcp_pdu_p != NULL) {
      /*
       * Create a Data PDU with header and append data
       *
       * Place User Plane PDCP Data PDU header first
       */

      if (srb_flagP) { //MP: Control plane PDCP Data PDU --> 5 bit SN + 3 bit reserved + 4 byte (MAC-I)
        pdcp_control_plane_data_pdu_header pdu_header;
        pdu_header.sn = pdcp_get_next_tx_seq_number(pdcp_p);
        current_sn = pdu_header.sn;
        memset(&pdu_header.mac_i[0],0,PDCP_CONTROL_PLANE_DATA_PDU_MAC_I_SIZE);
        memset(&pdcp_pdu_p->data[sdu_buffer_sizeP + pdcp_header_len],0,PDCP_CONTROL_PLANE_DATA_PDU_MAC_I_SIZE);

        if (pdcp_serialize_control_plane_data_pdu_with_SRB_sn_buffer((unsigned char*)pdcp_pdu_p->data, &pdu_header) == FALSE) {
          LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" Cannot fill PDU buffer with relevant header fields!\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_p));

          if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
            stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
          } else {
            stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
          }

          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_OUT);
          return FALSE;
        }
      } else { //MP: user plane (DRBs) --> NB-IoT use only short SN with 7 bits
    	  pdcp_user_plane_data_pdu_header_with_short_sn pdu_header;
        pdu_header.dc = (modeP == PDCP_TRANSMISSION_MODE_DATA) ? PDCP_DATA_PDU_BIT_SET :  PDCP_CONTROL_PDU_BIT_SET;
        pdu_header.sn = pdcp_get_next_tx_seq_number(pdcp_p);
        current_sn = pdu_header.sn ;

        //TODO: should change this function for NB-IoT --> define for short sn
        if (pdcp_serialize_user_plane_data_pdu_with_long_sn_buffer((unsigned char*)pdcp_pdu_p->data, &pdu_header) == FALSE) {
          LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" Cannot fill PDU buffer with relevant header fields!\n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_p));

          if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
            stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
          } else {
            stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
          }

          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_OUT);
          return FALSE;
        }
      }

      /*
       * Validate incoming sequence number, there might be a problem with PDCP initialization
       */
      if (current_sn > pdcp_calculate_max_seq_num_for_given_size(pdcp_p->seq_num_size)) {
        LOG_E(PDCP, PROTOCOL_PDCP_CTXT_FMT" Generated sequence number (%"PRIu16") is greater than a sequence number could ever be!\n"\
              "There must be a problem with PDCP initialization, ignoring this PDU...\n",
              PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP,pdcp_p),
              current_sn);

        free_mem_block(pdcp_pdu_p, __func__);

        if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
          stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
        } else {
          stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
        }

        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_OUT);
        return FALSE;
      }

      LOG_D(PDCP, "Sequence number %d is assigned to current PDU\n", current_sn);

      /* Then append data... */
      memcpy(&pdcp_pdu_p->data[pdcp_header_len], sdu_buffer_pP, sdu_buffer_sizeP);

      //For control plane data that are not integrity protected,
      // the MAC-I field is still present and should be padded with padding bits set to 0.
      // NOTE: user-plane data are never integrity protected
      for (i=0; i<pdcp_tailer_len; i++) {
        pdcp_pdu_p->data[pdcp_header_len + sdu_buffer_sizeP + i] = 0x00;// pdu_header.mac_i[i];
      }

#if defined(ENABLE_SECURITY)

      if ((pdcp_p->security_activated != 0) &&
          (((pdcp_p->cipheringAlgorithm) != 0) ||
           ((pdcp_p->integrityProtAlgorithm) != 0))) {

        if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
          start_meas(&eNB_pdcp_stats[ctxt_pP->module_id].apply_security);
        } else {
          start_meas(&UE_pdcp_stats[ctxt_pP->module_id].apply_security);
        }

        //FIXME also this function should be checked for NB-IoT
        pdcp_apply_security(ctxt_pP,
                            pdcp_p,
                            srb_flagP,
                            rb_idP % maxDRB_NB_r13,
                            pdcp_header_len,
                            current_sn,
                            pdcp_pdu_p->data,
                            sdu_buffer_sizeP);

        if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
          stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].apply_security);
        } else {
          stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].apply_security);
        }
      }

#endif

      /* Print octets of outgoing data in hexadecimal form */
      LOG_D(PDCP, "Following content with size %d will be sent over RLC (PDCP PDU header is the first two bytes)\n",
            pdcp_pdu_size);
      //util_print_hex_octets(PDCP, (unsigned char*)pdcp_pdu_p->data, pdcp_pdu_size);
      //util_flush_hex_octets(PDCP, (unsigned char*)pdcp_pdu->data, pdcp_pdu_size);
    } else {
      LOG_E(PDCP, "Cannot create a mem_block for a PDU!\n");

      if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
        stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
      } else {
        stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
      }

#if defined(STOP_ON_IP_TRAFFIC_OVERLOAD)
      AssertFatal(0, "[FRAME %5u][%s][PDCP][MOD %u/%u][RB %u] PDCP_DATA_REQ SDU DROPPED, OUT OF MEMORY \n",
                  ctxt_pP->frame,
                  (ctxt_pP->enb_flag) ? "eNB" : "UE",
                  ctxt_pP->enb_module_id,
                  ctxt_pP->ue_module_id,
                  rb_idP);
#endif
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_OUT);
      return FALSE;
    }

    /*
     * Ask sublayer to transmit data and check return value
     * to see if RLC succeeded
     */
#ifdef PDCP_MSG_PRINT
    int i=0;
    LOG_F(PDCP,"[MSG] PDCP DL %s PDU on rb_id %d\n", (srb_flagP)? "CONTROL" : "DATA", rb_idP);

    for (i = 0; i < pdcp_pdu_size; i++) {
      LOG_F(PDCP,"%02x ", ((uint8_t*)pdcp_pdu_p->data)[i]);
    }

    LOG_F(PDCP,"\n");
#endif
    rlc_status = NB_rlc_data_req(ctxt_pP, srb_flagP,rb_idP, muiP, confirmP, pdcp_pdu_size, pdcp_pdu_p);

  }

  switch (rlc_status) {
  case RLC_OP_STATUS_OK:
    LOG_D(PDCP, "Data sending request over RLC succeeded!\n");
    ret=TRUE;
    break;

  case RLC_OP_STATUS_BAD_PARAMETER:
    LOG_W(PDCP, "Data sending request over RLC failed with 'Bad Parameter' reason!\n");
    ret= FALSE;
    break;

  case RLC_OP_STATUS_INTERNAL_ERROR:
    LOG_W(PDCP, "Data sending request over RLC failed with 'Internal Error' reason!\n");
    ret= FALSE;
    break;

  case RLC_OP_STATUS_OUT_OF_RESSOURCES:
    LOG_W(PDCP, "Data sending request over RLC failed with 'Out of Resources' reason!\n");
    ret= FALSE;
    break;

  default:
    LOG_W(PDCP, "RLC returned an unknown status code after PDCP placed the order to send some data (Status Code:%d)\n", rlc_status);
    ret= FALSE;
    break;
  }

  if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
    stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_req);
  } else {
    stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_req);
  }

  /*
   * Control arrives here only if rlc_data_req() returns RLC_OP_STATUS_OK
   * so we return TRUE afterwards
   */

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_REQ,VCD_FUNCTION_OUT);
  return ret;
}

//defined in L2_interface
void NB_rrc_data_ind(
  const protocol_ctxt_t* const ctxt_pP,
  const rb_id_t                Srb_id,
  const sdu_size_t             sdu_sizeP,
  const uint8_t*   const       buffer_pP,
  const srb1bis_flag_t srb1bis_flag
)
//------------------------------------------------------------------------------
{

  rb_id_t    DCCH_index;
  if(srb1bis_flag == SRB1BIS_FLAG_YES)
	  DCCH_index = 3; //SRb1bis (LCID=3) over DCCH0
  else
	  DCCH_index = Srb_id;

  if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
    LOG_N(RRC, "[UE %x] Frame %d: received a DCCH %d message on SRB %d with Size %d from eNB %d\n",
          ctxt_pP->module_id, ctxt_pP->frame, DCCH_index,Srb_id,sdu_sizeP,  ctxt_pP->eNB_index);
  } else {
    LOG_N(RRC, "[eNB %d] Frame %d: received a DCCH %d message on SRB %d with Size %d from UE %x\n",
          ctxt_pP->module_id,
          ctxt_pP->frame,
          DCCH_index,
          Srb_id,
          sdu_sizeP,
          ctxt_pP->rnti);
  }

  //FiXME: MP: we should put out this messages from ITTI
#if defined(ENABLE_ITTI) //From PDCP to RRC
  {
    MessageDef *message_p;
    // Uses a new buffer to avoid issue with PDCP buffer content that could be changed by PDCP (asynchronous message handling).
    uint8_t *message_buffer;

    message_buffer = itti_malloc (ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE, ctxt_pP->enb_flag ? TASK_RRC_ENB : TASK_RRC_UE, sdu_sizeP);
    memcpy (message_buffer, buffer_pP, sdu_sizeP);

    message_p = itti_alloc_new_message (ctxt_pP->enb_flag ? TASK_PDCP_ENB : TASK_PDCP_UE, RRC_DCCH_DATA_IND);
    RRC_DCCH_DATA_IND (message_p).frame      = ctxt_pP->frame;
    RRC_DCCH_DATA_IND (message_p).dcch_index = DCCH_index;
    RRC_DCCH_DATA_IND (message_p).sdu_size   = sdu_sizeP;
    RRC_DCCH_DATA_IND (message_p).sdu_p      = message_buffer;
    RRC_DCCH_DATA_IND (message_p).rnti       = ctxt_pP->rnti;
    RRC_DCCH_DATA_IND (message_p).module_id  = ctxt_pP->module_id;
    RRC_DCCH_DATA_IND (message_p).eNB_index  = ctxt_pP->eNB_index;

    itti_send_msg_to_task (ctxt_pP->enb_flag ? TASK_RRC_ENB : TASK_RRC_UE, ctxt_pP->instance, message_p);
  }
#else

  if (ctxt_pP->enb_flag == ENB_FLAG_YES) {
    rrc_eNB_decode_dcch_NB(
      ctxt_pP,
	  DCCH_index, //--> becomes the srb_id in decode_dcch
      buffer_pP,
      sdu_sizeP);
  } else {
//#warning "LG put 0 to arg4 that is eNB index"
    rrc_ue_decode_dcch(
      ctxt_pP,
      DCCH_index,
      buffer_pP,
      0);
  }

#endif
}

//defined in pdcp.c
boolean_t NB_pdcp_remove_UE(
  const protocol_ctxt_t* const  ctxt_pP
);


/*---------------------------------RRC-RLC-----------------------------------*/

//defined in rlc_rrc.c
rlc_op_status_t NB_rrc_rlc_config_asn1_req (
	const protocol_ctxt_t   * const ctxt_pP,
    const SRB_ToAddModList_NB_r13_t   * const srb2add_listP,
    const DRB_ToAddModList_NB_r13_t   * const drb2add_listP,
    const DRB_ToReleaseList_NB_r13_t  * const drb2release_listP,
	srb1bis_flag_t							srb1bis_flag //may is not so much needed
    )
{
  //-----------------------------------------------------------------------------
  rb_id_t                rb_id           = 0;
  logical_chan_id_t      lc_id           = 0;
  DRB_Identity_t         drb_id          = 0;
  DRB_Identity_t*        pdrb_id         = NULL;
  long int               cnt             = 0;
  const SRB_ToAddMod_NB_r13_t  *srb_toaddmod_p  = NULL;
  const DRB_ToAddMod_NB_r13_t  *drb_toaddmod_p  = NULL;
  rlc_union_t           *rlc_union_p     = NULL;
  hash_key_t             key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;

  //no MBMS in NB-IoT( The RLC config for MBMS was UM)

  /* for no gcc warnings */
  (void)rlc_union_p;
  (void)key;
  (void)h_rc;

  LOG_D(RLC, PROTOCOL_CTXT_FMT" CONFIG REQ ASN1 \n",
        PROTOCOL_CTXT_ARGS(ctxt_pP));

#ifdef OAI_EMU

  CHECK_CTXT_ARGS(ctxt_pP)

#endif

  if (srb2add_listP != NULL) {
		if(srb1bis_flag == SRB1BIS_FLAG_YES){
	    	rb_id = DCCH0;
		}//srb1bis
	    else{
	    	rb_id = DCCH1;
	    }//srb1

	    lc_id = rb_id;

    for (cnt=0; cnt<srb2add_listP->list.count; cnt++) { //MP: should be only 1 iteration

      LOG_D(RLC, "Adding SRB, rb_id %d\n",rb_id);
      srb_toaddmod_p = srb2add_listP->list.array[cnt];

      if (srb_toaddmod_p->rlc_Config_r13) {
        switch (srb_toaddmod_p->rlc_Config_r13->present) {
        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_NOTHING:
          break;

        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_explicitValue:
          switch (srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.present) {
          case RLC_Config_NB_r13_PR_NOTHING:
            break;

          case RLC_Config_NB_r13_PR_am:
            if (NB_rrc_rlc_add_rlc (ctxt_pP, SRB_FLAG_YES, rb_id, lc_id, RLC_MODE_AM) != NULL) {
              NB_config_req_rlc_am_asn1 (
                ctxt_pP,
                SRB_FLAG_YES,
                &srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.choice.am,
                rb_id,
				lc_id);
            } else {
              LOG_E(RLC, PROTOCOL_CTXT_FMT" ERROR IN ALLOCATING SRB %d \n",
                    PROTOCOL_CTXT_ARGS(ctxt_pP),
                    rb_id);
            }

            break;

          default:
            LOG_E(RLC, PROTOCOL_CTXT_FMT" UNKNOWN RLC CONFIG %d \n",
                  PROTOCOL_CTXT_ARGS(ctxt_pP),
                  srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.present);
            break;
          }

          break; //RLC explicit value

        case SRB_ToAddMod_NB_r13__rlc_Config_r13_PR_defaultValue:
//#warning TO DO SRB_ToAddMod__rlc_Config_PR_defaultValue
          LOG_I(RRC, "RLC SRB1/SRB1bis is default value !!\n");
          struct RLC_Config_NB_r13__am  *  config_am_pP = &srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.choice.am;
          config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13     = T_PollRetransmit_NB_r13_ms25000;
          config_am_pP->ul_AM_RLC_r13.maxRetxThreshold_r13 = UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t4;
          config_am_pP->dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 = NULL; //should be disabled

          if (NB_rrc_rlc_add_rlc (ctxt_pP, SRB_FLAG_YES, rb_id, lc_id, RLC_MODE_AM) != NULL) {
            NB_config_req_rlc_am_asn1 (
              ctxt_pP,
              SRB_FLAG_YES,
              &srb_toaddmod_p->rlc_Config_r13->choice.explicitValue.choice.am,
              rb_id,lc_id);
          } else {
            LOG_E(RLC, PROTOCOL_CTXT_FMT" ERROR IN ALLOCATING SRB %d \n",
                  PROTOCOL_CTXT_ARGS(ctxt_pP),
                  rb_id);
          }
          break;

        default:
          ;
        }
      }
    }
  }

  if (drb2add_listP != NULL) {
    for (cnt=0; cnt<drb2add_listP->list.count; cnt++) {
      drb_toaddmod_p = drb2add_listP->list.array[cnt];

      drb_id = drb_toaddmod_p->drb_Identity_r13;
      if (drb_toaddmod_p->logicalChannelIdentity_r13) {
        lc_id = *drb_toaddmod_p->logicalChannelIdentity_r13;
      } else {
        LOG_E(RLC, PROTOCOL_CTXT_FMT" logicalChannelIdentity is missing from drb-ToAddMod information element!\n", PROTOCOL_CTXT_ARGS(ctxt_pP));
        continue;
      }

      if (lc_id == 1 || lc_id == 2 || lc_id == 3) {
        LOG_E(RLC, PROTOCOL_CTXT_FMT" logicalChannelIdentity = %d is invalid in RRC message when adding DRB!\n", PROTOCOL_CTXT_ARGS(ctxt_pP), lc_id);
        continue;
      }

      LOG_D(RLC, "Adding DRB %ld, lc_id %d\n",drb_id,lc_id);


      if (drb_toaddmod_p->rlc_Config_r13) {

        switch (drb_toaddmod_p->rlc_Config_r13->present) {
        case RLC_Config_NB_r13_PR_NOTHING:
          break;

        case RLC_Config_NB_r13_PR_am:
          if (NB_rrc_rlc_add_rlc (ctxt_pP, SRB_FLAG_NO, drb_id, lc_id, RLC_MODE_AM) != NULL) {
            NB_config_req_rlc_am_asn1 (
              ctxt_pP,
              SRB_FLAG_NO,
              &drb_toaddmod_p->rlc_Config_r13->choice.am,
              drb_id, lc_id);
          }

          break;

        default:
          LOG_W(RLC, PROTOCOL_CTXT_FMT"[RB %ld] unknown drb_toaddmod_p->rlc_Config_r13->present \n",
                PROTOCOL_CTXT_ARGS(ctxt_pP),
                drb_id);
        }
      }
    }
  }

  if (drb2release_listP != NULL) {
    for (cnt=0; cnt<drb2release_listP->list.count; cnt++) {
      pdrb_id = drb2release_listP->list.array[cnt];
      NB_rrc_rlc_remove_rlc(
        ctxt_pP,
        SRB_FLAG_NO,
        *pdrb_id);
    }
  }

//MP: MBMS not in NB-IoT

  LOG_D(RLC, PROTOCOL_CTXT_FMT" CONFIG REQ ASN1 END \n",
        PROTOCOL_CTXT_ARGS(ctxt_pP));
  return RLC_OP_STATUS_OK;
}

//defined in rlc_am.c
//from TS 36.331 V14.2.1
uint32_t pollRetransmit_NB_tab[T_PollRetransmit_NB_r13_spare1] = {250,500,1000,2000,3000,4000,6000,10000,15000,25000,40000,60000,90000,120000,180000};
uint32_t maxRetxThreshold_NB_tab[UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t32 +1] = {1,2,3,4,6,8,16,32};
//-----------------------------------------------------------------------------
void NB_config_req_rlc_am_asn1 (
  const protocol_ctxt_t* const         ctxt_pP,
  const srb_flag_t                     srb_flagP,
  const struct RLC_Config_NB_r13__am  * const config_am_pP, //extracted from the srb_toAddMod
  const rb_id_t                        rb_idP,
  const logical_chan_id_t              chan_idP)
{
  rlc_union_t     *rlc_union_p   = NULL;
  rlc_am_entity_t *l_rlc_p         = NULL;
  hash_key_t       key           = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  hashtable_rc_t   h_rc;

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    l_rlc_p = &rlc_union_p->rlc.am;

    //MP: TODO: check if this conditions are correct
    if ((config_am_pP->ul_AM_RLC_r13.maxRetxThreshold_r13 <= UL_AM_RLC_NB_r13__maxRetxThreshold_r13_t32) &&
        (config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13 < T_PollRetransmit_NB_r13_spare1))
		//&&(config_am_pP->dl_AM_RLC_r13.enableStatusReportSN_Gap_r13 == NULL) //MP: may this check is not needed
		{

      MSC_LOG_RX_MESSAGE(
        (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_ENB:MSC_RLC_UE,
        (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RRC_ENB:MSC_RRC_UE,
        NULL,
        0,
        MSC_AS_TIME_FMT" "PROTOCOL_RLC_AM_MSC_FMT" CONFIG-REQ t_PollRetx_NB %u",
        MSC_AS_TIME_ARGS(ctxt_pP),
        PROTOCOL_RLC_AM_MSC_ARGS(ctxt_pP, l_rlc_p),
        pollRetransmit_NB_tab[config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13]);

      LOG_D(RLC, PROTOCOL_RLC_AM_CTXT_FMT" CONFIG_REQ (max_retx_threshold_NB = %d t_poll_retransmit_NB = %d \n",
            PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,l_rlc_p),
            maxRetxThreshold_NB_tab[config_am_pP->ul_AM_RLC_r13.maxRetxThreshold_r13],
            pollRetransmit_NB_tab[config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13]);

      //FIXME: the following function are ok for NB_IoT??
      rlc_am_init(ctxt_pP, l_rlc_p);
      rlc_am_set_debug_infos(ctxt_pP, l_rlc_p, srb_flagP, rb_idP, chan_idP);
      NB_rlc_am_configure(ctxt_pP,
    		  	  	  	  l_rlc_p,
						  maxRetxThreshold_NB_tab[config_am_pP->ul_AM_RLC_r13.maxRetxThreshold_r13],
						  pollRetransmit_NB_tab[config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13],
						  config_am_pP->dl_AM_RLC_r13.enableStatusReportSN_Gap_r13); //may not needed to be passed
    } else {
      MSC_LOG_RX_DISCARDED_MESSAGE(
        (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_ENB:MSC_RLC_UE,
        (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RRC_ENB:MSC_RRC_UE,
        NULL,
        0,
        MSC_AS_TIME_FMT" "PROTOCOL_RLC_AM_MSC_FMT" CONFIG-REQ",
        MSC_AS_TIME_ARGS(ctxt_pP),
        PROTOCOL_RLC_AM_MSC_ARGS(ctxt_pP, l_rlc_p));

      LOG_D(RLC,
            PROTOCOL_RLC_AM_CTXT_FMT"ILLEGAL CONFIG_REQ (max_retx_threshold_NB=%ld t_poll_retransmit_NB=%ld), RLC-AM NOT CONFIGURED\n",
            PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,l_rlc_p),
            config_am_pP->ul_AM_RLC_r13.maxRetxThreshold_r13,
            config_am_pP->ul_AM_RLC_r13.t_PollRetransmit_r13);
    }
  } else {
    LOG_E(RLC, PROTOCOL_RLC_AM_CTXT_FMT"CONFIG_REQ --> RLC NOT FOUND\n",
          PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,l_rlc_p));
  }
}

//defined in rlc_am_init.c
//-----------------------------------------------------------------------------
void
NB_rlc_am_configure(
  const protocol_ctxt_t* const  ctxt_pP,
  rlc_am_entity_t *const        rlc_pP,
  const uint16_t                max_retx_thresholdP,
  const uint16_t                t_poll_retransmitP,
  const uint16_t* const			enableStatusReportSN_Gap
  )
{
  if (rlc_pP->configured == TRUE) {
    LOG_I(RLC, PROTOCOL_RLC_AM_CTXT_FMT"[RECONFIGURE] max_retx_threshold_NB %d t_poll_retransmit_NB %d\n",
          PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,rlc_pP),
          max_retx_thresholdP,
          t_poll_retransmitP
		  );

    //FIXME: rlc_am_entity_t should be modified??

    rlc_pP->max_retx_threshold = max_retx_thresholdP;
    rlc_pP->protocol_state     = RLC_DATA_TRANSFER_READY_STATE;
    rlc_pP->t_poll_retransmit.ms_duration   = t_poll_retransmitP;
    rlc_pP->enableStatusReportSN_Gap = enableStatusReportSN_Gap; //FIXME: new defined in rlc_am_entity


  } else {
    LOG_I(RLC, PROTOCOL_RLC_AM_CTXT_FMT"[CONFIGURE] max_retx_threshold_NB %d t_poll_retransmit_NB %d\n",
          PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,rlc_pP),
          max_retx_thresholdP,
          t_poll_retransmitP
		  //enableStatusReportSN_Gap
    		);

    rlc_pP->max_retx_threshold = max_retx_thresholdP;
    rlc_pP->protocol_state     = RLC_DATA_TRANSFER_READY_STATE;
    rlc_pP->enableStatusReportSN_Gap = enableStatusReportSN_Gap;


    rlc_am_init_timer_poll_retransmit(ctxt_pP, rlc_pP, t_poll_retransmitP);

    rlc_pP->configured = TRUE;
  }

}

//defined in rlc_rrc.c
//-----------------------------------------------------------------------------
rlc_union_t* NB_rrc_rlc_add_rlc   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t        srb_flagP,
  const rb_id_t           rb_idP,
  const logical_chan_id_t chan_idP,
  const rlc_mode_t        rlc_modeP)
{
  //-----------------------------------------------------------------------------
  hash_key_t             key         = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;
  hash_key_t             key_lcid    = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_lcid_rc;
  rlc_union_t           *rlc_union_p = NULL;
#if defined(Rel10) || defined(Rel14)
  logical_chan_id_t      lcid            = 0;
#endif

#ifdef OAI_EMU

  CHECK_CTXT_ARGS(ctxt_pP)

#endif

    AssertFatal (rb_idP < NB_RB_MAX_NB_IOT, "RB id is too high (%u/%d)!\n", rb_idP, NB_RB_MAX);
    AssertFatal (chan_idP < RLC_MAX_LC, "LC id is too high (%u/%d)!\n", chan_idP, RLC_MAX_LC);


  {
    key = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
    key_lcid = RLC_COLL_KEY_LCID_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, chan_idP, srb_flagP);
  }

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    LOG_W(RLC, PROTOCOL_CTXT_FMT"[%s %u] rrc_rlc_add_rlc , already exist %s\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          (srb_flagP) ? "SRB" : "DRB",
          rb_idP,
          (srb_flagP) ? "SRB" : "DRB");
    AssertFatal(rlc_union_p->mode == rlc_modeP, "Error rrc_rlc_add_rlc , already exist but RLC mode differ");
    return rlc_union_p;
  } else if (h_rc == HASH_TABLE_KEY_NOT_EXISTS) {
    rlc_union_p = calloc(1, sizeof(rlc_union_t));
    h_rc = hashtable_insert(rlc_coll_p, key, rlc_union_p);
    h_lcid_rc = hashtable_insert(rlc_coll_p, key_lcid, rlc_union_p);

    if ((h_rc == HASH_TABLE_OK) && (h_lcid_rc == HASH_TABLE_OK)) {

      {
        LOG_I(RLC, PROTOCOL_CTXT_FMT" [%s %u] rrc_rlc_add_rlc  %s\n",
              PROTOCOL_CTXT_ARGS(ctxt_pP),
              (srb_flagP) ? "SRB" : "DRB",
              rb_idP,
              (srb_flagP) ? "SRB" : "DRB");
      }

      rlc_union_p->mode = rlc_modeP;
      return rlc_union_p;
    } else {
      LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] rrc_rlc_add_rlc FAILED %s (add by RB_id=%d; add by LC_id=%d)\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            (srb_flagP) ? "SRB" : "DRB",
            rb_idP,
            (srb_flagP) ? "SRB" : "DRB",
            h_rc, h_lcid_rc);
      free(rlc_union_p);
      rlc_union_p = NULL;
      return NULL;
    }
  } else {
    LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] rrc_rlc_add_rlc , INTERNAL ERROR %s\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          (srb_flagP) ? "SRB" : "DRB",
          rb_idP,
          (srb_flagP) ? "SRB" : "DRB");
  }

  return NULL;
}

//defined in rlc_rrc.c
//-----------------------------------------------------------------------------
rlc_op_status_t NB_rrc_rlc_remove_rlc   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t  srb_flagP,
  const rb_id_t     rb_idP)
{
  //-----------------------------------------------------------------------------
  logical_chan_id_t      lcid            = 0;
  hash_key_t             key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;
  hash_key_t             key_lcid        = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_lcid_rc;
  rlc_union_t           *rlc_union_p = NULL;

#ifdef OAI_EMU
  CHECK_CTXT_ARGS(ctxt_pP)

#endif

  if(rb_idP == 2){
	  LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] NB_rrc_rlc_remove_rlc --> rb_id = 2 (SRB2) should not be used in NB-IoT!\n",
	              PROTOCOL_CTXT_ARGS(ctxt_pP),
	              (srb_flagP) ? "SRB" : "DRB",
	              rb_idP);
  }


  /* for no gcc warnings */
  (void)lcid;

  {
	  //MP: this function know that if i have to search for a DRB the rb_idP that i pass could also be 1,3,... so add something ??
	  //see NB_rrc_rlc_remove_ue
    key = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  }

  AssertFatal (rb_idP < NB_RB_MAX_NB_IOT, "RB id is too high (%u/%d)!\n", rb_idP, NB_RB_MAX_NB_IOT);

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    // also remove the hash-key created by LC-id
	//MP: UM mode not allowed for NB-IoT
    switch (rlc_union_p->mode) {
    case RLC_MODE_AM:
      lcid = rlc_union_p->rlc.am.channel_id;
      break;
    case RLC_MODE_TM:
      lcid = rlc_union_p->rlc.tm.channel_id;
      break;
    default:
      LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] RLC mode is unknown!\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            (srb_flagP) ? "SRB" : "DRB",
            rb_idP);
    }
    //MP:for lcid
    key_lcid = RLC_COLL_KEY_LCID_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, lcid, srb_flagP);
    h_lcid_rc = hashtable_get(rlc_coll_p, key_lcid, (void**)&rlc_union_p);
  } else {
    h_lcid_rc = HASH_TABLE_KEY_NOT_EXISTS;
  }

  if ((h_rc == HASH_TABLE_OK) && (h_lcid_rc == HASH_TABLE_OK)) {
    h_lcid_rc = hashtable_remove(rlc_coll_p, key_lcid);
    h_rc = hashtable_remove(rlc_coll_p, key);
    LOG_D(RLC, PROTOCOL_CTXT_FMT"[%s %u LCID %d] RELEASED %s\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          (srb_flagP) ? "SRB" : "DRB",
          rb_idP, lcid,
          (srb_flagP) ? "SRB" : "DRB");
  } else if ((h_rc == HASH_TABLE_KEY_NOT_EXISTS) || (h_lcid_rc == HASH_TABLE_KEY_NOT_EXISTS)) {
    LOG_D(RLC, PROTOCOL_CTXT_FMT"[%s %u LCID %d] RELEASE : RLC NOT FOUND %s, by RB-ID=%d, by LC-ID=%d\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          (srb_flagP) ? "SRB" : "DRB",
          rb_idP, lcid,
          (srb_flagP) ? "SRB" : "DRB",
          h_rc, h_lcid_rc);
  } else {
    LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u LCID %d] RELEASE : INTERNAL ERROR %s\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          (srb_flagP) ? "SRB" : "DRB",
          rb_idP, lcid,
          (srb_flagP) ? "SRB" : "DRB");
  }

  return RLC_OP_STATUS_OK;
}

//defined in rlc_rrc.c //used only for process_RRCConnectionReconfigurationComplete --> CONFIG_ACTION_REMOVE
//used also for rrc_t310_expiration --> I don't know if it is used (probably not)
rlc_op_status_t NB_rrc_rlc_config_req   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t      srb_flagP,
  const config_action_t actionP,
  const rb_id_t         rb_idP,
  const rlc_info_t      rlc_infoP)
{
  //-----------------------------------------------------------------------------
  //rlc_op_status_t status;

  LOG_D(RLC, PROTOCOL_CTXT_FMT" CONFIG_REQ for RAB %u\n",
        PROTOCOL_CTXT_ARGS(ctxt_pP),
        rb_idP);

#ifdef OAI_EMU

  CHECK_CTXT_ARGS(ctxt_pP)

#endif
  AssertFatal (rb_idP < NB_RB_MAX_NB_IOT, "RB id is too high (%u/%d)!\n", rb_idP, NB_RB_MAX_NB_IOT);

  switch (actionP) {

  //XXX MP:is not used for adding rlc instance??
  case CONFIG_ACTION_ADD:
    if (NB_rrc_rlc_add_rlc(ctxt_pP, srb_flagP, rb_idP, rb_idP, rlc_infoP.rlc_mode) != NULL) {
      return RLC_OP_STATUS_INTERNAL_ERROR;
    }

    // no break, fall to next case
  case CONFIG_ACTION_MODIFY:
    switch (rlc_infoP.rlc_mode) {
    case RLC_MODE_AM:
      LOG_I(RLC, PROTOCOL_CTXT_FMT"[RB %u] MODIFY RB AM\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            rb_idP);

      NB_config_req_rlc_am(
        ctxt_pP,
        srb_flagP,
        &rlc_infoP.rlc.rlc_am_info_NB, //MP: pass the volatile structure for NB_IoT protocol params in rlc_am_init.h
        rb_idP, rb_idP);
      break;

    case RLC_MODE_TM:
      LOG_I(RLC, PROTOCOL_CTXT_FMT"[RB %u] MODIFY RB TM\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            rb_idP);
      NB_config_req_rlc_tm( //MP: TM mode configuration
        ctxt_pP,
        srb_flagP,
        &rlc_infoP.rlc.rlc_tm_info,
        rb_idP,
		rb_idP);
      break;

    default:
      return RLC_OP_STATUS_BAD_PARAMETER;
    }

    break;

  case CONFIG_ACTION_REMOVE:
    return NB_rrc_rlc_remove_rlc(ctxt_pP, srb_flagP, rb_idP);
    break;

  default:
    return RLC_OP_STATUS_BAD_PARAMETER;
  }

  return RLC_OP_STATUS_OK;
}


//defined in rlc_am.c
//-----------------------------------------------------------------------------
void
NB_config_req_rlc_am (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t             srb_flagP,
  rlc_am_info_NB_t  * const       config_am_pP, //XXX: MP: rlc_am_init.c --> this structure has been modified for NB-IoT
  const rb_id_t                rb_idP,
  const logical_chan_id_t      chan_idP
)
{
  rlc_union_t       *rlc_union_p = NULL;
  rlc_am_entity_t *l_rlc_p         = NULL;
  hash_key_t       key           = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  hashtable_rc_t   h_rc;

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    l_rlc_p = &rlc_union_p->rlc.am;
    LOG_D(RLC,
          PROTOCOL_RLC_AM_CTXT_FMT" CONFIG_REQ (max_retx_threshold=%d t_poll_retransmit=%d)\n",
          PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,l_rlc_p),
          config_am_pP->max_retx_threshold,
          config_am_pP->t_poll_retransmit
		  //enableStatusReportSN_Gap_r13
		  );
    rlc_am_init(ctxt_pP, l_rlc_p);
    rlc_am_set_debug_infos(ctxt_pP, l_rlc_p, srb_flagP, rb_idP, chan_idP);
    NB_rlc_am_configure(ctxt_pP,
    					l_rlc_p,
						config_am_pP->max_retx_threshold,
                    	config_am_pP->t_poll_retransmit,
						config_am_pP->enableStatusReportSN_Gap);
  } else {
    LOG_E(RLC, PROTOCOL_RLC_AM_CTXT_FMT" CONFIG_REQ RLC NOT FOUND\n",
          PROTOCOL_RLC_AM_CTXT_ARGS(ctxt_pP,l_rlc_p));
  }
}


//defined in rlc_tm_init.c (nothing to be changed)
//-----------------------------------------------------------------------------
void NB_config_req_rlc_tm (
  const protocol_ctxt_t* const  ctxt_pP,
  const srb_flag_t  srb_flagP,
  const rlc_tm_info_t * const config_tmP,
  const rb_id_t rb_idP,
  const logical_chan_id_t chan_idP
)
{
  rlc_union_t     *rlc_union_p  = NULL;
  rlc_tm_entity_t *rlc_p        = NULL;
  hash_key_t       key          = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  hashtable_rc_t   h_rc;

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    rlc_p = &rlc_union_p->rlc.tm;
    LOG_D(RLC, PROTOCOL_RLC_TM_CTXT_FMT" CONFIG_REQ (is_uplink_downlink=%d) RB %u\n",
          PROTOCOL_RLC_TM_CTXT_ARGS(ctxt_pP, rlc_p),
          config_tmP->is_uplink_downlink,
          rb_idP);

    //MP: i think this functions are fine also for NB-IoT
    rlc_tm_init(ctxt_pP, rlc_p);
    rlc_p->protocol_state = RLC_DATA_TRANSFER_READY_STATE;
    rlc_tm_set_debug_infos(ctxt_pP, rlc_p, srb_flagP, rb_idP, chan_idP);
    rlc_tm_configure(ctxt_pP, rlc_p, config_tmP->is_uplink_downlink);
  } else {
    LOG_E(RLC, PROTOCOL_RLC_TM_CTXT_FMT" CONFIG_REQ RB %u RLC NOT FOUND\n",
          PROTOCOL_RLC_TM_CTXT_ARGS(ctxt_pP, rlc_p),
          rb_idP);
  }
}

//defined in rlc_rrc.c
rlc_op_status_t NB_rrc_rlc_remove_ue (
  const protocol_ctxt_t* const ctxt_pP)
{
  //-----------------------------------------------------------------------------
  rb_id_t                rb_id;

  //XXX MP: rb_id = 2 should be not used (SRB2 not defined for NB-IoT)
  for (rb_id = 1; rb_id <= 3; rb_id++) {
    if(rb_id != 2){
    	NB_rrc_rlc_remove_rlc(ctxt_pP,
    						  SRB_FLAG_YES,
						      rb_id);
    }
    else
    {
    	LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] NB_rrc_rlc_remove_ue --> removing an UE with rb_id = 2 in NB_IoT!\n",
    		              PROTOCOL_CTXT_ARGS(ctxt_pP), "SRB", rb_id);
    }
  }

  for (rb_id = 1; rb_id <= maxDRB_NB_r13; rb_id++) { //FIXME: maxDRB_NB_r13 +3 ??
	  if(rb_id != 2){
		  NB_rrc_rlc_remove_rlc(ctxt_pP,
                       SRB_FLAG_NO,
                       rb_id);
	  }
	  else
	      LOG_E(RLC, PROTOCOL_CTXT_FMT"[%s %u] NB_rrc_rlc_remove_ue --> removing an UE with rb_id = 2 in NB_IoT!\n",
	      		              PROTOCOL_CTXT_ARGS(ctxt_pP), "DRB", rb_id);
  }

  return RLC_OP_STATUS_OK;
}

//defined in rlc_rrc.c --> NO MORE USED PROBABLY
//void NB_rrc_rlc_register_rrc ( //what do?
//		rrc_data_ind_cb_t rrc_data_indP,
//		rrc_data_conf_cb_t rrc_data_confP
//		);

/*--------------------------------------------RLC-PDCP--------------------------------------------------*/

//defined in rlc.c
//--------------------------------------------
void NB_rlc_data_ind     (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t  srb_flagP,
  const srb1bis_flag_t srb1bis_flag,
  const rb_id_t     rb_idP,
  const sdu_size_t  sdu_sizeP,
  mem_block_t      *sdu_pP)
{
  //-----------------------------------------------------------------------------


#if defined(TRACE_RLC_PAYLOAD)
  LOG_D(RLC, PROTOCOL_CTXT_FMT"[%s %u] Display of rlc_data_ind: size %u\n",
        PROTOCOL_CTXT_ARGS(ctxt_pP),
        (srb_flagP) ? "SRB" : "DRB",
        rb_idP,
        sdu_sizeP);

  rlc_util_print_hex_octets(RLC, (unsigned char*)sdu_pP->data, sdu_sizeP);
#endif

#if T_TRACER
  if (ctxt_pP->enb_flag)
    T(T_ENB_RLC_UL, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->rnti), T_INT(rb_idP), T_INT(sdu_sizeP));
#endif


  NB_pdcp_data_ind (
    ctxt_pP,
    srb_flagP,
	srb1bis_flag,
    rb_idP,
    sdu_sizeP,
    sdu_pP);
}

//defined in rlc.c
//-----------------------------------------------------------------------------
rlc_op_status_t NB_rlc_data_req     (const protocol_ctxt_t* const ctxt_pP,
                                  const srb_flag_t   srb_flagP,
                                  const rb_id_t      rb_idP,
                                  const mui_t        muiP,
                                  confirm_t    confirmP,
                                  sdu_size_t   sdu_sizeP,
                                  mem_block_t *sdu_pP)
{
  //-----------------------------------------------------------------------------
  mem_block_t           *new_sdu_p    = NULL;
  rlc_mode_t             rlc_mode     = RLC_MODE_NONE;
  rlc_union_t           *rlc_union_p = NULL;
  hash_key_t             key         = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;

#if defined(Rel10) || defined(Rel14)

  logical_chan_id_t      log_ch_id  = 0;
#endif
#ifdef DEBUG_RLC_DATA_REQ
  LOG_D(RLC,PROTOCOL_CTXT_FMT"rlc_data_req:  rb_id %u (MAX %d), muip %d, confirmP %d, sdu_sizeP %d, sdu_pP %p\n",
        PROTOCOL_CTXT_ARGS(ctxt_pP),
        rb_idP,
        NB_RAB_MAX, //may to be cha ged for NB-IoT
        muiP,
        confirmP,
        sdu_sizeP,
        sdu_pP);
#endif
#if defined(Rel10) || defined(Rel14)
#else
  //AssertFatal(MBMS_flagP == 0, "MBMS_flagP %u", MBMS_flagP);
#endif
#ifdef OAI_EMU

  CHECK_CTXT_ARGS(ctxt_pP)

#endif

#if T_TRACER
  if (ctxt_pP->enb_flag)
    T(T_ENB_RLC_DL, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->rnti), T_INT(rb_idP), T_INT(sdu_sizeP));
#endif

//  if (MBMS_flagP) {
//    AssertFatal (rb_idP < NB_RB_MBMS_MAX, "RB id is too high (%u/%d)!\n", rb_idP, NB_RB_MBMS_MAX);
//  } else {
//    AssertFatal (rb_idP < NB_RB_MAX, "RB id is too high (%u/%d)!\n", rb_idP, NB_RB_MAX);
//  }

  DevAssert(sdu_pP != NULL);
  DevCheck(sdu_sizeP > 0, sdu_sizeP, 0, 0);

#if !defined(Rel10) && !defined(Rel14)
 // DevCheck(MBMS_flagP == 0, MBMS_flagP, 0, 0);
#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_IN);

#if defined(Rel10) || defined(Rel14)

//  if (MBMS_flagP == TRUE) {
//    if (ctxt_pP->enb_flag) {
//      log_ch_id = rlc_mbms_enb_get_lcid_by_rb_id(ctxt_pP->module_id,rb_idP);
//      mbms_id_p = &rlc_mbms_lcid2service_session_id_eNB[ctxt_pP->module_id][log_ch_id];
//    } else {
//      log_ch_id = rlc_mbms_ue_get_lcid_by_rb_id(ctxt_pP->rnti,rb_idP);
//      mbms_id_p = &rlc_mbms_lcid2service_session_id_ue[ctxt_pP->rnti][log_ch_id];
//    }
//
//    key = RLC_COLL_KEY_MBMS_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, mbms_id_p->service_id, mbms_id_p->session_id);
//  } else
#endif
  {
    key = RLC_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_idP, srb_flagP);
  }

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  //MP: the RLC entity should be already configured at configuration time
  if (h_rc == HASH_TABLE_OK) {
    rlc_mode = rlc_union_p->mode;
  } else {
    rlc_mode = RLC_MODE_NONE;
    AssertFatal (0 , "RLC not configured key %ju\n", key);
  }

 // if (MBMS_flagP == 0) {

    LOG_D(RLC, PROTOCOL_CTXT_FMT"[RB %u] Display of rlc_data_req:\n",
          PROTOCOL_CTXT_ARGS(ctxt_pP),
          rb_idP);
#if defined(TRACE_RLC_PAYLOAD)
    rlc_util_print_hex_octets(RLC, (unsigned char*)sdu_pP->data, sdu_sizeP);
#endif

#ifdef DEBUG_RLC_DATA_REQ
    LOG_D(RLC,"RLC_TYPE : %d\n", rlc_mode);
#endif

    switch (rlc_mode) {
    case RLC_MODE_NONE:
      free_mem_block(sdu_pP, __func__);
      LOG_E(RLC, PROTOCOL_CTXT_FMT" Received RLC_MODE_NONE as rlc_type for rb_id %u\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            rb_idP);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
      return RLC_OP_STATUS_BAD_PARAMETER;

    case RLC_MODE_AM:
#ifdef DEBUG_RLC_DATA_REQ
      msg("RLC_MODE_AM\n");
#endif
      new_sdu_p = get_free_mem_block (sdu_sizeP + sizeof (struct rlc_am_data_req_alloc), __func__);

      if (new_sdu_p != NULL) {
        // PROCESS OF COMPRESSION HERE:
        memset (new_sdu_p->data, 0, sizeof (struct rlc_am_data_req_alloc));
        memcpy (&new_sdu_p->data[sizeof (struct rlc_am_data_req_alloc)], &sdu_pP->data[0], sdu_sizeP);

        ((struct rlc_am_data_req *) (new_sdu_p->data))->data_size = sdu_sizeP;
        ((struct rlc_am_data_req *) (new_sdu_p->data))->conf = confirmP;
        ((struct rlc_am_data_req *) (new_sdu_p->data))->mui  = muiP;
        ((struct rlc_am_data_req *) (new_sdu_p->data))->data_offset = sizeof (struct rlc_am_data_req_alloc);
        free_mem_block(sdu_pP, __func__);
        rlc_am_data_req(ctxt_pP, &rlc_union_p->rlc.am, new_sdu_p);
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
        return RLC_OP_STATUS_OK;
      } else {
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
        return RLC_OP_STATUS_INTERNAL_ERROR;
      }

      break;

//MP: RLC mode UM not defined for NB-IoT

    case RLC_MODE_TM:
      new_sdu_p = get_free_mem_block (sdu_sizeP + sizeof (struct rlc_tm_data_req_alloc), __func__);

      if (new_sdu_p != NULL) {
        // PROCESS OF COMPRESSION HERE:
        memset (new_sdu_p->data, 0, sizeof (struct rlc_tm_data_req_alloc));
        memcpy (&new_sdu_p->data[sizeof (struct rlc_tm_data_req_alloc)], &sdu_pP->data[0], sdu_sizeP);

        ((struct rlc_tm_data_req *) (new_sdu_p->data))->data_size = sdu_sizeP;
        ((struct rlc_tm_data_req *) (new_sdu_p->data))->data_offset = sizeof (struct rlc_tm_data_req_alloc);
        free_mem_block(sdu_pP, __func__);
        rlc_tm_data_req(ctxt_pP, &rlc_union_p->rlc.tm, new_sdu_p);
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
        return RLC_OP_STATUS_OK;
      } else {
        //handle_event(ERROR,"FILE %s FONCTION rlc_data_req() LINE %s : out of memory\n", __FILE__, __LINE__);
        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
        return RLC_OP_STATUS_INTERNAL_ERROR;
      }

      break;

    default:
      free_mem_block(sdu_pP, __func__);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
      return RLC_OP_STATUS_INTERNAL_ERROR;

    }

    //MBMS not in NB-IoT
//#if defined(Rel10) || defined(Rel14)
//  } else { /* MBMS_flag != 0 */
//    //  LOG_I(RLC,"DUY rlc_data_req: mbms_rb_id in RLC instant is: %d\n", mbms_rb_id);
//    if (sdu_pP != NULL) {
//      if (sdu_sizeP > 0) {
//        LOG_I(RLC,"received a packet with size %d for MBMS \n", sdu_sizeP);
//        new_sdu_p = get_free_mem_block (sdu_sizeP + sizeof (struct rlc_um_data_req_alloc), __func__);
//
//        if (new_sdu_p != NULL) {
//          // PROCESS OF COMPRESSION HERE:
//          memset (new_sdu_p->data, 0, sizeof (struct rlc_um_data_req_alloc));
//          memcpy (&new_sdu_p->data[sizeof (struct rlc_um_data_req_alloc)], &sdu_pP->data[0], sdu_sizeP);
//          ((struct rlc_um_data_req *) (new_sdu_p->data))->data_size = sdu_sizeP;
//          ((struct rlc_um_data_req *) (new_sdu_p->data))->data_offset = sizeof (struct rlc_um_data_req_alloc);
//          free_mem_block(sdu_pP, __func__);
//          rlc_um_data_req(ctxt_pP, &rlc_union_p->rlc.um, new_sdu_p);
//
//          //free_mem_block(new_sdu, __func__);
//          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
//          return RLC_OP_STATUS_OK;
//        } else {
//          VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
//          return RLC_OP_STATUS_BAD_PARAMETER;
//        }
//      } else {
//        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
//        return RLC_OP_STATUS_BAD_PARAMETER;
//      }
//    } else {
//      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
//      return RLC_OP_STATUS_BAD_PARAMETER;
//    }
//  }
//
//#else
//  }
//  else  /* MBMS_flag != 0 */
//  {
//    free_mem_block(sdu_pP, __func__);
//    LOG_E(RLC, "MBMS_flag != 0 while Rel10/Rel14 is not defined...\n");
//    //handle_event(ERROR,"FILE %s FONCTION rlc_data_req() LINE %s : parameter module_id out of bounds :%d\n", __FILE__, __LINE__, module_idP);
//    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RLC_DATA_REQ,VCD_FUNCTION_OUT);
//    return RLC_OP_STATUS_BAD_PARAMETER;
//  }

//#endif
}


//defined in pdcp.c
//FIXME: should go transparent through the PDCP
//--------------------------------------------
boolean_t
NB_pdcp_data_ind(
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t   srb_flagP,
  const srb1bis_flag_t srb1bis_flag,
  const rb_id_t      rb_idP,
  const sdu_size_t   sdu_buffer_sizeP,
  mem_block_t* const sdu_buffer_pP
)
//-----------------------------------------------------------------------------
{
  pdcp_t      *pdcp_p          = NULL;
  list_t      *sdu_list_p      = NULL;
  mem_block_t *new_sdu_p       = NULL;
  uint8_t      pdcp_header_len = 0;
  uint8_t      pdcp_tailer_len = 0;
  pdcp_sn_t    sequence_number = 0;
  volatile sdu_size_t   payload_offset  = 0;
  rb_id_t      rb_id            = rb_idP;
  boolean_t    packet_forwarded = FALSE;
  hash_key_t      key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t  h_rc;
#if defined(LINK_ENB_PDCP_TO_GTPV1U)
  MessageDef  *message_p        = NULL;
  uint8_t     *gtpu_buffer_p    = NULL;
#endif


  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_IN);

#ifdef OAI_EMU

  CHECK_CTXT_ARGS(ctxt_pP);

#endif
#ifdef PDCP_MSG_PRINT
  int i=0;
  LOG_F(PDCP,"[MSG] PDCP UL %s PDU on rb_id %d\n", (srb_flagP)? "CONTROL" : "DATA", rb_idP);

  for (i = 0; i < sdu_buffer_sizeP; i++) {
    LOG_F(PDCP,"%02x ", ((uint8_t*)sdu_buffer_pP->data)[i]);
  }

  LOG_F(PDCP,"\n");
#endif

#if T_TRACER
  if (ctxt_pP->enb_flag != ENB_FLAG_NO)
    T(T_ENB_PDCP_UL, T_INT(ctxt_pP->module_id), T_INT(ctxt_pP->rnti), T_INT(rb_idP), T_INT(sdu_buffer_sizeP));
#endif

  if (srb1bis_flag) { //SRB1bis

	if(rb_id != 3)
	  	  LOG_E(PDCP, "Received SRB1bis but with rb_id = %d", rb_id);

    if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
      LOG_D(PDCP, "SRB1bis Data indication notification for PDCP entity from eNB %u to UE %x "
            "and radio bearer ID %d rlc sdu size %d ctxt_pP->enb_flag %d\n",
            ctxt_pP->module_id,
            ctxt_pP->rnti,
            rb_idP,
            sdu_buffer_sizeP,
            ctxt_pP->enb_flag);

    } else {
      LOG_D(PDCP, "SRB1bis Data indication notification for PDCP entity from UE %x to eNB %u "
            "and radio bearer ID %d rlc sdu size %d ctxt_pP->enb_flag %d\n",
            ctxt_pP->rnti,
            ctxt_pP->module_id ,
            rb_idP,
            sdu_buffer_sizeP,
            ctxt_pP->enb_flag);
    }

  } else { //SRB1 or DRBs

    rb_id = rb_idP % maxDRB_NB_r13; //XXX --> rb_id = rb_Idp(1,2not,3,4,5)%2 = 1,0,1,0,

    AssertError (rb_id < maxDRB_NB_r13, return FALSE, "RB id is too high (%u/%d) %u UE %x!\n",
                 rb_id,
				 maxDRB_NB_r13,
                 ctxt_pP->module_id,
                 ctxt_pP->rnti);
    AssertError (rb_id > 0, return FALSE, "RB id is too low (%u/%d) %u UE %x!\n",
                 rb_id,
				 maxDRB_NB_r13,
                 ctxt_pP->module_id,
                 ctxt_pP->rnti);

    key = PDCP_COLL_KEY_VALUE(ctxt_pP->module_id, ctxt_pP->rnti, ctxt_pP->enb_flag, rb_id, srb_flagP);
    h_rc = hashtable_get(pdcp_coll_p, key, (void**)&pdcp_p);

    if (h_rc != HASH_TABLE_OK) {
      LOG_W(PDCP,
            PROTOCOL_CTXT_FMT"Could not get PDCP instance key 0x%"PRIx64"\n",
            PROTOCOL_CTXT_ARGS(ctxt_pP),
            key);
      free_mem_block(sdu_buffer_pP, __func__);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
      return FALSE;
    }
  }

  sdu_list_p = &pdcp_sdu_list; //protected list in pdcp.h file

  if (sdu_buffer_sizeP == 0) {
    LOG_W(PDCP, "SDU buffer size is zero! Ignoring this chunk!\n");
    return FALSE;
  }

  if (ctxt_pP->enb_flag) {
    start_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
  } else {
    start_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
  }

  /*
   * Parse the PDU placed at the beginning of SDU to check
   * if incoming SN is in line with RX window
   */

  //if (MBMS_flagP == 0 ) {
    if (srb_flagP) { //SRB1/1bis
      pdcp_header_len = PDCP_CONTROL_PLANE_DATA_PDU_SN_SIZE;
      pdcp_tailer_len = PDCP_CONTROL_PLANE_DATA_PDU_MAC_I_SIZE;
      sequence_number =   pdcp_get_sequence_number_of_pdu_with_SRB_sn((unsigned char*)sdu_buffer_pP->data);
    } else { // DRB
      pdcp_tailer_len = 0;

      if (pdcp_p->seq_num_size == PDCP_SN_7BIT) { //MP: the only allowed for NB-IoT (2^7-1 = 127)
        pdcp_header_len = PDCP_USER_PLANE_DATA_PDU_SHORT_SN_HEADER_SIZE;
        sequence_number =     pdcp_get_sequence_number_of_pdu_with_short_sn((unsigned char*)sdu_buffer_pP->data);
      }
      else {
        //sequence_number = 128 (2^7);
        LOG_E(PDCP,
              PROTOCOL_PDCP_CTXT_FMT"wrong sequence number  (%d) for this NB-IoT pdcp entity (should be 7 bit) \n",
              PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
              pdcp_p->seq_num_size);
      }

    }

    /*
     * Check if incoming SDU is long enough to carry a PDU header
     */
    if (sdu_buffer_sizeP < pdcp_header_len + pdcp_tailer_len ) {
      LOG_W(PDCP,
            PROTOCOL_PDCP_CTXT_FMT"Incoming (from RLC) SDU is short of size (size:%d)! Ignoring...\n",
            PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
            sdu_buffer_sizeP);
      free_mem_block(sdu_buffer_pP, __func__);

      if (ctxt_pP->enb_flag) {
        stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
      } else {
        stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
      }

      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
      return FALSE;
    }

    if (pdcp_is_rx_seq_number_valid(sequence_number, pdcp_p, srb_flagP) == TRUE) {
#if 0
      LOG_T(PDCP, "Incoming PDU has a sequence number (%d) in accordance with RX window\n", sequence_number);
#endif

    } else {
      LOG_W(PDCP,
            PROTOCOL_PDCP_CTXT_FMT"Incoming PDU has an unexpected sequence number (%d), RX window synchronisation have probably been lost!\n",
            PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
            sequence_number);
      /*
       * XXX Till we implement in-sequence delivery and duplicate discarding
       * mechanism all out-of-order packets will be delivered to RRC/IP
       */
#if 0
      LOG_D(PDCP, "Ignoring PDU...\n");
      free_mem_block(sdu_buffer, __func__);
      return FALSE;
#else
      //LOG_W(PDCP, "Delivering out-of-order SDU to upper layer...\n");
#endif
    }

    // SRB1/SRB1bis: control-plane data
    if (srb_flagP) {

#if defined(ENABLE_SECURITY)

      if (pdcp_p->security_activated == 1) {
        if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
          start_meas(&eNB_pdcp_stats[ctxt_pP->module_id].validate_security);
        } else {
          start_meas(&UE_pdcp_stats[ctxt_pP->module_id].validate_security);
        }

        pdcp_validate_security(ctxt_pP,
                               pdcp_p,
                               srb_flagP,
                               rb_idP,
                               pdcp_header_len,
                               sequence_number,
                               sdu_buffer_pP->data,
                               sdu_buffer_sizeP - pdcp_tailer_len);

        if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
          stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].validate_security);
        } else {
          stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].validate_security);
        }
      }

#endif

  	MSC_LOG_TX_MESSAGE(
  	    (ctxt_pP->enb_flag == ENB_FLAG_NO)? MSC_PDCP_UE:MSC_PDCP_ENB,
        (ctxt_pP->enb_flag == ENB_FLAG_NO)? MSC_RRC_UE:MSC_RRC_ENB,
        NULL,0,
        PROTOCOL_PDCP_CTXT_FMT" DATA-IND len %u",
        PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
        sdu_buffer_sizeP - pdcp_header_len - pdcp_tailer_len);

      NB_rrc_data_ind(ctxt_pP,
    		  	  	  rb_id,
					  sdu_buffer_sizeP - pdcp_header_len - pdcp_tailer_len,
					  (uint8_t*)&sdu_buffer_pP->data[pdcp_header_len],
					(srb1bis_flag) ? SRB1BIS_FLAG_YES : SRB1BIS_FLAG_NO);

      free_mem_block(sdu_buffer_pP, __func__);

      if (ctxt_pP->enb_flag) {
        stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
      } else {
        stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
      }

      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
      return TRUE;
    }

    /*
     * DRBs
     */
    payload_offset=pdcp_header_len;// PDCP_USER_PLANE_DATA_PDU_SHORT_SN_HEADER_SIZE;
#if defined(ENABLE_SECURITY)

    if (pdcp_p->security_activated == 1) {
      if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
        start_meas(&eNB_pdcp_stats[ctxt_pP->module_id].validate_security);
      } else {
        start_meas(&UE_pdcp_stats[ctxt_pP->module_id].validate_security);
      }

      pdcp_validate_security(
        ctxt_pP,
        pdcp_p,
        srb_flagP,
        rb_idP,
        pdcp_header_len,
        sequence_number,
        sdu_buffer_pP->data,
        sdu_buffer_sizeP - pdcp_tailer_len);

      if (ctxt_pP->enb_flag == ENB_FLAG_NO) {
        stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].validate_security);
      } else {
        stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].validate_security);
      }

    }

#endif
//  } else {
//    payload_offset=0;
//  }

#if defined(USER_MODE) && defined(OAI_EMU)

  if (oai_emulation.info.otg_enabled == 1) {
    //unsigned int dst_instance;
    int    ctime;

    if ((pdcp_p->rlc_mode == RLC_MODE_AM)) { //MP: &&(MBMS_flagP==0)
      pdcp_p->last_submitted_pdcp_rx_sn = sequence_number;
    }

#if defined(DEBUG_PDCP_PAYLOAD)
    rlc_util_print_hex_octets(PDCP,
                              (unsigned char*)&sdu_buffer_pP->data[payload_offset],
                              sdu_buffer_sizeP - payload_offset);
#endif

    ctime = oai_emulation.info.time_ms; // avg current simulation time in ms : we may get the exact time through OCG?
//    if (MBMS_flagP == 0){
//      LOG_D(PDCP,
//	    PROTOCOL_PDCP_CTXT_FMT"Check received buffer :  (dst %d)\n",
//	    PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p),
//	    ctxt_pP->instance);
//    }
    if (otg_rx_pkt(
          ctxt_pP->instance,
          ctime,
          (const char*)(&sdu_buffer_pP->data[payload_offset]),
                   sdu_buffer_sizeP - payload_offset ) == 0 ) {
      free_mem_block(sdu_buffer_pP, __func__);

      if (ctxt_pP->enb_flag) {
        stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
      } else {
        stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
      }

      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
      return TRUE;
    }
  }

#else

  if (otg_enabled==1) {
    LOG_D(OTG,"Discarding received packed\n");
    free_mem_block(sdu_buffer_pP, __func__);

    if (ctxt_pP->enb_flag) {
      stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
    } else {
      stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
    }

    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
    return TRUE;
  }

#endif


  // XXX Decompression would be done at this point

  /*
   * After checking incoming sequence number PDCP header
   * has to be stripped off so here we copy SDU buffer starting
   * from its second byte (skipping 0th and 1st octets, i.e.
   * PDCP header)
   *
   *FIXME: to be check if up to know the length of the PDCP packet is ok also for Nb-IoT
   *
   */
#if defined(LINK_ENB_PDCP_TO_GTPV1U)

  if ((TRUE == ctxt_pP->enb_flag) && (FALSE == srb_flagP)) {
    MSC_LOG_TX_MESSAGE(
    		MSC_PDCP_ENB,
    		MSC_GTPU_ENB,
    		NULL,0,
    		"0 GTPV1U_ENB_TUNNEL_DATA_REQ  ue %x rab %u len %u",
    		ctxt_pP->rnti,
    		rb_id + 4,
    		sdu_buffer_sizeP - payload_offset);
    //LOG_T(PDCP,"Sending to GTPV1U %d bytes\n", sdu_buffer_sizeP - payload_offset);
    gtpu_buffer_p = itti_malloc(TASK_PDCP_ENB, TASK_GTPV1_U,
                                sdu_buffer_sizeP - payload_offset + GTPU_HEADER_OVERHEAD_MAX);
    AssertFatal(gtpu_buffer_p != NULL, "OUT OF MEMORY");
    memcpy(&gtpu_buffer_p[GTPU_HEADER_OVERHEAD_MAX], &sdu_buffer_pP->data[payload_offset], sdu_buffer_sizeP - payload_offset);
    message_p = itti_alloc_new_message(TASK_PDCP_ENB, GTPV1U_ENB_TUNNEL_DATA_REQ);
    AssertFatal(message_p != NULL, "OUT OF MEMORY");
    GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).buffer       = gtpu_buffer_p;
    GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).length       = sdu_buffer_sizeP - payload_offset;
    GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).offset       = GTPU_HEADER_OVERHEAD_MAX;
    GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).rnti         = ctxt_pP->rnti;
    GTPV1U_ENB_TUNNEL_DATA_REQ(message_p).rab_id       = rb_id + 4;
    itti_send_msg_to_task(TASK_GTPV1_U, INSTANCE_DEFAULT, message_p);
    packet_forwarded = TRUE;
  }

#else
  packet_forwarded = FALSE;
#endif

  if (FALSE == packet_forwarded) {
    new_sdu_p = get_free_mem_block(sdu_buffer_sizeP - payload_offset + sizeof (pdcp_data_ind_header_t), __func__);

    if (new_sdu_p) {
      if (pdcp_p->rlc_mode == RLC_MODE_AM ) {
        pdcp_p->last_submitted_pdcp_rx_sn = sequence_number;
      }

      /*
       * Prepend PDCP indication header which is going to be removed at pdcp_fifo_flush_sdus()
       */
      memset(new_sdu_p->data, 0, sizeof (pdcp_data_ind_header_t));
      ((pdcp_data_ind_header_t *) new_sdu_p->data)->data_size = sdu_buffer_sizeP - payload_offset;
      AssertFatal((sdu_buffer_sizeP - payload_offset >= 0), "invalid PDCP SDU size!");

      // Here there is no virtualization possible
      // set ((pdcp_data_ind_header_t *) new_sdu_p->data)->inst for IP layer here

      if (ctxt_pP->enb_flag == ENB_FLAG_NO) { //UE
        ((pdcp_data_ind_header_t *) new_sdu_p->data)->rb_id = rb_id;
#if defined(OAI_EMU)
        ((pdcp_data_ind_header_t*) new_sdu_p->data)->inst  = ctxt_pP->module_id + oai_emulation.info.nb_enb_local - oai_emulation.info.first_ue_local;
#else
#  if defined(ENABLE_USE_MME)
        /* for the UE compiled in S1 mode, we need 1 here
         * for the UE compiled in noS1 mode, we need 0
         * TODO: be sure of this
         */
        ((pdcp_data_ind_header_t*) new_sdu_p->data)->inst  = 1;
#  endif
#endif

      } else { //eNB
        ((pdcp_data_ind_header_t*) new_sdu_p->data)->rb_id = rb_id + (ctxt_pP->module_id * maxDRB_NB_r13);

#if defined(OAI_EMU)
        ((pdcp_data_ind_header_t*) new_sdu_p->data)->inst  = ctxt_pP->module_id - oai_emulation.info.first_enb_local;
#endif
      }
#ifdef DEBUG_PDCP_FIFO_FLUSH_SDU
      static uint32_t pdcp_inst = 0;
      ((pdcp_data_ind_header_t*) new_sdu_p->data)->inst = pdcp_inst++;
      LOG_D(PDCP, "inst=%d size=%d\n", ((pdcp_data_ind_header_t*) new_sdu_p->data)->inst, ((pdcp_data_ind_header_t *) new_sdu_p->data)->data_size);
#endif

      memcpy(&new_sdu_p->data[sizeof (pdcp_data_ind_header_t)], \
             &sdu_buffer_pP->data[payload_offset], \
             sdu_buffer_sizeP - payload_offset);
      list_add_tail_eurecom (new_sdu_p, sdu_list_p);

      /* Print octets of incoming data in hexadecimal form */
      LOG_D(PDCP, "Following content has been received from RLC (%d,%d)(PDCP header has already been removed):\n",
            sdu_buffer_sizeP  - payload_offset + (int)sizeof(pdcp_data_ind_header_t),
            sdu_buffer_sizeP  - payload_offset);

      //XXX MP: reactivated this utils for the moment
      LOG_D(PDCP, "HexPrint of the content");
      util_print_hex_octets(PDCP, &new_sdu_p->data[sizeof (pdcp_data_ind_header_t)], sdu_buffer_sizeP - payload_offset);
      util_flush_hex_octets(PDCP, &new_sdu_p->data[sizeof (pdcp_data_ind_header_t)], sdu_buffer_sizeP - payload_offset);

      /*
       * Update PDCP statistics
       * XXX Following two actions are identical, is there a merge error?
       */

      /*if (ctxt_pP->enb_flag == 1) {
          Pdcp_stats_rx[module_id][(rb_idP & RAB_OFFSET2) >> RAB_SHIFT2][(rb_idP & RAB_OFFSET) - DTCH]++;
          Pdcp_stats_rx_bytes[module_id][(rb_idP & RAB_OFFSET2) >> RAB_SHIFT2][(rb_idP & RAB_OFFSET) - DTCH] += sdu_buffer_sizeP;
        } else {
          Pdcp_stats_rx[module_id][(rb_idP & RAB_OFFSET2) >> RAB_SHIFT2][(rb_idP & RAB_OFFSET) - DTCH]++;
          Pdcp_stats_rx_bytes[module_id][(rb_idP & RAB_OFFSET2) >> RAB_SHIFT2][(rb_idP & RAB_OFFSET) - DTCH] += sdu_buffer_sizeP;
        }*/
    }
  }

#if defined(STOP_ON_IP_TRAFFIC_OVERLOAD)
  else {
    AssertFatal(0, PROTOCOL_PDCP_CTXT_FMT" PDCP_DATA_IND SDU DROPPED, OUT OF MEMORY \n",
                PROTOCOL_PDCP_CTXT_ARGS(ctxt_pP, pdcp_p));
  }

#endif

  free_mem_block(sdu_buffer_pP, __func__);

  if (ctxt_pP->enb_flag) {
    stop_meas(&eNB_pdcp_stats[ctxt_pP->module_id].data_ind);
  } else {
    stop_meas(&UE_pdcp_stats[ctxt_pP->module_id].data_ind);
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PDCP_DATA_IND,VCD_FUNCTION_OUT);
  return TRUE;
}

/*-----------------------------------MAC-RLC------------------------------------------*/

//-----------------------------------------------------------------------------
//defined in rlc_mac.c
void NB_mac_rlc_data_ind     (
  const module_id_t         module_idP,
  const rnti_t              rntiP,
  const module_id_t         eNB_index,
  const frame_t             frameP,
  const eNB_flag_t          enb_flagP,
//const MBMS_flag_t         MBMS_flagP,
  const logical_chan_id_t   channel_idP,
  char                     *buffer_pP,
  const tb_size_t           tb_sizeP,
  num_tb_t                  num_tbP,
  crc_t                    *crcs_pP)
{
  //-----------------------------------------------------------------------------
  rlc_mode_t             rlc_mode   = RLC_MODE_NONE;
  rlc_union_t           *rlc_union_p     = NULL;
  hash_key_t             key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;
  //XXX MP: modify the check on LCID
  srb_flag_t             srb_flag        = (channel_idP <= 3) ? SRB_FLAG_YES : SRB_FLAG_NO;
  srb1bis_flag_t         srb1bis_flag    = (channel_idP == 3) ? SRB1BIS_FLAG_YES : SRB1BIS_FLAG_NO; //may not needed?
  protocol_ctxt_t     ctxt;

  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_idP, enb_flagP, rntiP, frameP, 0, eNB_index);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MAC_RLC_DATA_IND,VCD_FUNCTION_IN);

#ifdef DEBUG_MAC_INTERFACE

  if (num_tbP) {
    LOG_D(RLC, PROTOCOL_CTXT_FMT" MAC_RLC_DATA_IND on channel %d (%d), rb max %d, Num_tb %d\n",
          PROTOCOL_CTXT_ARGS(&ctxt),
          channel_idP,
          RLC_MAX_LC, //XXX ???
		  NB_RB_MAX_NB_IOT,
          num_tbP);
  }

#endif // DEBUG_MAC_INTERFACE
#ifdef OAI_EMU


    AssertFatal (channel_idP < NB_RB_MAX_NB_IOT,        "channel id is too high (%u/%d)!\n",
                 channel_idP, NB_RB_MAX_NB_IOT);

  CHECK_CTXT_ARGS(&ctxt);

#endif

#if T_TRACER
  if (enb_flagP)
    T(T_ENB_RLC_MAC_UL, T_INT(module_idP), T_INT(rntiP), T_INT(channel_idP), T_INT(tb_sizeP));
#endif

  //no MBMS flag

  {
    key = RLC_COLL_KEY_LCID_VALUE(module_idP, rntiP, enb_flagP, channel_idP, srb_flag);
  }

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  //XXX MP: also for SRB1bis an RLC-AM mode should be configured
  if (h_rc == HASH_TABLE_OK) {
    rlc_mode = rlc_union_p->mode;
  } else {
    rlc_mode = RLC_MODE_NONE;
    //AssertFatal (0 , "%s RLC not configured rb id %u lcid %u module %u!\n", __FUNCTION__, rb_id, channel_idP, ue_module_idP);
  }

  struct mac_data_ind data_ind = mac_rlc_deserialize_tb(buffer_pP, tb_sizeP, num_tbP, crcs_pP);

  switch (rlc_mode) {
  case RLC_MODE_NONE:
    //handle_event(WARNING,"FILE %s FONCTION mac_rlc_data_ind() LINE %s : no radio bearer configured :%d\n", __FILE__, __LINE__, channel_idP);
    break;

  case RLC_MODE_AM:
    NB_rlc_am_mac_data_indication(&ctxt, &rlc_union_p->rlc.am, data_ind);
    break;

   //XXX MP: no UM mode for NB_IoT

  case RLC_MODE_TM:
    rlc_tm_mac_data_indication(&ctxt, &rlc_union_p->rlc.tm, data_ind);
    break;

  default:
	  LOG_E(RLC,PROTOCOL_PDCP_CTXT_FMT"mac_rlc_data_ind -> RLC mode unknown");
	break;
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MAC_RLC_DATA_IND,VCD_FUNCTION_OUT);

}

//-----------------------------------------------------------------------------
//defined in rlc_am.c
void
NB_rlc_am_mac_data_indication (
  const protocol_ctxt_t* const ctxt_pP,
  void * const                 rlc_pP,
  struct mac_data_ind          data_indP
)
{
  rlc_am_entity_t*           l_rlc_p = (rlc_am_entity_t*) rlc_pP;

#if TRACE_RLC_AM_PDU || MESSAGE_CHART_GENERATOR
  rlc_am_pdu_info_t   pdu_info;
  rlc_am_pdu_sn_10_t *rlc_am_pdu_sn_10_p;
  mem_block_t        *tb_p;
  sdu_size_t          tb_size_in_bytes;
  int                 num_nack;
  char                message_string[7000];
  size_t              message_string_size = 0;
#   if ENABLE_ITTI
  MessageDef         *msg_p;
#   endif
  int                 octet_index, index;
  /* for no gcc warnings */
  (void)num_nack;
  (void)message_string;
  (void)message_string_size;
  (void)octet_index;
  (void)index;
#endif

  (void)l_rlc_p; /* avoid gcc warning "unused variable" */

#if TRACE_RLC_AM_PDU || MESSAGE_CHART_GENERATOR

  if (data_indP.data.nb_elements > 0) {

    tb_p = data_indP.data.head;

    while (tb_p != NULL) {

      rlc_am_pdu_sn_10_p = (rlc_am_pdu_sn_10_t*)((struct mac_tb_ind *) (tb_p->data))->data_ptr;
      tb_size_in_bytes   = ((struct mac_tb_ind *) (tb_p->data))->size;

      if ((((struct mac_tb_ind *) (tb_p->data))->data_ptr[0] & RLC_DC_MASK) == RLC_DC_DATA_PDU ) {
        if (rlc_am_get_data_pdu_infos(ctxt_pP,l_rlc_p,rlc_am_pdu_sn_10_p, tb_size_in_bytes, &pdu_info) >= 0) {
#if MESSAGE_CHART_GENERATOR
          message_string_size = 0;
          message_string_size += sprintf(&message_string[message_string_size],
                                         MSC_AS_TIME_FMT" "PROTOCOL_RLC_AM_MSC_FMT" DATA SN %u size %u RF %u P %u FI %u",
                                         MSC_AS_TIME_ARGS(ctxt_pP),
                                         PROTOCOL_RLC_AM_MSC_ARGS(ctxt_pP, l_rlc_p),
                                         pdu_info.sn,
                                         tb_size_in_bytes,
                                         pdu_info.rf,
                                         pdu_info.p,
                                         pdu_info.fi);

          if (pdu_info.rf) {
            message_string_size += sprintf(&message_string[message_string_size], " LSF %u\n", pdu_info.lsf);
            message_string_size += sprintf(&message_string[message_string_size], " SO %u\n", pdu_info.so);
          }

          if (pdu_info.e) {
            message_string_size += sprintf(&message_string[message_string_size], "| HE:");

            for (index=0; index < pdu_info.num_li; index++) {
              message_string_size += sprintf(&message_string[message_string_size], " LI %u", pdu_info.li_list[index]);
            }
          }

          MSC_LOG_RX_MESSAGE(
            (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_ENB:MSC_RLC_UE,
            (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_UE:MSC_RLC_ENB,
            (char*)rlc_am_pdu_sn_10_p,
            tb_size_in_bytes,
            message_string);

#endif

#   if ENABLE_ITTI && TRACE_RLC_AM_PDU
          message_string_size += sprintf(&message_string[message_string_size], "Bearer      : %u\n", l_rlc_p->rb_id);
          message_string_size += sprintf(&message_string[message_string_size], "PDU size    : %u\n", tb_size_in_bytes);
          message_string_size += sprintf(&message_string[message_string_size], "Header size : %u\n", pdu_info.header_size);
          message_string_size += sprintf(&message_string[message_string_size], "Payload size: %u\n", pdu_info.payload_size);

          if (pdu_info.rf) {
            message_string_size += sprintf(&message_string[message_string_size], "PDU type    : RLC AM DATA IND: AMD PDU segment\n\n");
          } else {
            message_string_size += sprintf(&message_string[message_string_size], "PDU type    : RLC AM DATA IND: AMD PDU\n\n");
          }

          message_string_size += sprintf(&message_string[message_string_size], "Header      :\n");
          message_string_size += sprintf(&message_string[message_string_size], "  D/C       : %u\n", pdu_info.d_c);
          message_string_size += sprintf(&message_string[message_string_size], "  RF        : %u\n", pdu_info.rf);
          message_string_size += sprintf(&message_string[message_string_size], "  P         : %u\n", pdu_info.p);
          message_string_size += sprintf(&message_string[message_string_size], "  FI        : %u\n", pdu_info.fi);
          message_string_size += sprintf(&message_string[message_string_size], "  E         : %u\n", pdu_info.e);
          message_string_size += sprintf(&message_string[message_string_size], "  SN        : %u\n", pdu_info.sn);

          if (pdu_info.rf) {
            message_string_size += sprintf(&message_string[message_string_size], "  LSF       : %u\n", pdu_info.lsf);
            message_string_size += sprintf(&message_string[message_string_size], "  SO        : %u\n", pdu_info.so);
          }

          if (pdu_info.e) {
            message_string_size += sprintf(&message_string[message_string_size], "\nHeader extension  : \n");

            for (index=0; index < pdu_info.num_li; index++) {
              message_string_size += sprintf(&message_string[message_string_size], "  LI        : %u\n", pdu_info.li_list[index]);
            }
          }

          message_string_size += sprintf(&message_string[message_string_size], "\nPayload  : \n");
          message_string_size += sprintf(&message_string[message_string_size], "------+-------------------------------------------------|\n");
          message_string_size += sprintf(&message_string[message_string_size], "      |  0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f |\n");
          message_string_size += sprintf(&message_string[message_string_size], "------+-------------------------------------------------|\n");

          for (octet_index = 0; octet_index < pdu_info.payload_size; octet_index++) {
            if ((octet_index % 16) == 0) {
              if (octet_index != 0) {
                message_string_size += sprintf(&message_string[message_string_size], " |\n");
              }

              message_string_size += sprintf(&message_string[message_string_size], " %04d |", octet_index);
            }

            /*
             * Print every single octet in hexadecimal form
             */
            message_string_size += sprintf(&message_string[message_string_size], " %02x", pdu_info.payload[octet_index]);
            /*
             * Align newline and pipes according to the octets in groups of 2
             */
          }

          /*
           * Append enough spaces and put final pipe
           */
          for (index = octet_index; index < 16; ++index) {
            message_string_size += sprintf(&message_string[message_string_size], "   ");
          }

          message_string_size += sprintf(&message_string[message_string_size], " |\n");

          msg_p = itti_alloc_new_message_sized (ctxt_pP->enb_flag > 0 ? TASK_RLC_ENB:TASK_RLC_UE , RLC_AM_DATA_PDU_IND, message_string_size + sizeof (IttiMsgText));
          msg_p->ittiMsg.rlc_am_data_pdu_ind.size = message_string_size;
          memcpy(&msg_p->ittiMsg.rlc_am_data_pdu_ind.text, message_string, message_string_size);

          itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);

# else
          rlc_am_display_data_pdu_infos(ctxt_pP, l_rlc_p, &pdu_info);
# endif
        }
      } else {
        if (rlc_am_get_control_pdu_infos(rlc_am_pdu_sn_10_p, &tb_size_in_bytes, &l_rlc_p->control_pdu_info) >= 0) {
#if MESSAGE_CHART_GENERATOR
          message_string_size = 0;
          message_string_size += sprintf(&message_string[message_string_size],
                                         MSC_AS_TIME_FMT" "PROTOCOL_RLC_AM_MSC_FMT" STATUS size ACK_SN %u",
                                         MSC_AS_TIME_ARGS(ctxt_pP),
                                         PROTOCOL_RLC_AM_MSC_ARGS(ctxt_pP, l_rlc_p),
                                         l_rlc_p->control_pdu_info.ack_sn);

          for (num_nack = 0; num_nack < l_rlc_p->control_pdu_info.num_nack; num_nack++) {
            if (l_rlc_p->control_pdu_info.nack_list[num_nack].e2) {
              message_string_size += sprintf(&message_string[message_string_size], "  NACK SN %u SO START %u SO END %u",
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].nack_sn,
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].so_start,
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].so_end);

            } else {
              message_string_size += sprintf(&message_string[message_string_size], "  NACK SN %u",
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].nack_sn);
            }
          }

          MSC_LOG_RX_MESSAGE(
            (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_ENB:MSC_RLC_UE,
            (ctxt_pP->enb_flag == ENB_FLAG_YES) ? MSC_RLC_UE:MSC_RLC_ENB,
            (char*)rlc_am_pdu_sn_10_p,
            tb_size_in_bytes,
            message_string);

#endif

#   if ENABLE_ITTI && TRACE_RLC_AM_PDU
          message_string_size = 0;
          message_string_size += sprintf(&message_string[message_string_size], "Bearer      : %u\n", l_rlc_p->rb_id);
          message_string_size += sprintf(&message_string[message_string_size], "PDU size    : %u\n", ((struct mac_tb_ind *) (tb_p->data))->size);
          message_string_size += sprintf(&message_string[message_string_size], "PDU type    : RLC AM DATA IND: STATUS PDU\n\n");
          message_string_size += sprintf(&message_string[message_string_size], "Header      :\n");
          message_string_size += sprintf(&message_string[message_string_size], "  D/C       : %u\n", l_rlc_p->control_pdu_info.d_c);
          message_string_size += sprintf(&message_string[message_string_size], "  CPT       : %u\n", l_rlc_p->control_pdu_info.cpt);
          message_string_size += sprintf(&message_string[message_string_size], "  ACK_SN    : %u\n", l_rlc_p->control_pdu_info.ack_sn);
          message_string_size += sprintf(&message_string[message_string_size], "  E1        : %u\n", l_rlc_p->control_pdu_info.e1);

          for (num_nack = 0; num_nack < l_rlc_p->control_pdu_info.num_nack; num_nack++) {
            if (l_rlc_p->control_pdu_info.nack_list[num_nack].e2) {
              message_string_size += sprintf(&message_string[message_string_size], "  NACK SN %04d SO START %05d SO END %05d",
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].nack_sn,
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].so_start,
                                             l_rlc_p->control_pdu_info.nack_list[num_nack].so_end);
            } else {
              message_string_size += sprintf(&message_string[message_string_size], "  NACK SN %04d",  l_rlc_p->control_pdu_info.nack_list[num_nack].nack_sn);
            }
          }

          msg_p = itti_alloc_new_message_sized (ctxt_pP->enb_flag > 0 ? TASK_RLC_ENB:TASK_RLC_UE , RLC_AM_STATUS_PDU_IND, message_string_size + sizeof (IttiMsgText));
          msg_p->ittiMsg.rlc_am_status_pdu_ind.size = message_string_size;
          memcpy(&msg_p->ittiMsg.rlc_am_status_pdu_ind.text, message_string, message_string_size);

          itti_send_msg_to_task(TASK_UNKNOWN, ctxt_pP->instance, msg_p);

#   endif
        }
      }

      tb_p = tb_p->next;
    }
  }

#endif
  rlc_am_rx (ctxt_pP, rlc_pP, data_indP);
}

//-----------------------------------------------------------------------------
//defined in rlc_mac.c
//called by the schedule_ue_spec for getting SDU to be transmitted from SRB1/SRB1bis and DRBs
tbs_size_t NB_mac_rlc_data_req_eNB(
  const module_id_t       module_idP,
  const rnti_t            rntiP,
  const eNB_index_t       eNB_index,
  const frame_t           frameP,
  const MBMS_flag_t       MBMS_flagP,
  const logical_chan_id_t channel_idP,
  char             *buffer_pP)
{
  //-----------------------------------------------------------------------------
  struct mac_data_req    data_request;
  rlc_mode_t             rlc_mode        = RLC_MODE_NONE;
  rlc_union_t           *rlc_union_p     = NULL;
  hash_key_t             key             = HASHTABLE_NOT_A_KEY_VALUE;
  hashtable_rc_t         h_rc;
  srb_flag_t             srb_flag        = (channel_idP <= 3) ? SRB_FLAG_YES : SRB_FLAG_NO;
  srb1bis_flag_t			srb1bis_flag = (channel_idP == 3) ? SRB1BIS_FLAG_YES : SRB1BIS_FLAG_NO;
  tbs_size_t             ret_tb_size         = 0;
  protocol_ctxt_t     ctxt;

  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_idP, ENB_FLAG_YES, rntiP, frameP, 0,eNB_index);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MAC_RLC_DATA_REQ,VCD_FUNCTION_IN);
#ifdef DEBUG_MAC_INTERFACE
  LOG_D(RLC, PROTOCOL_CTXT_FMT" MAC_RLC_DATA_REQ channel %d (%d) MAX RB %d, Num_tb %d\n",
        PROTOCOL_CTXT_ARGS((&ctxt)),
        channel_idP,
        RLC_MAX_LC,
        NB_RB_MAX);

#endif // DEBUG_MAC_INTERFACE

  { //SRB1 or DRBs
    AssertFatal (channel_idP < NB_RB_MAX_NB_IOT,        "channel id is too high (%u/%d)!\n",     channel_idP, NB_RB_MAX);
  }

#ifdef OAI_EMU
  CHECK_CTXT_ARGS(&ctxt);
  //printf("MBMS_flagP %d, MBMS_FLAG_NO %d \n",MBMS_flagP, MBMS_FLAG_NO);
  //  AssertFatal (MBMS_flagP == MBMS_FLAG_NO ," MBMS FLAG SHOULD NOT BE SET IN mac_rlc_data_req in UE\n");

#endif

  //no MBMS in NB-IoT

  {
    key = RLC_COLL_KEY_LCID_VALUE(module_idP, rntiP, ENB_FLAG_YES, channel_idP, srb_flag);
  }

  h_rc = hashtable_get(rlc_coll_p, key, (void**)&rlc_union_p);

  if (h_rc == HASH_TABLE_OK) {
    rlc_mode = rlc_union_p->mode;
  } else {
    rlc_mode = RLC_MODE_NONE;
    AssertFatal (0 , "mac_rlc_data req --> RLC not configured lcid %u RNTI %x!\n", channel_idP, rntiP);
  }

  switch (rlc_mode) {
  case RLC_MODE_NONE:
    ret_tb_size =0;
    break;

  case RLC_MODE_AM:
    data_request = rlc_am_mac_data_request(&ctxt, &rlc_union_p->rlc.am, ENB_FLAG_YES);
    ret_tb_size =mac_rlc_serialize_tb(buffer_pP, data_request.data);
    break;

    //UM mode not for NB-IoT

  case RLC_MODE_TM:
    data_request = rlc_tm_mac_data_request(&ctxt, &rlc_union_p->rlc.tm);
    ret_tb_size = mac_rlc_serialize_tb(buffer_pP, data_request.data);
    break;

  default:
    ;
  }

#if T_TRACER
  if (enb_flagP)
    T(T_ENB_RLC_MAC_DL, T_INT(module_idP), T_INT(rntiP), T_INT(channel_idP), T_INT(ret_tb_size));
#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_MAC_RLC_DATA_REQ,VCD_FUNCTION_OUT);
  return ret_tb_size;
}

