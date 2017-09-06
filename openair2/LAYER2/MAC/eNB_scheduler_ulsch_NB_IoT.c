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

/*! \file eNB_scheduler_ulsch.c
 * \brief eNB procedures for the ULSCH transport channel
 * \author Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email: navid.nikaein@eurecom.fr
 * \version 1.0
 * @ingroup _mac

 */

//#include "assertions.h"
//#include "PHY/defs.h"
//#include "PHY/extern.h"
#include "PHY/extern_NB_IoT.h"

//#include "SCHED/defs.h"
//#include "SCHED/extern.h"

//#include "LAYER2/MAC/defs.h"
//#include "LAYER2/MAC/proto.h"

#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"  // for trace_pdu() function , description is in probe.c
//#include "OCG.h"
//#include "OCG_extern.h"

//#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"
//NB-IoT
//#include "PHY/defs_NB_IoT.h"
//#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "RRC/LITE/defs_NB_IoT.h"
//#include "LAYER2/MAC/pre_processor.c"
//#include "pdcp.h"

//#if defined(ENABLE_ITTI)
//# include "intertask_interface.h"
//#endif

#include "T.h"

#define ENABLE_MAC_PAYLOAD_DEBUG
//#define DEBUG_eNB_SCHEDULER 1
unsigned char *parse_ulsch_header_NB_IoT(unsigned char *mac_header,
                                         unsigned char *num_ce,
                                         unsigned char *num_sdu,
                                         unsigned char *rx_ces,
                                         unsigned char *rx_lcids,
                                         unsigned short *rx_lengths,
                                         unsigned short tb_length)
{
  //MAC_xface_NB_IoT *mac_xface_NB_IoT; //test_xface

  unsigned char not_done=1,num_ces=0,num_sdus=0,lcid,num_sdu_cnt;
  unsigned char *mac_header_ptr = mac_header;
  unsigned short length, ce_len=0;

  while (not_done==1) {

    if (((SCH_SUBHEADER_FIXED_NB_IoT*)mac_header_ptr)->E == 0) {
      not_done = 0;
    }

    lcid = ((SCH_SUBHEADER_FIXED_NB_IoT *)mac_header_ptr)->LCID;

    if (lcid < EXTENDED_POWER_HEADROOM_NB_IoT) {
      if (not_done==0) { // last MAC SDU, length is implicit
        mac_header_ptr++;
        length = tb_length-(mac_header_ptr-mac_header)-ce_len;

        for (num_sdu_cnt=0; num_sdu_cnt < num_sdus ; num_sdu_cnt++) {
          length -= rx_lengths[num_sdu_cnt];
        }
      } else {
        if (((SCH_SUBHEADER_SHORT_NB_IoT *)mac_header_ptr)->F == 0) {
          length = ((SCH_SUBHEADER_SHORT_NB_IoT *)mac_header_ptr)->L;
          mac_header_ptr += 2;//sizeof(SCH_SUBHEADER_SHORT);
        } else { // F = 1
          length = ((((SCH_SUBHEADER_LONG_NB_IoT *)mac_header_ptr)->L_MSB & 0x7f ) << 8 ) | (((SCH_SUBHEADER_LONG_NB_IoT *)mac_header_ptr)->L_LSB & 0xff);
          mac_header_ptr += 3;//sizeof(SCH_SUBHEADER_LONG);
        }
      }

      LOG_D(MAC,"[eNB] sdu %d lcid %d tb_length %d length %d (offset now %ld)\n",
            num_sdus,lcid,tb_length, length,mac_header_ptr-mac_header);
      rx_lcids[num_sdus] = lcid;
      rx_lengths[num_sdus] = length;
      num_sdus++;
    } else { // This is a control element subheader POWER_HEADROOM, BSR and CRNTI
      if (lcid == SHORT_PADDING_NB_IoT) {
        mac_header_ptr++;
      } else {
        rx_ces[num_ces] = lcid;
        num_ces++;
        mac_header_ptr++;

        if (lcid==LONG_BSR_NB_IoT) {
          ce_len+=3;
        } else if (lcid==CRNTI_NB_IoT) {
          ce_len+=2;
        } else if ((lcid==POWER_HEADROOM_NB_IoT) || (lcid==TRUNCATED_BSR_NB_IoT)|| (lcid== SHORT_BSR_NB_IoT)) {
          ce_len++;
        } else {
          LOG_E(MAC,"unknown CE %d \n", lcid);
          mac_xface_NB_IoT->macphy_exit("unknown CE");
        }
      }
    }
  }

  *num_ce = num_ces;
  *num_sdu = num_sdus;

  return(mac_header_ptr);
}


void rx_sdu_NB_IoT(const module_id_t enb_mod_idP,
	    const int         CC_idP,
	    const frame_t     frameP,
	    const sub_frame_t subframeP,
	    const rnti_t      rntiP,
	    uint8_t          *sduP,
	    const uint16_t    sdu_lenP,
	    const int         harq_pidP
      )
{

  unsigned char  rx_ces[MAX_NUM_CE_NB_IoT],num_ce,num_sdu,i,*payload_ptr;
  unsigned char  rx_lcids[NB_RB_MAX];//for NB-IoT, NB_RB_MAX should be fixed to 5 (2 DRB+ 3SRB) 
  unsigned short rx_lengths[NB_RB_MAX];
  int    UE_id = find_UE_id_NB_IoT(enb_mod_idP,rntiP);
  int ii,j;
  eNB_MAC_INST_NB_IoT *eNB = &eNB_mac_inst_NB_IoT[enb_mod_idP];
  UE_list_NB_IoT_t *UE_list= &eNB->UE_list;
  int crnti_rx=0;
  //int old_buffer_info;

  start_meas(&eNB->rx_ulsch_sdu);

  /*if there is an error for UE_id> max or UE_id==-1, set rx_lengths to 0*/
  if ((UE_id >  NUMBER_OF_UE_MAX_NB_IoT) || (UE_id == -1)  )
    for(ii=0; ii<NB_RB_MAX; ii++) {
      rx_lengths[ii] = 0;
    }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RX_SDU,1);
  if (opt_enabled == 1) {
    trace_pdu(0, sduP,sdu_lenP, 0, 3, rntiP, frameP, subframeP, 0,0);
    LOG_D(OPT,"[eNB %d][ULSCH] Frame %d  rnti %x  with size %d\n",
    		  enb_mod_idP, frameP, rntiP, sdu_lenP);
  }

  LOG_D(MAC,"[eNB %d] CC_id %d Received ULSCH sdu from PHY (rnti %x, UE_id %d), parsing header\n",enb_mod_idP,CC_idP,rntiP,UE_id);

  if (sduP==NULL) { // we've got an error after N rounds
    UE_list->UE_sched_ctrl[UE_id].ul_scheduled       &= (~(1<<harq_pidP)); //ul_scheduled: A kind of resource scheduling information
    return;
  }

  if (UE_id!=-1) {
    UE_list->UE_sched_ctrl[UE_id].ul_inactivity_timer=0;
    UE_list->UE_sched_ctrl[UE_id].ul_failure_timer   =0;
    UE_list->UE_sched_ctrl[UE_id].ul_scheduled       &= (~(1<<harq_pidP));
    /*RLF procedure this part just check UE context is NULL or not, if not, means UL in synch*/
    if (UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync > 0) {
      UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync=0;
      mac_eNB_rrc_ul_in_sync_NB_IoT(enb_mod_idP,CC_idP,frameP,subframeP,UE_RNTI_NB_IoT(enb_mod_idP,UE_id));
    }
  }


  payload_ptr = parse_ulsch_header_NB_IoT(sduP,&num_ce,&num_sdu,rx_ces,rx_lcids,rx_lengths,sdu_lenP);

  T(T_ENB_MAC_UE_UL_PDU, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
    T_INT(harq_pidP), T_INT(sdu_lenP), T_INT(num_ce), T_INT(num_sdu));
  T(T_ENB_MAC_UE_UL_PDU_WITH_DATA, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
    T_INT(harq_pidP), T_INT(sdu_lenP), T_INT(num_ce), T_INT(num_sdu), T_BUFFER(sduP, sdu_lenP));

  eNB->eNB_stats[CC_idP].ulsch_bytes_rx=sdu_lenP;
  eNB->eNB_stats[CC_idP].total_ulsch_bytes_rx+=sdu_lenP;
  eNB->eNB_stats[CC_idP].total_ulsch_pdus_rx+=1;
  // control element
  for (i=0; i<num_ce; i++) {

    T(T_ENB_MAC_UE_UL_CE, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
      T_INT(rx_ces[i]));
    /*rx_ces = lcid in parse_ulsch_header() if not short padding*/
    switch (rx_ces[i]) { // implement and process BSR + CRNTI + PHR
    case POWER_HEADROOM_NB_IoT:
      if (UE_id != -1) {
        UE_list->UE_template[CC_idP][UE_id].phr_info =  (payload_ptr[0] & 0x3f) - PHR_MAPPING_OFFSET_NB_IoT;
        LOG_D(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : Received PHR PH = %d (db)\n",
              enb_mod_idP, CC_idP, rx_ces[i], UE_list->UE_template[CC_idP][UE_id].phr_info);
        UE_list->UE_template[CC_idP][UE_id].phr_info_configured=1;
	UE_list->UE_sched_ctrl[UE_id].phr_received = 1;
      }
      payload_ptr+=sizeof(POWER_HEADROOM_CMD_NB_IoT);
      break;

    case CRNTI_NB_IoT:
      UE_id = find_UE_id_NB_IoT(enb_mod_idP,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1]);
      LOG_I(MAC, "[eNB %d] Frame %d, Subframe %d CC_id %d MAC CE_LCID %d (ce %d/%d): CRNTI %x (UE_id %d) in Msg3\n",
	    frameP,subframeP,enb_mod_idP, CC_idP, rx_ces[i], i,num_ce,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1],UE_id);
      if (UE_id!=-1) {
	UE_list->UE_sched_ctrl[UE_id].ul_inactivity_timer=0;
	UE_list->UE_sched_ctrl[UE_id].ul_failure_timer=0;
	if (UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync > 0) {
	  UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync=0;
	  /*In RRC branch*/
    //mac_eNB_rrc_ul_in_sync_NB_IoT(enb_mod_idP,CC_idP,frameP,subframeP,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1]);
	}
      }
      crnti_rx=1;
      payload_ptr+=2;
    
      break;
    /*For this moment, long bsr is not processed in the case*/
    //case TRUNCATED_BSR:
    /*DV lcid =???*/
    //case DATA_VOLUME_INDICATOR
    case SHORT_BSR_NB_IoT: {
      uint8_t lcgid;
      lcgid = (payload_ptr[0] >> 6);

      LOG_D(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : Received short BSR LCGID = %u bsr = %d\n",
	    enb_mod_idP, CC_idP, rx_ces[i], lcgid, payload_ptr[0] & 0x3f);

      if (crnti_rx==1)
	LOG_I(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : Received short BSR LCGID = %u bsr = %d\n",
	      enb_mod_idP, CC_idP, rx_ces[i], lcgid, payload_ptr[0] & 0x3f);
      if (UE_id  != -1) {

        UE_list->UE_template[CC_idP][UE_id].bsr_info[lcgid] = (payload_ptr[0] & 0x3f);

	// update buffer info
	
	UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid]=BSR_TABLE_NB_IoT[UE_list->UE_template[CC_idP][UE_id].bsr_info[lcgid]];

	UE_list->UE_template[CC_idP][UE_id].ul_total_buffer= UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid];

	PHY_vars_eNB_NB_IoT_g[enb_mod_idP][CC_idP]->pusch_stats_bsr[UE_id][(frameP*10)+subframeP] = (payload_ptr[0] & 0x3f);
	if (UE_id == UE_list->head)
	  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_BSR,PHY_vars_eNB_NB_IoT_g[enb_mod_idP][CC_idP]->pusch_stats_bsr[UE_id][(frameP*10)+subframeP]);	
        if (UE_list->UE_template[CC_idP][UE_id].ul_buffer_creation_time[lcgid] == 0 ) {
          UE_list->UE_template[CC_idP][UE_id].ul_buffer_creation_time[lcgid]=frameP;
        }
	if (mac_eNB_get_rrc_status(enb_mod_idP,UE_RNTI_NB_IoT(enb_mod_idP,UE_id)) < RRC_CONNECTED_NB_IoT)
	  LOG_I(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : ul_total_buffer = %d (lcg increment %d)\n",
		enb_mod_idP, CC_idP, rx_ces[i], UE_list->UE_template[CC_idP][UE_id].ul_total_buffer,
		UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid]);	
      }
      else {

      }
      payload_ptr += 1;//sizeof(SHORT_BSR_NB_IoT); // fixme
    }
    break;

	

    default:
      LOG_E(MAC, "[eNB %d] CC_id %d Received unknown MAC header (0x%02x)\n", enb_mod_idP, CC_idP, rx_ces[i]);
      break;

  }

  for (i=0; i<num_sdu; i++) {
    LOG_D(MAC,"SDU Number %d MAC Subheader SDU_LCID %d, length %d\n",i,rx_lcids[i],rx_lengths[i]);

    T(T_ENB_MAC_UE_UL_SDU, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
      T_INT(rx_lcids[i]), T_INT(rx_lengths[i]));
    T(T_ENB_MAC_UE_UL_SDU_WITH_DATA, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
      T_INT(rx_lcids[i]), T_INT(rx_lengths[i]), T_BUFFER(payload_ptr, rx_lengths[i]));

    switch (rx_lcids[i]) {
    case CCCH_NB_IoT :
      if (rx_lengths[i] > CCCH_PAYLOAD_SIZE_MAX_NB_IoT) {
        LOG_E(MAC, "[eNB %d/%d] frame %d received CCCH of size %d (too big, maximum allowed is %d), dropping packet\n",
              enb_mod_idP, CC_idP, frameP, rx_lengths[i], CCCH_PAYLOAD_SIZE_MAX_NB_IoT);
        break;
      }
      LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, Received CCCH:  %x.%x.%x.%x.%x.%x, Terminating RA procedure for UE rnti %x\n",
            enb_mod_idP,CC_idP,frameP,
            payload_ptr[0],payload_ptr[1],payload_ptr[2],payload_ptr[3],payload_ptr[4], payload_ptr[5], rntiP);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_TERMINATE_RA_PROC,1);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_TERMINATE_RA_PROC,0);
      for (ii=0; ii<RA_PROC_MAX_NB_IoT; ii++) {
        LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Checking proc %d : rnti (%x, %x), active %d\n",
              enb_mod_idP, CC_idP, ii,
              eNB->common_channels[CC_idP].RA_template[ii].rnti, rntiP,
              eNB->common_channels[CC_idP].RA_template[ii].RA_active);

        if ((eNB->common_channels[CC_idP].RA_template[ii].rnti==rntiP) &&
            (eNB->common_channels[CC_idP].RA_template[ii].RA_active==TRUE)) {

          //payload_ptr = parse_ulsch_header(msg3,&num_ce,&num_sdu,rx_ces,rx_lcids,rx_lengths,msg3_len);

          if (UE_id < 0) {
            memcpy(&eNB->common_channels[CC_idP].RA_template[ii].cont_res_id[0],payload_ptr,6);
            LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d CCCH: Received Msg3: length %d, offset %ld\n",
                  enb_mod_idP,CC_idP,frameP,rx_lengths[i],payload_ptr-sduP);

            if ((UE_id=add_new_ue_NB_IoT(enb_mod_idP,CC_idP,eNB->common_channels[CC_idP].RA_template[ii].rnti,harq_pidP)) == -1 ) {
              mac_xface_NB_IoT->macphy_exit("[MAC][eNB] Max user count reached\n");
	      // kill RA procedure
            } else
              LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Added user with rnti %x => UE %d\n",
                    enb_mod_idP,CC_idP,frameP,eNB->common_channels[CC_idP].RA_template[ii].rnti,UE_id);
          } else {
            LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d CCCH: Received Msg3 from already registered UE %d: length %d, offset %ld\n",
                  enb_mod_idP,CC_idP,frameP,UE_id,rx_lengths[i],payload_ptr-sduP);
	    // kill RA procedure
          }

          if (Is_rrc_registered_NB_IoT == 1)
        	 //MP: send directly the information to the RRC in case of CCCH (SRB0)

            mac_rrc_data_ind_eNB_NB_IoT(
              enb_mod_idP,
              CC_idP,
              frameP,
        	  subframeP,
              rntiP,
              CCCH_NB_IoT,
              (uint8_t*)payload_ptr,
              rx_lengths[i]);


          if (num_ce >0) {  // handle msg3 which is not RRCConnectionRequest
            //  process_ra_message(msg3,num_ce,rx_lcids,rx_ces);
          }

          eNB->common_channels[CC_idP].RA_template[ii].generate_Msg4 = 1;
          eNB->common_channels[CC_idP].RA_template[ii].wait_ack_Msg4 = 0;

        } // if process is active
      } // loop on RA processes
      
      break ;
    /*DCCH0 is for SRB1bis, DCCH1 is for SRB1*/
    case DCCH0_NB_IoT :
    case DCCH1_NB_IoT :
      //      if(eNB_mac_inst[module_idP][CC_idP].Dcch_lchan[UE_id].Active==1){
      

#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
      LOG_T(MAC,"offset: %d\n",(unsigned char)((unsigned char*)payload_ptr-sduP));
      for (j=0; j<32; j++) {
        LOG_T(MAC,"%x ",payload_ptr[j]);
      }
      LOG_T(MAC,"\n");
#endif

      if (UE_id != -1) {

	/*NO lcg in NB-IoT, anyway set to 0*/
	// adjust buffer occupancy of the correponding logical channel group
	/*if (UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] >= rx_lengths[i])
	  UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] -= rx_lengths[i];
	else
	  UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] = 0;*/

          LOG_D(MAC,"[eNB %d] CC_id %d Frame %d : ULSCH -> UL-DCCH, received %d bytes form UE %d on LCID %d \n",
                enb_mod_idP,CC_idP,frameP, rx_lengths[i], UE_id, rx_lcids[i]);

        mac_rlc_data_ind_NB_IoT(
			   enb_mod_idP,
			   rntiP,
			   enb_mod_idP,
			   frameP,
			   ENB_FLAG_YES,
			   rx_lcids[i],
			   (char *)payload_ptr,
			   rx_lengths[i],
			   1,
			   NULL);//(unsigned int*)crc_status);*/

          UE_list->eNB_UE_stats[CC_idP][UE_id].num_pdu_rx[rx_lcids[i]]+=1;
          UE_list->eNB_UE_stats[CC_idP][UE_id].num_bytes_rx[rx_lcids[i]]+=rx_lengths[i];
      } /* UE_id != -1 */
 
      // } 
      break;

      // all the DRBS
    case DTCH0_NB_IoT:
    default :

#if defined(ENABLE_MAC_PAYLOAD_DEBUG)
      LOG_T(MAC,"offset: %d\n",(unsigned char)((unsigned char*)payload_ptr-sduP));
      for (j=0; j<32; j++) {
        LOG_T(MAC,"%x ",payload_ptr[j]);
      }
      LOG_T(MAC,"\n");
#endif
      if (rx_lcids[i]  < NB_RB_MAX ) {
	LOG_D(MAC,"[eNB %d] CC_id %d Frame %d : ULSCH -> UL-DTCH, received %d bytes from UE %d for lcid %d\n",
	      enb_mod_idP,CC_idP,frameP, rx_lengths[i], UE_id, rx_lcids[i]);
	
	if (UE_id != -1) {
	  // adjust buffer occupancy of the correponding logical channel group
	  LOG_D(MAC,"[eNB %d] CC_id %d Frame %d : ULSCH -> UL-DTCH, received %d bytes from UE %d for lcid %d\n",
		enb_mod_idP,CC_idP,frameP, rx_lengths[i], UE_id,rx_lcids[i]);
	  
	  /*if (UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] >= rx_lengths[i])
	    UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] -= rx_lengths[i];
	  else
	    UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[UE_list->UE_template[CC_idP][UE_id].lcgidmap[rx_lcids[i]]] = 0;*/
	  if ((rx_lengths[i] <SCH_PAYLOAD_SIZE_MAX_NB_IoT) &&  (rx_lengths[i] > 0) ) {   // MAX SIZE OF transport block

	    mac_rlc_data_ind_NB_IoT(
			     enb_mod_idP,
			     rntiP,
			     enb_mod_idP,
			     frameP,
			     ENB_FLAG_YES,
			     rx_lcids[i],
			     (char *)payload_ptr,
			     rx_lengths[i],
			     1,
			     NULL);//(unsigned int*)crc_status);*/
	    
	    UE_list->eNB_UE_stats[CC_idP][UE_id].num_pdu_rx[rx_lcids[i]]+=1;
	    UE_list->eNB_UE_stats[CC_idP][UE_id].num_bytes_rx[rx_lcids[i]]+=rx_lengths[i];
	  }
	  else { /* rx_length[i] */
	    UE_list->eNB_UE_stats[CC_idP][UE_id].num_errors_rx+=1;
	    LOG_E(MAC,"[eNB %d] CC_id %d Frame %d : Max size of transport block reached LCID %d from UE %d ",
		  enb_mod_idP, CC_idP, frameP, rx_lcids[i], UE_id);
	  }
	}    
	else {/*(UE_id != -1*/ 
	  LOG_E(MAC,"[eNB %d] CC_id %d Frame %d : received unsupported or unknown LCID %d from UE %d ",
		enb_mod_idP, CC_idP, frameP, rx_lcids[i], UE_id);
	}
      }

      break;
    }
  
    payload_ptr+=rx_lengths[i];
  }

  /* NN--> FK: we could either check the payload, or use a phy helper to detect a false msg3 */
  if ((num_sdu == 0) && (num_ce==0)) {
    if (UE_id != -1)
      UE_list->eNB_UE_stats[CC_idP][UE_id].total_num_errors_rx+=1;

  } else {
    if (UE_id != -1) {
      UE_list->eNB_UE_stats[CC_idP][UE_id].pdu_bytes_rx=sdu_lenP;
      UE_list->eNB_UE_stats[CC_idP][UE_id].total_pdu_bytes_rx+=sdu_lenP;
      UE_list->eNB_UE_stats[CC_idP][UE_id].total_num_pdus_rx+=1;
    }
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_RX_SDU,0);
  stop_meas(&eNB->rx_ulsch_sdu);
}
}

/* This function is called by PHY layer when it schedules some
 * uplink for a random access message 3.
 * The MAC scheduler has to skip the RBs used by this message 3
 * (done below in schedule_ulsch).
 */
void set_msg3_subframe_NB_IoT(module_id_t Mod_id,
                              int CC_id,
                              int frame,
                              int subframe,
                              int rnti,
                              int Msg3_frame,
                              int Msg3_subframe)
{
  eNB_MAC_INST_NB_IoT *eNB=&eNB_mac_inst_NB_IoT[Mod_id];
  int i;
  for (i=0; i<RA_PROC_MAX_NB_IoT; i++) {
    if (eNB->common_channels[CC_id].RA_template[i].RA_active == TRUE &&
        eNB->common_channels[CC_id].RA_template[i].rnti == rnti) {
      eNB->common_channels[CC_id].RA_template[i].Msg3_subframe = Msg3_subframe;
      break;
    }
  }
}


