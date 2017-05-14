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

#include "assertions.h"
#include "PHY/defs.h"
#include "PHY/extern.h"

#include "SCHED/defs.h"
#include "SCHED/extern.h"

//#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/proto.h"
#include "LAYER2/MAC/extern.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"

#include "LAYER2/MAC/defs-nb.h"
#include "LAYER2/MAC/proto-nb.h"


//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#include "T.h"

#define ENABLE_MAC_PAYLOAD_DEBUG
#define DEBUG_eNB_SCHEDULER 1

// This table holds the allowable PRB sizes for ULSCH transmissions
uint8_t rb_table[33] = {1,2,3,4,5,6,8,9,10,12,15,16,18,20,24,25,27,30,32,36,40,45,48,50,54,60,72,75,80,81,90,96,100};



void NB_rx_sdu(const module_id_t enb_mod_idP,
	    const int         CC_idP,
	    const frame_t     frameP,
	    const sub_frame_t subframeP,
	    const rnti_t      rntiP,
	    uint8_t          *sduP,
	    const uint16_t    sdu_lenP,
	    const int         harq_pidP,
	    uint8_t          *msg3_flagP)
{

  unsigned char  rx_ces[MAX_NUM_CE],num_ce,num_sdu,i,*payload_ptr;
  unsigned char  rx_lcids[NB_RB_MAX];//for NB-IoT, NB_RB_MAX should be fixed to 5 (2 DRB+ 3SRB) 
  unsigned short rx_lengths[NB_RB_MAX];
  int    UE_id = find_UE_id(enb_mod_idP,rntiP);
  int ii,j;
  eNB_MAC_INST_NB *eNB = &eNB_mac_inst_NB[enb_mod_idP];
  UE_list_NB_t *UE_list= &eNB->UE_list;
  int crnti_rx=0;
  //int old_buffer_info;

  start_meas(&eNB->rx_ulsch_sdu);

  /*if there is an error for UE_id> max or UE_id==-1, set rx_lengths to 0*/
  if ((UE_id >  NUMBER_OF_UE_MAX) || (UE_id == -1)  )
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
      NB_mac_eNB_rrc_ul_in_sync(enb_mod_idP,CC_idP,frameP,subframeP,UE_RNTI(enb_mod_idP,UE_id));
    }
  }


  payload_ptr = parse_ulsch_header(sduP,&num_ce,&num_sdu,rx_ces,rx_lcids,rx_lengths,sdu_lenP);

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
    case POWER_HEADROOM:
      if (UE_id != -1) {
        UE_list->UE_template[CC_idP][UE_id].phr_info =  (payload_ptr[0] & 0x3f) - PHR_MAPPING_OFFSET;
        LOG_D(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : Received PHR PH = %d (db)\n",
              enb_mod_idP, CC_idP, rx_ces[i], UE_list->UE_template[CC_idP][UE_id].phr_info);
        UE_list->UE_template[CC_idP][UE_id].phr_info_configured=1;
	UE_list->UE_sched_ctrl[UE_id].phr_received = 1;
      }
      payload_ptr+=sizeof(POWER_HEADROOM_CMD);
      break;

    case CRNTI:
      UE_id = find_UE_id(enb_mod_idP,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1]);
      LOG_I(MAC, "[eNB %d] Frame %d, Subframe %d CC_id %d MAC CE_LCID %d (ce %d/%d): CRNTI %x (UE_id %d) in Msg3\n",
	    frameP,subframeP,enb_mod_idP, CC_idP, rx_ces[i], i,num_ce,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1],UE_id);
      if (UE_id!=-1) {
	UE_list->UE_sched_ctrl[UE_id].ul_inactivity_timer=0;
	UE_list->UE_sched_ctrl[UE_id].ul_failure_timer=0;
	if (UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync > 0) {
	  UE_list->UE_sched_ctrl[UE_id].ul_out_of_sync=0;
	  NB_mac_eNB_rrc_ul_in_sync(enb_mod_idP,CC_idP,frameP,subframeP,(((uint16_t)payload_ptr[0])<<8) + payload_ptr[1]);
	}
      }
      crnti_rx=1;
      payload_ptr+=2;

      if (msg3_flagP != NULL) {
		*msg3_flagP = 0;
    
      break;
    /*For this moment, long bsr is not processed in the case*/
    //case TRUNCATED_BSR:
    /*DV lcid =???*/
    //case DATA_VOLUME_INDICATOR
    case SHORT_BSR: {
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
	
	UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid]=BSR_TABLE[UE_list->UE_template[CC_idP][UE_id].bsr_info[lcgid]];

	UE_list->UE_template[CC_idP][UE_id].ul_total_buffer= UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid];

	PHY_vars_eNB_g[enb_mod_idP][CC_idP]->pusch_stats_bsr[UE_id][(frameP*10)+subframeP] = (payload_ptr[0] & 0x3f);
	if (UE_id == UE_list->head)
	  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_BSR,PHY_vars_eNB_g[enb_mod_idP][CC_idP]->pusch_stats_bsr[UE_id][(frameP*10)+subframeP]);	
        if (UE_list->UE_template[CC_idP][UE_id].ul_buffer_creation_time[lcgid] == 0 ) {
          UE_list->UE_template[CC_idP][UE_id].ul_buffer_creation_time[lcgid]=frameP;
        }
	if (mac_eNB_get_rrc_status(enb_mod_idP,UE_RNTI(enb_mod_idP,UE_id)) < RRC_CONNECTED)
	  LOG_I(MAC, "[eNB %d] CC_id %d MAC CE_LCID %d : ul_total_buffer = %d (lcg increment %d)\n",
		enb_mod_idP, CC_idP, rx_ces[i], UE_list->UE_template[CC_idP][UE_id].ul_total_buffer,
		UE_list->UE_template[CC_idP][UE_id].ul_buffer_info[lcgid]);	
      }
      else {

      }
      payload_ptr += 1;//sizeof(SHORT_BSR); // fixme
    }
    break;

	

    default:
      LOG_E(MAC, "[eNB %d] CC_id %d Received unknown MAC header (0x%02x)\n", enb_mod_idP, CC_idP, rx_ces[i]);
      break;
    }
  }

  for (i=0; i<num_sdu; i++) {
    LOG_D(MAC,"SDU Number %d MAC Subheader SDU_LCID %d, length %d\n",i,rx_lcids[i],rx_lengths[i]);

    T(T_ENB_MAC_UE_UL_SDU, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
      T_INT(rx_lcids[i]), T_INT(rx_lengths[i]));
    T(T_ENB_MAC_UE_UL_SDU_WITH_DATA, T_INT(enb_mod_idP), T_INT(CC_idP), T_INT(rntiP), T_INT(frameP), T_INT(subframeP),
      T_INT(rx_lcids[i]), T_INT(rx_lengths[i]), T_BUFFER(payload_ptr, rx_lengths[i]));

    switch (rx_lcids[i]) {
    case CCCH :
      if (rx_lengths[i] > CCCH_PAYLOAD_SIZE_MAX) {
        LOG_E(MAC, "[eNB %d/%d] frame %d received CCCH of size %d (too big, maximum allowed is %d), dropping packet\n",
              enb_mod_idP, CC_idP, frameP, rx_lengths[i], CCCH_PAYLOAD_SIZE_MAX);
        break;
      }
      LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, Received CCCH:  %x.%x.%x.%x.%x.%x, Terminating RA procedure for UE rnti %x\n",
            enb_mod_idP,CC_idP,frameP,
            payload_ptr[0],payload_ptr[1],payload_ptr[2],payload_ptr[3],payload_ptr[4], payload_ptr[5], rntiP);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_TERMINATE_RA_PROC,1);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_TERMINATE_RA_PROC,0);
      for (ii=0; ii<NB_RA_PROC_MAX; ii++) {
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

            if ((UE_id=add_new_ue(enb_mod_idP,CC_idP,eNB->common_channels[CC_idP].RA_template[ii].rnti,harq_pidP)) == -1 ) {
              mac_xface->macphy_exit("[MAC][eNB] Max user count reached\n");
	      // kill RA procedure
            } else
              LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Added user with rnti %x => UE %d\n",
                    enb_mod_idP,CC_idP,frameP,eNB->common_channels[CC_idP].RA_template[ii].rnti,UE_id);
          } else {
            LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d CCCH: Received Msg3 from already registered UE %d: length %d, offset %ld\n",
                  enb_mod_idP,CC_idP,frameP,UE_id,rx_lengths[i],payload_ptr-sduP);
	    // kill RA procedure
          }

          if (Is_rrc_registered == 1)
            NB_mac_rrc_data_ind(
              enb_mod_idP,
              CC_idP,
              frameP,subframeP,
              rntiP,
              CCCH,
              (uint8_t*)payload_ptr,
              rx_lengths[i],
              ENB_FLAG_YES,
              enb_mod_idP,
              0);


          if (num_ce >0) {  // handle msg3 which is not RRCConnectionRequest
            //  process_ra_message(msg3,num_ce,rx_lcids,rx_ces);
          }

          eNB->common_channels[CC_idP].RA_template[ii].generate_Msg4 = 1;
          eNB->common_channels[CC_idP].RA_template[ii].wait_ack_Msg4 = 0;

        } // if process is active
      } // loop on RA processes
      
      break ;
    /*DCCH0 is for SRB1bis, DCCH1 is for SRB1*/
    case DCCH0 :
    case DCCH1 :
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

		  /*TODO*/
          /*mac_rlc_data_ind(
			   enb_mod_idP,
			   rntiP,
			   enb_mod_idP,
			   frameP,
			   ENB_FLAG_YES,
			   MBMS_FLAG_NO,
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
    case DTCH0:
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
	  if ((rx_lengths[i] <SCH_PAYLOAD_SIZE_MAX) &&  (rx_lengths[i] > 0) ) {   // MAX SIZE OF transport block
	    /*mac_rlc_data_ind(
			     enb_mod_idP,
			     rntiP,
			     enb_mod_idP,
			     frameP,
			     ENB_FLAG_YES,
			     MBMS_FLAG_NO,
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

    if (msg3_flagP != NULL) {
      if( *msg3_flagP == 1 ) {
        LOG_N(MAC,"[eNB %d] CC_id %d frame %d : false msg3 detection: signal phy to canceling RA and remove the UE\n", enb_mod_idP, CC_idP, frameP);
        *msg3_flagP=0;
      }
    }
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
\

