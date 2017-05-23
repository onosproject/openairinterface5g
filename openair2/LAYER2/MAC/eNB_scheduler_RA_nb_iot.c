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

/*! \file eNB_scheduler_RA.c
 * \brief primitives used for random access
 * \author  Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email: navid.nikaein@eurecom.fr
 * \version 1.0
 * @ingroup _mac

 */

#include "assertions.h"
#include "platform_types.h"
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "msc.h"

#include "SCHED/defs.h"
#include "SCHED/extern.h"

#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"

#include "LAYER2/MAC/proto.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"

//NB-IoT
#include "proto_nb_iot.h"
#include "defs_nb_iot.h"
#include "math.h"
#include "openair1/PHY/LTE_TRANSPORT/dci_nb_iot.h"
//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#include "SIMULATION/TOOLS/defs.h" // for taus

#include "T.h"

void NB_schedule_RA(module_id_t module_idP,frame_t frameP, sub_frame_t subframeP)
{

  int CC_id;
  eNB_MAC_INST_NB *eNB = &eNB_mac_inst_NB[module_idP];


  RA_TEMPLATE_NB *RA_template;
  unsigned char i,harq_pid,round;
  int16_t rrc_sdu_length;
  unsigned char lcid,offset;
  int UE_id = -1;
  unsigned short TBsize = -1;
  unsigned short msg4_padding,msg4_post_padding,msg4_header;
  DCI_PDU_NB *DCI_pdu;

  // start_meas(&eNB->schedule_ra);

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {


    
    DCI_pdu = &eNB->common_channels[CC_id].DCI_pdu;

    for (i=0; i<NB_RA_PROC_MAX; i++) {

      RA_template = (RA_TEMPLATE_NB *)&eNB->common_channels[CC_id].RA_template[i];

      if (RA_template->RA_active == TRUE) {

        LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d RA %d is active (generate RAR %d, generate_Msg4 %d, wait_ack_Msg4 %d, rnti %x)\n",
              module_idP,CC_id,i,RA_template->generate_rar,RA_template->generate_Msg4,RA_template->wait_ack_Msg4, RA_template->rnti);

        if (RA_template->generate_rar == 1) {

          LOG_D(MAC,"[eNB %d] CC_id %d Frame %d, subframeP %d: Generating RAR DCI (proc %d), RA_active %d format 1A (%d,%d))\n",
                module_idP, CC_id, frameP, subframeP,i,
                RA_template->RA_active,
                RA_template->RA_dci_fmt1,
                RA_template->RA_dci_size_bits1);
              //directly fill DCI Filed base on DCI Nq for RAR
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->type=1;
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->orderIndicator=0;
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->Scheddly=1;//fixed delay approach?
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->ResAssign=0;
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->mcs=0;//fixes?//fixes? base on CE levels?
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->RepNum=0;//fixes? base on CE levels?
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->Reserved=0;
              ((DCIFormatN1_RAR_t*)&RA_template->RA_alloc_pdu1[0])->DCIRep=0;//fixes?
        }
    //New appoach for CCE allocaton, delete !CCE_allocation_infeasible..
        else if (RA_template->generate_Msg4 == 1) {

          // check for Msg4 Message
          UE_id = find_UE_id(module_idP,RA_template->rnti);
          if (UE_id == -1) { printf("%s:%d:%s: FATAL ERROR\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

          if (Is_rrc_registered == 1) {//Fixed mac_rrc_data_req

            // Get RRCConnectionSetup for Piggyback
            rrc_sdu_length = mac_rrc_data_req(module_idP,
                                              CC_id,
                                              frameP,
                                              CCCH,
                                              1, // 1 transport block
                                              &eNB->common_channels[CC_id].CCCH_pdu.payload[0],
                                              ENB_FLAG_YES,
                                              module_idP,
                                              0); // not used in this case

            if (rrc_sdu_length == -1) {
              mac_xface->macphy_exit("[MAC][eNB Scheduler] CCCH not allocated\n");
              return; // not reached
            } else {
              //msg("[MAC][eNB %d] Frame %d, subframeP %d: got %d bytes from RRC\n",module_idP,frameP, subframeP,rrc_sdu_length);
            }
          }

          LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, subframeP %d: UE_id %d, Is_rrc_registered %d, rrc_sdu_length %d\n",
                module_idP, CC_id, frameP, subframeP,UE_id, Is_rrc_registered,rrc_sdu_length);

          if (rrc_sdu_length>0) {
            LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, subframeP %d: Generating Msg4 with RRC Piggyback (RA proc %d, RNTI %x)\n",
                  module_idP, CC_id, frameP, subframeP,i,RA_template->rnti);
            // Compute MCS for 3 PRB
            msg4_header = 1+6+1;  // CR header, CR CE, SDU header

            //need to fixed ndi & msc base on NB-IoT DCI for Msg4
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->ndi=1;
            if ((rrc_sdu_length+msg4_header) <= 22) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=4;
              TBsize = 22;
            } else if ((rrc_sdu_length+msg4_header) <= 28) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=5;
              TBsize = 28;
            } else if ((rrc_sdu_length+msg4_header) <= 32) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=6;
              TBsize = 32;
            } else if ((rrc_sdu_length+msg4_header) <= 41) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=7;
              TBsize = 41;
            } else if ((rrc_sdu_length+msg4_header) <= 49) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=8;
              TBsize = 49;
            } else if ((rrc_sdu_length+msg4_header) <= 57) {
              ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->mcs=9;
              TBsize = 57;
            }

            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->type=1;
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->orderIndicator=0;
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->Scheddly=1;//fixed delay approach?
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->ResAssign=5;//fixed depend on mcs/tbs to Nsf
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->RepNum=1;//fixed base on CE levels
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->HARQackRes=0;//Avoid confict multiple Msg ACk
            ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->DCIRep=0;//fixed base on CE levels
            }
      }else if (RA_template->wait_ack_Msg4==1) {
  // check HARQ status and retransmit if necessary
  LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, subframeP %d: Checking if Msg4 was acknowledged: \n",
        module_idP,CC_id,frameP,subframeP);
  // Get candidate harq_pid from PHY
  mac_xface->get_ue_active_harq_pid(module_idP,CC_id,RA_template->rnti,frameP,subframeP,&harq_pid,&round,openair_harq_RA);
  
  if (round>0) {
    //RA_template->wait_ack_Msg4++;
    // we have to schedule a retransmission
    ((DCIFormatN1_t*)&RA_template->RA_alloc_pdu2[0])->ndi=1;
    
    //     if (!CCE_allocation_infeasible(module_idP,CC_id,0,subframeP,2,RA_template->rnti)) {
    //       add_ue_spec_dci(DCI_pdu,
    //           (void*)&RA_template->RA_alloc_pdu2[0],
    //           RA_template->rnti,
    //           RA_template->RA_dci_size_bytes2,
    //           2,
    //           RA_template->RA_dci_size_bits2,
    //           RA_template->RA_dci_fmt2,
    //           0);
    // printf("MAC: msg4 retransmission for rnti %x (round %d) fsf %d/%d\n", RA_template->rnti, round, frameP, subframeP);
    //     }
      }else
      printf("MAC: msg4 retransmission for rnti %x (round %d) fsf %d/%d CCE allocation failed!\n", RA_template->rnti, round, frameP, subframeP);
          LOG_W(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, subframeP %d: Msg4 not acknowledged, adding ue specific dci (rnti %x) for RA (Msg4 Retransmission)\n",
          module_idP,CC_id,frameP,subframeP,RA_template->rnti);
  } else {
    /*      msg4 not received
      if ((round == 0) && (RA_template->wait_ack_Msg4>1){
      remove UE instance across all the layers: mac_xface->cancel_RA();
      }
    */
printf("MAC: msg4 acknowledged for rnti %x fsf %d/%d, let's configure it\n", RA_template->rnti, frameP, subframeP);
    LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d, subframeP %d : Msg4 acknowledged\n",module_idP,CC_id,frameP,subframeP);
    RA_template->wait_ack_Msg4=0;
    RA_template->RA_active=FALSE;
    UE_id = find_UE_id(module_idP,RA_template->rnti);
    DevAssert( UE_id != -1 );
    eNB_mac_inst_NB[module_idP].UE_list.UE_template[UE_PCCID(module_idP,UE_id)][UE_id].configured=TRUE;
    
  }
  
    }// RA_active
  } // for i=0 .. N_RA_PROC-1 
} // CC_id

  // stop_meas(&eNB->schedule_ra);
}
/*This function should loop all over the preamble index*/
void NB_initiate_ra_proc(module_id_t module_idP, int CC_id,frame_t frameP, uint16_t preamble_index,int16_t timing_offset,sub_frame_t subframeP)
{

  uint8_t i;
  uint8_t carrier_id = 0;/*The index of the UL carrier associated with the NPRACH, the carrier_id of the anchor carrier is 0*/

  RA_TEMPLATE_NB *RA_template = (RA_TEMPLATE_NB *)&eNB_mac_inst_NB[module_idP].common_channels[CC_id].RA_template[0];
    /*preamble index will be a subcarrier index (0-47)*/
  LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Initiating RA procedure for preamble index %d\n",module_idP,CC_id,frameP,preamble_index);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC,1);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_INITIATE_RA_PROC,0);
  /*May up to 48 RA Procdeure MAX at the moment*/
  for (i=0; i<NB_RA_PROC_MAX; i++) {
    if (RA_template[i].RA_active==FALSE &&
        RA_template[i].wait_ack_Msg4 == 0) {
      int loop = 0;
      RA_template[i].RA_active=TRUE;
      RA_template[i].generate_rar=1;
      RA_template[i].generate_Msg4=0;
      RA_template[i].wait_ack_Msg4=0;
      RA_template[i].timing_offset=timing_offset;
      /* TODO: find better procedure to allocate RNTI */
      do {
        RA_template[i].rnti = taus();
        loop++;
      } while (loop != 100 &&
               /* TODO: this is not correct, the rnti may be in use without
                * being in the MAC yet. To be refined.
                */
               !(find_UE_id(module_idP, RA_template[i].rnti) == -1 &&
                 /* 1024 and 60000 arbirarily chosen, not coming from standard */
                 RA_template[i].rnti >= 1024 && RA_template[i].rnti < 60000));
      if (loop == 100) { printf("%s:%d:%s: FATAL ERROR! contact the authors\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
      //RA_template[i].RA_rnti = 1+subframeP+(10*f_id);
      /*for NB-IoT, RA_rnti is counted in 36.321 5.1.4*/
      RA_template[i].RA_rnti = 1+floor(frameP/4)+256*carrier_id;
      RA_template[i].preamble_index = preamble_index;
      LOG_D(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d Activating RAR generation for process %d, rnti %x, RA_active %d\n",
            module_idP,CC_id,frameP,i,RA_template[i].rnti,
            RA_template[i].RA_active);

      return;
    }
  }

  LOG_E(MAC,"[eNB %d][RAPROC] FAILURE: CC_id %d Frame %d Initiating RA procedure for preamble index %d\n",module_idP,CC_id,frameP,preamble_index);
}



