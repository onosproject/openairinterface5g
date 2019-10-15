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

/*! \file phy_procedures_lte_eNB.c
 * \brief Implementation of eNB procedures from 36.213 LTE specifications
 * \author R. Knopp, F. Kaltenberger, N. Nikaein, X. Foukas, Michele Paffetti, Nick Ho
 * \date 2011
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr,navid.nikaein@eurecom.fr, x.foukas@sms.ed.ac.uk, michele.paffetti@studio.unibo.it, nick133371@gmail.com
 * \note
 * \warning
 */
#include "PHY/defs_eNB.h"
#include "PHY/defs_UE.h"
#include "PHY/defs_L1_NB_IoT.h"
#include "PHY/phy_extern.h"
#include "PHY/LTE_ESTIMATION/defs_NB_IoT.h"
#include "PHY/NBIoT_TRANSPORT/defs_NB_IoT.h"
#include "PHY/NBIoT_TRANSPORT/proto_NB_IoT.h"
//#include "PHY/extern_NB_IoT.h" //where we get the global Sched_Rsp_t structure filled
//#include "SCHED/defs.h"
#include "SCHED_NBIOT/sched_common_extern_NB_IoT.h"
//#include "PHY/LTE_TRANSPORT/if4_tools.h"
//#include "PHY/LTE_TRANSPORT/if5_tools.h"
#include "RRC/NBIOT/proto_NB_IoT.h"
#include "SIMULATION/TOOLS/sim.h"  // purpose: included for taus() function
//#ifdef EMOS
//#include "SCHED/phy_procedures_emos.h"
//#endif

// for NB-IoT
#include "SCHED_NBIOT/defs_NB_IoT.h"
#include "openair2/RRC/NBIOT/proto_NB_IoT.h"
#include "openair2/RRC/NBIOT/extern_NB_IoT.h"
#include "RRC/NBIOT/MESSAGES/asn1_msg_NB_IoT.h"
//#define DEBUG_PHY_PROC (Already defined in cmake)
//#define DEBUG_ULSCH

#include "LAYER2/MAC/extern_NB_IoT.h"
#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/mac_proto.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "assertions.h"
#include "msc.h"

#include <time.h>

#if defined(ENABLE_ITTI)
#   include "intertask_interface.h"
#endif

#include "PHY/extern_NB_IoT.h"

#if defined(FLEXRAN_AGENT_SB_IF)
//Agent-related headers
#include "ENB_APP/flexran_agent_extern.h"
#include "ENB_APP/CONTROL_MODULES/MAC/flexran_agent_mac.h"
#include "LAYER2/MAC/flexran_agent_mac_proto.h"
#endif


//extern eNB_MAC_INST_NB_IoT *eNB_mac_inst; // For NB-IoT branch


/*

#if defined(FLEXRAN_AGENT_SB_IF)
//Agent-related headers
#include "ENB_APP/flexran_agent_extern.h"
#include "ENB_APP/CONTROL_MODULES/MAC/flexran_agent_mac.h"
#include "LAYER2/MAC/flexran_agent_mac_proto.h"
#endif
*/

//#define DIAG_PHY

///#define NS_PER_SLOT 500000

///#define PUCCH 1

//DCI_ALLOC_t dci_alloc[8];

///#ifdef EMOS
///fifo_dump_emos_eNB emos_dump_eNB;
///#endif

int npdsch_rep_to_array[3]      = {4,8,16}; //TS 36.213 Table 16.4.1.3-3
int sib1_startFrame_to_array[4] = {0,16,32,48};//TS 36.213 Table 16.4.1.3-4

//New----------------------------------------------------
//return -1 whenever no SIB1-NB transmission occur.
//return sib1_startFrame when transmission occur in the current frame
uint32_t is_SIB1_NB_IoT(const frame_t          frameP,
                        long                   schedulingInfoSIB1,   //from the mib
                        int                    physCellId,           //by configuration
                        NB_IoT_eNB_NDLSCH_t   *ndlsch_SIB1
                        )
{
  uint8_t    nb_rep=0; // number of sib1-nb repetitions within the 256 radio frames
  uint32_t   sib1_startFrame;
  uint32_t   sib1_period_NB_IoT = 256;//from specs TS 36.331 (rf)
  uint8_t    index;
  int        offset;
  int        period_nb; // the number of the actual period over the 1024 frames

        if(schedulingInfoSIB1 > 11 || schedulingInfoSIB1 < 0){
          LOG_E(RRC, "is_SIB1_NB_IoT: schedulingInfoSIB1 value not allowed");
          return 0;
        }


        //SIB1-NB period number
        period_nb = (int) frameP/sib1_period_NB_IoT;


        //number of repetitions
        nb_rep = npdsch_rep_to_array[schedulingInfoSIB1%3];

        //based on number of rep. and the physical cell id we derive the starting radio frame (TS 36.213 Table 16.4.1.3-3/4)
        switch(nb_rep)
        {
        case 4:
          //physCellId%4 possible value are 0,1,2,3
          sib1_startFrame = sib1_startFrame_to_array[physCellId%4];
          break;
        case 8:
          //physCellId%2possible value are 0,1
          sib1_startFrame = sib1_startFrame_to_array[physCellId%2];
          break;
        case 16:
          //physCellId%2 possible value are 0,1
          if(physCellId%2 == 0)
            sib1_startFrame = 0;
          else
            sib1_startFrame = 1; // the only case in which the starting frame is odd
          break;
        default:
          LOG_E(RRC, "Number of repetitions %d not allowed", nb_rep);
          return -1;
        }

        //check the actual frame w.r.t SIB1-NB starting frame
        if(frameP < sib1_startFrame + period_nb*256){
          LOG_T(RRC, "the actual frame %d is before the SIB1-NB starting frame %d of the period--> bcch_sdu_legnth = 0", frameP, sib1_startFrame + period_nb*256);
          return -1;
        }


        //calculate offset between SIB1-NB repetitions (repetitions are equally spaced)
        offset = (sib1_period_NB_IoT-(16*nb_rep))/nb_rep;

        //loop over the SIB1-NB period
        for( int i = 0; i < nb_rep; i++)
        {
          //find the correct sib1-nb repetition interval in which the actual frame is

          //this is the start frame of a repetition
          index = sib1_startFrame+ i*(16+offset) + period_nb*256;

          //the actual frame is in a gap between two consecutive repetitions
          if(frameP < index)
          {
              ndlsch_SIB1->sib1_rep_start      = 0;
              ndlsch_SIB1->relative_sib1_frame = 0;
                return -1;
          }
          //this is needed for ndlsch_procedure
          else if(frameP == index)
          {
            //the actual frame is the start of a new repetition (SIB1-NB should be retransmitted)
            ndlsch_SIB1->sib1_rep_start      = 1;
            ndlsch_SIB1->relative_sib1_frame = 1;
            return sib1_startFrame;
          }
          else
            ndlsch_SIB1->sib1_rep_start = 0;

          //check in the current SIB1_NB repetition
          if(frameP>= index && frameP <= (index+15))
          {
            //find if the actual frame is one of the "every other frame in 16 continuous frame" in which SIB1-NB is transmitted

            for(int y = 0; y < 16; y += 2) //every other frame (increment by 2)
            {
              if(frameP == index + y)
              {
                //this flag tell which is the number of the current frame w.r.t the 8th (over the continuous 16) in a repetition
                ndlsch_SIB1->relative_sib1_frame = y/2 + 1; //1st, 2nd, 3rd,...
                return sib1_startFrame;
              }
            }

            //if we are here means that the frame was inside the repetition interval but not considered for SIB1-NB transmission
            ndlsch_SIB1->relative_sib1_frame = 0;
            return -1;
         }

        }

        return -1;
}

/* For NB-IoT, we put NPBCH in later part, since it would be scheduled by MAC scheduler
* It generates NRS/NPSS/NSSS
*
*/
void common_signal_procedures_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc) 
{
  //LTE_DL_FRAME_PARMS   *fp       =  &eNB->frame_parms_NB_IoT;
  LTE_DL_FRAME_PARMS   *fp       =  &eNB->frame_parms;
  NB_IoT_eNB_NPBCH_t   *broadcast_str = &eNB->npbch;
  //NB_IoT_eNB_NDLSCH_t  *sib1          = &eNB->ndlsch_SIB;
  //NB_IoT_eNB_NDLSCH_t  *ndlsch        = &eNB->ndlsch_SIB1;
  NB_IoT_eNB_NDLSCH_t *sib1          = eNB->ndlsch_SIB1;
  NB_IoT_eNB_NDLSCH_t  *sib23         = eNB->ndlsch_SIB23;

  uint8_t      *npbch_pdu =  broadcast_str->pdu;

  int                     **txdataF =  eNB->common_vars.txdataF[0];
  uint32_t                subframe  =  proc->subframe_tx;
  uint32_t                frame     =  proc->frame_tx;
  //uint16_t                Ntti      =  10;                      //ntti = 10
  int                     RB_IoT_ID=22;                          // XXX should be initialized (RB reserved for NB-IoT, PRB index)
  int                     With_NSSS=0;                            // With_NSSS = 1; if the frame include a sub-Frame with NSSS signal
  uint8_t                 release_v13_5_0 = 0;

  uint32_t                hyper_frame=proc->HFN;

  fp->flag_free_sf =0;
  ////////////////////////////////////////////////////////////////////////////////////
  /*
  rrc_eNB_carrier_data_NB_IoT_t *carrier = &eNB_rrc_inst_NB_IoT->carrier[0];
      if(frame%64==0 && subframe ==0)
      {//printf("dooooo MIB");

     
       do_MIB_NB_IoT(carrier,1,frame,hyper_frame);
     
      }

     if(frame%64==1 && subframe ==0)
      {     
         do_SIB1_NB_IoT_x(0,0,carrier,208,92,1,3584,28,2,hyper_frame);
      }
    */  
  /////////////////////////////////////////////////////////////////////////////////
  //uint8_t      *control_region_size = get_NB_IoT_SIB1_eutracontrolregionsize();
  //int           G=0;
 //NSSS only happened in the even frame
  int nsss_state = 0;

  if(frame%2==0)
  {
      With_NSSS = 1;
  } else {
      With_NSSS = 0;
  }
  /////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// NPSS && NSSS ////////////////////////////////// 
  /////////////////////////////////////////////////////////////////////////////////
  if(subframe == 5)
    {

      generate_npss_NB_IoT(txdataF,
                           AMP,
                           fp,
                           3,
                           10,
                           RB_IoT_ID);
   }
   else if((subframe == 9) && (With_NSSS == 1))
    {
    
      generate_sss_NB_IoT(txdataF,
                          AMP,
                          fp,
                          3,
                          18,
                          frame,
                          RB_IoT_ID);
      nsss_state = 1;
    }
    /////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// MIB //////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////

    if(subframe == 0)
    {
      generate_npbch(broadcast_str,
                     txdataF,
                     AMP,
                     fp,
                     npbch_pdu,
                     frame%64,
                     RB_IoT_ID,
                     release_v13_5_0);
    }
    /////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////// SIB1 ////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    int sib1_state = 0;
    if(subframe == 4)
    {
       sib1_state = generate_SIB1(sib1,
                                  txdataF,
                                  AMP,
                                  fp,
                                  frame,
                                  subframe,
                                  RB_IoT_ID,
                                  0,
                                  release_v13_5_0);
    }
    /////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// SIB23 ////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////
    if( (subframe != 0) && (subframe != 5) && (sib1_state != 1) && (nsss_state != 1))
    {      
          generate_SIB23(sib23,
                         txdataF,
                         AMP,
                         fp,
                         frame,
                         subframe,
                         RB_IoT_ID,
                         release_v13_5_0);
    }
  
    if( (subframe != 0) && (subframe != 5) && (nsss_state != 1) && (fp->flag_free_sf == 0) )
    {
      NB_IoT_eNB_NPDCCH_t  *npdcch_str      = eNB->npdcch_DCI;
      NB_IoT_eNB_NDLSCH_t  *RAR             = eNB->ndlsch_RAR;
      NB_IoT_eNB_NDLSCH_t  *data            = eNB->ndlsch[0];
      /////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////// NPDCCH ////////////////////////////////////
      /////////////////////////////////////////////////////////////////////////////////
      generate_NPDCCH_NB_IoT(npdcch_str,
                             txdataF,
                             AMP,
                             fp,
                             frame,
                             subframe,
                             RB_IoT_ID);
      /////////////////////////////////////////////////////////////////////////////////
      ///////////////////////////////////// NPDSCH ////////////////////////////////////
      /////////////////////////////////////////////////////////////////////////////////
      if(eNB->ndlsch_RAR != NULL && RAR->active_msg2 == 1)
      {
          generate_NDLSCH_NB_IoT(eNB,
                                 RAR,
                                 txdataF,
                                 AMP,
                                 fp,
                                 frame,
                                 subframe,
                                 RB_IoT_ID,
                                 release_v13_5_0);

      } else if(eNB->ndlsch[0] != NULL) {
          generate_NDLSCH_NB_IoT(eNB,
                                 data,
                                 txdataF,
                                 AMP,
                                 fp,
                                 frame,
                                 subframe,
                                 RB_IoT_ID,
                                 release_v13_5_0);
      }
      ///////////////////////////////////////////////////////////////////////////////////
    }

    generate_pilots_NB_IoT(eNB,
                           txdataF,
                           AMP,
                           subframe,
                           RB_IoT_ID,
                           With_NSSS);

  if(proc->frame_rx==1023 && proc->subframe_rx==9)
  {
      //printf("%d",hyper_frame);
      if(proc->HFN==1023)
      {             
           proc->HFN=0;
      }else{ 
           proc->HFN++;
           //printf("Update HFN:%d when frame:%d subframe:%d\n",proc->HFN,proc->frame_rx,proc->subframe_rx);
      }
  }

  
}



void generate_eNB_dlsch_params_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t * proc,nfapi_dl_config_request_pdu_t *dl_config_pdu) 
{
  int                      UE_id         =  -1;
  LTE_DL_FRAME_PARMS    *fp           =  &eNB->frame_parms;
  int                      frame         =  proc->frame_tx;
  int                      subframe      =  proc->subframe_tx;
  DCI_CONTENT              *DCI_Content; 
  DCI_format_NB_IoT_t      DCI_format;
  NB_IoT_eNB_NDLSCH_t      *ndlsch;
  NB_IoT_eNB_NPDCCH_t      *npdcch;

  eNB->DCI_pdu = (DCI_PDU_NB_IoT*)malloc(sizeof(DCI_PDU_NB_IoT));

  DCI_Content = (DCI_CONTENT*) malloc(sizeof(DCI_CONTENT));

  // check DCI format is N1 (format 0)
  if(dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_format == 0)
    {
      //check DCI format N1 is for RAR  rnti_type  in FAPI specs table 4-45
      if(dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti_type == 1)
        {

        //mapping the fapi parameters to the oai parameters

          DCI_format = DCIFormatN1_RAR;

          //DCI format N1 to RAR
          DCI_Content->DCIN1_RAR.type           = 1;
          DCI_Content->DCIN1_RAR.orderIndicator = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.npdcch_order_indication;
          DCI_Content->DCIN1_RAR.Scheddly       = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.scheduling_delay;
          DCI_Content->DCIN1_RAR.ResAssign      = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.resource_assignment;
          DCI_Content->DCIN1_RAR.mcs            = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.mcs;
          DCI_Content->DCIN1_RAR.RepNum         = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.repetition_number;
          DCI_Content->DCIN1_RAR.ndi            = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.new_data_indicator;
          DCI_Content->DCIN1_RAR.HARQackRes     = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.harq_ack_resource;
          DCI_Content->DCIN1_RAR.DCIRep         = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_subframe_repetition_number;


          //TODO calculate the number of common repetitions
          //fp->nprach_config_common.number_repetition_RA = see TS 36.213 Table 16.1-3

          // fill the dlsch_ra_NB structure for RAR, and packed the DCI PDU

          ndlsch               =  eNB->ndlsch_RAR;
          ndlsch->ndlsch_type  =  RAR;

          ndlsch->rnti         =  dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti;

          npdcch               =  eNB->npdcch_DCI;
           
          LOG_D(PHY,"Generating pdcch params for DCIN1 RAR and packing DCI\n");
          //LOG_I(PHY,"Rep of DCI is : %d\n",DCI_Content->DCIN1_RAR.RepNum);

          //LOG_I(PHY,"Generating dlsch params for RA_RNTI and packing DCI\n");
          generate_eNB_dlsch_params_from_dci_NB_IoT(eNB,
                                                    frame,
                                                    subframe,
                                                    DCI_Content,
                                                    dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti,
                                                    DCI_format,
                                                    npdcch,
                                                    fp,
                                                    dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.aggregation_level,
                                                    dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.start_symbol,
                                                    dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.ncce_index);

        //printf("PHY_vars_eNB_NB_IoT_g[0][0]->ndlsch_RAR->rnti = %d\n",PHY_vars_eNB_NB_IoT_g[0][0]->ndlsch_RAR->rnti);
          //eNB->dlsch_ra_NB->nCCE[subframe] = eNB->DCI_pdu->dci_alloc.firstCCE;
        }
      else
        { //managing data
        LOG_I(PHY,"Handling the DCI for ue-spec data or MSG4!\n");
        // Temp: Add UE id when Msg4 trigger
        eNB->ndlsch[0][0]= (NB_IoT_eNB_NDLSCH_t*) malloc(sizeof(NB_IoT_eNB_NDLSCH_t));
        eNB->ndlsch[0][0]->harq_processes[0] = (NB_IoT_DL_eNB_HARQ_t*)malloc(sizeof(NB_IoT_DL_eNB_HARQ_t));
        eNB->ndlsch[0][0]->rnti=dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti; 
        //TODO target/SIMU/USER?init_lte/init_lte_eNB we should allocate the ndlsch structures
        UE_id = find_ue_NB_IoT(dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti, eNB);
        AssertFatal(UE_id != -1, "no ndlsch context available or no ndlsch context corresponding to that rnti\n");


            //mapping the fapi parameters to the oai parameters

            DCI_format = DCIFormatN1;

              //DCI format N1 to DLSCH
              DCI_Content->DCIN1.type           = 1;
              DCI_Content->DCIN1.orderIndicator = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.npdcch_order_indication;
              DCI_Content->DCIN1.Scheddly       = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.scheduling_delay;
              DCI_Content->DCIN1.ResAssign      = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.resource_assignment;
              DCI_Content->DCIN1.mcs            = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.mcs;
              DCI_Content->DCIN1.RepNum         = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.repetition_number;
              DCI_Content->DCIN1.ndi            = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.new_data_indicator;
              DCI_Content->DCIN1.HARQackRes     = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.harq_ack_resource;
              DCI_Content->DCIN1.DCIRep         = dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_subframe_repetition_number;
              npdcch               =  eNB->npdcch_DCI;

              //eNB->npdcch[(uint8_t)UE_id] = (NB_IoT_eNB_NPDCCH_t *) malloc(sizeof(NB_IoT_eNB_NPDCCH_t));
              //set the NPDCCH UE-specific structure  (calculate R)
              //npdcch=eNB->npdcch[(uint8_t)UE_id];
              //AssertFatal(npdcch != NULL, "NPDCCH structure for UE specific is not exist\n");
              //npdcch->repetition_idx[(uint8_t)UE_id] = 0; //this is used for the encoding mechanism to understand that is the first transmission

              //if(dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.aggregation_level) //whenever aggregation level is =1 we have only 1 repetition for USS
             // npdcch->repetition_number[(uint8_t)UE_id] = 1;
              //else
              //{
                //see TS 36.213 Table 16.1-1
              //}


              //fill the ndlsch structure for UE and packed the DCI PD

            ndlsch = eNB->ndlsch[(uint8_t)UE_id]; //in the old implementation they also consider UE_id = 1;
            ndlsch->ndlsch_type = UE_Data;

              //parameters we don't consider pdsch config dedicated since not calling the phy config dedicated step2

            LOG_I(PHY,"Generating dlsch params for DCIN1 data and packing DCI, res: %d\n",DCI_Content->DCIN1.ResAssign);
            generate_eNB_dlsch_params_from_dci_NB_IoT(eNB,
                                                      frame,
                                                      subframe,
                                                      DCI_Content,
                                                      dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti,
                                                      DCI_format,
                                                      npdcch,
                                                      fp,
                                                      dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.aggregation_level,
                                                      dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.start_symbol,
                                                      dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.ncce_index);

              //eNB->ndlsch[(uint8_t)UE_id]->nCCE[subframe] = eNB->DCI_pdu->dci_alloc[i].firstCCE;


        }
    }
  else if(dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.dci_format == 1)
    { 
      DCI_format = DCIFormatN2;
      LOG_D(PHY,"Paging procedure not implemented\n");
    }
  else
    LOG_E(PHY,"unknown DCI format for NB-IoT DL\n");


}



void generate_eNB_ulsch_params_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,nfapi_hi_dci0_request_pdu_t *hi_dci0_pdu) {

  //int UE_id = -1;
  //int harq_pid = 0;
  int                      frame         =  proc->frame_tx;
  int                      subframe      =  proc->subframe_tx;
  DCI_CONTENT *DCI_Content;
  DCI_Content = (DCI_CONTENT*) malloc(sizeof(DCI_CONTENT));
  NB_IoT_eNB_NPDCCH_t      *npdcch;

  //mapping the fapi parameters to the OAI parameters
  DCI_Content->DCIN0.type       = 0;
  DCI_Content->DCIN0.scind      = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.subcarrier_indication;
  DCI_Content->DCIN0.ResAssign  = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.resource_assignment;
  DCI_Content->DCIN0.mcs        = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.mcs;
  DCI_Content->DCIN0.ndi        = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.new_data_indicator;
  DCI_Content->DCIN0.Scheddly   = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.scheduling_delay;
  DCI_Content->DCIN0.RepNum     = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.repetition_number;
  DCI_Content->DCIN0.rv         = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.redudancy_version;
  DCI_Content->DCIN0.DCIRep     = hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.dci_subframe_repetition_number;

  npdcch               =  eNB->npdcch_DCI;

  /*Log for generate ULSCH DCI*/
  LOG_I(PHY,"packing DCI N0\n");

  LOG_I(PHY,"Dump DCI N0 : scind: %d, ResAssign: %d, mcs: %d, ndi: %d, Scheddly: %d, RepNum: %d, rv: %d, DCIRep: %d\n",DCI_Content->DCIN0.scind,DCI_Content->DCIN0.ResAssign,DCI_Content->DCIN0.mcs,DCI_Content->DCIN0.ndi,DCI_Content->DCIN0.Scheddly,DCI_Content->DCIN0.RepNum,DCI_Content->DCIN0.rv,DCI_Content->DCIN0.DCIRep);

  generate_eNB_ulsch_params_from_dci_NB_IoT(eNB,
                                            frame,
                                            subframe,
                                            DCI_Content,
                                            hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.rnti,
                                            npdcch,
                                            hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.aggregation_level,
                                            hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.start_symbol,
                                            hi_dci0_pdu->npdcch_dci_pdu.npdcch_dci_pdu_rel13.ncce_index
                                            );  

  
}



/*
 * for NB-IoT ndlsch procedure
 * this function is called by the PHy procedure TX in 3 possible occasion:
 * 1) we manage BCCH pdu (SI)
 * 2) we manage RA dlsch pdu
 * 3) UE-specific dlsch pdu
 * ** we need to know if exist and which value has the eutracontrolRegionSize (TS 36.213 ch 16.4.1.4) whenever we are in In-band mode
 * ** CQI and PMI are not present in NB-IoT
 * ** redundancy version exist only in UL for NB-IoT and not in DL
 */
void npdsch_procedures(PHY_VARS_eNB_NB_IoT      *eNB,
                       eNB_rxtx_proc_NB_IoT_t   *proc,     //Context data structure for RX/TX portion of subframe processing
                       NB_IoT_eNB_NDLSCH_t      *ndlsch,
                       //int num_npdcch_symbols,            //(BCOM says are not needed
                       uint8_t                  *pdu
                       )
{
  int                     frame                   =   proc->frame_tx;
  int                     subframe                =   proc->subframe_tx;
  NB_IoT_DL_eNB_HARQ_t    *ndlsch_harq            =   ndlsch->harq_processes;
  int                     input_buffer_length     =   ndlsch_harq->TBS/8;         // get in byte //the TBS is set in generate_dlsch_param
  NB_IoT_DL_FRAME_PARMS   *fp                     =   &eNB->frame_parms_NB_IoT;
  int                     G;
  uint8_t                 *DLSCH_pdu              =   NULL;
  uint8_t                 DLSCH_pdu_tmp[input_buffer_length+4];                   //[768*8];
  //uint8_t DLSCH_pdu_rar[256];
  int                     i;

  LOG_D(PHY,
        "[eNB %"PRIu8"][PDSCH rnti%"PRIx16"] Frame %d, subframe %d: Generating PDSCH/DLSCH with input size = %"PRIu16", mcs %"PRIu8"(round %"PRIu8")\n",
        eNB->Mod_id,
        ndlsch->rnti,
        frame, subframe, input_buffer_length,
        ndlsch_harq->mcs,
        ndlsch_harq->round
        );

  if(ndlsch_harq->round == 0) { //first transmission so we encode... because we generate the sequence

    if (eNB->mac_enabled == 1) { // set in lte-softmodem/main line 1646

        DLSCH_pdu = pdu;

      /*
       * we don't need to manage the RAR here since should be managed in the MAC layer for two reasons:
       * 1)we should receive directly the pdu containing the RAR from the MAC in the schedule_response
       * 2)all the parameters for getting the MSG3 should be given by the UL_CONFIG.request (all inside the next schedule_response function)
       *
       */

        //fill_rar shouduld be in the MAC
        //cancel ra procedure should be in the mac
        //scheduling request not implemented in NB-IoT
        //nulsch_param configuration for MSG3 should be considered in handling UL_Config.request
        //(in particular the nulsch structure for RAR is distinguished based on the harq_process->rar_alloc and the particular subframe in which we should have Msg3)
    }

    else {  //XXX we should change taus function???

      DLSCH_pdu = DLSCH_pdu_tmp;

      for (i=0; i<input_buffer_length; i++)

         DLSCH_pdu[i] = (unsigned char)(taus()&0xff);
    }
  }
  else {
    //We are doing a retransmission (harq round > 0
    #ifdef DEBUG_PHY_PROC
    #ifdef DEBUG_DLSCH
    LOG_D(PHY,"[eNB] This DLSCH is a retransmission\n");
    #endif
    #endif
  }

  if (eNB->abstraction_flag==0) { // used for simulation of the PHY??


    //we can distinguish among the different kind of NDLSCH structure (example)
    switch(ndlsch->ndlsch_type)
    {
    case SIB1:
      break;
    case SI_Message:
      break;
    case RAR: //maybe not needed
      break;
    case UE_Data: //maybe not needed
      break;
    }


   /*
    * in any case inside the encoding procedure is re-checked if this is round 0 or no
    * in the case of harq_process round = 0 --> generate the sequence and put it into the parameter *c[r]
    * otherwise do nothing(only rate maching)
    */


  /*
   * REASONING:
   * Encoding procedure will generate a Table with encoded data ( in ndlsch structure)
   * The table will go in input to the scrambling
   * --we should take care if there are repetitions of data or not because scrambling should be called at the first frame and subframe in which each repetition
   * begin (see params Nf, Ns)
   */


    // 36-212
    //encoding---------------------------

    /*
     *
     * REASONING:
   * Encoding procedure will generate a Table with encoded data ( in ndlsch structure)
   * The table will go in input to the scrambling
   * --we should take care if there are repetitions of data or not because scrambling should be called at the first frame and subframe in which each repetition
   *  begin (see params Nf, Ns)
     *
     * we should have as an iput parameter also G for the encoding based on the switch/case over eutracontrolRegionSize (if exist) and operationModeInfo if defined
     * NB: switch case of G is the same for npdsch and npdcch
     *
     * npdsch_start symbol index
     * -refers to TS 36.213 ch 16.4.1.4:
     * -if subframe k is a subframe for receiving the SIB1-NB
     *  -- if operationModeInfo set to 00 or 01 (in band) --> npdsch_start_sysmbol = 3
     *  -- otherwise --> npdsch_start_symbol = 0
     * -if the k subframe is not for SIB1-NB
     *  --npdsch_start_symbol = eutracontrolregionsize (defined for in-band operating mode (mode 0,1 for FAPI specs) and take values 1,2,3 [units in number of OFDM symbol])
     * - otherwise --> npdsch_start_symbol = 0
     * (is the starting OFDM for the NPDSCH transmission in the first slot in a subframe k)
     * FAPI style:
     * npdsch_start symbol is stored in the ndlsch structure from the reception of the NPDLSCH PDU in the DL_CONFIG.request (so should be set by the MAC and put inside the schedule response)
     * Nsf needed as an input (number of subframe)-->inside harq_process of ndlsch
     */

    switch(ndlsch->npdsch_start_symbol)
    {
      case 0:
        G = 304;
      break;
      case 1:
        G = 240;
      break;
      case 2:
        G = 224;
      break;
      case 3:
        G =200;
      break;
      default:
        LOG_E (PHY,"npdsch_start_index has unwanted value\n");
      break;

    }
    //start_meas_NB_IoT(&eNB->dlsch_encoding_stats);
    LOG_I(PHY, "NB-IoT Encoding step\n");

    //    eNB->te(eNB,
    //      DLSCH_pdu,
    //      num_npdcch_symbols,
    //      dlsch,
    //      frame,subframe,
    //      &eNB->dlsch_rate_matching_stats,
    //      &eNB->dlsch_turbo_encoding_stats,
    //      &eNB->dlsch_interleaving_stats);


   // stop_meas_NB_IoT(&eNB->dlsch_encoding_stats);

    // 36-211
    //scrambling-------------------------------------------

   // start_meas_NB_IoT(&eNB->dlsch_scrambling_stats);
    LOG_I(PHY, "NB-IoT Scrambling step\n");

    /*
     * SOME RELEVANT FACTS:
     *
     *
     */

      //    dlsch_scrambling(fp,
      //         0,
      //         dlsch,
      //         get_G(fp,
      //         dlsch_harq->nb_rb,
      //         dlsch_harq->rb_alloc,
      //         get_Qm(dlsch_harq->mcs),
      //         dlsch_harq->Nl,
      //         num_npdcch_symbols,
      //         frame,subframe,
      //         0),
      //         0,
      //         subframe<<1);

    //stop_meas_NB_IoT(&eNB->dlsch_scrambling_stats);


    //modulation-------------------------------------------
    //start_meas_NB_IoT(&eNB->dlsch_modulation_stats);
    LOG_I(PHY, "NB-IoT Modulation step\n");

    //    dlsch_modulation(eNB,
    //         eNB->common_vars.txdataF[0],
    //         AMP,
    //         subframe,
    //         num_npdcch_symbols,
    //         dlsch,
    //         dlsch1);

    //stop_meas_NB_IoT(&eNB->dlsch_modulation_stats);
  }


#ifdef PHY_ABSTRACTION
  else {
    //start_meas_NB_IoT(&eNB->dlsch_encoding_stats);
    //dlsch_encoding_emul(eNB,
      //DLSCH_pdu,
      //dlsch);
   // stop_meas_NB_IoT(&eNB->dlsch_encoding_stats);
  }

#endif
  ndlsch->active = 0;
}


/*
 * ASSUMPTION
 *
 * The MAC schedule the schedule_response in a SUBFRAME BASE (at least because otherwise we have problem with our assumptions on SI transmission)
 *
 *Since in FAPI specs seems to not manage the information for the sceduling of system information:
 * Assume that the MAC layer manage the scheduling for the System information (SI messages) transmission while MIB and SIB1 are done directly at PHY layer
 * This means that the MAC scheduler will send to the PHY the NDLSCH PDU and MIB PDU (DL_CONFIG.request)each time they should be transmitted. In particular:
 ***MIB-NB
 *schedule_response containing a n-BCH PDU is transmitted only at the beginning of the MIB period, then repetitions are made directly by the PHY layer (see FAPI specs pag 94 N-BCH 3.2.4.2)
 *if no new N-BCH PDU is trasmitted at SFN mod 64=0 then stop MIB transmission
 ***SIB1-NB
 *schedule response containing a NDLSCH pdu (with appropiate configuration) will be transmitted only at the beginning of each SIB1-NB period (256 rf)
 *then repetitions are managed directly by the PHY layer
 *if no new NDLSCH pdu (configured for SIB1-NB) at SFN mod 256 = 0 is transmitted. stop SIB1-NB transmission
 ****SI Messages
 * -schedule_response is transmitted by the MAC in every subframe needed for the SI transmission (NDLSCH should have a proper configuration)
 * -if the schedule_response carry any SDU for SI-Message (SDU!= NULL)--> put the SDU in the PHY buffer to be encoded ecc... and start the transmission
 * -if the schedule_response not carry any SDU (SDU == NULL) but NDLSCH is properly set for SI, then PHY continue transmit the remaining part of the previous SDU
 * (this because the PHY layer have no logic of repetition_pattern, si_window ecc.. so should be continuously instructed the PHY when to transmit.
 *
 * Furthermore, SI messages are transmitted in more that 1 subframe (2 or 8) and therefore MAC layer need to count how many subframes are available in the current frame for transmit it
 * and take in consideration that other frames are needed before starting the transmission of a new one)
 *
 *
 *We assume that whenever the NDLSCH pdu is a BCCH type, we consider as if it's a SIB1 while in other case can be data or SI-message depending on the RNTI
 *
 * **relevant aspects for the System information Transmission (Table 4-47 NDLSCH FAPi specs)
 * 1)RNTI type = 0 (contains a BCCH)
 * 2)Repetition number == scheduling info SIB1 mapped into 4-8-16
 * 3)RNTI (0xFFFF = SI-RNTI)
 * (see schedule_response implementation)
 *
 */





uint32_t rx_nprach_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB, int frame, uint8_t subframe, uint16_t *rnti, uint16_t *preamble_index, uint16_t *timing_advance) {

  uint32_t estimated_TA; 
  //int frame,frame_mod;    // subframe,
 // subframe = eNB->proc.subframe_prach; 
 // frame = eNB->proc.frame_prach;
    estimated_TA = process_nprach_NB_IoT(eNB,frame,subframe,rnti,preamble_index,timing_advance);
    //printf("estim = %i\n",estimated_TA);
 // }
  return estimated_TA;
}


void fill_crc_indication_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,int UE_id,int frame,int subframe,uint8_t decode_flag) {


  pthread_mutex_lock(&eNB->UL_INFO_mutex);
  
 
          nfapi_crc_indication_pdu_t *pdu            =  &eNB->UL_INFO.crc_ind.crc_pdu_list[0]; //[eNB->UL_INFO.crc_ind.crc_indication_body.number_of_crcs];
          pdu->rx_ue_information.rnti                =  eNB->ulsch_NB_IoT[0]->rnti;              /// OK
          pdu->crc_indication_rel8.crc_flag          =  decode_flag;

          if(decode_flag == 1)
          {
              eNB->UL_INFO.crc_ind.number_of_crcs++;
          } else {
             eNB->UL_INFO.crc_ind.number_of_crcs =0;
          }

    // nfapi_crc_indication_pdu_t* crc_pdu_list
  ///eNB->UL_INFO.crc_ind.sfn_sf                         = frame<<4 | subframe;
  //eNB->UL_INFO.crc_ind.header.message_id              = NFAPI_CRC_INDICATION;
  //eNB->UL_INFO.crc_ind.crc_indication_body.tl.tag     = NFAPI_CRC_INDICATION_BODY_TAG;

  //pdu->instance_length                                = 0; // don't know what to do with this
  //  pdu->rx_ue_information.handle                       = handle;
  ///////////////////////pdu->rx_ue_information.tl.tag                       = NFAPI_RX_UE_INFORMATION_TAG;
  //////////////////////////pdu->crc_indication_rel8.tl.tag                     = NFAPI_CRC_INDICATION_REL8_TAG;
  
  
  //LOG_D(PHY, "%s() rnti:%04x crcs:%d crc_flag:%d\n", __FUNCTION__, pdu->rx_ue_information.rnti, eNB->UL_INFO.crc_ind.crc_indication_body.number_of_crcs, crc_flag);

  pthread_mutex_unlock(&eNB->UL_INFO_mutex);
}

void fill_rx_indication_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,uint8_t data_or_control, uint8_t decode_flag)
{
      nfapi_rx_indication_pdu_t *pdu;
      nfapi_nb_harq_indication_pdu_t *ack_ind; // &eNB->UL_INFO.nb_harq_ind.nb_harq_indication_body.nb_harq_pdu_list[0] // nb_harq_indication_fdd_rel13->harq_tb1

      pthread_mutex_lock(&eNB->UL_INFO_mutex);

      if (data_or_control == 0)    // format 1
      {

            if(decode_flag == 1)
            { 
                eNB->UL_INFO.RX_NPUSCH.number_of_pdus  = 1;
            } else {
                eNB->UL_INFO.RX_NPUSCH.number_of_pdus  = 0;
            }
           
            pdu                                    = &eNB->UL_INFO.RX_NPUSCH.rx_pdu_list[0];
            pdu->rx_ue_information.rnti            = eNB->ulsch_NB_IoT[0]->rnti;
            pdu->rx_indication_rel8.length         = eNB->ulsch_NB_IoT[0]->harq_processes[0]->TBS; //eNB->ulsch_NB_IoT[0]->harq_process->TBS>>3;
            pdu->data                              = eNB->ulsch_NB_IoT[0]->harq_processes[0]->b;

      } else {             // format 2

           if(decode_flag == 1)
           { 
                  eNB->UL_INFO.nb_harq_ind.nb_harq_indication_body.number_of_harqs =1;
                  ack_ind                                             =  &eNB->UL_INFO.nb_harq_ind.nb_harq_indication_body.nb_harq_pdu_list[0];
                  ack_ind->nb_harq_indication_fdd_rel13.harq_tb1      =  1;
                  ack_ind->rx_ue_information.rnti                     =  eNB->ulsch_NB_IoT[0]->rnti;

            } else {
                  eNB->UL_INFO.nb_harq_ind.nb_harq_indication_body.number_of_harqs =1;
                  ack_ind                                             =  &eNB->UL_INFO.nb_harq_ind.nb_harq_indication_body.nb_harq_pdu_list[0];
                  ack_ind->nb_harq_indication_fdd_rel13.harq_tb1      =  2;
                  ack_ind->rx_ue_information.rnti                     =  eNB->ulsch_NB_IoT[0]->rnti;
            }
      }
      
       //eNB->UL_INFO.RX_NPUSCH.rx_pdu_list.rx_ue_information.tl.tag = NFAPI_RX_INDICATION_BODY_TAG;   // do we need this ?? 
      //eNB->UL_INFO.RX_NPUSCH.rx_pdu_list.rx_ue_information.rnti = rnti;  // rnti should be got from eNB structure
      //pdu                                    = &eNB->UL_INFO.RX_NPUSCH.rx_pdu_list[eNB->UL_INFO.rx_ind.rx_indication_body.number_of_pdus];
      //  pdu->rx_ue_information.handle          = eNB->ulsch[UE_id]->handle;
      // pdu->rx_ue_information.tl.tag          = NFAPI_RX_UE_INFORMATION_TAG;
      //pdu->rx_indication_rel8.tl.tag         = NFAPI_RX_INDICATION_REL8_TAG;
      
     

      /*if(msg3_flag == 1)
      {
          pdu->rx_indication_rel8.length         = 6; //eNB->ulsch_NB_IoT[0]->harq_process->TBS>>3;
          int m =0;
          for(m=0; m<6;m++)
          { 
              pdu->data[m]  = eNB->ulsch_NB_IoT[0]->harq_process->b[2+m];
              printf(" pdu content = %d \n", eNB->ulsch_NB_IoT[0]->harq_process->b[2+m]);
          }        
          
      } else { */  
         
      //}
      //pdu->data                              = eNB->ulsch_NB_IoT[UE_id]->harq_processes[harq_pid]->b;   
      //eNB->UL_INFO.rx_ind.rx_indication_body.number_of_pdus++;
      //eNB->UL_INFO.rx_ind.sfn_sf = frame<<4 | subframe;

      // do we need to transmit timing ?? however, the nfapi structure does not include timing paramters !!!!!

      pthread_mutex_unlock(&eNB->UL_INFO_mutex);

}



void npusch_procedures(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc)
{
  
  uint32_t i;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  NB_IoT_eNB_NULSCH_t *nulsch;
  NB_IoT_UL_eNB_HARQ_t *nulsch_harq;
  nulsch = eNB->ulsch_NB_IoT[0];
  nulsch_harq = nulsch->harq_processes;

  const int rx_subframe   =   proc->subframe_rx;
  const int rx_frame      =   proc->frame_rx;

  int   RB_IoT_ID         = 22;
  //for (i=0; i<NUMBER_OF_UE_MAX; i++)
  for (i=0; i<1; i++)
  {
      //ulsch_NB_IoT = eNB->ulsch_NB_IoT[i];
      //ulsch_harq = ulsch_NB_IoT->harq_process;
      // if eNB is ready to receive UL data 
      // define a flag to trigger on or off the decoding process
     rx_ulsch_Gen_NB_IoT(eNB,
                           proc,
                           0,                         // this is the effective sector id
                           0,
                           RB_IoT_ID,                        // 22 , to be included in // to be replaced by NB_IoT_start ??
                           rx_subframe,  // first received subframe 
                           rx_frame);     // first received frame
   }  // for UE loop

}

/*-----------------------------------------------------------------------------------------------------------------------------------------------*/

extern uint16_t hundred_times_log10_NPRB[100];
int harq_pid_updated[NUMBER_OF_UE_MAX][8] = {{0}};
int harq_pid_round[NUMBER_OF_UE_MAX][8] = {{0}};

#ifdef EMOS
fifo_dump_emos_eNB emos_dump_eNB;
#endif


uint8_t is_SR_subframe(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,uint8_t UE_id)
{

  const int subframe = proc->subframe_rx;
  const int frame = proc->frame_rx;

  LOG_D(PHY,"[eNB %d][SR %x] Frame %d subframe %d Checking for SR TXOp(sr_ConfigIndex %d)\n",
        eNB->Mod_id,eNB->ulsch[UE_id]->rnti,frame,subframe,
        eNB->scheduling_request_config[UE_id].sr_ConfigIndex);

  if (eNB->scheduling_request_config[UE_id].sr_ConfigIndex <= 4) {        // 5 ms SR period
    if ((subframe%5) == eNB->scheduling_request_config[UE_id].sr_ConfigIndex)
      return(1);
  } else if (eNB->scheduling_request_config[UE_id].sr_ConfigIndex <= 14) { // 10 ms SR period
    if (subframe==(eNB->scheduling_request_config[UE_id].sr_ConfigIndex-5))
      return(1);
  } else if (eNB->scheduling_request_config[UE_id].sr_ConfigIndex <= 34) { // 20 ms SR period
    if ((10*(frame&1)+subframe) == (eNB->scheduling_request_config[UE_id].sr_ConfigIndex-15))
      return(1);
  } else if (eNB->scheduling_request_config[UE_id].sr_ConfigIndex <= 74) { // 40 ms SR period
    if ((10*(frame&3)+subframe) == (eNB->scheduling_request_config[UE_id].sr_ConfigIndex-35))
      return(1);
  } else if (eNB->scheduling_request_config[UE_id].sr_ConfigIndex <= 154) { // 80 ms SR period
    if ((10*(frame&7)+subframe) == (eNB->scheduling_request_config[UE_id].sr_ConfigIndex-75))
      return(1);
  }

  return(0);
}


int mac_phy_remove_ue(module_id_t Mod_idP,rnti_t rntiP) {
  uint8_t i;
  int CC_id;
  PHY_VARS_eNB_NB_IoT *eNB;

  for (CC_id=0;CC_id<MAX_NUM_CCs;CC_id++) {
    eNB = PHY_vars_eNB_NB_IoT_g[Mod_idP][CC_id];
    for (i=0; i<NUMBER_OF_UE_MAX; i++) {
      if ((eNB->ndlsch[i]==NULL) || (eNB->ulsch[i]==NULL)) {
  MSC_LOG_EVENT(MSC_PHY_ENB, "0 Failed remove ue %"PRIx16" (ENOMEM)", rntiP);
  LOG_E(PHY,"Can't remove UE, not enough memory allocated\n");
  return(-1);
      } else {
  if (eNB->UE_stats[i].crnti==rntiP) {
    MSC_LOG_EVENT(MSC_PHY_ENB, "0 Removed ue %"PRIx16" ", rntiP);

    LOG_D(PHY,"eNB %d removing UE %d with rnti %x\n",eNB->Mod_id,i,rntiP);

    //LOG_D(PHY,("[PHY] UE_id %d\n",i);
    clean_eNb_dlsch(eNB->ndlsch[i][0]);
    clean_eNb_ulsch(eNB->ulsch[i]);
    //eNB->UE_stats[i].crnti = 0;
    memset(&eNB->UE_stats[i],0,sizeof(LTE_eNB_UE_stats));
    //  mac_exit_wrapper("Removing UE");
    

    return(i);
  }
      }
    }
  }
  MSC_LOG_EVENT(MSC_PHY_ENB, "0 Failed remove ue %"PRIx16" (not found)", rntiP);
  return(-1);
}

int8_t find_next_ue_index(PHY_VARS_eNB_NB_IoT *eNB)
{
  uint8_t i;

  for (i=0; i<NUMBER_OF_UE_MAX; i++) {
    if (eNB->UE_stats[i].crnti==0) {
      /*if ((eNB->dlsch[i]) &&
  (eNB->dlsch[i][0]) &&
  (eNB->dlsch[i][0]->rnti==0))*/
      LOG_D(PHY,"Next free UE id is %d\n",i);
      return(i);
    }
  }

  return(-1);
}

int get_ue_active_harq_pid(const uint8_t Mod_id,const uint8_t CC_id,const uint16_t rnti, const int frame, const uint8_t subframe,uint8_t *harq_pid,uint8_t *round,const uint8_t harq_flag)
{
  LTE_eNB_DLSCH_t *DLSCH_ptr;
  LTE_eNB_ULSCH_t *ULSCH_ptr;
  uint8_t ulsch_subframe,ulsch_frame;
  int i;
  int8_t UE_id = find_ue(rnti,PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]);

  if (UE_id==-1) {
    LOG_D(PHY,"Cannot find UE with rnti %x (Mod_id %d, CC_id %d)\n",rnti, Mod_id, CC_id);
    *round=0;
    return(-1);
  }

  if ((harq_flag == openair_harq_DL) || (harq_flag == openair_harq_RA))  {// this is a DL request

    DLSCH_ptr = PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->ndlsch[(uint32_t)UE_id][0];

    if (harq_flag == openair_harq_RA) {
      if (DLSCH_ptr->harq_processes[0] != NULL) {
  *harq_pid = 0;
  *round = DLSCH_ptr->harq_processes[0]->round;
  return 0;
      } else {
  return -1;
      }
    }

    /* let's go synchronous for the moment - maybe we can change at some point */
    i = (frame * 10 + subframe) % 8;

    if (DLSCH_ptr->harq_processes[i]->status == ACTIVE) {
      *harq_pid = i;
      *round = DLSCH_ptr->harq_processes[i]->round;
    } else if (DLSCH_ptr->harq_processes[i]->status == SCH_IDLE) {
      *harq_pid = i;
      *round = 0;
    } else {
      printf("%s:%d: bad state for harq process - PLEASE REPORT!!\n", __FILE__, __LINE__);
      abort();
    }
  } else { // This is a UL request

    ULSCH_ptr = PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->ulsch[(uint32_t)UE_id];
    ulsch_subframe = pdcch_alloc2ul_subframe(&PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms,subframe);
    ulsch_frame    = pdcch_alloc2ul_frame(&PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms,frame,subframe);
    // Note this is for TDD configuration 3,4,5 only
    *harq_pid = subframe2harq_pid(&PHY_vars_eNB_NB_IoT_g[Mod_id][CC_id]->frame_parms,
                                  ulsch_frame,
                                  ulsch_subframe);
    *round    = ULSCH_ptr->harq_processes[*harq_pid]->round;
    LOG_T(PHY,"[eNB %d][PUSCH %d] Frame %d subframe %d Checking HARQ, round %d\n",Mod_id,*harq_pid,frame,subframe,*round);
  }

  return(0);
}



#ifdef EMOS
void phy_procedures_emos_eNB_RX(unsigned char subframe,PHY_VARS_eNB_NB_IoT *eNB)
{

  uint8_t aa;
  uint16_t last_subframe_emos;
  uint16_t pilot_pos1 = 3 - eNB->frame_parms.Ncp, pilot_pos2 = 10 - 2*eNB->frame_parms.Ncp;
  uint32_t bytes;

  last_subframe_emos=0;




#ifdef EMOS_CHANNEL

  //if (last_slot%2==1) // this is for all UL subframes
  if (subframe==3)
    for (aa=0; aa<eNB->frame_parms.nb_antennas_rx; aa++) {
      memcpy(&emos_dump_eNB.channel[aa][last_subframe_emos*2*eNB->frame_parms.N_RB_UL*12],
             &eNB->pusch_vars[0]->drs_ch_estimates[0][aa][eNB->frame_parms.N_RB_UL*12*pilot_pos1],
             eNB->frame_parms.N_RB_UL*12*sizeof(int));
      memcpy(&emos_dump_eNB.channel[aa][(last_subframe_emos*2+1)*eNB->frame_parms.N_RB_UL*12],
             &eNB->pusch_vars[0]->drs_ch_estimates[0][aa][eNB->frame_parms.N_RB_UL*12*pilot_pos2],
             eNB->frame_parms.N_RB_UL*12*sizeof(int));
    }

#endif

  if (subframe==4) {
    emos_dump_eNB.timestamp = rt_get_time_ns();
    emos_dump_eNB.frame_tx = eNB->proc[subframe].frame_rx;
    emos_dump_eNB.rx_total_gain_dB = eNB->rx_total_gain_dB;
    emos_dump_eNB.mimo_mode = eNB->transmission_mode[0];
    memcpy(&emos_dump_eNB.measurements,
           &eNB->measurements[0],
           sizeof(PHY_MEASUREMENTS_eNB));
    memcpy(&emos_dump_eNB.UE_stats[0],&eNB->UE_stats[0],NUMBER_OF_UE_MAX*sizeof(LTE_eNB_UE_stats));

    bytes = rtf_put(CHANSOUNDER_FIFO_MINOR, &emos_dump_eNB, sizeof(fifo_dump_emos_eNB));

    //bytes = rtf_put(CHANSOUNDER_FIFO_MINOR, "test", sizeof("test"));
    if (bytes!=sizeof(fifo_dump_emos_eNB)) {
      LOG_W(PHY,"[eNB %d] Frame %d, subframe %d, Problem writing EMOS data to FIFO (bytes=%d, size=%d)\n",
            eNB->Mod_id,eNB->proc[(subframe+1)%10].frame_rx, subframe,bytes,sizeof(fifo_dump_emos_eNB));
    } else {
      if (eNB->proc[(subframe+1)%10].frame_tx%100==0) {
        LOG_I(PHY,"[eNB %d] Frame %d (%d), subframe %d, Writing %d bytes EMOS data to FIFO\n",
              eNB->Mod_id,eNB->proc[(subframe+1)%10].frame_rx, ((fifo_dump_emos_eNB*)&emos_dump_eNB)->frame_tx, subframe, bytes);
      }
    }
  }
}
#endif


void common_signal_procedures_nbiot (PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc) {

  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  int **txdataF = eNB->common_vars.txdataF[0];
  //////////////////////////////////////////////////////// to uncomment for LTE,      uint8_t *pbch_pdu=&eNB->pbch_pdu[0];
  int subframe = proc->subframe_tx;
  int frame = proc->frame_tx;
  RA_TEMPLATE_NB_IoT *RA_template = (RA_TEMPLATE_NB_IoT *)&eNB_mac_inst[eNB->Mod_id].common_channels[eNB->CC_id].RA_template[0];
  //int                     With_NSSS=0;
  int framerx = proc->frame_rx; 
  int subframerx = proc->subframe_rx;
  /////////////////////////////////////////////////ACK ///////////////////////////////////////
  
  NB_IoT_eNB_NULSCH_t    **ulsch_NB_IoT   =  &eNB->ulsch_NB_IoT[0];//[0][0];

  ////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////// Decoding ACK ////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////
  if(subframe==proc->subframe_msg5 && frame==proc->frame_msg5 && proc->flag_msg5==1 &&  proc->counter_msg5>0)
  {
          
          printf("\n\n msg5 received in frame %d subframe %d \n\n",framerx,subframerx);

          if (proc->counter_msg5 ==2)
          {
            proc->frame_dscr_msg5 = framerx; 
            proc->subframe_dscr_msg5 = subframerx;
          }


          proc->subframe_msg5++; 
          proc->counter_msg5--;
      

  }
  ////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////  RX NPUSH  //////////////////////////////////////

  if(subframe==proc->subframe_real && proc->flag_msg3==1 && frame==proc->frame_msg3 &&  proc->counter_msg3>0) ///&& frame == ????
  {
       proc->subframe_real++;
      if(proc->subframe_real==10) ///&& frame == ????
      {
          proc->subframe_real=0;
          proc->frame_msg3++;
      }
      if (proc->counter_msg3 ==8)
      {
        proc->frame_dscr_msg3 = framerx; 
        proc->subframe_dscr_msg3 = subframerx;
      }

    proc->counter_msg3--;

  }



//////////////////////////////////////////////////////////////////////////////////////////////////
if(proc->flag_msg4 == 1 && proc->counter_msg4 > 0)
{

        if(frame == proc->frame_msg4 && subframe == proc->subframe_msg4)
        {
                 NB_IoT_eNB_NDLSCH_t  *rar  =  eNB->ndlsch_RAR;
                //uint8_t   tab_rar[15];
                //uint8_t   tab_rar[18];
                uint8_t   tab_rar[7];
                uint8_t *nas_id = &eNB->msg3_pdu[0];
                //uint8_t   *NAS_tab = &eNB->tab_nas;
                // avoid subframe 9 and subframe 0 of next frame
                
                tab_rar[0]=28;
                tab_rar[1]=nas_id[0]; // NAS part 1
                tab_rar[2]=nas_id[1];  // NAS part 2 
                tab_rar[3]=nas_id[2];  // NAS part 3
                tab_rar[4]=nas_id[3]; // NAS part 4 
                tab_rar[5]=nas_id[4];  // NAS part 5
                tab_rar[6]=nas_id[5]; // NAS part 5   
                 
               
                proc->flag_scrambling =1;
                printf("\n RAR sentttttt frame %d, subframe %d", frame, subframe);
          

                 proc->counter_msg4--;
                 proc->subframe_msg4 =subframe+1;     
        }         
}

///////////////////////////////////////////////////////////////////////////////
  common_signal_procedures_NB_IoT(eNB,proc);
/////////////////////////// END ///////////////////////////////////////  

}



void phy_procedures_eNB_TX_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
         eNB_rxtx_proc_NB_IoT_t *proc,
                           relaying_type_t r_type,
         PHY_VARS_RN_NB_IoT *rn,
         int do_meas,
         int do_pdcch_flag)
{
  UNUSED(rn);
  int frame=proc->frame_tx;
  int subframe=proc->subframe_tx;
  //  uint16_t input_buffer_length;
  uint32_t i,aa;  //j;
  uint8_t harq_pid;
  DCI_PDU_NB_IoT *DCI_pdu;
  DCI_PDU_NB_IoT DCI_pdu_tmp;
  int8_t UE_id=0;
 // uint8_t num_npdcch_symbols=0;
  uint8_t ul_subframe;
  uint32_t ul_frame;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
 // DCI_ALLOC_t *dci_alloc=(DCI_ALLOC_t *)NULL;

  int offset = eNB->CC_id;//proc == &eNB->proc.proc_rxtx[0] ? 0 : 1;

#if defined(SMBV) 
  // counts number of allocations in subframe
  // there is at least one allocation for PDCCH
  uint8_t smbv_alloc_cnt = 1;Exiting eNB thread RXn_TXnp4

#endif

  if ((fp->frame_type == TDD) && (subframe_select(fp,subframe)==SF_UL)) return;

 // VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_TX+offset,1);
 // if (do_meas==1) start_meas(&eNB->phy_proc_tx);

  T(T_ENB_PHY_DL_TICK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe));

  for (i=0; i<NUMBER_OF_UE_MAX; i++) {
    // If we've dropped the UE, go back to PRACH mode for this UE
    if ((frame==0)&&(subframe==0)) {
      if (eNB->UE_stats[i].crnti > 0) {
  LOG_I(PHY,"UE %d : rnti %x\n",i,eNB->UE_stats[i].crnti);
      }
    }
    if (eNB->UE_stats[i].ulsch_consecutive_errors == ULSCH_max_consecutive_errors) {
      LOG_W(PHY,"[eNB %d, CC %d] frame %d, subframe %d, UE %d: ULSCH consecutive error count reached %u, triggering UL Failure\n",
            eNB->Mod_id,eNB->CC_id,frame,subframe, i, eNB->UE_stats[i].ulsch_consecutive_errors);
      eNB->UE_stats[i].ulsch_consecutive_errors=0;
      UL_failure_indication(eNB->Mod_id,
               eNB->CC_id,
               frame,
               eNB->UE_stats[i].crnti,
               subframe);
               
    }
  

  }


  // clear the transmit data array for the current subframe
  if (eNB->abstraction_flag==0) {
    for (aa=0; aa<fp->nb_antenna_ports_eNB; aa++) {      
      memset(&eNB->common_vars.txdataF[0][aa][subframe*fp->ofdm_symbol_size*(fp->symbols_per_tti)],
             0,fp->ofdm_symbol_size*(fp->symbols_per_tti)*sizeof(int32_t));
    }
  }

 if (eNB->mac_enabled==1) {
    // Parse DCI received from MAC
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_ENB_PDCCH_TX,1);
    DCI_pdu = get_dci_sdu(eNB->Mod_id,
             eNB->CC_id,
             frame,
             subframe);
  }
  else {
    DCI_pdu = &DCI_pdu_tmp;

#ifdef EMOS_CHANNEL
   // fill_dci_emos(DCI_pdu,eNB);
#else

#endif
  }

  /* save old HARQ information needed for PHICH generation */
  for (i=0; i<NUMBER_OF_UE_MAX; i++) {
    if (eNB->ulsch[i]) {
      /* Store first_rb and n_DMRS for correct PHICH generation below.
       * For PHICH generation we need "old" values of last scheduling
       * for this HARQ process. 'generate_eNB_dlsch_params' below will
       * overwrite first_rb and n_DMRS and 'generate_phich_top', done
       * after 'generate_eNB_dlsch_params', would use the "new" values
       * instead of the "old" ones.
       *
       * This has been tested for FDD only, may be wrong for TDD.
       *
       * TODO: maybe we should restructure the code to be sure it
       *       is done correctly. The main concern is if the code
       *       changes and first_rb and n_DMRS are modified before
       *       we reach here, then the PHICH processing will be wrong,
       *       using wrong first_rb and n_DMRS values to compute
       *       ngroup_PHICH and nseq_PHICH.
       *
       * TODO: check if that works with TDD.
       */
      if ((subframe_select(fp,ul_subframe)==SF_UL) ||
          (fp->frame_type == FDD)) {
        harq_pid = subframe2harq_pid(fp,ul_frame,ul_subframe);
        eNB->ulsch[i]->harq_processes[harq_pid]->previous_first_rb =
            eNB->ulsch[i]->harq_processes[harq_pid]->first_rb;
        eNB->ulsch[i]->harq_processes[harq_pid]->previous_n_DMRS =
            eNB->ulsch[i]->harq_processes[harq_pid]->n_DMRS;
      }
    }
  }

  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_DCI_INFO,(frame*10)+subframe);


  // if we have DCI to generate do it now
  if ((DCI_pdu->Num_common_dci + DCI_pdu->Num_ue_spec_dci)>0) {


  } else { // for emulation!!
    eNB->num_ue_spec_dci[(subframe)&1]=0;
    eNB->num_common_dci[(subframe)&1]=0;
  }

  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_DCI_INFO,DCI_pdu->num_npdcch_symbols);

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_ENB_PDCCH_TX,0);

  // Check for RA activity
  if ((eNB->dlsch_ra) && (eNB->dlsch_ra->active == 1)) {

    
    LOG_D(PHY,"[eNB %"PRIu8"][RAPROC] Frame %d, subframe %d: Calling generate_dlsch (RA),Msg3 frame %"PRIu32", Msg3 subframe %"PRIu8"\n",
    eNB->Mod_id,
    frame, subframe,
    eNB->ulsch[(uint32_t)UE_id]->Msg3_frame,
    eNB->ulsch[(uint32_t)UE_id]->Msg3_subframe);
    
   // pdsch_procedures(eNB,proc,eNB->dlsch_ra,(LTE_eNB_DLSCH_t*)NULL,(LTE_eNB_UE_stats*)NULL,1,num_npdcch_symbols);    
    
    eNB->dlsch_ra->active = 0;
  }

  // Now scan UE specific DLSCH
  for (UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++)
  {
      if ((eNB->ndlsch[(uint8_t)UE_id][0])&&
           (eNB->ndlsch[(uint8_t)UE_id][0]->rnti>0)&&
             (eNB->ndlsch[(uint8_t)UE_id][0]->active == 1)) {

     //pdsch_procedures(eNB,proc,eNB->dlsch[(uint8_t)UE_id][0],eNB->dlsch[(uint8_t)UE_id][1],&eNB->UE_stats[(uint32_t)UE_id],0,num_npdcch_symbols);

  }

      else if ((eNB->ndlsch[(uint8_t)UE_id][0])&&
         (eNB->ndlsch[(uint8_t)UE_id][0]->rnti>0)&&
         (eNB->ndlsch[(uint8_t)UE_id][0]->active == 0)) {

  // clear subframe TX flag since UE is not scheduled for PDSCH in this subframe (so that we don't look for PUCCH later)
  //eNB->dlsch[(uint8_t)UE_id][0]->subframe_tx[subframe]=0;
      }
    }

  // if we have PHICH to generate

  if (is_phich_subframe(fp,subframe))
    {
     /* generate_phich_top(eNB,
       proc,
       AMP,
       0);*/
    }



#ifdef EMOS
 // phy_procedures_emos_eNB_TX(subframe, eNB);
#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_TX+offset,0);
  if (do_meas==1) stop_meas(&eNB->phy_proc_tx);
  
}



void process_Msg3(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,uint8_t UE_id, uint8_t harq_pid)
{
  // this prepares the demodulation of the first PUSCH of a new user, containing Msg3
  int subframe = proc->subframe_rx;
  int frame = proc->frame_rx;

  LOG_D(PHY,"[eNB %d][RAPROC] frame %d : subframe %d : process_Msg3 UE_id %d (active %d, subframe %d, frame %d)\n",
        eNB->Mod_id,
        frame,subframe,
        UE_id,eNB->ulsch[(uint32_t)UE_id]->Msg3_active,
        eNB->ulsch[(uint32_t)UE_id]->Msg3_subframe,
        eNB->ulsch[(uint32_t)UE_id]->Msg3_frame);
  eNB->ulsch[(uint32_t)UE_id]->Msg3_flag = 0;

  if ((eNB->ulsch[(uint32_t)UE_id]->Msg3_active == 1) &&
      (eNB->ulsch[(uint32_t)UE_id]->Msg3_subframe == subframe) &&
      (eNB->ulsch[(uint32_t)UE_id]->Msg3_frame == (uint32_t)frame))   {

    //    harq_pid = 0;

    eNB->ulsch[(uint32_t)UE_id]->Msg3_active = 0;
    eNB->ulsch[(uint32_t)UE_id]->Msg3_flag = 1;
    eNB->ulsch[(uint32_t)UE_id]->harq_processes[harq_pid]->subframe_scheduling_flag=1;
    LOG_D(PHY,"[eNB %d][RAPROC] frame %d, subframe %d: Setting subframe_scheduling_flag (Msg3) for UE %d\n",
          eNB->Mod_id,
          frame,subframe,UE_id);
  }
}



// This function retrieves the harq_pid of the corresponding DLSCH process
// and updates the error statistics of the DLSCH based on the received ACK
// info from UE along with the round index.  It also performs the fine-grain
// rate-adaptation based on the error statistics derived from the ACK/NAK process

void process_HARQ_feedback(uint8_t UE_id,
                           PHY_VARS_eNB_NB_IoT *eNB,
         eNB_rxtx_proc_NB_IoT_t *proc,
                           uint8_t pusch_flag,
                           uint8_t *pucch_payload,
                           uint8_t pucch_sel,
                           uint8_t SR_payload)
{

  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  uint8_t dl_harq_pid[8],dlsch_ACK[8],dl_subframe;
  LTE_eNB_DLSCH_t *dlsch             =  eNB->ndlsch[(uint32_t)UE_id][0];
  LTE_eNB_UE_stats *ue_stats         =  &eNB->UE_stats[(uint32_t)UE_id];
  LTE_DL_eNB_HARQ_t *dlsch_harq_proc;
  uint8_t subframe_m4,M,m;
  int mp;
  int all_ACKed=1,nb_alloc=0,nb_ACK=0;
  int frame = proc->frame_rx;
  int subframe = proc->subframe_rx;
  int harq_pid = subframe2harq_pid( fp,frame,subframe);

  if (fp->frame_type == FDD) { //FDD
    subframe_m4 = (subframe<4) ? subframe+6 : subframe-4;

    dl_harq_pid[0] = dlsch->harq_ids[subframe_m4];
    M=1;

    if (pusch_flag == 1) {
      dlsch_ACK[0] = eNB->ulsch[(uint8_t)UE_id]->harq_processes[harq_pid]->o_ACK[0];
      if (dlsch->subframe_tx[subframe_m4]==1)
  LOG_D(PHY,"[eNB %d] Frame %d: Received ACK/NAK %d on PUSCH for subframe %d\n",eNB->Mod_id,
        frame,dlsch_ACK[0],subframe_m4);
    }
    else {
      dlsch_ACK[0] = pucch_payload[0];
      LOG_D(PHY,"[eNB %d] Frame %d: Received ACK/NAK %d on PUCCH for subframe %d\n",eNB->Mod_id,
      frame,dlsch_ACK[0],subframe_m4);
      /*
  if (dlsch_ACK[0]==0)
  AssertFatal(0,"Exiting on NAK on PUCCH\n");
      */
    }


#if defined(MESSAGE_CHART_GENERATOR_PHY)
    MSC_LOG_RX_MESSAGE(
           MSC_PHY_ENB,MSC_PHY_UE,
           NULL,0,
           "%05u:%02u %s received %s  rnti %x harq id %u  tx SF %u",
           frame,subframe,
           (pusch_flag == 1)?"PUSCH":"PUCCH",
           (dlsch_ACK[0])?"ACK":"NACK",
           dlsch->rnti,
           dl_harq_pid[0],
           subframe_m4
           );
#endif
  } else { // TDD Handle M=1,2 cases only

    M=ul_ACK_subframe2_M(fp,
                         subframe);

    // Now derive ACK information for TDD
    if (pusch_flag == 1) { // Do PUSCH ACK/NAK first
      // detect missing DAI
      //FK: this code is just a guess
      //RK: not exactly, yes if scheduled from PHICH (i.e. no DCI format 0)
      //    otherwise, it depends on how many of the PDSCH in the set are scheduled, we can leave it like this,
      //    but we have to adapt the code below.  For example, if only one out of 2 are scheduled, only 1 bit o_ACK is used

      dlsch_ACK[0] = eNB->ulsch[(uint8_t)UE_id]->harq_processes[harq_pid]->o_ACK[0];
      dlsch_ACK[1] = (eNB->pucch_config_dedicated[UE_id].tdd_AckNackFeedbackMode == bundling)
  ?eNB->ulsch[(uint8_t)UE_id]->harq_processes[harq_pid]->o_ACK[0]:eNB->ulsch[(uint8_t)UE_id]->harq_processes[harq_pid]->o_ACK[1];
    }

    else {  // PUCCH ACK/NAK
      if ((SR_payload == 1)&&(pucch_sel!=2)) {  // decode Table 7.3 if multiplexing and SR=1
        nb_ACK = 0;

        if (M == 2) {
          if ((pucch_payload[0] == 1) && (pucch_payload[1] == 1)) // b[0],b[1]
            nb_ACK = 1;
          else if ((pucch_payload[0] == 1) && (pucch_payload[1] == 0))
            nb_ACK = 2;
        } else if (M == 3) {
          if ((pucch_payload[0] == 1) && (pucch_payload[1] == 1))
            nb_ACK = 1;
          else if ((pucch_payload[0] == 1) && (pucch_payload[1] == 0))
            nb_ACK = 2;
          else if ((pucch_payload[0] == 0) && (pucch_payload[1] == 1))
            nb_ACK = 3;
        }
      } else if (pucch_sel == 2) { // bundling or M=1
        dlsch_ACK[0] = pucch_payload[0];
        dlsch_ACK[1] = pucch_payload[0];
      } else { // multiplexing with no SR, this is table 10.1
        if (M==1)
          dlsch_ACK[0] = pucch_payload[0];
        else if (M==2) {
          if (((pucch_sel == 1) && (pucch_payload[0] == 1) && (pucch_payload[1] == 1)) ||
              ((pucch_sel == 0) && (pucch_payload[0] == 0) && (pucch_payload[1] == 1)))
            dlsch_ACK[0] = 1;
          else
            dlsch_ACK[0] = 0;

          if (((pucch_sel == 1) && (pucch_payload[0] == 1) && (pucch_payload[1] == 1)) ||
              ((pucch_sel == 1) && (pucch_payload[0] == 0) && (pucch_payload[1] == 0)))
            dlsch_ACK[1] = 1;
          else
            dlsch_ACK[1] = 0;
        }
      }
    }
  }

  // handle case where positive SR was transmitted with multiplexing
  if ((SR_payload == 1)&&(pucch_sel!=2)&&(pusch_flag == 0)) {
    nb_alloc = 0;

    for (m=0; m<M; m++) {
      dl_subframe = ul_ACK_subframe2_dl_subframe(fp,
             subframe,
             m);

      if (dlsch->subframe_tx[dl_subframe]==1)
        nb_alloc++;
    }

    if (nb_alloc == nb_ACK)
      all_ACKed = 1;
    else
      all_ACKed = 0;
  }


  for (m=0,mp=-1; m<M; m++) {

    dl_subframe = ul_ACK_subframe2_dl_subframe(fp,
                 subframe,
                 m);

    if (dlsch->subframe_tx[dl_subframe]==1) {
      if (pusch_flag == 1)
        mp++;
      else
        mp = m;

      dl_harq_pid[m]     = dlsch->harq_ids[dl_subframe];
      harq_pid_updated[UE_id][dl_harq_pid[m]] = 1;

      if ((pucch_sel != 2)&&(pusch_flag == 0)) { // multiplexing
        if ((SR_payload == 1)&&(all_ACKed == 1))
          dlsch_ACK[m] = 1;
        else
          dlsch_ACK[m] = 0;
      }

      if (dl_harq_pid[m]<dlsch->Mdlharq) {
        dlsch_harq_proc = dlsch->harq_processes[dl_harq_pid[m]];
#ifdef DEBUG_PHY_PROC
        LOG_D(PHY,"[eNB %d][PDSCH %x/%d] subframe %d, status %d, round %d (mcs %d, rv %d, TBS %d)\n",eNB->Mod_id,
              dlsch->rnti,dl_harq_pid[m],dl_subframe,
              dlsch_harq_proc->status,dlsch_harq_proc->round,
              dlsch->harq_processes[dl_harq_pid[m]]->mcs,
              dlsch->harq_processes[dl_harq_pid[m]]->rvidx,
              dlsch->harq_processes[dl_harq_pid[m]]->TBS);

        if (dlsch_harq_proc->status==DISABLED)
          LOG_E(PHY,"dlsch_harq_proc is disabled? \n");

#endif

        if ((dl_harq_pid[m]<dlsch->Mdlharq) &&
            (dlsch_harq_proc->status == ACTIVE)) {
          // dl_harq_pid of DLSCH is still active

          if ( dlsch_ACK[mp]==0) {
            // Received NAK
#ifdef DEBUG_PHY_PROC
            LOG_D(PHY,"[eNB %d][PDSCH %x/%d] M = %d, m= %d, mp=%d NAK Received in round %d, requesting retransmission\n",eNB->Mod_id,
                  dlsch->rnti,dl_harq_pid[m],M,m,mp,dlsch_harq_proc->round);
#endif

            T(T_ENB_PHY_DLSCH_UE_NACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(UE_id), T_INT(dlsch->rnti),
              T_INT(dl_harq_pid[m]));

            if (dlsch_harq_proc->round == 0)
              ue_stats->dlsch_NAK_round0++;

            ue_stats->dlsch_NAK[dl_harq_pid[m]][dlsch_harq_proc->round]++;


            // then Increment DLSCH round index
            dlsch_harq_proc->round++;


            if (dlsch_harq_proc->round == dlsch->Mlimit) {
              // This was the last round for DLSCH so reset round and increment l2_error counter
#ifdef DEBUG_PHY_PROC
              LOG_W(PHY,"[eNB %d][PDSCH %x/%d] DLSCH retransmissions exhausted, dropping packet\n",eNB->Mod_id,
                    dlsch->rnti,dl_harq_pid[m]);
#endif
#if defined(MESSAGE_CHART_GENERATOR_PHY)
              MSC_LOG_EVENT(MSC_PHY_ENB, "0 HARQ DLSCH Failed RNTI %"PRIx16" round %u",
                            dlsch->rnti,
                            dlsch_harq_proc->round);
#endif

              dlsch_harq_proc->round = 0;
              ue_stats->dlsch_l2_errors[dl_harq_pid[m]]++;
              dlsch_harq_proc->status = SCH_IDLE;
              dlsch->harq_ids[0][dl_subframe] = dlsch->Mdlharq;
            }
          } else {
#ifdef DEBUG_PHY_PROC
            LOG_D(PHY,"[eNB %d][PDSCH %x/%d] ACK Received in round %d, resetting process\n",eNB->Mod_id,
                  dlsch->rnti,dl_harq_pid[m],dlsch_harq_proc->round);
#endif

            T(T_ENB_PHY_DLSCH_UE_ACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(UE_id), T_INT(dlsch->rnti),
              T_INT(dl_harq_pid[m]));

            ue_stats->dlsch_ACK[dl_harq_pid[m]][dlsch_harq_proc->round]++;

            // Received ACK so set round to 0 and set dlsch_harq_pid IDLE
            dlsch_harq_proc->round  = 0;
            dlsch_harq_proc->status = SCH_IDLE;
            dlsch->harq_ids[0][dl_subframe] = dlsch->Mdlharq;

            ue_stats->total_TBS = ue_stats->total_TBS +
        eNB->ndlsch[(uint8_t)UE_id][0]->harq_processes[dl_harq_pid[m]]->TBS;
            /*
              ue_stats->total_transmitted_bits = ue_stats->total_transmitted_bits +
              eNB->dlsch[(uint8_t)UE_id][0]->harq_processes[dl_harq_pid[m]]->TBS;
            */
          }
   
          // Do fine-grain rate-adaptation for DLSCH
          if (ue_stats->dlsch_NAK_round0 > dlsch->error_threshold) {
            if (ue_stats->dlsch_mcs_offset == 1)
              ue_stats->dlsch_mcs_offset=0;
            else
              ue_stats->dlsch_mcs_offset=-1;
          }

#ifdef DEBUG_PHY_PROC
          LOG_D(PHY,"[process_HARQ_feedback] Frame %d Setting round to %d for pid %d (subframe %d)\n",frame,
                dlsch_harq_proc->round,dl_harq_pid[m],subframe);
#endif
    harq_pid_round[UE_id][dl_harq_pid[m]] = dlsch_harq_proc->round;
          // Clear NAK stats and adjust mcs offset
          // after measurement window timer expires
          if (ue_stats->dlsch_sliding_cnt == dlsch->ra_window_size) {
            if ((ue_stats->dlsch_mcs_offset == 0) && (ue_stats->dlsch_NAK_round0 < 2))
              ue_stats->dlsch_mcs_offset = 1;

            if ((ue_stats->dlsch_mcs_offset == 1) && (ue_stats->dlsch_NAK_round0 > 2))
              ue_stats->dlsch_mcs_offset = 0;

            if ((ue_stats->dlsch_mcs_offset == 0) && (ue_stats->dlsch_NAK_round0 > 2))
              ue_stats->dlsch_mcs_offset = -1;

            if ((ue_stats->dlsch_mcs_offset == -1) && (ue_stats->dlsch_NAK_round0 < 2))
              ue_stats->dlsch_mcs_offset = 0;

            ue_stats->dlsch_NAK_round0 = 0;
            ue_stats->dlsch_sliding_cnt = 0;
          }
        }
      }
    }
  }
}



void get_n1_pucch_eNB(PHY_VARS_eNB_NB_IoT *eNB,
          eNB_rxtx_proc_NB_IoT_t *proc,
                      uint8_t UE_id,
                      int16_t *n1_pucch0,
                      int16_t *n1_pucch1,
                      int16_t *n1_pucch2,
                      int16_t *n1_pucch3)
{

  LTE_DL_FRAME_PARMS *frame_parms=&eNB->frame_parms;
  uint8_t nCCE0,nCCE1;
  int sf;
  int frame = proc->frame_rx;
  int subframe = proc->subframe_rx;

  if (frame_parms->frame_type == FDD ) {
    sf = (subframe<4) ? (subframe+6) : (subframe-4);

    if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[sf]>0) {
      *n1_pucch0 = frame_parms->pucch_config_common.n1PUCCH_AN + eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[sf];
      *n1_pucch1 = -1;
    } else {
      *n1_pucch0 = -1;
      *n1_pucch1 = -1;
    }
  } else {

    switch (frame_parms->tdd_config) {
    case 1:  // DL:S:UL:UL:DL:DL:S:UL:UL:DL
      if (subframe == 2) {  // ACK subframes 5 and 6
        /*  if (eNB->dlsch[(uint32_t)UE_id][0]->subframe_tx[6]>0) {
      nCCE1 = eNB->dlsch[(uint32_t)UE_id][0]->nCCE[6];
      *n1_pucch1 = get_Np(frame_parms->N_RB_DL,nCCE1,1) + nCCE1 + frame_parms->pucch_config_common.n1PUCCH_AN;
      }
      else
      *n1_pucch1 = -1;*/

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[5]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[5];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0+ frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;

        *n1_pucch1 = -1;
      } else if (subframe == 3) { // ACK subframe 9

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[9]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[9];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0 +frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;

        *n1_pucch1 = -1;

      } else if (subframe == 7) { // ACK subframes 0 and 1
        //harq_ack[0].nCCE;
        //harq_ack[1].nCCE;
        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[0]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[0];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0 + frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;

        *n1_pucch1 = -1;
      } else if (subframe == 8) { // ACK subframes 4
        //harq_ack[4].nCCE;
        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[4]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[4];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0 + frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;

        *n1_pucch1 = -1;
      } else {
        LOG_D(PHY,"[eNB %d] frame %d: phy_procedures_lte.c: get_n1pucch, illegal subframe %d for tdd_config %d\n",
              eNB->Mod_id,
              frame,
              subframe,frame_parms->tdd_config);
        return;
      }

      break;

    case 3:  // DL:S:UL:UL:UL:DL:DL:DL:DL:DL
      if (subframe == 2) {  // ACK subframes 5,6 and 1 (S in frame-2), forget about n-11 for the moment (S-subframe)
        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[6]>0) {
          nCCE1 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[6];
          *n1_pucch1 = get_Np(frame_parms->N_RB_DL,nCCE1,1) + nCCE1 + frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch1 = -1;

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[5]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[5];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0+ frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;
      } else if (subframe == 3) { // ACK subframes 7 and 8
        LOG_D(PHY,"get_n1_pucch_eNB : subframe 3, subframe_tx[7] %d, subframe_tx[8] %d\n",
              eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[7],eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[8]);

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[8]>0) {
          nCCE1 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[8];
          *n1_pucch1 = get_Np(frame_parms->N_RB_DL,nCCE1,1) + nCCE1 + frame_parms->pucch_config_common.n1PUCCH_AN;
          LOG_D(PHY,"nCCE1 %d, n1_pucch1 %d\n",nCCE1,*n1_pucch1);
        } else
          *n1_pucch1 = -1;

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[7]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[7];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0 +frame_parms->pucch_config_common.n1PUCCH_AN;
          LOG_D(PHY,"nCCE0 %d, n1_pucch0 %d\n",nCCE0,*n1_pucch0);
        } else
          *n1_pucch0 = -1;
      } else if (subframe == 4) { // ACK subframes 9 and 0
        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[0]>0) {
          nCCE1 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[0];
          *n1_pucch1 = get_Np(frame_parms->N_RB_DL,nCCE1,1) + nCCE1 + frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch1 = -1;

        if (eNB->ndlsch[(uint32_t)UE_id][0]->subframe_tx[9]>0) {
          nCCE0 = eNB->ndlsch[(uint32_t)UE_id][0]->nCCE[9];
          *n1_pucch0 = get_Np(frame_parms->N_RB_DL,nCCE0,0) + nCCE0 +frame_parms->pucch_config_common.n1PUCCH_AN;
        } else
          *n1_pucch0 = -1;
      } else {
        LOG_D(PHY,"[eNB %d] Frame %d: phy_procedures_lte.c: get_n1pucch, illegal subframe %d for tdd_config %d\n",
              eNB->Mod_id,frame,subframe,frame_parms->tdd_config);
        return;
      }

      break;
    }  // switch tdd_config

    // Don't handle the case M>2
    *n1_pucch2 = -1;
    *n1_pucch3 = -1;
  }
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////// NB-IoT testing ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void prach_procedures_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB) {

   uint16_t rnti[4],preamble_index[4],timing_advance_preamble[4];
 //  uint16_t i;
 //  int frame,subframe;

  uint8_t subframe = eNB->proc.subframe_prach;
  int frame = eNB->proc.frame_prach;
 // uint8_t CC_id = eNB->CC_id;
  uint32_t detection=0;
  //uint16_t estimated_TA=2;

  if (eNB->abstraction_flag == 0) {
         /* rx_prach(eNB,
                     preamble_energy_list,
                     preamble_delay_list,
                     frame,
                     0);*/
      detection = rx_nprach_NB_IoT(eNB,frame,subframe,rnti,preamble_index,timing_advance_preamble);
  }
 
 if(detection == 1)    ////////////////////////// to be moved to handle_rach_NB_IoT
  {
      

      pthread_mutex_lock(&eNB->UL_INFO_mutex);
                                                                                 //////////////////////////////////////////////////////////       
      eNB->UL_INFO.nrach_ind.number_of_initial_scs_detected  = 1;            //!!!!!!!!!!!!!   // should be set to zero in every call of UL_indication !!!!!!!!!!!!!!!!!!!!!!!
      eNB->UL_INFO.nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.rnti             = rnti[0];
      eNB->UL_INFO.nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.initial_sc       = preamble_index[0];
      eNB->UL_INFO.nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.timing_advance   = timing_advance_preamble[0];
      eNB->UL_INFO.nrach_ind.nrach_pdu_list[0].nrach_indication_rel13.nrach_ce_level   = 2;

      eNB->UL_INFO.frame = frame;
      eNB->UL_INFO.subframe = subframe;
      eNB->UL_INFO.hypersfn = eNB->proc.proc_rxtx[0].HFN; 

      pthread_mutex_unlock(&eNB->UL_INFO_mutex);
      
  }
  
}
//////////////////////////////////////////////////////////// END ///////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




void prach_procedures_NB_IoT_testing(PHY_VARS_eNB_NB_IoT *eNB) {

  int subframe = eNB->proc.subframe_prach;
  int frame = eNB->proc.frame_prach;
 // uint8_t CC_id = eNB->CC_id;


  uint32_t detection=0;
  uint16_t estimated_TA=2;


  if (eNB->abstraction_flag == 0) {

  }

  /////////////////////////////////////////// NB-IoT testing //////////////////////////
 if(detection == 1)
  {
    initiate_ra_proc(eNB->Mod_id,
            eNB->CC_id,
            frame,
            eNB->preamble_index_NB_IoT,
            estimated_TA,
            0,subframe,0);      

  }
  /////////////////////////////////////////////////////////////////////////////////////

}



void pucch_procedures(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,int UE_id,int harq_pid,uint8_t do_srs)
{
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  uint8_t SR_payload = 0,*pucch_payload=NULL,pucch_payload0[2]= {0,0},pucch_payload1[2]= {0,0};
  int16_t n1_pucch0 = -1, n1_pucch1 = -1, n1_pucch2 = -1, n1_pucch3 = -1;
  uint8_t do_SR = 0;
  uint8_t pucch_sel = 0;
  int32_t metric0=0,metric1=0,metric0_SR=0;
  ANFBmode_t bundling_flag;
  PUCCH_FMT_t format;
  const int subframe = proc->subframe_rx;
  const int frame = proc->frame_rx;

  if ((eNB->ndlsch[UE_id][0]) &&
      (eNB->ndlsch[UE_id][0]->rnti>0) &&
      (eNB->ulsch[UE_id]->harq_processes[harq_pid]->subframe_scheduling_flag==0)) {

    // check SR availability
    do_SR = is_SR_subframe(eNB,proc,UE_id);
    //      do_SR = 0;

    // Now ACK/NAK
    // First check subframe_tx flag for earlier subframes

    get_n1_pucch_eNB(eNB,
                     proc,
                     UE_id,
                     &n1_pucch0,
                     &n1_pucch1,
                     &n1_pucch2,
                     &n1_pucch3);

    LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d, subframe %d Checking for PUCCH (%d,%d,%d,%d) SR %d\n",
          eNB->Mod_id,eNB->ndlsch[UE_id][0]->rnti,
          frame,subframe,
          n1_pucch0,n1_pucch1,n1_pucch2,n1_pucch3,do_SR);

    if ((n1_pucch0==-1) && (n1_pucch1==-1) && (do_SR==0)) {  // no TX PDSCH that have to be checked and no SR for this UE_id
    } else {
      // otherwise we have some PUCCH detection to do

      // Null out PUCCH PRBs for noise measurement
      switch(fp->N_RB_UL) {
      case 6:
        eNB->rb_mask_ul[0] |= (0x1 | (1<<5)); //position 5
        break;
      case 15:
        eNB->rb_mask_ul[0] |= (0x1 | (1<<14)); // position 14
        break;
      case 25:
        eNB->rb_mask_ul[0] |= (0x1 | (1<<24)); // position 24
        break;
      case 50:
        eNB->rb_mask_ul[0] |= 0x1;
        eNB->rb_mask_ul[1] |= (1<<17); // position 49 (49-32)
        break;
      case 75:
        eNB->rb_mask_ul[0] |= 0x1;
        eNB->rb_mask_ul[2] |= (1<<10); // position 74 (74-64)
        break;
      case 100:
        eNB->rb_mask_ul[0] |= 0x1;
        eNB->rb_mask_ul[3] |= (1<<3); // position 99 (99-96)
        break;
      default:
        LOG_E(PHY,"Unknown number for N_RB_UL %d\n",fp->N_RB_UL);
        break;
      }

      if (do_SR == 1) {
        eNB->UE_stats[UE_id].sr_total++;


        if (eNB->abstraction_flag == 0) {
          metric0_SR = rx_pucch(eNB,
                                pucch_format1,
                                UE_id,
                                eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex,
                                0, // n2_pucch
                                do_srs, // shortened format
                                &SR_payload,
                                frame,
                                subframe,
                                PUCCH1_THRES);
          LOG_D(PHY,"[eNB %d][SR %x] Frame %d subframe %d Checking SR is %d (SR n1pucch is %d)\n",
                eNB->Mod_id,
                eNB->ulsch[UE_id]->rnti,
                frame,
                subframe,
                SR_payload,
                eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex);
        }
#ifdef PHY_ABSTRACTION
        else {
          metric0_SR = rx_pucch_emul(eNB,
                                     proc,
                                     UE_id,
                                     pucch_format1,
                                     0,
                                     &SR_payload);
          LOG_D(PHY,"[eNB %d][SR %x] Frame %d subframe %d Checking SR (UE SR %d/%d)\n",eNB->Mod_id,
                eNB->ulsch[UE_id]->rnti,frame,subframe,SR_payload,eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex);
        }
#endif
      }// do_SR==1

      if ((n1_pucch0==-1) && (n1_pucch1==-1)) { // just check for SR
      } else if (fp->frame_type==FDD) { // FDD
        // if SR was detected, use the n1_pucch from SR, else use n1_pucch0
        //          n1_pucch0 = (SR_payload==1) ? eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex:n1_pucch0;

        LOG_D(PHY,"Demodulating PUCCH for ACK/NAK: n1_pucch0 %d (%d), SR_payload %d\n",n1_pucch0,eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex,SR_payload);

        if (eNB->abstraction_flag == 0) {
          metric0 = rx_pucch(eNB,
                             pucch_format1a,
                             UE_id,
                             (uint16_t)n1_pucch0,
                             0, //n2_pucch
                             do_srs, // shortened format
                             pucch_payload0,
                             frame,
                             subframe,
                             PUCCH1a_THRES);
        }
#ifdef PHY_ABSTRACTION
        else {
          metric0 = rx_pucch_emul(eNB,
                                  proc,
                                  UE_id,
                                  pucch_format1a,
                                  0,
                                  pucch_payload0);
        }
#endif

        /* cancel SR detection if reception on n1_pucch0 is better than on SR PUCCH resource index */
        if (do_SR && metric0 > metric0_SR) SR_payload = 0;

        if (do_SR && metric0 <= metric0_SR) {
          /* when transmitting ACK/NACK on SR PUCCH resource index, SR payload is always 1 */
          SR_payload = 1;

          if (eNB->abstraction_flag == 0) {
            metric0=rx_pucch(eNB,
                             pucch_format1a,
                             UE_id,
                             eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex,
                             0, //n2_pucch
                             do_srs, // shortened format
                             pucch_payload0,
                             frame,
                             subframe,
                             PUCCH1a_THRES);
          }
#ifdef PHY_ABSTRACTION
          else {
            metric0 = rx_pucch_emul(eNB,
                                    proc,
                                    UE_id,
                                    pucch_format1a,
                                    0,
                                    pucch_payload0);
          }
#endif
        }

#ifdef DEBUG_PHY_PROC
        LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d pucch1a (FDD) payload %d (metric %d)\n",
            eNB->Mod_id,
            eNB->ndlsch[UE_id][0]->rnti,
            frame,subframe,
            pucch_payload0[0],metric0);
#endif

        process_HARQ_feedback(UE_id,eNB,proc,
                            0,// pusch_flag
                            pucch_payload0,
                            2,
                            SR_payload);
      } // FDD
      else {  //TDD

        bundling_flag = eNB->pucch_config_dedicated[UE_id].tdd_AckNackFeedbackMode;

        // fix later for 2 TB case and format1b

        if ((fp->frame_type==FDD) ||
          (bundling_flag==bundling)    ||
          ((fp->frame_type==TDD)&&(fp->tdd_config==1)&&((subframe!=2)&&(subframe!=7)))) {
          format = pucch_format1a;
        } else {
          format = pucch_format1b;
        }

        // if SR was detected, use the n1_pucch from SR
        if (SR_payload==1) {
#ifdef DEBUG_PHY_PROC
          LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d Checking ACK/NAK (%d,%d,%d,%d) format %d with SR\n",eNB->Mod_id,
                eNB->ndlsch[UE_id][0]->rnti,
                frame,subframe,
                n1_pucch0,n1_pucch1,n1_pucch2,n1_pucch3,format);
#endif

          if (eNB->abstraction_flag == 0)
            metric0 = rx_pucch(eNB,
                               format,
                               UE_id,
                               eNB->scheduling_request_config[UE_id].sr_PUCCH_ResourceIndex,
                               0, //n2_pucch
                               do_srs, // shortened format
                               pucch_payload0,
                               frame,
                               subframe,
                               PUCCH1a_THRES);
          else {
#ifdef PHY_ABSTRACTION
            metric0 = rx_pucch_emul(eNB,proc,
                                    UE_id,
                                    format,
                                    0,
                                    pucch_payload0);
#endif
          }
        } else { //using n1_pucch0/n1_pucch1 resources
#ifdef DEBUG_PHY_PROC
          LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d Checking ACK/NAK (%d,%d,%d,%d) format %d\n",eNB->Mod_id,
                eNB->ndlsch[UE_id][0]->rnti,
                frame,subframe,
                n1_pucch0,n1_pucch1,n1_pucch2,n1_pucch3,format);
#endif
          metric0=0;
          metric1=0;

          // Check n1_pucch0 metric
          if (n1_pucch0 != -1) {
            if (eNB->abstraction_flag == 0)
              metric0 = rx_pucch(eNB,
                                 format,
                                 UE_id,
                                 (uint16_t)n1_pucch0,
                                 0, // n2_pucch
         do_srs, // shortened format
                                 pucch_payload0,
                                 frame,
                                 subframe,
                                 PUCCH1a_THRES);
            else {
#ifdef PHY_ABSTRACTION
              metric0 = rx_pucch_emul(eNB,
                                      proc,
                                      UE_id,
                                      format,
                                      0,
                                      pucch_payload0);
#endif
            }
          }

          // Check n1_pucch1 metric
          if (n1_pucch1 != -1) {
            if (eNB->abstraction_flag == 0)
              metric1 = rx_pucch(eNB,
                                 format,
                                 UE_id,
                                 (uint16_t)n1_pucch1,
                                 0, //n2_pucch
                                 do_srs, // shortened format
                                 pucch_payload1,
                                 frame,
                                 subframe,
                                 PUCCH1a_THRES);
            else {
#ifdef PHY_ABSTRACTION
              metric1 = rx_pucch_emul(eNB,
                                      proc,
                                      UE_id,
                                      format,
                                      1,
                                      pucch_payload1);
#endif
            }
          }
        }

        if (SR_payload == 1) {
          pucch_payload = pucch_payload0;

          if (bundling_flag == bundling)
            pucch_sel = 2;
        } else if (bundling_flag == multiplexing) { // multiplexing + no SR
          pucch_payload = (metric1>metric0) ? pucch_payload1 : pucch_payload0;
          pucch_sel     = (metric1>metric0) ? 1 : 0;
        } else { // bundling + no SR
          if (n1_pucch1 != -1)
            pucch_payload = pucch_payload1;
          else if (n1_pucch0 != -1)
            pucch_payload = pucch_payload0;

          pucch_sel = 2;  // indicate that this is a bundled ACK/NAK
        }

#ifdef DEBUG_PHY_PROC
        LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d ACK/NAK metric 0 %d, metric 1 %d, sel %d, (%d,%d)\n",eNB->Mod_id,
              eNB->ndlsch[UE_id][0]->rnti,
              frame,subframe,
              metric0,metric1,pucch_sel,pucch_payload[0],pucch_payload[1]);
#endif
        process_HARQ_feedback(UE_id,eNB,proc,
                              0,// pusch_flag
                              pucch_payload,
                              pucch_sel,
                              SR_payload);
      } // TDD
    }

    if (SR_payload == 1) {
      LOG_D(PHY,"[eNB %d][SR %x] Frame %d subframe %d Got SR for PUSCH, transmitting to MAC\n",eNB->Mod_id,
            eNB->ulsch[UE_id]->rnti,frame,subframe);
      eNB->UE_stats[UE_id].sr_received++;

      if (eNB->first_sr[UE_id] == 1) { // this is the first request for uplink after Connection Setup, so clear HARQ process 0 use for Msg4
        eNB->first_sr[UE_id] = 0;
        eNB->ndlsch[UE_id][0]->harq_processes[0]->round=0;
        eNB->ndlsch[UE_id][0]->harq_processes[0]->status=SCH_IDLE;
        LOG_D(PHY,"[eNB %d][SR %x] Frame %d subframe %d First SR\n",
              eNB->Mod_id,
              eNB->ulsch[UE_id]->rnti,frame,subframe);
      }

      if (eNB->mac_enabled==1) {
        SR_indication(eNB->Mod_id,
                                 eNB->CC_id,
                                 frame,
                                 eNB->ndlsch[UE_id][0]->rnti,subframe,0);
      }
    }
  }
}




void cba_procedures(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,int UE_id,int harq_pid) {

  uint8_t access_mode;
  int num_active_cba_groups;
  const int subframe = proc->subframe_rx;
  const int frame = proc->frame_rx;
  uint16_t rnti=0;
  int ret=0;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;

  if (eNB->ulsch[UE_id]==NULL) return;

  num_active_cba_groups = eNB->ulsch[UE_id]->num_active_cba_groups;
 
  if ((num_active_cba_groups > 0) &&
      (eNB->ulsch[UE_id]->cba_rnti[UE_id%num_active_cba_groups]>0) &&
      (eNB->ulsch[UE_id]->harq_processes[harq_pid]->subframe_cba_scheduling_flag==1)) {
    rnti=0;
    
#ifdef DEBUG_PHY_PROC
    LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d Checking PUSCH/ULSCH CBA Reception for UE %d with cba rnti %x mode %s\n",
    eNB->Mod_id,harq_pid,
    frame,subframe,
    UE_id, (uint16_t)eNB->ulsch[UE_id]->cba_rnti[UE_id%num_active_cba_groups],mode_string[eNB->UE_stats[UE_id].mode]);
#endif
    
    if (eNB->abstraction_flag==0) {
      rx_ulsch(eNB,proc,
         eNB->UE_stats[UE_id].sector,  // this is the effective sector id
         UE_id,
         eNB->ulsch,
         0);
    }
    
#ifdef PHY_ABSTRACTION
    else {
      rx_ulsch_emul(eNB,proc,
        eNB->UE_stats[UE_id].sector,  // this is the effective sector id
        UE_id);
    }
    
#endif
    
    if (eNB->abstraction_flag == 0) {
      ret = ulsch_decoding(eNB,proc,
         UE_id,
         0, // control_only_flag
         eNB->ulsch[UE_id]->harq_processes[harq_pid]->V_UL_DAI,
         eNB->ulsch[UE_id]->harq_processes[harq_pid]->nb_rb>20 ? 1 : 0);
    }
    
#ifdef PHY_ABSTRACTION
    else {
      ret = ulsch_decoding_emul(eNB,
        proc,
        UE_id,
        &rnti);
    }
    
#endif
    
    if (eNB->ulsch[UE_id]->harq_processes[harq_pid]->cqi_crc_status == 1) {
#ifdef DEBUG_PHY_PROC
      
      print_CQI(eNB->ulsch[UE_id]->harq_processes[harq_pid]->o,eNB->ulsch[UE_id]->harq_processes[harq_pid]->uci_format,0,fp->N_RB_DL);
#endif
      access_mode = UNKNOWN_ACCESS;
      extract_CQI_NB_IoT(eNB->ulsch[UE_id]->harq_processes[harq_pid]->o,
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->uci_format,
      &eNB->UE_stats[UE_id],
      fp->N_RB_DL,
      &rnti, &access_mode);
      eNB->UE_stats[UE_id].rank = eNB->ulsch[UE_id]->harq_processes[harq_pid]->o_RI[0];
    }
    
    eNB->ulsch[UE_id]->harq_processes[harq_pid]->subframe_cba_scheduling_flag=0;
    eNB->ulsch[UE_id]->harq_processes[harq_pid]->status= SCH_IDLE;
      
    if ((num_active_cba_groups > 0) &&
  (UE_id + num_active_cba_groups < NUMBER_OF_UE_MAX) &&
  (eNB->ulsch[UE_id+num_active_cba_groups]->cba_rnti[UE_id%num_active_cba_groups] > 0 ) &&
  (eNB->ulsch[UE_id+num_active_cba_groups]->num_active_cba_groups> 0)) {
#ifdef DEBUG_PHY_PROC
      LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d UE %d harq_pid %d resetting the subframe_scheduling_flag for Ue %d cba groups %d members\n",
      eNB->Mod_id,harq_pid,frame,subframe,UE_id,harq_pid,
      UE_id+num_active_cba_groups, UE_id%eNB->ulsch[UE_id]->num_active_cba_groups);
#endif
      eNB->ulsch[UE_id+num_active_cba_groups]->harq_processes[harq_pid]->subframe_cba_scheduling_flag=1;
      eNB->ulsch[UE_id+num_active_cba_groups]->harq_processes[harq_pid]->status= CBA_ACTIVE;
      eNB->ulsch[UE_id+num_active_cba_groups]->harq_processes[harq_pid]->TBS=eNB->ulsch[UE_id]->harq_processes[harq_pid]->TBS;
    }

    if (ret == (1+MAX_TURBO_ITERATIONS)) {
      eNB->UE_stats[UE_id].ulsch_round_errors[harq_pid][eNB->ulsch[UE_id]->harq_processes[harq_pid]->round]++;
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->phich_active = 1;
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->phich_ACK = 0;
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->round++;
    } // ulsch in error
    else {
      LOG_D(PHY,"[eNB %d][PUSCH %d] Frame %d subframe %d ULSCH received, setting round to 0, PHICH ACK\n",
      eNB->Mod_id,harq_pid,
      frame,subframe);

      eNB->ulsch[UE_id]->harq_processes[harq_pid]->phich_active = 1;
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->phich_ACK = 1;
      eNB->ulsch[UE_id]->harq_processes[harq_pid]->round = 0;
      eNB->UE_stats[UE_id].ulsch_consecutive_errors = 0;
#ifdef DEBUG_PHY_PROC
#ifdef DEBUG_ULSCH
      LOG_D(PHY,"[eNB] Frame %d, Subframe %d : ULSCH SDU (RX harq_pid %d) %d bytes:",
      frame,subframe,
      harq_pid,eNB->ulsch[UE_id]->harq_processes[harq_pid]->TBS>>3);

      for (j=0; j<eNB->ulsch[UE_id]->harq_processes[harq_pid]->TBS>>3; j++)
  LOG_T(PHY,"%x.",eNB->ulsch[UE_id]->harq_processes[harq_pid]->b[j]);

      LOG_T(PHY,"\n");
#endif
#endif

      if (access_mode > UNKNOWN_ACCESS) {
  LOG_D(PHY,"[eNB %d] Frame %d, Subframe %d : received ULSCH SDU from CBA transmission, UE (%d,%x), CBA (group %d, rnti %x)\n",
        eNB->Mod_id, frame,subframe,
        UE_id, eNB->ulsch[UE_id]->rnti,
        UE_id % eNB->ulsch[UE_id]->num_active_cba_groups, eNB->ulsch[UE_id]->cba_rnti[UE_id%num_active_cba_groups]);

  // detect if there is a CBA collision
  if ((eNB->cba_last_reception[UE_id%num_active_cba_groups] == 0 ) && 
      (eNB->mac_enabled==1)) {
    rx_sdu(eNB->Mod_id,
          eNB->CC_id,
          frame,subframe,
          eNB->ulsch[UE_id]->rnti,
          eNB->ulsch[UE_id]->harq_processes[harq_pid]->b,
          eNB->ulsch[UE_id]->harq_processes[harq_pid]->TBS>>3,
          harq_pid,
          NULL);

    eNB->cba_last_reception[UE_id%num_active_cba_groups]+=1;//(subframe);
  } else {
    if (eNB->cba_last_reception[UE_id%num_active_cba_groups] == 1 )
      LOG_N(PHY,"[eNB%d] Frame %d subframe %d : first CBA collision detected \n ",
      eNB->Mod_id,frame,subframe);

    LOG_N(PHY,"[eNB%d] Frame %d subframe %d : CBA collision set SR for UE %d in group %d \n ",
    eNB->Mod_id,frame,subframe,
    eNB->cba_last_reception[UE_id%num_active_cba_groups],UE_id%num_active_cba_groups );

    eNB->cba_last_reception[UE_id%num_active_cba_groups]+=1;

    SR_indication(eNB->Mod_id,
           eNB->CC_id,
           frame,
           eNB->ndlsch[UE_id][0]->rnti,subframe,0);
  }
      } // UNKNOWN_ACCESS
    } // ULSCH CBA not in error
  }

}



void phy_procedures_eNB_uespec_RX_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc,const relaying_type_t r_type)
{
  //RX processing for ue-specific resources (i
  UNUSED(r_type);
  uint32_t ret=0,i,j,k;
  uint32_t harq_pid, harq_idx, round;
  uint8_t nPRS;
  int sync_pos;
  uint16_t rnti=0;
  uint8_t access_mode;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;

  const int subframe = proc->subframe_rx;
  const int frame    = proc->frame_rx;
  int offset         = eNB->CC_id;//(proc == &eNB->proc.proc_rxtx[0]) ? 0 : 1;

  uint16_t srsPeriodicity;
  uint16_t srsOffset;
  uint16_t do_srs=0;
  uint16_t is_srs_pos=0;

  T(T_ENB_PHY_UL_TICK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe));

  T(T_ENB_PHY_INPUT_SIGNAL, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(0),
    T_BUFFER(&eNB->common_vars.rxdata[0][0][subframe*eNB->frame_parms.samples_per_tti],
             eNB->frame_parms.samples_per_tti * 4));

  if ((fp->frame_type == TDD) && (subframe_select(fp,subframe)!=SF_UL)) return;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_RX_UESPEC+offset, 1 );

#ifdef DEBUG_PHY_PROC
  LOG_D(PHY,"[eNB %d] Frame %d: Doing phy_procedures_eNB_uespec_RX(%d)\n",eNB->Mod_id,frame, subframe);
#endif


  eNB->rb_mask_ul[0]=0;
  eNB->rb_mask_ul[1]=0;
  eNB->rb_mask_ul[2]=0;
  eNB->rb_mask_ul[3]=0;

  // Check for active processes in current subframe
  harq_pid = subframe2harq_pid(fp,
                               frame,subframe);

  // reset the cba flag used for collision detection
  for (i=0; i < NUM_MAX_CBA_GROUP; i++) {
    eNB->cba_last_reception[i]=0;
  }

  is_srs_pos = is_srs_occasion_common(fp,frame,subframe);
  
  for (i=0; i<NUMBER_OF_UE_MAX; i++) {

    // Do SRS processing 
    // check if there is SRS and we have to use shortened format
    // TODO: check for exceptions in transmission of SRS together with ACK/NACK
    do_srs=0;
    if (is_srs_pos && eNB->soundingrs_ul_config_dedicated[i].srsConfigDedicatedSetup ) {
      compute_srs_pos(fp->frame_type, eNB->soundingrs_ul_config_dedicated[i].srs_ConfigIndex, &srsPeriodicity, &srsOffset);
      if (((10*frame+subframe) % srsPeriodicity) == srsOffset) {
  do_srs = 1;
      }
    }

    if (do_srs==1) {
      if (lte_srs_channel_estimation(fp,
             &eNB->common_vars,
             &eNB->srs_vars[i],
             &eNB->soundingrs_ul_config_dedicated[i],
             subframe,
             0/*eNB_id*/)) {
  LOG_E(PHY,"problem processing SRS\n");
      }
    }

    // Do PUCCH processing 

    pucch_procedures(eNB,proc,i,harq_pid, do_srs);


    // check for Msg3
    if (eNB->mac_enabled==1) {
      if (eNB->UE_stats[i].mode == RA_RESPONSE) {
  process_Msg3(eNB,proc,i,harq_pid);
      }
    }


    eNB->pusch_stats_rb[i][(frame*10)+subframe] = -63;
    eNB->pusch_stats_round[i][(frame*10)+subframe] = 0;
    eNB->pusch_stats_mcs[i][(frame*10)+subframe] = -63;

    if ((eNB->ulsch[i]) &&
        (eNB->ulsch[i]->rnti>0) &&
        (eNB->ulsch[i]->harq_processes[harq_pid]->subframe_scheduling_flag==1)) {
      // UE is has ULSCH scheduling
      round = eNB->ulsch[i]->harq_processes[harq_pid]->round;
 
      for (int rb=0;
           rb<=eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb;
     rb++) {
  int rb2 = rb+eNB->ulsch[i]->harq_processes[harq_pid]->first_rb;
  eNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
      }


      if (eNB->ulsch[i]->Msg3_flag == 1) {
        LOG_D(PHY,"[eNB %d] frame %d, subframe %d: Scheduling ULSCH Reception for Msg3 in Sector %d\n",
              eNB->Mod_id,
              frame,
              subframe,
              eNB->UE_stats[i].sector);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_ENB_ULSCH_MSG3,1);
      } else {

        LOG_D(PHY,"[eNB %d] frame %d, subframe %d: Scheduling ULSCH Reception for UE %d Mode %s\n",
              eNB->Mod_id,
              frame,
              subframe,
              i,
              mode_string[eNB->UE_stats[i].mode]);
      }


      nPRS = fp->pusch_config_common.ul_ReferenceSignalsPUSCH.nPRS[subframe<<1];

      eNB->ulsch[i]->cyclicShift = (eNB->ulsch[i]->harq_processes[harq_pid]->n_DMRS2 + fp->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift +
            nPRS)%12;

      if (fp->frame_type == FDD ) {
        int sf = (subframe<4) ? (subframe+6) : (subframe-4);

        if (eNB->ndlsch[i][0]->subframe_tx[sf]>0) { // we have downlink transmission
          eNB->ulsch[i]->harq_processes[harq_pid]->O_ACK = 1;
        } else {
          eNB->ulsch[i]->harq_processes[harq_pid]->O_ACK = 0;
        }
      }

      LOG_D(PHY,
            "[eNB %d][PUSCH %d] Frame %d Subframe %d Demodulating PUSCH: dci_alloc %d, rar_alloc %d, round %d, first_rb %d, nb_rb %d, mcs %d, TBS %d, rv %d, cyclic_shift %d (n_DMRS2 %d, cyclicShift_common %d, nprs %d), O_ACK %d \n",
            eNB->Mod_id,harq_pid,frame,subframe,
            eNB->ulsch[i]->harq_processes[harq_pid]->dci_alloc,
            eNB->ulsch[i]->harq_processes[harq_pid]->rar_alloc,
            eNB->ulsch[i]->harq_processes[harq_pid]->round,
            eNB->ulsch[i]->harq_processes[harq_pid]->first_rb,
            eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb,
            eNB->ulsch[i]->harq_processes[harq_pid]->mcs,
            eNB->ulsch[i]->harq_processes[harq_pid]->TBS,
            eNB->ulsch[i]->harq_processes[harq_pid]->rvidx,
            eNB->ulsch[i]->cyclicShift,
            eNB->ulsch[i]->harq_processes[harq_pid]->n_DMRS2,
            fp->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift,
            nPRS,
            eNB->ulsch[i]->harq_processes[harq_pid]->O_ACK);
      eNB->pusch_stats_rb[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb;
      eNB->pusch_stats_round[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->round;
      eNB->pusch_stats_mcs[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->mcs;
      start_meas(&eNB->ulsch_demodulation_stats);

      if (eNB->abstraction_flag==0) {
        rx_ulsch(eNB,proc,
                 eNB->UE_stats[i].sector,  // this is the effective sector id
                 i,
                 eNB->ulsch,
                 0);
      }

#ifdef PHY_ABSTRACTION
      else {
        rx_ulsch_emul(eNB,proc,
                      eNB->UE_stats[i].sector,  // this is the effective sector id
                      i);
      }

#endif
      stop_meas(&eNB->ulsch_demodulation_stats);


      start_meas(&eNB->ulsch_decoding_stats);

      if (eNB->abstraction_flag == 0) {
        ret = ulsch_decoding(eNB,proc,
                             i,
                             0, // control_only_flag
                             eNB->ulsch[i]->harq_processes[harq_pid]->V_UL_DAI,
           eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb>20 ? 1 : 0);
      }

#ifdef PHY_ABSTRACTION
      else {
        ret = ulsch_decoding_emul(eNB,
          proc,
                                  i,
                                  &rnti);
      }

#endif
      stop_meas(&eNB->ulsch_decoding_stats);

      LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d RNTI %x RX power (%d,%d) RSSI (%d,%d) N0 (%d,%d) dB ACK (%d,%d), decoding iter %d\n",
            eNB->Mod_id,harq_pid,
            frame,subframe,
            eNB->ulsch[i]->rnti,
            dB_fixed(eNB->pusch_vars[i]->ulsch_power[0]),
            dB_fixed(eNB->pusch_vars[i]->ulsch_power[1]),
            eNB->UE_stats[i].UL_rssi[0],
            eNB->UE_stats[i].UL_rssi[1],
            eNB->measurements->n0_power_dB[0],
            eNB->measurements->n0_power_dB[1],
            eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[0],
            eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[1],
            ret);

      //compute the expected ULSCH RX power (for the stats)
      eNB->ulsch[(uint32_t)i]->harq_processes[harq_pid]->delta_TF =
        get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 0); // 0 means bw_factor is not considered

      eNB->UE_stats[i].nulsch_decoding_attempts[harq_pid][eNB->ulsch[i]->harq_processes[harq_pid]->round]++;
#ifdef DEBUG_PHY_PROC
      LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d UE %d harq_pid %d Clearing subframe_scheduling_flag\n",
            eNB->Mod_id,harq_pid,frame,subframe,i,harq_pid);
#endif
      eNB->ulsch[i]->harq_processes[harq_pid]->subframe_scheduling_flag=0;

      if (eNB->ulsch[i]->harq_processes[harq_pid]->cqi_crc_status == 1) {
#ifdef DEBUG_PHY_PROC
        //if (((frame%10) == 0) || (frame < 50))
        print_CQI(eNB->ulsch[i]->harq_processes[harq_pid]->o,eNB->ulsch[i]->harq_processes[harq_pid]->uci_format,0,fp->N_RB_DL);
#endif
        extract_CQI_NB_IoT(eNB->ulsch[i]->harq_processes[harq_pid]->o,
                    eNB->ulsch[i]->harq_processes[harq_pid]->uci_format,
                    &eNB->UE_stats[i],
                    fp->N_RB_DL,
                    &rnti, &access_mode);
        eNB->UE_stats[i].rank = eNB->ulsch[i]->harq_processes[harq_pid]->o_RI[0];

      }

      if (eNB->ulsch[i]->Msg3_flag == 1)
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_ENB_ULSCH_MSG3,0);

      if (ret == (1+MAX_TURBO_ITERATIONS)) {
        T(T_ENB_PHY_ULSCH_UE_NACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(i), T_INT(eNB->ulsch[i]->rnti),
          T_INT(harq_pid));

        eNB->UE_stats[i].ulsch_round_errors[harq_pid][eNB->ulsch[i]->harq_processes[harq_pid]->round]++;
        eNB->ulsch[i]->harq_processes[harq_pid]->phich_active = 1;
        eNB->ulsch[i]->harq_processes[harq_pid]->phich_ACK = 0;
        eNB->ulsch[i]->harq_processes[harq_pid]->round++;

        LOG_D(PHY,"[eNB][PUSCH %d] Increasing to round %d\n",harq_pid,eNB->ulsch[i]->harq_processes[harq_pid]->round);

        if (eNB->ulsch[i]->Msg3_flag == 1) {

          LOG_D(PHY,"[eNB %d/%d][RAPROC] frame %d, subframe %d, UE %d: Error receiving ULSCH (Msg3), round %d/%d\n",
                eNB->Mod_id,
                eNB->CC_id,
                frame,subframe, i,
                eNB->ulsch[i]->harq_processes[harq_pid]->round-1,
                fp->maxHARQ_Msg3Tx-1);
    /*dump_ulsch(eNB,proc,i);
      exit(-1);*/

    LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d RNTI %x RX power (%d,%d) RSSI (%d,%d) N0 (%d,%d) dB ACK (%d,%d), decoding iter %d\n",
    eNB->Mod_id,harq_pid,
    frame,subframe,
    eNB->ulsch[i]->rnti,
    dB_fixed(eNB->pusch_vars[i]->ulsch_power[0]),
    dB_fixed(eNB->pusch_vars[i]->ulsch_power[1]),
    eNB->UE_stats[i].UL_rssi[0],
    eNB->UE_stats[i].UL_rssi[1],
    eNB->measurements->n0_power_dB[0],
    eNB->measurements->n0_power_dB[1],
    eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[0],
    eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[1],
    ret);

          if (eNB->ulsch[i]->harq_processes[harq_pid]->round ==
              fp->maxHARQ_Msg3Tx) {
            LOG_D(PHY,"[eNB %d][RAPROC] maxHARQ_Msg3Tx reached, abandoning RA procedure for UE %d\n",
                  eNB->Mod_id, i);
            eNB->UE_stats[i].mode = PRACH;
      if (eNB->mac_enabled==1) {
        cancel_ra_proc(eNB->Mod_id,
          eNB->CC_id,
          frame,
          eNB->UE_stats[i].crnti);
      }
            mac_phy_remove_ue(eNB->Mod_id,eNB->UE_stats[i].crnti);

            eNB->ulsch[(uint32_t)i]->Msg3_active = 0;
            //eNB->ulsch[i]->harq_processes[harq_pid]->phich_active = 0;

          } else {
            // activate retransmission for Msg3 (signalled to UE PHY by PHICH (not MAC/DCI)
            eNB->ulsch[(uint32_t)i]->Msg3_active = 1;

            get_Msg3_alloc_ret(fp,
                               subframe,
                               frame,
                               &eNB->ulsch[i]->Msg3_frame,
                               &eNB->ulsch[i]->Msg3_subframe);

            set_msg3_subframe(eNB->Mod_id, eNB->CC_id, frame, subframe, eNB->ulsch[i]->rnti,
                                         eNB->ulsch[i]->Msg3_frame, eNB->ulsch[i]->Msg3_subframe);

            T(T_ENB_PHY_MSG3_ALLOCATION, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe),
              T_INT(i), T_INT(eNB->ulsch[i]->rnti), T_INT(0 /* 0 is for retransmission*/),
              T_INT(eNB->ulsch[i]->Msg3_frame), T_INT(eNB->ulsch[i]->Msg3_subframe));
          }
          LOG_D(PHY,"[eNB] Frame %d, Subframe %d: Msg3 in error, i = %d \n", frame,subframe,i);
        } // This is Msg3 error

        else { //normal ULSCH
          LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d UE %d Error receiving ULSCH, round %d/%d (ACK %d,%d)\n",
                eNB->Mod_id,harq_pid,
                frame,subframe, i,
                eNB->ulsch[i]->harq_processes[harq_pid]->round-1,
                eNB->ulsch[i]->Mlimit,
                eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[0],
                eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[1]);

#if defined(MESSAGE_CHART_GENERATOR_PHY)
          MSC_LOG_RX_DISCARDED_MESSAGE(
               MSC_PHY_ENB,MSC_PHY_UE,
               NULL,0,
               "%05u:%02u ULSCH received rnti %x harq id %u round %d",
               frame,subframe,
               eNB->ulsch[i]->rnti,harq_pid,
               eNB->ulsch[i]->harq_processes[harq_pid]->round-1
               );
#endif

          if (eNB->ulsch[i]->harq_processes[harq_pid]->round== eNB->ulsch[i]->Mlimit) {
            LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d UE %d ULSCH Mlimit %d reached\n",
                  eNB->Mod_id,harq_pid,
                  frame,subframe, i,
                  eNB->ulsch[i]->Mlimit);

            eNB->ulsch[i]->harq_processes[harq_pid]->round=0;
            eNB->ulsch[i]->harq_processes[harq_pid]->phich_active=0;
            eNB->UE_stats[i].ulsch_errors[harq_pid]++;
            eNB->UE_stats[i].ulsch_consecutive_errors++;

     /*if (eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb > 20) {
    dump_ulsch(eNB,proc,i);
    exit(-1);
           }*/
      // indicate error to MAC
      if (eNB->mac_enabled == 1)
        rx_sdu(eNB->Mod_id,
        eNB->CC_id,
        frame,subframe,
        eNB->ulsch[i]->rnti,
        NULL,
        0,
        harq_pid,
        &eNB->ulsch[i]->Msg3_flag);
          }
        }
      }  // ulsch in error
      else {



        T(T_ENB_PHY_ULSCH_UE_ACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(i), T_INT(eNB->ulsch[i]->rnti),
          T_INT(harq_pid));

        if (eNB->ulsch[i]->Msg3_flag == 1) {
    LOG_D(PHY,"[eNB %d][PUSCH %d] Frame %d subframe %d ULSCH received, setting round to 0, PHICH ACK\n",
    eNB->Mod_id,harq_pid,
    frame,subframe);
    LOG_D(PHY,"[eNB %d][PUSCH %d] frame %d subframe %d RNTI %x RX power (%d,%d) RSSI (%d,%d) N0 (%d,%d) dB ACK (%d,%d), decoding iter %d\n",
    eNB->Mod_id,harq_pid,
    frame,subframe,
    eNB->ulsch[i]->rnti,
    dB_fixed(eNB->pusch_vars[i]->ulsch_power[0]),
    dB_fixed(eNB->pusch_vars[i]->ulsch_power[1]),
    eNB->UE_stats[i].UL_rssi[0],
    eNB->UE_stats[i].UL_rssi[1],
    eNB->measurements->n0_power_dB[0],
    eNB->measurements->n0_power_dB[1],
    eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[0],
    eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[1],
    ret);
  }
#if defined(MESSAGE_CHART_GENERATOR_PHY)
        MSC_LOG_RX_MESSAGE(
         MSC_PHY_ENB,MSC_PHY_UE,
         NULL,0,
         "%05u:%02u ULSCH received rnti %x harq id %u",
         frame,subframe,
         eNB->ulsch[i]->rnti,harq_pid
         );
#endif
        for (j=0; j<fp->nb_antennas_rx; j++)
          //this is the RSSI per RB
          eNB->UE_stats[i].UL_rssi[j] =
      
            dB_fixed(eNB->pusch_vars[i]->ulsch_power[j]*
                     (eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb*12)/
                     fp->ofdm_symbol_size) -
            eNB->rx_total_gain_dB -
            hundred_times_log10_NPRB[eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb-1]/100 -
            get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 0)/100;
      
        eNB->ulsch[i]->harq_processes[harq_pid]->phich_active = 1;
        eNB->ulsch[i]->harq_processes[harq_pid]->phich_ACK = 1;
        eNB->ulsch[i]->harq_processes[harq_pid]->round = 0;
        eNB->UE_stats[i].ulsch_consecutive_errors = 0;

        if (eNB->ulsch[i]->Msg3_flag == 1) {
    if (eNB->mac_enabled==1) {

      LOG_I(PHY,"[eNB %d][RAPROC] Frame %d Terminating ra_proc for harq %d, UE %d\n",
      eNB->Mod_id,
      frame,harq_pid,i);
      if (eNB->mac_enabled)
        rx_sdu(eNB->Mod_id,
        eNB->CC_id,
        frame,subframe,
        eNB->ulsch[i]->rnti,
        eNB->ulsch[i]->harq_processes[harq_pid]->b,
        eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3,
        harq_pid,
        &eNB->ulsch[i]->Msg3_flag);
      
      // one-shot msg3 detection by MAC: empty PDU (e.g. CRNTI)
      if (eNB->ulsch[i]->Msg3_flag == 0 ) {
        eNB->UE_stats[i].mode = PRACH;
        cancel_ra_proc(eNB->Mod_id,
          eNB->CC_id,
          frame,
          eNB->UE_stats[i].crnti);
        mac_phy_remove_ue(eNB->Mod_id,eNB->UE_stats[i].crnti);
        eNB->ulsch[(uint32_t)i]->Msg3_active = 0;
      } // Msg3_flag == 0
      
    } // mac_enabled==1

          eNB->UE_stats[i].mode = PUSCH;
          eNB->ulsch[i]->Msg3_flag = 0;

    LOG_D(PHY,"[eNB %d][RAPROC] Frame %d : RX Subframe %d Setting UE %d mode to PUSCH\n",eNB->Mod_id,frame,subframe,i);

          for (k=0; k<8; k++) { //harq_processes
            for (j=0; j<eNB->ndlsch[i][0]->Mlimit; j++) {
              eNB->UE_stats[i].dlsch_NAK[k][j]=0;
              eNB->UE_stats[i].dlsch_ACK[k][j]=0;
              eNB->UE_stats[i].dlsch_trials[k][j]=0;
            }

            eNB->UE_stats[i].dlsch_l2_errors[k]=0;
            eNB->UE_stats[i].ulsch_errors[k]=0;
            eNB->UE_stats[i].ulsch_consecutive_errors=0;

            for (j=0; j<eNB->ulsch[i]->Mlimit; j++) {
              eNB->UE_stats[i].nulsch_decoding_attempts[k][j]=0;
              eNB->UE_stats[i].ulsch_decoding_attempts_last[k][j]=0;
              eNB->UE_stats[i].ulsch_round_errors[k][j]=0;
              eNB->UE_stats[i].ulsch_round_fer[k][j]=0;
            }
          }

          eNB->UE_stats[i].dlsch_sliding_cnt=0;
          eNB->UE_stats[i].dlsch_NAK_round0=0;
          eNB->UE_stats[i].dlsch_mcs_offset=0;
        } // Msg3_flag==1
  else {  // Msg3_flag == 0

#ifdef DEBUG_PHY_PROC
#ifdef DEBUG_ULSCH
          LOG_D(PHY,"[eNB] Frame %d, Subframe %d : ULSCH SDU (RX harq_pid %d) %d bytes:",frame,subframe,
                harq_pid,eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3);

          for (j=0; j<eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3; j++)
            LOG_T(PHY,"%x.",eNB->ulsch[i]->harq_processes[harq_pid]->b[j]);

          LOG_T(PHY,"\n");
#endif
#endif

    if (eNB->mac_enabled==1) {

      rx_sdu(eNB->Mod_id,
            eNB->CC_id,
            frame,subframe,
            eNB->ulsch[i]->rnti,
            eNB->ulsch[i]->harq_processes[harq_pid]->b,
            eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3,
            harq_pid,
            NULL);

#ifdef LOCALIZATION
      start_meas(&eNB->localization_stats);
      aggregate_eNB_UE_localization_stats(eNB,
            i,
            frame,
            subframe,
            get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 1)/100);
      stop_meas(&eNB->localization_stats);
#endif
      
    } // mac_enabled==1
        } // Msg3_flag == 0

        // estimate timing advance for MAC
        if (eNB->abstraction_flag == 0) {
          sync_pos = lte_est_timing_advance_pusch(eNB,i);
          eNB->UE_stats[i].timing_advance_update = sync_pos - fp->nb_prefix_samples/4; //to check
        }

#ifdef DEBUG_PHY_PROC
        LOG_D(PHY,"[eNB %d] frame %d, subframe %d: user %d: timing advance = %d\n",
              eNB->Mod_id,
              frame, subframe,
              i,
              eNB->UE_stats[i].timing_advance_update);
#endif


      }  // ulsch not in error

      // process HARQ feedback
#ifdef DEBUG_PHY_PROC
      LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d, Processing HARQ feedback for UE %d (after PUSCH)\n",eNB->Mod_id,
            eNB->ndlsch[i][0]->rnti,
            frame,subframe,
            i);
#endif
      process_HARQ_feedback(i,
                            eNB,proc,
                            1, // pusch_flag
                            0,
                            0,
                            0);

#ifdef DEBUG_PHY_PROC
      LOG_D(PHY,"[eNB %d] Frame %d subframe %d, sect %d: received ULSCH harq_pid %d for UE %d, ret = %d, CQI CRC Status %d, ACK %d,%d, ulsch_errors %d/%d\n",
            eNB->Mod_id,frame,subframe,
            eNB->UE_stats[i].sector,
            harq_pid,
            i,
            ret,
            eNB->ulsch[i]->harq_processes[harq_pid]->cqi_crc_status,
            eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[0],
            eNB->ulsch[i]->harq_processes[harq_pid]->o_ACK[1],
            eNB->UE_stats[i].ulsch_errors[harq_pid],
            eNB->UE_stats[i].nulsch_decoding_attempts[harq_pid][0]);
#endif
      
      // dump stats to VCD
      if (i==0) {
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_MCS0+harq_pid,eNB->pusch_stats_mcs[0][(frame*10)+subframe]);
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_RB0+harq_pid,eNB->pusch_stats_rb[0][(frame*10)+subframe]);
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_ROUND0+harq_pid,eNB->pusch_stats_round[0][(frame*10)+subframe]);
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_RSSI0+harq_pid,dB_fixed(eNB->pusch_vars[0]->ulsch_power[0]));
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_RES0+harq_pid,ret);
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME(VCD_SIGNAL_DUMPER_VARIABLES_UE0_SFN0+harq_pid,(frame*10)+subframe);
      }
    } // ulsch[0] && ulsch[0]->rnti>0 && ulsch[0]->subframe_scheduling_flag == 1


    // update ULSCH statistics for tracing
    if ((frame % 100 == 0) && (subframe == 4)) {
      for (harq_idx=0; harq_idx<8; harq_idx++) {
        for (round=0; round<eNB->ulsch[i]->Mlimit; round++) {
          if ((eNB->UE_stats[i].nulsch_decoding_attempts[harq_idx][round] -
               eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round]) != 0) {
            eNB->UE_stats[i].ulsch_round_fer[harq_idx][round] =
              (100*(eNB->UE_stats[i].ulsch_round_errors[harq_idx][round] -
                    eNB->UE_stats[i].ulsch_round_errors_last[harq_idx][round]))/
              (eNB->UE_stats[i].nulsch_decoding_attempts[harq_idx][round] -
               eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round]);
          } else {
            eNB->UE_stats[i].ulsch_round_fer[harq_idx][round] = 0;
          }

          eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round] =
            eNB->UE_stats[i].nulsch_decoding_attempts[harq_idx][round];
          eNB->UE_stats[i].ulsch_round_errors_last[harq_idx][round] =
            eNB->UE_stats[i].ulsch_round_errors[harq_idx][round];
        }
      }
    }

    if ((frame % 100 == 0) && (subframe==4)) {
      eNB->UE_stats[i].dlsch_bitrate = (eNB->UE_stats[i].total_TBS -
          eNB->UE_stats[i].total_TBS_last);

      eNB->UE_stats[i].total_TBS_last = eNB->UE_stats[i].total_TBS;
    }

    // CBA (non-LTE)
    cba_procedures(eNB,proc,i,harq_pid);
  } // loop i=0 ... NUMBER_OF_UE_MAX-1

  if (eNB->abstraction_flag == 0) {
    lte_eNB_I0_measurements(eNB,
          subframe,
          0,
          eNB->first_run_I0_measurements);
    eNB->first_run_I0_measurements = 0;
  }

#ifdef PHY_ABSTRACTION
  else {
    lte_eNB_I0_measurements_emul(eNB,
         0);
  }

#endif
  //}

#ifdef EMOS
  phy_procedures_emos_eNB_RX(subframe,eNB);
#endif

#if defined(FLEXRAN_AGENT_SB_IF)
#ifndef DISABLE_SF_TRIGGER
  //Send subframe trigger to the controller
  if (mac_agent_registered[eNB->Mod_id]) {
    agent_mac_xface[eNB->Mod_id]->flexran_agent_send_sf_trigger(eNB->Mod_id);
  }
#endif
#endif

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_RX_UESPEC+offset, 0 );

  stop_meas(&eNB->phy_proc_rx);

}






