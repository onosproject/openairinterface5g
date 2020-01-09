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

#include "PHY/defs.h"
#include "PHY/defs_NB_IoT.h"
#include "PHY/extern.h"
#include "PHY/LTE_ESTIMATION/defs_NB_IoT.h"
#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
#include "PHY/LTE_TRANSPORT/proto_NB_IoT.h"
//#include "PHY/extern_NB_IoT.h" //where we get the global Sched_Rsp_t structure filled
//#include "SCHED/defs.h"
#include "SCHED/extern_NB_IoT.h"
//#include "PHY/LTE_TRANSPORT/if4_tools.h"
//#include "PHY/LTE_TRANSPORT/if5_tools.h"
#include "RRC/LITE/proto_NB_IoT.h"
#include "SIMULATION/TOOLS/defs.h"  // purpose: included for taus() function
//#ifdef EMOS
//#include "SCHED/phy_procedures_emos.h"
//#endif

// for NB-IoT
#include "SCHED/defs_NB_IoT.h"
#include "openair2/RRC/LITE/proto_NB_IoT.h"
#include "openair2/RRC/LITE/extern_NB_IoT.h"
#include "RRC/LITE/MESSAGES/asn1_msg_NB_IoT.h"
//#define DEBUG_PHY_PROC (Already defined in cmake)
//#define DEBUG_ULSCH

//#include "LAYER2/MAC/extern.h"
//#include "LAYER2/MAC/defs.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "assertions.h"
#include "msc.h"

#include <time.h>

#if defined(ENABLE_ITTI)
#   include "intertask_interface.h"
#endif

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
* It generates all the downlink message : NPBCH, NSSS, NPSS, NRS, NPDCCH and NPDSCH
*
*/
void NB_IoT_TX_procedure(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc) 
{
  LTE_DL_FRAME_PARMS   *fp       =  &eNB->frame_parms;
  NB_IoT_eNB_NPBCH_t   *broadcast_str = &eNB->npbch;
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
       	  //printf("Going to generate_NDLSCH_NB_IoT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	  generate_NDLSCH_NB_IoT(eNB,
                                 data,
                                 txdataF,
                                 AMP,
                                 fp,
                                 frame,
                                 subframe,
                                 RB_IoT_ID,
                                 release_v13_5_0);
          //printf("Finish doing generate_NDLSCH_NB_IoT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
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
      LOG_D(PHY,"Hyper_frame is %d",hyper_frame);
      if(proc->HFN==1023)
      {             
           proc->HFN=0;
      }else{ 
           proc->HFN++;
           LOG_D(PHY,"Update HFN:%d when frame:%d subframe:%d\n",proc->HFN,proc->frame_rx,proc->subframe_rx);
      }
  }

  
}

void phy_procedures_eNB_uespec_RX_NB_IoT(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc) //UL_IND_NB_IoT_t *UL_INFO)
{
  //RX processing for ue-specific resources (i
  npusch_procedures(eNB,proc);

     
}

/////Generate eNB ndlsch params for NB-IoT from the NPDCCH PDU of the DCI, modify the input to the Sched Rsp variable////
void generate_eNB_dlsch_params_NB_IoT(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t * proc,nfapi_dl_config_request_pdu_t *dl_config_pdu) 
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
           
          LOG_I(PHY,"Generating pdcch params for DCIN1 RAR and packing DCI\n");
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

        //printf("PHY_vars_eNB_g[0][0]->ndlsch_RAR->rnti = %d\n",PHY_vars_eNB_g[0][0]->ndlsch_RAR->rnti);
          //eNB->dlsch_ra_NB->nCCE[subframe] = eNB->DCI_pdu->dci_alloc.firstCCE;
        }
      else
        { //managing data
        LOG_I(PHY,"Handling the DCI for ue-spec data or MSG4!\n");
        // Temp: Add UE id when Msg4 trigger
        eNB->ndlsch[0]= (NB_IoT_eNB_NDLSCH_t*) malloc(sizeof(NB_IoT_eNB_NDLSCH_t));
        eNB->ndlsch[0]->harq_process = (NB_IoT_DL_eNB_HARQ_t*)malloc(sizeof(NB_IoT_DL_eNB_HARQ_t));
        eNB->ndlsch[0]->rnti=dl_config_pdu->npdcch_pdu.npdcch_pdu_rel13.rnti; 
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



void generate_eNB_ulsch_params_NB_IoT(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,nfapi_hi_dci0_request_pdu_t *hi_dci0_pdu) {

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





extern int oai_exit;

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



uint32_t rx_nprach_NB_IoT(PHY_VARS_eNB *eNB, int frame, uint8_t subframe, uint16_t *rnti, uint16_t *preamble_index, uint16_t *timing_advance) {

  uint32_t estimated_TA; 
  //int frame,frame_mod;    // subframe,
 // subframe = eNB->proc.subframe_prach; 
 // frame = eNB->proc.frame_prach;
    estimated_TA = process_nprach_NB_IoT(eNB,frame,subframe,rnti,preamble_index,timing_advance);
    //printf("estim = %i\n",estimated_TA);
 // }
  return estimated_TA;
}


void fill_crc_indication_NB_IoT(PHY_VARS_eNB *eNB,int UE_id,int frame,int subframe,uint8_t decode_flag) {


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

void fill_rx_indication_NB_IoT(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,uint8_t data_or_control, uint8_t decode_flag)
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
            pdu->rx_indication_rel8.length         = eNB->ulsch_NB_IoT[0]->harq_process->TBS; //eNB->ulsch_NB_IoT[0]->harq_process->TBS>>3;
            pdu->data                              = eNB->ulsch_NB_IoT[0]->harq_process->b;

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



void npusch_procedures(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc)
{
  
  uint32_t i;
  NB_IoT_eNB_NULSCH_t *nulsch;
  nulsch = eNB->ulsch_NB_IoT[0];

  const int rx_subframe   =   proc->subframe_rx;
  const int rx_frame      =   proc->frame_rx;

  int   RB_IoT_ID         = 22;
  for (i=0; i<1; i++)
  {
      // if eNB is ready to receive UL data 
      // define a flag to trigger on or off the decoding process
     rx_ulsch_Gen_NB_IoT(eNB,
                           proc,
                           0,                         // this is the effective sector id
                           0,
                           RB_IoT_ID,    // 22 , to be included in // to be replaced by NB_IoT_start ??
                           rx_subframe,  // first received subframe 
                           rx_frame);     // first received frame
   }  // for UE loop

}
