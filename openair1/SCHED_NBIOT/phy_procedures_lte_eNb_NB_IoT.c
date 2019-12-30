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

/*! \file phy_procedures_lte_eNB_NB_IoT.c
 * \brief Implementation of eNB procedures from 36.213 NB-IoT R13 specifications
 * \author M. KANJ, Nick Ho
 * \date 2019
 * \version 1.0
 * \company B-COM, NTUST
 * \email:
 * \note
 * \warning
 */

#include "PHY/defs_eNB.h"
#include "PHY/defs_L1_NB_IoT.h"
#include "PHY/phy_extern.h"
#include "PHY/defs_common.h"
#include "PHY/NBIoT_ESTIMATION/defs_NB_IoT.h"
#include "PHY/NBIoT_TRANSPORT/defs_NB_IoT.h"
#include "PHY/NBIoT_TRANSPORT/proto_NB_IoT.h"
//#include "PHY/extern_NB_IoT.h" //where we get the global Sched_Rsp_t structure filled
//#include "SCHED/defs.h"
#include "SCHED_NBIOT/sched_common_extern_NB_IoT.h"
#include "SIMULATION/TOOLS/sim.h"  // purpose: included for taus() function

// for NB-IoT
#include "SCHED_NBIOT/defs_NB_IoT.h"
#include "openair2/RRC/NBIOT/proto_NB_IoT.h"
#include "openair2/RRC/NBIOT/extern_NB_IoT.h"
//#include "RRC/LITE/MESSAGES/asn1_msg_NB_IoT.h"

//#include "UTIL/LOG/log.h"
//#include "UTIL/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "assertions.h"
#include "msc.h"

#include <time.h>

#if defined(ENABLE_ITTI)
#   include "intertask_interface.h"
#endif

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
*  Actually phy_procedures_eNB_TX here
*/
void common_signal_procedures_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc) 
{
  NB_IoT_DL_FRAME_PARMS   *fp       =  &eNB->frame_parms_NB_IoT;
  NB_IoT_eNB_NPBCH_t   *broadcast_str = eNB->npbch;
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


void npusch_procedures(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc)
{
  
  uint32_t i;
  //NB_IoT_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  NB_IoT_eNB_NULSCH_t *nulsch;
  NB_IoT_UL_eNB_HARQ_t *nulsch_harq;
  nulsch = eNB->ulsch_NB_IoT[0];
  nulsch_harq = nulsch->harq_process;

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

void phy_procedures_eNB_uespec_RX_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc) //UL_IND_NB_IoT_t *UL_INFO)
{
  //RX processing for ue-specific resources (i
  //NB_IoT_DL_FRAME_PARMS     *fp=&eNB->frame_parms_NB_IoT;

  const int subframe    =   proc->subframe_rx;
  const int frame       =   proc->frame_rx;


  npusch_procedures(eNB,proc);


  //pthread_mutex_lock(&eNB->UL_INFO_mutex);

  // Fix me here, these should be locked
  //eNB->UL_INFO.RX_NPUSCH.number_of_pdus  = 0;
 // eNB->UL_INFO.crc_ind.number_of_crcs = 0;

 // pthread_mutex_unlock(&eNB->UL_INFO_mutex);
 // if (nfapi_mode == 0 || nfapi_mode == 1) { // If PNF or monolithic
     
  //}          

     
}

// the function called by l1 IF-Module 

void generate_eNB_dlsch_params_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t * proc,nfapi_dl_config_request_pdu_t *dl_config_pdu)
{
  int                      UE_id         =  -1;
  NB_IoT_DL_FRAME_PARMS    *fp           =  &eNB->frame_parms_NB_IoT;
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
                       //int num_pdcch_symbols,            //(BCOM says are not needed
                       uint8_t                  *pdu
                       )
{
  int                     frame                   =   proc->frame_tx;
  int                     subframe                =   proc->subframe_tx;
  NB_IoT_DL_eNB_HARQ_t    *ndlsch_harq            =   ndlsch->harq_process;
  int                     input_buffer_length     =   ndlsch_harq->TBS/8;         // get in byte //the TBS is set in generate_dlsch_param
  //NB_IoT_DL_FRAME_PARMS   *fp                     =   &eNB->frame_parms_NB_IoT;
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
    //      num_pdcch_symbols,
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
      //         num_pdcch_symbols,
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
    //         num_pdcch_symbols,
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


extern volatile int oai_exit;

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

/*
 * This function is triggered by the schedule_response
 * (the frequency at which is transmitted to the PHY depends on the MAC scheduler implementation)
 * (in OAI in principle is every subframe)
 */

void phy_procedures_eNB_TX_NB_IoT(PHY_VARS_eNB_NB_IoT     *eNB,
                                  eNB_rxtx_proc_NB_IoT_t  *proc,
                                  int                     do_meas)
{
  int                    frame           = proc->frame_tx;
  int                    subframe        = proc->subframe_tx;
  uint32_t               aa;
  DCI_PDU_NB_IoT         *dci_pdu        = eNB->DCI_pdu;
  NB_IoT_DL_FRAME_PARMS  *fp             = &eNB->frame_parms_NB_IoT;
  int8_t                 UE_id           = 0;
  //int                    **txdataF       = eNB->common_vars.txdataF[0];
  uint32_t               sib1_startFrame = -1;
  //NB_IoT_eNB_NPDCCH_t*npdcch;

  if(do_meas == 1)
    //start_meas_NB_IoT(&eNB->phy_proc_tx);


  /*the original scheduler "eNB_dlsch_ulsch_scheduler" now is no more done here but is triggered directly from UL_Indication (IF-Module Function)*/

  // clear the transmit data array for the current subframe
  for (aa=0; aa<fp->nb_antenna_ports_eNB; aa++) 
    {      
      memset(&eNB->common_vars.txdataF[0][aa][subframe*fp->ofdm_symbol_size*(fp->symbols_per_tti)],
                  0,fp->ofdm_symbol_size*(fp->symbols_per_tti)*sizeof(int32_t));
    } 

  //generate NPSS/NSSS
 // common_signal_procedures_NB_IoT(eNB,proc);  // to uncomment after NB-IoT testing

    //Generate MIB
    if(subframe ==0 && (eNB->npbch != NULL))
     {
          if(eNB->npbch->pdu != NULL)
          {
            //BCOM function
            /*
             * -the function get the MIB pdu and schedule the transmission over the 64 radio frame
             * -need to check the subframe #0 (since encoding functions only check the frame)
             * this functions should be called every frame (the function will transmit the remaining part of MIB)
             * ( XXX Should check when the schedule_responce is transmitted by MAC scheduler)
             * RB-ID only for the case of in-band operation but should be always considered
             * (in stand alone i can put whatever the number)in other case consider the PRB index in the Table R&Shwartz pag 9
             *
             */
/*
            generate_npbch(eNB->npbch,
                           txdataF,
                           AMP,
                           fp,
                           eNB->npbch->pdu,
                           frame%64,
                           fp->NB_IoT_RB_ID);*/
                        
          }

          //In the last frame in which the MIB-NB should be transmitted after we point to NULL since maybe we stop MIB trasnmission
          //this should be in line with FAPI specs pag 94 (BCH procedure in Downlink 3.2.4.2 for NB-IoT)
          if(frame%64 == 63)
          {
            eNB->npbch->pdu = NULL;
          }
      }


    //Check for SIB1-NB transmission
    /*
     *
     * the function should be called for each frame
     * Parameters needed:
     * -sib1-NB pdu if new one (should be given by the MAC at the start of each SIB1-NB period)
     * -when start a new SIB1-NB repetition (sib1_rep_start)
     * -the frame number relative to the 16 continuous frame within a repetition (relative_sib1_frame) 1st, 2nd ...
     *
     * we check that the transmission should occurr in subframe #4
     *
     * consider that if at the start of the new SIB1-NB period the MAC will not send an NPDSCH for the SIB1-NB transmission then SIB1-NB will be not transmitted (pdu = NULL)
     *
     */
    if(subframe == 4 && eNB->ndlsch_SIB1 != NULL && eNB->ndlsch_SIB1->harq_process->status == ACTIVE_NB_IoT)
    {
      //check if current frame is for SIB1-NB transmission (if yes get the starting frame of SIB1-NB) and set the flag for the encoding
      sib1_startFrame = is_SIB1_NB_IoT(frame,
                                       (long)eNB->ndlsch_SIB1->harq_process->repetition_number,
                                       fp->Nid_cell,
                                       eNB->ndlsch_SIB1); //set the flags
                                   
      if(sib1_startFrame != -1 && eNB->ndlsch_SIB1->harq_process->pdu != NULL)
      {
         npdsch_procedures(eNB,
                           proc,
                           eNB->ndlsch_SIB1, //since we have no DCI for system information, this is filled directly when we receive the NDLSCH pdu from DL_CONFIG.request message
                           eNB->ndlsch_SIB1->harq_process->pdu);
      }

      //at the end of the period we put the PDU to NULL since we have to wait for the new one from the MAC for starting the next SIB1-NB transmission
      if((frame-sib1_startFrame)%256 == 255)
      {
          //whenever we will not receive a new sdu from MAC at the start of the next SIB1-NB period we prevent future SIB1-NB transmission (may just only of the two condition is necessary)
          eNB->ndlsch_SIB1->harq_process->status = DISABLED;
          eNB->ndlsch_SIB1->harq_process->pdu = NULL;
      }

    }


    //Check for SI transmission
    /*
     *Parameters needed:
     * -total number of subframes for the transmission (2-8) (inside the NDLSCH structure --> HARQ process -->resource_assignment)
     * XXX: in reality this flag is not needed because is enough to check if the PDU is NULL (continue the transmission) or not (new SI transmission)
     * -SI_start (inside ndlsch structure): flag for indicate the starting of the SI transmission within the SI window (new PDU is received by the MAC) otherwise the PHY continue to transmit
     *  what have in its buffer (so check the remaining encoded data continuously)
     *
     * SI transmission should not occurr in reserved subframes
     * subframe = 0 (MIB-NB)
     * subframe = 4 (SIB1-NB) but depends on the frame
     * subframe = 5 (NPSS)
     * subframe = 9 (NSSS) but depends on the frame (if is even)
     *
     * [This condition should be known by the MAC layer so it should trigger an DLSCH pdu only at proper instants]
     *
     * XXX Important: in the case the SI-window finish the PHY layer should have also being able to conclude all the SI transmission in time
     * (because this is managed by the MAC layer that stops transmitting the SDU to PHY in advance because is counting the remaining subframe for the transmission)
     *
     *
     *XXX important: set the flag HARQ process->status to DISABLE when PHY finished the SI-transmission over the 2 or 8 subframes
     *XXX important: whenever we enter for some error in the ndlsch_procedure with a pdu that is NULL but all the data of the SI have been transmitted (pdu_buffer_index = 0)
     *XXX  --> generate error
     *XXX: the npdlsch_procedure in this case should be only called when is triggered by the MAC schedule_response (use the status flag set by the schedule_response)
     *
     */

  if(eNB->ndlsch_SI->harq_process->status == ACTIVE_NB_IoT && (eNB->ndlsch_SIB1->harq_process->status != ACTIVE_NB_IoT || subframe != 4)) //condition on SIB1-NB
  {
      if(frame%2 == 0)//condition on NSSS (subframe 9 not available)
      {
        if(eNB->ndlsch_SI != NULL &&  subframe!= 0 && subframe != 5 && subframe != 9)
         {
          //check if the PDU != NULL will be done inside just for understanding if a new SI message need to be transmitted or not
          npdsch_procedures(eNB,
                            proc,
                            eNB->ndlsch_SI, //since we have no DCI for system information, this is filled directly when we receive the DL_CONFIG.request message
                            eNB->ndlsch_SI->harq_process->pdu);

          eNB->ndlsch_SI->harq_process->status = DISABLED_NB_IoT;
        }

       } else {//this frame not foresee the transmission of NSSS (subframe 9 is available)
      
              if(eNB->ndlsch_SI != NULL &&  subframe!= 0 && subframe != 5)
              {
                   npdsch_procedures(eNB,
                                     proc,
                                     eNB->ndlsch_SI, //since we have no DCI for system information, this is filled directly when we receive the DL_CONFIG.request message
                                     eNB->ndlsch_SI->harq_process->pdu);

                   eNB->ndlsch_SI->harq_process->status = DISABLED_NB_IoT;

              }
           }

  }

      ///check for RAR transmission
      if(eNB->ndlsch_ra != NULL && eNB->ndlsch_ra->active == 1 && (eNB->ndlsch_SIB1->harq_process->status != ACTIVE_NB_IoT || subframe != 4)) //condition on SIB1-NB
      {
        if(frame%2 == 0)//condition on NSSS (subframe 9 not available)
         {
          if(eNB->ndlsch_SI != NULL &&  subframe!= 0 && subframe != 5 && subframe != 9)
           {

            npdsch_procedures(eNB,
                              proc,
                              eNB->ndlsch_ra, //should be filled ?? (in the old implementation was filled when from DCI we generate_dlsch_params
                              eNB->ndlsch_ra->harq_process->pdu);

            //it should be activated only when we receive the proper DCIN1_RAR
            eNB->ndlsch_ra->active= 0;
           }
         }
      else //this frame not foresee the transmission of NSSS (subframe 9 is available)
      {
        if(eNB->ndlsch_SI != NULL &&  subframe!= 0 && subframe != 5)
           {
              npdsch_procedures(eNB,
                                proc,
                                eNB->ndlsch_ra, //should be filled ?? (in the old implementation was filled when from DCI we generate_dlsch_params
                                eNB->ndlsch_ra->harq_process->pdu);

              //it should be activated only when we receive the proper DCIN1_RAR
              eNB->ndlsch_ra->active= 0; // maybe this is already done inside the ndlsch_procedure

           }
      }
      }


      //check for UE specific transmission
      /*
       * Delays between DCI transmission and NDLSCH transmission are taken in consideration by the MAC scheduler by sending in the proper subframe the scheduler_response
       * (TS 36.213 ch 16.4.1: DCI format N1, N2, ending in subframe n intended for the UE, the UE shall decode, starting from subframe n+5 DL subframe,
       * the corresponding NPDSCH transmission over the N consecutive NB/IoT DL subframes according to NPDCCH information)
       * Transmission over more subframe and Repetitions are managed directly by the PHY layer
       * We should have only 1 ue-specific ndlsch structure active at each time (active flag is set = 1 only at the corresponding NDLSCH pdu reception and not at the DCI time
       * (NDLSCH transmission should be compliant with the FAPI procedure Figure 3-49)
       *
       * XXX how are managed the transmission and repetitions over the NPDSCH:
       * -repetitions over the NPDSCH channel are defined inside the DCI
       * -need to know the repetition number R (see specs)
       * -repetition are made following a pattern rule (e.g. 00, 11 ...) (see specs)
       * --whenever R>4 then repetition pattern rule changes
       * -possibility to have DL-GAP (OPTIONAL) otherwise no gap in DCI transmission
       *
       * XXX During repetitions of DCI or NDLSCH we receive no schedule_response form MAC
       *
       */

      //this should give only 1 result (since only 1 ndlsch procedure is activated at once) so we brak after the transmission
      for (UE_id = 0; UE_id < NUMBER_OF_UE_MAX_NB_IoT; UE_id++)
      {

        if(eNB->ndlsch[(uint8_t)UE_id] != NULL && eNB->ndlsch[(uint8_t)UE_id]->active == 1 && (eNB->ndlsch_SIB1->harq_process->status != ACTIVE_NB_IoT || subframe != 4)) //condition on sib1-NB
        {
          if(frame%2 == 0)//condition on NSSS (subframe 9 not available)
            {
              if( subframe!= 0 && subframe != 5 && subframe != 9)
               {
                npdsch_procedures(eNB,
                                  proc,
                                  eNB->ndlsch[(uint8_t)UE_id],
                                  eNB->ndlsch[(uint8_t)UE_id]->harq_process->pdu);
                break;
                 }
               }
          else //this frame not foresee the transmission of NSSS (subframe 9 is available)
          {
            if( subframe!= 0 && subframe != 5)
               {
                npdsch_procedures(eNB,
                                  proc,
                                  eNB->ndlsch[(uint8_t)UE_id],
                                  eNB->ndlsch[(uint8_t)UE_id]->harq_process->pdu);
                break;

               }
          }
        }


      }


      //no dedicated phy config


      /*If we have DCI to generate do it now
       *
       * DCI in NB-IoT are transmitted over NPDCCH search spaces as described in TS 36.213 ch 16.6
       *
       * Don-t care about the concept of search space since will be managed by the MAC.
       * MAC also evaluate the starting position of NPDCCH transmission and will send the corresponding scheduling_response
       *
       *
       * The PHY layer should evaluate R (repetitions of DCI) based on:
       *  -L (aggregation level) --> inside the NPDCCH PDU
       *  -Rmax
       *  -DCI subframe repetition number (2 bits) --> inside the NPDCCH PDU
       *  -TS 36.213 Table 16.6/1/2/3
       *
       *
       *  The higher layer parms (Rmax):
       * -npdcch-NumRepetitions (UE-specific) [inside the NPDCCH UE-specific strucuture] --> configured through phyconfigDedicated
       * -npdcch-NumRepetitionPaging (common)
       * -npdcch-NumRepetitions-RA (common) [inside the NB_IoT_DL_FRAME_PARMS-> nprach_ParametersList] --> configured in phy_config_sib2
       *
       *  PROBLEM: in FAPI specs seems there is no way to trasnmit Rmax to the PHY (waiting for answers)
       *
       * *Rmax is also needed for evaluate the scheduling delay for NDLSCH (see scheduling delay field in NPDCCH PDU FAPI)
       *
       * *Scrambling re-initialization is needed at the beginning of the Search Space or every 4th NPDCCH subframe (See TS 36.211)
       * (this is taken in cosideration by the NPDCCH parameter "scrambling re-initialization batch index" in FAPI specs (Table 4-45)
       *
       ****whenever we have aggregation level = 1 for UE-specific the R is always = 1 (see table 16.6-1)
       ****DCI DL transmission should not happen in case of reference signals or SI messages (this function should be triggered every subframe)
       *
       * */


      for(UE_id = 0 ; UE_id < NUMBER_OF_UE_MAX_NB_IoT; UE_id++)
      {
        if(eNB->npdcch[(uint8_t)UE_id] != NULL && eNB->npdcch[(uint8_t)UE_id]->rnti[(uint8_t)UE_id] == dci_pdu->dci_alloc->rnti && (eNB->ndlsch_SIB1->harq_process->status != ACTIVE_NB_IoT || subframe != 4))
        {
            if(frame%2 == 0)//condition on NSSS (subframe 9 not available)
              {
                if( subframe!= 0 && subframe != 5 && subframe != 9)
                 {

                  generate_dci_top_NB_IoT(eNB->npdcch[(uint8_t)UE_id],
                                          dci_pdu->Num_dci,
                                          dci_pdu->dci_alloc,
                                          AMP,
                                          fp,
                                          eNB->common_vars.txdataF[0],
                                          subframe,
                                          dci_pdu->npdcch_start_symbol); //this parameter depends by eutraControlRegionSize (see TS36.213 16.6.1)
                                          eNB->npdcch[(uint8_t)UE_id]->repetition_idx[(uint8_t)UE_id]++; //can do also inside also the management

                  break;
                 }
              }
          else //this frame not foresee the transmission of NSSS (subframe 9 is available)
             {
             if( subframe!= 0 && subframe != 5)
              {
                generate_dci_top_NB_IoT(eNB->npdcch[(uint8_t)UE_id],
                                        dci_pdu->Num_dci,
                                        dci_pdu->dci_alloc,
                                        AMP,
                                        fp,
                                        eNB->common_vars.txdataF[0],
                                        subframe,
                                        dci_pdu->npdcch_start_symbol); //this parameter depends by eutraControlRegionSize (see TS36.213 16.6.1)
                
                eNB->npdcch[(uint8_t)UE_id]->repetition_idx[(uint8_t)UE_id]++; //can do also inside also the management

              break;
              }

             }
           }
        }

}

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


void prach_procedures_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB) {
 // LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
 // uint16_t preamble_energy_list[64],preamble_delay_list[64];
 // uint16_t preamble_max,preamble_energy_max;
  // uint16_t preamble_max=0;
 // uint16_t i;
 // int8_t UE_id;
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
      
      /*initiate_ra_proc(UL_info->module_id,
                       UL_info->CC_id,
                       NFAPI_SFNSF2SFN(UL_info->rach_ind.sfn_sf),
                       NFAPI_SFNSF2SF(UL_info->rach_ind.sfn_sf),
                       UL_info->rach_ind.rach_indication_body.preamble_list[0].preamble_rel8.preamble,
                       UL_info->rach_ind.rach_indication_body.preamble_list[0].preamble_rel8.timing_advance,
                       UL_info->rach_ind.rach_indication_body.preamble_list[0].preamble_rel8.rnti); 
      mac_xface->initiate_ra_proc(eNB->Mod_id,
                                  eNB->CC_id,
                                  frame,
                                  preamble_index[0],
                                  (int16_t) timing_advance_preamble[0],
                                  0,subframe,0);*/      
  }
  
}