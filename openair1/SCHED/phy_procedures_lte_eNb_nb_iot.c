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
 * \author R. Knopp, F. Kaltenberger, N. Nikaein, X. Foukas
 * \date 2011
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr,navid.nikaein@eurecom.fr, x.foukas@sms.ed.ac.uk
 * \note
 * \warning
 */

//NB-IoT test
#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"

#include "PHY/defs.h"
#include "PHY/extern.h"
#include "SCHED/defs.h"
#include "SCHED/extern.h"
#include "PHY/LTE_TRANSPORT/if4_tools.h"
#include "PHY/LTE_TRANSPORT/if5_tools.h"

#ifdef EMOS
#include "SCHED/phy_procedures_emos.h"
#endif

//#define DEBUG_PHY_PROC (Already defined in cmake)
//#define DEBUG_ULSCH

#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/defs.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "assertions.h"
#include "msc.h"
#include "PHY/defs_nb_iot.h"
#include <time.h>

#if defined(ENABLE_ITTI)
#   include "intertask_interface.h"
#endif


#if defined(FLEXRAN_AGENT_SB_IF)
//Agent-related headers
#include "ENB_APP/flexran_agent_extern.h"
#include "ENB_APP/CONTROL_MODULES/MAC/flexran_agent_mac.h"
#include "LAYER2/MAC/flexran_agent_mac_proto.h"
#endif

//#define DIAG_PHY

#define NS_PER_SLOT 500000

#define PUCCH 1

//DCI_ALLOC_t dci_alloc[8];

#ifdef EMOS
fifo_dump_emos_eNB emos_dump_eNB;
#endif

#if defined(SMBV) 
extern const char smbv_fname[];
extern unsigned short config_frames[4];
extern uint8_t smbv_frame_cnt;
#endif

#ifdef DIAG_PHY
extern int rx_sig_fifo;
#endif



/* For NB-IoT, we put NPBCH in later part, since it would be scheduled by MAC scheduler
* It generates NRS/NPSS/NSSS
*
*/

void NB_common_signal_procedures (PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc) 
{
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  int **txdataF = eNB->common_vars.txdataF[0];
  int subframe = proc->subframe_tx;
  int frame = proc->frame_tx;
  uint16_t Ntti = 10;//ntti = 10
  int RB_IoT_ID;// RB reserved for NB-IoT, PRB index
  int With_NSSS;// With_NSSS = 1; if the frame include a sub-Frame with NSSS signal
  
  /*NSSS only happened in the even frame*/
  if(frame%2==0)
    {
      With_NSSS = 1;
    }
  else
    {
      With_NSSS = 0;
    }
    
  /*NRS*/
    generate_pilots_NB_IoT(eNB,
               txdataF,
               AMP,
               Ntti,
               RB_IoT_ID,
               With_NSSS);
               
  /*NPSS when subframe 5*/
  if(subframe == 5)
    {
      generate_npss_NB_IoT(txdataF,
                 AMP,
                 fp,
                 3,
                 0,
                 RB_IoT_ID);
    }
    
  /*NSSS when subframe 9 on even frame*/
  if((subframe == 9)&&(With_NSSS == 1))
    {
      generate_nsss_NB_IoT(txdataF,
                          AMP,
                          fp,
                          3,
                          0,
                          frame,
                          RB_IoT_ID);
    }

  
}


void NB_phy_procedures_eNB_uespec_RX(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,const relaying_type_t r_type)
{
  //RX processing for ue-specific resources (i
  UNUSED(r_type);
  uint32_t ret=0,i,j,k;
  uint32_t harq_pid, harq_idx, round;
  int sync_pos;
  uint16_t rnti=0;
  uint8_t access_mode;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;

  const int subframe = proc->subframe_rx;
  const int frame    = proc->frame_rx;
  int offset         = eNB->CC_id;//(proc == &eNB->proc.proc_rxtx[0]) ? 0 : 1;
  
  /*NB-IoT IF module Common setting*/
  
  UL_IND_t UL_INFO;


  UL_INFO.module_id = eNB->Mod_id;
  UL_INFO.CC_id = eNB->CC_id;
  UL_INFO.frame =  frame;
  UL_INFO.subframe = subframe;


  T(T_ENB_PHY_UL_TICK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe));

  T(T_ENB_PHY_INPUT_SIGNAL, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(0),
    T_BUFFER(&eNB->common_vars.rxdata[0][0][subframe*eNB->frame_parms.samples_per_tti],
             eNB->frame_parms.samples_per_tti * 4));

  //if ((fp->frame_type == TDD) && (subframe_select(fp,subframe)!=SF_UL)) return;


  //check if any RB using in this UL subframe
  eNB->rb_mask_ul[0]=0;
  eNB->rb_mask_ul[1]=0;
  eNB->rb_mask_ul[2]=0;
  eNB->rb_mask_ul[3]=0;

  // Check for active processes in current subframe
  // NB-IoT subframe2harq_pid is in dci_tools, always set the frame type to FDD, this would become simpler.
  harq_pid = subframe2harq_pid(fp,frame,subframe);

  // delete the cba
  // delete the srs
  
  /*Loop over the UE, i is the UE ID */
  for (i=0; i<NUMBER_OF_UE_MAX; i++) 
    {

      // delete srs 
      // delete Pucch procedure

      // check for Msg3
      if (eNB->mac_enabled==1) 
        {
          if (eNB->UE_stats[i].mode == RA_RESPONSE) 
            {
	             /*Process Msg3 TODO*/
              //process_Msg3(eNB,proc,i,harq_pid);
            }
        }

      eNB->pusch_stats_rb[i][(frame*10)+subframe] = -63;
      eNB->pusch_stats_round[i][(frame*10)+subframe] = 0;
      eNB->pusch_stats_mcs[i][(frame*10)+subframe] = -63;

      /*Check if this UE is has ULSCH scheduling*/
      if ((eNB->ulsch[i]) &&
          (eNB->ulsch[i]->rnti>0) &&
          (eNB->ulsch[i]->harq_processes[harq_pid]->subframe_scheduling_flag==1)) 
        {
          // UE is has ULSCH scheduling
          round = eNB->ulsch[i]->harq_processes[harq_pid]->round;
          /*NB-IoT The nb_rb always set to 1 */
          for (int rb=0;rb<=eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb;rb++) 
            {
	           int rb2 = rb+eNB->ulsch[i]->harq_processes[harq_pid]->first_rb;
              eNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
            }

          /*Log for what kind of the ULSCH Reception*/

          /*Calculate for LTE C-RS*/
          //nPRS = fp->pusch_config_common.ul_ReferenceSignalsPUSCH.nPRS[subframe<<1];

          //eNB->ulsch[i]->cyclicShift = (eNB->ulsch[i]->harq_processes[harq_pid]->n_DMRS2 + fp->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift +nPRS)%12;

          if (fp->frame_type == FDD ) 
            {
              int sf = (subframe<4) ? (subframe+6) : (subframe-4);
              /*After Downlink Data transmission, simply have a notice to received ACK from PUCCH, I think it's not use for now */
              if (eNB->dlsch[i][0]->subframe_tx[sf]>0) // we have downlink transmission
                { 
                  eNB->ulsch[i]->harq_processes[harq_pid]->O_ACK = 1;
                } 
              else 
                {
                  eNB->ulsch[i]->harq_processes[harq_pid]->O_ACK = 0;
                }
            }

          eNB->pusch_stats_rb[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb;
          eNB->pusch_stats_round[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->round;
          eNB->pusch_stats_mcs[i][(frame*10)+subframe] = eNB->ulsch[i]->harq_processes[harq_pid]->mcs;

          rx_ulsch(eNB,proc,
                  eNB->UE_stats[i].sector,  // this is the effective sector id
                  i,
                  eNB->ulsch,
                  0);

          ret = ulsch_decoding(eNB,proc,
                             i,
                             0, // control_only_flag
                             eNB->ulsch[i]->harq_processes[harq_pid]->V_UL_DAI,
			                       eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb>20 ? 1 : 0);

          //compute the expected ULSCH RX power (for the stats)
          eNB->ulsch[(uint32_t)i]->harq_processes[harq_pid]->delta_TF = get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 0); // 0 means bw_factor is not considered
          eNB->UE_stats[i].ulsch_decoding_attempts[harq_pid][eNB->ulsch[i]->harq_processes[harq_pid]->round]++;
          eNB->ulsch[i]->harq_processes[harq_pid]->subframe_scheduling_flag=0;
          if (eNB->ulsch[i]->harq_processes[harq_pid]->cqi_crc_status == 1) 
            {
              extract_CQI(eNB->ulsch[i]->harq_processes[harq_pid]->o,
                        eNB->ulsch[i]->harq_processes[harq_pid]->uci_format,
                        &eNB->UE_stats[i],
                      fp->N_RB_DL,
                      &rnti, &access_mode);
              eNB->UE_stats[i].rank = eNB->ulsch[i]->harq_processes[harq_pid]->o_RI[0];
            }

          if (ret == (1+MAX_TURBO_ITERATIONS)) 
            {
              T(T_ENB_PHY_ULSCH_UE_NACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(i), T_INT(eNB->ulsch[i]->rnti),
                T_INT(harq_pid));

              eNB->UE_stats[i].ulsch_round_errors[harq_pid][eNB->ulsch[i]->harq_processes[harq_pid]->round]++;
              eNB->ulsch[i]->harq_processes[harq_pid]->phich_active = 1;
              eNB->ulsch[i]->harq_processes[harq_pid]->phich_ACK = 0;
              eNB->ulsch[i]->harq_processes[harq_pid]->round++;

              LOG_D(PHY,"[eNB][PUSCH %d] Increasing to round %d\n",harq_pid,eNB->ulsch[i]->harq_processes[harq_pid]->round);

          if (eNB->ulsch[i]->Msg3_flag == 1) 
            {
	           /*dump_ulsch(eNB,proc,i);
	             exit(-1);*/

            /*In NB-IoT MSG3 */
            // activate retransmission for Msg3 (signalled to UE PHY by DCI
            eNB->ulsch[(uint32_t)i]->Msg3_active = 1;
            /* Need to check the procedure for NB-IoT (MSG3) retransmission
            get_Msg3_alloc_ret(fp,subframe,frame,&eNB->ulsch[i]->Msg3_frame,&eNB->ulsch[i]->Msg3_subframe);
            mac_xface->set_msg3_subframe(eNB->Mod_id, eNB->CC_id, frame, subframe, eNB->ulsch[i]->rnti,eNB->ulsch[i]->Msg3_frame, eNB->ulsch[i]->Msg3_subframe);
            */
            T(T_ENB_PHY_MSG3_ALLOCATION, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe),
                  T_INT(i), T_INT(eNB->ulsch[i]->rnti), T_INT(0 /* 0 is for retransmission*/),
                  T_INT(eNB->ulsch[i]->Msg3_frame), T_INT(eNB->ulsch[i]->Msg3_subframe));     
            } // This is Msg3 error
          else 
            { //normal ULSCH
              if (eNB->ulsch[i]->harq_processes[harq_pid]->round== eNB->ulsch[i]->Mlimit) 
                {
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
                    {
                      //instead rx_sdu to report The Uplink data not received successfully to MAC
                      (UL_INFO.crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 1;
                       UL_INFO.crc_ind.number_of_crcs++;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->data= NULL;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = 0;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid = harq_pid;
                       UL_INFO.RX_NPUSCH.number_of_pdus++;
                    }
                }
            }
        }  // ulsch in error
        else 
          {
            T(T_ENB_PHY_ULSCH_UE_ACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(i), T_INT(eNB->ulsch[i]->rnti),
              T_INT(harq_pid));

          // Delete MSG3  log for the PHICH 

          for (j=0; j<fp->nb_antennas_rx; j++)
          //this is the RSSI per RB
          eNB->UE_stats[i].UL_rssi[j] =
            dB_fixed(eNB->pusch_vars[i]->ulsch_power[j]*
                     (eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb*12)/
                     fp->ofdm_symbol_size) -
            eNB->rx_total_gain_dB -
            hundred_times_log10_NPRB[eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb-1]/100 -
            get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 0)/100;
          //for NB-IoT PHICH not work
	        /*eNB->ulsch[i]->harq_processes[harq_pid]->phich_active = 1;
          eNB->ulsch[i]->harq_processes[harq_pid]->phich_ACK = 1;*/
          eNB->ulsch[i]->harq_processes[harq_pid]->round = 0;
          eNB->UE_stats[i].ulsch_consecutive_errors = 0;

          if (eNB->ulsch[i]->Msg3_flag == 1) 
            {
	            if (eNB->mac_enabled==1) 
                {
	                LOG_I(PHY,"[eNB %d][RAPROC] Frame %d Terminating ra_proc for harq %d, UE %d\n",
		                    eNB->Mod_id,frame,harq_pid,i);
	                if (eNB->mac_enabled)
                    {
                      // store successful MSG3 in UL_Info instead rx_sdu
                      (UL_INFO.crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 0;
                      UL_INFO.crc_ind.number_of_crcs++;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->data = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                      (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid = harq_pid;
                      UL_INFO.RX_NPUSCH.number_of_pdus++;
                    }

	                /* Need check if this needed in NB-IoT
	                // one-shot msg3 detection by MAC: empty PDU (e.g. CRNTI)
	                if (eNB->ulsch[i]->Msg3_flag == 0 ) {
	               eNB->UE_stats[i].mode = PRACH;
	               mac_xface->cancel_ra_proc(eNB->Mod_id,
					       eNB->CC_id,
					       frame,
					       eNB->UE_stats[i].crnti);
	               mac_phy_remove_ue(eNB->Mod_id,eNB->UE_stats[i].crnti);
	               eNB->ulsch[(uint32_t)i]->Msg3_active = 0;
	               } // Msg3_flag == 0*/
	    
	            } // mac_enabled==1

            eNB->UE_stats[i].mode = PUSCH;
            eNB->ulsch[i]->Msg3_flag = 0;

	          LOG_D(PHY,"[eNB %d][RAPROC] Frame %d : RX Subframe %d Setting UE %d mode to PUSCH\n",eNB->Mod_id,frame,subframe,i);

            /*Init HARQ parameters, need to check*/
            for (k=0; k<8; k++) 
              { //harq_processes
                for (j=0; j<eNB->dlsch[i][0]->Mlimit; j++) 
                  {
                    eNB->UE_stats[i].dlsch_NAK[k][j]=0;
                    eNB->UE_stats[i].dlsch_ACK[k][j]=0;
                    eNB->UE_stats[i].dlsch_trials[k][j]=0;
                  }

                eNB->UE_stats[i].dlsch_l2_errors[k]=0;
                eNB->UE_stats[i].ulsch_errors[k]=0;
                eNB->UE_stats[i].ulsch_consecutive_errors=0;

                for (j=0; j<eNB->ulsch[i]->Mlimit; j++) 
                  {
                    eNB->UE_stats[i].ulsch_decoding_attempts[k][j]=0;
                    eNB->UE_stats[i].ulsch_decoding_attempts_last[k][j]=0;
                    eNB->UE_stats[i].ulsch_round_errors[k][j]=0;
                    eNB->UE_stats[i].ulsch_round_fer[k][j]=0;
                  }
              }

            eNB->UE_stats[i].dlsch_sliding_cnt=0;
            eNB->UE_stats[i].dlsch_NAK_round0=0;
            eNB->UE_stats[i].dlsch_mcs_offset=0;
          } // Msg3_flag==1
	       else 
          {  // Msg3_flag == 0
	          if (eNB->mac_enabled==1) 
              {
                  // store successful Uplink data in UL_Info instead rx_sdu
                  (UL_INFO.crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 0;
                  UL_INFO.crc_ind.number_of_crcs++;
                  (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                  (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->data = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                  (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                  (UL_INFO.RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid  = harq_pid;
                  UL_INFO.RX_NPUSCH.number_of_pdus++;
	    
	            } // mac_enabled==1
          } // Msg3_flag == 0

            // estimate timing advance for MAC
              sync_pos = lte_est_timing_advance_pusch(eNB,i);
              eNB->UE_stats[i].timing_advance_update = sync_pos - fp->nb_prefix_samples/4; //to check

      }  // ulsch not in error


      // Process HARQ only in NPUSCH
      /*process_HARQ_feedback(i,
                            eNB,proc,
                            1, // pusch_flag
                            0,
                            0,
                            0);*/


      

    } // ulsch[0] && ulsch[0]->rnti>0 && ulsch[0]->subframe_scheduling_flag == 1


    // update ULSCH statistics for tracing




  } // loop i=0 ... NUMBER_OF_UE_MAX-1



  /*Exact not here, but use to debug*/
  if_inst->UL_indication(UL_INFO);

}

#undef DEBUG_PHY_PROC

/*Generate eNB dlsch params for NB-IoT, modify the input to the Sched Rsp variable*/

void NB_generate_eNB_dlsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t * proc,Sched_Rsp_t *Sched_Rsp,const int UE_id) 
{
  //LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;
  int frame = proc->frame_tx;
  //int subframe = proc->subframe_tx;

  // In NB-IoT, there is no DCI for SI, we might use the scheduling infomation from SIB1-NB to get the phyical layer configuration.
  
  if (Sched_Rsp->NB_DL.NB_DCI.DCI_Format == DCIFormatN1_RAR) // This is format 1A allocation for RA
    {  
      // configure dlsch parameters and CCE index
      LOG_D(PHY,"Generating dlsch params for RA_RNTI\n");

      //NB_generate_eNB_dlsch_params_from_dci();
      
      //eNB->dlsch_ra->nCCE[subframe] = dci_alloc->firstCCE;    
      /*Log for common DCI*/
    }
  else if ((Sched_Rsp->NB_DL.NB_DCI.DCI_Format != DCIFormatN0)&&(Sched_Rsp->NB_DL.NB_DCI.DCI_Format != DCIFormatN2_Ind)&&(Sched_Rsp->NB_DL.NB_DCI.DCI_Format != DCIFormatN2_Pag))
    { // this is a normal DLSCH allocation
      if (UE_id>=0) 
        {
          LOG_D(PHY,"Generating dlsch params for RNTI %x\n",Sched_Rsp->NB_DL.NB_DCI.DL_DCI.npdcch_pdu_rel13.rnti);      
          //NB_generate_eNB_dlsch_params_from_dci();

          /*Log for remaining DCI*/

          //eNB->dlsch[(uint8_t)UE_id][0]->nCCE[subframe] = dci_alloc->firstCCE;
      
          /*LOG for DCI resource allocation and some detail*/
        } 
      else 
        {
          LOG_D(PHY,"[eNB %"PRIu8"][PDSCH] Frame %d : No UE_id with corresponding rnti %"PRIx16", dropping DLSCH\n",
                      eNB->Mod_id,frame,Sched_Rsp->NB_DL.NB_DCI.DL_DCI.npdcch_pdu_rel13.rnti);
        }
    }
  
}

void NB_generate_eNB_ulsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,Sched_Rsp_t *Sched_Rsp,const int UE_id) {

  int harq_pid = 0;

  /*Log for generate ULSCH DCI*/

  //NB_generate_eNB_ulsch_params_from_dci();  
  
  //LOG for ULSCH DCI Resource allocation
  
  if ((Sched_Rsp->NB_DL.NB_DCI.UL_DCI.npdcch_dci_pdu_rel13.rnti  >= CBA_RNTI) && (Sched_Rsp->NB_DL.NB_DCI.UL_DCI.npdcch_dci_pdu_rel13.rnti < P_RNTI))
    eNB->ulsch[(uint32_t)UE_id]->harq_processes[harq_pid]->subframe_cba_scheduling_flag = 1;
  else
    eNB->ulsch[(uint32_t)UE_id]->harq_processes[harq_pid]->subframe_scheduling_flag = 1;
  
}




/*
r_type, rn is only used in PMCH procedure so I remove it.
*/
void NB_phy_procedures_eNB_TX(PHY_VARS_eNB *eNB,
         eNB_rxtx_proc_t *proc,
         int do_meas)
{
  int frame = proc->frame_tx;
  int subframe = proc->subframe_tx;
  uint32_t i,aa;
  uint8_t harq_pid;
  //DCI_PDU_NB *DCI_pdu;
  //DCI_PDU_NB DCI_pdu_tmp;
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
 // DCI_ALLOC_t *dci_alloc = (DCI_ALLOC_t *)NULL;
  int oai_exit = 0;
  int8_t UE_id = 0;
  uint8_t ul_subframe;
  uint32_t ul_frame;
  //uint8_t num_npdcch_symbols = 0;

  //for NB-IoT

  Sched_Rsp_t *Sched_Rsp;

  if(do_meas == 1)
    start_meas(&eNB->phy_proc_tx);

  for(i = 0;i<NUMBER_OF_UE_MAX;i++)
    {
      if((frame==0)&&(subframe==0))
        {
          if(eNB->UE_stats[i].crnti > 0)
              LOG_I(PHY,"UE%d : rnti %x\n",i,eNB->UE_stats[i].crnti);
        }
    }

  // Original scheduler 

  // clear the transmit data array for the current subframe

  for (aa=0; aa<fp->nb_antenna_ports_eNB; aa++) 
    {      
      memset(&eNB->common_vars.txdataF[0][aa][subframe*fp->ofdm_symbol_size*(fp->symbols_per_tti)],
                  0,fp->ofdm_symbol_size*(fp->symbols_per_tti)*sizeof(int32_t));
    } 


  //ignore the PMCH part only do the generate PSS/SSS, note: Seperate MIB from here
  NB_common_signal_procedures(eNB,proc);

  while(!oai_exit)
    {


      /*Not test yet , mutex_l2, cond_l2, instance_cnt_l2
        if(wait_on_condition(&proc->mutex_l2,&proc->cond_l2,&proc->instance_cnt_l2,"eNB_L2_thread") < 0) 
        break;*/

      /*Take the structures from the shared structures*/
      //Sched_Rsp = ;

      /*clear the existing ulsch dci allocations before applying info from MAC*/
      ul_subframe = (subframe+4)%10;
      ul_frame = frame+(ul_subframe >= 6 ? 1 :0);
      harq_pid = ((ul_frame<<1)+ul_subframe)&7;

      /*clear the DCI allocation maps for new subframe*/
      for(i=0;i<NUMBER_OF_UE_MAX;i++)
        {
          if(eNB->ulsch[i])
            {
              eNB->ulsch[i]->harq_processes[harq_pid]->dci_alloc = 0;
              eNB->ulsch[i]->harq_processes[harq_pid]->rar_alloc = 0;
            }
        }

      /*clear previous allocation information for all UEs*/
      for(i=0;i<NUMBER_OF_UE_MAX;i++)
        {
          if(eNB->dlsch[i][0])
            eNB->dlsch[i][0]->subframe_tx[subframe]=0;
        }

      /*remove the part save old HARQ information for PHICH generation*/


      /*Loop over all the dci to generate DLSCH allocation, there is only 1 or 2 DCIs for NB-IoT in the same time*/
      /*Also Packed the DCI here*/
      
      if (Sched_Rsp->NB_DL.NB_DCI.DL_DCI.npdcch_pdu_rel13.rnti<= P_RNTI) 
        {
          UE_id = find_ue((int16_t)Sched_Rsp->NB_DL.NB_DCI.DL_DCI.npdcch_pdu_rel13.rnti,eNB);
        }
      else 
        UE_id=0;
    
      NB_generate_eNB_dlsch_params(eNB,proc,Sched_Rsp,UE_id);
      
      /* Apply physicalConfigDedicated if needed, don't know if needed in NB-IoT or not
       This is for UEs that have received this IE, which changes these DL and UL configuration, we apply after a delay for the eNodeB UL parameters
      phy_config_dedicated_eNB_step2(eNB);*/

      //dci_alloc = &DCI_pdu->dci_alloc[i];

      if (Sched_Rsp->NB_DL.NB_DCI.DCI_Format == DCIFormatN0) // this is a ULSCH allocation
        {  
          UE_id = find_ue((int16_t)Sched_Rsp->NB_DL.NB_DCI.UL_DCI.npdcch_dci_pdu_rel13.rnti,eNB);
          NB_generate_eNB_ulsch_params(eNB,proc,Sched_Rsp,UE_id);
        }

      /*If we have DCI to generate do it now TODO : have a generate dci top for NB_IoT */      
      //NB_generate_dci_top();

      if(Sched_Rsp->NB_DL.NB_DLSCH.NPDSCH_pdu.segments)
        {
            /*TODO: MPDSCH procedures for NB-IoT*/
            //npdsch_procedures();
        }

    }
}