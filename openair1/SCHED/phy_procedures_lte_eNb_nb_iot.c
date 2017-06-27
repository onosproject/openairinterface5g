

#include "PHY/defs.h"
#include "PHY/defs_nb_iot.h"
#include "PHY/extern.h" //where we get the global Sched_Rsp_t structure filled
#include "SCHED/defs.h"
#include "SCHED/extern.h"
#include "PHY/LTE_TRANSPORT/if4_tools.h"
#include "PHY/LTE_TRANSPORT/if5_tools.h"

#ifdef EMOS
#include "SCHED/phy_procedures_emos.h"
#endif

// for NB-IoT
#include "SCHED/defs_nb_iot.h"

//#define DEBUG_PHY_PROC (Already defined in cmake)
//#define DEBUG_ULSCH

#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/defs.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"

#include "T.h"

#include "assertions.h"
#include "msc.h"

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
  else if((subframe == 9)&&(With_NSSS == 1))
    {
      generate_sss_NB_IoT(txdataF,
                          AMP,
                          fp,
                          3,
                          0,
                          frame,
                          RB_IoT_ID);
    }

  else
  {
    /*NRS*/
    generate_pilots_NB_IoT(eNB,
               txdataF,
               AMP,
               Ntti,
               RB_IoT_ID,
               With_NSSS);
  }
  
}

void NB_phy_procedures_eNB_uespec_RX(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,UL_IND_t *UL_INFO)
{
  //RX processing for ue-specific resources (i

  uint32_t ret=0,i,j,k;
  uint32_t harq_pid,round;
  int sync_pos;
  uint16_t rnti=0;
  uint8_t access_mode;
  LTE_DL_FRAME_PARMS *fp=&eNB->frame_parms;

  const int subframe = proc->subframe_rx;
  const int frame    = proc->frame_rx;
  
  /*NB-IoT IF module Common setting*/

  UL_INFO->module_id = eNB->Mod_id;
  UL_INFO->CC_id = eNB->CC_id;
  UL_INFO->frame =  frame;
  UL_INFO->subframe = subframe;


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
                      (UL_INFO->crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 1;
                       UL_INFO->crc_ind.number_of_crcs++;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->data= NULL;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = 0;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid = harq_pid;
                       UL_INFO->RX_NPUSCH.number_of_pdus++;
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
                      (UL_INFO->crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 0;
                      UL_INFO->crc_ind.number_of_crcs++;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->data = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                      (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid = harq_pid;
                      UL_INFO->RX_NPUSCH.number_of_pdus++;
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
                  (UL_INFO->crc_ind.crc_pdu_list+i)->crc_indication_rel8.crc_flag= 0;
                  UL_INFO->crc_ind.number_of_crcs++;
                  (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.rnti= eNB->ulsch[i]->rnti;
                  (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->data = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                  (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_indication_rel8.length = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                  (UL_INFO->RX_NPUSCH.rx_pdu_list+i)->rx_ue_information.harq_pid  = harq_pid;
                  UL_INFO->RX_NPUSCH.number_of_pdus++;
	    
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

}

#undef DEBUG_PHY_PROC

/*Generate eNB dlsch params for NB-IoT, modify the input to the Sched Rsp variable*/

void NB_generate_eNB_dlsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t * proc,Sched_Rsp_t *Sched_Rsp,const int UE_id) 
{
  NB_DL_FRAME_PARMS *fp=&eNB->frame_parms_nb_iot;
  int frame = proc->frame_tx;
  int subframe = proc->subframe_tx;
  DCI_CONTENT DCI_Content[2]; //max number of DCI in a single subframe = 2 (may put this as a global variable)

  //DCI_Content = (DCI_CONTENT*) malloc(sizeof(DCI_CONTENT));

  // In NB-IoT, there is no DCI for SI, we might use the scheduling infomation from SIB1-NB to get the phyical layer configuration.

  //mapping the fapi parameters to the oai parameters

  for (int i = 0; i< Sched_Rsp->NB_DL.NB_DCI->DL_DCI.number_dci; i++){
  switch (Sched_Rsp->NB_DL.NB_DCI->DCI_Format){
    case DCIFormatN1_RAR:
      //DCI format N1 to RAR
      DCI_Content[i].DCIN1_RAR.type = 1;
      //check if this work
      DCI_Content[i].DCIN1_RAR.orderIndicator = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.npdcch_order_indication;
      DCI_Content[i].DCIN1_RAR.Scheddly = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.scheduling_delay;
      DCI_Content[i].DCIN1_RAR.ResAssign = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.resource_assignment;
      DCI_Content[i].DCIN1_RAR.mcs = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.mcs;
      DCI_Content[i].DCIN1_RAR.ndi = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.new_data_indicator;
      DCI_Content[i].DCIN1_RAR.HARQackRes = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.harq_ack_resource;
      DCI_Content[i].DCIN1_RAR.DCIRep = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.dci_subframe_repetition_number;

      // configure dlsch parameters and CCE index (fill the dlsch_ra structure???)
      LOG_D(PHY,"Generating dlsch params for RA_RNTI\n");

      NB_generate_eNB_dlsch_params_from_dci(eNB,
                                            frame,
                                            subframe,
											&DCI_Content[i],
                                            Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list->npdcch_pdu.npdcch_pdu_rel13.rnti,
                                            DCIFormatN1_RAR,
                                            &eNB->dlsch_ra_NB,
                                            fp,
                                            Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list->npdcch_pdu.npdcch_pdu_rel13.aggregation_level,
											Sched_Rsp->NB_DL.NB_DCI->DL_DCI.number_dci
                                            );
      break;
    case DCIFormatN1:
      //DCI format N1 to DLSCH
    	 DCI_Content[i].DCIN1.type = 1;
    	 DCI_Content[i].DCIN1.orderIndicator = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.npdcch_order_indication;
    	 DCI_Content[i].DCIN1.Scheddly = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.scheduling_delay;
    	 DCI_Content[i].DCIN1.ResAssign = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.resource_assignment;
    	 DCI_Content[i].DCIN1.mcs = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.mcs;
    	 DCI_Content[i].DCIN1.ndi = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.new_data_indicator;
    	 DCI_Content[i].DCIN1.HARQackRes = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.harq_ack_resource;
    	 DCI_Content[i].DCIN1.DCIRep = Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.dci_subframe_repetition_number;


    	//fill the dlsch[] structure???
      NB_generate_eNB_dlsch_params_from_dci(eNB,
                                            frame,
                                            subframe,
											                      &DCI_Content[i],
                                            Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list->npdcch_pdu.npdcch_pdu_rel13.rnti,
                                            DCIFormatN1,
                                            eNB->ndlsch[(uint8_t)UE_id],
                                            fp,
                                            Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list->npdcch_pdu.npdcch_pdu_rel13.aggregation_level,
                                            Sched_Rsp->NB_DL.NB_DCI->DL_DCI.number_dci
                                            );
      break;
    /*TODO reserve for the N2 DCI*/
    case DCIFormatN2_Pag:
    	LOG_I(PHY, "Paging is not implemented, DCIFormatN2_Pag cannot be elaborated\n");
    	break;

  }

  }



  
}

void NB_generate_eNB_ulsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,Sched_Rsp_t *Sched_Rsp,const int UE_id) {

  int harq_pid = 0;

  DCI_CONTENT DCI_Content[2]; //max number of DCI in a single subframe = 2 (may put this as a global variable)

  for(int i = 0; i<Sched_Rsp->NB_DL.NB_DCI->UL_DCI.number_of_dci; i++)
  {

  //mapping the fapi parameters to the OAI parameters
  DCI_Content[i].DCIN0.type = 0;
  DCI_Content[i].DCIN0.scind = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.subcarrier_indication;
  DCI_Content[i].DCIN0.ResAssign = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.subcarrier_indication;
  DCI_Content[i].DCIN0.mcs = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.mcs;
  DCI_Content[i].DCIN0.ndi = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.new_data_indicator;
  DCI_Content[i].DCIN0.Scheddly = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.scheduling_delay;
  DCI_Content[i].DCIN0.RepNum = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.repetition_number;
  DCI_Content[i].DCIN0.rv = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.redudancy_version;
  DCI_Content[i].DCIN0.DCIRep = Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.dci_subframe_repetition_number;


  /*Log for generate ULSCH DCI*/

  NB_generate_eNB_ulsch_params_from_dci(eNB,
                                        proc,
                                        &DCI_Content[i],
                                        Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list->npdcch_dci_pdu.npdcch_dci_pdu_rel13.rnti,
                                        DCIFormatN0,
                                        UE_id,
                                        Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list->npdcch_dci_pdu.npdcch_dci_pdu_rel13.aggregation_level,
										Sched_Rsp->NB_DL.NB_DCI->UL_DCI.number_of_dci
                                        );  

  
  //LOG for ULSCH DCI Resource allocation
  
  //CBA is not used in NB-IoT
//  if ((Sched_Rsp->NB_DL.NB_DCI.UL_DCI.npdcch_dci_pdu_rel13.rnti  >= CBA_RNTI) && (Sched_Rsp->NB_DL.NB_DCI.UL_DCI.npdcch_dci_pdu_rel13.rnti < P_RNTI))
//    eNB->ulsch[(uint32_t)UE_id]->harq_processes[harq_pid]->subframe_cba_scheduling_flag = 1;
//  else
    eNB->ulsch[(uint32_t)UE_id]->harq_processes[harq_pid]->subframe_scheduling_flag = 1;
  
  }
}


/* process the following message 
 * mainly for filling DLSCH and ULSCH data structures in PHY_vars
 * if there is a DCI we do the packing
* 
*/
void process_schedule_rsp(Sched_Rsp_t *sched_rsp,
                          PHY_VARS_eNB *eNB,
                          eNB_rxtx_proc_NB_t *proc)

{

	int UE_id = 0;

	  //First check for DCI
	  if(sched_rsp->NB_DL.NB_DCI != NULL)
	  {

		  /*
		   * In NB-IoT we could have at most 2 DCI but not of two different type (only 2 for the DL or two for the UL)
		   * DCI N0 is the only format for UL
		   *
		   */

		  if(Sched_Rsp->NB_DL.NB_DCI->DCI_Format != DCIFormatN0) //this is a DLSCH allocation
		  {


		      /*Loop over all the dci to generate DLSCH allocation, there is only 1 or 2 DCIs for NB-IoT in the same time*/
		      // Add dci fapi structure for contain two dcis
		      /*Also Packed the DCI here*/
		      for(int i= 0; i< Sched_Rsp->NB_DL.NB_DCI->DL_DCI.number_dci; i++)
		      {

		      if (Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.rnti<= P_RNTI)
		        {
		    	  //is not system iformation but cound be paging
		    	  //in any case we generate dlsch for not system information
		          UE_id = find_ue_NB((int16_t)Sched_Rsp->NB_DL.NB_DCI->DL_DCI.dl_config_pdu_list[i].npdcch_pdu.npdcch_pdu_rel13.rnti,eNB);
		          if (UE_id<0)
		        	  LOG_E(PHY, "process_schedule_rsp: UE_id for DL_DCI is negative\n");
		        }
		      else
		        UE_id=0;

		    	//inside we have nFAPI to OAI parameters
		    	NB_generate_eNB_dlsch_params(eNB,proc,Sched_Rsp,UE_id);

		      }

		  }
		   else //ULSCH allocation
		   {

		      /* Apply physicalConfigDedicated if needed, don't know if needed in NB-IoT or not
		       This is for UEs that have received this IE, which changes these DL and UL configuration, we apply after a delay for the eNodeB UL parameters
		      phy_config_dedicated_eNB_step2(eNB);*/


			//HI_DCI0_request
			for (int i = 0; Sched_Rsp->NB_DL.NB_DCI->UL_DCI.number_of_dci; i ++)
			{

				UE_id = find_ue_NB((int16_t)Sched_Rsp->NB_DL.NB_DCI->UL_DCI.hi_dci0_pdu_list[i].npdcch_dci_pdu.npdcch_dci_pdu_rel13.rnti,eNB);

				if (UE_id<0)
				LOG_E(PHY, "process_schedule_rsp: UE_id for UL_DCI is negative\n");

				NB_generate_eNB_ulsch_params(eNB,proc,Sched_Rsp,UE_id);

				}
		    }
	  }

	  else if(sched_rsp->NB_DL.NB_DLSCH != NULL)
	   {

		  //check for the SI (FAPI specs rnti_type = 1 Other or rnti_type = 0 BCCH information)
		  if(sched_rsp->NB_DL.NB_DLSCH->ndlsch.rnti_type == 0)
		  {
		  //TODO fill the dlsch_SI
		  }
		  else
		  {
		  //TODO fill dlsch ue data
		  }

	   }
	  else if(sched_rsp->NB_DL.NB_UL_config != NULL)
	  {
		  //TODO : manage the configuration for UL
		  /*
		   * UL_CONFIG.request body
		   * pdu_type = 16 (NULSCH PDU)
		   * pdu_type = 17 (NRACH PDU) just for nprach configuration change (not managed here)
		   *
		   */

		  //NULSCH PDU (FAPI specs table 4-49)
		  if(sched_rsp->NB_DL.NB_UL_config->ulsch_pdu.pdu_type == 16)
		  {

		  }
	  }
	  else
		  LOG_D(PHY, "No DL data contained in the Sched_rsp!!\n");



};



/*
 * for NB-IoT ndlsch procedure
 * this function is called by the PHy procedure TX in 3 possible occasion:
 * 1) we manage BCCH pdu
 * 2) we manage RA dlsch pdu (to be checked if needed in our case)
 * 3) UE specific dlsch pdu
 * ** we need to know if exist and which value has the eutracontrolRegionSize (TS 36.213 ch 16.4.1.4) whenever we are in In-band mode
 * ** CQI and PMI are not present in NB-IoT
 * ** redundancy version exist only in UL for NB-IoT and not in DL
 */
//void npdsch_procedures(PHY_VARS_eNB *eNB,
//						eNB_rxtx_proc_t *proc, //Context data structure for RX/TX portion of subframe processing
//						NB_IoT_eNB_NDLSCH_t *dlsch,
////NB_IoT_eNB_DLSCH_t *dlsch1,//this is the second colum of the UE specific LTE_eNB_DLSCH_t (see the PHY/defs.h line 471) is used only in ue specific dlsch for two parallel streams (but we don't have it in NB-IoT)
//						LTE_eNB_UE_stats *ue_stats,
//						int ra_flag,// set to 1 only in case of RAR as a segment data
//						//int num_pdcch_symbols, (BCOM says are not needed
//						uint32_t segment_length, //lenght of the  DLSCH PDU from the Sched_rsp (FAPI nomenclature)
//						uint8_t* segment_data // the DLSCH PDU itself from the Sched_rsp (FAPI nomenclature)
//							)
//{
//  int frame=proc->frame_tx;
//  int subframe=proc->subframe_tx;
//  //int harq_pid = dlsch->current_harq_pid;
//  LTE_DL_eNB_HARQ_t *dlsch_harq=dlsch->harq_process; //TODO: review the HARQ process for NB_IoT
//  int input_buffer_length = dlsch_harq->TBS/8; // get in byte //to be changed for NB_IoT????
//  NB_DL_FRAME_PARMS *fp=&eNB->frame_parms_nb_iot;
//  uint8_t *DLSCH_pdu=NULL;
//  uint8_t DLSCH_pdu_tmp[input_buffer_length+4]; //[768*8];
//  //uint8_t DLSCH_pdu_rar[256];
//  int i;
//
//
//
//  LOG_D(PHY,
//	"[eNB %"PRIu8"][PDSCH rnti%"PRIx16"] Frame %d, subframe %d: Generating PDSCH/DLSCH with input size = %"PRIu16", G %d, nb_rb %"PRIu16", mcs %"PRIu8"(round %"PRIu8")\n",
//	eNB->Mod_id,
//	dlsch->rnti,
//	frame, subframe, input_buffer_length,
//	get_G(fp,dlsch_harq->nb_rb,dlsch_harq->rb_alloc,get_Qm(dlsch_harq->mcs),dlsch_harq->Nl, num_pdcch_symbols, frame, subframe, dlsch_harq->mimo_mode==TM7?7:0),
//	dlsch_harq->nb_rb, //in NB_IoT we not need it??? (Current Number of RBs should be only 1)
//	dlsch_harq->mcs,
//	dlsch_harq->round);
//
//
/////XXX skip this for the moment and all the ue stats
////#if defined(MESSAGE_CHART_GENERATOR_PHY)
////  MSC_LOG_TX_MESSAGE(
////		     MSC_PHY_ENB,MSC_PHY_UE,
////		     NULL,0,
////		     "%05u:%02u PDSCH/DLSCH input size = %"PRIu16", G %d, nb_rb %"PRIu16", mcs %"PRIu8", pmi_alloc %"PRIx16", rv %"PRIu8" (round %"PRIu8")",
////		     frame, subframe,
////		     input_buffer_length,
////		     get_G(fp,
////			   dlsch_harq->nb_rb,
////			   dlsch_harq->rb_alloc,
////			   get_Qm(dlsch_harq->mcs),
////			   dlsch_harq->Nl,
////			   num_pdcch_symbols,
////			   frame,
////			   subframe,
////			   dlsch_harq->mimo_mode==TM7?7:0),
////		     dlsch_harq->nb_rb,
////		     dlsch_harq->mcs,
////		     pmi2hex_2Ar1(dlsch_harq->pmi_alloc),
////		     dlsch_harq->rvidx,
////		     dlsch_harq->round);
////#endif
//
////if (ue_stats) ue_stats->dlsch_sliding_cnt++; //used to compute the mcs offset
//
//  if(dlsch_harq->round == 0) { //first transmission
//
////    if (ue_stats)
////      ue_stats->dlsch_trials[harq_pid][0]++;
//
//    if (eNB->mac_enabled==1) { // set in lte-softmodem/main line 1646
//      if (ra_flag == 0)  {
//    	  DLSCH_pdu =segment_data;
//
//      }
//    else { //manage the RAR
//
//  	  /*
//  	   * In FAPI style we don-t need to process the RAR because we have all the parameters for getting the MSG3 given by the
//  	   * UL_CONFIG.request (all inside the next Sched_RSP function)
//  	   *
//  	   */
//
////    	  int16_t crnti = mac_xface->fill_rar(eNB->Mod_id,
////					    eNB->CC_id,
////					    frame,
////					    DLSCH_pdu_rar,
////					    fp->N_RB_UL,
////					    input_buffer_length);
//
//    	  DLSCH_pdu = segment_data; //the proper PDU should be passed in the function when the RA flag is activated
//
//    	  int UE_id;
//
//    	  if (crnti!=0)
//    		  UE_id = add_ue(crnti,eNB);
//    	  else
//	 	UE_id = -1;
//
//    	  if (UE_id==-1) {
//    		  LOG_W(PHY,"[eNB] Max user count reached.\n");
//    		  mac_xface->cancel_ra_proc(eNB->Mod_id,
//    				  	  	  	  	  eNB->CC_id,
//									  frame,
//									  crnti);
//    	  } else {
//    		  eNB->UE_stats[(uint32_t)UE_id].mode = RA_RESPONSE;
//    		  // Initialize indicator for first SR (to be cleared after ConnectionSetup is acknowledged)
//    		  eNB->first_sr[(uint32_t)UE_id] = 1;
//
//
//
//
//
//    		  generate_eNB_ulsch_params_from_rar(DLSCH_pdu,
//					     	 	 	 	 	 	 frame,
//												 subframe,
//												 eNB->ulsch[(uint32_t)UE_id],
//												 fp);
//
//    		  LOG_D(PHY,"[eNB][RAPROC] Frame %d subframe %d, Activated Msg3 demodulation for UE %"PRId8" in frame %"PRIu32", subframe %"PRIu8"\n",
//    				  frame,
//					  subframe,
//					  UE_id,
//					  eNB->ulsch[(uint32_t)UE_id]->Msg3_frame,
//					  eNB->ulsch[(uint32_t)UE_id]->Msg3_subframe);
//
//    		  /* TODO: get rid of this hack. The problem is that the eNodeB may
//    		   * sometimes wrongly generate PHICH because somewhere 'phich_active' was
//    		   * not reset to 0, due to an unidentified reason. When adding this
//    		   * resetting here the problem seems to disappear completely.
//    		   */
//    		  LOG_D(PHY, "hack: set phich_active to 0 for UE %d fsf %d %d all HARQs\n", UE_id, frame, subframe);
//    		  for (i = 0; i < 8; i++)
//    			  eNB->ulsch[(uint32_t)UE_id]->harq_processes[i]->phich_active = 0;
//
//    		  mac_xface->set_msg3_subframe(eNB->Mod_id, eNB->CC_id, frame, subframe, (uint16_t)crnti,
//                                       eNB->ulsch[UE_id]->Msg3_frame, eNB->ulsch[UE_id]->Msg3_subframe);
//
//    		  T(T_ENB_PHY_MSG3_ALLOCATION, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe),
//    				  T_INT(UE_id), T_INT((uint16_t)crnti), T_INT(1 /* 1 is for initial transmission*/),
//					  T_INT(eNB->ulsch[UE_id]->Msg3_frame), T_INT(eNB->ulsch[UE_id]->Msg3_subframe));
//    	  }
//    	  if (ue_stats) ue_stats->total_TBS_MAC += dlsch_harq->TBS;
//
//      }// ra_flag = 1
//
//    } //mac_eabled = 1
//    else {  //XXX we should change taus function???
//      DLSCH_pdu = DLSCH_pdu_tmp;
//
//      for (i=0; i<input_buffer_length; i++)
//	DLSCH_pdu[i] = (unsigned char)(taus()&0xff);
//    }
//
//#if defined(SMBV)
//
//    // Configures the data source of allocation (allocation is configured by DCI)
//    if (smbv_is_config_frame(frame) && (smbv_frame_cnt < 4)) {
//      LOG_D(PHY,"[SMBV] Frame %3d, Configuring PDSCH payload in SF %d alloc %"PRIu8"\n",frame,(smbv_frame_cnt*10) + (subframe),smbv_alloc_cnt);
//      //          smbv_configure_datalist_for_user(smbv_fname, find_ue(dlsch->rnti,eNB)+1, DLSCH_pdu, input_buffer_length);
//    }
//#endif
//
//
//
//#ifdef DEBUG_PHY_PROC
//#ifdef DEBUG_DLSCH
//    LOG_T(PHY,"eNB DLSCH SDU: \n");
//
//    //eNB->dlsch[(uint8_t)UE_id][0]->nCCE[subframe] = DCI_pdu->dci_alloc[i].firstCCE;
//
//    LOG_D(PHY,"[eNB %"PRIu8"] Frame %d subframe %d : CCE resource for ue DCI (PDSCH %"PRIx16")  => %"PRIu8"/%u\n",eNB->Mod_id,eNB->proc[sched_subframe].frame_tx,subframe,
//	  DCI_pdu->dci_alloc[i].rnti,eNB->dlsch[(uint8_t)UE_id][0]->nCCE[subframe],DCI_pdu->dci_alloc[i].firstCCE);
//
//
//    for (i=0; i<dlsch_harq->TBS>>3; i++)
//      LOG_T(PHY,"%"PRIx8".",DLSCH_pdu[i]);
//
//    LOG_T(PHY,"\n");
//#endif
//#endif
//  } //harq round == 0
//  else {
//	  //We are doing a retransmission
//
//    ue_stats->dlsch_trials[harq_pid][dlsch_harq->round]++;
//
//#ifdef DEBUG_PHY_PROC
//#ifdef DEBUG_DLSCH
//    LOG_D(PHY,"[eNB] This DLSCH is a retransmission\n");
//#endif
//#endif
//  }
//
//  if (eNB->abstraction_flag==0) { // used for simulation of the PHY??
//
//    LOG_D(PHY,"Generating NDLSCH/NPDSCH %d\n",ra_flag);
//
//
//    // 36-212
//    //encoding---------------------------
//
//    /*
//     * we should have as an iput parameter also G for the encoding based on the switch/case
//     * G is evaluated based on the switch/case done over eutracontrolRegionSize (if exist) and operationModeInfo
//     * NB: switch case of G is the same for npdsch and npdcch
//     *
//     * Nsf needed as an input (number of subframe)
//     */
//
//    start_meas(&eNB->dlsch_encoding_stats);
//
//    LOG_I(PHY, "NB-IoT Encoding step\n");
//
//    eNB->te(eNB,
//	    DLSCH_pdu,
//	    num_pdcch_symbols,
//	    dlsch,
//	    frame,subframe,
//	    &eNB->dlsch_rate_matching_stats,
//	    &eNB->dlsch_turbo_encoding_stats,
//	    &eNB->dlsch_interleaving_stats);
//
//
//    stop_meas(&eNB->dlsch_encoding_stats);
//    //scrambling-------------------------------------------
//    // 36-211
//    start_meas(&eNB->dlsch_scrambling_stats);
//    LOG_I(PHY, "NB-IoT Scrambling step\n");
//
//    dlsch_scrambling(fp,
//		     0,
//		     dlsch,
//		     get_G(fp,
//			   dlsch_harq->nb_rb,
//			   dlsch_harq->rb_alloc,
//			   get_Qm(dlsch_harq->mcs),
//			   dlsch_harq->Nl,
//			   num_pdcch_symbols,
//			   frame,subframe,
//			   0),
//		     0,
//		     subframe<<1);
//
//    stop_meas(&eNB->dlsch_scrambling_stats);
//
//
//    //modulation-------------------------------------------
//    start_meas(&eNB->dlsch_modulation_stats);
//    LOG_I(PHY, "NB-IoT Modulation step\n");
//
//    dlsch_modulation(eNB,
//		     eNB->common_vars.txdataF[0],
//		     AMP,
//		     subframe,
//		     num_pdcch_symbols,
//		     dlsch,
//		     dlsch1);
//
//    stop_meas(&eNB->dlsch_modulation_stats);
//  }
//
//
//#ifdef PHY_ABSTRACTION
//  else {
//    start_meas(&eNB->dlsch_encoding_stats);
//    dlsch_encoding_emul(eNB,
//			DLSCH_pdu,
//			dlsch);
//    stop_meas(&eNB->dlsch_encoding_stats);
//  }
//
//#endif
//  dlsch->active = 0;
//}



extern int oai_exit;

/*
r_type, rn is only used in PMCH procedure so I remove it.
*/
void NB_phy_procedures_eNB_TX(PHY_VARS_eNB *eNB,
         	 	 	 	 	  eNB_rxtx_proc_NB_t *proc,
							  int do_meas)
{
  int frame = proc->frame_tx;
  int subframe = proc->subframe_tx;
  uint32_t i,aa;
  //uint8_t harq_pid; only one HARQ process
  //DCI_PDU_NB *DCI_pdu; we already have inside Sched_Rsp
  //DCI_PDU_NB DCI_pdu_tmp;
  NB_DL_FRAME_PARMS *fp = &eNB->frame_parms_nb_iot;
  // DCI_ALLOC_t *dci_alloc = (DCI_ALLOC_t *)NULL;
  int8_t UE_id = 0;
  uint8_t ul_subframe;
  uint32_t ul_frame;

  int **txdataF = eNB->common_vars.txdataF[0];

  Sched_Rsp_t *sched_rsp;

  // are needed??? (maybe not)
  //uint8_t num_npdcch_symbols = 0;

  if(do_meas == 1)
    start_meas(&eNB->phy_proc_tx);


  //UE_stats can be not considered
//  for(i = 0;i<NUMBER_OF_UE_MAX;i++)
//    {
//      if((frame==0)&&(subframe==0))
//        {
//          if(eNB->UE_stats[i].crnti > 0)
//              LOG_I(PHY,"UE%d : rnti %x\n",i,eNB->UE_stats[i].crnti);
//        }
//    }
  //ULSCH consecutive error count from Ue_stats reached has been deleted

  /*called the original scheduler "eNB_dlsch_ulsch_scheduler" now is no more done here but is triggered directly from UL_Indication (IF-Module Function)*/

  // clear the transmit data array for the current subframe
  for (aa=0; aa<fp->nb_antenna_ports_eNB; aa++) 
    {      
      memset(&eNB->common_vars.txdataF[0][aa][subframe*fp->ofdm_symbol_size*(fp->symbols_per_tti)],
                  0,fp->ofdm_symbol_size*(fp->symbols_per_tti)*sizeof(int32_t));
    } 


  //follow similar approach of (eNB_thread_prach)
  while(!oai_exit)
    {

	  if(oai_exit) break;

	  //generate NPSS/NSSS
	  NB_common_signal_procedures(eNB,proc);

        //Not test yet , mutex_l2, cond_l2, instance_cnt_l2
        //cond_l2 should be given by sched_rsp after the scheduling
	  	//WE should unlock this thread through the Schedule_rsp content
        if(wait_on_condition(&proc->mutex_l2,&proc->cond_l2,&proc->instance_cnt_l2,"eNB_L2_thread") < 0) 
        break;

      /*At this point it means that we have the Sched_rsp structure filled by the MAC scheduler*/

       //Take the Sched_rsp structures from the shared structures
       if(Sched_Rsp == NULL)
    	   LOG_E(PHY, "We don't have the Sched_Rsp_t structure ready\n");

      sched_rsp = Sched_Rsp;



      /*Generate MIB
       *
       *
       *Sched_Rsp_t content:
       *
       * DL_Config.request--> dl_config_request_pdu --> nfapi_dl_config_nbch_pdu_rel13_t --> NBCH PDU
       *
       * TX.request --> nfapi_tx_request_pdu_t --> MAC PDU (MIB)
       * 	Content of tx_request_pdu
       * 	-pdu length 14 (bytes)???
       * 	-pdu index = 1
       * 	-num segments = 1,
       * 	-segment length 5 bytes,
       * 	-segment data  34 bits for MIB (5 bytes)
       *
       *XXX we are assuming that for the moment we are not segmenting the pdu (this should happen only when we have separate machines)
       *XXX so the data is always contained in the first segment (segments[0])
       *
       *Problem: NB_IoT_RB_ID should be the ID of the RB dedicated to NB-IoT
       *but if we are in stand alone mode?? i should pass the DC carrier???
       *in general this RB-ID should be w.r.t LTE bandwidht???
       **allowed indexes for Nb-IoT PRBs are reported in R&Shwartz pag 9
       *
       */
      if((subframe==0) && (sched_rsp->NB_DL.NB_BCH->MIB_pdu.segments[0].segment_data)!=NULL)
        {
          generate_npbch(&eNB->npbch,
                         txdataF,
                         AMP,
                         fp,
                         &sched_rsp->NB_DL.NB_BCH->MIB_pdu.segments[0].segment_data,
                         frame%64,
                         fp->NB_IoT_RB_ID
                         );
        }


      /*process the schedule response
       * fill the PHY config structures
       * */
      process_schedule_rsp(sched_rsp, eNB, proc);


      /*
       * Generate BCCH transmission (System Information)
       */
      // check for BCCH
      //rnti_type = 0 BCCH information
      //rnti_type = 1 Other
      if(sched_rsp->NB_DL.NB_DLSCH->ndlsch.rnti_type == 0 && (sched_rsp->NB_DL.NB_DLSCH->NPDSCH_pdu.segments[0].segment_data)!= NULL)
        {
            /*TODO: NPDSCH procedures for BCCH for NB-IoT*/
            npdsch_procedures(eNB,
            				  proc,
							  eNB->dlsch_SI_NB, //should be filled ?? (in the old implementation was filled when from DCI we generate_dlsch_params
							  sched_rsp->NB_DL.NB_DLSCH->NPDSCH_pdu.segments[0].segment_length,
							  sched_rsp->NB_DL.NB_DLSCH->NPDSCH_pdu.segments[0].segment_data);

        }


      //no HARQ pid (we have only 1 single process for each user)
      //clear previous possible allocation
      for(int i = 0; i < NUMBER_OF_UE_MAX_NB_IoT; i++)
      {
    	  if(eNB->ndlsch[i])
    	  {
    		  eNB->ndlsch[i]->harq_process->round=0; // may not needed
    		  /*clear previous allocation information for all UEs*/
    		  eNB->ndlsch[i]->subframe_tx[subframe] = 0;
    	  }


    	  /*clear the DCI allocation maps for new subframe*/
    	  if(eNB->nulsch[i])
    	  {
    		  eNB->nulsch[i]->harq_process->dci_alloc = 0; //flag for indicating that a DCI has been allocated for UL
    		  eNB->nulsch[i]->harq_process->rar_alloc = 0; //Flag indicating that this ULSCH has been allocated by a RAR
    	  }

      }


      //no PHICH in NB-IoT
      //num_pdcch_symbols?? (maybe later when we have the DCI)




      //now we should check if Sched_Rsp contains data
      //rnti_type = 0 BCCH information
      //rnti_type = 1 Other
      if(Sched_Rsp->NB_DL.NB_DLSCH->ndlsch.rnti_type == 1 && (sched_rsp->NB_DL.NB_DLSCH->NPDSCH_pdu.segments[0].segment_data)!= NULL)
      {

    	  //we not need between RAR PDUS
              /*TODO: NPDSCH procedures for BCCH for NB-IoT*/
              //npdsch_procedures();
      }


      /*If we have DCI to generate do it now TODO : have a generate dci top for NB_IoT */
      //to be modified but inside we have the nuew function for dci transmission
      generate_dci_top_NB();


      //maybe not needed
      if (release_thread(&proc->mutex_l2,&proc->instance_cnt_l2,"eNB_L2_thread") < 0) break;

    }



}
