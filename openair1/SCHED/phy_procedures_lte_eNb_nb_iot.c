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

void exit_fun(const char* s);

extern int exit_openair;
struct timespec start_fh, start_fh_prev;
int start_fh_sf, start_fh_prev_sf;
// Fix per CC openair rf/if device update
// extern openair0_device openair0;

unsigned char dlsch_input_buffer[2700] __attribute__ ((aligned(32)));
int eNB_sync_buffer0[640*6] __attribute__ ((aligned(32)));
int eNB_sync_buffer1[640*6] __attribute__ ((aligned(32)));
int *eNB_sync_buffer[2] = {eNB_sync_buffer0, eNB_sync_buffer1};

extern uint16_t hundred_times_log10_NPRB[100];

unsigned int max_peak_val;
int max_sync_pos;

int harq_pid_updated[NUMBER_OF_UE_MAX][8] = {{0}};
int harq_pid_round[NUMBER_OF_UE_MAX][8] = {{0}};

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




void NB_phy_procedures_eNB_uespec_RX(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,const relaying_type_t r_type)
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
  
  /*NB-IoT IF module Common setting*/
  
  UL_IND_t UL_Info;


  UL_Info.module_id = eNB->Mod_id;
  UL_Info.CC_id = eNB->CC_id;
  UL_Info.frame =  frame;
  UL_Info.subframe = subframe;


  T(T_ENB_PHY_UL_TICK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe));

  T(T_ENB_PHY_INPUT_SIGNAL, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(0),
    T_BUFFER(&eNB->common_vars.rxdata[0][0][subframe*eNB->frame_parms.samples_per_tti],
             eNB->frame_parms.samples_per_tti * 4));

  //if ((fp->frame_type == TDD) && (subframe_select(fp,subframe)!=SF_UL)) return;

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_PROCEDURES_ENB_RX_UESPEC+offset, 1 );

#ifdef DEBUG_PHY_PROC
  LOG_D(PHY,"[eNB %d] Frame %d: Doing phy_procedures_eNB_uespec_RX(%d)\n",eNB->Mod_id,frame, subframe);
  LOG_D(PHY,"[eNB %d] Frame %d: Set the Common part in UL_IND\n",eNB->Mod_id,frame);
#endif

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
  for (i=0; i<NUMBER_OF_UE_MAX; i++) {

    // delete srs 

    // delete Pucch procedure

    // check for Msg3
    if (eNB->mac_enabled==1) {
      if (eNB->UE_stats[i].mode == RA_RESPONSE) {
	     process_Msg3(eNB,proc,i,harq_pid);
      }
    }


    eNB->pusch_stats_rb[i][(frame*10)+subframe] = -63;
    eNB->pusch_stats_round[i][(frame*10)+subframe] = 0;
    eNB->pusch_stats_mcs[i][(frame*10)+subframe] = -63;

    /*Check if this UE is has ULSCH scheduling*/
    if ((eNB->ulsch[i]) &&
        (eNB->ulsch[i]->rnti>0) &&
        (eNB->ulsch[i]->harq_processes[harq_pid]->subframe_scheduling_flag==1)) {
      // UE is has ULSCH scheduling
      round = eNB->ulsch[i]->harq_processes[harq_pid]->round;
      /*NB-IoT The nb_rb always set to 1 */
      for (int rb=0;
           rb<=eNB->ulsch[i]->harq_processes[harq_pid]->nb_rb;
	         rb++) 
      {
	     int rb2 = rb+eNB->ulsch[i]->harq_processes[harq_pid]->first_rb;
       eNB->rb_mask_ul[rb2>>5] |= (1<<(rb2&31));
      }

      /*Log for what kind of the ULSCH Reception*/
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

      /*Calculate for LTE C-RS*/
      //nPRS = fp->pusch_config_common.ul_ReferenceSignalsPUSCH.nPRS[subframe<<1];

      //eNB->ulsch[i]->cyclicShift = (eNB->ulsch[i]->harq_processes[harq_pid]->n_DMRS2 + fp->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift +nPRS)%12;

      if (fp->frame_type == FDD ) {
        int sf = (subframe<4) ? (subframe+6) : (subframe-4);
        /*After Downlink Data transmission, simply have a notice to received ACK from PUCCH, I think it's not use for now */
        if (eNB->dlsch[i][0]->subframe_tx[sf]>0) { // we have downlink transmission
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

      eNB->UE_stats[i].ulsch_decoding_attempts[harq_pid][eNB->ulsch[i]->harq_processes[harq_pid]->round]++;
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
        extract_CQI(eNB->ulsch[i]->harq_processes[harq_pid]->o,
                    eNB->ulsch[i]->harq_processes[harq_pid]->uci_format,
                    &eNB->UE_stats[i],
                    fp->N_RB_DL,
                    &rnti, &access_mode);
        eNB->UE_stats[i].rank = eNB->ulsch[i]->harq_processes[harq_pid]->o_RI[0];

      }

      if (eNB->ulsch[i]->Msg3_flag == 1)
	VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_ENB_ULSCH_MSG3,0);

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

          /*In NB-IoT MSG3 */
          // activate retransmission for Msg3 (signalled to UE PHY by DCI
          eNB->ulsch[(uint32_t)i]->Msg3_active = 1;
          /* Need to check the procedure for NB-IoT retransmission
          get_Msg3_alloc_ret(fp,subframe,frame,&eNB->ulsch[i]->Msg3_frame,&eNB->ulsch[i]->Msg3_subframe);
          mac_xface->set_msg3_subframe(eNB->Mod_id, eNB->CC_id, frame, subframe, eNB->ulsch[i]->rnti,eNB->ulsch[i]->Msg3_frame, eNB->ulsch[i]->Msg3_subframe);
          */
          T(T_ENB_PHY_MSG3_ALLOCATION, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe),
                  T_INT(i), T_INT(eNB->ulsch[i]->rnti), T_INT(0 /* 0 is for retransmission*/),
                  T_INT(eNB->ulsch[i]->Msg3_frame), T_INT(eNB->ulsch[i]->Msg3_subframe));
              
          LOG_D(PHY,"[eNB] Frame %d, Subframe %d: Msg3 in error, i = %d \n", frame,subframe,i);
      } // This is Msg3 error

      else 
      { //normal ULSCH
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

              if (eNB->ulsch[i]->harq_processes[harq_pid]->round== eNB->ulsch[i]->Mlimit) 
              {
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
                    {
                      //instead rx_sdu to report NAK to MAC
                      UL_Info.UL_SPEC_Info[i].rntiP= eNB->ulsch[i]->rnti;
                      UL_Info.UL_SPEC_Info[i].sdu = NULL;
                      UL_Info.UL_SPEC_Info[i].sdu_lenP = 0;
                      UL_Info.UL_SPEC_Info[i].harq_pidP = harq_pid;
                      UL_Info.UL_SPEC_Info[i].msg3_flagP = &eNB->ulsch[i]->Msg3_flag;

                    }
              }
        }
      }  // ulsch in error
      else {

        
        T(T_ENB_PHY_ULSCH_UE_ACK, T_INT(eNB->Mod_id), T_INT(frame), T_INT(subframe), T_INT(i), T_INT(eNB->ulsch[i]->rnti),
          T_INT(harq_pid));

        // Delete MSG3  log for the PHICH 
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
                UL_Info.UL_SPEC_Info[i].rntiP= eNB->ulsch[i]->rnti;
                UL_Info.UL_SPEC_Info[i].sdu = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                UL_Info.UL_SPEC_Info[i].sdu_lenP = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                UL_Info.UL_SPEC_Info[i].harq_pidP = harq_pid;
                UL_Info.UL_SPEC_Info[i].msg3_flagP = &eNB->ulsch[i]->Msg3_flag;
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
          for (k=0; k<8; k++) { //harq_processes
            for (j=0; j<eNB->dlsch[i][0]->Mlimit; j++) {
              eNB->UE_stats[i].dlsch_NAK[k][j]=0;
              eNB->UE_stats[i].dlsch_ACK[k][j]=0;
              eNB->UE_stats[i].dlsch_trials[k][j]=0;
            }

            eNB->UE_stats[i].dlsch_l2_errors[k]=0;
            eNB->UE_stats[i].ulsch_errors[k]=0;
            eNB->UE_stats[i].ulsch_consecutive_errors=0;

            for (j=0; j<eNB->ulsch[i]->Mlimit; j++) {
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

	        if (eNB->mac_enabled==1) 
              {
                  // store successful Uplink data in UL_Info instead rx_sdu
                  UL_Info.UL_SPEC_Info[i].rntiP= eNB->ulsch[i]->rnti;
                  UL_Info.UL_SPEC_Info[i].sdu = eNB->ulsch[i]->harq_processes[harq_pid]->b;
                  UL_Info.UL_SPEC_Info[i].sdu_lenP = eNB->ulsch[i]->harq_processes[harq_pid]->TBS>>3;
                  UL_Info.UL_SPEC_Info[i].harq_pidP = harq_pid;
                  UL_Info.UL_SPEC_Info[i].msg3_flagP = NULL;

#ifdef LOCALIZATION
	    start_meas(&eNB->localization_stats);
	    aggregate_eNB_UE_localization_stats(eNB,i,frame,subframe,get_hundred_times_delta_IF_eNB(eNB,i,harq_pid, 1)/100);
	    stop_meas(&eNB->localization_stats);
#endif
	    
	           } // mac_enabled==1
      } // Msg3_flag == 0

          // estimate timing advance for MAC
          if (eNB->abstraction_flag == 0) 
            {
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

#ifdef DEBUG_PHY_PROC
      LOG_D(PHY,"[eNB %d][PDSCH %x] Frame %d subframe %d, Processing HARQ feedback for UE %d (after PUSCH)\n",eNB->Mod_id,
            eNB->dlsch[i][0]->rnti,
            frame,subframe,
            i);
#endif
      // Process HARQ only in NPUSCH
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
            eNB->UE_stats[i].ulsch_decoding_attempts[harq_pid][0]);
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
          if ((eNB->UE_stats[i].ulsch_decoding_attempts[harq_idx][round] -
               eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round]) != 0) {
            eNB->UE_stats[i].ulsch_round_fer[harq_idx][round] =
              (100*(eNB->UE_stats[i].ulsch_round_errors[harq_idx][round] -
                    eNB->UE_stats[i].ulsch_round_errors_last[harq_idx][round]))/
              (eNB->UE_stats[i].ulsch_decoding_attempts[harq_idx][round] -
               eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round]);
          } else {
            eNB->UE_stats[i].ulsch_round_fer[harq_idx][round] = 0;
          }

          eNB->UE_stats[i].ulsch_decoding_attempts_last[harq_idx][round] =
            eNB->UE_stats[i].ulsch_decoding_attempts[harq_idx][round];
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

#undef DEBUG_PHY_PROC


