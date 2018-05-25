/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*! \file slschsim.c
 \brief Top-level UL simulator
 \author R. Knopp
 \date 2011 - 2018
 \version 0.1
 \company Eurecom
 \email: knopp@eurecom.fr
 \note
 \warning
*/

#include <string.h>
#include <math.h>
#include <unistd.h>
#include "SIMULATION/TOOLS/defs.h"
#include "PHY/types.h"
#include "PHY/defs.h"
#include "PHY/vars.h"

#include "SCHED/defs.h"
#include "SCHED/vars.h"
#include "LAYER2/MAC/vars.h"
#include "OCG_vars.h"
#include "intertask_interface_init.h"

#include "unitary_defs.h"

//#define PSBCH_DEBUG 1

int nfapi_mode=0;
double cpuf;
extern int32_t* sync_corr_ue1;

#define PSBCH_A 40
#define PSBCH_E 1008 //12REs/PRB*6PRBs*7symbols*2 bits/RB

int main(int argc, char **argv) {

  int frame,subframe;
  int trials;
  int n_trials=10;
  int N_RB_DL=50;
  int n_rx=1;
  channel_desc_t *UE2UE[2][2][2];
  PHY_VARS_UE *UE;
  int log_level = LOG_INFO;
  int tx_offset=0;
  SLSCH_t slsch;
  SLDCH_t sldch;
  SLSS_t slss;
  
  SCM_t channel_model=AWGN;
  UE_rxtx_proc_t proc;
  double snr0 = 35;
  double snr_step=1;
  double snr_int=1;
  uint8_t slsch_payload[768*9];
  uint8_t sldch_payload[32];
  int mcs=10;
  int nb_rb=20;
  char channel_model_input[20];
  int pscch_errors=0;
  int do_SLSS=0;

  
  AssertFatal(load_configmodule(argc,argv) != NULL,
	      "cannot load configuration module, exiting\n");
  logInit();
  // enable these lines if you need debug info
  set_comp_log(PHY,LOG_INFO,LOG_HIGH,1);
  set_comp_log(MAC,LOG_INFO,LOG_HIGH,1);
  set_comp_log(RRC,LOG_INFO,LOG_HIGH,1);
  set_comp_log(OCM,LOG_INFO,LOG_HIGH,1);
  set_glog(log_level,LOG_HIGH);
  

  char c;

  while ((c = getopt (argc, argv, "hf:m:n:g:r:z:w:s:S:")) != -1) {
    switch (c) {
    case 'f':
      snr_step= atof(optarg);
      break;
    case 'm':
      mcs = atoi(optarg);
      break;
    case 'n':
      n_trials = atoi(optarg);
      break;
    case 'r':
      nb_rb = atoi(optarg);
      break;

    case 's':
      snr0 = atof(optarg);
      break;
    case 'g':
      memcpy(channel_model_input,optarg,10);

      switch((char)*optarg) {
      case 'A':
        channel_model=SCM_A;
        break;

      case 'B':
        channel_model=SCM_B;
        break;

      case 'C':
        channel_model=SCM_C;
        break;

      case 'D':
        channel_model=SCM_D;
        break;

      case 'E':
        channel_model=EPA;
        break;

      case 'F':
        channel_model=EVA;
        break;

      case 'G':
        channel_model=ETU;
        break;

      case 'H':
        channel_model=Rayleigh8;
        break;

      case 'I':
        channel_model=Rayleigh1;
        break;

      case 'J':
        channel_model=Rayleigh1_corr;
        break;

      case 'K':
        channel_model=Rayleigh1_anticorr;
        break;

      case 'L':
        channel_model=Rice8;
        break;

      case 'M':
        channel_model=Rice1;
        break;

      case 'N':
        channel_model=AWGN;
        break;
      default:
        printf("Unsupported channel model!\n");
        exit(-1);
      }

      break;
    
    case 'z':
      n_rx=atoi(optarg);

      if ((n_rx==0) || (n_rx>2)) {
        printf("Unsupported number of rx antennas %d\n",n_rx);
        exit(-1);
      }

      break;
    case 'S':
      do_SLSS=1;
      tx_offset=atoi(optarg);
      printf("Running TX/RX synchornization signals with timing offset %d\n",tx_offset);
      break;
      
    case 'h':
    default:
      printf("%s -h(elp) -a(wgn on) -m mcs -n n_frames -s snr0 -z RXant \n",argv[0]);
      printf("-h This message\n");
      printf("-a Use AWGN channel and not multipath\n");
      printf("-m MCS for SL TB \n");
      printf("-n Number of SL periods to simulate\n");
      printf("-s Starting SNR, runs from SNR to SNR+%.1fdB in steps of %.1fdB. If n_frames is 1 then just SNR is simulated and MATLAB/OCTAVE output is generated\n", snr_int, snr_step);
      printf("-f step size of SNR, default value is 1.\n");
      printf("-r number of resource blocks\n");
      printf("-g [A:M] Use 3GPP 25.814 SCM-A/B/C/D('A','B','C','D') or 36-101 EPA('E'), EVA ('F'),ETU('G') models (ignores delay spread and Ricean factor), Rayghleigh8 ('H'), Rayleigh1('I'), Rayleigh1_corr('J'), Rayleigh1_anticorr ('K'), Rice8('L'), Rice1('M')\n");
      printf("-z Number of RX antennas used in UE\n");
      printf("-S Run SL synchronization procedures\n");
      exit(1);
      break;
    }
  }

  lte_param_init(NULL,
		 &UE,
		 NULL,
		 1, //nb_tx_port
		 1, //nb_tx_phy
		 1,
		 n_rx, //n_rx
		 1, //transmission_mode
		 0, // extended_prefix_flag
		 FDD,
		 0,
		 3,
		 N_RB_DL,
		 0, //pa
		 0, //threequart_fs
		 1, //osf
		 0, //perfect_ce
		 1,  //sidelink_active
		 1); //SLonly 

  UE2UE[0][0][0] = new_channel_desc_scm(UE->frame_parms.nb_antennas_tx,
					UE->frame_parms.nb_antennas_rx,
					channel_model,
					N_RB2sampling_rate(UE->frame_parms.N_RB_DL),
					N_RB2channel_bandwidth(UE->frame_parms.N_RB_DL),
					0.0,
					0,
			       0);


  // for a call to phy_reset_ue later we need PHY_vars_UE_g allocated and pointing to UE
  PHY_vars_UE_g = (PHY_VARS_UE***)malloc(sizeof(PHY_VARS_UE**));
  PHY_vars_UE_g[0] = (PHY_VARS_UE**) malloc(sizeof(PHY_VARS_UE*));
  PHY_vars_UE_g[0][0] = UE;

  
  init_lte_ue_transport(UE,0);

  for (int i=0;i<768*9;i++) slsch_payload[i] = taus()&255;
  for (int i=0;i<32;i++) sldch_payload[i] = taus()&255;

  if (do_SLSS==1) lte_sync_time_init(&UE->frame_parms);
  
  UE->rx_total_gain_dB = 120.0;
  UE->rx_offset = 0;
  UE->timing_advance = 0;
  UE->N_TA_offset = 0;
  UE->hw_timing_advance = 0;
  UE->slsch = &slsch;
  UE->sldch = &sldch;
  UE->slss  = &slss;

  // SLSCH/CCH Configuration
  slsch.N_SL_RB_data                   = 20;
  slsch.prb_Start_data                 = 5;
  slsch.prb_End_data                   = 44;
  slsch.N_SL_RB_SC                     = 4;
  slsch.prb_Start_SC                   = 5;
  slsch.prb_End_SC                     = 44;
  slsch.SL_SC_Period                   = 320;
  slsch.SubframeBitmapSL_length        = 4;
  slsch.SL_OffsetIndicator        = 0;
  slsch.SL_OffsetIndicator_data   = 0;
  // This can be 40,60,70,80,120,140,160,240,280,320
  slsch.SL_SC_Period              = 320;
  slsch.bitmap1                   = 0xffffffffff;
  // SCI Paramters and Payload
  slsch.n_pscch                   = 13;
  slsch.format                    = 0;
  slsch.freq_hopping_flag         = 0;
  slsch.resource_block_coding     = 127;
  slsch.time_resource_pattern     = 106; // all subframes for Nrp=8
  slsch.mcs                       = mcs;
  slsch.timing_advance_indication = 0;
  slsch.group_destination_id      = 0;

  // SLSCH parameters
  slsch.Nsb                       = 1; // can be 1,2,4
  slsch.N_RB_HO                   = 0; // 0-110
  slsch.n_ss_PSSCH                = 0; // TM 1 parameter
  slsch.n_ssf_PSSCH               = 0; // TM1 parameter
  slsch.cinit                     = 0;
  slsch.rvidx                     = 0; // changed dynamically below
  slsch.RB_start                  = 0;
  slsch.L_CRBs                    = nb_rb;
  slsch.payload_length            = get_TBS_UL(slsch.mcs,slsch.L_CRBs);
  slsch.payload                   = slsch_payload;
  // SLDCH Configuration
  sldch.type                      = disc_type1;
  sldch.N_SL_RB                   = 8;
  sldch.prb_Start                 = 15;
  sldch.prb_End                   = 34;
  sldch.offsetIndicator        = 1;
  /// 128 frame
  sldch.discPeriod                = 128;
  // 1 transmission per period
  sldch.numRepetitions            = 1;
  // 4 transmissions per SLDCH sdu
  sldch.numRetx                   = 3;
  // 16 TXops
  sldch.bitmap1                   = 0xffff;
  sldch.bitmap_length                   = 16;
  sldch.payload_length            = 256;
  memcpy((void*)sldch.payload,(void*)sldch_payload,32);
  memcpy((void*)&UE->sldch_rx,(void*)&sldch,sizeof(SLDCH_t));
  memset((void*)&slss,0,sizeof(slss));
  
  if (do_SLSS == 1) {
    slss.SL_OffsetIndicator         = 0;
    slss.slss_id                    = 170;
    UE->frame_parms.Nid_SL          = slss.slss_id;
    slss.slmib_length               = 5;
    slss.slmib[0]                   = 0;
    slss.slmib[1]                   = 1;
    slss.slmib[2]                   = 2;
    slss.slmib[3]                   = 3;
    slss.slmib[4]                   = 4;
  }
  // copy sidelink parameters, PSCCH and PSSCH payloads will get overwritten
  memcpy((void*)&UE->slsch_rx,(void*)UE->slsch,sizeof(SLSCH_t));
  
  for (int i=0;i<768*9;i++) slsch_payload[i] = taus()&255;
  for (int i=0;i<32;i++) sldch_payload[i] = taus()&255;

  // 0dBm transmit power for PSCCH = 0dBm - 10*log10(12) dBm/RE

  generate_sl_grouphop(UE);
  
  for (double snr=snr0;snr<snr+snr_int;snr+=snr_step) {
    printf("*****************SNR %f\n",snr);
    UE2UE[0][0][0]->path_loss_dB = -132.24 + snr + 10*log10(12);

    UE->slsch_errors=0;
    UE->slsch_txcnt = 0;
    UE->slsch_rxcnt[0] = 0;    UE->slsch_rxcnt[1] = 0;    UE->slsch_rxcnt[2] = 0;    UE->slsch_rxcnt[3] = 0;
    pscch_errors=0;
    proc.sl_fep_done = 0;
    UE->slbch_errors=0;
    
    for (trials = 0;trials<n_trials;trials++) {
      UE->pscch_coded=0;
      UE->pscch_generated=0;
      UE->psdch_generated=0;

      for (int absSF=0;absSF<10240;absSF++) {
	UE->slss_generated=0;
	frame = absSF/10;
	subframe= absSF%10;
        if (do_SLSS==0) {
	   check_and_generate_psdch(UE,frame,subframe);
/*	   UE->slsch_active = 1;
	   check_and_generate_pscch(UE,frame,subframe);*/
	   proc.subframe_tx = subframe;
	   proc.frame_tx    = frame;
//	   check_and_generate_pssch(UE,&proc,frame,subframe);
        }
	   check_and_generate_slss(UE,frame,subframe);
	if (UE->psdch_generated>0 || UE->pscch_generated > 0 || UE->pssch_generated > 0 || UE->slss_generated > 0) {
	  AssertFatal(UE->pscch_generated<3,"Illegal pscch_generated %d\n",UE->pscch_generated);
	  // FEP
	  ulsch_common_procedures(UE,proc.frame_tx,proc.subframe_tx,0);
	  write_output("txsig0SL.m","txs0",&UE->common_vars.txdata[0][UE->frame_parms.samples_per_tti*subframe],UE->frame_parms.samples_per_tti,1,1);
	  printf("Running do_SL_sig for frame %d subframe %d (%d,%d,%d,%d)\n",frame,subframe,UE->slss_generated,UE->pscch_generated,UE->psdch_generated,UE->pssch_generated);
	  do_SL_sig(0,UE2UE,subframe,UE->pscch_generated,
		    3*(UE->frame_parms.ofdm_symbol_size)+2*(UE->frame_parms.nb_prefix_samples)+UE->frame_parms.nb_prefix_samples0,
		    &UE->frame_parms,frame,0);
	  //	  write_output("rxsig0.m","rxs0",&UE->common_vars.rxdata[0][UE->frame_parms.samples_per_tti*subframe],UE->frame_parms.samples_per_tti,1,1);
	  if (do_SLSS==1) 
	    memcpy((void*)&UE->common_vars.rxdata_syncSL[0][2*tx_offset+(((frame&3)*10)+subframe)*2*UE->frame_parms.samples_per_tti],
		   (void*)&UE->common_vars.rxdata[0][subframe*UE->frame_parms.samples_per_tti],
		   2*UE->frame_parms.samples_per_tti*sizeof(int16_t));
	  //	  write_output("rxsyncb0.m","rxsyncb0",(void*)UE->common_vars.rxdata_syncSL[0],(UE->frame_parms.samples_per_tti),1,1);
	  //	  exit(-1);
          if (UE->psdch_generated==1) rx_sldch(UE,&proc,frame,subframe);

	  UE->pscch_generated = 0;
	  UE->pssch_generated = 0;
	  UE->psdch_generated = 0;
	}
/*
	rx_slcch(UE,&proc,frame,subframe);
	rx_slsch(UE,&proc,frame,subframe);
*/
      //  rx_sldch(UE,&proc,frame,subframe);
        if ((absSF % 40) == 3 && do_SLSS==1) {
	  printf("Running Initial synchronization for SL\n");
	  // initial synch for SL
	  initial_syncSL(UE);

	}
	  
	proc.sl_fep_done = 0;
	if ((absSF%320) == 319)  {
	  if (UE->slcch_received == 0) pscch_errors++;
	  break;
	}    
      
      }
      printf("SNR %f : pscch_errors %d, slsch_errors %d (%d.%d.%d.%d)\n",
	     snr,pscch_errors,UE->slsch_errors,
	     UE->slsch_rxcnt[0],
	     UE->slsch_rxcnt[1],
	     UE->slsch_rxcnt[2],
	     UE->slsch_rxcnt[3]);
    }
  }
}
