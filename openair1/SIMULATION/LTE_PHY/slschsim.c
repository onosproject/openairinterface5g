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



int nfapi_mode=0;
double cpuf;

int main(int argc, char **argv) {

  int frame,subframe;
  int trials;
  int n_trials=10;
  int N_RB_DL=50;
  int n_rx=1;
  channel_desc_t *UE2UE[2][2][2];
  PHY_VARS_UE *UE;
  int log_level = LOG_INFO;
  SLSCH_t slsch;
  SCM_t channel_model=AWGN;
  UE_rxtx_proc_t proc;
  double snr0 = 35;
  double snr_step=1;
  double snr_int=1;
  uint8_t slsch_payload[768*9];
  int mcs=10;
  int nb_rb=20;
  char channel_model_input[20];
  int pscch_errors=0;


  AssertFatal(load_configmodule(argc,argv) != NULL,
	      "cannot load configuration module, exiting\n");
  logInit();
  // enable these lines if you need debug info
  set_comp_log(PHY,LOG_INFO,LOG_HIGH,1);
  set_glog(log_level,LOG_HIGH);
  

  char c;

  while ((c = getopt (argc, argv, "hf:m:n:g:r:z:w:s:")) != -1) {
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
		 0); //perfect_ce

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

  UE->rx_total_gain_dB = 120.0;
  UE->rx_offset = 0;
  UE->timing_advance = 0;
  UE->N_TA_offset = 0;
  UE->hw_timing_advance = 0;
  UE->slsch = &slsch;
  // SL Configuration
  slsch.N_SL_RB                   = 20;
  slsch.prb_Start                 = 5;
  slsch.prb_End                   = 44;
  slsch.SL_OffsetIndicator        = 0;
  // This can be 40,60,70,80,120,140,160,240,280,320
  slsch.SL_SC_Period              = 320;
  slsch.bitmap1                   = 0xffffffffff;
  // SCI Paramters and Payload
  slsch.n_pscch                   = 1111;
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
  // copy sidelink parameters, PSCCH and PSSCH payloads will get overwritten
  memcpy((void*)&UE->slsch_rx,(void*)UE->slsch,sizeof(SLSCH_t));
  
  for (int i=0;i<768*9;i++) slsch_payload[i] = taus()&255;

  // 0dBm transmit power for PSCCH = 0dBm - 10*log10(12) dBm/RE

  for (double snr=snr0;snr<snr+snr_int;snr+=snr_step) {
    printf("*****************SNR %f\n",snr);
    UE2UE[0][0][0]->path_loss_dB = -132.24 + snr + 10*log10(12);

    UE->slsch_errors=0;
    UE->slsch_txcnt = 0;
    UE->slsch_rxcnt[0] = 0;    UE->slsch_rxcnt[1] = 0;    UE->slsch_rxcnt[2] = 0;    UE->slsch_rxcnt[3] = 0;
    pscch_errors=0;

    for (trials = 0;trials<n_trials;trials++) {
      UE->pscch_coded=0;
      UE->pscch_generated=0;
      for (int absSF=0;absSF<10240;absSF++) {
	frame = absSF/10;
	subframe= absSF%10;
	UE->slsch_active = 1;
	check_and_generate_pscch(UE,frame,subframe);
	proc.subframe_tx = subframe;
	proc.frame_tx    = frame;
	check_and_generate_pssch(UE,&proc,frame,subframe);
	if (UE->pscch_generated > 0 || UE->pssch_generated > 0) {
	  AssertFatal(UE->pscch_generated<3,"Illegal pscch_generated %d\n",UE->pscch_generated);
	  // FEP
	  ulsch_common_procedures(UE,&proc,0);
	  //	write_output("txsig0SL.m","txs0",&UE->common_vars.txdata[0][UE->frame_parms.samples_per_tti*subframe],UE->frame_parms.samples_per_tti,1,1);
	  printf("Running do_SL_sig for subframe %d, slot_ind %d\n",subframe,UE->pscch_generated);
	  do_SL_sig(0,UE2UE,subframe,UE->pscch_generated,&UE->frame_parms,frame,0);
	  //	write_output("rxsig0.m","rxs0",&UE->common_vars.rxdata[0][UE->frame_parms.samples_per_tti*subframe],UE->frame_parms.samples_per_tti,1,1);
	  UE->pscch_generated = 0;
	  UE->pssch_generated = 0;
	  
	  
	}
	rx_slcch(UE,frame,subframe);
	rx_slsch(UE,&proc,frame,subframe);
	
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
