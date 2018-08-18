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

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include "SIMULATION/TOOLS/sim.h"
#include "SIMULATION/RF/rf.h"
#include "PHY/types.h"
#include "PHY/defs_eNB.h"
#include "PHY/defs_UE.h"
#include "PHY/phy_vars.h"
#include "PHY/phy_vars_ue.h"
#include "SCHED/sched_common_vars.h"
#include "LAYER2/MAC/mac_vars.h"

#include "PHY/MODULATION/modulation_common.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/LTE_TRANSPORT/transport_proto.h"
#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"
#include "SCHED/sched_eNB.h"
#include "SCHED_UE/sched_UE.h"
#include "common/config/config_load_configmodule.h"
#include "PHY/INIT/phy_init.h"

#ifdef XFORMS
#include "PHY/TOOLS/lte_phy_scope.h"
#endif //XFORMS


#include "OCG_vars.h"
#include "unitary_defs.h"
#include "dummy_functions.c"



void feptx_ofdm(RU_t *ru);
void feptx_prec(RU_t *ru);

double cpuf;



uint16_t m_rnti=0x1234;
int nfapi_mode=0;
int codingw = 0;
int emulate_rf = 0;

void DL_channel(RU_t *ru,PHY_VARS_UE *UE,uint subframe,int awgn_flag,double SNR, int tx_lev,int hold_channel,int trials, 
		  channel_desc_t *eNB2UE,
		  double *s_re[2],double *s_im[2],double *r_re[2],double *r_im[2]) {

  int i,u;
  int aa,aarx,aatx;
  double channelx,channely;
  double sigma2_dB,sigma2;
  double iqim=0.0;

  //    printf("Copying tx ..., nsymb %d (n_tx %d), awgn %d\n",nsymb,eNB->frame_parms.nb_antennas_tx,awgn_flag);
  for (i=0; i<2*UE->frame_parms.samples_per_tti; i++) {
    for (aa=0; aa<ru->frame_parms.nb_antennas_tx; aa++) {
      if (awgn_flag == 0) {
	s_re[aa][i] = ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) + (i<<1)]);
	s_im[aa][i] = ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) +(i<<1)+1]);


      } else {
	for (aarx=0; aarx<UE->frame_parms.nb_antennas_rx; aarx++) {
	  if (aa==0) {
	    r_re[aarx][i] = ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) +(i<<1)]);
	    r_im[aarx][i] = ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) +(i<<1)+1]);
	  } else {
	    r_re[aarx][i] += ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) +(i<<1)]);
	    r_im[aarx][i] += ((double)(((short *)ru->common.txdata[aa]))[(2*subframe*UE->frame_parms.samples_per_tti) +(i<<1)+1]);
	  }

	}
      }
    }
  }

  // Multipath channel
  if (awgn_flag == 0) {
    multipath_channel(eNB2UE,s_re,s_im,r_re,r_im,
		      2*UE->frame_parms.samples_per_tti,hold_channel);



    if (UE->perfect_ce==1) {
      // fill in perfect channel estimates
      freq_channel(eNB2UE,UE->frame_parms.N_RB_DL,12*UE->frame_parms.N_RB_DL + 1);
    }
  }


  //AWGN
  // tx_lev is the average energy over the whole subframe
  // but SNR should be better defined wrt the energy in the reference symbols
  sigma2_dB = 10*log10((double)tx_lev) +10*log10((double)ru->frame_parms.ofdm_symbol_size/(double)(ru->frame_parms.N_RB_DL*12)) - SNR;
  sigma2 = pow(10,sigma2_dB/10);

  for (i=0; i<2*UE->frame_parms.samples_per_tti; i++) {
    for (aa=0; aa<UE->frame_parms.nb_antennas_rx; aa++) {
      //printf("s_re[0][%d]=> %f , r_re[0][%d]=> %f\n",i,s_re[aa][i],i,r_re[aa][i]);
      ((short*) UE->common_vars.rxdata[aa])[(2*subframe*UE->frame_parms.samples_per_tti)+2*i] =
	(short) (r_re[aa][i] + sqrt(sigma2/2)*gaussdouble(0.0,1.0));
      ((short*) UE->common_vars.rxdata[aa])[(2*subframe*UE->frame_parms.samples_per_tti)+2*i+1] =
	(short) (r_im[aa][i] + (iqim*r_re[aa][i]) + sqrt(sigma2/2)*gaussdouble(0.0,1.0));
    }
  }
}


uint16_t
fill_tx_req(nfapi_tx_request_body_t *tx_req_body,
	    uint16_t                absSF,
	    uint16_t                pdu_length,
	    uint16_t                pdu_index,
	    uint8_t                 *pdu)
{
  nfapi_tx_request_pdu_t *TX_req = &tx_req_body->tx_pdu_list[tx_req_body->number_of_pdus];
  LOG_D(MAC, "Filling TX_req %d for pdu length %d\n",
	tx_req_body->number_of_pdus, pdu_length);

  TX_req->pdu_length                 = pdu_length;
  TX_req->pdu_index                  = pdu_index;
  TX_req->num_segments               = 1;
  TX_req->segments[0].segment_length = pdu_length;
  TX_req->segments[0].segment_data   = pdu;
  tx_req_body->tl.tag                = NFAPI_TX_REQUEST_BODY_TAG;
  tx_req_body->number_of_pdus++;

  return (((absSF / 10) << 4) + (absSF % 10));
}

void fill_MCH_config(PHY_VARS_eNB *eNB,
		     int frame,
		     int subframe,
		     Sched_Rsp_t *sched_resp,
		     uint8_t input_buffer[20000],
		     int mcs,
		     int mbsfn_id,
		     int m_rnti,
		     int NB_RB,
		     int MCH_RB_ALLOC) {


  nfapi_dl_config_request_body_t *dl_req=&sched_resp->DL_req->dl_config_request_body;
  nfapi_tx_request_body_t        *TX_req=&sched_resp->TX_req->tx_request_body;

  dl_req->number_dci=0;
  dl_req->number_pdu=0;
  TX_req->number_of_pdus=0;
  dl_req->tl.tag = NFAPI_DL_CONFIG_REQUEST_BODY_TAG;

  nfapi_dl_config_request_pdu_t *dl_config_pdu = &dl_req->dl_config_pdu_list[dl_req->number_pdu];

  memset((void *) dl_config_pdu, 0,
	 sizeof(nfapi_dl_config_request_pdu_t));

  dl_config_pdu->pdu_type                                                    = NFAPI_DL_CONFIG_MCH_PDU_TYPE;
  dl_config_pdu->pdu_size                                                    = (uint8_t) (2 + sizeof(nfapi_dl_config_mch_pdu));

  dl_config_pdu->mch_pdu.mch_pdu_rel8.tl.tag                                 = NFAPI_DL_CONFIG_REQUEST_MCH_PDU_REL8_TAG;
  dl_config_pdu->mch_pdu.mch_pdu_rel8.length                                 = get_TBS_DL(mcs,NB_RB);
  dl_config_pdu->mch_pdu.mch_pdu_rel8.pdu_index                              = 0;
  dl_config_pdu->mch_pdu.mch_pdu_rel8.rnti                                   = m_rnti;
  dl_config_pdu->mch_pdu.mch_pdu_rel8.resource_allocation_type               = 0;
  dl_config_pdu->mch_pdu.mch_pdu_rel8.resource_block_coding                  = MCH_RB_ALLOC;
  dl_config_pdu->mch_pdu.mch_pdu_rel8.modulation                             = get_Qm(mcs);
  dl_config_pdu->mch_pdu.mch_pdu_rel8.mbsfn_area_id                          = mbsfn_id;
  dl_req->number_pdu++;
  fill_tx_req(TX_req,
	      (frame * 10) + subframe,
	      get_TBS_DL(mcs,NB_RB),
	      0,
	      input_buffer);
 
  eNB->frame_parms.Nid_cell_mbsfn = mbsfn_id;
}


int main(int argc, char **argv)
{

  char c;

  int i,l,l2,aa,aarx,k;
  double sigma2, sigma2_dB=0,SNR,snr0=-2.0,snr1=0.0;
  uint8_t snr1set=0;
  double snr_step=1,input_snr_step=1;
  int **txdata;
  double s_re0[2*30720],s_im0[2*30720],s_re1[2*30720],s_im1[2*30720];
  double r_re0[2*30720],r_im0[2*30720],r_re1[2*30720],r_im1[2*30720];
  double *s_re[2]={s_re0,s_re1},*s_im[2]={s_im0,s_im1},*r_re[2]={r_re0,r_re1},*r_im[2]={r_im0,r_im1};
  double iqim = 0.0;
  int subframe=1;
  char fname[40];//, vname[40];
  uint8_t transmission_mode = 1,n_tx=1,n_rx=2;
  uint16_t Nid_cell=0;
  uint16_t mbsfn_id=0;

  FILE *fd;

  int eNB_id = 0;
  unsigned char mcs=0,awgn_flag=0,round;

  int n_frames=1;
  channel_desc_t *eNB2UE;
  uint32_t nsymb,tx_lev,tx_lev_dB;
  uint8_t extended_prefix_flag=1;
  LTE_DL_FRAME_PARMS *frame_parms;
  int hold_channel=0;


  uint16_t NB_RB=25;
  int MCH_RB_ALLOC = 0;

  int tdd_config=3;

  SCM_t channel_model=MBSFN;
  PHY_VARS_eNB *eNB;
  RU_t *ru;
  PHY_VARS_UE *UE;

  nfapi_dl_config_request_t DL_req;
  nfapi_ul_config_request_t UL_req;
  nfapi_hi_dci0_request_t HI_DCI0_req;
  nfapi_dl_config_request_pdu_t dl_config_pdu_list[MAX_NUM_DL_PDU];
  nfapi_tx_request_pdu_t tx_pdu_list[MAX_NUM_TX_REQUEST_PDU];
  nfapi_tx_request_t TX_req;
  Sched_Rsp_t sched_resp;

  unsigned char *input_buffer;
  unsigned short input_buffer_length;
  unsigned int ret;

  unsigned int trials;

  uint8_t N_RB_DL=25,osf=1;
  uint32_t perfect_ce = 0;

  lte_frame_type_t frame_type = FDD;

  uint32_t Nsoft = 1827072;

  switch (N_RB_DL) {
  case 6:
    MCH_RB_ALLOC = 0x3f;

    break;
    
  case 25:
    MCH_RB_ALLOC = 0x1fff;
    break;
    
  case 50:
    MCH_RB_ALLOC = 0x1ffff;
    break;
    
  case 100:
    MCH_RB_ALLOC = 0x1ffffff;
    break;
  }
  
  NB_RB=conv_nprb(0,MCH_RB_ALLOC,N_RB_DL);
  

  /*
    #ifdef XFORMS
  FD_lte_phy_scope_ue *form_ue;
  char title[255];

  fl_initialize (&argc, argv, NULL, 0, 0);
  form_ue = create_lte_phy_scope_ue();
  sprintf (title, "LTE DL SCOPE UE");
  fl_show_form (form_ue->lte_phy_scope_ue, FL_PLACE_HOTSPOT, FL_FULLBORDER, title);
#endif
  */

  cpuf = get_cpu_freq_GHz();

  memset((void*)&sched_resp,0,sizeof(sched_resp));
  sched_resp.DL_req = &DL_req;
  sched_resp.UL_req = &UL_req;
  sched_resp.HI_DCI0_req = &HI_DCI0_req;
  sched_resp.TX_req = &TX_req;
  memset((void*)&DL_req,0,sizeof(DL_req));
  memset((void*)&UL_req,0,sizeof(UL_req));
  memset((void*)&HI_DCI0_req,0,sizeof(HI_DCI0_req));
  memset((void*)&TX_req,0,sizeof(TX_req));

  DL_req.dl_config_request_body.dl_config_pdu_list = dl_config_pdu_list;
  TX_req.tx_request_body.tx_pdu_list = tx_pdu_list;


  while ((c = getopt (argc, argv, "ahA:Cp:n:s:S:t:x:y:z:N:F:R:O:dm:i:Y")) != -1) {
    switch (c) {
    case 'a':
      awgn_flag=1;
      break;

    case 'd':
      frame_type = 0;
      break;

    case 'n':
      n_frames = atoi(optarg);
      break;

    case 'm':
      mcs=atoi(optarg);
      break;

    case 's':
      snr0 = atof(optarg);
      msg("Setting SNR0 to %f\n",snr0);
      break;

    case 'i':
      input_snr_step = atof(optarg);
      break;

    case 'S':
      snr1 = atof(optarg);
      snr1set=1;
      msg("Setting SNR1 to %f\n",snr1);
      break;

    case 'p': // subframe no;
      subframe=atoi(optarg);
      break;

    case 'z':
      n_rx=atoi(optarg);

      if ((n_rx==0) || (n_rx>2)) {
        msg("Unsupported number of rx antennas %d\n",n_rx);
        exit(-1);
      }

      break;

    case 'N':
      Nid_cell = atoi(optarg);
      break;

    case 'R':
      N_RB_DL = atoi(optarg);

      if ((N_RB_DL!=6) && (N_RB_DL!=25) && (N_RB_DL!=50) && (N_RB_DL!=100))  {
        printf("Unsupported Bandwidth %d\n",N_RB_DL);
        exit(-1);
      }

      break;

    case 'O':
      osf = atoi(optarg);
      break;

    case 'Y':
      perfect_ce = 1;
      break;

    default:
    case 'h':
      printf("%s -h(elp) -p(subframe) -N cell_id -g channel_model -n n_frames -t Delayspread -s snr0 -S snr1 -i snr increment -z RXant \n",argv[0]);
      printf("-h This message\n");
      printf("-a Use AWGN Channel\n");
      printf("-p Use extended prefix mode\n");
      printf("-d Use TDD\n");
      printf("-n Number of frames to simulate\n");
      printf("-s Starting SNR, runs from SNR0 to SNR0 + 5 dB.  If n_frames is 1 then just SNR is simulated\n");
      printf("-S Ending SNR, runs from SNR0 to SNR1\n");
      printf("-t Delay spread for multipath channel\n");
      printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
      printf("-x Transmission mode (1,2,6 for the moment)\n");
      printf("-y Number of TX antennas used in eNB\n");
      printf("-z Number of RX antennas used in UE\n");
      printf("-i Relative strength of first intefering eNB (in dB) - cell_id mod 3 = 1\n");
      printf("-j Relative strength of second intefering eNB (in dB) - cell_id mod 3 = 2\n");
      printf("-N Nid_cell\n");
      printf("-R N_RB_DL\n");
      printf("-O oversampling factor (1,2,4,8,16)\n");
      printf("-A Interpolation_filname Run with Abstraction to generate Scatter plot using interpolation polynomial in file\n");
      printf("-C Generate Calibration information for Abstraction (effective SNR adjustment to remove Pe bias w.r.t. AWGN)\n");
      printf("-f Output filename (.txt format) for Pe/SNR results\n");
      printf("-F Input filename (.txt format) for RX conformance testing\n");
      exit (-1);
      break;
    }
  }



  if (awgn_flag == 1)
    channel_model = AWGN;

  // check that subframe is legal for eMBMS

  if ((subframe == 0) || (subframe == 5) ||    // TDD and FDD SFn 0,5;
      ((frame_type == FDD) && ((subframe == 4) || (subframe == 9))) || // FDD SFn 4,9;
      ((frame_type == TDD ) && ((subframe<3) || (subframe==6))))    {  // TDD SFn 1,2,6;

    printf("Illegal subframe %d for eMBMS transmission (frame_type %d)\n",subframe,frame_type);
    exit(-1);
  }

  if (transmission_mode==2)
    n_tx=2;

  AssertFatal(load_configmodule(argc,argv) != NULL,
	      "cannot load configuration module, exiting\n");
  logInit();
  set_glog(LOG_INFO);

  RC.nb_L1_inst = 1;
  RC.nb_RU = 1;

  lte_param_init(&eNB,&UE,&ru,
		 n_tx,
                 n_tx,
		 1,
		 n_rx,
		 transmission_mode,
		 1,
		 FDD,
		 Nid_cell,
		 tdd_config,
		 N_RB_DL,
		 0,
		 0,
		 osf,
		 perfect_ce);

  RC.eNB = (PHY_VARS_eNB ***)malloc(sizeof(PHY_VARS_eNB **));
  RC.eNB[0] = (PHY_VARS_eNB **)malloc(sizeof(PHY_VARS_eNB *));
  RC.ru = (RU_t **)malloc(sizeof(RC.ru));
  RC.eNB[0][0] = eNB;
  RC.ru[0] = ru;

  if (snr1set==0) {
    if (n_frames==1)
      snr1 = snr0+.1;
    else
      snr1 = snr0+5.0;
  }

  printf("SNR0 %f, SNR1 %f\n",snr0,snr1);

  frame_parms = &eNB->frame_parms;

  if (awgn_flag == 0)
    sprintf(fname,"embms_%d_%d.m",mcs,N_RB_DL);
  else
    sprintf(fname,"embms_awgn_%d_%d.m",mcs,N_RB_DL);

  if (!(fd = fopen(fname,"w"))) {
    printf("Cannot open %s, check permissions\n",fname);
    exit(-1);
  }
	

  if (awgn_flag==0)
    fprintf(fd,"SNR_%d_%d=[];errs_mch_%d_%d=[];mch_trials_%d_%d=[];\n",
            mcs,N_RB_DL,
            mcs,N_RB_DL,
            mcs,N_RB_DL);
  else
    fprintf(fd,"SNR_awgn_%d_%d=[];errs_mch_awgn_%d_%d=[];mch_trials_awgn_%d_%d=[];\n",
            mcs,N_RB_DL,
            mcs,N_RB_DL,
            mcs,N_RB_DL);

  fflush(fd);


  nsymb = 12;

  printf("FFT Size %d, Extended Prefix %d, Samples per subframe %d, Symbols per subframe %d, AWGN %d\n",NUMBER_OF_OFDM_CARRIERS,
         frame_parms->Ncp,frame_parms->samples_per_tti,nsymb,awgn_flag);

  eNB2UE = new_channel_desc_scm(eNB->frame_parms.nb_antennas_tx,
                                UE->frame_parms.nb_antennas_rx,
                                channel_model,
				N_RB2sampling_rate(eNB->frame_parms.N_RB_DL),
				N_RB2channel_bandwidth(eNB->frame_parms.N_RB_DL),
                                0,
                                0,
                                0);

  // Create transport channel structures for 2 transport blocks (MIMO)
  eNB->dlsch_MCH = new_eNB_dlsch(1,8,Nsoft,N_RB_DL,0,&eNB->frame_parms);

  if (!eNB->dlsch_MCH) {
    printf("Can't get eNB dlsch structures\n");
    exit(-1);
  }

  UE->dlsch_MCH[0]  = new_ue_dlsch(1,8,Nsoft,MAX_TURBO_ITERATIONS_MBSFN,N_RB_DL,0);

  eNB->frame_parms.num_MBSFN_config = 1;
  eNB->frame_parms.MBSFN_config[0].radioframeAllocationPeriod = 0;
  eNB->frame_parms.MBSFN_config[0].radioframeAllocationOffset = 0;
  eNB->frame_parms.MBSFN_config[0].fourFrames_flag = 0;
  eNB->frame_parms.MBSFN_config[0].mbsfn_SubframeConfig=0xff; // activate all possible subframes
  UE->frame_parms.num_MBSFN_config = 1;
  UE->frame_parms.MBSFN_config[0].radioframeAllocationPeriod = 0;
  UE->frame_parms.MBSFN_config[0].radioframeAllocationOffset = 0;
  UE->frame_parms.MBSFN_config[0].fourFrames_flag = 0;
  UE->frame_parms.MBSFN_config[0].mbsfn_SubframeConfig=0xff; // activate all possible subframes

  //  fill_eNB_dlsch_MCH(eNB,mcs,1,0);
  fill_UE_dlsch_MCH(UE,mcs,1,0,0);
  UE->frame_parms.Nid_cell_mbsfn = mbsfn_id;

  eNB_rxtx_proc_t *proc_eNB = &eNB->proc.proc_rxtx[0];//UE->current_thread_id[subframe]];

  if (is_pmch_subframe(0,subframe,&eNB->frame_parms)==0) {
    printf("eNB is not configured for MBSFN in subframe %d\n",subframe);
    exit(-1);
  } else if (is_pmch_subframe(0,subframe,&UE->frame_parms)==0) {
    printf("UE is not configured for MBSFN in subframe %d\n",subframe);
    exit(-1);
  }


  input_buffer_length = eNB->dlsch_MCH->harq_processes[0]->TBS/8;
  input_buffer = (unsigned char *)malloc(input_buffer_length+4);
  memset(input_buffer,0,input_buffer_length+4);

  for (i=0; i<input_buffer_length+4; i++) {
    input_buffer[i]= (unsigned char)(taus()&0xff);
  }


  fill_MCH_config(eNB,0,1,&sched_resp,input_buffer,mcs,mbsfn_id,m_rnti,NB_RB,MCH_RB_ALLOC);

  snr_step = input_snr_step;

  for (SNR=snr0; SNR<snr1; SNR+=snr_step) {
    UE->proc.proc_rxtx[0].frame_tx=0;
    UE->proc.proc_rxtx[UE->current_thread_id[subframe]].frame_rx=0;
    eNB->proc.proc_rxtx[0].frame_tx=0;
    eNB->proc.proc_rxtx[0].subframe_tx=subframe;

    UE->dlsch_mtch_errors[0][0]=0;

    printf("********************** SNR %f (step %f)\n",SNR,snr_step);

    for (trials = 0; trials<n_frames; trials++) {
      //        printf("Trial %d\n",trials);
      fflush(stdout);
      round=0;


      DL_req.sfn_sf = (proc_eNB->frame_tx<<4)+subframe;
      TX_req.sfn_sf = (proc_eNB->frame_tx<<4)+subframe;

      proc_eNB->subframe_tx = subframe;
      sched_resp.subframe=subframe;
      sched_resp.frame=proc_eNB->frame_tx;
      
      eNB->abstraction_flag=0;
      schedule_response(&sched_resp);
      phy_procedures_eNB_TX(eNB,proc_eNB,1);
      
      ru->proc.subframe_tx=(subframe+1)%10;
      feptx_prec(ru);
      feptx_ofdm(ru);
      
      
      proc_eNB->frame_tx++;
      
      
      if (n_frames==1) {
        LOG_M("txsigF0.m","txsF0", &eNB->common_vars.txdataF[0][subframe*nsymb*eNB->frame_parms.ofdm_symbol_size],
                     nsymb*eNB->frame_parms.ofdm_symbol_size,1,1);
        //if (eNB->frame_parms.nb_antennas_tx>1)
        //LOG_M("txsigF1.m","txsF1", &eNB->lte_eNB_common_vars.txdataF[eNB_id][1][subframe*nsymb*eNB->frame_parms.ofdm_symbol_size],nsymb*eNB->frame_parms.ofdm_symbol_size,1,1);
      }

      tx_lev = 0;

      for (aa=0; aa<eNB->frame_parms.nb_antennas_tx; aa++) {
        tx_lev += signal_energy(&ru->common.txdata[aa]
                                [subframe*eNB->frame_parms.samples_per_tti],
                                eNB->frame_parms.samples_per_tti);
      }


      tx_lev_dB = (unsigned int) dB_fixed(tx_lev);

      if (n_frames==1) {
        printf("tx_lev = %d (%d dB)\n",tx_lev,tx_lev_dB);
        //    LOG_M("txsig0.m","txs0", &eNB->common_vars.txdata[0][0][subframe* eNB->frame_parms.samples_per_tti],

        //     eNB->frame_parms.samples_per_tti,1,1);
      }

      DL_channel(ru,UE,subframe,awgn_flag,SNR,tx_lev,hold_channel,trials,eNB2UE,s_re,s_im,r_re,r_im);
      
      
      UE_rxtx_proc_t *proc = &UE->proc.proc_rxtx[UE->current_thread_id[subframe]];
      proc->subframe_rx = subframe;
      UE->UE_mode[0] = PUSCH;
      
      slot_fep_mbsfn(UE,
		     0,
		     proc->subframe_rx,
		     UE->rx_offset,
		     0);

      if (n_frames==1) printf("Running phy_procedures_UE_RX\n");
      phy_procedures_UE_RX(UE,proc,0,0,0,normal_txrx);


      UE->proc.proc_rxtx[0].frame_tx++;
      eNB->proc.proc_rxtx[0].frame_tx++;

    }

    printf("errors %d/%d (Pe %e)\n",UE->dlsch_mtch_errors[0][0],trials,(double)UE->dlsch_mtch_errors[0][0]/trials);

    if (awgn_flag==0)
      fprintf(fd,"SNR_%d_%d = [SNR_%d_%d %f]; errs_mch_%d_%d =[errs_mch_%d_%d  %d]; mch_trials_%d_%d =[mch_trials_%d_%d  %d];\n",
              mcs,N_RB_DL,mcs,N_RB_DL,SNR,
              mcs,N_RB_DL,mcs,N_RB_DL,UE->dlsch_mtch_errors[0][0],
              mcs,N_RB_DL,mcs,N_RB_DL,trials);
    else
      fprintf(fd,"SNR_awgn_%d = [SNR_awgn_%d %f]; errs_mch_awgn_%d =[errs_mch_awgn_%d  %d]; mch_trials_awgn_%d =[mch_trials_awgn_%d %d];\n",
              N_RB_DL,N_RB_DL,SNR,
              N_RB_DL,N_RB_DL,UE->dlsch_mtch_errors[0][0],
              N_RB_DL,N_RB_DL,trials);

    fflush(fd);

    if (UE->dlsch_mtch_errors[0][0] == 0)
      break;
  }


  if (n_frames==1) {
    printf("Dumping PMCH files ( G %d)\n",UE->dlsch_MCH[0]->harq_processes[0]->G);
    dump_mch(UE,0,
             UE->dlsch_MCH[0]->harq_processes[0]->G,
             subframe);
  }

  printf("Freeing dlsch structures\n");
  free_eNB_dlsch(eNB->dlsch_MCH);
  free_ue_dlsch(UE->dlsch_MCH[0]);

  fclose(fd);

  return(0);
}

