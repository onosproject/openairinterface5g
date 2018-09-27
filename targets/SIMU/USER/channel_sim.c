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
 *-------------------------------------------------------------------------------
 * Optimization using SIMD instructions
 * Frecuency Domain Analysis
 * Luis Felipe Ariza Vesga, email:lfarizav@unal.edu.co
 * Functions: do_DL_sig_freq(), do_UL_sig_freq(), init_channel_vars_freq(),
 * do_UL_sig_freq_prach.
 *-------------------------------------------------------------------------------
 */

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "SIMULATION/TOOLS/defs.h"
#include "SIMULATION/RF/defs.h"
#include "PHY/types.h"
#include "PHY/defs.h"
#include "PHY/extern.h"

#ifdef OPENAIR2
#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"
#include "UTIL/LOG/log_if.h"
#include "UTIL/LOG/log_extern.h"
#include "RRC/LITE/extern.h"
#include "PHY_INTERFACE/extern.h"
#include "UTIL/OCG/OCG.h"
#include "UTIL/OPT/opt.h" // to test OPT
#endif

#include "UTIL/FIFO/types.h"

#ifdef IFFT_FPGA
#include "PHY/LTE_REFSIG/mod_table.h"
#endif

#include "SCHED/defs.h"
#include "SCHED/extern.h"

#ifdef XFORMS
#include "forms.h"
#include "phy_procedures_sim_form.h"
#endif

#include "oaisim.h"

#define RF
//#define DEBUG_SIM

int number_rb_ul;
int first_rbUL ;

double r_re_DL[NUMBER_OF_UE_MAX][2][30720];
double r_im_DL[NUMBER_OF_UE_MAX][2][30720];
double r_re_UL[NUMBER_OF_eNB_MAX][2][30720];
double r_im_UL[NUMBER_OF_eNB_MAX][2][30720];

#ifdef    __AVX2__
float r_re_DL_f[NUMBER_OF_UE_MAX][2][2048*14];
float r_im_DL_f[NUMBER_OF_UE_MAX][2][2048*14];
float r_re_UL_f[NUMBER_OF_eNB_MAX][2][2048*14];
float r_im_UL_f[NUMBER_OF_eNB_MAX][2][2048*14];
float r_re_UL_f_prach[NUMBER_OF_eNB_MAX][2][2048*14];
float r_im_UL_f_prach[NUMBER_OF_eNB_MAX][2][2048*14];
#else
double r_re_DL_f[NUMBER_OF_UE_MAX][2][2048*14];
double r_im_DL_f[NUMBER_OF_UE_MAX][2][2048*14];
double r_re_UL_f[NUMBER_OF_eNB_MAX][2][2048*14];
double r_im_UL_f[NUMBER_OF_eNB_MAX][2][2048*14];
double r_re_UL_f_prach[NUMBER_OF_eNB_MAX][2][2048*14];
double r_im_UL_f_prach[NUMBER_OF_eNB_MAX][2][2048*14];
#endif

int eNB_output_mask[NUMBER_OF_UE_MAX];
int UE_output_mask[NUMBER_OF_eNB_MAX];
pthread_mutex_t eNB_output_mutex[NUMBER_OF_UE_MAX];
pthread_mutex_t UE_output_mutex[NUMBER_OF_eNB_MAX];
pthread_mutex_t UE_PRACH_output_mutex[NUMBER_OF_eNB_MAX];

void do_DL_sig(channel_desc_t *eNB2UE[NUMBER_OF_eNB_MAX][NUMBER_OF_UE_MAX][MAX_NUM_CCs],
	       node_desc_t *enb_data[NUMBER_OF_eNB_MAX],
	       node_desc_t *ue_data[NUMBER_OF_UE_MAX],
	       uint16_t subframe,uint8_t abstraction_flag,LTE_DL_FRAME_PARMS *frame_parms,
	       uint8_t UE_id,
	       int CC_id)
{

  int32_t att_eNB_id=-1;
  int32_t **txdata,**rxdata;

  uint8_t eNB_id=0;
  double tx_pwr;
  double rx_pwr;
  int32_t rx_pwr2;
  uint32_t i,aa;
  uint32_t sf_offset;

  double min_path_loss=-200;
  uint8_t hold_channel=0;
  uint8_t nb_antennas_rx = eNB2UE[0][0][CC_id]->nb_rx; // number of rx antennas at UE
  uint8_t nb_antennas_tx = eNB2UE[0][0][CC_id]->nb_tx; // number of tx antennas at eNB

  double s_re0[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double s_re1[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *s_re[2];
  double s_im0[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double s_im1[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *s_im[2];
  double r_re00[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double r_re01[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *r_re0[2];
  double r_im00[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double r_im01[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *r_im0[2];

  s_re[0] = s_re0;
  s_im[0] = s_im0;
  s_re[1] = s_re1;
  s_im[1] = s_im1;

  r_re0[0] = r_re00;
  r_im0[0] = r_im00;
  r_re0[1] = r_re01;
  r_im0[1] = r_im01;

  if (subframe==0)
    hold_channel = 0;
  else
    hold_channel = 1;

  if (abstraction_flag != 0) {
    //for (UE_id=0;UE_id<NB_UE_INST;UE_id++) {

    if (!hold_channel) {
      // calculate the random channel from each eNB
      for (eNB_id=0; eNB_id<NB_eNB_INST; eNB_id++) {

        random_channel(eNB2UE[eNB_id][UE_id][CC_id],abstraction_flag);
        /*
        for (i=0;i<eNB2UE[eNB_id][UE_id]->nb_taps;i++)
        printf("eNB2UE[%d][%d]->a[0][%d] = (%f,%f)\n",eNB_id,UE_id,i,eNB2UE[eNB_id][UE_id]->a[0][i].x,eNB2UE[eNB_id][UE_id]->a[0][i].y);
        */
        freq_channel(eNB2UE[eNB_id][UE_id][CC_id], frame_parms->N_RB_DL,frame_parms->N_RB_DL*12+1);
      }

      // find out which eNB the UE is attached to
      for (eNB_id=0; eNB_id<NB_eNB_INST; eNB_id++) {
        if (find_ue(PHY_vars_UE_g[UE_id][CC_id]->pdcch_vars[0][0]->crnti,PHY_vars_eNB_g[eNB_id][CC_id])>=0) {
          // UE with UE_id is connected to eNb with eNB_id
          att_eNB_id=eNB_id;
          LOG_D(OCM,"A: UE attached to eNB (UE%d->eNB%d)\n",UE_id,eNB_id);
        }
      }

      // if UE is not attached yet, find assume its the eNB with the smallest pathloss
      if (att_eNB_id<0) {
        for (eNB_id=0; eNB_id<NB_eNB_INST; eNB_id++) {
          if (min_path_loss<eNB2UE[eNB_id][UE_id][CC_id]->path_loss_dB) {
            min_path_loss = eNB2UE[eNB_id][UE_id][CC_id]->path_loss_dB;
            att_eNB_id=eNB_id;
            LOG_D(OCM,"B: UE attached to eNB (UE%d->eNB%d)\n",UE_id,eNB_id);
          }
        }
      }

      if (att_eNB_id<0) {
        LOG_E(OCM,"Cannot find eNB for UE %d, return\n",UE_id);
        return; //exit(-1);
      }

#ifdef DEBUG_SIM
      rx_pwr = signal_energy_fp2(eNB2UE[att_eNB_id][UE_id][CC_id]->ch[0],
                                 eNB2UE[att_eNB_id][UE_id][CC_id]->channel_length)*eNB2UE[att_eNB_id][UE_id][CC_id]->channel_length;
      LOG_D(OCM,"Channel (CCid %d) eNB %d => UE %d : tx_power %d dBm, path_loss %f dB\n",
            CC_id,att_eNB_id,UE_id,
            frame_parms->pdsch_config_common.referenceSignalPower,
            eNB2UE[att_eNB_id][UE_id][CC_id]->path_loss_dB);
#endif

      //dlsch_abstraction(PHY_vars_UE_g[UE_id]->sinr_dB, rb_alloc, 8);
      // fill in perfect channel estimates
      channel_desc_t *desc1 = eNB2UE[att_eNB_id][UE_id][CC_id];
      int32_t **dl_channel_est = PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].dl_ch_estimates[0];
      //      double scale = pow(10.0,(enb_data[att_eNB_id]->tx_power_dBm + eNB2UE[att_eNB_id][UE_id]->path_loss_dB + (double) PHY_vars_UE_g[UE_id]->rx_total_gain_dB)/20.0);
      double scale = pow(10.0,(frame_parms->pdsch_config_common.referenceSignalPower+eNB2UE[att_eNB_id][UE_id][CC_id]->path_loss_dB + (double) PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB)/20.0);
      LOG_D(OCM,"scale =%lf (%d dB)\n",scale,(int) (20*log10(scale)));
      // freq_channel(desc1,frame_parms->N_RB_DL,nb_samples);
      //write_output("channel.m","ch",desc1->ch[0],desc1->channel_length,1,8);
      //write_output("channelF.m","chF",desc1->chF[0],nb_samples,1,8);
      int count,count1,a_rx,a_tx;

      for(a_tx=0; a_tx<nb_antennas_tx; a_tx++) {
        for (a_rx=0; a_rx<nb_antennas_rx; a_rx++) {
          //for (count=0;count<frame_parms->symbols_per_tti/2;count++)
          for (count=0; count<1; count++) {
            for (count1=0; count1<frame_parms->N_RB_DL*12; count1++) {
              ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count1+(count*frame_parms->ofdm_symbol_size+LTE_CE_FILTER_LENGTH)*2]=(int16_t)(desc1->chF[a_rx+(a_tx*nb_antennas_rx)][count1].x*scale);
              ((int16_t *) dl_channel_est[(a_tx<<1)+a_rx])[2*count1+1+(count*frame_parms->ofdm_symbol_size+LTE_CE_FILTER_LENGTH)*2]=(int16_t)(desc1->chF[a_rx+(a_tx*nb_antennas_rx)][count1].y*scale) ;
            }
          }
        }
      }

      // calculate the SNR for the attached eNB (this assumes eNB always uses PMI stored in eNB_UE_stats; to be improved)
      init_snr(eNB2UE[att_eNB_id][UE_id][CC_id], enb_data[att_eNB_id], ue_data[UE_id], PHY_vars_UE_g[UE_id][CC_id]->sinr_dB, &PHY_vars_UE_g[UE_id][CC_id]->N0,
               PHY_vars_UE_g[UE_id][CC_id]->transmission_mode[att_eNB_id], PHY_vars_eNB_g[att_eNB_id][CC_id]->UE_stats[UE_id].DL_pmi_single,
	       PHY_vars_eNB_g[att_eNB_id][CC_id]->mu_mimo_mode[UE_id].dl_pow_off,PHY_vars_eNB_g[att_eNB_id][CC_id]->frame_parms.N_RB_DL);

      // calculate sinr here
      for (eNB_id = 0; eNB_id < NB_eNB_INST; eNB_id++) {
        if (att_eNB_id != eNB_id) {
          calculate_sinr(eNB2UE[eNB_id][UE_id][CC_id], enb_data[eNB_id], ue_data[UE_id], PHY_vars_UE_g[UE_id][CC_id]->sinr_dB,PHY_vars_eNB_g[att_eNB_id][CC_id]->frame_parms.N_RB_DL);
        }
      }
    } // hold channel
  }
  else { //abstraction_flag
    //eNB_id = PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id;
    pthread_mutex_lock(&eNB_output_mutex[UE_id]);
 
    if (eNB_output_mask[UE_id] == 0) {  //  This is the first eNodeB for this UE, clear the buffer
      
      for (aa=0; aa<nb_antennas_rx; aa++) {
	memset((void*)r_re_DL[UE_id][aa],0,(frame_parms->samples_per_tti)*sizeof(double));
	memset((void*)r_im_DL[UE_id][aa],0,(frame_parms->samples_per_tti)*sizeof(double));
      }
    }
    pthread_mutex_unlock(&eNB_output_mutex[UE_id]);

    for (eNB_id=0; eNB_id<NB_eNB_INST; eNB_id++) {
      txdata = PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.txdata[0];
      sf_offset = subframe*frame_parms->samples_per_tti;
      //for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d, eNB_id %d: txdata[%d] = (%d,%d)\n", subframe,eNB_id, idx, ((short*)&txdata[0][sf_offset+idx])[0], ((short*)&txdata[0][sf_offset+idx])[1]);
      start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain);
      tx_pwr = dac_fixed_gain(s_re,
                              s_im,
                              txdata,
                              sf_offset,
                              nb_antennas_tx,
                              frame_parms->samples_per_tti,
                              sf_offset,
                              frame_parms->ofdm_symbol_size,
                              14,
                              frame_parms->pdsch_config_common.referenceSignalPower, // dBm/RE
                              frame_parms->N_RB_DL*12);
      stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain);

#ifdef DEBUG_SIM
      LOG_D(OCM,"[SIM][DL] eNB %d (CCid %d): tx_pwr %.1f dBm/RE (target %d dBm/RE), for subframe %d\n",
            eNB_id,CC_id,
            10*log10(tx_pwr),
            frame_parms->pdsch_config_common.referenceSignalPower,
            subframe);

#endif
      //eNB2UE[eNB_id][UE_id]->path_loss_dB = 0;
      start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel);
      multipath_channel(eNB2UE[eNB_id][UE_id][CC_id],s_re,s_im,r_re0,r_im0,
                        frame_parms->samples_per_tti,hold_channel);
      stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel);
#ifdef DEBUG_SIM
      rx_pwr = signal_energy_fp2(eNB2UE[eNB_id][UE_id][CC_id]->ch[0],
                                 eNB2UE[eNB_id][UE_id][CC_id]->channel_length)*eNB2UE[eNB_id][UE_id][CC_id]->channel_length;
      LOG_D(OCM,"[SIM][DL] Channel eNB %d => UE %d (CCid %d): Channel gain %f dB (%f)\n",eNB_id,UE_id,CC_id,10*log10(rx_pwr),rx_pwr);
#endif


#ifdef DEBUG_SIM

      for (i=0; i<eNB2UE[eNB_id][UE_id][CC_id]->channel_length; i++)
        LOG_D(OCM,"channel(%d,%d)[%d] : (%f,%f)\n",eNB_id,UE_id,i,eNB2UE[eNB_id][UE_id][CC_id]->ch[0][i].x,eNB2UE[eNB_id][UE_id][CC_id]->ch[0][i].y);

#endif

      LOG_D(OCM,"[SIM][DL] Channel eNB %d => UE %d (CCid %d): tx_power %.1f dBm/RE, path_loss %1.f dB\n",
            eNB_id,UE_id,CC_id,
            (double)frame_parms->pdsch_config_common.referenceSignalPower,
            //         enb_data[eNB_id]->tx_power_dBm,
            eNB2UE[eNB_id][UE_id][CC_id]->path_loss_dB);

#ifdef DEBUG_SIM
      rx_pwr = signal_energy_fp(r_re0,r_im0,nb_antennas_rx,
                                frame_parms->ofdm_symbol_size,
                                sf_offset)/(12.0*frame_parms->N_RB_DL);
      LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr %f dBm/RE (%f dBm RSSI)for subframe %d\n",UE_id,
            10*log10(rx_pwr),
            10*log10(rx_pwr*(double)frame_parms->N_RB_DL*12),subframe);
      LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr (noise) -132 dBm/RE (N0fs = %.1f dBm, N0B = %.1f dBm) for subframe %d\n",
            UE_id,
            10*log10(eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate*1e6)-174,
            10*log10(eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate*1e6*12*frame_parms->N_RB_DL/(double)frame_parms->ofdm_symbol_size)-174,
            subframe);
#endif

      if (eNB2UE[eNB_id][UE_id][CC_id]->first_run == 1)
        eNB2UE[eNB_id][UE_id][CC_id]->first_run = 0;


      // RF model
#ifdef DEBUG_SIM
      LOG_D(OCM,"[SIM][DL] UE %d (CCid %d): rx_gain %d dB (-ADC %f) for subframe %d\n",UE_id,CC_id,PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB,
            PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB-66.227,subframe);
#endif
      start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple);
      rf_rx_simple(r_re0,
                   r_im0,
                   nb_antennas_rx,
                   frame_parms->samples_per_tti,
                   1e3/eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate,  // sampling time (ns)
                   (double)PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB - 66.227);   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
      stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple);

#ifdef DEBUG_SIM
      rx_pwr = signal_energy_fp(r_re0,r_im0,
                                nb_antennas_rx,
                                frame_parms->ofdm_symbol_size,
                                sf_offset)/(12.0*frame_parms->N_RB_DL);
      LOG_D(OCM,"[SIM][DL] UE %d : ADC in (eNB %d) %f dBm/RE for subframe %d\n",
            UE_id,eNB_id,
            10*log10(rx_pwr),subframe);
#endif
      
      pthread_mutex_lock(&eNB_output_mutex[UE_id]);
      for (i=0; i<frame_parms->samples_per_tti; i++) {
        for (aa=0; aa<nb_antennas_rx; aa++) {
          r_re_DL[UE_id][aa][i]+=r_re0[aa][i];
          r_im_DL[UE_id][aa][i]+=r_im0[aa][i];
        }
      }
      eNB_output_mask[UE_id] |= (1<<eNB_id);
      if (eNB_output_mask[UE_id] == (1<<NB_eNB_INST)-1) {
	eNB_output_mask[UE_id]=0;

	double *r_re_p[2] = {r_re_DL[UE_id][0],r_re_DL[UE_id][1]};
	double *r_im_p[2] = {r_im_DL[UE_id][0],r_im_DL[UE_id][1]};

#ifdef DEBUG_SIM
	rx_pwr = signal_energy_fp(r_re_p,r_im_p,nb_antennas_rx,frame_parms->ofdm_symbol_size,sf_offset)/(12.0*frame_parms->N_RB_DL);
	LOG_D(OCM,"[SIM][DL] UE %d : ADC in %f dBm for subframe %d\n",UE_id,10*log10(rx_pwr),subframe);
#endif
	
	rxdata = PHY_vars_UE_g[UE_id][CC_id]->common_vars.rxdata;
	sf_offset = subframe*frame_parms->samples_per_tti;

        start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc);
	adc(r_re_p,
	    r_im_p,
	    0,
	    sf_offset,
	    rxdata,
	    nb_antennas_rx,
	    frame_parms->samples_per_tti,
	    12);
        stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc);
	//for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d, eNB_id %d: rxdata[%d] = (%d,%d)\n", subframe,eNB_id, idx, ((short*)&rxdata[0][sf_offset+idx])[0], ((short*)&rxdata[0][sf_offset+idx])[1]);
#ifdef DEBUG_SIM
	rx_pwr2 = signal_energy(rxdata[0]+sf_offset,frame_parms->ofdm_symbol_size)/(12.0*frame_parms->N_RB_DL);
	LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr (ADC out) %f dB/RE (%d) for subframe %d, writing to %p\n",UE_id, 10*log10((double)rx_pwr2),rx_pwr2,subframe,rxdata);
#else
	UNUSED_VARIABLE(rx_pwr2);
	UNUSED_VARIABLE(tx_pwr);
	UNUSED_VARIABLE(rx_pwr);
#endif
		
      } // eNB_output_mask
      pthread_mutex_unlock(&eNB_output_mutex[UE_id]);      
    } // eNB_id

  }
}
void do_DL_sig_freq(channel_desc_t *eNB2UE[NUMBER_OF_eNB_MAX][NUMBER_OF_UE_MAX][MAX_NUM_CCs],
	       node_desc_t *enb_data[NUMBER_OF_eNB_MAX],
	       node_desc_t *ue_data[NUMBER_OF_UE_MAX],
	       uint16_t subframe,uint8_t abstraction_flag,LTE_DL_FRAME_PARMS *frame_parms,
	       uint8_t UE_id,
	       int CC_id)
{
  /*static int first_run=0;
  static double sum;
  static int count;
  if (!first_run)
  {
     first_run=1;
     sum=0;
     count=0;
  }
  count++;*/
  int32_t att_eNB_id=-1;
  int32_t **txdataF,**rxdataF;

  uint8_t eNB_id=0;
#ifdef    __AVX2__
  float tx_pwr;
  //printf ("AVX2 instruction set activated\n");
#else
  double tx_pwr;
#endif
  double rx_pwr;
  int32_t rx_pwr0,rx_pwr1,rx_pwr2, rx_pwr3;
  uint32_t i,aa;
  uint32_t sf_offset;

  //double min_path_loss=-200;
  uint8_t hold_channel=0;
  uint8_t nb_antennas_rx = eNB2UE[0][0][CC_id]->nb_rx; // number of rx antennas at UE
  uint8_t nb_antennas_tx = eNB2UE[0][0][CC_id]->nb_tx; // number of tx antennas at eNB
#ifdef    __AVX2__
  float s_re0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float s_re1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *s_re_f[2];
  float s_im0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float s_im1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *s_im_f[2];
  float r_re00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float r_re01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *r_re0_f[2];
  float r_im00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float r_im01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *r_im0_f[2];
#else
  double s_re0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double s_re1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *s_re_f[2];
  double s_im0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double s_im1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *s_im_f[2];
  double r_re00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double r_re01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *r_re0_f[2];
  double r_im00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double r_im01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *r_im0_f[2];
#endif

  s_re_f[0] = s_re0_f;
  s_im_f[0] = s_im0_f;
  s_re_f[1] = s_re1_f;
  s_im_f[1] = s_im1_f;

  r_re0_f[0] = r_re00_f;
  r_im0_f[0] = r_im00_f;
  r_re0_f[1] = r_re01_f;
  r_im0_f[1] = r_im01_f;

  //FILE *file1;
  //file1 = fopen("chsim_s_re_im.m","w");
  //printf("chsim thread %d. ue->proc->frame_rx %d, ue->subframe_rx %d, ue->proc->frame_tx %d, ue->subframe_tx %d\n",subframe&0x1,PHY_vars_UE_g[0][0]->proc.proc_rxtx[subframe&0x1].frame_rx,PHY_vars_UE_g[0][0]->proc.proc_rxtx[subframe&0x1].subframe_rx,PHY_vars_UE_g[0][0]->proc.proc_rxtx[subframe&0x1].frame_tx,PHY_vars_UE_g[0][0]->proc.proc_rxtx[subframe&0x1].subframe_tx);


  if (subframe==0)
    hold_channel = 0;
  else
    hold_channel = 1;

  if (abstraction_flag) {
	LOG_D(OCM,"[SIM][DL] Abstraction for do_DL_sig_freq is not implemented in frequency domain\n");
	exit(-1);
  }
  else { //abstraction_flag = 0

    //printf("UE association: (UE%d->eNB%d)\n",UE_id,PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id);
    eNB_id = PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id;
    pthread_mutex_lock(&eNB_output_mutex[UE_id]);
    //printf("eNB_output_mask[UE_id] is %d\n",eNB_output_mask[UE_id]);
    //if (eNB_output_mask[UE_id] == 0) {  //  This is the first eNodeB for this UE, clear the buffer
	      for (aa=0; aa<nb_antennas_rx; aa++) {
			memset((void*)r_re_DL_f[UE_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
			memset((void*)r_im_DL_f[UE_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
    			//memset(&PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF[aa][subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(int));		
	      }
    //}
    pthread_mutex_unlock(&eNB_output_mutex[UE_id]);
    //for (eNB_id=0; eNB_id<NB_eNB_INST; eNB_id++) {

	      	txdataF = PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.txdataF[0];
	      	sf_offset = subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;              
              	//for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d,UE %d, eNB_id %d: txdataF[%d] = (%d,%d)\n", subframe,UE_id,eNB_id, idx, ((short*)&txdataF[0][sf_offset+idx])[0], ((short*)&txdataF[0][sf_offset+idx])[1]);
              
	      
#ifdef    __AVX2__
	      	start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain_freq);
	      	tx_pwr = dac_fixed_gain_AVX_float(s_re_f,
		                      s_im_f,
		                      txdataF,
		                      sf_offset,
		                      nb_antennas_tx,
		                      frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		                      sf_offset,
		                      frame_parms->ofdm_symbol_size,
		                      14,
		                      frame_parms->pdsch_config_common.referenceSignalPower, // dBm/RE
		                      frame_parms->N_RB_DL*12);
	      	stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain_freq);
		printf("UE%d,eNB%d: dac_fixed_gain: referenceSignalPower %d\n",UE_id,eNB_id,frame_parms->pdsch_config_common.referenceSignalPower);
#else
	      	start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain_freq);
	      	tx_pwr = dac_fixed_gain(s_re_f,
		                      s_im_f,
		                      txdataF,
		                      sf_offset,
		                      nb_antennas_tx,
		                      frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		                      sf_offset,
		                      frame_parms->ofdm_symbol_size,
		                      14,
		                      frame_parms->pdsch_config_common.referenceSignalPower, // dBm/RE
		                      frame_parms->N_RB_DL*12);
	      	stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain_freq);
#endif
	      
	      	//print_meas (&eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain,"[DL][dac_fixed_gain]", &eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain, &eNB2UE[eNB_id][UE_id][CC_id]->DL_dac_fixed_gain);

			//for (x=0;x<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;x++){
			//	fprintf(file1,"%d\t%e\t%e\n",x,s_re_f[0][x],s_im_f[0][x]);
			//}
		/*if (eNB_id==0 && subframe ==9)
	      		write_output("txsigF0.m","txF0", PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.txdataF[0][0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
		else if (eNB_id==1 && subframe ==9)
	      		write_output("txsigF1.m","txF1", PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.txdataF[0][0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);*/
#ifdef DEBUG_SIM
      		LOG_D(OCM,"[SIM][DL] eNB %d (CCid %d): tx_pwr %.1f dBm/RE (target %d dBm/RE), for subframe %d\n",
            	eNB_id,CC_id,
            	10*log10(tx_pwr),
            	frame_parms->pdsch_config_common.referenceSignalPower,
            	subframe);
#endif
      		//eNB2UE[eNB_id][UE_id]->path_loss_dB = 0;
                //clock_t start=clock();

#ifdef    __AVX2__
		start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel_freq);
      		multipath_channel_freq_AVX_float(eNB2UE[eNB_id][UE_id][CC_id],s_re_f,s_im_f,r_re0_f,r_im0_f,
                frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,hold_channel,eNB_id,UE_id,CC_id,subframe&0x1,frame_parms->N_RB_DL,frame_parms->N_RB_DL*12+1,
		frame_parms->ofdm_symbol_size,frame_parms->symbols_per_tti);
		stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel_freq);
#else
		start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel_freq);
      		multipath_channel_freq(eNB2UE[eNB_id][UE_id][CC_id],s_re_f,s_im_f,r_re0_f,r_im0_f,
                        frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,hold_channel,eNB_id,UE_id,CC_id,subframe&0x1);
		stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_multipath_channel_freq);
#endif

		//for (int idx=0;idx<10;idx++) printf("dumping DL raw tx subframe (input) %d: s_f[%d] = (%f,%f)\n", subframe, idx, s_re_f[0][idx],s_im_f[0][idx]);
		//for (int idx=0;idx<10;idx++) printf("dumping DL raw tx subframe (input) %d: r_f[%d] = (%f,%f)\n", subframe, idx, r_re0_f[0][idx],r_im0_f[0][idx]);
       		/*clock_t stop=clock();
  		printf("multipath_channel DL time is %f s, AVERAGE time is %f s, count %d, sum %e\n",(float) (stop-start)/CLOCKS_PER_SEC,(float) (sum+stop-start)/(count*CLOCKS_PER_SEC),count,sum+stop-start);
  		sum=(sum+stop-start);*/
			/*for (int x=0;x<frame_parms->N_RB_DL*12;x++){
				fprintf(file1,"%d\t%e\t%e\n",x,eNB2UE[eNB_id][UE_id][CC_id]->chF[0][x].x,eNB2UE[eNB_id][UE_id][CC_id]->chF[0][x].y);
			}*/

#ifdef DEBUG_SIM
      		rx_pwr = signal_energy_fp2(eNB2UE[eNB_id][UE_id][CC_id]->chF[0],
                                 frame_parms->N_RB_DL*12+1)*(frame_parms->N_RB_DL*12+1);
      		LOG_D(OCM,"[SIM][DL] Channel eNB %d => UE %d (CCid %d): Channel gain %f dB (%f)\n",eNB_id,UE_id,CC_id,10*log10(rx_pwr),rx_pwr);
#endif

      		rx_pwr = signal_energy_fp2(eNB2UE[eNB_id][UE_id][CC_id]->chF[0],
                                 frame_parms->N_RB_DL*12+1)*(frame_parms->N_RB_DL*12+1);
      		//printf("[SIM][DL] Channel eNB %d => UE %d (CCid %d): Channel gain %f dB (%f)\n",eNB_id,UE_id,CC_id,10*log10(rx_pwr),rx_pwr);
//#ifdef DEBUG_SIM
		/*if (eNB_id==0 && UE_id ==0 && subframe ==9)
	      		write_output_chFf("channelF00.m","chF00", eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].x,eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].y,frame_parms->ofdm_symbol_size,1);
		else if (eNB_id==1 && UE_id ==0 && subframe ==9)
	      		write_output_chFf("channelF10.m","chF10", eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].x,eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].y,frame_parms->ofdm_symbol_size,1);
		else if (eNB_id==0 && UE_id ==1 && subframe ==9)
	      		write_output_chFf("channelF01.m","chF01", eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].x,eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].y,frame_parms->ofdm_symbol_size,1);
		else if (eNB_id==1 && UE_id ==1 && subframe ==9)
	      		write_output_chFf("channelF11.m","chF11", eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].x,eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].y,frame_parms->ofdm_symbol_size,1);*/
      		/*for (i=0; i<frame_parms->N_RB_DL*12; i++){
        		printf("do_DL_sig channel(eNB%d,UE%d)[%d] : (%f,%f)\n",eNB_id,UE_id,i,[i],eNB2UE[eNB_id][UE_id][CC_id]->chFf[0].y[i]);
		}*/

//#endif

      		LOG_D(OCM,"[SIM][DL] Channel eNB %d => UE %d (CCid %d): tx_power %.1f dBm/RE, path_loss %1.f dB\n",
            	eNB_id,UE_id,CC_id,
            	(double)frame_parms->pdsch_config_common.referenceSignalPower,
            	//         enb_data[eNB_id]->tx_power_dBm,
            	eNB2UE[eNB_id][UE_id][CC_id]->path_loss_dB);

#ifdef DEBUG_SIM
#ifdef    __AVX2__
      		rx_pwr = signal_energy_fp_AVX_float(r_re0_f,r_im0_f,nb_antennas_rx,
                                frame_parms->ofdm_symbol_size,
                                sf_offset)/(12.0*frame_parms->N_RB_DL);
#else
      		rx_pwr = signal_energy_fp(r_re0_f,r_im0_f,nb_antennas_rx,
                                frame_parms->ofdm_symbol_size,
                                sf_offset)/(12.0*frame_parms->N_RB_DL);
#endif

      		LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr %f dBm/RE (%f dBm RSSI)for subframe %d\n",UE_id,
            	10*log10(rx_pwr),
            	10*log10(rx_pwr*(double)frame_parms->N_RB_DL*12),subframe);
      		LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr (noise) -132 dBm/RE (N0fs = %.1f dBm, N0B = %.1f dBm) for slot %d (subframe %d)\n",
            	UE_id,10*log10(eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate*1e6)-174,
            	10*log10(eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate*1e6*12*frame_parms->N_RB_DL/(double)frame_parms->ofdm_symbol_size)-174,
            	subframe);
#endif

      		if (eNB2UE[eNB_id][UE_id][CC_id]->first_run == 1)
        		eNB2UE[eNB_id][UE_id][CC_id]->first_run = 0;


      		// RF model
#ifdef DEBUG_SIM
      		LOG_D(OCM,"[SIM][DL] UE %d (CCid %d): rx_gain %d dB (-ADC %f) for subframe %d\n",UE_id,CC_id,PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB,
            	PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB-66.227,subframe);
#endif
		/*count++;
		clock_t start=clock();*/

#ifdef    __AVX2__
		start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq);
      		rf_rx_simple_freq_AVX_float(r_re0_f,
                   		r_im0_f,
                   		nb_antennas_rx,
                   		frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
                   		(float)1e3/eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate,  // sampling time (ns)
                   		(float)PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB - 66.227,   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
				frame_parms->symbols_per_tti,
				frame_parms->ofdm_symbol_size,
				12.0*frame_parms->N_RB_DL);
		stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq);
#else
		start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq);
      		rf_rx_simple_freq(r_re0_f,
                   		r_im0_f,
                   		nb_antennas_rx,
                   		frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
                   		1e3/eNB2UE[eNB_id][UE_id][CC_id]->sampling_rate,  // sampling time (ns)
                   		(double)PHY_vars_UE_g[UE_id][CC_id]->rx_total_gain_dB - 66.227,   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
				frame_parms->symbols_per_tti,
				frame_parms->ofdm_symbol_size,
				12.0*frame_parms->N_RB_DL);
		stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq);
#endif

		//for (int idx=0;idx<10;idx++) printf("dumping DL raw tx subframe (input) %d: r_f[%d] = (%f,%f)\n", subframe, idx, r_re0_f[0][idx],r_im0_f[0][idx]);
      	        //print_meas (&eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq,"[DL][rf_rx_simple_freq]", &eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq, &eNB2UE[eNB_id][UE_id][CC_id]->DL_rf_rx_simple_freq);

  		/*clock_t stop=clock();
  		printf("rf_rx DL time is %f s, AVERAGE time is %f s, count %d, sum %e\n",(float) (stop-start)/CLOCKS_PER_SEC,(float) (sum+stop-start)/(count*CLOCKS_PER_SEC),count,sum+stop-start);
  		sum=(sum+stop-start);*/
		/*for (int x=0;x<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;x++){
			fprintf(file1,"%d\t%e\t%e\n",x,r_re0_f[0][x],r_im0_f[0][x]);
		}*/
#ifdef DEBUG_SIM
      		rx_pwr = signal_energy_fp(r_re0_f,r_im0_f,
                                nb_antennas_rx,
                                frame_parms->ofdm_symbol_size,//?
                                (sf_offset)/(12.0*frame_parms->N_RB_DL));

      		LOG_D(OCM,"[SIM][DL] UE %d : ADC in (eNB %d) %f dBm/RE for subframe %d\n",
            	UE_id,eNB_id,
            	10*log10(rx_pwr),subframe);
#endif
      
      		pthread_mutex_lock(&eNB_output_mutex[UE_id]);
	      	for (i=0; i<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti; i++) {
			for (aa=0; aa<nb_antennas_rx; aa++) {
			  r_re_DL_f[UE_id][aa][i]+=r_re0_f[aa][i];
			  r_im_DL_f[UE_id][aa][i]+=r_im0_f[aa][i];
			}
	      	}

      		/*eNB_output_mask[UE_id] |= (1<<eNB_id);
      		if (eNB_output_mask[UE_id] == (1<<NB_eNB_INST)-1) {
			eNB_output_mask[UE_id]=0;
		}*/
#ifdef    __AVX2__
			float *r_re_p_f[2] = {r_re_DL_f[UE_id][0],r_re_DL_f[UE_id][1]};
			float *r_im_p_f[2] = {r_im_DL_f[UE_id][0],r_im_DL_f[UE_id][1]};
#else      
			double *r_re_p_f[2] = {r_re_DL_f[UE_id][0],r_re_DL_f[UE_id][1]};
			double *r_im_p_f[2] = {r_im_DL_f[UE_id][0],r_im_DL_f[UE_id][1]};
#endif

#ifdef DEBUG_SIM
			rx_pwr0 = signal_energy_fp(r_re_p_f,r_im_p_f,nb_antennas_rx,frame_parms->ofdm_symbol_size,sf_offset)/(12.0*frame_parms->N_RB_DL);
			LOG_D(OCM,"[SIM][DL] UE %d : (r_re_p_f) ADC in %f dBm for subframe %d\n",UE_id,10*log10(rx_pwr1),subframe);

			LOG_D(OCM,"[SIM][DL] UE %d : ADC in %f dBm for subframe %d\n",UE_id,10*log10(rx_pwr),subframe);
#endif
			//for (int idx=0;idx<10;idx++) printf("dumping DL raw rx subframe (input) %d: rxdataF[%d] = (%f,%f)\n", subframe, idx, r_re_p_f[0][idx],r_im_p_f[0][idx]);
			rxdataF = PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF;
			sf_offset = subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;
		
			//printf("[ch_sim] sf_offset %d\n",sf_offset);

#ifdef    __AVX2__
			start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc_freq);
	        	adc_AVX_float(r_re_p_f,
		    	r_im_p_f,
		    	0,
		    	sf_offset,
		    	rxdataF,
		    	nb_antennas_rx,
		    	frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		    	12,
		    	PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_DL*12,
		    	frame_parms->ofdm_symbol_size);
			stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc_freq);
#else
			start_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc_freq);
	        	adc(r_re_p_f,
		    	r_im_p_f,
		    	0,
		    	sf_offset,
		    	rxdataF,
		    	nb_antennas_rx,
		    	frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		    	12);
			stop_meas(&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc_freq);
#endif
    			/*if (eNB_id==0 && subframe ==9){
				if (UE_id==0)
    				   write_output("rxsigF00.m","rxF00", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
				else if (UE_id==1)
    				   write_output("rxsigF01.m","rxF01", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
			}
    			else if (eNB_id==1 && subframe ==9){
				if (UE_id==0)
				  write_output("rxsigF10.m","rxF10", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
				else if (UE_id==1)
				  write_output("rxsigF11.m","rxF11", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[subframe&0x1].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
			}*/
	      		//for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d: r_re_p_f[%d] = (%e,%e)\n", subframe, idx, r_re_p_f[0][idx], r_im_p_f[0][idx]);
              		//for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d,UE %d, eNB_id %d: rxdataF0[%d] = (%d,%d)\n", subframe,UE_id,eNB_id, idx, ((short*)&rxdataF[0][sf_offset+idx])[0], ((short*)&rxdataF[0][sf_offset+idx])[1]);
      	        	//print_meas (&eNB2UE[eNB_id][UE_id][CC_id]->DL_adc,"[DL][adc]", &eNB2UE[eNB_id][UE_id][CC_id]->DL_adc, &eNB2UE[eNB_id][UE_id][CC_id]->DL_adc);
             		//for (int idx=0;idx<10;idx++) printf("dumping DL raw rx subframe %d: rxdataF[%d] = (%d,%d)=====>%s,txdataF[%d] = (%d,%d), r_re_im_p_f(%e,%e)\n", subframe, idx, ((short*)&rxdataF[0][sf_offset+idx])[0], ((short*)&rxdataF[0][sf_offset+idx])[1],(((((r_re_p_f[0][idx]<0)&&(((short*)&rxdataF[0][sf_offset+idx])[0]<0))||((r_re_p_f[0][idx]>=0)&&(((short*)&rxdataF[0][sf_offset+idx])[0]>=0))))&&(((r_im_p_f[0][idx]<0)&&(((short*)&rxdataF[0][sf_offset+idx])[1]<0))||((r_im_p_f[0][idx]>=0)&&(((short*)&rxdataF[0][sf_offset+idx])[1]>=0))))?"OK":"ERROR",idx,((short*)&txdataF[0][sf_offset+idx])[0],((short*)&txdataF[0][sf_offset+idx])[1],r_re_p_f[0][idx],r_im_p_f[0][idx]);
			/*if (UE_id==0)
				write_output("chsim0_rxsigF_subframe0.m","chsm0_rxsF0", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[0].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
			else
				write_output("chsim1_rxsigF_subframe0.m","chsm1_rxsF0", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[0].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);*/
			//write_output("chsim_rxsigF_subframe1.m","chsm_rxsF1", PHY_vars_UE_g[UE_id][CC_id]->common_vars.common_vars_rx_data_per_thread[1].rxdataF[0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
	
#ifdef DEBUG_SIM
			rx_pwr2 = signal_energy((rxdataF[0])+sf_offset,frame_parms->ofdm_symbol_size)/(12.0*frame_parms->N_RB_DL);
			LOG_D(OCM,"[SIM][DL] UE %d : rx_pwr(rxdafaF) (ADC out) %f dB/RE (%d) for subframe %d, writing to %p\n",UE_id, 10*log10((double)rx_pwr2),rx_pwr2,subframe,rxdataF);
#else
			//UNUSED_VARIABLE(rx_pwr2);
			//UNUSED_VARIABLE(tx_pwr);
			//UNUSED_VARIABLE(rx_pwr);
#endif	
      		//} // eNB_output_mask
      		pthread_mutex_unlock(&eNB_output_mutex[UE_id]);      
    //} // eNB_id

  }
}


void do_UL_sig(channel_desc_t *UE2eNB[NUMBER_OF_UE_MAX][NUMBER_OF_eNB_MAX][MAX_NUM_CCs],
               node_desc_t *enb_data[NUMBER_OF_eNB_MAX],node_desc_t *ue_data[NUMBER_OF_UE_MAX],
	       uint16_t subframe,uint8_t abstraction_flag,LTE_DL_FRAME_PARMS *frame_parms, 
	       uint32_t frame,int eNB_id,uint8_t CC_id)
{

  int32_t **txdata,**rxdata;
#ifdef PHY_ABSTRACTION_UL
  int32_t att_eNB_id=-1;
#endif
  uint8_t UE_id=0;

  uint8_t nb_antennas_rx = UE2eNB[0][0][CC_id]->nb_rx; // number of rx antennas at eNB
  uint8_t nb_antennas_tx = UE2eNB[0][0][CC_id]->nb_tx; // number of tx antennas at UE

  double tx_pwr, rx_pwr;
  int32_t rx_pwr2;
  uint32_t i,aa;
  uint32_t sf_offset;

  uint8_t hold_channel=0;

#ifdef PHY_ABSTRACTION_UL
  double min_path_loss=-200;
  uint16_t ul_nb_rb=0 ;
  uint16_t ul_fr_rb=0;
  int ulnbrb2 ;
  int ulfrrb2 ;
  uint8_t harq_pid;
#endif
  double s_re0[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double s_re1[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *s_re[2];
  double s_im0[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double s_im1[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *s_im[2];
  double r_re00[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double r_re01[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *r_re0[2];
  double r_im00[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double r_im01[30720];//PHY_vars_UE_g[UE_id][CC_id]->frame_parms.samples_per_tti];
  double *r_im0[2];

  s_re[0] = s_re0;
  s_im[0] = s_im0;
  s_re[1] = s_re1;
  s_im[1] = s_im1;

  r_re0[0] = r_re00;
  r_im0[0] = r_im00;
  r_re0[1] = r_re01;
  r_im0[1] = r_im01;

  if (abstraction_flag!=0)  {
#ifdef PHY_ABSTRACTION_UL

    for (UE_id=0; UE_id<NB_UE_INST; UE_id++) {
      if (!hold_channel) {
	random_channel(UE2eNB[UE_id][eNB_id][CC_id],abstraction_flag);
	freq_channel(UE2eNB[UE_id][eNB_id][CC_id], frame_parms->N_RB_UL,frame_parms->N_RB_UL*12+1);
	
	// REceived power at the eNB
	rx_pwr = signal_energy_fp2(UE2eNB[UE_id][eNB_id][CC_id]->ch[0],
				   UE2eNB[UE_id][eNB_id][CC_id]->channel_length)*UE2eNB[UE_id][att_eNB_id][CC_id]->channel_length; // calculate the rx power at the eNB
      }
      
      //  write_output("SINRch.m","SINRch",PHY_vars_eNB_g[att_eNB_id]->sinr_dB_eNB,frame_parms->N_RB_UL*12+1,1,1);
      if(subframe>1 && subframe <5) {
	harq_pid = subframe2harq_pid(frame_parms,frame,subframe);
	ul_nb_rb = PHY_vars_eNB_g[att_eNB_id][CC_id]->ulsch_eNB[(uint8_t)UE_id]->harq_processes[harq_pid]->nb_rb;
	ul_fr_rb = PHY_vars_eNB_g[att_eNB_id][CC_id]->ulsch_eNB[(uint8_t)UE_id]->harq_processes[harq_pid]->first_rb;
      }
      
      if(ul_nb_rb>1 && (ul_fr_rb < 25 && ul_fr_rb > -1)) {
	number_rb_ul = ul_nb_rb;
	first_rbUL = ul_fr_rb;
	init_snr_up(UE2eNB[UE_id][att_eNB_id][CC_id],enb_data[att_eNB_id], ue_data[UE_id],PHY_vars_eNB_g[att_eNB_id][CC_id]->sinr_dB,&PHY_vars_UE_g[att_eNB_id][CC_id]->N0,ul_nb_rb,ul_fr_rb);
	
      }
    } //UE_id

#else

#endif
  } else { //without abstraction

    pthread_mutex_lock(&UE_output_mutex[eNB_id]);
    // Clear RX signal for eNB = eNB_id
    for (i=0; i<frame_parms->samples_per_tti; i++) {
      for (aa=0; aa<nb_antennas_rx; aa++) {
	r_re_UL[eNB_id][aa][i]=0.0;
	r_im_UL[eNB_id][aa][i]=0.0;
      }
    }
    pthread_mutex_unlock(&UE_output_mutex[eNB_id]);

    // Compute RX signal for eNB = eNB_id
    for (UE_id=0; UE_id<NB_UE_INST; UE_id++) {
      //printf("ue->generate_ul_signal[%d] %d\n",eNB_id,PHY_vars_UE_g[UE_id][CC_id]->generate_ul_signal[eNB_id]);
      //if (PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id != eNB_id)
	//	continue;
      //printf("[channel_sim_UL_time] subframe %d\n",subframe);
      txdata = PHY_vars_UE_g[UE_id][CC_id]->common_vars.txdata;
      sf_offset = subframe*frame_parms->samples_per_tti;
      //for (int idx=0;idx<10;idx++) printf("dumping UL raw subframe %d: txdata[%d] = (%d,%d)\n", subframe, idx, ((short*)&txdata[0][sf_offset+idx])[0], ((short*)&txdata[0][sf_offset+idx])[1]);
     //write_output("chsim_txsigF_UL.m","chsm_txsF_UL", &PHY_vars_UE_g[UE_id][CC_id]->common_vars.txdataF[0][0],10*frame_parms->samples_per_tti,1,16);
      
      if (((double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe] +
	   UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB) <= -125.0) {
	// don't simulate a UE that is too weak
	LOG_D(OCM,"[SIM][UL] UE %d tx_pwr %d dBm (num_RE %d) for subframe %d (sf_offset %d)\n",
	      UE_id,
	      PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
	      PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
	      subframe,sf_offset);	
      } else {
        start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain);
	tx_pwr = dac_fixed_gain((double**)s_re,
				(double**)s_im,
				txdata,
				sf_offset,
				nb_antennas_tx,
				frame_parms->samples_per_tti,
				sf_offset,
				frame_parms->ofdm_symbol_size,
				14,
				(double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe]-10*log10((double)PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]),
				PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]);  // This make the previous argument the total power
        stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain);
	LOG_D(OCM,"[SIM][UL] UE %d tx_pwr %f dBm (target %d dBm, num_RE %d) for subframe %d (sf_offset %d)\n",
	      UE_id,
	      10*log10(tx_pwr),
	      PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
	      PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
	      subframe,sf_offset);
       
        start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel);		
	multipath_channel(UE2eNB[UE_id][eNB_id][CC_id],s_re,s_im,r_re0,r_im0,
			  frame_parms->samples_per_tti,hold_channel);
	stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel);

	rx_pwr = signal_energy_fp2(UE2eNB[UE_id][eNB_id][CC_id]->ch[0],
				   UE2eNB[UE_id][eNB_id][CC_id]->channel_length)*UE2eNB[UE_id][eNB_id][CC_id]->channel_length;

	LOG_D(OCM,"[SIM][UL] subframe %d Channel UE %d => eNB %d : %f dB (hold %d,length %d, PL %f)\n",subframe,UE_id,eNB_id,10*log10(rx_pwr),
	      hold_channel,UE2eNB[UE_id][eNB_id][CC_id]->channel_length,
	      UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB);

	rx_pwr = signal_energy_fp(r_re0,r_im0,nb_antennas_rx,frame_parms->samples_per_tti,0);
	LOG_D(OCM,"[SIM][UL] eNB %d : rx_pwr %f dBm (%f) for subframe %d, sptti %d\n",
	      eNB_id,10*log10(rx_pwr),rx_pwr,subframe,frame_parms->samples_per_tti);
	
	
	if (UE2eNB[UE_id][eNB_id][CC_id]->first_run == 1)
	  UE2eNB[UE_id][eNB_id][CC_id]->first_run = 0;
	
	
	pthread_mutex_lock(&UE_output_mutex[eNB_id]);
	for (aa=0; aa<nb_antennas_rx; aa++) {
	  for (i=0; i<frame_parms->samples_per_tti; i++) {
	    r_re_UL[eNB_id][aa][i]+=r_re0[aa][i];
	    r_im_UL[eNB_id][aa][i]+=r_im0[aa][i];
	  }
	}
	pthread_mutex_unlock(&UE_output_mutex[eNB_id]);
      }
    } //UE_id
    
    double *r_re_p[2] = {r_re_UL[eNB_id][0],r_re_UL[eNB_id][1]};
    double *r_im_p[2] = {r_im_UL[eNB_id][0],r_im_UL[eNB_id][1]};
    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple); 
    rf_rx_simple(r_re_p,
		 r_im_p,
		 nb_antennas_rx,
		 frame_parms->samples_per_tti,
		 1e3/UE2eNB[0][eNB_id][CC_id]->sampling_rate,  // sampling time (ns)
		 (double)PHY_vars_eNB_g[eNB_id][CC_id]->rx_total_gain_dB - 66.227);   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple); 
    
#ifdef DEBUG_SIM
    rx_pwr = signal_energy_fp(r_re_p,r_im_p,nb_antennas_rx,frame_parms->samples_per_tti,0)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
    LOG_D(OCM,"[SIM][UL] rx_pwr (ADC in) %f dB for subframe %d\n",10*log10(rx_pwr),subframe);
#endif
    
    rxdata = PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.rxdata[0];
    sf_offset = subframe*frame_parms->samples_per_tti;

    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc);
    adc(r_re_p,
	r_im_p,
	0,
	sf_offset,
	rxdata,
	nb_antennas_rx,
	frame_parms->samples_per_tti,
	12);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc);
#ifdef DEBUG_SIM
    rx_pwr2 = signal_energy(rxdata[0]+sf_offset,frame_parms->samples_per_tti)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
    LOG_D(OCM,"[SIM][UL] eNB %d rx_pwr (ADC out) %f dB (%d) for subframe %d (offset %d)\n",eNB_id,10*log10((double)rx_pwr2),rx_pwr2,subframe,sf_offset);
#else
    UNUSED_VARIABLE(tx_pwr);
    UNUSED_VARIABLE(rx_pwr);
    UNUSED_VARIABLE(rx_pwr2);
#endif
    
  } // abstraction_flag==0
}
void do_UL_sig_freq(channel_desc_t *UE2eNB[NUMBER_OF_UE_MAX][NUMBER_OF_eNB_MAX][MAX_NUM_CCs],
               node_desc_t *enb_data[NUMBER_OF_eNB_MAX],node_desc_t *ue_data[NUMBER_OF_UE_MAX],
	       uint16_t subframe,uint8_t abstraction_flag,LTE_DL_FRAME_PARMS *frame_parms, 
	       uint32_t frame,int eNB_id,uint8_t CC_id)
{
  int32_t **txdataF,**rxdataF;
#ifdef PHY_ABSTRACTION_UL
  int32_t att_eNB_id=-1;
#endif
  uint8_t UE_id=0;

  uint8_t nb_antennas_rx = UE2eNB[0][0][CC_id]->nb_rx; // number of rx antennas at eNB
  uint8_t nb_antennas_tx = UE2eNB[0][0][CC_id]->nb_tx; // number of tx antennas at UE
#ifdef    __AVX2__
  float tx_pwr, rx_pwr;
#else
  double tx_pwr, rx_pwr;
#endif
  int32_t rx_pwr2;
  uint32_t i,aa;
  uint32_t sf_offset;

  uint8_t hold_channel=0;
  //FILE *file1;
  //file1 = fopen("chsim_chF_UL.m","w");

#ifdef PHY_ABSTRACTION_UL
  double min_path_loss=-200;
  uint16_t ul_nb_rb=0 ;
  uint16_t ul_fr_rb=0;
  int ulnbrb2 ;
  int ulfrrb2 ;
  uint8_t harq_pid;
#endif
#ifdef    __AVX2__
  float s_re0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float s_re1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *s_re_f[2];
  float s_im0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float s_im1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *s_im_f[2];
  float r_re00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float r_re01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *r_re0_f[2];
  float r_im00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float r_im01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  float *r_im0_f[2];
#else
  double s_re0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double s_re1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *s_re_f[2];
  double s_im0_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double s_im1_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *s_im_f[2];
  double r_re00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double r_re01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *r_re0_f[2];
  double r_im00_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double r_im01_f[2048*14];//ofdm_symbol_size*symbols_per_tti;
  double *r_im0_f[2];
#endif

  s_re_f[0] = s_re0_f;
  s_im_f[0] = s_im0_f;
  s_re_f[1] = s_re1_f;
  s_im_f[1] = s_im1_f;

  r_re0_f[0] = r_re00_f;
  r_im0_f[0] = r_im00_f;
  r_re0_f[1] = r_re01_f;
  r_im0_f[1] = r_im01_f;

  //uint8_t do_ofdm_mod = PHY_vars_UE_g[0][0]->do_ofdm_mod;

  if (abstraction_flag!=0)  {
/*#ifdef PHY_ABSTRACTION_UL

    for (UE_id=0; UE_id<NB_UE_INST; UE_id++) {
      if (!hold_channel) {
	random_channel(UE2eNB[UE_id][eNB_id][CC_id],abstraction_flag);
	freq_channel(UE2eNB[UE_id][eNB_id][CC_id], frame_parms->N_RB_UL,frame_parms->N_RB_UL*12+1);
	
	// REceived power at the eNB
	rx_pwr = signal_energy_fp2(UE2eNB[UE_id][eNB_id][CC_id]->ch[0],
				   UE2eNB[UE_id][eNB_id][CC_id]->channel_length)*UE2eNB[UE_id][att_eNB_id][CC_id]->channel_length; // calculate the rx power at the eNB
      }
      
      //  write_output_xrange("SINRch.m","SINRch",PHY_vars_eNB_g[att_eNB_id]->sinr_dB_eNB,frame_parms->N_RB_UL*12+1,1,1);
      if(subframe>1 && subframe <5) {
	harq_pid = subframe2harq_pid(frame_parms,frame,subframe);
	ul_nb_rb = PHY_vars_eNB_g[att_eNB_id][CC_id]->ulsch_eNB[(uint8_t)UE_id]->harq_processes[harq_pid]->nb_rb;
	ul_fr_rb = PHY_vars_eNB_g[att_eNB_id][CC_id]->ulsch_eNB[(uint8_t)UE_id]->harq_processes[harq_pid]->first_rb;
      }
      
      if(ul_nb_rb>1 && (ul_fr_rb < 25 && ul_fr_rb > -1)) {
	number_rb_ul = ul_nb_rb;
	first_rbUL = ul_fr_rb;
	init_snr_up(UE2eNB[UE_id][att_eNB_id][CC_id],enb_data[att_eNB_id], ue_data[UE_id],PHY_vars_eNB_g[att_eNB_id][CC_id]->sinr_dB,&PHY_vars_UE_g[att_eNB_id][CC_id]->N0,ul_nb_rb,ul_fr_rb);
	
      }
    } //UE_id

#else
#endif*/
	LOG_D(OCM,"[SIM][DL] Abstraction for do_UL_sig_freq is not implemented in frequency domain\n");
	exit(-1);

  } else { //without abstraction

    pthread_mutex_lock(&UE_output_mutex[eNB_id]);
    // Clear RX signal for eNB = eNB_id
    	//for (i=0; i<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti; i++) {
	      for (aa=0; aa<nb_antennas_rx; aa++) {
		/*r_re_UL_f[eNB_id][aa][i]=0.0;
		r_im_UL_f[eNB_id][aa][i]=0.0;
	      }*/
		memset((void*)r_re_UL_f[eNB_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
		memset((void*)r_im_UL_f[eNB_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
        }
    pthread_mutex_unlock(&UE_output_mutex[eNB_id]);

    // Compute RX signal for eNB = eNB_id
    for (UE_id=0; UE_id<NB_UE_INST; UE_id++) {
	//printf("ue->generate_ul_signal[%d] %d\n",eNB_id,PHY_vars_UE_g[UE_id][CC_id]->generate_ul_signal[eNB_id]);
	if (PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id != eNB_id)
		continue;
	txdataF = PHY_vars_UE_g[UE_id][CC_id]->common_vars.txdataF;
        AssertFatal(txdataF != NULL,"txdataF is null\n");
      	sf_offset = subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;
        //printf("[channel_sim_UL_freq] subframe %d\n",subframe);
 	/*for (int idx=subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx<(subframe+1)*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx++){
		if (((short*)&txdataF[0][idx])[0]!=0 || ((short*)&txdataF[0][idx])[1]!=0)
			printf("dumping UL raw subframet %d: txdataF[%d] = (%d,%d)\n", subframe, idx, ((short*)&txdataF[0][idx])[0], ((short*)&txdataF[0][idx])[1]);
	}*/
	//write_output("chsim_txsigF_UL.m","chsm_txsF_UL", &txdataF[0][0],10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
      
      if (((double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe] +
	   	UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB) <= -125.0) {
		// don't simulate a UE that is too weak
		LOG_D(OCM,"[SIM][UL] UE %d tx_pwr %d dBm (num_RE %d) for subframe %d (sf_offset %d)\n",
	      	UE_id,
	      	PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
	      	PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
	      	subframe,sf_offset);
		//printf("multipath_channel, UE too weak %e\n", ((double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe] +
		//   UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB));	
      } else {

#ifdef    __AVX2__
		start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain_freq);
		tx_pwr = dac_fixed_gain_AVX_float((float**)s_re_f,
					(float**)s_im_f,
					txdataF,
					sf_offset,
					nb_antennas_tx,
					frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
					sf_offset,
					frame_parms->ofdm_symbol_size,
					14,
					(float)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe]-10*log10((double)PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]),
					PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]);  // This make the previous argument the total power
		stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain_freq);
#else
		start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain_freq);
		tx_pwr = dac_fixed_gain((double**)s_re_f,
					(double**)s_im_f,
					txdataF,
					sf_offset,
					nb_antennas_tx,
					frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
					sf_offset,
					frame_parms->ofdm_symbol_size,
					14,
					(double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe]-10*log10((double)PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]),
					PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]);  // This make the previous argument the total power
		stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain_freq);
#endif

		//print_meas (&UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain,"[UL][dac_fixed_gain]", &UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain, &UE2eNB[UE_id][eNB_id][CC_id]->UL_dac_fixed_gain);
	        /*for (int idx=subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx<(subframe+1)*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx++){
			if (((short*)&txdataF[0][idx])[0]!=0 || ((short*)&txdataF[0][idx])[1]!=0)
				printf("dumping raw UL tx subframe (output) %d: s_re_f[%d] = (%f,%f)\n", subframe, idx, s_re_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti],s_im_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti]);
		}*/
		LOG_D(OCM,"[SIM][UL] UE %d tx_pwr %f dBm (target %d dBm, num_RE %d) for subframe %d (sf_offset %d)\n",
		      UE_id,
		      10*log10(tx_pwr),
		      PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
		      PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
		      subframe,sf_offset);
		//write_output("chsim_s_re_f_UL.m","chsm_sref_UL", s_re_f,10*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);

#ifdef    __AVX2__
		start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel_freq);
	      	multipath_channel_freq_AVX_float(UE2eNB[UE_id][eNB_id][CC_id],s_re_f,s_im_f,r_re0_f,r_im0_f,
			  frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,hold_channel,eNB_id,UE_id,CC_id,subframe&0x1,frame_parms->N_RB_DL,frame_parms->N_RB_DL*12+1,frame_parms->ofdm_symbol_size,frame_parms->symbols_per_tti);//ue timer subframe&0x1
		stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel_freq);
#else
		start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel_freq);
	      	multipath_channel_freq(UE2eNB[UE_id][eNB_id][CC_id],s_re_f,s_im_f,r_re0_f,r_im0_f,
			  frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,hold_channel,eNB_id,UE_id,CC_id,subframe&0x1);//ue timer subframe&0x1
		stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel_freq);
#endif

		//print_meas (&UE2eNB[UE_id][eNB_id][CC_id]->UL_multipath_channel_freq,"[UL][multipath_channel_freq]", NULL, NULL);
			//for (int x=0;x<frame_parms->N_RB_DL*12;x++){
			//	fprintf(file1,"%d\t%e\t%e\n",x,UE2eNB[UE_id][eNB_id][CC_id]->chF[0][x].x,UE2eNB[UE_id][eNB_id][CC_id]->chF[0][x].y);
			//}
	        /*for (int idx=subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx<(subframe+1)*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx++){
			if (((short*)&txdataF[0][idx])[0]!=0 || ((short*)&txdataF[0][idx])[1]!=0)
				printf("dumping raw UL tx subframe (output) %d: r_re0_f[%d] = (%f,%f)\n", subframe, idx, r_re0_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti],r_im0_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti]);
		}*/
		//write_output("chsim_r_re0_f_UL.m","chsim_rre0f_UL.m", r_re0_f,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);

		rx_pwr = signal_energy_fp2(UE2eNB[UE_id][eNB_id][CC_id]->ch[0],UE2eNB[UE_id][eNB_id][CC_id]->channel_length)*UE2eNB[UE_id][eNB_id][CC_id]->channel_length;
#ifdef DEBUG_SIM
      		for (i=0; i<10; i++){
        		LOG_D(OCM,"do_UL_sig channel(%d,%d)[%d] : (%f,%f)\n",UE_id,eNB_id,i,UE2eNB[UE_id][eNB_id][CC_id]->ch[0][i].x,UE2eNB[UE_id][eNB_id][CC_id]->ch[0][i].y);
		}
      		for (i=frame_parms->N_RB_DL*12-10; i<frame_parms->N_RB_DL*12; i++){
        		LOG_D(OCM,"do_UL_sig channel(%d,%d)[%d] : (%f,%f)\n",UE_id,eNB_id,i,UE2eNB[UE_id][eNB_id][CC_id]->ch[0][i].x,UE2eNB[UE_id][eNB_id][CC_id]->ch[0][i].y);
		}

#endif
		LOG_D(OCM,"[SIM][UL] subframe %d Channel UE %d => eNB %d : %f dB (hold %d,length %d, PL %f)\n",subframe,UE_id,eNB_id,10*log10(rx_pwr),
	     	 hold_channel,12*frame_parms->N_RB_DL+1,
	      	 UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB);
#ifdef    __AVX2__ 
		rx_pwr = signal_energy_fp_SSE_float(r_re0_f,r_im0_f,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,0);
#else
		rx_pwr = signal_energy_fp(r_re0_f,r_im0_f,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,0);
#endif
		LOG_D(OCM,"[SIM][UL] eNB %d : rx_pwr %f dBm (%f) for subframe %d, sptti %d\n",
	     	eNB_id,10*log10(rx_pwr),rx_pwr,subframe,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti);
	
	
		if (UE2eNB[UE_id][eNB_id][CC_id]->first_run == 1)
	  		UE2eNB[UE_id][eNB_id][CC_id]->first_run = 0;
	
	
		pthread_mutex_lock(&UE_output_mutex[eNB_id]);
		for (aa=0; aa<nb_antennas_rx; aa++) {
		  for (i=0; i<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti; i++) {
		    r_re_UL_f[eNB_id][aa][i]+=r_re0_f[aa][i];
		    r_im_UL_f[eNB_id][aa][i]+=r_im0_f[aa][i];
		  }
		}
	pthread_mutex_unlock(&UE_output_mutex[eNB_id]);
      }
    } //UE_id
#ifdef    __AVX2__    
    float *r_re_p_f[2] = {r_re_UL_f[eNB_id][0],r_re_UL_f[eNB_id][1]};
    float *r_im_p_f[2] = {r_im_UL_f[eNB_id][0],r_im_UL_f[eNB_id][1]};
#else
    double *r_re_p_f[2] = {r_re_UL_f[eNB_id][0],r_re_UL_f[eNB_id][1]};
    double *r_im_p_f[2] = {r_im_UL_f[eNB_id][0],r_im_UL_f[eNB_id][1]};
#endif

#ifdef    __AVX2__ 
    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple_freq); 
    rf_rx_simple_freq_AVX_float(r_re_p_f,
		 r_im_p_f,
		 nb_antennas_rx,
		 frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
  		 (float)1e3/UE2eNB[0][eNB_id][CC_id]->sampling_rate,  // sampling time (ns)
		 (float)PHY_vars_eNB_g[eNB_id][CC_id]->rx_total_gain_dB - 66.227,   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
		 frame_parms->symbols_per_tti,
		 frame_parms->ofdm_symbol_size,
		 12.0*frame_parms->N_RB_DL);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple_freq);
#else
    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple_freq);
    rf_rx_simple_freq(r_re_p_f,
		 r_im_p_f,
		 nb_antennas_rx,
		 frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
  		 1e3/UE2eNB[0][eNB_id][CC_id]->sampling_rate,  // sampling time (ns)
		 (double)PHY_vars_eNB_g[eNB_id][CC_id]->rx_total_gain_dB - 66.227,   // rx_gain (dB) (66.227 = 20*log10(pow2(11)) = gain from the adc that will be applied later)
		 frame_parms->symbols_per_tti,
		 frame_parms->ofdm_symbol_size,
		 12.0*frame_parms->N_RB_DL);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_rf_rx_simple_freq);
#endif

    //print_meas (&UE2eNB[UE_id][eNB_id][CC_id]->UL_rf_rx_simple_freq,"[UL][rf_rx_simple_freq]", &UE2eNB[UE_id][eNB_id][CC_id]->UL_rf_rx_simple_freq, &UE2eNB[UE_id][eNB_id][CC_id]->UL_rf_rx_simple_freq);
    /*for (int idx=subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx<(subframe+1)*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx++){
    if (((short*)&txdataF[0][idx])[0]!=0 || ((short*)&txdataF[0][idx])[1]!=0)
	printf("dumping UL raw rx subframe (input) %d: rxdataF[%d] = (%f,%f)\n", subframe, idx, r_re_p_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti],r_im_p_f[0][idx-subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti]);
		}*/
	
#ifdef DEBUG_SIM
    rx_pwr = signal_energy_fp(r_re_p_f,r_im_p_f,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,0)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
    LOG_D(OCM,"[SIM][UL] rx_pwr (ADC in) %f dB for subframe %d\n",10*log10(rx_pwr),subframe);
#endif
    rxdataF = PHY_vars_eNB_g[eNB_id][CC_id]->common_vars.rxdataF[0];
    sf_offset = 0;//subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;

#ifdef    __AVX2__
    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc_freq);
    adc_AVX_float(r_re_p_f,
		r_im_p_f,
		sf_offset,
		sf_offset,
		rxdataF,
		nb_antennas_rx,
		frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		12,
		PHY_vars_eNB_g[eNB_id][CC_id]->frame_parms.N_RB_DL*12,
		frame_parms->ofdm_symbol_size);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc_freq);
#else
    start_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc_freq);
    adc(r_re_p_f,
		r_im_p_f,
		sf_offset,
		sf_offset,
		rxdataF,
		nb_antennas_rx,
		frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,
		12);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->UL_adc_freq);
#endif
    /*for (int idx=0;idx<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti;idx++){
	if (((short*)&txdataF[0][idx+subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti])[0]!=0 || ((short*)&txdataF[0][idx+subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti])[1]!=0)
		printf("dumping UL raw rx subframe %d: rxdataF[%d] = (%d,%d)\n", subframe, idx, ((short*)&rxdataF[0][idx])[0], ((short*)&rxdataF[0][idx])[1]);
	}*/
    //write_output("chsim_rxsigF_UL.m","chsm_rxsF_UL", (void*)rxdataF[0],2*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16); 
    
#ifdef DEBUG_SIM
    //rx_pwr2 = signal_energy(rxdataF[0]+sf_offset,subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
    //LOG_D(OCM,"[SIM][UL] eNB %d rx_pwr (ADC out) %f dB (%d) for subframe %d (offset %d)\n",eNB_id,10*log10((double)rx_pwr2),rx_pwr2,subframe,sf_offset);

#else
    UNUSED_VARIABLE(tx_pwr);
    UNUSED_VARIABLE(rx_pwr);
    UNUSED_VARIABLE(rx_pwr2);
#endif
  } // abstraction_flag==0

}

void do_UL_sig_freq_prach(channel_desc_t *UE2eNB[NUMBER_OF_UE_MAX][NUMBER_OF_eNB_MAX][MAX_NUM_CCs],
               node_desc_t *enb_data[NUMBER_OF_eNB_MAX],node_desc_t *ue_data[NUMBER_OF_UE_MAX],
	       uint16_t subframe,uint8_t abstraction_flag,LTE_DL_FRAME_PARMS *frame_parms, 
	       uint32_t frame,int eNB_id,uint8_t CC_id)
{
  /*static int first_run=0;
  static double sum;
  static int count;
  if (!first_run)
  {
     first_run=1;
     sum=0;
     count=0;
  } 
  count++;*/
 

#ifdef PHY_ABSTRACTION_UL
  int32_t att_eNB_id=-1;
#endif
  uint8_t UE_id=0;
  int16_t **rx_prachF;
  int16_t *tx_prachF;
  uint8_t nb_antennas_rx = UE2eNB[0][0][CC_id]->nb_rx; // number of rx antennas at eNB
  uint8_t nb_antennas_tx = UE2eNB[0][0][CC_id]->nb_tx; // number of tx antennas at UE
#ifdef    __AVX2__
  float tx_pwr, rx_pwr;
#else
  double tx_pwr, rx_pwr;
#endif
  int32_t rx_pwr2;
  uint32_t i,aa;
  uint32_t sf_offset;

  uint8_t hold_channel=0;
  int n_ra_prb;

#ifdef PHY_ABSTRACTION_UL
  double min_path_loss=-200;
  uint16_t ul_nb_rb=0 ;
  uint16_t ul_fr_rb=0;
  int ulnbrb2 ;
  int ulfrrb2 ;
  uint8_t harq_pid;
#endif
#ifdef    __AVX2__
  float s_re0_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float s_re1_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float *s_re_f_prach[2];
  float s_im0_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float s_im1_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float *s_im_f_prach[2];
  float r_re00_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float r_re01_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float *r_re0_f_prach[2];
  float r_im00_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float r_im01_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  float *r_im0_f_prach[2];
#else
  double s_re0_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double s_re1_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double *s_re_f_prach[2];
  double s_im0_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double s_im1_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double *s_im_f_prach[2];
  double r_re00_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double r_re01_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double *r_re0_f_prach[2];
  double r_im00_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double r_im01_f_prach[2048*14*12];//ofdm_symbol_size*symbols_per_tti;
  double *r_im0_f_prach[2];
#endif

  s_re_f_prach[0] = s_re0_f_prach;
  s_im_f_prach[0] = s_im0_f_prach;
  s_re_f_prach[1] = s_re1_f_prach;
  s_im_f_prach[1] = s_im1_f_prach;

  r_re0_f_prach[0] = r_re00_f_prach;
  r_im0_f_prach[0] = r_im00_f_prach;
  r_re0_f_prach[1] = r_re01_f_prach;
  r_im0_f_prach[1] = r_im01_f_prach;

  uint8_t prach_ConfigIndex;
  uint8_t prach_fmt=0;
  int pointer_firstvalue_PRACH=0;

  if (abstraction_flag!=0)  {
#ifdef PHY_ABSTRACTION_UL
	LOG_D(OCM,"[SIM][UL] UE %d, Abstraction for do_UL_prach is not implemented in frequency domain\n",UE_id);
	exit(-1);
#endif
  } else { //without abstraction

    pthread_mutex_lock(&UE_output_mutex[eNB_id]);
    // Clear RX signal for eNB = eNB_id
    	//for (i=0; i<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti; i++) {
	      for (aa=0; aa<nb_antennas_rx; aa++) {
		/*r_re_UL_f_prach[eNB_id][aa][i]=0.0;
		r_im_UL_f_prach[eNB_id][aa][i]=0.0;
	      }*/
	      memset((void*)r_re_UL_f_prach[eNB_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
	      memset((void*)r_im_UL_f_prach[eNB_id][aa],0,(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*sizeof(double));
        }
    pthread_mutex_unlock(&UE_output_mutex[eNB_id]);

    //for (int i=0;i<NB_UE_INST;i++)
    // Compute RX signal for eNB = eNB_id
    for (UE_id=0; UE_id<NB_UE_INST; UE_id++) {
	if (PHY_vars_UE_g[UE_id][CC_id]->common_vars.eNb_id != eNB_id)
		continue;      
        lte_frame_type_t frame_type = PHY_vars_UE_g[UE_id][CC_id]->frame_parms.frame_type;
        prach_ConfigIndex   = PHY_vars_UE_g[UE_id][CC_id]->frame_parms.prach_config_common.prach_ConfigInfo.prach_ConfigIndex;
        prach_fmt = get_prach_fmt(prach_ConfigIndex,frame_type);
        n_ra_prb = get_prach_prb_offset(frame_parms, PHY_vars_UE_g[UE_id][CC_id]->prach_resources[eNB_id]->ra_TDD_map_index, PHY_vars_UE_g[UE_id][CC_id]->proc.proc_rxtx[subframe&0x1].frame_tx);

	tx_prachF = PHY_vars_UE_g[UE_id][CC_id]->prach_vars[eNB_id]->prachF;
	//write_output("txprachF.m","prach_txF", PHY_vars_UE_g[0][0]->prach_vars[0]->prachF,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti*12,1,1);
      	
        //for (int idx=0;idx<10;idx++) printf("dumping DL raw subframe %d: txdataF[%d] = (%d,%d)\n", subframe, idx, ((short*)&txdataF[0][sf_offset+idx])[0], ((short*)&txdataF[0][sf_offset+idx])[1]);

	pointer_firstvalue_PRACH=((12*n_ra_prb) - 6*PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_UL<0)?(((12*n_ra_prb) - 6*PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_UL+PHY_vars_UE_g[UE_id][CC_id]->frame_parms.ofdm_symbol_size)*12+13)*2:(((12*n_ra_prb) - 6*PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_UL)*12+13)*2;
	sf_offset = pointer_firstvalue_PRACH;

        //for (int idx=pointer_firstvalue_PRACH;idx<pointer_firstvalue_PRACH+20;idx+=2) printf("dumping PRACH UL raw, subframe %d: ue->prachF[%d] = (%d,%d)\n", subframe, idx/2, (short)tx_prachF[idx], (short)tx_prachF[idx+1]);

        //for (int idx=pointer_firstvalue_PRACH+2*839-20;idx<pointer_firstvalue_PRACH+2*839;idx+=2) printf("dumping PRACH UL raw, subframe %d: ue->prachF[%d] = (%d,%d)\n", subframe, idx/2, (short)tx_prachF[idx], (short)tx_prachF[idx+1]);
      if (((double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe] +
	   UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB) <= -125.0) {
	// don't simulate a UE that is too weak
	/*LOG_D(OCM,"[SIM][UL] PRACH:UE %d tx_pwr %d dBm (num_RE %d) for subframe %d (sf_offset %d)\n",
	      UE_id,
	      PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
	      PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
	      subframe,sf_offset);*/
	      //printf("multipath_channel_prach, UE too weak %e\n", ((double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe] +
	   //UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB));	
      } else {
	     
#ifdef    __AVX2__
	     start_meas(&UE2eNB[0][eNB_id][CC_id]->dac_fixed_gain_PRACH);
	     tx_pwr = dac_fixed_gain_prach_AVX_float((float**)s_re_f_prach,
					(float**)s_im_f_prach,
					(int *)tx_prachF,
					pointer_firstvalue_PRACH,
					nb_antennas_tx,
					(prach_fmt<4)?839:139,
					pointer_firstvalue_PRACH,
					(prach_fmt<4)?839:139,
					14,
					(float)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe]-10*log10((double)PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]),
					PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
					PHY_vars_UE_g[UE_id][CC_id]->frame_parms.ofdm_symbol_size);  // This make the previous argument the total power
	     stop_meas(&UE2eNB[0][eNB_id][CC_id]->dac_fixed_gain_PRACH);
#else
	     start_meas(&UE2eNB[0][eNB_id][CC_id]->dac_fixed_gain_PRACH);
	     tx_pwr = dac_fixed_gain_prach((double**)s_re_f_prach,
					(double**)s_im_f_prach,
					(int *)tx_prachF,
					pointer_firstvalue_PRACH,
					nb_antennas_tx,
					(prach_fmt<4)?839:139,
					pointer_firstvalue_PRACH,
					(prach_fmt<4)?839:139,
					14,
					(double)PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe]-10*log10((double)PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe]),
					PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
					PHY_vars_UE_g[UE_id][CC_id]->frame_parms.ofdm_symbol_size);  // This make the previous argument the total power
	     stop_meas(&UE2eNB[0][eNB_id][CC_id]->dac_fixed_gain_PRACH);
#endif
	    
	    //for (int idx=0;idx<10;idx++) printf("dumping raw PRACH UL tx subframe (input) %d: s_f[%d] = (%f,%f)\n", subframe, idx, s_re_f_prach[0][idx],s_im_f_prach[0][idx]);
	    //for (int idx=829;idx<839;idx++) printf("dumping raw PRACH UL tx subframe (input) %d: s_f[%d] = (%f,%f)\n", subframe, idx, s_re_f_prach[0][idx],s_im_f_prach[0][idx]);
	    /*LOG_D(OCM,"[SIM][UL] UE %d tx_pwr %f dBm (target %d dBm, num_RE %d) for subframe %d (sf_offset %d)\n",
		      UE_id,
		      10*log10(tx_pwr),
		      PHY_vars_UE_g[UE_id][CC_id]->tx_power_dBm[subframe],
		      PHY_vars_UE_g[UE_id][CC_id]->tx_total_RE[subframe],
		      subframe,sf_offset);*/

	    

	    // write_output("s_re_f_prach.m","s_re_f_prach_txF", s_re_f_prach,frame_parms->ofdm_symbol_size*12,1,1);
	    
#ifdef    __AVX2__
	    start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->multipath_channel_freq_PRACH);
	    multipath_channel_prach_AVX_float(UE2eNB[UE_id][eNB_id][CC_id],s_re_f_prach,s_im_f_prach,r_re0_f_prach,r_im0_f_prach,&PHY_vars_UE_g[UE_id][CC_id]->frame_parms,
			  (prach_fmt<4)?13+839+12:3+139+2,hold_channel,eNB_id,prach_fmt,n_ra_prb);
	    stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->multipath_channel_freq_PRACH);
#else
	    start_meas(&UE2eNB[UE_id][eNB_id][CC_id]->multipath_channel_freq_PRACH);
	    multipath_channel_prach(UE2eNB[UE_id][eNB_id][CC_id],s_re_f_prach,s_im_f_prach,r_re0_f_prach,r_im0_f_prach,&PHY_vars_UE_g[UE_id][CC_id]->frame_parms,
			  (prach_fmt<4)?13+839+12:3+139+2,hold_channel,eNB_id,prach_fmt,n_ra_prb);
	    stop_meas(&UE2eNB[UE_id][eNB_id][CC_id]->multipath_channel_freq_PRACH);
#endif

	    //for (int idx=0;idx<10;idx++) printf("dumping raw PRACH UL tx subframe (output) %d: r_f[%d] = (%f,%f)\n", subframe, idx, r_re0_f_prach[0][idx],r_im0_f_prach[0][idx]);
	    //for (int idx=829;idx<839;idx++) printf("dumping raw PRACH UL tx subframe (output) %d: r_f[%d] = (%f,%f)\n", subframe, idx, r_re0_f_prach[0][idx],r_im0_f_prach[0][idx]);
		//write_output("txprachF.m","prach_txF", PHY_vars_UE_g[0][0]->prach_vars[0]->prachF,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,1);
	     rx_pwr = signal_energy_fp2(UE2eNB[UE_id][eNB_id][CC_id]->chF[0],(12*frame_parms->N_RB_DL+1))*(12*frame_parms->N_RB_DL+1);
	     /*LOG_D(OCM,"[SIM][UL] subframe %d Channel UE %d => eNB %d : %f dB (hold %d,length %d, PL %f)\n",subframe,UE_id,eNB_id,10*log10(rx_pwr),
	     hold_channel,12*frame_parms->N_RB_DL+1,
	     UE2eNB[UE_id][eNB_id][CC_id]->path_loss_dB);*/
#ifdef    __AVX2__ 
	     rx_pwr = signal_energy_fp_AVX_float(r_re0_f_prach,r_im0_f_prach,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti*12,0);
#else
	     rx_pwr = signal_energy_fp(r_re0_f_prach,r_im0_f_prach,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti*12,0);
#endif
	     /*LOG_D(OCM,"[SIM][UL] eNB %d : rx_pwr %f dBm (%f) for subframe %d, sptti %d\n",
	      eNB_id,10*log10(rx_pwr),rx_pwr,subframe,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti*12);*/
#ifdef DEBUG_SIM

      	/*for (i=0; i<10; i++)
        LOG_D(OCM,"do_UL_prach channel(%d,%d)[%d] : (%f,%f)\n",UE_id,eNB_id,i,UE2eNB[UE_id][eNB_id][CC_id]->chF_prach[0][i].x,UE2eNB[UE_id][eNB_id][CC_id]->chF_prach[0][i].y);*/

#endif	
	     /*for (i=0; i<864; i++){
        	LOG_D(OCM,"channel_prach(%d,%d)[%d] : (%f,%f)\n",eNB_id,UE_id,i,UE2eNB[UE_id][eNB_id][CC_id]->chF_prach[0][i].x,UE2eNB[UE_id][eNB_id][CC_id]->chF_prach[0][i].y);
	     }*/
	if (UE2eNB[UE_id][eNB_id][CC_id]->first_run == 1)
	  UE2eNB[UE_id][eNB_id][CC_id]->first_run = 0;
	
	__m256 r_re0_f_prach_256,r_im0_f_prach_256,r_re_UL_f_prach_256,r_im_UL_f_prach_256;
	pthread_mutex_lock(&UE_output_mutex[eNB_id]);
#ifdef    __AVX2__
	for (aa=0; aa<nb_antennas_rx; aa++) {
		for (i=0; i<(frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)>>3; i++) {
		    //r_re_UL_f_prach[eNB_id][aa][i]+=r_re0_f_prach[aa][i];
		    //r_im_UL_f_prach[eNB_id][aa][i]+=r_im0_f_prach[aa][i];

		    r_re0_f_prach_256 = _mm256_loadu_ps(&r_re0_f_prach[aa][8*i]);
		    r_im0_f_prach_256 = _mm256_loadu_ps(&r_im0_f_prach[aa][8*i]);
		    r_re_UL_f_prach_256 = _mm256_loadu_ps(&r_re_UL_f_prach[eNB_id][aa][8*i]);
		    r_im_UL_f_prach_256 = _mm256_loadu_ps(&r_im_UL_f_prach[eNB_id][aa][8*i]);

		    r_re_UL_f_prach_256 = _mm256_add_ps(r_re_UL_f_prach_256,r_re0_f_prach_256);
		    r_im_UL_f_prach_256 = _mm256_add_ps(r_im_UL_f_prach_256,r_im0_f_prach_256);

		    _mm256_storeu_ps(&r_re_UL_f_prach[eNB_id][aa][8*i],r_re_UL_f_prach_256);
		    _mm256_storeu_ps(&r_im_UL_f_prach[eNB_id][aa][8*i],r_im_UL_f_prach_256);
		}
	}
#else
	for (aa=0; aa<nb_antennas_rx; aa++) {
		for (i=0; i<frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti; i++) {
		    r_re_UL_f_prach[eNB_id][aa][i]+=r_re0_f_prach[aa][i];
		    r_im_UL_f_prach[eNB_id][aa][i]+=r_im0_f_prach[aa][i];
		}
	}
#endif
	pthread_mutex_unlock(&UE_output_mutex[eNB_id]);
      }
    } //UE_id
#ifdef    __AVX2__    
    float *r_re_p_f_prach[2] = {r_re_UL_f_prach[eNB_id][0],r_re_UL_f_prach[eNB_id][1]};
    float *r_im_p_f_prach[2] = {r_im_UL_f_prach[eNB_id][0],r_im_UL_f_prach[eNB_id][1]};
#else
    double *r_re_p_f_prach[2] = {r_re_UL_f_prach[eNB_id][0],r_re_UL_f_prach[eNB_id][1]};
    double *r_im_p_f_prach[2] = {r_im_UL_f_prach[eNB_id][0],r_im_UL_f_prach[eNB_id][1]};
#endif
    /*for (int idx=0;idx<10;idx++) printf("dumping raw PRACH UL tx subframe (output) %d: r_re_im_p_f_prach[%d] = (%d,%d)\n", subframe, idx, (short)(r_re_p_f_prach[0][idx]),(short)(r_im_p_f_prach[0][idx]));
    for (int idx=829;idx<839;idx++) printf("dumping raw PRACH UL tx subframe (output) %d: r_re_im_p_f_prach[%d] = (%d,%d)\n", subframe, idx, (short)(r_re_p_f_prach[0][idx]),(short)(r_im_p_f_prach[0][idx]));*/
		//clock_t start=clock();
#ifdef    __AVX2__  
    start_meas(&UE2eNB[0][eNB_id][CC_id]->rf_rx_simple_freq_PRACH);
    rf_rx_simple_freq_AVX_float(r_re_p_f_prach,
		 r_im_p_f_prach,
		 nb_antennas_rx,
		 (prach_fmt<4)?839:139,
		 (float)1e3/UE2eNB[0][eNB_id][CC_id]->sampling_rate,  // sampling time (ns)
		 (float)PHY_vars_eNB_g[eNB_id][CC_id]->rx_total_gain_dB - 66.227,
		 frame_parms->symbols_per_tti,
		 frame_parms->ofdm_symbol_size,
		 12.0*frame_parms->N_RB_DL);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->rf_rx_simple_freq_PRACH);
#else
    start_meas(&UE2eNB[0][eNB_id][CC_id]->rf_rx_simple_freq_PRACH);
    rf_rx_simple_freq(r_re_p_f_prach,
		 r_im_p_f_prach,
		 nb_antennas_rx,
		 (prach_fmt<4)?839:139,
		 1e3/UE2eNB[0][eNB_id][CC_id]->sampling_rate,  // sampling time (ns)
		 (double)PHY_vars_eNB_g[eNB_id][CC_id]->rx_total_gain_dB - 66.227,
		 frame_parms->symbols_per_tti,
		 frame_parms->ofdm_symbol_size,
		 12.0*frame_parms->N_RB_DL);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->rf_rx_simple_freq_PRACH);
#endif
    

  /*clock_t stop=clock();
  printf("UE_PRACH_channel time is %f s, AVERAGE time is %f s, count %d, sum %e, subframe %d\n",(float) (stop-start)/CLOCKS_PER_SEC,(float) (sum+stop-start)/(count*CLOCKS_PER_SEC),count,sum+stop-start,subframe);
  sum=(sum+stop-start);*/
#ifdef DEBUG_SIM

    rx_pwr = signal_energy_fp(r_re_p_f_prach,r_im_p_f_prach,nb_antennas_rx,frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti*12,0)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
    /*LOG_D(OCM,"[SIM][UL] rx_pwr (ADC in) %f dB for subframe %d\n",10*log10(rx_pwr),subframe);*/

#endif
     rx_prachF = PHY_vars_eNB_g[eNB_id][CC_id]->prach_vars.rxsigF;
     sf_offset = pointer_firstvalue_PRACH;

#ifdef    __AVX2__ 
     start_meas(&UE2eNB[0][eNB_id][CC_id]->adc_PRACH);
     adc_prach_AVX_float(r_re_p_f_prach,
		r_im_p_f_prach,
		0,
		sf_offset,
		(int **)rx_prachF,
		nb_antennas_rx,
		(prach_fmt<4)?839:139,
		12);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->adc_PRACH);
#else
     start_meas(&UE2eNB[0][eNB_id][CC_id]->adc_PRACH);
     adc_prach(r_re_p_f_prach,
		r_im_p_f_prach,
		0,
		sf_offset,
		(int **)rx_prachF,
		nb_antennas_rx,
		(prach_fmt<4)?839:139,
		12);
    stop_meas(&UE2eNB[0][eNB_id][CC_id]->adc_PRACH);
#endif
     
    //write_output("chsim_rxsigF.m","chsim_rx_sigF", PHY_vars_eNB_g[eNB_id][CC_id]->prach_vars.rxsigF[0],4*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti,1,16);
#ifdef DEBUG_SIM

	    //rx_pwr2 = signal_energy(rxdataF[0]+sf_offset,subframe*frame_parms->ofdm_symbol_size*frame_parms->symbols_per_tti)*(double)frame_parms->ofdm_symbol_size/(12.0*frame_parms->N_RB_DL);
	    //LOG_D(OCM,"[SIM][UL] eNB %d rx_pwr (ADC out) %f dB (%d) for subframe %d (offset %d)\n",eNB_id,10*log10((double)rx_pwr2),rx_pwr2,subframe,sf_offset);

#else
    UNUSED_VARIABLE(tx_pwr);
    UNUSED_VARIABLE(rx_pwr);
    UNUSED_VARIABLE(rx_pwr2);
#endif
    
  } // abstraction_flag==0
}

void init_channel_vars(LTE_DL_FRAME_PARMS *frame_parms, double ***s_re,double ***s_im,double ***r_re,double ***r_im,double ***r_re0,double ***r_im0)
{

  int i;

  memset(eNB_output_mask,0,sizeof(int)*NUMBER_OF_UE_MAX);
  for (i=0;i<NB_UE_INST;i++)
    pthread_mutex_init(&eNB_output_mutex[i],NULL);

  memset(UE_output_mask,0,sizeof(int)*NUMBER_OF_eNB_MAX);
  for (i=0;i<NB_eNB_INST;i++)
    pthread_mutex_init(&UE_output_mutex[i],NULL);

}
void init_channel_vars_freq(LTE_DL_FRAME_PARMS *frame_parms, double ***s_re_f,double ***s_im_f,double ***r_re_f,double ***r_im_f,double ***r_re0_f,double ***r_im0_f)
{

  int i, eNB_id, UE_id;

  memset(eNB_output_mask,0,sizeof(int)*NUMBER_OF_UE_MAX);
  for (i=0;i<NB_UE_INST;i++)
    pthread_mutex_init(&eNB_output_mutex[i],NULL);

  memset(UE_output_mask,0,sizeof(int)*NUMBER_OF_eNB_MAX);
  for (i=0;i<NB_eNB_INST;i++){
    pthread_mutex_init(&UE_output_mutex[i],NULL);
    pthread_mutex_init(&UE_PRACH_output_mutex[i],NULL);
  }
}
