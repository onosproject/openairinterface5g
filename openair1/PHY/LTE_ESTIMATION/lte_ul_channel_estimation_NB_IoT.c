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

/*! \file PHY/LTE_TRANSPORT/lte_ul_channel_estimation_NB_IoT.c
* \brief Channel estimation 
* \author Vincent Savaux
* \date 2017
* \version 0.1
* \company b<>com
* \email: vincent.savaux@b<>com.com
* \note
* \warning
*/

#include "PHY/defs_NB_IoT.h"
#include "PHY/extern_NB_IoT.h"
//#include "PHY/sse_intrin.h"
#include <math.h>

#include "PHY/LTE_ESTIMATION/defs_NB_IoT.h"
#include "PHY/LTE_TRANSPORT/extern_NB_IoT.h"
//#define DEBUG_CH

#include "T.h"

// For Channel Estimation in Distributed Alamouti Scheme
//static int16_t temp_out_ifft[2048*4] __attribute__((aligned(16)));
/*
static int16_t temp_out_fft_0[2048*4] __attribute__((aligned(16)));
static int16_t temp_out_fft_1[2048*4] __attribute__((aligned(16)));
static int16_t temp_out_ifft_0[2048*4] __attribute__((aligned(16)));
static int16_t temp_out_ifft_1[2048*4] __attribute__((aligned(16)));
*/

// static int32_t temp_in_ifft_0[2048*2] __attribute__((aligned(32)));
//static int32_t temp_in_ifft_1[2048*2] __attribute__((aligned(32)));
//static int32_t temp_in_fft_0[2048*2] __attribute__((aligned(16)));
// static int32_t temp_in_fft_1[2048*2] __attribute__((aligned(16)));

// round(exp(sqrt(-1)*(pi/2)*[0:1:N-1]/N)*pow2(15))
// static int16_t ru_90[2*128] = {32767, 0,32766, 402,32758, 804,32746, 1206,32729, 1608,32706, 2009,32679, 2411,32647, 2811,32610, 3212,32568, 3612,32522, 4011,32470, 4410,32413, 4808,32352, 5205,32286, 5602,32214, 5998,32138, 6393,32058, 6787,31972, 7180,31881, 7571,31786, 7962,31686, 8351,31581, 8740,31471, 9127,31357, 9512,31238, 9896,31114, 10279,30986, 10660,30853, 11039,30715, 11417,30572, 11793,30425, 12167,30274, 12540,30118, 12910,29957, 13279,29792, 13646,29622, 14010,29448, 14373,29269, 14733,29086, 15091,28899, 15447,28707, 15800,28511, 16151,28311, 16500,28106, 16846,27897, 17190,27684, 17531,27467, 17869,27246, 18205,27020, 18538,26791, 18868,26557, 19195,26320, 19520,26078, 19841,25833, 20160,25583, 20475,25330, 20788,25073, 21097,24812, 21403,24548, 21706,24279, 22006,24008, 22302,23732, 22595,23453, 22884,23170, 23170,22884, 23453,22595, 23732,22302, 24008,22006, 24279,21706, 24548,21403, 24812,21097, 25073,20788, 25330,20475, 25583,20160, 25833,19841, 26078,19520, 26320,19195, 26557,18868, 26791,18538, 27020,18205, 27246,17869, 27467,17531, 27684,17190, 27897,16846, 28106,16500, 28311,16151, 28511,15800, 28707,15447, 28899,15091, 29086,14733, 29269,14373, 29448,14010, 29622,13646, 29792,13279, 29957,12910, 30118,12540, 30274,12167, 30425,11793, 30572,11417, 30715,11039, 30853,10660, 30986,10279, 31114,9896, 31238,9512, 31357,9127, 31471,8740, 31581,8351, 31686,7962, 31786,7571, 31881,7180, 31972,6787, 32058,6393, 32138,5998, 32214,5602, 32286,5205, 32352,4808, 32413,4410, 32470,4011, 32522,3612, 32568,3212, 32610,2811, 32647,2411, 32679,2009, 32706,1608, 32729,1206, 32746,804, 32758,402, 32766};

// static int16_t ru_90c[2*128] = {32767, 0,32766, -402,32758, -804,32746, -1206,32729, -1608,32706, -2009,32679, -2411,32647, -2811,32610, -3212,32568, -3612,32522, -4011,32470, -4410,32413, -4808,32352, -5205,32286, -5602,32214, -5998,32138, -6393,32058, -6787,31972, -7180,31881, -7571,31786, -7962,31686, -8351,31581, -8740,31471, -9127,31357, -9512,31238, -9896,31114, -10279,30986, -10660,30853, -11039,30715, -11417,30572, -11793,30425, -12167,30274, -12540,30118, -12910,29957, -13279,29792, -13646,29622, -14010,29448, -14373,29269, -14733,29086, -15091,28899, -15447,28707, -15800,28511, -16151,28311, -16500,28106, -16846,27897, -17190,27684, -17531,27467, -17869,27246, -18205,27020, -18538,26791, -18868,26557, -19195,26320, -19520,26078, -19841,25833, -20160,25583, -20475,25330, -20788,25073, -21097,24812, -21403,24548, -21706,24279, -22006,24008, -22302,23732, -22595,23453, -22884,23170, -23170,22884, -23453,22595, -23732,22302, -24008,22006, -24279,21706, -24548,21403, -24812,21097, -25073,20788, -25330,20475, -25583,20160, -25833,19841, -26078,19520, -26320,19195, -26557,18868, -26791,18538, -27020,18205, -27246,17869, -27467,17531, -27684,17190, -27897,16846, -28106,16500, -28311,16151, -28511,15800, -28707,15447, -28899,15091, -29086,14733, -29269,14373, -29448,14010, -29622,13646, -29792,13279, -29957,12910, -30118,12540, -30274,12167, -30425,11793, -30572,11417, -30715,11039, -30853,10660, -30986,10279, -31114,9896, -31238,9512, -31357,9127, -31471,8740, -31581,8351, -31686,7962, -31786,7571, -31881,7180, -31972,6787, -32058,6393, -32138,5998, -32214,5602, -32286,5205, -32352,4808, -32413,4410, -32470,4011, -32522,3612, -32568,3212, -32610,2811, -32647,2411, -32679,2009, -32706,1608, -32729,1206, -32746,804, -32758,402, -32766};

#define SCALE 0x3FFF 


void rotate_channel_single_carrier_NB_IoT(int16_t *estimated_channel,unsigned char l, uint8_t Qm)
{
  int16_t pi_2_re[2] = {32767 , 0}; 
  int16_t pi_2_im[2] = {0 , 32768}; 
  int16_t pi_4_re[2] = {32767 , 25735}; 
  int16_t pi_4_im[2] = {0 , 25736}; 
  int k; 
  int16_t est_channel_re, est_channel_im;    

  for (k=0;k<12;k++){
    est_channel_re = estimated_channel[k<<1]; 
    est_channel_im = estimated_channel[(k<<1)+1]; 

    if (Qm == 1){
      estimated_channel[k<<1] = (int16_t)(((int32_t)pi_2_re[l%2] * (int32_t)est_channel_re + 
                          (int32_t)pi_2_im[l%2] * (int32_t)est_channel_im)>>15); 
      estimated_channel[(k<<1)+1] = (int16_t)(((int32_t)pi_2_re[l%2] * (int32_t)est_channel_im - 
                          (int32_t)pi_2_im[l%2] * (int32_t)est_channel_re)>>15); 
    }
    if(Qm == 2){
      estimated_channel[k<<1] = (int16_t)(((int32_t)pi_4_re[l%2] * (int32_t)est_channel_re + 
                          (int32_t)pi_4_im[l%2] * (int32_t)est_channel_im)>>15); 
      estimated_channel[(k<<1)+1] = (int16_t)(((int32_t)pi_4_re[l%2] * (int32_t)est_channel_im - 
                          (int32_t)pi_4_im[l%2] * (int32_t)est_channel_re)>>15); 
    }

  }

} 

int32_t ul_channel_estimation_NB_IoT(PHY_VARS_eNB_NB_IoT      *eNB,
                                         eNB_rxtx_proc_NB_IoT_t   *proc,
                                         uint8_t                  eNB_id,
                                         uint8_t                  UE_id,
                                         unsigned char            l,
                                         unsigned char            Ns,
                                         uint8_t                  cooperation_flag)
{

  NB_IoT_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  NB_IoT_eNB_PUSCH *pusch_vars = eNB->pusch_vars[UE_id];
  int32_t **ul_ch_estimates=pusch_vars->drs_ch_estimates[eNB_id];
  //int32_t **ul_ch_estimates_time=  pusch_vars->drs_ch_estimates_time[eNB_id];
  //int32_t **ul_ch_estimates_0=  pusch_vars->drs_ch_estimates_0[eNB_id];
  //int32_t **ul_ch_estimates_1=  pusch_vars->drs_ch_estimates_1[eNB_id];
  int32_t **rxdataF_ext=  pusch_vars->rxdataF_ext[eNB_id];
  int subframe = proc->subframe_rx;
  //uint8_t harq_pid = subframe2harq_pid_NB_IoT(frame_parms,proc->frame_rx,subframe);
  //int16_t delta_phase = 0;
  // int16_t *ru1 = ru_90;
  // int16_t *ru2 = ru_90;
  //int16_t current_phase1,current_phase2;
  // uint16_t N_rb_alloc = eNB->ulsch[UE_id]->harq_process->nb_rb;
  uint16_t aa; //,Msc_RS,Msc_RS_idx;
  //uint16_t * Msc_idx_ptr;
  // int k,pilot_pos1 = 3 - frame_parms->Ncp, pilot_pos2 = 10 - 2*frame_parms->Ncp; 
  int pilot_pos1_15k = 3, pilot_pos2_15k = 10; // holds for npusch format 1, and 15 kHz subcarrier bandwidth
  int pilot_pos_format2_15k[6] = {2,3,4,9,10,11}; // holds for npusch format 2, and 15 kHz subcarrier bandwidth 
  int pilot_pos1_3_75k = 4, pilot_pos2_3_75k = 11; // holds for npusch format 1, and 3.75 kHz subcarrier bandwidth
  int pilot_pos_format2_3_75k[6] = {0,1,2,7,8,9}; // holds for npusch format 2, and 3.75 kHz subcarrier bandwidth 

  int pilot_pos1, pilot_pos2; // holds for npusch format 1, and 15 kHz subcarrier bandwidth
  int *pilot_pos_format2; // holds for npusch format 2, and 15 kHz subcarrier bandwidth
  int16_t *ul_ch1=NULL, *ul_ch2=NULL, *ul_ch3=NULL, *ul_ch4=NULL, *ul_ch5=NULL, *ul_ch6=NULL; 
  int16_t average_channel[24]; // average channel over a RB and 2 slots
  int32_t *p_average_channel = (int32_t *)&average_channel; 
  //int32_t *ul_ch1_0=NULL,*ul_ch2_0=NULL,*ul_ch1_1=NULL,*ul_ch2_1=NULL;
  int16_t ul_ch_estimates_re,ul_ch_estimates_im;

  //uint8_t nb_antennas_rx = frame_parms->nb_antenna_ports_eNB;
  uint8_t nb_antennas_rx = frame_parms->nb_antennas_rx; 
  uint8_t subcarrier_spacing = frame_parms->subcarrier_spacing; // 15 kHz or 3.75 kHz 

  uint8_t Qm; // needed to rotate the estimated channel
  //uint32_t alpha_ind;
  uint32_t u;//=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[Ns+(subframe<<1)];
  //uint32_t v=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.seqhop[Ns+(subframe<<1)];
  // int32_t tmp_estimates[N_rb_alloc*12] __attribute__((aligned(16)));

  int symbol_offset, k, i, n, p; 
  uint16_t Ncell_ID = frame_parms->Nid_cell; 
  uint32_t x1, x2, s=0;
  int n_s; // slot in frame 
  uint8_t reset=1, index_w; 

  ////// NB-IoT specific ///////////////////////////////////////////////////////////////////////////////////////

  uint32_t I_sc = eNB->ulsch[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
  uint16_t ul_sc_start; // subcarrier start index into UL RB 

  // 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-3 
  int16_t alpha3_re[9] = {32767 , 32767, 32767, 
                      32767, -16384, -16384, 
                      32767, -16384, -16384}; 
  int16_t alpha3_im[9] = {0 , 0, 0, 
                      0, 28377, -28378, 
                      0, -28378, 28377};                     
  int16_t alpha6_re[24] = {32767 , 32767, 32767, 32767, 32767, 32767, 
                      32767, 16383, -16384, -32768, -16384, 16383, 
                      32767, -16384, -16384, 32767, -16384, -16384, 
                      32767, -16384, -16384, 32767, -16384, -16384}; 
  int16_t alpha6_im[24] = {0 , 0, 0, 0, 0, 0, 
                      0, 28377, 28377, 0, -28378, -28378, 
                      0, 28377, -28378, 0, 28377, -28378,
                      0, -28378, 28377, 0, -28378, 28377}; 

  // 36.211, Table 5.5.2.2.1-2 --> used for pilots in NPUSCH format 2
  int16_t w_format2_re[9] = {32767 , 32767, 32767, 
                      32767, -16384, -16384, 
                      32767, -16384, -16384}; 
  int16_t w_format2_im[9] = {0 , 0, 0, 
                      0, 28377, -28378, 
                      0, -28378, 28377};                        

  int16_t *p_alpha_re, *p_alpha_im; // pointers to tables alpha above;                     
  uint8_t threetnecyclicshift=0, sixtonecyclichift=0; // NB-IoT: to be defined from higher layer, see 36.211 Section 10.1.4.1.2
  uint8_t actual_cyclicshift; 
  uint8_t Nsc_RU = eNB->ulsch[UE_id]->harq_process->N_sc_RU; // Vincent: number of sc 1,3,6,12 
  unsigned int index_Nsc_RU=4; // Vincent: index_Nsc_RU 0,1,2,3 ---> number of sc 1,3,6,12 
  int16_t *received_data, *estimated_channel, *pilot_sig; // pointers to 
  uint8_t npusch_format = 1; // NB-IoT: format 1 (data), or 2: ack. Should be defined in higher layer 

  ///////////////////////////////////////////////////////////////////////////////////////

  //////// get pseudo-random sequence for NPUSCH format 2 ////////////// 
  n_s = (int)Ns+(subframe<<1); 
  x2 = (uint32_t) Ncell_ID; 
  
  for (p=0;p<n_s+1;p++){ // this should be outsourced to avoid computation in each subframe
    if ((p&3) == 0) {
            s = lte_gold_generic_NB_IoT(&x1,&x2,reset);
            reset = 0;
    }
  } 

  ///////////////////////////////////////////////////////////////////////////////////////

  switch (Nsc_RU){
    case 1: 
      index_Nsc_RU = 0;
      break; 
    case 3: 
      index_Nsc_RU = 1;
      break;
    case 6: 
      index_Nsc_RU = 2; 
      break; 
    case 12: 
      index_Nsc_RU = 3; 
      break; 
    default: 
      printf("Error in number of subcarrier in channel estimation\n"); 
      break;
  }

  if (subcarrier_spacing){
    pilot_pos_format2 = pilot_pos_format2_15k; 
    pilot_pos1 = pilot_pos1_15k; 
    pilot_pos2 = pilot_pos2_15k;
  }else{
    pilot_pos_format2 = pilot_pos_format2_3_75k; 
    pilot_pos1 = pilot_pos1_3_75k; 
    pilot_pos2 = pilot_pos2_3_75k;
  }

  ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
  u=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[n_s][index_Nsc_RU]; // Vincent: may be adapted for Nsc_RU, see 36.211, Section 10.1.4.1.3
  switch (npusch_format){
  case 1: 
      if (l == pilot_pos1) { // NB-IoT: no extended CP 

        symbol_offset = frame_parms->N_RB_UL*12*(l+(7*(Ns&1)));

        for (aa=0; aa<nb_antennas_rx; aa++) {

          received_data = (int16_t *)&rxdataF_ext[aa][symbol_offset];
          estimated_channel   = (int16_t *)&ul_ch_estimates[aa][symbol_offset]; 
          if (index_Nsc_RU){ // NB-IoT: a shift ul_sc_start is added in order to get the same position of the first pilot in rxdataF_ext and ul_ref_sigs_rx_NB_IoT
            pilot_sig  = &ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][24-(ul_sc_start<<1)]; // pilot values are the same every slots
          }else{
            pilot_sig  = &ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][24 + 2*12*(n_s)-(ul_sc_start<<1)]; // pilot values depends on the slots
          }

          for (k=0;k<12;k++){
            // Multiplication by the complex conjugate of the pilot
            estimated_channel[k<<1] = (int16_t)(((int32_t)received_data[k<<1]*(int32_t)pilot_sig[k<<1] + 
                        (int32_t)received_data[(k<<1)+1]*(int32_t)pilot_sig[(k<<1)+1])>>15); //real part of estimated channel 
            estimated_channel[(k<<1)+1] = (int16_t)(((int32_t)received_data[(k<<1)+1]*(int32_t)pilot_sig[k<<1] - 
                        (int32_t)received_data[k<<1]*(int32_t)pilot_sig[(k<<1)+1])>>15); //imaginary part of estimated channel 
          }

          if (Nsc_RU == 1){ // rotate the estimated channel by pi/2 or pi/4, due to mapping b2c
            Qm       = get_Qm_ul_NB_IoT(eNB->ulsch[UE_id]->harq_process->mcs,Nsc_RU);
            rotate_channel_single_carrier_NB_IoT(estimated_channel,l,Qm); 

          }

          if(Nsc_RU != 1 && Nsc_RU != 12) {
            // Compensating for the phase shift introduced at the transmitter
            // In NB-IoT NPUSCH format 1, phase alpha is zero when 1 and 12 subcarriers are allocated
            // else (still format 1), alpha is defined in 36.211, Table 10.1.4.1.2-3
            if (Nsc_RU == 3){
              p_alpha_re = alpha3_re; 
              p_alpha_im = alpha3_im; 
              actual_cyclicshift = threetnecyclicshift;
            }else if (Nsc_RU == 6){
              p_alpha_re = alpha6_re; 
              p_alpha_im = alpha6_im; 
              actual_cyclicshift = sixtonecyclichift; 
            }else{
              msg("lte_ul_channel_estimation_NB-IoT: wrong Nsc_RU value, Nsc_RU=%d\n",Nsc_RU);
              return(-1);
            }

            for(i=symbol_offset+ul_sc_start; i<symbol_offset+ul_sc_start+Nsc_RU; i++) {
              ul_ch_estimates_re = ((int16_t*) ul_ch_estimates[aa])[i<<1];
              ul_ch_estimates_im = ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1];
              //    ((int16_t*) ul_ch_estimates[aa])[i<<1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_re;
              ((int16_t*) ul_ch_estimates[aa])[i<<1] =
                (int16_t) (((int32_t) (p_alpha_re[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_re) +
                            (int32_t) (p_alpha_im[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_im))>>15); 

              //((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_im;
              ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =
                (int16_t) (((int32_t) (p_alpha_re[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_im) -
                            (int32_t) (p_alpha_im[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_re))>>15); 

            }

          }

          if (Ns&1) {//we are in the second slot of the sub-frame, so do the interpolation

            ul_ch1 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos1];
            ul_ch2 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos2]; 

            // Here, the channel is supposed to be quasi-static during one subframe
            // Then, an average over 2 pilot symbols is performed to increase the SNR
            // This part may be improved
            for (k=0;k<12;k++){
              average_channel[k<<1] = (int16_t)(((int32_t)ul_ch1[k<<1] + (int32_t)ul_ch2[k<<1])/2); 
              average_channel[1+(k<<1)] = (int16_t)(((int32_t)ul_ch1[1+(k<<1)] + (int32_t)ul_ch2[1+(k<<1)])/2);
            }

            for (n=0; n<frame_parms->symbols_per_tti; n++) {
              
              if ((n != pilot_pos1) && (n != pilot_pos2))  {

                for (k=0;k<12;k++){
                  ul_ch_estimates[aa][frame_parms->N_RB_UL*12*n + k] = p_average_channel[k]; 
                }

              }

            }

          }

        }

      }
    break; 
  case 2: 
    if (l == pilot_pos_format2[0] || l == pilot_pos_format2[1] || l == pilot_pos_format2[2]){ 

      symbol_offset = frame_parms->N_RB_UL*12*(l+(7*(Ns&1))); 
      index_w = (uint8_t)l-2 + 3*(((uint8_t*)&s)[n_s&3]%3); // base index in w_format2_re and w_format2_im, see 36.211, Section 10.1.4.1.1

      for (aa=0; aa<nb_antennas_rx; aa++) { 

        received_data = (int16_t *)&rxdataF_ext[aa][symbol_offset];
        estimated_channel   = (int16_t *)&ul_ch_estimates[aa][symbol_offset]; 
        pilot_sig  = &ul_ref_sigs_rx_NB_IoT[u][index_Nsc_RU][24 + 2*12*(n_s)-(ul_sc_start<<1)]; // pilot values is the same during 3 symbols l = 1, 2, 3

        for (k=0;k<12;k++){
          // Multiplication by the complex conjugate of the pilot
          estimated_channel[k<<1] = (int16_t)(((int32_t)received_data[k<<1]*(int32_t)pilot_sig[k<<1] + 
                      (int32_t)received_data[(k<<1)+1]*(int32_t)pilot_sig[(k<<1)+1])>>15); //real part of estimated channel 
          estimated_channel[(k<<1)+1] = (int16_t)(((int32_t)received_data[(k<<1)+1]*(int32_t)pilot_sig[k<<1] - 
                      (int32_t)received_data[k<<1]*(int32_t)pilot_sig[(k<<1)+1])>>15); //imaginary part of estimated channel 
        }

        // rotate the estimated channel by pi/2 or pi/4, due to mapping b2c
        rotate_channel_single_carrier_NB_IoT(estimated_channel,l,1); // last input: Qm = 1 in format 2

        // Compensating for the phase shift introduced at the transmitter
        // In NB-IoT NPUSCH format 1, phase alpha is zero when 1 and 12 subcarriers are allocated
        // else (still format 1), alpha is defined in 36.211, Table 10.1.4.1.2-3
        ul_ch_estimates_re = ((int16_t*) ul_ch_estimates[aa])[(symbol_offset+ul_sc_start)<<1]; 
        ul_ch_estimates_im = ((int16_t*) ul_ch_estimates[aa])[((symbol_offset+ul_sc_start)<<1)+1]; 

        //    ((int16_t*) ul_ch_estimates[aa])[i<<1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_re;
        ((int16_t*) ul_ch_estimates[aa])[i<<1] =
        (int16_t) (((int32_t) (w_format2_re[index_w]) * (int32_t) (ul_ch_estimates_re) +
                    (int32_t) (w_format2_im[index_w]) * (int32_t) (ul_ch_estimates_im))>>15); 

        //((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_im;
        ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =
          (int16_t) (((int32_t) (w_format2_re[index_w]) * (int32_t) (ul_ch_estimates_im) -
                      (int32_t) (w_format2_im[index_w]) * (int32_t) (ul_ch_estimates_re))>>15); 

        if (Ns&1 && l==pilot_pos_format2[2]) {//we are in the second slot of the sub-frame, so do the interpolation

          ul_ch1 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[0]];
          ul_ch2 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[1]]; 
          ul_ch3 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[2]];
          ul_ch4 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[3]]; 
          ul_ch5 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[4]];
          ul_ch6 = (int16_t *)&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos_format2[5]]; 

          // Here, the channel is supposed to be quasi-static during one subframe
          // Then, an average over 6 pilot symbols is performed to increase the SNR
          // This part may be improved
          for (k=0;k<12;k++){
            average_channel[k<<1] = (int16_t)(((int32_t)ul_ch1[k<<1] + (int32_t)ul_ch2[k<<1] + 
                                               (int32_t)ul_ch3[k<<1] + (int32_t)ul_ch4[k<<1] + 
                                               (int32_t)ul_ch5[k<<1] + (int32_t)ul_ch6[k<<1])/6); 
            average_channel[1+(k<<1)] = (int16_t)(((int32_t)ul_ch1[1+(k<<1)] + (int32_t)ul_ch2[1+(k<<1)] + 
                                                   (int32_t)ul_ch3[1+(k<<1)] + (int32_t)ul_ch4[1+(k<<1)] + 
                                                   (int32_t)ul_ch5[1+(k<<1)] + (int32_t)ul_ch6[1+(k<<1)])/2);
          }

          for (n=0; n<frame_parms->symbols_per_tti; n++) {
            
            if ((n != pilot_pos_format2[0]) && (n != pilot_pos_format2[1])
                && (n != pilot_pos_format2[2]) && (n != pilot_pos_format2[3])
                && (n != pilot_pos_format2[4]) && (n != pilot_pos_format2[5]))  {

              for (k=0;k<12;k++){
                ul_ch_estimates[aa][frame_parms->N_RB_UL*12*n + k] = p_average_channel[k]; 
              }

            }

          }

        }

      }

    }

    break; 
  default: 
    printf("Error in NPUSCH format, npusch_format=%i \n", npusch_format); 
    break; 

  }
  return(0); 

}






// int32_t lte_ul_channel_estimation_NB_IoT(PHY_VARS_eNB_NB_IoT      *eNB,
// 				                                 eNB_rxtx_proc_NB_IoT_t   *proc,
//                                          uint8_t                  eNB_id,
//                                          uint8_t                  UE_id,
//                                          unsigned char            l,
//                                          unsigned char            Ns,
//                                          uint8_t                  cooperation_flag)
// {

//   NB_IoT_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
//   NB_IoT_eNB_PUSCH *pusch_vars = eNB->pusch_vars[UE_id];
//   int32_t **ul_ch_estimates=pusch_vars->drs_ch_estimates[eNB_id];
//   int32_t **ul_ch_estimates_time=  pusch_vars->drs_ch_estimates_time[eNB_id];
//   //int32_t **ul_ch_estimates_0=  pusch_vars->drs_ch_estimates_0[eNB_id];
//   //int32_t **ul_ch_estimates_1=  pusch_vars->drs_ch_estimates_1[eNB_id];
//   int32_t **rxdataF_ext=  pusch_vars->rxdataF_ext[eNB_id];
//   int subframe = proc->subframe_rx;
//   //uint8_t harq_pid = subframe2harq_pid_NB_IoT(frame_parms,proc->frame_rx,subframe);
//   int16_t delta_phase = 0;
//   int16_t *ru1 = ru_90;
//   int16_t *ru2 = ru_90;
//   int16_t current_phase1,current_phase2;
//   uint16_t N_rb_alloc = eNB->ulsch[UE_id]->harq_process->nb_rb;
//   uint16_t aa,Msc_RS,Msc_RS_idx;
//   uint16_t * Msc_idx_ptr;
//   int32_t rx_power_correction; 
//   // int k,pilot_pos1 = 3 - frame_parms->Ncp, pilot_pos2 = 10 - 2*frame_parms->Ncp; 
//   int k,pilot_pos1 = 3, pilot_pos2 = 10;
//   int16_t alpha, beta;
//   int32_t *ul_ch1=NULL, *ul_ch2=NULL;
//   //int32_t *ul_ch1_0=NULL,*ul_ch2_0=NULL,*ul_ch1_1=NULL,*ul_ch2_1=NULL;
//   int16_t ul_ch_estimates_re,ul_ch_estimates_im;

//   //uint8_t nb_antennas_rx = frame_parms->nb_antenna_ports_eNB;
//   uint8_t nb_antennas_rx = frame_parms->nb_antennas_rx;
//   uint8_t cyclic_shift;

//   uint32_t alpha_ind;
//   uint32_t u;//=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[Ns+(subframe<<1)];
//   //uint32_t v=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.seqhop[Ns+(subframe<<1)];
//   int32_t tmp_estimates[N_rb_alloc*12] __attribute__((aligned(16)));

//   int symbol_offset,i;
//   //int j;

//   //debug_msg("lte_ul_channel_estimation: cyclic shift %d\n",cyclicShift);

//   // int16_t alpha_re[12] = {32767, 28377, 16383,     0,-16384,  -28378,-32768,-28378,-16384,    -1, 16383, 28377};
//   // int16_t alpha_im[12] = {0,     16383, 28377, 32767, 28377,   16383,     0,-16384,-28378,-32768,-28378,-16384}; 

//   ////// NB-IoT specific ///////////////////////////////////////////////////////////////////////////////////////

//   uint32_t I_sc = eNB->ulsch[UE_id]->harq_process->I_sc;  // NB_IoT: subcarrier indication field: must be defined in higher layer
//   uint16_t ul_sc_start; // subcarrier start index into UL RB 

//   // 36.211, Section 10.1.4.1.2, Table 10.1.4.1.2-3 
//   int16_t alpha3_re[9] = {32767 , 32767, 32767, 
//                       32767, -16384, -16384,
//                       32767, -16384, -16384}; 
//   int16_t alpha3_im[9] = {0 , 0, 0, 
//                       0, 28377, -28378, 
//                       0, -28378, 28377};                     
//   int16_t alpha6_re[24] = {32767 , 32767, 32767, 32767, 32767, 32767, 
//                       32767, 16383, -16384, -32768, -16384, 16383, 
//                       32767, -16384, -16384, 32767, -16384, -16384, 
//                       32767, -16384, -16384, 32767, -16384, -16384}; 
//   int16_t alpha6_im[24] = {0 , 0, 0, 0, 0, 0, 
//                       0, 28377, 28377, 0, -28378, -28378, 
//                       0, 28377, -28378, 0, 28377, -28378,
//                       0, -28378, 28377, 0, -28378, 28377}; 
//   int16_t *p_alpha_re, *p_alpha_im; // pointers to tables above;                     
//   uint8_t threetnecyclicshift=0, sixtonecyclichift=0; // NB-IoT: to be defined from higher layer, see 36.211 Section 10.1.4.1.2
//   uint8_t actual_cyclicshift; 
//   uint16_t Nsc_RU; // Vincent: number of sc 1,3,6,12 
//   unsigned int index_Nsc_RU; // Vincent: index_Nsc_RU 0,1,2,3 ---> number of sc 1,3,6,12 

//   ///////////////////////////////////////////////////////////////////////////////////////

//  /* 
//       int32_t *in_fft_ptr_0 = (int32_t*)0,*in_fft_ptr_1 = (int32_t*)0,
//            *temp_out_fft_0_ptr = (int32_t*)0,*out_fft_ptr_0 = (int32_t*)0,
//             *temp_out_fft_1_ptr = (int32_t*)0,*out_fft_ptr_1 = (int32_t*)0,
//              *temp_in_ifft_ptr = (int32_t*)0; 
// */

// #if defined(__x86_64__) || defined(__i386__)
//   __m128i *rxdataF128,*ul_ref128,*ul_ch128;
//   __m128i mmtmpU0,mmtmpU1,mmtmpU2,mmtmpU3;
// #elif defined(__arm__)
//   int16x8_t *rxdataF128,*ul_ref128,*ul_ch128;
//   int32x4_t mmtmp0,mmtmp1,mmtmp_re,mmtmp_im;
// #endif
//   Msc_RS = N_rb_alloc*12;

//   cyclic_shift = (frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.cyclicShift +
//                   eNB->ulsch[UE_id]->harq_process->n_DMRS2 +
//                   frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.nPRS[(subframe<<1)+Ns]) % 12;

// #if defined(USER_MODE)
//   Msc_idx_ptr = (uint16_t*) bsearch(&Msc_RS, dftsizes, 33, sizeof(uint16_t), compareints);

//   if (Msc_idx_ptr)
//     Msc_RS_idx = Msc_idx_ptr - dftsizes;
//   else {
//     msg("lte_ul_channel_estimation: index for Msc_RS=%d not found\n",Msc_RS);
//     return(-1);
//   }

// #else
//   uint8_t b;

//   for (b=0; b<33; b++)
//     if (Msc_RS==dftsizes[b])
//       Msc_RS_idx = b;

// #endif

//   //  LOG_I(PHY,"subframe %d, Ns %d, l %d, Msc_RS = %d, Msc_RS_idx = %d, u %d, v %d, cyclic_shift %d\n",subframe,Ns,l,Msc_RS, Msc_RS_idx,u,v,cyclic_shift);
// #ifdef DEBUG_CH

// #ifdef USER_MODE

//   if (Ns==0)
//     write_output("drs_seq0.m","drsseq0",ul_ref_sigs_rx[u][Msc_RS_idx],2*Msc_RS,2,1);
//   else
//     write_output("drs_seq1.m","drsseq1",ul_ref_sigs_rx[u][Msc_RS_idx],2*Msc_RS,2,1);

// #endif
// #endif

//   rx_power_correction = 1;
//   ul_sc_start = get_UL_sc_start_NB_IoT(I_sc); // NB-IoT: get the used subcarrier in RB
//   u=frame_parms->npusch_config_common.ul_ReferenceSignalsNPUSCH.grouphop[Ns+(subframe<<1)][index_Nsc_RU]; // Vincent: may be adapted for Nsc_RU, see 36.211, Section 10.1.4.1.3

//   // if (l == (3 - frame_parms->Ncp)) { 
//   if (l == 3) { // NB-IoT: no extended CP 

//     // symbol_offset = frame_parms->N_RB_UL*12*(l+((7-frame_parms->Ncp)*(Ns&1))); 
//     symbol_offset = frame_parms->N_RB_UL*12*(l+(7*(Ns&1)));

//     for (aa=0; aa<nb_antennas_rx; aa++) {
//       //           msg("Componentwise prod aa %d, symbol_offset %d,ul_ch_estimates %p,ul_ch_estimates[aa] %p,ul_ref_sigs_rx[0][0][Msc_RS_idx] %p\n",aa,symbol_offset,ul_ch_estimates,ul_ch_estimates[aa],ul_ref_sigs_rx[0][0][Msc_RS_idx]);

// #if defined(__x86_64__) || defined(__i386__)
//       rxdataF128 = (__m128i *)&rxdataF_ext[aa][symbol_offset];
//       ul_ch128   = (__m128i *)&ul_ch_estimates[aa][symbol_offset];
//       if (index_Nsc_RU){ // NB-IoT: a shift ul_sc_start is added in order to get the same position of the first pilot in rxdataF_ext and ul_ref_sigs_rx
//         ul_ref128  = (__m128i *)ul_ref_sigs_rx[u][index_Nsc_RU][24-(ul_sc_start<<1)]; // pilot values are the same every slots
//         }else{
//         ul_ref128  = (__m128i *)ul_ref_sigs_rx[u][index_Nsc_RU][24 + 12*(subframe<<1)-(ul_sc_start<<1)]; // pilot values depends on the slots
//       }
// #elif defined(__arm__)
//       rxdataF128 = (int16x8_t *)&rxdataF_ext[aa][symbol_offset];
//       ul_ch128   = (int16x8_t *)&ul_ch_estimates[aa][symbol_offset]; 
//       if (index_Nsc_RU){
//         ul_ref128  = (int16x8_t *)ul_ref_sigs_rx[u][index_Nsc_RU][24-(ul_sc_start<<1)]; 
//       }else{
//         ul_ref128  = (int16x8_t *)ul_ref_sigs_rx[u][index_Nsc_RU][24 + 12*(subframe<<1)-(ul_sc_start<<1)]; 
//       }
// #endif

//       // for (i=0; i<Msc_RS/12; i++) {
// #if defined(__x86_64__) || defined(__i386__)
//         // multiply by conjugated channel
//         mmtmpU0 = _mm_madd_epi16(ul_ref128[0],rxdataF128[0]);
//         // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
//         mmtmpU1 = _mm_shufflelo_epi16(ul_ref128[0],_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)&conjugate[0]);
//         mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[0]);
//         // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
//         mmtmpU0 = _mm_srai_epi32(mmtmpU0,15);
//         mmtmpU1 = _mm_srai_epi32(mmtmpU1,15);
//         mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
//         mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);

//         ul_ch128[0] = _mm_packs_epi32(mmtmpU2,mmtmpU3);
//         //  printf("rb %d ch: %d %d\n",i,((int16_t*)ul_ch128)[0],((int16_t*)ul_ch128)[1]);
//         // multiply by conjugated channel
//         mmtmpU0 = _mm_madd_epi16(ul_ref128[1],rxdataF128[1]);
//         // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
//         mmtmpU1 = _mm_shufflelo_epi16(ul_ref128[1],_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)conjugate);
//         mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[1]);
//         // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
//         mmtmpU0 = _mm_srai_epi32(mmtmpU0,15);
//         mmtmpU1 = _mm_srai_epi32(mmtmpU1,15);
//         mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
//         mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);

//         ul_ch128[1] = _mm_packs_epi32(mmtmpU2,mmtmpU3);

//         mmtmpU0 = _mm_madd_epi16(ul_ref128[2],rxdataF128[2]);
//         // mmtmpU0 contains real part of 4 consecutive outputs (32-bit)
//         mmtmpU1 = _mm_shufflelo_epi16(ul_ref128[2],_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_shufflehi_epi16(mmtmpU1,_MM_SHUFFLE(2,3,0,1));
//         mmtmpU1 = _mm_sign_epi16(mmtmpU1,*(__m128i*)conjugate);
//         mmtmpU1 = _mm_madd_epi16(mmtmpU1,rxdataF128[2]);
//         // mmtmpU1 contains imag part of 4 consecutive outputs (32-bit)
//         mmtmpU0 = _mm_srai_epi32(mmtmpU0,15);
//         mmtmpU1 = _mm_srai_epi32(mmtmpU1,15);
//         mmtmpU2 = _mm_unpacklo_epi32(mmtmpU0,mmtmpU1);
//         mmtmpU3 = _mm_unpackhi_epi32(mmtmpU0,mmtmpU1);

//         ul_ch128[2] = _mm_packs_epi32(mmtmpU2,mmtmpU3);
// #elif defined(__arm__)
//       mmtmp0 = vmull_s16(((int16x4_t*)ul_ref128)[0],((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(((int16x4_t*)ul_ref128)[1],((int16x4_t*)rxdataF128)[1]);
//       mmtmp_re = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));
//       mmtmp0 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[0],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[1],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[1]);
//       mmtmp_im = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));

//       ul_ch128[0] = vcombine_s16(vmovn_s32(mmtmp_re),vmovn_s32(mmtmp_im));
//       ul_ch128++;
//       ul_ref128++;
//       rxdataF128++;
//       mmtmp0 = vmull_s16(((int16x4_t*)ul_ref128)[0],((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(((int16x4_t*)ul_ref128)[1],((int16x4_t*)rxdataF128)[1]);
//       mmtmp_re = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));
//       mmtmp0 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[0],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[1],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[1]);
//       mmtmp_im = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));

//       ul_ch128[0] = vcombine_s16(vmovn_s32(mmtmp_re),vmovn_s32(mmtmp_im));
//       ul_ch128++;
//       ul_ref128++;
//       rxdataF128++;

//       mmtmp0 = vmull_s16(((int16x4_t*)ul_ref128)[0],((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(((int16x4_t*)ul_ref128)[1],((int16x4_t*)rxdataF128)[1]);
//       mmtmp_re = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));
//       mmtmp0 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[0],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[0]);
//       mmtmp1 = vmull_s16(vrev32_s16(vmul_s16(((int16x4_t*)ul_ref128)[1],*(int16x4_t*)conjugate)), ((int16x4_t*)rxdataF128)[1]);
//       mmtmp_im = vcombine_s32(vpadd_s32(vget_low_s32(mmtmp0),vget_high_s32(mmtmp0)),
//                               vpadd_s32(vget_low_s32(mmtmp1),vget_high_s32(mmtmp1)));

//       ul_ch128[0] = vcombine_s16(vmovn_s32(mmtmp_re),vmovn_s32(mmtmp_im));
//       ul_ch128++;
//       ul_ref128++;
//       rxdataF128++;


// #endif
//       //   ul_ch128+=3;
//       //   ul_ref128+=3;
//       //   rxdataF128+=3;
//       // }

// //       alpha_ind = 0;

// //       if((cyclic_shift != 0) && Msc_RS != 12) {
// //       // if(Nsc_RU != 1 && Nsc_RU != 12) {
// //         // Compensating for the phase shift introduced at the transmitter
// //         // In NB-IoT, phase alpha is zero when 12 subcarriers are allocated
// // #ifdef DEBUG_CH
// //         write_output("drs_est_pre.m","drsest_pre",ul_ch_estimates[0],300*12,1,1);
// // #endif

// //         for(i=symbol_offset; i<symbol_offset+Msc_RS; i++) {
// //           ul_ch_estimates_re = ((int16_t*) ul_ch_estimates[aa])[i<<1];
// //           ul_ch_estimates_im = ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1];
// //           //    ((int16_t*) ul_ch_estimates[aa])[i<<1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_re;
// //           ((int16_t*) ul_ch_estimates[aa])[i<<1] =
// //             (int16_t) (((int32_t) (alpha_re[alpha_ind]) * (int32_t) (ul_ch_estimates_re) +
// //                         (int32_t) (alpha_im[alpha_ind]) * (int32_t) (ul_ch_estimates_im))>>15);

// //           //((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_im;
// //           ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =
// //             (int16_t) (((int32_t) (alpha_re[alpha_ind]) * (int32_t) (ul_ch_estimates_im) -
// //                         (int32_t) (alpha_im[alpha_ind]) * (int32_t) (ul_ch_estimates_re))>>15);

// //           alpha_ind+=cyclic_shift;

// //           if (alpha_ind>11)
// //             alpha_ind-=12;
// //         }

// // #ifdef DEBUG_CH
// //         write_output("drs_est_post.m","drsest_post",ul_ch_estimates[0],300*12,1,1);
// // #endif
// //       } 

//       alpha_ind = 0;

//       if(Nsc_RU != 1 && Nsc_RU != 12) {
//         // Compensating for the phase shift introduced at the transmitter
//         // In NB-IoT, phase alpha is zero when 1 and 12 subcarriers are allocated
// #ifdef DEBUG_CH
//         write_output("drs_est_pre.m","drsest_pre",ul_ch_estimates[0],300*12,1,1);
// #endif
//         if (Nsc_RU == 3){
//           p_alpha_re = alpha3_re; 
//           p_alpha_im = alpha3_im; 
//           actual_cyclicshift = threetnecyclicshift;
//         }else if (Nsc_RU == 6){
//           p_alpha_re = alpha6_re; 
//           p_alpha_im = alpha6_im; 
//           actual_cyclicshift= sixtonecyclichift; 
//         }else{
//           msg("lte_ul_channel_estimation_NB-IoT: wrong Nsc_RU value, Nsc_RU=%d\n",Nsc_RU);
//           return(-1);
//         }

//         for(i=symbol_offset+ul_sc_start; i<symbol_offset+ul_sc_start+Nsc_RU; i++) {
//           ul_ch_estimates_re = ((int16_t*) ul_ch_estimates[aa])[i<<1];
//           ul_ch_estimates_im = ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1];
//           //    ((int16_t*) ul_ch_estimates[aa])[i<<1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_re;
//           ((int16_t*) ul_ch_estimates[aa])[i<<1] =
//             (int16_t) (((int32_t) (p_alpha_re[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_re) +
//                         (int32_t) (p_alpha_im[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_im))>>15);

//           //((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_im;
//           ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =
//             (int16_t) (((int32_t) (p_alpha_re[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_im) -
//                         (int32_t) (p_alpha_im[actual_cyclicshift*Nsc_RU+i]) * (int32_t) (ul_ch_estimates_re))>>15); 

//         }

// #ifdef DEBUG_CH
//         write_output("drs_est_post.m","drsest_post",ul_ch_estimates[0],300*12,1,1);
// #endif
//       }

//       //copy MIMO channel estimates to temporary buffer for EMOS
//       //memcpy(&ul_ch_estimates_0[aa][symbol_offset],&ul_ch_estimates[aa][symbol_offset],frame_parms->ofdm_symbol_size*sizeof(int32_t)*2);

//       memset(temp_in_ifft_0,0,frame_parms->ofdm_symbol_size*sizeof(int32_t));

//       // Convert to time domain for visualization
//       for(i=0; i<Msc_RS; i++)
//         ((int32_t*)temp_in_ifft_0)[i] = ul_ch_estimates[aa][symbol_offset+i];
//       switch(frame_parms->N_RB_DL) {
//       case 6:
	
// 	idft128((int16_t*) temp_in_ifft_0,
// 	       (int16_t*) ul_ch_estimates_time[aa],
// 	       1);
// 	break;
//       case 25:
	
// 	idft512((int16_t*) temp_in_ifft_0,
// 	       (int16_t*) ul_ch_estimates_time[aa],
// 	       1);
// 	break;
//       case 50:
	
// 	idft1024((int16_t*) temp_in_ifft_0,
// 	       (int16_t*) ul_ch_estimates_time[aa],
// 	       1);
// 	break;
//       case 100:
	
// 	idft2048((int16_t*) temp_in_ifft_0,
// 	       (int16_t*) ul_ch_estimates_time[aa],
// 	       1);
// 	break;
//       }

// #if T_TRACER
//       if (aa == 0)
//         T(T_ENB_PHY_UL_CHANNEL_ESTIMATE, T_INT(eNB_id), T_INT(UE_id),
//           T_INT(proc->frame_rx), T_INT(subframe),
//           T_INT(0), T_BUFFER(ul_ch_estimates_time[0], 512  * 4));
// #endif

// #ifdef DEBUG_CH

//       if (aa==0) {
//         if (Ns == 0) {
//           write_output("rxdataF_ext.m","rxF_ext",&rxdataF_ext[aa][symbol_offset],512*2,2,1);
//           write_output("tmpin_ifft.m","drs_in",temp_in_ifft_0,512,1,1);
//           write_output("drs_est0.m","drs0",ul_ch_estimates_time[aa],512,1,1);
//         } else
//           write_output("drs_est1.m","drs1",ul_ch_estimates_time[aa],512,1,1);
//       }

// #endif


// //       if(cooperation_flag == 2) {
// //         memset(temp_in_ifft_0,0,frame_parms->ofdm_symbol_size*sizeof(int32_t*)*2);
// //         memset(temp_in_ifft_1,0,frame_parms->ofdm_symbol_size*sizeof(int32_t*)*2);
// //         memset(temp_in_fft_0,0,frame_parms->ofdm_symbol_size*sizeof(int32_t*)*2);
// //         memset(temp_in_fft_1,0,frame_parms->ofdm_symbol_size*sizeof(int32_t*)*2);

// //         temp_in_ifft_ptr = &temp_in_ifft_0[0];

// //         i = symbol_offset;

// //         for(j=0; j<(frame_parms->N_RB_UL*12); j++) {
// //           temp_in_ifft_ptr[j] = ul_ch_estimates[aa][i];
// //           i++;
// //         }

// //         alpha_ind = 0;

// //         // Compensating for the phase shift introduced at the transmitter
// //         for(i=symbol_offset; i<symbol_offset+Msc_RS; i++) {
// //           ul_ch_estimates_re = ((int16_t*) ul_ch_estimates[aa])[i<<1];
// //           ul_ch_estimates_im = ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1];
// //           //    ((int16_t*) ul_ch_estimates[aa])[i<<1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_re;
// //           ((int16_t*) ul_ch_estimates[aa])[i<<1] =
// //             (int16_t) (((int32_t) (alpha_re[alpha_ind]) * (int32_t) (ul_ch_estimates_re) +
// //                         (int32_t) (alpha_im[alpha_ind]) * (int32_t) (ul_ch_estimates_im))>>15);

// //           //((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =  (i%2 == 1? 1:-1) * ul_ch_estimates_im;
// //           ((int16_t*) ul_ch_estimates[aa])[(i<<1)+1] =
// //             (int16_t) (((int32_t) (alpha_re[alpha_ind]) * (int32_t) (ul_ch_estimates_im) -
// //                         (int32_t) (alpha_im[alpha_ind]) * (int32_t) (ul_ch_estimates_re))>>15);

// //           alpha_ind+=10;

// //           if (alpha_ind>11)
// //             alpha_ind-=12;
// //         }

// //         //Extracting Channel Estimates for Distributed Alamouti Receiver Combining

// //         temp_in_ifft_ptr = &temp_in_ifft_1[0];

// //         i = symbol_offset;

// //         for(j=0; j<(frame_parms->N_RB_UL*12); j++) {
// //           temp_in_ifft_ptr[j] = ul_ch_estimates[aa][i];
// //           i++;
// //         }

// // 	switch (frame_parms->N_RB_DL) {
// // 	case 6:
// // 	  idft128((int16_t*) &temp_in_ifft_0[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_0,
// // 		  1);
// // 	  idft128((int16_t*) &temp_in_ifft_1[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_1,
// // 		  1);
// // 	  break;
// // 	case 25:
// // 	  idft512((int16_t*) &temp_in_ifft_0[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_0,
// // 		  1);
// // 	  idft512((int16_t*) &temp_in_ifft_1[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_1,
// // 		  1);
// // 	  break;
// // 	case 50:
// // 	  idft1024((int16_t*) &temp_in_ifft_0[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_0,
// // 		  1);
// // 	  idft1024((int16_t*) &temp_in_ifft_1[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_1,
// // 		  1);
// // 	  break;
// // 	case 100:
// // 	  idft2048((int16_t*) &temp_in_ifft_0[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_0,
// // 		  1);
// // 	  idft2048((int16_t*) &temp_in_ifft_1[0],                          // Performing IFFT on Combined Channel Estimates
// // 		  temp_out_ifft_1,
// // 		  1);
// // 	  break;
// // 	}

// //         // because the ifft is not power preserving, we should apply the factor sqrt(power_correction) here, but we rather apply power_correction here and nothing after the next fft
// //         in_fft_ptr_0 = &temp_in_fft_0[0];
// //         in_fft_ptr_1 = &temp_in_fft_1[0];

// //         for(j=0; j<(frame_parms->ofdm_symbol_size)/12; j++) {
// //           if (j>19) {
// //             ((int16_t*)in_fft_ptr_0)[-40+(2*j)] = ((int16_t*)temp_out_ifft_0)[-80+(2*j)]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_0)[-40+(2*j)+1] = ((int16_t*)temp_out_ifft_0)[-80+(2*j+1)]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_1)[-40+(2*j)] = ((int16_t*)temp_out_ifft_1)[-80+(2*j)]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_1)[-40+(2*j)+1] = ((int16_t*)temp_out_ifft_1)[-80+(2*j)+1]*rx_power_correction;
// //           } else {
// //             ((int16_t*)in_fft_ptr_0)[2*(frame_parms->ofdm_symbol_size-20+j)] = ((int16_t*)temp_out_ifft_0)[2*(frame_parms->ofdm_symbol_size-20+j)]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_0)[2*(frame_parms->ofdm_symbol_size-20+j)+1] = ((int16_t*)temp_out_ifft_0)[2*(frame_parms->ofdm_symbol_size-20+j)+1]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_1)[2*(frame_parms->ofdm_symbol_size-20+j)] = ((int16_t*)temp_out_ifft_1)[2*(frame_parms->ofdm_symbol_size-20+j)]*rx_power_correction;
// //             ((int16_t*)in_fft_ptr_1)[2*(frame_parms->ofdm_symbol_size-20+j)+1] = ((int16_t*)temp_out_ifft_1)[2*(frame_parms->ofdm_symbol_size-20+j)+1]*rx_power_correction;
// //           }
// //         }

// // 	switch (frame_parms->N_RB_DL) {
// //         case 6:
// // 	  dft128((int16_t*) &temp_in_fft_0[0],     
// // 		 // Performing FFT to obtain the Channel Estimates for UE0 to eNB1
// // 		 temp_out_fft_0,
// // 		 1);
// // 	  break;
// //         case 25:
// // 	  dft512((int16_t*) &temp_in_fft_0[0],     
// // 		 // Performing FFT to obtain the Channel Estimates for UE0 to eNB1
// // 		 temp_out_fft_0,
// // 		 1);
// // 	  break;
// //         case 50:
// // 	  dft1024((int16_t*) &temp_in_fft_0[0],     
// // 		 // Performing FFT to obtain the Channel Estimates for UE0 to eNB1
// // 		 temp_out_fft_0,
// // 		 1);
// // 	  break;
// //         case 100:
// // 	  dft2048((int16_t*) &temp_in_fft_0[0],     
// // 		 // Performing FFT to obtain the Channel Estimates for UE0 to eNB1
// // 		 temp_out_fft_0,
// // 		 1);
// // 	  break;
// // 	}

// //         out_fft_ptr_0 = &ul_ch_estimates_0[aa][symbol_offset]; // CHANNEL ESTIMATES FOR UE0 TO eNB1
// //         temp_out_fft_0_ptr = (int32_t*) temp_out_fft_0;

// //         i=0;

// //         for(j=0; j<frame_parms->N_RB_UL*12; j++) {
// //           out_fft_ptr_0[i] = temp_out_fft_0_ptr[j];
// //           i++;
// //         }
// // 	switch (frame_parms->N_RB_DL) {
// // 	case 6:
// // 	  dft128((int16_t*) &temp_in_fft_1[0],                          // Performing FFT to obtain the Channel Estimates for UE1 to eNB1
// // 		 temp_out_fft_1,
// // 		 1);
// // 	  break;
// // 	case 25:
// // 	  dft512((int16_t*) &temp_in_fft_1[0],                          // Performing FFT to obtain the Channel Estimates for UE1 to eNB1
// // 		 temp_out_fft_1,
// // 		 1);
// // 	  break;
// // 	case 50:
// // 	  dft1024((int16_t*) &temp_in_fft_1[0],                          // Performing FFT to obtain the Channel Estimates for UE1 to eNB1
// // 		 temp_out_fft_1,
// // 		 1);
// // 	  break;
// // 	case 100:
// // 	  dft2048((int16_t*) &temp_in_fft_1[0],                          // Performing FFT to obtain the Channel Estimates for UE1 to eNB1
// // 		 temp_out_fft_1,
// // 		 1);
// // 	  break;
// // 	}

// //         out_fft_ptr_1 = &ul_ch_estimates_1[aa][symbol_offset];   // CHANNEL ESTIMATES FOR UE1 TO eNB1
// //         temp_out_fft_1_ptr = (int32_t*) temp_out_fft_1;

// //         i=0;

// //         for(j=0; j<frame_parms->N_RB_UL*12; j++) {
// //           out_fft_ptr_1[i] = temp_out_fft_1_ptr[j];
// //           i++;
// //         }

// // #ifdef DEBUG_CH
// // #ifdef USER_MODE

// //         if((aa == 0)&& (cooperation_flag == 2)) {
// //           write_output("test1.m","t1",temp_in_ifft_0,512,1,1);
// //           write_output("test2.m","t2",temp_out_ifft_0,512*2,2,1);
// //           write_output("test3.m","t3",temp_in_fft_0,512,1,1);
// //           write_output("test4.m","t4",temp_out_fft_0,512,1,1);
// //           write_output("test5.m","t5",temp_in_fft_1,512,1,1);
// //           write_output("test6.m","t6",temp_out_fft_1,512,1,1);
// //         }

// // #endif
// // #endif

// //       }//cooperation_flag == 2

//       if (Ns&1) {//we are in the second slot of the sub-frame, so do the interpolation

//         ul_ch1 = &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos1];
//         ul_ch2 = &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*pilot_pos2];


//         // if(cooperation_flag == 2) { // For Distributed Alamouti
//         //   ul_ch1_0 = &ul_ch_estimates_0[aa][frame_parms->N_RB_UL*12*pilot_pos1];
//         //   ul_ch2_0 = &ul_ch_estimates_0[aa][frame_parms->N_RB_UL*12*pilot_pos2];

//         //   ul_ch1_1 = &ul_ch_estimates_1[aa][frame_parms->N_RB_UL*12*pilot_pos1];
//         //   ul_ch2_1 = &ul_ch_estimates_1[aa][frame_parms->N_RB_UL*12*pilot_pos2];
//         // }

//         // Estimation of phase difference between the 2 channel estimates
//         // delta_phase = lte_ul_freq_offset_estimation_NB_IoT(frame_parms,
//         //               ul_ch_estimates[aa],
//         //               N_rb_alloc);
//         delta_phase = lte_ul_freq_offset_estimation_NB_IoT(frame_parms,
//                       ul_ch_estimates[aa],
//                       1); // NB-IoT: only 1 RB
//         // negative phase index indicates negative Im of ru
//         //    msg("delta_phase: %d\n",delta_phase);

// #ifdef DEBUG_CH
//         msg("lte_ul_channel_estimation: ul_ch1 = %p, ul_ch2 = %p, pilot_pos1=%d, pilot_pos2=%d\n",ul_ch1, ul_ch2, pilot_pos1,pilot_pos2);
// #endif

//         for (k=0; k<frame_parms->symbols_per_tti; k++) {

//           // we scale alpha and beta by SCALE (instead of 0x7FFF) to avoid overflows
//           alpha = (int16_t) (((int32_t) SCALE * (int32_t) (pilot_pos2-k))/(pilot_pos2-pilot_pos1));
//           beta  = (int16_t) (((int32_t) SCALE * (int32_t) (k-pilot_pos1))/(pilot_pos2-pilot_pos1));


// #ifdef DEBUG_CH
//           msg("lte_ul_channel_estimation: k=%d, alpha = %d, beta = %d\n",k,alpha,beta);
// #endif
//           //symbol_offset_subframe = frame_parms->N_RB_UL*12*k;

//           // interpolate between estimates
//           if ((k != pilot_pos1) && (k != pilot_pos2))  {
//             //          multadd_complex_vector_real_scalar((int16_t*) ul_ch1,alpha,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);
//             //          multadd_complex_vector_real_scalar((int16_t*) ul_ch2,beta ,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//             //          multadd_complex_vector_real_scalar((int16_t*) ul_ch1,SCALE,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);
//             //          multadd_complex_vector_real_scalar((int16_t*) ul_ch2,SCALE,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);
//             //          msg("phase = %d\n",ru[2*cmax(((delta_phase/7)*(k-3)),0)]);

//             // the phase is linearly interpolated
//             current_phase1 = (delta_phase/7)*(k-pilot_pos1);
//             current_phase2 = (delta_phase/7)*(k-pilot_pos2);
//             //          msg("sym: %d, current_phase1: %d, current_phase2: %d\n",k,current_phase1,current_phase2);
//             // set the right quadrant
//             (current_phase1 > 0) ? (ru1 = ru_90) : (ru1 = ru_90c);
//             (current_phase2 > 0) ? (ru2 = ru_90) : (ru2 = ru_90c);
//             // take absolute value and clip
//             current_phase1 = cmin(abs(current_phase1),127);
//             current_phase2 = cmin(abs(current_phase2),127);

//             //          msg("sym: %d, current_phase1: %d, ru: %d + j%d, current_phase2: %d, ru: %d + j%d\n",k,current_phase1,ru1[2*current_phase1],ru1[2*current_phase1+1],current_phase2,ru2[2*current_phase2],ru2[2*current_phase2+1]);

//             // rotate channel estimates by estimated phase
//             rotate_cpx_vector((int16_t*) ul_ch1,
//                               &ru1[2*current_phase1],
//                               (int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],
//                               Msc_RS,
//                               15);

//             rotate_cpx_vector((int16_t*) ul_ch2,
//                               &ru2[2*current_phase2],
//                               (int16_t*) &tmp_estimates[0],
//                               Msc_RS,
//                               15);

//             // Combine the two rotated estimates
//             multadd_complex_vector_real_scalar((int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],SCALE,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);
//             multadd_complex_vector_real_scalar((int16_t*) &tmp_estimates[0],SCALE,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//             /*
//             if ((k<pilot_pos1) || ((k>pilot_pos2))) {

//                 multadd_complex_vector_real_scalar((int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],SCALE>>1,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);

//                 multadd_complex_vector_real_scalar((int16_t*) &tmp_estimates[0],SCALE>>1,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//             } else {

//                 multadd_complex_vector_real_scalar((int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],SCALE>>1,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);

//                 multadd_complex_vector_real_scalar((int16_t*) &tmp_estimates[0],SCALE>>1,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//                 //              multadd_complex_vector_real_scalar((int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],alpha,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);

//                 //              multadd_complex_vector_real_scalar((int16_t*) &tmp_estimates[0],beta ,(int16_t*) &ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//             }
//             */

//             //      memcpy(&ul_ch_estimates[aa][frame_parms->N_RB_UL*12*k],ul_ch1,Msc_RS*sizeof(int32_t));
//             // if(cooperation_flag == 2) { // For Distributed Alamouti
//             //   multadd_complex_vector_real_scalar((int16_t*) ul_ch1_0,beta ,(int16_t*) &ul_ch_estimates_0[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);
//             //   multadd_complex_vector_real_scalar((int16_t*) ul_ch2_0,alpha,(int16_t*) &ul_ch_estimates_0[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);

//             //   multadd_complex_vector_real_scalar((int16_t*) ul_ch1_1,beta ,(int16_t*) &ul_ch_estimates_1[aa][frame_parms->N_RB_UL*12*k],1,Msc_RS);
//             //   multadd_complex_vector_real_scalar((int16_t*) ul_ch2_1,alpha,(int16_t*) &ul_ch_estimates_1[aa][frame_parms->N_RB_UL*12*k],0,Msc_RS);
//             // }

//           }
//         } //for(k=...

//         // because of the scaling of alpha and beta we also need to scale the final channel estimate at the pilot positions

//         //    multadd_complex_vector_real_scalar((int16_t*) ul_ch1,SCALE,(int16_t*) ul_ch1,1,Msc_RS);
//         //    multadd_complex_vector_real_scalar((int16_t*) ul_ch2,SCALE,(int16_t*) ul_ch2,1,Msc_RS);

//         // if(cooperation_flag == 2) { // For Distributed Alamouti
//         //   multadd_complex_vector_real_scalar((int16_t*) ul_ch1_0,SCALE,(int16_t*) ul_ch1_0,1,Msc_RS);
//         //   multadd_complex_vector_real_scalar((int16_t*) ul_ch2_0,SCALE,(int16_t*) ul_ch2_0,1,Msc_RS);

//         //   multadd_complex_vector_real_scalar((int16_t*) ul_ch1_1,SCALE,(int16_t*) ul_ch1_1,1,Msc_RS);
//         //   multadd_complex_vector_real_scalar((int16_t*) ul_ch2_1,SCALE,(int16_t*) ul_ch2_1,1,Msc_RS);
//         // }


//       } //if (Ns&1)

//     } //for(aa=...

//   } //if(l==...



//   return(0);
// }


// int16_t lte_ul_freq_offset_estimation_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
//                                              int32_t *ul_ch_estimates,
//                                              uint16_t nb_rb)
// {

// #if defined(__x86_64__) || defined(__i386__)
//   int k, rb;
//   int a_idx = 64;
//   uint8_t conj_flag = 0;
//   uint8_t output_shift;
//   // int pilot_pos1 = 3 - frame_parms->Ncp;
//   // int pilot_pos2 = 10 - 2*frame_parms->Ncp;
//   int pilot_pos1 = 3; 
//   int pilot_pos2 = 10; 
//   __m128i *ul_ch1 = (__m128i*)&ul_ch_estimates[pilot_pos1*frame_parms->N_RB_UL*12];
//   __m128i *ul_ch2 = (__m128i*)&ul_ch_estimates[pilot_pos2*frame_parms->N_RB_UL*12];
//   int32_t avg[2];
//   int16_t Ravg[2];
//   Ravg[0]=0;
//   Ravg[1]=0;
//   int16_t iv, rv, phase_idx;
//   __m128i avg128U1, avg128U2, R[3], mmtmpD0,mmtmpD1,mmtmpD2,mmtmpD3;

//   // round(tan((pi/4)*[1:1:N]/N)*pow2(15))
//   int16_t alpha[128] = {201, 402, 603, 804, 1006, 1207, 1408, 1610, 1811, 2013, 2215, 2417, 2619, 2822, 3024, 3227, 3431, 3634, 3838, 4042, 4246, 4450, 4655, 4861, 5066, 5272, 5479, 5686, 5893, 6101, 6309, 6518, 6727, 6937, 7147, 7358, 7570, 7782, 7995, 8208, 8422, 8637, 8852, 9068, 9285, 9503, 9721, 9940, 10160, 10381, 10603, 10825, 11049, 11273, 11498, 11725, 11952, 12180, 12410, 12640, 12872, 13104, 13338, 13573, 13809, 14046, 14285, 14525, 14766, 15009, 15253, 15498, 15745, 15993, 16243, 16494, 16747, 17001, 17257, 17515, 17774, 18035, 18298, 18563, 18829, 19098, 19368, 19640, 19915, 20191, 20470, 20750, 21033, 21318, 21605, 21895, 22187, 22481, 22778, 23078, 23380, 23685, 23992, 24302, 24615, 24931, 25250, 25572, 25897, 26226, 26557, 26892, 27230, 27572, 27917, 28266, 28618, 28975, 29335, 29699, 30067, 30440, 30817, 31198, 31583, 31973, 32368, 32767};

//   // compute log2_maxh (output_shift)
//   avg128U1 = _mm_setzero_si128();
//   avg128U2 = _mm_setzero_si128();

//   for (rb=0; rb<nb_rb; rb++) {
//     avg128U1 = _mm_add_epi32(avg128U1,_mm_madd_epi16(ul_ch1[0],ul_ch1[0]));
//     avg128U1 = _mm_add_epi32(avg128U1,_mm_madd_epi16(ul_ch1[1],ul_ch1[1]));
//     avg128U1 = _mm_add_epi32(avg128U1,_mm_madd_epi16(ul_ch1[2],ul_ch1[2]));

//     avg128U2 = _mm_add_epi32(avg128U2,_mm_madd_epi16(ul_ch2[0],ul_ch2[0]));
//     avg128U2 = _mm_add_epi32(avg128U2,_mm_madd_epi16(ul_ch2[1],ul_ch2[1]));
//     avg128U2 = _mm_add_epi32(avg128U2,_mm_madd_epi16(ul_ch2[2],ul_ch2[2]));

//     ul_ch1+=3;
//     ul_ch2+=3;
//   }

//   avg[0] = (((int*)&avg128U1)[0] +
//             ((int*)&avg128U1)[1] +
//             ((int*)&avg128U1)[2] +
//             ((int*)&avg128U1)[3])/(nb_rb*12);

//   avg[1] = (((int*)&avg128U2)[0] +
//             ((int*)&avg128U2)[1] +
//             ((int*)&avg128U2)[2] +
//             ((int*)&avg128U2)[3])/(nb_rb*12);

//   //    msg("avg0 = %d, avg1 = %d\n",avg[0],avg[1]);
//   avg[0] = cmax(avg[0],avg[1]);
//   avg[1] = log2_approx(avg[0]);
//   output_shift = cmax(0,avg[1]-10);
//   //output_shift  = (log2_approx(avg[0])/2)+ log2_approx(frame_parms->nb_antennas_rx-1)+1;
//   //    msg("avg= %d, shift = %d\n",avg[0],output_shift);

//   ul_ch1 = (__m128i*)&ul_ch_estimates[pilot_pos1*frame_parms->N_RB_UL*12];
//   ul_ch2 = (__m128i*)&ul_ch_estimates[pilot_pos2*frame_parms->N_RB_UL*12];

//   // correlate and average the 2 channel estimates ul_ch1*ul_ch2
//   for (rb=0; rb<nb_rb; rb++) {
//     mmtmpD0 = _mm_madd_epi16(ul_ch1[0],ul_ch2[0]);
//     mmtmpD1 = _mm_shufflelo_epi16(ul_ch1[0],_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)&conjugate);
//     mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch2[0]);
//     mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
//     mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
//     mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
//     mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);
//     R[0] = _mm_packs_epi32(mmtmpD2,mmtmpD3);

//     mmtmpD0 = _mm_madd_epi16(ul_ch1[1],ul_ch2[1]);
//     mmtmpD1 = _mm_shufflelo_epi16(ul_ch1[1],_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)&conjugate);
//     mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch2[1]);
//     mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
//     mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
//     mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
//     mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);
//     R[1] = _mm_packs_epi32(mmtmpD2,mmtmpD3);

//     mmtmpD0 = _mm_madd_epi16(ul_ch1[2],ul_ch2[2]);
//     mmtmpD1 = _mm_shufflelo_epi16(ul_ch1[2],_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_shufflehi_epi16(mmtmpD1,_MM_SHUFFLE(2,3,0,1));
//     mmtmpD1 = _mm_sign_epi16(mmtmpD1,*(__m128i*)&conjugate);
//     mmtmpD1 = _mm_madd_epi16(mmtmpD1,ul_ch2[2]);
//     mmtmpD0 = _mm_srai_epi32(mmtmpD0,output_shift);
//     mmtmpD1 = _mm_srai_epi32(mmtmpD1,output_shift);
//     mmtmpD2 = _mm_unpacklo_epi32(mmtmpD0,mmtmpD1);
//     mmtmpD3 = _mm_unpackhi_epi32(mmtmpD0,mmtmpD1);
//     R[2] = _mm_packs_epi32(mmtmpD2,mmtmpD3);

//     R[0] = _mm_add_epi16(_mm_srai_epi16(R[0],1),_mm_srai_epi16(R[1],1));
//     R[0] = _mm_add_epi16(_mm_srai_epi16(R[0],1),_mm_srai_epi16(R[2],1));

//     Ravg[0] += (((short*)&R)[0] +
//                 ((short*)&R)[2] +
//                 ((short*)&R)[4] +
//                 ((short*)&R)[6])/(nb_rb*4);

//     Ravg[1] += (((short*)&R)[1] +
//                 ((short*)&R)[3] +
//                 ((short*)&R)[5] +
//                 ((short*)&R)[7])/(nb_rb*4);

//     ul_ch1+=3;
//     ul_ch2+=3;
//   }

//   // phase estimation on Ravg
//   //   Ravg[0] = 56;
//   //   Ravg[1] = 0;
//   rv = Ravg[0];
//   iv = Ravg[1];

//   if (iv<0)
//     iv = -Ravg[1];

//   if (rv<iv) {
//     rv = iv;
//     iv = Ravg[0];
//     conj_flag = 1;
//   }

//   //   msg("rv = %d, iv = %d\n",rv,iv);
//   //   msg("max_avg = %d, log2_approx = %d, shift = %d\n",avg[0], avg[1], output_shift);

//   for (k=0; k<6; k++) {
//     (iv<(((int32_t)(alpha[a_idx]*rv))>>15)) ? (a_idx -= 32>>k) : (a_idx += 32>>k);
//   }

//   (conj_flag==1) ? (phase_idx = 127-(a_idx>>1)) : (phase_idx = (a_idx>>1));

//   if (Ravg[1]<0)
//     phase_idx = -phase_idx;

//   return(phase_idx);
// #elif defined(__arm__)
//   return(0);
// #endif
// }
