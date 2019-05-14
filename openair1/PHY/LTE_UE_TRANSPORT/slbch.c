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

/*! \file PHY/LTE_TRANSPORT/slbch.c
* \brief Top-level routines for transmitting and receiving the sidelink broadcast channel
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
/* file: slbch.c
   purpose: TX and RX procedures for Sidelink Broadcast Channel
   author: raymond.knopp@eurecom.fr
   date: 02.05.2018
*/

//#include "defs.h"
#include "PHY/defs_UE.h"
#include "PHY/phy_extern.h"
#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/LTE_ESTIMATION/lte_estimation.h"

#define PSBCH_A 40
#define PSBCH_E 1008 //12REs/PRB*6PRBs*7symbols*2 bits/RB

//#define PSBCH_DEBUG 1

void dft_lte(int32_t *z,int32_t *d, int32_t Msc_PUSCH, uint8_t Nsymb);
void ulsch_channel_level(int32_t **drs_ch_estimates_ext, LTE_DL_FRAME_PARMS *frame_parms, int32_t *avg, uint16_t nb_rb, int symbol_offset);
void lte_idft(LTE_DL_FRAME_PARMS *frame_parms,uint32_t *z, uint16_t Msc_PUSCH);
void pbch_quantize(int8_t *pbch_llr8, int16_t *pbch_llr, uint16_t len);
void ulsch_extract_rbs_single(int32_t **rxdataF,
                              int32_t **rxdataF_ext,
                              uint32_t first_rb,
                              uint32_t nb_rb,
                              uint8_t l,
                              uint8_t Ns,
                              LTE_DL_FRAME_PARMS *frame_parms);
void lte_idft(LTE_DL_FRAME_PARMS *frame_parms,uint32_t *z, uint16_t Msc_PUSCH);

int dlsch_encoding0(LTE_DL_FRAME_PARMS *frame_parms,
                    unsigned char *a,
                    uint8_t num_pdcch_symbols,
                    LTE_eNB_DLSCH_t *dlsch,
                    int frame,
                    uint8_t subframe,
                    time_stats_t *rm_stats,
                    time_stats_t *te_stats,
                    time_stats_t *i_stats);

	  
int generate_slbch(int32_t **txdataF,
		   short amp,
		   LTE_DL_FRAME_PARMS *frame_parms,
		   int subframe,
		   uint8_t *slmib) {
  
  uint8_t slbch_a[PSBCH_A>>3];
  uint32_t psbch_D;
  uint8_t psbch_d[96+(3*(16+PSBCH_A))];
  uint8_t psbch_w[3*3*(16+PSBCH_A)];
  uint8_t psbch_e[PSBCH_E];
  uint8_t RCC;
  int a;

  psbch_D    = 16+PSBCH_A;
  
  AssertFatal(frame_parms->Ncp==NORMAL,"Only Normal Prefix supported for Sidelink\n");
  AssertFatal(frame_parms->Ncp==NORMAL,"Only Normal Prefix supported for Sidelink\n");

  bzero(slbch_a,PSBCH_A>>3);
  bzero(psbch_e,PSBCH_E);
  memset(psbch_d,LTE_NULL,96);
    
  for (int i=0; i<(PSBCH_A>>3); i++)
    slbch_a[(PSBCH_A>>3)-i-1] = slmib[i];

  ccodelte_encode(PSBCH_A,2,slbch_a,psbch_d+96,0);
  RCC = sub_block_interleaving_cc(psbch_D,psbch_d+96,psbch_w);
  
  lte_rate_matching_cc(RCC,PSBCH_E,psbch_w,psbch_e);
  //  for (int i=0;i<PSBCH_E;i++) printf("PSBCH E[%d] %d\n",i,psbch_e[i]);
  pbch_scrambling(frame_parms,
		  psbch_e,
		  PSBCH_E,
		  1);
  int symb=0;
  uint8_t *eptr = psbch_e;
  int16_t *txptr;
  int k;

  a = (amp*SQRT_18_OVER_32_Q15)>>(15-2);
  int Nsymb=14;

  int16_t precin[144*12];
  int16_t precout[144*12];

  for (int i=0;i<144*7;i++)
    if (*eptr++ == 1) precin[i] =-a;
    else              precin[i] = a;
  
  dft_lte((int32_t*)precout,
	  (int32_t*)precin,
	  72,
	  12);

  int j=0;
  for (symb=0;symb<10;symb++) { 
    k = (frame_parms->ofdm_symbol_size<<1)-72;
    //    printf("Generating PSBCH in symbol %d offset %d\n",symb,
    //	       (subframe*Nsymb*frame_parms->ofdm_symbol_size)+(symb*frame_parms->ofdm_symbol_size));

    txptr = (int16_t*)&txdataF[0][(subframe*Nsymb*frame_parms->ofdm_symbol_size)+(symb*frame_parms->ofdm_symbol_size)];


    
    // first half (negative frequencies)
    for (int i=0;i<72;i++,j++,k++) txptr[k] = precout[j];
    // second half (positive frequencies)
    for (int i=0,k=0;i<72;i++,j++,k++) txptr[k] = precout[j];
     
    if (symb==0) symb+=3;
  }

  // scale by sqrt(72/62)
  // note : we have to scale for TX power requirements too, beta_PSBCH !

  //  //printf("[PSS] amp=%d, a=%d\n",amp,a);
  
  
  return(0);
}

int rx_psbch(PHY_VARS_UE *ue,int frame_rx,int subframe_rx) {
  
  
  int16_t **rxdataF      = ue->sl_rxdataF[ue->current_thread_id[subframe_rx]];
  int32_t **rxdataF_ext  = ue->pusch_slbch->rxdataF_ext;
  int32_t **drs_ch_estimates = ue->pusch_slbch->drs_ch_estimates;
  int32_t **rxdataF_comp     = ue->pusch_slbch->rxdataF_comp;
  int32_t **ul_ch_mag        = ue->pusch_slbch->ul_ch_mag;
  int32_t avgs;
  uint8_t log2_maxh=0;
  int32_t avgU[2];
  //int Nsymb=7;

  RU_t ru_tmp;
  memset((void*)&ru_tmp,0,sizeof(RU_t));
  
  memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
  ru_tmp.N_TA_offset=0;
  ru_tmp.common.rxdata_7_5kHz     = (int32_t**)malloc16(ue->frame_parms.nb_antennas_rx*sizeof(int32_t*)); 
  for (int aa=0;aa<ue->frame_parms.nb_antennas_rx;aa++) 
    ru_tmp.common.rxdata_7_5kHz[aa] = (int32_t*)ue->sl_rxdata_7_5kHz[ue->current_thread_id[0]][aa];//(int32_t*)&ue->common_vars.rxdata_syncSL[aa][ue->rx_offsetSL*2];
  ru_tmp.common.rxdataF = (int32_t**)rxdataF;
  ru_tmp.nb_rx = ue->frame_parms.nb_antennas_rx;

  int SLaoffset=0;
  if (ue->SLonly==0) SLaoffset=1;
  // if SLonly then all antennas are SL only, else they are inteleaved with legacy RX antennas

  if (ue->is_synchronizedSL == 1) { // Run front-end processing
    ru_tmp.common.rxdata            = (int32_t**)malloc16(ue->frame_parms.nb_antennas_rx*sizeof(int32_t*));
    for (int aa=SLaoffset;aa<(ue->frame_parms.nb_antennas_rx<<SLaoffset);aa+=(1<<SLaoffset)) {
      ru_tmp.common.rxdata[aa]        = (int32_t*)&ue->common_vars.rxdata[aa][0];
    }


    remove_7_5_kHz(&ru_tmp,0);
    remove_7_5_kHz(&ru_tmp,1);

    free(ru_tmp.common.rxdata);


  }
  LOG_D(PHY,"Running PBCH detection with Nid_SL %d (is_synchronizedSL %d) rxdata %p\n",ue->frame_parms.Nid_SL,ue->is_synchronizedSL,ue->common_vars.rxdata[0]);
  LOG_D(PHY,"slbch_decoding: FEP in %d.%d rx signal energy %d dB %d dB\n",frame_rx,subframe_rx,
         dB_fixed((uint32_t)signal_energy(&ue->common_vars.rxdata[0][ue->frame_parms.samples_per_tti*subframe_rx],ue->frame_parms.samples_per_tti)),
         dB_fixed((uint32_t)signal_energy((int32_t*)ue->sl_rxdata_7_5kHz[ue->current_thread_id[0]][0],ue->frame_parms.samples_per_tti)));
 
  for (int l=0; l<11; l++) {
    slot_fep_ul(&ru_tmp,l%7,(l>6)?1:0,0);
    ulsch_extract_rbs_single((int32_t**)rxdataF,
			     (int32_t**)rxdataF_ext,
			     (ue->frame_parms.N_RB_UL/2)-3,
			     6,
			     l,
			     0,
			     &ue->frame_parms);
    if (l==0) l+=2;
  }
  free(ru_tmp.common.rxdata_7_5kHz);
#ifdef PSBCH_DEBUG
  if (ue->is_synchronizedSL == 1 && ue->frame_parms.Nid_SL==170) {
     LOG_M("slbch.m","slbchrx",ue->common_vars.rxdata[0],ue->frame_parms.samples_per_tti,1,1);
     LOG_M("slbch_rxF.m",
	       "slbchrxF",
	       &rxdataF[0][0],
	       14*ue->frame_parms.ofdm_symbol_size,1,1);
     LOG_M("slbch_rxF_ext.m","slbchrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
  }
#endif
  
  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    6,
			    0,
			    0,
			    ue->gh[0][0], //u
			    0, //v
			    (ue->frame_parms.Nid_SL>>1)&7, //cyclic_shift
			    3,
			    0, // interpolation
			    0);
  
  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    6,
			    0,
			    0,
			    ue->gh[0][1],//u
			    0,//v
			    (ue->frame_parms.Nid_SL>>1)&7,//cyclic_shift,
			    10,
			    0, // interpolation
			    0);
  
  ulsch_channel_level(drs_ch_estimates,
		      &ue->frame_parms,
		      avgU,
		      2,0);
  
#ifdef PSBCH_DEBUG
  if (ue->is_synchronizedSL == 1 && ue->frame_parms.Nid_SL == 170) LOG_M("drsbch_est0.m","drsbchest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif
  
  avgs = 0;
  
  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++)
    avgs = cmax(avgs,avgU[aarx]);
  
  //      log2_maxh = 4+(log2_approx(avgs)/2);
  
  log2_maxh = (log2_approx(avgs)/2)+ log2_approx(ue->frame_parms.nb_antennas_rx-1)+4;
  
  
  for (int l=0; l<10; l++) {
    
    
    ulsch_channel_compensation(
			       rxdataF_ext,
			       drs_ch_estimates,
			       ul_ch_mag,
			       NULL,
			       rxdataF_comp,
			       &ue->frame_parms,
			       l,
			       2, //Qm
			       6, //nb_rb
			       log2_maxh); // log2_maxh+I0_shift
    
    if (ue->frame_parms.nb_antennas_rx > 1)
      ulsch_detection_mrc(&ue->frame_parms,
			  rxdataF_comp,
			  ul_ch_mag,
			  NULL,
			  l,
			  6 //nb_rb
			  );
    
    freq_equalization(&ue->frame_parms,
		      rxdataF_comp,
		      ul_ch_mag,
		      NULL,
		      l,
		      72,
		      2);
    
    if (l==0) l=3;
  }
  lte_idft(&ue->frame_parms,
	   (uint32_t*)rxdataF_comp[0],
	   72);
  
#ifdef PSBCH_DEBUG
  if (ue->frame_parms.Nid_SL == 170) LOG_M("slbch_rxF_comp.m","slbchrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif
  int8_t llr[PSBCH_E];
  int8_t *llrp = llr;
  
  for (int l=0; l<10; l++) {
    pbch_quantize(llrp,
		  (int16_t*)&rxdataF_comp[0][l*ue->frame_parms.N_RB_UL*12],
		  72*2);
    llrp += 72*2;
    if (l==0) l=3;
  }
  pbch_unscrambling(&ue->frame_parms,
		    llr,
		    PSBCH_E,
		    0,
		    1);
  
#ifdef PSBCH_DEBUG
  if (ue->frame_parms.Nid_SL == 170)  LOG_M("slbch_llr.m","slbch_llr",llr,PSBCH_E,1,4);
#endif
  
  uint8_t slbch_a[2+(PSBCH_A>>3)];
  uint32_t psbch_D;
  int8_t psbch_d_rx[96+(3*(16+PSBCH_A))];
  uint8_t dummy_w_rx[3*3*(16+PSBCH_A)];
  int8_t psbch_w_rx[3*3*(16+PSBCH_A)];
  int8_t *psbch_e_rx=llr;
  uint8_t RCC;
  //int a;
  uint8_t *decoded_output = ue->slss_rx.slmib;
  
  psbch_D    = 16+PSBCH_A;
  
  
  memset(dummy_w_rx,0,3*3*(psbch_D));
  RCC = generate_dummy_w_cc(psbch_D,
			    dummy_w_rx);
  
  
  lte_rate_matching_cc_rx(RCC,PSBCH_E,psbch_w_rx,dummy_w_rx,psbch_e_rx);
  
  sub_block_deinterleaving_cc(psbch_D,
			      &psbch_d_rx[96],
			      &psbch_w_rx[0]);
  
  memset(slbch_a,0,((16+PSBCH_A)>>3));
  
  
  
  
  phy_viterbi_lte_sse2(psbch_d_rx+96,slbch_a,16+PSBCH_A);
  
  // Fix byte endian of PSBCH (bit 39 goes in first)
  for (int i=0; i<(PSBCH_A>>3); i++)
    decoded_output[(PSBCH_A>>3)-i-1] = slbch_a[i];
  
  if (ue->is_synchronizedSL==0) LOG_I(PHY,"SFN.SF %d.%d SLBCH  : %x.%x.%x.%x.%x\n",frame_rx,subframe_rx,decoded_output[0],decoded_output[1],decoded_output[2],decoded_output[3],decoded_output[4]);
  
#ifdef DEBUG_PSBCH
  LOG_I(PHY,"PSBCH CRC %x : %x\n",
	crc16(slbch_a,PSBCH_A),
	((uint16_t)slbch_a[PSBCH_A>>3]<<8)+slbch_a[(PSBCH_A>>3)+1]);
#endif
  
  uint16_t crc = (crc16(slbch_a,PSBCH_A)>>16) ^
    (((uint16_t)slbch_a[PSBCH_A>>3]<<8)+slbch_a[(PSBCH_A>>3)+1]);

  ue->slbch_rxops++;

  if (crc>0)  {
     LOG_I(PHY,"SLBCH not received in %d.%d\n", frame_rx,subframe_rx);
     ue->slbch_errors++;
     return(-1);
  }
  else {
     //SLSS_t dummy_slss;
     int testframe;
     int testsubframe;
     ue_decode_si(ue->Mod_id,
                   0, // CC_id
                   0, // frame
                   0, // eNB_index
                   NULL, // pdu, NULL for MIB-SL
                   0,    // len, 0 for MIB-SL
                   &ue->slss_rx,
                   &testframe,
                   &testsubframe);
     if (ue->is_synchronizedSL!=0 || ue->is_synchronized!=0) 
        AssertFatal(testframe==frame_rx && testsubframe==subframe_rx,
	         "SFN.SF %d.%d != %d.%d\n",testframe,testsubframe,frame_rx,subframe_rx);
     return(0);
  }
}
