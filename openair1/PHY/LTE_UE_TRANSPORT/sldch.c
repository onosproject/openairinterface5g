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

/*! \file PHY/LTE_TRANSPORT/sldch.c
 * \brief Functions to Generate and Receive PSDCH
 * \author R. Knopp
 * \date 2017
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */
#ifndef __LTE_TRANSPORT_SLSS__C__
#define __LTE_TRANSPORT_SLSS__C__
#include "PHY/defs_UE.h"
#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/LTE_ESTIMATION/lte_estimation.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "LAYER2/MAC/mac_proto.h"
#include "SCHED_UE/sched_UE.h"

extern uint8_t D2D_en;
void ulsch_channel_level(int32_t **drs_ch_estimates_ext, LTE_DL_FRAME_PARMS *frame_parms, int32_t *avg, uint16_t nb_rb, int symbol_offset);
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


//#define PSDCH_DEBUG 1

void sldch_decoding(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_rx,int subframe_rx,int npsdch,int nprb,int rvidx) {

  int Nsymb = 7;
  int16_t **rxdataF_ext      = (int16_t**)ue->pusch_sldch[ue->current_thread_id[subframe_rx]]->rxdataF_ext;
  int16_t **drs_ch_estimates = (int16_t**)ue->pusch_sldch[ue->current_thread_id[subframe_rx]]->drs_ch_estimates;
  int16_t **rxdataF_comp     = (int16_t**)ue->pusch_sldch[ue->current_thread_id[subframe_rx]]->rxdataF_comp;
  int16_t **ul_ch_mag        = (int16_t**)ue->pusch_sldch[ue->current_thread_id[subframe_rx]]->ul_ch_mag;
  int16_t **rxdata_7_5kHz    = (int16_t**)ue->sl_rxdata_7_5kHz[ue->current_thread_id[subframe_rx]];
  int16_t **rxdataF          = (int16_t**)ue->sl_rxdataF[ue->current_thread_id[subframe_rx]];
  int32_t avgs;
  uint8_t log2_maxh=0;
  int32_t avgU[2];


  LOG_D(PHY,"sldch_decoding %d.%d npsdch %d rvidx %d sl_fep_done %d\n",frame_rx,subframe_rx,npsdch,rvidx,proc->sl_fep_done);

  if (ue->sldch_received[npsdch] > 0) return;

  int SLaoffset=0;
  if (ue->SLonly==0) SLaoffset=1;

  // slot FEP
  if (proc->sl_fep_done == 0) {
    proc->sl_fep_done = 1;
    RU_t ru_tmp;
    memset((void*)&ru_tmp,0,sizeof(RU_t));
    
    memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
    ru_tmp.N_TA_offset=0;
    //    ru_tmp.common.rxdata = ue->common_vars.rxdata;

    ru_tmp.common.rxdata            = (int32_t**)malloc16((1+ue->frame_parms.nb_antennas_rx)*sizeof(int32_t*));
    int aaSL=0;
    for (int aa=SLaoffset;aa<(ue->frame_parms.nb_antennas_rx<<SLaoffset);aa+=(1<<SLaoffset)) {
      ru_tmp.common.rxdata[aaSL]        = (int32_t*)&ue->common_vars.rxdata[aa][0];
      aaSL++;
    }
    ru_tmp.common.rxdata_7_5kHz = (int32_t**)rxdata_7_5kHz;
    ru_tmp.common.rxdataF = (int32_t**)rxdataF;
    ru_tmp.nb_rx = ue->frame_parms.nb_antennas_rx;
    
    remove_7_5_kHz(&ru_tmp,(subframe_rx<<1));
    remove_7_5_kHz(&ru_tmp,(subframe_rx<<1)+1);

    free(ru_tmp.common.rxdata);

    // extract symbols from slot  
    for (int l=0; l<Nsymb; l++) {
      slot_fep_ul(&ru_tmp,l,(subframe_rx<<1),0);
      if (l<Nsymb-1)  // skip last symbol in second slot
	slot_fep_ul(&ru_tmp,l,(subframe_rx<<1)+1,0);
    }
  }
   LOG_D(PHY,"sldch_decoding: FEP in %d.%d for npsdch %d rvidx %d rx signal energy %d (%p) dB %d dB\n",frame_rx,subframe_rx,npsdch,rvidx,
         dB_fixed(signal_energy((int32_t*)&ue->common_vars.rxdata[SLaoffset][ue->frame_parms.samples_per_tti*subframe_rx],ue->frame_parms.samples_per_tti)),
         &ue->common_vars.rxdata[SLaoffset][ue->frame_parms.samples_per_tti*subframe_rx],
         dB_fixed(signal_energy((int32_t*)ue->sl_rxdata_7_5kHz[ue->current_thread_id[subframe_rx]][0],ue->frame_parms.samples_per_tti)));

  for (int l=0; l<Nsymb; l++) {
    ulsch_extract_rbs_single((int32_t**)rxdataF,
			     (int32_t**)rxdataF_ext,
			     nprb,
			     2,
			     l,
			     (subframe_rx<<1),
			     &ue->frame_parms);

    if (l<Nsymb-1)  // skip last symbol in second slot
      ulsch_extract_rbs_single((int32_t**)rxdataF,
			       (int32_t**)rxdataF_ext,
			       nprb,
			       2,
			       l,
			       (subframe_rx<<1)+1,
			       &ue->frame_parms);
    
  }

#ifdef PSDCH_DEBUG
  write_output("sldch_rxF.m",
	       "sldchrxF",
	       &rxdataF[0][0],
	       14*ue->frame_parms.ofdm_symbol_size,1,1);
  write_output("sldch_rxF_ext.m","sldchrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
#endif

  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    2,
			    frame_rx,
			    subframe_rx,
			    0, //u
			    0, //v
			    0, //cyclic_shift
			    3,
			    0, // interpolation
			    0);

  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    2,
			    frame_rx,
			    subframe_rx,
			    0,//u
			    0,//v
			    0,//cyclic_shift,
			    10,
			    0, // interpolation
			    0);

  ulsch_channel_level((int32_t**)drs_ch_estimates,
		      &ue->frame_parms,
		      avgU,
		      2,0);

#ifdef PSDCH_DEBUG
  write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif

  avgs = 0;
  
  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++)
    avgs = cmax(avgs,avgU[aarx]);
  
  //      log2_maxh = 4+(log2_approx(avgs)/2);
  
  log2_maxh = (log2_approx(avgs)/2)+ log2_approx(ue->frame_parms.nb_antennas_rx-1)+4;

  if (log2_maxh > 5) LOG_D(PHY,"sldch_decoding %d.%d npsdch %d log2_maxh %d\n",frame_rx,subframe_rx,npsdch,log2_maxh);



  for (int l=0; l<(Nsymb<<1)-1; l++) {

    if (((ue->frame_parms.Ncp == 0) && ((l==3) || (l==10)))||   // skip pilots
        ((ue->frame_parms.Ncp == 1) && ((l==2) || (l==8)))) {
      l++;
    }

    ulsch_channel_compensation(
			       (int32_t**)rxdataF_ext,
			       (int32_t**)drs_ch_estimates,
			       (int32_t**)ul_ch_mag,
			       NULL,
			       (int32_t**)rxdataF_comp,
			       &ue->frame_parms,
			       l,
			       2, //Qm
			       2, //nb_rb
			       log2_maxh); // log2_maxh+I0_shift

    if (ue->frame_parms.nb_antennas_rx > 1)
      ulsch_detection_mrc(&ue->frame_parms,
			  (int32_t**)rxdataF_comp,
			  (int32_t**)ul_ch_mag,
			  NULL,
			  l,
			  2 //nb_rb
			  );
    
    freq_equalization(&ue->frame_parms,
		      (int32_t**)rxdataF_comp,
		      (int32_t**)ul_ch_mag,
		      NULL,
		      l,
		      24,
		      2);
  
  }
  lte_idft(&ue->frame_parms,
           (uint32_t*)rxdataF_comp[0],
           24);

#ifdef PSDCH_DEBUG
  write_output("sldch_rxF_comp.m","sldchrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif

  int E = 12*2*2*((Nsymb-1)<<1);

  int16_t *llrp = ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]];

  for (int l=0; l<(Nsymb<<1)-1; l++) {

    if (((ue->frame_parms.Ncp == 0) && ((l==3) || (l==10)))||   // skip pilots
        ((ue->frame_parms.Ncp == 1) && ((l==2) || (l==8)))) {
      l++;
    }

    ulsch_qpsk_llr(&ue->frame_parms,
		   (int32_t **)rxdataF_comp,
		   (int16_t *)ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]],
		   l,
		   2,
		   &llrp);
  }
#ifdef PSDCH_DEBUG  
  write_output("sldch_llr.m","sldchllr",ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]],
               12*2*(ue->frame_parms.symbols_per_tti),
               1,0);
#endif  

  // unscrambling

  uint32_t x1,x2=510;

  uint32_t s = lte_gold_generic(&x1, &x2, 1);
  int k=0;
  int16_t c;

  

  for (int i=0; i<(1+(E>>5)); i++) {
    for (int j=0; j<32; j++,k++) {
        c = (int16_t)((((s>>j)&1)<<1)-1);
        ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]][k] = c*ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]][k];
    }    
    s = lte_gold_generic(&x1, &x2, 0);
  }

  // Deinterleaving

  int Cmux = (Nsymb-1)*2;
  for (int i=0,j=0;i<Cmux;i++) {
    for (int r=0;r<24;r++) {
      ue->sldch_dlsch_llr[ue->current_thread_id[subframe_rx]][((r*Cmux)+i)<<1]     = ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]][j++];
      ue->sldch_dlsch_llr[ue->current_thread_id[subframe_rx]][(((r*Cmux)+i)<<1)+1] = ue->sldch_ulsch_llr[ue->current_thread_id[subframe_rx]][j++];
 //     	printf("dlsch_llr[%d] %d(%d) dlsch_llr[%d] %d(%d)\n",
 //     	       ((r*Cmux)+i)<<1,ue->sldch_dlsch_llr[((r*Cmux)+i)<<1],j-2,(((r*Cmux)+i)<<1)+1,ue->sldch_dlsch_llr[(((r*Cmux)+i)<<1)+1],j-1);
    }
  }


  // Decoding
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->rvidx = rvidx;
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->nb_rb = 2;
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->TBS   = 256;
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->Qm    = 2;
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->G     = E;
  ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->Nl    = 1;

  //  for (int i=0;i<E/16;i++) printf("decoding: E[%d] %d\n",i,ue->slsch_dlsch_llr[i]);

  int ret = dlsch_decoding(ue,
			   ue->sldch_dlsch_llr[ue->current_thread_id[subframe_rx]],
			   &ue->frame_parms,
			   ue->dlsch_rx_sldch[npsdch],
			   ue->dlsch_rx_sldch[npsdch]->harq_processes[0],
			   frame_rx,
			   subframe_rx,
			   0,
			   0,
			   1);

//  printf("slsch decoding round %d ret %d\n",ue->dlsch_rx_sldch->harq_processes[0]->round,ret);
  if (ret<ue->dlsch_rx_sldch[npsdch]->max_turbo_iterations) {
    LOG_I(PHY,"SLDCH received for npsdch %d (rvidx %d, iter %d)\n",
	  npsdch,
	  rvidx,ret);
    ue->sldch_received[npsdch] = 1;
    ue->sldch_rxcnt[npsdch]++;
    ue_send_sl_sdu(0,
                   0,
                   frame_rx,
                   subframe_rx,
                   ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->b,
                   ue->dlsch_rx_sldch[npsdch]->harq_processes[0]->TBS>>3,
                   0,
                   SL_DISCOVERY_FLAG_YES);

  }

  //exit(-1);
}

void rx_sldch(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc, int frame_rx,int subframe_rx) {

  SLDCH_t *sldch = &ue->sldch_rx;

  AssertFatal(frame_rx<1024 && frame_rx>=0,"frame %d is illegal\n",frame_rx);
  AssertFatal(subframe_rx<10 && subframe_rx>=0,"subframe %d is illegal\n",subframe_rx);
  AssertFatal(sldch!=NULL,"SLDCH is null\n");

  uint32_t O = sldch->offsetIndicator;
  uint32_t P = sldch->discPeriod;
  uint32_t absSF = (frame_rx*10)+subframe_rx;
  uint32_t absSF_offset,absSF_modP;
  int rvtab[4]={0,2,3,1};

  absSF_offset = absSF-O;

  if (absSF_offset < O) return;

  absSF_modP = absSF_offset%P;
  // compute parameters
  AssertFatal(sldch->bitmap_length==4 || sldch->bitmap_length==8 ||
              sldch->bitmap_length==12 || sldch->bitmap_length==16 ||
              sldch->bitmap_length==30 || sldch->bitmap_length==40 ||
              sldch->bitmap_length==42,"SLDCH Bitmap_length %x not supported\n",
              sldch->bitmap_length);

  if (absSF_modP == 0) memset((void*)ue->sldch_received,0,sizeof(ue->sldch_received));

  if (absSF_modP >= sldch->bitmap_length) return;
  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & sldch->bitmap1) == 0) return;

  // if we get here, then there is a PSDCH subframe for a potential reception

  // compute parameters
  AssertFatal(sldch->bitmap_length==4 || sldch->bitmap_length==8 ||
	      sldch->bitmap_length==12 || sldch->bitmap_length==16 ||
	      sldch->bitmap_length==30 || sldch->bitmap_length==40 ||
	      sldch->bitmap_length==42,"SLDCH Bitmap_length %x not supported\n",
	      sldch->bitmap_length);

  int LPSDCH=0;
  for (int i=0;i<sldch->bitmap_length;i++) if (((((uint64_t)1)<<i)&sldch->bitmap1) > 00) LPSDCH++;
  
  AssertFatal(sldch->type == disc_type1 || sldch->type == disc_type2B,
	      "unknown Discovery type %d\n",sldch->type);

  int N_TX_SLD = 1+sldch->numRetx;
  uint32_t M_RB_PSDCH_RP = sldch->N_SL_RB;

  if (sldch->type == disc_type1) {
    int Ni = LPSDCH/N_TX_SLD;
    int Nf = M_RB_PSDCH_RP>>1;
    //    int a_ji = (((sldch->j-1)*(Nf/N_TX_SLD)) + (sldch->n_psdch/Ni))%Nf;
    //    int b_1i = sldch->n_psdch%Ni;
    // repetition number
    int jrx;    
    int npsdch;
    int nprb;

    LOG_D(PHY,"LPSDCH %d (%llx), N_TX_SLD %d, Ni %d, Nf %d\n",LPSDCH,(long long unsigned int)sldch->bitmap1,N_TX_SLD,Ni,Nf); 
    // loop over all candidate PRBs 
    for (int a_ji=0;a_ji<Nf;a_ji++){
      jrx = absSF_modP%N_TX_SLD;//i/(Nf/N_TX_SLD);
      // b_1= absSF_ModP/N_TX_SLD
      npsdch = Ni*((a_ji+Nf-(jrx*Nf/N_TX_SLD))%Nf) + ((absSF_modP/N_TX_SLD)%Ni);
      nprb = a_ji<<1;
      if (nprb<(sldch->N_SL_RB>>1)) nprb+=sldch->prb_Start;
      else                          nprb+=(sldch->prb_End-(sldch->N_SL_RB>>1));
      // call decoding for candidate npsdch
//      if (npsdch==0) {
//        LOG_I(PHY,"SLDCH (RX): absSF_modP %d Trying npsdch %d, j %d rvidx %d (nprb %d)\n",absSF_modP,npsdch,jrx,rvtab[jrx],nprb);
        sldch_decoding(ue,proc,frame_rx,subframe_rx,npsdch,nprb,rvtab[jrx]);
 //     }
    }

  }
  else {
    AssertFatal(1==0,"Discovery Type 2B not supported yet\n");
  }
}

extern int multicast_link_write_sock(int groupP, char *dataP, uint32_t sizeP);

void generate_sldch(PHY_VARS_UE *ue,SLDCH_t *sldch,int frame_tx,int subframe_tx) {

  UE_tport_t pdu;
  size_t sldch_header_len = sizeof(UE_tport_header_t);

  pdu.header.packet_type = SLDCH;
  pdu.header.absSF = (frame_tx*10)+subframe_tx;


  AssertFatal(sldch->payload_length <=1500-sldch_header_len - sizeof(SLDCH_t) + sizeof(uint8_t*),
                "SLDCH payload length > %lu\n",
                1500-sldch_header_len - sizeof(SLDCH_t) + sizeof(uint8_t*));
  memcpy((void*)&pdu.sldch,
         (void*)sldch,
         sizeof(SLDCH_t));

  LOG_D(PHY,"SLDCH configuration %lu bytes, TBS payload %d bytes => %lu bytes\n",
        sizeof(SLDCH_t)-sizeof(uint8_t*),
        sldch->payload_length,
        sldch_header_len+sizeof(SLDCH_t)-sizeof(uint8_t*)+sldch->payload_length);

  multicast_link_write_sock(0,
                            (char *)&pdu,
                            sldch_header_len+sizeof(SLDCH_t));

}


#endif

void sldch_codingmodulation(PHY_VARS_UE *ue,int frame_tx,int subframe_tx,int nprb,int rvidx) {

  SLDCH_t *sldch         = ue->sldch;
  LTE_eNB_DLSCH_t *dlsch = ue->dlsch_sldch;
  LTE_UE_ULSCH_t *ulsch  = ue->ulsch_sldch;
  int tx_amp;
  uint32_t Nsymb = 7;
  // 24 REs/PRB * 2*(Nsymb-1) symbols * 2 bits/RE 

  AssertFatal(sldch!=NULL,"ue->sldch is null\n");
  



  LOG_D(PHY,"Generating SLDCH in %d.%d for rvidx %d, npsdch %d, first rb %d\n",
	frame_tx,subframe_tx,rvidx,sldch->n_psdch,nprb);

  if (rvidx==0) ue->sldch_txcnt[sldch->n_psdch]++;

  dlsch->harq_processes[0]->nb_rb       = 2;
  dlsch->harq_processes[0]->TBS         = 256;
  dlsch->harq_processes[0]->Qm          = 2;
  dlsch->harq_processes[0]->mimo_mode   = SISO;
  dlsch->harq_processes[0]->rb_alloc[0] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[1] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[2] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[3] = 0; // unused for SL
  dlsch->harq_processes[0]->Nl          = 1;
  dlsch->harq_processes[0]->round       = sldch->j; 
  dlsch->harq_processes[0]->rvidx       = rvidx;

  dlsch_encoding0(&ue->frame_parms,
		 sldch->payload,
		 0, // means SL
		 dlsch,
		 frame_tx,
		 subframe_tx,
		 &ue->ulsch_rate_matching_stats,
		 &ue->ulsch_turbo_encoding_stats,
		 &ue->ulsch_interleaving_stats);

  int Cmux = (Nsymb-1)<<1;
  uint8_t *eptr;
  for (int i=0,j=0; i<Cmux; i++)
    // 24 = 12*(Nsymb-1)*2/(Nsymb-1)
    for (int r=0; r<24; r++) {
       if (dlsch->harq_processes[0]->Qm == 2) {
           eptr=&dlsch->harq_processes[0]->e[((r*Cmux)+i)<<1];
           ulsch->h[j++] = *eptr++;
           ulsch->h[j++] = *eptr++;
       }
       else if (dlsch->harq_processes[0]->Qm == 4) {
           eptr=&dlsch->harq_processes[0]->e[((r*Cmux)+i)<<2];
           ulsch->h[j++] = *eptr++;
           ulsch->h[j++] = *eptr++;
           ulsch->h[j++] = *eptr++;
           ulsch->h[j++] = *eptr++;
       }
       else {
            AssertFatal(1==0,"64QAM not supported for SL\n");
       }
    }

  // scrambling
  uint32_t cinit=510;

  ulsch->harq_processes[0]->nb_rb       = 2;
  ulsch->harq_processes[0]->first_rb    = nprb;
  ulsch->harq_processes[0]->mcs         = 8;
  ulsch->Nsymb_pusch                    = ((Nsymb-1)<<1);

  ue->sl_chan = PSDCH;

  ue->tx_power_dBm[subframe_tx] = 3;
  ue->tx_total_RE[subframe_tx] = 24;
#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[subframe_tx],
		      ue->tx_power_max_dBm,
		      ue->frame_parms.N_RB_UL,
		      2);
#else
  tx_amp = AMP;
#endif  

  if (ue->generate_ul_signal[subframe_tx][0] == 0) 
    for (int aa=0; aa<ue->frame_parms.nb_antennas_tx; aa++) {
      memset(&ue->common_vars.txdataF[aa][subframe_tx*ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti],
	     0,
	     ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti*sizeof(int32_t));
    }
 
  ulsch_modulation(ue->common_vars.txdataF,
		   tx_amp,
		   frame_tx,
                   subframe_tx,
		   &ue->frame_parms,
                   ulsch,
                   1,
                   cinit);

  generate_drs_pusch(ue,
		     NULL,
		     0,
		     tx_amp,
		     subframe_tx,
		     nprb,
		     2,
                     0,
                     NULL,
                     0);

  ue->generate_ul_signal[subframe_tx][0] = 1;

}
  
void check_and_generate_psdch(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {
  
  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);
  
  SLDCH_t *sldch = ue->sldch;
  AssertFatal(sldch!=NULL,"SLDCH is null\n");
  uint32_t O = ue->sldch->offsetIndicator;
  uint32_t P = ue->sldch->discPeriod;
  uint32_t absSF = (frame_tx*10)+subframe_tx;
  uint32_t absSF_offset,absSF_modP;


  absSF_offset = absSF-O;

  if (absSF_offset < O) return;

  if (absSF_offset == 0) sldch->j = 0;

  absSF_modP = absSF_offset%P;

  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  LOG_D(PHY,"SLDCH: SFN.SF %d.%d : absSF_modP %d, bitmap1 %llx\n",frame_tx,subframe_tx,absSF_modP,(unsigned long long)sldch->bitmap1);

  if ((SFpos & sldch->bitmap1) == 0) return;

  // if we get here, then this is a potential PSDCH subframe 

  int rvtab[4] = {0,2,3,1};

  int rvidx = rvtab[sldch->j&3];
  int nprb;

  // compute parameters
  AssertFatal(sldch->bitmap_length==4 || sldch->bitmap_length==8 ||
	      sldch->bitmap_length==12 || sldch->bitmap_length==16 ||
	      sldch->bitmap_length==30 || sldch->bitmap_length==40 ||
	      sldch->bitmap_length==42,"SLDCH Bitmap_length %x not supported\n",
	      sldch->bitmap_length);

  int LPSDCH=0;
  for (int i=0;i<sldch->bitmap_length;i++) if (((((uint64_t)1)<<i)&sldch->bitmap1) >0) LPSDCH++;

  AssertFatal(LPSDCH>0,"LPSDCH is 0 (bitmap1 %lx, bitmap_length %d)\n",sldch->bitmap1,sldch->bitmap_length);
  AssertFatal(sldch->type == disc_type1 || sldch->type == disc_type2B,
	      "unknown Discovery type %d\n",sldch->type);

  int N_TX_SLD = 1+sldch->numRetx;
  uint32_t M_RB_PSDCH_RP = sldch->N_SL_RB;
  unsigned int Ni,Nf; 
  if (sldch->type == disc_type1) {
    Ni = LPSDCH/N_TX_SLD;
    Nf = M_RB_PSDCH_RP>>1;
    if (ue->is_SynchRef == 0) sldch->n_psdch = 2; //taus()%(Ni*Nf);
    int a_ji = ((sldch->j*(Nf/N_TX_SLD)) + (sldch->n_psdch/Ni))%Nf;
    
    int b_1i = sldch->n_psdch%Ni;

    if (absSF_modP != ((b_1i*N_TX_SLD)+sldch->j)) return;
    
    nprb = 2*a_ji;
    LOG_D(PHY,"Generating SLDCH in SFN.SF %d.%d (O %d, P %d, n_psdch %d, Nf %d, Ni %d, j %d, a_ji %d) \n",frame_tx,subframe_tx,O,P,sldch->n_psdch,Nf,Ni,sldch->j,a_ji);

  }
  else {
    AssertFatal(1==0,"Discovery Type 2B not supported yet\n");
  }


  if (nprb <(sldch->N_SL_RB>>1)) nprb+=sldch->prb_Start;
  else                           nprb+=(sldch->prb_End-(sldch->N_SL_RB>>1));

  if (D2D_en){ // Flag for D2D operation in emulation mode
	  generate_sldch(ue,sldch,frame_tx,subframe_tx);
  }
  else{ // D2D in full PHY (RF mode)
	  sldch_codingmodulation(ue,frame_tx,subframe_tx,nprb,rvidx);
  }
  ue->psdch_generated=1;
  sldch->j++;
  sldch->j&=3;
  if (sldch->j==0) ue->sldch_active=0;

}
