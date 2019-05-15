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

/*! \file PHY/LTE_TRANSPORT/slss.c
 * \brief Functions to Generate and Receive PSSCH
 * \author R. Knopp
 * \date 2017
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */
#include "PHY/defs_UE.h"
#include "pssch.h"
#include "PHY/LTE_UE_TRANSPORT/transport_proto_ue.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "SCHED_UE/sched_UE.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/LTE_ESTIMATION/lte_estimation.h"
#include "LAYER2/MAC/mac_proto.h"
#include "LAYER2/PDCP_v10.1.0/pdcp.h"
//#define PSSCH_DEBUG 1
//#define DEBUG_SCI_DECODING 1

extern int
multicast_link_write_sock(int groupP, char *dataP, uint32_t sizeP);
extern uint8_t D2D_en;
extern UE_MAC_INST *UE_mac_inst;

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

void dci_encoding(uint8_t *a,
                  uint8_t A,
                  uint16_t E,
                  uint8_t *e,
                  uint16_t rnti);

void generate_sl_grouphop(PHY_VARS_UE *ue)
{

  uint8_t ns;
  uint8_t reset=1;
  uint32_t x1, x2, s=0;
  uint32_t fss_pusch;
  uint32_t destid;

  for (int index=0;index<257;index++) {
    // This is from Section 5.5.1.3
    if (index > 0) { // PSSCH
      destid=index-1;
      fss_pusch = destid%30;
      
      x2 = destid/30;
#ifdef DEBUG_SLGROUPHOP
      printf("[PHY] SL GroupHop %d:",destid);
#endif
    }
    else { // PSBCH
      fss_pusch =(ue->frame_parms.Nid_SL/16)%30;
      
      x2 = ue->frame_parms.Nid_SL/30;

    }
    for (ns=0; ns<20; ns++) {
      if ((ns&3) == 0) {
	s = lte_gold_generic(&x1,&x2,reset);
	reset = 0;
      }
	
      ue->gh[index][ns] = (((uint8_t*)&s)[ns&3]+fss_pusch)%30;
    
      
#ifdef DEBUG_SLGROUPHOP
      printf("%d.",ue->gh[destid][ns]);
#endif
    }
    
#ifdef DEBUG_SLGROUPHOP
    printf("\n");
#endif
  }
} 
 
uint64_t sci_mapping(PHY_VARS_UE *ue) {
  SLSCH_t *slsch                = ue->slsch;
  
  AssertFatal(slsch->freq_hopping_flag<2,"freq_hop %d >= 2\n",slsch->freq_hopping_flag);
  uint64_t freq_hopping_flag     = (uint64_t)slsch->freq_hopping_flag;
  
  uint64_t RAbits                = log2_approx(slsch->N_SL_RB_data*((slsch->N_SL_RB_data+1)>>1));
  AssertFatal(slsch->resource_block_coding<(1<<RAbits),"slsch->resource_block_coding %x >= %x\n",slsch->resource_block_coding,(1<<RAbits));
  uint64_t resource_block_coding     = (uint64_t)slsch->resource_block_coding; 
  
  AssertFatal(slsch->time_resource_pattern<128,"slsch->time_resource_pattern %d>=128\n",slsch->time_resource_pattern);
  uint64_t time_resource_pattern     = (uint64_t)slsch->time_resource_pattern;
  
  AssertFatal(slsch->mcs<32,"slsch->mcs %d >= 32\n",slsch->mcs);
  uint64_t mcs                       = (uint64_t)slsch->mcs;
  
  AssertFatal(slsch->timing_advance_indication<2048,"slsch->timing_advance_indication %d >= 2048\n",slsch->timing_advance_indication);
  uint64_t timing_advance_indication = (uint64_t)slsch->timing_advance_indication;
  
  AssertFatal(slsch->group_destination_id<256,"slsch->group_destination_id %d >= 256\n",slsch->group_destination_id);
  uint64_t group_destination_id = (uint64_t)slsch->group_destination_id;
 
  LOG_D(PHY,"SCI : RAbits %llu\n",(long long unsigned int)RAbits); 
  // map bitfields
  // frequency-hopping 1-bit
  return( (freq_hopping_flag <<63) | 
	  (resource_block_coding << (63-1-RAbits+1)) | 
	  (time_resource_pattern<<(63-1-7-RAbits+1)) | 
	  (mcs<<(63-1-RAbits-7-5+1)) | 
	  (timing_advance_indication<<(63-1-RAbits-7-5-11+1)) | 
	  (group_destination_id<<(63-1-RAbits-7-5-11-8+1))
	  );
  
}

void dft_slcch(int32_t *z,int32_t *d, uint8_t Nsymb) {

#if defined(__x86_64__) || defined(__i386__)
  __m128i dft_in128[4][12],dft_out128[4][12];
#elif defined(__arm__)
  int16x8_t dft_in128[4][12],dft_out128[4][12];
#endif
  uint32_t *dft_in0=(uint32_t*)dft_in128[0],*dft_out0=(uint32_t*)dft_out128[0];
  uint32_t *dft_in1=(uint32_t*)dft_in128[1],*dft_out1=(uint32_t*)dft_out128[1];
  
  uint32_t *d0,*d1,*d2,*d3,*d4,*d5;
  
  uint32_t *z0,*z1,*z2,*z3,*z4,*z5;
  uint32_t i,ip;
#if defined(__x86_64__) || defined(__i386__)
  __m128i norm128;
#elif defined(__arm__)
  int16x8_t norm128;
#endif
  
  
  d0 = (uint32_t *)d;
  d1 = d0+12;
  d2 = d1+12;
  d3 = d2+12;
  d4 = d3+12;
  d5 = d4+12;

  //  printf("symbol 0 (d0 %p, d %p)\n",d0,d);
  for (i=0,ip=0; i<12; i++,ip+=4) {
    dft_in0[ip]   =  d0[i];
    dft_in0[ip+1] =  d1[i];
    dft_in0[ip+2] =  d2[i];
    dft_in0[ip+3] =  d3[i];
    dft_in1[ip]   =  d4[i];
    dft_in1[ip+1] =  d5[i];
    dft_in1[ip+2] = 0;
  }
  
  dft12((int16_t *)dft_in0,(int16_t *)dft_out0);
  dft12((int16_t *)dft_in1,(int16_t *)dft_out1);
  
#if defined(__x86_64__) || defined(__i386__)
  norm128 = _mm_set1_epi16(9459);
#elif defined(__arm__)
  norm128 = vdupq_n_s16(9459);
#endif
  for (i=0; i<12; i++) {
#if defined(__x86_64__) || defined(__i386__)
    ((__m128i*)dft_out0)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)dft_out0)[i],norm128),1);
    ((__m128i*)dft_out1)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)dft_out1)[i],norm128),1);
#elif defined(__arm__)
    ((int16x8_t*)dft_out0)[i] = vqdmulhq_s16(((int16x8_t*)dft_out0)[i],norm128);
    ((int16x8_t*)dft_out1)[i] = vqdmulhq_s16(((int16x8_t*)dft_out1)[i],norm128);
#endif
  }
  
  
  
  z0 = (uint32_t *)z;
  z1 = z0+12;
  z2 = z1+12;
  z3 = z2+12;
  z4 = z3+12;
  z5 = z4+12;
  
  
  //  printf("symbol0 (dft)\n");
  for (i=0,ip=0; i<12; i++,ip+=4) {
    z0[i]     = dft_out0[ip];
    z1[i]     = dft_out0[ip+1];
    z2[i]     = dft_out0[ip+2];
    z3[i]     = dft_out0[ip+3];
    z4[i]     = dft_out1[ip+0];
    z5[i]     = dft_out1[ip+1];

    
  }
  
  //  printf("\n");
}

extern short conjugate2[8];

void idft_slcch(LTE_DL_FRAME_PARMS *frame_parms,int32_t *z,int slot)
{
#if defined(__x86_64__) || defined(__i386__)
  __m128i idft_in128[3][12],idft_out128[3][12];
  __m128i norm128;
#elif defined(__arm__)
  int16x8_t idft_in128[3][12],idft_out128[3][12];
  int16x8_t norm128;
#endif
  int16_t *idft_in0=(int16_t*)idft_in128[0],*idft_out0=(int16_t*)idft_out128[0];
  int16_t *idft_in1=(int16_t*)idft_in128[1],*idft_out1=(int16_t*)idft_out128[1];

  int32_t *z0,*z1,*z2,*z3,*z4,*z5=NULL;
  int i,ip;

  //  printf("Doing lte_idft for Msc_PUSCH %d\n",Msc_PUSCH);

  if (frame_parms->Ncp == 0) { // Normal prefix
    z0 = z+slot*(frame_parms->N_RB_DL*12*7);
    z1 = z0+(frame_parms->N_RB_DL*12);
    z2 = z1+(frame_parms->N_RB_DL*12);
    //pilot
    z3 = z2+(2*frame_parms->N_RB_DL*12);
    z4 = z3+(frame_parms->N_RB_DL*12);
    z5 = z4+(frame_parms->N_RB_DL*12);
  } else { // extended prefix
    z0 = z+slot*(frame_parms->N_RB_DL*12*6);
    z1 = z0+(frame_parms->N_RB_DL*12);
    //pilot
    z2 = z1+(2*frame_parms->N_RB_DL*12);
    z3 = z2+(frame_parms->N_RB_DL*12);
    z4 = z3+(frame_parms->N_RB_DL*12);
  }

  // conjugate input
  for (i=0; i<3; i++) {
#if defined(__x86_64__)||defined(__i386__)
    *&(((__m128i*)z0)[i])=_mm_sign_epi16(*&(((__m128i*)z0)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z1)[i])=_mm_sign_epi16(*&(((__m128i*)z1)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z2)[i])=_mm_sign_epi16(*&(((__m128i*)z2)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z3)[i])=_mm_sign_epi16(*&(((__m128i*)z3)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z4)[i])=_mm_sign_epi16(*&(((__m128i*)z4)[i]),*(__m128i*)&conjugate2[0]);
    *&(((__m128i*)z5)[i])=_mm_sign_epi16(*&(((__m128i*)z5)[i]),*(__m128i*)&conjugate2[0]);
    
  
#elif defined(__arm__)
    *&(((int16x8_t*)z0)[i])=vmulq_s16(*&(((int16x8_t*)z0)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z1)[i])=vmulq_s16(*&(((int16x8_t*)z1)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z2)[i])=vmulq_s16(*&(((int16x8_t*)z2)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z3)[i])=vmulq_s16(*&(((int16x8_t*)z3)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z4)[i])=vmulq_s16(*&(((int16x8_t*)z4)[i]),*(int16x8_t*)&conjugate2[0]);
    if (frame_parms->Ncp==NORMAL)  
      *&(((int16x8_t*)z5)[i])=vmulq_s16(*&(((int16x8_t*)z5)[i]),*(int16x8_t*)&conjugate2[0]);
    
#endif
  }

  for (i=0,ip=0; i<12; i++,ip+=4) {
    ((uint32_t*)idft_in0)[ip+0] =  z0[i];
    ((uint32_t*)idft_in0)[ip+1] =  z1[i];
    ((uint32_t*)idft_in0)[ip+2] =  z2[i];
    ((uint32_t*)idft_in0)[ip+3] =  z3[i];
    ((uint32_t*)idft_in1)[ip+0] =  z4[i];

    if (frame_parms->Ncp==0) ((uint32_t*)idft_in1)[ip+1] =  z5[i];
  }


  
  dft12((int16_t *)idft_in0,(int16_t *)idft_out0);
  dft12((int16_t *)idft_in1,(int16_t *)idft_out1);

#if defined(__x86_64__)||defined(__i386__)
  norm128 = _mm_set1_epi16(9459);
#elif defined(__arm__)
  norm128 = vdupq_n_s16(9459);
#endif
  for (i=0; i<12; i++) {
#if defined(__x86_64__)||defined(__i386__)
    ((__m128i*)idft_out0)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out0)[i],norm128),1);
    ((__m128i*)idft_out1)[i] = _mm_slli_epi16(_mm_mulhi_epi16(((__m128i*)idft_out1)[i],norm128),1);
#elif defined(__arm__)
    ((int16x8_t*)idft_out0)[i] = vqdmulhq_s16(((int16x8_t*)idft_out0)[i],norm128);
    ((int16x8_t*)idft_out1)[i] = vqdmulhq_s16(((int16x8_t*)idft_out1)[i],norm128);
#endif
  }
  
  for (i=0,ip=0; i<12; i++,ip+=4) {
    z0[i]     = ((uint32_t*)idft_out0)[ip];
    /*
      printf("out0 (%d,%d),(%d,%d),(%d,%d),(%d,%d)\n",
      ((int16_t*)&idft_out0[ip])[0],((int16_t*)&idft_out0[ip])[1],
      ((int16_t*)&idft_out0[ip+1])[0],((int16_t*)&idft_out0[ip+1])[1],
      ((int16_t*)&idft_out0[ip+2])[0],((int16_t*)&idft_out0[ip+2])[1],
      ((int16_t*)&idft_out0[ip+3])[0],((int16_t*)&idft_out0[ip+3])[1]);
    */
    z1[i]     = ((uint32_t*)idft_out0)[ip+1];
    z2[i]     = ((uint32_t*)idft_out0)[ip+2];
    z3[i]     = ((uint32_t*)idft_out0)[ip+3];
    z4[i]     = ((uint32_t*)idft_out1)[ip+0];

    if (frame_parms->Ncp==0)     z5[i]     = ((uint32_t*)idft_out1)[ip+1];
  }

  // conjugate output
  for (i=0; i<3; i++) {
#if defined(__x86_64__) || defined(__i386__)
    ((__m128i*)z0)[i]=_mm_sign_epi16(((__m128i*)z0)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z1)[i]=_mm_sign_epi16(((__m128i*)z1)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z2)[i]=_mm_sign_epi16(((__m128i*)z2)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z3)[i]=_mm_sign_epi16(((__m128i*)z3)[i],*(__m128i*)&conjugate2[0]);
    ((__m128i*)z4)[i]=_mm_sign_epi16(((__m128i*)z4)[i],*(__m128i*)&conjugate2[0]);

    if (frame_parms->Ncp==NORMAL) 
      ((__m128i*)z5)[i]=_mm_sign_epi16(((__m128i*)z5)[i],*(__m128i*)&conjugate2[0]);
    
#elif defined(__arm__)
    *&(((int16x8_t*)z0)[i])=vmulq_s16(*&(((int16x8_t*)z0)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z1)[i])=vmulq_s16(*&(((int16x8_t*)z1)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z2)[i])=vmulq_s16(*&(((int16x8_t*)z2)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z3)[i])=vmulq_s16(*&(((int16x8_t*)z3)[i]),*(int16x8_t*)&conjugate2[0]);
    *&(((int16x8_t*)z4)[i])=vmulq_s16(*&(((int16x8_t*)z4)[i]),*(int16x8_t*)&conjugate2[0]);

    if (frame_parms->Ncp==NORMAL) *&(((int16x8_t*)z5)[i])=vmulq_s16(*&(((int16x8_t*)z5)[i]),*(int16x8_t*)&conjugate2[0]);
    

#endif
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif

}


void pscch_codingmodulation(PHY_VARS_UE *ue,int frame_tx,int subframe_tx,uint32_t a,int slot) {

  AssertFatal(ue!=NULL,"UE is null\n");
  AssertFatal(frame_tx>=0 && frame_tx<1024,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx>=0 && subframe_tx<10,"subframe %d is illegal\n",subframe_tx);
  AssertFatal(slot>=0 && slot<2, "slot %d is illegal\n",slot);
  LTE_UE_PSCCH_TX *pscch = ue->pscch_vars_tx;
  SLSCH_t *slsch         = ue->slsch;
  int tx_amp;
  int nprb;
  uint64_t sci;
  // Note this should depend on configuration of slsch/slcch
  uint32_t Nsymb = 7;
  int Nsymb2  = slot==0 ? Nsymb : Nsymb-1;
  uint32_t E = 12*(Nsymb-1)*2;


  // coding part
  if (ue->pscch_coded == 0) {

    LOG_D(PHY,"pscch_coding\n");
    sci = sci_mapping(ue);

    int length = log2_approx(slsch->N_SL_RB_data*((ue->slsch_rx.N_SL_RB_data+1)>>1))+32;

    LOG_D(PHY,"sci %lx (%d bits): freq_hopping_flag %d,resource_block_coding %d,time_resource_pattern %d,mcs %d,timing_advance_indication %d, group_destination_id %d\n",sci,length,
           slsch->freq_hopping_flag, 
           slsch->resource_block_coding,
           slsch->time_resource_pattern,
           slsch->mcs,
           slsch->timing_advance_indication,
           slsch->group_destination_id);

    //   for (int i=0;i<(length+7)/8;i++) printf("sci[%d] %x\n",i,((uint8_t *)&sci)[i]);
    uint8_t sci_flip[8];
    sci_flip[0] = ((uint8_t *)&sci)[7];
    sci_flip[1] = ((uint8_t *)&sci)[6];
    sci_flip[2] = ((uint8_t *)&sci)[5];
    sci_flip[3] = ((uint8_t *)&sci)[4];
    sci_flip[4] = ((uint8_t *)&sci)[3];
    sci_flip[5] = ((uint8_t *)&sci)[2];
    sci_flip[6] = ((uint8_t *)&sci)[1];
    sci_flip[7] = ((uint8_t *)&sci)[0];
    dci_encoding(sci_flip,
		 length,
		 E,
		 pscch->f,
		 0);

    // interleaving
    // Cmux assumes configuration 0
    int Cmux = Nsymb-1;
    uint8_t *fptr;
    for (int i=0,j=0; i<Cmux; i++)
      // 24 = 12*(Nsymb-1)*2/(Nsymb-1)
      for (int r=0; r<12; r++) {
        fptr=&pscch->f[((r*Cmux)+i)<<1];
        pscch->h[j++] = *fptr++;
        pscch->h[j++] = *fptr++;
      }

    // scrambling
    uint32_t x1,x2=510;
    
    uint32_t s = lte_gold_generic(&x1,&x2,1); 
    uint8_t c;

    for (int i=0,k=0;i<(1+(E>>5));i++) {
      for (int j=0;(j<32)&&(k<E);j++,k++) {
	c = (uint8_t)((s>>j)&1);
	pscch->b_tilde[k] = (pscch->h[k]+c)&1;
      }
      s = lte_gold_generic(&x1,&x2,0); 
    }		 
    ue->pscch_coded=1;
  }
  // convert a to prb number and compute slot

  // get index within slot (first half of the prbs in slot 0, second half in 1)
  uint32_t amod = a%(slsch->N_SL_RB_SC);

  if (amod<(slsch->N_SL_RB_SC>>1)) nprb = slsch->prb_Start_SC + amod;
  else                             nprb = slsch->prb_End_SC-(slsch->N_SL_RB_SC>>1)+amod;

  LOG_D(PHY,"%d.%d: nprb %d, slot %d\n",frame_tx,subframe_tx,nprb,slot);
  // Fill in power control later
  //  pssch_power_cntl(ue,proc,eNB_id,1, abstraction_flag);
  //  ue->tx_power_dBm[subframe_tx] = ue->slcch[eNB_id]->Po_PUSCH;
  ue->tx_power_dBm[subframe_tx] = 0;
  ue->tx_total_RE[subframe_tx] = 12;
#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[subframe_tx],
		      ue->tx_power_max_dBm,
		      ue->frame_parms.N_RB_UL,
		      1);
#else
  tx_amp = AMP;
#endif
  
  // modulation
  int Msymb = E/2;
  int32_t d[Msymb];
  int16_t gain_lin_QPSK = (int16_t)((tx_amp*ONE_OVER_SQRT2_Q15)>>15);

  LOG_D(PHY,"pscch modulation, Msymb %d\n",Msymb);
  for (int i=0,j=0; i<Msymb; i++,j+=2) {
    
    ((int16_t*)&d[i])[0] = (pscch->b_tilde[j] == 1)  ? (-gain_lin_QPSK) : gain_lin_QPSK;
    ((int16_t*)&d[i])[1] = (pscch->b_tilde[j+1] == 1)? (-gain_lin_QPSK) : gain_lin_QPSK;
  }

  // precoding
  int32_t z[Msymb];
  int re_offset,re_offset0,symbol_offset;
  LTE_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  dft_slcch(z,d,Nsymb);
  
  // RE mapping
  re_offset0 = frame_parms->first_carrier_offset + (nprb*12);

  if (re_offset0>frame_parms->ofdm_symbol_size) {
    re_offset0 -= frame_parms->ofdm_symbol_size;
  }

  AssertFatal(slot==0||slot==1,"Slot %d is illegal\n",slot);
  int loffset = slot==0 ? 0 : Nsymb;

  int32_t *txptr;

  if (ue->generate_ul_signal[subframe_tx][0] == 0){ 
    LOG_D(PHY,"%d.%d: clearing ul_signal\n",frame_tx,subframe_tx);
    for (int aa=0; aa<ue->frame_parms.nb_antennas_tx; aa++) {
      memset(&ue->common_vars.txdataF[aa][subframe_tx*ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti],
	     0,
	     ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti*sizeof(int32_t));
    }
  }

  for (int j=0,l=0; l<Nsymb2; l++) {
    re_offset = re_offset0;
    symbol_offset = (uint32_t)frame_parms->ofdm_symbol_size*(loffset+l+(subframe_tx*2*Nsymb));
    txptr = &ue->common_vars.txdataF[0][symbol_offset];

    if (((frame_parms->Ncp == 0) && ((l==3) || (l==10)))||
        ((frame_parms->Ncp == 1) && ((l==2) || (l==8)))) {
    }
    // Skip reference symbols
    else {
      for (int i=0; i<12; i++,j++) {
        txptr[re_offset++] = z[j];
        if (re_offset==frame_parms->ofdm_symbol_size) re_offset = 0;
      }
    }
  }

  ue->sl_chan = (slot==0) ? PSCCH_12_EVEN : PSCCH_12_ODD;
  // DMRS insertion
  for (int aa=0; aa<1/*frame_parms->nb_antennas_tx*/; aa++)
    generate_drs_pusch(ue,
                       NULL, // no proc means this is SLSCH/SLCCH
                       0,
                       tx_amp,
                       subframe_tx,
                       nprb,
                       1,
                       aa,
                       NULL,
                       0);

  ue->pscch_generated |= (1+slot);
  ue->generate_ul_signal[subframe_tx][0] = 1;
  LOG_D(PHY, "PSCCH signal generated \n");

}

void slsch_codingmodulation(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_tx,int subframe_tx,int ljmod10) {

  SLSCH_t *slsch               = ue->slsch;
  LTE_eNB_DLSCH_t *dlsch = ue->dlsch_slsch;
  LTE_UE_ULSCH_t *ulsch  = ue->ulsch_slsch;
  uint32_t Nsymb = 7;
  int tx_amp;
  
  AssertFatal(slsch!=NULL,"ue->slsch is null\n");
  
  AssertFatal(ue->slsch_sdu_active > 0,"ue->slsch_sdu_active isn't active\n");

  LOG_D(PHY,"Generating SLSCH for rvidx %d, group_id %d, mcs %d, resource first rb %d, L_crbs %d\n",
	slsch->rvidx,slsch->group_destination_id,slsch->mcs,slsch->RB_start+slsch->prb_Start_data,slsch->L_CRBs);

  


  dlsch->harq_processes[0]->nb_rb       = slsch->L_CRBs;
  dlsch->harq_processes[0]->TBS         = get_TBS_UL(slsch->mcs,slsch->L_CRBs)<<3;
  dlsch->harq_processes[0]->Qm          = get_Qm_ul(slsch->mcs);
  dlsch->harq_processes[0]->mimo_mode   = SISO;
  dlsch->harq_processes[0]->rb_alloc[0] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[1] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[2] = 0; // unused for SL
  dlsch->harq_processes[0]->rb_alloc[3] = 0; // unused for SL
  dlsch->harq_processes[0]->Nl          = 1;
  dlsch->harq_processes[0]->round       = (slsch->rvidx == 0) ? 0 : (dlsch->harq_processes[0]->round+1); 
  dlsch->harq_processes[0]->rvidx       = slsch->rvidx;

  //  int E = dlsch->harq_processes[0]->Qm * 12 * slsch->L_CRBs * ((Nsymb-1)<<1);
  if (slsch->rvidx == 0) ue->slsch_txcnt++;

  dlsch_encoding0(&ue->frame_parms,
		 slsch->payload,
		 0, // means SL
		 dlsch,
		 frame_tx,
		 subframe_tx,
		 &ue->ulsch_rate_matching_stats,
		 &ue->ulsch_turbo_encoding_stats,
		 &ue->ulsch_interleaving_stats);

  //  for (int i=0;i<2*12*slsch->L_CRBs*((Nsymb-1)<<1)/16;i++) printf("encoding: E[%d] %d\n",i,dlsch->harq_processes[0]->e[i]);
  // interleaving
  // Cmux assumes configuration 0
  int Cmux = (Nsymb-1)<<1;
  uint8_t *eptr;
  for (int i=0,j=0; i<Cmux; i++)
    // 24 = 12*(Nsymb-1)*2/(Nsymb-1)
    for (int r=0; r<12*slsch->L_CRBs; r++) {
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
  uint32_t cinit=510+(((uint32_t)slsch->group_destination_id)<<14)+(ljmod10<<9);
 
  LOG_D(PHY,"SLSCH cinit %x (%d,%d)\n",cinit,slsch->group_destination_id,ljmod10); 
  ulsch->harq_processes[0]->nb_rb       = slsch->L_CRBs;
  ulsch->harq_processes[0]->first_rb    = slsch->RB_start + slsch->prb_Start_data;
  ulsch->harq_processes[0]->mcs         = slsch->mcs;
  ulsch->Nsymb_pusch                    = ((Nsymb-1)<<1);

  LOG_D(PHY,"%d.%d : SLSCH nbrb %d, first rb %d\n",frame_tx,subframe_tx,slsch->L_CRBs,slsch->RB_start+slsch->prb_Start_data);

  ue->sl_chan = PSSCH_12;

  // Fill in power control later
  //  pssch_power_cntl(ue,proc,eNB_id,1, abstraction_flag);
  //  ue->tx_power_dBm[subframe_tx] = ue->slcch[eNB_id]->Po_PUSCH;
  ue->tx_power_dBm[subframe_tx] = 0;
  ue->tx_total_RE[subframe_tx] = slsch->L_CRBs*12;
#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[subframe_tx],
		      ue->tx_power_max_dBm,
		      ue->frame_parms.N_RB_UL,
		      1);
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
		     slsch->RB_start+slsch->prb_Start_data,
		     slsch->L_CRBs,
                     0,
                     ue->gh[1+slsch->group_destination_id],
                     ljmod10);

  ue->pssch_generated = 1;
  ue->generate_ul_signal[subframe_tx][0] = 1;
  LOG_D(PHY, "PSSCH signal generated \n");
}

void check_and_generate_pssch(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_tx,int subframe_tx) {


  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);
  SLSCH_t *slsch = ue->slsch;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = ue->slsch->SL_OffsetIndicator_data;
  uint32_t P = ue->slsch->SL_SC_Period;
  uint32_t absSF = (frame_tx*10)+subframe_tx;
  uint32_t absSF_offset,absSF_modP;


  absSF_offset = absSF-O;

  LOG_D(PHY,"Checking pssch for absSF %d (slsch_active %d)\n",
	absSF, ue->slsch_active);
  
  if (ue->slsch_active == 0) return;

  if (absSF < O) return;

  absSF_modP = absSF_offset%P;

  LOG_D(PHY,"Checking pssch for absSF_mod P %d, SubframeBitmapSL_length %d\n", absSF_modP, slsch->SubframeBitmapSL_length);
  // This is the condition for short SCCH bitmap (slsch->SubframeBitmapSL_length bits), check that the current subframe is for SLSCH
  if (absSF_modP < slsch->SubframeBitmapSL_length) return;
 
  if (absSF_modP == slsch->SubframeBitmapSL_length) {
    ue->pscch_coded =0;
    ue->pscch_generated=0;
  }
 
  absSF_modP-=slsch->SubframeBitmapSL_length;


  AssertFatal(slsch->time_resource_pattern <= TRP8_MAX,
	      "received Itrp %d: TRP8 is used with Itrp in 0...%d\n",
	      slsch->time_resource_pattern,TRP8_MAX);

  LOG_D(PHY,"Checking pssch for absSF %d (trp mask %d, rv %d)\n",
	absSF, trp8[slsch->time_resource_pattern][absSF_modP&7],
	slsch->rvidx);
  // Note : this assumes Ntrp=8 for now
  if (trp8[slsch->time_resource_pattern][absSF_modP&7]==0) return;
/*
  // we have an opportunity in this subframe
  if (absSF_modP == 10) slsch->ljmod10 = 0;
  else slsch->ljmod10++;
*/	 

  if(D2D_en){
	  generate_slsch(ue,proc,slsch,frame_tx,subframe_tx);
  }
  else{
	  slsch_codingmodulation(ue,proc,frame_tx,subframe_tx,slsch->ljmod10);
  }
}

void check_and_generate_pscch(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {
  
  AssertFatal(frame_tx<1024 && frame_tx>=0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>=0,"subframe %d is illegal\n",subframe_tx);
  SLSCH_t *slsch = ue->slsch;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = ue->slsch->SL_OffsetIndicator;
  uint32_t P = ue->slsch->SL_SC_Period;
  uint32_t absSF = (frame_tx*10)+subframe_tx;
  uint32_t absSF_offset,absSF_modP;

  LOG_D(PHY,"Checking pscch for absSF %d (pscch_generate = %d)\n",absSF,ue->pscch_generated);


  
  absSF_offset = absSF-O;

  if (absSF < O) return;

  absSF_modP = absSF_offset%P;

  // This is the condition for short SCCH bitmap (slsch->SubframeBitmapSL_length bits), check that the current subframe is for SCCH
  if (absSF_modP >= slsch->SubframeBitmapSL_length) { 
    ue->pscch_coded =0; 
    ue->pscch_generated=0;
    return;
  }
  LOG_D(PHY,"Checking pscch for absSF_modP %d (SubframeBitmalSL_length %d,mask %llx)\n",absSF_modP,slsch->SubframeBitmapSL_length,(long long unsigned int)slsch->bitmap1);

  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & slsch->bitmap1) == 0) return;

  // if we get here, then there is a PSCCH subframe for a potential transmission
  uint32_t sf_index=slsch->SubframeBitmapSL_length,LPSCCH=0;
  for (int i=0;i<slsch->SubframeBitmapSL_length;i++) {
    if (i==absSF_modP) sf_index=LPSCCH;
    if (((((uint64_t)1)<<i) & slsch->bitmap1)>0) LPSCCH++;
  }
  AssertFatal(sf_index<slsch->SubframeBitmapSL_length,"sf_index not set, should not happen (absSF_modP %d)\n",absSF_modP);

  // sf_index now contains the SF index in 0...LPSCCH-1
  // LPSCCH has the number of PSCCH subframes

 
  // number of resources blocks per slot times 2 slots
  uint32_t M_RB_PSCCH_RP = slsch->N_SL_RB_SC;
  AssertFatal(slsch->n_pscch < (M_RB_PSCCH_RP>>1)*LPSCCH,"n_pscch not in 0..%d\n",
	      ((M_RB_PSCCH_RP>>1)*LPSCCH)-1);
  // hard-coded to transmission mode one for now (Section 14.2.1.1 from 36.213 Rel14.3)
  uint32_t a1=slsch->n_pscch/LPSCCH;
  uint32_t a2=a1+(M_RB_PSCCH_RP>>1);
  uint32_t b1=slsch->n_pscch%LPSCCH;
  uint32_t b2=(slsch->n_pscch + 1 + (a1%(LPSCCH-1)))%LPSCCH;


  LOG_D(PHY,"Checking pscch for absSF %d / n_pscch %d (N_SL_RB_SC %d, LPSCCH %d, M_RB_PSCCH_RP %d, a1 %d, a2 %d, b1 %d, b2 %d) pscch_coded %d\n",
	absSF, slsch->n_pscch,slsch->N_SL_RB_SC,LPSCCH, M_RB_PSCCH_RP,a1,a2,b1,b2,ue->pscch_coded);

  ue->slsch_sdu_active = 1;

  if(!D2D_en) {
	  if (absSF_modP == b1)      pscch_codingmodulation(ue,frame_tx,subframe_tx,a1,0);
	  else if (absSF_modP == b2) pscch_codingmodulation(ue,frame_tx,subframe_tx,a2,1);
	  else return;
  }

}

void generate_slsch(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc, SLSCH_t *slsch,int frame_tx,int subframe_tx) {
    
  UE_tport_t pdu;
  size_t slsch_header_len = sizeof(UE_tport_header_t);


  if (ue->sidelink_l2_emulation == 1) {
    if (slsch->rvidx==0) {
      pdu.header.packet_type = SLSCH;
      pdu.header.absSF = (frame_tx*10)+subframe_tx;
      
      memcpy((void*)&pdu.slsch,(void*)slsch,sizeof(SLSCH_t)-sizeof(uint8_t*));
      
      AssertFatal(slsch->payload_length <=1500-slsch_header_len - sizeof(SLSCH_t) + sizeof(uint8_t*),
		  "SLSCH payload length > %lu\n",
		  1500-slsch_header_len - sizeof(SLSCH_t) + sizeof(uint8_t*));
      memcpy((void*)&pdu.payload[0],
	     (void*)slsch->payload,
	     slsch->payload_length);
      
      LOG_D(PHY,"SLSCH configuration %lu bytes, TBS payload %d bytes => %lu bytes\n",
	    sizeof(SLSCH_t)-sizeof(uint8_t*),
	    slsch->payload_length,
	    slsch_header_len+sizeof(SLSCH_t)-sizeof(uint8_t*)+slsch->payload_length);
      
      multicast_link_write_sock(0, 
				(char *)&pdu, 
				slsch_header_len+sizeof(SLSCH_t)-sizeof(uint8_t*)+slsch->payload_length);
  
    }
  } // sidelink_emulation=1
  //Panos: Remove this part as generate_slsch() will be called only for emulation mode now.
  /*else if (ue->sidelink_active==0){ // This is first indication of sidelink in this period
    ue->sidelink_active = 1;
    ue->slsch           = slsch;
  }
  // check and flll SCI portion
  LOG_D(PHY,"pscch: SFN.SF %d.%d\n",frame_tx,subframe_tx); 
  check_and_generate_pscch(ue,frame_tx,subframe_tx);
  // check and flll SLSCH portion
  LOG_D(PHY,"pssch: SFN.SF %d.%d\n",frame_tx,subframe_tx); 
  check_and_generate_pssch(ue,proc,frame_tx,subframe_tx);*/
}


void pscch_decoding(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_rx,int subframe_rx,int a,int slot) {

  int Nsymb = 7 - slot;
  SLSCH_t *slsch = &ue->slsch_rx;

  uint32_t amod = a%(slsch->N_SL_RB_SC);
  int16_t **rxdataF_ext      = (int16_t**)ue->pusch_slcch->rxdataF_ext;
  int16_t **drs_ch_estimates = (int16_t**)ue->pusch_slcch->drs_ch_estimates;
  int16_t **rxdataF_comp     = (int16_t**)ue->pusch_slcch->rxdataF_comp;
  int16_t **ul_ch_mag        = (int16_t**)ue->pusch_slcch->ul_ch_mag;
  int16_t **rxdata_7_5kHz    = ue->sl_rxdata_7_5kHz[ue->current_thread_id[subframe_rx]];
  int16_t **rxdataF          = ue->sl_rxdataF[ue->current_thread_id[subframe_rx]];
  int32_t avgs;
  uint8_t log2_maxh=0;
  int32_t avgU[2];
  int nprb;

  AssertFatal(slot==0 || slot==1, "slot %d is illegal\n",slot);


			   
  if (amod<(slsch->N_SL_RB_SC>>1)) nprb = slsch->prb_Start_SC + amod;
  else                             nprb = slsch->prb_End_SC-(slsch->N_SL_RB_SC>>1)+amod;

  if (frame_rx < 100) LOG_D(PHY,"%d.%d: Running pscch decoding slot %d, nprb %d, a %d, amod %d,N_SL_RB_SC %d\n",frame_rx,subframe_rx,slot,nprb,a,amod,slsch->N_SL_RB_SC); 
  // slot FEP
  int SLaoffset=0;
  if (ue->SLonly==0) SLaoffset=1;
  if (proc->sl_fep_done == 0) {
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

#ifdef PSCCH_DEBUG
    write_output("rxsig0_input.m","rxs0_in",&ue->common_vars.rxdata[0][((subframe_rx<<1)+slot)*ue->frame_parms.samples_per_tti>>1],ue->frame_parms.samples_per_tti>>1,1,1);
    write_output("rxsig0_7_5kHz.m","rxs0_7_5kHz",rxdata_7_5kHz[0],ue->frame_parms.samples_per_tti,1,1);
#endif
    for (int l=0; l<Nsymb; l++) {
      slot_fep_ul(&ru_tmp,l,(subframe_rx<<1),0);
      slot_fep_ul(&ru_tmp,l,(subframe_rx<<1)+1,0);
    }
    proc->sl_fep_done = 1;
  }
  // extract symbols from slot  
  for (int l=0; l<Nsymb; l++) {
    ulsch_extract_rbs_single((int32_t **)rxdataF,
			     (int32_t **)rxdataF_ext,
			     nprb,
			     1,
			     l,
			     (subframe_rx<<1)+slot,
			     &ue->frame_parms);
  }
#ifdef PSCCH_DEBUG
  write_output("slcch_rxF.m",
	       "slcchrxF",
	       &rxdataF[0][0],
	       14*ue->frame_parms.ofdm_symbol_size,1,1);
  write_output("slcch_rxF_ext.m","slcchrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
#endif

  // channel estimation
  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    1,
			    frame_rx,
			    subframe_rx,
			    0,
			    0,
			    0,
			    (slot==0) ? 3 : 10,
			    0, // interpolation
			    0);
#ifdef PSCCH_DEBUG
  write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif

  ulsch_channel_level((int32_t**)drs_ch_estimates,
		      &ue->frame_parms,
		      avgU,
		      1,slot*7);
  avgs = 0;
  
  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++)
    avgs = cmax(avgs,avgU[aarx]);
  
  //      log2_maxh = 4+(log2_approx(avgs)/2);
  
  log2_maxh = (log2_approx(avgs)/2)+ log2_approx(ue->frame_parms.nb_antennas_rx-1)+4;

  if (log2_maxh > 5) LOG_D(PHY,"%d.%d: nprb %d slot %d, pscch log2_maxh %d (avgs %d, slot energy %d dB)\n",frame_rx,subframe_rx,nprb,slot,log2_maxh, avgs,
                  dB_fixed(signal_energy(&ue->common_vars.rxdata[0][((subframe_rx<<1)+slot)*(ue->frame_parms.samples_per_tti>>1)],ue->frame_parms.samples_per_tti>>1)));


  for (int l=0; l<(ue->frame_parms.symbols_per_tti>>1)-slot; l++) {
  
    int l2 = l + slot*(ue->frame_parms.symbols_per_tti>>1);
    if (((ue->frame_parms.Ncp == 0) && ((l2==3) || (l2==10)))||   // skip pilots
	((ue->frame_parms.Ncp == 1) && ((l2==2) || (l2==8)))) {
      l2++;l++;
    }

    //    printf("Doing SLCCH reception for symbol %d\n",l2);

    ulsch_channel_compensation((int32_t**)rxdataF_ext,
			       (int32_t**)drs_ch_estimates,
			       (int32_t**)ul_ch_mag,
			       NULL,
			       (int32_t**)rxdataF_comp,
			       &ue->frame_parms,
			       l2,
			       2,
			       1,
			       log2_maxh); // log2_maxh+I0_shift
    
    
    
    
    if (ue->frame_parms.nb_antennas_rx > 1)
      ulsch_detection_mrc(&ue->frame_parms,
			  (int32_t**)rxdataF_comp,
			  (int32_t**)ul_ch_mag,
			  NULL,
			  l2,
			  1);
    
    
    
    
    //    if (23<ulsch_power[0]) {
    freq_equalization(&ue->frame_parms,
		      (int32_t**)rxdataF_comp,
		      (int32_t**)ul_ch_mag,
		      NULL,
		      l2,
		      12,
		      2);
    
  }
	    
  idft_slcch(&ue->frame_parms,
	     (int32_t*)rxdataF_comp[0],
	     slot);
	    
#ifdef PSCCH_DEBUG
  write_output("slcch_rxF_comp.m","slcchrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif


  int E=144;
  int16_t llr[E] __attribute__((aligned(32)));
  int16_t *llrp = (int16_t*)&llr[0];
  memset((void*)llr,0,E*sizeof(int16_t));
  for (int l=0; l<(ue->frame_parms.symbols_per_tti>>1)-slot; l++) {

    int l2 = l + slot*(ue->frame_parms.symbols_per_tti>>1);
    if (((ue->frame_parms.Ncp == 0) && ((l2==3) || (l2==10)))||   // skip pilots
	((ue->frame_parms.Ncp == 1) && ((l2==2) || (l2==8)))) {
      l2++;l++;
    }
    //    printf("Running ulsch_qpsk_llr for symbol %d\n",l2);
    ulsch_qpsk_llr(&ue->frame_parms,
		   (int32_t**)rxdataF_comp,
                   llr,
		   l2,
		   1,
		   &llrp);
  }
  /*
  write_output("slcch_llr.m","slcchllr",llr,
               12*2*(ue->frame_parms.symbols_per_tti>>1),
               1,0);
  */
  // unscrambling
  uint32_t x1,x2=510;
  
  uint32_t s = lte_gold_generic(&x1,&x2,1); 
  
  for (int i=0,k=0;i<(1+(E>>5));i++) {
    //  printf("s[%d] %x\n",i,s);
    for (int j=0;(j<32)&&(k<E);j++,k++) {
      //      printf("llr[%d]=%d c %d => %d\n",k,llr[k],(int16_t)((((s>>j)&1)<<1)-1),(int16_t)((((s>>j)&1)<<1)-1) * llr[k]);
      llr[k] = (int16_t)((((s>>j)&1)<<1)-1) * llr[k];
    }
    s = lte_gold_generic(&x1,&x2,0); 
  }
  // deinterleaving
  int8_t f[144];
  int Cmux = (ue->frame_parms.symbols_per_tti>>1)-1;
  for (int i=0,j=0;i<Cmux;i++) {
    // 12 = 12*(Nsymb-1)/(Nsymb-1)
//    printf("******* i %d\n",i);
    for (int r=0;r<12;r++) {
      f[((r*Cmux)+i)<<1]     = (int8_t)(llr[j++]>>4);
      f[(((r*Cmux)+i)<<1)+1]     = (int8_t)(llr[j++]>>4);
//      printf("f[%d] %d(%d) f[%d] %d(%d)\n",
//	     ((r*Cmux)+i)<<1,f[((r*Cmux)+i)<<1],j-2,(((r*Cmux)+i)<<1)+1,f[(((r*Cmux)+i)<<1)+1],j-1);
    }
  }
  uint16_t res;
  uint64_t sci_rx=0,sci_rx_flip=0;
  //decoding
  int length = log2_approx(slsch->N_SL_RB_data*(slsch->N_SL_RB_data+1)>>1)+32;
  
  //Panos: Modification here to comply with the new definition of dci_decoding()
  dci_decoding(length,1,f,(uint8_t*)&sci_rx);
  //dci_decoding(length,E,f,(uint8_t*)&sci_rx);
  ((uint8_t *)&sci_rx_flip)[0] = ((uint8_t *)&sci_rx)[7];
  ((uint8_t *)&sci_rx_flip)[1] = ((uint8_t *)&sci_rx)[6];
  ((uint8_t *)&sci_rx_flip)[2] = ((uint8_t *)&sci_rx)[5];
  ((uint8_t *)&sci_rx_flip)[3] = ((uint8_t *)&sci_rx)[4];
  ((uint8_t *)&sci_rx_flip)[4] = ((uint8_t *)&sci_rx)[3];
  ((uint8_t *)&sci_rx_flip)[5] = ((uint8_t *)&sci_rx)[2];
  ((uint8_t *)&sci_rx_flip)[6] = ((uint8_t *)&sci_rx)[1];
  ((uint8_t *)&sci_rx_flip)[7] = ((uint8_t *)&sci_rx)[0];
  //  for (int i=0;i<((length+7)/8);i++) printf("sci_rx[%d] %x\n",i,((uint8_t *)&sci_rx_flip)[i]);
  res = (crc16((uint8_t*)&sci_rx,length)>>16) ^ extract_crc((uint8_t*)&sci_rx,length);

  // extract SCI bit fields
  int RAbits = length-32;

  if (res==0) {
    slsch->freq_hopping_flag         = (sci_rx_flip>>63)&1;
    slsch->resource_block_coding     = (sci_rx_flip>>(63-1-RAbits+1))&((1<<RAbits)-1);
    RIV2_alloc(slsch->N_SL_RB_data,
               slsch->resource_block_coding,
	       (int *)&slsch->L_CRBs,(int *)&slsch->RB_start);
    slsch->time_resource_pattern     = (sci_rx_flip>>(63-1-7-RAbits+1))&127;
    slsch->mcs                       = (sci_rx_flip>>(63-1-7-5-RAbits+1))&31;
    slsch->timing_advance_indication = (sci_rx_flip>>(63-1-7-5-11-RAbits+1))&2047;
    slsch->group_destination_id      = (sci_rx_flip>>(63-1-7-5-11-8-RAbits+1))&255;

    uint8_t group_id_found = 0;
    for (int j = 0; j< MAX_NUM_LCID; j++){
    	//PC5-S (default RX)
    	if (UE_mac_inst[ue->Mod_id].sl_info[j].groupL2Id == slsch->group_destination_id) {
    		group_id_found = 1;
    		break;
    	}
    }
    //if(slsch->group_destination_id == UE_mac_inst[ue->Mod_id].groupL2Id || slsch->group_destination_id == UE_mac_inst[ue->Mod_id].sourceL2Id)
    if(slsch->mcs<=20 && slsch->freq_hopping_flag==0 && (group_id_found|| slsch->group_destination_id == UE_mac_inst[ue->Mod_id].sourceL2Id)) {
    	ue->slcch_received                     = 1;
	ue->slsch_rx_sdu_active=1;
    }
    else
    	ue->slcch_received                     = 0;
    
    ue->slsch_decoded                      = 0;
#ifdef DEBUG_SCI_DECODING
    printf("%d.%d sci %lx (%d bits,RAbits %d) : freq_hop %d, resource_block_coding %d, time_resource_pattern %d, mcs %d, timing_advance_indication %d, group_destination_id %d (gid shift %d result %lx => %lx\n",
       frame_rx,subframe_rx,
       sci_rx_flip,length,RAbits,
	   slsch->freq_hopping_flag,
	   slsch->resource_block_coding,
	   slsch->time_resource_pattern,
	   slsch->mcs,
	   slsch->timing_advance_indication,
    slsch->group_destination_id,
    63-1-7-5-11-8-RAbits+1,
    (sci_rx_flip>>(63-1-7-5-11-8-RAbits+1)),
    (sci_rx_flip>>(63-1-7-5-11-8-RAbits+1))&255
    );
#endif
    // check group_id here (not done yet)
    /*
    write_output("rxsig0_input.m","rxs0_in",&ue->common_vars.rxdata[0][((subframe_rx<<1)+slot)*ue->frame_parms.samples_per_tti>>1],ue->frame_parms.samples_per_tti>>1,1,1);
    write_output("rxsig0_7_5kHz.m","rxs0_7_5kHz",rxdata_7_5kHz[0],ue->frame_parms.samples_per_tti,1,1);
    write_output("slcch_rxF.m",
		 "slcchrxF",
		 &rxdataF[0][0],
		 14*ue->frame_parms.ofdm_symbol_size,1,1);
    write_output("slcch_rxF_ext.m","slcchrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
    write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
    write_output("slcch_rxF_comp.m","slcchrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
    write_output("slcch_llr.m","slcchllr",llr,
		 12*2*(ue->frame_parms.symbols_per_tti>>1),
		 1,0);    

		 exit(-1);
		 */
  }
  else {
    /*
    write_output("rxsig0_input.m","rxs0_in",&ue->common_vars.rxdata[0][((subframe_rx<<1)+slot)*ue->frame_parms.samples_per_tti>>1],ue->frame_parms.samples_per_tti>>1,1,1);
    write_output("rxsig0_7_5kHz.m","rxs0_7_5kHz",rxdata_7_5kHz[0],ue->frame_parms.samples_per_tti,1,1);
    write_output("slcch_rxF.m",
		 "slcchrxF",
		 &rxdataF[0][0],
		 14*ue->frame_parms.ofdm_symbol_size,1,1);
    write_output("slcch_rxF_ext.m","slcchrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
    write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
    write_output("slcch_rxF_comp.m","slcchrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
    write_output("slcch_llr.m","slcchllr",llr,
		 12*2*(ue->frame_parms.symbols_per_tti>>1),
		 1,0);    

		 exit(-1);*/
  }



}	

void rx_slcch(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_rx,int subframe_rx) {

  AssertFatal(frame_rx<1024 && frame_rx>=0,"frame %d is illegal\n",frame_rx);
  AssertFatal(subframe_rx<10 && subframe_rx>=0,"subframe %d is illegal\n",subframe_rx);
  SLSCH_t *slsch = &ue->slsch_rx;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = slsch->SL_OffsetIndicator;
  uint32_t P = slsch->SL_SC_Period;
  uint32_t absSF = (frame_rx*10)+subframe_rx;
  uint32_t absSF_offset,absSF_modP;



  absSF_offset = absSF-O;

  if (absSF < O) return;

  absSF_modP = absSF_offset%P;

  if (absSF_modP == 0) {
     ue->slcch_received=0;
     ue->slsch_rx_sdu_active = 0;
  }

  // This is the condition for short SCCH bitmap (slsch->SubframeBitmapSL_length bits), check that the current subframe is for SCCH

  if (ue->slcch_received == 1) return;

  if (absSF_modP >= slsch->SubframeBitmapSL_length) return;
  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & slsch->bitmap1) == 0) return;

  // if we get here, then there is a PSCCH subframe for a potential transmission
  uint32_t sf_index=slsch->SubframeBitmapSL_length,LPSCCH=0;
  for (int i=0;i<slsch->SubframeBitmapSL_length;i++) {
    if (i==absSF_modP) sf_index=LPSCCH;
    if (((((uint64_t)1)<<i) & slsch->bitmap1)>0) LPSCCH++;
  }
  AssertFatal(sf_index<slsch->SubframeBitmapSL_length,"sf_index not set, should not happen\n");

  // sf_index now contains the SF index in 0...LPSCCH-1
  // LPSCCH has the number of PSCCH subframes

  uint32_t M_RB_PSCCH_RP = slsch->N_SL_RB_SC;


  //AssertFatal(slsch->n_pscch < (M_RB_PSCCH_RP>>1)*LPSCCH,"n_pscch not in 0..%d\n",
  //	      ((M_RB_PSCCH_RP>>1)*LPSCCH)-1);

  for (int n_pscch = 0; n_pscch <  (M_RB_PSCCH_RP>>1)*LPSCCH ; n_pscch++) {
    // hard-coded to transmission mode one for now (Section 14.2.1.1 from 36.213 Rel14.3)
    uint32_t a1=n_pscch/LPSCCH;
    uint32_t a2=a1+(M_RB_PSCCH_RP>>1);
    uint32_t b1=n_pscch%LPSCCH;
    uint32_t b2=(n_pscch + 1 + (a1%(LPSCCH-1)))%LPSCCH;
  
    if (frame_rx < 100) LOG_D(PHY,"%d.%d: Checking n_pscch %d => a1 %d, a2 %d, b1 %d, b2 %d (LPSCCH %d, M_RB_PSCCH_RP %d)\n",
                                frame_rx,subframe_rx,n_pscch,a1,a2,b1,b2,LPSCCH,M_RB_PSCCH_RP); 
    if (absSF_modP == b1) {
    	LOG_D(PHY, "About to decode SCI at b1: %d \n \n \n", b1);
    	pscch_decoding(ue,proc,frame_rx,subframe_rx,a1,0);
    }
    else if (absSF_modP == b2){
    	LOG_D(PHY, "About to decode SCI at b2: %d \n \n \n", b2);
    	pscch_decoding(ue,proc,frame_rx,subframe_rx,a2,1);
    }
    else continue;
  }

}

void slsch_decoding(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc,int frame_rx,int subframe_rx,int ljmod10) {

  int Nsymb = 7;
  SLSCH_t *slsch = &ue->slsch_rx;
  int16_t **rxdataF_ext      = (int16_t**)ue->pusch_slsch->rxdataF_ext;
  int16_t **drs_ch_estimates = (int16_t**)ue->pusch_slsch->drs_ch_estimates;
  int16_t **rxdataF_comp     = (int16_t**)ue->pusch_slsch->rxdataF_comp;
  int16_t **ul_ch_mag        = (int16_t**)ue->pusch_slsch->ul_ch_mag;
  int16_t **rxdata_7_5kHz    = ue->sl_rxdata_7_5kHz[ue->current_thread_id[subframe_rx]];
  int16_t **rxdataF          = ue->sl_rxdataF[ue->current_thread_id[subframe_rx]];
  int32_t avgs;
  uint8_t log2_maxh=0;
  int32_t avgU[2];


  LOG_D(PHY,"slsch_decoding %d.%d => lmod10 %d\n",frame_rx,subframe_rx,ljmod10);

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

    for (int l=0; l<Nsymb; l++) {
      slot_fep_ul(&ru_tmp,l,(subframe_rx<<1),0);
      if (l<Nsymb-1)  // skip last symbol in second slot
	slot_fep_ul(&ru_tmp,l,(subframe_rx<<1)+1,0);
    }
    LOG_D(PHY,"SLSCH Slot FEP %d.%d\n",frame_rx,subframe_rx);
  }
  LOG_D(PHY,"SLSCH RBstart %d, L_CRBs %d\n",slsch->RB_start+slsch->prb_Start_data,slsch->L_CRBs);
  // extract symbols from slot 
   for (int l=0; l<Nsymb; l++) {
    ulsch_extract_rbs_single((int32_t**)rxdataF,
			     (int32_t**)rxdataF_ext,
			     slsch->RB_start+slsch->prb_Start_data,
			     slsch->L_CRBs,
			     l,
			     (subframe_rx<<1),
			     &ue->frame_parms);

    if (l<Nsymb-1) { // skip last symbol in second slot
      ulsch_extract_rbs_single((int32_t**)rxdataF,
			       (int32_t**)rxdataF_ext,
			       slsch->RB_start+slsch->prb_Start_data,
			       slsch->L_CRBs,
			       l,
			       (subframe_rx<<1)+1,
			       &ue->frame_parms);
    }
  }

#ifdef PSSCH_DEBUG
  write_output("slsch_rxF.m",
	       "slschrxF",
	       &rxdataF[0][0],
	       14*ue->frame_parms.ofdm_symbol_size,1,1);
  write_output("slsch_rxF_ext.m","slschrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
#endif

  AssertFatal(slsch->group_destination_id < 256,"Illegal group_destination_id %d\n",ue->slsch->group_destination_id);
  
  uint32_t u = ue->gh[1+slsch->group_destination_id][ljmod10<<1];
  uint32_t v = 0;
  uint32_t cyclic_shift=(slsch->group_destination_id>>1)&7;

  LOG_D(PHY,"SLSCH, u0 %d, cyclic_shift %d (ljmod10 %d)\n",u,cyclic_shift,ljmod10);
  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    slsch->L_CRBs,
			    frame_rx,
			    subframe_rx,
			    u,
			    v,
			    cyclic_shift,
			    3,
			    0, // interpolation
			    0);
  u = ue->gh[1+slsch->group_destination_id][1+(ljmod10<<1)];
  LOG_D(PHY,"SLSCH, u1 %d\n",u);

  lte_ul_channel_estimation(&ue->frame_parms,
			    (int32_t**)drs_ch_estimates,
			    (int32_t**)NULL,
			    (int32_t**)rxdataF_ext,
			    slsch->L_CRBs,
			    frame_rx,
			    subframe_rx,
			    u,
			    v,
			    cyclic_shift,
			    10,
			    0, // interpolation
			    0);

  ulsch_channel_level((int32_t**)drs_ch_estimates,
		      &ue->frame_parms,
		      avgU,
		      slsch->L_CRBs,0);
 
#ifdef PSSCH_DEBUG
  write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif

  avgs = 0;
  
  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++)
    avgs = cmax(avgs,avgU[aarx]);
  
  //      log2_maxh = 4+(log2_approx(avgs)/2);
  
  log2_maxh = (log2_approx(avgs)/2)+ log2_approx(ue->frame_parms.nb_antennas_rx-1)+4;
  int Qm = get_Qm_ul(slsch->mcs);

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
			       Qm,
			       slsch->L_CRBs,
			       log2_maxh); // log2_maxh+I0_shift

    if (ue->frame_parms.nb_antennas_rx > 1)
      ulsch_detection_mrc(&ue->frame_parms,
			  (int32_t**)rxdataF_comp,
			  (int32_t**)ul_ch_mag,
			  NULL,
			  l,
			  slsch->L_CRBs);
    
    freq_equalization(&ue->frame_parms,
		      (int32_t**)rxdataF_comp,
		      (int32_t**)ul_ch_mag,
		      NULL,
		      l,
		      slsch->L_CRBs*12,
		      Qm);
  
  }
  lte_idft(&ue->frame_parms,
           (uint32_t*)rxdataF_comp[0],
           slsch->L_CRBs*12);

#ifdef PSSCH_DEBUG
  write_output("slsch_rxF_comp.m","slschrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
#endif

  int E = 12*Qm*slsch->L_CRBs*((Nsymb-1)<<1);

  int16_t *llrp = ue->slsch_ulsch_llr;

  for (int l=0; l<(Nsymb<<1)-1; l++) {

    if (((ue->frame_parms.Ncp == 0) && ((l==3) || (l==10)))||   // skip pilots
        ((ue->frame_parms.Ncp == 1) && ((l==2) || (l==8)))) {
      l++;
    }

    switch (Qm) {
    case 2 :
      ulsch_qpsk_llr(&ue->frame_parms,
                     (int32_t**)rxdataF_comp,
                     (int16_t*)ue->slsch_ulsch_llr,
                     l,
                     slsch->L_CRBs,
                     &llrp);
      break;

    case 4 :
      ulsch_16qam_llr(&ue->frame_parms,
                      (int32_t**)rxdataF_comp,
                      (int16_t*)ue->slsch_ulsch_llr,
                      (int32_t**)ul_ch_mag,
                      l,slsch->L_CRBs,
                      &llrp);
      break;

    case 6 :
      AssertFatal(1==0,"64QAM not supported for SL\n");
      /*
      ulsch_64qam_llr(frame_parms,
                      pusch_vars->rxdataF_comp,
                      pusch_vars->llr,
                      pusch_vars->ul_ch_mag,
                      pusch_vars->ul_ch_magb,
                      l,ulsch[UE_id]->harq_processes[harq_pid]->nb_rb,
                      &llrp);
      */
      break;

    default:
      AssertFatal(1==0,"Unknown Qm !!!!\n");
      break;
    }
  }
  
#ifdef PSSCH_DEBUG
  write_output("slsch_llr.m","slschllr",ue->slsch_ulsch_llr,
               12*slsch->L_CRBs*Qm*(ue->frame_parms.symbols_per_tti-2),
               1,0);  
#endif

  // unscrambling

  uint32_t x1,x2=510+(((uint32_t)slsch->group_destination_id)<<14)+(ljmod10<<9);
  LOG_D(PHY,"Setting seed (unscrambling) for SL to %x (%x,%d)\n",x2,slsch->group_destination_id,ljmod10);

  uint32_t s = lte_gold_generic(&x1, &x2, 1);
  int k=0;
  int16_t c;

  


  for (int i=0; i<(1+(E>>5)); i++) {
    for (int j=0; j<32; j++,k++) {
        c = (int16_t)((((s>>j)&1)<<1)-1);
	//	printf("i %d : %d (llr %d c %d)\n",(i<<5)+j,c*ue->slsch_ulsch_llr[k],ue->slsch_ulsch_llr[k],c);
        ue->slsch_ulsch_llr[k] = c*ue->slsch_ulsch_llr[k];
    }    
    s = lte_gold_generic(&x1, &x2, 0);
  }

  // Deinterleaving

  int Cmux = (Nsymb-1)*2;
  for (int i=0,j=0;i<Cmux;i++) {
    // 12 = 12*(Nsymb-1)/(Nsymb-1)
//    printf("******* i %d\n",i);
    if (Qm == 2) {
      for (int r=0;r<12*slsch->L_CRBs;r++) {
	ue->slsch_dlsch_llr[((r*Cmux)+i)<<1]     = ue->slsch_ulsch_llr[j++];
	ue->slsch_dlsch_llr[(((r*Cmux)+i)<<1)+1] = ue->slsch_ulsch_llr[j++];
	//	printf("dlsch_llr[%d] %d(%d) dlsch_llr[%d] %d(%d)\n",
	//	       ((r*Cmux)+i)<<1,ue->slsch_dlsch_llr[((r*Cmux)+i)<<1],j-2,(((r*Cmux)+i)<<1)+1,ue->slsch_dlsch_llr[(((r*Cmux)+i)<<1)+1],j-1);
      }
    }
    else if (Qm == 4) {
      for (int r=0;r<12*slsch->L_CRBs;r++) {
	ue->slsch_dlsch_llr[((r*Cmux)+i)<<2]     = ue->slsch_ulsch_llr[j++];
	ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+1] = ue->slsch_ulsch_llr[j++];
	ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+2] = ue->slsch_ulsch_llr[j++];
	ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+3] = ue->slsch_ulsch_llr[j++];
	//	printf("dlsch_llr[%d] %d(%d) dlsch_llr[%d] %d(%d)\n",
	//	       ((r*Cmux)+i)<<2,ue->slsch_dlsch_llr[((r*Cmux)+i)<<2],j-4,(((r*Cmux)+i)<<2)+1,ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+1],j-3);
	//	printf("dlsch_llr[%d] %d(%d) dlsch_llr[%d] %d(%d)\n",
	//	       (((r*Cmux)+i)<<2)+2,ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+2],j-2,(((r*Cmux)+i)<<2)+3,ue->slsch_dlsch_llr[(((r*Cmux)+i)<<2)+3],j-1);
      }
    }
  }


  // Decoding

  ue->dlsch_rx_slsch->harq_processes[0]->rvidx = slsch->rvidx;
  ue->dlsch_rx_slsch->harq_processes[0]->nb_rb = slsch->L_CRBs;
  ue->dlsch_rx_slsch->harq_processes[0]->TBS   = get_TBS_UL(slsch->mcs,slsch->L_CRBs)<<3;
  ue->dlsch_rx_slsch->harq_processes[0]->Qm    = Qm;
  ue->dlsch_rx_slsch->harq_processes[0]->G     = E;
  ue->dlsch_rx_slsch->harq_processes[0]->Nl    = 1;

  //  for (int i=0;i<E/16;i++) printf("decoding: E[%d] %d\n",i,ue->slsch_dlsch_llr[i]);

  int ret = dlsch_decoding(ue,
		 ue->slsch_dlsch_llr,
		 &ue->frame_parms,
		 ue->dlsch_rx_slsch,
		 ue->dlsch_rx_slsch->harq_processes[0],
		 frame_rx,
		 subframe_rx,
		 0,
		 0,
		 ue->dlsch_rx_slsch->harq_processes[0]->TBS>256?1:0);

  LOG_D(PHY,"slsch decoding round %d ret %d (%d,%d)\n",(ue->dlsch_rx_slsch->harq_processes[0]->round+3)&3,ret,
	dB_fixed(ue->pusch_slsch->ulsch_power[0]),
	dB_fixed(ue->pusch_slsch->ulsch_power[1]));
  if (ret<ue->dlsch_rx_slsch->max_turbo_iterations) {
    ue->slsch_decoded=1;
    LOG_D(PHY,"SLSCH received for group_id %d (L_CRBs %d, mcs %d,rvidx %d, iter %d)\n",
	  slsch->group_destination_id,slsch->L_CRBs,slsch->mcs,
	  slsch->rvidx,ret);

    /*LOG_I(PDCP, "In slsch_decoding() before calling ue_send_sl_sdu() 1 \n");
    list_display_memory_head_tail(&pdcp_sdu_list);*/

    ue_send_sl_sdu(0,
                   0,
                   frame_rx,subframe_rx,
                   ue->dlsch_rx_slsch->harq_processes[0]->b,
                   ue->dlsch_rx_slsch->harq_processes[0]->TBS>>3,
                   0,
                   SL_DISCOVERY_FLAG_NO);
    /*LOG_I(PDCP, "In slsch_decoding() before calling ue_send_sl_sdu() 1 \n");
    list_display_memory_head_tail(&pdcp_sdu_list);*/

    if      (slsch->rvidx == 0 ) ue->slsch_rxcnt[0]++;
    else if (slsch->rvidx == 2 ) ue->slsch_rxcnt[1]++;
    else if (slsch->rvidx == 3 ) ue->slsch_rxcnt[2]++;
    else if (slsch->rvidx == 1 ) ue->slsch_rxcnt[3]++;
  }
  else if (ue->dlsch_rx_slsch->harq_processes[0]->rvidx == 1 &&
	   ret==ue->dlsch_rx_slsch->max_turbo_iterations) {
    LOG_D(PHY,"sLSCH received in error for group_id %d (L_CRBs %d, mcs %d) power (%d,%d)\n",
	  slsch->group_destination_id,slsch->L_CRBs,slsch->mcs,
	  dB_fixed(ue->pusch_slsch->ulsch_power[0]),
	  dB_fixed(ue->pusch_slsch->ulsch_power[1]));
    ue->slsch_errors++;
  }
  else {
     LOG_D(PHY,"sLSCH received in error for rvidx %d round %d (L_CRBs %d, mcs %d)\n",
           slsch->rvidx,(ue->dlsch_rx_slsch->harq_processes[0]->round+3)&3,slsch->L_CRBs,slsch->mcs);
/*
     if (slsch->rvidx == 0) {
        write_output("slsch_rxF_comp.m","slschrxF_comp",rxdataF_comp[0],ue->frame_parms.N_RB_UL*12*14,1,1);
	write_output("slsch_rxF_ext.m","slschrxF_ext",rxdataF_ext[0],14*12*ue->frame_parms.N_RB_DL,1,1);
	write_output("drs_ext0.m","drsest0",drs_ch_estimates[0],ue->frame_parms.N_RB_UL*12*14,1,1);
	exit(-1);

     }
*/
  }


}

void rx_slsch(PHY_VARS_UE *ue,UE_rxtx_proc_t *proc, int frame_rx,int subframe_rx) {

  AssertFatal(frame_rx<1024 && frame_rx>=0,"frame %d is illegal\n",frame_rx);
  AssertFatal(subframe_rx<10 && subframe_rx>=0,"subframe %d is illegal\n",subframe_rx);
  SLSCH_t *slsch = &ue->slsch_rx;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = slsch->SL_OffsetIndicator;
  uint32_t P = slsch->SL_SC_Period;
  uint32_t absSF = (frame_rx*10)+subframe_rx;
  uint32_t absSF_offset,absSF_modP;

  if (ue->slcch_received == 0) return;

  if (ue->slsch_rx_sdu_active == 0) return;

  absSF_offset = absSF-O;

  if (absSF < O) return;

  absSF_modP = absSF_offset%P;

  // This is the condition for short SCCH bitmap (slsch->SubframeBitmapSL_length bits), check that the current subframe is for SCCH
  if (absSF_modP < slsch->SubframeBitmapSL_length) return;

  LOG_D(PHY,"Checking pssch for absSF %d (trp mask %d, rv %d, slsch_decoded %d)\n",
	absSF, trp8[slsch->time_resource_pattern][absSF_modP&7],
	slsch->rvidx,
        ue->slsch_decoded);
  // Note : this assumes Ntrp=8 for now
  if (trp8[slsch->time_resource_pattern][absSF_modP&7]==0) return;
  // we have an opportunity in this subframe
  if (absSF_modP == slsch->SubframeBitmapSL_length) slsch->ljmod10 = 0;
  else slsch->ljmod10++;
  if (slsch->ljmod10 == 10) slsch->ljmod10 = 0;

  LOG_D(PHY,"Receiving slsch for rvidx %d\n",slsch->rvidx);

  if (slsch->rvidx == 0) { // first new transmission in period, get a new packet
    ue->slsch_decoded = 0;
    slsch_decoding(ue,proc,frame_rx,subframe_rx,slsch->ljmod10);
    slsch->rvidx=2;
  }
  else {
    if (ue->slsch_decoded == 0) slsch_decoding(ue,proc,frame_rx,subframe_rx,slsch->ljmod10);
    if      (slsch->rvidx == 2) slsch->rvidx = 3;
    else if (slsch->rvidx == 3) slsch->rvidx = 1;
    else if (slsch->rvidx == 1) slsch->rvidx = 0;
    else                        AssertFatal(1==0,"rvidx %d isn't possible\n",slsch->rvidx);
  }
  LOG_D(PHY,"%d.%d : returning\n",frame_rx,subframe_rx);

}

