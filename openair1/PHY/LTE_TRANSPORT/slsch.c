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
#ifndef __LTE_TRANSPORT_SLSS__C__
#define __LTE_TRANSPORT_SLSS__C__
#include "PHY/defs.h"

int64_t sci_mapping(PHY_VARS_UE *ue) {
  SLSCH_t *slsch                = ue->slsch;
  
  AssertFatal(slsch->freq_hopping_flag<2,"freq_hop %d >= 2\n",slsch->freq_hopping_flag);
  int64_t freq_hopping_flag     = (uint64_t)slsch->freq_hopping_flag;
  
  int64_t RAbits                = log2_approx(slsch->N_SL_RB*((slsch->N_SL_RB+1)>>1));
  AssertFatal(slsch->resource_block_coding<(1<<RAbits),"slsch->resource_block_coding %x >= %x\n",slsch->resource_block_coding,(1<<RAbits));
  int64_t resource_block_coding     = (uint64_t)slsch->resource_block_coding; 
  
  AssertFatal(slsch->time_resource_pattern<128,"slsch->time_resource_pattern %d>=128\n",slsch->time_resource_pattern);
  int64_t time_resource_pattern     = (uint64_t)slsch->time_resource_pattern;

  AssertFatal(slsch->mcs<32,"slsch->mcs %d >= 32\n",slsch->mcs);
  int64_t mcs                       = (uint64_t)slsch->mcs;

  AssertFatal(slsch->timing_advance_indication<2048,"slsch->timing_advance_indication %d >= 2048\n",slsch->timing_advance_indication);
  int64_t timing_advance_indication = (uint64_t)slsch->timing_advance_indication;

  AssertFatal(slsch->group_destination_id<256,"slsch->group_destination_id %d >= 256\n",slsch->group_destination_id);
  int64_t group_destination_id = (uint64_t)slsch->group_destination_id;

  // map bitfields
  // frequency-hopping 1-bit
  return( freq_hopping_flag | 
	  (resource_block_coding << 1) | 
	  (time_resource_pattern<<(1+RAbits)) | 
	  (mcs<<(1+RAbits+7)) | 
	  (timing_advance_indication<<(1+RAbits+7+5)) | 
	  (group_destination_id<<(1+RAbits+7+5+11))
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

void idft_slcch(LTE_DL_FRAME_PARMS *frame_parms,int32_t *z)
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
    z0 = z;
    z1 = z0+(frame_parms->N_RB_DL*12);
    z2 = z1+(frame_parms->N_RB_DL*12);
    //pilot
    z3 = z2+(2*frame_parms->N_RB_DL*12);
    z4 = z3+(frame_parms->N_RB_DL*12);
    z5 = z4+(frame_parms->N_RB_DL*12);
  } else { // extended prefix
    z0 = z;
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

  LTE_UE_PSCCH_TX *pscch = ue->pscch_vars_tx;
  SLSCH_t *slsch         = ue->slsch;
  int tx_amp;
  int nprb;
  uint64_t sci;
  // Note this should depend on configuration of slsch/slcch
  uint32_t Nsymb = 7;
  uint32_t E = 12*(Nsymb-1)*2;

  // coding part
  if (ue->pscch_coded == 0) {
    sci = sci_mapping(ue);
    dci_encoding((uint8_t *)sci,
		 log2_approx(slsch->N_SL_RB*((slsch->N_SL_RB+1)>>1))+31,
		 E,
		 pscch->f,
		 0);

    // interleaving
    // Cmux assumes configuration 0
    int Cmux = Nsymb-1;
    uint8_t *fptr;
    for (int i=0,j=0; i<Cmux; i++)
      // 24 = 12*(Nsymb-1)*2/(Nsymb-1)
      for (int r=0; r<24; r++) {
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
    }		 
    ue->pscch_coded=1;
  }
  // convert a to prb number and compute slot

  // get index within slot (first half of the prbs in slot 0, second half in 1)
  uint32_t amod = a%(slsch->N_SL_RB);

  if (amod<(slsch->N_SL_RB>>1)) nprb = slsch->prb_Start + amod;
  else                          nprb = slsch->prb_End-slsch->N_SL_RB+amod;


  // Fill in power control later
  //  pssch_power_cntl(ue,proc,eNB_id,1, abstraction_flag);
  //  ue->tx_power_dBm[subframe_tx] = ue->slcch[eNB_id]->Po_PUSCH;
  ue->tx_power_dBm[subframe_tx] = 0;
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

  for (int j=0,l=0; l<(Nsymb-1); l++) {
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
		       aa);



}

void check_and_generate_pscch(PHY_VARS_UE *ue,int frame_tx,int subframe_tx) {
  
  AssertFatal(frame_tx<1024 && frame_tx>0,"frame %d is illegal\n",frame_tx);
  AssertFatal(subframe_tx<10 && subframe_tx>0,"subframe %d is illegal\n",subframe_tx);
  SLSCH_t *slsch = ue->slsch;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = ue->slsch->SL_OffsetIndicator;
  uint32_t P = ue->slsch->SL_SC_Period;
  uint32_t absSF = (frame_tx*10)+subframe_tx;
  uint32_t absSF_offset,absSF_modP;

  if (ue->pscch_generated == 1) return;


  
  absSF_offset = absSF-O;

  if (absSF_offset < O) return;

  absSF_modP = absSF_offset%P;

  // This is the condition for short SCCH bitmap (40 bits), check that the current subframe is for SCCH
  if (absSF_modP > 39) return;


  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & slsch->bitmap1) == 0) return;

  // if we get here, then there is a PSCCH subframe for a potential transmission
  uint32_t sf_index=40,LPSCCH=0;
  for (int i=0;i<40;i++) {
    if (i==absSF_modP) sf_index=LPSCCH;
    if (((((uint64_t)1)<<i) & slsch->bitmap1)>0) sf_index++;
  }
  AssertFatal(sf_index<40,"sf_index not set, should not happen\n");
  LPSCCH++;
  // sf_index now contains the SF index in 0...LPSCCH-1
  // LPSCCH has the number of PSCCH subframes

  // 2 SLSCH/SLCCH resource block regions subframe times number of resources blocks per slot times 2 slots
  uint32_t M_RB_PSCCH_RP = slsch->N_SL_RB*LPSCCH<<2;
  AssertFatal(slsch->n_pscch < (M_RB_PSCCH_RP>>1)*LPSCCH,"n_pscch not in 0..%d\n",
	      ((M_RB_PSCCH_RP>>1)*LPSCCH)-1);
  // hard-coded to transmission mode one for now (Section 14.2.1.1 from 36.213 Rel14.3)
  uint32_t a1=slsch->n_pscch/LPSCCH;
  uint32_t a2=a1+slsch->n_pscch/LPSCCH+(M_RB_PSCCH_RP>>1);
  uint32_t b1=slsch->n_pscch%LPSCCH;
  uint32_t b2=(slsch->n_pscch + 1 + (a1%(LPSCCH-1)))%LPSCCH;

  if (absSF_modP == b1)      pscch_codingmodulation(ue,frame_tx,subframe_tx,a1,0);	
  else if (absSF_modP == b2) pscch_codingmodulation(ue,frame_tx,subframe_tx,a2,1);
  else return;

}

void generate_slsch(PHY_VARS_UE *ue,SLSCH_t *slsch,int frame_tx,int subframe_tx) {
    
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
      
      LOG_I(PHY,"SLSCH configuration %lu bytes, TBS payload %d bytes => %lu bytes\n",
	    sizeof(SLSCH_t)-sizeof(uint8_t*),
	    slsch->payload_length,
	    slsch_header_len+sizeof(SLSCH_t)-sizeof(uint8_t*)+slsch->payload_length);
      
      multicast_link_write_sock(0, 
				&pdu, 
				slsch_header_len+sizeof(SLSCH_t)-sizeof(uint8_t*)+slsch->payload_length);
  
    }
  } // sidelink_emulation=1
  else if (ue->sidelink_active==0){ // This is first indication of sidelink in this period
    ue->sidelink_active = 1;
    ue->slsch           = slsch;
  }
  // check and flll SCI portion
  check_and_generate_pscch(ue,frame_tx,subframe_tx);
  // check_and_generate_pssch(ue,frame_tx,subframe_tx);
}


void pscch_decoding(PHY_VARS_UE *ue,int frame_rx,int subframe_rx,int a,int slot) {

  AssertFatal(slot==0 || slot==1, "slot %d is illegal\n",slot);
  int Nsymb = 7 - slot;
  SLSCH_t *slsch = &ue->slsch_rx;

  uint32_t amod = a%(slsch->N_SL_RB);
  int32_t rxdataF_ext[2][12];
  int32_t drs_ch_estimates[2][12] __attribute__ ((aligned (32)));
  int32_t rxdataF_comp[2][12] __attribute__ ((aligned (32)));
  int32_t ul_ch_mag[2][12] __attribute__ ((aligned (32)));
  int32_t avgs;
  uint8_t log2_maxh=0;
  int32_t avgU[2];
  int nprb;

  if (amod<(slsch->N_SL_RB>>1)) nprb = slsch->prb_Start + amod;
  else                          nprb = slsch->prb_End-slsch->N_SL_RB+amod;

  // slot FEP
  RU_t ru_tmp;
  int rxdata_7_5kHz[2][ue->frame_parms.samples_per_tti] __attribute__ ((aligned (32)));
  int rxdataF[2][ue->frame_parms.ofdm_symbol_size*ue->frame_parms.symbols_per_tti] __attribute__ ((aligned (32)));

  memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS)); 
  ru_tmp.common.rxdata = ue->common_vars.rxdata;
  ru_tmp.common.rxdata_7_5kHz = (int32_t**)rxdata_7_5kHz;
  ru_tmp.common.rxdataF = (int32_t**)rxdataF;

  remove_7_5_kHz(&ru_tmp,(subframe_rx<<1)+slot);

  // extract symbols from slot  
  for (int l=0; l<Nsymb; l++) {
    slot_fep_ul(&ru_tmp,l,(subframe_rx<<1)+slot,0);
    ulsch_extract_rbs_single((int32_t**)rxdataF,
			     (int32_t**)rxdataF_ext,
			     nprb,
			     1,
			     l,
			     (subframe_rx<<1)+slot,
			     &ue->frame_parms);
  }
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
			    3,
			    slot,
			    0);
  
  avgs = 0;
  
  for (int aarx=0; aarx<ue->frame_parms.nb_antennas_rx; aarx++)
    avgs = cmax(avgs,avgU[aarx]);
  
  //      log2_maxh = 4+(log2_approx(avgs)/2);
  
  log2_maxh = (log2_approx(avgs)/2)+ log2_approx(ue->frame_parms.nb_antennas_rx-1)+4;

  for (int l=0; l<(ue->frame_parms.symbols_per_tti>>1)-1; l++) {

    if (((ue->frame_parms.Ncp == 0) && ((l==3) || (l==10)))||   // skip pilots
        ((ue->frame_parms.Ncp == 1) && ((l==2) || (l==8)))) {
      l++;
    }

    ulsch_channel_compensation((int32_t**)rxdataF_ext,
			       (int32_t**)drs_ch_estimates,
			       (int32_t**)ul_ch_mag,
			       NULL,
			       (int32_t**)rxdataF_comp,
			       &ue->frame_parms,
			       l,
			       2,
			       1,
			       log2_maxh); // log2_maxh+I0_shift

  


    if (ue->frame_parms.nb_antennas_rx > 1)
      ulsch_detection_mrc(&ue->frame_parms,
			  (int32_t**)rxdataF_comp,
			  (int32_t**)ul_ch_mag,
			  NULL,
			  l,
			  1);
    
    
    

    //    if (23<ulsch_power[0]) {
      freq_equalization(&ue->frame_parms,
			(int32_t**)rxdataF_comp,
			(int32_t**)ul_ch_mag,
			NULL,
			l,
			12,
			2);
    
  }

  idft_slcch(&ue->frame_parms,
	     rxdataF_comp[0]);

  int E=144;
  int16_t llr[144];
  int16_t *llrp = (int16_t*)&llr[0];

  for (int l=0; l<(ue->frame_parms.symbols_per_tti>>1)-slot; l++) {

    if (((ue->frame_parms.Ncp == 0) && ((l==3) || (l==10)))||   // skip pilots
        ((ue->frame_parms.Ncp == 1) && ((l==2) || (l==8)))) {
      l++;
    }
    
    ulsch_qpsk_llr(&ue->frame_parms,
		   (int32_t**)rxdataF_comp,
		   llr,
		   l,
		   1,
		   &llrp);
  }

  // unscrambling
  uint32_t x1,x2=510,k;
  
  uint32_t s = lte_gold_generic(&x1,&x2,1); 
  
  for (int i=0,k=0;i<(1+(E>>5));i++) 
    for (int j=0;(j<32)&&(k<E);j++,k++) 
      llr[k] = (int16_t)((((s>>j)&1)<<1)-1) * llr[k];

  // deinterleaving
  int8_t f[144];
  int Cmux = (ue->frame_parms.symbols_per_tti>>1)-1-slot;
  for (int i=0,j=0;i<Cmux;i++)
    // 24 = 12*(Nsymb-1)*2/(Nsymb-1)
    for (int r=0;r<24;r++) {
      f[((r*Cmux)+1)<<1]     = (int8_t)llr[j++];
      f[(((r*Cmux)+1)<<1)+1] = (int8_t)llr[j++];
    }
  uint16_t res;
  uint64_t sci_rx;
  //decoding
  int length = log2_approx(slsch->N_SL_RB*((ue->slsch_rx.N_SL_RB+1)>>1))+31;
  dci_decoding(length,144,f,(uint8_t*)&sci_rx);
  res = (crc16((uint8_t*)&sci_rx,length)>>16) ^ extract_crc((uint8_t*)&sci_rx,length);
#ifdef DEBUG_SCI_DECODING
  printf("crc res =>%x\n",res);
#endif
  // unpopulate SCI bit fields
  int RAbits = length-31;
  if (res==0) {
    ue->slsch_rx.freq_hopping_flag         = sci_rx&1;
    ue->slsch_rx.resource_block_coding     = (sci_rx>>1)&((1<<RAbits)-1);
    ue->slsch_rx.time_resource_pattern     = (sci_rx>>(1+RAbits))&127;
    ue->slsch_rx.mcs                       = (sci_rx>>(1+7+RAbits))&31;
    ue->slsch_rx.timing_advance_indication = (sci_rx>>(1+7+5+RAbits))&2047;
    ue->slsch_rx.group_destination_id      = (sci_rx>>(1+7+5+11+RAbits))&255;
    ue->slcch_received                     = 1;
  }
}	

void rx_slcch(PHY_VARS_UE *ue,int frame_rx,int subframe_rx) {

  AssertFatal(frame_rx<1024 && frame_rx>0,"frame %d is illegal\n",frame_rx);
  AssertFatal(subframe_rx<10 && subframe_rx>0,"subframe %d is illegal\n",subframe_rx);
  SLSCH_t *slsch = &ue->slsch_rx;
  AssertFatal(slsch!=NULL,"SLSCH is null\n");
  uint32_t O = slsch->SL_OffsetIndicator;
  uint32_t P = slsch->SL_SC_Period;
  uint32_t absSF = (frame_rx*10)+subframe_rx;
  uint32_t absSF_offset,absSF_modP;

  if (ue->slcch_received == 1) return;

  absSF_offset = absSF-O;

  if (absSF_offset < O) return;

  absSF_modP = absSF_offset%P;

  // This is the condition for short SCCH bitmap (40 bits), check that the current subframe is for SCCH
  if (absSF_modP > 39) return;


  uint64_t SFpos = ((uint64_t)1) << absSF_modP;
  if ((SFpos & slsch->bitmap1) == 0) return;

  // if we get here, then there is a PSCCH subframe for a potential transmission
  uint32_t sf_index=40,LPSCCH=0;
  for (int i=0;i<40;i++) {
    if (i==absSF_modP) sf_index=LPSCCH;
    if (((((uint64_t)1)<<i) & slsch->bitmap1)>0) sf_index++;
  }
  AssertFatal(sf_index<40,"sf_index not set, should not happen\n");
  LPSCCH++;
  // sf_index now contains the SF index in 0...LPSCCH-1
  // LPSCCH has the number of PSCCH subframes

  // 2 SLSCH/SLCCH resource block regions subframe times number of resources blocks per slot times 2 slots
  uint32_t M_RB_PSCCH_RP = slsch->N_SL_RB*LPSCCH<<2;
  AssertFatal(slsch->n_pscch < (M_RB_PSCCH_RP>>1)*LPSCCH,"n_pscch not in 0..%d\n",
	      ((M_RB_PSCCH_RP>>1)*LPSCCH)-1);
  // hard-coded to transmission mode one for now (Section 14.2.1.1 from 36.213 Rel14.3)
  uint32_t a1=slsch->n_pscch/LPSCCH;
  uint32_t a2=a1+slsch->n_pscch/LPSCCH+(M_RB_PSCCH_RP>>1);
  uint32_t b1=slsch->n_pscch%LPSCCH;
  uint32_t b2=(slsch->n_pscch + 1 + (a1%(LPSCCH-1)))%LPSCCH;

  if (absSF_modP == b1)      pscch_decoding(ue,frame_rx,subframe_rx,a1,0);	
  else if (absSF_modP == b2) pscch_decoding(ue,frame_rx,subframe_rx,a2,1);
  else return;


}

#endif
