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

/*! \file PHY/LTE_TRANSPORT/sss.c
* \brief Top-level routines for generating and decoding the secondary synchronization signal (SSS) V8.6 2009-03
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
#include "PHY/defs_UE.h"

//Panos: Substituting defs.h with transport_ue.h after merge of sidelink with develop. Not sure if this is correct
//#include "defs.h"
#include "transport_ue.h"

#include "PHY/phy_extern.h"
#include "PHY/MODULATION/modulation_eNB.h"

//#define DEBUG_SSS


int generate_slsss(int32_t **txdataF,
		   int subframe,
		   int16_t amp,
		   LTE_DL_FRAME_PARMS *frame_parms,
		   uint16_t symbol)
{

  uint8_t i,aa,Nsymb;
  int16_t *d,k;
  uint8_t Nid2;
  uint16_t Nid1;
  int16_t a;

  Nid2 = frame_parms->Nid_SL / 168;
  AssertFatal(Nid2<2,"Nid2 %d >= 2\n",Nid2);
  
  Nid1 = frame_parms->Nid_SL%168;


  AssertFatal((frame_parms->Ncp == NORMAL && (symbol==11 || symbol==12)) ||
	      (frame_parms->Ncp == EXTENDED && (symbol==10 || symbol==11)),
	      "Symbol %d not possible for SLSSS\n",
	      symbol);
  
  if (((symbol == 11) && frame_parms->Ncp == NORMAL) ||
      ((symbol == 10) && frame_parms->Ncp == EXTENDED))
    d = &d0_sss[62*(Nid2 + (Nid1*2))];
  else 
    d = &d5_sss[62*(Nid2 + (Nid1*2))];

  Nsymb = (frame_parms->Ncp==NORMAL)?14:12;
  k = frame_parms->ofdm_symbol_size-3*12+5;
  a = (frame_parms->nb_antenna_ports_eNB == 1) ? amp : (amp*ONE_OVER_SQRT2_Q15)>>15;

  for (i=0; i<62; i++) {
    for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {

      ((int16_t*)txdataF[aa])[subframe*Nsymb*frame_parms->ofdm_symbol_size + 
			      (2*(symbol*frame_parms->ofdm_symbol_size + k))] =
	(a * d[i]);
      ((int16_t*)txdataF[aa])[subframe*Nsymb*frame_parms->ofdm_symbol_size + 
			      (2*(symbol*frame_parms->ofdm_symbol_size + k))+1] = 
	0;
    }

    k+=1;

    if (k >= frame_parms->ofdm_symbol_size) {
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  return(0);
}

int slpss_ch_est(PHY_VARS_UE *ue,
		 int32_t pss0_ext[4][72],
		 int32_t sss0_ext[4][72],
		 int32_t pss1_ext[4][72],
		 int32_t sss1_ext[4][72],
                 int64_t sss0_comp[72],
		 int64_t sss1_comp[72],
		 int Nid2)
{

  int16_t *pss;
  int16_t *pss0_ext2,*sss0_ext2,tmp0_re,tmp0_im,tmp1_re,tmp1_im;
  int32_t *sss0comp,*sss1comp,tmp0_re2,tmp0_im2,tmp1_re2,tmp1_im2;
  int16_t *pss1_ext2,*sss1_ext2;
  uint8_t aarx,i;
  LTE_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;

  switch (Nid2) {

  case 0:
    pss = &primary_synch0SL[10];
    break;

  case 1:
    pss = &primary_synch1SL[10];
    break;

  default:
    AssertFatal(1==0,"Impossible Nid2 %d\n",Nid2);
    break;
  }

  sss0comp = (int32_t*)&sss0_comp[5];
  sss1comp = (int32_t*)&sss1_comp[5];

  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

    sss0_ext2 = (int16_t*)&sss0_ext[aarx][5];
    pss0_ext2 = (int16_t*)&pss0_ext[aarx][5];
    sss1_ext2 = (int16_t*)&sss1_ext[aarx][5];
    pss1_ext2 = (int16_t*)&pss1_ext[aarx][5];

    for (i=0; i<62; i++) {

      // This is H*(PSS) = R* \cdot PSS
      tmp0_re = (int16_t)((((pss0_ext2[i<<1]) * (int32_t)pss[i<<1])>>15)     + (((pss0_ext2[1+(i<<1)]) * (int32_t)pss[1+(i<<1)])>>15));
      tmp0_im = (int16_t)((((pss0_ext2[i<<1]) * (int32_t)pss[1+(i<<1)])>>15) - (((pss0_ext2[1+(i<<1)]) * (int32_t)pss[(i<<1)])>>15));
      tmp1_re = (int16_t)((((pss1_ext2[i<<1]) * (int32_t)pss[i<<1])>>15)     + (((pss1_ext2[1+(i<<1)]) * (int32_t)pss[1+(i<<1)])>>15));
      tmp1_im = (int16_t)((((pss1_ext2[i<<1]) * (int32_t)pss[1+(i<<1)])>>15) - (((pss1_ext2[1+(i<<1)]) * (int32_t)pss[(i<<1)])>>15));

      //printf("H*(%d,%d) : (%d,%d)\n",aarx,i,tmp0_re,tmp0_im);
      // This is R(SSS0) \cdot H*(PSS)
      tmp0_re2 = (tmp0_re * (int32_t)sss0_ext2[i<<1])  - 
		 (tmp0_im * (int32_t)sss0_ext2[1+(i<<1)]);
      tmp0_im2 = (tmp0_re * (int32_t)sss0_ext2[1+(i<<1)]) + 
	         (tmp0_im * (int32_t)sss0_ext2[(i<<1)]);
      // This is R(SSS1) \cdot H*(PSS)
      tmp1_re2 = (tmp1_re * (int32_t)sss1_ext2[i<<1])  - 
		 (tmp1_im * (int32_t)sss1_ext2[1+(i<<1)]);
      tmp1_im2 = (tmp1_re * (int32_t)sss1_ext2[1+(i<<1)]) + 
		 (tmp1_im * (int32_t)sss1_ext2[(i<<1)]);
    
      //      printf("SSSi(%d,%d) : (%d,%d)\n",aarx,i,sss_ext2[i<<1],sss_ext2[1+(i<<1)]);
            //printf("SSScomp0(%d,%d) : (%d,%d)\n",aarx,i,tmp0_re2,tmp0_im2);
            //printf("SSScomp1(%d,%d) : (%d,%d)\n",aarx,i,tmp1_re2,tmp1_im2);

      // MRC on RX antennas
      if (aarx==0) {
        sss0comp[i<<1]      = tmp0_re2;
        sss0comp[1+(i<<1)]  = tmp0_im2;
        sss1comp[i<<1]      = tmp1_re2;
        sss1comp[1+(i<<1)]  = tmp1_im2;
      } else {
        sss0comp[i<<1]      += tmp0_re2;
        sss0comp[1+(i<<1)]  += tmp0_im2;
        sss1comp[i<<1]      += tmp1_re2;
        sss1comp[1+(i<<1)]  += tmp1_im2;
      }
    }
  }

  // sss_ext now contains the compensated SSS
  return(0);
}


int _do_slpss_sss_extract(PHY_VARS_UE *ue,
			  int32_t pss0_ext[4][72],
			  int32_t sss0_ext[4][72],
			  int32_t pss1_ext[4][72],
			  int32_t sss1_ext[4][72],
			  uint8_t doPss, uint8_t doSss,
			  int subframe)
{



  uint16_t rb,nb_rb=6;
  uint8_t i,aarx;
  int32_t *pss0_rxF,*pss0_rxF_ext;
  int32_t *pss1_rxF,*pss1_rxF_ext;
  int32_t *sss0_rxF,*sss0_rxF_ext;
  int32_t *sss1_rxF,*sss1_rxF_ext;
  LTE_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;


  int rx_offset = frame_parms->ofdm_symbol_size-3*12;
  uint8_t pss0_symb,pss1_symb,sss0_symb,sss1_symb;

  int32_t **rxdataF;


  //LOG_I(PHY,"do_pss_sss_extract subframe %d \n",subframe);
  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

    if (frame_parms->frame_type == FDD) {
      pss0_symb = 1;
      pss1_symb = 2;
      sss0_symb = 11;
      sss1_symb = 12;
      rxdataF   =  (int32_t**)ue->sl_rxdataF[ue->current_thread_id[0]];
;

      pss0_rxF  =  &rxdataF[aarx][(rx_offset + (pss0_symb*(frame_parms->ofdm_symbol_size)))];
      sss0_rxF  =  &rxdataF[aarx][(rx_offset + (sss0_symb*(frame_parms->ofdm_symbol_size)))];
      pss1_rxF  =  &rxdataF[aarx][(rx_offset + (pss1_symb*(frame_parms->ofdm_symbol_size)))];
      sss1_rxF  =  &rxdataF[aarx][(rx_offset + (sss1_symb*(frame_parms->ofdm_symbol_size)))];
      
    } else {
      AssertFatal(1==0,"TDD not supported for Sidelink\n");
    }
    
  
    //printf("extract_rbs: symbol_mod=%d, rx_offset=%d, ch_offset=%d\n",symbol_mod,
    //   (rx_offset + (symbol*(frame_parms->ofdm_symbol_size)))*2,
    //   LTE_CE_OFFSET+ch_offset+(symbol_mod*(frame_parms->ofdm_symbol_size)));

    pss0_rxF_ext    = &pss0_ext[aarx][0];
    sss0_rxF_ext    = &sss0_ext[aarx][0];
    pss1_rxF_ext    = &pss1_ext[aarx][0];
    sss1_rxF_ext    = &sss1_ext[aarx][0];

    for (rb=0; rb<nb_rb; rb++) {
      // skip DC carrier
      if (rb==3) {
        if(frame_parms->frame_type == FDD)
	  {
	    sss0_rxF       = &rxdataF[aarx][((sss0_symb*(frame_parms->ofdm_symbol_size)))];
	    pss0_rxF       = &rxdataF[aarx][((pss0_symb*(frame_parms->ofdm_symbol_size)))];
	    sss1_rxF       = &rxdataF[aarx][((sss1_symb*(frame_parms->ofdm_symbol_size)))];
	    pss1_rxF       = &rxdataF[aarx][((pss1_symb*(frame_parms->ofdm_symbol_size)))];
	  }
        else
	  AssertFatal(0,"");
      }
    

      for (i=0; i<12; i++) {
        if (doPss) {
	  pss0_rxF_ext[i]=pss0_rxF[i];
	  pss1_rxF_ext[i]=pss1_rxF[i];
	}
        if (doSss) {
	  sss0_rxF_ext[i]=sss0_rxF[i];
	  sss1_rxF_ext[i]=sss1_rxF[i];
	  //	  printf("rb %d: sss0 %d (%d,%d)\n",rb,i,((int16_t*)&sss0_rxF[i])[0],((int16_t*)&sss0_rxF[i])[1]);
	  //	  printf("rb %d: sss1 %d (%d,%d)\n",rb,i,((int16_t*)&sss1_rxF[i])[0],((int16_t*)&sss1_rxF[i])[1]);
	}
      }

      pss0_rxF+=12;
      sss0_rxF+=12;
      pss0_rxF_ext+=12;
      sss0_rxF_ext+=12;
      pss1_rxF+=12;
      sss1_rxF+=12;
      pss1_rxF_ext+=12;
      sss1_rxF_ext+=12;

    }
  }
  
  return(0);
}

int slpss_sss_extract(PHY_VARS_UE *phy_vars_ue,
		      int32_t pss0_ext[4][72],
		      int32_t sss0_ext[4][72],
		      int32_t pss1_ext[4][72],
		      int32_t sss1_ext[4][72],
		      uint8_t subframe)
{
  return _do_slpss_sss_extract(phy_vars_ue, pss0_ext, sss0_ext, pss1_ext, sss1_ext, 1 /* doPss */, 1 /* doSss */, subframe);
}



int16_t phaseSL_re[7] = {16383, 25101, 30791, 32767, 30791, 25101, 16383};
int16_t phaseSL_im[7] = {-28378, -21063, -11208, 0, 11207, 21062, 28377};


int rx_slsss(PHY_VARS_UE *ue,int32_t *tot_metric,uint8_t *phase_max,int Nid2)
{

  uint8_t i;
  int32_t pss0_ext[4][72],pss1_ext[4][72];
  int32_t sss0_ext[4][72],sss1_ext[4][72];
  int64_t sss0_comp[72],sss1_comp[72];
  int16_t sss0_comp16[144],sss1_comp16[144];

  uint8_t phase;
  uint16_t Nid1;
  int16_t *sss0,*sss1;
  LTE_DL_FRAME_PARMS *frame_parms=&ue->frame_parms;
  int32_t metric;
  int16_t *d0,*d5;
  int16_t **rxdataF          = ue->sl_rxdataF[ue->current_thread_id[0]];

  if (frame_parms->frame_type == FDD) {
#ifdef DEBUG_SSS

    if (frame_parms->Ncp == NORMAL)
      printf("[PHY][UE%d] Doing SSS for FDD Normal Prefix\n",ue->Mod_id);
    else
      printf("[PHY][UE%d] Doing SSS for FDD Extended Prefix\n",ue->Mod_id);

#endif
    // Do FFTs for SSS/PSS
    // SSS
    LOG_I(PHY,"Doing SSS detection at offset %d\n",ue->rx_offsetSL);
    
    RU_t ru_tmp;
    memset((void*)&ru_tmp,0,sizeof(RU_t));
    
    memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
    ru_tmp.N_TA_offset=0;
    ru_tmp.common.rxdata_7_5kHz     = (int32_t**)malloc16(ue->frame_parms.nb_antennas_rx*sizeof(int32_t*)); 
    ru_tmp.common.rxdata            = (int32_t**)malloc16(ue->frame_parms.nb_antennas_rx*sizeof(int32_t*)); 
    for (int aa=0;aa<ue->frame_parms.nb_antennas_rx;aa++) { 
      ru_tmp.common.rxdata_7_5kHz[aa] = (int32_t*)ue->sl_rxdata_7_5kHz[ue->current_thread_id[0]][aa];
      ru_tmp.common.rxdata[aa]        = (int32_t*)&ue->common_vars.rxdata_syncSL[aa][2*ue->rx_offsetSL];
    }
    ru_tmp.common.rxdataF = (int32_t**)rxdataF;
    ru_tmp.nb_rx = ue->frame_parms.nb_antennas_rx;
    
    
    remove_7_5_kHz(&ru_tmp,0);
    remove_7_5_kHz(&ru_tmp,1);
    // PSS
    slot_fep_ul(&ru_tmp,1,0,0);
    slot_fep_ul(&ru_tmp,2,0,0);
    // SSS
    slot_fep_ul(&ru_tmp,4,1,0);
    slot_fep_ul(&ru_tmp,5,1,0);
    
    free(ru_tmp.common.rxdata_7_5kHz); 
    free(ru_tmp.common.rxdata); 
  } else { // TDD
    AssertFatal(1==0,"TDD not supported for Sidelink\n");
  }
  // pss sss extract for subframe 0
  slpss_sss_extract(ue,
		    pss0_ext,
		    sss0_ext,
		    pss1_ext,
		    sss1_ext,
		    0);

  //  write_output("rxdataF0.m","rxF0",&rxdataF[0][0],2*14*ue->frame_parms.ofdm_symbol_size,1,1);
   /* write_output("pss0_ext.m","pss0ext",pss0_ext,72,1,1);
    write_output("sss0_ext.m","sss0ext",sss0_ext,72,1,1);
    write_output("pss1_ext.m","pss1ext",pss1_ext,72,1,1);
    write_output("sss1_ext.m","sss1ext",sss1_ext,72,1,1); */



  //  exit(-1);

  // get conjugated channel estimate from PSS (symbol 6), H* = R* \cdot PSS
  // and do channel estimation and compensation based on PSS

  slpss_ch_est(ue,
	       pss0_ext,
	       sss0_ext,
	       pss1_ext,
	       sss1_ext,
	       sss0_comp,
	       sss1_comp,
	       Nid2);

  // rescale from 64 to 16 bit resolution keeping 8 bits of dynamic range
  uint32_t maxval=0;
  int32_t *sss0comp=(int32_t*)sss0_comp,*sss1comp=(int32_t*)sss1_comp;
  for (i=10;i<134;i++) {
      if (sss0comp[i] >=0) maxval=(uint64_t)max(maxval,sss0comp[i]);
      else maxval=(uint64_t)max(maxval,-sss0comp[i]);
      if (sss1comp[i] >=0) maxval=(uint64_t)max(maxval,sss1comp[i]);
      else maxval=(uint64_t)max(maxval,-sss1comp[i]);
  }
  uint8_t log2maxval = log2_approx64(maxval);
  uint8_t shift;
  if (log2maxval < 8) shift = 0; else shift = log2maxval-8; 




  for (i=0;i<144;i++) {
      sss0_comp16[i] = (int16_t)(sss0comp[i]>>shift);
      sss1_comp16[i] = (int16_t)(sss1comp[i]>>shift);
  }
/*
    write_output("sss0_comp0.m","sss0comp0",sss0_comp16,72,1,1);
    write_output("sss1_comp0.m","sss1comp0",sss1_comp16,72,1,1);
    exit(-1); */
  // now do the SSS detection based on the precomputed sequences in PHY/LTE_TRANSPORT/sss.h

  *tot_metric = -99999999;


  sss0 = &sss0_comp16[10];
  sss1 = &sss1_comp16[10];


  for (phase=0; phase<7; phase++) { // phase offset between PSS and SSS
    for (Nid1 = 0 ; Nid1 <= 167; Nid1++) {  // 168 possible Nid1 values
      metric = 0;
      
      d0 = &d0_sss[62*(Nid2 + (Nid1*2))];
      d5 = &d5_sss[62*(Nid2 + (Nid1*2))];
      // This is the inner product using one particular value of each unknown parameter
      for (i=0; i<62; i++) {
	metric += 
	  (int16_t)(((d0[i]*((((phaseSL_re[phase]*(int32_t)sss0[i<<1])>>15)-((phaseSL_im[phase]*(int32_t)sss0[1+(i<<1)])>>15)))))) + 
	  (int16_t)(((d5[i]*((((phaseSL_re[phase]*(int32_t)sss1[i<<1])>>15)-((phaseSL_im[phase]*(int32_t)sss1[1+(i<<1)])>>15))))));
      }
      
      // if the current metric is better than the last save it
      if (metric > *tot_metric) {
	*tot_metric = metric;
	ue->frame_parms.Nid_SL = (Nid2*168)+Nid1;
	*phase_max = phase;
	//#ifdef DEBUG_SSS
	  LOG_I(PHY,"(phase,Nid_SL) (%d,%d), metric_phase %d tot_metric %d, phase_max %d\n",phase,ue->frame_parms.Nid_SL,metric,*tot_metric,*phase_max);
	//#endif
	
      }
    }
  }

  return(0);
}

