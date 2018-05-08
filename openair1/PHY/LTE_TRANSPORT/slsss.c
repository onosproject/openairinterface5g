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
#include "PHY/defs.h"
#include "defs.h"
#include "PHY/extern.h"

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
    d = &d0_sss[62*(Nid2 + (Nid1*3))];
  else 
    d = &d5_sss[62*(Nid2 + (Nid1*3))];

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
		 int Nid2)
{

  int16_t *pss;
  int16_t *pss0_ext2,*sss0_ext2,*sss0_ext3,*sss1_ext3,tmp_re,tmp_im,tmp0_re2,tmp0_im2,tmp1_re2,tmp1_im2;

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

  sss0_ext3 = (int16_t*)&sss0_ext[0][5];
  sss1_ext3 = (int16_t*)&sss1_ext[0][5];

  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

    sss0_ext2 = (int16_t*)&sss0_ext[aarx][5];
    pss0_ext2 = (int16_t*)&pss0_ext[aarx][5];
    sss1_ext2 = (int16_t*)&sss1_ext[aarx][5];
    pss1_ext2 = (int16_t*)&pss1_ext[aarx][5];

    for (i=0; i<62; i++) {

      // This is H*(PSS) = R* \cdot PSS
      tmp_re = (int16_t)((((pss0_ext2[i<<1]+pss1_ext2[i<<1]) * (int32_t)pss[i<<1])>>15)     + (((pss0_ext2[1+(i<<1)]+pss1_ext2[1+(i<<1)]) * (int32_t)pss[1+(i<<1)])>>15));
      tmp_im = (int16_t)((((pss0_ext2[i<<1]+pss1_ext2[i<<1]) * (int32_t)pss[1+(i<<1)])>>15) - (((pss0_ext2[1+(i<<1)]+pss1_ext2[1+(i<<1)]) * (int32_t)pss[(i<<1)])>>15));
      //      printf("H*(%d,%d) : (%d,%d)\n",aarx,i,tmp_re,tmp_im);
      // This is R(SSS0) \cdot H*(PSS)
      tmp0_re2 = (int16_t)(((tmp_re * (int32_t)sss0_ext2[i<<1])>>15)  - 
			   ((tmp_im * (int32_t)sss0_ext2[1+(i<<1)])>>15));
      tmp0_im2 = (int16_t)(((tmp_re * (int32_t)sss0_ext2[1+(i<<1)])>>15) + 
			   ((tmp_im * (int32_t)sss0_ext2[(i<<1)])>>15));
      // This is R(SSS1) \cdot H*(PSS)
      tmp1_re2 = (int16_t)(((tmp_re * (int32_t)sss1_ext2[i<<1])>>15)  - 
			   ((tmp_im * (int32_t)sss1_ext2[1+(i<<1)])>>15));
      tmp1_im2 = (int16_t)(((tmp_re * (int32_t)sss1_ext2[1+(i<<1)])>>15) + 
			   ((tmp_im * (int32_t)sss1_ext2[(i<<1)])>>15));
    
      //      printf("SSSi(%d,%d) : (%d,%d)\n",aarx,i,sss_ext2[i<<1],sss_ext2[1+(i<<1)]);
      //      printf("SSSo(%d,%d) : (%d,%d)\n",aarx,i,tmp_re2,tmp_im2);
      // MRC on RX antennas
      if (aarx==0) {
        sss0_ext3[i<<1]      = tmp0_re2;
        sss0_ext3[1+(i<<1)]  = tmp0_im2;
        sss1_ext3[i<<1]      = tmp1_re2;
        sss1_ext3[1+(i<<1)]  = tmp1_im2;
      } else {
        sss0_ext3[i<<1]      += tmp0_re2;
        sss0_ext3[1+(i<<1)]  += tmp0_im2;
        sss1_ext3[i<<1]      += tmp1_re2;
        sss1_ext3[1+(i<<1)]  += tmp1_im2;
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


  int rx_offset = (subframe*frame_parms->samples_per_tti) + frame_parms->ofdm_symbol_size-3*12;
  uint8_t pss0_symb,pss1_symb,sss0_symb,sss1_symb;

  int32_t **rxdataF;


  //LOG_I(PHY,"do_pss_sss_extract subframe %d \n",subframe);
  for (aarx=0; aarx<frame_parms->nb_antennas_rx; aarx++) {

    if (frame_parms->frame_type == FDD) {
      pss0_symb = 1;
      pss1_symb = 2;
      sss0_symb = 10;
      sss1_symb = 11;
      rxdataF  =  ue->common_vars.common_vars_rx_data_per_thread[ue->current_thread_id[subframe]].rxdataF;
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
          sss0_rxF       = &rxdataF[aarx][(1 + (sss0_symb*(frame_parms->ofdm_symbol_size)))];
          pss0_rxF       = &rxdataF[aarx][(1 + (pss0_symb*(frame_parms->ofdm_symbol_size)))];
          sss1_rxF       = &rxdataF[aarx][(1 + (sss1_symb*(frame_parms->ofdm_symbol_size)))];
          pss1_rxF       = &rxdataF[aarx][(1 + (pss1_symb*(frame_parms->ofdm_symbol_size)))];
        }
        else
	  AssertFatal(0,"");
        }
      }

      for (i=0; i<12; i++) {
        if (doPss) {
	  pss0_rxF_ext[i]=pss0_rxF[i];
	  pss1_rxF_ext[i]=pss1_rxF[i];
	}
        if (doSss) {
	  sss0_rxF_ext[i]=sss0_rxF[i];
	  sss1_rxF_ext[i]=sss1_rxF[i];
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


int rx_slsss(PHY_VARS_UE *ue,int32_t *tot_metric,uint8_t *phase_max,int Nid2,int subframe_rx)
{

  uint8_t i;
  int32_t pss0_ext[4][72],pss1_ext[4][72];
  int32_t sss0_ext[4][72],sss1_ext[4][72];
  uint8_t phase;
  uint16_t Nid1;
  int16_t *sss0,*sss1;
  LTE_DL_FRAME_PARMS *frame_parms=&ue->frame_parms;
  int32_t metric;
  int16_t *d0,*d5;
  int16_t **rxdata_7_5kHz    = ue->sl_rxdata_7_5kHz;
  int16_t **rxdataF          = ue->sl_rxdataF;

  if (frame_parms->frame_type == FDD) {
#ifdef DEBUG_SSS

    if (frame_parms->Ncp == NORMAL)
      printf("[PHY][UE%d] Doing SSS for FDD Normal Prefix\n",ue->Mod_id);
    else
      printf("[PHY][UE%d] Doing SSS for FDD Extended Prefix\n",ue->Mod_id);

#endif
    // Do FFTs for SSS/PSS
    // SSS
    RU_t ru_tmp;
    memset((void*)&ru_tmp,0,sizeof(RU_t));
    
    memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
    ru_tmp.N_TA_offset=0;
    ru_tmp.common.rxdata = ue->common_vars.rxdata;
    ru_tmp.common.rxdata_7_5kHz = (int32_t**)rxdata_7_5kHz;
    ru_tmp.common.rxdataF = (int32_t**)rxdataF;
    ru_tmp.nb_rx = ue->frame_parms.nb_antennas_rx;
    

    remove_7_5_kHz(&ru_tmp,(subframe_rx<<1));
    remove_7_5_kHz(&ru_tmp,(subframe_rx<<1)+1);
    // PSS
    slot_fep_ul(&ru_tmp,1,0,0);
    slot_fep_ul(&ru_tmp,2,0,0);
    // SSS
    slot_fep_ul(&ru_tmp,11,1,0);
    slot_fep_ul(&ru_tmp,12,1,0);

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
  /*
  write_output("rxsig0.m","rxs0",&ue->common_vars.rxdata[0][0],ue->frame_parms.samples_per_tti,1,1);
  write_output("rxdataF0.m","rxF0",&ue->common_vars.rxdataF[0][0],2*14*ue->frame_parms.ofdm_symbol_size,2,1);
  write_output("pss_ext0.m","pssext0",pss_ext,72,1,1);
  write_output("sss0_ext0.m","sss0ext0",sss0_ext,72,1,1);
  */

  // get conjugated channel estimate from PSS (symbol 6), H* = R* \cdot PSS
  // and do channel estimation and compensation based on PSS

  slpss_ch_est(ue,
	       pss0_ext,
	       sss0_ext,
	       pss1_ext,
	       sss1_ext,
	       Nid2);

  //  write_output("sss0_comp0.m","sss0comp0",sss0_ext,72,1,1);

  // now do the SSS detection based on the precomputed sequences in PHY/LTE_TRANSPORT/sss.h

  *tot_metric = -99999999;


  sss0 = (int16_t*)&sss0_ext[0][5];
  sss1 = (int16_t*)&sss1_ext[0][5];


  for (phase=0; phase<7; phase++) { // phase offset between PSS and SSS
    for (Nid1 = 0 ; Nid1 <= 167; Nid1++) {  // 168 possible Nid1 values
      metric = 0;
      
      d0 = &d0_sss[62*(Nid2 + (Nid1*3))];
      d5 = &d5_sss[62*(Nid2 + (Nid1*3))];
      // This is the inner product using one particular value of each unknown parameter
      for (i=0; i<62; i++) {
	metric += 
	  (int16_t)(((d0[i]*((((phaseSL_re[phase]*(int32_t)sss0[i<<1])>>19)-((phaseSL_im[phase]*(int32_t)sss0[1+(i<<1)])>>19)))))) + 
	  (int16_t)(((d5[i]*((((phaseSL_re[phase]*(int32_t)sss1[i<<1])>>19)-((phaseSL_im[phase]*(int32_t)sss1[1+(i<<1)])>>19))))));
      }
      
      // if the current metric is better than the last save it
      if (metric > *tot_metric) {
	*tot_metric = metric;
	ue->frame_parms.Nid_SL = Nid2+(2*Nid1);
	*phase_max = phase;
#ifdef DEBUG_SSS
	printf("(phase,Nid1) (%d,%d), metric_phase %d tot_metric %d, phase_max %d\n",phase,Nid1,metric,*tot_metric,*phase_max);
#endif
	
      }
    }
  }


  return(0);
}

