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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "defs.h"
#include "SIMULATION/RF/defs.h"
#include "PHY/extern.h"

//#define DEBUG_CH
uint8_t multipath_channel_nosigconv(channel_desc_t *desc)
{

  random_channel(desc,0);
  return(1);
}

//#define CHANNEL_SSE
#ifdef CHANNEL_SSE
void multipath_channel(channel_desc_t *desc,
                       double tx_sig_re[2][30720*2],
                       double tx_sig_im[2][30720*2],
                       double rx_sig_re[2][30720*2],
                       double rx_sig_im[2][30720*2],
                       uint32_t length,
                       uint8_t keep_channel)
{

  int i,ii,j,l;
  int length1, length2, tail;
  __m128d rx_tmp128_re_f,rx_tmp128_im_f,rx_tmp128_re,rx_tmp128_im, rx_tmp128_1,rx_tmp128_2,rx_tmp128_3,rx_tmp128_4,tx128_re,tx128_im,ch128_x,ch128_y,pathloss128;

  double path_loss = pow(10,desc->path_loss_dB/20);
  int dd = abs(desc->channel_offset);

  pathloss128 = _mm_set1_pd(path_loss);

#ifdef DEBUG_CH
  printf("[CHANNEL] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %d, len %d \n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,dd,desc->channel_length);
#endif

  if (keep_channel) {
    // do nothing - keep channel
  } else {
    random_channel(desc,0);
  }

  start_meas(&desc->convolution);

#ifdef DEBUG_CH

  for (l = 0; l<(int)desc->channel_length; l++) {
    printf("%p (%f,%f) ",desc->ch[0],desc->ch[0][l].x,desc->ch[0][l].y);
  }

  printf("\n");
#endif

  tail = ((int)length-dd)%2;

  if(tail)
    length1 = ((int)length-dd)-1;
  else
    length1 = ((int)length-dd);

  length2 = length1/2;

  for (i=0; i<length2; i++) { //
    for (ii=0; ii<desc->nb_rx; ii++) {
      // rx_tmp.x = 0;
      // rx_tmp.y = 0;
      rx_tmp128_re_f = _mm_setzero_pd();
      rx_tmp128_im_f = _mm_setzero_pd();

      for (j=0; j<desc->nb_tx; j++) {
        for (l = 0; l<(int)desc->channel_length; l++) {
          if ((i>=0) && (i-l)>=0) { //SIMD correct only if length1 > 2*channel_length...which is almost always satisfied
            // tx.x = tx_sig_re[j][i-l];
            // tx.y = tx_sig_im[j][i-l];
            tx128_re = _mm_loadu_pd(&tx_sig_re[j][2*i-l]); // tx_sig_re[j][i-l+1], tx_sig_re[j][i-l]
            tx128_im = _mm_loadu_pd(&tx_sig_im[j][2*i-l]);
          } else {
            //tx.x =0;
            //tx.y =0;
            tx128_re = _mm_setzero_pd();
            tx128_im = _mm_setzero_pd();
          }

          ch128_x = _mm_set1_pd(desc->ch[ii+(j*desc->nb_rx)][l].x);
          ch128_y = _mm_set1_pd(desc->ch[ii+(j*desc->nb_rx)][l].y);
          //  rx_tmp.x += (tx.x * desc->ch[ii+(j*desc->nb_rx)][l].x) - (tx.y * desc->ch[ii+(j*desc->nb_rx)][l].y);
          //  rx_tmp.y += (tx.y * desc->ch[ii+(j*desc->nb_rx)][l].x) + (tx.x * desc->ch[ii+(j*desc->nb_rx)][l].y);
          rx_tmp128_1 = _mm_mul_pd(tx128_re,ch128_x);
          rx_tmp128_2 = _mm_mul_pd(tx128_re,ch128_y);
          rx_tmp128_3 = _mm_mul_pd(tx128_im,ch128_x);
          rx_tmp128_4 = _mm_mul_pd(tx128_im,ch128_y);
          rx_tmp128_re = _mm_sub_pd(rx_tmp128_1,rx_tmp128_4);
          rx_tmp128_im = _mm_add_pd(rx_tmp128_2,rx_tmp128_3);
          rx_tmp128_re_f = _mm_add_pd(rx_tmp128_re_f,rx_tmp128_re);
          rx_tmp128_im_f = _mm_add_pd(rx_tmp128_im_f,rx_tmp128_im);
        } //l
      }  // j

      //rx_sig_re[ii][i+dd] = rx_tmp.x*path_loss;
      //rx_sig_im[ii][i+dd] = rx_tmp.y*path_loss;
      rx_tmp128_re_f = _mm_mul_pd(rx_tmp128_re_f,pathloss128);
      rx_tmp128_im_f = _mm_mul_pd(rx_tmp128_im_f,pathloss128);
      _mm_storeu_pd(&rx_sig_re[ii][2*i+dd],rx_tmp128_re_f); // max index: length-dd -1 + dd = length -1
      _mm_storeu_pd(&rx_sig_im[ii][2*i+dd],rx_tmp128_im_f);
      /*
      if ((ii==0)&&((i%32)==0)) {
      printf("%p %p %f,%f => %e,%e\n",rx_sig_re[ii],rx_sig_im[ii],rx_tmp.x,rx_tmp.y,rx_sig_re[ii][i-dd],rx_sig_im[ii][i-dd]);
      }
      */
      //rx_sig_re[ii][i] = sqrt(.5)*(tx_sig_re[0][i] + tx_sig_re[1][i]);
      //rx_sig_im[ii][i] = sqrt(.5)*(tx_sig_im[0][i] + tx_sig_im[1][i]);

    } // ii
  } // i

  stop_meas(&desc->convolution);

}

#else
void multipath_channel(channel_desc_t *desc,
                       double *tx_sig_re[2],
                       double *tx_sig_im[2],
                       double *rx_sig_re[2],
                       double *rx_sig_im[2],
                       uint32_t length,
                       uint8_t keep_channel)
{

  int i,ii,j,l;
  struct complex rx_tmp,tx;

  double path_loss = pow(10,desc->path_loss_dB/20);
  int dd;
  dd = abs(desc->channel_offset);

#ifdef DEBUG_CH
  printf("[CHANNEL] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %d, len %d \n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,dd,desc->channel_length);
#endif

  if (keep_channel) {
    // do nothing - keep channel
  } else {
    random_channel(desc,0);
  }

#ifdef DEBUG_CH

  for (l = 0; l<(int)desc->channel_length; l++) {
    printf("%p (%f,%f) ",desc->ch[0],desc->ch[0][l].x,desc->ch[0][l].y);
  }

  printf("\n");
#endif

  for (i=0; i<((int)length-dd); i++) {
    for (ii=0; ii<desc->nb_rx; ii++) {
      rx_tmp.x = 0;
      rx_tmp.y = 0;

      for (j=0; j<desc->nb_tx; j++) {
        for (l = 0; l<(int)desc->channel_length; l++) {
          if ((i>=0) && (i-l)>=0) {
            tx.x = tx_sig_re[j][i-l];
            tx.y = tx_sig_im[j][i-l];
          } else {
            tx.x =0;
            tx.y =0;
          }

          rx_tmp.x += (tx.x * desc->ch[ii+(j*desc->nb_rx)][l].x) - (tx.y * desc->ch[ii+(j*desc->nb_rx)][l].y);
          rx_tmp.y += (tx.y * desc->ch[ii+(j*desc->nb_rx)][l].x) + (tx.x * desc->ch[ii+(j*desc->nb_rx)][l].y);
        } //l
      }  // j

      rx_sig_re[ii][i+dd] = rx_tmp.x*path_loss;
      rx_sig_im[ii][i+dd] = rx_tmp.y*path_loss;
      /*
      if ((ii==0)&&((i%32)==0)) {
      printf("%p %p %f,%f => %e,%e\n",rx_sig_re[ii],rx_sig_im[ii],rx_tmp.x,rx_tmp.y,rx_sig_re[ii][i-dd],rx_sig_im[ii][i-dd]);
      }
      */
      //rx_sig_re[ii][i] = sqrt(.5)*(tx_sig_re[0][i] + tx_sig_re[1][i]);
      //rx_sig_im[ii][i] = sqrt(.5)*(tx_sig_im[0][i] + tx_sig_im[1][i]);

    } // ii
  } // i
}
#endif
void multipath_channel_freq(channel_desc_t *desc,
                       double *tx_sig_re[2],
                       double *tx_sig_im[2],
                       double *rx_sig_re[2],
                       double *rx_sig_im[2],
                       uint32_t length,
                       uint8_t keep_channel,
		       uint8_t eNB_id,
		       uint8_t UE_id,
		       uint8_t CC_id,
		       uint8_t th_id)
{

  int ii,j,k,f,f2;
  struct complex rx_tmp;

  double path_loss = pow(10,desc->path_loss_dB/20);
  int dd;
  dd = abs(desc->channel_offset);

  int nb_rb, n_samples, ofdm_symbol_size, symbols_per_tti;
  nb_rb=PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_DL;
  n_samples=PHY_vars_UE_g[UE_id][CC_id]->frame_parms.N_RB_DL*12+1;
  ofdm_symbol_size=length/PHY_vars_UE_g[UE_id][CC_id]->frame_parms.symbols_per_tti;
  symbols_per_tti=length/PHY_vars_UE_g[UE_id][CC_id]->frame_parms.ofdm_symbol_size;

  //FILE *file;
  //file = fopen("multipath.txt","w");

#ifdef DEBUG_CH
  printf("[CHANNEL_FREQ] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %d, len %d \n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,dd,desc->channel_length);
#endif		
  printf("[CHANNEL_FREQ] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %d, len %d , symbols tti %d\n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,dd,desc->channel_length,symbols_per_tti);

  if (keep_channel) {
  	// do nothing - keep channel
  } else {
  	random_channel(desc,0);//Find a(l)
  	freq_channel(desc,nb_rb,n_samples);//Find desc->chF
  	//freq_channel_prach(desc,nb_rb,n_samples,1,44);//Find desc->chF
  }
  for (k=0;k<symbols_per_tti;k++){//k = 0-13  normal cyclic prefix
	f2 = 0;	  
	for (f=0;f<ofdm_symbol_size; f++) {//f2 = 0-1023 for 10 Mhz BW
		for (ii=0; ii<desc->nb_rx; ii++) {
			rx_tmp.x = 0;
			rx_tmp.y = 0;
			if (f<=(n_samples>>1) && f>0)
			{
				for (j=0; j<desc->nb_tx; j++) {	
					//first n_samples>>1 samples of each frequency ofdm symbol out of ofdm_symbol_size
					//RX_RE(k) += TX_RE(k).chF(k).x	- TX_IM(k).chF(k).y	
					//RX_IM(k) += TX_IM(k).chF(k).x + TX_RE(k).chF(k).y
					rx_tmp.x += (tx_sig_re[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f+(n_samples>>1)-1].x)
						     -(tx_sig_im[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f+(n_samples>>1)-1].y);
					rx_tmp.y += (tx_sig_im[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f+(n_samples>>1)-1].x)
						     +(tx_sig_re[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f+(n_samples>>1)-1].y);
				}  // j    
				rx_sig_re[ii][f+k*ofdm_symbol_size] =  rx_tmp.x*path_loss;
				rx_sig_im[ii][f+k*ofdm_symbol_size] =  rx_tmp.y*path_loss;
			}
			else if (f>=ofdm_symbol_size-(n_samples>>1))
			{
				for (j=0; j<desc->nb_tx; j++) {	
					//last n_samples>>1 samples of each frequency ofdm symbol out of ofdm_symbol_size
					//RX_RE(k) += TX_RE(k).chF(k).x - TX_IM(k).chF(k).y
					//RX_IM(k) += TX_IM(k).chF(k).x + TX_RE(k).chF(k).y
					rx_tmp.x += (tx_sig_re[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f2].x)
						     -(tx_sig_im[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f2].y);
					rx_tmp.y += (tx_sig_im[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f2].x)
						     +(tx_sig_re[j][f+k*ofdm_symbol_size] * desc->chF[ii+(j*desc->nb_rx)][f2].y);
				}  // j    
				rx_sig_re[ii][f+k*ofdm_symbol_size] = rx_tmp.x*path_loss;
				rx_sig_im[ii][f+k*ofdm_symbol_size] = rx_tmp.y*path_loss;
				f2++;
			}
			else
			{
				rx_sig_re[ii][f+k*ofdm_symbol_size] =  0;
				rx_sig_im[ii][f+k*ofdm_symbol_size] =  0;
			}

			//fprintf(file,"%d\t%d\t%d\t%e\t%e\t%e\t%e\t%e\t%e\n",f,f2,k,tx_sig_re[ii][f+k*ofdm_symbol_size],tx_sig_im[ii][f+k*ofdm_symbol_size],rx_sig_re[ii][f+k*ofdm_symbol_size],rx_sig_im[ii][f+k*ofdm_symbol_size],desc->chF[0][f].x,desc->chF[0][f].y);
		//fflush(file);
		//printf("number of taps%d\n",desc->channel_length); 
		} // ii
	} // f,f2,f3
  }//k	 
//fclose(file);    	
}
void multipath_channel_freq_test(channel_desc_t *desc,
                       double *tx_sig_re[2],
                       double *tx_sig_im[2],
                       double *rx_sig_re[2],
                       double *rx_sig_im[2],
                       uint32_t length,
                       uint8_t keep_channel)
{

  int ii,k,f;

  double path_loss = pow(10,desc->path_loss_dB/20);
  int dd;
  dd = abs(desc->channel_offset);

  //int nb_rb, n_samples, ofdm_symbol_size, symbols_per_tti;
  //nb_rb=PHY_vars_UE_g[0][0]->frame_parms.N_RB_DL;
  //n_samples=PHY_vars_UE_g[0][0]->frame_parms.N_RB_DL*12+1;
  int ofdm_symbol_size=length/PHY_vars_UE_g[0][0]->frame_parms.symbols_per_tti;
  int symbols_per_tti=length/PHY_vars_UE_g[0][0]->frame_parms.ofdm_symbol_size;


  printf("[CHANNEL_FREQ] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, dd %d, len %d , symbols tti %d\n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,dd,desc->channel_length,symbols_per_tti);


  for (k=0;k<symbols_per_tti;k++){//k = 0-13  normal cyclic prefix	  
	for (f=0;f<ofdm_symbol_size; f++) {//f2 = 0-1024 for 10 Mhz
		for (ii=0; ii<desc->nb_rx; ii++) {

			{
				rx_sig_re[ii][f+k*ofdm_symbol_size] =  tx_sig_re[ii][f+k*ofdm_symbol_size]*path_loss;
				rx_sig_im[ii][f+k*ofdm_symbol_size] =  tx_sig_im[ii][f+k*ofdm_symbol_size]*path_loss;
			}

		} // ii
	} // f
  }//k	     	
}
void multipath_channel_prach(channel_desc_t *desc,
                       double *tx_sig_re[2],
                       double *tx_sig_im[2],
                       double *rx_sig_re[2],
                       double *rx_sig_im[2],
		       uint32_t length,
                       uint8_t keep_channel,
		       uint8_t eNB_id,
		       uint8_t UE_id,
		       uint8_t CC_id,
		       uint8_t th_id,
		       uint8_t subframe)
{
  LTE_DL_FRAME_PARMS* const fp      = &PHY_vars_UE_g[UE_id][CC_id]->frame_parms;
  int prach_samples;
  lte_frame_type_t frame_type = PHY_vars_UE_g[UE_id][CC_id]->frame_parms.frame_type;
  uint8_t prach_ConfigIndex   = PHY_vars_UE_g[UE_id][CC_id]->frame_parms.prach_config_common.prach_ConfigInfo.prach_ConfigIndex;
  uint8_t prach_fmt = get_prach_fmt(prach_ConfigIndex,frame_type);
  int n_ra_prb;
  int ii,j,k,f,l;
  struct complex rx_tmp;
  double delta_f;
  prach_samples = (prach_fmt<4)?13+839+12:3+139+2;
  double path_loss = pow(10,desc->path_loss_dB/20);
  int nb_rb, n_samples, ofdm_symbol_size, symbols_per_tti;
  
  n_ra_prb = get_prach_prb_offset(fp, PHY_vars_UE_g[UE_id][CC_id]->prach_resources[eNB_id]->ra_TDD_map_index, PHY_vars_UE_g[UE_id][CC_id]->proc.proc_rxtx[th_id].frame_tx);
  nb_rb=fp->N_RB_DL;
  n_samples=fp->N_RB_DL*12+1;
  ofdm_symbol_size=fp->ofdm_symbol_size;
  symbols_per_tti=fp->symbols_per_tti;
  delta_f = (prach_fmt<4)?nb_rb*180000/((n_samples-1)*12):nb_rb*180000/((n_samples-1)*2);
  printf("prach_samples %d, n_ra_prb %d, delta_f %e, prach_fmt %d\n",prach_samples,get_prach_prb_offset(fp, PHY_vars_UE_g[UE_id][CC_id]->prach_resources[0]->ra_TDD_map_index, PHY_vars_eNB_g[0][0]->proc.frame_prach), delta_f,prach_fmt);
  #ifdef DEBUG_CH
  printf("[CHANNEL_PRACH] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, len %d \n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,desc->channel_length);
#endif		
  printf("[CHANNEL_PRACH] keep = %d : path_loss = %g (%f), nb_rx %d, nb_tx %d, len %d , symbols tti %d\n",keep_channel,path_loss,desc->path_loss_dB,desc->nb_rx,desc->nb_tx,desc->channel_length,symbols_per_tti);
   		if (keep_channel) {
		// do nothing - keep channel
		} else {
		random_channel(desc,0);//Find a(l)
		freq_channel_prach(desc,nb_rb,n_samples,prach_fmt,n_ra_prb);//Find desc->chF_prach
		}	
			for (f=0;f<prach_samples; f++) {
				rx_tmp.x = 0;
				rx_tmp.y = 0;
				for (ii=0; ii<desc->nb_rx; ii++) {
					for (j=0; j<desc->nb_tx; j++) {		
						//RX_RE(k) = TX_RE(k).chF(k).x	- TX_IM(k).chF(k).y	
						 rx_tmp.x += (tx_sig_re[ii][f] * desc->chF_prach[ii+(j*desc->nb_rx)][f+(prach_fmt<4)?13:3].x)-(tx_sig_im[ii][f] * desc->chF_prach[ii+(j*desc->nb_rx)][f+(prach_fmt<4)?13:3].y);
						//RX_IM(k) = TX_IM(k).chF(k).x + TX_RE(k).chF(k).y
						 rx_tmp.y += (tx_sig_im[ii][f] * desc->chF_prach[ii+(j*desc->nb_rx)][f+(prach_fmt<4)?13:3].x)+(tx_sig_re[ii][f] * desc->chF_prach[ii+(j*desc->nb_rx)][f+(prach_fmt<4)?13:3].y);
				         }  // j 
				//printf("[multipath prach] k: %d\n",k/2);
				rx_sig_re[ii][f]  =   rx_tmp.x*path_loss;
				rx_sig_im[ii][f]  =   rx_tmp.y*path_loss;	
				} // ii
			} // f
}


