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

#ifndef __LTE_ESTIMATION_DEFS_NB_IOT__H__
#define __LTE_ESTIMATION_DEFS_NB_IOT__H__

#include "PHY/defs_NB_IoT.h"

/*
int lte_est_timing_advance(NB_IoT_DL_FRAME_PARMS *frame_parms,
                           NB_IoT_eNB_SRS *lte_eNb_srs,
                           unsigned int *eNb_id,
                           unsigned char clear,
                           unsigned char number_of_cards,
                           short coef);
*/

////////// Vincent: NB-IoT DL synchronization //////////////////////////////////////////////////

int lte_sync_time_init_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms ); 

void lte_sync_time_free_NB_IoT(void); 

int lte_sync_time_NB_IoT(int **rxdata, ///rx data in time domain
                  NB_IoT_DL_FRAME_PARMS *frame_parms,
                  int *eNB_id); 

void lte_sync_timefreq_NB_IoT(PHY_VARS_UE_NB_IoT *ue,int band,unsigned int DL_freq); 


////////////////////////////////////////////////////////////////////////////////////////////////

int NB_IoT_est_timing_advance_pusch(PHY_VARS_eNB_NB_IoT* phy_vars_eNB,module_id_t UE_id);



int lte_ul_channel_estimation_NB_IoT(PHY_VARS_eNB_NB_IoT      *phy_vars_eNB,
			      					 eNB_rxtx_proc_NB_IoT_t   *proc,
                              		 module_id_t              eNB_id,
                              		 module_id_t              UE_id,
                              		 uint8_t                  l,
                              		 uint8_t                  Ns,
                              		 uint8_t                  cooperation_flag);

////////// Vincent: NB-IoT specific adapted function for channel estimation ////////////////////

int ul_channel_estimation_NB_IoT(PHY_VARS_eNB_NB_IoT      *eNB,
                                         eNB_rxtx_proc_NB_IoT_t   *proc,
                                         uint8_t                  eNB_id,
                                         uint8_t                  UE_id,
                                         unsigned char            l,
                                         unsigned char            Ns,
                                         uint8_t                  cooperation_flag); 
////////////////////////////////////////////////////////////////////////////////////////////////

int16_t lte_ul_freq_offset_estimation_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
                                      		 int32_t *ul_ch_estimates,
                                      		 uint16_t nb_rb);

void freq_equalization_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
                         		  int **rxdataF_comp,
                         		  int **ul_ch_mag,
                         		  int **ul_ch_mag_b,
                         		  unsigned char symbol,
                       	      unsigned short Msc_RS,
                              unsigned char Qm);

/** @} */
#endif
