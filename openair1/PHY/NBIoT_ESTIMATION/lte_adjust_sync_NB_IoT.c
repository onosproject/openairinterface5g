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

//#include "PHY/types.h"
#include "PHY/defs_L1_NB_IoT.h"
//#include "PHY/extern.h"

//#include "UTIL/LOG/vcd_signal_dumper.h"

#define DEBUG_PHY

int NB_IoT_est_timing_advance_pusch(PHY_VARS_eNB_NB_IoT* eNB,uint8_t UE_id)
{
  static int first_run=1;
  static int max_pos_fil2=0;
  int temp, i, aa, max_pos=0, max_val=0;
  short Re,Im,coef=24576;
  short ncoef = 32768 - coef;

  NB_IoT_DL_FRAME_PARMS *frame_parms = &eNB->frame_parms;
  NB_IoT_eNB_PUSCH *eNB_pusch_vars = eNB->pusch_vars[UE_id];
  int32_t **ul_ch_estimates_time=  eNB_pusch_vars->drs_ch_estimates_time[0];
  uint8_t cyclic_shift = 0;
  int sync_pos = (frame_parms->ofdm_symbol_size-cyclic_shift*frame_parms->ofdm_symbol_size/12)%(frame_parms->ofdm_symbol_size);


  for (i = 0; i < frame_parms->ofdm_symbol_size; i++) {
    temp = 0;

    for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {
      Re = ((int16_t*)ul_ch_estimates_time[aa])[(i<<1)];
      Im = ((int16_t*)ul_ch_estimates_time[aa])[1+(i<<1)];
      temp += (Re*Re/2) + (Im*Im/2);
    }

    if (temp > max_val) {
      max_pos = i;
      max_val = temp;
    }
  }

  if (max_pos>frame_parms->ofdm_symbol_size/2)
    max_pos = max_pos-frame_parms->ofdm_symbol_size;

  // filter position to reduce jitter
  if (first_run == 1) {
    first_run=0;
    max_pos_fil2 = max_pos;
  } else
    max_pos_fil2 = ((max_pos_fil2 * coef) + (max_pos * ncoef)) >> 15;

#ifdef DEBUG_PHY
  LOG_D(PHY,"frame %d: max_pos = %d, max_pos_fil = %d, sync_pos=%d\n",eNB->proc.frame_rx,max_pos,max_pos_fil2,sync_pos);
#endif //DEBUG_PHY

  return(max_pos_fil2-sync_pos);
}
