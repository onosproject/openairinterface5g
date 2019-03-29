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

#include "PHY/defs_UE.h"
#include "PHY/defs_nr_UE.h"
#include "modulation_UE.h"
#include "PHY/LTE_ESTIMATION/lte_estimation.h"
#include "PHY/NR_UE_ESTIMATION/nr_estimation.h"

#define DEBUG_FEP(a...)

#define SOFFSET 0

/*#ifdef LOG_I
#undef LOG_I
#define LOG_I(A,B...) printf(A)
#endif*/

int nr_slot_fep(PHY_VARS_NR_UE *ue,
                unsigned char symbol,
                unsigned char Ns,
                int sample_offset,
                int no_prefix,
                NR_CHANNEL_EST_t channel) {
  NR_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
  NR_UE_COMMON *common_vars   = &ue->common_vars;
  uint8_t eNB_id = 0;
  unsigned char aa;
  unsigned int nb_prefix_samples;
  unsigned int nb_prefix_samples0;
  if (ue->is_synchronized) {
    nb_prefix_samples = (no_prefix ? 0 : frame_parms->nb_prefix_samples);
    nb_prefix_samples0 = (no_prefix ? 0 : frame_parms->nb_prefix_samples0);
  }
  else {
    nb_prefix_samples = (no_prefix ? 0 : frame_parms->nb_prefix_samples);
    nb_prefix_samples0 = (no_prefix ? 0 : frame_parms->nb_prefix_samples);
  }
  //unsigned int subframe_offset;//,subframe_offset_F;
  unsigned int slot_offset;
  //int i;
  unsigned int frame_length_samples = frame_parms->samples_per_subframe * 10;
  unsigned int rx_offset;
  NR_UE_PDCCH *pdcch_vars  = ue->pdcch_vars[ue->current_thread_id[Ns]][0];
  uint16_t coreset_start_subcarrier = frame_parms->first_carrier_offset;//+((int)floor(frame_parms->ssb_start_subcarrier/NR_NB_SC_PER_RB)+pdcch_vars->coreset[0].rb_offset)*NR_NB_SC_PER_RB;
  uint16_t nb_rb_coreset = 0;
  uint16_t bwp_start_subcarrier = frame_parms->first_carrier_offset;//+516;
  uint16_t nb_rb_pdsch = 50;
  uint8_t p=0;
  uint8_t l0 = pdcch_vars->coreset[0].duration;
  uint64_t coreset_freq_dom  = pdcch_vars->coreset[0].frequencyDomainResources;

  for (int i = 0; i < 45; i++) {
    if (((coreset_freq_dom & 0x1FFFFFFFFFFF) >> i) & 0x1) nb_rb_coreset++;
  }

  nb_rb_coreset = 6 * nb_rb_coreset;
  //printf("corset duration %d nb_rb_coreset %d\n", l0, nb_rb_coreset);
  void (*dft)(int16_t *,int16_t *, int);
  struct l_s {
    int key;
    void (*val)(int16_t *,int16_t *, int);
  }
  listFunc[] = {
    {128, dft128},{256,dft256},{512,dft512},{1024,dft1024},
    {1536,dft1536},{2048,dft2048},{4096,dft4096},{8192,dft8192}
  };
  findInList(frame_parms->ofdm_symbol_size, dft, listFunc, struct l_s);

  if (no_prefix) {
    slot_offset = frame_parms->ofdm_symbol_size * (frame_parms->symbols_per_slot) * (Ns);
  } else {
    slot_offset = (frame_parms->samples_per_slot) * (Ns);
  }

  int32_t **rxdf=common_vars->common_vars_rx_data_per_thread[ue->current_thread_id[Ns]].rxdataF;
  AssertFatal (Ns>=0 && Ns<20, "slot_fep: Ns must be between 0 and 19\n");

  for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {
    memset(&rxdf[aa][frame_parms->ofdm_symbol_size*symbol],0,frame_parms->ofdm_symbol_size*sizeof(int));

    rx_offset = sample_offset + slot_offset + nb_prefix_samples0 - SOFFSET+
                (frame_parms->ofdm_symbol_size+nb_prefix_samples)*symbol;
    rx_offset %= frame_length_samples;
    // Align with 256 bit
    //    rx_offset = rx_offset&0xfffffff8;
    LOG_D(PHY,"slot_fep: slot %d, symbol %d, nb_prefix_samples %d, nb_prefix_samples0 %d, slot_offset %d,  sample_offset %d,rx_offset %d, frame_length_samples %d\n",
          Ns, symbol, nb_prefix_samples,nb_prefix_samples0,slot_offset,sample_offset,rx_offset,frame_length_samples);

    if (rx_offset > (frame_length_samples - frame_parms->ofdm_symbol_size)) {
      if (rx_offset+frame_parms->ofdm_symbol_size > frame_parms->samples_per_subframe*10+2048)
	LOG_E(PHY,"rx_offset get out of buffer\n");
      else
	LOG_W(PHY,"DFT on a buffer outside regular space, code looks wrong\n");
      memcpy((short *)&common_vars->rxdataTime[aa][frame_length_samples],
             (short *)&common_vars->rxdataTime[aa][0],
             frame_parms->ofdm_symbol_size*sizeof(int));
    }
    if ((rx_offset&7)!=0) {  // if input to dft is not 256-bit aligned, issue for size 6,15 and 25 PRBs
      LOG_D(PHY,"dft on unaligned buffer: %d offset\n", rx_offset);
      int tmp_dft_in[8192] __attribute__ ((aligned (32)));
      memcpy((void *)tmp_dft_in,
             (void *)&common_vars->rxdataTime[aa][rx_offset],
             frame_parms->ofdm_symbol_size*sizeof(int));
      UE_meas(ue->rx_dft_stats,
              dft((int16_t *)tmp_dft_in,
                  (int16_t *)&rxdf[aa][frame_parms->ofdm_symbol_size*symbol],1));
    } else { // use dft input from RX buffer directly
      UE_meas(ue->rx_dft_stats,
              dft((int16_t *)&common_vars->rxdataTime[aa][rx_offset],
                  (int16_t *)&rxdf[aa][frame_parms->ofdm_symbol_size*symbol],1));
    }

    //  if (ue->frame <100)
    DEBUG_FEP("slot_fep: frame %d: symbol %d rx_offset %d\n", ue->proc.proc_rxtx[(Ns)&1].frame_rx, symbol,rx_offset);
  }

  if (ue->perfect_ce == 0) {
    switch(channel) {
      case NR_PBCH_EST:
        break;

      case NR_PDCCH_EST:
        DEBUG_FEP("PDCCH Channel estimation eNB %d, aatx %d, slot %d, symbol %d start_sc %d\n",eNB_id,aa,Ns,l,coreset_start_subcarrier);
        UE_meas(ue->dlsch_channel_estimation_stats,
                nr_pdcch_channel_estimation(ue,eNB_id,
                                            Ns,
                                            symbol,
                                            coreset_start_subcarrier,
                                            nb_rb_coreset));
        break;

      case NR_PDSCH_EST:
        DEBUG_FEP("Channel estimation eNB %d, aatx %d, slot %d, symbol %d\n",eNB_id,aa,Ns,l);
        ue->frame_parms.nushift =  (p>>1)&1;

        if (symbol ==l0)
          UE_meas(ue->dlsch_channel_estimation_stats,
                  nr_pdsch_channel_estimation(ue,eNB_id,
                                              Ns,
                                              p,
                                              symbol,
                                              bwp_start_subcarrier,
                                              nb_rb_pdsch));

        break;

      case NR_SSS_EST:
        break;

      default:
        LOG_E(PHY,"[UE][FATAL] Unknown channel format %d\n",channel);
        return(-1);
        break;
    }
  }

  DEBUG_FEP("slot_fep: done\n");
  return(0);
}

