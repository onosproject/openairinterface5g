



#include "../impl_defs_lte_NB_IoT.h"
#include "../defs_L1_NB_IoT.h"
#include "phy_init_NB_IoT.h"
#include "common/config/config_paramdesc.h"
#include "openair2/ENB_APP/NB_IoT_paramdef.h"
#include "PHY/phy_vars.h"

PHY_VARS_eNB_NB_IoT* init_lte_eNB_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
                                         uint8_t eNB_id,
                                         uint8_t Nid_cell,
                                         eNB_func_NB_IoT_t node_function,
                                         uint8_t abstraction_flag)
{

  //int i;
  PHY_VARS_eNB_NB_IoT* PHY_vars_eNB = malloc(sizeof(PHY_VARS_eNB_NB_IoT));
  memset(PHY_vars_eNB,0,sizeof(PHY_VARS_eNB_NB_IoT));
  PHY_vars_eNB->Mod_id=eNB_id;
  PHY_vars_eNB->cooperation_flag=0;//cooperation_flag;
  memcpy(&(PHY_vars_eNB->frame_parms), frame_parms, sizeof(NB_IoT_DL_FRAME_PARMS));
  //PHY_vars_eNB->frame_parms.Nid_cell = ((Nid_cell/3)*3)+((eNB_id+Nid_cell)%3);
  //PHY_vars_eNB->frame_parms.nushift = PHY_vars_eNB->frame_parms.Nid_cell%6;
  PHY_vars_eNB->frame_parms.Nid_cell =  Nid_cell;                                             ///////((Nid_cell/3)*3)+((eNB_id+Nid_cell)%3);
  PHY_vars_eNB->frame_parms.nushift = PHY_vars_eNB->frame_parms.Nid_cell%6;
  phy_init_lte_eNB_NB_IoT(PHY_vars_eNB,0,abstraction_flag);

// for NB-IoT testing
//  PHY_vars_eNB->ndlsch_SIB.content_sib1.si_rnti = 0xffff;
//  PHY_vars_eNB->ndlsch_SIB.content_sib23.si_rnti = 0xffff;
////////////////////////////
  
  /*LOG_I(PHY,"init eNB: Node Function %d\n",node_function);
  LOG_I(PHY,"init eNB: Nid_cell %d\n", frame_parms->Nid_cell);
  LOG_I(PHY,"init eNB: frame_type %d,tdd_config %d\n", frame_parms->frame_type,frame_parms->tdd_config);
  LOG_I(PHY,"init eNB: number of ue max %d number of enb max %d number of harq pid max %d\n",
        NUMBER_OF_UE_MAX, NUMBER_OF_eNB_MAX, NUMBER_OF_HARQ_PID_MAX);
  LOG_I(PHY,"init eNB: N_RB_DL %d\n", frame_parms->N_RB_DL);
  LOG_I(PHY,"init eNB: prach_config_index %d\n", frame_parms->prach_config_common.prach_ConfigInfo.prach_ConfigIndex);
  */
/*
  if (node_function >= NGFI_RRU_IF5)
    // For RRU, don't allocate DLSCH/ULSCH Transport channel buffers
    return (PHY_vars_eNB);

*/
  /*
  for (i=0; i<NUMBER_OF_UE_MAX_NB_IoT; i++) {
    LOG_I(PHY,"Allocating Transport Channel Buffers for DLSCH, UE %d\n",i);
    for (j=0; j<2; j++) {
      PHY_vars_eNB->dlsch[i][j] = new_eNB_dlsch(1,8,NSOFT,frame_parms->N_RB_DL,abstraction_flag,frame_parms);
      if (!PHY_vars_eNB->dlsch[i][j]) {
  LOG_E(PHY,"Can't get eNB dlsch structures for UE %d \n", i);
  exit(-1);
      } else {
  LOG_D(PHY,"dlsch[%d][%d] => %p\n",i,j,PHY_vars_eNB->dlsch[i][j]);
  PHY_vars_eNB->dlsch[i][j]->rnti=0;
      }
    }
    
    LOG_I(PHY,"Allocating Transport Channel Buffer for ULSCH, UE %d\n", i);
    PHY_vars_eNB->ulsch[1+i] = new_eNB_ulsch(MAX_TURBO_ITERATIONS,frame_parms->N_RB_UL, abstraction_flag);
    
    if (!PHY_vars_eNB->ulsch[1+i]) {
      LOG_E(PHY,"Can't get eNB ulsch structures\n");
      exit(-1);
    }
    
*/

    // this is the transmission mode for the signalling channels
    // this will be overwritten with the real transmission mode by the RRC once the UE is connected
    PHY_vars_eNB->transmission_mode[0] =  1 ;
/*#ifdef LOCALIZATION
    PHY_vars_eNB->ulsch[1+i]->aggregation_period_ms = 5000; // 5000 milliseconds // could be given as an argument (TBD))
    struct timeval ts;
    gettimeofday(&ts, NULL);
    PHY_vars_eNB->ulsch[1+i]->reference_timestamp_ms = ts.tv_sec * 1000 + ts.tv_usec / 1000;
    int j;
    
    for (j=0; j<10; j++) {
      initialize(&PHY_vars_eNB->ulsch[1+i]->loc_rss_list[j]);
      initialize(&PHY_vars_eNB->ulsch[1+i]->loc_rssi_list[j]);
      initialize(&PHY_vars_eNB->ulsch[1+i]->loc_subcarrier_rss_list[j]);
      initialize(&PHY_vars_eNB->ulsch[1+i]->loc_timing_advance_list[j]);
      initialize(&PHY_vars_eNB->ulsch[1+i]->loc_timing_update_list[j]);
    }
    
    initialize(&PHY_vars_eNB->ulsch[1+i]->tot_loc_rss_list);
    initialize(&PHY_vars_eNB->ulsch[1+i]->tot_loc_rssi_list);
    initialize(&PHY_vars_eNB->ulsch[1+i]->tot_loc_subcarrier_rss_list);
    initialize(&PHY_vars_eNB->ulsch[1+i]->tot_loc_timing_advance_list);
    initialize(&PHY_vars_eNB->ulsch[1+i]->tot_loc_timing_update_list);
#endif*/
 // }
 
  /*
  // ULSCH for RA
  PHY_vars_eNB->ulsch[0] = new_eNB_ulsch(MAX_TURBO_ITERATIONS, frame_parms->N_RB_UL, abstraction_flag);
  
  if (!PHY_vars_eNB->ulsch[0]) {
    LOG_E(PHY,"Can't get eNB ulsch structures\n");
    exit(-1);
  }
  PHY_vars_eNB->dlsch_SI  = new_eNB_dlsch(1,8,NSOFT,frame_parms->N_RB_DL, abstraction_flag, frame_parms);
  LOG_D(PHY,"eNB %d : SI %p\n",eNB_id,PHY_vars_eNB->dlsch_SI);
  PHY_vars_eNB->dlsch_ra  = new_eNB_dlsch(1,8,NSOFT,frame_parms->N_RB_DL, abstraction_flag, frame_parms);
  LOG_D(PHY,"eNB %d : RA %p\n",eNB_id,PHY_vars_eNB->dlsch_ra);
  PHY_vars_eNB->dlsch_MCH = new_eNB_dlsch(1,8,NSOFT,frame_parms->N_RB_DL, 0, frame_parms);
  LOG_D(PHY,"eNB %d : MCH %p\n",eNB_id,PHY_vars_eNB->dlsch_MCH);
  */
  
  PHY_vars_eNB->rx_total_gain_dB=130;
  
 /* for(i=0; i<NUMBER_OF_UE_MAX; i++)
    PHY_vars_eNB->mu_mimo_mode[i].dl_pow_off = 2;
  
  PHY_vars_eNB->check_for_total_transmissions = 0;
  
  PHY_vars_eNB->check_for_MUMIMO_transmissions = 0;
  
  PHY_vars_eNB->FULL_MUMIMO_transmissions = 0;
  
  PHY_vars_eNB->check_for_SUMIMO_transmissions = 0;
  
    PHY_vars_eNB->frame_parms.pucch_config_common.deltaPUCCH_Shift = 1;
*/
  return (PHY_vars_eNB);

}


int phy_init_lte_eNB_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
                     unsigned char is_secondary_eNB,
                     unsigned char abstraction_flag)
{

  // shortcuts
  NB_IoT_DL_FRAME_PARMS* const fp      = &eNB->frame_parms;
  NB_IoT_eNB_COMMON* const common_vars = &eNB->common_vars;
  NB_IoT_eNB_PUSCH** const pusch_vars  = eNB->pusch_vars;
  NB_IoT_eNB_SRS* const srs_vars       = eNB->srs_vars;
  NB_IoT_eNB_PRACH* const prach_vars   = &eNB->prach_vars;

  int i, j, eNB_id, UE_id;
  int re;


  eNB->total_dlsch_bitrate = 0;
  eNB->total_transmitted_bits = 0;
  eNB->total_system_throughput = 0;
  eNB->check_for_MUMIMO_transmissions=0;


  if (eNB->node_function != NGFI_RRU_IF4p5_NB_IoT) {
  //  lte_gold_NB_IoT(fp,eNB->lte_gold_table_NB_IoT,fp->Nid_cell);   ****** uncomment when this function is used - 16/02/2018
  //  generate_pcfich_reg_mapping(fp);
  //  generate_phich_reg_mapping(fp);

    for (UE_id=0; UE_id<NUMBER_OF_UE_MAX_NB_IoT; UE_id++) {
      eNB->first_run_timing_advance[UE_id] =
        1; ///This flag used to be static. With multiple eNBs this does no longer work, hence we put it in the structure. However it has to be initialized with 1, which is performed here.

      // clear whole structure
      bzero( &eNB->UE_stats[UE_id], sizeof(NB_IoT_eNB_UE_stats) );

      eNB->physicalConfigDedicated[UE_id] = NULL;
    }

    eNB->first_run_I0_measurements = 1; ///This flag used to be static. With multiple eNBs this does no longer work, hence we put it in the structure. However it has to be initialized with 1, which is performed here.
  }

  //  for (eNB_id=0; eNB_id<3; eNB_id++) {
  {
    eNB_id=0;
    if (abstraction_flag==0) {

      // TX vars
      if (eNB->node_function != NGFI_RCC_IF4p5_NB_IoT)

        common_vars->txdata[eNB_id]  = (int32_t**)malloc16(fp->nb_antennas_tx*sizeof(int32_t*));
      common_vars->txdataF[eNB_id] = (int32_t **)malloc16(NB_ANTENNA_PORTS_ENB*sizeof(int32_t*));
      common_vars->txdataF_BF[eNB_id] = (int32_t **)malloc16(fp->nb_antennas_tx*sizeof(int32_t*));

      if (eNB->node_function != NGFI_RRU_IF5_NB_IoT) {
        for (i=0; i<NB_ANTENNA_PORTS_ENB; i++) {
          if (i<fp->nb_antenna_ports_eNB || i==5) {
            common_vars->txdataF[eNB_id][i] = (int32_t*)malloc16_clear(fp->ofdm_symbol_size*fp->symbols_per_tti*10*sizeof(int32_t) );
#ifdef DEBUG_PHY
            printf("[openair][LTE_PHY][INIT] common_vars->txdataF[%d][%d] = %p (%lu bytes)\n",
                   eNB_id,i,common_vars->txdataF[eNB_id][i],
                   fp->ofdm_symbol_size*fp->symbols_per_tti*10*sizeof(int32_t));
#endif
          }
        }
      }

      for (i=0; i<fp->nb_antennas_tx; i++) {
        common_vars->txdataF_BF[eNB_id][i] = (int32_t*)malloc16_clear(fp->ofdm_symbol_size*sizeof(int32_t) );
        if (eNB->node_function != NGFI_RCC_IF4p5_NB_IoT)

          // Allocate 10 subframes of I/Q TX signal data (time) if not
          common_vars->txdata[eNB_id][i]  = (int32_t*)malloc16_clear( fp->samples_per_tti*10*sizeof(int32_t) );

#ifdef DEBUG_PHY
        printf("[openair][LTE_PHY][INIT] common_vars->txdata[%d][%d] = %p (%lu bytes)\n",eNB_id,i,common_vars->txdata[eNB_id][i],
               fp->samples_per_tti*10*sizeof(int32_t));
#endif
      }

      for (i=0; i<NB_ANTENNA_PORTS_ENB; i++) {
        if (i<fp->nb_antenna_ports_eNB || i==5) {
          common_vars->beam_weights[eNB_id][i] = (int32_t **)malloc16_clear(fp->nb_antennas_tx*sizeof(int32_t*));
          for (j=0; j<fp->nb_antennas_tx; j++) {
            common_vars->beam_weights[eNB_id][i][j] = (int32_t *)malloc16_clear(fp->ofdm_symbol_size*sizeof(int32_t));
            // antenna ports 0-3 are mapped on antennas 0-3
            // antenna port 4 is mapped on antenna 0
            // antenna ports 5-14 are mapped on all antennas
            if (((i<4) && (i==j)) || ((i==4) && (j==0))) {
              for (re=0; re<fp->ofdm_symbol_size; re++)
                common_vars->beam_weights[eNB_id][i][j][re] = 0x00007fff;
            }
            else if (i>4) {
              for (re=0; re<fp->ofdm_symbol_size; re++)
                common_vars->beam_weights[eNB_id][i][j][re] = 0x00007fff/fp->nb_antennas_tx;
            }
#ifdef DEBUG_PHY
            msg("[openair][LTE_PHY][INIT] lte_common_vars->beam_weights[%d][%d][%d] = %p (%zu bytes)\n",
                eNB_id,i,j,common_vars->beam_weights[eNB_id][i][j],
                fp->ofdm_symbol_size*sizeof(int32_t));
#endif
          }
        }
      }


      // RX vars
      if (eNB->node_function != NGFI_RCC_IF4p5_NB_IoT) {
        common_vars->rxdata[eNB_id]        = (int32_t**)malloc16(fp->nb_antennas_rx*sizeof(int32_t*) );
        common_vars->rxdata_7_5kHz[eNB_id] = (int32_t**)malloc16(fp->nb_antennas_rx*sizeof(int32_t*) );
      }

      common_vars->rxdataF[eNB_id]       = (int32_t**)malloc16(fp->nb_antennas_rx*sizeof(int32_t*) );


      for (i=0; i<fp->nb_antennas_rx; i++) {
        if (eNB->node_function != NGFI_RCC_IF4p5_NB_IoT) {
          // allocate 2 subframes of I/Q signal data (time) if not an RCC (no time-domain signals)
          common_vars->rxdata[eNB_id][i] = (int32_t*)malloc16_clear( fp->samples_per_tti*10*sizeof(int32_t) );

          if (eNB->node_function != NGFI_RRU_IF5_NB_IoT)
            // allocate 2 subframes of I/Q signal data (time, 7.5 kHz offset)
            common_vars->rxdata_7_5kHz[eNB_id][i] = (int32_t*)malloc16_clear( 2*fp->samples_per_tti*2*sizeof(int32_t) );
        }

        if (eNB->node_function != NGFI_RRU_IF5_NB_IoT)
          // allocate 2 subframes of I/Q signal data (frequency)
          common_vars->rxdataF[eNB_id][i] = (int32_t*)malloc16_clear(sizeof(int32_t)*(2*fp->ofdm_symbol_size*fp->symbols_per_tti) );

#ifdef DEBUG_PHY
        printf("[openair][LTE_PHY][INIT] common_vars->rxdata[%d][%d] = %p (%lu bytes)\n",eNB_id,i,common_vars->rxdata[eNB_id][i],fp->samples_per_tti*10*sizeof(int32_t));
        if (eNB->node_function != NGFI_RRU_IF5_NB_IoT)
          printf("[openair][LTE_PHY][INIT] common_vars->rxdata_7_5kHz[%d][%d] = %p (%lu bytes)\n",eNB_id,i,common_vars->rxdata_7_5kHz[eNB_id][i],fp->samples_per_tti*2*sizeof(int32_t));
#endif
        common_vars->rxdataF[eNB_id][i] = (int32_t*)malloc16_clear(sizeof(int32_t)*(fp->ofdm_symbol_size*fp->symbols_per_tti) );
      }


      if ((eNB->node_function != NGFI_RRU_IF4p5_NB_IoT)&&(eNB->node_function != NGFI_RRU_IF5_NB_IoT)) {

        // Channel estimates for SRS
        for (UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++) {

          srs_vars[UE_id].srs_ch_estimates[eNB_id]      = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          srs_vars[UE_id].srs_ch_estimates_time[eNB_id] = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );

          for (i=0; i<fp->nb_antennas_rx; i++) {
            srs_vars[UE_id].srs_ch_estimates[eNB_id][i]      = (int32_t*)malloc16_clear( sizeof(int32_t)*fp->ofdm_symbol_size );
            srs_vars[UE_id].srs_ch_estimates_time[eNB_id][i] = (int32_t*)malloc16_clear( sizeof(int32_t)*fp->ofdm_symbol_size*2 );
          }
        } //UE_id

        common_vars->sync_corr[eNB_id] = (uint32_t*)malloc16_clear( LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(uint32_t)*fp->samples_per_tti );
      }
    } // abstraction_flag = 0
    else { //UPLINK abstraction = 1
      eNB->sinr_dB = (double*) malloc16_clear( fp->N_RB_DL*12*sizeof(double) );
    }
  } //eNB_id



  if (abstraction_flag==0) {
    if ((eNB->node_function != NGFI_RRU_IF4p5_NB_IoT)&&(eNB->node_function != NGFI_RRU_IF5_NB_IoT)) {
      generate_ul_ref_sigs_rx();

      // SRS
      for (UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++) {
        srs_vars[UE_id].srs = (int32_t*)malloc16_clear(2*fp->ofdm_symbol_size*sizeof(int32_t));
      }
    }
  }



  // ULSCH VARS, skip if NFGI_RRU_IF4


  if ((eNB->node_function!=NGFI_RRU_IF4p5_NB_IoT)&&(eNB->node_function != NGFI_RRU_IF5_NB_IoT))

    prach_vars->prachF = (int16_t*)malloc16_clear( 1024*2*sizeof(int16_t) );

  /* number of elements of an array X is computed as sizeof(X) / sizeof(X[0]) */
  AssertFatal(fp->nb_antennas_rx <= sizeof(prach_vars->rxsigF) / sizeof(prach_vars->rxsigF[0]),
              "nb_antennas_rx too large");
  for (i=0; i<fp->nb_antennas_rx; i++) {
    prach_vars->rxsigF[i] = (int16_t*)malloc16_clear( fp->ofdm_symbol_size*12*2*sizeof(int16_t) );
#ifdef DEBUG_PHY
    printf("[openair][LTE_PHY][INIT] prach_vars->rxsigF[%d] = %p\n",i,prach_vars->rxsigF[i]);
#endif
  }

  if ((eNB->node_function != NGFI_RRU_IF4p5_NB_IoT)&&(eNB->node_function != NGFI_RRU_IF5_NB_IoT)) {

    AssertFatal(fp->nb_antennas_rx <= sizeof(prach_vars->prach_ifft) / sizeof(prach_vars->prach_ifft[0]),
                "nb_antennas_rx too large");
    for (i=0; i<fp->nb_antennas_rx; i++) {
      prach_vars->prach_ifft[i] = (int16_t*)malloc16_clear(1024*2*sizeof(int16_t));
#ifdef DEBUG_PHY
      printf("[openair][LTE_PHY][INIT] prach_vars->prach_ifft[%d] = %p\n",i,prach_vars->prach_ifft[i]);
#endif
    }

    for (UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++) {

      //FIXME
      pusch_vars[UE_id] = (NB_IoT_eNB_PUSCH*)malloc16_clear( NUMBER_OF_UE_MAX*sizeof(NB_IoT_eNB_PUSCH) );

      if (abstraction_flag==0) {
        for (eNB_id=0; eNB_id<3; eNB_id++) {

          pusch_vars[UE_id]->rxdataF_ext[eNB_id]      = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->rxdataF_ext2[eNB_id]     = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->drs_ch_estimates[eNB_id] = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->drs_ch_estimates_time[eNB_id] = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->rxdataF_comp[eNB_id]     = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->ul_ch_mag[eNB_id]  = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );
          pusch_vars[UE_id]->ul_ch_magb[eNB_id] = (int32_t**)malloc16( fp->nb_antennas_rx*sizeof(int32_t*) );

          for (i=0; i<fp->nb_antennas_rx; i++) {
            // RK 2 times because of output format of FFT!
            // FIXME We should get rid of this
            pusch_vars[UE_id]->rxdataF_ext[eNB_id][i]      = (int32_t*)malloc16_clear( 2*sizeof(int32_t)*fp->N_RB_UL*12*fp->symbols_per_tti );
            pusch_vars[UE_id]->rxdataF_ext2[eNB_id][i]     = (int32_t*)malloc16_clear( sizeof(int32_t)*fp->N_RB_UL*12*fp->symbols_per_tti );
            pusch_vars[UE_id]->drs_ch_estimates[eNB_id][i] = (int32_t*)malloc16_clear( sizeof(int32_t)*fp->N_RB_UL*12*fp->symbols_per_tti );
            pusch_vars[UE_id]->drs_ch_estimates_time[eNB_id][i] = (int32_t*)malloc16_clear( 2*2*sizeof(int32_t)*fp->ofdm_symbol_size );
            pusch_vars[UE_id]->rxdataF_comp[eNB_id][i]     = (int32_t*)malloc16_clear( sizeof(int32_t)*fp->N_RB_UL*12*fp->symbols_per_tti );
            pusch_vars[UE_id]->ul_ch_mag[eNB_id][i]  = (int32_t*)malloc16_clear( fp->symbols_per_tti*sizeof(int32_t)*fp->N_RB_UL*12 );
            pusch_vars[UE_id]->ul_ch_magb[eNB_id][i] = (int32_t*)malloc16_clear( fp->symbols_per_tti*sizeof(int32_t)*fp->N_RB_UL*12 );
          }
        } //eNB_id

        pusch_vars[UE_id]->llr = (int16_t*)malloc16_clear( (8*((3*8*6144)+12))*sizeof(int16_t) );
      } // abstraction_flag
    } //UE_id


    if (abstraction_flag==0) {
      if (is_secondary_eNB) {
        for (eNB_id=0; eNB_id<3; eNB_id++) {
          eNB->dl_precoder_SeNB[eNB_id] = (int **)malloc16(4*sizeof(int*));

          if (eNB->dl_precoder_SeNB[eNB_id]) {
#ifdef DEBUG_PHY
            printf("[openair][SECSYS_PHY][INIT] eNB->dl_precoder_SeNB[%d] allocated at %p\n",eNB_id,
                eNB->dl_precoder_SeNB[eNB_id]);
#endif
          } else {
            printf("[openair][SECSYS_PHY][INIT] eNB->dl_precoder_SeNB[%d] not allocated\n",eNB_id);
            return(-1);
          }

          for (j=0; j<fp->nb_antennas_tx; j++) {
            eNB->dl_precoder_SeNB[eNB_id][j] = (int *)malloc16(2*sizeof(int)*(fp->ofdm_symbol_size)); // repeated format (hence the '2*')

            if (eNB->dl_precoder_SeNB[eNB_id][j]) {
#ifdef DEBUG_PHY
              printf("[openair][LTE_PHY][INIT] eNB->dl_precoder_SeNB[%d][%d] allocated at %p\n",eNB_id,j,
                  eNB->dl_precoder_SeNB[eNB_id][j]);
#endif
              memset(eNB->dl_precoder_SeNB[eNB_id][j],0,2*sizeof(int)*(fp->ofdm_symbol_size));
            } else {
              printf("[openair][LTE_PHY][INIT] eNB->dl_precoder_SeNB[%d][%d] not allocated\n",eNB_id,j);
              return(-1);
            }
          } //for(j=...nb_antennas_tx

        } //for(eNB_id...
      }
    }
/*
    for (UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++)
      eNB->UE_stats_ptr[UE_id] = &eNB->UE_stats[UE_id];

    //defaul value until overwritten by RRCConnectionReconfiguration
    if (fp->nb_antenna_ports_eNB==2)
      eNB->pdsch_config_dedicated->p_a = dBm3;
    else
      eNB->pdsch_config_dedicated->p_a = dB0;

    init_prach_tables(839);
    */
  } // node_function != NGFI_RRU_IF4p5

  return (0);

}

void phy_init_lte_top_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms)
{

  crcTableInit_NB_IoT();

  //ccodedot11_init();
  //ccodedot11_init_inv();
  ccodelte_init_NB_IoT();
  ccodelte_init2_NB_IoT();
  //ccodelte_init_inv();

  //treillis_table_init();

  //phy_generate_viterbi_tables();
  //phy_generate_viterbi_tables_lte();

  //init_td8();
 // init_td16();
#ifdef __AVX2__
 // init_td16avx2();
#endif

  //lte_sync_time_init_NB_IoT(frame_parms);

  //generate_ul_ref_sigs();
  //generate_ul_ref_sigs_rx();
  generate_ul_ref_sigs_rx_NB_IoT();

 // generate_64qam_table();
  //generate_16qam_table();
 // generate_RIV_tables();

 init_unscrambling_lut_NB_IoT();
 // init_scrambling_lut();

  //set_taus_seed(1328);

}


//for NB-IoT layer1 to get informstion from layer2
int
l1_north_init_NB_IoT()
{
  int j;
  paramlist_def_t NbIoT_L1_ParamList = {NBIOT_L1LIST_CONFIG_STRING,NULL,0};

  if (RC.L1_NB_IoT != NULL)
  {
    AssertFatal(RC.L1_NB_IoT!=NULL,"RC.L1_NB_IoT is null\n");
    LOG_I(PHY,"RC.L1_NB_IoT = %p\n",RC.L1_NB_IoT);

    for (j=0; j<NbIoT_L1_ParamList.numelt; j++) {
      AssertFatal(RC.L1_NB_IoT[j]!=NULL,"RC.eNB_NB_IoT[%d] is null\n",j);
      LOG_I(PHY,"RC.L1_NB_IoT = %p\n",RC.L1_NB_IoT);


      if ((RC.L1_NB_IoT[j]->if_inst_NB_IoT =  IF_Module_init_NB_IoT(j))<0) return(-1); 
      LOG_I(PHY,"RC.L1_NB_IoT = %p\n",RC.L1_NB_IoT);

      RC.L1_NB_IoT[j]->if_inst_NB_IoT->PHY_config_req = PHY_config_req_NB_IoT;
      RC.L1_NB_IoT[j]->if_inst_NB_IoT->schedule_response = schedule_response_NB_IoT;
      
    }
  }
  else
  {
    LOG_I(PHY,"%s() Not installing PHY callbacks - RC.nb_nb_iot_L1_inst:%d RC.L1_NB_IoT:%p\n", __FUNCTION__, RC.nb_nb_iot_L1_inst, RC.L1_NB_IoT);
  }
  return(0);
}
