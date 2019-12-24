#include "defs.h"
#include "common/utils/LOG/log.h"
#include "PHY/impl_defs_lte_NB_IoT.h"

int init_frame_parms_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,uint8_t osf)
{

  uint8_t log2_osf;

#if DISABLE_LOG_X
  printf("Initializing frame parms for N_RB_DL %d, Ncp %d, osf %d\n",frame_parms->N_RB_DL,frame_parms->Ncp,osf);
#else
  LOG_I(PHY,"Initializing frame parms for N_RB_DL %d, Ncp %d, osf %d\n",frame_parms->N_RB_DL,frame_parms->Ncp,osf);
#endif

  
    frame_parms->nb_prefix_samples0 = 160;
    frame_parms->nb_prefix_samples = 144;
    frame_parms->symbols_per_tti = 14;
      


  switch(osf) {
  case 1:
    log2_osf = 0;
    break;

  case 2:
    log2_osf = 1;
    break;

  case 4:
    log2_osf = 2;
    break;

  case 8:
    log2_osf = 3;
    break;

  case 16:
    log2_osf = 4;
    break;

  default:
    printf("Illegal oversampling %d\n",osf);
    return(-1);
  }

  switch (frame_parms->N_RB_DL) {

  case 100:
    if (osf>1) {
      printf("Illegal oversampling %d for N_RB_DL %d\n",osf,frame_parms->N_RB_DL);
      return(-1);
    }

    if (frame_parms->threequarter_fs) {
      frame_parms->ofdm_symbol_size = 1536;
      frame_parms->samples_per_tti = 23040;
      frame_parms->first_carrier_offset = 1536-600;
      frame_parms->nb_prefix_samples=(frame_parms->nb_prefix_samples*3)>>2;
      frame_parms->nb_prefix_samples0=(frame_parms->nb_prefix_samples0*3)>>2;
    }
    else {
      frame_parms->ofdm_symbol_size = 2048;
      frame_parms->samples_per_tti = 30720;
      frame_parms->first_carrier_offset = 2048-600;
    }

    break;

  case 75:
    if (osf>1) {
      printf("Illegal oversampling %d for N_RB_DL %d\n",osf,frame_parms->N_RB_DL);
      return(-1);
    }


    frame_parms->ofdm_symbol_size = 1536;
    frame_parms->samples_per_tti = 23040;
    frame_parms->first_carrier_offset = 1536-450;
    frame_parms->nb_prefix_samples=(frame_parms->nb_prefix_samples*3)>>2;
    frame_parms->nb_prefix_samples0=(frame_parms->nb_prefix_samples0*3)>>2;

    break;

  case 50:
    if (osf>1) {
      printf("Illegal oversampling %d for N_RB_DL %d\n",osf,frame_parms->N_RB_DL);
      return(-1);
    }

    frame_parms->ofdm_symbol_size = 1024*osf;
    frame_parms->samples_per_tti = 15360*osf;
    frame_parms->first_carrier_offset = frame_parms->ofdm_symbol_size - 300;
    frame_parms->nb_prefix_samples>>=(1-log2_osf);
    frame_parms->nb_prefix_samples0>>=(1-log2_osf);

    break;

  case 25:
    if (osf>2) {
      printf("Illegal oversampling %d for N_RB_DL %d\n",osf,frame_parms->N_RB_DL);
      return(-1);
    }

    frame_parms->ofdm_symbol_size = 512*osf;


    frame_parms->samples_per_tti = 7680*osf;
    frame_parms->first_carrier_offset = frame_parms->ofdm_symbol_size - 150;
    frame_parms->nb_prefix_samples>>=(2-log2_osf);
    frame_parms->nb_prefix_samples0>>=(2-log2_osf);



    break;

  case 15:
    frame_parms->ofdm_symbol_size = 256*osf;
    frame_parms->samples_per_tti = 3840*osf;
    frame_parms->first_carrier_offset = frame_parms->ofdm_symbol_size - 90;
    frame_parms->nb_prefix_samples>>=(3-log2_osf);
    frame_parms->nb_prefix_samples0>>=(3-log2_osf);

    break;

  case 6:
    frame_parms->ofdm_symbol_size = 128*osf;
    frame_parms->samples_per_tti = 1920*osf;
    frame_parms->first_carrier_offset = frame_parms->ofdm_symbol_size - 36;
    frame_parms->nb_prefix_samples>>=(4-log2_osf);
    frame_parms->nb_prefix_samples0>>=(4-log2_osf);
    break;

  default:
    printf("init_frame_parms: Error: Number of resource blocks (N_RB_DL %d) undefined, frame_parms = %p \n",frame_parms->N_RB_DL, frame_parms);
    return(-1);
    break;
  }

  printf("lte_parms.c: Setting N_RB_DL to %d, ofdm_symbol_size %d\n",frame_parms->N_RB_DL, frame_parms->ofdm_symbol_size);


  //  frame_parms->tdd_config=3;
  return(0);
}
