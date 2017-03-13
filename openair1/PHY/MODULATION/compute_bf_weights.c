#include <stdio.h>
#include <stdlib.h> // contains the header information or prototype of the malloc
#include <string.h>
#include "UTIL/LOG/log.h"
#include "PHY/impl_defs_lte.h"
#include "PHY/TOOLS/defs.h"

extern char tdd_recip_calib_file[1024];

int read_calibration_matrix(int32_t **tdd_calib_coeffs, char *calibF_fname, LTE_DL_FRAME_PARMS *frame_parms) {

  FILE *calibF_fd;
  int aa,re,calibF_e ;

  //char* openair_dir = getenv("OPENAIR_DIR");
  
  calibF_fd = fopen(tdd_recip_calib_file, "r"); 

  if (calibF_fd == NULL) {
   printf("Warning: %s not found, running with defaults\n", tdd_recip_calib_file);
   return(1);
  }

  //printf("Loading Calibration matrix from %s\n", calibF_file_name);
  printf("Loading Calibration matrix from %s\n", tdd_recip_calib_file);
  
  for (aa=0;aa<frame_parms->nb_antennas_tx;aa++) {
    for(re=0;re<frame_parms->N_RB_DL*12;re++) {
      fscanf(calibF_fd, "%d", &calibF_e) ;
      //printf("aa=%d, re=%d, tdd_calib[0]=%d\n", aa, re, calibF_e);
      ((int16_t*)(&tdd_calib_coeffs[aa][re]))[0] = calibF_e;
      fscanf(calibF_fd, "%d", &calibF_e) ;
      //printf("aa=%d, re=%d, tdd_calib[1]=%d\n", aa, re, calibF_e);
      ((int16_t*)(&tdd_calib_coeffs[aa][re]))[1] = calibF_e;
      //printf("aa=%d, re=%d, tdd_calib=%d+i%d\n", aa, re, (int16_t*)(&tdd_calib_coeffs[aa][re])[0],(int16_t*)(&tdd_calib_coeffs[aa][re])[1]);
    }
  }
}

//TODO: ULCSI should be obtained from SRS in order to ensure ULCSI covers all freq band
void estimate_DLCSI_from_ULCSI(int32_t **calib_dl_ch_estimates, LTE_eNB_PUSCH *pusch_vars, int32_t **tdd_calib_coeffs, LTE_DL_FRAME_PARMS *frame_parms, int eNB_id) {
  int aa,re;
  uint8_t shift;
  int32_t **ul_ch_estimates = pusch_vars->drs_ch_estimates[eNB_id];
  int *ulsch_power = pusch_vars->ulsch_power;
  int beta = 1.2;

  for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
     shift = log2_approx(ulsch_power[aa]*beta)>>1;
     //printf("ulsch_power[%d]=%d\n", aa, ulsch_power[aa]);
     // Temporal implementation for 5MHz band
     multadd_cpx_vector((int16_t*)(&tdd_calib_coeffs[aa][12]),(int16_t*)(&ul_ch_estimates[aa][12]),(int16_t*)(&calib_dl_ch_estimates[aa][12]),1,20*12,shift);

    //multadd_cpx_vector((int16_t*)(&tdd_calib_coeffs[aa][0]),(int16_t*)(&ul_ch_estimates[aa][0]),(int16_t*)(&calib_dl_ch_estimates[aa][0]),1,frame_parms->N_RB_DL*12,15);

/*
    for (re=12; re<20*12; re++) {
      ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0] = (((int16_t*)(&tdd_calib_coeffs[aa][re]))[0]*((int16_t*)(&ul_ch_estimates[aa][re]))[0])>>shift;
      ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0] -= (((int16_t*)(&tdd_calib_coeffs[aa][re]))[1]*((int16_t*)(&ul_ch_estimates[aa][re]))[1])>>shift;
      ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1] = (((int16_t*)(&tdd_calib_coeffs[aa][re]))[0]*((int16_t*)(&ul_ch_estimates[aa][re]))[1])>>shift;
      ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1] += (((int16_t*)(&tdd_calib_coeffs[aa][re]))[1]*((int16_t*)(&ul_ch_estimates[aa][re]))[0])>>shift;
    }
*/

/*
    for (re=12; re<20*12; re++) {
      printf("aa=%d, re=%d, tdd_calib_coeffs=(%d, %d), ul_ch_estimates=(%d, %d), calib_dl_ch_estimates=(%d,%d)\n",aa,re,
             ((int16_t*)(&tdd_calib_coeffs[aa][re]))[0], ((int16_t*)(&tdd_calib_coeffs[aa][re]))[1],
             ((int16_t*)(&ul_ch_estimates[aa][re]))[0], ((int16_t*)(&ul_ch_estimates[aa][re]))[1],
             ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0],((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1]);
    }
*/
  }
}

void compute_BF_weights(int32_t **beam_weights, int32_t **calib_dl_ch_estimates, PRECODE_TYPE_t precode_type, LTE_DL_FRAME_PARMS *frame_parms) {

  int aa, re;
  int norm_factor = 0;

  switch (precode_type) {
  //case MRT
  case MRT :
  for (aa=0 ; aa<frame_parms->nb_antennas_tx ; aa++) {
    //for (re=0; re<frame_parms->N_RB_DL*6; re++) {
    for (re=12; re<frame_parms->N_RB_DL*6; re++) {
      //normalisation simplied by a constent shift
      ((int16_t*)(&beam_weights[aa][frame_parms->first_carrier_offset+re]))[0] = ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0]<<norm_factor;
      ((int16_t*)(&beam_weights[aa][frame_parms->first_carrier_offset+re]))[1] = -((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1]<<norm_factor;
     /* printf("calib_dl_ch_estimates[%d][%d]=(%d,%d),beam_weight[%d][%d]=(%d,%d)\n",
             aa, re,
             ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0],
             ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1],
             aa, frame_parms->first_carrier_offset+re, 
             ((int16_t*)(&beam_weights[aa][frame_parms->first_carrier_offset+re]))[0],
             ((int16_t*)(&beam_weights[aa][frame_parms->first_carrier_offset+re]))[1]);*/
    }

    //for (re=frame_parms->N_RB_DL*6; re<frame_parms->N_RB_DL*12; re++) {
    for (re=frame_parms->N_RB_DL*6; re<20*12; re++) {
      //normalisation simplied by a constent shift
      ((int16_t*)(&beam_weights[aa][re-frame_parms->N_RB_DL*6+1]))[0] = ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0]<<norm_factor;
      ((int16_t*)(&beam_weights[aa][re-frame_parms->N_RB_DL*6+1]))[1] = -((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1]<<norm_factor;
      /*printf("calib_dl_ch_estimates[%d][%d]=(%d,%d),beam_weight[%d][%d]=(%d,%d)\n",
             aa, re,
             ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[0],
             ((int16_t*)(&calib_dl_ch_estimates[aa][re]))[1],
             aa, frame_parms->first_carrier_offset+re, 
             ((int16_t*)(&beam_weights[aa][re-frame_parms->N_RB_DL*6+1]))[0],
             ((int16_t*)(&beam_weights[aa][re-frame_parms->N_RB_DL*6+1]))[1]);*/
    }
  }
  break ;

  //case ZF
  case ZF :
  break;

  //case MMSE
  case MMSE :
  break;

  default :
  break;  
  }

} 



//unitary test function
/*
void main() {
  printf("Test Compute BF weights.\n");

  int32_t **tdd_calib_coeffs, **calib_dl_ch_estimates, **ul_ch_estimates, **beam_weights;
  int nb_ant, nb_freq, aa, re;
  char calibF_fname[] = "calibF.m";
  char BF_fname[] = "BF_weights.m";
  FILE *BF_weights_fd;

  nb_ant = 8;
  nb_freq = 300;

  // memory allocation
  tdd_calib_coeffs = (int32_t **)malloc(nb_ant*sizeof(int32_t *));
  calib_dl_ch_estimates = (int32_t **)malloc(nb_ant*sizeof(int32_t *));
  ul_ch_estimates = (int32_t **)malloc(nb_ant*sizeof(int32_t *));
  beam_weights = (int32_t **)malloc(nb_ant*sizeof(int32_t *));

  for (aa=0; aa<nb_ant; aa++) {
    tdd_calib_coeffs[aa] = (int32_t *)malloc(nb_freq*sizeof(int32_t));
    calib_dl_ch_estimates[aa] = (int32_t *)malloc(nb_freq*sizeof(int32_t));
    ul_ch_estimates[aa] = (int32_t *)malloc(nb_freq*sizeof(int32_t));
    beam_weights[aa] = (int32_t *)malloc(nb_freq*sizeof(int32_t));
  }

  // ul channel estimation initilisation
  for (aa=0; aa<nb_ant; aa++)
    for (re=0; re<nb_freq; re++)
      ul_ch_estimates[aa][re] = 0x7fff7fff;

  // calibration coefficients loading
  read_calibration_matrix(calibF_fname, nb_ant, nb_freq, tdd_calib_coeffs);

  // DL calib channel estimation
  estimate_DLCSI_from_ULCSI(calib_dl_ch_estimates, ul_ch_estimates, tdd_calib_coeffs, nb_ant, nb_freq);

  // Beamforming weights calculation
  compute_BF_weights(beam_weights, calib_dl_ch_estimates, MRT, nb_ant, nb_freq);

  // writing beam_weights into a .m file
  BF_weights_fd = fopen(BF_fname,"w");
  for (aa=0; aa<nb_ant; aa++) {
    for (re=0; re<nb_freq; re++) {
      fprintf(BF_weights_fd, "%d", ((int16_t *)&beam_weights[aa][re])[0]);
      fprintf(BF_weights_fd, "%s", " ");
      fprintf(BF_weights_fd, "%d", ((int16_t *)&beam_weights[aa][re])[1]);
      fprintf(BF_weights_fd, "%s", " ");
    }
    fprintf(BF_weights_fd, "\n");
  }
}
*/
