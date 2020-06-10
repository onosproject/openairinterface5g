#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "PHY/defs_common.h"
#include "PHY/defs_eNB.h"

int f_read(char *calibF_fname, int nb_ant, int nb_freq, int32_t **tdd_calib_coeffs){

  FILE *calibF_fd;
  int i,j,calibF_e;
  
  calibF_fd = fopen(calibF_fname,"r");
 
  if (calibF_fd) {
    printf("Loading Calibration matrix from %s\n", calibF_fname);
  
    for(i=0;i<nb_ant;i++){
      for(j=0;j<nb_freq*2;j++){
        if (fscanf(calibF_fd, "%d", &calibF_e) != 1) abort();
        tdd_calib_coeffs[i][j] = (int16_t)calibF_e;
      }
    }
    printf("%d\n",(int)tdd_calib_coeffs[0][0]);
    printf("%d\n",(int)tdd_calib_coeffs[1][599]);
    fclose(calibF_fd);
  } else
   printf("%s not found, running with defaults\n",calibF_fname);
  /* TODO: what to return? is this code used at all? */
  return 0;
}


int estimate_DLCSI_from_ULCSI(int32_t **calib_dl_ch_estimates, int32_t **ul_ch_estimates, int32_t **tdd_calib_coeffs, int nb_ant, int nb_freq) {

  /* TODO: what to return? is this code used at all? */
  return 0;

}

int compute_BF_weights(int32_t **beam_weights, int32_t **calib_dl_ch_estimates, PRECODE_TYPE_t precode_type, int nb_ant, int nb_freq) {
  switch (precode_type) {
  //case MRT
  case 0 :
  //case ZF
  break;
  case 1 :
  //case MMSE
  break;
  case 2 :
  break;
  default :
  break;  
}
  /* TODO: what to return? is this code used at all? */
  return 0;
}

int compute_beam_weights(int32_t **beam_weights[NUMBER_OF_eNB_MAX+1][15], int32_t **calib_coeffs, int32_t **ul_ch_estimates, PHY_VARS_eNB *eNB, int l1_id, int p, int aa, int ru_id) {

//PHY_VARS_eNB *eNB = RC.eNB[0][0];
LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
int d_f = 597;

//LOG_I(PHY,"compute_beam_weights : l1_id %d, p %d, aa %d, ru_id %d \n",l1_id,p,aa,ru_id);
//LOG_I(PHY,"(int16_t*)&beam_weights[%d][%d][%d][0] %p\n", l1_id,p,aa,(int16_t*)&beam_weights[l1_id][p][aa][0]);
//LOG_I(PHY,"[compute_beam_weights] : calib_coeffs[1][%d] : %d %d i\n",d_f,calib_coeffs[1][d_f],calib_coeffs[1][d_f+1]);
mult_cpx_vector((int16_t *)calib_coeffs[ru_id], 
                    (int16_t*)&ul_ch_estimates[aa][0],
                    (int16_t*)&beam_weights[l1_id][p][aa][0],
                    fp->N_RB_UL*12,
                    15);

return 0;
} 

// temporal test function
/*
void main(){
  // initialization
  // compare
  printf("Hello world!\n");
}
*/
