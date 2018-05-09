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

/* file: lte_sync_time.c
   purpose: coarse timing synchronization for LTE (using PSS)
   author: florian.kaltenberger@eurecom.fr, oscar.tonelli@yahoo.it
   date: 22.10.2009
*/

//#include <string.h>
#include "defs.h"
#include "PHY/defs.h"
#include "PHY/extern.h"
#include "SCHED/extern.h"
#include <math.h>

#ifdef OPENAIR2
#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"
#include "RRC/LITE/extern.h"
#include "PHY_INTERFACE/extern.h"
#endif
//#define DEBUG_PHY

int32_t* sync_corr_ue0 = NULL;
int32_t* sync_corr_ue1 = NULL;
int32_t* sync_corr_ue2 = NULL;
int32_t sync_tmp[2048*4] __attribute__((aligned(32)));
int16_t syncF_tmp[2048*2] __attribute__((aligned(32)));



int lte_sync_time_init(LTE_DL_FRAME_PARMS *frame_parms )   // LTE_UE_COMMON *common_vars
{

  int i,k;

  sync_corr_ue0 = (int32_t *)malloc16(LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int32_t)*frame_parms->samples_per_tti);
  sync_corr_ue1 = (int32_t *)malloc16(LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int32_t)*frame_parms->samples_per_tti);
  sync_corr_ue2 = (int32_t *)malloc16(LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int32_t)*frame_parms->samples_per_tti);

  if (sync_corr_ue0) {
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue allocated at %p\n", sync_corr_ue0);
#endif
    //common_vars->sync_corr = sync_corr;
  } else {
    LOG_E(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue0 not allocated\n");
    return(-1);
  }

  if (sync_corr_ue1) {
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue allocated at %p\n", sync_corr_ue1);
#endif
    //common_vars->sync_corr = sync_corr;
  } else {
    LOG_E(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue1 not allocated\n");
    return(-1);
  }

  if (sync_corr_ue2) {
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue allocated at %p\n", sync_corr_ue2);
#endif
    //common_vars->sync_corr = sync_corr;
  } else {
    LOG_E(PHY,"[openair][LTE_PHY][SYNC] sync_corr_ue2 not allocated\n");
    return(-1);
  }

  //  primary_synch0_time = (int *)malloc16((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
  primary_synch0_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);

  if (primary_synch0_time) {
    //    bzero(primary_synch0_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
    bzero(primary_synch0_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch0_time allocated at %p\n", primary_synch0_time);
#endif
  } else  AssertFatal(1==0,"primary_synch0_time not allocated\n");


  //  primary_synch1_time = (int *)malloc16((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
  primary_synch1_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);

  if (primary_synch1_time) {
    //    bzero(primary_synch1_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
    bzero(primary_synch1_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch1_time allocated at %p\n", primary_synch1_time);
#endif
  } else  AssertFatal(1==0,"primary_synch1_time not allocated\n");

  //  primary_synch2_time = (int *)malloc16((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
  primary_synch2_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);

  if (primary_synch2_time) {
    //    bzero(primary_synch2_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
    bzero(primary_synch2_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch2_time allocated at %p\n", primary_synch2_time);
#endif
  } else  AssertFatal(1==0,"primary_synch2_time not allocated\n");


  primary_synch0SL_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
  if (primary_synch0SL_time) {
    //    bzero(primary_synch0_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
    bzero(primary_synch0SL_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch0SL_time allocated at %p\n", primary_synch0SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch0SL_time not allocated\n");


  //  primary_synch1_time = (int *)malloc16((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
  primary_synch1SL_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);

  if (primary_synch1SL_time) {
    //    bzero(primary_synch1_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int));
    bzero(primary_synch1SL_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch1SL_time allocated at %p\n", primary_synch1SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch1SL_time not allocated\n");


  // generate oversampled sync_time sequences
  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch0[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch0[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k++;  // skip DC carrier
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp,
	     1); /// complex output
    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    break;
  }

  for (i=0; i<frame_parms->ofdm_symbol_size; i++)
    ((int32_t*)primary_synch0_time)[i] = sync_tmp[i];

  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch1[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch1[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k++;  // skip DC carrier
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    break;
  }

  for (i=0; i<frame_parms->ofdm_symbol_size; i++)
    ((int32_t*)primary_synch1_time)[i] = sync_tmp[i];

  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch2[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch2[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k++;  // skip DC carrier
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    break;
  }

  for (i=0; i<frame_parms->ofdm_symbol_size; i++)
    ((int32_t*)primary_synch2_time)[i] = sync_tmp[i];

  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch0SL[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch0SL[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp,
	     1); /// complex output
    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    break;
  }

  for (i=0; i<frame_parms->ofdm_symbol_size; i++)
    ((int32_t*)primary_synch0SL_time)[i] = sync_tmp[i];

  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch1SL[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch1SL[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    break;
  }

  for (i=0; i<frame_parms->ofdm_symbol_size; i++)
    ((int32_t*)primary_synch1SL_time)[i] = sync_tmp[i];

  /*
  write_output("primary_sync0.m","psync0",primary_synch0_time,frame_parms->ofdm_symbol_size,1,1);
  write_output("primary_sync1.m","psync1",primary_synch1_time,frame_parms->ofdm_symbol_size,1,1);
  write_output("primary_sync2.m","psync2",primary_synch2_time,frame_parms->ofdm_symbol_size,1,1);
  write_output("primary_syncSL0.m","psyncSL0",primary_synch0SL_time,frame_parms->ofdm_symbol_size,1,1);
  write_output("primary_syncSL1.m","psyncSL1",primary_synch1SL_time,frame_parms->ofdm_symbol_size,1,1);
  */
  
  return (1);
}


void lte_sync_time_free(void)
{


  if (sync_corr_ue0) {
    LOG_D(PHY,"Freeing sync_corr_ue (%p)...\n",sync_corr_ue0);
    free(sync_corr_ue0);
  }

  if (sync_corr_ue1) {
    LOG_D(PHY,"Freeing sync_corr_ue (%p)...\n",sync_corr_ue1);
    free(sync_corr_ue1);
  }

  if (sync_corr_ue2) {
    LOG_D(PHY,"Freeing sync_corr_ue (%p)...\n",sync_corr_ue2);
    free(sync_corr_ue2);
  }

  if (primary_synch0_time) {
    LOG_D(PHY,"Freeing primary_sync0_time ...\n");
    free(primary_synch0_time);
  }

  if (primary_synch1_time) {
    LOG_D(PHY,"Freeing primary_sync1_time ...\n");
    free(primary_synch1_time);
  }

  if (primary_synch2_time) {
    LOG_D(PHY,"Freeing primary_sync2_time ...\n");
    free(primary_synch2_time);
  }

  if (primary_synch0SL_time) {
    LOG_D(PHY,"Freeing primary_sync0SL_time ...\n");
    free(primary_synch0SL_time);
  }

  if (primary_synch1SL_time) {
    LOG_D(PHY,"Freeing primary_sync1SL_time ...\n");
    free(primary_synch1SL_time);
  }

  
  sync_corr_ue0 = NULL;
  sync_corr_ue1 = NULL;
  sync_corr_ue2 = NULL;
  primary_synch0_time = NULL;
  primary_synch1_time = NULL;
  primary_synch2_time = NULL;
}

static inline int32_t abs32(int32_t x)
{
  return (((int32_t)((int16_t*)&x)[0])*((int32_t)((int16_t*)&x)[0]) + ((int32_t)((int16_t*)&x)[1])*((int32_t)((int16_t*)&x)[1]));
}

#ifdef DEBUG_PHY
int debug_cnt=0;
#endif

#define SHIFT 17

int lte_sync_time(int **rxdata, ///rx data in time domain
                  LTE_DL_FRAME_PARMS *frame_parms,
                  int *eNB_id)
{



  // perform a time domain correlation using the oversampled sync sequence

  unsigned int n, ar, s, peak_pos, peak_val, sync_source;
  int32_t result,result2;
  int32_t sync_out[3] = {0,0,0},sync_out2[3] = {0,0,0};
  int32_t tmp[3] = {0,0,0};
  int length =   LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*frame_parms->samples_per_tti>>1;

  //LOG_D(PHY,"[SYNC TIME] Calling sync_time.\n");
  if (sync_corr_ue0 == NULL) {
    LOG_E(PHY,"[SYNC TIME] sync_corr_ue0 not yet allocated! Exiting.\n");
    return(-1);
  }

  if (sync_corr_ue1 == NULL) {
    LOG_E(PHY,"[SYNC TIME] sync_corr_ue1 not yet allocated! Exiting.\n");
    return(-1);
  }

  if (sync_corr_ue2 == NULL) {
    LOG_E(PHY,"[SYNC TIME] sync_corr_ue2 not yet allocated! Exiting.\n");
    return(-1);
  }

  peak_val = 0;
  peak_pos = 0;
  sync_source = 0;


  for (n=0; n<length; n+=4) {

    sync_corr_ue0[n] = 0;
    sync_corr_ue0[n+length] = 0;
    sync_corr_ue1[n] = 0;
    sync_corr_ue1[n+length] = 0;
    sync_corr_ue2[n] = 0;
    sync_corr_ue2[n+length] = 0;

    for (s=0; s<3; s++) {
      sync_out[s]=0;
      sync_out2[s]=0;
    }

    //    if (n<(length-frame_parms->ofdm_symbol_size-frame_parms->nb_prefix_samples)) {
    if (n<(length-frame_parms->ofdm_symbol_size)) {

      //calculate dot product of primary_synch0_time and rxdata[ar][n] (ar=0..nb_ant_rx) and store the sum in temp[n];
      for (ar=0; ar<frame_parms->nb_antennas_rx; ar++) {

        result  = dot_product((int16_t*)primary_synch0_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);
        result2 = dot_product((int16_t*)primary_synch0_time, (int16_t*) &(rxdata[ar][n+length]), frame_parms->ofdm_symbol_size, SHIFT);

        ((int16_t*)sync_corr_ue0)[2*n] += ((int16_t*) &result)[0];
        ((int16_t*)sync_corr_ue0)[2*n+1] += ((int16_t*) &result)[1];
        ((int16_t*)sync_corr_ue0)[2*(length+n)] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_corr_ue0)[(2*(length+n))+1] += ((int16_t*) &result2)[1];
        ((int16_t*)sync_out)[0] += ((int16_t*) &result)[0];
        ((int16_t*)sync_out)[1] += ((int16_t*) &result)[1];
        ((int16_t*)sync_out2)[0] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_out2)[1] += ((int16_t*) &result2)[1];
      }

      for (ar=0; ar<frame_parms->nb_antennas_rx; ar++) {
        result = dot_product((int16_t*)primary_synch1_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);
        result2 = dot_product((int16_t*)primary_synch1_time, (int16_t*) &(rxdata[ar][n+length]), frame_parms->ofdm_symbol_size, SHIFT);
        ((int16_t*)sync_corr_ue1)[2*n] += ((int16_t*) &result)[0];
        ((int16_t*)sync_corr_ue1)[2*n+1] += ((int16_t*) &result)[1];
        ((int16_t*)sync_corr_ue1)[2*(length+n)] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_corr_ue1)[(2*(length+n))+1] += ((int16_t*) &result2)[1];

        ((int16_t*)sync_out)[2] += ((int16_t*) &result)[0];
        ((int16_t*)sync_out)[3] += ((int16_t*) &result)[1];
        ((int16_t*)sync_out2)[2] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_out2)[3] += ((int16_t*) &result2)[1];
      }

      for (ar=0; ar<frame_parms->nb_antennas_rx; ar++) {

        result = dot_product((int16_t*)primary_synch2_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);
        result2 = dot_product((int16_t*)primary_synch2_time, (int16_t*) &(rxdata[ar][n+length]), frame_parms->ofdm_symbol_size, SHIFT);
        ((int16_t*)sync_corr_ue2)[2*n] += ((int16_t*) &result)[0];
        ((int16_t*)sync_corr_ue2)[2*n+1] += ((int16_t*) &result)[1];
        ((int16_t*)sync_corr_ue2)[2*(length+n)] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_corr_ue2)[(2*(length+n))+1] += ((int16_t*) &result2)[1];
        ((int16_t*)sync_out)[4] += ((int16_t*) &result)[0];
        ((int16_t*)sync_out)[5] += ((int16_t*) &result)[1];
        ((int16_t*)sync_out2)[4] += ((int16_t*) &result2)[0];
        ((int16_t*)sync_out2)[5] += ((int16_t*) &result2)[1];
      }

    }

    // calculate the absolute value of sync_corr[n]

    sync_corr_ue0[n] = abs32(sync_corr_ue0[n]);
    sync_corr_ue0[n+length] = abs32(sync_corr_ue0[n+length]);
    sync_corr_ue1[n] = abs32(sync_corr_ue1[n]);
    sync_corr_ue1[n+length] = abs32(sync_corr_ue1[n+length]);
    sync_corr_ue2[n] = abs32(sync_corr_ue2[n]);
    sync_corr_ue2[n+length] = abs32(sync_corr_ue2[n+length]);

    for (s=0; s<3; s++) {
      tmp[s] = (abs32(sync_out[s])>>1) + (abs32(sync_out2[s])>>1);

      if (tmp[s]>peak_val) {
        peak_val = tmp[s];
        peak_pos = n;
        sync_source = s;
        /*
        printf("s %d: n %d sync_out %d, sync_out2  %d (sync_corr %d,%d), (%d,%d) (%d,%d)\n",s,n,abs32(sync_out[s]),abs32(sync_out2[s]),sync_corr_ue0[n],
               sync_corr_ue0[n+length],((int16_t*)&sync_out[s])[0],((int16_t*)&sync_out[s])[1],((int16_t*)&sync_out2[s])[0],((int16_t*)&sync_out2[s])[1]);
        */
      }
    }
  }

  *eNB_id = sync_source;

  LOG_I(PHY,"[UE] lte_sync_time: Sync source = %d, Peak found at pos %d, val = %d (%d dB)\n",sync_source,peak_pos,peak_val,dB_fixed(peak_val)/2);


#ifdef DEBUG_PHY
  if (debug_cnt == 0) {
    write_output("sync_corr0_ue.m","synccorr0",sync_corr_ue0,2*length,1,2);
    write_output("sync_corr1_ue.m","synccorr1",sync_corr_ue1,2*length,1,2);
    write_output("sync_corr2_ue.m","synccorr2",sync_corr_ue2,2*length,1,2);
    write_output("rxdata0.m","rxd0",rxdata[0],length<<1,1,1);
    //    exit(-1);
  } else {
    debug_cnt++;
  }


#endif


  return(peak_pos);

}

int lte_sync_timeSL(PHY_VARS_UE *ue,
		    int *ind,
		    int64_t *lev,
		    int64_t *avg)
{


  LTE_DL_FRAME_PARMS *frame_parms = &ue->frame_parms;
		      
  // perform a time domain correlation using the oversampled sync sequence

  int length =   4*LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*frame_parms->samples_per_tti;
  
  int32_t sync_corr0[frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples];
  int32_t sync_corr1[frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples];


  // circular copy of beginning to end of rxdata buffer. Note: buffer should be big enough upon calling this function
  for (int ar=0;ar<frame_parms->nb_antennas_rx;ar++) memcpy((void*)&ue->common_vars.rxdata_syncSL[ar][2*length],
							    (void*)&ue->common_vars.rxdata_syncSL[ar][0],
							    frame_parms->ofdm_symbol_size);

  int32_t tmp0,tmp1;
  int32_t magtmp0,magtmp1,lev0,lev1,maxlev0=0,maxlev1=0;
  int     maxpos0=0,maxpos1=0;
  int64_t avg0=0,avg1=0;
  int32_t result;
  int32_t **rxdata = ue->common_vars.rxdata_syncSL; ///rx data in time domain
  RU_t ru_tmp;
  int16_t **rxdata_7_5kHz    = ue->sl_rxdata_7_5kHz;

  memset((void*)&ru_tmp,0,sizeof(RU_t));
  
  memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
  ru_tmp.N_TA_offset=0;
  ru_tmp.common.rxdata = rxdata;
  ru_tmp.common.rxdata_7_5kHz = (int32_t**)rxdata_7_5kHz;
  ru_tmp.nb_rx = frame_parms->nb_antennas_rx;
  


  // remove 7.5 kHz
  for (int slot=0;slot<80;slot++) {
    remove_7_5_kHz(&ru_tmp,slot);
    for (int ar=0;ar<frame_parms->nb_antennas_rx;ar++) {
      memcpy((void*)&rxdata[ar][slot*2*(frame_parms->samples_per_tti/2)],
	     (void*)&rxdata_7_5kHz[ar][(slot&1)*2*(frame_parms->samples_per_tti/2)],
	     sizeof(int16_t)*(2*frame_parms->samples_per_tti/2));
    }
  }
  
  for (int n=0; n<length; n+=4) {

    int nprime = n % (frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples);
    
    tmp0 = 0;
    tmp1 = 0;

    //calculate dot product of primary_synch0_time and rxdata[ar][n] (ar=0..nb_ant_rx) and store the sum in temp[n];
    for (int ar=0; ar<frame_parms->nb_antennas_rx; ar++) {
      
      result  = dot_product((int16_t*)primary_synch0SL_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);
      
      ((int16_t*)&tmp0)[0] += ((int16_t*) &result)[0];
      ((int16_t*)&tmp0)[1] += ((int16_t*) &result)[1];

      result  = dot_product((int16_t*)primary_synch1SL_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);

      ((int16_t*)&tmp1)[0] += ((int16_t*) &result)[0];
      ((int16_t*)&tmp1)[1] += ((int16_t*) &result)[1];

    }

    // tmpi holds <synchi,rx0>+<synci,rx1>+...+<synchi,rx_{nbrx-1}>

    magtmp0 = abs32(tmp0);
    magtmp1 = abs32(tmp1);

    // this does max |tmpi(n)|^2 + |tmpi(n-L)|^2 and argmax |tmpi(n)|^2 + |tmpi(n-L)|^2
    
    if (n>(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)) {

      //      if (n<4096) printf("syncSL: sample %d (nprime %d) : , mag0 %d, prev0 %d, mag1 %d, prev1 %d\n",
      //			 n,nprime,magtmp0,sync_corr0[nprime],magtmp1,sync_corr1[nprime]);
      lev0 = magtmp0 + sync_corr0[nprime];
      lev1 = magtmp1 + sync_corr1[nprime];
      
      if (lev0>maxlev0) { maxlev0 = lev0; maxpos0 = n-(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples); }
      if (lev1>maxlev1) { maxlev1 = lev1; maxpos1 = n-(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples); }
      avg0 += magtmp0;
      avg1 += magtmp1;
      
    }
    sync_corr0[nprime]=magtmp0;
    sync_corr1[nprime]=magtmp1;
  }
  avg0/=length;
  avg1/=length;

  // PSS in symbol 1
  int pssoffset = frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples0 + frame_parms->nb_prefix_samples;
  
  if (maxlev0 > maxlev1) {
    if ((int64_t)maxlev0 > (5*avg0)) {*lev = maxlev0; *ind=0; *avg=avg0; return((length+maxpos0-pssoffset)%length);};
  }
  else {
    if ((int64_t)maxlev1 > (5*avg1)) {*lev = maxlev1; *ind=1; *avg=avg1; return((length+maxpos1-pssoffset)%length);};
  }
  return(-1);


}

//#define DEBUG_PHY

int lte_sync_time_eNB(int32_t **rxdata, ///rx data in time domain
                      LTE_DL_FRAME_PARMS *frame_parms,
                      uint32_t length,
                      uint32_t *peak_val_out,
                      uint32_t *sync_corr_eNB)
{

  // perform a time domain correlation using the oversampled sync sequence

  unsigned int n, ar, peak_val, peak_pos;
  uint64_t mean_val;
  int result;
  int16_t *primary_synch_time;
  int eNB_id = frame_parms->Nid_cell%3;

  // LOG_E(PHY,"[SYNC TIME] Calling sync_time_eNB(%p,%p,%d,%d)\n",rxdata,frame_parms,eNB_id,length);
  if (sync_corr_eNB == NULL) {
    LOG_E(PHY,"[SYNC TIME] sync_corr_eNB not yet allocated! Exiting.\n");
    return(-1);
  }

  switch (eNB_id) {
  case 0:
    primary_synch_time = (int16_t*)primary_synch0_time;
    break;

  case 1:
    primary_synch_time = (int16_t*)primary_synch1_time;
    break;

  case 2:
    primary_synch_time = (int16_t*)primary_synch2_time;
    break;

  default:
    LOG_E(PHY,"[SYNC TIME] Illegal eNB_id!\n");
    return (-1);
  }

  peak_val = 0;
  peak_pos = 0;
  mean_val = 0;

  for (n=0; n<length; n+=4) {

    sync_corr_eNB[n] = 0;

    if (n<(length-frame_parms->ofdm_symbol_size-frame_parms->nb_prefix_samples)) {

      //calculate dot product of primary_synch0_time and rxdata[ar][n] (ar=0..nb_ant_rx) and store the sum in temp[n];
      for (ar=0; ar<frame_parms->nb_antennas_rx; ar++)  {

        result = dot_product((int16_t*)primary_synch_time, (int16_t*) &(rxdata[ar][n]), frame_parms->ofdm_symbol_size, SHIFT);
        //((int16_t*)sync_corr)[2*n]   += ((int16_t*) &result)[0];
        //((int16_t*)sync_corr)[2*n+1] += ((int16_t*) &result)[1];
        sync_corr_eNB[n] += abs32(result);

      }

    }

    /*
    if (eNB_id == 2) {
      printf("sync_time_eNB %d : %d,%d (%d)\n",n,sync_corr_eNB[n],mean_val,
       peak_val);
    }
    */
    mean_val += sync_corr_eNB[n];

    if (sync_corr_eNB[n]>peak_val) {
      peak_val = sync_corr_eNB[n];
      peak_pos = n;
    }
  }

  mean_val/=length;

  *peak_val_out = peak_val;

  if (peak_val <= (40*(uint32_t)mean_val)) {
    LOG_D(PHY,"[SYNC TIME] No peak found (%u,%u,%"PRIu64",%"PRIu64")\n",peak_pos,peak_val,mean_val,40*mean_val);
    return(-1);
  } else {
    LOG_D(PHY,"[SYNC TIME] Peak found at pos %u, val = %u, mean_val = %"PRIu64"\n",peak_pos,peak_val,mean_val);
    return(peak_pos);
  }

}

