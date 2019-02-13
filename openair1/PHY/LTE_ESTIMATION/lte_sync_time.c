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
#include "PHY/defs_UE.h"
//#include "PHY/phy_vars_ue.h"
#include "PHY/phy_extern_ue.h"
#include <math.h>
#include "PHY/MODULATION/modulation_extern.h"

#include "LAYER2/MAC/mac.h"
#include "RRC/LTE/rrc_extern.h"
#include "PHY_INTERFACE/phy_interface.h"


int64_t* sync_corr_ue0 = NULL;
int64_t* sync_corr_ue1 = NULL;
int64_t* sync_corr_ue2 = NULL;

/*
extern int16_t s6n_kHz_7_5[1920];
extern int16_t s6e_kHz_7_5[1920];
extern int16_t s25n_kHz_7_5[7680];
extern int16_t s25e_kHz_7_5[7680];
extern int16_t s50n_kHz_7_5[15360];
extern int16_t s50e_kHz_7_5[15360];
extern int16_t s75n_kHz_7_5[24576];
extern int16_t s75e_kHz_7_5[24576];
extern int16_t s100n_kHz_7_5[30720];
extern int16_t s100e_kHz_7_5[30720];
*/

int lte_sync_time_init(LTE_DL_FRAME_PARMS *frame_parms )   // LTE_UE_COMMON *common_vars
{

  int i,k;
  int32_t sync_tmp[2048*4] __attribute__((aligned(32)));
  int16_t syncF_tmp[2048*2] __attribute__((aligned(32)));

  sync_corr_ue0 = (int64_t *)malloc16(4*LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int64_t)*frame_parms->samples_per_tti);
  sync_corr_ue1 = (int64_t *)malloc16(4*LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int64_t)*frame_parms->samples_per_tti);
  sync_corr_ue2 = (int64_t *)malloc16(LTE_NUMBER_OF_SUBFRAMES_PER_FRAME*sizeof(int64_t)*frame_parms->samples_per_tti);

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


  primary_synch0SL_time = (int16_t *)malloc16((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int16_t)*2);
  if (primary_synch0SL_time) {
    bzero(primary_synch0SL_time,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch0SL_time allocated at %p\n", primary_synch0SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch0SL_time not allocated\n");
  primary_synch0SL_time_rx = (int16_t *)malloc16(2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int16_t)*2);
  if (primary_synch0SL_time_rx) {
    bzero(primary_synch0SL_time_rx,(frame_parms->ofdm_symbol_size)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch0SL_time_rx allocated at %p\n", primary_synch0SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch0SL_time_rx not allocated\n");



  primary_synch1SL_time = (int16_t *)malloc16(((frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples))*sizeof(int16_t)*2);

  if (primary_synch1SL_time) {
    bzero(primary_synch1SL_time,(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch1SL_time allocated at %p\n", primary_synch1SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch1SL_time not allocated\n");
  primary_synch1SL_time_rx = (int16_t *)malloc16(2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int16_t)*2);
  if (primary_synch1SL_time_rx) {
    bzero(primary_synch1SL_time_rx,2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*sizeof(int16_t)*2);
#ifdef DEBUG_PHY
    LOG_D(PHY,"[openair][LTE_PHY][SYNC] primary_synch1SL_time_rx allocated at %p\n", primary_synch1SL_time);
#endif
  } else  AssertFatal(1==0,"primary_synch1SL_time_rx not allocated\n");


  memset((void*)syncF_tmp,0,2048*sizeof(int32_t));
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

  memset((void*)syncF_tmp,0,2048*sizeof(int32_t));

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

  memset((void*)syncF_tmp,0,2048*sizeof(int32_t));

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

  memset((void*)syncF_tmp,0,2048*sizeof(int32_t));

  k=frame_parms->ofdm_symbol_size-36;

  for (i=0; i<72; i++) {
    syncF_tmp[2*k] = primary_synch0SL[2*i]>>2;  //we need to shift input to avoid overflow in fft
    syncF_tmp[2*k+1] = primary_synch0SL[2*i+1]>>2;
    k++;

    if (k >= frame_parms->ofdm_symbol_size) {
      k-=frame_parms->ofdm_symbol_size;
    }
  }

  int16_t *kHz7_5ptr;


  switch (frame_parms->N_RB_DL) {
  case 6:
    idft128((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    kHz7_5ptr = (frame_parms->Ncp==0) ? ((int16_t*)s6n_kHz_7_5)+(2*138): ((int16_t*)s6e_kHz_7_5)+(2*160);

    break;
  case 25:
    idft512((int16_t*)syncF_tmp,          /// complex input
	   (int16_t*)sync_tmp, /// complex output
	   1);
    kHz7_5ptr = (frame_parms->Ncp==0) ? ((int16_t*)s25n_kHz_7_5)+(2*552) : ((int16_t*)s25e_kHz_7_5)+(2*640);

    break;
  case 50:
    idft1024((int16_t*)syncF_tmp,          /// complex input
	    (int16_t*)sync_tmp, /// complex output
	    1);
    kHz7_5ptr = (frame_parms->Ncp==0) ? ((int16_t*)s50n_kHz_7_5)+(2*1104) : ((int16_t*)s50e_kHz_7_5)+(2*1280);
    break;
    
  case 75:
    idft1536((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp,
	     1); /// complex output
    kHz7_5ptr = (frame_parms->Ncp==0) ? ((int16_t*)s75n_kHz_7_5)+(2*1656): ((int16_t*)s75e_kHz_7_5)+(2*1920);

    break;
  case 100:
    idft2048((int16_t*)syncF_tmp,          /// complex input
	     (int16_t*)sync_tmp, /// complex output
	     1);
    kHz7_5ptr = (frame_parms->Ncp==0) ? ((int16_t*)s100n_kHz_7_5)+(2*2208) : ((int16_t*)s100e_kHz_7_5)+(2*2560);

    break;
  default:
    LOG_E(PHY,"Unsupported N_RB_DL %d\n",frame_parms->N_RB_DL);
    kHz7_5ptr = NULL;
    break;
  }
  int imod;
  for (i=0; i<(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2; i++) {
    imod = i%(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples);
    if (i<(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples))
	((int32_t*)primary_synch0SL_time)[i] = sync_tmp[(i+(frame_parms->ofdm_symbol_size-frame_parms->nb_prefix_samples))%frame_parms->ofdm_symbol_size];

    primary_synch0SL_time_rx[i<<1]     = (int16_t)(((int32_t)primary_synch0SL_time[imod<<1]*kHz7_5ptr[i<<1])>>15) - (int16_t)(((int32_t)primary_synch0SL_time[1+(imod<<1)]*kHz7_5ptr[1+(i<<1)])>>15);
    primary_synch0SL_time_rx[1+(i<<1)] = (int16_t)(((int32_t)primary_synch0SL_time[imod<<1]*kHz7_5ptr[1+(i<<1)])>>15) + (int16_t)(((int32_t)primary_synch0SL_time[1+(imod<<1)]*kHz7_5ptr[i<<1])>>15);
  }
 
  memset((void*)syncF_tmp,0,2048*sizeof(int32_t));

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

  for (i=0;i<(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples);i++)
      ((int32_t*)primary_synch1SL_time)[i] = sync_tmp[(i+(frame_parms->ofdm_symbol_size-frame_parms->nb_prefix_samples))%frame_parms->ofdm_symbol_size];

  for (i=0; i<(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2; i++) {
    imod = i%(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples);

    primary_synch1SL_time_rx[i<<1]     = (int16_t)(((int32_t)primary_synch1SL_time[imod<<1]*kHz7_5ptr[i<<1])>>15) + 
                                         (int16_t)(((int32_t)primary_synch1SL_time[1+(imod<<1)]*kHz7_5ptr[1+(i<<1)])>>15);
    primary_synch1SL_time_rx[1+(i<<1)] = -(int16_t)(((int32_t)primary_synch1SL_time[imod<<1]*kHz7_5ptr[1+(i<<1)])>>15) +
                                         (int16_t)(((int32_t)primary_synch1SL_time[1+(imod<<1)]*kHz7_5ptr[i<<1])>>15);
 /*   printf("sync_timeSL1(%d) : (%d,%d) x (%d,%d)' = (%d,%d)\n",
          i,
          primary_synch1SL_time[imod<<1],
          primary_synch1SL_time[1+(imod<<1)],
          kHz7_5ptr[i<<1],
          kHz7_5ptr[1+(i<<1)],
          primary_synch1SL_time_rx[i<<1],
          primary_synch1SL_time_rx[1+(i<<1)]);
*/
  }


  
  LOG_M("primary_sync0.m","psync0",primary_synch0_time,frame_parms->ofdm_symbol_size,1,1);
  LOG_M("primary_sync1.m","psync1",primary_synch1_time,frame_parms->ofdm_symbol_size,1,1);
  LOG_M("primary_sync2.m","psync2",primary_synch2_time,frame_parms->ofdm_symbol_size,1,1);
  LOG_M("primary_syncSL0.m","psyncSL0",primary_synch0SL_time,frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples,1,1);
  LOG_M("primary_syncSL1.m","psyncSL1",primary_synch1SL_time,frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples,1,1);

    
  LOG_M("primary_syncSL1rx.m","psyncSL1rx",primary_synch1SL_time_rx,2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples),1,1);
  LOG_M("primary_syncSL0rx.m","psyncSL0rx",primary_synch0SL_time_rx,2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples),1,1);
  LOG_M("kHz75.m","kHz75",kHz7_5ptr,2*1096,1,1);
  
  if ( LOG_DUMPFLAG(DEBUG_LTEESTIM)){
    LOG_M("primary_sync0.m","psync0",primary_synch0_time,frame_parms->ofdm_symbol_size,1,1);
    LOG_M("primary_sync1.m","psync1",primary_synch1_time,frame_parms->ofdm_symbol_size,1,1);
    LOG_M("primary_sync2.m","psync2",primary_synch2_time,frame_parms->ofdm_symbol_size,1,1);
  }
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

static inline int64_t abs64(int64_t x)
{
  return (((int64_t)((int32_t*)&x)[0])*((int64_t)((int32_t*)&x)[0]) + ((int64_t)((int32_t*)&x)[1])*((int64_t)((int32_t*)&x)[1]));
}

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


  if ( LOG_DUMPFLAG(DEBUG_LTEESTIM)){
    static int debug_cnt;
    if (debug_cnt == 0) {
      LOG_M("sync_corr0_ue.m","synccorr0",sync_corr_ue0,2*length,1,2);
      LOG_M("sync_corr1_ue.m","synccorr1",sync_corr_ue1,2*length,1,2);
      LOG_M("sync_corr2_ue.m","synccorr2",sync_corr_ue2,2*length,1,2);
      LOG_M("rxdata0.m","rxd0",rxdata[0],length<<1,1,1);
      //    exit(-1);
    } else {
    debug_cnt++;
  }
} 


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
  
  // circular copy of beginning to end of rxdata buffer. Note: buffer should be big enough upon calling this function
  for (int ar=0;ar<frame_parms->nb_antennas_rx;ar++) memcpy((void*)&ue->common_vars.rxdata_syncSL[ar][2*length],
							    (void*)&ue->common_vars.rxdata_syncSL[ar][0],
							    frame_parms->ofdm_symbol_size);

  int64_t tmp0,tmp1;
  int64_t magtmp0,magtmp1,maxlev0=0,maxlev1=0;
  int     maxpos0=0,maxpos1=0;
  int64_t avg0=0,avg1=0;
  int64_t result;
  int32_t **rxdata = (int32_t**)ue->common_vars.rxdata_syncSL; ///rx data in time domain
  RU_t ru_tmp;
  int16_t **rxdata_7_5kHz    = (int16_t**)ue->sl_rxdata_7_5kHz;

  memset((void*)&ru_tmp,0,sizeof(RU_t));
  
  memcpy((void*)&ru_tmp.frame_parms,(void*)&ue->frame_parms,sizeof(LTE_DL_FRAME_PARMS));
  ru_tmp.N_TA_offset=0;
  ru_tmp.common.rxdata = rxdata;
  ru_tmp.common.rxdata_7_5kHz = (int32_t**)rxdata_7_5kHz;
  ru_tmp.nb_rx = frame_parms->nb_antennas_rx;
  
  int maxval=0;
  for (int i=0;i<2*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples);i++) {
    maxval = max(maxval,primary_synch0SL_time_rx[i]);
    maxval = max(maxval,-primary_synch0SL_time_rx[i]);
    maxval = max(maxval,primary_synch1SL_time_rx[i]);
    maxval = max(maxval,-primary_synch1SL_time_rx[i]);
  }
  int shift = log2_approx(maxval);//*(frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2);
  printf("Synchtime SL : shifting by %d bits\n",shift);
  for (int n=0; n<length; n+=4) 
  {

    tmp0 = 0;
    tmp1 = 0;
    int32_t tmp0_re=((int32_t*)&tmp0)[0], tmp0_im=((int32_t*)&tmp0)[1];
    int32_t tmp1_re=((int32_t*)&tmp1)[0], tmp1_im=((int32_t*)&tmp1)[1];

    //calculate dot product of primary_synch0_time and rxdata[ar][n] (ar=0..nb_ant_rx) and store the sum in temp[n];
    for (int ar=0; ar<frame_parms->nb_antennas_rx; ar++) {
      
      result  = dot_product(primary_synch0SL_time_rx,
			    (int16_t*) &(rxdata[ar][n]),
			    (frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2,
			    shift);
      
      tmp0_re += ((int32_t*) &result)[0];
      tmp0_im += ((int32_t*) &result)[1];

      result  = dot_product(primary_synch1SL_time_rx,
			    (int16_t*) &(rxdata[ar][n]),
			    (frame_parms->ofdm_symbol_size+frame_parms->nb_prefix_samples)*2,
			    shift);

      tmp1_re += ((int32_t*) &result)[0];
      tmp1_im += ((int32_t*) &result)[1];

    }

    // tmpi holds <synchi,rx0>+<synci,rx1>+...+<synchi,rx_{nbrx-1}>

    magtmp0 = (int64_t)tmp0_re*tmp0_re + (int64_t)tmp0_im*tmp0_im;
    magtmp1 = (int64_t)tmp1_re*tmp1_re + (int64_t)tmp1_im*tmp1_im;

    //printf("0: n %d (%d,%d) => %lld\n",n,tmp0_re,tmp0_im,magtmp0);
    //printf("1: n %d (%d,%d) => %lld\n",n,tmp1_re,tmp1_im,magtmp1);
    // this does max |tmpi(n)|^2 + |tmpi(n-L)|^2 and argmax |tmpi(n)|^2 + |tmpi(n-L)|^2
    
    if (magtmp0>maxlev0) { maxlev0 = magtmp0; maxpos0 = n; }
    if (magtmp1>maxlev1) { maxlev1 = magtmp1; maxpos1 = n; }
    avg0 += magtmp0;
    avg1 += magtmp1;
    if (n<4*FRAME_LENGTH_COMPLEX_SAMPLES) {
      sync_corr_ue1[n] = magtmp1;     
      sync_corr_ue0[n] = magtmp0;       
    }
  }
  avg0/=(length/4);
  avg1/=(length/4);

  // PSS in symbol 1
  int pssoffset = frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples0;
  printf("maxpos = (%d,%d), pssoffset = %d, maxlev= (%lld,%lld) avglev (%lld,%lld)\n",maxpos0,maxpos1,pssoffset,
         (long long int)maxlev0,(long long int)maxlev1,(long long int)avg0,(long long int)avg1);
 
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

  mean_val/=(length/4); 

  *peak_val_out = peak_val;

  if (peak_val <= (40*(uint32_t)mean_val)) {
    LOG_D(PHY,"[SYNC TIME] No peak found (%u,%u,%"PRIu64",%"PRIu64")\n",peak_pos,peak_val,mean_val,40*mean_val);
    return(-1);
  } else {
    LOG_D(PHY,"[SYNC TIME] Peak found at pos %u, val = %u, mean_val = %"PRIu64"\n",peak_pos,peak_val,mean_val);
    return(peak_pos);
  }

}




