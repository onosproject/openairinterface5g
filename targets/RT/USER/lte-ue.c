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

/*! \file lte-ue.c
 * \brief threads and support functions for real-time LTE UE target
 * \author R. Knopp, F. Kaltenberger, Navid Nikaein
 * \date 2015
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr, navid.nikaein@eurecom.fr
 * \note
 * \warning
 */
#include "lte-softmodem.h"

#include "rt_wrapper.h"

#ifdef OPENAIR2
#include "LAYER2/MAC/defs.h"
#include "RRC/LITE/extern.h"
#endif
#include "PHY_INTERFACE/extern.h"

#undef MALLOC //there are two conflicting definitions, so we better make sure we don't use it at all
//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all

#include "PHY/extern.h"
#include "SCHED/extern.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto.h"

#include "UTIL/LOG/log_extern.h"
#include "UTIL/OTG/otg_tx.h"
#include "UTIL/OTG/otg_externs.h"
#include "UTIL/MATH/oml.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"

#include "T.h"

#ifdef UE_EXPANSION_SIM2
#include "common/utils/udp/udp_com.h"
#endif

extern double cpuf;

#define FRAME_PERIOD    100000000ULL
#define DAQ_PERIOD      66667ULL
#define FIFO_PRIORITY   40

#ifdef UE_EXPANSION_SIM2
extern long TTI;

volatile int start_time_flag = 0;
struct timespec enb_start_time = { .tv_sec = 0, .tv_nsec = 0 };
void* UE_time_sync(void *arg);
static void* UE_phy_rev(void* arg);
static void* UE_phy_send(void* param);
#endif

typedef enum {
    pss=0,
    pbch=1,
    si=2
} sync_mode_t;

void init_UE_threads(int);
void *UE_thread(void *arg);
void init_UE(int nb_inst,int,int);

int32_t **rxdata;
int32_t **txdata;

#ifdef UE_EXPANSION_SIM2
UE_RX_RECEIVE_INFO ue_rx_receive_info[10];
//uint8_t pdcp_end_flag[RX_NB_TH][256];
int ue_sd_c;
int ue_sd_s;
extern int udp_socket_ip_enb;
extern int udp_socket_port_enb;
extern int udp_socket_ip_ue;
extern int udp_socket_port_ue;

int instance_cnt_send[RX_NB_TH] = {-1,-1};

extern pthread_mutex_t mutex_send[RX_NB_TH];
extern pthread_cond_t cond_send[RX_NB_TH];

extern pthread_cond_t cond_rxtx[RX_NB_TH];
/// mutex for RXn-TXnp4 processing thread
extern pthread_mutex_t mutex_rxtx[RX_NB_TH];

#endif

#define KHz (1000UL)
#define MHz (1000*KHz)

typedef struct eutra_band_s {
    int16_t band;
    uint32_t ul_min;
    uint32_t ul_max;
    uint32_t dl_min;
    uint32_t dl_max;
    lte_frame_type_t frame_type;
} eutra_band_t;

typedef struct band_info_s {
    int nbands;
    eutra_band_t band_info[100];
} band_info_t;

band_info_t bands_to_scan;

static const eutra_band_t eutra_bands[] = {
    { 1, 1920    * MHz, 1980    * MHz, 2110    * MHz, 2170    * MHz, FDD},
    { 2, 1850    * MHz, 1910    * MHz, 1930    * MHz, 1990    * MHz, FDD},
    { 3, 1710    * MHz, 1785    * MHz, 1805    * MHz, 1880    * MHz, FDD},
    { 4, 1710    * MHz, 1755    * MHz, 2110    * MHz, 2155    * MHz, FDD},
    { 5,  824    * MHz,  849    * MHz,  869    * MHz,  894    * MHz, FDD},
    { 6,  830    * MHz,  840    * MHz,  875    * MHz,  885    * MHz, FDD},
    { 7, 2500    * MHz, 2570    * MHz, 2620    * MHz, 2690    * MHz, FDD},
    { 8,  880    * MHz,  915    * MHz,  925    * MHz,  960    * MHz, FDD},
    { 9, 1749900 * KHz, 1784900 * KHz, 1844900 * KHz, 1879900 * KHz, FDD},
    {10, 1710    * MHz, 1770    * MHz, 2110    * MHz, 2170    * MHz, FDD},
    {11, 1427900 * KHz, 1452900 * KHz, 1475900 * KHz, 1500900 * KHz, FDD},
    {12,  698    * MHz,  716    * MHz,  728    * MHz,  746    * MHz, FDD},
    {13,  777    * MHz,  787    * MHz,  746    * MHz,  756    * MHz, FDD},
    {14,  788    * MHz,  798    * MHz,  758    * MHz,  768    * MHz, FDD},
    {17,  704    * MHz,  716    * MHz,  734    * MHz,  746    * MHz, FDD},
    {20,  832    * MHz,  862    * MHz,  791    * MHz,  821    * MHz, FDD},
    {22, 3510    * MHz, 3590    * MHz, 3410    * MHz, 3490    * MHz, FDD},
    {33, 1900    * MHz, 1920    * MHz, 1900    * MHz, 1920    * MHz, TDD},
    {34, 2010    * MHz, 2025    * MHz, 2010    * MHz, 2025    * MHz, TDD},
    {35, 1850    * MHz, 1910    * MHz, 1850    * MHz, 1910    * MHz, TDD},
    {36, 1930    * MHz, 1990    * MHz, 1930    * MHz, 1990    * MHz, TDD},
    {37, 1910    * MHz, 1930    * MHz, 1910    * MHz, 1930    * MHz, TDD},
    {38, 2570    * MHz, 2620    * MHz, 2570    * MHz, 2630    * MHz, TDD},
    {39, 1880    * MHz, 1920    * MHz, 1880    * MHz, 1920    * MHz, TDD},
    {40, 2300    * MHz, 2400    * MHz, 2300    * MHz, 2400    * MHz, TDD},
    {41, 2496    * MHz, 2690    * MHz, 2496    * MHz, 2690    * MHz, TDD},
    {42, 3400    * MHz, 3600    * MHz, 3400    * MHz, 3600    * MHz, TDD},
    {43, 3600    * MHz, 3800    * MHz, 3600    * MHz, 3800    * MHz, TDD},
    {44, 703    * MHz, 803    * MHz, 703    * MHz, 803    * MHz, TDD},
};




pthread_t                       main_ue_thread;
pthread_attr_t                  attr_UE_thread;
struct sched_param              sched_param_UE_thread;

void phy_init_lte_ue_transport(PHY_VARS_UE *ue,int absraction_flag);

PHY_VARS_UE* init_ue_vars(LTE_DL_FRAME_PARMS *frame_parms,
			  uint8_t UE_id,
			  uint8_t abstraction_flag)

{

  PHY_VARS_UE* ue;

  if (frame_parms!=(LTE_DL_FRAME_PARMS *)NULL) { // if we want to give initial frame parms, allocate the PHY_VARS_UE structure and put them in
    ue = (PHY_VARS_UE *)malloc(sizeof(PHY_VARS_UE));
    memset(ue,0,sizeof(PHY_VARS_UE));
    memcpy(&(ue->frame_parms), frame_parms, sizeof(LTE_DL_FRAME_PARMS));
  }					
  else ue = PHY_vars_UE_g[UE_id][0];


  ue->Mod_id      = UE_id;
  ue->mac_enabled = 1;
  // initialize all signal buffers
  init_lte_ue_signal(ue,1,abstraction_flag);
  // intialize transport
  init_lte_ue_transport(ue,abstraction_flag);

  return(ue);
}


char uecap_xer[1024];



void init_thread(int sched_runtime, int sched_deadline, int sched_fifo, cpu_set_t *cpuset, char * name) {

#ifdef DEADLINE_SCHEDULER
    if (sched_runtime!=0) {
        struct sched_attr attr= {0};
        attr.size = sizeof(attr);
        attr.sched_policy = SCHED_DEADLINE;
        attr.sched_runtime  = sched_runtime;
        attr.sched_deadline = sched_deadline;
        attr.sched_period   = 0;
        AssertFatal(sched_setattr(0, &attr, 0) == 0,
                    "[SCHED] %s thread: sched_setattr failed %s \n", name, strerror(errno));
        LOG_I(HW,"[SCHED][eNB] %s deadline thread %lu started on CPU %d\n",
              name, (unsigned long)gettid(), sched_getcpu());
    }
#else
    if (CPU_COUNT(cpuset) > 0)
        AssertFatal( 0 == pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), cpuset), "");
    struct sched_param sp;
    sp.sched_priority = sched_fifo;
    AssertFatal(pthread_setschedparam(pthread_self(),SCHED_FIFO,&sp)==0,
                "Can't set thread priority, Are you root?\n");
    /* Check the actual affinity mask assigned to the thread */
    cpu_set_t *cset=CPU_ALLOC(CPU_SETSIZE);
    if (0 == pthread_getaffinity_np(pthread_self(), CPU_ALLOC_SIZE(CPU_SETSIZE), cset)) {
      char txt[512]={0};
      for (int j = 0; j < CPU_SETSIZE; j++)
        if (CPU_ISSET(j, cset))
	  sprintf(txt+strlen(txt), " %d ", j);
      printf("CPU Affinity of thread %s is %s\n", name, txt);
    }
    CPU_FREE(cset);
#endif

}

void init_UE(int nb_inst,int eMBMS_active, int uecap_xer_in) {

  PHY_VARS_UE *UE;
  int         inst;
  int         ret;

  LOG_I(PHY,"UE : Calling Layer 2 for initialization\n");
    
#ifdef UE_EXPANSION_SIM2
  uecap_xer_in=0;

  pthread_cond_init(&cond_rxtx[0],NULL);
  pthread_cond_init(&cond_rxtx[1],NULL);
  pthread_mutex_init(&mutex_rxtx[0], NULL);
  pthread_mutex_init(&mutex_rxtx[1], NULL);

#endif
  l2_init_ue(eMBMS_active,(uecap_xer_in==1)?uecap_xer:NULL,
	     0,// cba_group_active
	     0); // HO flag
  
  for (inst=0;inst<nb_inst;inst++) {

    LOG_I(PHY,"Initializing memory for UE instance %d (%p)\n",inst,PHY_vars_UE_g[inst]);
#ifndef UE_EXPANSION_SIM2
    PHY_vars_UE_g[inst][0] = init_ue_vars(NULL,inst,0);
#endif

    LOG_I(PHY,"Intializing UE Threads for instance %d (%p,%p)...\n",inst,PHY_vars_UE_g[inst],PHY_vars_UE_g[inst][0]);
    init_UE_threads(inst);

    UE = PHY_vars_UE_g[inst][0];
#ifndef UE_EXPANSION_SIM2
    if (oaisim_flag == 0) {
      ret = openair0_device_load(&(UE->rfdevice), &openair0_cfg[0]);
      if (ret !=0){
	exit_fun("Error loading device library");
      }
    }

    UE->rfdevice.host_type = RAU_HOST;
    //    UE->rfdevice.type      = NONE_DEV;
    PHY_VARS_UE *UE = PHY_vars_UE_g[inst][0];
   AssertFatal(0 == pthread_create(&UE->proc.pthread_ue,
                                    &UE->proc.attr_ue,
                                    UE_thread,
                                    (void*)UE), "");
#else
#ifdef NAS_UE
    MessageDef *message_p;
    message_p = itti_alloc_new_message(TASK_NAS_UE, INITIALIZE_MESSAGE);
    itti_send_msg_to_task (TASK_NAS_UE, UE->Mod_id + NB_eNB_INST, message_p);
#endif
#endif
  }

#ifdef UE_EXPANSION_SIM2
  pthread_mutex_init(&mutex_send[0], NULL);
  pthread_mutex_init(&mutex_send[1], NULL);
  pthread_cond_init(&cond_send[0], NULL);
  pthread_cond_init(&cond_send[1], NULL);

  for(inst=0;inst<RX_NB_TH;inst++){
    pthread_create(&PHY_vars_UE_g[0][0]->proc.pthread_phy_send, NULL, UE_phy_send, (void*)&inst);
    usleep(1000);
  }
  pthread_create(&PHY_vars_UE_g[0][0]->proc.pthread_time, NULL, UE_time_sync,(void*)NULL);
  pthread_create(&PHY_vars_UE_g[0][0]->proc.pthread_phy_stub, NULL, UE_phy_rev, (void*)NULL);
#endif
  printf("UE threads created by %ld\n", gettid());
#if 0
#if defined(ENABLE_USE_MME)
  extern volatile int start_UE;
  while (start_UE == 0) {
    sleep(1);
  }
#endif
#endif
}

/*!
 * \brief This is the UE synchronize thread.
 * It performs band scanning and synchonization.
 * \param arg is a pointer to a \ref PHY_VARS_UE structure.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */

static void *UE_thread_synch(void *arg)
{
  static int UE_thread_synch_retval;
  int i, hw_slot_offset;
  PHY_VARS_UE *UE = (PHY_VARS_UE*) arg;
  int current_band = 0;
  int current_offset = 0;
  sync_mode_t sync_mode = pbch;
  int CC_id = UE->CC_id;
  int ind;
  int found;
  int freq_offset=0;
  char threadname[128];

  UE->is_synchronized = 0;
  printf("UE_thread_sync in with PHY_vars_UE %p\n",arg);

   cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  if ( threads.iq != -1 )
    CPU_SET(threads.iq, &cpuset);
  // this thread priority must be lower that the main acquisition thread
  sprintf(threadname, "sync UE %d\n", UE->Mod_id);
  init_thread(100000, 500000, FIFO_PRIORITY-1, &cpuset, threadname);
  
  printf("starting UE synch thread (IC %d)\n",UE->proc.instance_cnt_synch);
  ind = 0;
  found = 0;


  if (UE->UE_scan == 0) {
    do  {
      current_band = eutra_bands[ind].band;
      printf( "Scanning band %d, dl_min %"PRIu32", ul_min %"PRIu32"\n", current_band, eutra_bands[ind].dl_min,eutra_bands[ind].ul_min);

      if ((eutra_bands[ind].dl_min <= UE->frame_parms.dl_CarrierFreq) && (eutra_bands[ind].dl_max >= UE->frame_parms.dl_CarrierFreq)) {
	for (i=0; i<4; i++)
	  uplink_frequency_offset[CC_id][i] = eutra_bands[ind].ul_min - eutra_bands[ind].dl_min;

        found = 1;
        break;
      }

      ind++;
    } while (ind < sizeof(eutra_bands) / sizeof(eutra_bands[0]));
  
    if (found == 0) {
      LOG_E(PHY,"Can't find EUTRA band for frequency %d",UE->frame_parms.dl_CarrierFreq);
      exit_fun("Can't find EUTRA band for frequency");
      return &UE_thread_synch_retval;
    }


    LOG_I( PHY, "[SCHED][UE] Check absolute frequency DL %"PRIu32", UL %"PRIu32" (oai_exit %d, rx_num_channels %d)\n", UE->frame_parms.dl_CarrierFreq, UE->frame_parms.ul_CarrierFreq,oai_exit, openair0_cfg[0].rx_num_channels);

    for (i=0;i<openair0_cfg[UE->rf_map.card].rx_num_channels;i++) {
      openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = UE->frame_parms.dl_CarrierFreq;
      openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] = UE->frame_parms.ul_CarrierFreq;
      openair0_cfg[UE->rf_map.card].autocal[UE->rf_map.chain+i] = 1;
      if (uplink_frequency_offset[CC_id][i] != 0) // 
	openair0_cfg[UE->rf_map.card].duplex_mode = duplex_mode_FDD;
      else //FDD
	openair0_cfg[UE->rf_map.card].duplex_mode = duplex_mode_TDD;
    }

    sync_mode = pbch;

  } else if  (UE->UE_scan == 1) {
    current_band=0;

    for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
      downlink_frequency[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[CC_id].dl_min;
      uplink_frequency_offset[UE->rf_map.card][UE->rf_map.chain+i] =
	bands_to_scan.band_info[CC_id].ul_min-bands_to_scan.band_info[CC_id].dl_min;
      openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i];
      openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] =
	downlink_frequency[CC_id][i]+uplink_frequency_offset[CC_id][i];
      openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;
    }
  }

  while (sync_var<0)     
    pthread_cond_wait(&sync_cond, &sync_mutex);   
  pthread_mutex_unlock(&sync_mutex);   

  printf("Started device, unlocked sync_mutex (UE_sync_thread)\n");   

  if (UE->rfdevice.trx_start_func(&UE->rfdevice) != 0 ) {     
    LOG_E(HW,"Could not start the device\n");     
    oai_exit=1;   
  }

  while (oai_exit==0) {
    AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
    while (UE->proc.instance_cnt_synch < 0)
      // the thread waits here most of the time
      pthread_cond_wait( &UE->proc.cond_synch, &UE->proc.mutex_synch );
    AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");
    
    switch (sync_mode) {
    case pss:
      LOG_I(PHY,"[SCHED][UE] Scanning band %d (%d), freq %u\n",bands_to_scan.band_info[current_band].band, current_band,bands_to_scan.band_info[current_band].dl_min+current_offset);
      lte_sync_timefreq(UE,current_band,bands_to_scan.band_info[current_band].dl_min+current_offset);
      current_offset += 20000000; // increase by 20 MHz
      
      if (current_offset > bands_to_scan.band_info[current_band].dl_max-bands_to_scan.band_info[current_band].dl_min) {
	current_band++;
                current_offset=0;
            }

            if (current_band==bands_to_scan.nbands) {
                current_band=0;
                oai_exit=1;
            }

            for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
                downlink_frequency[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[current_band].dl_min+current_offset;
                uplink_frequency_offset[UE->rf_map.card][UE->rf_map.chain+i] = bands_to_scan.band_info[current_band].ul_min-bands_to_scan.band_info[0].dl_min + current_offset;

                openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i];
                openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i]+uplink_frequency_offset[CC_id][i];
                openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;
                if (UE->UE_scan_carrier) {
                    openair0_cfg[UE->rf_map.card].autocal[UE->rf_map.chain+i] = 1;
                }
	    }

	    break;
 
    case pbch:

#if DISABLE_LOG_X
            printf("[UE thread Synch] Running Initial Synch (mode %d)\n",UE->mode);
#else
            LOG_I(PHY, "[UE thread Synch] Running Initial Synch (mode %d)\n",UE->mode);
#endif
            if (initial_sync( UE, UE->mode ) == 0) {

                hw_slot_offset = (UE->rx_offset<<1) / UE->frame_parms.samples_per_tti;
                LOG_I( HW, "Got synch: hw_slot_offset %d, carrier off %d Hz, rxgain %d (DL %u, UL %u), UE_scan_carrier %d\n",
                       hw_slot_offset,
                       freq_offset,
                       UE->rx_total_gain_dB,
                       downlink_frequency[0][0]+freq_offset,
                       downlink_frequency[0][0]+uplink_frequency_offset[0][0]+freq_offset,
                       UE->UE_scan_carrier );


                    // rerun with new cell parameters and frequency-offset
                    for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
                        openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;//-USRP_GAIN_OFFSET;
			if (UE->UE_scan_carrier == 1) {
                        if (freq_offset >= 0)
                            openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] += abs(UE->common_vars.freq_offset);
                        else
                            openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] -= abs(UE->common_vars.freq_offset);
                        openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] =
                            openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i]+uplink_frequency_offset[CC_id][i];
                        downlink_frequency[CC_id][i] = openair0_cfg[CC_id].rx_freq[i];
                        freq_offset=0;
                    }
	  }

                    // reconfigure for potentially different bandwidth
                    switch(UE->frame_parms.N_RB_DL) {
                    case 6:
                        openair0_cfg[UE->rf_map.card].sample_rate =1.92e6;
                        openair0_cfg[UE->rf_map.card].rx_bw          =.96e6;
                        openair0_cfg[UE->rf_map.card].tx_bw          =.96e6;
                        //            openair0_cfg[0].rx_gain[0] -= 12;
                        break;
                    case 25:
                        openair0_cfg[UE->rf_map.card].sample_rate =7.68e6;
                        openair0_cfg[UE->rf_map.card].rx_bw          =2.5e6;
                        openair0_cfg[UE->rf_map.card].tx_bw          =2.5e6;
                        //            openair0_cfg[0].rx_gain[0] -= 6;
                        break;
                    case 50:
                        openair0_cfg[UE->rf_map.card].sample_rate =15.36e6;
                        openair0_cfg[UE->rf_map.card].rx_bw          =5.0e6;
                        openair0_cfg[UE->rf_map.card].tx_bw          =5.0e6;
                        //            openair0_cfg[0].rx_gain[0] -= 3;
                        break;
                    case 100:
                        openair0_cfg[UE->rf_map.card].sample_rate=30.72e6;
                        openair0_cfg[UE->rf_map.card].rx_bw=10.0e6;
                        openair0_cfg[UE->rf_map.card].tx_bw=10.0e6;
                        //            openair0_cfg[0].rx_gain[0] -= 0;
                        break;
                    }

                    UE->rfdevice.trx_set_freq_func(&UE->rfdevice,&openair0_cfg[0],0);
                    //UE->rfdevice.trx_set_gains_func(&openair0,&openair0_cfg[0]);
                    //UE->rfdevice.trx_stop_func(&UE->rfdevice);
                    sleep(1);
                    init_frame_parms(&UE->frame_parms,1);
                    /*if (UE->rfdevice.trx_start_func(&UE->rfdevice) != 0 ) {
                        LOG_E(HW,"Could not start the device\n");
                        oai_exit=1;
                    }*/

		if (UE->UE_scan_carrier == 1) {

		  UE->UE_scan_carrier = 0;
                } else {
                    AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
                    UE->is_synchronized = 1;
                    AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");

                    if( UE->mode == rx_dump_frame ) {
                        FILE *fd;
                        if ((UE->proc.proc_rxtx[0].frame_rx&1) == 0) {  // this guarantees SIB1 is present
                            if ((fd = fopen("rxsig_frame0.dat","w")) != NULL) {
                                fwrite((void*)&UE->common_vars.rxdata[0][0],
                                       sizeof(int32_t),
                                       10*UE->frame_parms.samples_per_tti,
                                       fd);
                                LOG_I(PHY,"Dummping Frame ... bye bye \n");
                                fclose(fd);
                                exit(0);
                            } else {
                                LOG_E(PHY,"Cannot open file for writing\n");
                                exit(0);
                            }
                        } else {
                            AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
                            UE->is_synchronized = 0;
                            AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");

                        }
                    }
                }
            } else {
                // initial sync failed
                // calculate new offset and try again
                if (UE->UE_scan_carrier == 1) {
                    if (freq_offset >= 0)
                        freq_offset += 100;
                    freq_offset *= -1;

                    if (abs(freq_offset) > 7500) {
                        LOG_I( PHY, "[initial_sync] No cell synchronization found, abandoning\n" );
                        FILE *fd;
                        if ((fd = fopen("rxsig_frame0.dat","w"))!=NULL) {
                            fwrite((void*)&UE->common_vars.rxdata[0][0],
                                   sizeof(int32_t),
                                   10*UE->frame_parms.samples_per_tti,
                                   fd);
                            LOG_I(PHY,"Dummping Frame ... bye bye \n");
                            fclose(fd);
                            exit(0);
                        }
                        AssertFatal(1==0,"No cell synchronization found, abandoning");
                        return &UE_thread_synch_retval; // not reached
                    }
                }
#if DISABLE_LOG_X
                printf("[initial_sync] trying carrier off %d Hz, rxgain %d (DL %u, UL %u)\n",
                       freq_offset,
                       UE->rx_total_gain_dB,
                       downlink_frequency[0][0]+freq_offset,
                       downlink_frequency[0][0]+uplink_frequency_offset[0][0]+freq_offset );
#else
                LOG_I(PHY, "[initial_sync] trying carrier off %d Hz, rxgain %d (DL %u, UL %u)\n",
                       freq_offset,
                       UE->rx_total_gain_dB,
                       downlink_frequency[0][0]+freq_offset,
                       downlink_frequency[0][0]+uplink_frequency_offset[0][0]+freq_offset );
#endif

                for (i=0; i<openair0_cfg[UE->rf_map.card].rx_num_channels; i++) {
                    openair0_cfg[UE->rf_map.card].rx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i]+freq_offset;
                    openair0_cfg[UE->rf_map.card].tx_freq[UE->rf_map.chain+i] = downlink_frequency[CC_id][i]+uplink_frequency_offset[CC_id][i]+freq_offset;
                    openair0_cfg[UE->rf_map.card].rx_gain[UE->rf_map.chain+i] = UE->rx_total_gain_dB;//-USRP_GAIN_OFFSET;
                    if (UE->UE_scan_carrier==1)
                        openair0_cfg[UE->rf_map.card].autocal[UE->rf_map.chain+i] = 1;
                }
                UE->rfdevice.trx_set_freq_func(&UE->rfdevice,&openair0_cfg[0],0);
            }// initial_sync=0
            break;
        case si:
        default:
            break;
        }

        AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
        // indicate readiness
        UE->proc.instance_cnt_synch--;
        AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");

        VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_UE_THREAD_SYNCH, 0 );
    }  // while !oai_exit

    return &UE_thread_synch_retval;
}

/*!
 * \brief This is the UE thread for RX subframe n and TX subframe n+4.
 * This thread performs the phy_procedures_UE_RX() on every received slot.
 * then, if TX is enabled it performs TX for n+4.
 * \param arg is a pointer to a \ref PHY_VARS_UE structure.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */

static void *UE_thread_rxn_txnp4(void *arg) {
    static __thread int UE_thread_rxtx_retval;
    struct rx_tx_thread_data *rtd = arg;
    UE_rxtx_proc_t *proc = rtd->proc;
    PHY_VARS_UE    *UE   = rtd->UE;
    int ret;

    proc->instance_cnt_rxtx=-1;
    proc->subframe_rx=proc->sub_frame_start;

    char threadname[256];
    sprintf(threadname,"UE_%d_proc_%d", UE->Mod_id, proc->sub_frame_start);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
#if 0
    if ( (proc->sub_frame_start+1)%RX_NB_TH == 0 && threads.one != -1 )
        CPU_SET(threads.one, &cpuset);
    if ( (proc->sub_frame_start+1)%RX_NB_TH == 1 && threads.two != -1 )
        CPU_SET(threads.two, &cpuset);
    if ( (proc->sub_frame_start+1)%RX_NB_TH == 2 && threads.three != -1 )
        CPU_SET(threads.three, &cpuset);
            //CPU_SET(threads.three, &cpuset);
#endif
    CPU_SET(18+(UE->Mod_id%8), &cpuset);
    init_thread(900000,1000000 , FIFO_PRIORITY-1, &cpuset,
                threadname);

    while (!oai_exit) {
          // most of the time, the thread is waiting here
#ifdef UE_EXPANSION_SIM2
        if (pthread_mutex_lock(&mutex_rxtx[proc->sub_frame_start]) != 0){
          LOG_E( PHY, "[SCHED][UE] error locking mutex for RXn_TXnp4\n" );
          exit_fun("nothing to add");
        }

        pthread_cond_wait(&cond_rxtx[proc->sub_frame_start],&mutex_rxtx[proc->sub_frame_start]) ;

        if (pthread_mutex_unlock(&mutex_rxtx[proc->sub_frame_start]) != 0) {
          LOG_E( PHY, "[SCHED][UE] error unlocking mutex for UE RXn_TXnp4\n" );
          exit_fun("nothing to add");
        }
#else
        if (pthread_mutex_lock(&proc->mutex_rxtx) != 0) {
          LOG_E( PHY, "[SCHED][UE] error locking mutex for UE RXTX\n" );
          exit_fun("nothing to add");
        }
        while (proc->instance_cnt_rxtx < 0) {
          // most of the time, the thread is waiting here
          pthread_cond_wait( &proc->cond_rxtx, &proc->mutex_rxtx );
        }
        if (pthread_mutex_unlock(&proc->mutex_rxtx) != 0) {
          LOG_E( PHY, "[SCHED][UE] error unlocking mutex for UE RXn_TXnp4\n" );
          exit_fun("nothing to add");
        }
#endif


        initRefTimes(t2);
        initRefTimes(t3);
        pickTime(current);
        updateTimes(proc->gotIQs, &t2, 10000, "Delay to wake up UE_Thread_Rx (case 2)");

#ifdef UE_EXPANSION_SIM2
        if (UE->Mod_id != 0) {
          proc->frame_rx = PHY_vars_UE_g[0][UE->CC_id]->proc.proc_rxtx[proc->sub_frame_start].frame_rx;
          proc->subframe_rx = PHY_vars_UE_g[0][UE->CC_id]->proc.proc_rxtx[proc->sub_frame_start].subframe_rx;
          proc->frame_tx = PHY_vars_UE_g[0][UE->CC_id]->proc.proc_rxtx[proc->sub_frame_start].frame_tx;
          proc->subframe_tx = PHY_vars_UE_g[0][UE->CC_id]->proc.proc_rxtx[proc->sub_frame_start].subframe_tx;
        }
        if(UE->UE_mode[0] >= PRACH){
#endif
        // Process Rx data for one sub-frame
        lte_subframe_t sf_type = subframe_select( &UE->frame_parms, proc->subframe_rx);
        if ((sf_type == SF_DL) ||
                (UE->frame_parms.frame_type == FDD) ||
                (sf_type == SF_S)) {

            if (UE->frame_parms.frame_type == TDD) {
                LOG_D(PHY, "%s,TDD%d,%s: calling UE_RX\n",
                      threadname,
                      UE->frame_parms.tdd_config,
                      (sf_type==SF_DL? "SF_DL" :
                       (sf_type==SF_UL? "SF_UL" :
                        (sf_type==SF_S ? "SF_S"  : "UNKNOWN_SF_TYPE"))));
            } else {
                LOG_D(PHY, "%s,%s,%s: calling UE_RX\n",
                      threadname,
                      (UE->frame_parms.frame_type==FDD? "FDD":
                       (UE->frame_parms.frame_type==TDD? "TDD":"UNKNOWN_DUPLEX_MODE")),
                      (sf_type==SF_DL? "SF_DL" :
                       (sf_type==SF_UL? "SF_UL" :
                        (sf_type==SF_S ? "SF_S"  : "UNKNOWN_SF_TYPE"))));
            }
#ifndef UE_EXPANSION_SIM2
#ifdef UE_SLOT_PARALLELISATION
            phy_procedures_slot_parallelization_UE_RX( UE, proc, 0, 0, 1, UE->mode, no_relay, NULL );
#else
            phy_procedures_UE_RX( UE, proc, 0, 0, 1, UE->mode, no_relay, NULL );
#endif
#else
            //DLSCH RX
            UE_RX_RECEIVE_INFO*    rx_info = &ue_rx_receive_info[proc->subframe_rx];
            //ULSCH ACK
            if (is_phich_subframe(&UE->frame_parms,proc->subframe_rx)) {
                uint8_t harq_pid = phich_subframe_to_harq_pid(&UE->frame_parms,proc->frame_rx,proc->subframe_rx);
                if (UE->ulsch[0]->harq_processes[harq_pid]->status == ACTIVE) {
                    UE->ulsch[0]->harq_processes[harq_pid]->status                   = SCH_IDLE;
                	UE->ulsch[0]->harq_processes[harq_pid]->round                    = 0;
                	UE->ulsch[0]->harq_processes[harq_pid]->subframe_scheduling_flag = 0;
                	UE->ulsch_Msg3_active[0] = 0;
                }
            }
            //DCI
            for(int i = 0;i< rx_info->dci_num;i++){
                //SI DCI
                if((rx_info->dci_alloc[i].format == format1A) && (rx_info->dci_alloc[i].rnti == SI_RNTI)){
                    for(int j = 0;j<rx_info->pdu_num;j++){
                        if(rx_info->pdu_info[j].pdsch_type == PDSCH_SI){
                          ue_decode_si(UE->Mod_id,
                                       UE->CC_id,
                                       proc->frame_rx,
                                       0,
                                       &rx_info->pdu_buffer[rx_info->pdu_info[j].pdu_start_index],
                                       rx_info->pdu_info[j].pdu_length);
                          break;
                        }
                    }
                }else if((rx_info->dci_alloc[i].format == format1A) && (rx_info->dci_alloc[i].rnti == P_RNTI)){  //P DCI
                    for(int j = 0;j<rx_info->pdu_num;j++){
                        if(rx_info->pdu_info[j].pdsch_type == PDSCH_P){
                          ue_decode_p(UE->Mod_id,
                                       UE->CC_id,
                                       proc->frame_rx,
                                       0,
                                       &rx_info->pdu_buffer[rx_info->pdu_info[j].pdu_start_index],
                                       rx_info->pdu_info[j].pdu_length);
                          break;
                        }
                    }
                }else if((UE->UE_mode[0] == PRACH) && (UE->prach_resources[0]) &&//RA DCI
                   (rx_info->dci_alloc[i].format == format1A) &&
                   (rx_info->dci_alloc[i].rnti == UE->prach_resources[0]->ra_RNTI)){
                    for(int j = 0;j<rx_info->pdu_num;j++){
                        if(rx_info->pdu_info[j].pdsch_type == PDSCH_RA){
                          memcpy(UE->dlsch_ra[0]->harq_processes[0]->b,
                                   &rx_info->pdu_buffer[rx_info->pdu_info[j].pdu_start_index],
                                   rx_info->pdu_info[j].pdu_length);
                          process_rar(UE,proc,0,UE->mode,0);
                          break;
                        }
                    }
                }else if((UE->UE_mode[0] > PRACH) &&  //DLSCH DCI
                   (rx_info->dci_alloc[i].rnti == UE->pdcch_vars[UE->current_thread_id[proc->subframe_rx]][0]->crnti) &&
                   (rx_info->dci_alloc[i].format != format0)){
                    for(int j = 0;j<rx_info->ue_num;j++){
                        if(rx_info->ue_rx_info[j].rnti == rx_info->dci_alloc[i].rnti){
                          ue_send_sdu(UE->Mod_id,
                                UE->CC_id,
                                proc->frame_rx,
                                proc->subframe_rx,
                                &rx_info->ue_rx_info[j].pdu_buffer[0],
                                rx_info->ue_rx_info[j].pdu_length,
                                0);
                          break;
                        }
                    }
                }else if((UE->UE_mode[0] > PRACH) &&  //ULSCH DCI
                   (rx_info->dci_alloc[i].rnti == UE->pdcch_vars[UE->current_thread_id[proc->subframe_rx]][0]->crnti) &&
                   (rx_info->dci_alloc[i].format == format0)){
                    if (generate_ue_ulsch_params_from_dci(
                                (void *)&rx_info->dci_alloc[i].dci_pdu,
                                 UE->pdcch_vars[UE->current_thread_id[proc->subframe_rx]][0]->crnti,
                                 proc->subframe_rx,
                                 format0,
                                 UE,
                                 proc,
                                 SI_RNTI,
                                 0,
                                 P_RNTI,
                                 CBA_RNTI,
                                 0,
                                 0)==0) {
                        LOG_D(PHY,"[UE  %d] Generate UE ULSCH C_RNTI format 0 (subframe %d)\n",UE->Mod_id,proc->subframe_rx);
                    }
                }
            }
            //memset(rx_info,0,sizeof(UE_RX_RECEIVE_INFO));
#endif
        }

#if UE_TIMING_TRACE
        start_meas(&UE->generic_stat);
#endif
        if (UE->mac_enabled==1) {

            ret = ue_scheduler(UE->Mod_id,
			       proc->frame_rx,
			       proc->subframe_rx,
			       proc->frame_tx,
			       proc->subframe_tx,
			       subframe_select(&UE->frame_parms,proc->subframe_tx),
			       0,
			       0/*FIXME CC_id*/
#ifdef UE_EXPANSION_SIM2
                   ,proc->sub_frame_start
#endif
                   );
            if ( ret != CONNECTION_OK) {
                char *txt;
                switch (ret) {
                case CONNECTION_LOST:
                    txt="RRC Connection lost, returning to PRACH";
#ifdef UE_EXPANSION_SIM2
                    UE->UE_mode[0] = PRACH;
#endif
                    break;
                case PHY_RESYNCH:
                    txt="RRC Connection lost, trying to resynch";
                    break;
                case RESYNCH:
                    txt="return to PRACH and perform a contention-free access";
                    break;
                default:
                    txt="UNKNOWN RETURN CODE";
                };
                LOG_E( PHY, "[UE %"PRIu8"] Frame %"PRIu32", subframe %u %s\n",
                       UE->Mod_id, proc->frame_rx, proc->subframe_tx,txt );
            }
        }
#if UE_TIMING_TRACE
        stop_meas(&UE->generic_stat);
#endif


        // Prepare the future Tx data

        if ((subframe_select( &UE->frame_parms, proc->subframe_tx) == SF_UL) ||
	    (UE->frame_parms.frame_type == FDD) )
            if (UE->mode != loop_through_memory)
                phy_procedures_UE_TX(UE,proc,0,0,UE->mode,no_relay);


#ifndef UE_EXPANSION_SIM2

        if ((subframe_select( &UE->frame_parms, proc->subframe_tx) == SF_S) &&
                (UE->frame_parms.frame_type == TDD))
            if (UE->mode != loop_through_memory)
                phy_procedures_UE_S_TX(UE,0,0,no_relay);
#else
        }else{
            //pdcp_end_flag[proc->sub_frame_start][UE->Mod_id] = 1;
        }
        ue_tx_info[proc->sub_frame_start][UE->Mod_id].flag = 1;
#endif
        updateTimes(current, &t3, 10000, "Delay to process sub-frame (case 3)");



#ifndef UE_EXPANSION_SIM2
        if (pthread_mutex_lock(&proc->mutex_rxtx) != 0) {
          LOG_E( PHY, "[SCHED][UE] error locking mutex for UE RXTX\n" );
          exit_fun("noting to add");
        }
        proc->instance_cnt_rxtx--;
        if (pthread_mutex_unlock(&proc->mutex_rxtx) != 0) {
          LOG_E( PHY, "[SCHED][UE] error unlocking mutex for UE RXTX\n" );
          exit_fun("noting to add");
        }
#endif

    }

// thread finished
    free(arg);
    return &UE_thread_rxtx_retval;
}

#ifdef UE_EXPANSION_SIM2
static void wait_system_ready (char *message, volatile int *start_flag) {

  static char *indicator[] = {".    ", "..   ", "...  ", ".... ", ".....",
                  " ....", "  ...", "   ..", "    .", "     "};
  int i = 0;

  while ((!oai_exit) && (*start_flag == 0)) {
    LOG_N(EMU, message, indicator[i]);
    fflush(stdout);
    i = (i + 1) % (sizeof(indicator) / sizeof(indicator[0]));
    usleep(10000);
  }

  LOG_D(EMU,"\n");
}
void* UE_time_sync(void *arg){
    static __thread int UE_time_sync_retval;

    //PHY_VARS_UE *UE = (PHY_VARS_UE*) arg;
    long n;
    uint16_t  frame;
    uint8_t   subframe;
    uint8_t next_ue_start = 0;
    struct timespec now_time,req_time;
    struct timespec tti_time = { .tv_sec = 0, .tv_nsec = 0 };
    struct timespec rem_time = { .tv_sec = 0, .tv_nsec = 0 };
    char threadname[16];
    long int diff_sec,diff_nsec,usleep_time;
    int inst;
    uint8_t CC_id;
    int ue_inst = 0;
    protocol_ctxt_t ctxt;
    
    volatile uint8_t thread_idx = 0;
    sprintf(threadname,"UE_time_sync");
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    if ( threads.iq != -1 )
      CPU_SET(threads.iq, &cpuset);
    init_thread(900000,1000000 , FIFO_PRIORITY-1, &cpuset,
                threadname);

    for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
      PHY_vars_UE_g[0][CC_id]->UE_mode[0] = PRACH;
    }
    for (inst = 1; inst < NB_UE_INST; inst++) {
      for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
        PHY_vars_UE_g[inst][CC_id]->UE_mode[0] = NOT_SYNCHED;
      }
    }
    memset(&ue_tx_info[0][0], 0, sizeof(UE_TX_INFO)*NUMBER_OF_UE_MAX*RX_NB_TH);
    wait_system_ready ("Waiting for eNB start time %s\r", &start_time_flag);
    clock_gettime( CLOCK_REALTIME, &now_time );

    diff_sec = now_time.tv_sec - enb_start_time.tv_sec;
    diff_nsec = now_time.tv_nsec - enb_start_time.tv_nsec;
    n = (diff_sec * 1000000000 + diff_nsec)/TTI;
    
    subframe = n%10;
    frame = n/10;

    tti_time.tv_nsec = enb_start_time.tv_nsec + (n*TTI)%1000000000;
    tti_time.tv_sec  = enb_start_time.tv_sec + (n*TTI)/1000000000;

    while (!oai_exit) {
      if (ue_inst < NB_UE_INST-1 ) {
        next_ue_start = 0;
        for (CC_id = 0; CC_id < MAX_NUM_CCs; CC_id++) {
          if (PHY_vars_UE_g[ue_inst][CC_id]->UE_mode[0] != PUSCH) {
            next_ue_start = 1;
            break;
          }
        }

        if (next_ue_start == 0) {
            ue_inst++;
          for (int CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
            PHY_vars_UE_g[ue_inst][CC_id]->UE_mode[0] = PRACH;
          }
        }
      }

      tti_time.tv_nsec = tti_time.tv_nsec + TTI;

      if (tti_time.tv_nsec > 999999999) {
          tti_time.tv_nsec -= 1000000000;
          tti_time.tv_sec += 1;
      }
      clock_gettime( CLOCK_REALTIME, &now_time );

      diff_sec = tti_time.tv_sec - now_time.tv_sec;
      diff_nsec = tti_time.tv_nsec - now_time.tv_nsec;
      usleep_time = diff_sec * 1000000000 + diff_nsec;

      if ( usleep_time > 0 ) {
          req_time.tv_sec = usleep_time / 1000000000;
          req_time.tv_nsec = usleep_time % 1000000000;
          nanosleep(&req_time,&rem_time);
      }

      if (subframe==9) {
          subframe=0;
          frame++;
          frame&=1023;
      } else {
          subframe++;
      }

      if ((instance_cnt_send[thread_idx] < 0) || (oai_exit == 1)) {
        for (int CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
          PHY_vars_UE_g[0][CC_id]->proc.proc_rxtx[thread_idx].frame_rx = frame;
          PHY_vars_UE_g[0][CC_id]->proc.proc_rxtx[thread_idx].subframe_rx = subframe;
          PHY_vars_UE_g[0][CC_id]->proc.proc_rxtx[thread_idx].frame_tx = (frame + (subframe > 5 ? 1: 0))&1023;
          PHY_vars_UE_g[0][CC_id]->proc.proc_rxtx[thread_idx].subframe_tx = (subframe +4)%10;
        }
        //memset(&pdcp_end_flag[thread_idx][0],0,256);
        PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, 0, ENB_FLAG_NO, 0, 
                PHY_vars_UE_g[0][0]->proc.proc_rxtx[thread_idx].frame_tx,
                PHY_vars_UE_g[0][0]->proc.proc_rxtx[thread_idx].subframe_tx, 0);
        pdcp_run(&ctxt);

        if(pthread_mutex_lock(&mutex_rxtx[thread_idx]) != 0){
            LOG_E( MAC, "[UE] ERROR locking mutex for cond rxtx[%d] \n", thread_idx );
            exit_fun( "ERROR pthread_mutex_lock cond_rxtx" );
        }

        pthread_cond_broadcast(&cond_rxtx[thread_idx]);

        if(pthread_mutex_unlock(&mutex_rxtx[thread_idx]) != 0){
            LOG_E( MAC, "[UE] ERROR unlocking mutex_signal for cond rxtx[%d] \n", thread_idx );
            exit_fun( "ERROR pthread_mutex_unlock cond_rxtx" );
        }

        if (pthread_mutex_lock(&mutex_send[thread_idx])!= 0) {
          LOG_E( MAC, "[UE] error locking MAC proc mutex for mutex send\n");
          exit_fun("error locking mutex_send");
        }
        instance_cnt_send[thread_idx]++;
        pthread_cond_signal(&cond_send[thread_idx]);

        if (pthread_mutex_unlock(&mutex_send[thread_idx])!= 0) {
          LOG_E( MAC, "[UE] error unlocking MAC proc mutex for mutex send\n");
          exit_fun("error unlocking mutex_send");
        }
      }else{
            LOG_E(PHY,"frame %d subframe %d : thread_rxn_txnp4 busy (instance_cnt_send[%d] %d UE_num %d) \n",
        		frame,subframe,thread_idx,instance_cnt_send[thread_idx],NB_UE_INST);
      }
      thread_idx = (thread_idx+1)%RX_NB_TH;
    }
    return &UE_time_sync_retval;
}
#endif

/*!
 * \brief This is the main UE thread.
 * This thread controls the other three UE threads:
 * - UE_thread_rxn_txnp4 (even subframes)
 * - UE_thread_rxn_txnp4 (odd subframes)
 * - UE_thread_synch
 * \param arg unused
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */

void *UE_thread(void *arg) {


    PHY_VARS_UE *UE = (PHY_VARS_UE *) arg;
    //  int tx_enabled = 0;
    int dummy_rx[UE->frame_parms.nb_antennas_rx][UE->frame_parms.samples_per_tti] __attribute__((aligned(32)));
    openair0_timestamp timestamp,timestamp1;
    void* rxp[NB_ANTENNAS_RX], *txp[NB_ANTENNAS_TX];
    int start_rx_stream = 0;
    int i;
    int th_id;

    static uint8_t thread_idx = 0;

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if ( threads.iq != -1 )
        CPU_SET(threads.iq, &cpuset);
    init_thread(100000, 500000, FIFO_PRIORITY, &cpuset,
                "UHD Threads");

#ifdef NAS_UE
    MessageDef *message_p;
    message_p = itti_alloc_new_message(TASK_NAS_UE, INITIALIZE_MESSAGE);
    itti_send_msg_to_task (TASK_NAS_UE, UE->Mod_id + NB_eNB_INST, message_p);
#endif

#ifndef UE_EXPANSION_SIM2
    int sub_frame=-1;
    //int cumulated_shift=0;

    while (!oai_exit) {
        AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
        int instance_cnt_synch = UE->proc.instance_cnt_synch;
        int is_synchronized    = UE->is_synchronized;
        AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");

        if (is_synchronized == 0) {
            if (instance_cnt_synch < 0) {  // we can invoke the synch
                // grab 10 ms of signal and wakeup synch thread
                for (int i=0; i<UE->frame_parms.nb_antennas_rx; i++)
                    rxp[i] = (void*)&UE->common_vars.rxdata[i][0];

                if (UE->mode != loop_through_memory)
                    AssertFatal( UE->frame_parms.samples_per_tti*10 ==
                                 UE->rfdevice.trx_read_func(&UE->rfdevice,
                                                            &timestamp,
                                                            rxp,
                                                            UE->frame_parms.samples_per_tti*10,
                                                            UE->frame_parms.nb_antennas_rx), "");
		AssertFatal ( 0== pthread_mutex_lock(&UE->proc.mutex_synch), "");
                instance_cnt_synch = ++UE->proc.instance_cnt_synch;
                if (instance_cnt_synch == 0) {
                    AssertFatal( 0 == pthread_cond_signal(&UE->proc.cond_synch), "");
                } else {
                    LOG_E( PHY, "[SCHED][UE] UE sync thread busy!!\n" );
                    exit_fun("nothing to add");
                }
		AssertFatal ( 0== pthread_mutex_unlock(&UE->proc.mutex_synch), "");
            } else {
#if OAISIM
              (void)dummy_rx; /* avoid gcc warnings */
              usleep(500);
#else
                // grab 10 ms of signal into dummy buffer
                if (UE->mode != loop_through_memory) {
                    for (int i=0; i<UE->frame_parms.nb_antennas_rx; i++)
                        rxp[i] = (void*)&dummy_rx[i][0];
                    for (int sf=0; sf<10; sf++)
                        //	    printf("Reading dummy sf %d\n",sf);
                          UE->rfdevice.trx_read_func(&UE->rfdevice,
                                              &timestamp,
                                              rxp,
                                              UE->frame_parms.samples_per_tti,
                                              UE->frame_parms.nb_antennas_rx);
                }
#endif
            }

        } // UE->is_synchronized==0
        else {
            if (start_rx_stream==0) {
                start_rx_stream=1;
                if (UE->mode != loop_through_memory) {
                    if (UE->no_timing_correction==0) {
                        LOG_I(PHY,"Resynchronizing RX by %d samples (mode = %d)\n",UE->rx_offset,UE->mode);
                        AssertFatal(UE->rx_offset ==
                                    UE->rfdevice.trx_read_func(&UE->rfdevice,
                                                               &timestamp,
                                                               (void**)UE->common_vars.rxdata,
                                                               UE->rx_offset,
                                                               UE->frame_parms.nb_antennas_rx),"");
                    }
                    UE->rx_offset=0;
                    UE->time_sync_cell=0;
                    //UE->proc.proc_rxtx[0].frame_rx++;
                    //UE->proc.proc_rxtx[1].frame_rx++;
                    for (th_id=0; th_id < RX_NB_TH; th_id++) {
                        UE->proc.proc_rxtx[th_id].frame_rx++;
                    }

                    // read in first symbol
                    AssertFatal (UE->frame_parms.ofdm_symbol_size+UE->frame_parms.nb_prefix_samples0 ==
                                 UE->rfdevice.trx_read_func(&UE->rfdevice,
                                                            &timestamp,
                                                            (void**)UE->common_vars.rxdata,
                                                            UE->frame_parms.ofdm_symbol_size+UE->frame_parms.nb_prefix_samples0,
                                                            UE->frame_parms.nb_antennas_rx),"");
                    slot_fep(UE,0, 0, 0, 0, 0);
                } //UE->mode != loop_through_memory
                else
                    rt_sleep_ns(1000*1000);

            } else {
                sub_frame++;
                sub_frame%=10;
                UE_rxtx_proc_t *proc = &UE->proc.proc_rxtx[thread_idx];
                // update thread index for received subframe
                UE->current_thread_id[sub_frame] = thread_idx;

                LOG_D(PHY,"Process Subframe %d thread Idx %d \n", sub_frame, UE->current_thread_id[sub_frame]);

                thread_idx++;
                if(thread_idx>=RX_NB_TH)
                    thread_idx = 0;


                if (UE->mode != loop_through_memory) {
                    for (i=0; i<UE->frame_parms.nb_antennas_rx; i++)
                        rxp[i] = (void*)&UE->common_vars.rxdata[i][UE->frame_parms.ofdm_symbol_size+
                                 UE->frame_parms.nb_prefix_samples0+
                                 sub_frame*UE->frame_parms.samples_per_tti];
                    for (i=0; i<UE->frame_parms.nb_antennas_tx; i++)
                        txp[i] = (void*)&UE->common_vars.txdata[i][((sub_frame+2)%10)*UE->frame_parms.samples_per_tti];

                    int readBlockSize, writeBlockSize;
                    if (sub_frame<9) {
                        readBlockSize=UE->frame_parms.samples_per_tti;
                        writeBlockSize=UE->frame_parms.samples_per_tti;
                    } else {
                        // set TO compensation to zero
                        UE->rx_offset_diff = 0;
                        // compute TO compensation that should be applied for this frame
                        if ( UE->rx_offset < 5*UE->frame_parms.samples_per_tti  &&
                                UE->rx_offset > 0 )
                            UE->rx_offset_diff = -1 ;
                        if ( UE->rx_offset > 5*UE->frame_parms.samples_per_tti &&
                                UE->rx_offset < 10*UE->frame_parms.samples_per_tti )
                            UE->rx_offset_diff = 1;

                        LOG_D(PHY,"AbsSubframe %d.%d SET rx_off_diff to %d rx_offset %d \n",proc->frame_rx,sub_frame,UE->rx_offset_diff,UE->rx_offset);
                        readBlockSize=UE->frame_parms.samples_per_tti -
                                      UE->frame_parms.ofdm_symbol_size -
                                      UE->frame_parms.nb_prefix_samples0 -
                                      UE->rx_offset_diff;
                        writeBlockSize=UE->frame_parms.samples_per_tti -
                                       UE->rx_offset_diff;
                    }

                    AssertFatal(readBlockSize ==
                                UE->rfdevice.trx_read_func(&UE->rfdevice,
                                                           &timestamp,
                                                           rxp,
                                                           readBlockSize,
                                                           UE->frame_parms.nb_antennas_rx),"");
                    AssertFatal( writeBlockSize ==
                                 UE->rfdevice.trx_write_func(&UE->rfdevice,
                                         timestamp+
                                         (2*UE->frame_parms.samples_per_tti) -
                                         UE->frame_parms.ofdm_symbol_size-UE->frame_parms.nb_prefix_samples0 -
                                         openair0_cfg[0].tx_sample_advance,
                                         txp,
                                         writeBlockSize,
                                         UE->frame_parms.nb_antennas_tx,
                                         1),"");
                    if( sub_frame==9) {
                        // read in first symbol of next frame and adjust for timing drift
                        int first_symbols=writeBlockSize-readBlockSize;
                        if ( first_symbols > 0 )
                            AssertFatal(first_symbols ==
                                        UE->rfdevice.trx_read_func(&UE->rfdevice,
                                                                   &timestamp1,
                                                                   (void**)UE->common_vars.rxdata,
                                                                   first_symbols,
                                                                   UE->frame_parms.nb_antennas_rx),"");
                        if ( first_symbols <0 )
                            LOG_E(PHY,"can't compensate: diff =%d\n", first_symbols);
                    }
                    pickTime(gotIQs);
                    // operate on thread sf mod 2
                    AssertFatal(pthread_mutex_lock(&proc->mutex_rxtx) ==0,"");
                    if(sub_frame == 0) {
                        //UE->proc.proc_rxtx[0].frame_rx++;
                        //UE->proc.proc_rxtx[1].frame_rx++;
                        for (th_id=0; th_id < RX_NB_TH; th_id++) {
                            UE->proc.proc_rxtx[th_id].frame_rx++;
                        }
                    }
                    //UE->proc.proc_rxtx[0].gotIQs=readTime(gotIQs);
                    //UE->proc.proc_rxtx[1].gotIQs=readTime(gotIQs);
                    for (th_id=0; th_id < RX_NB_TH; th_id++) {
                        UE->proc.proc_rxtx[th_id].gotIQs=readTime(gotIQs);
                    }
                    proc->subframe_rx=sub_frame;
                    proc->subframe_tx=(sub_frame+4)%10;
                    proc->frame_tx = proc->frame_rx + (proc->subframe_rx>5?1:0);
                    proc->timestamp_tx = timestamp+
                                         (4*UE->frame_parms.samples_per_tti)-
                                         UE->frame_parms.ofdm_symbol_size-UE->frame_parms.nb_prefix_samples0;

                    proc->instance_cnt_rxtx++;
                    LOG_D( PHY, "[SCHED][UE %d] UE RX instance_cnt_rxtx %d subframe %d !!\n", UE->Mod_id, proc->instance_cnt_rxtx,proc->subframe_rx);
                    if (proc->instance_cnt_rxtx == 0) {
                      if (pthread_cond_signal(&proc->cond_rxtx) != 0) {
                        LOG_E( PHY, "[SCHED][UE %d] ERROR pthread_cond_signal for UE RX thread\n", UE->Mod_id);
                        exit_fun("nothing to add");
                      }
                    } else {
                      LOG_E( PHY, "[SCHED][UE %d] UE RX thread busy (IC %d)!!\n", UE->Mod_id, proc->instance_cnt_rxtx);
                      if (proc->instance_cnt_rxtx > 2)
                        exit_fun("instance_cnt_rxtx > 2");
                    }

                    AssertFatal (pthread_cond_signal(&proc->cond_rxtx) ==0 ,"");
                    AssertFatal(pthread_mutex_unlock(&proc->mutex_rxtx) ==0,"");
                    initRefTimes(t1);
                    initStaticTime(lastTime);
                    updateTimes(lastTime, &t1, 20000, "Delay between two IQ acquisitions (case 1)");
                    pickStaticTime(lastTime);

                } else {
                    printf("Processing subframe %d",proc->subframe_rx);
                    getchar();
                }
            } // start_rx_stream==1
        } // UE->is_synchronized==1

    } // while !oai_exit
#endif
    return NULL;
}

#ifdef UE_EXPANSION_SIM2
void ue_time_sync_info()
 {
  int header_len = sizeof(T_MSGHEAD);
  int data_len = 0;
  T_UDP_MSG data = {0};

  data_len = header_len;

  data.msgHead.msgid = STUB_ENB_TIME_SYNC;
  data.msgHead.msgLen = data_len;

  int retval = PacketWrite(ue_sd_c,
                           &data,
                           data_len,
                           udp_socket_ip_enb,
                           udp_socket_port_enb);
  if (retval == -1) {  // error
    LOG_E(PHY, "ue_time_sync_info notify from UE to eNB failed\n");
    return;
  }
  // success
  LOG_D(PHY, "ue_time_sync_info notify from UE to eNB successfully\n");
}

/**
 * param arg unused
 */
static void* UE_phy_rev( void* arg ) {
  static int UE_phy_stub_status;

  T_UDP_MSG buffer;
  eNB_TX_INFO *enb_tx_info_p;
  eNB_TX_PDU_INFO *enb_tx_pdu_p;
  int addr;
  int port;
  int ue_num;

  // set default return value
  UE_phy_stub_status = 0;

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
//  if ( threads.iq != -1 )
      CPU_SET(15, &cpuset);
  init_thread(100000, 500000, FIFO_PRIORITY-1, &cpuset,
              "UE_phy_rev");

  addr = udp_socket_ip_ue;
  port = udp_socket_port_ue;

  if (DataLinkSocket(addr, port, &ue_sd_s) == -1) {
    LOG_E(PHY, "socket descriptor(ue_sd_s) create error\n");
    printf("socket ue_sd_s create error\n");
    exit(-1);
  }
  LOG_D(PHY, "[UE STUB] receive socket descriptor[%d]\n", ue_sd_s);

  if (SendSocket(&ue_sd_c) == -1) {
    closeSocket_fd(ue_sd_s);
    LOG_E(PHY, "socket descriptor(ue_sd_c) create error\n");
    printf("socket ue_sd_c create error\n");
    exit(-1);
  }
  LOG_D(PHY, "[UE STUB] send socket descriptor[%d]\n", ue_sd_c);

  memset(&ue_rx_receive_info[0], 0, sizeof(UE_RX_RECEIVE_INFO)*10);

  ue_time_sync_info();

  while (!oai_exit) {

    if (oai_exit) break;

    if (PacketRead(ue_sd_s, &buffer, sizeof(buffer)) > 0) {
      switch (buffer.msgHead.msgid) {
      case STUB_UE_RX:
        enb_tx_info_p = (eNB_TX_INFO *)(buffer.data);
        memcpy(&ue_rx_receive_info[enb_tx_info_p->subframe], enb_tx_info_p, sizeof(eNB_TX_INFO));
        break;
      case STUB_UE_RX_PDU:
        enb_tx_pdu_p = (eNB_TX_PDU_INFO *)(buffer.data);
        ue_num = ue_rx_receive_info[enb_tx_pdu_p->subframe].ue_num;
        ue_rx_receive_info[enb_tx_pdu_p->subframe].ue_rx_info[ue_num].rnti = enb_tx_pdu_p->rnti;
        ue_rx_receive_info[enb_tx_pdu_p->subframe].ue_rx_info[ue_num].pdu_length = enb_tx_pdu_p->pdu_length;

        memcpy(&ue_rx_receive_info[enb_tx_pdu_p->subframe].ue_rx_info[ue_num].pdu_buffer,
            enb_tx_pdu_p->pdu_buffer,
            enb_tx_pdu_p->pdu_length);
        ue_rx_receive_info[enb_tx_pdu_p->subframe].ue_num++;
        break;
      case STUB_UE_TIME_SYNC:
        enb_start_time = *(struct timespec *)(buffer.data);
        start_time_flag = 1;
        break;
      default:
        LOG_E(PHY, "[UE STUB] receive unknown message(id:0x%X)\n", buffer.msgHead.msgid);
        break;
      }
    }
  }

  closeSocket_fd(ue_sd_s);
  closeSocket_fd(ue_sd_c);

  LOG_I(PHY, "Exiting UE PHY STUB\n");

  UE_phy_stub_status = 0;
  return &UE_phy_stub_status;
}

static void* UE_phy_send( void* param ) {
  static int UE_phy_send_status;

  int thread_idx = *((int *)param);
  int end_flag;
  int inst;
  UES_TX_INFO enb_rx_info;
  // set default return value
  UE_phy_send_status = 0;
  uint8_t eNB_index = 0;
  UE_rxtx_proc_t *proc_p = &PHY_vars_UE_g[0][0]->proc.proc_rxtx[thread_idx];

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
//  if ( threads.iq != -1 )
  CPU_SET(13+thread_idx, &cpuset);
  init_thread(100000, 500000, FIFO_PRIORITY-1, &cpuset,
              "UE_phy_send");

  while (!oai_exit) {

    if (pthread_mutex_lock(&mutex_send[thread_idx]) != 0){
      LOG_E( PHY, "[SCHED][UE] error locking mutex for phy_send\n" );
      exit_fun("nothing to add");
    }
    while (instance_cnt_send[thread_idx] < 0) {
      pthread_cond_wait(&cond_send[thread_idx], &mutex_send[thread_idx]);
    }

    if (pthread_mutex_unlock(&mutex_send[thread_idx]) != 0){
      LOG_E( PHY, "[SCHED][UE] error unlocking mutex for phy_send\n" );
      exit_fun("nothing to add");
    }

    memset(&enb_rx_info, 0, sizeof(enb_rx_info));
    enb_rx_info.Mod_id = eNB_index;
    enb_rx_info.CC_id = PHY_vars_UE_g[0][0]->CC_id;
    enb_rx_info.frame = proc_p->frame_tx;
    enb_rx_info.subframe = proc_p->subframe_tx;

    do {
      end_flag = 1;

      for (inst = 0; inst < NB_UE_INST; inst++) {
        if (ue_tx_info[thread_idx][inst].flag == 1) {
          ue_tx_info[thread_idx][inst].flag = 2;

          if (ue_tx_info[thread_idx][inst].sr_flag == 1) {
            enb_rx_info.sr_rnti_list[enb_rx_info.sr_num] = ue_tx_info[thread_idx][inst].rnti;
            enb_rx_info.sr_num++;
          }

          if (ue_tx_info[thread_idx][inst].pusch_type == MSG1_PUSCH) {
            enb_rx_info.pusch_type = ue_tx_info[thread_idx][inst].pusch_type;
            enb_rx_info.preamble = ue_tx_info[thread_idx][inst].preamble;
          }
        }

        if (ue_tx_info[thread_idx][inst].flag == 0) {
          end_flag = 0;
        }
      }
    } while (end_flag == 0);

    if (oai_exit) break;

    if (enb_rx_info.sr_num > 0 || enb_rx_info.pusch_type == MSG1_PUSCH) {
      ue_tx_send_info(&enb_rx_info);
    }

    memset(&ue_tx_info[thread_idx][0], 0, sizeof(UE_TX_INFO)*NUMBER_OF_UE_MAX);
    memset(&ue_rx_receive_info[proc_p->subframe_rx],0,sizeof(UE_RX_RECEIVE_INFO));

    pthread_mutex_lock(&mutex_send[thread_idx]);
    instance_cnt_send[thread_idx]--;
    pthread_mutex_unlock(&mutex_send[thread_idx]);
  }

  LOG_I(PHY, "Exiting UE PHY SEND\n");

  UE_phy_send_status = 0;
  return &UE_phy_send_status;
}
#endif

/*!
 * \brief Initialize the UE theads.
 * Creates the UE threads:
 * - UE_thread_rxtx0
 * - UE_thread_rxtx1
 * - UE_thread_synch
 * - UE_thread_fep_slot0
 * - UE_thread_fep_slot1
 * - UE_thread_dlsch_proc_slot0
 * - UE_thread_dlsch_proc_slot1
 * and the locking between them.
 */
void init_UE_threads(int inst) {
    struct rx_tx_thread_data *rtd;
    PHY_VARS_UE *UE;

    AssertFatal(PHY_vars_UE_g!=NULL,"PHY_vars_UE_g is NULL\n");
    AssertFatal(PHY_vars_UE_g[inst]!=NULL,"PHY_vars_UE_g[inst] is NULL\n");
    AssertFatal(PHY_vars_UE_g[inst][0]!=NULL,"PHY_vars_UE_g[inst][0] is NULL\n");
    UE = PHY_vars_UE_g[inst][0];

    pthread_attr_init (&UE->proc.attr_ue);
    pthread_attr_setstacksize(&UE->proc.attr_ue,8192);//5*PTHREAD_STACK_MIN);

    pthread_mutex_init(&UE->proc.mutex_synch,NULL);
    pthread_cond_init(&UE->proc.cond_synch,NULL);

    // the threads are not yet active, therefore access is allowed without locking
    int nb_threads=RX_NB_TH;
    for (int i=0; i<nb_threads; i++) {
        rtd = calloc(1, sizeof(struct rx_tx_thread_data));
        if (rtd == NULL) abort();
        rtd->UE = UE;
        rtd->proc = &UE->proc.proc_rxtx[i];

        pthread_mutex_init(&UE->proc.proc_rxtx[i].mutex_rxtx,NULL);
        pthread_cond_init(&UE->proc.proc_rxtx[i].cond_rxtx,NULL);
        UE->proc.proc_rxtx[i].sub_frame_start=i;
        UE->proc.proc_rxtx[i].sub_frame_step=nb_threads;
        printf("Init_UE_threads rtd %d proc %d nb_threads %d i %d\n",rtd->proc->sub_frame_start, UE->proc.proc_rxtx[i].sub_frame_start,nb_threads, i);
        pthread_create(&UE->proc.proc_rxtx[i].pthread_rxtx, NULL, UE_thread_rxn_txnp4, rtd);

#ifdef UE_SLOT_PARALLELISATION
        //pthread_mutex_init(&UE->proc.proc_rxtx[i].mutex_slot0_dl_processing,NULL);
        //pthread_cond_init(&UE->proc.proc_rxtx[i].cond_slot0_dl_processing,NULL);
        //pthread_create(&UE->proc.proc_rxtx[i].pthread_slot0_dl_processing,NULL,UE_thread_slot0_dl_processing, rtd);

        pthread_mutex_init(&UE->proc.proc_rxtx[i].mutex_slot1_dl_processing,NULL);
        pthread_cond_init(&UE->proc.proc_rxtx[i].cond_slot1_dl_processing,NULL);
        pthread_create(&UE->proc.proc_rxtx[i].pthread_slot1_dl_processing,NULL,UE_thread_slot1_dl_processing, rtd);
#endif

    }
#ifndef UE_EXPANSION_SIM2
    pthread_create(&UE->proc.pthread_synch,NULL,UE_thread_synch,(void*)UE);
#endif
}


#ifdef OPENAIR2
void fill_ue_band_info(void) {

    UE_EUTRA_Capability_t *UE_EUTRA_Capability = UE_rrc_inst[0].UECap->UE_EUTRA_Capability;
    int i,j;

    bands_to_scan.nbands = UE_EUTRA_Capability->rf_Parameters.supportedBandListEUTRA.list.count;

    for (i=0; i<bands_to_scan.nbands; i++) {

        for (j=0; j<sizeof (eutra_bands) / sizeof (eutra_bands[0]); j++)
            if (eutra_bands[j].band == UE_EUTRA_Capability->rf_Parameters.supportedBandListEUTRA.list.array[i]->bandEUTRA) {
                memcpy(&bands_to_scan.band_info[i],
                       &eutra_bands[j],
                       sizeof(eutra_band_t));

                printf("Band %d (%lu) : DL %u..%u Hz, UL %u..%u Hz, Duplex %s \n",
                       bands_to_scan.band_info[i].band,
                       UE_EUTRA_Capability->rf_Parameters.supportedBandListEUTRA.list.array[i]->bandEUTRA,
                       bands_to_scan.band_info[i].dl_min,
                       bands_to_scan.band_info[i].dl_max,
                       bands_to_scan.band_info[i].ul_min,
                       bands_to_scan.band_info[i].ul_max,
                       (bands_to_scan.band_info[i].frame_type==FDD) ? "FDD" : "TDD");
                break;
            }
    }
}
#endif

int setup_ue_buffers(PHY_VARS_UE **phy_vars_ue, openair0_config_t *openair0_cfg) {

    int i, CC_id;
    LTE_DL_FRAME_PARMS *frame_parms;
    openair0_rf_map *rf_map;

    for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
      rf_map = &phy_vars_ue[CC_id]->rf_map;
      
      AssertFatal( phy_vars_ue[CC_id] !=0, "");
      frame_parms = &(phy_vars_ue[CC_id]->frame_parms);
      
      // replace RX signal buffers with mmaped HW versions
      rxdata = (int32_t**)malloc16( frame_parms->nb_antennas_rx*sizeof(int32_t*) );
      txdata = (int32_t**)malloc16( frame_parms->nb_antennas_tx*sizeof(int32_t*) );
      
      for (i=0; i<frame_parms->nb_antennas_rx; i++) {
	LOG_I(PHY, "Mapping UE CC_id %d, rx_ant %d, freq %u on card %d, chain %d\n",
	      CC_id, i, downlink_frequency[CC_id][i], rf_map->card, rf_map->chain+i );
	free( phy_vars_ue[CC_id]->common_vars.rxdata[i] );
	rxdata[i] = (int32_t*)malloc16_clear( 307200*sizeof(int32_t) );
	phy_vars_ue[CC_id]->common_vars.rxdata[i] = rxdata[i]; // what about the "-N_TA_offset" ? // N_TA offset for TDD
      }
		
      for (i=0; i<frame_parms->nb_antennas_tx; i++) {
	LOG_I(PHY, "Mapping UE CC_id %d, tx_ant %d, freq %u on card %d, chain %d\n",
	      CC_id, i, downlink_frequency[CC_id][i], rf_map->card, rf_map->chain+i );
	free( phy_vars_ue[CC_id]->common_vars.txdata[i] );
	txdata[i] = (int32_t*)malloc16_clear( 307200*sizeof(int32_t) );
	phy_vars_ue[CC_id]->common_vars.txdata[i] = txdata[i];
      }
      
      // rxdata[x] points now to the same memory region as phy_vars_ue[CC_id]->common_vars.rxdata[x]
      // txdata[x] points now to the same memory region as phy_vars_ue[CC_id]->common_vars.txdata[x]
      // be careful when releasing memory!
      // because no "release_ue_buffers"-function is available, at least rxdata and txdata memory will leak (only some bytes)
    }
    return 0;
}

