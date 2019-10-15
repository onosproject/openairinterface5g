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

/*! \file lte-enb.c
 * \brief Top-level threads for eNodeB
 * \author R. Knopp, F. Kaltenberger, Navid Nikaein
 * \date 2012
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr, navid.nikaein@eurecom.fr
 * \note
 * \warning
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sched.h>
#include <linux/sched.h>
#include <signal.h>
#include <execinfo.h>
#include <getopt.h>
#include <sys/sysinfo.h>
#include "rt_wrapper.h"

//#include "time_utils.h"

#undef MALLOC //there are two conflicting definitions, so we better make sure we don't use it at all

#include "assertions.h"
#include "msc.h"

#include "PHY/types.h"

#undef MALLOC //there are two conflicting definitions, so we better make sure we don't use it at all
//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all

#include "../../ARCH/COMMON/common_lib.h"

//#undef FRAME_LENGTH_COMPLEX_SAMPLES //there are two conflicting definitions, so we better make sure we don't use it at all

#include "PHY/LTE_TRANSPORT/if4_tools.h"
#include "PHY/LTE_TRANSPORT/if5_tools.h"

#include "PHY/phy_extern.h"
//#include "SCHED/extern.h"
#include "SCHED/sched_eNB.h"

#include "../../SIMU/USER/init_lte.h"

//NB-IoT 
#include "PHY/defs_eNB.h"
#include "PHY/defs_L1_NB_IoT.h"
//#include "PHY/defs_L1_NB_IoT.h"
#include "SCHED_NBIOT/defs_NB_IoT.h"
#include "SCHED/sched_common.h" // for calling prach_procedures_NB_IoT()
#include "PHY_INTERFACE/IF_Module_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "PHY/extern_NB_IoT.h"
#include "LAYER2/MAC/defs.h"
#include "PHY_INTERFACE/phy_interface_extern.h"
//#include "PHY/LTE_TRANSPORT/transport_proto.h"
#include "PHY/NBIoT_TRANSPORT/proto_NB_IoT.h"

#ifdef SMBV
#include "PHY/TOOLS/smbv.h"
unsigned short config_frames[4] = {2,9,11,13};
#endif
#include "UTIL/OTG/otg_tx.h"
#include "UTIL/OTG/otg_externs.h"
#include "UTIL/MATH/oml.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "enb_config.h"
//#include "PHY/TOOLS/time_meas.h"

#ifndef OPENAIR2
#include "UTIL/OTG/otg_extern.h"
#endif

#if defined(ENABLE_ITTI)
# if defined(ENABLE_USE_MME)
#   include "s1ap_eNB.h"
#ifdef PDCP_USE_NETLINK
#   include "SIMULATION/ETH_TRANSPORT/proto.h"
#endif
# endif
#endif

#include "T.h"

extern volatile int             start_eNB;
extern volatile int                    oai_exit;
extern int oaisim_flag;
extern openair0_config_t openair0_cfg[MAX_CARDS];
extern uint16_t sf_ahead;

uint8_t seqno; //sequence number
static int                      time_offset[4] = {0,0,0,0};

static int recv_if_count = 0;

static struct {
  pthread_mutex_t  mutex_phy_proc_tx;
  pthread_cond_t   cond_phy_proc_tx;
  volatile uint8_t phy_proc_CC_id;
} sync_phy_proc;

struct timespec start_rf_new, start_rf_prev, start_rf_prev2, end_rf;
openair0_timestamp start_rf_new_ts, start_rf_prev_ts, start_rf_prev2_ts, end_rf_ts;

extern struct timespec start_fh, start_fh_prev;
extern int start_fh_sf, start_fh_prev_sf;
struct timespec end_fh;
int end_fh_sf;

void init_eNB_NB_IoT(eNB_func_NB_IoT_t node_function[], eNB_timing_NB_IoT_t node_timing[],int nb_inst,eth_params_t *,int,int);

/**********************************************************Other structure***************************************************************/




void proc_tx_high0_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
       eNB_rxtx_proc_NB_IoT_t *proc,
       relaying_type_t r_type,
       PHY_VARS_RN_NB_IoT *rn) {

  int offset = proc == &eNB->proc.proc_rxtx[0] ? 0 : 1;

  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME( VCD_SIGNAL_DUMPER_VARIABLES_FRAME_NUMBER_TX0_ENB+offset, proc->frame_tx );
  VCD_SIGNAL_DUMPER_DUMP_VARIABLE_BY_NAME( VCD_SIGNAL_DUMPER_VARIABLES_SUBFRAME_NUMBER_TX0_ENB+offset, proc->subframe_tx );

  // issue here
  phy_procedures_eNB_TX_NB_IoT(eNB,proc,r_type,rn,1,1);

  /* we're done, let the next one proceed */
  if (pthread_mutex_lock(&sync_phy_proc.mutex_phy_proc_tx) != 0) {
    LOG_E(PHY, "[SCHED][eNB] error locking PHY proc mutex for eNB TX proc\n");
    exit_fun("nothing to add");
  } 
  sync_phy_proc.phy_proc_CC_id++;
  sync_phy_proc.phy_proc_CC_id %= MAX_NUM_CCs;
  pthread_cond_broadcast(&sync_phy_proc.cond_phy_proc_tx);
  if (pthread_mutex_unlock(&sync_phy_proc.mutex_phy_proc_tx) != 0) {
    LOG_E(PHY, "[SCHED][eNB] error unlocking PHY proc mutex for eNB TX proc\n");
    exit_fun("nothing to add");
  }

}

void proc_tx_high_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
      eNB_rxtx_proc_NB_IoT_t *proc,
      relaying_type_t r_type,
      PHY_VARS_RN_NB_IoT *rn) {


  // do PHY high
  proc_tx_high0_NB_IoT(eNB,proc,r_type,rn);

  // if TX fronthaul go ahead 
  if (eNB->tx_fh) eNB->tx_fh(eNB,proc);

}


void proc_tx_full_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,
      eNB_rxtx_proc_NB_IoT_t *proc,
      relaying_type_t r_type,
      PHY_VARS_RN_NB_IoT *rn) {


  // do PHY high
  proc_tx_high0_NB_IoT(eNB,proc,r_type,rn);

}

#if defined(ENABLE_ITTI) && defined(ENABLE_USE_MME)
/* Wait for eNB application initialization to be complete (eNB registration to MME) */
static void wait_system_ready (char *message, volatile int *start_flag) {
  
  static char *indicator[] = {".    ", "..   ", "...  ", ".... ", ".....",
            " ....", "  ...", "   ..", "    .", "     "};
  int i = 0;
  
  while ((!oai_exit) && (*start_flag == 0)) {
    //LOG_N(EMU, message, indicator[i]);
    fflush(stdout);
    i = (i + 1) % (sizeof(indicator) / sizeof(indicator[0]));
    usleep(200000);
  }
  
    //LOG_D(EMU,"\n");
}
#endif


static inline int rxtx_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,eNB_rxtx_proc_NB_IoT_t *proc, char *thread_name) {

  ///start_meas(&softmodem_stats_rxtx_sf);

  // ****************************************
  // Common RX procedures subframe n

  
   if ((eNB->do_prach)&&((eNB->node_function != NGFI_RCC_IF4p5_NB_IoT)))
    eNB->do_prach(eNB,proc->frame_rx,proc->subframe_rx);

  
  // UE-specific RX processing for subframe n
  ///////////////////////////////////// for NB-IoT testing  ////////////////////////
  // for NB-IoT testing  // activating only TX part
  if (eNB->proc_uespec_rx) eNB->proc_uespec_rx(eNB, proc, no_relay );
   ////////////////////////////////////END///////////////////////

  //npusch_procedures(eNB,proc,data_or_control);
  //fill_rx_indication(eNB,i,frame,subframe);
  //////////////////////////////////// for IF Module/scheduler testing
 
  pthread_mutex_lock(&eNB->UL_INFO_mutex);

  eNB->UL_INFO.frame     = proc->frame_rx;
  eNB->UL_INFO.subframe  = proc->subframe_rx;
  eNB->UL_INFO.module_id = eNB->Mod_id;
  eNB->UL_INFO.CC_id     = eNB->CC_id;
  eNB->UL_INFO.hypersfn  = proc->HFN;

  eNB->if_inst_NB_IoT->UL_indication(&eNB->UL_INFO);

  pthread_mutex_unlock(&eNB->UL_INFO_mutex);

  //LOG_I(PHY,"After UL_indication\n");
  // *****************************************
  // TX processing for subframe n+4
  // run PHY TX procedures the one after the other for all CCs to avoid race conditions
  // (may be relaxed in the future for performance reasons)
  // *****************************************
  //if (wait_CCs(proc)<0) return(-1);
  
  if (oai_exit) return(-1);
  
  if (eNB->proc_tx) eNB->proc_tx(eNB, proc, no_relay, NULL );
  
  if (release_thread(&proc->mutex_rxtx,&proc->instance_cnt_rxtx,thread_name)<0) return(-1);

 /// stop_meas( &softmodem_stats_rxtx_sf );
  
  return(0);
}

/*
static void* eNB_thread_single_NB_IoT( void* param ) {

  static int eNB_thread_single_status;

  eNB_proc_NB_IoT_t             *proc = (eNB_proc_NB_IoT_t*)param;
  eNB_rxtx_proc_NB_IoT_t        *proc_rxtx = &proc->proc_rxtx[0];
  PHY_VARS_eNB_NB_IoT *eNB = PHY_vars_eNB_NB_IoT_g[0][proc->CC_id];
  //PHY_VARS_eNB_NB_IoT *eNB_NB_IoT = PHY_vars_eNB_NB_IoT_g[0][proc->CC_id];
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;
 // NB_IoT_DL_FRAME_PARMS *fp_NB_IoT = &eNB_NB_IoT->frame_parms_NB_IoT;
  eNB->CC_id =  proc->CC_id;

  void *rxp[2],*rxp2[2];

  int subframe=0, frame=0; 

  int32_t dummy_rx[fp->nb_antennas_rx][fp->samples_per_tti] __attribute__((aligned(32)));

  int ic;

  int rxs;

  int i;

  // initialize the synchronization buffer to the common_vars.rxdata
  for (int i=0;i<fp->nb_antennas_rx;i++)
    rxp[i] = &eNB->common_vars.rxdata[0][i][0];

  // set default return value
  eNB_thread_single_status = 0;

  thread_top_init("eNB_thread_single",0,870000,1000000,1000000);

  wait_sync("eNB_thread_single");

#if defined(ENABLE_ITTI) && defined(ENABLE_USE_MME)
  if ((eNB->node_function < NGFI_RRU_IF5_NB_IoT) && (eNB->mac_enabled==1))
    wait_system_ready ("Waiting for eNB application to be ready %s\r", &start_eNB);
#endif 

  // Start IF device if any
  if (eNB->start_if) 
    if (eNB->start_if(eNB) != 0)
      LOG_E(HW,"Could not start the IF device\n");

  // Start RF device if any
  if (eNB->start_rf)
    if (eNB->start_rf(eNB) != 0)
      LOG_E(HW,"Could not start the RF device\n");

  // wakeup asnych_rxtx thread because the devices are ready at this point
  pthread_mutex_lock(&proc->mutex_asynch_rxtx);
  proc->instance_cnt_asynch_rxtx=0;
  pthread_mutex_unlock(&proc->mutex_asynch_rxtx);
  pthread_cond_signal(&proc->cond_asynch_rxtx);



  // if this is a slave eNB, try to synchronize on the DL frequency
  if ((eNB->is_slave) &&
      ((eNB->node_function >= NGFI_RRU_IF5))) {
    // if FDD, switch RX on DL frequency
    
    double temp_freq1 = eNB->rfdevice.openair0_cfg->rx_freq[0];
    double temp_freq2 = eNB->rfdevice.openair0_cfg->tx_freq[0];
    for (i=0;i<4;i++) {
      eNB->rfdevice.openair0_cfg->rx_freq[i] = eNB->rfdevice.openair0_cfg->tx_freq[i];
      eNB->rfdevice.openair0_cfg->tx_freq[i] = temp_freq1;
    }
    eNB->rfdevice.trx_set_freq_func(&eNB->rfdevice,eNB->rfdevice.openair0_cfg,0);

    while ((eNB->in_synch ==0)&&(!oai_exit)) {
      // read in frame
      rxs = eNB->rfdevice.trx_read_func(&eNB->rfdevice,
          &(proc->timestamp_rx),
          rxp,
          fp->samples_per_tti*10,
          fp->nb_antennas_rx);

      if (rxs != (fp->samples_per_tti*10))
  exit_fun("Problem receiving samples\n");

      // wakeup synchronization processing thread
      wakeup_synch(eNB);
      ic=0;
      
      while ((ic>=0)&&(!oai_exit)) {
  // continuously read in frames, 1ms at a time, 
  // until we are done with the synchronization procedure
  
  for (i=0; i<fp->nb_antennas_rx; i++)
    rxp2[i] = (void*)&dummy_rx[i][0];
  for (i=0;i<10;i++)
    rxs = eNB->rfdevice.trx_read_func(&eNB->rfdevice,
              &(proc->timestamp_rx),
              rxp2,
              fp->samples_per_tti,
              fp->nb_antennas_rx);
  if (rxs != fp->samples_per_tti)
    exit_fun( "problem receiving samples" );

  pthread_mutex_lock(&eNB->proc.mutex_synch);
  ic = eNB->proc.instance_cnt_synch;
  pthread_mutex_unlock(&eNB->proc.mutex_synch);
      } // ic>=0
    } // in_synch==0
    // read in rx_offset samples
    LOG_I(PHY,"Resynchronizing by %d samples\n",eNB->rx_offset);
    rxs = eNB->rfdevice.trx_read_func(&eNB->rfdevice,
              &(proc->timestamp_rx),
              rxp,
              eNB->rx_offset,
              fp->nb_antennas_rx);
    if (rxs != eNB->rx_offset)
      exit_fun( "problem receiving samples" );

    for (i=0;i<4;i++) {
      eNB->rfdevice.openair0_cfg->rx_freq[i] = temp_freq1;
      eNB->rfdevice.openair0_cfg->tx_freq[i] = temp_freq2;
    }
    eNB->rfdevice.trx_set_freq_func(&eNB->rfdevice,eNB->rfdevice.openair0_cfg,1);
  } // if RRU and slave


  // This is a forever while loop, it loops over subframes which are scheduled by incoming samples from HW devices
  while (!oai_exit) {

    // these are local subframe/frame counters to check that we are in synch with the fronthaul timing.
    // They are set on the first rx/tx in the underly FH routines.
    if (subframe==9) { 
      subframe=0;
      frame++;
      frame&=1023;
    } else {
      subframe++;
    }      

    if (eNB->CC_id==1) 
  LOG_D(PHY,"eNB thread single (proc %p, CC_id %d), frame %d (%p), subframe %d (%p)\n",
    proc, eNB->CC_id, frame,&frame,subframe,&subframe);
 
    // synchronization on FH interface, acquire signals/data and block
    if (eNB->rx_fh) eNB->rx_fh(eNB,&frame,&subframe);
    else AssertFatal(1==0, "No fronthaul interface : eNB->node_function %d",eNB->node_function);

    T(T_ENB_MASTER_TICK, T_INT(0), T_INT(proc->frame_rx), T_INT(proc->subframe_rx));

    proc_rxtx->subframe_rx = proc->subframe_rx;
    proc_rxtx->frame_rx    = proc->frame_rx;
    proc_rxtx->subframe_tx = (proc->subframe_rx+4)%10;
    proc_rxtx->frame_tx    = (proc->subframe_rx>5) ? (1+proc->frame_rx)&1023 : proc->frame_rx;
    proc->frame_tx         = proc_rxtx->frame_tx;
    proc_rxtx->timestamp_tx = proc->timestamp_tx;
    // adjust for timing offset between RRU
    if (eNB->CC_id!=0) proc_rxtx->frame_tx = (proc_rxtx->frame_tx+proc->frame_offset)&1023;

    // At this point, all information for subframe has been received on FH interface
    // If this proc is to provide synchronization, do so
    wakeup_slaves(proc);

   
    if (rxtx_NB_IoT(eNB,proc_rxtx,"eNB_thread_single") < 0) break;
    //if (rxtx(eNB,proc_rxtx,"eNB_thread_single") < 0) break;
  }
  

  printf( "Exiting eNB_single thread \n");
 
  eNB_thread_single_status = 0;
  return &eNB_thread_single_status;

}
*/

/*!
 * \brief The RX UE-specific and TX thread of eNB.
 * \param param is a \ref eNB_proc_t structure which contains the info what to process.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */
/*
static void* eNB_thread_rxtx_NB_IoT( void* param ) {

  static int eNB_thread_rxtx_status;

  eNB_rxtx_proc_NB_IoT_t *proc = (eNB_rxtx_proc_NB_IoT_t*)param;
  ///eNB_rxtx_proc_NB_IoT_t *proc_NB_IoT = (eNB_rxtx_proc_NB_IoT_t*)param;  // to remove when eNB_thread_rxtx_status is duplicated for NB-IoT

  PHY_VARS_eNB_NB_IoT *eNB = PHY_vars_eNB_NB_IoT_g[0][proc->CC_id];

  ///PHY_VARS_eNB_NB_IoT *eNB_NB_IoT = PHY_vars_eNB_NB_IoT_g[0][proc_NB_IoT->CC_id]; // to remove when eNB_thread_rxtx_status is duplicated for NB-IoT

  char thread_name[100];


  // set default return value
  eNB_thread_rxtx_status = 0;


  sprintf(thread_name,"RXn_TXnp4_%d\n",&eNB->proc.proc_rxtx[0] == proc ? 0 : 1);
  thread_top_init(thread_name,1,850000L,1000000L,2000000L);

  while (!oai_exit) {
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_eNB_PROC_RXTX0+(proc->subframe_rx&1), 0 );

    if (wait_on_condition(&proc->mutex_rxtx,&proc->cond_rxtx,&proc->instance_cnt_rxtx,thread_name)<0) break;

    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_eNB_PROC_RXTX0+(proc->subframe_rx&1), 1 );

    
  
    if (oai_exit) break;

    if (eNB->CC_id==0)
    {
      if (rxtx_NB_IoT(eNB,proc,thread_name) < 0) break;
    }

  } // while !oai_exit

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_eNB_PROC_RXTX0+(proc->subframe_rx&1), 0 );

  printf( "Exiting eNB thread RXn_TXnp4\n");

  eNB_thread_rxtx_status = 0;
  return &eNB_thread_rxtx_status;
}

*/

void eNB_nb_iot_top(PHY_VARS_eNB_NB_IoT *eNB, int frame_rx, int subframe_rx, char *string,RU_t *ru) {
  eNB_proc_NB_IoT_t *proc           = &eNB->proc;
  eNB_rxtx_proc_NB_IoT_t *L1_proc = &proc->proc_rxtx[0];
  LTE_DL_FRAME_PARMS *fp = &ru->frame_parms;
  RU_proc_t *ru_proc_nbiot = &ru->proc;
  proc->frame_rx    = frame_rx;
  proc->subframe_rx = subframe_rx;

  if (!oai_exit) {
    T(T_ENB_MASTER_TICK, T_INT(0), T_INT(ru_proc_nbiot->frame_rx), T_INT(ru_proc_nbiot->subframe_rx));
    L1_proc->timestamp_tx = ru_proc_nbiot->timestamp_rx + (sf_ahead*fp->samples_per_tti);
    L1_proc->frame_rx     = ru_proc_nbiot->frame_rx;
    L1_proc->subframe_rx  = ru_proc_nbiot->subframe_rx;
    L1_proc->frame_tx     = (L1_proc->subframe_rx > (9-sf_ahead)) ? (L1_proc->frame_rx+1)&1023 : L1_proc->frame_rx;
    L1_proc->subframe_tx  = (L1_proc->subframe_rx + sf_ahead)%10;

    if (rxtx_NB_IoT(eNB,L1_proc,string) < 0) LOG_E(PHY,"eNB %d CC_id %d failed during execution\n",eNB->Mod_id,eNB->CC_id);

    ru_proc_nbiot->timestamp_tx = L1_proc->timestamp_tx;
    ru_proc_nbiot->subframe_tx  = L1_proc->subframe_tx;
    ru_proc_nbiot->frame_tx     = L1_proc->frame_tx;
  }
}


extern void do_prach_NB_IoT(PHY_VARS_eNB_NB_IoT *eNB,int frame,int subframe);


///Modify to NB-IoT merge
void init_eNB_NB_IoT(eNB_func_NB_IoT_t node_function[], eNB_timing_NB_IoT_t node_timing[],int nb_inst,eth_params_t *eth_params,int single_thread_flag,int wait_for_sync) {
  
  int CC_id;
  int inst;
  PHY_VARS_eNB_NB_IoT *eNB;
  int ret;

  for (inst=0;inst<nb_inst;inst++) {
    for (CC_id=0;CC_id<MAX_NUM_CCs;CC_id++) {
      eNB = PHY_vars_eNB_NB_IoT_g[inst][CC_id]; 
      eNB->node_function      = node_function[CC_id];
      eNB->node_timing        = node_timing[CC_id];
      eNB->eth_params         = eth_params+CC_id;
      eNB->abstraction_flag   = 0;
      eNB->single_thread_flag = single_thread_flag;
      eNB->ts_offset          = 0;
      eNB->in_synch           = 0;
      eNB->is_slave           = (wait_for_sync>0) ? 1 : 0;

      /////// IF-Module initialization ///////////////

      LOG_I(PHY,"Registering with MAC interface module start\n");
      AssertFatal((eNB->if_inst_NB_IoT         = IF_Module_init_NB_IoT(inst))!=NULL,"Cannot register interface");
      eNB->if_inst_NB_IoT->schedule_response   = schedule_response_NB_IoT;
      eNB->if_inst_NB_IoT->PHY_config_req      = PHY_config_req_NB_IoT;
      LOG_I(PHY,"Registering with MAC interface module sucessfully\n");




      if (setup_eNB_buffers(PHY_vars_eNB_NB_IoT_g[inst],&openair0_cfg[CC_id])!=0) {
  	printf("Exiting, cannot initialize eNodeB Buffers\n");
 	exit(-1);
      }
    }

    //init_eNB_proc_NB_IoT(inst);
  }

  sleep(1);
  LOG_D(HW,"[lte-softmodem.c] eNB threads created\n");
  

}






