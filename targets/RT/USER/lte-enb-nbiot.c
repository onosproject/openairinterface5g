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

#ifdef SMBV
#include "PHY/TOOLS/smbv.h"
unsigned short config_frames[4] = {2,9,11,13};
#endif
#include "UTIL/OTG/otg_tx.h"
#include "UTIL/OTG/otg_externs.h"
#include "UTIL/MATH/oml.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
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

void init_eNB_NB_IoT(eNB_func_t node_function[], eNB_timing_t node_timing[],int nb_inst,eth_params_t *,int,int);
extern void do_prach(PHY_VARS_eNB *eNB,int frame,int subframe);



//Modify for NB-IoT merge
static inline int rxtx_NB_IoT(PHY_VARS_eNB *eNB,L1_rxtx_proc_t *proc, char *thread_name) {
 ///start_meas(&softmodem_stats_rxtx_sf);

  // ****************************************
  // Common RX procedures subframe n

  
   if ((eNB->do_prach)&&((eNB->node_function != NGFI_RCC_IF4p5)))
    eNB->do_prach(eNB,proc->frame_rx,proc->subframe_rx);
  phy_procedures_eNB_common_RX(eNB,proc);
  
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

  eNB->if_inst->UL_indication(&eNB->UL_INFO);

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



/*!
 * \brief The prach receive thread of eNB.
 * \param param is a \ref eNB_proc_t structure which contains the info what to process.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */
static void* eNB_thread_prach_NB_IoT( void* param ) {
  static int eNB_thread_prach_status;

  PHY_VARS_eNB *eNB= (PHY_VARS_eNB *)param;
  L1_proc_t *proc = &eNB->proc;

  // set default return value
  eNB_thread_prach_status = 0;

  thread_top_init("eNB_thread_prach",1,500000L,1000000L,20000000L);

  while (!oai_exit) {
    
    if (oai_exit) break;

    if (wait_on_condition(&proc->mutex_prach,&proc->cond_prach,&proc->instance_cnt_prach,"eNB_prach_thread") < 0) break;
    
    //prach_procedures(eNB);
    
    ////// NB_IoT testing ///////
    prach_procedures_NB_IoT(eNB);
    /////////////////////////////
    
    if (release_thread(&proc->mutex_prach,&proc->instance_cnt_prach,"eNB_prach_thread") < 0) break;
  }

  printf( "Exiting eNB thread PRACH\n");

  eNB_thread_prach_status = 0;
  return &eNB_thread_prach_status;
}



///Modify to NB-IoT merge
void init_eNB_NB_IoT(eNB_func_t node_function[], eNB_timing_t node_timing[],int nb_inst,eth_params_t *eth_params,int single_thread_flag,int wait_for_sync) {
  
  int CC_id;
  int inst;
  PHY_VARS_eNB *eNB;
  int ret;

  for (inst=0;inst<nb_inst;inst++) {
    for (CC_id=0;CC_id<MAX_NUM_CCs;CC_id++) {
      eNB = PHY_vars_eNB_g[inst][CC_id]; 
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
      AssertFatal((eNB->if_inst         = IF_Module_init_NB_IoT(inst))!=NULL,"Cannot register interface");
      eNB->if_inst->schedule_response   = schedule_response_NB_IoT;
      eNB->if_inst->PHY_config_req      = PHY_config_req_NB_IoT;
      LOG_I(PHY,"Registering with MAC interface module sucessfully\n");


#ifndef OCP_FRAMEWORK
      LOG_I(PHY,"Initializing eNB %d CC_id %d : (%s,%s)\n",inst,CC_id,eNB_functions[node_function[CC_id]],eNB_timing[node_timing[CC_id]]);
#endif

      switch (node_function[CC_id]) {
      case NGFI_RRU_IF5:
  eNB->do_prach             = NULL;
  eNB->do_precoding         = 0;
  eNB->fep                  = eNB_fep_rru_if5;
  eNB->td                   = NULL;
  eNB->te                   = NULL;
  eNB->proc_uespec_rx       = NULL;
  eNB->proc_tx              = NULL;
  eNB->tx_fh                = NULL;
  eNB->rx_fh                = rx_rf;
  eNB->start_rf             = start_rf;
  eNB->start_if             = start_if;
  eNB->fh_asynch            = fh_if5_asynch_DL;
  if (oaisim_flag == 0) {
    ret = openair0_device_load(&eNB->rfdevice, &openair0_cfg[CC_id]);
    if (ret<0) {
      printf("Exiting, cannot initialize rf device\n");
      exit(-1);
    }
  }
  eNB->rfdevice.host_type   = RRH_HOST;
  eNB->ifdevice.host_type   = RRH_HOST;
        ret = openair0_transport_load(&eNB->ifdevice, &openair0_cfg[CC_id], eNB->eth_params);
  printf("openair0_transport_init returns %d for CC_id %d\n",ret,CC_id);
        if (ret<0) {
          printf("Exiting, cannot initialize transport protocol\n");
          exit(-1);
        }
  malloc_IF5_buffer(eNB);
  break;
      case NGFI_RRU_IF4p5:
  eNB->do_precoding         = 0;
  eNB->do_prach             = do_prach;
  eNB->fep                  = eNB_fep_full;//(single_thread_flag==1) ? eNB_fep_full_2thread : eNB_fep_full;
  eNB->td                   = NULL;
  eNB->te                   = NULL;
  eNB->proc_uespec_rx       = NULL;
  eNB->proc_tx              = NULL;//proc_tx_rru_if4p5;
  eNB->tx_fh                = NULL;
  eNB->rx_fh                = rx_rf;
  eNB->fh_asynch            = fh_if4p5_asynch_DL;
  eNB->start_rf             = start_rf;
  eNB->start_if             = start_if;
  if (oaisim_flag == 0) {
    ret = openair0_device_load(&eNB->rfdevice, &openair0_cfg[CC_id]);
    if (ret<0) {
      printf("Exiting, cannot initialize rf device\n");
      exit(-1);
    }
  }
  eNB->rfdevice.host_type   = RRH_HOST;
  eNB->ifdevice.host_type   = RRH_HOST;
  printf("loading transport interface ...\n");
        ret = openair0_transport_load(&eNB->ifdevice, &openair0_cfg[CC_id], eNB->eth_params);
  printf("openair0_transport_init returns %d for CC_id %d\n",ret,CC_id);
        if (ret<0) {
          printf("Exiting, cannot initialize transport protocol\n");
          exit(-1);
        }

  malloc_IF4p5_buffer(eNB);

  break;
      case eNodeB_3GPP:
  eNB->do_precoding         = eNB->frame_parms.nb_antennas_tx!=eNB->frame_parms.nb_antenna_ports_eNB;
  eNB->do_prach             = do_prach;
  eNB->fep                  = eNB_fep_full;//(single_thread_flag==1) ? eNB_fep_full_2thread : eNB_fep_full;
  eNB->td                   = ulsch_decoding_data;//(single_thread_flag==1) ? ulsch_decoding_data_2thread : ulsch_decoding_data;
  eNB->te                   = dlsch_encoding;//(single_thread_flag==1) ? dlsch_encoding_2threads : dlsch_encoding;
  ////////////////////// NB-IoT testing ////////////////////
  //eNB->proc_uespec_rx       = phy_procedures_eNB_uespec_RX;
  eNB->proc_uespec_rx       = phy_procedures_eNB_uespec_RX_NB_IoT;

  eNB->proc_tx              = proc_tx_full;
  eNB->tx_fh                = NULL;
  eNB->rx_fh                = rx_rf;
  eNB->start_rf             = start_rf;
  eNB->start_if             = NULL;
        eNB->fh_asynch            = NULL;
        if (oaisim_flag == 0) {
      ret = openair0_device_load(&eNB->rfdevice, &openair0_cfg[CC_id]);
          if (ret<0) {
            printf("Exiting, cannot initialize rf device\n");
            exit(-1);
          }
        }
  eNB->rfdevice.host_type   = BBU_HOST;
  eNB->ifdevice.host_type   = BBU_HOST;
  break;
      case eNodeB_3GPP_BBU:
  eNB->do_precoding         = eNB->frame_parms.nb_antennas_tx!=eNB->frame_parms.nb_antenna_ports_eNB;
  eNB->do_prach             = do_prach;
  eNB->fep                  = eNB_fep_full;//(single_thread_flag==1) ? eNB_fep_full_2thread : eNB_fep_full;
  eNB->td                   = ulsch_decoding_data;//(single_thread_flag==1) ? ulsch_decoding_data_2thread : ulsch_decoding_data;
  eNB->te                   = dlsch_encoding;//(single_thread_flag==1) ? dlsch_encoding_2threads : dlsch_encoding;
  eNB->proc_uespec_rx       = phy_procedures_eNB_uespec_RX;
  eNB->proc_tx              = proc_tx_full;
        if (eNB->node_timing == synch_to_other) {
           eNB->tx_fh             = tx_fh_if5_mobipass;
           eNB->rx_fh             = rx_fh_slave;
           eNB->fh_asynch         = fh_if5_asynch_UL;

        }
        else {
           eNB->tx_fh             = tx_fh_if5;
           eNB->rx_fh             = rx_fh_if5;
           eNB->fh_asynch         = NULL;
        }

  eNB->start_rf             = NULL;
  eNB->start_if             = start_if;
  eNB->rfdevice.host_type   = BBU_HOST;

  eNB->ifdevice.host_type   = BBU_HOST;

        ret = openair0_transport_load(&eNB->ifdevice, &openair0_cfg[CC_id], eNB->eth_params);
        printf("openair0_transport_init returns %d for CC_id %d\n",ret,CC_id);
        if (ret<0) {
          printf("Exiting, cannot initialize transport protocol\n");
          exit(-1);
        }
  malloc_IF5_buffer(eNB);
  break;
      case NGFI_RCC_IF4p5:
  eNB->do_precoding         = 0;
  eNB->do_prach             = do_prach;
  eNB->fep                  = NULL;
  eNB->td                   = ulsch_decoding_data;//(single_thread_flag==1) ? ulsch_decoding_data_2thread : ulsch_decoding_data;
  eNB->te                   = dlsch_encoding;//(single_thread_flag==1) ? dlsch_encoding_2threads : dlsch_encoding;
  eNB->proc_uespec_rx       = phy_procedures_eNB_uespec_RX;
  eNB->proc_tx              = proc_tx_high;
  eNB->tx_fh                = tx_fh_if4p5;
  eNB->rx_fh                = rx_fh_if4p5;
  eNB->start_rf             = NULL;
  eNB->start_if             = start_if;
        eNB->fh_asynch            = (eNB->node_timing == synch_to_other) ? fh_if4p5_asynch_UL : NULL;
  eNB->rfdevice.host_type   = BBU_HOST;
  eNB->ifdevice.host_type   = BBU_HOST;
        ret = openair0_transport_load(&eNB->ifdevice, &openair0_cfg[CC_id], eNB->eth_params);
        printf("openair0_transport_init returns %d for CC_id %d\n",ret,CC_id);
        if (ret<0) {
          printf("Exiting, cannot initialize transport protocol\n");
          exit(-1);
        }
  malloc_IF4p5_buffer(eNB);

  break;
      case NGFI_RAU_IF4p5:
  eNB->do_precoding   = 0;
  eNB->do_prach       = do_prach;
  eNB->fep            = NULL;

  eNB->td             = ulsch_decoding_data;//(single_thread_flag==1) ? ulsch_decoding_data_2thread : ulsch_decoding_data;
  eNB->te             = dlsch_encoding;//(single_thread_flag==1) ? dlsch_encoding_2threads : dlsch_encoding;
  eNB->proc_uespec_rx = phy_procedures_eNB_uespec_RX;
  eNB->proc_tx        = proc_tx_high;
  eNB->tx_fh          = tx_fh_if4p5; 
  eNB->rx_fh          = rx_fh_if4p5; 
        eNB->fh_asynch      = (eNB->node_timing == synch_to_other) ? fh_if4p5_asynch_UL : NULL;
  eNB->start_rf       = NULL;
  eNB->start_if       = start_if;

  eNB->rfdevice.host_type   = BBU_HOST;
  eNB->ifdevice.host_type   = BBU_HOST;
        ret = openair0_transport_load(&eNB->ifdevice, &openair0_cfg[CC_id], eNB->eth_params);
        printf("openair0_transport_init returns %d for CC_id %d\n",ret,CC_id);
        if (ret<0) {
          printf("Exiting, cannot initialize transport protocol\n");
          exit(-1);
        }
  break;  
  malloc_IF4p5_buffer(eNB);

      }

      if (setup_eNB_buffers(PHY_vars_eNB_g[inst],&openair0_cfg[CC_id])!=0) {
  printf("Exiting, cannot initialize eNodeB Buffers\n");
  exit(-1);
      }
    }

    init_eNB_proc(inst);
  }

  sleep(1);
  LOG_D(HW,"[lte-softmodem.c] eNB threads created\n");
  

}

