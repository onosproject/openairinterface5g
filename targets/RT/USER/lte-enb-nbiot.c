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
extern volatile int                    oai_exit;

void init_eNB_NB_IoT(eNB_func_NB_IoT_t node_function[], eNB_timing_NB_IoT_t node_timing[],int nb_inst,eth_params_t *,int,int);
extern void do_prach(PHY_VARS_eNB_NB_IoT *eNB,int frame,int subframe);





/*!
 * \brief The prach receive thread of eNB.
 * \param param is a \ref eNB_proc_t structure which contains the info what to process.
 * \returns a pointer to an int. The storage is not on the heap and must not be freed.
 */
static void* eNB_thread_prach_NB_IoT( void* param ) {
  static int eNB_thread_prach_status;

  PHY_VARS_eNB_NB_IoT *eNB= (PHY_VARS_eNB_NB_IoT *)param;
  L1_proc_t *proc = &eNB->proc;

  // set default return value
  eNB_thread_prach_status = 0;

  thread_top_init("eNB_thread_prach_NB_IoT",1,500000L,1000000L,20000000L);

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
void init_eNB_NB_IoT(eNB_func_NB_IoT_t node_function[], eNB_timing_NB_IoT_t node_timing[],int nb_inst,eth_params_t *eth_params,int single_thread_flag,int wait_for_sync) {
  
  int CC_id;
  int inst;
  PHY_VARS_eNB_NB_IoT *eNB;
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

    init_eNB_proc_NB_IoT(inst);
  }

  sleep(1);
  LOG_D(HW,"[lte-softmodem.c] eNB threads created\n");
  

}


void init_eNB_proc_NB_IoT(int inst) {
  
  int i=0;
  int CC_id;
  PHY_VARS_eNB_NB_IoT *eNB;
  eNB_proc_t *proc;
  L1_rxtx_proc_t *proc_rxtx;
  pthread_attr_t *attr0=NULL,*attr1=NULL,*attr_FH=NULL,*attr_prach=NULL,*attr_asynch=NULL,*attr_single=NULL,*attr_fep=NULL,*attr_td=NULL,*attr_te=NULL,*attr_synch=NULL;

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    eNB = PHY_vars_eNB_g[inst][CC_id];
#ifndef OCP_FRAMEWORK
    LOG_I(PHY,"Initializing eNB %d CC_id %d (%s,%s),\n",inst,CC_id,eNB_functions[eNB->node_function],eNB_timing[eNB->node_timing]);
#endif
    proc = &eNB->proc;

    proc_rxtx = proc->proc_rxtx;
    proc_rxtx[0].instance_cnt_rxtx = -1;
    proc_rxtx[1].instance_cnt_rxtx = -1;
    proc->instance_cnt_prach       = -1;
    proc->instance_cnt_FH          = -1;
    proc->instance_cnt_asynch_rxtx = -1;
    proc->CC_id = CC_id;    
    proc->instance_cnt_synch        =  -1;

    proc->first_rx=1;
    proc->first_tx=1;
    proc->frame_offset = 0;

    for (i=0;i<10;i++) proc->symbol_mask[i]=0;

    pthread_mutex_init( &proc_rxtx[0].mutex_rxtx, NULL);
    pthread_mutex_init( &proc_rxtx[1].mutex_rxtx, NULL);
    pthread_cond_init( &proc_rxtx[0].cond_rxtx, NULL);
    pthread_cond_init( &proc_rxtx[1].cond_rxtx, NULL);

    pthread_mutex_init( &proc->mutex_prach, NULL);
    pthread_mutex_init( &proc->mutex_asynch_rxtx, NULL);
    pthread_mutex_init( &proc->mutex_synch,NULL);

    pthread_cond_init( &proc->cond_prach, NULL);
    pthread_cond_init( &proc->cond_FH, NULL);
    pthread_cond_init( &proc->cond_asynch_rxtx, NULL);
    pthread_cond_init( &proc->cond_synch,NULL);

    pthread_attr_init( &proc->attr_FH);
    pthread_attr_init( &proc->attr_prach);
    pthread_attr_init( &proc->attr_synch);
    pthread_attr_init( &proc->attr_asynch_rxtx);
    pthread_attr_init( &proc->attr_single);
    pthread_attr_init( &proc->attr_fep);
    pthread_attr_init( &proc->attr_td);
    pthread_attr_init( &proc->attr_te);
    pthread_attr_init( &proc_rxtx[0].attr_rxtx);
    pthread_attr_init( &proc_rxtx[1].attr_rxtx);
#ifndef DEADLINE_SCHEDULER
    attr0       = &proc_rxtx[0].attr_rxtx;
    attr1       = &proc_rxtx[1].attr_rxtx;
    attr_FH     = &proc->attr_FH;
    attr_prach  = &proc->attr_prach;
    attr_synch  = &proc->attr_synch;
    attr_asynch = &proc->attr_asynch_rxtx;
    attr_single = &proc->attr_single;
    attr_fep    = &proc->attr_fep;
    attr_td     = &proc->attr_td;
    attr_te     = &proc->attr_te; 
#endif

    if (eNB->single_thread_flag==0) {
      //the two threads that manage tw consecutive subframes
      pthread_create( &proc_rxtx[0].pthread_rxtx, attr0, eNB_thread_rxtx, &proc_rxtx[0] );
      pthread_create( &proc_rxtx[1].pthread_rxtx, attr1, eNB_thread_rxtx, &proc_rxtx[1] );
      pthread_create( &proc->pthread_FH, attr_FH, eNB_thread_FH, &eNB->proc );
    }
    else {
      pthread_create(&proc->pthread_single, attr_single, eNB_thread_single, &eNB->proc);
      init_fep_thread(eNB,attr_fep);
      /*
      init_td_thread(eNB,attr_td);
      init_te_thread(eNB,attr_te);
      */
    }
    pthread_create( &proc->pthread_prach, attr_prach, eNB_thread_prach_NB_IoT, &eNB->proc );
    pthread_create( &proc->pthread_synch, attr_synch, eNB_thread_synch, eNB);
    if ((eNB->node_timing == synch_to_other) ||
  (eNB->node_function == NGFI_RRU_IF5) ||
  (eNB->node_function == NGFI_RRU_IF4p5))


      pthread_create( &proc->pthread_asynch_rxtx, attr_asynch, eNB_thread_asynch_rxtx, &eNB->proc );

    char name[16];
    if (eNB->single_thread_flag == 0) {
      snprintf( name, sizeof(name), "RXTX0 %d", i );
      pthread_setname_np( proc_rxtx[0].pthread_rxtx, name );
      snprintf( name, sizeof(name), "RXTX1 %d", i );
      pthread_setname_np( proc_rxtx[1].pthread_rxtx, name );
      snprintf( name, sizeof(name), "FH %d", i );
      pthread_setname_np( proc->pthread_FH, name );
    }
    else {
      snprintf( name, sizeof(name), " %d", i );
      pthread_setname_np( proc->pthread_single, name );
    }
  }

  //for multiple CCs: setup master and slaves
 /* 
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    eNB = PHY_vars_eNB_g[inst][CC_id];

    if (eNB->node_timing == synch_to_ext_device) { //master
      eNB->proc.num_slaves = MAX_NUM_CCs-1;
      eNB->proc.slave_proc = (eNB_proc_t**)malloc(eNB->proc.num_slaves*sizeof(eNB_proc_t*));

      for (i=0; i< eNB->proc.num_slaves; i++) {
        if (i < CC_id)  eNB->proc.slave_proc[i] = &(PHY_vars_eNB_g[inst][i]->proc);
        if (i >= CC_id)  eNB->proc.slave_proc[i] = &(PHY_vars_eNB_g[inst][i+1]->proc);
      }
    }
    }
*/

  /* setup PHY proc TX sync mechanism */
  pthread_mutex_init(&sync_phy_proc.mutex_phy_proc_tx, NULL);
  pthread_cond_init(&sync_phy_proc.cond_phy_proc_tx, NULL);
  sync_phy_proc.phy_proc_CC_id = 0;
}



/************************************************************************************************************************/
/* this function maps the phy_vars_eNB tx and rx buffers to the available rf chains.
   Each rf chain is is addressed by the card number and the chain on the card. The
   rf_map specifies for each CC, on which rf chain the mapping should start. Multiple
   antennas are mapped to successive RF chains on the same card. */
int setup_eNB_buffers(PHY_VARS_eNB_NB_IoT **phy_vars_eNB, openair0_config_t *openair0_cfg) {

  int i,j; 
  int CC_id,card,ant;

  //uint16_t N_TA_offset = 0;

  LTE_DL_FRAME_PARMS *frame_parms;

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    if (phy_vars_eNB[CC_id]) {
      frame_parms = &(phy_vars_eNB[CC_id]->frame_parms);
      printf("setup_eNB_buffers: frame_parms = %p\n",frame_parms);
    } else {
      printf("phy_vars_eNB[%d] not initialized\n", CC_id);
      return(-1);
    }

    /*
    if (frame_parms->frame_type == TDD) {
      if (frame_parms->N_RB_DL == 100)
        N_TA_offset = 624;
      else if (frame_parms->N_RB_DL == 50)
        N_TA_offset = 624/2;
      else if (frame_parms->N_RB_DL == 25)
        N_TA_offset = 624/4;
    }
    */
 

    if (openair0_cfg[CC_id].mmapped_dma == 1) {
    // replace RX signal buffers with mmaped HW versions
      
      for (i=0; i<frame_parms->nb_antennas_rx; i++) {
  card = i/4;
  ant = i%4;
  printf("Mapping eNB CC_id %d, rx_ant %d, on card %d, chain %d\n",CC_id,i,phy_vars_eNB[CC_id]->rf_map.card+card, phy_vars_eNB[CC_id]->rf_map.chain+ant);
  free(phy_vars_eNB[CC_id]->common_vars.rxdata[0][i]);
  phy_vars_eNB[CC_id]->common_vars.rxdata[0][i] = openair0_cfg[phy_vars_eNB[CC_id]->rf_map.card+card].rxbase[phy_vars_eNB[CC_id]->rf_map.chain+ant];
  
  printf("rxdata[%d] @ %p\n",i,phy_vars_eNB[CC_id]->common_vars.rxdata[0][i]);
  for (j=0; j<16; j++) {
    printf("rxbuffer %d: %x\n",j,phy_vars_eNB[CC_id]->common_vars.rxdata[0][i][j]);
    phy_vars_eNB[CC_id]->common_vars.rxdata[0][i][j] = 16-j;
  }
      }
      
      for (i=0; i<frame_parms->nb_antennas_tx; i++) {
  card = i/4;
  ant = i%4;
  printf("Mapping eNB CC_id %d, tx_ant %d, on card %d, chain %d\n",CC_id,i,phy_vars_eNB[CC_id]->rf_map.card+card, phy_vars_eNB[CC_id]->rf_map.chain+ant);
  free(phy_vars_eNB[CC_id]->common_vars.txdata[0][i]);
  phy_vars_eNB[CC_id]->common_vars.txdata[0][i] = openair0_cfg[phy_vars_eNB[CC_id]->rf_map.card+card].txbase[phy_vars_eNB[CC_id]->rf_map.chain+ant];
  
  printf("txdata[%d] @ %p\n",i,phy_vars_eNB[CC_id]->common_vars.txdata[0][i]);
  
  for (j=0; j<16; j++) {
    printf("txbuffer %d: %x\n",j,phy_vars_eNB[CC_id]->common_vars.txdata[0][i][j]);
    phy_vars_eNB[CC_id]->common_vars.txdata[0][i][j] = 16-j;
  }
      }
    }
    else {  // not memory-mapped DMA 
      //nothing to do, everything already allocated in lte_init
      /*
      rxdata = (int32_t**)malloc16(frame_parms->nb_antennas_rx*sizeof(int32_t*));
      txdata = (int32_t**)malloc16(frame_parms->nb_antennas_tx*sizeof(int32_t*));
      
      for (i=0; i<frame_parms->nb_antennas_rx; i++) {
  free(phy_vars_eNB[CC_id]->common_vars.rxdata[0][i]);
  rxdata[i] = (int32_t*)(32 + malloc16(32+frame_parms->samples_per_tti*10*sizeof(int32_t))); // FIXME broken memory allocation
  phy_vars_eNB[CC_id]->common_vars.rxdata[0][i] = rxdata[i]; //-N_TA_offset; // N_TA offset for TDD         FIXME! N_TA_offset > 16 => access of unallocated memory
  memset(rxdata[i], 0, frame_parms->samples_per_tti*10*sizeof(int32_t));
  printf("rxdata[%d] @ %p (%p) (N_TA_OFFSET %d)\n", i, phy_vars_eNB[CC_id]->common_vars.rxdata[0][i],rxdata[i],N_TA_offset);      
      }
      
      for (i=0; i<frame_parms->nb_antennas_tx; i++) {
  free(phy_vars_eNB[CC_id]->common_vars.txdata[0][i]);
  txdata[i] = (int32_t*)(32 + malloc16(32 + frame_parms->samples_per_tti*10*sizeof(int32_t))); // FIXME broken memory allocation
  phy_vars_eNB[CC_id]->common_vars.txdata[0][i] = txdata[i];
  memset(txdata[i],0, frame_parms->samples_per_tti*10*sizeof(int32_t));
  printf("txdata[%d] @ %p\n", i, phy_vars_eNB[CC_id]->common_vars.txdata[0][i]);
      }
      */
    }
  }

  return(0);
}
