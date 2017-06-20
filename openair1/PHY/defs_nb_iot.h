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

/*! \file PHY/defs.h
 \brief Top-level defines and structure definitions
 \author R. Knopp, F. Kaltenberger
 \date 2011
 \version 0.1
 \company Eurecom
 \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr
 \note
 \warning
*/
#ifndef __PHY_DEFS_NB_IOT__H__
#define __PHY_DEFS_NB_IOT__H__

#define _GNU_SOURCE 
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include "common_lib.h"

//#include <complex.h>
#include "assertions.h"
#ifdef MEX
# define msg mexPrintf
#else
# ifdef OPENAIR2
#   if ENABLE_RAL
#     include "collection/hashtable/hashtable.h"
#     include "COMMON/ral_messages_types.h"
#     include "UTIL/queue.h"
#   endif
#   include "log.h"
#   define msg(aRGS...) LOG_D(PHY, ##aRGS)
# else
#   define msg printf
# endif
#endif
//use msg in the real-time thread context
#define msg_nrt printf
//use msg_nrt in the non real-time context (for initialization, ...)
#ifndef malloc16
#  ifdef __AVX2__
#    define malloc16(x) memalign(32,x)
#  else
#    define malloc16(x) memalign(16,x)
#  endif
#endif
#define free16(y,x) free(y)
#define bigmalloc malloc
#define bigmalloc16 malloc16
#define openair_free(y,x) free((y))
#define PAGE_SIZE 4096

//#ifdef SHRLIBDEV
//extern int rxrescale;
//#define RX_IQRESCALELEN rxrescale
//#else
//#define RX_IQRESCALELEN 15
//#endif





#define PAGE_MASK 0xfffff000
#define virt_to_phys(x) (x)

#define openair_sched_exit() exit(-1)


#define max(a,b)  ((a)>(b) ? (a) : (b))
#define min(a,b)  ((a)<(b) ? (a) : (b))


#define bzero(s,n) (memset((s),0,(n)))

#define cmax(a,b)  ((a>b) ? (a) : (b))
#define cmin(a,b)  ((a<b) ? (a) : (b))

#define cmax3(a,b,c) ((cmax(a,b)>c) ? (cmax(a,b)) : (c))

/// suppress compiler warning for unused arguments
#define UNUSED(x) (void)x;


#include "impl_defs_top.h"
#include "impl_defs_lte.h"
#include "impl_defs_lte_nb_iot.h"

#include "PHY/TOOLS/time_meas.h"
#include "PHY/CODING/defs.h"
#include "PHY/TOOLS/defs.h"
#include "platform_types.h"

#ifdef OPENAIR_LTE

#include "PHY/LTE_TRANSPORT/defs.h"
#include "PHY/LTE_TRANSPORT/defs_nb_iot.h"
#include <pthread.h>

#include "targets/ARCH/COMMON/common_lib.h"

#define NUM_DCI_MAX 32

#define NUMBER_OF_eNB_SECTORS_MAX 3

#define NB_BANDS_MAX 8






/// Context data structure for RX/TX portion of subframe processing
typedef struct {
  /// Component Carrier index
  uint8_t              CC_id;
  /// timestamp transmitted to HW
  openair0_timestamp timestamp_tx;
  /// subframe to act upon for transmission
  int subframe_tx;
  /// subframe to act upon for reception
  int subframe_rx;
  /// frame to act upon for transmission
  int frame_tx;
  /// frame to act upon for reception
  int frame_rx;
  /// \brief Instance count for RXn-TXnp4 processing thread.
  /// \internal This variable is protected by \ref mutex_rxtx.
  int instance_cnt_rxtx;
  /// pthread structure for RXn-TXnp4 processing thread
  pthread_t pthread_rxtx;
  /// pthread attributes for RXn-TXnp4 processing thread
  pthread_attr_t attr_rxtx;
  /// condition variable for tx processing thread
  pthread_cond_t cond_rxtx;
  /// mutex for RXn-TXnp4 processing thread
  pthread_mutex_t mutex_rxtx;
  /// scheduling parameters for RXn-TXnp4 thread
  struct sched_param sched_param_rxtx;
  
  /// for IF_Module
  pthread_t pthread_l2;
  pthread_cond_t cond_l2;
  pthread_mutex_t mutex_l2;
  int instance_cnt_l2;
} eNB_rxtx_proc_NB_t;


/// Context data structure for eNB subframe processing
typedef struct eNB_proc_NB_t_s {
  /// Component Carrier index
  uint8_t              CC_id;
  /// thread index
  int thread_index;
  /// timestamp received from HW
  openair0_timestamp timestamp_rx;
  /// timestamp to send to "slave rru"
  openair0_timestamp timestamp_tx;
  /// subframe to act upon for reception
  int subframe_rx;
  /// symbol mask for IF4p5 reception per subframe
  uint32_t symbol_mask[10];
  /// subframe to act upon for PRACH
  int subframe_prach;
  /// frame to act upon for reception
  int frame_rx;
  /// frame to act upon for transmission
  int frame_tx;
  /// frame offset for secondary eNBs (to correct for frame asynchronism at startup)
  int frame_offset;
  /// frame to act upon for PRACH
  int frame_prach;
  /// \internal This variable is protected by \ref mutex_fep.
  int instance_cnt_fep;
  /// \internal This variable is protected by \ref mutex_td.
  int instance_cnt_td;
  /// \internal This variable is protected by \ref mutex_te.
  int instance_cnt_te;
  /// \brief Instance count for FH processing thread.
  /// \internal This variable is protected by \ref mutex_FH.
  int instance_cnt_FH;
  /// \brief Instance count for rx processing thread.
  /// \internal This variable is protected by \ref mutex_prach.
  int instance_cnt_prach;
  // instance count for over-the-air eNB synchronization
  int instance_cnt_synch;
  /// \internal This variable is protected by \ref mutex_asynch_rxtx.
  int instance_cnt_asynch_rxtx;
  /// pthread structure for FH processing thread
  pthread_t pthread_FH;
  /// pthread structure for eNB single processing thread
  pthread_t pthread_single;
  /// pthread structure for asychronous RX/TX processing thread
  pthread_t pthread_asynch_rxtx;
  /// flag to indicate first RX acquisition
  int first_rx;
  /// flag to indicate first TX transmission
  int first_tx;
  /// pthread attributes for parallel fep thread
  pthread_attr_t attr_fep;
  /// pthread attributes for parallel turbo-decoder thread
  pthread_attr_t attr_td;
  /// pthread attributes for parallel turbo-encoder thread
  pthread_attr_t attr_te;
  /// pthread attributes for FH processing thread
  pthread_attr_t attr_FH;
  /// pthread attributes for single eNB processing thread
  pthread_attr_t attr_single;
  /// pthread attributes for prach processing thread
  pthread_attr_t attr_prach;
  /// pthread attributes for over-the-air synch thread
  pthread_attr_t attr_synch;
  /// pthread attributes for asynchronous RX thread
  pthread_attr_t attr_asynch_rxtx;
  /// scheduling parameters for parallel fep thread
  struct sched_param sched_param_fep;
  /// scheduling parameters for parallel turbo-decoder thread
  struct sched_param sched_param_td;
  /// scheduling parameters for parallel turbo-encoder thread
  struct sched_param sched_param_te;
  /// scheduling parameters for FH thread
  struct sched_param sched_param_FH;
  /// scheduling parameters for single eNB thread
  struct sched_param sched_param_single;
  /// scheduling parameters for prach thread
  struct sched_param sched_param_prach;
  /// scheduling parameters for over-the-air synchronization thread
  struct sched_param sched_param_synch;
  /// scheduling parameters for asynch_rxtx thread
  struct sched_param sched_param_asynch_rxtx;
  /// pthread structure for parallel fep thread
  pthread_t pthread_fep;
  /// pthread structure for parallel turbo-decoder thread
  pthread_t pthread_td;
  /// pthread structure for parallel turbo-encoder thread
  pthread_t pthread_te;
  /// pthread structure for PRACH thread
  pthread_t pthread_prach;
  /// pthread structure for eNB synch thread
  pthread_t pthread_synch;
  /// condition variable for parallel fep thread
  pthread_cond_t cond_fep;
  /// condition variable for parallel turbo-decoder thread
  pthread_cond_t cond_td;
  /// condition variable for parallel turbo-encoder thread
  pthread_cond_t cond_te;
  /// condition variable for FH thread
  pthread_cond_t cond_FH;
  /// condition variable for PRACH processing thread;
  pthread_cond_t cond_prach;
  // condition variable for over-the-air eNB synchronization
  pthread_cond_t cond_synch;
  /// condition variable for asynch RX/TX thread
  pthread_cond_t cond_asynch_rxtx;
  /// mutex for parallel fep thread
  pthread_mutex_t mutex_fep;
  /// mutex for parallel turbo-decoder thread
  pthread_mutex_t mutex_td;
  /// mutex for parallel turbo-encoder thread
  pthread_mutex_t mutex_te;
  /// mutex for FH
  pthread_mutex_t mutex_FH;
  /// mutex for PRACH thread
  pthread_mutex_t mutex_prach;
  // mutex for over-the-air eNB synchronization
  pthread_mutex_t mutex_synch;
  /// mutex for asynch RX/TX thread
  pthread_mutex_t mutex_asynch_rxtx;
  /// parameters for turbo-decoding worker thread
  td_params tdp;
  /// parameters for turbo-encoding worker thread
  te_params tep;
  /// set of scheduling variables RXn-TXnp4 threads
  eNB_rxtx_proc_NB_t proc_rxtx[2];

} eNB_proc_NB_t;


/// Top-level PHY Data Structure for eNB
typedef struct PHY_VARS_eNB_NB_s {
  /// Module ID indicator for this instance
  module_id_t          Mod_id;
  uint8_t              CC_id;
  eNB_proc_NB_t           proc;
  NB_DL_FRAME_PARMS   frame_parms;

  //number of UE max = 1 // in fapy is udated dinamically each subframe  and is no more a table
  LTE_eNB_DLSCH_t     *dlsch[NUMBER_OF_UE_MAX][2];   // Nusers times two spatial streams
  //how many ULSCH we keep???? should be more that 1 for sure
  LTE_eNB_ULSCH_t     *ulsch[NUMBER_OF_UE_MAX+1];      // Nusers + number of RA
  LTE_eNB_DLSCH_t     *dlsch_SI,*dlsch_ra;
  LTE_eNB_DLSCH_t     *dlsch_MCH;

} PHY_VARS_eNB_NB;



#include "PHY/INIT/defs.h"
#include "PHY/LTE_REFSIG/defs.h"
#include "PHY/MODULATION/defs.h"
#include "PHY/LTE_TRANSPORT/proto.h"
#include "PHY/LTE_TRANSPORT/proto_nb_iot.h"
#include "PHY/LTE_ESTIMATION/defs.h"

#include "SIMULATION/ETH_TRANSPORT/defs.h"
#endif
#endif //  __PHY_DEFS__H__
