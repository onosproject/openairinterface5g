#ifndef NR_SOFTMODEM_COMMON_H
#define NR_SOFTMODEM_COMMON_H

#ifndef _GNU_SOURCE
  #define _GNU_SOURCE
#endif

#include <execinfo.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/sched.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syscall.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/sysinfo.h>
#include "targets/ARCH/COMMON/common_lib.h"
#undef MALLOC
#include "assertions.h"
#include "PHY/types.h"
#include <threadPool/thread-pool.h>

#include "s1ap_eNB.h"
#include "SIMULATION/ETH_TRANSPORT/proto.h"
#include "executables/softmodem-common.h"


/***************************************************************************************************************************************/


extern pthread_cond_t sync_cond;
extern pthread_mutex_t sync_mutex;
extern int sync_var;


extern int32_t uplink_frequency_offset[MAX_NUM_CCs][4];

extern int rx_input_level_dBm;

extern int oaisim_flag;
extern volatile int  oai_exit;

extern openair0_config_t openair0_cfg[MAX_CARDS];
extern pthread_cond_t sync_cond;
extern pthread_mutex_t sync_mutex;
extern int sync_var;
extern int transmission_mode;
extern double cpuf;

#if defined(ENABLE_ITTI)
  extern volatile int start_eNB;
  extern volatile int start_UE;
#endif

#endif
