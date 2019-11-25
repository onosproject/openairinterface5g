#ifndef NR_SOFTMODEM_H
#define NR_SOFTMODEM_H

#include <executables/nr-softmodem-common.h>

#include "flexran_agent.h"
#include "PHY/defs_gNB.h"
#include "proto_agent.h"







extern int T_port;
extern int T_nowait;
extern int T_dont_fork;


#include "threads_t.h"
extern threads_t threads;

// In nr-gnb.c
extern void init_gNB(int single_thread_flag,int wait_for_sync);
extern void stop_gNB(int);
extern void kill_gNB_proc(int inst);

// In nr-ru.c
extern void init_NR_RU(char *);
extern void init_RU_proc(RU_t *ru);
extern void stop_RU(int nb_ru);
extern void kill_NR_RU_proc(int inst);
extern void set_function_spec_param(RU_t *ru);

extern void reset_opp_meas(void);
extern void print_opp_meas(void);

extern void init_fep_thread(PHY_VARS_gNB *);

void init_gNB_afterRU(void);

extern int stop_L1L2(module_id_t gnb_id);
extern int restart_L1L2(module_id_t gnb_id);

#endif
