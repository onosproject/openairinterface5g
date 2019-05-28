#include <common/utils/system.h>
#include <common/utils/LOG/log.h>
#include "common/ran_context.h"
#include "PHY/defs_gNB.h"
#include "common/config/config_userapi.h"
#include "common/utils/load_module_shlib.h"
#include "intertask_interface.h"
#include "otg_defs.h"
#include "UTIL/OTG/otg_vars.h"
#include <openair2/UTIL/OPT/opt.h>
#include <nr-softmodem.h>
#include "NB_IoT_interface.h"
#include <openair2/GNB_APP/gnb_config.h>
#include <openair2/GNB_APP/gnb_app.h>
#include <openair1/SCHED_NR/sched_nr.h>
#include <openair1/SCHED_NR/fapi_nr_l1.h>
#include <openair1/PHY/INIT/phy_init.h>
#include <targets/ARCH/ETHERNET/USERSPACE/LIB/if_defs.h>

#include <openair1/PHY/NR_REFSIG/nr_mod_table.h>
#include <openair1/PHY/phy_vars.h>
#include <openair2/LAYER2/MAC/mac_vars.h>
#include <openair2/RRC/LTE/rrc_vars.h>
#include <openair1/SCHED/sched_common_vars.h>
#include <common/utils/threadPool/thread-pool.h>

RAN_CONTEXT_t RC;
tpool_t * Tpool;
volatile int oai_exit = 0;
char   rf_config_file[1024]="";
unsigned int mmapped_dma=0;
int single_thread_flag=1;
int phy_test = 0;
uint8_t usim_test = 0;
static clock_source_t clock_source = internal;
static int wait_for_sync = 0;
static int8_t threequarter_fs=0;
static char   do_forms=0;
static int DEFBANDS[] = {7};
static int DEFENBS[] = {0};
#include "ENB_APP/enb_paramdef.h"
// Fuck dirty code
//#include <openair2/GNB_APP/gnb_paramdef.h>

uint32_t timing_advance = 0;
uint32_t target_dl_mcs = 28; //maximum allowed mcs
uint32_t target_ul_mcs = 20;
static char *itti_dump_file = NULL;
int transmission_mode=1;
int otg_enabled=0;
int emulate_rf = 0;
int numerology = 0;
char *parallel_config = NULL;
char *worker_config = NULL;
int chain_offset=0;
unsigned char NB_gNB_INST = 1;
uint8_t nfapi_mode = 0;
runmode_t mode = normal_txrx;
uint32_t  downlink_frequency[MAX_NUM_CCs][4];
int32_t   uplink_frequency_offset[MAX_NUM_CCs][4];
openair0_config_t openair0_cfg[MAX_CARDS];
uint8_t exit_missed_slots=1;
pthread_cond_t sync_cond;
pthread_mutex_t sync_mutex;
int sync_var=-1;
int config_sync_var=-1;
pthread_cond_t nfapi_sync_cond;
pthread_mutex_t nfapi_sync_mutex;
int nfapi_sync_var=-1;
static struct {
  pthread_mutex_t  mutex_phy_proc_tx;
  pthread_cond_t   cond_phy_proc_tx;
  volatile uint8_t phy_proc_CC_id;
} sync_phy_proc;

unsigned char scope_enb_num_ue = 2;
double cpuf;
uint16_t sf_ahead=4;
uint16_t sl_ahead=4;
static notifiedFIFO_t mainThreadFIFO;
typedef enum {
  AbortRU,
  doneInitRU
} localMsg;

WORKER_CONF_t get_thread_worker_conf(void) {
  return  WORKER_DISABLE;
}
void set_parallel_conf(char *parallel_conf) {
}
void set_worker_conf(char *worker_conf) {
}
int stop_L1L2(module_id_t gnb_id) {
  return 0;
}

int start_rf(RU_t *ru) {
  return(ru->rfdevice.trx_start_func(&ru->rfdevice));
}

int stop_rf(RU_t *ru) {
  ru->rfdevice.trx_end_func(&ru->rfdevice);
  return 0;
}

void rx_rf(RU_t *ru,int *frame,int *slot) {
  RU_proc_t *proc = &ru->proc;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  void *rxp[ru->nb_rx];
  unsigned int rxs;
  int i;
  openair0_timestamp ts,old_ts;
  AssertFatal(*slot<fp->slots_per_frame && *slot>=0, "slot %d is illegal (%d)\n",*slot,fp->slots_per_frame);

  for (i=0; i<ru->nb_rx; i++)
    rxp[i] = (void *)&ru->common.rxdata[i][*slot*fp->samples_per_slot];

  old_ts = proc->timestamp_rx;
  LOG_D(PHY,"Reading %d samples for slot %d (%p)\n",fp->samples_per_slot,*slot,rxp[0]);
  rxs = ru->rfdevice.trx_read_func(&ru->rfdevice,
                                   &ts,
                                   rxp,
                                   fp->samples_per_slot,
                                   ru->nb_rx);
  proc->timestamp_rx = ts-ru->ts_offset;

  if (rxs != fp->samples_per_slot)
    LOG_E(PHY, "rx_rf: Asked for %d samples, got %d from USRP\n",fp->samples_per_slot,rxs);

  if (proc->first_rx == 1) {
    ru->ts_offset = proc->timestamp_rx;
    proc->timestamp_rx = 0;
  } else {
    if (proc->timestamp_rx - old_ts != fp->samples_per_slot) {
      LOG_I(PHY,"rx_rf: rfdevice timing drift of %"PRId64" samples (ts_off %"PRId64")\n",
            proc->timestamp_rx - old_ts - fp->samples_per_slot,ru->ts_offset);
      ru->ts_offset += (proc->timestamp_rx - old_ts - fp->samples_per_slot);
      proc->timestamp_rx = ts-ru->ts_offset;
    }
  }

  proc->frame_rx     = (proc->timestamp_rx / (fp->samples_per_slot*fp->slots_per_frame))&1023;
  proc->tti_rx       = (proc->timestamp_rx / fp->samples_per_slot)%fp->slots_per_frame;
  // synchronize first reception to frame 0 subframe 0
  proc->timestamp_tx = proc->timestamp_rx+(sl_ahead*fp->samples_per_slot);
  proc->tti_tx  = (proc->tti_rx+sl_ahead)%fp->slots_per_frame;
  proc->frame_tx     = (proc->tti_rx>(fp->slots_per_frame-1-sl_ahead)) ? (proc->frame_rx+1)&1023 : proc->frame_rx;
  LOG_I(PHY,"RU %d/%d TS %lu (off %ld), frame %d, slot %d.%d / %d\n",
        ru->idx,
        0,
        proc->timestamp_rx,
        ru->ts_offset,proc->frame_rx,proc->tti_rx,proc->tti_tx,fp->slots_per_frame);

  if (proc->first_rx == 0) {
    AssertFatal(proc->tti_rx == *slot,
                "Received Timestamp (%lu) doesn't correspond to the time we think it is (proc->tti_rx %d, subframe %d)\n",
                proc->timestamp_rx,proc->tti_rx,*slot);
    AssertFatal(proc->frame_rx == *frame,
                "Received Timestamp (%lu) doesn't correspond to the time we think it is (proc->frame_rx %d frame %d)\n",
                proc->timestamp_rx,proc->frame_rx,*frame);
  } else {
    proc->first_rx = 0;
    *frame = proc->frame_rx;
    *slot  = proc->tti_rx;
  }

  if (rxs != fp->samples_per_slot) {
    //exit_fun( "problem receiving samples" );
    LOG_E(PHY, "problem receiving samples\n");
  }
}


void tx_rf(RU_t *ru) {
  RU_proc_t *proc = &ru->proc;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  nfapi_nr_config_request_t *cfg = &ru->gNB_list[0]->gNB_config;
  void *txp[ru->nb_tx];
  unsigned int txs;
  int i;
  nr_subframe_t SF_type     = nr_slot_select(cfg,proc->tti_tx%fp->slots_per_frame);
  int sf_extension = 0;

  if ((SF_type == SF_DL) ||
      (SF_type == SF_S)) {
    int siglen=fp->samples_per_slot,flags=1;
    /*
        if (SF_type == SF_S) {
          siglen = fp->dl_symbols_in_S_subframe*(fp->ofdm_symbol_size+fp->nb_prefix_samples0);
          flags=3; // end of burst
        }
        if ((fp->frame_type == TDD) &&
      (SF_type == SF_DL)&&
      (prevSF_type == SF_UL) &&
      (nextSF_type == SF_DL)) {
          flags = 2; // start of burst
          sf_extension = ru->N_TA_offset<<1;
        }

        if ((cfg->subframe_config.duplex_mode == TDD) &&
      (SF_type == SF_DL)&&
      (prevSF_type == SF_UL) &&
      (nextSF_type == SF_UL)) {
          flags = 4; // start of burst and end of burst (only one DL SF between two UL)
          sf_extension = ru->N_TA_offset<<1;
        } */

    for (i=0; i<ru->nb_tx; i++)
      txp[i] = (void *)&ru->common.txdata[i][(proc->tti_tx*fp->samples_per_slot)-sf_extension];

    // prepare tx buffer pointers
    txs = ru->rfdevice.trx_write_func(&ru->rfdevice,
                                      proc->timestamp_tx+ru->ts_offset-ru->openair0_cfg.tx_sample_advance-sf_extension,
                                      txp,
                                      siglen+sf_extension,
                                      ru->nb_tx,
                                      flags);
    LOG_D(PHY,"[TXPATH] RU %d tx_rf, writing to TS %llu, frame %d, unwrapped_frame %d, subframe %d\n",ru->idx,
          (long long unsigned int)proc->timestamp_tx,proc->frame_tx,proc->frame_tx_unwrap,proc->tti_tx);
    AssertFatal(txs == siglen+sf_extension,"TX : Timeout (sent %d/%d)\n",txs, siglen);
  }
}

void init_gNB_proc(int inst) {
  PHY_VARS_gNB *gNB;
  gNB_L1_proc_t *proc;
  gNB_L1_rxtx_proc_t *L1_proc,*L1_proc_tx;
  LOG_I(PHY,"%s(inst:%d) RC.nb_nr_CC[inst]:%d \n",__FUNCTION__,inst,RC.nb_nr_CC[inst]);

  for (int CC_id=0; CC_id<RC.nb_nr_CC[inst]; CC_id++) {
    gNB = RC.gNB[inst][CC_id];
    proc = &gNB->proc;
    L1_proc                        = &proc->L1_proc;
    L1_proc_tx                     = &proc->L1_proc_tx;
    L1_proc->instance_cnt          = -1;
    L1_proc_tx->instance_cnt       = -1;
    L1_proc->instance_cnt_RUs      = 0;
    L1_proc_tx->instance_cnt_RUs   = 0;
    proc->instance_cnt_prach       = -1;
    proc->instance_cnt_asynch_rxtx = -1;
    proc->CC_id                    = CC_id;
    proc->first_rx                 =1;
    proc->first_tx                 =1;
    proc->RU_mask                  =0;
    proc->RU_mask_tx               = (1<<gNB->num_RU)-1;
    proc->RU_mask_prach            =0;
    pthread_mutex_init( &gNB->UL_INFO_mutex, NULL);
    pthread_mutex_init( &L1_proc->mutex, NULL);
    pthread_mutex_init( &L1_proc_tx->mutex, NULL);
    pthread_cond_init( &L1_proc->cond, NULL);
    pthread_cond_init( &L1_proc_tx->cond, NULL);
    pthread_mutex_init( &proc->mutex_prach, NULL);
    pthread_mutex_init( &proc->mutex_asynch_rxtx, NULL);
    pthread_mutex_init( &proc->mutex_RU,NULL);
    pthread_mutex_init( &proc->mutex_RU_tx,NULL);
    pthread_mutex_init( &proc->mutex_RU_PRACH,NULL);
    pthread_cond_init( &proc->cond_prach, NULL);
    pthread_cond_init( &proc->cond_asynch_rxtx, NULL);
    LOG_I(PHY,"gNB->single_thread_flag:%d\n", gNB->single_thread_flag);
    //pthread_create( &proc->pthread_prach, attr_prach, gNB_thread_prach, gNB );
    AssertFatal(proc->instance_cnt_prach == -1,"instance_cnt_prach = %d\n",proc->instance_cnt_prach);
  }

  /* setup PHY proc TX sync mechanism */
  pthread_mutex_init(&sync_phy_proc.mutex_phy_proc_tx, NULL);
  pthread_cond_init(&sync_phy_proc.cond_phy_proc_tx, NULL);
  sync_phy_proc.phy_proc_CC_id = 0;
}

static inline int rxtx(PHY_VARS_gNB *gNB,gNB_L1_rxtx_proc_t *proc, char *thread_name) {
  // *******************************************************************
  /// NR disabling
  // ****************************************
  // Common RX procedures subframe n
  /*
    // if this is IF5 or 3GPP_gNB
    if (gNB && gNB->RU_list && gNB->RU_list[0] && gNB->RU_list[0]->function < NGFI_RAU_IF4p5) {
      wakeup_prach_gNB(gNB,NULL,proc->frame_rx,proc->slot_rx);
    }

    // UE-specific RX processing for subframe n
    if (nfapi_mode == 0 || nfapi_mode == 1) {
      phy_procedures_gNB_uespec_RX(gNB, proc, no_relay );
    }
  */
  pthread_mutex_lock(&gNB->UL_INFO_mutex);
  gNB->UL_INFO.frame     = proc->frame_rx;
  gNB->UL_INFO.slot      = proc->slot_rx;
  gNB->UL_INFO.module_id = gNB->Mod_id;
  gNB->UL_INFO.CC_id     = gNB->CC_id;
  gNB->if_inst->NR_UL_indication(&gNB->UL_INFO);
  pthread_mutex_unlock(&gNB->UL_INFO_mutex);
  /// end
  // *****************************************
  // TX processing for subframe n+sl_ahead
  // run PHY TX procedures the one after the other for all CCs to avoid race conditions
  // (may be relaxed in the future for performance reasons)
  // *****************************************
  //if (wait_CCs(proc)<0) return(-1);
  uint64_t a=rdtsc();
  phy_procedures_gNB_TX(gNB, proc, 1);
  uint64_t b=rdtsc() -a;
  if (b/3500.0 > 100 )
    printf("processin: %d, %ld \n", proc->slot_rx, b/3500);
  return(0);
}


void gNB_top(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx, char *string, struct RU_t_s *ru) {
  gNB_L1_proc_t *proc           = &gNB->proc;
  gNB_L1_rxtx_proc_t *L1_proc = &proc->L1_proc;
  NR_DL_FRAME_PARMS *fp = ru->nr_frame_parms;
  RU_proc_t *ru_proc=&ru->proc;
  proc->frame_rx    = frame_rx;
  proc->slot_rx = slot_rx;

  if (!oai_exit) {
    L1_proc->timestamp_tx = ru_proc->timestamp_rx + (sl_ahead*fp->samples_per_slot);
    L1_proc->frame_rx     = ru_proc->frame_rx;
    L1_proc->slot_rx      = ru_proc->tti_rx;
    L1_proc->frame_tx     = (L1_proc->slot_rx > (fp->slots_per_frame-1-sl_ahead)) ? (L1_proc->frame_rx+1)&1023 : L1_proc->frame_rx;
    L1_proc->slot_tx      = (L1_proc->slot_rx + sl_ahead)%fp->slots_per_frame;

    if (rxtx(gNB,L1_proc,string) < 0)
      LOG_E(PHY,"gNB %d CC_id %d failed during execution\n",gNB->Mod_id,gNB->CC_id);

    ru_proc->timestamp_tx = L1_proc->timestamp_tx;
    ru_proc->tti_tx       = L1_proc->slot_tx;
    ru_proc->frame_tx     = L1_proc->frame_tx;
  }
}

/// eNB kept in function name for nffapi calls, TO FIX
void init_eNB_afterRU(void) {
  PHY_VARS_gNB *gNB;
  LOG_I(PHY,"%s() RC.nb_nr_inst:%d\n", __FUNCTION__, RC.nb_nr_inst);

  for (int inst=0; inst<RC.nb_nr_inst; inst++) {
    LOG_I(PHY,"RC.nb_nr_CC[inst]:%d\n", RC.nb_nr_CC[inst]);

    for (int CC_id=0; CC_id<RC.nb_nr_CC[inst]; CC_id++) {
      LOG_I(PHY,"RC.nb_nr_CC[inst:%d][CC_id:%d]:%p\n", inst, CC_id, RC.gNB[inst][CC_id]);
      gNB                                  =  RC.gNB[inst][CC_id];
      phy_init_nr_gNB(gNB,0,0);

      // map antennas and PRACH signals to gNB RX
      if (0) AssertFatal(gNB->num_RU>0,"Number of RU attached to gNB %d is zero\n",gNB->Mod_id);

      LOG_I(PHY,"Mapping RX ports from %d RUs to gNB %d\n",gNB->num_RU,gNB->Mod_id);
      //LOG_I(PHY,"Overwriting gNB->prach_vars.rxsigF[0]:%p\n", gNB->prach_vars.rxsigF[0]);
      gNB->prach_vars.rxsigF[0] = (int16_t **)malloc16(64*sizeof(int16_t *));
      LOG_I(PHY,"gNB->num_RU:%d\n", gNB->num_RU);

      for (int ru_id=0,aa=0; ru_id<gNB->num_RU; ru_id++) {
        AssertFatal(gNB->RU_list[ru_id]->common.rxdataF!=NULL,
                    "RU %d : common.rxdataF is NULL\n",
                    gNB->RU_list[ru_id]->idx);
        AssertFatal(gNB->RU_list[ru_id]->prach_rxsigF!=NULL,
                    "RU %d : prach_rxsigF is NULL\n",
                    gNB->RU_list[ru_id]->idx);

        for (int i=0; i<gNB->RU_list[ru_id]->nb_rx; aa++,i++) {
          LOG_I(PHY,"Attaching RU %d antenna %d to gNB antenna %d\n",gNB->RU_list[ru_id]->idx,i,aa);
          gNB->prach_vars.rxsigF[0][aa]    =  gNB->RU_list[ru_id]->prach_rxsigF[i];
          gNB->common_vars.rxdataF[aa]     =  gNB->RU_list[ru_id]->common.rxdataF[i];
        }
      }

      /* TODO: review this code, there is something wrong.
       * In monolithic mode, we come here with nb_antennas_rx == 0
       * (not tested in other modes).
       */
      //init_precoding_weights(RC.gNB[inst][CC_id]);
    }

    init_gNB_proc(inst);
  }

  for (int ru_id=0; ru_id<RC.nb_RU; ru_id++) {
    AssertFatal(RC.ru[ru_id]!=NULL,"ru_id %d is null\n",ru_id);
    RC.ru[ru_id]->nr_wakeup_rxtx         = NULL;
    //    RC.ru[ru_id]->wakeup_prach_eNB    = wakeup_prach_gNB;
    RC.ru[ru_id]->gNB_top             = gNB_top;
  }
}

void set_function_spec_param(RU_t *ru) {
  ru->do_prach             = 0;                       // no prach processing in RU
  ru->feprx                = fep_full ;
  ru->feptx_ofdm           = nr_feptx_ofdm ;
  ru->feptx_prec           = feptx_prec;              // this is fep with idft and precoding
  ru->fh_north_in          = NULL;                    // no incoming fronthaul from north
  ru->fh_north_out         = NULL;                    // no outgoing fronthaul to north
  ru->start_if             = NULL;                    // no if interface
  ru->rfdevice.host_type   = RAU_HOST;
  ru->fh_south_in            = rx_rf;                               // local synchronous RF RX
  ru->fh_south_out           = tx_rf;                               // local synchronous RF TX
  ru->start_rf               = start_rf;                            // need to start the local RF interface
  ru->stop_rf                = stop_rf;
}

static void *ru_thread_prach( void *param ) {
  static int ru_thread_prach_status;
  RU_t *ru        = (RU_t *)param;
  RU_proc_t *proc = (RU_proc_t *)&ru->proc;
  // set default return value
  ru_thread_prach_status = 0;

  while (RC.ru_mask>0) {
    usleep(1e6);
    LOG_I(PHY,"%s() RACH waiting for RU to be configured\n", __FUNCTION__);
  }

  LOG_I(PHY,"%s() RU configured - RACH processing thread running\n", __FUNCTION__);

  while (!oai_exit) {
    if (oai_exit) break;

    if (wait_on_condition(&proc->mutex_prach,&proc->cond_prach,&proc->instance_cnt_prach,"ru_prach_thread") < 0) break;

    /*if (ru->gNB_list[0]){
      prach_procedures(
        ru->gNB_list[0]
    #if (RRC_VERSION >= MAKE_VERSION(14, 0, 0))
        ,0
    #endif
        );
    }
    else {
       rx_prach(NULL,
            ru,
          NULL,
                NULL,
                NULL,
                proc->frame_prach,
                0
    #if (RRC_VERSION >= MAKE_VERSION(14, 0, 0))
          ,0
    #endif
          );
    }
    VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME( VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_RU_PRACH_RX, 0 ); */
    if (release_thread(&proc->mutex_prach,&proc->instance_cnt_prach,"ru_prach_thread") < 0) break;
  }

  LOG_I(PHY, "Exiting RU thread PRACH\n");
  ru_thread_prach_status = 0;
  return &ru_thread_prach_status;
}

void fill_rf_config(RU_t *ru, char *rf_config_file) {
  int i;
  NR_DL_FRAME_PARMS *fp   = ru->nr_frame_parms;
  nfapi_nr_config_request_t *gNB_config = &ru->gNB_list[0]->gNB_config; //tmp index
  openair0_config_t *cfg   = &ru->openair0_cfg;
  int N_RB = gNB_config->rf_config.dl_carrier_bandwidth.value;
  int mu = gNB_config->subframe_config.numerology_index_mu.value;

  if (mu == NR_MU_0) { //or if LTE
    if(N_RB == 100) {
      if (fp->threequarter_fs) {
        cfg->sample_rate=23.04e6;
        cfg->samples_per_frame = 230400;
        cfg->tx_bw = 10e6;
        cfg->rx_bw = 10e6;
      } else {
        cfg->sample_rate=30.72e6;
        cfg->samples_per_frame = 307200;
        cfg->tx_bw = 10e6;
        cfg->rx_bw = 10e6;
      }
    } else if(N_RB == 50) {
      cfg->sample_rate=15.36e6;
      cfg->samples_per_frame = 153600;
      cfg->tx_bw = 5e6;
      cfg->rx_bw = 5e6;
    } else if (N_RB == 25) {
      cfg->sample_rate=7.68e6;
      cfg->samples_per_frame = 76800;
      cfg->tx_bw = 2.5e6;
      cfg->rx_bw = 2.5e6;
    } else if (N_RB == 6) {
      cfg->sample_rate=1.92e6;
      cfg->samples_per_frame = 19200;
      cfg->tx_bw = 1.5e6;
      cfg->rx_bw = 1.5e6;
    } else AssertFatal(1==0,"Unknown N_RB %d\n",N_RB);
  } else if (mu == NR_MU_1) {
    if(N_RB == 217) {
      if (fp->threequarter_fs) {
        cfg->sample_rate=92.16e6;
        cfg->samples_per_frame = 921600;
        cfg->tx_bw = 40e6;
        cfg->rx_bw = 40e6;
      } else {
        cfg->sample_rate=122.88e6;
        cfg->samples_per_frame = 1228800;
        cfg->tx_bw = 40e6;
        cfg->rx_bw = 40e6;
      }
    } else if(N_RB == 106) {
      cfg->sample_rate=61.44e6;
      cfg->samples_per_frame = 614400;
      cfg->tx_bw = 20e6;
      cfg->rx_bw = 20e6;
    } else {
      AssertFatal(0==1,"N_RB %d not yet supported for numerology %d\n",N_RB,mu);
    }
  } else {
    AssertFatal(0 == 1,"Numerology %d not supported for the moment\n",mu);
  }

  if (gNB_config->subframe_config.duplex_mode.value==TDD)
    cfg->duplex_mode = duplex_mode_TDD;
  else //FDD
    cfg->duplex_mode = duplex_mode_FDD;

  cfg->Mod_id = 0;
  cfg->num_rb_dl=N_RB;
  cfg->tx_num_channels=ru->nb_tx;
  cfg->rx_num_channels=ru->nb_rx;

  for (i=0; i<ru->nb_tx; i++) {
    cfg->tx_freq[i] = (double)fp->dl_CarrierFreq;
    cfg->rx_freq[i] = (double)fp->ul_CarrierFreq;
    cfg->tx_gain[i] = ru->att_tx;
    cfg->rx_gain[i] = ru->max_rxgain-ru->att_rx;
    cfg->configFilename = rf_config_file;
    printf("channel %d, Setting tx_gain offset %f, rx_gain offset %f, tx_freq %f, rx_freq %f\n",
           i, cfg->tx_gain[i],
           cfg->rx_gain[i],
           cfg->tx_freq[i],
           cfg->rx_freq[i]);
  }
}

void configure_rru(int idx,
                   void *arg) {
  RRU_config_t *config = (RRU_config_t *)arg;
  RU_t         *ru         = RC.ru[idx];
  nfapi_nr_config_request_t *gNB_config = &ru->gNB_list[0]->gNB_config;
  ru->nr_frame_parms->eutra_band                                               = config->band_list[0];
  ru->nr_frame_parms->dl_CarrierFreq                                           = config->tx_freq[0];
  ru->nr_frame_parms->ul_CarrierFreq                                           = config->rx_freq[0];

  if (ru->nr_frame_parms->dl_CarrierFreq == ru->nr_frame_parms->ul_CarrierFreq) {
    gNB_config->subframe_config.duplex_mode.value                         = TDD;
    //ru->nr_frame_parms->tdd_config                                            = config->tdd_config[0];
    //ru->nr_frame_parms->tdd_config_S                                          = config->tdd_config_S[0];
  } else
    gNB_config->subframe_config.duplex_mode.value                            = FDD;

  ru->att_tx                                                               = config->att_tx[0];
  ru->att_rx                                                               = config->att_rx[0];
  gNB_config->rf_config.dl_carrier_bandwidth.value                         = config->N_RB_DL[0];
  gNB_config->rf_config.ul_carrier_bandwidth.value                         = config->N_RB_UL[0];
  ru->nr_frame_parms->threequarter_fs                                       = config->threequarter_fs[0];

  //ru->nr_frame_parms->pdsch_config_common.referenceSignalPower                 = ru->max_pdschReferenceSignalPower-config->att_tx[0];
  if (ru->function==NGFI_RRU_IF4p5) {
    ru->nr_frame_parms->att_rx = ru->att_rx;
    ru->nr_frame_parms->att_tx = ru->att_tx;
    /*
        LOG_I(PHY,"Setting ru->function to NGFI_RRU_IF4p5, prach_FrequOffset %d, prach_ConfigIndex %d, att (%d,%d)\n",
        config->prach_FreqOffset[0],config->prach_ConfigIndex[0],ru->att_tx,ru->att_rx);
        ru->nr_frame_parms->prach_config_common.prach_ConfigInfo.prach_FreqOffset  = config->prach_FreqOffset[0];
        ru->nr_frame_parms->prach_config_common.prach_ConfigInfo.prach_ConfigIndex = config->prach_ConfigIndex[0]; */
  }

  fill_rf_config(ru,ru->rf_config_file);
  nr_init_frame_parms(&ru->gNB_list[0]->gNB_config, ru->nr_frame_parms);
  nr_phy_init_RU(ru);
}

int connect_rau(RU_t *ru) {
  RRU_CONFIG_msg_t   rru_config_msg;
  ssize_t      msg_len;
  int                tick_received          = 0;
  int                configuration_received = 0;
  RRU_capabilities_t *cap;
  int                i;
  int                len;

  // wait for RAU_tick
  while (tick_received == 0) {
    msg_len  = sizeof(RRU_CONFIG_msg_t)-MAX_RRU_CONFIG_SIZE;

    if ((len = ru->ifdevice.trx_ctlrecv_func(&ru->ifdevice,
               &rru_config_msg,
               msg_len))<0) {
      LOG_I(PHY,"Waiting for RAU\n");
    } else {
      if (rru_config_msg.type == RAU_tick) {
        LOG_I(PHY,"Tick received from RAU\n");
        tick_received = 1;
      } else LOG_E(PHY,"Received erroneous message (%d)from RAU, expected RAU_tick\n",rru_config_msg.type);
    }
  }

  // send capabilities
  rru_config_msg.type = RRU_capabilities;
  rru_config_msg.len  = sizeof(RRU_CONFIG_msg_t)-MAX_RRU_CONFIG_SIZE+sizeof(RRU_capabilities_t);
  cap                 = (RRU_capabilities_t *)&rru_config_msg.msg[0];
  LOG_I(PHY,"Sending Capabilities (len %d, num_bands %d,max_pdschReferenceSignalPower %d, max_rxgain %d, nb_tx %d, nb_rx %d)\n",
        (int)rru_config_msg.len,ru->num_bands,ru->max_pdschReferenceSignalPower,ru->max_rxgain,ru->nb_tx,ru->nb_rx);

  switch (ru->function) {
    case NGFI_RRU_IF4p5:
      cap->FH_fmt                                   = OAI_IF4p5_only;
      break;

    case NGFI_RRU_IF5:
      cap->FH_fmt                                   = OAI_IF5_only;
      break;

    case MBP_RRU_IF5:
      cap->FH_fmt                                   = MBP_IF5;
      break;

    default:
      AssertFatal(1==0,"RU_function is unknown %d\n",RC.ru[0]->function);
      break;
  }

  cap->num_bands                                  = ru->num_bands;

  for (i=0; i<ru->num_bands; i++) {
    LOG_I(PHY,"Band %d: nb_rx %d nb_tx %d pdschReferenceSignalPower %d rxgain %d\n",
          ru->band[i],ru->nb_rx,ru->nb_tx,ru->max_pdschReferenceSignalPower,ru->max_rxgain);
    cap->band_list[i]                             = ru->band[i];
    cap->nb_rx[i]                                 = ru->nb_rx;
    cap->nb_tx[i]                                 = ru->nb_tx;
    cap->max_pdschReferenceSignalPower[i]         = ru->max_pdschReferenceSignalPower;
    cap->max_rxgain[i]                            = ru->max_rxgain;
  }

  AssertFatal((ru->ifdevice.trx_ctlsend_func(&ru->ifdevice,&rru_config_msg,rru_config_msg.len)!=-1),
              "RU %d failed send capabilities to RAU\n",ru->idx);
  // wait for configuration
  rru_config_msg.len  = sizeof(RRU_CONFIG_msg_t)-MAX_RRU_CONFIG_SIZE+sizeof(RRU_config_t);

  while (configuration_received == 0) {
    if ((len = ru->ifdevice.trx_ctlrecv_func(&ru->ifdevice,
               &rru_config_msg,
               rru_config_msg.len))<0) {
      LOG_I(PHY,"Waiting for configuration from RAU\n");
    } else {
      LOG_I(PHY,"Configuration received from RAU  (num_bands %d,band0 %d,txfreq %u,rxfreq %u,att_tx %d,att_rx %d,N_RB_DL %d,N_RB_UL %d,3/4FS %d, prach_FO %d, prach_CI %d)\n",
            ((RRU_config_t *)&rru_config_msg.msg[0])->num_bands,
            ((RRU_config_t *)&rru_config_msg.msg[0])->band_list[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->tx_freq[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->rx_freq[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->att_tx[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->att_rx[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->N_RB_DL[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->N_RB_UL[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->threequarter_fs[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->prach_FreqOffset[0],
            ((RRU_config_t *)&rru_config_msg.msg[0])->prach_ConfigIndex[0]);
      configure_rru(ru->idx,
                    (void *)&rru_config_msg.msg[0]);
      configuration_received = 1;
    }
  }

  return 0;
}

bool setup_RU_buffers(RU_t *ru) {
  int i,j;
  int card,ant;
  //uint16_t N_TA_offset = 0;
  NR_DL_FRAME_PARMS *frame_parms;
  //nfapi_nr_config_request_t *gNB_config = ru->gNB_list[0]->gNB_config; //tmp index

  if (ru) {
    frame_parms = ru->nr_frame_parms;
    printf("setup_RU_buffers: frame_parms = %p\n",frame_parms);
  } else {
    printf("RU[%d] not initialized\n", ru->idx);
    return(false);
  }

  /*  if (frame_parms->frame_type == TDD) {
      if      (frame_parms->N_RB_DL == 100) ru->N_TA_offset = 624;
      else if (frame_parms->N_RB_DL == 50)  ru->N_TA_offset = 624/2;
      else if (frame_parms->N_RB_DL == 25)  ru->N_TA_offset = 624/4;
    } */
  if (ru->openair0_cfg.mmapped_dma == 1) {
    // replace RX signal buffers with mmaped HW versions
    for (i=0; i<ru->nb_rx; i++) {
      card = i/4;
      ant = i%4;
      printf("Mapping RU id %d, rx_ant %d, on card %d, chain %d\n",ru->idx,i,ru->rf_map.card+card, ru->rf_map.chain+ant);
      free(ru->common.rxdata[i]);
      ru->common.rxdata[i] = ru->openair0_cfg.rxbase[ru->rf_map.chain+ant];
      printf("rxdata[%d] @ %p\n",i,ru->common.rxdata[i]);

      for (j=0; j<16; j++) {
        printf("rxbuffer %d: %x\n",j,ru->common.rxdata[i][j]);
        ru->common.rxdata[i][j] = 16-j;
      }
    }

    for (i=0; i<ru->nb_tx; i++) {
      card = i/4;
      ant = i%4;
      printf("Mapping RU id %d, tx_ant %d, on card %d, chain %d\n",ru->idx,i,ru->rf_map.card+card, ru->rf_map.chain+ant);
      free(ru->common.txdata[i]);
      ru->common.txdata[i] = ru->openair0_cfg.txbase[ru->rf_map.chain+ant];
      printf("txdata[%d] @ %p\n",i,ru->common.txdata[i]);

      for (j=0; j<16; j++) {
        printf("txbuffer %d: %x\n",j,ru->common.txdata[i][j]);
        ru->common.txdata[i][j] = 16-j;
      }
    }
  } else { // not memory-mapped DMA
    //nothing to do, everything already allocated in lte_init
  }

  return(true);
}

static void modulateSend (void* arg) {
  RU_t *ru=*(RU_t**)arg;
  if(ru->num_eNB==0) {
    // do TX front-end processing if needed (precoding and/or IDFTs)
    if (ru->feptx_prec)
      ru->feptx_prec(ru);
    
    // do OFDM if needed
    if ((ru->fh_north_asynch_in == NULL) && (ru->feptx_ofdm))
      ru->feptx_ofdm(ru);
    
    // do outgoing fronthaul (south) if needed
    if ((ru->fh_north_asynch_in == NULL) && (ru->fh_south_out))
      ru->fh_south_out(ru);
    
    if (ru->fh_north_out)
      ru->fh_north_out(ru);
  }
}

static void *ru_thread( void *param ) {
  RU_t               *ru      = (RU_t *)param;
  RU_proc_t          *proc    = &ru->proc;
  NR_DL_FRAME_PARMS *fp      = ru->nr_frame_parms;
  int                slot = fp->slots_per_frame-1;
  int                frame    =1023;
  char               threadname[40];
  // set default return value
  // set default return value
  sprintf(threadname,"ru_thread %d",ru->idx);
  LOG_I(PHY,"Starting RU %d (%s,%s),\n",ru->idx,NB_functions[ru->function],NB_timing[ru->if_timing]);


  if (ru->if_south == LOCAL_RF) { // configure RF parameters only
    fill_rf_config(ru,ru->rf_config_file);
    nr_init_frame_parms(&ru->gNB_list[0]->gNB_config, fp);
    nr_dump_frame_parms(fp);
    nr_phy_init_RU(ru);
    AssertFatal(openair0_device_load(&ru->rfdevice,&ru->openair0_cfg)==0,"Cannot connect to local radio\n");
  }

  AssertFatal(setup_RU_buffers(ru),"Exiting, cannot initialize RU Buffers\n");

  // Start RF device if any
  if (ru->start_rf) {
    if (ru->start_rf(ru) != 0)
      LOG_E(HW,"Could not start the RF device\n");
    else LOG_I(PHY,"RU %d rf device ready\n",ru->idx);
  } else LOG_I(PHY,"RU %d no rf device\n",ru->idx);

  // The RU initializationis in the middle of gNB one
  // so , we signal and wait the gNB finishes it's init
  // before running the main loop
  static notifiedFIFO_t ruThreadFIFO;
  initNotifiedFIFO(&ruThreadFIFO);
  LOG_I(PHY, "Signaling main thread that RU %d is ready\n",ru->idx);
  notifiedFIFO_elt_t *msg=newNotifiedFIFO_elt(0,doneInitRU,&ruThreadFIFO,NULL);
  pushNotifiedFIFO(&mainThreadFIFO,msg);
  LOG_I(PHY, "wait main thread that RU %d is ready\n",ru->idx);
  delNotifiedFIFO_elt(pullNotifiedFIFO(&ruThreadFIFO));

  // Start IF device if any
  if (ru->start_if) {
    LOG_I(PHY,"Starting IF interface for RU %d\n",ru->idx);
    AssertFatal(ru->start_if(ru,NULL) == 0, "Could not start the IF device\n");
    AssertFatal(connect_rau(ru)==0,"Cannot connect to remote radio\n");
  }
  
  // This is a forever while loop, it loops over subframes which are scheduled by incoming samples from HW devices
  initRefTimes(rx);
  initRefTimes(frx);
  initRefTimes(mainProc);

  while (!oai_exit) {
    // these are local subframe/frame counters to check that we are in synch with the fronthaul timing.
    // They are set on the first rx/tx in the underly FH routines.
    if (slot==(fp->slots_per_frame-1)) {
      slot=0;
      frame++;
      frame&=1023;
    } else {
      slot++;
    }

    pickTime(beg);
    // synchronization on input FH interface, acquire signals/data and block
    AssertFatal(ru->fh_south_in, "No fronthaul interface at south port");
    ru->fh_south_in(ru,&frame,&slot);
    LOG_D(PHY,"AFTER fh_south_in - SFN/SL:%d%d RU->proc[RX:%d.%d TX:%d.%d] RC.gNB[0][0]:[RX:%d%d TX(SFN):%d]\n",
          frame,slot,
          proc->frame_rx,proc->tti_rx,
          proc->frame_tx,proc->tti_tx,
          RC.gNB[0][0]->proc.frame_rx,RC.gNB[0][0]->proc.slot_rx,
          RC.gNB[0][0]->proc.frame_tx);
    /*
      LOG_D(PHY,"RU thread (do_prach %d, is_prach_subframe %d), received frame %d, subframe %d\n",
      ru->do_prach,
      is_prach_subframe(fp, proc->frame_rx, proc->tti_rx),
      proc->frame_rx,proc->tti_rx);

      if ((ru->do_prach>0) && (is_prach_subframe(fp, proc->frame_rx, proc->tti_rx)==1)) {
    wakeup_prach_ru(ru);
        }*/

    // adjust for timing offset between RU
    //printf("~~~~~~~~~~~~~~~~~~~~~~~~~~%d.%d in ru_thread is in process\n", proc->frame_rx, proc->tti_rx);
    if (ru->idx!=0)
      proc->frame_tx = (proc->frame_tx+proc->frame_offset)&1023;

    updateTimes(beg, &rx, 1000, "trx_read");
    pickTime(beg2);

    if (rx.iterations%1000 == 0 ) printf("%d \n", rx.iterations);

    // do RX front-end processing (frequency-shift, dft) if needed
    if (ru->feprx)
      ru->feprx(ru);

    // At this point, all information for subframe has been received on FH interface
    updateTimes(beg2, &frx, 1000, "feprx");
    pickTime(beg3);

    // wakeup all gNB processes waiting for this RU
    for (int gnb=0; gnb < ru->num_gNB; gnb++)
      ru->gNB_top(ru->gNB_list[gnb],ru->proc.frame_rx,ru->proc.tti_rx,"not def",ru);

    updateTimes(beg3, &mainProc, 1000, "rxtx");
    
    notifiedFIFO_elt_t *txWork=newNotifiedFIFO_elt(0,0,NULL,modulateSend);
    void **tmp= (void**)NotifiedFifoData(txWork);
    *tmp=(void*)ru;
    pushTpool(Tpool,txWork);
  }

  notifiedFIFO_elt_t *msg2=newNotifiedFIFO_elt(sizeof(ru),AbortRU,NULL,modulateSend);
  pushNotifiedFIFO(&mainThreadFIFO,msg2);
  return NULL;
}

void init_RU_proc(RU_t *ru) {
  int i=0;
  RU_proc_t *proc;
  proc = &ru->proc;
  memset((void *)proc,0,sizeof(RU_proc_t));
  proc->ru = ru;
  proc->instance_cnt_prach       = -1;
  proc->instance_cnt_synch       = -1;     ;
  proc->instance_cnt_FH          = -1;
  proc->instance_cnt_FH1         = -1;
  proc->instance_cnt_gNBs        = -1;
  proc->instance_cnt_asynch_rxtx = -1;
  proc->instance_cnt_emulateRF   = -1;
  proc->first_rx                 = 1;
  proc->first_tx                 = 1;
  proc->frame_offset             = 0;
  proc->num_slaves               = 0;
  proc->frame_tx_unwrap          = 0;

  for (i=0; i<10; i++) proc->symbol_mask[i]=0;

  pthread_mutex_init( &proc->mutex_prach, NULL);
  pthread_mutex_init( &proc->mutex_asynch_rxtx, NULL);
  pthread_mutex_init( &proc->mutex_synch,NULL);
  pthread_mutex_init( &proc->mutex_FH,NULL);
  pthread_mutex_init( &proc->mutex_FH1,NULL);
  pthread_mutex_init( &proc->mutex_emulateRF,NULL);
  pthread_mutex_init( &proc->mutex_gNBs, NULL);
  pthread_cond_init( &proc->cond_prach, NULL);
  pthread_cond_init( &proc->cond_FH, NULL);
  pthread_cond_init( &proc->cond_FH1, NULL);
  pthread_cond_init( &proc->cond_emulateRF, NULL);
  pthread_cond_init( &proc->cond_asynch_rxtx, NULL);
  pthread_cond_init( &proc->cond_synch,NULL);
  pthread_cond_init( &proc->cond_gNBs, NULL);
  threadCreate( &proc->pthread_FH, ru_thread, (void *)ru, "thread_FH", -1, OAI_PRIORITY_RT_MAX );
  threadCreate( &proc->pthread_prach, ru_thread_prach, (void *)ru,"RACH", -1, OAI_PRIORITY_RT );
}

int restart_L1L2(module_id_t gnb_id) {
  return 0;
}

void exit_function(const char *file, const char *function, const int line, const char *s) {
  int ru_id;

  if (s != NULL) {
    printf("%s:%d %s() Exiting OAI softmodem: %s\n",file,line, function, s);
  }

  oai_exit = 1;

  if (RC.ru == NULL)
    exit(-1); // likely init not completed, prevent crash or hang, exit now...

  for (ru_id=0; ru_id<RC.nb_RU; ru_id++) {
    if (RC.ru[ru_id] && RC.ru[ru_id]->rfdevice.trx_end_func) {
      RC.ru[ru_id]->rfdevice.trx_end_func(&RC.ru[ru_id]->rfdevice);
      RC.ru[ru_id]->rfdevice.trx_end_func = NULL;
    }

    if (RC.ru[ru_id] && RC.ru[ru_id]->ifdevice.trx_end_func) {
      RC.ru[ru_id]->ifdevice.trx_end_func(&RC.ru[ru_id]->ifdevice);
      RC.ru[ru_id]->ifdevice.trx_end_func = NULL;
    }
  }

  sleep(1); //allow lte-softmodem threads to exit first
#if defined(ENABLE_ITTI)
  itti_terminate_tasks (TASK_UNKNOWN);
#endif
  exit(1);
}

void signal_handler_itti(int sig) {
  // Call exit function
  char msg[256];
  memset(msg, 0, 256);
  sprintf(msg, "caught signal %s\n", strsignal(sig));
  exit_function(__FILE__, __FUNCTION__, __LINE__, msg);
}

static void get_options(void) {
  int tddflag, nonbiotflag;
  uint32_t online_log_messages;
  uint32_t glog_level, glog_verbosity;
  uint32_t start_telnetsrv;
  paramdef_t cmdline_params[] =CMDLINE_PARAMS_DESC ;
  paramdef_t cmdline_logparams[] =CMDLINE_LOGPARAMS_DESC ;
  config_process_cmdline( cmdline_params,sizeof(cmdline_params)/sizeof(paramdef_t),NULL);

  if (strlen(in_path) > 0) {
    opt_type = OPT_PCAP;
    opt_enabled=1;
    printf("Enabling OPT for PCAP  with the following file %s \n",in_path);
  }

  if (strlen(in_ip) > 0) {
    opt_enabled=1;
    opt_type = OPT_WIRESHARK;
    printf("Enabling OPT for wireshark for local interface");
  }

  config_process_cmdline( cmdline_logparams,sizeof(cmdline_logparams)/sizeof(paramdef_t),NULL);

  if(config_isparamset(cmdline_logparams,CMDLINE_ONLINELOG_IDX))
    set_glog_onlinelog(online_log_messages);

  if(config_isparamset(cmdline_logparams,CMDLINE_GLOGLEVEL_IDX))
    set_glog(glog_level);

  if (start_telnetsrv)
    load_module_shlib("telnetsrv",NULL,0,NULL);

  if ( !(CONFIG_ISFLAGSET(CONFIG_ABORT)) ) {
    memset((void *)&RC,0,sizeof(RC));
    /* Read RC configuration file */
    NRRCConfig();
    NB_gNB_INST = RC.nb_nr_inst;
    NB_RU   = RC.nb_RU;
    printf("Configuration: nb_rrc_inst %d, nb_nr_L1_inst %d, nb_ru %d\n",NB_gNB_INST,RC.nb_nr_L1_inst,NB_RU);
  }
}

void set_default_frame_parms(nfapi_nr_config_request_t *config[MAX_NUM_CCs], NR_DL_FRAME_PARMS *frame_parms[MAX_NUM_CCs]) {
  int CC_id;

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    frame_parms[CC_id] = (NR_DL_FRAME_PARMS *) malloc(sizeof(NR_DL_FRAME_PARMS));
    config[CC_id] = (nfapi_nr_config_request_t *) malloc(sizeof(nfapi_nr_config_request_t));
    config[CC_id]->subframe_config.numerology_index_mu.value =1;
    config[CC_id]->subframe_config.duplex_mode.value = 1; //FDD
    config[CC_id]->subframe_config.dl_cyclic_prefix_type.value = 0; //NORMAL
    config[CC_id]->rf_config.dl_carrier_bandwidth.value = 106;
    config[CC_id]->rf_config.ul_carrier_bandwidth.value = 106;
    config[CC_id]->sch_config.physical_cell_id.value = 0;
    ///dl frequency to be filled in
    /*  //Set some default values that may be overwritten while reading options
        frame_parms[CC_id]->frame_type          = FDD;
        frame_parms[CC_id]->tdd_config          = 3;
        frame_parms[CC_id]->tdd_config_S        = 0;
        frame_parms[CC_id]->N_RB_DL             = 100;
        frame_parms[CC_id]->N_RB_UL             = 100;
        frame_parms[CC_id]->Ncp                 = NORMAL;
        frame_parms[CC_id]->Ncp_UL              = NORMAL;
        frame_parms[CC_id]->Nid_cell            = 0;
        frame_parms[CC_id]->num_MBSFN_config    = 0;
        frame_parms[CC_id]->nb_antenna_ports_eNB  = 1;
        frame_parms[CC_id]->nb_antennas_tx      = 1;
        frame_parms[CC_id]->nb_antennas_rx      = 1;

        frame_parms[CC_id]->nushift             = 0;

        frame_parms[CC_id]->phich_config_common.phich_resource = oneSixth;
        frame_parms[CC_id]->phich_config_common.phich_duration = normal;
        // UL RS Config
        frame_parms[CC_id]->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift = 0;//n_DMRS1 set to 0
        frame_parms[CC_id]->pusch_config_common.ul_ReferenceSignalsPUSCH.groupHoppingEnabled = 0;
        frame_parms[CC_id]->pusch_config_common.ul_ReferenceSignalsPUSCH.sequenceHoppingEnabled = 0;
        frame_parms[CC_id]->pusch_config_common.ul_ReferenceSignalsPUSCH.groupAssignmentPUSCH = 0;

        frame_parms[CC_id]->prach_config_common.rootSequenceIndex=22;
        frame_parms[CC_id]->prach_config_common.prach_ConfigInfo.zeroCorrelationZoneConfig=1;
        frame_parms[CC_id]->prach_config_common.prach_ConfigInfo.prach_ConfigIndex=0;
        frame_parms[CC_id]->prach_config_common.prach_ConfigInfo.highSpeedFlag=0;
        frame_parms[CC_id]->prach_config_common.prach_ConfigInfo.prach_FreqOffset=0;

    //    downlink_frequency[CC_id][0] = 2680000000; // Use float to avoid issue with frequency over 2^31.
    //    downlink_frequency[CC_id][1] = downlink_frequency[CC_id][0];
    //    downlink_frequency[CC_id][2] = downlink_frequency[CC_id][0];
    //    downlink_frequency[CC_id][3] = downlink_frequency[CC_id][0];
        //printf("Downlink for CC_id %d frequency set to %u\n", CC_id, downlink_frequency[CC_id][0]);
        frame_parms[CC_id]->dl_CarrierFreq=downlink_frequency[CC_id][0];
    */
  }
}

void init_gNB(int single_thread_flag,int wait_for_sync) {
  PHY_VARS_gNB *gNB;

  if (RC.gNB == NULL) RC.gNB = (PHY_VARS_gNB ***) malloc(RC.nb_nr_L1_inst*sizeof(PHY_VARS_gNB **));

  for (int inst=0; inst<RC.nb_nr_L1_inst; inst++) {
    if (RC.gNB[inst] == NULL) RC.gNB[inst] = (PHY_VARS_gNB **) malloc(RC.nb_nr_CC[inst]*sizeof(PHY_VARS_gNB *));

    for (int CC_id=0; CC_id<RC.nb_nr_L1_CC[inst]; CC_id++) {
      if (RC.gNB[inst][CC_id] == NULL) RC.gNB[inst][CC_id] = (PHY_VARS_gNB *) malloc(sizeof(PHY_VARS_gNB));

      gNB                     = RC.gNB[inst][CC_id];
      gNB->abstraction_flag   = 0;
      gNB->single_thread_flag = single_thread_flag;
      /*nr_polar_init(&gNB->nrPolar_params,
                NR_POLAR_PBCH_MESSAGE_TYPE,
          NR_POLAR_PBCH_PAYLOAD_BITS,
          NR_POLAR_PBCH_AGGREGATION_LEVEL);*/
      LOG_I(PHY,"Initializing gNB %d CC_id %d single_thread_flag:%d\n",inst,CC_id,single_thread_flag);
      LOG_I(PHY,"Registering with MAC interface module\n");
      AssertFatal((gNB->if_inst         = NR_IF_Module_init(inst))!=NULL,"Cannot register interface");
      gNB->if_inst->NR_Schedule_response   = nr_schedule_response;
      gNB->if_inst->NR_PHY_config_req      = nr_phy_config_request;
      memset((void *)&gNB->UL_INFO,0,sizeof(gNB->UL_INFO));
      memset((void *)&gNB->Sched_INFO,0,sizeof(gNB->Sched_INFO));
      LOG_I(PHY,"Setting indication lists\n");
      gNB->UL_INFO.rx_ind.rx_indication_body.rx_pdu_list   = gNB->rx_pdu_list;
      gNB->UL_INFO.crc_ind.crc_indication_body.crc_pdu_list = gNB->crc_pdu_list;
      gNB->UL_INFO.sr_ind.sr_indication_body.sr_pdu_list = gNB->sr_pdu_list;
      gNB->UL_INFO.harq_ind.harq_indication_body.harq_pdu_list = gNB->harq_pdu_list;
      gNB->UL_INFO.cqi_ind.cqi_pdu_list = gNB->cqi_pdu_list;
      gNB->UL_INFO.cqi_ind.cqi_raw_pdu_list = gNB->cqi_raw_pdu_list;
      gNB->prach_energy_counter = 0;
    }
  }

  LOG_I(PHY,"[nr-softmodem.c] gNB structure allocated\n");
}

void RCconfig_RU(void) {
  int               j                             = 0;
  int               i                             = 0;
  paramdef_t RUParams[] = RUPARAMS_DESC;
  paramlist_def_t RUParamList = {CONFIG_STRING_RU_LIST,NULL,0};
  config_getlist( &RUParamList,RUParams,sizeof(RUParams)/sizeof(paramdef_t), NULL);

  if ( RUParamList.numelt > 0) {
    RC.ru = (RU_t **)malloc(RC.nb_RU*sizeof(RU_t *));
    RC.ru_mask=(1<<NB_RU) - 1;
    printf("Set RU mask to %lx\n",RC.ru_mask);

    for (j = 0; j < RC.nb_RU; j++) {
      RC.ru[j]                                    = (RU_t *)malloc(sizeof(RU_t));
      memset((void *)RC.ru[j],0,sizeof(RU_t));
      RC.ru[j]->idx                                 = j;
      RC.ru[j]->nr_frame_parms                      = (NR_DL_FRAME_PARMS *)malloc(sizeof(NR_DL_FRAME_PARMS));
      RC.ru[j]->frame_parms                      = (LTE_DL_FRAME_PARMS *)malloc(sizeof(LTE_DL_FRAME_PARMS));
      printf("Creating RC.ru[%d]:%p\n", j, RC.ru[j]);
      RC.ru[j]->if_timing                           = synch_to_ext_device;

      if (RC.nb_nr_L1_inst >0)
        RC.ru[j]->num_gNB                           = RUParamList.paramarray[j][RU_ENB_LIST_IDX].numelt;
      else
        RC.ru[j]->num_gNB                           = 0;

      for (i=0; i<RC.ru[j]->num_gNB; i++) RC.ru[j]->gNB_list[i] = RC.gNB[RUParamList.paramarray[j][RU_ENB_LIST_IDX].iptr[i]][0];

      if (config_isparamset(RUParamList.paramarray[j], RU_SDR_ADDRS)) {
        RC.ru[j]->openair0_cfg.sdr_addrs = strdup(*(RUParamList.paramarray[j][RU_SDR_ADDRS].strptr));
      }

      if (config_isparamset(RUParamList.paramarray[j], RU_SDR_CLK_SRC)) {
        if (strcmp(*(RUParamList.paramarray[j][RU_SDR_CLK_SRC].strptr), "internal") == 0) {
          RC.ru[j]->openair0_cfg.clock_source = internal;
          LOG_D(PHY, "RU clock source set as internal\n");
        } else if (strcmp(*(RUParamList.paramarray[j][RU_SDR_CLK_SRC].strptr), "external") == 0) {
          RC.ru[j]->openair0_cfg.clock_source = external;
          LOG_D(PHY, "RU clock source set as external\n");
        } else if (strcmp(*(RUParamList.paramarray[j][RU_SDR_CLK_SRC].strptr), "gpsdo") == 0) {
          RC.ru[j]->openair0_cfg.clock_source = gpsdo;
          LOG_D(PHY, "RU clock source set as gpsdo\n");
        } else {
          LOG_E(PHY, "Erroneous RU clock source \n");
        }
      }

      if (strcmp(*(RUParamList.paramarray[j][RU_LOCAL_RF_IDX].strptr), "yes") == 0) {
        if ( !(config_isparamset(RUParamList.paramarray[j],RU_LOCAL_IF_NAME_IDX)) ) {
          RC.ru[j]->if_south                        = LOCAL_RF;
          RC.ru[j]->function                        = gNodeB_3GPP;
          printf("Setting function for RU %d to gNodeB_3GPP\n",j);
        } else {
          RC.ru[j]->eth_params.local_if_name  = strdup(*(RUParamList.paramarray[j][RU_LOCAL_IF_NAME_IDX].strptr));
          RC.ru[j]->eth_params.my_addr        = strdup(*(RUParamList.paramarray[j][RU_LOCAL_ADDRESS_IDX].strptr));
          RC.ru[j]->eth_params.remote_addr    = strdup(*(RUParamList.paramarray[j][RU_REMOTE_ADDRESS_IDX].strptr));
          RC.ru[j]->eth_params.my_portc       = *(RUParamList.paramarray[j][RU_LOCAL_PORTC_IDX].uptr);
          RC.ru[j]->eth_params.remote_portc   = *(RUParamList.paramarray[j][RU_REMOTE_PORTC_IDX].uptr);
          RC.ru[j]->eth_params.my_portd       = *(RUParamList.paramarray[j][RU_LOCAL_PORTD_IDX].uptr);
          RC.ru[j]->eth_params.remote_portd   = *(RUParamList.paramarray[j][RU_REMOTE_PORTD_IDX].uptr);

          if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "udp") == 0) {
            RC.ru[j]->if_south                        = LOCAL_RF;
            RC.ru[j]->function                        = NGFI_RRU_IF5;
            RC.ru[j]->eth_params.transp_preference    = ETH_UDP_MODE;
            printf("Setting function for RU %d to NGFI_RRU_IF5 (udp)\n",j);
          } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "raw") == 0) {
            RC.ru[j]->if_south                        = LOCAL_RF;
            RC.ru[j]->function                        = NGFI_RRU_IF5;
            RC.ru[j]->eth_params.transp_preference    = ETH_RAW_MODE;
            printf("Setting function for RU %d to NGFI_RRU_IF5 (raw)\n",j);
          } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "udp_if4p5") == 0) {
            RC.ru[j]->if_south                        = LOCAL_RF;
            RC.ru[j]->function                        = NGFI_RRU_IF4p5;
            RC.ru[j]->eth_params.transp_preference    = ETH_UDP_IF4p5_MODE;
            printf("Setting function for RU %d to NGFI_RRU_IF4p5 (udp)\n",j);
          } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "raw_if4p5") == 0) {
            RC.ru[j]->if_south                        = LOCAL_RF;
            RC.ru[j]->function                        = NGFI_RRU_IF4p5;
            RC.ru[j]->eth_params.transp_preference    = ETH_RAW_IF4p5_MODE;
            printf("Setting function for RU %d to NGFI_RRU_IF4p5 (raw)\n",j);
          }
        }

        RC.ru[j]->max_pdschReferenceSignalPower     = *(RUParamList.paramarray[j][RU_MAX_RS_EPRE_IDX].uptr);;
        RC.ru[j]->max_rxgain                        = *(RUParamList.paramarray[j][RU_MAX_RXGAIN_IDX].uptr);
        RC.ru[j]->num_bands                         = RUParamList.paramarray[j][RU_BAND_LIST_IDX].numelt;

        for (i=0; i<RC.ru[j]->num_bands; i++) RC.ru[j]->band[i] = RUParamList.paramarray[j][RU_BAND_LIST_IDX].iptr[i];
      } //strcmp(local_rf, "yes") == 0
      else {
        printf("RU %d: Transport %s\n",j,*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr));
        RC.ru[j]->eth_params.local_if_name        = strdup(*(RUParamList.paramarray[j][RU_LOCAL_IF_NAME_IDX].strptr));
        RC.ru[j]->eth_params.my_addr          = strdup(*(RUParamList.paramarray[j][RU_LOCAL_ADDRESS_IDX].strptr));
        RC.ru[j]->eth_params.remote_addr        = strdup(*(RUParamList.paramarray[j][RU_REMOTE_ADDRESS_IDX].strptr));
        RC.ru[j]->eth_params.my_portc         = *(RUParamList.paramarray[j][RU_LOCAL_PORTC_IDX].uptr);
        RC.ru[j]->eth_params.remote_portc       = *(RUParamList.paramarray[j][RU_REMOTE_PORTC_IDX].uptr);
        RC.ru[j]->eth_params.my_portd         = *(RUParamList.paramarray[j][RU_LOCAL_PORTD_IDX].uptr);
        RC.ru[j]->eth_params.remote_portd       = *(RUParamList.paramarray[j][RU_REMOTE_PORTD_IDX].uptr);

        if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "udp") == 0) {
          RC.ru[j]->if_south                     = REMOTE_IF5;
          RC.ru[j]->function                     = NGFI_RAU_IF5;
          RC.ru[j]->eth_params.transp_preference = ETH_UDP_MODE;
        } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "raw") == 0) {
          RC.ru[j]->if_south                     = REMOTE_IF5;
          RC.ru[j]->function                     = NGFI_RAU_IF5;
          RC.ru[j]->eth_params.transp_preference = ETH_RAW_MODE;
        } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "udp_if4p5") == 0) {
          RC.ru[j]->if_south                     = REMOTE_IF4p5;
          RC.ru[j]->function                     = NGFI_RAU_IF4p5;
          RC.ru[j]->eth_params.transp_preference = ETH_UDP_IF4p5_MODE;
        } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "raw_if4p5") == 0) {
          RC.ru[j]->if_south                     = REMOTE_IF4p5;
          RC.ru[j]->function                     = NGFI_RAU_IF4p5;
          RC.ru[j]->eth_params.transp_preference = ETH_RAW_IF4p5_MODE;
        } else if (strcmp(*(RUParamList.paramarray[j][RU_TRANSPORT_PREFERENCE_IDX].strptr), "raw_if5_mobipass") == 0) {
          RC.ru[j]->if_south                     = REMOTE_IF5;
          RC.ru[j]->function                     = NGFI_RAU_IF5;
          RC.ru[j]->if_timing                    = synch_to_other;
          RC.ru[j]->eth_params.transp_preference = ETH_RAW_IF5_MOBIPASS;
        }
      }  /* strcmp(local_rf, "yes") != 0 */

      RC.ru[j]->nb_tx   = *(RUParamList.paramarray[j][RU_NB_TX_IDX].uptr);
      RC.ru[j]->nb_rx   = *(RUParamList.paramarray[j][RU_NB_RX_IDX].uptr);
      RC.ru[j]->att_tx  = *(RUParamList.paramarray[j][RU_ATT_TX_IDX].uptr);
      RC.ru[j]->att_rx  = *(RUParamList.paramarray[j][RU_ATT_RX_IDX].uptr);
    }// j=0..num_rus
  } else {
    RC.nb_RU = 0;
  } // setting != NULL

  return;
}

void init_RU(const char *rf_config_file) {
  int ru_id;
  RU_t *ru;
  PHY_VARS_gNB *gNB0= (PHY_VARS_gNB *)NULL;
  NR_DL_FRAME_PARMS *fp = (NR_DL_FRAME_PARMS *)NULL;
  int i;
  int CC_id;
  // create status mask
  RC.ru_mask = 0;
  pthread_mutex_init(&RC.ru_mutex,NULL);
  pthread_cond_init(&RC.ru_cond,NULL);
  // read in configuration file)
  printf("configuring RU from file\n");
  RCconfig_RU();
  LOG_I(PHY,"number of L1 instances %d, number of RU %d, number of CPU cores %d\n",RC.nb_nr_L1_inst,RC.nb_RU,get_nprocs());

  if (RC.nb_nr_CC != 0)
    for (i=0; i<RC.nb_nr_L1_inst; i++)
      for (CC_id=0; CC_id<RC.nb_nr_CC[i]; CC_id++) RC.gNB[i][CC_id]->num_RU=0;

  LOG_D(PHY,"Process RUs RC.nb_RU:%d\n",RC.nb_RU);

  for (ru_id=0; ru_id<RC.nb_RU; ru_id++) {
    LOG_D(PHY,"Process RC.ru[%d]\n",ru_id);
    ru               = RC.ru[ru_id];
    ru->rf_config_file = rf_config_file;
    ru->idx          = ru_id;
    ru->ts_offset    = 0;
    // use gNB_list[0] as a reference for RU frame parameters
    // NOTE: multiple CC_id are not handled here yet!

    if (ru->num_gNB > 0) {
      LOG_D(PHY, "%s() RC.ru[%d].num_gNB:%d ru->gNB_list[0]:%p RC.gNB[0][0]:%p rf_config_file:%s\n", __FUNCTION__, ru_id, ru->num_gNB, ru->gNB_list[0], RC.gNB[0][0], ru->rf_config_file);

      if (ru->gNB_list[0] == 0) {
        LOG_E(PHY,"%s() DJP - ru->gNB_list ru->num_gNB are not initialized - so do it manually\n", __FUNCTION__);
        ru->gNB_list[0] = RC.gNB[0][0];
        ru->num_gNB=1;
        //
        // DJP - feptx_prec() / feptx_ofdm() parses the gNB_list (based on num_gNB) and copies the txdata_F to txdata in RU
        //
      } else {
        LOG_E(PHY,"DJP - delete code above this %s:%d\n", __FILE__, __LINE__);
      }
    }

    gNB0             = ru->gNB_list[0];
    fp               = ru->nr_frame_parms;
    LOG_D(PHY, "RU FUnction:%d ru->if_south:%d\n", ru->function, ru->if_south);

    if (gNB0) {
      if ((ru->function != NGFI_RRU_IF5) && (ru->function != NGFI_RRU_IF4p5))
        AssertFatal(gNB0!=NULL,"gNB0 is null!\n");

      if (gNB0) {
        LOG_I(PHY,"Copying frame parms from gNB %d to ru %d\n",gNB0->Mod_id,ru->idx);
        memcpy((void *)fp,(void *)&gNB0->frame_parms,sizeof(NR_DL_FRAME_PARMS));
        memset((void *)ru->frame_parms, 0, sizeof(LTE_DL_FRAME_PARMS));
        // attach all RU to all gNBs in its list/
        LOG_D(PHY,"ru->num_gNB:%d gNB0->num_RU:%d\n", ru->num_gNB, gNB0->num_RU);

        for (i=0; i<ru->num_gNB; i++) {
          gNB0 = ru->gNB_list[i];
          gNB0->RU_list[gNB0->num_RU++] = ru;
        }
      }
    }

    //    LOG_I(PHY,"Initializing RRU descriptor %d : (%s,%s,%d)\n",ru_id,ru_if_types[ru->if_south],gNB_timing[ru->if_timing],ru->function);
    set_function_spec_param(ru);
    LOG_I(PHY,"Starting ru_thread %d\n",ru_id);
    init_RU_proc(ru);
  } // for ru_id

  //  sleep(1);
  LOG_D(HW,"[nr-softmodem.c] RU threads created\n");
}


int main( int argc, char **argv ) {
  AssertFatal(load_configmodule(argc,argv) != NULL, "");
  logInit();
  start_background_system();
  mode = normal_txrx;
  configure_linux();
  printf("Reading in command-line options\n");
  get_options ();
  set_taus_seed (0);
  initNotifiedFIFO(&mainThreadFIFO);

  if (opt_type != OPT_NONE) {
    if (init_opt(in_path, in_ip) == -1)
      LOG_E(OPT,"failed to run OPT \n");
  }

  cpuf=get_cpu_freq_GHz();
  itti_init(TASK_MAX, THREAD_MAX, MESSAGES_ID_MAX, tasks_info, messages_info);
  AssertFatal(itti_create_task (TASK_GNB_APP, gNB_app_task, NULL) ==0, "");
  AssertFatal(itti_create_task (TASK_RRC_GNB, rrc_gnb_task, NULL) ==0, "");
  pthread_mutex_init(&ue_pf_po_mutex, NULL);
  memset (&UE_PF_PO[0][0], 0, sizeof(UE_PF_PO_t)*NUMBER_OF_UE_MAX*MAX_NUM_CCs);
  pthread_cond_init(&sync_cond,NULL);
  pthread_mutex_init(&sync_mutex, NULL);

  tpool_t pool;
  Tpool=&pool;
  char params[]="-1,-1";
  initTpool(params, Tpool, false);

  if (do_forms==1) {
    loader_shlibfunc_t shlib_fdesc[1]= {0};
    shlib_fdesc[0].fname="startScope";
    AssertFatal(load_module_shlib("gnbScope",shlib_fdesc,1,NULL)>=0,
                "Error loading scope library");
    scopeParms_t p= {&argc,argv};
    void (*f) (scopeParms_t *) = (void (*) (scopeParms_t *)) shlib_fdesc[0].fptr;
    f(&p);
  }

  // Fixme: a sleep is needed because some external (thread?)
  // has to populate something
  // remove the sleep, you will see the assert
  sleep(1);
  number_of_cards = 1;
  init_gNB(single_thread_flag,wait_for_sync);
  init_RU(rf_config_file);
  notifiedFIFO_elt_t *msgs[RC.nb_RU];

  for (int ru=0; ru<RC.nb_RU; ru++)
    msgs[ru]=pullNotifiedFIFO(&mainThreadFIFO);

  init_eNB_afterRU();

  for (int ru=0; ru<RC.nb_RU; ru++)
    pushNotifiedFIFO(msgs[ru]->reponseFifo, msgs[ru]);

  pthread_mutex_lock(&sync_mutex);
  sync_var=0;
  pthread_cond_broadcast(&sync_cond);
  pthread_mutex_unlock(&sync_mutex);
  // When threads leaves, they  send a final message to main
  notifiedFIFO_elt_t *msg=pullNotifiedFIFO(&mainThreadFIFO);
  return 0;
}
