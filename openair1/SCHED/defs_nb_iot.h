

#ifndef __openair_SCHED_NB_IOT_H__
#define __openair_SCHED_NB_IOT_H__

#include "PHY/defs.h"
#include "PHY/defs_nb_iot.h"
#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"

void process_schedule_rsp(Sched_Rsp_t *sched_rsp,
                          PHY_VARS_eNB *eNB,
                          eNB_rxtx_proc_t *proc);

/*Processing the ue-specific resources for uplink in NB-IoT*/
void NB_phy_procedures_eNB_uespec_RX(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc, UL_IND_t *UL_INFO);

/* For NB-IoT, we put NPBCH in later part, since it would be scheduled by MAC scheduler,this generates NRS/NPSS/NSSS*/
void NB_common_signal_procedures (PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc); 

/*Generate the ulsch params and do the mapping for the FAPI style parameters to OAI, and then do the packing*/
void NB_generate_eNB_ulsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,nfapi_hi_dci0_request_pdu_t *hi_dci0_pdu);

/*Generate the dlsch params and do the mapping for the FAPI style parameters to OAI, and then do the packing*/
void NB_generate_eNB_dlsch_params(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t * proc,nfapi_dl_config_request_pdu_t *dl_config_pdu);

/*Process all the scheduling result from MAC and also common signals.*/
void NB_phy_procedures_eNB_TX(PHY_VARS_eNB *eNB,eNB_rxtx_proc_t *proc,int do_meas);
#endif


