/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2014 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is
    included in this distribution in the file called "COPYING". If not,
    see <http://www.gnu.org/licenses/>.

  Contact Information
  OpenAirInterface Admin: openair_admin@eurecom.fr
  OpenAirInterface Tech : openair_tech@eurecom.fr
  OpenAirInterface Dev  : openair4g-devel@lists.eurecom.fr

  Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

*******************************************************************************/
/*! \file eNB_scheduler.c
 * \brief eNB scheduler top level function operates on per subframe basis
 * \author  Navid Nikaein and Raymond Knopp
 * \date 2010 - 2014
 * \email: navid.nikaein@eurecom.fr
 * \version 0.5
 * @ingroup _mac

 */

#include "assertions.h"
#include "PHY/defs.h"
#include "PHY/extern.h"

#include "SCHED/defs.h"
#include "SCHED/extern.h"

#include "LAYER2/MAC/defs.h"
#include "LAYER2/MAC/extern.h"

#include "LAYER2/MAC/proto.h"
#include "UTIL/LOG/log.h"
#include "UTIL/LOG/vcd_signal_dumper.h"
#include "UTIL/OPT/opt.h"
#include "OCG.h"
#include "OCG_extern.h"

#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"

//#include "LAYER2/MAC/pre_processor.c"
#include "pdcp.h"

#if defined(ENABLE_ITTI)
# include "intertask_interface.h"
#endif

#define ENABLE_MAC_PAYLOAD_DEBUG
#define DEBUG_eNB_SCHEDULER 1
//#define DEBUG_HEADER_PARSING 1
//#define DEBUG_PACKET_TRACE 1

/*
  #ifndef USER_MODE
  #define msg debug_msg
  #endif
 */





#if FAPI

/* global variables to deal with TPC command - we send one only once per frame maximum */
static long global_subframe;                        /* global subframe - incremented at each TTI */
static long ue_last_pucch_tpc[NUMBER_OF_UE_MAX];    /* last global subframe where a PUCCH TPC was sent to the UE */
static long ue_last_pusch_tpc[NUMBER_OF_UE_MAX];    /* last global subframe where a PUSCH TPC was sent to the UE */

static void fapi_dl_tpc(int module_id, int CC_id, struct DlDciListElement_s *dci)
{
  int              UE_id                = find_UE_id(module_id, dci->rnti);
  LTE_eNB_UE_stats *eNB_UE_stats        = NULL;
  int32_t          normalized_rx_power;
  int32_t          target_rx_power;
  static int       tpc_accumulated = 0;

  if (UE_id == -1) { printf("%s:%d: rnti %x not found\n", __FILE__, __LINE__, dci->rnti); abort(); }

  // this assumes accumulated tpc
  // make sure that we are only sending a tpc update once a frame, otherwise the control loop will freak out
  if (global_subframe < ue_last_pucch_tpc[UE_id] + 10) {
printf("TPC PUCCH global sf %ld (%ld/%ld) UE %x: no TPC, last one is too recent (accumulated %d)\n", global_subframe, global_subframe/10, global_subframe%10, dci->rnti, tpc_accumulated);
    dci->tpc = 0;
    return;
  }

  eNB_UE_stats        = mac_xface->get_eNB_UE_stats(module_id, CC_id, dci->rnti);
  if (eNB_UE_stats == NULL) { printf("%s:%d: stats for rnti %x not found\n", __FILE__, __LINE__, dci->rnti); abort(); }
  normalized_rx_power = eNB_UE_stats->Po_PUCCH_dBm;
  target_rx_power     = mac_xface->get_target_pucch_rx_power(module_id, CC_id) + 20;

  if (eNB_UE_stats->Po_PUCCH_update == 1) {
    eNB_UE_stats->Po_PUCCH_update = 0;

    ue_last_pucch_tpc[UE_id] = global_subframe;

    if (normalized_rx_power > target_rx_power + 1) {
      dci->tpc = -1;
      tpc_accumulated--;
    } else if (normalized_rx_power < target_rx_power - 1) {
      dci->tpc = 1;
      tpc_accumulated++;
    } else {
      dci->tpc = 0;
    }
  }
printf("TPC PUCCH global sf %ld (%ld/%ld) UE %x: TPC %d (accumulated %d)\n", global_subframe, global_subframe/10, global_subframe%10, dci->rnti, dci->tpc, tpc_accumulated);
}

static void fapi_ul_tpc(int module_id, int CC_id, struct UlDciListElement_s *dci)
{
  int              UE_id                = find_UE_id(module_id, dci->rnti);
  LTE_eNB_UE_stats *eNB_UE_stats        = NULL;
  int32_t          normalized_rx_power;
  int32_t          target_rx_power;
  static int       tpc_accumulated = 0;

  if (UE_id == -1) { printf("%s:%d: rnti %x not found\n", __FILE__, __LINE__, dci->rnti); abort(); }

  // this assumes accumulated tpc
  // make sure that we are only sending a tpc update once a frame, otherwise the control loop will freak out
  if (global_subframe < ue_last_pusch_tpc[UE_id] + 10) {
printf("TPC PUSCH global sf %ld (%ld/%ld) UE %x: no TPC, last one is too recent (accumulated %d)\n", global_subframe, global_subframe/10, global_subframe%10, dci->rnti, tpc_accumulated);
    dci->tpc = 0;
    return;
  }

  eNB_UE_stats        = mac_xface->get_eNB_UE_stats(module_id, CC_id, dci->rnti);
  if (eNB_UE_stats == NULL) { printf("%s:%d: stats for rnti %x not found\n", __FILE__, __LINE__, dci->rnti); abort(); }
  normalized_rx_power = eNB_UE_stats->UL_rssi[0];
  target_rx_power     = mac_xface->get_target_pusch_rx_power(module_id, CC_id);

  if (normalized_rx_power > target_rx_power + 1) {
    dci->tpc = -1;
    tpc_accumulated--;
  } else if (normalized_rx_power < target_rx_power - 1) {
    dci->tpc = 1;
    tpc_accumulated++;
  } else {
    dci->tpc = 0;
  }

  if (dci->tpc != 0) {
    ue_last_pusch_tpc[UE_id] = global_subframe;
  }
printf("TPC PUSCH global sf %ld (%ld/%ld) UE %x: TPC %d (accumulated %d)\n", global_subframe, global_subframe/10, global_subframe%10, dci->rnti, dci->tpc, tpc_accumulated);
}

/* this structure is used to store downlink ack/nack information
 * to be sent to FAPI by SchedDlTriggerReq
 */
/* TODO: lock access to it or not? */
/* TODO: do it per CC */
static struct {
  struct {
    int rnti;
    int harq_pid;
    int ack[MAX_TB_LIST];
    int ack_count;
  } ack[MAX_DL_INFO_LIST];
  int count;
} fapi_dl_ack_nack_data;

/* this array is used to store uplink ack/nack information
 * to be sent to FAPI by SchedUlTriggerReq
 * one element per TTI
 */
/* TODO: do it per CC */
static struct {
  struct {
    int reception_frame;          /* the frame where the ACK/NACK has been received */
    int reception_subframe;       /* the subframe where the ACK/NACK has been received */
    int rnti;
    int ack;
    int length[MAX_LC_LIST+1];
  } ack[MAX_UL_INFO_LIST];
  int count;
} fapi_ul_ack_nack_data[10];

/* this function is called by the PHY to signal UE's ACK/NACK */
/* TODO: do it per CC */
void fapi_dl_ack_nack(int rnti, int harq_pid, int transport_block, int ack)
{
  int pos = fapi_dl_ack_nack_data.count;
printf("GOT DOWNLINK ack %d for rnti %x harq_pid %d transport_block %d\n", ack, rnti, harq_pid, transport_block);
  /* TODO: handle more than 1 TB */
  if (transport_block) { printf("%s:%d:%s: TODO: tb != 0\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  if (pos == MAX_DL_INFO_LIST) {
    LOG_E(MAC, "fapi_dl_ack_nack: full!\n");
    abort();
  }

  fapi_dl_ack_nack_data.ack[pos].rnti      = rnti;
  fapi_dl_ack_nack_data.ack[pos].harq_pid  = harq_pid;
  fapi_dl_ack_nack_data.ack[pos].ack[0]    = ack;       /* TODO: use transport_block here */
  fapi_dl_ack_nack_data.ack[pos].ack_count = 1;         /* TODO: take care of transport block */

  fapi_dl_ack_nack_data.count++;
}

/* this function is called by the PHY to inform about correct or wrong
 * reception by the eNodeB of an uplink UE transmission
 */
/* TODO: do it per CC */
void fapi_ul_ack_nack(int frame, int subframe, int harq_pid, int rnti, int ack)
{
printf("GOT UPLINK ack %d for rnti %x harq_pid %d (f/sf %d/%d)\n", ack, rnti, harq_pid, frame, subframe);
  int pos = fapi_ul_ack_nack_data[subframe].count;
  if (pos == MAX_UL_INFO_LIST) {
    LOG_E(MAC, "fapi_ul_ack_nack: full! (f/sf %d/%d)\n", frame, subframe);
    abort();
  }
  fapi_ul_ack_nack_data[subframe].ack[pos].reception_frame    = frame;
  fapi_ul_ack_nack_data[subframe].ack[pos].reception_subframe = subframe;
  fapi_ul_ack_nack_data[subframe].ack[pos].rnti               = rnti;
  fapi_ul_ack_nack_data[subframe].ack[pos].ack                = ack;

  /* the values in length are set later in the function fapi_ul_lc_length */
  memset(fapi_ul_ack_nack_data[subframe].ack[pos].length, 0, sizeof(int) * MAX_LC_LIST+1);

  fapi_ul_ack_nack_data[subframe].count++;
}

/* this function is called by rx_sdu to indicate the number of
 * bytes received by the given Logical Channel
 */
void fapi_ul_lc_length(int frame, int subframe, int lcid, int length, int rnti)
{
printf("GOT lcid %d length %d (f/sf %d/%d)\n", lcid, length, frame, subframe);
  int pos = fapi_ul_ack_nack_data[subframe].count - 1;
  if (pos < 0) { printf("%s:%d:%s: fatal error\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  /* TODO: remove this check (and the rnti parameter)? */
  if (rnti != fapi_ul_ack_nack_data[subframe].ack[pos].rnti) {
    printf("%s:%d:%s: fatal error: LCID %d wrong RNTI %x (expected %x)\n",
           __FILE__, __LINE__, __FUNCTION__, lcid,
           rnti, fapi_ul_ack_nack_data[subframe].ack[pos].rnti);
    abort();
  }

  if (lcid < 0 || lcid > 3)
    { printf("%s:%d:%s: fatal error: unhandled LCID %d\n", __FILE__, __LINE__, __FUNCTION__, lcid); abort(); }

  fapi_ul_ack_nack_data[subframe].ack[pos].length[lcid] = length;
}

void fapi_dl_cqi_report(int module_id, int rnti, int frame, int subframe, int cqi_wideband, int *cqi_subband, int rank_indication)
{
  /* TODO: 2 TBs, other reporting modes - we suppose 3-0 (see 36.213 7.2.1) */
  fapi_interface_t                   *fapi;
  struct SchedDlCqiInfoReqParameters params;
  struct CqiListElement_s            cqi;
  int                                i;

  fapi = eNB_mac_inst[module_id].fapi;

  cqi.rnti                          = rnti;
  cqi.csiReport.ri                  = rank_indication;
  cqi.csiReport.mode                = A30;          /* TODO: get real value */
  cqi.csiReport.report.A30Csi.wbCqi = cqi_wideband;
  for (i = 0; i < MAX_HL_SB; i++)
    cqi.csiReport.report.A30Csi.sbCqi[i] = cqi_subband[i];

  /* TODO: remove this oaisim fix
   * value 0 is not allowed in wideband CQI reporting,
   * see 36.213 7.2.3 table 7.2.3-1
   * but we get value 0 in oaisim
   */
  if (cqi.csiReport.report.A30Csi.wbCqi == 0) cqi.csiReport.report.A30Csi.wbCqi++;

  cqi.servCellIndex                 = 0;         /* TODO: get correct value */

  params.sfnSf                 = frame * 16 + subframe;
  params.nrcqiList             = 1;
  params.cqiList               = &cqi;
  params.nr_vendorSpecificList = 0;
  params.vendorSpecificList    = NULL;

  SchedDlCqiInfoReq(fapi->sched, &params);
}

static void fapi_convert_dl_1A_5MHz_FDD(struct DlDciListElement_s *dci, DCI_ALLOC_t *a)
{
  DCI1A_5MHz_FDD_t *d = (DCI1A_5MHz_FDD_t *)a->dci_pdu;

  if (dci->nr_of_tbs != 1) { printf("%s:%d: TODO\n", __FUNCTION__, __LINE__); exit(1); }

  d->type       = 1;    /* type = 0 => DCI Format 0, type = 1 => DCI Format 1A */
  d->vrb_type   = dci->vrbFormat == VRB_LOCALIZED ? 0 :
                  dci->vrbFormat == VRB_DISTRIBUTED ? 1 :
                  (printf("%s:%d: error\n", __FUNCTION__, __LINE__), abort(), 0);
  d->rballoc    = dci->rbBitmap;
  d->mcs        = dci->mcs[0];        /* TODO: take care of transport block index */
  d->harq_pid   = dci->harqProcess;
  d->ndi        = dci->ndi[0];        /* TODO: take care of transport block index */
  d->rv         = dci->rv[0];         /* TODO: take care of transport block index */
  d->TPC        = dci->tpc == -1 ? 0 :    /* see 36.213 table 5.1.2.1-1 */
                  dci->tpc ==  0 ? 1 :
                  dci->tpc ==  1 ? 2 :
                  dci->tpc ==  3 ? 3 :
                  (printf("%s:%d: error\n", __FUNCTION__, __LINE__), abort(), 0);
  d->padding    = 0;

  a->dci_length = sizeof_DCI1A_5MHz_FDD_t;
  a->format     = format1A;
}

static uint32_t revert(uint32_t x, int len)
{
  int i;
  int ret = 0;
  for (i = 0; i < len; i++) {
    ret <<= 1;
    ret |= x & 1;
    x >>= 1;
  }
  return ret;
}

static void fapi_convert_dl_1_5MHz_FDD(struct DlDciListElement_s *dci, DCI_ALLOC_t *a)
{
  DCI1_5MHz_FDD_t *d = (DCI1_5MHz_FDD_t *)a->dci_pdu;

  if (dci->nr_of_tbs != 1) { printf("%s:%d: TODO\n", __FUNCTION__, __LINE__); exit(1); }

  d->rah        = dci->resAlloc == 0 ? 0 :
                  dci->resAlloc == 1 ? 1 :
                  (printf("%s:%d: error\n", __FUNCTION__, __LINE__), abort(), 0);
  d->rballoc    = revert(dci->rbBitmap, 13);
  d->mcs        = dci->mcs[0];        /* TODO: take care of transport block index */
  d->harq_pid   = dci->harqProcess;
  d->ndi        = dci->ndi[0];        /* TODO: take care of transport block index */
  d->rv         = dci->rv[0];         /* TODO: take care of transport block index */
  d->TPC        = dci->tpc == -1 ? 0 :    /* see 36.213 table 5.1.2.1-1 */
                  dci->tpc ==  0 ? 1 :
                  dci->tpc ==  1 ? 2 :
                  dci->tpc ==  3 ? 3 :
                  (printf("%s:%d: error (dci->tpc = %d)\n", __FUNCTION__, __LINE__, dci->tpc), abort(), 0);
  d->dummy      = 0;

  a->dci_length = sizeof_DCI1_5MHz_FDD_t;
  a->format     = format1;
}

static void fapi_convert_dl_dci(int module_id, int CC_id,
    struct DlDciListElement_s *dci, DCI_ALLOC_t *a)
{
  /* set TPC in the DCI */
  /* TODO: remove it if/when the scheduler does it */
  if (dci->rnti != 0)
    fapi_dl_tpc(module_id, CC_id, dci);

  /* 5MHz FDD supposed, not checked */
  switch (dci->format) {
  case ONE_A:
    fapi_convert_dl_1A_5MHz_FDD(dci, a);
    break;
  case ONE:
    fapi_convert_dl_1_5MHz_FDD(dci, a);
    break;
  default: printf("%s:%d: TODO\n", __FUNCTION__, __LINE__); abort();
  }
  a->L = dci->aggrLevel == 1 ? 0 :
         dci->aggrLevel == 2 ? 1 :
         dci->aggrLevel == 4 ? 2 :
         dci->aggrLevel == 8 ? 3 :
         (printf("%s:%d: error\n", __FUNCTION__, __LINE__), abort(), 0);
  a->firstCCE = dci->cceIndex;
  a->ra_flag  = 0;              /* TODO: set to 1 only by fapi_schedule_RAR, is it ok? */
  a->rnti     = dci->rnti;
}

static void fapi_convert_ul_5MHz_FDD(module_id_t module_idP, int CC_id,
    struct UlDciListElement_s *dci, DCI_ALLOC_t *a)
{
  DCI0_5MHz_FDD_t *d = (DCI0_5MHz_FDD_t *)a->dci_pdu;

  d->type       = 0;    /* type = 0 => DCI Format 0, type = 1 => DCI Format 1A */
  d->hopping    = dci->hopping;
  d->rballoc    = mac_xface->computeRIV(PHY_vars_eNB_g[module_idP][CC_id]->lte_frame_parms.N_RB_DL, dci->rbStart, dci->rbLen);
  d->mcs        = dci->mcs;
  d->ndi        = dci->ndi;
  d->TPC        = dci->tpc == -1 ? 0 :    /* see 36.213 table 5.1.1.1-2, accumulated case supposed */
                  dci->tpc ==  0 ? 1 :
                  dci->tpc ==  1 ? 2 :
                  dci->tpc ==  3 ? 3 :
                  (printf("%s:%d: error (tpc = %d)\n", __FUNCTION__, __LINE__, dci->tpc), abort(), 0);
  d->cshift     = dci->n2Dmrs;
  d->cqi_req    = dci->cqiRequest;
  d->padding    = 0;

  a->dci_length = sizeof_DCI0_5MHz_FDD_t;
  a->format     = format0;
}

static void fapi_convert_ul_dci(module_id_t module_idP, int CC_id,
    struct UlDciListElement_s *dci, DCI_ALLOC_t *a)
{
  /* set TPC in the DCI */
  /* TODO: remove it if/when the scheduler does it */
  fapi_ul_tpc(module_idP, CC_id, dci);

  /* 5MHz FDD supposed, not checked */
  fapi_convert_ul_5MHz_FDD(module_idP, CC_id, dci, a);

  a->L = dci->aggrLevel == 1 ? 0 :
         dci->aggrLevel == 2 ? 1 :
         dci->aggrLevel == 4 ? 2 :
         dci->aggrLevel == 8 ? 3 :
         (printf("%s:%d: error\n", __FUNCTION__, __LINE__), abort(), 0);
  a->firstCCE = dci->cceIndex;
  a->ra_flag  = 0;
  a->rnti     = dci->rnti;
}

/* index 0 for SIB1, 1 for SIB23 */
static void fapi_schedule_SI(module_id_t module_idP, int CC_id, frame_t frameP,
    sub_frame_t subframeP, uint8_t index, struct DlDciListElement_s *dci)
{
  eNB_MAC_INST *eNB = &eNB_mac_inst[module_idP];
  DCI_PDU *dci_pdu = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
  DCI_ALLOC_t *a = &dci_pdu->dci_alloc[dci_pdu->Num_common_dci];
  int bcch_sdu_length;

  if (dci_pdu->Num_common_dci >= NUM_DCI_MAX) { printf("%s:%d:%s: too much DCIs\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  /* we can increment Num_common_dci or Num_ue_spec_dci, there is no difference */
  dci_pdu->Num_common_dci++;

  fapi_convert_dl_dci(module_idP, CC_id, dci, a);

  /* bug in libscheduler.a? rnti is 0 */
  a->rnti = SI_RNTI;

  if (index == 0)
    bcch_sdu_length = mac_rrc_get_SIB1(module_idP, CC_id, &eNB->common_channels[CC_id].BCCH_pdu.payload[0]);
  else if (index == 1)
    bcch_sdu_length = mac_rrc_get_SIB23(module_idP, CC_id, &eNB->common_channels[CC_id].BCCH_pdu.payload[0]);
  else { printf("%s:%d: fatal error\n", __FUNCTION__, __LINE__); abort(); }

  eNB->eNB_stats[CC_id].total_num_bcch_pdu+=1;
  eNB->eNB_stats[CC_id].bcch_buffer=bcch_sdu_length;
  eNB->eNB_stats[CC_id].total_bcch_buffer+=bcch_sdu_length;
  eNB->eNB_stats[CC_id].bcch_mcs=dci->mcs[0];         /* TODO: take care of transport block index */
}

static void fapi_schedule_RAR(int module_idP, int CC_id, frame_t frameP,
    sub_frame_t subframeP, uint16_t rnti, uint32_t grant,
    struct DlDciListElement_s *dci)
{
  eNB_MAC_INST *eNB = &eNB_mac_inst[module_idP];
  DCI_PDU *dci_pdu = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
  DCI_ALLOC_t *a = &dci_pdu->dci_alloc[dci_pdu->Num_common_dci];
  RA_TEMPLATE *RA_template = NULL;
  int i;

  if (dci_pdu->Num_common_dci >= NUM_DCI_MAX) { printf("%s:%d:%s: too much DCIs\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  for (i = 0; i < NB_RA_PROC_MAX; i++) {
    RA_template = (RA_TEMPLATE *)&eNB->common_channels[CC_id].RA_template[i];
    if (RA_template->RA_active != TRUE) continue;
    if (RA_template->rnti != rnti) continue;
    break;
  }
  if (i == NB_RA_PROC_MAX) { printf("%s:%d:%s: possible?\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  RA_template->generate_rar = 1;
  RA_template->UL_grant = grant;

  /* we can increment Num_common_dci or Num_ue_spec_dci, there is no difference */
  dci_pdu->Num_common_dci++;

  fapi_convert_dl_dci(module_idP, CC_id, dci, a);
  a->ra_flag = 1;

  /* bug in libscheduler.a? rnti is 0 */
  a->rnti = RA_template->RA_rnti;
}

static void fapi_schedule_RA_Msg4(int module_idP, int CC_id, int RA,
    int frameP, int subframeP, struct DlDciListElement_s *dci, int pdu_maxsize)
{
  int           UE_id;
  int           rrc_sdu_length;
  eNB_MAC_INST  *eNB = &eNB_mac_inst[module_idP];
  RA_TEMPLATE   *RA_template;
  int           msg4_header_length;
  int           msg4_padding, msg4_post_padding;
  int           TBsize;
  DCI_PDU       *dci_pdu = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
  DCI_ALLOC_t   *a = &dci_pdu->dci_alloc[dci_pdu->Num_common_dci];
  int           offset;
  unsigned char lcid;

  if (dci_pdu->Num_common_dci >= NUM_DCI_MAX) { printf("%s:%d:%s: too much DCIs\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  RA_template = (RA_TEMPLATE *)&eNB->common_channels[CC_id].RA_template[RA];
  UE_id = find_UE_id(module_idP,RA_template->rnti);
  rrc_sdu_length = mac_rrc_data_req(module_idP,
                                    CC_id,
                                    frameP,
                                    CCCH,
                                    1, // 1 transport block
                                    &eNB->common_channels[CC_id].CCCH_pdu.payload[0],
                                    ENB_FLAG_YES,
                                    module_idP,
                                    0); // not used in this case
  if (rrc_sdu_length <= 0) {
    LOG_E(MAC, "fapi_schedule_RA_Msg4: rrc_sdu_length (%d) is not > 0\n", rrc_sdu_length);
    abort();
  }
  if (rrc_sdu_length > pdu_maxsize) {
    LOG_E(MAC, "fapi_schedule_RA_Msg4: rrc_sdu_length (%d) > pdu_maxsize (%d)\n", rrc_sdu_length, pdu_maxsize);
    abort();
  }

  msg4_header_length = 1+6+1;  // CR header, CR CE, SDU header

  TBsize = dci->tbsSize[CC_id] / 8;

  if ((TBsize - rrc_sdu_length - msg4_header_length) <= 2) {
    msg4_padding = TBsize - rrc_sdu_length - msg4_header_length;
    msg4_post_padding = 0;
  } else {
    msg4_padding = 0;
    msg4_post_padding = TBsize - rrc_sdu_length - msg4_header_length -1;
  }

  LOG_I(MAC,"[eNB %d][RAPROC] CC_id %d Frame %d subframeP %d Msg4 : TBS %d, sdu_len %d, msg4_header %d, msg4_padding %d, msg4_post_padding %d\n",
        module_idP, CC_id, frameP, subframeP, TBsize, rrc_sdu_length, msg4_header_length, msg4_padding, msg4_post_padding);

  lcid = 0;
  offset = generate_dlsch_header((unsigned char*)eNB->UE_list.DLSCH_pdu[CC_id][0][(unsigned char)UE_id].payload[0],
                                 1,                           //num_sdus
                                 (unsigned short*)&rrc_sdu_length,             //
                                 &lcid,                       // sdu_lcid
                                 255,                         // no drx
                                 0,                           // no timing advance
                                 RA_template->cont_res_id,  // contention res id
                                 msg4_padding,                // no padding
                                 msg4_post_padding);

  memcpy((void*)&eNB->UE_list.DLSCH_pdu[CC_id][0][UE_id].payload[0][offset],
         &eNB->common_channels[CC_id].CCCH_pdu.payload[0],
         rrc_sdu_length);

  RA_template->generate_Msg4=0;
  RA_template->wait_ack_Msg4=1;
  RA_template->RA_active = FALSE;

  /* we can increment Num_common_dci or Num_ue_spec_dci, there is no difference */
  dci_pdu->Num_common_dci++;

  fapi_convert_dl_dci(module_idP, CC_id, dci, a);
}

/* returns 1 if the given LCID has a MAC header of fixed size (for DL-SCH) */
static int fixed_size(int lcid)
{
  /* see 36.321, especially table 6.2.1-1 (rel. 10 is used here) */
  return lcid >= 27;
}

/* TODO: deal with more than one transport block */
static void fapi_schedule_ue(int module_id, int CC_id, int frame, int subframe, struct BuildDataListElement_s *d)
{
  int header_size;
  int payload_size;
  int padding_size;
  int tbs;
  int i;
  mac_rlc_status_resp_t rlc_status;
  unsigned char         dlsch_buffer[MAX_DLSCH_PAYLOAD_BYTES];
  int                   dlsch_filled = 0;
  int                   output_length;
  int                   UE_id;
  eNB_MAC_INST          *eNB      = &eNB_mac_inst[module_id];
  UE_list_t             *UE_list  = &eNB->UE_list;
  int                   num_sdus;
  unsigned short        sdu_lengths[MAX_RLC_PDU_LIST];
  unsigned char         sdu_lcids[MAX_RLC_PDU_LIST];
  int                   offset;
  DCI_PDU               *dci_pdu = &eNB->common_channels[CC_id].DCI_pdu;
  DCI_ALLOC_t           *a = &dci_pdu->dci_alloc[dci_pdu->Num_common_dci];

  /* generate DCI */
  if (dci_pdu->Num_common_dci >= NUM_DCI_MAX) { printf("%s:%d:%s: too much DCIs\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  /* we can increment Num_common_dci or Num_ue_spec_dci, there is no difference */
  dci_pdu->Num_common_dci++;

  fapi_convert_dl_dci(module_id, CC_id, &d->dci, a);
printf("RUN fapi_schedule_ue\n");

  if (d->nr_rlcPDU_List[0] != 1) { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
  if (d->nr_rlcPDU_List[1] != 0) { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
  if (d->ceBitmap[0])            { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
  if (d->ceBitmap[1])            { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
  if (d->servCellIndex != 0)     { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  tbs = d->dci.tbsSize[0] / 8;

  num_sdus = 0;

  /* get DLSCH buffer and adjust size according to what RLC says */
  for (i = 0; i < d->nr_rlcPDU_List[0]; i++) {
    rlc_status = mac_rlc_status_ind(
        module_id,
        d->rnti,
        module_id,
        frame,
        ENB_FLAG_YES,
        MBMS_FLAG_NO,
        d->rlcPduList[0][i].logicalChannelIdentity,
        d->rlcPduList[0][i].size);
printf("RLC_SIZE in fapi_schedule_ue %d (asked %d) lcid %d rnti %x f/sf %d/%d\n", rlc_status.bytes_in_buffer, d->rlcPduList[0][i].size, d->rlcPduList[0][i].logicalChannelIdentity, d->rnti, frame, subframe);
    if (rlc_status.bytes_in_buffer <= 0) { printf("%s:%d:%s: error\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
    if (rlc_status.bytes_in_buffer > d->rlcPduList[0][i].size) abort(); /* that can't happen */
    if (dlsch_filled + d->rlcPduList[0][i].size > MAX_DLSCH_PAYLOAD_BYTES) {
      printf("dlsch buffer filled too much\n");
      abort();
    }
    output_length = mac_rlc_data_req(
        module_id,
        d->rnti,
        module_id,
        frame,
        ENB_FLAG_YES,
        MBMS_FLAG_NO,
        d->rlcPduList[0][i].logicalChannelIdentity,
        (char *)&dlsch_buffer[dlsch_filled]);
    if (output_length <= 0 || output_length > d->rlcPduList[0][i].size) abort();
    d->rlcPduList[0][i].size = output_length;
    dlsch_filled += output_length;
    sdu_lengths[num_sdus] = output_length;
    sdu_lcids[num_sdus]   = d->rlcPduList[0][i].logicalChannelIdentity;
    num_sdus++;
printf("FILLED %d bytes\n", output_length);
  }

  /* get size of header and payload */
  header_size = 0;
  payload_size = 0;

  if (d->nr_rlcPDU_List[0]) {
    header_size++;
    payload_size += d->rlcPduList[0][0].size;
  }
  for (i = 1; i < d->nr_rlcPDU_List[0]; i++) {
    if (!fixed_size(d->rlcPduList[0][i-1].logicalChannelIdentity)) {
      header_size++;
      if (d->rlcPduList[0][i-1].size >= 128)
        header_size++;
    }
    header_size++;
    payload_size += d->rlcPduList[0][i].size;
  }

  padding_size = tbs - header_size - payload_size;
printf("PADDING_SIZE %d\n", padding_size);

  UE_id = find_UE_id(module_id, d->rnti);

  /* generate dlsch header */
  offset = generate_dlsch_header((unsigned char*)UE_list->DLSCH_pdu[CC_id][0][UE_id].payload[0],
      num_sdus,
      sdu_lengths,
      sdu_lcids,
      255,                                   // no drx
      0,                                     /* TODO: timing advance */
      NULL,                                  // contention res id
      padding_size <= 2 ? padding_size : 0,
      padding_size > 2 ? padding_size : 0);

  /* fill payload */
  memcpy(&UE_list->DLSCH_pdu[CC_id][0][UE_id].payload[0][offset], dlsch_buffer, dlsch_filled);

  add_ue_dlsch_info(module_id,
      CC_id,
      UE_id,
      subframe,
      /* S_DL_SCHEDULED */ S_DL_WAITING);
}

static void fapi_schedule_uplink(int module_idP, int CC_id, struct UlDciListElement_s *dci)
{
  DCI_PDU       *dci_pdu = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
  DCI_ALLOC_t   *a = &dci_pdu->dci_alloc[dci_pdu->Num_common_dci];

  if (dci_pdu->Num_common_dci >= NUM_DCI_MAX) { printf("%s:%d:%s: too much DCIs\n", __FILE__, __LINE__, __FUNCTION__); abort(); }

  /* we can increment Num_common_dci or Num_ue_spec_dci, there is no difference */
  dci_pdu->Num_common_dci++;

  fapi_convert_ul_dci(module_idP, CC_id, dci, a);
}

char *binary(unsigned x)
{
  static char r[33];
  char *s = r+31;
  int i;
  r[32] = 0;
  for (i = 0; i < 32; i++) {
    *s = '0' + (x&1);
    s--;
    x >>= 1;
  }
  return r;
}

static char *dci_format_to_string(DCI_format_t f)
{
  switch (f) {
    case format0: return "format 0";
    case format1: return "format 1";
    case format1A: return "format 1A";
    case format1B: return "format 1B";
    case format1C: return "format 1C";
    case format1D: return "format 1D";
    case format1E_2A_M10PRB: return "format 1E_2A_M10PRB";
    case format2: return "format 2";
    case format2A: return "format 2A";
    case format2B: return "format 2B";
    case format2C: return "format 2C";
    case format2D: return "format 2D";
    case format3: return "format 3";
  }
  printf("%s:%d: unhandled DCI format\n", __FILE__, __LINE__);
  abort();
}

void eNB_dlsch_ulsch_scheduler(module_id_t module_idP,uint8_t cooperation_flag, frame_t frameP, sub_frame_t subframeP)  //, int calibration_flag) {
{
  int                                 CC_id;
  int                                 i, j;
  DCI_PDU                             *DCI_pdu[MAX_NUM_CCs];
  int                                 mbsfn_status[MAX_NUM_CCs];
  UE_list_t                           *UE_list = &eNB_mac_inst[module_idP].UE_list;
  fapi_interface_t                    *fapi;
  struct SchedDlTriggerReqParameters  dlreq;
  struct SchedDlConfigIndParameters   dlind;
  struct DlInfoListElement_s          dlinfo[MAX_DL_INFO_LIST];
  protocol_ctxt_t                     ctxt;
  int                                 UE_id;
  eNB_MAC_INST                        *eNB = &eNB_mac_inst[module_idP];
  struct SchedUlTriggerReqParameters  ulreq;
  struct SchedUlConfigIndParameters   ulind;
  struct UlInfoListElement_s          ulinfo[MAX_UL_INFO_LIST];
  int                                 ulsf;

printf("SCHEDULER called for f/sf %d/%d\n", frameP, subframeP);
#if defined(ENABLE_ITTI)
  while (1) {
    // Checks if a message has been sent to MAC sub-task
    MessageDef   *msg_p;
    int           result;
    itti_poll_msg (TASK_MAC_ENB, &msg_p);
    if (msg_p == NULL) break;
    result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
    AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
  }
#endif

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    DCI_pdu[CC_id] = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
    DCI_pdu[CC_id]->nCCE=0;
    DCI_pdu[CC_id]->num_pdcch_symbols=1;
    mbsfn_status[CC_id]=0;
    // clear vrb_map
    memset(eNB_mac_inst[module_idP].common_channels[CC_id].vrb_map,0,100);
  }

  // refresh UE list based on UEs dropped by PHY in previous subframe
  i = UE_list->head;
  while (i>=0) {
    int rnti, next_i;
    rnti = UE_RNTI(module_idP, i);
    CC_id = UE_PCCID(module_idP, i);
    next_i= UE_list->next[i];
    if (mac_xface->get_eNB_UE_stats(module_idP, CC_id, rnti)==NULL) {
      mac_remove_ue(module_idP, i, frameP, subframeP);
    }
    i = next_i;
  }

  // clear DCI and BCCH contents before scheduling
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    DCI_pdu[CC_id]->Num_common_dci  = 0;
    DCI_pdu[CC_id]->Num_ue_spec_dci = 0;
#ifdef Rel10
    eNB_mac_inst[module_idP].common_channels[CC_id].mcch_active = 0;
#endif
    eNB_mac_inst[module_idP].frame    = frameP;
    eNB_mac_inst[module_idP].subframe = subframeP;
  }

  /* run PDCP */
  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_idP, ENB_FLAG_YES, NOT_A_RNTI, frameP, 0, module_idP);
  pdcp_run(&ctxt);

  // check HO
  rrc_rx_tx(&ctxt,
            0, // eNB index, unused in eNB
            CC_id);

  /* MBMS thing not done */
  mbsfn_status[0]++; /* avoid gcc warning */

  /* let's FAPI schedule! */
  fapi = eNB_mac_inst[module_idP].fapi;

  /*************************************************/
  /*           downlink scheduling                 */
  /*************************************************/

  /* check RA - is there one in generate_Msg4 mode with some CCCH data ready in the RRC Srb0? */
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    for (i=0; i < NB_RA_PROC_MAX; i++) {
      if (eNB->common_channels[CC_id].RA_template[i].RA_active == TRUE &&
          eNB->common_channels[CC_id].RA_template[i].generate_Msg4 == 1 &&
          mac_rrc_get_ccch_size(module_idP, CC_id) > 0) {
        struct SchedDlRlcBufferReqParameters rlc;
        memset(&rlc, 0, sizeof(rlc));
        rlc.rnti = eNB->common_channels[CC_id].RA_template[i].rnti;
        rlc.logicalChannelIdentity = CCCH;
        rlc.rlcTransmissionQueueSize = mac_rrc_get_ccch_size(module_idP, CC_id);
        LOG_I(MAC, "calling SchedDlRlcBufferReq on CCCH rnti %x queue_size %d\n", rlc.rnti, rlc.rlcTransmissionQueueSize);
printf("MAC to FAPI downlink BUF CCCH %d\n", rlc.rlcTransmissionQueueSize);
        SchedDlRlcBufferReq(fapi->sched, &rlc);
        /* let's report only once, so put generate_Msg4 to 2
         * (we will generate later on, when FAPI tells us to do so)
         */
        eNB->common_channels[CC_id].RA_template[i].generate_Msg4 = 2;
      }
    }
  }

  /* update RLC buffers status in the scheduler */
  for (UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id]) {
    struct SchedDlRlcBufferReqParameters rlc;
    mac_rlc_status_resp_t                rlc_status;

    memset(&rlc, 0, sizeof(rlc));
    rlc.rnti = UE_RNTI(module_idP, UE_id);

    /* this code is wrong: we should loop over all configured RaB */

    /* DCCH   (srb 1, lcid 1) */
    rlc_status = mac_rlc_status_ind(module_idP, rlc.rnti, module_idP, frameP, ENB_FLAG_YES, MBMS_FLAG_NO, DCCH, 0);
    rlc.logicalChannelIdentity = DCCH;
    rlc.rlcTransmissionQueueSize = rlc_status.bytes_in_buffer;
    LOG_I(MAC, "calling SchedDlRlcBufferReq on DCCH rnti %x queue_size %d\n", rlc.rnti, rlc_status.bytes_in_buffer);
    SchedDlRlcBufferReq(fapi->sched, &rlc);
printf("MAC to FAPI downlink BUF DCCH %d\n", rlc_status.bytes_in_buffer);

    /* DCCH+1 (srb 2, lcid 2) */
    rlc_status = mac_rlc_status_ind(module_idP, rlc.rnti, module_idP, frameP, ENB_FLAG_YES, MBMS_FLAG_NO, DCCH+1, 0);
    rlc.logicalChannelIdentity = DCCH+1;
    rlc.rlcTransmissionQueueSize = rlc_status.bytes_in_buffer;
    LOG_I(MAC, "calling SchedDlRlcBufferReq on DCCH+1 rnti %x queue_size %d\n", rlc.rnti, rlc_status.bytes_in_buffer);
    SchedDlRlcBufferReq(fapi->sched, &rlc);
printf("MAC to FAPI downlink BUF DCCH+1 %d\n", rlc_status.bytes_in_buffer);

    /* DTCH   (drb 1, lcid 3) */
    rlc_status = mac_rlc_status_ind(module_idP, rlc.rnti, module_idP, frameP, ENB_FLAG_YES, MBMS_FLAG_NO, DTCH, 0);
    rlc.logicalChannelIdentity = DTCH;
    rlc.rlcTransmissionQueueSize = rlc_status.bytes_in_buffer;
    LOG_I(MAC, "calling SchedDlRlcBufferReq on DTCH rnti %x queue_size %d\n", rlc.rnti, rlc_status.bytes_in_buffer);
    SchedDlRlcBufferReq(fapi->sched, &rlc);
printf("MAC to FAPI downlink BUF DTCH %d\n", rlc_status.bytes_in_buffer);
  }

  dlreq.sfnSf                 = frameP * 16 + subframeP;
  dlreq.nr_dlInfoList         = 0;
  dlreq.dlInfoList            = NULL;
  dlreq.nr_vendorSpecificList = 0;
  dlreq.vendorSpecificList    = NULL;

  /* fill harq feedback data */
  /* TODO: deal with more than one TB */
  for (i = 0; i < fapi_dl_ack_nack_data.count; i++) {
    dlinfo[i].rnti          = fapi_dl_ack_nack_data.ack[i].rnti;
    dlinfo[i].harqProcessId = fapi_dl_ack_nack_data.ack[i].harq_pid;
    dlinfo[i].nr_harqStatus = 1;                                                         /* TODO: deal with more than 1 TB */
    dlinfo[i].harqStatus[0] = fapi_dl_ack_nack_data.ack[i].ack[0] ? ff_ACK : ff_NACK;    /* TODO: more than 1 TB */
    dlinfo[i].servCellIndex = 0;                                                         /* TODO: get real value for the servCellIndex */
printf("MAC to FAPI downlink ack/nack from PHY f/sf %d/%d rnti %x harq %d ack %d\n", frameP, subframeP, dlinfo[i].rnti, dlinfo[i].harqProcessId, fapi_dl_ack_nack_data.ack[i].ack[0]);
  }
  if (fapi_dl_ack_nack_data.count) {
    dlreq.nr_dlInfoList = fapi_dl_ack_nack_data.count;
    dlreq.dlInfoList    = dlinfo;
  }
  fapi_dl_ack_nack_data.count = 0;

  LOG_I(MAC, "calling SchedDlTriggerReq\n");
  SchedDlTriggerReq(fapi->sched, &dlreq);

  LOG_I(MAC, "calling SchedDlConfigInd\n");
  SchedDlConfigInd(fapi, &dlind);

  LOG_I(MAC, "SchedDlConfigInd returns dlind.nr_buildDataList %d f/sf %d/%d\n", dlind.nr_buildDataList, frameP, subframeP);
  LOG_I(MAC, "SchedDlConfigInd returns dlind.nr_buildRARList %d f/sf %d/%d\n", dlind.nr_buildRARList, frameP, subframeP);
  LOG_I(MAC, "SchedDlConfigInd returns dlind.nr_buildBroadcastList %d f/sf %d/%d\n", dlind.nr_buildBroadcastList, frameP, subframeP);

  /* TODO: rewrite. All should go into fapi_schedule_ue where special cases should be handled */
  for (i = 0; i < dlind.nr_buildDataList; i++) {
    if (dlind.buildDataList[i].ceBitmap[1] != 0 || dlind.buildDataList[i].nr_rlcPDU_List[1] != 0) { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
    if (dlind.buildDataList[i].nr_rlcPDU_List[0] != 1) { printf("%s:%d:%s: TODO\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
printf("FAPI to MAC downlink schedule ue %x channel %d f/sf %d/%d\n", dlind.buildDataList[i].rnti, dlind.buildDataList[i].rlcPduList[0][0].logicalChannelIdentity, frameP, subframeP);
    switch (dlind.buildDataList[i].rlcPduList[0][0].logicalChannelIdentity) {
    case 0: /* CCCH */
      /* TODO: get the right CC_id from servCellIndex, depending on the UE rnti/pcell/scell settings */
      CC_id = dlind.buildDataList[i].servCellIndex;
      /* look for an active RA with generate_Msg4 == 2 for this rnti */
      for (j=0; j < NB_RA_PROC_MAX; j++) {
        if (eNB->common_channels[CC_id].RA_template[j].RA_active == TRUE &&
            eNB->common_channels[CC_id].RA_template[j].generate_Msg4 == 2 &&
            eNB->common_channels[CC_id].RA_template[j].rnti == dlind.buildDataList[i].rnti) {
          if (dlind.buildDataList[i].ceBitmap[CC_id] != ff_CR) { printf("%s:%d:%s: TODO)\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
          fapi_schedule_RA_Msg4(module_idP, CC_id, j, frameP, subframeP,
              &dlind.buildDataList[i].dci, dlind.buildDataList[i].rlcPduList[0][0].size);
          break;
        }
      }
      if (j == NB_RA_PROC_MAX) { printf("%s:%d:%s: possible?\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
      break;
    case 1: /* DCCH   (SRB1) */
    case 2: /* DCCH+1 (SRB2) */
    case 3: /* DTCH   (DRB1) */
      /* TODO: get the correct CC_id from servCellIndex */
      fapi_schedule_ue(module_idP, dlind.buildDataList[i].servCellIndex /* CC_id */, frameP, subframeP, &dlind.buildDataList[i]);
      break;
    default: printf("%s:%d:%s: bad channel ID (%d)\n", __FILE__, __LINE__, __FUNCTION__, dlind.buildDataList[i].rlcPduList[0][0].logicalChannelIdentity); abort();
    }
  }

  if (dlind.nr_buildRARList) {
    if (dlind.nr_buildRARList != 1) { printf("%s:%d: more than 1 RAR, todo\n", __FUNCTION__, __LINE__); exit(0); }
    if (dlind.buildRarList[0].carrierIndex != 0) { printf("%s:%d: 2nd CC: todo properly\n", __FUNCTION__, __LINE__); exit(0); }
    /* force bit 0 of grant to 0
     * according to 36.213 6.2 the value
     * is reserved but openair needs it to be 0
     * this is for contention-based RA
     * maybe we will need to review that if/when we do contention-free RA
     */
printf("FAPI to MAC downlink schedule RAR ue %x f/sf %d/%d\n", dlind.buildRarList[0].rnti, frameP, subframeP);
    dlind.buildRarList[0].grant &= ~1;
    fapi_schedule_RAR(module_idP, dlind.buildRarList[0].carrierIndex, frameP, subframeP,
        dlind.buildRarList[0].rnti, dlind.buildRarList[0].grant, &dlind.buildRarList[0].dci);
  }

  if (dlind.nr_buildBroadcastList) {
    if (dlind.nr_buildBroadcastList != 1) { printf("%s:%d: more than 1 broadcast SI, todo\n", __FUNCTION__, __LINE__); exit(0); }
    if (dlind.buildBroadcastList[0].type == ff_PCCH) { printf("%s:%d: PCCH: todo\n", __FUNCTION__, __LINE__); exit(0); }
    if (dlind.buildBroadcastList[0].carrierIndex != 0) { printf("%s:%d: 2nd CC: todo properly\n", __FUNCTION__, __LINE__); exit(0); }
printf("FAPI to MAC downlink schedule SI %d f/sf %d/%d\n", dlind.buildBroadcastList[0].index, frameP, subframeP);
    fapi_schedule_SI(module_idP, dlind.buildBroadcastList[0].carrierIndex, frameP, subframeP,
        dlind.buildBroadcastList[0].index,
        &dlind.buildBroadcastList[0].dci);
  }

  /* TODO: do it correctly */
  if (dlind.nr_ofdmSymbolsCount != 0) {
    if (dlind.nr_ofdmSymbolsCount != 1) { printf("%s:%d:%s: what to do?\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
    for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
      if (dlind.nrOfPdcchOfdmSymbols[CC_id] != NULL) {
        int cc = dlind.nrOfPdcchOfdmSymbols[CC_id]->carrierIndex;
        DCI_pdu[cc]->num_pdcch_symbols = dlind.nrOfPdcchOfdmSymbols[CC_id]->pdcchOfdmSymbolCount;
printf("FAPI to MAC downlink DCI_pdu[%d]->num_pdcch_symbols %d f/sf %d/%d\n", cc, DCI_pdu[cc]->num_pdcch_symbols, frameP, subframeP);
      }
    }
  }

#if 0
  // Allocate CCEs for good after scheduling is done
  for (CC_id=0;CC_id<MAX_NUM_CCs;CC_id++)
    allocate_CCEs(module_idP,CC_id,subframeP,0);
#endif

  /*************************************************/
  /*            uplink scheduling                  */
  /*************************************************/

  ulreq.sfnSf         = frameP * 16 + subframeP;
  ulreq.nr_ulInfoList = 0;
  ulreq.ulInfoList    = NULL;
  ulreq.nr_vendorSpecificList = 0;
  ulreq.vendorSpecificList    = NULL;

  /* fill ulInfoList */
  ulsf = (subframeP + 10 - 3) % 10;
  if (fapi_ul_ack_nack_data[ulsf].count) {
    ulreq.nr_ulInfoList = fapi_ul_ack_nack_data[ulsf].count;
    ulreq.ulInfoList    = ulinfo;
    for (i = 0; i < ulreq.nr_ulInfoList; i++) {
printf("MAC to FAPI uplink acknack ue %x f/sf %d/%d ulsf %d [reception_subframe %d] ack %d\n", fapi_ul_ack_nack_data[ulsf].ack[i].rnti, frameP, subframeP, ulsf, fapi_ul_ack_nack_data[ulsf].ack[i].reception_subframe, fapi_ul_ack_nack_data[ulsf].ack[i].ack);
      ulinfo[i].puschTransmissionTimestamp = fapi_ul_ack_nack_data[ulsf].ack[i].reception_frame * 16
                                             + fapi_ul_ack_nack_data[ulsf].ack[i].reception_subframe;
      ulinfo[i].rnti                       = fapi_ul_ack_nack_data[ulsf].ack[i].rnti;
      ulinfo[i].receptionStatus            = fapi_ul_ack_nack_data[ulsf].ack[i].ack == 1 ? Ok : NotOk;
      ulinfo[i].tpc                        = 0;           /* TODO */
      ulinfo[i].servCellIndex              = 0;           /* TODO: get correct value */
      for (j = 0; j < MAX_LC_LIST+1; j++) {
        ulinfo[i].ulReception[j] = fapi_ul_ack_nack_data[ulsf].ack[i].length[j];
printf("MAC to FAPI uplink ue %x f/sf %d/%d lcid %d size acked %d\n", fapi_ul_ack_nack_data[ulsf].ack[i].rnti, frameP, subframeP, j, fapi_ul_ack_nack_data[ulsf].ack[i].length[j]);
}
    }
    fapi_ul_ack_nack_data[ulsf].count = 0;
  }

  LOG_I(MAC, "calling SchedUlTriggerReq\n");
  SchedUlTriggerReq(fapi->sched, &ulreq);

  LOG_I(MAC, "calling SchedUlConfigInd\n");
  SchedUlConfigInd(fapi, &ulind);

printf("FAPI to MAC uplink nr_dclList %d nr_phichList %d\n", ulind.nr_dciList, ulind.nr_phichList);
  for (i = 0; i < ulind.nr_dciList; i++) {
    /* TODO: get the right CC_id from servCellIndex, depending on the UE rnti/pcell/scell settings */
    CC_id = ulind.dciList[i].servCellIndex;
printf("FAPI to MAC uplink schedule ue %x ndi %d (fsf %d %d)\n", ulind.dciList[i].rnti, ulind.dciList[i].ndi, frameP, subframeP);
    fapi_schedule_uplink(module_idP, CC_id, &ulind.dciList[i]);
  }

printf("RECAP dci pdu count %d\n", eNB_mac_inst[0].common_channels[0].DCI_pdu.Num_common_dci);
for (i = 0; i < eNB_mac_inst[0].common_channels[0].DCI_pdu.Num_common_dci; i++) {
printf("    RECAP %i rnti %x %s dci pdu %s\n", i,
  eNB_mac_inst[0].common_channels[0].DCI_pdu.dci_alloc[i].rnti,
  dci_format_to_string(eNB_mac_inst[0].common_channels[0].DCI_pdu.dci_alloc[i].format),
  binary(*(uint32_t *)eNB_mac_inst[0].common_channels[0].DCI_pdu.dci_alloc[i].dci_pdu)
  );
}

  global_subframe++;
}

#else /* FAPI */

void eNB_dlsch_ulsch_scheduler(module_id_t module_idP,uint8_t cooperation_flag, frame_t frameP, sub_frame_t subframeP)  //, int calibration_flag) {
{

  int mbsfn_status[MAX_NUM_CCs];
  protocol_ctxt_t   ctxt;
#ifdef EXMIMO
  int ret;
#endif
#if defined(ENABLE_ITTI)
  MessageDef   *msg_p;
  const char   *msg_name;
  instance_t    instance;
  int           result;
#endif
  DCI_PDU *DCI_pdu[MAX_NUM_CCs];
  int CC_id,i,next_i;
  UE_list_t *UE_list=&eNB_mac_inst[module_idP].UE_list;
  rnti_t rnti;

  LOG_D(MAC,"[eNB %d] Frame %d, Subframe %d, entering MAC scheduler (UE_list->head %d)\n",module_idP, frameP, subframeP,UE_list->head);

  start_meas(&eNB_mac_inst[module_idP].eNB_scheduler);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_ENB_DLSCH_ULSCH_SCHEDULER,VCD_FUNCTION_IN);

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    DCI_pdu[CC_id] = &eNB_mac_inst[module_idP].common_channels[CC_id].DCI_pdu;
    DCI_pdu[CC_id]->nCCE=0;
    DCI_pdu[CC_id]->num_pdcch_symbols=1;
    mbsfn_status[CC_id]=0;
    // clear vrb_map
    memset(eNB_mac_inst[module_idP].common_channels[CC_id].vrb_map,0,100);
  }

  // refresh UE list based on UEs dropped by PHY in previous subframe
  i = UE_list->head;

  while (i>=0) {
    rnti = UE_RNTI(module_idP, i);
    CC_id = UE_PCCID(module_idP, i);
    LOG_D(MAC,"UE %d: rnti %x (%p)\n", i, rnti,
          mac_xface->get_eNB_UE_stats(module_idP, CC_id, rnti));
    next_i= UE_list->next[i];

    if (mac_xface->get_eNB_UE_stats(module_idP, CC_id, rnti)==NULL) {
      mac_remove_ue(module_idP, i, frameP, subframeP);
    }
    i = next_i;
  }

#if defined(ENABLE_ITTI)

  do {
    // Checks if a message has been sent to MAC sub-task
    itti_poll_msg (TASK_MAC_ENB, &msg_p);

    if (msg_p != NULL) {
      msg_name = ITTI_MSG_NAME (msg_p);
      instance = ITTI_MSG_INSTANCE (msg_p);

      switch (ITTI_MSG_ID(msg_p)) {
      case MESSAGE_TEST:
        LOG_D(MAC, "Received %s\n", ITTI_MSG_NAME(msg_p));
        break;

      case RRC_MAC_BCCH_DATA_REQ:
        LOG_D(MAC, "Received %s from %s: instance %d, frameP %d, eNB_index %d\n",
              msg_name, ITTI_MSG_ORIGIN_NAME(msg_p), instance,
              RRC_MAC_BCCH_DATA_REQ (msg_p).frame, RRC_MAC_BCCH_DATA_REQ (msg_p).enb_index);

        // TODO process BCCH data req.
        break;

      case RRC_MAC_CCCH_DATA_REQ:
        LOG_D(MAC, "Received %s from %s: instance %d, frameP %d, eNB_index %d\n",
              msg_name, ITTI_MSG_ORIGIN_NAME(msg_p), instance,
              RRC_MAC_CCCH_DATA_REQ (msg_p).frame, RRC_MAC_CCCH_DATA_REQ (msg_p).enb_index);

        // TODO process CCCH data req.
        break;

#ifdef Rel10

      case RRC_MAC_MCCH_DATA_REQ:
        LOG_D(MAC, "Received %s from %s: instance %d, frameP %d, eNB_index %d, mbsfn_sync_area %d\n",
              msg_name, ITTI_MSG_ORIGIN_NAME(msg_p), instance,
              RRC_MAC_MCCH_DATA_REQ (msg_p).frame, RRC_MAC_MCCH_DATA_REQ (msg_p).enb_index, RRC_MAC_MCCH_DATA_REQ (msg_p).mbsfn_sync_area);

        // TODO process MCCH data req.
        break;
#endif

      default:
        LOG_E(MAC, "Received unexpected message %s\n", msg_name);
        break;
      }

      result = itti_free (ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
      AssertFatal (result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
    }
  } while(msg_p != NULL);

#endif

  // clear DCI and BCCH contents before scheduling
  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    DCI_pdu[CC_id]->Num_common_dci  = 0;
    DCI_pdu[CC_id]->Num_ue_spec_dci = 0;


#ifdef Rel10
    eNB_mac_inst[module_idP].common_channels[CC_id].mcch_active =0;
#endif

    eNB_mac_inst[module_idP].frame    = frameP;
    eNB_mac_inst[module_idP].subframe = subframeP;


  }

  //if (subframeP%5 == 0)
  //#ifdef EXMIMO
  PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_idP, ENB_FLAG_YES, NOT_A_RNTI, frameP, 0,module_idP);
  pdcp_run(&ctxt);
  //#endif

  // check HO
  rrc_rx_tx(&ctxt,
            0, // eNB index, unused in eNB
            CC_id);

#ifdef Rel10

  for (CC_id=0; CC_id<MAX_NUM_CCs; CC_id++) {
    if (eNB_mac_inst[module_idP].common_channels[CC_id].MBMS_flag >0) {
      start_meas(&eNB_mac_inst[module_idP].schedule_mch);
      mbsfn_status[CC_id] = schedule_MBMS(module_idP,CC_id,frameP,subframeP);
      stop_meas(&eNB_mac_inst[module_idP].schedule_mch);
    }
  }

#endif
  // refresh UE list based on UEs dropped by PHY in previous subframe
  /*
  i=UE_list->head;
  while (i>=0) {
    next_i = UE_list->next[i];
    LOG_T(MAC,"UE %d : rnti %x, stats %p\n",i,UE_RNTI(module_idP,i),mac_xface->get_eNB_UE_stats(module_idP,0,UE_RNTI(module_idP,i)));
    if (mac_xface->get_eNB_UE_stats(module_idP,0,UE_RNTI(module_idP,i))==NULL) {
      mac_remove_ue(module_idP,i,frameP);
    }
    i=next_i;
  }
  */

  switch (subframeP) {
  case 0:

    // FDD/TDD Schedule Downlink RA transmissions (RA response, Msg4 Contention resolution)
    // Schedule ULSCH for FDD or subframeP 4 (TDD config 0,3,6)
    // Schedule Normal DLSCH


    schedule_RA(module_idP,frameP,subframeP,2);


    if (mac_xface->lte_frame_parms->frame_type == FDD) {  //FDD
      schedule_ulsch(module_idP,frameP,cooperation_flag,0,4);//,calibration_flag);
    } else if  ((mac_xface->lte_frame_parms->tdd_config == TDD) || //TDD
                (mac_xface->lte_frame_parms->tdd_config == 3) ||
                (mac_xface->lte_frame_parms->tdd_config == 6)) {
      //schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,4);//,calibration_flag);
    }

    schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
    fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);

    break;

  case 1:

    // TDD, schedule UL for subframeP 7 (TDD config 0,1) / subframeP 8 (TDD Config 6)
    // FDD, schedule normal UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == TDD) { // TDD
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 0:
      case 1:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,7);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 6:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,8);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD
      schedule_ulsch(module_idP,frameP,cooperation_flag,1,5);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 2:

    // TDD, nothing
    // FDD, normal UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == FDD) {  //FDD
      schedule_ulsch(module_idP,frameP,cooperation_flag,2,6);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 3:

    // TDD Config 2, ULSCH for subframeP 7
    // TDD Config 2/5 normal DLSCH
    // FDD, normal UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == TDD) {
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 2:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,7);

        // no break here!
      case 5:
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD

      schedule_ulsch(module_idP,frameP,cooperation_flag,3,7);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 4:

    // TDD Config 1, ULSCH for subframeP 8
    // TDD Config 1/2/4/5 DLSCH
    // FDD UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == 1) { // TDD
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 1:
        //        schedule_RA(module_idP,frameP,subframeP);
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,8);

        // no break here!
      case 2:

        // no break here!
      case 4:

        // no break here!
      case 5:

        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else {
      if (mac_xface->lte_frame_parms->frame_type == FDD) {  //FDD

	//        schedule_RA(module_idP,frameP, subframeP, 0);
	schedule_ulsch(module_idP, frameP, cooperation_flag, 4, 8);
	schedule_ue_spec(module_idP, frameP, subframeP,  mbsfn_status);
        fill_DLSCH_dci(module_idP, frameP, subframeP,   mbsfn_status);
      }
    }

    break;

  case 5:
    // TDD/FDD Schedule SI
    // TDD Config 0,6 ULSCH for subframes 9,3 resp.
    // TDD normal DLSCH
    // FDD normal UL/DLSCH
    schedule_SI(module_idP,frameP,subframeP);

    //schedule_RA(module_idP,frameP,subframeP,5);
    if (mac_xface->lte_frame_parms->frame_type == FDD) {
      schedule_RA(module_idP,frameP,subframeP,1);
      schedule_ulsch(module_idP,frameP,cooperation_flag,5,9);
      schedule_ue_spec(module_idP, frameP, subframeP,  mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    } else if ((mac_xface->lte_frame_parms->tdd_config == 0) || // TDD Config 0
               (mac_xface->lte_frame_parms->tdd_config == 6)) { // TDD Config 6
      //schedule_ulsch(module_idP,cooperation_flag,subframeP);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    } else {
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 6:

    // TDD Config 0,1,6 ULSCH for subframes 2,3
    // TDD Config 3,4,5 Normal DLSCH
    // FDD normal ULSCH/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == TDD) { // TDD
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 0:
        break;

      case 1:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,2);
        //  schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 6:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,3);
        //  schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 5:
        schedule_RA(module_idP,frameP,subframeP,2);
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 3:
      case 4:
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD
      //      schedule_ulsch(module_idP,frameP,cooperation_flag,6,0);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 7:

    // TDD Config 3,4,5 Normal DLSCH
    // FDD Normal UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == TDD) { // TDD
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 3:
      case 4:
        schedule_RA(module_idP,frameP,subframeP,3);  // 3 = Msg3 subframeP, not
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 5:
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD
      //schedule_ulsch(module_idP,frameP,cooperation_flag,7,1);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 8:

    // TDD Config 2,3,4,5 ULSCH for subframeP 2
    //
    // FDD Normal UL/DLSCH
    if (mac_xface->lte_frame_parms->frame_type == TDD) { // TDD
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 2:
      case 3:
      case 4:
      case 5:

        //  schedule_RA(module_idP,subframeP);
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,2);
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD
      //schedule_ulsch(module_idP,frameP,cooperation_flag,8,2);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  case 9:

    // TDD Config 1,3,4,6 ULSCH for subframes 3,3,3,4
    if (mac_xface->lte_frame_parms->frame_type == TDD) {
      switch (mac_xface->lte_frame_parms->tdd_config) {
      case 1:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,3);
        schedule_RA(module_idP,frameP,subframeP,7);  // 7 = Msg3 subframeP, not
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 3:
      case 4:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,3);
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 6:
        schedule_ulsch(module_idP,frameP,cooperation_flag,subframeP,4);
        //schedule_RA(module_idP,frameP,subframeP);
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      case 2:
      case 5:
        //schedule_RA(module_idP,frameP,subframeP);
        schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
        fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
        break;

      default:
        break;
      }
    } else { //FDD
      //     schedule_ulsch(module_idP,frameP,cooperation_flag,9,3);
      schedule_ue_spec(module_idP,frameP,subframeP,mbsfn_status);
      fill_DLSCH_dci(module_idP,frameP,subframeP,mbsfn_status);
    }

    break;

  }

  LOG_D(MAC,"FrameP %d, subframeP %d : Scheduling CCEs\n",frameP,subframeP);

  // Allocate CCEs for good after scheduling is done
  for (CC_id=0;CC_id<MAX_NUM_CCs;CC_id++)
    allocate_CCEs(module_idP,CC_id,subframeP,0);

  LOG_D(MAC,"frameP %d, subframeP %d\n",frameP,subframeP);

  stop_meas(&eNB_mac_inst[module_idP].eNB_scheduler);
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_ENB_DLSCH_ULSCH_SCHEDULER,VCD_FUNCTION_OUT);

}

#endif /* FAPI */
