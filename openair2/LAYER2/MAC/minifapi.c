/* this is a very primitive FAPI scheduler
 * it was done to develop the eNodeB side of FAPI
 * only one UE supported, MCS always 4, 25 RBs, 1 CC
 * and many other unrealistic assumptions
 */
#include "ff-mac-sched-sap.h"
#include "ff-mac-csched-sap.h"
#include "ff-mac-init.h"
#include "ff-mac-callback.h"

#include <stdlib.h>
#include <stdio.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

//#define LOG(...) do { printf("minifapi:%s:%d: ", __FUNCTION__, __LINE__); printf(__VA_ARGS__); printf("\n"); } while (0)
#define LOG(...) /**/

struct scheduler {
  void                                 *callback_data;
  SchedDlConfigInd_callback_t          *SchedDlConfigInd;
  SchedUlConfigInd_callback_t          *SchedUlConfigInd;
  CschedCellConfigCnf_callback_t       *CschedCellConfigCnf;
  CschedUeConfigCnf_callback_t         *CschedUeConfigCnf;
  CschedLcConfigCnf_callback_t         *CschedLcConfigCnf;
  CschedLcReleaseCnf_callback_t        *CschedLcReleaseCnf;
  CschedUeReleaseCnf_callback_t        *CschedUeReleaseCnf;
  CschedUeConfigUpdateInd_callback_t   *CschedUeConfigUpdateInd;
  CschedCellConfigUpdateInd_callback_t *CschedCellConfigUpdateInd;
  /* below data is for testing purpose - it's hackish */
  /* has_ue: hackishly set to 1 in schedule_ue_spec */
  int                                  has_ue;
  /* the RNTI of the UE */
  int                                  ue_rnti;
  /* the transmission queue size of each RLC logical channel for the UE */
  int                                  lc_rlc_queuesize[11];
  int                                  old_ndi;
};

/* hack function to set has_ue to 1 */
void has_ue(struct scheduler *s, int rnti)
{
  s->has_ue = 1;
  s->ue_rnti = rnti;
}

void *SchedInit(
    void                                 *callback_data,
    SchedDlConfigInd_callback_t          *SchedDlConfigInd,
    SchedUlConfigInd_callback_t          *SchedUlConfigInd,
    CschedCellConfigCnf_callback_t       *CschedCellConfigCnf,
    CschedUeConfigCnf_callback_t         *CschedUeConfigCnf,
    CschedLcConfigCnf_callback_t         *CschedLcConfigCnf,
    CschedLcReleaseCnf_callback_t        *CschedLcReleaseCnf,
    CschedUeReleaseCnf_callback_t        *CschedUeReleaseCnf,
    CschedUeConfigUpdateInd_callback_t   *CschedUeConfigUpdateInd,
    CschedCellConfigUpdateInd_callback_t *CschedCellConfigUpdateInd)
{
  struct scheduler *ret;

  LOG("enter");

  ret = calloc(1, sizeof(struct scheduler));
  if (ret == NULL) {
    printf("minifapi:%s:%d: memory allocation error\n", __FUNCTION__, __LINE__);
    return NULL;
  }

  ret->callback_data             = callback_data;
  ret->SchedDlConfigInd          = SchedDlConfigInd;
  ret->SchedUlConfigInd          = SchedUlConfigInd;
  ret->CschedCellConfigCnf       = CschedCellConfigCnf;
  ret->CschedUeConfigCnf         = CschedUeConfigCnf;
  ret->CschedLcConfigCnf         = CschedLcConfigCnf;
  ret->CschedLcReleaseCnf        = CschedLcReleaseCnf;
  ret->CschedUeReleaseCnf        = CschedUeReleaseCnf;
  ret->CschedUeConfigUpdateInd   = CschedUeConfigUpdateInd;
  ret->CschedCellConfigUpdateInd = CschedCellConfigUpdateInd;

  ret->has_ue = 0;

  LOG("leave");

  return ret;
}

/*************** SCHED functions ***************/

void SchedDlRlcBufferReq(void *_sched, const struct SchedDlRlcBufferReqParameters *params)
{
  struct scheduler                  *sched = _sched;

  LOG("enter lcid %d queue size %d", params->logicalChannelIdentity, params->rlcTransmissionQueueSize);

  sched->lc_rlc_queuesize[params->logicalChannelIdentity] = params->rlcTransmissionQueueSize;

  LOG("leave");
}

void SchedDlPagingBufferReq(void *_sched, const struct SchedDlPagingBufferReqParameters *params)
{
}

void SchedDlMacBufferReq(void *_sched, const struct SchedDlMacBufferReqParameters *params)
{
}

void SchedDlTriggerReq(void *_sched, const struct SchedDlTriggerReqParameters *params)
{
  struct scheduler                  *sched = _sched;
  struct SchedDlConfigIndParameters ind;
  struct BuildDataListElement_s     *d;
  int                               lcid = -1;

  LOG("enter frame %d subframe %d", params->sfnSf >> 4, params->sfnSf & 0xf);

  ind.nr_buildDataList = 0;
  ind.nr_buildRARList = 0;
  ind.nr_buildBroadcastList = 0;
  ind.nr_ofdmSymbolsCount = 0;
  ind.nr_vendorSpecificList = 0;
  ind.nr_ofdmSymbolsCount = 0;

  /* basic scheduler: only one UE accepted, we send max 20 RLC bytes,
   * only from one logical channel (srb1, srb2 or drb1, in that priority
   * order), 5MHz, SISO, no retransmission (ACK/NACK processing not done)
   */
  if (sched->lc_rlc_queuesize[1] == 0) goto no_srb1;
  lcid = 1;
  goto fill;

no_srb1:
  if (sched->lc_rlc_queuesize[2] == 0) goto no_srb2;
  lcid = 2;
  goto fill;

no_srb2:
  if (sched->lc_rlc_queuesize[3] == 0) goto no_sched;
  lcid = 3;

fill:
  ind.nr_ofdmSymbolsCount = 1;
  ind.nr_buildDataList = 1;
  d = &ind.buildDataList[0];
  d->rnti = sched->ue_rnti;
  /* fill DCI */
  d->dci.rnti        = d->rnti;
  d->dci.rbBitmap    = 0x1800;       /* 1 1000 0000 0000 allocate the 2 low RBG = 4 RBs */
  d->dci.rbShift     = 0;
  d->dci.resAlloc    = 0;
  d->dci.tbsSize[0]  = 32;
  d->dci.mcs[0]      = 4;
  d->dci.ndi[0]      = 1 - sched->old_ndi;   sched->old_ndi = 1 - sched->old_ndi;  /* always new */
  d->dci.rv[0]       = 0;
  d->dci.cceIndex    = 0;
  d->dci.aggrLevel   = 1;
  /* d->dci.precodingInfo not relevant */
  d->dci.format      = ONE;
  d->dci.tpc         = 0;
  d->dci.harqProcess = 0;      /* always use harq pid 0 */
  /* d->dci.dai not relevant */
  /* d->dci.vrbFormat not relevant */
  /* d->dci.tbSwap not relevant */
  d->dci.spsRelease  = 0;
  /* the rest of d->dci is not relevant */
  d->ceBitmap[0]    = 0;
  d->nr_rlcPDU_List = 1;
  d->rlcPduList[0][0].logicalChannelIdentity = lcid;
  d->rlcPduList[0][0].size                   = MIN(sched->lc_rlc_queuesize[lcid], 20);
  /* servCellIndex/activationDeactivationCE not set, not used for the moment */

no_sched:
  sched->SchedDlConfigInd(sched->callback_data, &ind);

  LOG("leave");
}

void SchedDlRachInfoReq(void *_sched, const struct SchedDlRachInfoReqParameters *params)
{
}

void SchedDlCqiInfoReq(void *_sched, const struct SchedDlCqiInfoReqParameters *params)
{
}

void SchedUlTriggerReq(void *_sched, const struct SchedUlTriggerReqParameters *params)
{
}

void SchedUlNoiseInterferenceReq(void *_sched, const struct SchedUlNoiseInterferenceReqParameters *params)
{
}

void SchedUlSrInfoReq(void *_sched, const struct SchedUlSrInfoReqParameters *params)
{
}

void SchedUlMacCtrlInfoReq(void *_sched, const struct SchedUlMacCtrlInfoReqParameters *params)
{
}

void SchedUlCqiInfoReq(void *_sched, const struct SchedUlCqiInfoReqParameters *params)
{
}

/*************** CSCHED functions ***************/

void CschedCellConfigReq(void *_sched, const struct CschedCellConfigReqParameters *params)
{
  struct scheduler *sched = _sched;
  struct CschedCellConfigCnfParameters conf;

  LOG("enter");

  conf.result = ff_SUCCESS;
  conf.nr_vendorSpecificList = 0;

  sched->CschedCellConfigCnf(sched->callback_data, &conf);

  LOG("leave");
}

void CschedUeConfigReq(void *_sched, const struct CschedUeConfigReqParameters *params)
{
}

void CschedLcConfigReq(void *_sched, const struct CschedLcConfigReqParameters *params)
{
}

void CschedLcReleaseReq(void *_sched, const struct CschedLcReleaseReqParameters *params)
{
}

void CschedUeReleaseReq(void *_sched, const struct CschedUeReleaseReqParameters *params)
{
}
