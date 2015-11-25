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

#define LOG(...) do { printf("minifapi:%s:%d: ", __FUNCTION__, __LINE__); printf(__VA_ARGS__); printf("\n"); } while (0)

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
};

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

  LOG("leave");

  return ret;
}

/*************** SCHED functions ***************/

void SchedDlRlcBufferReq(void *_sched, const struct SchedDlRlcBufferReqParameters *params)
{
}

void SchedDlPagingBufferReq(void *_sched, const struct SchedDlPagingBufferReqParameters *params)
{
}

void SchedDlMacBufferReq(void *_sched, const struct SchedDlMacBufferReqParameters *params)
{
}

void SchedDlTriggerReq(void *_sched, const struct SchedDlTriggerReqParameters *params)
{
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
