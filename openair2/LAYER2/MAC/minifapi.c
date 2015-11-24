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

struct scheduler {
  void *callback_data;
  SchedDlConfigInd_callback          *SchedDlConfigInd;
  SchedUlConfigInd_callback          *SchedUlConfigInd;
  CschedCellConfigCnf_callback       *CschedCellConfigCnf;
  CschedUeConfigCnf_callback         *CschedUeConfigCnf;
  CschedLcConfigCnf_callback         *CschedLcConfigCnf;
  CschedLcReleaseCnf_callback        *CschedLcReleaseCnf;
  CschedUeReleaseCnf_callback        *CschedUeReleaseCnf;
  CschedUeConfigUpdateInd_callback   *CschedUeConfigUpdateInd;
  CschedCellConfigUpdateInd_callback *CschedCellConfigUpdateInd;
};

void *SchedInit(
    void                               *callback_data,
    SchedDlConfigInd_callback          *SchedDlConfigInd,
    SchedUlConfigInd_callback          *SchedUlConfigInd,
    CschedCellConfigCnf_callback       *CschedCellConfigCnf,
    CschedUeConfigCnf_callback         *CschedUeConfigCnf,
    CschedLcConfigCnf_callback         *CschedLcConfigCnf,
    CschedLcReleaseCnf_callback        *CschedLcReleaseCnf,
    CschedUeReleaseCnf_callback        *CschedUeReleaseCnf,
    CschedUeConfigUpdateInd_callback   *CschedUeConfigUpdateInd,
    CschedCellConfigUpdateInd_callback *CschedCellConfigUpdateInd)
{
  struct scheduler *ret;

  printf("minifapi:%s:%d: start\n", __FUNCTION__, __LINE__);

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
  ret-> CschedLcReleaseCnf       = CschedLcReleaseCnf;
  ret->CschedUeReleaseCnf        = CschedUeReleaseCnf;
  ret->CschedUeConfigUpdateInd   = CschedUeConfigUpdateInd;
  ret->CschedCellConfigUpdateInd = CschedCellConfigUpdateInd;

  printf("minifapi:%s:%d: done\n", __FUNCTION__, __LINE__);

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
