/*
gcc -Wall fapi_replay.c ./libscheduler.a -pthread -lm -g -o fapi_replay
 */

#define MAX_NUM_CCs 2

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "ff-mac.h"
#include "ff-mac-csched-sap.h"
#include "ff-mac-sched-sap.h"
#include "ff-mac-init.h"

#undef CschedCellConfigReq
#undef CschedUeConfigReq
#undef CschedLcConfigReq
#undef CschedLcReleaseReq
#undef CschedUeReleaseReq
#undef SchedDlRlcBufferReq
#undef SchedDlPagingBufferReq
#undef SchedDlMacBufferReq
#undef SchedDlTriggerReq
#undef SchedDlRachInfoReq
#undef SchedDlCqiInfoReq
#undef SchedUlTriggerReq
#undef SchedUlNoiseInterferenceReq
#undef SchedUlSrInfoReq
#undef SchedUlMacCtrlInfoReq
#undef SchedUlCqiInfoReq

int pipe_queue_read;
int pipe_queue_write;

void ok(int x)
{
  if (write(pipe_queue_write, &x, sizeof(int)) != sizeof(int)) abort();
}

void wt(int x)
{
  int y;
  if (read(pipe_queue_read, &y, sizeof(int)) != sizeof(int)) abort();
  if (x != y) { printf("possible?\n"); abort(); }
}

enum __ {
e_SchedDlConfigInd,
e_SchedUlConfigInd,
e_CschedCellConfigCnf,
e_CschedUeConfigCnf,
e_CschedLcConfigCnf,
e_CschedLcReleaseCnf,
e_CschedUeReleaseCnf,
e_CschedUeConfigUpdateInd,
e_CschedCellConfigUpdateInd
};

void SchedDlConfigInd(fapi_interface_t *x, struct SchedDlConfigIndParameters* params) { wt(e_SchedDlConfigInd); }
void SchedUlConfigInd(fapi_interface_t *x, struct SchedUlConfigIndParameters* params) { wt(e_SchedUlConfigInd); }
void CschedCellConfigCnf(fapi_interface_t *x, struct CschedCellConfigCnfParameters *params) { wt(e_CschedCellConfigCnf); }
void CschedUeConfigCnf(fapi_interface_t *x, struct CschedUeConfigCnfParameters *params) { wt(e_CschedUeConfigCnf); }
void CschedLcConfigCnf(fapi_interface_t *x, struct CschedLcConfigCnfParameters *params) { wt(e_CschedLcConfigCnf); }
void CschedLcReleaseCnf(fapi_interface_t *x, struct CschedLcReleaseCnfParameters *params) { wt(e_CschedLcReleaseCnf); }
void CschedUeReleaseCnf(fapi_interface_t *x, struct CschedUeReleaseCnfParameters *params) { wt(e_CschedUeReleaseCnf); }
void CschedUeConfigUpdateInd(fapi_interface_t *x, struct CschedUeConfigUpdateIndParameters *params) { wt(e_CschedUeConfigUpdateInd); }
void CschedCellConfigUpdateInd(fapi_interface_t *x, struct CschedCellConfigUpdateIndParameters *params) { wt(e_CschedCellConfigUpdateInd); }

void SchedDlConfigInd_callback(void *callback_data, const struct SchedDlConfigIndParameters *params) { ok(e_SchedDlConfigInd); }
void SchedUlConfigInd_callback(void *callback_data, const struct SchedUlConfigIndParameters *params) { ok(e_SchedUlConfigInd); }
void CschedCellConfigCnf_callback(void *callback_data, const struct CschedCellConfigCnfParameters *params) { ok(e_CschedCellConfigCnf); }
void CschedUeConfigCnf_callback(void *callback_data, const struct CschedUeConfigCnfParameters *params) { ok(e_CschedUeConfigCnf); }
void CschedLcConfigCnf_callback(void *callback_data, const struct CschedLcConfigCnfParameters *params) { ok(e_CschedLcConfigCnf); }
void CschedLcReleaseCnf_callback(void *callback_data, const struct CschedLcReleaseCnfParameters *params) { ok(e_CschedLcReleaseCnf); }
void CschedUeReleaseCnf_callback(void *callback_data, const struct CschedUeReleaseCnfParameters *params) { ok(e_CschedUeReleaseCnf); }
void CschedUeConfigUpdateInd_callback(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params) { abort(); }
void CschedCellConfigUpdateInd_callback(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params) { abort(); }


int main(void)
{
  int p[2];
  if (pipe(p)) abort();
  pipe_queue_read = p[0];
  pipe_queue_write = p[1];

  void *y;
  void * x = SchedInit(&y,
                      SchedDlConfigInd_callback,
                      SchedUlConfigInd_callback,
                      CschedCellConfigCnf_callback,
                      CschedUeConfigCnf_callback,
                      CschedLcConfigCnf_callback,
                      CschedLcReleaseCnf_callback,
                      CschedUeReleaseCnf_callback,
                      CschedUeConfigUpdateInd_callback,
                      CschedCellConfigUpdateInd_callback);
#include "/tmp/fapi.c"
  return 0;
}
