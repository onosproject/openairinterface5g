#include "ff-mac.h"
#include "ff-mac-sched-sap.h"
#include "ff-mac-csched-sap.h"
#include "ff-mac-init.h"
#include "log.h"
#include "assertions.h"

#undef LOG_D
#define LOG_D LOG_I

#include <stdlib.h>
#include <pthread.h>
#include <errno.h>

/* callback IDs */
#define SCHED_DL_CONFIG_IND           0
#define SCHED_UL_CONFIG_IND           1
#define CSCHED_CELL_CONFIG_CNF        2
#define CSCHED_UE_CONFIG_CNF          3
#define CSCHED_LC_CONFIG_CNF          4
#define CSCHED_LC_RELEASE_CNF         5
#define CSCHED_UE_RELEASE_CNF         6
#define CSCHED_UE_CONFIG_UPDATE_IND   7
#define CSCHED_CELL_CONFIG_UPDATE_IND 8
#define N_IDs 9

/* this structure stores required data for OAI to work with FAPI */
/* it is the private version of fapi_interface_t */
struct fapi {
  fapi_interface_t fi;   /* the start of the structure matches fapi_interface_t */
  pthread_mutex_t mutex[N_IDs];
  pthread_cond_t cond[N_IDs];
  volatile int req_id[N_IDs];
  volatile int rsp_id[N_IDs];
  struct CschedCellConfigCnfParameters CschedCellConfigCnfParameters;
  struct SchedDlConfigIndParameters SchedDlConfigIndParameters;
  struct SchedUlConfigIndParameters SchedUlConfigIndParameters;
  struct CschedUeConfigCnfParameters CschedUeConfigCnfParameters;
  struct CschedLcConfigCnfParameters CschedLcConfigCnfParameters;
};

#define LOCK(fi, fn) do { \
    LOG_D(MAC, "%s: locking fn %d fi %p mutex %p\n", __FUNCTION__, fn, fi, &fi->mutex[fn]); \
    if (pthread_mutex_lock(&fi->mutex[fn])) \
      AssertFatal(0, "%s:%d:%s: mutex error\n", __FILE__, __LINE__, __FUNCTION__); \
  } while (0)

#define UNLOCK(fi, fn) do { \
    LOG_D(MAC, "%s: unlocking fn %d fi %p mutex %p\n", __FUNCTION__, fn, fi, &fi->mutex[fn]); \
    if (pthread_mutex_unlock(&fi->mutex[fn])) \
      AssertFatal(0, "%s:%d:%s: mutex error\n", __FILE__, __LINE__, __FUNCTION__); \
  } while (0)

#define CHECK(fi, fn) do { \
    if (fi->req_id[fn] != fi->rsp_id[fn]) \
      AssertFatal(0, "%s:%d:%s: check error red_id %d rsp_id %d\n", __FILE__, __LINE__, __FUNCTION__, fi->req_id[fn], fi->rsp_id[fn]); \
  } while (0)

#define WAIT(fi, fn) do { \
    LOG_D(MAC, "%s: WAIT fn %d req %d rsp %d\n", __FUNCTION__, fn, fi->req_id[fn], fi->rsp_id[fn]); \
    while (fi->req_id[fn] == fi->rsp_id[fn]) \
      if (pthread_cond_wait(&fi->cond[fn], &fi->mutex[fn])) \
        AssertFatal(0, "%s:%d:%s: cond error\n", __FILE__, __LINE__, __FUNCTION__); \
  } while (0)

#define WAIT_TIMEOUT(fi, fn, delay_ms, has_timed_out) do { \
    int ret = 0; \
    has_timed_out = 0; \
    LOG_D(MAC, "%s: WAIT fn %d req %d rsp %d\n", __FUNCTION__, fn, fi->req_id[fn], fi->rsp_id[fn]); \
    struct timespec tout; \
    if (clock_gettime(CLOCK_REALTIME, &tout)) \
      AssertFatal(0, "%s:%d:%s: clock_gettime error\n", __FILE__, __LINE__, __FUNCTION__); \
    tout.tv_nsec += (unsigned long)(delay_ms) * 1000000; \
    while (tout.tv_nsec > 999999999) { \
      tout.tv_sec++; \
      tout.tv_nsec -= 1000000000; \
    } \
    while (ret == 0 && fi->req_id[fn] == fi->rsp_id[fn]) \
      ret = pthread_cond_timedwait(&fi->cond[fn], &fi->mutex[fn], &tout); \
    if (ret && ret != ETIMEDOUT) \
      AssertFatal(0, "%s:%d:%s: cond error\n", __FILE__, __LINE__, __FUNCTION__); \
    has_timed_out = ret == ETIMEDOUT; \
  } while (0)

#define DONE_callback(fi, fn) do { \
    fi->req_id[fn]++; \
/* printf("DONE_callback: req id %d rsp id %d\n", fi->req_id[fn], fi->rsp_id[fn]); */ \
    if (pthread_cond_signal(&fi->cond[fn])) \
      AssertFatal(0, "%s:%d:%s: mutex error\n", __FILE__, __LINE__, __FUNCTION__); \
  } while (0)

#define DONE_wrapper(fi, fn) do { \
    fi->rsp_id[fn]++; \
  } while (0)

/* SCHED "wrappers" */

void SchedDlConfigInd(fapi_interface_t *_fi, struct SchedDlConfigIndParameters *params)
{
  struct fapi *fi = (struct fapi *)_fi;
  int fn = SCHED_DL_CONFIG_IND;
  int has_timed_out;

  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  //WAIT_TIMEOUT(fi, fn, 100, has_timed_out);
  WAIT(fi, fn);

#if 0
  if (has_timed_out) {
    LOG_E(MAC, "SchedDlConfigInd timed out\n");
    memset(params, 0, sizeof(*params));
    goto end;
  }
#endif

  *params = fi->SchedDlConfigIndParameters;

  DONE_wrapper(fi, fn);
end:
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void SchedUlConfigInd(fapi_interface_t *_fi, struct SchedUlConfigIndParameters *params)
{
  struct fapi *fi = (struct fapi *)_fi;
  int fn = SCHED_UL_CONFIG_IND;

  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  WAIT(fi, fn);

  *params = fi->SchedUlConfigIndParameters;

  DONE_wrapper(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

/* CSCHED "wrappers" */

void CschedCellConfigCnf(fapi_interface_t *_fi, struct CschedCellConfigCnfParameters *params)
{
  struct fapi *fi = (struct fapi *)_fi;
  int fn = CSCHED_CELL_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  WAIT(fi, fn);

  *params = fi->CschedCellConfigCnfParameters;

  DONE_wrapper(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedUeConfigCnf(fapi_interface_t *_fi, struct CschedUeConfigCnfParameters *params)
{
  struct fapi *fi = (struct fapi *)_fi;
  int fn = CSCHED_UE_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  WAIT(fi, fn);

  *params = fi->CschedUeConfigCnfParameters;

  DONE_wrapper(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedLcConfigCnf(fapi_interface_t *_fi, struct CschedLcConfigCnfParameters *params)
{
  struct fapi *fi = (struct fapi *)_fi;
  int fn = CSCHED_LC_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  WAIT(fi, fn);

  *params = fi->CschedLcConfigCnfParameters;

  DONE_wrapper(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedLcReleaseCnf(fapi_interface_t *_fi, struct CschedLcReleaseCnfParameters *params)
{
  int fn = CSCHED_LC_RELEASE_CNF;
}

void CschedUeReleaseCnf(fapi_interface_t *_fi, struct CschedUeReleaseCnfParameters *params)
{
  int fn = CSCHED_UE_RELEASE_CNF;
}

void CschedUeConfigUpdateInd(fapi_interface_t *_fi, struct CschedUeConfigUpdateIndParameters *params)
{
  int fn = CSCHED_UE_CONFIG_UPDATE_IND;
}

void CschedCellConfigUpdateInd(fapi_interface_t *_fi, struct CschedCellConfigUpdateIndParameters *params)
{
  int fn = CSCHED_CELL_CONFIG_UPDATE_IND;
}

/* SCHED callbacks */

void SchedDlConfigInd_callback(void *callback_data, const struct SchedDlConfigIndParameters *params)
{
  struct fapi *fi = callback_data;
  int fn = SCHED_DL_CONFIG_IND;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  CHECK(fi, fn);

  fi->SchedDlConfigIndParameters = *params;

  DONE_callback(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void SchedUlConfigInd_callback(void *callback_data, const struct SchedUlConfigIndParameters *params)
{
  struct fapi *fi = callback_data;
  int fn = SCHED_UL_CONFIG_IND;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  CHECK(fi, fn);

  fi->SchedUlConfigIndParameters = *params;

  DONE_callback(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

/* CSCHED callbacks */

void CschedCellConfigCnf_callback(void *callback_data, const struct CschedCellConfigCnfParameters *params)
{
  struct fapi *fi = callback_data;
  int fn = CSCHED_CELL_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  CHECK(fi, fn);

  fi->CschedCellConfigCnfParameters = *params;

  DONE_callback(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedUeConfigCnf_callback(void *callback_data, const struct CschedUeConfigCnfParameters *params)
{
  struct fapi *fi = callback_data;
  int fn = CSCHED_UE_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  CHECK(fi, fn);

  fi->CschedUeConfigCnfParameters = *params;

  DONE_callback(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedLcConfigCnf_callback(void *callback_data, const struct CschedLcConfigCnfParameters *params)
{
  struct fapi *fi = callback_data;
  int fn = CSCHED_LC_CONFIG_CNF;
  LOG_D(MAC, "%s enter\n", __FUNCTION__);

  LOCK(fi, fn);
  CHECK(fi, fn);

  fi->CschedLcConfigCnfParameters = *params;

  DONE_callback(fi, fn);
  UNLOCK(fi, fn);

  LOG_D(MAC, "%s leave\n", __FUNCTION__);
}

void CschedLcReleaseCnf_callback(void *callback_data, const struct CschedLcReleaseCnfParameters *params)
{
  int fn = CSCHED_LC_RELEASE_CNF;
abort();
}

void CschedUeReleaseCnf_callback(void *callback_data, const struct CschedUeReleaseCnfParameters *params)
{
  int fn = CSCHED_UE_RELEASE_CNF;
abort();
}

void CschedUeConfigUpdateInd_callback(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params)
{
  int fn = CSCHED_UE_CONFIG_UPDATE_IND;
abort();
}

void CschedCellConfigUpdateInd_callback(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params)
{
  int fn = CSCHED_CELL_CONFIG_UPDATE_IND;
abort();
}

fapi_interface_t *init_fapi(void)
{
  struct fapi *ret;
  int i;

  LOG_I(MAC, "FAPI initialization\n");

  ret = calloc(1, sizeof(struct fapi));
  if (ret == NULL) {
    LOG_E(MAC, "init_fapi: memory allocation error\n");
    return NULL;
  }

  for (i = 0; i < N_IDs; i++) {
    if (pthread_mutex_init(&ret->mutex[i], NULL)) {
      LOG_E(MAC, "init_fapi: mutex init error\n");
      exit(1);
    }
    if (pthread_cond_init(&ret->cond[i], NULL)) {
      LOG_E(MAC, "init_fapi: cond init error\n");
      exit(1);
    }
    ret->req_id[i] = 0;
    ret->rsp_id[i] = 0;
  }

  ret->fi.sched = SchedInit(ret,
                      SchedDlConfigInd_callback,
                      SchedUlConfigInd_callback,
                      CschedCellConfigCnf_callback,
                      CschedUeConfigCnf_callback,
                      CschedLcConfigCnf_callback,
                      CschedLcReleaseCnf_callback,
                      CschedUeReleaseCnf_callback,
                      CschedUeConfigUpdateInd_callback,
                      CschedCellConfigUpdateInd_callback);

  if (ret->fi.sched == NULL) {
    LOG_E(MAC, "init_fapi: SchedInit failed\n");
    free(ret);
    return NULL;
  }

  return (fapi_interface_t *)ret;
}
