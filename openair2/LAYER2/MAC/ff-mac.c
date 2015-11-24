#include "ff-mac.h"
#include "ff-mac-sched-sap.h"
#include "ff-mac-csched-sap.h"
#include "ff-mac-init.h"
#include "log.h"

#include <stdlib.h>

/* this structure stores required data for OAI to work with FAPI */
/* it is the private version of fapi_interface_t */
struct fapi {
  fapi_interface_t fi;   /* the start of the structure matches fapi_interface_t */
};

/* here come the callbacks */
void SchedDlConfigInd_callback(void *callback_data, const struct SchedDlConfigIndParameters *params)
{
}

void SchedUlConfigInd_callback(void *callback_data, const struct SchedUlConfigIndParameters *params)
{
}

void CschedCellConfigCnf_callback(void *callback_data, const struct CschedCellConfigCnfParameters *params)
{
}

void CschedUeConfigCnf_callback(void *callback_data, const struct CschedUeConfigCnfParameters *params)
{
}

void CschedLcConfigCnf_callback(void *callback_data, const struct CschedLcConfigCnfParameters *params)
{
}

void CschedLcReleaseCnf_callback(void *callback_data, const struct CschedLcReleaseCnfParameters *params)
{
}

void CschedUeReleaseCnf_callback(void *callback_data, const struct CschedUeReleaseCnfParameters *params)
{
}

void CschedUeConfigUpdateInd_callback(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params)
{
}

void CschedCellConfigUpdateInd_callback(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params)
{
}

fapi_interface_t *init_fapi(void)
{
  struct fapi *ret;

  LOG_I(MAC, "FAPI initialization\n");

  ret = calloc(1, sizeof(struct fapi));
  if (ret == NULL) LOG_E(MAC, "init_fapi: memory allocation error\n");

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
