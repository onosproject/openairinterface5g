#ifndef FF_MAC_H
#define FF_MAC_H

/** @defgroup _fapi  FAPI
 * @ingroup _mac
 * @{
 */


/* this file contains OAI related FAPI definitions */

/* this is the public view of the FAPI's OAI interface */
typedef struct {
  void *sched;     /* this is the pointer returned by SchedInit */
                   /* to be used when calling FAPI functions */
} fapi_interface_t;

/* this function initializes OAI's FAPI interfacing
 * it returns the opaque pointer given by SchedInit
 */
fapi_interface_t *init_fapi(void);

/* the following functions are called by OAI
 * they wait for the corresponding callback
 * to be called by the FAPI scheduler
 */

#include "ff-mac-sched-sap.h"
#include "ff-mac-csched-sap.h"

/* from SCHED */
void SchedDlConfigInd(fapi_interface_t *, struct SchedDlConfigIndParameters* params);
void SchedUlConfigInd(fapi_interface_t *, struct SchedUlConfigIndParameters* params);

/* from CSCHED */
void CschedCellConfigCnf(fapi_interface_t *, struct CschedCellConfigCnfParameters *params);
void CschedUeConfigCnf(fapi_interface_t *, struct CschedUeConfigCnfParameters *params);
void CschedLcConfigCnf(fapi_interface_t *, struct CschedLcConfigCnfParameters *params);
void CschedLcReleaseCnf(fapi_interface_t *, struct CschedLcReleaseCnfParameters *params);
void CschedUeReleaseCnf(fapi_interface_t *, struct CschedUeReleaseCnfParameters *params);
void CschedUeConfigUpdateInd(fapi_interface_t *, struct CschedUeConfigUpdateIndParameters *params);
void CschedCellConfigUpdateInd(fapi_interface_t *, struct CschedCellConfigUpdateIndParameters *params);

/*@}*/

#endif /* FF_MAC_H */
