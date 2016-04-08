#ifndef FF_MAC_INIT_H
#define FF_MAC_INIT_H

#if defined (__cplusplus)
extern "C" {
#endif

#include "ff-mac-callback.h"

/* this function is called to create and initialize a scheduler */
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
    CschedCellConfigUpdateInd_callback_t *CschedCellConfigUpdateInd);

void SchedShutdown(void* scheduler);

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_INIT_H */
