#ifndef FF_MAC_INIT_H
#define FF_MAC_INIT_H

#if defined (__cplusplus)
extern "C" {
#endif

#include "ff-mac-callback.h"

/* this function is called to create and initialize a scheduler */
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
    CschedCellConfigUpdateInd_callback *CschedCellConfigUpdateInd);

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_INIT_H */
