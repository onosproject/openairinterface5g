#ifndef FF_MAC_CALLBACK_H
#define FF_MAC_CALLBACK_H

/*!
 * @ingroup _fapi
 */


#if defined (__cplusplus)
extern "C" {
#endif

/* these are the callback function types from the scheduler to openair */

/* sched */
typedef void (SchedDlConfigInd_callback_t)(void *callback_data, const struct SchedDlConfigIndParameters *params);
typedef void (SchedUlConfigInd_callback_t)(void *callback_data, const struct SchedUlConfigIndParameters *params);

/* csched */
typedef void (CschedCellConfigCnf_callback_t)(void *callback_data, const struct CschedCellConfigCnfParameters *params);
typedef void (CschedUeConfigCnf_callback_t)(void *callback_data, const struct CschedUeConfigCnfParameters *params);
typedef void (CschedLcConfigCnf_callback_t)(void *callback_data, const struct CschedLcConfigCnfParameters *params);
typedef void (CschedLcReleaseCnf_callback_t)(void *callback_data, const struct CschedLcReleaseCnfParameters *params);
typedef void (CschedUeReleaseCnf_callback_t)(void *callback_data, const struct CschedUeReleaseCnfParameters *params);
typedef void (CschedUeConfigUpdateInd_callback_t)(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params);
typedef void (CschedCellConfigUpdateInd_callback_t)(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params);

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_CALLBACK_H */
