#ifndef FF_MAC_CALLBACK_H
#define FF_MAC_CALLBACK_H

/* these are the callback function types from the scheduler to openair */

/* sched */
typedef void (SchedDlConfigInd_callback)(void *callback_data, const struct SchedDlConfigIndParameters *params);
typedef void (SchedUlConfigInd_callback)(void *callback_data, const struct SchedUlConfigIndParameters *params);

/* csched */
typedef void (CschedCellConfigCnf_callback)(void *callback_data, const struct CschedCellConfigCnfParameters *params);
typedef void (CschedUeConfigCnf_callback)(void *callback_data, const struct CschedUeConfigCnfParameters *params);
typedef void (CschedLcConfigCnf_callback)(void *callback_data, const struct CschedLcConfigCnfParameters *params);
typedef void (CschedLcReleaseCnf_callback)(void *callback_data, const struct CschedLcReleaseCnfParameters *params);
typedef void (CschedUeReleaseCnf_callback)(void *callback_data, const struct CschedUeReleaseCnfParameters *params);
typedef void (CschedUeConfigUpdateInd_callback)(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params);
typedef void (CschedCellConfigUpdateInd_callback)(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params);

#endif /* FF_MAC_CALLBACK_H */
