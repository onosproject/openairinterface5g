/**
 * @file ff-mac-callback.h
 * @brief Implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 with extensions.
 * @details Contains definitions of function callback types, that the scheduler may call upon the MAC.
 * @author Florian Kaltenberger, Maciej Wewior
 * @date March 2015
 * @email: florian.kaltenberger@eurecom.fr, m.wewior@is-wireless.com
 * @ingroup _mac
 */

#ifndef FF_MAC_CALLBACK_H
#define FF_MAC_CALLBACK_H

#if defined (__cplusplus)
extern "C" {
#endif

//CSCHED SAP scheduler->MAC primitives

/**
 * Cell configuration and scheduler configuration applied.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params Cell configuration result
 */
typedef void (CschedCellConfigCnf_callback_t)(void *callback_data, const struct CschedCellConfigCnfParameters *params);

/**
 * UE specific configuration applied.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params UE configuration result
 */
typedef void (CschedUeConfigCnf_callback_t)(void *callback_data, const struct CschedUeConfigCnfParameters *params);

/**
 * Logical channel configuration applied.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params LC configuration result
 */
typedef void (CschedLcConfigCnf_callback_t)(void *callback_data, const struct CschedLcConfigCnfParameters *params);

/**
 * Logical Channel specific configuration removed.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params Logical channel release result
 */
typedef void (CschedLcReleaseCnf_callback_t)(void *callback_data, const struct CschedLcReleaseCnfParameters *params);

/**
 * UE specific configuration removed.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params UE release result
 */
typedef void (CschedUeReleaseCnf_callback_t)(void *callback_data, const struct CschedUeReleaseCnfParameters *params);

/**
 * Update of UE specific parameters from MAC scheduler to RRC.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params UE configuration update parameters
 */
typedef void (CschedUeConfigUpdateInd_callback_t)(void *callback_data, const struct CschedUeConfigUpdateIndParameters *params);

/**
 * Update of cell configuration from MAC scheduler to RRC.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params Cell configuration update parameters
 */
typedef void (CschedCellConfigUpdateInd_callback_t)(void *callback_data, const struct CschedCellConfigUpdateIndParameters *params);


//SCHED SAP scheduler->MAC primitives

/**
 * Passes the DL scheduling decision to MAC, triggers building of DL MAC PDUs and Subframe Configuration.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params DL scheduling decision
 */
typedef void (SchedDlConfigInd_callback_t)(void *callback_data, const struct SchedDlConfigIndParameters *params);

/**
 * Passes the UL scheduling decision (Format 0 DCIs)to MAC.
 * @param callback_data Callback data pointer provided in SchedInit()
 * @param params UL scheduling decision
 */
typedef void (SchedUlConfigInd_callback_t)(void *callback_data, const struct SchedUlConfigIndParameters *params);

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_CALLBACK_H */
