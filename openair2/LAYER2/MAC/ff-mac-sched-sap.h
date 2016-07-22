/**
 * @file ff-mac-sched-sap.h
 * @brief Implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 with extensions.
 * @details Contains SCHED SAP MAC->scheduler primitive declarations and all primitives data structures definitions.
 * @author Florian Kaltenberger, Maciej Wewior
 * @date March 2015
 * @email: florian.kaltenberger@eurecom.fr, m.wewior@is-wireless.com
 * @ingroup _mac
 */

#ifndef FF_MAC_SCHED_SAP_H
#define FF_MAC_SCHED_SAP_H

#include <stdint.h>
#include <stdbool.h>

#include "ff-mac-common.h"

#if defined (__cplusplus)
extern "C" {
#endif

//forward declarations
struct SchedDlRlcBufferReqParameters;
struct SchedDlPagingBufferReqParameters;
struct SchedDlMacBufferReqParameters;
struct SchedDlTriggerReqParameters;
struct SchedDlRachInfoReqParameters;
struct SchedDlCqiInfoReqParameters;
struct SchedUlTriggerReqParameters;
struct SchedUlNoiseInterferenceReqParameters;
struct SchedUlSrInfoReqParameters;
struct SchedUlMacCtrlInfoReqParameters;
struct SchedUlCqiInfoReqParameters;

//SCHED SAP MAC->scheduler primitives

/**
 * Update buffer status of logical channel data in RLC. The update rate with which the buffer status is updated in the scheduler is outside of the scope of the document.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params RLC buffer status
 */
void SchedDlRlcBufferReq (void * scheduler, const struct SchedDlRlcBufferReqParameters *params);

/**
 * Update buffer status of paging messages.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params Paging buffer status
 */
void SchedDlPagingBufferReq (void * scheduler, const struct SchedDlPagingBufferReqParameters *params);

/**
 * Update buffer status of MAC control elements. The update rate with which the buffer status is updated in the scheduler is outside of the scope of the document.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params MAC buffer status
 */
void SchedDlMacBufferReq (void * scheduler, const struct SchedDlMacBufferReqParameters *params);

/**
 * Starts the DL MAC scheduler for this subframe.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params DL HARQ information
 */
void SchedDlTriggerReq (void * scheduler, const struct SchedDlTriggerReqParameters *params);

/**
 * Provides RACH reception information to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params RACH reception information
 */
void SchedDlRachInfoReq (void * scheduler, const struct SchedDlRachInfoReqParameters *params);

/**
 * Provides CQI measurement report information to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params DL CQI information
 */
void SchedDlCqiInfoReq (void * scheduler, const struct SchedDlCqiInfoReqParameters *params);

/**
 * Starts the UL MAC scheduler for this subframe.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params  UL HARQ information
 */
void SchedUlTriggerReq (void * scheduler, const struct SchedUlTriggerReqParameters *params);

/**
 * Provides noise and interference measurement information to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params Noise and interference measurements
 */
void SchedUlNoiseInterferenceReq (void * scheduler, const struct SchedUlNoiseInterferenceReqParameters *params);

/**
 * Provides scheduling request reception information to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params Scheduling request information.
 */
void SchedUlSrInfoReq (void * scheduler, const struct SchedUlSrInfoReqParameters *params);

/**
 * Provides mac control information (power headroom, ul buffer status) to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params MAC control information received
 */
void SchedUlMacCtrlInfoReq (void * scheduler, const struct SchedUlMacCtrlInfoReqParameters *params);

/**
 * Provides UL CQI measurement information to the scheduler.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params UL CQI information
 */
void SchedUlCqiInfoReq (void * scheduler, const struct SchedUlCqiInfoReqParameters *params);

//SCHED primitives parameters

/**
 * RLC buffer status.
 */
struct SchedDlRlcBufferReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The logical channel ID, see \ref ref1 "[1]".
	 * \n Range: 0..10
	 */
	uint8_t logicalChannelIdentity;

	/**
	 * The current size of the transmission queue in byte.
	 */
	uint32_t  rlcTransmissionQueueSize;

	/**
	 * Head of line delay of new transmissions in ms.
	 */
	uint16_t rlcTransmissionQueueHolDelay;

	/**
	 * The current size of the retransmission queue in byte.
	 */
	uint32_t rlcRetransmissionQueueSize;

	/**
	 * Head of line delay of retransmissions in ms.
	 */
	uint16_t rlcRetransmissionHolDelay;

	/**
	 * The current size of the pending STATUS message in byte.
	 */
	uint16_t rlcStatusPduSize;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Paging buffer status.
 */
struct SchedDlPagingBufferReqParameters
{
	/**
	 * The number of elements in \ref pagingInfoList.
	 * \n Range: 0..#MAX_PAGING_LIST
	 */
	uint8_t nr_pagingInfoList;

	/**
	 * List holding paging information to be sent.
	 */
	struct PagingInfoListElement_s *pagingInfoList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * MAC buffer status.
 */
struct SchedDlMacBufferReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t  rnti;

	/**
	 * The CE element which is scheduled to be sent by the MAC. Can be Timing Advance CE, DRX Command CE and Contention Resolution CE (Activation/Deactivation CE is generated by the scheduler itself).
	 * This bitmap holds \ref CeBitmap_e enum flags.
	 */
	uint8_t ceBitmap;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 *
 */
struct SchedDlTriggerReqParameters
{
	/**
	 * The SFN and SF for which the scheduling is to be done, see \ref fapiExtDoc_timing_sec "Scheduler timing" and \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * Number of elements in \ref dlInfoList array.
	 * \n Range: 0..#MAX_DL_INFO_LIST
	 */
	uint8_t nr_dlInfoList;

	/**
	 * The list of UE DL information.
	 */
	struct DlInfoListElement_s *dlInfoList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * RACH reception information.
 */
struct SchedDlRachInfoReqParameters
{
	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * The number of elements in \ref rachList array.
	 * \n Range: 0..#MAX_RACH_LIST
	 */
	uint8_t nrrachList;

	/**
	 * The list of detected RACHs.
	 */
	struct RachListElement_s *rachList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * DL CQI information.
 */
struct SchedDlCqiInfoReqParameters
{
	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * The number of elements in \ref cqiList array.
	 * \n Range: 0..#MAX_CQI_LIST
	 */
	uint8_t nrcqiList;

	/**
	 * The list of DL CQI reports received in one subframe.
	 */
	struct CqiListElement_s *cqiList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * UL HARQ information.
 */
struct SchedUlTriggerReqParameters
{
	/**
	 * The SFN and SF for which the scheduling is to be done, see \ref fapiExtDoc_timing_sec "Scheduler timing" and \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * The number of elements in \ref ulInfoList array.
	 * \n Range: 0..#MAX_UL_INFO_LIST
	 */
	uint8_t nr_ulInfoList;

	/**
	 * The list of UL information for the scheduler.
	 */
	struct UlInfoListElement_s *ulInfoList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Noise and interference measurements.
 */
struct SchedUlNoiseInterferenceReqParameters
{
	/**
	 * Component carrier identifier, uniquely identifies carrier in the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t	carrierIndex;

	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * Received Interference Power, see \ref ref9 "[9]". In dBm.
	 * \n Range: -126.0..-75.0
	 * In fixed point format Q7.8.
	 */
	int16_t rip;

	/**
	 * Thermal Noise Power, see \ref ref9 "[9]". In dBm.
	 * \n Range: -146.0..-75
	 * In fixed point format Q7.8.
	 */
	int16_t tnp;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Scheduling request information.
 */
struct SchedUlSrInfoReqParameters
{
	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * The number of elements on \ref srList array.
	 * \n Range: 0..#MAX_SR_LIST
	 */
	uint8_t nr_srList;

	/**
	 * The list of SRs received in one subframe.
	 */
	struct SrListElement_s *srList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * MAC control information received.
 */
struct SchedUlMacCtrlInfoReqParameters
{
	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * Number of elements in \ref macCeUlList array.
	 * \n Range: 0..#MAX_MAC_CE_LIST
	 */
	uint8_t nr_macCEUL_List;

	/**
	 * The list of MAC control elements received in one subframe.
	 */
	struct MacCeUlListElement_s *macCeUlList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * UL CQI information.
 */
struct SchedUlCqiInfoReqParameters
{
	/**
	 * The SFN and SF in which the information was received, see \ref fapiExtDoc_timestamp_sec "Timestamp coding".
	 */
	uint16_t sfnSf;

	/**
	 * The number of elements in \ref ulCqiList array.
	 */
	uint8_t	nr_ulCqiList;

	/**
	 * List of UL CQI information received in one subframe.
	 */
	struct UlCqi_s* ulCqiList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};


/**
 * DL scheduling decision.
 */
struct SchedDlConfigIndParameters
{
	/**
	 * The number of elements in \ref buildDataList array.
	 * \n Range: 0..#MAX_BUILD_DATA_LIST
	 */
	uint8_t nr_buildDataList;

	/**
	 * The number of elements in \ref buildRarList array.
	 * \n Range: 0..#MAX_BUILD_RAR_LIST
	 */
	uint8_t nr_buildRARList;

	/**
	 * The number of elements in \ref buildBroadcastList array.
	 * \n Range: 0..#MAX_BUILD_BC_LIST
	 */
	uint8_t nr_buildBroadcastList;

	/**
	 * The list of resource allocation for UEs and LCs.
	 */
	struct BuildDataListElement_s *buildDataList;

	/**
	 * The list of resource allocation for RAR.
	 */
	struct BuildRarListElement_s *buildRarList;

	/**
	 * The list of resource allocation for BCCH, PCCH.
	 */
	struct BuildBroadcastListElement_s *buildBroadcastList;

	/**
	 * The number of elements in \ref nrOfPdcchOfdmSymbols array.
	 */
	uint8_t nr_ofdmSymbolsCount;

	/**
	 * Current size of PDCCH for each CC.
	 */
	struct PdcchOfdmSymbolCountListElement_s* nrOfPdcchOfdmSymbols[2 /*MAX_NUM_CCs*/];

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * UL scheduling decision.
 */
struct SchedUlConfigIndParameters
{
	/**
	 * The number of elements in \ref dciList array.
	 * \n Range: 0..MAX_DCI_LIST
	 */
	uint8_t nr_dciList;

	/**
	 * The number of elements in \ref phichList array.
	 * \n Range: 0..MAX_PHICH_LIST
	 */
	uint8_t nr_phichList;

	/**
	 * The list of UL DCI (Format 0) elements.
	 */
	struct UlDciListElement_s *dciList;

	/**
	 * The list of PHICH elements.
	 */
	struct PhichListElement_s *phichList;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

// Primitives defined as callbacks in separate file ff-mac-callback.h

#if FAPI_TRACER

#define SchedDlRlcBufferReq _SchedDlRlcBufferReq
#define SchedDlPagingBufferReq _SchedDlPagingBufferReq
#define SchedDlMacBufferReq _SchedDlMacBufferReq
#define SchedDlTriggerReq _SchedDlTriggerReq
#define SchedDlRachInfoReq _SchedDlRachInfoReq
#define SchedDlCqiInfoReq _SchedDlCqiInfoReq
#define SchedUlTriggerReq _SchedUlTriggerReq
#define SchedUlNoiseInterferenceReq _SchedUlNoiseInterferenceReq
#define SchedUlSrInfoReq _SchedUlSrInfoReq
#define SchedUlMacCtrlInfoReq _SchedUlMacCtrlInfoReq
#define SchedUlCqiInfoReq _SchedUlCqiInfoReq

#endif /* FAPI_TRACER */

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_SCHED_SAP_H */
