/**
 * @file ff-mac-csched-sap.h
 * @brief Implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 with extensions.
 * @details Contains CSCHED SAP MAC->scheduler primitive declarations and all primitives data structures definitions.
 * @author Florian Kaltenberger, Maciej Wewior
 * @date March 2015
 * @email: florian.kaltenberger@eurecom.fr, m.wewior@is-wireless.com
 * @ingroup _mac
 */

#ifndef FF_MAC_CSCHED_SAP_H
#define FF_MAC_CSCHED_SAP_H

#include <stdint.h>
#include <stdbool.h>

#include "ff-mac-common.h"

#if defined (__cplusplus)
extern "C" {
#endif

//forward declarations
struct CschedCellConfigReqParameters;
struct CschedUeConfigReqParameters;
struct CschedLcConfigReqParameters;
struct CschedLcReleaseReqParameters;
struct CschedUeReleaseReqParameters;

//CSCHED SAP MAC->scheduler primitives

/**
 * @brief Configure cell.
 * @details (Re-)configure MAC scheduler with cell configuration and scheduler configuration. The cell configuration will also setup the BCH, BCCH, PCCH and CCCH LC configuration (for each component carrier).
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params Cell configuration
 */
void CschedCellConfigReq (void* scheduler, const struct CschedCellConfigReqParameters *params);

/**
 * @brief Configure single UE.
 * @details (Re-)configure MAC scheduler with single UE specific parameters. A UE can only be configured when a cell configuration has been received.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params UE configuration
 */
void CschedUeConfigReq (void* scheduler, const struct CschedUeConfigReqParameters *params);

/**
 * @brief Configure UE's logical channel(s).
 * @details (Re-)configure MAC scheduler with UE's logical channel configuration. A logical channel can only be configured when a UE configuration has been received.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params UE's logical channel configuration
 */
void CschedLcConfigReq (void* scheduler, const struct CschedLcConfigReqParameters *params);

/**
 * @brief Release UE's logical channel(s).
 * @details Release UE's logical channel(s) in the MAC scheduler. A logical channel can only be released if it has been configured previously.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params UE's logical channel(s) to be released
 */
void CschedLcReleaseReq (void* scheduler, const struct CschedLcReleaseReqParameters *params);

/**
 * @brief Release UE.
 * @details Release a UE in the MAC scheduler. The release of the UE configuration implies the release of LCs, which are still active. A UE can only be released if it has been configured previously.
 * @param scheduler Scheduler context pointer, see SchedInit()
 * @param params UE to be released
 */
void CschedUeReleaseReq (void* scheduler, const struct CschedUeReleaseReqParameters *params);


//CSCHED SAP primitives parameters

/**
 * Cell configuration parameters.
 */
struct CschedCellConfigReqParameters
{
	/**
	 * The number of elements in the \ref ccConfigList array.
	 * \n Range: 0..#MAX_NUM_CCs
	 */
	uint8_t nr_carriers;

	/**
	 * The list of component carrier’s configurations.
	 */
	struct CschedCellConfigReqParametersListElement* ccConfigList[2 /*MAX_NUM_CCs*/];

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
 * Component carrier configuration parameters.
 */
struct CschedCellConfigReqParametersListElement
{
	/**
	 * PUSCH resources in RBs. used for hopping. see \ref ref2 "[2]" section 5.3.4.
	 * \n Range: 0..98
	 */
	uint8_t puschHoppingOffset;

	/**
	 * Physical cell ID.
	 * \n Range: 0..503
	 */
	unsigned NcellID;

	/**
	 * See \ref ref2 "[2]" section 5.3.4
	 */
	enum HoppingMode_e
	{
	  inter,
	  interintra
	} hoppingMode;

	/**
	 * Number of subbands, see \ref ref2 "[2]" section 5.3.4.
	 * \n Range: 1..4
	 */
	uint8_t nSb;

	/**
	 * The number of resources element groups used for PHICH.
	 */
	enum PhichResource_e
	{
	  PHICH_R_ONE_SIXTH,
	  PHICH_R_HALF,
	  PHICH_R_ONE,
	  PHICH_R_TWO
	} phichResource;


	/**
	 * See \ref ref2 "[2]" table 6.9.3-1.
	 */
	enum NormalExtended_e phichDuration;

	/**
	 * Nr of PDCCH OFDM symbols, see \ref ref2 "[2]" section 6.9.
	 * \n Range: 0..4
	 */
	uint8_t initialNrOfPdcchOfdmSymbols;

	/**
	 * The SI configuration.
	 */
	struct SiConfiguration_s siConfiguration;

	/**
	 * UL transmission bandwidth in RBs.
	 * \n Range: 6,15,25,50,75,100
	 */
	uint8_t ulBandwidth;

	/**
	 * DL transmission bandwidth in RBs.
	 * \n Range: 6,15,25,50,75,100
	 */
	uint8_t dlBandwidth;

	/**
	 * See \ref ref2 "[2]" section 5.2.1.
	 */
	enum NormalExtended_e ulCyclicPrefixLength;

	/**
	 * DL cyclic prefix.
	 */
	enum NormalExtended_e dlCyclicPrefixLength;

	/**
	 * Number of cell specific antenna ports, see \ref ref2 "[2]" section 6.2.1.
	 * \n Range: 1,2,4
	 */
	uint8_t antennaPortsCount;

	/**
	 * Cell is configured in TDD or FDD mode.
	 */
	enum DuplexMode_e
	{
	  DTDD,
	  DFDD
	} duplexMode;

	/**
	 * DL/UL subframe assignment. Only TDD, see \ref ref2 "[2]" table 4.2.2.
	 * \n Range: 0..6
	 */
	uint8_t subframeAssignment;

	/**
	 * TDD configuration. Only TDD, see \ref ref2 "[2]" table 4.2.1.
	 * \n Range: 0..8
	 */
	uint8_t specialSubframePatterns;

	/**
	 * Indicates if the \ref mbsfnSubframeConfigRfPeriod, \ref mbsfnSubframeConfigRfOffset, \ref mbsfnSubframeConfigSfAllocation fields are valid or not.
	 */
	bool mbsfn_SubframeConfigPresent;

	/**
	 * The MBSFN radio frame period.
	 * \n Range: 1,2,4,8,16,32
	 */
	uint8_t mbsfnSubframeConfigRfPeriod[MAX_MBSFN_CONFIG];

	/**
	 * The radio frame offset.
	 * \n Range: 0..7
	 */
	uint8_t mbsfnSubframeConfigRfOffset[MAX_MBSFN_CONFIG];

	/**
	 * Indicates the MBSFN subframes.
	 * \n Range: bitmap 0..9
	 */
	uint8_t mbsfnSubframeConfigSfAllocation[MAX_MBSFN_CONFIG];

	/**
	 * See \ref ref2 "[2]" section 5.7.1.
	 * \n Range: 0..63
	 */
	uint8_t prachConfigurationIndex;

	/**
	 * See \ref ref2 "[2]" section 5.7.1.
	 * \n Range: 0..94
	 */
	uint8_t prachFreqOffset;

	/**
	 * Duration of RA response window in SF, see \ref ref1 "[1]".
	 * \n Range: 2..8,10
	 */
	uint8_t raResponseWindowSize;

	/**
	 * Contention resolution timer used during random access, see \ref ref1 "[1]".
	 * \n Range: 8,16,24,32,40,48,56,64
	 */
	uint8_t macContentionResolutionTimer;

	/**
	 * See \ref ref1 "[1]".
	 * \n Range: 1..8
	 */
	uint8_t maxHarqMsg3Tx;

	/**
	 * See \ref ref4 "[4]" section 10.1.
	 * \n Range: 0..2047
	 */
	uint16_t n1PucchAn;

	/**
	 * See \ref ref2 "[2]" section 5.4.
	 * \n Range: 1..3
	 */
	uint8_t deltaPucchShift;

	/**
	 * See \ref ref2 "[2]" section 5.4.
	 * \n Range: 0..98
	 */
	uint8_t nrbCqi;

	/**
	 * See \ref ref2 "[2]" section 5.4.
	 * \n Range: 0..7
	 */
	uint8_t ncsAn;

	/**
	 * See \ref ref2 "[2]" table 5.5.3.3-1 and 5.5.3.3-2.
	 * \n Range: 0..15
	 */
	uint8_t srsSubframeConfiguration;

	/**
	 * See \ref ref2 "[2]" section 5.5.3.2.
	 * \n Range: 0..9
	 */
	uint8_t srsSubframeOffset;

	/**
	 * SRS bandwidth, see \ref ref2 "[2]" section 5.5.3.2.
	 * \n Range: 0..7
	 */
	uint8_t srsBandwidthConfiguration;

	/**
	 * See \ref ref2 "[2]" section 5.5.3.2. Only TDD.
	 */
	bool srsMaxUpPts;

	/**
	 * Maximum UL modulation supported, see \ref ref4 "[4]" section 8.6.1.
	 */
	enum Enable64Qae
	{
	  MOD_16QAM,
	  MOD_64QAM
	} enable64Qam;

	/**
	 * Component carrier identifier.
	 */
	uint8_t carrierIndex;
};

/**
 * UE configuration parameters.
 */
struct CschedUeConfigReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * Indicates if this is a reconfiguration for an existing UE or if a new UE is added.
	 */
	bool reconfigureFlag;

	/**
	 * Indicates if the \ref drxConfig sub-structure is valid or not.
	 */
	bool drxConfigPresent;

	/**
	 * The DRX configuration.
	 */
	struct DrxConfig_s drxConfig;

	/**
	 * In subframes, see \ref ref1 "[1]". Used for controlling synchronization status of the UE, not for the actual timing advance procedure.
	 * \n Range: 500,750,1280,1920,2560,5120,10240,inf
	 */
	uint16_t timeAlignmentTimer;

	/**
	 * Specifies the measurement gap configuration or that it is not applicable, see \ref ref6 "[6]".
	 */
	enum MeasGapConfigPattern_e
	{
	  MGP_GP1,
	  MGP_GP2,
	  OFF
	} measGapConfigPattern;

	/**
	 * Specifies the measurement gap offset, if applicable, see \ref ref6 "[6]".
	 * \n Range: 0..79
	 */
	uint8_t measGapConfigSubframeOffset;

	/**
	 * Indicates if the \ref spsConfig is valid.
	 */
	bool spsConfigPresent;

	/**
	 * The SPS configuration.
	 */
	struct SpsConfig_s spsConfig;

	/**
	 * Indicates if \ref srConfig struct is valid.
	 */
	bool srConfigPresent;

	/**
	 * The SR configuration.
	 */
	struct SrConfig_s srConfig;

	/**
	 * Indicates if \ref cqiConfig struct is valid.
	 */
	bool cqiConfigPresent;

	/**
	 * The CQI configuration.
	 */
	struct CqiConfig_s cqiConfig;

	/**
	 * The configured transmission mode, see \ref ref4 "[4]" section 7.1.
	 * \n Range: 1..7
	 */
	uint8_t transmissionMode;

	/**
	 * Aggregated bit rate of non-gbr bearer per UE, see \ref ref7 "[7]".
	 * \n Range: 0..10000000000
	 */
	uint64_t ueAggregatedMaximumBitrateUl;

	/**
	 * Aggregated bit rate of non-gbr bearer per UE, see \ref ref7 "[7]".
	 * \n Range: 0..10000000000
	 */
	uint64_t ueAggregatedMaximumBitrateDl;

	/**
	 * The UE capabilities.
	 */
	struct UeCapabilities_s ueCapabilities;

	/**
	 * See \ref ref4 "[4]" section 8.7.
	 */
	enum OpenClosedLoop_e
	{
	  noneloop,
	  openloop,
	  closedloop
	} ueTransmitAntennaSelection;

	/**
	 * See \ref ref1 "[1]".
	 */
	bool ttiBundling;

	/**
	 * The maximum HARQ retransmission for uplink HARQ, see \ref ref1 "[1]".
	 * \n Range: 1..8,10,12,16,20,24,28
	 */
	uint8_t maxHarqTx;

	/**
	 * See \ref ref4 "[4]" table 8.6.3-1.
	 * \n Range: 0..15
	 */
	uint8_t betaOffsetAckIndex;

	/**
	 * See \ref ref4 "[4]" table 8.6.3-2.
	 * \n Range: 0..15
	 */
	uint8_t betaOffsetRiIndex;

	/**
	 * See \ref ref4 "[4]" table 8.6.3-3.
	 * \n Range: 0..15
	 */
	uint8_t betaOffsetCqiIndex;

	/**
	 * See \ref ref4 "[4]" section 8.2.
	 */
	bool ackNackSrsSimultaneousTransmission;

	/**
	 * See \ref ref4 "[4]" section 10.1.
	 */
	bool simultaneousAckNackAndCqi;

	/**
	 * Reporting mode for aperiodic CQI, see \ref ref4 "[4]" section 7.2.1.
	 */
	enum RepMode_e
	{
	  ff_rm12, ff_rm20, ff_rm22, ff_rm30, ff_rm31, ff_nonemode
	} aperiodicCqiRepMode;

	/**
	 * See \ref ref3 "[3]" section 7.3. Only TDD.
	 */
	enum FeedbackMode_e
	{
	  ff_bundling,
	  ff_multiplexing
	} tddAckNackFeedbackMode;

	/**
	 * See \ref ref4 "[4]" section 10.1. 0 means no repetition.
	 * \n Range: 0,2,4,6
	 */
	uint8_t ackNackRepetitionFactor;

	/**
	 * Indicates if extended BSR sizes shall be used, see \ref ref1 "[1]" section 6.1.3.1.
	 */
	bool extendedBSRSizes;

	/**
	 * Indicates if the UE supports CA.
	 */
	bool caSupport;

	/**
	 * Indicates if the supports cross carrier scheduling.
	 */
	bool crossCarrierSchedSupport;

	/**
	 * Carrier index of the UE’s PCell. Indicates which of available carriers in eNB is the PCell for the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t pcellCarrierIndex;

	/**
	 * Number of SCells configured for the UE, indicates number of valid elements in \ref scellConfigList.
	 * \n Range: 0..#MAX_NUM_CCs-1
	 */
	uint8_t nr_scells;

	/**
	 * The list of SCell configurations.
	 */
	struct 	ScellConfig_s* scellConfigList[2 /*MAX_NUM_CCs*/ -1];

	/**
	 * SCell deactivation timer, see \ref ref1 "[1]".
	 */
	uint8_t	scellDeactivationTimer;

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
 * Logical channel configuration parameters.
 */
struct CschedLcConfigReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * Indicates if this is a reconfiguration for an existing UE or if a new UE is added. See \ref fapiExtDoc_lcreconf_sec "Logical channel reconfiguration".
	 */
	bool reconfigureFlag;

	/**
	 * The number of elements in \ref logicalChannelConfigList array.
	 * \n Range: 1..#MAX_LC_LIST
	 */
	uint8_t nr_logicalChannelConfigList;

	/**
	 * The array of logical channel configurations to be configured.
	 */
	struct LogicalChannelConfigListElement_s *logicalChannelConfigList;

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
 * Logical channel release parameters.
 */
struct CschedLcReleaseReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The number of elements in the \ref logicalChannelIdentity array.
	 * \n Range: 1..#MAX_LC_LIST
	 */
	uint8_t nr_logicalChannelIdendity;

	/**
	 * The array of logical channel ID which shall be released.
	 */
	uint8_t *logicalChannelIdentity;

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
 * UE release parameters.
 */
struct CschedUeReleaseReqParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};


//CSCHED SAP scheduler->MAC primitives parameters

/**
 * Cell configuration confirmation parameters.
 */
struct CschedCellConfigCnfParameters
{
	/**
	 * The outcome of the cell configuration request.
	 */
	enum Result_e result;

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
 * UE configuration confirmation parameters.
 */
struct CschedUeConfigCnfParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The outcome of the UE configuration request.
	 */
	enum Result_e result;

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
 * Logical channel configuration confirmation parameters.
 */
struct CschedLcConfigCnfParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The outcome of the LC configuration request.
	 */
	enum Result_e result;

	/**
	 * The number of elements in the \ref logicalChannelIdentity array.
	 * \n Range: 1..#MAX_LC_LIST
	 */
	uint8_t nr_logicalChannelIdendity;

	/**
	 * The array of logical channel ID which have been configured/updated.
	 */
	uint8_t *logicalChannelIdentity;

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
 * Logical channel release confirmation parameters.
 */
struct CschedLcReleaseCnfParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The outcome of the LC release request.
	 */
	enum Result_e result;

	/**
	 * The number of elements in the \ref logicalChannelIdentity array.
	 * \n Range: 1..#MAX_LC_LIST
	 */
	uint8_t nr_logicalChannelIdendity;

	/**
	 * The array of logical channel ID which have been released.
	 */
	uint8_t *logicalChannelIdentity;

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
 * UE release confirmation parameters.
 */
struct CschedUeReleaseCnfParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The outcome of the UE release request.
	 */
	enum Result_e result;

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
 * UE configuration update indication parameters.
 */
struct CschedUeConfigUpdateIndParameters
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The configured transmission mode, see \ref ref4 "[4]" section 7.1.
	 * \n Range: 1..7
	 */
	uint8_t transmissionMode;

	/**
	 * Indicates if \ref spsConfig struct is present.
	 */
	bool spsConfigPresent;

	/**
	 * The SPS configuration request.
	 */
	struct SpsConfig_s spsConfig;

	/**
	 * Indicates if \ref srConfig struct is present.
	 */
	bool srConfigPresent;

	/**
	 * The SR configuration request.
	 */
	struct SrConfig_s srConfig;

	/**
	 * Indicates if \ref cqiConfig struct is present.
	 */
	bool cqiConfigPresent;

	/**
	 * The CQI configuration request.
	 */
	struct CqiConfig_s cqiConfig;

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
 * Cell configuration update indication parameters.
 */
struct CschedCellConfigUpdateIndParameters
{
	/**
	 * Component carrier identifier, uniquely identifies carrier in the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t	carrierIndex;

	/**
	 * Percentage as defined in \ref ref8 "[8]".
	 * \n Range: 0..100
	 */
	uint8_t prbUtilizationDl;

	/**
	 * Percentage as defined in \ref ref8 "[8]".
	 * \n Range: 0..100
	 */
	uint8_t prbUtilizationUl;

	/**
	 * The number of elements in the \ref vendorSpecificList array.
	 */
	uint8_t nr_vendorSpecificList;

	/**
	 * Contains scheduler specific configuration received from the OAM subsystem for use by a specific scheduler.
	 */
	struct VendorSpecificListElement_s *vendorSpecificList;
};

//CSCHED SAP MAC->scheduler primitives are defined as callbacks in separate file ff-mac-callback.h

#if FAPI_TRACER

#define CschedCellConfigReq _CschedCellConfigReq
#define CschedUeConfigReq _CschedUeConfigReq
#define CschedLcConfigReq _CschedLcConfigReq
#define CschedLcReleaseReq _CschedLcReleaseReq
#define CschedUeReleaseReq _CschedUeReleaseReq

#endif /* FAPI_TRACER */

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_CSCHED_SAP_H */
