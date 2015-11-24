/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2015 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is
    included in this distribution in the file called "COPYING". If not,
    see <http://www.gnu.org/licenses/>.

  Contact Information
  OpenAirInterface Admin: openair_admin@eurecom.fr
  OpenAirInterface Tech : openair_tech@eurecom.fr
  OpenAirInterface Dev  : openair4g-devel@eurecom.fr

  Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

*******************************************************************************/
/*! \file ff-mac-csched-sap.h
 * \brief this is the implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 
 * \author Florian Kaltenberger
 * \date March 2015
 * \version 1.0
 * \email: florian.kaltenberger@eurecom.fr
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

/**
 * Parameters of the API primitives
 */

/**
 * Parameters of the CSCHED_CELL_CONFIG_REQ primitive.
 * See section 4.1.1 for a detailed description of the parameters.
 */
struct CschedCellConfigReqParameters
{
  uint8_t   nr_carriers;
  struct CschedCellConfigReqParametersListElement* ccConfigList[MAX_NUM_CCs];
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

struct CschedCellConfigReqParametersListElement
{
  uint8_t puschHoppingOffset;

  enum HoppingMode_e
    {
      inter,
      interintra
    } hoppingMode;

  uint8_t nSb;

  enum PhichResource_e
    {
      PHICH_R_ONE_SIXTH,
      PHICH_R_HALF,
      PHICH_R_ONE,
      PHICH_R_TWO
    } phichResource;


  enum NormalExtended_e phichDuration;

  uint8_t initialNrOfPdcchOfdmSymbols;

  struct SiConfiguration_s siConfiguration;

  uint8_t ulBandwidth;
  uint8_t dlBandwidth;

  enum NormalExtended_e ulCyclicPrefixLength;
  enum NormalExtended_e dlCyclicPrefixLength;

  uint8_t antennaPortsCount;

  enum DuplexMode_e
    {
      DTDD,
      DFDD
    } duplexMode;

  uint8_t subframeAssignment;
  uint8_t specialSubframePatterns;

  uint8_t mbsfnSubframeConfigRfPeriod[MAX_MBSFN_CONFIG];
  uint8_t mbsfnSubframeConfigRfOffset[MAX_MBSFN_CONFIG];
  uint8_t mbsfnSubframeConfigSfAllocation[MAX_MBSFN_CONFIG];
  uint8_t prachConfigurationIndex;
  uint8_t prachFreqOffset;
  uint8_t raResponseWindowSize;
  uint8_t macContentionResolutionTimer;
  uint8_t maxHarqMsg3Tx;
  uint16_t n1PucchAn;
  uint8_t deltaPucchShift;
  uint8_t nrbCqi;
  uint8_t ncsAn;
  uint8_t srsSubframeConfiguration;
  uint8_t srsSubframeOffset;
  uint8_t srsBandwidthConfiguration;
  bool    srsMaxUpPts;

  enum Enable64Qae
    {
      MOD_16QAM,
      MOD_64QAM
    } enable64Qam;

  uint8_t   carrierIndex;
};

/**
 * Parameters of the CSCHED_UE_CONFIG_REQ primitive.
 * See section 4.1.3 for a detailed description of the parameters.
 */
struct CschedUeConfigReqParameters
{
  uint16_t  rnti;
  bool      reconfigureFlag;
  bool      drxConfigPresent;
  struct DrxConfig_s drxConfig;
  uint16_t  timeAlignmentTimer;

  enum MeasGapConfigPattern_e
    {
      MGP_GP1,
      MGP_GP2,
      OFF
    } measGapConfigPattern;

  uint8_t   measGapConfigSubframeOffset;
  bool      spsConfigPresent;
  struct SpsConfig_s spsConfig;
  bool      srConfigPresent;
  struct SrConfig_s srConfig;
  bool      cqiConfigPresent;
  struct CqiConfig_s cqiConfig;
  uint8_t   transmissionMode;
  uint64_t  ueAggregatedMaximumBitrateUl;
  uint64_t  ueAggregatedMaximumBitrateDl;
  struct UeCapabilities_s ueCapabilities;

  enum OpenClosedLoop_e
    {
      noneloop,
      openloop,
      closedloop
    } ueTransmitAntennaSelection;

  bool      ttiBundling;
  uint8_t   maxHarqTx;
  uint8_t   betaOffsetAckIndex;
  uint8_t   betaOffsetRiIndex;
  uint8_t   betaOffsetCqiIndex;
  bool      ackNackSrsSimultaneousTransmission;
  bool      simultaneousAckNackAndCqi;

  enum RepMode_e
    {
      ff_rm12, ff_rm20, ff_rm22, ff_rm30, ff_rm31, ff_nonemode
    } aperiodicCqiRepMode;

  enum FeedbackMode_e
    {
      ff_bundling,
      ff_multiplexing
    } tddAckNackFeedbackMode;

  uint8_t   ackNackRepetitionFactor;
  bool		extendedBSRSizes;

  bool      caSupport;
  bool		crossCarrierSchedSupport;
  uint8_t   pcellCarrierIndex;
  uint8_t   nr_scells;
  struct 	ScellConfig_s* scellConfigList[MAX_NUM_CCs-1];
  uint8_t	scellDeactivationTimer;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_LC_CONFIG_REQ primitive.
 * See section 4.1.5 for a detailed description of the parameters.
 */
struct CschedLcConfigReqParameters
{
  uint16_t  rnti;
  bool      reconfigureFlag;

  uint8_t   nr_logicalChannelConfigList;
  struct LogicalChannelConfigListElement_s *logicalChannelConfigList;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_LC_RELEASE_REQ primitive.
 * See section 4.1.7 for a detailed description of the parameters.
 */
struct CschedLcReleaseReqParameters
{
  uint16_t  rnti;

  uint8_t   nr_logicalChannelIdendity;
  uint8_t   *logicalChannelIdentity;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_UE_RELEASE_REQ primitive.
 * See section 4.1.9 for a detailed description of the parameters.
 */
struct CschedUeReleaseReqParameters
{
  uint16_t  rnti;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

//
// CSCHED - MAC Scheduler Control SAP primitives
// (See 4.1 for description of the primitives)
//

/**
 * \brief CSCHED_CELL_CONFIG_REQ
 */
void CschedCellConfigReq(void *, const struct CschedCellConfigReqParameters *params);
void CschedUeConfigReq(void *, const struct CschedUeConfigReqParameters *params);
void CschedLcConfigReq(void *, const struct CschedLcConfigReqParameters *params);
void CschedLcReleaseReq(void *, const struct CschedLcReleaseReqParameters *params);
void CschedUeReleaseReq(void *, const struct CschedUeReleaseReqParameters *params);

/**
 * Parameters of the API primitives
 */

/**
 * Parameters of the CSCHED_CELL_CONFIG_CNF primitive.
 * See section 4.1.2 for a detailed description of the parameters.
 */
struct CschedCellConfigCnfParameters
{
  enum Result_e result;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_UE_CONFIG_CNF primitive.
 * See section 4.1.4 for a detailed description of the parameters.
 */
struct CschedUeConfigCnfParameters
{
  uint16_t  rnti;
  enum Result_e result;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_LC_CONFIG_CNF primitive.
 * See section 4.1.6 for a detailed description of the parameters.
 */
struct CschedLcConfigCnfParameters
{
  uint16_t  rnti;
  enum Result_e result;
  uint8_t   nr_logicalChannelIdendity;
  uint8_t   *logicalChannelIdentity;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_LC_RELEASE_CNF primitive.
 * See section 4.1.8 for a detailed description of the parameters.
 */
struct CschedLcReleaseCnfParameters
{
  uint16_t  rnti;
  enum Result_e result;

  uint8_t   nr_logicalChannelIdendity;
  uint8_t   *logicalChannelIdentity;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_UE_RELEASE_CNF primitive.
 * See section 4.1.10 for a detailed description of the parameters.
 */
struct CschedUeReleaseCnfParameters
{
  uint16_t  rnti;
  enum Result_e result;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_UE_CONFIG_UPDATE_IND primitive.
 * See section 4.1.11 for a detailed description of the parameters.
 */
struct CschedUeConfigUpdateIndParameters
{
  uint16_t  rnti;
  uint8_t   transmissionMode;
  bool      spsConfigPresent;
  struct SpsConfig_s spsConfig;
  bool      srConfigPresent;
  struct SrConfig_s srConfig;
  bool      cqiConfigPresent;
  struct CqiConfig_s cqiConfig;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the CSCHED_CELL_CONFIG_UPDATE_IND primitive.
 * See section 4.1.12 for a detailed description of the parameters.
 */
struct CschedCellConfigUpdateIndParameters
{
  uint8_t	carrierIndex;
  uint8_t   prbUtilizationDl;
  uint8_t   prbUtilizationUl;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

//
// CSCHED - MAC Scheduler Control SAP primitives
// (See 4.1 for description of the primitives)
//

#if 0
/* not used - the scheduler has callbacks for those */
void CschedCellConfigCnf(const struct CschedCellConfigCnfParameters *params);
void CschedUeConfigCnf(const struct CschedUeConfigCnfParameters *params);
void CschedLcConfigCnf(const struct CschedLcConfigCnfParameters *params);
void CschedLcReleaseCnf(const struct CschedLcReleaseCnfParameters *params);
void CschedUeReleaseCnf(const struct CschedUeReleaseCnfParameters *params);
void CschedUeConfigUpdateInd(const struct CschedUeConfigUpdateIndParameters *params);
void CschedCellConfigUpdateInd(const struct CschedCellConfigUpdateIndParameters *params);
#endif

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_CSCHED_SAP_H */
