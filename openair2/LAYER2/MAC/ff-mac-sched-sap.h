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
/*! \file ff-mac-sched-sap.h
 * \brief this is the implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 
 * \author Florian Kaltenberger
 * \date March 2015
 * \version 1.0
 * \email: florian.kaltenberger@eurecom.fr
 * @ingroup _fapi
 */

#ifndef FF_MAC_SCHED_SAP_H
#define FF_MAC_SCHED_SAP_H

#include <stdint.h>
#include <stdbool.h>

#include "ff-mac-common.h"

#if defined (__cplusplus)
extern "C" {
#endif

/** @defgroup _fapi  FAPI
 * @ingroup _mac
 * @{
 */

/**
 * Parameters of the API primitives
 */

/**
 * Parameters of the SCHED_DL_RLC_BUFFER_REQ primitive.
 * See section 4.2.1 for a detailed description of the parameters.
 */
struct SchedDlRlcBufferReqParameters
{
  uint16_t  rnti;
  uint8_t   logicalChannelIdentity;
  uint32_t  rlcTransmissionQueueSize;
  uint16_t  rlcTransmissionQueueHolDelay;
  uint32_t  rlcRetransmissionQueueSize;
  uint16_t  rlcRetransmissionHolDelay;
  uint16_t  rlcStatusPduSize;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_DL_PAGING_BUFFER_REQ primitive.
 * See section 4.2.2 for a detailed description of the parameters.
 */
struct SchedDlPagingBufferReqParameters
{
  uint16_t  rnti;
  uint8_t   nr_pagingInfoList;
  struct PagingInfoListElement_s *pagingInfoList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_DL_MAC_BUFFER_REQ primitive.
 * See section 4.2.3 for a detailed description of the parameters.
 */
struct SchedDlMacBufferReqParameters
{
  uint16_t  rnti;
  uint8_t   nr_macCEDL_List;
  struct MacCeDlListElement_s *macCeDlList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_DL_TRIGGER_REQ primitive.
 * See section 4.2.4 for a detailed description of the parameters.
 */
struct SchedDlTriggerReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nr_dlInfoList;
  struct DlInfoListElement_s *dlInfoList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_DL_RACH_INFO_REQ primitive.
 * See section 4.2.5 for a detailed description of the parameters.
 */
struct SchedDlRachInfoReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nrrachList;
  struct RachListElement_s *rachList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_DL_CQI_INFO_REQ primitive.
 * See section 4.2.6 for a detailed description of the parameters.
 */
struct SchedDlCqiInfoReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nrcqiList;
  struct CqiListElement_s *cqiList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_TRIGGER_REQ primitive.
 * See section 4.2.8 for a detailed description of the parameters.
 */
struct SchedUlTriggerReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nr_ulInfoList;
  struct UlInfoListElement_s *ulInfoList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_NOISE_INTERFERENCE_REQ primitive.
 * See section 4.2.9 for a detailed description of the parameters.
 */
struct SchedUlNoiseInterferenceReqParameters
{
  uint8_t	carrierIndex;
  uint16_t  sfnSf;
  uint16_t  rip;
  uint16_t  tnp;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_SR_INFO_REQ primitive.
 * See section 4.2.10 for a detailed description of the parameters.
 */
struct SchedUlSrInfoReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nr_srList;
  struct SrListElement_s *srList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_MAC_CTRL_INFO_REQ primitive.
 * See section 4.2.11 for a detailed description of the parameters.
 */
struct SchedUlMacCtrlInfoReqParameters
{
  uint16_t  sfnSf;
  uint8_t   nr_macCEUL_List;
  struct MacCeUlListElement_s *macCeUlList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_CQI_INFO_REQ primitive.
 * See section 4.2.12 for a detailed description of the parameters.
 */
struct SchedUlCqiInfoReqParameters
{
  uint16_t  sfnSf;
  uint8_t	nr_ulCqiList;
  struct UlCqi_s* ulCqiList;
  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

//
// SCHED - MAC Scheduler SAP primitives
// (See 4.2 for description of the primitives)
//

void SchedDlRlcBufferReq(void *, const struct SchedDlRlcBufferReqParameters *params);
void SchedDlPagingBufferReq(void *, const struct SchedDlPagingBufferReqParameters *params);
void SchedDlMacBufferReq(void *, const struct SchedDlMacBufferReqParameters *params);
void SchedDlTriggerReq(void *, const struct SchedDlTriggerReqParameters *params);
void SchedDlRachInfoReq(void *, const struct SchedDlRachInfoReqParameters *params);
void SchedDlCqiInfoReq(void *, const struct SchedDlCqiInfoReqParameters *params);
void SchedUlTriggerReq(void *, const struct SchedUlTriggerReqParameters *params);
void SchedUlNoiseInterferenceReq(void *, const struct SchedUlNoiseInterferenceReqParameters *params);
void SchedUlSrInfoReq(void *, const struct SchedUlSrInfoReqParameters *params);
void SchedUlMacCtrlInfoReq(void *, const struct SchedUlMacCtrlInfoReqParameters *params);
void SchedUlCqiInfoReq(void *, const struct SchedUlCqiInfoReqParameters *params);

/**
 * Parameters of the API primitives
 */

/**
 * Parameters of the SCHED_DL_CONFIG_IND primitive.
 * See section 4.2.7 for a detailed description of the parameters.
 */
struct SchedDlConfigIndParameters
{
  uint8_t nr_buildDataList;
  uint8_t nr_buildRARList;
  uint8_t nr_buildBroadcastList;
  struct BuildDataListElement_s      *buildDataList;
  struct BuildRarListElement_s       *buildRarList;
  struct BuildBroadcastListElement_s *buildBroadcastList;

  /* mind: this is just number of elems in the next array (not actual number of PDCCH OFDM symbols) */
  uint8_t nr_ofdmSymbolsCount;
#warning [31;46mMAX_NUM_CCs forced to 2 in structure SchedDlConfigIndParameters!![0m
  struct PdcchOfdmSymbolCountListElement_s* nrOfPdcchOfdmSymbols[2 /* MAX_NUM_CCs */];

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

/**
 * Parameters of the SCHED_UL_CONFIG_IND primitive.
 * See section 4.2.13 for a detailed description of the parameters.
 */
struct SchedUlConfigIndParameters
{
  uint8_t nr_dciList;
  uint8_t nr_phichList;
  struct UlDciListElement_s *dciList;
  struct PhichListElement_s *phichList;

  uint8_t   nr_vendorSpecificList;
  struct VendorSpecificListElement_s *vendorSpecificList;
};

//
// SCHED - MAC Scheduler SAP primitives
// (See 4.2 for description of the primitives)
//
// Primitives defined as callbacks in separate file ff-mac-callback.h

#if 0
/* not used - the scheduler has callbacks for those */
void SchedDlConfigInd(const struct SchedDlConfigIndParameters* params);
void SchedUlConfigInd(const struct SchedUlConfigIndParameters* params);
#endif

/*@}*/

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_SCHED_SAP_H */
