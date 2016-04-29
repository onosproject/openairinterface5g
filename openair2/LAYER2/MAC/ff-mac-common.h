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
/*! \file ff-mac-common.h
 * \brief this is the implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 
 * \author Florian Kaltenberger
 * \date March 2015
 * \version 1.0
 * \email: florian.kaltenberger@eurecom.fr
 * @ingroup _fapi
 */


#ifndef FF_MAC_COMMON_H
#define FF_MAC_COMMON_H

#include <stdint.h>
#include <stdbool.h>

#if defined (__cplusplus)
extern "C" {
#endif

/** @defgroup _fapi  FAPI
 * @ingroup _mac
 * @{
 */

/**
 * Constants. See section 4.4
 */
#define MAX_SCHED_CFG_LIST    10
#define MAX_LC_LIST           10

#define MAX_RACH_LIST         30
#define MAX_DL_INFO_LIST      30
#define MAX_BUILD_DATA_LIST   30
#define MAX_BUILD_RAR_LIST    10
#define MAX_BUILD_BC_LIST     3
#define MAX_UL_INFO_LIST      30
#define MAX_DCI_LIST          30
#define MAX_PHICH_LIST        30
#define MAX_TB_LIST           2
#define MAX_RLC_PDU_LIST      30
#define MAX_NR_LCG            4
#define MAX_MBSFN_CONFIG      5
#define MAX_SI_MSG_LIST       32
#define MAX_SI_MSG_SIZE       65535

#define MAX_CQI_LIST          30
#define MAX_UE_SELECTED_SB    6		//comes from Table 7.2.1-5, 36.213
#define MAX_HL_SB             13	//comes from Table 7.2.1-3, 36.213
#define MAX_SINR_RB_LIST      100
#define MAX_SR_LIST           30
#define MAX_MAC_CE_LIST       30

#ifndef MAX_NUM_CCs
#error MAX_NUM_CCs not defined
#endif

//#define MAX_NUM_CCs           2

enum Result_e
{
  ff_SUCCESS,
  ff_FAILURE
};

enum SetupRelease_e
{
  ff_setup,
  ff_release
};

enum CeBitmap_e
{
  ff_TA		= 1 << 0,
  ff_DRX	= 1 << 1,
  ff_CR		= 1 << 2,
  /// activation/deactivation of CCs
  ff_AD		= 1 << 3
};

enum NormalExtended_e
{
  ff_normal,
  ff_extended
};

/**
 * \brief See section 4.3.1 dlDciListElement
 */
struct DlDciListElement_s
{
  uint16_t  rnti;
  uint32_t  rbBitmap;
  uint8_t   rbShift;
  uint8_t   rbgSubset;      //resource allocation type1 field
  uint8_t   resAlloc;
  uint8_t   nr_of_tbs;
  uint16_t  tbsSize[MAX_TB_LIST];
  uint8_t   mcs[MAX_TB_LIST];
  uint8_t   ndi[MAX_TB_LIST];
  uint8_t   rv[MAX_TB_LIST];
  uint8_t   cceIndex;
  uint8_t   aggrLevel;
  uint8_t   precodingInfo;
  enum Format_e
  {
    ONE, ONE_A, ONE_B, ONE_C, ONE_D, TWO, TWO_A, TWO_B
  } format;
  int8_t    tpc;
  uint8_t   harqProcess;
  uint8_t   dai;
  enum VrbFormat_e
  {
    VRB_DISTRIBUTED,
    VRB_LOCALIZED
  } vrbFormat;
  bool      tbSwap;
  bool      spsRelease;
  bool      pdcchOrder;
  uint8_t   preambleIndex;
  uint8_t   prachMaskIndex;
  enum Ngap_e
  {
    GAP1, GAP2
  } nGap;
  uint8_t   tbsIdx;
  uint8_t   dlPowerOffset;
  uint8_t   pdcchPowerOffset;
  /// this is the DCI field (for potential cross carrier scheduling)
  bool      cifPresent;
  uint8_t   cif;
};

/**
 * \brief See section 4.3.2 ulDciListElement
 */
struct UlDciListElement_s
{
  uint16_t  rnti;
  uint8_t   rbStart;
  uint8_t   rbLen;
  uint16_t  tbSize;
  uint8_t   mcs;
  uint8_t   ndi;
  uint8_t   cceIndex;
  uint8_t   aggrLevel;
  uint8_t   ueTxAntennaSelection;
  bool      hopping;
  uint8_t   n2Dmrs;
  int8_t    tpc;
  bool      cqiRequest;
  uint8_t   ulIndex;
  uint8_t   dai;
  uint8_t   freqHopping;
  int8_t    pdcchPowerOffset;
  /// this is the DCI field (for potential cross carrier scheduling)
  bool      cifPresent;
  uint8_t   cif;
  /// this is the carrier index where the DCI will be transmitted on
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
};

/**
* \brief Base class for storing the values of vendor specific parameters
*/
struct VendorSpecificValue 
{ 
  uint32_t dummy;
  /*to be extended*/
};

/**
 * \brief See section 4.3.3 vendorSpecifiListElement
 */
struct VendorSpecificListElement_s
{
  uint32_t type;
  uint32_t length;
  struct VendorSpecificValue *value;
};

/**
 * \brief See section 4.3.4 logicalChannelConfigListElement
 */
struct LogicalChannelConfigListElement_s
{
  uint8_t   logicalChannelIdentity;
  uint8_t   logicalChannelGroup;

  enum Direction_e
  {
    DIR_UL,
    DIR_DL,
    DIR_BOTH
  } direction;

  enum QosBearerType_e
  {
    QBT_NON_GBR,
    QBT_GBR
  } qosBearerType;

  uint8_t   qci;
  uint64_t  eRabMaximulBitrateUl;
  uint64_t  eRabMaximulBitrateDl;
  uint64_t  eRabGuaranteedBitrateUl;
  uint64_t  eRabGuaranteedBitrateDl;
};

/**
 * \brief See section 4.3.6 rachListElement
 */
struct RachListElement_s
{
  uint16_t  rnti;
  uint16_t  estimatedSize;
  uint8_t   carrierIndex;
};

/**
 * \brief See section 4.3.7 phichListElement
 */
struct PhichListElement_s
{
  uint16_t  rnti;
  enum Phich_e
  {
    ACK, NACK
  } phich;
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
};

/**
 * \brief See section 4.3.9 rlcPDU_ListElement
 */
struct RlcPduListElement_s
{
  uint8_t   logicalChannelIdentity;
  uint16_t  size;
};

/**
 * \brief See section 4.3.8 builDataListElement
 */
struct BuildDataListElement_s
{
  uint16_t  rnti;
  struct DlDciListElement_s dci;
  /* This is an array of CeBitmap_e enum flags. If one wants for example to signal TA in 1st TB and DRX and AD in 2nd one should:
   * ceBitmap[0] = ff_TA;  ceBitmap[1] = ff_DRX | ff_AD; */
  uint8_t ceBitmap[MAX_TB_LIST];
  uint8_t   nr_rlcPDU_List[MAX_TB_LIST];
  struct RlcPduListElement_s* rlcPduList[MAX_TB_LIST];
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
  /* Hex content of Activation/Deactivation MAC CE */
  uint8_t	activationDeactivationCE;
};

/**
 * \brief See section 4.3.10 buildRARListElement
 */
struct BuildRarListElement_s
{
  uint16_t  rnti;
  uint32_t  grant; 
  struct DlDciListElement_s dci;
  uint8_t   carrierIndex;
};

/**
 * \brief See section 4.3.11 buildBroadcastListElement
 */
struct BuildBroadcastListElement_s
{
  enum BroadcastType_e
  {
    ff_BCCH, ff_PCCH
  } type;
  uint8_t index;
  struct DlDciListElement_s dci;
  uint8_t   carrierIndex;
};

/**
 * \brief See section 4.3.12 ulInfoListElement
 */
struct UlInfoListElement_s
{
  uint16_t  puschTransmissionTimestamp;	//this timestamp identifies PUSCH transmission to which below information relates to
  uint16_t  rnti;
  uint16_t  ulReception[MAX_LC_LIST+1];
  enum ReceptionStatus_e
  {
    Ok, NotOk, NotValid
  } receptionStatus;
  int8_t    tpc;
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
};

/**
 * \brief See section 4.3.13 srListElement
 */
struct SrListElement_s
{
  uint16_t  rnti;
};

/**
 * \brief See section 4.3.15 macCEValue
 */
struct MacCeUlValue_u
{
  uint8_t   phr;
  uint8_t   crnti;
  uint8_t   bufferStatus[MAX_NR_LCG];
};

/**
 * \brief See section 4.3.14 macCEListElement
 */
struct MacCeUlListElement_s
{
  uint16_t  rnti;
  enum MacCeType_e
  {
    ff_BSR, ff_PHR, ff_CRNTI
  } macCeType;
  struct MacCeUlValue_u macCeValue;
};

/**
 * \brief macCEDLValue (new)
 */
struct MacCeDlValue_u
{
  // timing advance value is not included here as it is supposed to be filled in by the MAC (scheduler does not care)
  // dtx CE does not have a value
  // contention resolution value should also be filled in by the MAC and not by the scheduler
  uint8_t   ActicationDeactivation;
};

/**
 * \brief macCEDLListElement (new)
 */
struct MacCeDlListElement_s
{
  uint16_t  rnti;
  enum CeBitmap_e macCeType;
  struct MacCeDlValue_u macCeValue;
};


/**
 * \brief See section 4.3.16 drxConfig
 */
struct DrxConfig_s
{
  uint8_t   onDurationTimer;
  uint16_t  drxInactivityTimer;
  uint16_t  drxRetransmissionTimer;
  uint16_t  longDrxCycle;
  uint16_t  longDrxCycleStartOffset;
  uint16_t  shortDrxCycle;
  uint8_t   drxShortCycleTimer;
};

/**
 * \brief See section 4.3.17 spsConfig
 */
struct SpsConfig_s
{
  uint16_t  semiPersistSchedIntervalUl;
  uint16_t  semiPersistSchedIntervalDl;
  uint8_t   numberOfConfSpsProcesses;
  uint8_t   n1PucchAnPersistentListSize;
  uint16_t  n1PucchAnPersistentList[4];
  uint8_t   implicitReleaseAfter;
};

/**
 * \brief See section 4.3.18 srConfig
 */
struct SrConfig_s
{
  enum SetupRelease_e action;
  uint8_t   schedInterval;
  uint8_t   dsrTransMax;
};

/**
 * \brief See section 4.3.19 cqiConfig
 */
struct CqiConfig_s
{
  enum SetupRelease_e action;
  uint16_t  cqiSchedInterval;
  uint8_t   riSchedInterval;
};

/**
 * \brief See section 4.3.20 ueCapabilities
 */
struct UeCapabilities_s
{
  bool      halfDuplex;
  bool      intraSfHopping;
  bool      type2Sb1;
  uint8_t   ueCategory;
  bool      resAllocType1;
};

/**
 * \brief See section 4.3.22 siMessageListElement
 */
struct SiMessageListElement_s
{
  uint16_t  periodicity;
  uint16_t  length;
};

/**
 * \brief See section 4.3.21 siConfiguration
 */
struct SiConfiguration_s
{
  uint16_t  sfn;
  uint16_t  sib1Length;
  uint8_t   siWindowLength;
  uint8_t   nrSI_Message_List;
  struct SiMessageListElement_s *siMessageList;
};

/**
 * \brief See section 4.3.23 dlInfoListElement
 */
struct DlInfoListElement_s
{
  uint16_t  rnti;
  uint8_t   harqProcessId;
  uint8_t nr_harqStatus;
  enum HarqStatus_e
    {
      ff_ACK, ff_NACK, ff_DTX
    } harqStatus[MAX_TB_LIST];
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
};


/**
 * \brief Represents types of SCI reports for all CSI reporting modes. \a mode indicates which structure is held in \a report union.
 */
struct CsiReport_s
{
	uint8_t ri;

	enum CsiRepMode_e
	{
		P10, P11, P20, P21, A12, A22, A20, A30, A31
	} mode;

	union
	{
		struct A12Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t sbPmi[MAX_HL_SB];
		} A12Csi;

		struct A30Csi_s
		{
			uint8_t wbCqi;
			uint8_t sbCqi[MAX_HL_SB];
		} A30Csi;

		struct A31Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t sbCqi[MAX_HL_SB][MAX_TB_LIST];
			uint8_t wbPmi;
		} A31Csi;

		struct A20Csi_s
		{
			uint8_t wbCqi;
			uint8_t sbCqi;
			uint8_t sbList[MAX_UE_SELECTED_SB];
		} A20Csi;

		struct A22Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t sbCqi[MAX_TB_LIST];
			uint8_t wbPmi;
			uint8_t sbPmi;
			uint8_t sbList[MAX_UE_SELECTED_SB];
		} A22Csi;

		struct P10Csi_s
		{
			uint8_t wbCqi;
		} P10Csi;

		struct P11Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t wbPmi;
		} P11Csi;

		struct P20Csi_s
		{
			uint8_t wbCqi;
			uint8_t sbCqi;
			uint8_t bwPartIndex;	//range 0-3; to cover maximum number of BW parts (J)
			uint8_t sbIndex;		//range 0-3; to cover maximum number of subbands inside BW part (Nj)
		} P20Csi;

		struct P21Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t wbPmi;
			uint8_t sbCqi[MAX_TB_LIST];
			uint8_t bwPartIndex;	//range 0-3; to cover maximum number of BW parts (J)
			uint8_t sbIndex;		//range 0-3; to cover maximum number of subbands inside BW part (Nj)
		} P21Csi;
	} report;
};


/**
 * \brief Modified structure holding CSI report for single UE (for original structure see section 4.3.24 cqiListElement).
 */
struct CqiListElement_s
{
  uint16_t  rnti;
  struct CsiReport_s csiReport;
  uint8_t   servCellIndex;	//definition according to 36.331 'ServCellIndex'
};


/**
 * \brief See section 4.3.29 ulCQI
 */
struct UlCqi_s
{
  uint16_t rnti;
  uint16_t sinr[MAX_SINR_RB_LIST];
  enum UlCqiType_e
    {
      ff_SRS,
      ff_PUSCH,
      ff_PUCCH_1,
      ff_PUCCH_2,
      ff_PRACH
    } type;
    uint8_t servCellIndex;	//definition according to 36.331 'ServCellIndex'
};

/**
 * \brief See section 4.3.30 pagingInfoListElement
 */
struct PagingInfoListElement_s
{
  uint8_t   pagingIndex;
  uint16_t  pagingMessageSize;
  uint8_t   pagingSubframe;
  uint8_t   carrierIndex;
};

/**
 * \brief Describes the SCell configuration.
 */
struct ScellConfig_s
{
	/* Unique carrier identifier */
	uint8_t carrierIndex;
	/* Index of this SCell (RRC SCellIndex) */
	uint8_t	scellIndex;
	/* Indicates if cross-carrier scheduling shall be used or not on the SCell */
	bool useCrossCarrierScheduling;
	/* Index of the cell responsible for delivering scheduling for this SCell */
	uint8_t schedulingCellIndex;
	/* Starting OFDM symbol of PDSCH data region for this SCell */
	uint8_t pdschStart;
};

struct PdcchOfdmSymbolCountListElement_s
{
	/* Unique carrier identifier */
	uint8_t carrierIndex;
	/* Size of PDCCH in OFDM symbols */
	uint8_t pdcchOfdmSymbolCount;
};

/*@}*/

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_COMMON_H */
