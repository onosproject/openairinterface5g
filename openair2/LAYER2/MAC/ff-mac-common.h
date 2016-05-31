/**
 * @file ff-mac-common.h
 * @brief Implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 with extensions.
 * @details Contains definitions of common FAPI constants and structures.
 * @author Florian Kaltenberger, Maciej Wewior
 * @date March 2015
 * @email: florian.kaltenberger@eurecom.fr, m.wewior@is-wireless.com
 * @ingroup _mac
 */

#ifndef FF_MAC_COMMON_H
#define FF_MAC_COMMON_H

#include <stdint.h>
#include <stdbool.h>

#if defined (__cplusplus)
extern "C" {
#endif

/**
 * MAX_SCHED_CFG_LIST
 */
#define MAX_SCHED_CFG_LIST    10

/**
 * MAX_LC_LIST
 */
#define MAX_LC_LIST           10

/**
 * MAX_RACH_LIST
 */
#define MAX_RACH_LIST         30

/**
 * MAX_DL_INFO_LIST
 */
#define MAX_DL_INFO_LIST      30

/**
 * MAX_BUILD_DATA_LIST
 */
#define MAX_BUILD_DATA_LIST   30

/**
 * MAX_BUILD_RAR_LIST
 */
#define MAX_BUILD_RAR_LIST    10

/**
 * MAX_BUILD_BC_LIST
 */
#define MAX_BUILD_BC_LIST     3

/**
 * MAX_UL_INFO_LIST
 */
#define MAX_UL_INFO_LIST      30

/**
 * MAX_DCI_LIST
 */
#define MAX_DCI_LIST          30

/**
 * MAX_PHICH_LIST
 */
#define MAX_PHICH_LIST        30

/**
 * MAX_TB_LIST
 */
#define MAX_TB_LIST           2

/**
 * MAX_RLC_PDU_LIST
 */
#define MAX_RLC_PDU_LIST      30

/**
 * MAX_NR_LCG
 */
#define MAX_NR_LCG            4

/**
 * MAX_MBSFN_CONFIG
 */
#define MAX_MBSFN_CONFIG      5

/**
 * MAX_SI_MSG_LIST
 */
#define MAX_SI_MSG_LIST       32

/**
 * MAX_SI_MSG_SIZE
 */
#define MAX_SI_MSG_SIZE       65535

/**
 * MAX_PAGING_LIST
 */
#define MAX_PAGING_LIST		  30

/**
 * MAX_CQI_LIST
 */
#define MAX_CQI_LIST          30

/**
 * MAX_UE_SELECTED_SB
 * comes from Table 7.2.1-5 in \ref ref4 "[4]"
 */
#define MAX_UE_SELECTED_SB    6

/**
 * MAX_HL_SB
 * comes from Table 7.2.1-3 in \ref ref4 "[4]"
 */
#define MAX_HL_SB             13

/**
 * MAX_SINR_RB_LIST
 */
#define MAX_SINR_RB_LIST      100

/**
 * MAX_SR_LIST
 */
#define MAX_SR_LIST           30

/**
 * MAX_MAC_CE_LIST
 */
#define MAX_MAC_CE_LIST       30

/**
 * MAX_NUM_CCs
 */
#ifndef MAX_NUM_CCs
#error MAX_NUM_CCs not defined
#endif
//#define MAX_NUM_CCs           2

/**
 * Result of operation.
 */
enum Result_e
{
  ff_SUCCESS,//!< operation successful
  ff_FAILURE //!< operation failed
};

/**
 * Type of action.
 */
enum SetupRelease_e
{
  ff_setup, //!< setup
  ff_release//!< release
};

/**
 * Type of MAC CE transmitted from eNB.
 */
enum CeBitmap_e
{
  ff_TA		= 1 << 0,//!< Timing Advance Command MAC Control Element
  ff_DRX	= 1 << 1,//!< DRX Command MAC Control Element
  ff_CR		= 1 << 2,//!< UE Contention Resolution Identity MAC Control Element
  ff_AD		= 1 << 3 //!< Activation/Deactivation MAC Control Element
};

/**
 * Cyclic prefix length.
 */
enum NormalExtended_e
{
  ff_normal, //!< Normal cyclic prefix
  ff_extended//!< Extended cyclic prefix
};

/**
 * DL DCI representation.
 */
struct DlDciListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The RBs allocated to the UE.
	 */
	uint32_t rbBitmap;

	/**
	 * Resource allocation type1 field, see \ref ref4 "[4]", section 7.1.6.2.
	 * \n Range: 0,1
	 */
	uint8_t rbShift;

	/**
	 * Selected resource blocks subset for DL resource allocation type 1, see \ref ref4 "[4]", section 7.1.6.2.
	 * \n Range: 0..3
	 */
	uint8_t	rbgSubset;

	/**
	 * Type of resource allocation.
	 * \n Range: 0,1,2
	 */
	uint8_t resAlloc;

	/**
	 * The number of transport blocks.
	 * \n Range: 1..#MAX_TB_LIST
	 */
	uint8_t	nr_of_tbs;

	/**
	 * The size of the transport blocks in byte.
	 */
	uint16_t tbsSize[MAX_TB_LIST];

	/**
	 * The modulation and coding scheme of each TB, see \ref ref4 "[4]" section 7.1.7.
	 * \n Range: 0..31
	 */
	uint8_t mcs[MAX_TB_LIST];

	/**
	 * New data Indicator.
	 * \n Range: 0..1
	 */
	uint8_t ndi[MAX_TB_LIST];

	/**
	 * Redundancy version.
	 * \n Range: 0..3
	 */
	uint8_t rv[MAX_TB_LIST];

	/**
	 * CCE index used to send the DCI.
	 * \n Range: 0.88
	 */
	uint8_t cceIndex;

	/**
	 * The aggregation level.
	 * \n Range: 1,2,4,8
	 */
	uint8_t aggrLevel;

	/**
	 * Precoding information.
	 * \n Range:
	 * - 2 antenna_ports: 0..6
	 * - 4 antenna_ports: 0..50
	 */
	uint8_t precodingInfo;

	/**
	 * Format of the DCI.
	 */
	enum Format_e
	{
		ONE, ONE_A, ONE_B, ONE_C, ONE_D, TWO, TWO_A, TWO_B
	} format;

	/**
	 * See \ref ref4 "[4]", section 5.1.1.1.
	 * \n Range: -4,-1,0,1,3,4
	 */
	int8_t tpc;

	/**
	 * HARQ process number.
	 * \n Range: 0..7
	 */
	uint8_t harqProcess;

	/**
	 * Only for TDD.
	 * \n Range: 1,2,3,4
	 */
	uint8_t dai;

	/**
	 * See \ref ref4 "[4]", section 7.1.6.3.
	 */
	enum VrbFormat_e
	{
		VRB_DISTRIBUTED,//!< VRBs of distributed type
		VRB_LOCALIZED   //!< VRBs of localized type
	} vrbFormat;

	/**
	 * TB to CW swap flag, see \ref ref3 "[3]", section 5.3.3.1.5.
	 */
	bool tbSwap;

	/**
	 *
	 */
	bool spsRelease;

	/**
	 * Indicates if PDCCH is for PDCCH order.
	 */
	bool pdcchOrder;

	/**
	 * Preamble index. Only valid if \ref pdcchOrder == TRUE.
	 * \n Range: 0..63
	 */
	uint8_t preambleIndex;

	/**
	 * PRACH Mask index. Only valid if \ref pdcchOrder == TRUE.
	 * \n Range: 0..15
	 */
	uint8_t prachMaskIndex;

	/**
	 * The value for N_GAP.
	 */
	enum Ngap_e
	{
		GAP1, GAP2
	} nGap;

	/**
	 * The TBS index for Format 1A.
	 * \n Range: 2,3
	 */
	uint8_t tbsIdx;

	/**
	 * For Format 1D, see \ref ref4 "[4]" section 7.1.5.
	 * \n Range: 0,1
	 */
	uint8_t dlPowerOffset;

	/**
	 * DL PDCCH power boosting in dB.
	 * \n Range: -6..4
	 */
	int8_t pdcchPowerOffset;

	/**
	 * Indicates if Carrier Indicator Field shall be present in DCI (for cross carrier scheduling).
	 */
	bool cifPresent;

	/**
	 * Carrier Indicator Field value.
	 * \n Range: 0..7
	 */
	uint8_t cif;
};

/**
 * UL DCI (DCI type0) representation.
 */
struct UlDciListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The start RB allocated to the UE, see \ref ref4 "[4]", section 8.1.
	 * \n Range: 0..99
	 */
	uint8_t rbStart;

	/**
	 * The number of RBs allocated to the UE. see \ref ref4 "[4]", section 8.1.
	 * \n Range: 1..100
	 */
	uint8_t rbLen;

	/**
	 * The size of the transport block in byte.
	 */
	uint16_t tbSize;

	/**
	 * The modulation and coding scheme of each TB, see \ref ref4 "[4]" section 8.6.
	 * \n Range: 0..32
	 */
	uint8_t mcs;

	/**
	 * New data Indicator.
	 * \n Range: 0..1
	 */
	uint8_t ndi;

	/**
	 * CCE index used to send the DCI.
	 * \n Range: 0..88
	 */
	uint8_t cceIndex;

	/**
	 * The aggregation level.
	 * \n Range: 1,2,4,8
	 */
	uint8_t aggrLevel;

	/**
	 * See \ref ref3 "[3]", section 5.3.3.2. 3 means antenna selection is off.
	 * \n Range: 0,1,3
	 */
	uint8_t ueTxAntennaSelection;

	/**
	 * Hopping enabled flag, see \ref ref4 "[4]" section 8.4.
	 */
	bool hopping;

	/**
	 * Cyclic shift.
	 * \n Range: 0..7
	 */
	uint8_t n2Dmrs;

	/**
	 * Tx power control command, see \ref ref4 "[4]" section 5.1.1.1.
	 * \n Range: -4,-1,0,1,3,4
	 */
	int8_t tpc;

	/**
	 * Aperiodic CQI request flag. see\ref ref4 "[4]" section 7.2.1.
	 */
	bool cqiRequest;

	/**
	 * UL index, only for TDD.
	 * \n Range: 0,1,2,3
	 */
	uint8_t ulIndex;

	/**
	 * DL assignment index, only for TDD.
	 * \n Range: 1,2,3,4
	 */
	uint8_t dai;

	/**
	 * The frequency hopping bits, see \ref ref4 "[4]" section 8.4.
	 * \n Range: 0..4
	 */
	uint8_t freqHopping;

	/**
	 *  DL PDCCH power boosting in dB.
	 * \n Range: -6..4
	 */
	int8_t pdcchPowerOffset;

	/**
	 * Indicates if Carrier Indicator Field shall be present in DCI (for cross carrier scheduling).
	 */
	bool cifPresent;

	/**
	 * Carrier Indicator Field value.
	 * \n Range: 0..7
	 */
	uint8_t cif;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};

/**
* Base class for storing the values of vendor specific parameters.
*/
struct VendorSpecificValue 
{ 
  uint32_t dummy;
  /*to be extended*/
};

/**
 * Vendor specific parameter in TLV format.
 */
struct VendorSpecificListElement_s
{
	/**
	 * Indicating the type of the value. This types are examples, real types are implementation specific, examples are:
	 * PF_WEIGHT1 – The first weight used by a proportional fair scheduler
	 * PF_WEIGHT2 – The second weight used by a proportional fair scheduler
	 * CQI_AVG_FACTOR – The factor used for averaging CQIs in the scheduler.
	 */
	uint32_t type;

	/**
	 * The length of the actual value.
	 */
	uint32_t length;

	/**
	 * The actual value which will be set.
	 */
	struct VendorSpecificValue *value;
};

/**
 * Logical channel configuration.
 */
struct LogicalChannelConfigListElement_s
{
	/**
	 * The logical channel id, see \ref ref1 "[1]". Note: CCCH is preconfigured.
	 * \n Range: 1..10
	 */
	uint8_t logicalChannelIdentity;

	/**
	 * The LC group the LC is mapped to. 4 means no LCG is associated with the logical channel.
	 * \n Range: 0..3, 4
	 */
	uint8_t logicalChannelGroup;

	/**
	 * The direction of the logical channel.
	 */
	enum Direction_e
	{
		DIR_UL,
		DIR_DL,
		DIR_BOTH
	} direction;

	/**
	 * Guaranteed or non-guaranteed bit rate bearer.
	 * \n Range:
	 */
	enum QosBearerType_e
	{
		QBT_NON_GBR,
		QBT_GBR
	} qosBearerType;

	/**
	 * The QCI defined in \ref ref10 "[10]". The QCI is coded as defined in \ref ref7 "[7]", i.e the value indicates one less than the actual QCI value.
	 */
	uint8_t qci;

	/**
	 * In bit/s. For QBT_GBR only.
	 * \n Range: 0..10000000000
	 */
	uint64_t eRabMaximulBitrateUl;

	/**
	 * In bit/s. For QBT_GBR only.
	 * \n Range: 0..10000000000
	 */
	uint64_t eRabMaximulBitrateDl;

	/**
	 * In bit/s. For QBT_GBR only.
	 * \n Range: 0..10000000000
	 */
	uint64_t eRabGuaranteedBitrateUl;

	/**
	 * In bit/s. For QBT_GBR only.
	 * \n Range: 0..10000000000
	 */
	uint64_t eRabGuaranteedBitrateDl;
};

/**
 * RACH information.
 */
struct RachListElement_s
{
	/**
	 * The newly allocated t-c-rnti.
	 */
	uint16_t rnti;

	/**
	 * Estimated minimum size of first UL message in bits, based on received RACH preamble.
	 * \n Range: 56,144,208,256
	 */
	uint16_t estimatedSize;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;
};

/**
 * PHICH information.
 */
struct PhichListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t  rnti;

	/**
	 * ACK or NACK to be passed to the UE in the PHICH.
	 */
	enum Phich_e
	{
		ACK, NACK
	} phich;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};

/**
 * RLC PDU information.
 */
struct RlcPduListElement_s
{
	/**
	 * The logical channel ID, see \ref ref1 "[1]".
	 * \n Range: 0..10
	 */
	uint8_t logicalChannelIdentity;

	/**
	 * Maximum length of RLC PDU in bytes.
	 * \n Range: 1..9420
	 */
	uint16_t size;
};

/**
 * DL user data scheduling decision element.
 */
struct BuildDataListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t  rnti;

	/**
	 * The DL DCI configured for this UE. This may also indicate PDCCH order or SPS release or format 3/3A, in which case there is no associated PDSCH.
	 */
	struct DlDciListElement_s dci;

	/**
	 * The CEs scheduled for transmission for this TB. This is array of \ref CeBitmap_e enum flags. (TA-Timing Advance Command, DRX-DRX Command, CR-Contention Resolution, AD-Activation/Deactivation)
	 * \n Range: TA, DRX, CR, AD
	 */
	uint8_t ceBitmap[MAX_TB_LIST];

	/**
	 * The number of RLC PDUs to be built for each Transport Block.
	 * \n Range: 1..#MAX_RLC_PDU_LIST
	 */
	uint8_t nr_rlcPDU_List[MAX_TB_LIST];

	/**
	 * List of parameters for RLC PDU creation.
	 */
	struct RlcPduListElement_s* rlcPduList[MAX_TB_LIST];

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;

	/**
	 * Activation/Deactivation MAC Control Element value built by the scheduler, to be sent in this TTI.
	 */
	uint8_t	activationDeactivationCE;
};

/**
 * RAR scheduling decision element.
 */
struct BuildRarListElement_s
{
	/**
	 * The RNTI identifying the UE (in this case it is the Temporary C-RNTI).
	 */
	uint16_t rnti;

	/**
	 * 20 bit UL grant, see \ref ref4 "[4]" section 6.2.
	 */
	uint32_t grant;

	/**
	 * The DL DCI configured for this RAR.
	 */
	struct DlDciListElement_s dci;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;
};

/**
 * Broadcast scheduling decision element.
 */
struct BuildBroadcastListElement_s
{
	/**
	 * The type identifying the broadcast message.
	 */
	enum BroadcastType_e
	{
		ff_BCCH, ff_PCCH
	} type;

	/**
	 * The index of the broadcast message. This identifies which broadcast message (either SIB1, SIx or PCCH) should be transmitted.
	 * 0 – SIB1
	 * 1..31 – SIx
	 * 32..63 - PCCH
	 */
	uint8_t index;

	/**
	 * The DL DCI configured for BCCH and PCCH.
	 */
	struct DlDciListElement_s dci;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;
};

/**
 * PUSCH data reception information.
 */
struct UlInfoListElement_s
{
	/**
	 * Timestamp identifying PUSCH transmission to which this information relates to.
	 */
	uint16_t puschTransmissionTimestamp;

	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t  rnti;

	/**
	 * The amount of data in bytes in the MAC SDU received in subframe identified by \ref puschTransmissionTimestamp for the given logical channel.
	 */
	uint16_t  ulReception[MAX_LC_LIST+1];

	/**
	 * NotValid is used when no TB is expected. Ok/notOk indicates successful/unsuccessful reception of UL TB.
	 */
	enum ReceptionStatus_e
	{
		Ok, NotOk, NotValid
	} receptionStatus;

	/**
	 * Tx power control command, see \ref ref4 "[4]" section 5.1.1.1.
	 * \n Range: -4,-1,0,1,3,4
	 */
	int8_t   tpc;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};

/**
 * Scheduling Request information.
 */
struct SrListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;
};

/**
 * UL MAC CE value.
 */
struct MacCeUlValue_u
{
	/**
	 * The power headroom, see \ref ref1 "[1]" section 6.1.3.6. 64 means no valid PHR is available.
	 *\n Range: 0..63,64
	 */
	uint8_t phr;

	/**
	 * Indicates that a C-RNTI MAC CE was received. The value is not used.
	 */
	uint8_t crnti;

	/**
	 * The value 64 indicates that the buffer status for this LCG should not to be updated. Always all 4 LCGs are present, see \ref ref1 "[1]" 6.1.3.1.
	 */
	uint8_t bufferStatus[MAX_NR_LCG];
};

/**
 * UL MAC CE information.
 */
struct MacCeUlListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t  rnti;

	/**
	 * Mac Control Element Type.
	 */
	enum MacCeType_e
	{
		ff_BSR, ff_PHR, ff_CRNTI
	} macCeType;

	/**
	 * MAC CE value.
	 */
	struct MacCeUlValue_u macCeValue;
};


/**
 * DRX configuration.
 */
struct DrxConfig_s
{
	/**
	 * Timer in subframes, see \ref ref1 "[1]".
	 * \n Range: 1,2,3,4,5,6,8,10,20,30,40,50,60,80,100,200
	 */
	uint8_t onDurationTimer;

	/**
	 * Timer in subframes, see \ref ref1 "[1]".
	 * \n Range: 1,2,3,4,5,6,8,10,20,30,40,50,60,80,100,200,300,500,750,1280,1920,2560
	 */
	uint16_t drxInactivityTimer;

	/**
	 * Timer in subframes, see \ref ref1 "[1]".
	 * \n Range: 1,2,4,6,8,16,24,33
	 */
	uint16_t drxRetransmissionTimer;

	/**
	 * Long DRX cycle in subframes, see \ref ref1 "[1]".
	 * \n Range: 10,20,32,40,64,80,128,160,256,320,512,640,1024,1280,2048,2560
	 */
	uint16_t longDrxCycle;

	/**
	 * Long DRX cycle offset, see \ref ref1 "[1]".
	 * \n Range: 0..2559
	 */
	uint16_t longDrxCycleStartOffset;

	/**
	 * Short DRX cycle in subframes, see \ref ref1 "[1]".
	 * \n Range: 2,5,8,10,16,10,21,40,64,80,128,160,256,320,512,640,OFF
	 */
	uint16_t shortDrxCycle;

	/**
	 * Timer in subframes, see \ref ref1 "[1]".
	 * \n Range: 1..16
	 */
	uint8_t drxShortCycleTimer;
};

/**
 * Semi-persistent scheduling configuration.
 */
struct SpsConfig_s
{
	/**
	 * SPS scheduling interval in UL in subframes.
	 * \n Range: 10,20,32,40,64,80,128,160,320,640
	 */
	uint16_t semiPersistSchedIntervalUl;

	/**
	 * SPS scheduling interval in DL in subframes.
	 * \n Range: 10,20,32,40,64,80,128,160,320,640
	 */
	uint16_t semiPersistSchedIntervalDl;

	/**
	 * Number of SPS HARQ processes, see \ref ref1 "[1]".
	 * \n Range: 1..8
	 */
	uint8_t numberOfConfSpsProcesses;

	/**
	 * The size of the \ref n1PucchAnPersistentList list. When spsConfig is included in CschedUeConfigReq() this parameters is ignored.
	 * \n Range: 0..4
	 */
	uint8_t n1PucchAnPersistentListSize;

	/**
	 * See \ref ref4 "[4]" section 10.1. When spsConfig is included in CschedUeConfigReq() this parameters is ignored.
	 * \n Range: 0..2047
	 */
	uint16_t n1PucchAnPersistentList[4];

	/**
	 * Number of empty transmission, see \ref ref1 "[1]" section 5.10.2. When spsConfig is included in CschedUeConfigReq() this parameters is ignored.
	 * \n Range: 2,3,4,8
	 */
	uint8_t implicitReleaseAfter;
};

/**
 * Scheduling Request configuration.
 */
struct SrConfig_s
{
	/**
	 * Indicates if SR config should be released or changed.
	 */
	enum SetupRelease_e action;

	/**
	 * SR scheduling interval in subframes.
	 * \n Range: 5,10,20,40,80
	 */
	uint8_t schedInterval;

	/**
	 * See \ref ref1 "[1]", section 5.4.4.
	 * \n Range: 4,8,16,32,64
	 */
	uint8_t dsrTransMax;
};

/**
 * CQI configuration.
 */
struct CqiConfig_s
{
	/**
	 * Indicates if CQI config should be released or changed.
	 */
	enum SetupRelease_e action;

	/**
	 * CQI scheduling interval in subframes.
	 * \n Range: 1,2,5,10,20,32,40,64,80,128,160
	 */
	uint16_t cqiSchedInterval;

	/**
	 * RI scheduling interval in subframes.
	 * \n Range: 1,2,4,8,16,20
	 */
	uint8_t riSchedInterval;
};

/**
 * UE capabilities.
 */
struct UeCapabilities_s
{
	/**
	 * UE only supports half-duplex FDD operation.
	 *
	 */
	bool halfDuplex;

	/**
	 * UE support of intra-subframe hopping.
	 */
	bool intraSfHopping;

	/**
	 * UE supports type 2 hopping with n_sb > 1.
	 */
	bool type2Sb1;

	/**
	 * The UE category
	 * \n Range: 1..5
	 */
	uint8_t ueCategory;

	/**
	 * UE support for resource allocation type 1.
	 */
	bool resAllocType1;
};

/**
 * System Information message.
 */
struct SiMessageListElement_s
{
	/**
	 * Periodicity of the SI-message Unit in radio frames.
	 * \n Range: 8, 16, 32, 64, 128, 256, 512
	 */
	uint16_t periodicity;

	/**
	 * The length of SI message. Unit in bytes.
	 * \n Range: 1..#MAX_SI_MSG_SIZE
	 */
	uint16_t length;
};

/**
 * System Information configuration.
 */
struct SiConfiguration_s
{
	/**
	 * Frame number to apply this configuration.
	 * \n Range: 0..1023
	 */
	uint16_t sfn;

	/**
	 * The length of the SIB 1 message. Unit in bytes.
	 * \n Range: 1..#MAX_SI_MSG_SIZE
	 */
	uint16_t sib1Length;

	/**
	 * Common SI scheduling window for all SIs. Unit in subframes.
	 * \n Range: 1,2,5,10,15,20,40
	 */
	uint8_t siWindowLength;

	/**
	 * The number of SI messages on the \ref siMessageList list.
	 * \n Range: 0..#MAX_SI_MSG_LIST
	 */
	uint8_t nrSI_Message_List;

	/**
	 * List of SI messages to be sent. The index will later be used to identify the message in the \ref BuildBroadcastListElement_s.
	 */
	struct SiMessageListElement_s *siMessageList;
};

/**
 * DL HARQ information.
 */
struct DlInfoListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * HARQ process ID. 8 is not present.
	 */
	uint8_t harqProcessId;

	/**
	 * The size of the HARQ status list.
	 * \n Range: 1..#MAX_TB_LIST
	 */
	uint8_t nr_harqStatus;

	/**
	 * HARQ status for the above process.
	 */
	enum HarqStatus_e
	{
		ff_ACK, ff_NACK, ff_DTX
	} harqStatus[MAX_TB_LIST];

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};


/**
 * Represents types of CSI reports for all CSI reporting modes.
 */
struct CsiReport_s
{
	/**
	 * The last received rank indication.
	 * \n Range: 1..4
	 */
	uint8_t ri;

	/**
	 * CSI reporting mode.
	 */
	enum CsiRepMode_e
	{
		P10, P11, P20, P21, A12, A22, A20, A30, A31
	} mode;

	/**
	 * Union discriminated by \ref mode. See \ref ref4 "[4]" sections 7.2.1 and 7.2.2.
	 */
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
			/**
			 * Range: 0-3, to cover maximum number of BW parts (J).
			 */
			uint8_t bwPartIndex;
			/**
			 * Range: 0-3, to cover maximum number of subbands inside BW part (Nj)
			 */
			uint8_t sbIndex;
		} P20Csi;

		struct P21Csi_s
		{
			uint8_t wbCqi[MAX_TB_LIST];
			uint8_t wbPmi;
			uint8_t sbCqi[MAX_TB_LIST];
			/**
			 * Range: 0-3, to cover maximum number of BW parts (J).
			 */
			uint8_t bwPartIndex;
			/**
			 * Range: 0-3, to cover maximum number of subbands inside BW part (Nj)
			 */
			uint8_t sbIndex;
		} P21Csi;
	} report;
};


/**
 * CSI report for single UE.
 */
struct CqiListElement_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * CSI report.
	 */
	struct CsiReport_s csiReport;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};


/**
 * UL channel state.
 */
struct UlCqi_s
{
	/**
	 * The RNTI identifying the UE.
	 */
	uint16_t rnti;

	/**
	 * The SINR measurement based on the resource given in type. In case of PUCCH only the first index is used. For PRACH the first 6 indices are used. For PUSCH and SRS each index represents one RB.
	 * The SINR is given in dB. See \ref fapiExtDoc_ul_sinr "UL channel state reporting" for exact interpretation of the array.
	 * Format of each SINR value: fixed point Q12.3 format with sign.
	 */
	int16_t sinr[MAX_SINR_RB_LIST];

	/**
	 * Type of the SINR measurement.
	 */
	enum UlCqiType_e
	{
		ff_SRS,    //!< measurement comes from SRS
		ff_PUSCH,  //!< measurement comes from last PUSCH transmission
		ff_PUCCH_1,//!< measurement done on Format 1 resource
		ff_PUCCH_2,//!< measurement done on Format 2 resource
		ff_PRACH   //!< measurement comes from PRACH transmission
	} type;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the UE, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t servCellIndex;
};

/**
 * Paging information.
 */
struct PagingInfoListElement_s
{
	/**
	 * The index used to identify the scheduled message, will be returned in \ref SchedDlConfigInd_callback_t "SchedDlConfigInd()".
	 * \n Range: 32..63
	 */
	uint8_t pagingIndex;

	/**
	 * The size of the paging message.
	 */
	uint16_t pagingMessageSize;

	/**
	 * The subframe during which the message shall be sent.
	 * \n Range: 0..9
	 */
	uint8_t pagingSubframe;

	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;
};

/**
 * SCell configuration.
 */
struct ScellConfig_s
{
	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;

	/**
	 * SCell index as to be sent in RRCConnectionReconfiguration (IE SCellIndex defined in RRC specification), see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 * \n Range: 1..7
	 */
	uint8_t	scellIndex;

	/**
	 * Indicates if cross carrier scheduling shall be used on this SCell. If so, detailed cross configuration is given in \ref schedulingCellIndex and \ref pdschStart.
	 */
	bool useCrossCarrierScheduling;

	/**
	 * Indicates which cell signals the downlink allocations and uplink grants, if applicable, for the concerned SCell.
	 * \n Range: 0..7
	 */
	uint8_t schedulingCellIndex;

	/**
	 * Starting OFDM symbol of PDSCH data region for this Scell.
	 * \n Range: 1..4
	 */
	uint8_t pdschStart;
};

/**
 * PDCCH symbol count.
 */
struct PdcchOfdmSymbolCountListElement_s
{
	/**
	 * Component carrier identifier, uniquely identifies carrier within the eNB, see \ref fapiExtDoc_indices_sec "PcellIndex/ScellIndex’ing".
	 */
	uint8_t carrierIndex;


	/**
	 * Current size of PDCCH.
	 * \n Range: 0..4
	 */
	uint8_t pdcchOfdmSymbolCount;
};

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_COMMON_H */
