/**
 * @file ff-mac-init.h
 * @brief Implementation of the Femto Forum LTE MAC Scheduler Interface Specification v1.11 with extensions.
 * @details Contains init and shutdown function declarations, that the MAC shall call upon the scheduler.
 * @author Florian Kaltenberger, Maciej Wewior
 * @date March 2015
 * @email: florian.kaltenberger@eurecom.fr, m.wewior@is-wireless.com
 * @ingroup _mac
 */

/**
 * @mainpage LTE MAC Scheduler Interface
 * This is an implementation of FemtoForum (today <a href="http://www.smallcellforum.org">Small Cell Forum</a>) originated LTE MAC Scheduler Interface v1.11.
 * Current version of the API was developed by <a href="http://www.is-wireless.com">IS-Wireless</a> and <a href="http://www.eurecom.fr/en">Eurecom</a> and is an extension of original FAPI. Extensions are:
 * @li support for Carrier Aggregation
 * @li improved CQI report structures
 * @li bug fixes
 *
 * <hr>
 * @section fapiOrigDoc_sec Original FAPI documentation
 * This section contains the copy of FAPI documentation that was included in the original "LTE MAC Scheduler Interface v1.11" document by FemtoForum (Document number: FF_Tech_001_v1.11
 * , Date issued: 12-10-2010, Document status: Document for public distribution).
 * @subsection fapiOrigDoc_scope_sec Scope
 * This document specifies the MAC Scheduler interface. The goal of this interface specification is to allow the use of a wide range of schedulers which can be plugged
 * into the eNodeB and to allow for standardized interference coordination interface to the scheduler.
 * @subsection fapiOrigDoc_overview_sec Interface overview
 * The MAC scheduler is part of MAC from a logical view and the MAC scheduler should be independent from the PHY interface.
 * \n The description in this interface does not foresee any specific implementation of the interface. What is specified in this document is the structure of the parameters.
 * In order to describe the interface in detail the following model is used:
 * \n The interface is defined as a service access point offered by the MAC scheduler to the remaining MAC functionality, as shown in Figure 1. A _REQ primitive is from MAC to the MAC scheduler.
 * A _IND/_CNF primitives are from the MAC scheduler to the MAC. The description using primitives does not foresee any specific implementation and is used for illustration purposes.
 * Therefore an implementation could be message-based or function-based interface. Timing constrains applicable to the MAC scheduler are not yet specified.
 * \n For the MAC scheduler interface specification a push-based concept is employed, that is all parameters needed by the scheduler are passed to the scheduler at specific times
 * rather than using a pull-based concept (i.e. fetching the parameters from different places as needed). The parameters specified are as far as possible aligned with the 3GPP specifications.
 * @image html fapi_overview.jpg
 * @image latex fapi_overview.eps
 * Figure 1 shows the functionality split between the MAC scheduler and the remaining MAC. For the purposes of describing the MAC scheduler interface the MAC consists of a control block and a subframe
 * block, which uses the CSCHED and SCHED SAP respectively. The subframe block triggers the MAC scheduler every TTI and receives the scheduler results. The control block forwards control information to the MAC
 * scheduler as necessary. The scheduler consists of the following blocks:
 * @li <b>UL</b> Is responsible for scheduling of the PUSCH resources.
 * @li <b>DL</b> Is responsible for scheduling of the PDSCH resources.
 * @li <b>PDCCH/RACH</b> Is responsible for shared resources between UL and DL.
 * @li <b>HARQ</b> Is responsible for handling HARQ retransmissions, keeping track of the number of retransmissions and redundancy versions.
 * @li <b>Cell Cfg</b> Stores the UE configuration needed by the MAC scheduler.
 * @li <b>UE Cfg</b> Stores the UE configuration needed by the MAC scheduler.
 * @li <b>LC Cfg</b> Stores the logical channel configuration needed by the MAC scheduler.
 * @li <b>Sched Cfg</b> Stores the scheduler-specific configuration needed by the MAC scheduler.
 * @subsection fapiOrigDoc_detailed_sec Detailed interface description
 * In the following section the messages exchanged at the SAPs are specified.
 * @subsubsection fapiOrigDoc_detailed_csched_sec CSCHED – MAC Scheduler Control SAP
 * Below list specifies which configuration messages can be used to configure the scheduler. There is no restriction on the timing of when these messages can be sent, except where otherwise
 * noted in the message description. The RNTI and, if available, the LCID are used to identity the UE/LC between the MAC scheduler and the MAC. In case of a reconfiguration message
 * all parameters previously configured in a message have to be resend, otherwise parameters not present are removed in the scheduler (i.e. no delta configuration is possible).
 * @paragraph csched_par_1 CSCHED primitives from MAC to scheduler
 * @li CschedCellConfigReq()
 * @li CschedUeConfigReq()
 * @li CschedLcConfigReq()
 * @li CschedLcReleaseReq()
 * @li CschedUeReleaseReq()
 * @paragraph csched_par_2 CSCHED Primitives from scheduler to MAC
 * @li \ref CschedCellConfigCnf_callback_t "CschedCellConfigCnf()"
 * @li \ref CschedUeConfigCnf_callback_t "CschedUeConfigCnf()"
 * @li \ref CschedLcConfigCnf_callback_t "CschedLcConfigCnf()"
 * @li \ref CschedLcReleaseCnf_callback_t "CschedLcReleaseCnf()"
 * @li \ref CschedUeReleaseCnf_callback_t "CschedUeReleaseCnf()"
 * @li \ref CschedUeConfigUpdateInd_callback_t "CschedUeConfigUpdateInd()"
 * @li \ref CschedCellConfigUpdateInd_callback_t "CschedCellConfigUpdateInd()"
 * @subsubsection fapiOrigDoc_detailed_sched_sec SCHED - MAC Scheduler SAP
 * @paragraph sched_par_1 SCHED primitives from MAC to scheduler
 * @li SchedDlTriggerReq()
 * @li SchedDlRlcBufferReq()
 * @li SchedDlPagingBufferReq()
 * @li SchedDlMacBufferReq()
 * @li SchedDlRachInfoReq()
 * @li SchedDlCqiInfoReq()
 * @li SchedUlTriggerReq()
 * @li SchedUlNoiseInterferenceReq()
 * @li SchedUlSrInfoReq()
 * @li SchedUlMacCtrlInfoReq()
 * @li SchedUlCqiInfoReq()
 * @paragraph sched_par_2 SCHED primitives from scheduler to MAC
 * @li \ref SchedDlConfigInd_callback_t "SchedDlConfigInd()"
 * @li \ref SchedUlConfigInd_callback_t "SchedUlConfigInd()"
 * @subsection fapiOrigDoc_scenarios_sec Scenarios
 * @subsubsection scenarios_1 Cell Setup
 * @image html cell_setup.jpg
 * @image latex cell_setup.eps
 * @subsubsection scenarios_2 RACH procedure
 * Mind!: this original sequence diagram has been found bugs and the new random access diagram has been created, see \ref fapiExtDoc_rap_sec "new random access procedure".
 * @image html random_access_old.jpg
 * @image latex random_access_old.eps
 * @subsubsection scenarios_3 UE configuration
 * @image html ue_conf.jpg
 * @image latex ue_conf.eps
 * @subsubsection scenarios_4 Radio Bearer Setup
 * @image html radio_bearer_setup.jpg
 * @image latex radio_bearer_setup.eps
 * @subsubsection scenarios_5 Handling of logical channel buffer status
 * Mind!: this original sequence diagram has improper name of the primitive for reporting DL buffer status, the proper name is SCHED_DL_RLC_BUFFER_REQ.
 * @image html buffer_status.jpg
 * @image latex buffer_status.eps
 * @subsubsection scenarios_6 DRB release
 * @image html drb_release.jpg
 * @image latex drb_release.eps
 * @subsubsection scenarios_7 DRB UE release
 * @image html ue_release.jpg
 * @image latex ue_release.eps
 * @subsubsection scenarios_8 UE configuration update by MAC scheduler
 * @image html conf_update.jpg
 * @image latex conf_update.eps
 * @subsubsection scenarios_9 Scheduler Subframe flow
 * @image html subframe_flow.jpg
 * @image latex subframe_flow.eps
 * @subsection fapiOrigDoc_appendix Appendix A: Performance and Functional Requirements for the LTE Femtocell Scheduler API
 * This appendix provides a high level overview of performance and functionality requirements for LTE schedulers that utilize the LTE femtocell scheduler API framework being defined by Femto Forum WG2.
 * These requirements are not totally comprehensive but represent a set of basic requirements that would be reasonably expected by an operator from an LTE scheduler residing in an LTE home eNodeB.
 * @li Satisfy latency and packet error loss characteristics of each QCI class standardized in 3GPP 23.203 Table 6.1.7 under the following conditions:
 * - Single user case: one user accesses any one of the example services in below table via a home eNodeB.
 * - Multiple user/services case: one or several users simultaneously access more than one of the example services in below table via a home eNodeB.
 * @li Satisfy Guaranteed Bit Rate (GBR), Minimum Bit Rate (MBR, as applicable, for each service data flow managed by the scheduler under the following conditions:
 * - Single user case: one user accesses any one of the example services in below table via a home eNodeB.
 * - Multiple user/services case: one or several users simultaneously access more than one of the example services in below table via a home eNodeB.
 * @li Enforce downlink maximum bit rate for sum of downlink bearers based on UE-AMBR and APN-AMBR (for non-GBR flows). Enforce corresponding uplink maximum bit rates.
 * @li Interact with admission and load control mechanisms to ensure that new users are admitted only when QoS requirements of existing and newly added users/bearers can be met.
 * @li When system load exceeds certain pre-defined thresholds, judiciously select lowest priority bearers for service downgrade.
 * @li Dynamically perform frequency selective and frequency diverse scheduling (localized and distributed virtual resource blocks) depending upon channel conditions, QoS requirements, etc.
 * @li Dynamically adapt transport block size selection, MIMO mode selection, and rank depending upon Channel Quality Indicator (CQI), Pre-coding Matrix Indicator (PMI), and Rank Indication (RI) feedback
 * from UEs while taking into account the status of data buffers.
 * @li Provide higher priority to HARQ re-transmissions versus new transmissions for a bearer.
 * @li Monitor current packet allocations and overall system load.
 *
 * <hr>
 *
 * @section fapiExtDoc_sec FAPI extensions
 * This section contains information related to FAPI extensions.
 *
 * @subsection fapiExtDoc_cellAct_sec SCell activation/deactivation
 * Decision about SCell activation/deactivation is always taken inside the scheduler and in such cases the scheduler generates proper MAC CE and schedules it for sending to the UE.
 *
 * @subsection fapiExtDoc_indices_sec PcellIndex/ScellIndex’ing
 * There are 3 kinds of component carrier indices in the interface:
 * @li carrierIndex – this is global eNB identifier that clearly and uniquely identifies component carrier within eNB. It does not need to be related to any 3gpp defined indices,
 * it is not related to RRC ServCellIndex value at all. Scheduler (and the other side of the interface) uses this index to signal which component carrier the data concerns.
 * @li servCellIndex - this is 1 to 1 copy of ServCellIndex from RRCConnectionReconfiguration message that configures SCell(s) for the UE. It is used to identify serving cells for given UE.
 * @li scellIndex – this is 1 to 1 copy of SCellIndex from RRCConnectionReconfiguration message that configures SCell(s) for the UE. It is used to identify SCells for given UE.
 * Example: \n
 * We have 3 carriers configured in the eNB:
 * @li CC1 – carrierIndex = 10
 * @li CC2 – carrierIndex = 15
 * @li CC3 – carrierIndex = 4
 * At some moment we may have 3 UEs connected to eNB having example CC configurations:
 * @li UE1 – has 2 CCs:
 * - PCell – CC with carrierIndex 10, servCellIndex 0
 * - SCell – CC with carrierIndex 15 and scellIndex=servCellIndex=1
 * @li UE2 – has 3 CCs:
 * - PCell – CC with carrierIndex 4, servCellIndex 0
 * - SCell – CC with carrierIndex 10 and scellIndex=servCellIndex=1
 * - SCell – CC with carrierIndex 15 and scellIndex=servCellIndex=2
 * @li UE1 – has 2 CCs:
 * - PCell – CC with carrierIndex 15, servCellIndex 0
 * - SCell – CC with carrierIndex 10 and scellIndex=servCellIndex=3
 *
 * @subsection fapiExtDoc_rap_sec Random access procedure
 * This section contains new fixed random access procedure sequence diagram.
 * @image html random_access_new.jpg
 * @image latex random_access_new.eps
 *
 * @subsection fapiExtDoc_lcreconf_sec Logical channel reconfiguration
 * Logical channel (re)configuration CschedLcConfigReqParameters::reconfigureFlag flag works as follows:
 * @li 'true' means that CschedLcConfigReq() contains the whole new logical channel configuration (except CCCH, which is always by default configured) - as the result
 * all current logical channels (except logical channel 0, CCCH) are removed and replaced by the new config
 * @li 'false' means that CschedLcConfigReq() contains new logical channel(s) to be added to existing channel config - as the result current logical channels are not touched just new one(s) are appended
 *
 * @subsection fapiExtDoc_timing_sec Scheduler timing
 * There're a few issues that must be taken into account regarding timing in the FAPI scheduler:
 * @li PUSCH scheduling is always 4 subframes ahead (PDCCH carries DCI0s that inform about PUSCH allocations +4 subframes)
 * @li the stack may, to have the scheduling decision on time, send the scheduling trigger in advance, that is send the scheduling trigger containing timestamp N at time N-k
 * @li HARQ on UL is synchronous meaning that the moment when the HARQ ACK/NACK is received indicates which HARQ process it relates to
 *
 * Let's see the following example of 2 radio frames below consisting of 20 subframes A-U:
 * <table border="1" style="width:50%">
 * <tr><td>A</td><td bgcolor="#00FF00">B</td><td>C</td><td bgcolor="#FF0000">D</td><td>E</td><td>F</td><td>G</td><td bgcolor="#FF0000">H</td><td>I</td><td bgcolor="#00FF00">J</td><td>K</td><td>L</td><td>M</td><td>N</td><td>O</td><td>P</td><td>R</td><td>S</td><td>T</td><td>U</td></tr>
 * </table>
 * The following example scenario is valid:
 * 1. B - scheduler receives DL and UL scheduling triggers with 'sfnSf' indicating D. The scheduler schedules D's PDCCH and PDSCH and D+4 (thus H) PUSCH and returns the scheduling decision when ready.
 * 2. D - eNB stack uses allocated PDSCH resources to send DL data.
 * 3. H - UE uses allocated PUSCH resources to send UL data. eNB stack starts decoding PUSCH data.
 * 4. J - eNB stack passes (to the scheduler) information about decoding of PUSCH data received in H.
 *
 * @subsection fapiExtDoc_timestamp_sec Timestamp coding
 * Timestamps in FAPI are conveyed by means of 16 bit unsigned integers, whose bits are coded as below:
 * - bit 0-3 subframe number
 * - bit 4-13 frame number
 *
 * @subsection fapiExtDoc_ul_sinr UL channel state reporting
 * UL channel state is reported as the array of SINRs for all/subset of PRBs. Channel state might be reported based on the following information:
 * @li SRS measurement - in this case the SINR array contains SINR values for the whole UL bandwidth thus number of valid elements is N_UL_PRB, where element '0' holds SINR for PRB '0', while element 'N_UL_PRB' holds SINR
 * for PRB 'N_UL_RB-1'
 * @li PRACH transmission - in this case the SINR array contains SINR values for the PRACH band - number of valid elements is 6 (PRACH bandwidth), where element '0' holds SINR for PRB 'prachFreqOffset'
 * (\ref CschedCellConfigReqParametersListElement::prachFreqOffset), while element '6' holds SINR for PRB 'prachFreqOffset+5'
 * @li PUSCH transmission - in this case the SINR array contains measurements measured during very last PUSCH transmission; here's the example when PUSCH transmission occurred upon the DCI0 allocation <2,7>.
 * 2 cases are possible here, depending on if PUSCH hopping was used:
 * - no PUSCH hopping is used - in such case the transmission happened on PRBs <2,7> so the channel measurement comes exactly from this range - number of valid elements in the SINR array is 6, where element '0'
 * holds SINR for PRB 2, while element '5' holds SINR for PRB 7
 * - PUSCH hopping is used - in such case the transmission happens on different set of PRBs for both slots, let's say transmission in slot0 occurred on PRBs <2,7>, while on the 2nd slot on PRBs <20,25>; in such case
 * number of valid elements is 12, where element '0' holds SINR for PRB 2, element '5' SINR for PRB 7, element '6' SINR for PRB 20, element '11' SINR for PRB 25; always slot0 then slot1, always from smaller PRBs to bigger
 *
 * @section fapiDoc_sec External documentation
 * @anchor ref1 [1] 3GPP TS 36.321: "Evolved Universal Terrestrial Radio Access (E-UTRA); Medium Access Control (MAC) protocol specification (Release 10)" \n
 * @anchor ref2 [2] 3GPP TS 36.211: "Evolved Universal Terrestrial Radio Access (E-UTRA); Physical Channels and Modulation (Release 10)" \n
 * @anchor ref3 [3] 3GPP TS 36.212: "Evolved Universal Terrestrial Radio Access (E-UTRA); Multiplexing and channel coding (Release 10)" \n
 * @anchor ref4 [4] 3GPP TS 36.213: "Evolved Universal Terrestrial Radio Access (E-UTRA); Physical Layer Procedures (Release 10)" \n
 * @anchor ref5 [5] 3GPP TS 36.331: "Evolved Universal Terrestrial Radio Access (E-UTRA); Radio Resource Control (RRC); Protocol Specification (Release 10)" \n
 * @anchor ref6 [6] 3GPP TS 36.133: "Evolved Universal Terrestrial Radio Access (E-UTRA); Requirements for support of radio resource management (Release 10)" \n
 * @anchor ref7 [7] 3GPP TS 36.413: "Evolved Universal Terrestrial Radio Access Network (E-UTRAN); S1 Application Protocol (S1AP) (Release 10)" \n
 * @anchor ref8 [8] 3GPP TS 36.314: "Evolved Universal Terrestrial Radio Access (E-UTRA); Layer 2 - Measurements (Release 10)" \n
 * @anchor ref9 [9] 3GPP TS 36.214: "Evolved Universal Terrestrial Radio Access (E-UTRA); Physical layer; Measurements (Release 10)" \n
 * @anchor ref10 [10] 3GPP TS 23.203 "Universal Mobile Telecommunications System (UMTS); LTE; Policy and charging control architecture (Release 10)" \n
 */

#ifndef FF_MAC_INIT_H
#define FF_MAC_INIT_H

#if defined (__cplusplus)
extern "C" {
#endif

#include "ff-mac-callback.h"

/**
 * Initialize the scheduler instance. Must be called during system initialization before any communication with the scheduler occurs.
 * @param callback_data Pointer to the callback data to be put by the scheduler when calling callback functions.
 * @param SchedDlConfigInd SchedDlConfigInd callback pointer
 * @param SchedUlConfigInd SchedUlConfigInd callback pointer
 * @param CschedCellConfigCnf CschedCellConfigCnf callback pointer
 * @param CschedUeConfigCnf CschedUeConfigCnf callback pointer
 * @param CschedLcConfigCnf CschedLcConfigCnf callback pointer
 * @param CschedLcReleaseCnf CschedLcReleaseCnf callback pointer
 * @param CschedUeReleaseCnf CschedUeReleaseCnf callback pointer
 * @param CschedUeConfigUpdateInd CschedUeConfigUpdateInd callback pointer
 * @param CschedCellConfigUpdateInd CschedCellConfigUpdateInd callback pointer
 * @return Scheduler context pointer
 */
void *SchedInit(
    void                                 *callback_data,
    SchedDlConfigInd_callback_t          *SchedDlConfigInd,
    SchedUlConfigInd_callback_t          *SchedUlConfigInd,
    CschedCellConfigCnf_callback_t       *CschedCellConfigCnf,
    CschedUeConfigCnf_callback_t         *CschedUeConfigCnf,
    CschedLcConfigCnf_callback_t         *CschedLcConfigCnf,
    CschedLcReleaseCnf_callback_t        *CschedLcReleaseCnf,
    CschedUeReleaseCnf_callback_t        *CschedUeReleaseCnf,
    CschedUeConfigUpdateInd_callback_t   *CschedUeConfigUpdateInd,
    CschedCellConfigUpdateInd_callback_t *CschedCellConfigUpdateInd);

/**
 * Shutdown the scheduler instance.
 * @param scheduler Scheduler context pointer
 */
void SchedShutdown(void* scheduler);

#if defined (__cplusplus)
}
#endif

#endif /* FF_MAC_INIT_H */
