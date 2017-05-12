/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file proto_nb_iot.h
 * \brief RRC functions prototypes for eNB and UE for NB-IoT
 * \author Navid Nikaein, Raymond Knopp and Michele Paffetti
 * \date 2010 - 2014
 * \email navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
 * \version 1.0

 */
/** \addtogroup _rrc
 *  @{
 */

#include "RRC/LITE/defs_nb_iot.h"
#include "pdcp.h"
#include "rlc.h"
#include "extern_nb_iot.h"

/*NOTE: no static function should be declared in this header file (e.g. init_SI_NB)*/

/*------------------------common_nb_iot.c----------------------------------------*/

/** \brief configure  BCCH & CCCH Logical Channels and associated rrc_buffers, configure associated SRBs
 */
void openair_rrc_on_NB(const protocol_ctxt_t* const ctxt_pP);

void rrc_config_buffer_NB(SRB_INFO_NB *srb_info, uint8_t Lchan_type, uint8_t Role);

int L3_xface_init_NB(void);

void openair_rrc_top_init_NB( char *uecap_xer);

//void rrc_top_cleanup(void); -->seems not to be used

//rrc_t310_expiration-->seems not to be used

/** \brief Function to update timers every subframe.  For UE it updates T300,T304 and T310.
@param ctxt_pP  running context
@param enb_index
@param CC_id
*/
RRC_status_t rrc_rx_tx_NB(protocol_ctxt_t* const ctxt_pP, const uint8_t  enb_index, const int CC_id);

//long binary_search_int(int elements[], long numElem, int value);--> seems not to be used
//long binary_search_float(float elements[], long numElem, float value);--> used only at UE side

/*------------------------L2_interface_nb_iot.c (temporary location for most of the interfaces)--------*/

//defined in MAC/config.c
//FIXME: this function has no implication in terms of logical channel configuration for MAC
int NB_rrc_mac_config_req_eNB(
			   module_id_t       				Mod_idP,
			   int                              CC_idP,
			   int								rntiP, //FIXME: Raymond bug?
			   int                              physCellId,
			   int                              p_eNB,
			   int                              Ncp,
			   long                             eutra_band,//FIXME: frequencyBandIndicator in sib1 (is a long not an int!!)
			   struct NS_PmaxList_NB_r13        *frequencyBandInfo, //optional SIB1
			   struct MultiBandInfoList_NB_r13  *multiBandInfoList, //optional SIB1
			   struct DL_Bitmap_NB_r13          *dl_bitmap, //optional SIB1
			   long*                            eutraControlRegionSize, //optional sib1
			   long*							nrs_CRS_PowerOffset, //optional
			   uint8_t                          *SIwindowsize, //maybe no more needed because TDD only
			   uint16_t                         *SIperiod, //maybe no more needed because TDD only
			   uint32_t                         dl_CarrierFreq,
			   uint32_t                         ul_CarrierFreq,
			   BCCH_BCH_Message_NB_t            *mib_NB,
			   RadioResourceConfigCommonSIB_NB_r13_t   *radioResourceConfigCommon,
			   struct PhysicalConfigDedicated_NB_r13  *physicalConfigDedicated,
			   MAC_MainConfig_NB_r13_t                *mac_MainConfig,
			   long                             logicalChannelIdentity,//FIXME: decide how to use it
			   LogicalChannelConfig_NB_r13_t          *logicalChannelConfig //FIXME: decide how to use it
			   );

//----------------------------------------
//defined in L2_interface
int8_t NB_mac_rrc_data_req_eNB(
  const module_id_t Mod_idP,
  const int         CC_id,
  const frame_t     frameP,
  const rb_id_t     Srb_id,
  uint8_t*    const buffer_pP,
  uint32_t				start_sfn_sib1_NB, //starting frame for sib1 transmission
  uint32_t				nb_sib1_NB_rep, //number of sib1 repetitions
  const frame_t		h_frameP, //HSFN
  long				si_periodicity, //SI-periodicity
  long				si_windowLength //Si-windwolength
);
//---------------------------------------

//defined in L2_interface
//called by rx_sdu only in case of CCCH message (e.g RRCConnectionRequest-NB)
int8_t NB_mac_rrc_data_ind_eNB(
  const module_id_t     module_idP,
  const int             CC_id,
  const frame_t         frameP,
  const sub_frame_t     sub_frameP,
  const rnti_t          rntiP,
  const rb_id_t         srb_idP,//could be skipped since always go through the CCCH channel
  const uint8_t*        sduP,
  const sdu_size_t      sdu_lenP
);
//-------------------------------------------

//defined in L2_interface
void NB_mac_eNB_rrc_ul_failure(
		const module_id_t mod_idP,
	    const int CC_idP,
	    const frame_t frameP,
	    const sub_frame_t subframeP,
	    const rnti_t rntiP);
//------------------------------------------

//defined in eNB_scheduler_primitives.c
int NB_rrc_mac_remove_ue(
		module_id_t mod_idP,
		rnti_t rntiP);
//------------------------------------------
//defined in L2_interface
void NB_mac_eNB_rrc_ul_in_sync(
				const module_id_t mod_idP,
			    const int CC_idP,
			    const frame_t frameP,
			    const sub_frame_t subframeP,
			    const rnti_t rntiP);
//------------------------------------------
//defined in L2_interface
int NB_mac_eNB_get_rrc_status(
  const module_id_t Mod_idP,
  const rnti_t      rntiP
);
//---------------------------

//FIXME for the moment we not configure PDCP for SRB1bis (but used as it is SRB1)
//defined in pdcp.c
boolean_t NB_rrc_pdcp_config_asn1_req (
  const protocol_ctxt_t* const  ctxt_pP,
  SRB_ToAddModList_NB_r13_t  *const srb2add_list_pP,
  DRB_ToAddModList_NB_r13_t  *const drb2add_list_pP,
  DRB_ToReleaseList_NB_r13_t *const drb2release_list_pP,
  const uint8_t                   security_modeP,
  uint8_t                  *const kRRCenc_pP,
  uint8_t                  *const kRRCint_pP,
  uint8_t                  *const kUPenc_pP,
  rb_id_t                 *const defaultDRB,
  long						LCID
);
//--------------------------------------------------

//defined in pdcp.c --> should be called only by a SRB1 (is internal to PDCP so is not an interface)
//-----------------------------------------------------------------------------
boolean_t
NB_pdcp_config_req_asn1 (
  const protocol_ctxt_t* const  ctxt_pP,
  pdcp_t         * const        pdcp_pP,
  const srb_flag_t              srb_flagP,
  const rlc_mode_t              rlc_modeP, //rlc_type
  const config_action_t         actionP,
  const uint16_t                lc_idP, // 1 = SRB1 // 3 = SRB1bis // >= 4 for DRBs
  const rb_id_t                 rb_idP,
  const uint8_t                 rb_snP, //5 if srb_sn // 7 is drb_sn // 0 if drb_sn to be removed
  const uint8_t                 rb_reportP, //not for SRBand not for NB-IOT
  const uint16_t                header_compression_profileP, //not for SRB only DRB
  const uint8_t                 security_modeP,
  uint8_t         *const        kRRCenc_pP,
  uint8_t         *const        kRRCint_pP,
  uint8_t         *const        kUPenc_pP);
//-----------------------------------------------------------------------------

//defined in L2_interface/pdcp.c
//FIXME SRB1bis should bypass the pdcp
//Distinction between different SRBs will be done by means of rd_id
uint8_t NB_rrc_data_req(
  const protocol_ctxt_t*   const ctxt_pP,
  const rb_id_t                  rb_idP,
  const mui_t                    muiP,
  const confirm_t                confirmP,
  const sdu_size_t               sdu_sizeP,
  uint8_t*                 const buffer_pP,
  const pdcp_transmission_mode_t modeP //when go through SRB1bis should be set as Transparent mode
);
//-------------------------------------------------------------
//we distinguish the SRBs based on the logical channel id and the transmission mode
boolean_t NB_pdcp_data_req(
  protocol_ctxt_t*  ctxt_pP,
  const srb_flag_t     srb_flagP, //SRB_FLAG_YES if called by RRC
  const rb_id_t        rb_idP,
  const mui_t          muiP,
  const confirm_t      confirmP,
  const sdu_size_t     sdu_buffer_sizeP, //the size of message that i should transmit
  unsigned char *const sdu_buffer_pP,
  const pdcp_transmission_mode_t modeP
);

//----------------------------------------------------------------

//defined in L2_interface
void NB_rrc_data_ind(
  const protocol_ctxt_t* const ctxt_pP,
  const rb_id_t                Srb_id,
  const sdu_size_t             sdu_sizeP,
  const uint8_t*   const       buffer_pP,
  const srb1bis_flag_t srb1bis_flag
);
//------------------------------------------------------------------------------

//defined in rlc_rrc.c
rlc_op_status_t NB_rrc_rlc_config_asn1_req (
	const protocol_ctxt_t   * const ctxt_pP,
    const SRB_ToAddModList_NB_r13_t   * const srb2add_listP,
    const DRB_ToAddModList_NB_r13_t   * const drb2add_listP,
    const DRB_ToReleaseList_NB_r13_t  * const drb2release_listP,
	srb1bis_flag_t							srb1bis_flag
    );
//-------------------------------------------------------------------------

// defined in rlc_am.c
void NB_config_req_rlc_am_asn1 (
  const protocol_ctxt_t* const         ctxt_pP,
  const srb_flag_t                     srb_flagP,
  const struct RLC_Config_NB_r13__am  * const config_am_pP, //extracted from the srb_toAddMod
  const rb_id_t                        rb_idP,
  const logical_chan_id_t              chan_idP);
//------------------------------------------------------------

//defined in rlc_am_init.c
//------------------------------------------------------------
void
NB_rlc_am_configure(
  const protocol_ctxt_t* const  ctxt_pP,
  rlc_am_entity_t *const        rlc_pP,
  const uint16_t                max_retx_thresholdP,
  const uint16_t                t_poll_retransmitP,
  const uint16_t* const			enableStatusReportSN_Gap
  );
//--------------------------------------------------------------

//defined in rlc_rrc.c
//-------------------------------------------------------------
rlc_union_t* NB_rrc_rlc_add_rlc   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t        srb_flagP,
  const rb_id_t           rb_idP,
  const logical_chan_id_t chan_idP,
  const rlc_mode_t        rlc_modeP);
//--------------------------------------------------------------

//defined in rlc_rrc.c
//--------------------------------------------------------------
rlc_op_status_t NB_rrc_rlc_remove_rlc   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t  srb_flagP,
  const rb_id_t     rb_idP);
//----------------------------------------------

//defined in rlc_rrc.c //used only for process_RRCConnectionReconfigurationComplete --> CONFIG_ACTION_REMOVE
//used also for rrc_t310_expiration --> I don't know if it is used (probably not)
rlc_op_status_t NB_rrc_rlc_config_req   (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t      srb_flagP,
  const config_action_t actionP,
  const rb_id_t         rb_idP,
  const rlc_info_t      rlc_infoP);
//-----------------------------------------------------


//defined in rlc_am.c
//------------------------------------------------------
void
NB_config_req_rlc_am (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t             srb_flagP,
  rlc_am_info_NB_t  * const       config_am_pP, //XXX: MP: rlc_am_init.c --> this structure has been modified for NB-IoT
  const rb_id_t                rb_idP,
  const logical_chan_id_t      chan_idP
);
//--------------------------------------------------------

//defined in rlc_tm_init.c (nothing to be changed)
//-----------------------------------------------------------------------------
void NB_config_req_rlc_tm (
  const protocol_ctxt_t* const  ctxt_pP,
  const srb_flag_t  srb_flagP,
  const rlc_tm_info_t * const config_tmP,
  const rb_id_t rb_idP,
  const logical_chan_id_t chan_idP
);
//------------------------------------------------------
//defined in rlc_rrc.c
rlc_op_status_t NB_rrc_rlc_remove_ue (
  const protocol_ctxt_t* const ctxt_pP);
//----------------------------------------------------

//defined in rlc.c
//--------------------------------------------
void NB_rlc_data_ind     (
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t  srb_flagP,
  const srb1bis_flag_t srb1bis_flag,
  const rb_id_t     rb_idP,
  const sdu_size_t  sdu_sizeP,
  mem_block_t      *sdu_pP);
//---------------------------------------------

//defined in rlc.c
//-----------------------------------------------------------------------------
rlc_op_status_t NB_rlc_data_req     (const protocol_ctxt_t* const ctxt_pP,
                                  const srb_flag_t   srb_flagP,
                                  const rb_id_t      rb_idP,
                                  const mui_t        muiP,
                                  confirm_t    confirmP,
                                  sdu_size_t   sdu_sizeP,
                                  mem_block_t *sdu_pP);
//-----------------------------------------------------------------------------

//defined in pdcp.c
//FIXME: should go transparent through the PDCP
//--------------------------------------------
boolean_t
NB_pdcp_data_ind(
  const protocol_ctxt_t* const ctxt_pP,
  const srb_flag_t   srb_flagP,
  const srb1bis_flag_t srb1bis_flag,
  const rb_id_t      rb_idP,
  const sdu_size_t   sdu_buffer_sizeP,
  mem_block_t* const sdu_buffer_pP
);
//---------------------------------------------

//defined in rlc_mac.c
void NB_mac_rlc_data_ind     (
  const module_id_t         module_idP,
  const rnti_t              rntiP,
  const module_id_t         eNB_index,
  const frame_t             frameP,
  const eNB_flag_t          enb_flagP,
//const MBMS_flag_t         MBMS_flagP,
  const logical_chan_id_t   channel_idP,
  char                     *buffer_pP,
  const tb_size_t           tb_sizeP,
  num_tb_t                  num_tbP,
  crc_t                    *crcs_pP);
//-------------------------------------------

//defined in rlc_am.c
void
NB_rlc_am_mac_data_indication (
  const protocol_ctxt_t* const ctxt_pP,
  void * const                 rlc_pP,
  struct mac_data_ind          data_indP
);
//--------------------------------------------

//defined in rlc_mac.c
//called by the schedule_ue_spec for getting SDU to be transmitted from SRB1/SRB1bis and DRBs
tbs_size_t NB_mac_rlc_data_req_eNB(
  const module_id_t       module_idP,
  const rnti_t            rntiP,
  const eNB_index_t       eNB_index,
  const frame_t           frameP,
  const MBMS_flag_t       MBMS_flagP,
  const logical_chan_id_t channel_idP,
  char             *buffer_pP);
//--------------------------------------------



/*UE procedures*/

/*eNB procedures*/

//---Initialization--------------
void
openair_eNB_rrc_on_NB(
  const protocol_ctxt_t* const ctxt_pP
);

void
rrc_config_buffer_NB(
  SRB_INFO_NB* Srb_info,
  uint8_t Lchan_type,
  uint8_t Role
);

char
openair_rrc_eNB_configuration_NB(
  const module_id_t enb_mod_idP,
  RrcConfigurationReq* configuration
);

//-----------------------------
/**\brief RRC eNB task. (starting of the RRC state machine)
   \param void *args_p Pointer on arguments to start the task. */
void *rrc_enb_task_NB(void *args_p);

/**\brief Entry routine to decode a UL-CCCH-Message-NB.  Invokes PER decoder and parses message.
   \param ctxt_pP Running context
   \param Srb_info Pointer to SRB0 information structure (buffer, etc.)*/
int
rrc_eNB_decode_ccch_NB(
  protocol_ctxt_t* const ctxt_pP,
  const SRB_INFO_NB*        const Srb_info,
  const int              CC_id
);

/**\brief Entry routine to decode a UL-DCCH-Message-NB.  Invokes PER decoder and parses message.
   \param ctxt_pP Context
   \param Rx_sdu Pointer Received Message
   \param sdu_size Size of incoming SDU*/
int
rrc_eNB_decode_dcch_NB(
  const protocol_ctxt_t* const ctxt_pP,
  const rb_id_t                Srb_id,
  const uint8_t*    const      Rx_sdu,
  const sdu_size_t             sdu_sizeP
);

/**\brief Generate RRCConnectionReestablishmentReject-NB
   \param ctxt_pP       Running context
   \param ue_context_pP UE context
   \param CC_id         Component Carrier ID*/
void
rrc_eNB_generate_RRCConnectionReestablishmentReject_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
);

void
rrc_eNB_generate_RRCConnectionReject_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
);

void
rrc_eNB_generate_RRCConnectionSetup_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP,
  const int                    CC_id
);

void
rrc_eNB_process_RRCConnectionReconfigurationComplete_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*        ue_context_pP,
  const uint8_t xid //transaction identifier
);


void //was under ITTI
rrc_eNB_reconfigure_DRBs_NB(const protocol_ctxt_t* const ctxt_pP,
			       rrc_eNB_ue_context_NB_t*  ue_context_pP);

void //was under ITTI
rrc_eNB_generate_dedicatedRRCConnectionReconfiguration_NB(
		const protocol_ctxt_t* const ctxt_pP,
	    rrc_eNB_ue_context_NB_t*          const ue_context_pP
        //no ho state
	     );

void
rrc_eNB_process_RRCConnectionSetupComplete_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*         ue_context_pP,
  RRCConnectionSetupComplete_NB_r13_IEs_t * rrcConnectionSetupComplete_NB
);

void
rrc_eNB_generate_SecurityModeCommand_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP
);

void
rrc_eNB_generate_UECapabilityEnquiry_NB(
  const protocol_ctxt_t* const ctxt_pP,
  rrc_eNB_ue_context_NB_t*          const ue_context_pP
);

void
rrc_eNB_generate_defaultRRCConnectionReconfiguration_NB(const protocol_ctxt_t* const ctxt_pP,
						     rrc_eNB_ue_context_NB_t*          const ue_context_pP
						//no HO flag
						     );


/// Utilities------------------------------------------------

void
rrc_eNB_free_mem_UE_context_NB(
  const protocol_ctxt_t*               const ctxt_pP,
  struct rrc_eNB_ue_context_NB_s*         const ue_context_pP
);



/**\brief Function to get the next transaction identifier.
   \param module_idP Instance ID for CH/eNB
   \return a transaction identifier*/
uint8_t rrc_eNB_get_next_transaction_identifier_NB(module_id_t module_idP);

