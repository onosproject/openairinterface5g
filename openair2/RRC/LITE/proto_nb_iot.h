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
/*NOTE: no static function should be declared in this header file*/

/*rrc_common.c*/
//int rrc_init_global_param(void);
//int L3_xface_init(void);
//void openair_rrc_top_init(int eMBMS_active, char *uecap_xer, uint8_t cba_group_active,uint8_t HO_enabled);
//void rrc_config_buffer(SRB_INFO *srb_info, uint8_t Lchan_type, uint8_t Role);
//void openair_rrc_on(const protocol_ctxt_t* const ctxt_pP);
//void rrc_top_cleanup(void);
//RRC_status_t rrc_rx_tx(protocol_ctxt_t* const ctxt_pP, const uint8_t  enb_index, const int CC_id);
//long binary_search_int(int elements[], long numElem, int value);
//long binary_search_float(float elements[], long numElem, float value);

/*L2_interface.c*/


/*UE procedures*/

/*eNB procedures*/

//char openair_rrc_eNB_init(const module_id_t module_idP);


/**\brief RRC eNB task.
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

char
openair_rrc_eNB_configuration_NB(
  const module_id_t enb_mod_idP,
  RrcConfigurationReq* configuration
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

