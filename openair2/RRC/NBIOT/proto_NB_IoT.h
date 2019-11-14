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

/*! \file proto_NB_IoT.h
 * \brief RRC functions prototypes for eNB and UE for NB-IoT
 * \author Navid Nikaein, Raymond Knopp and Michele Paffetti
 * \date 2010 - 2014
 * \email navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
 * \version 1.0

 */
/** \addtogroup _rrc
 *  @{
 */

#include "RRC/NBIOT/defs_NB_IoT.h"
#include "pdcp.h"
#include "rlc.h"
#include "extern_NB_IoT.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "platform_types_NB_IoT.h"

uint8_t* generate_msg4_NB_IoT(rrc_eNB_carrier_data_NB_IoT_t *carrier);

uint8_t* mac_rrc_msg3_ind_NB_IoT(uint8_t *payload_ptr, uint16_t rnti, uint32_t length);

uint8_t *get_NB_IoT_MIB(
    rrc_eNB_carrier_data_NB_IoT_t *carrier,
    uint16_t N_RB_DL,//may not needed--> for NB_IoT only 1 PRB is used
    uint32_t subframe,
    uint32_t frame,
    uint32_t hyper_frame);

uint8_t get_NB_IoT_MIB_size(void);

uint8_t *get_NB_IoT_SIB1(uint8_t Mod_id, int CC_id,
        rrc_eNB_carrier_data_NB_IoT_t *carrier,
        uint16_t mcc, //208
        uint16_t mnc, //92
        uint16_t tac, //1
        uint32_t cell_identity, //3584
        uint16_t band,  // 7
        uint16_t mnc_digit_length,
        uint32_t subframe,
        uint32_t frame,
        uint32_t hyper_frame);
uint8_t get_NB_IoT_SIB1_size(void);

uint8_t *get_NB_IoT_SIB23(void);

uint8_t get_NB_IoT_SIB23_size(void);

long *get_NB_IoT_SIB1_eutracontrolregionsize(void);

void init_testing_NB_IoT(uint8_t Mod_id, int CC_id, rrc_eNB_carrier_data_NB_IoT_t *carrier, NbIoTRrcConfigurationReq *configuration, uint32_t frame, uint32_t hyper_frame);

//defined in L2_interface/pdcp.c
//FIXME SRB1bis should bypass the pdcp
//Distinction between different SRBs will be done by means of rd_id
uint8_t rrc_data_req_NB_IoT(
  const protocol_ctxt_t*   const ctxt_pP,
  const rb_id_t                  rb_idP,
  const mui_t                    muiP,
  const confirm_t                confirmP,
  const sdu_size_t               sdu_sizeP,
  uint8_t*                 const buffer_pP,
  const pdcp_transmission_mode_t modeP //when go through SRB1bis should be set as Transparent mode
);
//--------------------------------------------------

//XXX for the moment we not configure PDCP for SRB1bis (but used as it is SRB1)
//defined in pdcp.c
boolean_t rrc_pdcp_config_asn1_req_NB_IoT (
  const protocol_ctxt_t* const  ctxt_pP,
  LTE_SRB_ToAddModList_NB_r13_t  *const srb2add_list_pP,
  LTE_DRB_ToAddModList_NB_r13_t  *const drb2add_list_pP,
  LTE_DRB_ToReleaseList_NB_r13_t *const drb2release_list_pP,
  const uint8_t                   security_modeP,
  uint8_t                  *const kRRCenc_pP,
  uint8_t                  *const kRRCint_pP,
  uint8_t                  *const kUPenc_pP,
  rb_id_t                 *const defaultDRB,
  long                      LCID
);
//--------------------------------------------------

//defined in rlc_rrc.c
rlc_op_status_t rrc_rlc_config_asn1_req_NB_IoT (
    const protocol_ctxt_t   * const ctxt_pP,
    const LTE_SRB_ToAddModList_NB_r13_t   * const srb2add_listP,
    const LTE_DRB_ToAddModList_NB_r13_t   * const drb2add_listP,
    const LTE_DRB_ToReleaseList_NB_r13_t  * const drb2release_listP,
    srb1bis_flag_t                          srb1bis_flag
    );
//-------------------------------------------------------------------------

/*-----------eNB procedures (rrc_eNB_NB_IoT.c)---------------*/

//---Initialization--------------
void openair_eNB_rrc_on_NB_IoT(
  const protocol_ctxt_t* const ctxt_pP
);

/// Utilities------------------------------------------------

void rrc_eNB_free_mem_UE_context_NB_IoT(
  const protocol_ctxt_t*               const ctxt_pP,
  struct rrc_eNB_ue_context_NB_IoT_s*         const ue_context_pP
);



