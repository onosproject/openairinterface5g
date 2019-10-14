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
//#include "platform_types_NB_IoT.h"

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





