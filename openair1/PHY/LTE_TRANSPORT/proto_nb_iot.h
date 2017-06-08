/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

/*! \file PHY/LTE_TRANSPORT/proto.h
 * \brief Function prototypes for PHY physical/transport channel processing and generation V8.6 2009-03
 * \author R. Knopp, F. Kaltenberger
 * \date 2011
 * \version 0.1
 * \company Eurecom
 * \email: knopp@eurecom.fr
 * \note
 * \warning
 */
#ifndef __LTE_TRANSPORT_PROTO_NB_IOT__H__
#define __LTE_TRANSPORT_PROTO_NB_IOT__H__
#include "PHY/defs_nb_iot.h"
#include <math.h>

// Functions below implement 36-211 and 36-212

/*Function to pack the DCI*/
void NB_add_dci(DCI_PDU_NB *DCI_pdu,void *pdu,rnti_t rnti,unsigned char dci_size_bytes,unsigned char aggregation,unsigned char dci_size_bits,unsigned char dci_fmt);

/*Use the UL DCI Information to configure PHY and also Packed*/
int NB_generate_eNB_ulsch_params_from_dci(PHY_VARS_eNB_NB *eNB,
                                       eNB_rxtx_proc_NB_t *proc,
                                       DCI_CONTENT *DCI_Content,
                                       uint16_t rnti,
                                       DCI_format_NB_t dci_format,
                                       uint8_t UE_id,
                                       uint8_t aggregation,
                                       uint8_t Num_dci
                                      );
/*Use the DL DCI Information to configure PHY and also Packed*/
int NB_generate_eNB_dlsch_params_from_dci(int frame,
                                       uint8_t subframe,
                                       DCI_CONTENT *DCI_Content,
                                       uint16_t rnti,
                                       DCI_format_NB_t dci_format,
                                       LTE_eNB_DLSCH_t **dlsch,
                                       NB_DL_FRAME_PARMS *frame_parms,
                                       uint8_t aggregation,
                                       uint8_t Num_dci
                                       );
#endif
