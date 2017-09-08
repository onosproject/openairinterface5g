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
#include "PHY/defs_NB_IoT.h"
//#include <math.h>

//NPSS

int generate_npss_NB_IoT(int32_t                **txdataF,
                         short                  amp,
                         NB_IoT_DL_FRAME_PARMS  *frame_parms,
                         unsigned short         symbol_offset,          // symbol_offset should equal to 3 for NB-IoT 
                         unsigned short         slot_offset,
                         unsigned short         RB_IoT_ID);             // new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE

//NSSS

int generate_sss_NB_IoT(int32_t                **txdataF,
                        int16_t                amp,
                        NB_IoT_DL_FRAME_PARMS  *frame_parms, 
                        uint16_t               symbol_offset,             // symbol_offset = 3 for NB-IoT 
                        uint16_t               slot_offset, 
                        unsigned short         frame_number,        // new attribute (Get value from higher layer), it does not exist for LTE
                        unsigned short         RB_IoT_ID);          // new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE

//NRS

void generate_pilots_NB_IoT(PHY_VARS_eNB_NB_IoT  *phy_vars_eNB,
                            int32_t              **txdataF,
                            int16_t              amp,
                            uint16_t             Ntti,                // Ntti = 10
                            unsigned short       RB_IoT_ID,       // RB reserved for NB-IoT
                            unsigned short       With_NSSS);      // With_NSSS = 1; if the frame include a sub-Frame with NSSS signal


//NPBCH

int allocate_npbch_REs_in_RB(NB_IoT_DL_FRAME_PARMS  *frame_parms,
                             int32_t                **txdataF,
                             uint32_t               *jj,
                             uint32_t               symbol_offset,
                             uint8_t                *x0,
                             uint8_t                pilots,
                             int16_t                amp,
                             unsigned short         id_offset,
                             uint32_t               *re_allocated);


int generate_npbch(NB_IoT_eNB_NPBCH_t     *eNB_npbch,
                   int32_t                **txdataF,
                   int                    amp,
                   NB_IoT_DL_FRAME_PARMS  *frame_parms,
                   uint8_t                *npbch_pdu,
                   uint8_t                frame_mod64,
                   unsigned short         NB_IoT_RB_ID);


void npbch_scrambling(NB_IoT_DL_FRAME_PARMS  *frame_parms,
                      uint8_t                *npbch_e,
                      uint32_t               length);

// Functions below implement 36-211 and 36-212

/*Function to pack the DCI*/ 
// newly added function for NB-IoT , does not exist for LTE
void add_dci_NB_IoT(DCI_PDU_NB_IoT    *DCI_pdu,
                    void              *pdu,
                    rnti_t            rnti,
                    unsigned char     dci_size_bytes,
                    unsigned char     aggregation, 
                    unsigned char     dci_size_bits,
                    unsigned char     dci_fmt,
                    uint8_t           npdcch_start_symbol);


/*Use the UL DCI Information to configure PHY and also Pack the DCI*/
int generate_eNB_ulsch_params_from_dci_NB_IoT(PHY_VARS_eNB_NB_IoT     *eNB,
                                              eNB_rxtx_proc_NB_IoT_t  *proc,
                                              DCI_CONTENT             *DCI_Content,
                                              uint16_t                rnti,
                                              DCI_format_NB_IoT_t     dci_format,
                                              uint8_t                 UE_id,
                                              uint8_t                 aggregation,
									                            uint8_t                 npdcch_start_symbol);


/*Use the DL DCI Information to configure PHY and also Pack the DCI*/
int generate_eNB_dlsch_params_from_dci_NB_IoT(PHY_VARS_eNB_NB_IoT    *eNB,
                                              int                    frame,
                                              uint8_t                subframe,
                                              DCI_CONTENT            *DCI_Content,
                                              uint16_t               rnti,
                                              DCI_format_NB_IoT_t    dci_format,
                                              NB_IoT_eNB_NDLSCH_t    *ndlsch,
                                              NB_IoT_DL_FRAME_PARMS  *frame_parms,
                                              uint8_t                aggregation,
									                            uint8_t                npdcch_start_symbol);


/*Function for DCI encoding, scrambling, modulation*/
uint8_t generate_dci_top_NB_IoT(NB_IoT_eNB_NPDCCH_t     *npdcch,
						                    uint8_t                 Num_dci,
                                DCI_ALLOC_NB_IoT_t      *dci_alloc,
                                int16_t                 amp,
                                NB_IoT_DL_FRAME_PARMS   *fp,
                                int32_t                 **txdataF,
                                uint32_t                subframe,
						                    uint8_t                 npdcch_start_symbol);

/*!
  \brief Decoding of PUSCH/ACK/RI/ACK from 36-212.
  @param phy_vars_eNB Pointer to eNB top-level descriptor
  @param proc Pointer to RXTX proc variables
  @param UE_id ID of UE transmitting this PUSCH
  @param subframe Index of subframe for PUSCH
  @param control_only_flag Receive PUSCH with control information only
  @param Nbundled Nbundled parameter for ACK/NAK scrambling from 36-212/36-213
  @param llr8_flag If 1, indicate that the 8-bit turbo decoder should be used
  @returns 0 on success
*/
unsigned int  ulsch_decoding_NB_IoT(PHY_VARS_eNB_NB_IoT     *phy_vars_eNB,
                                    eNB_rxtx_proc_NB_IoT_t  *proc,
                                    uint8_t                 UE_id,
                                    uint8_t                 control_only_flag,
                                    uint8_t                 Nbundled,
                                    uint8_t                 llr8_flag);

//NB-IoT version
NB_IoT_eNB_NDLSCH_t *new_eNB_dlsch_NB_IoT(//unsigned char Kmimo,
                                          //unsigned char Mdlharq,
                                          uint32_t Nsoft,
                                          //unsigned char N_RB_DL,
                                          uint8_t abstraction_flag,
                                          NB_IoT_DL_FRAME_PARMS* frame_parms);


NB_IoT_eNB_NULSCH_t *new_eNB_ulsch_NB(uint8_t abstraction_flag);

//5555
/*void rx_ulsch_NB_IoT(PHY_VARS_eNB *phy_vars_eNB,
                     eNB_rxtx_proc_t *proc,
                     uint8_t eNB_id,  // this is the effective sector id
                     uint8_t UE_id,
                     LTE_eNB_ULSCH_t **ulsch,
                     uint8_t cooperation_flag);
*/

uint8_t subframe2harq_pid_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,uint32_t frame,uint8_t subframe);


/** \brief Compute Q (modulation order) based on I_MCS for PUSCH.  Implements table 8.6.1-1 from 36.213.
    @param I_MCS */
uint8_t get_Qm_ul_NB_IoT(uint8_t I_MCS);

/** \fn dlsch_encoding(PHY_VARS_eNB *eNB,
    uint8_t *input_buffer,
    LTE_DL_FRAME_PARMS *frame_parms,
    uint8_t num_pdcch_symbols,
    LTE_eNB_DLSCH_t *dlsch,
    int frame,
    uint8_t subframe)
    \brief This function performs a subset of the bit-coding functions for LTE as described in 36-212, Release 8.Support is limited to turbo-coded channels (DLSCH/ULSCH). The implemented functions are:
    - CRC computation and addition
    - Code block segmentation and sub-block CRC addition
    - Channel coding (Turbo coding)
    - Rate matching (sub-block interleaving, bit collection, selection and transmission
    - Code block concatenation
    @param eNB Pointer to eNB PHY context
    @param input_buffer Pointer to input buffer for sub-frame
    @param frame_parms Pointer to frame descriptor structure
    @param num_pdcch_symbols Number of PDCCH symbols in this subframe
    @param dlsch Pointer to dlsch to be encoded
    @param frame Frame number
    @param subframe Subframe number
    @param rm_stats Time statistics for rate-matching
    @param te_stats Time statistics for turbo-encoding
    @param i_stats Time statistics for interleaving
    @returns status
*/

int32_t dlsch_encoding_NB_IoT(unsigned char              *a,
                              NB_IoT_eNB_DLSCH_t         *dlsch,
                              uint8_t                    Nsf,        // number of subframes required for npdsch pdu transmission calculated from Isf (3GPP spec table)
                              unsigned int               G,          // G (number of available RE) is implicitly multiplied by 2 (since only QPSK modulation)
                              time_stats_t_NB_IoT        *rm_stats,
                              time_stats_t_NB_IoT        *te_stats,
                              time_stats_t_NB_IoT        *i_stats);


void rx_ulsch_NB_IoT(PHY_VARS_eNB_NB_IoT      *phy_vars_eNB,
                     eNB_rxtx_proc_NB_IoT_t   *proc,
                     uint8_t                  eNB_id,               // this is the effective sector id
                     uint8_t                  UE_id,
                     NB_IoT_eNB_ULSCH_t       **ulsch,
                     uint8_t                  cooperation_flag);

void ulsch_extract_rbs_single_NB_IoT(int32_t                **rxdataF,
                                     int32_t                **rxdataF_ext,
                                     uint32_t               first_rb,
                                     uint32_t               nb_rb,
                                     uint8_t                l,
                                     uint8_t                Ns,
                                     NB_IoT_DL_FRAME_PARMS  *frame_parms);


#endif
