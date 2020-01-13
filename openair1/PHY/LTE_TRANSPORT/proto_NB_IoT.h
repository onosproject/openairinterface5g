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
 * \last changes: M. Kanj, V. Savaux
 * \date: 2018
 * \company: b<>com
 * \email: matthieu.kanj@b-com.com, vincent.savaux@b-com.com
 * \note
 * \warning
 */
#ifndef __LTE_TRANSPORT_PROTO_NB_IOT__H__
#define __LTE_TRANSPORT_PROTO_NB_IOT__H__
#include "PHY/defs_NB_IoT.h"
#include "PHY/impl_defs_lte.h"
#include "PHY/defs.h"
//#include "PHY/LTE_TRANSPORT/defs_NB_IoT.h"
//#include <math.h>

//NPSS
void free_eNB_dlsch_NB_IoT(NB_IoT_eNB_NDLSCH_t *dlsch);
void free_eNB_dlcch_NB_IoT(NB_IoT_eNB_NPDCCH_t *dlcch);

void init_unscrambling_lut_NB_IoT(void);

int generate_npss_NB_IoT(int32_t                **txdataF,
                         short                  amp,
                         LTE_DL_FRAME_PARMS  *frame_parms,
                         unsigned short         symbol_offset,          // symbol_offset should equal to 3 for NB-IoT 
                         unsigned short         slot_offset,
                         unsigned short         RB_IoT_ID);             // new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE

//NSSS

int generate_sss_NB_IoT(int32_t                **txdataF,
                        int16_t                amp,
                        LTE_DL_FRAME_PARMS  *frame_parms, 
                        uint16_t               symbol_offset,             // symbol_offset = 3 for NB-IoT 
                        uint16_t               slot_offset, 
                        unsigned short         frame_number,        // new attribute (Get value from higher layer), it does not exist for LTE
                        unsigned short         RB_IoT_ID);          // new attribute (values are between 0.. Max_RB_number-1), it does not exist for LTE 

//*****************Vincent part for Cell ID estimation from NSSS ******************// 

int rx_nsss_NB_IoT(PHY_VARS_UE_NB_IoT *ue,int32_t *tot_metric); 

int nsss_extract_NB_IoT(PHY_VARS_UE_NB_IoT *ue,
            NB_IoT_DL_FRAME_PARMS *frame_parms,
            int32_t **nsss_ext,
            int l);

//NRS

void generate_pilots_NB_IoT(PHY_VARS_eNB  *phy_vars_eNB,
                            int32_t              **txdataF,
                            int16_t              amp,
                            uint16_t             Ntti,                // Ntti = 10
                            unsigned short       RB_IoT_ID,       // RB reserved for NB-IoT
                            unsigned short       With_NSSS);      // With_NSSS = 1; if the frame include a sub-Frame with NSSS signal


//NPBCH

int allocate_npbch_REs_in_RB(LTE_DL_FRAME_PARMS  *frame_parms,
                             int32_t                **txdataF,
                             uint32_t               *jj,
                             uint32_t               symbol_offset,
                             uint8_t                *x0,
                             uint8_t                pilots,
                             int16_t                amp,
                             unsigned short         id_offset,
                             uint32_t               *re_allocated);

// NPDSCH
int allocate_REs_in_RB_NB_IoT(LTE_DL_FRAME_PARMS    *frame_parms,
                              int32_t               **txdataF,
                              uint32_t              *jj,
                              uint32_t              symbol_offset,
                              uint8_t               *x0,
                              uint8_t               pilots,
                              int16_t               amp,
                              unsigned short        id_offset,
                              uint8_t               pilot_shift,
                              uint32_t              *re_allocated); 

int generate_NDLSCH_NB_IoT(PHY_VARS_eNB               *eNB,
                           NB_IoT_eNB_NDLSCH_t        *RAR,
                           int32_t                    **txdataF,
                           int16_t                    amp,
                           LTE_DL_FRAME_PARMS         *frame_parms,
                           uint32_t                   frame,
                           uint32_t                   subframe,
                           int                        RB_IoT_ID,
                           uint8_t                    release_v13_5_0);

int generate_NPDCCH_NB_IoT(NB_IoT_eNB_NPDCCH_t    *DCI,
                           int32_t                **txdataF,
                           int16_t                amp,
                           LTE_DL_FRAME_PARMS     *frame_parms,
                           uint32_t               frame,
                           uint32_t               subframe,
                           int                    RB_IoT_ID);

int generate_SIB23(NB_IoT_eNB_NDLSCH_t    *SIB23,
                   int32_t                **txdataF,
                   int16_t                amp,
                   LTE_DL_FRAME_PARMS     *frame_parms,
                   uint32_t               frame,
                   uint32_t               subframe,
                   int                    RB_IoT_ID,
                   uint8_t                release_v13_5_0);

int generate_SIB1(NB_IoT_eNB_NDLSCH_t     *sib1_struct,
                   int32_t                **txdataF,
                   int16_t                amp,
                   LTE_DL_FRAME_PARMS     *frame_parms,
                   uint32_t               frame,
                   uint32_t               subframe,
                   int                    RB_IoT_ID,
                   uint8_t                operation_mode,
                   uint8_t                release_v13_5_0);

int generate_npbch(NB_IoT_eNB_NPBCH_t     *eNB_npbch,
                   int32_t                **txdataF,
                   int                    amp,
                   LTE_DL_FRAME_PARMS     *frame_parms,
                   uint8_t                *npbch_pdu,
                   uint8_t                frame_mod64,
                   unsigned short         NB_IoT_RB_ID,
                   uint8_t                release_v13_5_0);


void npbch_scrambling(LTE_DL_FRAME_PARMS  *frame_parms,
                      uint8_t                *npbch_e,
                      uint32_t               length);


void dlsch_scrambling_Gen_NB_IoT(LTE_DL_FRAME_PARMS         *frame_parms,
                                  NB_IoT_eNB_NDLSCH_t       *dlsch,
                                  int                       tot_bits,            // total number of bits to transmit
                                  uint16_t                  Nf,                  // Nf is the frame number (0..9)
                                  uint8_t                   Ns,
                                  uint32_t                  rnti,
                                  uint8_t                   release_v13_5_0,
                                  uint8_t                   SIB); 

NB_IoT_eNB_NDLSCH_t *new_eNB_dlsch_NB_IoT(uint8_t length, LTE_DL_FRAME_PARMS* frame_parms);

NB_IoT_eNB_NPDCCH_t *new_eNB_dlcch_NB_IoT(LTE_DL_FRAME_PARMS* frame_parms);

/*void dlsch_scrambling_Gen_NB_IoT(LTE_DL_FRAME_PARMS      *frame_parms,
                                  NB_IoT_eNB_NDLSCH_t    *dlsch,
                                  int                    tot_bits,         // total number of bits to transmit
                                  uint16_t               Nf,               // Nf is the frame number (0..9)
                                  uint8_t                Ns,
                                  uint32_t               rnti,     /// for SIB1 the SI_RNTI should be get from the DL request
                                  uint8_t                type);*/
/*
int scrambling_npbch_REs_rel_14(LTE_DL_FRAME_PARMS      *frame_parms,
                                int32_t                 **txdataF,
                                uint32_t                *jj,
                                int                     l,
                                uint32_t                symbol_offset,
                                uint8_t                 pilots,
                                unsigned short          id_offset,
                                uint8_t                 *reset,
                                uint32_t                *x1,
                                uint32_t                *x2,
                                uint32_t                *s);
*/
// Functions below implement 36-211 and 36-212

/*Use the UL DCI Information to configure PHY and also Pack the DCI*/
int generate_eNB_ulsch_params_from_dci_NB_IoT(PHY_VARS_eNB            *eNB,
                                              int                     frame,
                                              uint8_t                 subframe,
                                              DCI_CONTENT             *DCI_Content,
                                              uint16_t                rnti,
                                              NB_IoT_eNB_NPDCCH_t     *ndlcch,
                                              uint8_t                 aggregation,
                                              uint8_t                 npdcch_start_symbol,
                                              uint8_t                 ncce_index);


/*Use the DL DCI Information to configure PHY and also Pack the DCI*/
int generate_eNB_dlsch_params_from_dci_NB_IoT(PHY_VARS_eNB    *eNB,
                                              int                    frame,
                                              uint8_t                subframe,
                                              DCI_CONTENT            *DCI_Content,
                                              uint16_t               rnti,
                                              DCI_format_NB_IoT_t    dci_format,
                                              NB_IoT_eNB_NPDCCH_t      *ndlcch,
                                              LTE_DL_FRAME_PARMS  *frame_parms,
                                              uint8_t                aggregation,
									                            uint8_t                npdcch_start_symbol,
                                              uint8_t                ncce_index);




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
/*unsigned int  ulsch_decoding_NB_IoT(PHY_VARS_eNB     *phy_vars_eNB,
                                    eNB_rxtx_proc_t  *proc,
                                    uint8_t                 UE_id,
                                    uint8_t                 control_only_flag,
                                    uint8_t                 Nbundled,
                                    uint8_t                 llr8_flag);
*/

//  NB_IoT_eNB_NULSCH_t *new_eNB_ulsch_NB_IoT(uint8_t abstraction_flag);


uint8_t subframe2harq_pid_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,uint32_t frame,uint8_t subframe);


/** \brief Compute Q (modulation order) based on I_MCS for PUSCH.  Implements table 8.6.1-1 from 36.213.
    @param I_MCS */

//uint8_t get_Qm_ul_NB_IoT(uint8_t I_MCS);

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

int dci_modulation_NB_IoT(int32_t              **txdataF,
                          int16_t              amp,
                          LTE_DL_FRAME_PARMS   *frame_parms,
                          uint8_t              control_region_size,      
                          NB_IoT_eNB_NPDCCH_t  *dlcch,            
                          unsigned int         npdsch_data_subframe,        
                          uint8_t              agr_level,
                          uint8_t              ncce_index,
                          unsigned int         subframe,
                          unsigned short       NB_IoT_RB_ID);

int dci_allocate_REs_in_RB_NB_IoT(LTE_DL_FRAME_PARMS  *frame_parms,
                                  int32_t             **txdataF,
                                  uint32_t            *jj,
                                  uint32_t            symbol_offset,
                                  uint8_t             *x0,
                                  uint8_t             pilots,
                                  uint8_t             pilot_shift,
                                  int16_t             amp,
                                  unsigned short      id_offset,
                                  uint8_t             ncce_index,
                                  uint8_t             agr_level,
                                  uint32_t            *re_allocated);


void dci_encoding_NB_IoT(uint8_t                  *a,
                         NB_IoT_eNB_NPDCCH_t      *dlcch,                  
                         uint8_t                  A,
                         uint16_t                 G,              
                         uint8_t                  ncce_index,
                         uint8_t                  agr_level);


void npdcch_scrambling_NB_IoT(LTE_DL_FRAME_PARMS     *frame_parms,
                              NB_IoT_eNB_NPDCCH_t     *dlcch,     // Input data
                              int                     G,          // Total number of bits to transmit in one subframe(case of DCI = G)
                              uint8_t                 Ns,       //XXX we pass the subframe  // Slot number (0..19)
                              uint8_t                 ncce_index,
                              uint8_t                 agr_level); 


int dlsch_modulation_NB_IoT(int32_t               **txdataF,
                            int16_t               amp,
                            LTE_DL_FRAME_PARMS    *frame_parms,
                            uint8_t               control_region_size,      // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                            NB_IoT_eNB_NDLSCH_t   *dlsch0,  //NB_IoT_eNB_NDLSCH_t
                            int                   G,              // number of bits per subframe
                            unsigned int          npdsch_data_subframe,     // subframe index of the data table of npdsch channel (G*Nsf)  , values are between 0..Nsf        
                            unsigned int          subframe,
                            unsigned short        NB_IoT_RB_ID);
/*
int dlsch_modulation_rar_NB_IoT(int32_t         **txdataF,
                                int16_t         amp,
                                LTE_DL_FRAME_PARMS      *frame_parms,
                                uint8_t         control_region_size,      // control region size for LTE , values between 0..3, (0 for stand-alone / 1, 2 or 3 for in-band)
                                NB_IoT_DL_eNB_HARQ_t      *dlsch0, //NB_IoT_eNB_NDLSCH_t
                                int           G,              // number of bits per subframe
                                unsigned int        npdsch_data_subframe,     // subframe index of the data table of npdsch channel (G*Nsf)  , values are between 0..Nsf        
                                unsigned int        subframe,
                                unsigned short      NB_IoT_RB_ID,
                                uint8_t             option);
*/
int32_t dlsch_encoding_NB_IoT(unsigned char              *a,
                              NB_IoT_eNB_NDLSCH_t        *dlsch, // NB_IoT_eNB_NDLSCH_t
                              uint8_t                    Nsf,        // number of subframes required for npdsch pdu transmission calculated from Isf (3GPP spec table)
                              unsigned int               G);         // G (number of available RE) is implicitly multiplied by 2 (since only QPSK modulation)
 

void get_pilots_position(uint8_t npusch_format,uint8_t  subcarrier_spacing,uint8_t *pilot_pos1,uint8_t *pilot_pos2,uint8_t *pilots_slot);

void UL_channel_estimation_NB_IoT(PHY_VARS_eNB        *eNB,
                                  LTE_DL_FRAME_PARMS  *fp,
                                  uint16_t            UL_RB_ID_NB_IoT,
                                  uint16_t            Nsc_RU,
                                  uint8_t             pilot_pos1,
                                  uint8_t             pilot_pos2,
                                  uint16_t            ul_sc_start,
                                  uint8_t             Qm,
                                  uint16_t            N_SF_per_word,
                                  uint8_t             rx_subframe);

void get_llr_per_sf_NB_IoT(PHY_VARS_eNB        *eNB,
                           LTE_DL_FRAME_PARMS  *fp,
                           uint8_t             npusch_format,
                           uint8_t             counter_sf,
                           uint16_t            N_SF_per_word,
                           uint8_t             pilot_pos1,
                           uint8_t             pilot_pos2,
                           uint16_t            ul_sc_start,
                           uint16_t            Nsc_RU);

void descrambling_NPUSCH_data_NB_IoT(LTE_DL_FRAME_PARMS  *fp,
                                     int16_t             *ulsch_llr,
                                     int16_t             *y,
                                     uint8_t             Qm,
                                     unsigned int        Cmux,
                                     uint32_t            rnti_tmp,
                                     uint8_t             rx_subframe,
                                     uint32_t            rx_frame);

void descrambling_NPUSCH_ack_NB_IoT(LTE_DL_FRAME_PARMS  *fp,
                                    int32_t             *y_msg5,
                                    int32_t             *llr_msg5,
                                    uint32_t            rnti_tmp,
                                    uint16_t            *counter_ack,
                                    uint8_t             rx_subframe,
                                    uint32_t            rx_frame);

void  turbo_decoding_NB_IoT(PHY_VARS_eNB           *eNB,
                                NB_IoT_eNB_NULSCH_t    *ulsch_NB_IoT,
                                eNB_rxtx_proc_t        *proc,
                                uint8_t                 npusch_format,
                                unsigned int            G,
                                uint8_t                 rvdx,
                                uint8_t                 Qm,
                                uint32_t                rx_frame,
                                uint8_t                 rx_subframe);

void decode_NPUSCH_msg_NB_IoT(PHY_VARS_eNB        *eNB,
                              LTE_DL_FRAME_PARMS  *fp,
                              eNB_rxtx_proc_t     *proc,
                              uint8_t             npusch_format,
                              uint16_t            N_SF_per_word,
                              uint16_t            Nsc_RU,
                              uint16_t            N_UL_slots,
                              uint8_t             Qm,
                              uint8_t             pilots_slot,
                              uint32_t            rnti_tmp,
                              uint8_t             rx_subframe,
                              uint32_t            rx_frame);

void deinterleaving_NPUSCH_data_NB_IoT(NB_IoT_UL_eNB_HARQ_t *ulsch_harq, int16_t *y, unsigned int G);


uint8_t rx_ulsch_Gen_NB_IoT(PHY_VARS_eNB             *eNB,
                             eNB_rxtx_proc_t         *proc,
                             uint8_t                 eNB_id,  // this is the effective sector id
                             uint8_t                 UE_id,
                             uint16_t                UL_RB_ID_NB_IoT,  // 22 , to be included in // to be replaced by NB_IoT_start ??
                             uint8_t                 subframe,
                             uint32_t                frame);

void ulsch_extract_rbs_single_NB_IoT(int32_t **rxdataF,
                                     int32_t **rxdataF_ext, 
                                     uint16_t UL_RB_ID_NB_IoT, // index of UL NB_IoT resource block !!! may be defined twice : in frame_parms and in NB_IoT_UL_eNB_HARQ_t
                                     uint16_t N_sc_RU, // number of subcarriers in UL 
                                     uint8_t l,
                                     uint8_t Ns,
                                     LTE_DL_FRAME_PARMS *frame_parms);

void ulsch_channel_level_NB_IoT(int32_t **drs_ch_estimates_ext,
                                LTE_DL_FRAME_PARMS *frame_parms,
                                int32_t *avg,
                                uint16_t nb_rb);

void ulsch_channel_compensation_NB_IoT(int32_t **rxdataF_ext,
                                int32_t **ul_ch_estimates_ext,
                                int32_t **ul_ch_mag,
                                int32_t **ul_ch_magb,
                                int32_t **rxdataF_comp,
                                LTE_DL_FRAME_PARMS *frame_parms,
                                uint8_t symbol,
                                uint8_t Qm,
                                uint16_t nb_rb,
                                uint8_t output_shift);

void lte_idft_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,uint32_t *z, uint16_t Msc_PUSCH);

void extract_CQI_NB_IoT(void *o,UCI_format_NB_IoT_t uci_format,NB_IoT_eNB_UE_stats *stats,uint8_t N_RB_DL, uint16_t * crnti, uint8_t * access_mode);

//*****************Vincent part for nprach ******************//
uint32_t process_nprach_NB_IoT(PHY_VARS_eNB *eNB, int frame, uint8_t subframe,uint16_t *rnti, uint16_t *preamble_index, uint16_t *timing_advance); 

uint32_t TA_estimation_NB_IoT(PHY_VARS_eNB *eNB, 
                              int16_t *Rx_sub_sampled_buffer, 
                              uint16_t sub_sampling_rate, 
                              uint16_t FRAME_LENGTH_COMPLEX_SUB_SAMPLES, 
                              uint32_t estimated_TA_coarse, 
                              uint8_t coarse); 

uint8_t NPRACH_detection_NB_IoT(int16_t *input_buffer,uint32_t FRAME_LENGTH_COMPLEX_SAMPLESx); 

int16_t* sub_sampling_NB_IoT(int16_t *input_buffer, uint32_t length_input, uint32_t *length_ouput, uint16_t sub_sampling_rate); 

void filtering_signal(int16_t *input_buffer, int16_t *filtered_buffer, uint32_t FRAME_LENGTH_COMPLEX_SAMPLESx); 
//************************************************************//
///////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
//uint8_t get_UL_I_TBS_from_MCS_NB_IoT(uint8_t I_mcs, uint8_t N_sc_RU, uint8_t Msg3_flag);

uint8_t get_Qm_UL_NB_IoT(unsigned char I_mcs, uint8_t N_sc_RU, uint8_t I_sc, uint8_t Msg3_flag);

//uint16_t get_UL_sc_start_NB_IoT(uint16_t I_sc);

uint16_t get_UL_sc_ACK_NB_IoT(uint8_t subcarrier_spacing,uint16_t harq_ack_resource); 

uint16_t get_UL_sc_index_start_NB_IoT(uint8_t subcarrier_spacing, uint16_t I_sc, uint8_t npush_format);

uint16_t get_UL_N_ru_NB_IoT(uint8_t I_mcs, uint8_t I_ru, uint8_t flag_msg3);

uint16_t get_UL_N_rep_NB_IoT(uint8_t I_rep);

uint8_t get_numb_UL_sc_NB_IoT(uint8_t subcarrier_spacing, uint8_t I_sc, uint8_t npush_format);

uint8_t get_UL_slots_per_RU_NB_IoT(uint8_t subcarrier_spacing, uint8_t subcarrier_indcation, uint8_t UL_format);

///////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

void generate_grouphop_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms); 

void init_ul_hopping_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms); 

void rotate_single_carrier_NB_IoT(PHY_VARS_eNB          *eNB, 
                                  LTE_DL_FRAME_PARMS    *frame_parms,
                                  int32_t               **rxdataF_comp, 
                                  uint8_t               eNB_id,
                                  uint8_t               symbol, //symbol within subframe
                                  uint8_t               counter_msg3,          ///  to be replaced by the number of received part
                                  uint16_t              ul_sc_start,
                                  uint8_t               Qm,
                                  uint16_t              N_SF_per_word, 
                                  uint8_t               option);

void fill_rbs_zeros_NB_IoT(PHY_VARS_eNB *eNB, 
                                  LTE_DL_FRAME_PARMS *frame_parms,
                                  int32_t **rxdataF_comp,
                                  uint16_t ul_sc_start, 
                                  uint8_t UE_id,
                                  uint8_t symbol); 

int32_t ulsch_bpsk_llr_NB_IoT(PHY_VARS_eNB *eNB, 
                              LTE_DL_FRAME_PARMS *frame_parms,
                              int32_t **rxdataF_comp,
                              int16_t *ulsch_llr,
                              uint8_t symbol, 
                              uint16_t ul_sc_start,
                              uint8_t UE_id, 
                              int16_t **llrp); 

int32_t ulsch_qpsk_llr_NB_IoT(PHY_VARS_eNB *eNB, 
                              LTE_DL_FRAME_PARMS *frame_parms,
                              int32_t **rxdataF_comp,
                              int16_t *ulsch_llr, 
                              uint8_t symbol, 
                              uint8_t UE_id,
                              uint16_t ul_sc_start,
                              uint8_t Nsc_RU, 
                              int16_t *llrp);

void rotate_bpsk_NB_IoT(PHY_VARS_eNB *eNB, 
                        LTE_DL_FRAME_PARMS *frame_parms,
                        int32_t **rxdataF_comp,
                        uint16_t ul_sc_start, 
                        uint8_t UE_id,
                        uint8_t symbol); 
//************************************************************// 


int rx_npdsch_NB_IoT(PHY_VARS_UE_NB_IoT *ue,
                      unsigned char eNB_id,
                      unsigned char eNB_id_i, //if this == ue->n_connected_eNB, we assume MU interference
                      uint32_t frame,
                      uint8_t subframe,
                      unsigned char symbol,
                      unsigned char first_symbol_flag,
                      unsigned char i_mod,
                      unsigned char harq_pid); 

unsigned short dlsch_extract_rbs_single_NB_IoT(int **rxdataF,
                                        int **dl_ch_estimates,
                                        int **rxdataF_ext,
                                        int **dl_ch_estimates_ext,
                                        unsigned short pmi,
                                        unsigned char *pmi_ext,
                                        unsigned int *rb_alloc,
                                        unsigned char symbol,
                                        unsigned char subframe,
                                        uint32_t frame,
                                        uint32_t high_speed_flag,
                                        NB_IoT_DL_FRAME_PARMS *frame_parms); 

void dlsch_channel_level_NB_IoT(int **dl_ch_estimates_ext,
                                NB_IoT_DL_FRAME_PARMS *frame_parms,
                                int32_t *avg,
                                uint8_t symbol,
                                unsigned short nb_rb); 

void dlsch_channel_compensation_NB_IoT(int **rxdataF_ext,
                                        int **dl_ch_estimates_ext,
                                        int **dl_ch_mag,
                                        int **dl_ch_magb,
                                        int **rxdataF_comp,
                                        int **rho,
                                        NB_IoT_DL_FRAME_PARMS *frame_parms,
                                        unsigned char symbol,
                                        uint8_t first_symbol_flag,
                                        unsigned char mod_order,
                                        unsigned short nb_rb,
                                        unsigned char output_shift,
                                        PHY_MEASUREMENTS_NB_IoT *measurements); 

int dlsch_qpsk_llr_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
                           int32_t **rxdataF_comp,
                           int16_t *dlsch_llr,
                           uint8_t symbol,
                           uint8_t first_symbol_flag,
                           uint16_t nb_rb,
                           int16_t **llr32p,
                           uint8_t beamforming_mode); 

/// Vincent: temporary functions 

int ul_chest_tmp_NB_IoT(int32_t             **rxdataF_ext,
                        int32_t             **ul_ch_estimates,
                        uint8_t             l, //symbol within slot 
                        uint8_t             Ns,
                        uint8_t             counter_msg3,
                        uint8_t             pilot_pos1,
                        uint8_t             pilot_pos2,
                        uint16_t            ul_sc_start,
                        uint8_t             Qm,
                        uint16_t            N_SF_per_word,
                        LTE_DL_FRAME_PARMS  *frame_parms); 

/// Channel estimation for NPUSCH format 2
int ul_chest_tmp_f2_NB_IoT(int32_t **rxdataF_ext,
                           int32_t **ul_ch_estimates,
                           uint8_t l, //symbol within slot 
                           uint8_t Ns,
                           uint8_t counter_msg3,
                           uint8_t flag,
                           uint8_t subframerx,
                           uint8_t Qm, 
                           uint16_t ul_sc_start,
                           LTE_DL_FRAME_PARMS *frame_parms);

void rotate_channel_sc_tmp_NB_IoT(int16_t *estimated_channel,
                                  uint8_t l, 
                                  uint8_t Qm, 
                                  uint8_t counter_msg3,
                                  uint16_t N_SF_per_word,
                                  uint16_t  ul_sc_start,
                                  uint8_t flag); 

int ul_chequal_tmp_NB_IoT(int32_t **rxdataF_ext,
      int32_t **rxdataF_comp,
      int32_t **ul_ch_estimates,
      uint8_t l, //symbol within slot 
      uint8_t Ns,
      LTE_DL_FRAME_PARMS *frame_parms);
///

////////////////////////////NB-IoT testing ///////////////////////////////
void clean_eNb_ulsch_NB_IoT(NB_IoT_eNB_NULSCH_t *ulsch);

int get_G_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms);

int get_G_SIB1_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms, uint8_t operation_mode_info);

int get_rep_num_SIB1_NB_IoT(uint8_t scheduling_info_sib1);

int get_start_frame_SIB1_NB_IoT(LTE_DL_FRAME_PARMS *frame_parms,uint8_t repetition);

NB_IoT_eNB_NULSCH_t *new_eNB_ulsch_NB_IoT(uint8_t max_turbo_iterations,uint8_t N_RB_UL, uint8_t abstraction_flag);



#endif
