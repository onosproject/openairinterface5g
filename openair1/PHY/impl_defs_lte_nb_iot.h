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

/*! \file PHY/impl_defs_lte.h
* \brief LTE Physical channel configuration and variable structure definitions
* \author R. Knopp, F. Kaltenberger
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr,florian.kaltenberger@eurecom.fr
* \note
* \warning
*/

#ifndef __PHY_IMPL_DEFS_NB_IOT__H__
#define __PHY_IMPL_DEFS_NB_IOT__H__

#include "types.h"
//#include "defs.h"

typedef enum {TDD_NB_IoT=1,FDD_NB_IoT=0} NB_IoT_frame_type_t;
typedef enum {EXTENDED_NB_IoT=1,NORMAL_NB_IoT=0} NB_IoT_prefix_type_t;

#define	A_SEQUENCE_OF(type)	A_SET_OF(type)

#define	A_SET_OF(type)					\
	struct {					\
		type **array;				\
		int count;	/* Meaningful size */	\
		int size;	/* Allocated size */	\
		void (*free)(type *);			\
	}


/// NPRACH-ParametersList-NB-r13 from 36.331 RRC spec
typedef struct NPRACH_Parameters_NB_IoT{
  /// the period time for nprach
  uint16_t nprach_Periodicity;
  /// for the start time for the NPRACH resource from 40ms-2560ms
  uint16_t nprach_StartTime;
  /// for the subcarrier of set to the NPRACH preamble from n0 - n34
  uint16_t nprach_SubcarrierOffset;
  ///number of subcarriers in a NPRACH resource allowed values (n12,n24,n36,n48)
  uint16_t nprach_NumSubcarriers;
  /// where is the region that in NPRACH resource to indicate if this UE support MSG3 for multi-tone or not. from 0 - 1
  uint16_t nprach_SubcarrierMSG3_RangeStart;
  /// The max preamble transmission attempt for the CE level from 1 - 128
  uint16_t maxNumPreambleAttemptCE;
  /// Number of NPRACH repetitions per attempt for each NPRACH resource
  uint16_t numRepetitionsPerPreambleAttempt;
  /// The number of the repetition for DCI use in RAR/MSG3/MSG4 from 1 - 2048 (Rmax)
  uint16_t npdcch_NumRepetitions_RA;
  /// Starting subframe for NPDCCH Common searching space for (RAR/MSG3/MSG4)
  uint16_t npdcch_StartSF_CSS_RA;
  /// Fractional period offset of starting subframe for NPDCCH common search space
  uint16_t npdcch_Offset_RA;
} nprach_parameters_NB_IoT_t;

typedef struct{
  A_SEQUENCE_OF(nprach_parameters_NB_IoT_t) list;
}NPRACH_List_NB_IoT_t;

typedef long RSRP_Range_t;

typedef struct {
  A_SEQUENCE_OF(RSRP_Range_t) list;
}rsrp_ThresholdsNPrachInfoList;


/// NPRACH_ConfigSIB-NB from 36.331 RRC spec
typedef struct {
  /// nprach_CP_Length_r13, for the CP length(unit us) only 66.7 and 266.7 is implemented
  uint16_t nprach_CP_Length;
  /// The criterion for UEs to select a NPRACH resource. Up to 2 RSRP threshold values can be signalled.  \vr{[1..2]}
  struct rsrp_ThresholdsNPrachInfoList *rsrp_ThresholdsPrachInfoList;
  /// NPRACH Parameters List
  NPRACH_List_NB_IoT_t nprach_ParametersList;

} NPRACH_CONFIG_COMMON;

/// NPDSCH-ConfigCommon from 36.331 RRC spec
typedef struct {
  ///see TS 36.213 (16.2). \vr{[-60..50]}\n Provides the downlink reference-signal EPRE. The actual value in dBm.
  uint16_t nrs_Power;
} NPDSCH_CONFIG_COMMON;

typedef struct{
  /// The base sequence of DMRS sequence in a cell for 3 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 12. Value 12 is not used.
  uint16_t threeTone_BaseSequence;
  /// Define 3 cyclic shifts for the 3-tone case, see TS 36.211 [21, 10.1.4.1.2].
  uint16_t threeTone_CyclicShift;
  /// The base sequence of DMRS sequence in a cell for 6 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 14. Value 14 is not used.
  uint16_t sixTone_BaseSequence;
  /// Define 4 cyclic shifts for the 6-tone case, see TS 36.211 [21, 10.1.4.1.2].
  uint16_t sixTone_CyclicShift;
  /// The base sequence of DMRS sequence in a cell for 12 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 30. Value 30 is not used.
  uint16_t twelveTone_BaseSequence;

}DMRS_CONFIG_t;

/// UL-ReferenceSignalsNPUSCH from 36.331 RRC spec
typedef struct {
  /// Parameter: Group-hopping-enabled, see TS 36.211 (5.5.1.3). \vr{[0..1]}
  uint8_t groupHoppingEnabled;
  /// , see TS 36.211 (5.5.1.3). \vr{[0..29]}
  uint8_t groupAssignmentNPUSCH;
} UL_REFERENCE_SIGNALS_NPUSCH_t;


/// PUSCH-ConfigCommon from 36.331 RRC spec.
typedef struct {
  /// Number of repetitions for ACK/NACK HARQ response to NPDSCH containing Msg4 per NPRACH resource, see TS 36.213 [23, 16.4.2].
  uint8_t ack_NACK_NumRepetitions_Msg4[3];
  /// SRS SubframeConfiguration. See TS 36.211 [21, table 5.5.3.3-1]. Value sc0 corresponds to value 0, sc1 to value 1 and so on.
  uint8_t srs_SubframeConfig;
  /// Parameter: \f$N^{HO}_{RB}\f$, see TS 36.211 (5.3.4). \vr{[0..98]}
  DMRS_CONFIG_t dmrs_Config;
  /// Ref signals configuration
  UL_REFERENCE_SIGNALS_NPUSCH_t ul_ReferenceSignalsNPUSCH;
} NPUSCH_CONFIG_COMMON;


typedef struct{
  /// See TS 36.213 [23, 16.2.1.1], unit dBm.
  uint8_t p0_NominalNPUSCH;
  /// See TS 36.213 [23, 16.2.1.1] where al0 corresponds to 0, al04 corresponds to value 0.4, al05 to 0.5, al06 to 0.6, al07 to 0.7, al08 to 0.8, al09 to 0.9 and al1 corresponds to 1. 
  uint8_t alpha;
  /// See TS 36.213 [23, 16.2.1.1]. Actual value = IE value * 2 [dB].
  uint8_t deltaPreambleMsg3;
}UplinkPowerControlCommon_NB_IoT;


/* DL-GapConfig-NB-r13 */
typedef struct {
	uint16_t	 dl_GapThreshold;
	uint16_t	 dl_GapPeriodicity;
	uint16_t	 dl_GapDurationCoeff;
} DL_GapConfig_NB_IoT;

typedef struct {

  /// Frame type (0 FDD, 1 TDD)
  NB_IoT_frame_type_t frame_type;
  /// Number of resource blocks (RB) in DL of the LTE (for knowing the bandwidth)
  uint8_t N_RB_DL;
  /// Number of resource blocks (RB) in UL of the LTE ((for knowing the bandwidth)
  uint8_t N_RB_UL;

  /// Cell ID
  uint16_t Nid_cell;
  /// Cyclic Prefix for DL (0=Normal CP, 1=Extended CP)
  NB_IoT_prefix_type_t Ncp;
  /// Cyclic Prefix for UL (0=Normal CP, 1=Extended CP)
  NB_IoT_prefix_type_t Ncp_UL;
  /// shift of pilot position in one RB
  uint8_t nushift;
  /// indicates if node is a UE (NODE=2) or eNB (PRIMARY_CH=0).
  uint8_t node_id;
  /// Frequency index of CBMIMO1 card
  uint8_t freq_idx;
  /// RX Frequency for ExpressMIMO/LIME
  uint32_t carrier_freq[4];
  /// TX Frequency for ExpressMIMO/LIME
  uint32_t carrier_freqtx[4];
  /// RX gain for ExpressMIMO/LIME
  uint32_t rxgain[4];
  /// TX gain for ExpressMIMO/LIME
  uint32_t txgain[4];
  /// RF mode for ExpressMIMO/LIME
  uint32_t rfmode[4];
  /// RF RX DC Calibration for ExpressMIMO/LIME
  uint32_t rxdc[4];
  /// RF TX DC Calibration for ExpressMIMO/LIME
  uint32_t rflocal[4];
  /// RF VCO calibration for ExpressMIMO/LIME
  uint32_t rfvcolocal[4];
  /// Turns on second TX of CBMIMO1 card
  uint8_t dual_tx;
  /// flag to indicate SISO transmission
  uint8_t mode1_flag;
  /// Indicator that 20 MHz channel uses 3/4 sampling frequency
  //uint8_t threequarter_fs;
  /// Size of FFT
  uint16_t ofdm_symbol_size;
  /// Number of prefix samples in all but first symbol of slot
  uint16_t nb_prefix_samples;
  /// Number of prefix samples in first symbol of slot
  uint16_t nb_prefix_samples0;
  /// Carrier offset in FFT buffer for first RE in PRB0
  uint16_t first_carrier_offset;
  /// Number of samples in a subframe
  uint32_t samples_per_tti;
  /// Number of OFDM/SC-FDMA symbols in one subframe (to be modified to account for potential different in UL/DL)
  uint16_t symbols_per_tti;
  /// Number of Physical transmit antennas in node
  uint8_t nb_antennas_tx;
  /// Number of Receive antennas in node
  uint8_t nb_antennas_rx;
  /// Number of common transmit antenna ports in eNodeB (1 or 2)
  uint8_t nb_antenna_ports_eNB;
  /// Number of common receiving antenna ports in eNodeB (1 or 2)
  uint8_t nb_antenna_ports_rx_eNB;
  /// NPRACH Config Common (from 36-331 RRC spec)
  NPRACH_CONFIG_COMMON nprach_config_common;
  /// NPDSCH Config Common (from 36-331 RRC spec)
  NPDSCH_CONFIG_COMMON npdsch_config_common;
  /// PUSCH Config Common (from 36-331 RRC spec)
  NPUSCH_CONFIG_COMMON npusch_config_common;
  /// UL Power Control (from 36-331 RRC spec)
  UplinkPowerControlCommon_NB_IoT ul_power_control_config_common;
  /// DL Gap
  DL_GapConfig_NB_IoT DL_gap_config;
  /// Size of SI windows used for repetition of one SI message (in frames)
  uint8_t SIwindowsize;
  /// Period of SI windows used for repetition of one SI message (in frames)
  uint16_t SIPeriod;
  int                 eutra_band;
  uint32_t            dl_CarrierFreq;
  uint32_t            ul_CarrierFreq;
  // CE level to determine the NPRACH Configuration (one CE for each NPRACH config.)
  uint8_t             CE;

  /*
   * index of the PRB assigned to NB-IoT carrier in in-band/guard-band operating mode
   */
  unsigned short NB_IoT_RB_ID;


  /*Following FAPI approach:
   * 0 = in-band with same PCI
   * 1 = in-band with diff PCI
   * 2 = guard band
   * 3 =stand alone
   */
  uint16_t operating_mode;

  /*
   * Only for In-band operating mode with same PCI
   * its measured in number of OFDM symbols
   * allowed values:
   * 1, 2, 3, 4(this value is written in FAPI specs but not exist in TS 36.331 v14.2.1 pag 587)
   * -1 (we put this value when is not defined - other operating mode)
   */
  uint16_t control_region_size;

  /*Number of EUTRA CRS antenna ports (AP)
   * valid only for in-band different PCI mode
   * value 0 = indicates the same number of AP as NRS APs
   * value 1 = four CRS APs
   */
  uint16_t eutra_NumCRS_ports;


} NB_IoT_DL_FRAME_PARMS;

typedef struct {
  /// \brief Holds the transmit data in time domain.
  /// For IFFT_FPGA this points to the same memory as PHY_vars->rx_vars[a].RX_DMA_BUFFER.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: tx antenna [0..nb_antennas_tx[
  /// - third index:
  int32_t **txdata[3];
  /// \brief holds the transmit data in the frequency domain.
  /// For IFFT_FPGA this points to the same memory as PHY_vars->rx_vars[a].RX_DMA_BUFFER. //?
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: tx antenna [0..14[ where 14 is the total supported antenna ports.
  /// - third index: sample [0..]
  int32_t **txdataF[3];
  /// \brief holds the transmit data after beamforming in the frequency domain.
  /// For IFFT_FPGA this points to the same memory as PHY_vars->rx_vars[a].RX_DMA_BUFFER. //?
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: tx antenna [0..nb_antennas_tx[
  /// - third index: sample [0..]
  int32_t **txdataF_BF[3];
  /// \brief Holds the received data in time domain.
  /// Should point to the same memory as PHY_vars->rx_vars[a].RX_DMA_BUFFER.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna [0..nb_antennas_rx[
  /// - third index: sample [0..]
  int32_t **rxdata[3];
  /// \brief Holds the last subframe of received data in time domain after removal of 7.5kHz frequency offset.
  /// - first index: secotr id [0..2] (hard coded)
  /// - second index: rx antenna [0..nb_antennas_rx[
  /// - third index: sample [0..samples_per_tti[
  int32_t **rxdata_7_5kHz[3];
  /// \brief Holds the received data in the frequency domain.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna [0..nb_antennas_rx[
  /// - third index: ? [0..2*ofdm_symbol_size*frame_parms->symbols_per_tti[
  int32_t **rxdataF[3];
  /// \brief Holds output of the sync correlator.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: sample [0..samples_per_tti*10[
  uint32_t *sync_corr[3];
  /// \brief Holds the beamforming weights
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: eNB antenna port index (hard coded)
  /// - third index: tx antenna [0..nb_antennas_tx[
  /// - fourth index: sample [0..]
  int32_t **beam_weights[3][15];
  /// \brief Holds the tdd reciprocity calibration coefficients
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: tx antenna [0..nb_antennas_tx[
  /// - third index: frequency [0..]
  int32_t **tdd_calib_coeffs[3];
} NB_IoT_eNB_COMMON;

typedef struct {
  /// \brief Hold the channel estimates in frequency domain based on SRS.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..ofdm_symbol_size[
  int32_t **srs_ch_estimates[3];
  /// \brief Hold the channel estimates in time domain based on SRS.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..2*ofdm_symbol_size[
  int32_t **srs_ch_estimates_time[3];
  /// \brief Holds the SRS for channel estimation at the RX.
  /// - first index: ? [0..ofdm_symbol_size[
  int32_t *srs;
} NB_IoT_eNB_SRS;

typedef struct {
  /// \brief Holds the received data in the frequency domain for the allocated RBs in repeated format.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..2*ofdm_symbol_size[
  /// - third index (definition from phy_init_lte_eNB()): ? [0..24*N_RB_UL*frame_parms->symbols_per_tti[
  /// \warning inconsistent third index definition
  int32_t **rxdataF_ext[3];
  /// \brief Holds the received data in the frequency domain for the allocated RBs in normal format.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index (definition from phy_init_lte_eNB()): ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **rxdataF_ext2[3];
  /// \brief Hold the channel estimates in time domain based on DRS.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..4*ofdm_symbol_size[
  int32_t **drs_ch_estimates_time[3];
  /// \brief Hold the channel estimates in frequency domain based on DRS.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **drs_ch_estimates[3];
  /// \brief Hold the channel estimates for UE0 in case of Distributed Alamouti Scheme.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **drs_ch_estimates_0[3];
  /// \brief Hold the channel estimates for UE1 in case of Distributed Almouti Scheme.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **drs_ch_estimates_1[3];
  /// \brief Holds the compensated signal.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **rxdataF_comp[3];
  /// \brief Hold the compensated data (y)*(h0*) in case of Distributed Alamouti Scheme.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **rxdataF_comp_0[3];
  /// \brief Hold the compensated data (y*)*(h1) in case of Distributed Alamouti Scheme.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **rxdataF_comp_1[3];
  /// \brief ?.
  /// - first index: sector id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_mag[3];
  /// \brief ?.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_magb[3];
  /// \brief Hold the channel mag for UE0 in case of Distributed Alamouti Scheme.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_mag_0[3];
  /// \brief Hold the channel magb for UE0 in case of Distributed Alamouti Scheme.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_magb_0[3];
  /// \brief Hold the channel mag for UE1 in case of Distributed Alamouti Scheme.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_mag_1[3];
  /// \brief Hold the channel magb for UE1 in case of Distributed Alamouti Scheme.
  /// - first index: eNB id [0..2] (hard coded)
  /// - second index: rx antenna id [0..nb_antennas_rx[
  /// - third index: ? [0..12*N_RB_UL*frame_parms->symbols_per_tti[
  int32_t **ul_ch_magb_1[3];
  /// measured RX power based on DRS
  int ulsch_power[2];
  /// measured RX power based on DRS for UE0 in case of Distributed Alamouti Scheme
  int ulsch_power_0[2];
  /// measured RX power based on DRS for UE0 in case of Distributed Alamouti Scheme
  int ulsch_power_1[2];
  /// \brief llr values.
  /// - first index: ? [0..1179743] (hard coded)
  int16_t *llr;
#ifdef LOCALIZATION
  /// number of active subcarrier for a specific UE
  int32_t active_subcarrier;
  /// subcarrier power in dBm
  int32_t *subcarrier_power;
#endif
} NB_IoT_eNB_PUSCH;

#define PBCH_A_NB_IoT 24
typedef struct {
  uint8_t pbch_d[96+(3*(16+PBCH_A_NB_IoT))];
  uint8_t pbch_w[3*3*(16+PBCH_A_NB_IoT)];
  uint8_t pbch_e[1920];
} NB_IoT_eNB_PBCH;


typedef enum {
  /// TM1
  SISO_NB_IoT=0,
  /// TM2
  ALAMOUTI_NB_IoT=1,
  /// TM3
  LARGE_CDD_NB_IoT=2,
  /// the next 6 entries are for TM5
  UNIFORM_PRECODING11_NB_IoT=3,
  UNIFORM_PRECODING1m1_NB_IoT=4,
  UNIFORM_PRECODING1j_NB_IoT=5,
  UNIFORM_PRECODING1mj_NB_IoT=6,
  PUSCH_PRECODING0_NB_IoT=7,
  PUSCH_PRECODING1_NB_IoT=8,
  /// the next 3 entries are for TM4
  DUALSTREAM_UNIFORM_PRECODING1_NB_IoT=9,
  DUALSTREAM_UNIFORM_PRECODINGj_NB_IoT=10,
  DUALSTREAM_PUSCH_PRECODING_NB_IoT=11,
  TM7_NB_IoT=12,
  TM8_NB_IoT=13,
  TM9_10_NB_IoT=14
} MIMO_mode_NB_IoT_t;

typedef struct {
  /// \brief ?.
  /// first index: ? [0..1023] (hard coded)
  int16_t *prachF;
  /// \brief ?.
  /// first index: rx antenna [0..63] (hard coded) \note Hard coded array size indexed by \c nb_antennas_rx.
  /// second index: ? [0..ofdm_symbol_size*12[
  int16_t *rxsigF[64];
  /// \brief local buffer to compute prach_ifft (necessary in case of multiple CCs)
  /// first index: rx antenna [0..63] (hard coded) \note Hard coded array size indexed by \c nb_antennas_rx.
  /// second index: ? [0..2047] (hard coded)
  int16_t *prach_ifft[64];
} NB_IoT_eNB_PRACH;

typedef enum {
  NOT_SYNCHED_NB_IoT=0,
  PRACH_NB_IoT=1,
  RA_RESPONSE_NB_IoT=2,
  PUSCH_NB_IoT=3,
  RESYNCH_NB_IoT=4
} UE_MODE_NB_IoT_t;


#endif
