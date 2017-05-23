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



#include "types.h"
//#include "defs.h"

#define	A_SEQUENCE_OF(type)	A_SET_OF(type)

#define	A_SET_OF(type)					\
	struct {					\
		type **array;				\
		int count;	/* Meaningful size */	\
		int size;	/* Allocated size */	\
		void (*free)(type *);			\
	}


/// NPRACH-ParametersList-NB-r13 from 36.331 RRC spec
typedef struct {
  /// the period time for nprach
  uint8_t nprach_Periodicity;
  /// for the start time for the NPRACH resource from 40ms-2560ms
  uint8_t nprach_StartTime;	
  /// for the subcarrier of set to the NPRACH preamble from n0 - n34
  uint8_t nprach_SubcarrierOffset;
  /// where is the region that in NPRACH resource to indicate if this UE support MSG3 for multi-tone or not. from 0 - 1
  uint8_t nprach_SubcarrierMSG3_RangeStart;
  /// The max preamble transmission attempt for the CE level from 1 - 128
  uint8_t maxNumPreambleAttemptCE;
  /// The number of the repetition for DCI use in RAR/MSG3/MSG4 from 1 - 2048
  uint8_t npdcch_NumRepetitions_RA;
  /// Starting subframe for NPDCCH Common searching space for (RAR/MSG3/MSG4)
  uint8_t npdcch_StartSF_CSS_RA;
  /// Fractional period offset of starting subframe for NPDCCH common search space
  uint8_t npdcch_Offset_RA;
} nprach_parameters_NB_t;

typedef struct{
  A_SEQUENCE_OF(struct NPRACH_Parameters_NB) list;
}NPRACH_List_NB_t;

typedef long RSRP_Range_t;

typedef struct {
  A_SEQUENCE_OF(RSRP_Range_t) list;
}rsrp_ThresholdsNPrachInfoList;


/// NPRACH_ConfigSIB-NB from 36.331 RRC spec
typedef struct {
  /// nprach_CP_Length_r13, for the CP length(unit us) only 66.7 and 266.7 is implemented
  uint8_t nprach_CP_Length;
  /// The criterion for UEs to select a NPRACH resource. Up to 2 RSRP threshold values can be signalled.  \vr{[1..2]}
  struct rsrp_ThresholdsNPrachInfoList *rsrp_ThresholdsPrachInfoList;
  /// NPRACH Parameters List
  NPRACH_List_NB_t nprach_ParametersList;
} NPRACH_CONFIG_COMMON;

/// NPDSCH-ConfigCommon from 36.331 RRC spec
typedef struct {
  ///see TS 36.213 (16.2). \vr{[-60..50]}\n Provides the downlink reference-signal EPRE. The actual value in dBm.
  int8_t nrs_Power;
} NPDSCH_CONFIG_COMMON;

typedef struct{
  /// The base sequence of DMRS sequence in a cell for 3 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 12. Value 12 is not used.
  uint8_t threeTone_BaseSequence;
  /// Define 3 cyclic shifts for the 3-tone case, see TS 36.211 [21, 10.1.4.1.2].
  uint8_t threeTone_CyclicShift;
  /// The base sequence of DMRS sequence in a cell for 6 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 14. Value 14 is not used.
  uint8_t sixTone_BaseSequence;
  /// Define 4 cyclic shifts for the 6-tone case, see TS 36.211 [21, 10.1.4.1.2].
  uint8_t sixTone_CyclicShift;
  /// The base sequence of DMRS sequence in a cell for 12 tones transmission; see TS 36.211 [21, 10.1.4.1.2]. If absent, it is given by NB-IoT CellID mod 30. Value 30 is not used.
  uint8_t twelveTone_BaseSequence;

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
}UplinkPowerControlCommon_NB;


/* DL-GapConfig-NB-r13 */
typedef struct {
	uint8_t	 dl_GapThreshold;
	uint8_t	 dl_GapPeriodicity;
	uint8_t	 dl_GapDurationCoeff;
} DL_GapConfig_NB;

typedef struct {
  /// Cell ID
  uint16_t Nid_cell;
  /// Cyclic Prefix for DL (0=Normal CP, 1=Extended CP)
  lte_prefix_type_t Ncp;
  /// Cyclic Prefix for UL (0=Normal CP, 1=Extended CP)
  lte_prefix_type_t Ncp_UL;
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
  /// NPRACH Config Common (from 36-331 RRC spec)
  NPRACH_CONFIG_COMMON nprach_config_common;
  /// NPDSCH Config Common (from 36-331 RRC spec)
  NPDSCH_CONFIG_COMMON npdsch_config_common;
  /// PUSCH Config Common (from 36-331 RRC spec)
  NPUSCH_CONFIG_COMMON npusch_config_common;
  /// UL Power Control (from 36-331 RRC spec)
  UplinkPowerControlCommon_NB ul_power_control_config_common;
  /// DL Gap
  DL_GapConfig_NB DL_gap_config;
  /// Size of SI windows used for repetition of one SI message (in frames)
  uint8_t SIwindowsize;
  /// Period of SI windows used for repetition of one SI message (in frames)
  uint16_t SIPeriod;
  int                 eutra_band;
  uint32_t            dl_CarrierFreq;
  uint32_t            ul_CarrierFreq;
  uint8_t             CE;// CE level to determine the NPRACH Configuration

} NB_DL_FRAME_PARMS;

