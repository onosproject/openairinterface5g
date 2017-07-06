/*******************************************************************************
 
 *******************************************************************************/
/*! \file PHY/LTE_TRANSPORT/defs_NB_IoT.h
* \brief data structures for NPDSCH/NDLSCH/NPUSCH/NULSCH physical and transport channel descriptors (TX/RX) of NB-IoT
* \author M. KANJ
* \date 2017
* \version 0.0
* \company bcom
* \email: matthieu.kanj@b-com.com
* \note
* \warning
*/
#ifndef __LTE_TRANSPORT_DEFS_NB_IOT__H__
#define __LTE_TRANSPORT_DEFS_NB_IOT__H__
#include "PHY/defs.h"
#include "dci_nb_iot.h"
#include "dci.h"
#include "uci.h"
#ifndef STANDALONE_COMPILE
#include "UTIL/LISTS/list.h"
#endif

#define MOD_TABLE_QPSK_OFFSET 1
#define MOD_TABLE_16QAM_OFFSET 5
#define MOD_TABLE_64QAM_OFFSET 21
#define MOD_TABLE_PSS_OFFSET 85

// structures below implement 36-211 and 36-212

#define NSOFT 1827072
#define LTE_NULL 2

// maximum of 3 segments before each coding block if data length exceeds 6144 bits.

#define MAX_NUM_DLSCH_SEGMENTS 16
#define MAX_NUM_ULSCH_SEGMENTS MAX_NUM_DLSCH_SEGMENTS
#define MAX_DLSCH_PAYLOAD_BYTES (MAX_NUM_DLSCH_SEGMENTS*768)
#define MAX_ULSCH_PAYLOAD_BYTES (MAX_NUM_ULSCH_SEGMENTS*768)

#define MAX_NUM_CHANNEL_BITS (14*1200*6)  // 14 symbols, 1200 REs, 12 bits/RE
#define MAX_NUM_RE (14*1200)

#if !defined(SI_RNTI)
#define SI_RNTI  (rnti_t)0xffff
#endif
#if !defined(M_RNTI)
#define M_RNTI   (rnti_t)0xfffd
#endif
#if !defined(P_RNTI)
#define P_RNTI   (rnti_t)0xfffe
#endif
#if !defined(CBA_RNTI)
#define CBA_RNTI (rnti_t)0xfff4
#endif
#if !defined(C_RNTI)
#define C_RNTI   (rnti_t)0x1234
#endif

#define PMI_2A_11 0
#define PMI_2A_1m1 1
#define PMI_2A_1j 2
#define PMI_2A_1mj 3

// for NB-IoT
#define MAX_NUM_CHANNEL_BITS_NB_IOT 3360 			//14 symbols * 12 sub-carriers * 10 SF * 2bits/RE  // to check during real tests
#define MAX_DL_SIZE_BITS_NB_IOT 680 				// in release 13 // in release 14 = 2048      // ??? **** not sure
//#define MAX_NUM_CHANNEL_BITS_NB_IOT 3*680  			/// ??? ****not sure

// to be created LTE_eNB_DLSCH_t --> is duplicated for each number of UE and then indexed in the table

typedef struct {    																		// LTE_eNB_DLSCH_t
  /// TX buffers for UE-spec transmission (antenna ports 5 or 7..14, prior to precoding)
  uint32_t *txdataF[8];
  /// Allocated RNTI (0 means DLSCH_t is not currently used)
  uint16_t rnti;
  /// Active flag for baseband transmitter processing
  uint8_t active;
  /// Indicator of TX activation per subframe.  Used during PUCCH detection for ACK/NAK.
  uint8_t subframe_tx[10];
  /// First CCE of last PDSCH scheduling per subframe.  Again used during PUCCH detection for ACK/NAK.
  uint8_t nCCE[10];
  /// Current HARQ process id
  uint8_t current_harq_pid;
  /// Process ID's per subframe.  Used to associate received ACKs on PUSCH/PUCCH to DLSCH harq process ids
  uint8_t harq_ids[10];
  /// Window size (in outgoing transport blocks) for fine-grain rate adaptation
  uint8_t ra_window_size;
  /// First-round error threshold for fine-grain rate adaptation
  uint8_t error_threshold;
  /// Pointers to 8 HARQ processes for the DLSCH
  NB_IoT_DL_eNB_HARQ_t harq_processe;
  /// circular list of free harq PIDs (the oldest come first)
  /// (10 is arbitrary value, must be > to max number of DL HARQ processes in LTE)
  int harq_pid_freelist[10];
  /// the head position of the free list (if list is free then head=tail)
  int head_freelist;
  /// the tail position of the free list
  int tail_freelist;
  /// Number of soft channel bits
  uint32_t G;
  /// Codebook index for this dlsch (0,1,2,3)
  uint8_t codebook_index;
  /// Maximum number of HARQ processes (for definition see 36-212 V8.6 2009-03, p.17)
  uint8_t Mdlharq;
  /// Maximum number of HARQ rounds
  uint8_t Mlimit;
  /// MIMO transmission mode indicator for this sub-frame (for definition see 36-212 V8.6 2009-03, p.17)
  uint8_t Kmimo;
  /// Nsoft parameter related to UE Category
  uint32_t Nsoft;
  /// amplitude of PDSCH (compared to RS) in symbols without pilots
  int16_t sqrt_rho_a;
  /// amplitude of PDSCH (compared to RS) in symbols containing pilots
  int16_t sqrt_rho_b;

} NB_IoT_eNB_DLSCH_t;



typedef struct { 															// LTE_DL_eNB_HARQ_t
  /// Status Flag indicating for this DLSCH (idle,active,disabled)
  SCH_status_t status;
  /// Transport block size
  uint32_t TBS;
  /// The payload + CRC size in bits, "B" from 36-212
  uint32_t B;        // keep this parameter
  /// Pointer to the payload
  uint8_t *b;		// keep this parameter
  /// Pointers to transport block segments
  //uint8_t *c[MAX_NUM_DLSCH_SEGMENTS];
  /// RTC values for each segment (for definition see 36-212 V8.6 2009-03, p.15)
 // uint32_t RTC[MAX_NUM_DLSCH_SEGMENTS];
  /// Frame where current HARQ round was sent
  uint32_t frame;
  /// Subframe where current HARQ round was sent
  uint32_t subframe;
  /// Index of current HARQ round for this DLSCH
  uint8_t round;
  /// MCS format for this DLSCH
  uint8_t mcs;
  /// Redundancy-version of the current sub-frame
  uint8_t rvidx;
  /// MIMO mode for this DLSCH
  MIMO_mode_t mimo_mode;
  /// Current RB allocation
  uint32_t rb_alloc[4];
  /// distributed/localized flag
  vrb_t vrb_type;
  /// Current subband PMI allocation
  uint16_t pmi_alloc;
  /// Current subband RI allocation
  uint32_t ri_alloc;
  /// Current subband CQI1 allocation
  uint32_t cqi_alloc1;
  /// Current subband CQI2 allocation
  uint32_t cqi_alloc2;
  /// Current Number of RBs
  uint16_t nb_rb;
  /// downlink power offset field
  uint8_t dl_power_off;
  /// Concatenated "e"-sequences (for definition see 36-212 V8.6 2009-03, p.17-18)
  uint8_t e[MAX_NUM_CHANNEL_BITS_NB_IOT];
  /// data after scrambling
  uint8_t s_e[MAX_NUM_CHANNEL_BITS_NB_IOT];
  /// length of the table e                 
  uint16_t length_e									// new parameter
  /// Tail-biting convolutional coding outputs
  uint8_t d[96+(3*(24+MAX_DL_SIZE_BITS_NB_IOT))];  // new parameter
  /// Sub-block interleaver outputs 
  uint8_t w[3*3*(MAX_DL_SIZE_BITS_NB_IOT+24)];  	  // new parameter
  /// Number of MIMO layers (streams) (for definition see 36-212 V8.6 2009-03, p.17, TM3-4)
  uint8_t Nl;
  /// Number of layers for this PDSCH transmission (TM8-10)
  uint8_t Nlayers;
  /// First layer for this PSCH transmission
  uint8_t first_layer;
} NB_IoT_DL_eNB_HARQ_t;


#endif
