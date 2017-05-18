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

/*! \file PHY/LTE_TRANSPORT/dci.h
* \brief typedefs for LTE DCI structures from 36-212, V8.6 2009-03.  Limited to 5 MHz formats for the moment.Current LTE compliance V8.6 2009-03.
* \author R. Knopp
* \date 2011
* \version 0.1
* \company Eurecom
* \email: knopp@eurecom.fr
* \note
* \warning
*/
#ifndef USER_MODE
#include "PHY/types.h"
#else
#include <stdint.h>
#endif

typedef enum DCI_format_NB
{
  DCIFormatN0 = 0,
  DCIFormatN1,
  DCIFormatN1_RA,
  DCIFormatN1_RAR,
  DCIFormatN2_Ind,
  DCIFormatN2_Pag,
}e_DCI_format_NB;

///  DCI Format Type 0 (180 kHz, 23 bits)
typedef struct DCIFormatN0{
  /// type = 0 => DCI Format N0, type = 1 => DCI Format N1, 1 bits
  uint8_t type;
  /// Subcarrier indication, 6 bits
  uint8_t scind;
  /// Resourse Assignment (RU Assignment), 3 bits
  uint8_t ResAssign;
  /// Modulation and Coding Scheme, 4 bits
  uint8_t mcs;
  /// New Data Indicator, 1 bits
  uint8_t ndi;
  /// Scheduling Delay, 2 bits
  uint8_t Scheddly;
  /// Repetition Number, 3 bits
  uint8_t RepNum;
  /// Redundancy version for HARQ (only use 0 and 2), 1 bits
  uint8_t rv;
  /// DCI subframe repetition Number, 2 bits
  uint8_t DCIRep;
};

typedef struct DCIFormatN0 DCIFormatN0_t;
#define sizeof_DDCIFormatN0_t 23

///  DCI Format Type N1 for User data
struct DCIFormatN1{
  /// type = 0 => DCI Format N0, type = 1 => DCI Format N1,1bits
  uint8_t type;
  //NPDCCH order indicator (set to 0), 1 bits
  uint8_t orderIndicator;
  // Scheduling Delay,3 bits
  uint8_t Scheddly;
  // Resourse Assignment (RU Assignment),3 bits
  uint8_t ResAssign;
  // Modulation and Coding Scheme,4 bits
  uint8_t mcs;
  // Repetition Number,4 bits
  uint8_t RepNum;
  // New Data Indicator,1 bits
  uint8_t ndi;
  // HARQ-ACK resource,4 bits
  uint8_t HARQackRes;
  // DCI subframe repetition Number,2 bits
  uint8_t DCIRep;
};


typedef struct DCIFormatN1 DCIFormatN1_t;
#define sizeof_DCIFormatN1_t 23

///  DCI Format Type N1 for initial RA
struct DCIFormatN1_RA{
  /// type = 0 => DCI Format N0, type = 1 => DCI Format N1, 1 bits
  uint8_t type;
  //NPDCCH order indicator (set to 0),1 bits
  uint8_t orderIndicator;
  // Start number of NPRACH repetiiton, 2 bits
  uint8_t Scheddly;
  // Subcarrier indication of NPRACH, 6 bits
  uint8_t scind;
  // All the remainging bits, 13 bits
  uint8_t remaingingBits;
};

typedef struct DCIFormatN1_RA DCIFormatN1_RA_t;
#define sizeof_DCIFormatN1_RA_t 23

///  DCI Format Type N1 for RAR
struct DCIFormatN1_RAR{
  /// type = 0 => DCI Format N0, type = 1 => DCI Format N1, 1 bits
  uint8_t type;
  //NPDCCH order indicator (set to 0),1 bits
  uint8_t orderIndicator;
  // Scheduling Delay, 3 bits
  uint8_t Scheddly;
  // Resourse Assignment (RU Assignment), 3 bits
  uint8_t ResAssign;
  // Modulation and Coding Scheme, 4 bits
  uint8_t mcs;
  // Repetition Number, 4 bits
  uint8_t RepNum;
  // Reserved 5 bits
  uint8_t Reserved;
  // DCI subframe repetition Number, 2 bits
  uint8_t DCIRep;
};

typedef struct DCIFormatN1_RAR DCIFormatN1_RAR_t;
#define sizeof_DCIFormatN1_RAR_t 23

//  DCI Format Type N2 for direct indication, 15 bits
struct DCIFormatN2_Ind{
  //Flag for paging(1)/direct indication(0), set to 0,1 bits
  uint8_t type;
  //Direct indication information, 8 bits
  uint8_t directIndInf;
  // Reserved information bits, 6 bits
  uint8_t resInfoBits;
};

typedef struct DCIFormatN2_Ind DCIFormatN2_Ind_t;
#define sizeof_DCIFormatN2_Ind_t 15

//  DCI Format Type N2 for Paging, 15 bits
struct DCIFormatN2_Pag{
  //Flag for paging(1)/direct indication(0), set to 1,1 bits
  uint8_t type;
  // Resourse Assignment (RU Assignment), 3 bits
  uint8_t ResAssign;
  // Modulation and Coding Scheme, 4 bits
  uint8_t mcs;
  // Repetition Number, 4 bits
  uint8_t RepNum;
  // Reserved 3 bits
  uint8_t DCIRep;
};

typedef struct DCIFormatN2_Pag DCIFormatN2_Pag_t;
#define sizeof_DCIFormatN2_Pag_t 15

// struct DCI0_5MHz_TDD0 {
//   /// type = 0 => DCI Format 0, type = 1 => DCI Format 1A
//   uint32_t type:1;
//   /// Hopping flag
//   uint32_t hopping:1;
//   /// RB Assignment (ceil(log2(N_RB_UL*(N_RB_UL+1)/2)) bits)
//   uint32_t rballoc:9;
//   /// Modulation and Coding Scheme and Redundancy Version
//   uint32_t mcs:5;
//   /// New Data Indicator
//   uint32_t ndi:1;
//   /// Power Control
//   uint32_t TPC:2;
//   /// Cyclic shift
//   uint32_t cshift:3;
//   /// DAI (TDD)
//   uint32_t ulindex:2;
//   /// CQI Request
//   uint32_t cqi_req:1;
//   /// Padding to get to size of DCI1A
//   uint32_t padding:2;
// } __attribute__ ((__packed__));

// typedef struct DCI0_5MHz_TDD0 DCI0_5MHz_TDD0_t;
// #define sizeof_DCI0_5MHz_TDD_0_t 27

//Not sure if needed in NB-IoT
// struct DCI_INFO_EXTRACTED {
//     /// type = 0 => DCI Format 0, type = 1 => DCI Format 1A
//     uint8_t type;
//     /// Resource Allocation Header
//     uint8_t rah;
//     /// HARQ Process
//     uint8_t harq_pid;
//     /// CQI Request
//     uint8_t cqi_req;
//     /// SRS Request
//     uint8_t srs_req;
//     /// Power Control
//     uint8_t TPC;
//     /// Localized/Distributed VRB
//     uint8_t vrb_type;
//     /// RB Assignment (ceil(log2(N_RB_DL/P)) bits)
//     uint32_t rballoc;
//     // Applicable only when vrb_type = 1
//     uint8_t Ngap;
//     /// Cyclic shift
//     uint8_t cshift;
//     /// Hopping flag
//     uint8_t hopping;
//     /// Downlink Assignment Index
//     uint8_t dai;
//     /// DAI (TDD)
//     uint8_t ulindex;

//     /// TB swap
//     uint8_t tb_swap;
//     /// TPMI information for precoding
//     uint8_t tpmi;
//     /// Redundancy version 2
//     uint8_t rv2;
//     /// New Data Indicator 2
//     uint8_t ndi2;
//     /// Modulation and Coding Scheme and Redundancy Version 2
//     uint8_t mcs2;
//     /// Redundancy version 1
//     uint8_t rv1;
//     /// New Data Indicator 1
//     uint8_t ndi1;
//     /// Modulation and Coding Scheme and Redundancy Version 1
//     uint8_t mcs1;

//     /// Scrambling ID
//     uint64_t ap_si_nl_id:3;
// };
// typedef struct DCI_INFO_EXTRACTED DCI_INFO_EXTRACTED_t;
