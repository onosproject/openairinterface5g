
/*! \file extern_NB_IoT.h
 * \brief MAC extern
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#ifndef __MAC_EXTERN_NB_IOT_H__
#define __MAC_EXTERN_NB_IOT_H__


// #ifdef USER_MODE
// //#include "stdio.h"
// #endif //USER_MODE
// #include "PHY/defs.h"
// #include "defs.h"
// #include "COMMON/mac_rrc_primitives.h"
// #ifdef PHY_EMUL
// //#include "SIMULATION/simulation_defs.h"
// #endif //PHY_EMUL
#include "openair2/PHY_INTERFACE/defs_NB_IoT.h" 
//#include "RRC/LITE/defs_NB_IoT.h"

//#include "LAYER2/MAC/vars_NB_IoT.h"

//NB-IoT

extern eNB_MAC_INST_NB_IoT *mac_inst;

// //extern uint32_t EBSR_Level[63];
// extern const uint32_t Extended_BSR_TABLE[BSR_TABLE_SIZE];
// //extern uint32_t Extended_BSR_TABLE[63];  ----currently not used 

// extern const uint8_t cqi2fmt0_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern const uint8_t cqi2fmt1x_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern const uint8_t cqi2fmt2x_agg[MAX_SUPPORTED_BW][CQI_VALUE_RANGE];

// extern UE_MAC_INST *UE_mac_inst;
// extern eNB_MAC_INST *eNB_mac_inst;
// extern eNB_RRC_INST *eNB_rrc_inst;
//extern UE_RRC_INST_NB_IoT *UE_rrc_inst_NB_IoT;
// extern UE_MAC_INST *ue_mac_inst;
// extern MAC_RLC_XFACE *Mac_rlc_xface;
// extern uint8_t Is_rrc_registered;


//#ifndef USER_MODE

// extern RRC_XFACE *Rrc_xface;          //// to uncomment when it is used

extern uint8_t Is_rrc_registered;

#ifndef PHY_EMUL
#ifndef PHYSIM
#define NB_INST 1
#else
extern unsigned char NB_INST;
#endif
extern unsigned char NB_eNB_INST;
extern unsigned char NB_UE_INST;
extern unsigned char NB_RN_INST;
extern unsigned short NODE_ID[1];
extern void* bigphys_malloc(int);
#else
extern EMULATION_VARS *Emul_vars;
#endif //PHY_EMUL




//NB-IoT---------------------------------

extern eNB_MAC_INST_NB_IoT *mac_inst;
extern uint8_t Is_rrc_registered_NB_IoT;
//extern BCCH_BCH_Message_NB_t               MIB;
//extern BCCH_DL_SCH_Message_NB_t            SIB;
//extern RRCConnectionSetup_NB_r13_IEs_t     DED_Config;

extern available_resource_DL_t *available_resource_DL;
extern available_resource_tones_UL_t *available_resource_UL;
extern available_resource_DL_t *available_resource_DL_last;
extern schedule_result_t *schedule_result_list_UL;
extern schedule_result_t *schedule_result_list_DL;

// array will be active when they are used

// 10 -> single-tone / 12 -> multi-tone
extern const uint32_t max_mcs[2];

// [CE level] [0 - 3] -> single-tone / [CE level] [4-7] -> multi-tone
extern const uint32_t mapped_mcs[3][8];

//TBS table for NPUSCH transmission TS 36.213 v14.2 table Table 16.5.1.2-2:
extern const int UL_TBS_Table[14][8];

extern const int rachperiod[8];
extern const int rachstart[8];
extern const int rachrepeat[8];
extern const int rachscofst[7];
extern const int rachnumsc[4];

extern const uint32_t RU_table[8];

extern const uint32_t scheduling_delay[4];
extern const uint32_t msg3_scheduling_delay_table[4];

extern const uint32_t ack_nack_delay[4];
extern const uint32_t R_dl_table[16];

// TBS table for the case not containing SIB1-NB_IoT, Table 16.4.1.5.1-1 in TS 36.213 v14.2
extern const uint32_t MAC_TBStable_NB_IoT[14][8];

//TBS table for the case containing S1B1-NB_IoT, Table 16.4.1.5.2-1 in TS 36.213 v14.2 (Itbs = 12 ~ 15 is reserved field
//mapping ITBS to SIB1-NB_IoT
extern const unsigned int MAC_TBStable_NB_IoT_SIB1[16];

extern const int DV_table[16];
extern const int BSR_table[64];

extern const int dl_rep[3];
extern const uint32_t dci_rep[3];
extern const uint32_t harq_rep[3];



#endif //DEF_H


