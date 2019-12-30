#ifndef __PHY_INIT_NB_IOT__H__
#define __PHY_INIT_NB_IOT__H__

#include "../defs_L1_NB_IoT.h"

// for NB-IoT testing  

PHY_VARS_eNB_NB_IoT* init_lte_eNB_NB_IoT(NB_IoT_DL_FRAME_PARMS *frame_parms,
                                          uint8_t eNB_id,
                                          uint8_t Nid_cell,
                                          eNB_func_NB_IoT_t node_function,
                                          uint8_t abstraction_flag);

int phy_init_lte_eNB_NB_IoT(PHY_VARS_eNB_NB_IoT *phy_vars_eNb,
                     unsigned char is_secondary_eNb,
                     unsigned char abstraction_flag);

#endif