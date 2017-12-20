#include "PHY/defs.h"
#include "PHY/extern.h"
#include "PHY/CUDA/LTE_TRANSPORT/defs.h"
void rx_ulsch_cu(PHY_VARS_eNB *phy_vars_eNB,
              uint32_t sched_subframe,
              uint8_t eNB_id,  // this is the effective sector id
              uint8_t UE_id,
              LTE_eNB_ULSCH_t **ulsch,
              uint8_t cooperation_flag);