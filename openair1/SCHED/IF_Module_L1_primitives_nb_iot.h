#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_nb_iot.h"


#ifndef __IF_MODULE_L1_PRIMITIVES_NB_IOT_H__
#define __IF_MODULE_L1_PRIMITIVES_NB_IOT_H__

/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void schedule_response(Sched_Rsp_t *Sched_INFO);

/*Interface for PHY Configuration
 * Trigger the phy_config_xxx functions using parameters from the shared PHY_Config structure
 * */
void PHY_config_req(PHY_Config_t* config_INFO);

#endif
