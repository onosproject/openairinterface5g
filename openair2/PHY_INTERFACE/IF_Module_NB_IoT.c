#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"
#include "openair2/PHY_INTERFACE/IF_Module_L2_primitives_NB_IoT.h"
#include "openair1/SCHED/IF_Module_L1_primitives_NB_IoT.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_NB_IoT.h"


//called at initialization of L2
void IF_Module_init_L2(void) // northbound IF-Module Interface
{
	//mapping the IF-Module function to L2 definition
	if_inst->UL_indication = UL_indication;

	//return if_inst;
}


//called at initialization of L1 (phy_init_lte_eNB)
void IF_Module_init_L1(void) //southbound IF-Module Interface
{
	//mapping the IF-module function to L1 definition
	if_inst->schedule_response = schedule_response;
	if_inst->PHY_config_req  = PHY_config_req;

	//return if_inst;
}

