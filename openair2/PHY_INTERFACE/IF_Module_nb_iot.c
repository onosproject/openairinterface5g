#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "openair2/PHY_INTERFACE/IF_Module_L2_primitives_nb_iot.h"
#include "openair1/SCHED/IF_Module_L1_primitives_nb_iot.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_nb_iot.h"


//called at initialization of L2
//TODO: define the input
void IF_Module_init_L2(void) //southbound IF-Module Interface
{
	//register the IF Module to MAC
	if_inst->UL_indication = UL_indication;

	//return if_inst;
}

//called at initialization of L1
//TODO: define the input
void IF_Module_init_L1(void) //northbound IF-Module Interface
{
	//fill the Sched_Rsp_t
	//fill the PHY_Config_t -->already done in rrc_mac_config
	if_inst->schedule_response = schedule_response;
	if_inst->PHY_config_req  = PHY_config_req;

	//return if_inst;
}

