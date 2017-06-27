#include "../SCHED/IF_Module_L1_primitives_nb_iot.h"




//to be integrated in the scheduling procedure of L1
void schedule_response(Sched_Rsp_t *Sched_INFO){
      //todo
}

void PHY_config_req(PHY_Config_t* config_INFO){


	if(config_INFO->get_MIB != 0){
		//MIB-NB configuration
		NB_phy_config_mib_eNB(config_INFO->mod_id,
							  config_INFO->CC_id,
							  config_INFO->frequency_band_indicator,
							  config_INFO->sch_config.physical_cell_id.value,
							  config_INFO->subframe_config.dl_cyclic_prefix_type.value,
							  config_INFO->subframe_config.ul_cyclic_prefix_type.value,
							  config_INFO->rf_config.tx_antenna_ports.value,
							  config_INFO->dl_CarrierFreq,
							  config_INFO->ul_CarrierFreq);
	}

	if(config_INFO->get_COMMON != 0)
	{
		//Common Configuration included in SIB2-NB
		NB_phy_config_sib2_eNB(config_INFO->mod_id,
							   config_INFO->CC_id,
						       &config_INFO->nb_iot_config, // FIXME to be evaluated is should be passed a pointer
						       &config_INFO->rf_config,
							   &config_INFO->uplink_reference_signal_config,
							   &config_INFO->extra_phy_parms
							   );
	}

	if(config_INFO->get_DEDICATED!= 0)
	{
	//Dedicated Configuration
		if(config_INFO->phy_config_dedicated != NULL){

			NB_phy_config_dedicated_eNB(config_INFO->mod_id,
										config_INFO->CC_id,
										config_INFO->rnti,
										config_INFO->phy_config_dedicated //not defined by fapi specs
										);
		}

	}
}

