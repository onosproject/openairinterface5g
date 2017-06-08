#include "openair2/PHY_INTERFACE/IF_Module_nb_iot.h"
#include "LAYER2/MAC/extern.h"
#include "LAYER2/MAC/proto_nb_iot.h"

void UL_indication(UL_IND_t UL_INFO)
{
    int i=0;
    UL_INFO.test=1;
    if(UL_INFO.test == 1)
        {
          /*If there is a preamble, do the initiate RA procedure*/
          if(UL_INFO.Number_SC>0)
            {
              for(i=0;i<UL_INFO.Number_SC;i++)
              {
                NB_initiate_ra_proc(UL_INFO.module_id,
                                    UL_INFO.CC_id,
                                    UL_INFO.frame,
                                    UL_INFO.Preamble_list[UL_INFO.Number_SC].preamble_index,
                                    UL_INFO.Preamble_list[UL_INFO.Number_SC].timing_offset,
                                    UL_INFO.subframe
                                    );
              }             
            } 
          /*If there is a Uplink SDU (even MSG3, NAK) need to send to MAC*/
          for(i=0;i<UL_INFO.UE_NUM;i++)
              {
                  /*For MSG3, Normal Uplink Data, NAK*/
                  if(UL_INFO.UL_SPEC_Info[i].RNTI)
                      NB_rx_sdu(UL_INFO.module_id,
                                UL_INFO.CC_id,
                                UL_INFO.frame,
                                UL_INFO.subframe,
                                UL_INFO.UL_SPEC_Info[i].RNTI,
                                UL_INFO.UL_SPEC_Info[i].sdu,
                                UL_INFO.UL_SPEC_Info[i].sdu_lenP,
                                UL_INFO.UL_SPEC_Info[i].harq_pidP,
                                UL_INFO.UL_SPEC_Info[i].msg3_flagP
                                );
                    

              }
        }

    //NB_eNB_dlsch_ulsch_scheduler(UL_INFO.module_id,0,UL_INFO.frame,UL_INFO.subframe); TODO: to be implemented
}

void schedule_response(Sched_Rsp_t Sched_INFO){
      //todo
}

void PHY_config_req(PHY_Config_t* config_INFO){


	if(config_INFO->get_MIB != 0){
		//MIB-NB configuration
		NB_phy_config_mib_eNB(config_INFO->mod_id,
							  config_INFO->CC_id,
							  config_INFO->frequency_band_indicator,
							  config_INFO->sch_config.physical_cell_id,
							  config_INFO->subframe_config.dl_cyclic_prefix_type,
							  config_INFO->rf_config.tx_antenna_ports,
							  config_INFO->dl_CarrierFreq,
							  config_INFO->ul_CarrierFreq);
	}

	if(config_INFO->get_COMMON != 0)
	{
		//Common Configuration included in SIB2-NB
		NB_phy_config_sib2_eNB(config_INFO->mod_id,
							   config_INFO->CC_id,
						       &config_INFO->nb_iot_config, // FIXME to be evaluated is should be passed a pointer
						       &config_INFO->rf_config); // FIXME to be evaluated is should be passed a pointer
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


int IF_Module_init(IF_Module_t *if_inst){
  
  if_inst->UL_indication      = UL_indication;
  if_inst->schedule_response  = schedule_response;
  if_inst->PHY_config_req 	  = PHY_config_req;
  
  //create the UL_IND_t , Sched_Resp_t and PHY_Config_t structures

  return 0;
}

