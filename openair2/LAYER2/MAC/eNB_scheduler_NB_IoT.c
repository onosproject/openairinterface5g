
/*! \file eNB_scheduler_NB_IoT.c
 * \brief top level of the scheduler, it scheduled in pdcch period based.
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"


//	scheduler
#define flag_css_type1 0x1
#define flag_css_type2 0x2
#define flag_uss_v	   0x4


//	common
#define flag_mib      0x1
#define flag_sib1     0x2
#define flag_npss     0x4
#define flag_nsss     0x8


//extern BCCH_DL_SCH_Message_NB_IoT_t SIB;

void eNB_scheduler_computing_flag_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t abs_subframe, uint32_t *scheduler_flags, uint32_t *common_flags, uint32_t *max_subframe){
	uint32_t subframe = abs_subframe % 10;
	uint32_t frame = abs_subframe / 10;
	int i;
    uint32_t max = 0;
    
    //NPRACH_Parameters_NB_IoT_r13_t **type2_css_info = SIB.message.choice.c1.choice.systemInformation_r13.criticalExtensions.choice.systemInformation_r13.sib_TypeAndInfo_r13.choice.sib2_r13.radioResourceConfigCommon_r13.nprach_Config_r13.nprach_ParametersList_r13.list.array;
    
    //	fixed scheduling part (e.g. MIB, NPSS, NSSS, SIB1)
	if(subframe == 0){
		*common_flags |= flag_mib;
	}else if(subframe == 5){
		*common_flags |= flag_npss;
	}else if(subframe == 9 && (frame&0x1)==0){
		*common_flags |= flag_nsss;
	}else if(subframe == 4 && mac_inst->sib1_flag[frame%mac_inst->sib1_period]){
		*common_flags |= flag_sib1;
	}

/*	uint32_t type2_css_pp[3] = { 	type2_css_info[0]->npdcch_NumRepetitions_RA_r13*type2_css_info[0]->npdcch_StartSF_CSS_RA_r13,		type2_css_info[1]->npdcch_NumRepetitions_RA_r13*type2_css_info[1]->npdcch_StartSF_CSS_RA_r13,		type2_css_info[2]->npdcch_NumRepetitions_RA_r13*type2_css_info[2]->npdcch_StartSF_CSS_RA_r13	};*/
	uint32_t type2_css_pp[3] = {256, 256, 256};	//	TODO RRC config should get from structure
	uint32_t start_subframe;
	for(i=0; i<1; ++i){	//	only CE0
		start_subframe = 0;						
		/*switch(type2_css_info[i]->npdcch_Offset_RA_r13){
			case NPRACH_Parameters_NB_IoT_r13__npdcch_Offset_RA_r13_zero:			
				start_subframe = 0;						
				break;
			case NPRACH_Parameters_NB_IoT_r13__npdcch_Offset_RA_r13_oneEighth:		
				start_subframe = type2_css_pp[i]>>3;	
				break;
			case NPRACH_Parameters_NB_IoT_r13__npdcch_Offset_RA_r13_oneFourth:		
				start_subframe = type2_css_pp[i]>>2;	
				break;
			case NPRACH_Parameters_NB_IoT_r13__npdcch_Offset_RA_r13_threeEighth:	
				start_subframe = (type2_css_pp[i]>>3)+(type2_css_pp[i]>>2);
				break;
			default:	break;
		}*/
	
		if(((abs_subframe+1)%type2_css_pp[i])==start_subframe){
			*scheduler_flags |= flag_css_type2;
			max = MAX(max, extend_space[i]);
			LOG_D(MAC,"[%d][computing flags] common searching space: %d, num subframe: %d\n", mac_inst->current_subframe, i, extend_space[i]);
		}
	}
	
	//USS trigger flag
	for(i=0;i<mac_inst->num_uss_list;++i)
	{
		if(((abs_subframe+1)%mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T)==mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.ss_start_uss)
		{
			*scheduler_flags |= (flag_uss_v<<i);
			max = MAX(max, mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T);
			LOG_D(MAC,"[%d][computing flags] UE-spec searching space: %d, num subframe: %d\n", mac_inst->current_subframe, i, mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T);
		}
	}
	
	*max_subframe = max;	//	the maximum subframe to be extend
}

/*function description:
* top level of the scheduler, this will trigger in every subframe,
* and determined if do the schedule by checking this current subframe is the start of the NPDCCH period or not
*/
void eNB_dlsch_ulsch_scheduler_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t abs_subframe){
	
	int i;
	uint8_t tx_mib=0, tx_sib1=0;
	uint32_t scheduler_flags, max_subframe, common_flags;
	/*Check this subframe should schedule something, set the flag*/
	scheduler_flags = 0;
	common_flags = 0;

	uint32_t h,f,sf;
	eNB_scheduler_computing_flag_NB_IoT(mac_inst, abs_subframe, &scheduler_flags, &common_flags, &max_subframe);

	if(scheduler_flags > 0){
	    extend_available_resource_DL(mac_inst, mac_inst->current_subframe +1 + max_subframe);
	}
	
	maintain_available_resource(mac_inst);

	if((abs_subframe % rachperiod[4]) == rachstart[0]){	//TODO, configuration should be pass by configuration module
		add_UL_Resource();
	}

	//Check if type2 searching space scheduling
	if((scheduler_flags&flag_css_type2)>0){
		schedule_RA_NB_IoT(mac_inst);
		scheduler_flags &= ~(flag_css_type2);
	}

	//Check if type1 searching space scheduling
	if((scheduler_flags&flag_css_type1)>0){
		//	paging, direct indication
		scheduler_flags &= ~(flag_css_type1);
 	}
 	//The scheduling time is current subframe + 1
	convert_system_number(abs_subframe+1, &h, &f, &sf);
	
	// loop all USS period
	for(i=0;i<mac_inst->num_uss_list;++i)
	{
		if((scheduler_flags&(flag_uss_v<<i))>0){
			LOG_D(MAC,"--------------[%04d][SchedulerUSS] Schedule USS list %d------------\n", mac_inst->current_subframe, (scheduler_flags&(flag_uss_v<<i))>>3);
			schedule_uss_NB_IoT(0, mac_inst,sf, f, h, i);
			LOG_D(MAC,"--------------[%04d][SchedulerUSS] Schedule USS list %d end------------\n", mac_inst->current_subframe, (scheduler_flags&(flag_uss_v<<i))>>3);
			scheduler_flags &= ~(flag_uss_v<<i);
		}
	}
	
	if(common_flags&flag_mib){
		tx_mib = 1;
	}
	if(common_flags&flag_sib1){
		tx_sib1 = 1;
	}

	convert_system_number(abs_subframe, &h, &f, &sf);

	if(0 != output_handler(mac_inst, (module_id_t)0, 0, h, f, sf, tx_mib, tx_sib1, abs_subframe)){
		LOG_D(MAC,"output handler error\n");
	}
}

void schedule_uss_NB_IoT(module_id_t module_id, eNB_MAC_INST_NB_IoT *mac_inst, uint32_t subframe, uint32_t frame, uint32_t hypersfn, int index_ss)
{

	//SCHEDULE_NB_IoT_t *scheduler =  &eNB->scheduler;
	mac_inst->scheduling_flag.flag_uss[0]=1;
	mac_inst->scheduling_flag.flag_uss[1]=0;
	mac_inst->scheduling_flag.flag_uss[2]=0;
	mac_inst->scheduling_flag.num_uss_run = 0;

	//search space index
	//int index_ss=0;
	int UE_ID;
	UE_TEMPLATE_NB_IoT *UE_template_temp;

	if(mac_inst->scheduling_flag.num_uss_run > 1)
	{
		//spectial case
	}
	else
	{
		//general case
	}

  	UE_ID = mac_inst->UE_list_spec[index_ss].head;
  	while(UE_ID>-1)
  	{
    
    	UE_template_temp = &(mac_inst->UE_list_spec[index_ss].UE_template_NB_IoT[UE_ID]);
    	LOG_D(MAC,"------Start Scheduling USS UE RNTI %d------\n", UE_template_temp->rnti);
    	if(UE_template_temp->RRC_connected!=1)
    	{
	  		LOG_D(MAC,"[%04d][USS scheduler][UE%d] rrc not connected\n", mac_inst->current_subframe, UE_template_temp->rnti);
    	}
    	else
    	{
    		LOG_D(MAC,"[%04d][USS scheduler][UE%d] ", mac_inst->current_subframe, UE_template_temp->rnti);
      
      		switch(UE_template_temp->direction)
      		{
		        case 1:		//	Downlink Scheduling
					LOG_D(MAC,"uss downlink scheduling.. \n");
          			schedule_DL_NB_IoT((module_id_t)0, mac_inst, UE_template_temp,  hypersfn, frame, subframe);
          			break;
        		case 0:		//	Uplink Scheduling
					LOG_D(MAC,"uss uplink scheduling.. \n");
          			schedule_UL_NB_IoT(mac_inst, UE_template_temp, subframe, frame, hypersfn);
          			break;
				case -1:	//	Idle
					LOG_D(MAC,"current idle.. \n");
				default:
					break;
      		}
    	}

    	UE_ID = UE_template_temp->next;
  	}
}

