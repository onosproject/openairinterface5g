
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


unsigned char str22[] = "UL_Data";
unsigned char str23[] = "DCI_N0";

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

	uint32_t type2_css_pp[3] = { 	mac_inst->npdcch_config_common[0].R_max*mac_inst->npdcch_config_common[0].G,
									mac_inst->npdcch_config_common[1].R_max*mac_inst->npdcch_config_common[1].G,
									mac_inst->npdcch_config_common[2].R_max*mac_inst->npdcch_config_common[2].G	};
	uint32_t start_subframe;
	for(i=0; i<1; ++i){	//	only CE0
		if(mac_inst->npdcch_config_common[i].a_offset==0)
		{
			start_subframe = 0;	
		}
		else if(mac_inst->npdcch_config_common[i].a_offset==1/8)
		{
			start_subframe = type2_css_pp[i]>>3;
		}
		else if(mac_inst->npdcch_config_common[i].a_offset==1/4)
		{
			start_subframe = type2_css_pp[i]>>2;
		}
		else if(mac_inst->npdcch_config_common[i].a_offset==3/8)
		{
			start_subframe = (type2_css_pp[i]>>3)+(type2_css_pp[i]>>2);
		}
	
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
	uint8_t MIB_flag = 0, SIB1_flag = 0;
	uint32_t scheduler_flags, max_subframe, common_flags;
	/*Check this subframe should schedule something, set the flag*/
	scheduler_flags = 0;
	common_flags = 0;
	uint32_t h,f,sf,a;
	mac_inst->current_subframe = abs_subframe;

        
	protocol_ctxt_t ctxt;
	convert_system_number(abs_subframe, &h, &f, &sf);
	//*************************RUN PDCP****************************
        PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, 0, ENB_FLAG_YES, NOT_A_RNTI, f, sf, 0);
        pdcp_run(&ctxt);
        //*************************************************************


	eNB_scheduler_computing_flag_NB_IoT(mac_inst, abs_subframe, &scheduler_flags, &common_flags, &max_subframe);

	if(scheduler_flags > 0){
	    extend_available_resource_DL(mac_inst, mac_inst->current_subframe +1 + max_subframe);
	}
	
	maintain_available_resource(mac_inst);

	if((abs_subframe % nprach_list->nprach_Periodicity) == rachstart[0]){	//TODO, configuration should be pass by configuration module
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
	
	if(common_flags ==  flag_mib)
		MIB_flag = 1;
	if(common_flags == flag_sib1)
		SIB1_flag = 1;

	convert_system_number(abs_subframe, &h, &f, &sf);
	a = output_handler(mac_inst, 0,0,h,f,sf,MIB_flag,SIB1_flag, abs_subframe);

	if(a==-1)
	LOG_I(MAC,"[%04d][SchedulerUSS] schedule result is empty------------\n", mac_inst->current_subframe);
}

void USS_scheduling_module(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t abs_subframe, uint8_t total_num_UE_list)
{
	int i, max_subframe,MIB_flag,SIB1_flag;

	/*Check this subframe should schedule something, set the flag*/
	MIB_flag = 0;
	SIB1_flag = 0;
	max_subframe=0;
	uint32_t h,f,sf;
	int a;
	int UE_list_index;
	// how many scheduling is triggered this sunframe
	uint8_t num_sched_UE_list=0;
	uint8_t *UE_list_flag=(uint8_t*)malloc(total_num_UE_list*sizeof(uint8_t));
	//DEBUG("[%04d][USS_scheduling_module] check scheduling trigger\n", mac_inst->current_subframe);
	//eNB_scheduler_computing_flag_NB_IoT(mac_inst, abs_subframe, &scheduler_flags, &common_flags, &max_subframe);
	// Check which scheduling period of UE_list is triggered
	for(i=0;i<total_num_UE_list;++i)
	{
		if(((abs_subframe+1)%mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T)==mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.ss_start_uss)
		//if((abs_subframe+1)%16==0)
		{
			if(mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T>max_subframe)
			{
				max_subframe=mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T;
			}
			UE_list_flag[i]=1;
			num_sched_UE_list++;
			//*scheduler_flags |= (flag_uss_v<<i);
			//max = MAX(max, mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T);
			LOG_D(MAC,"[%d][USS_scheduling_module] UE_list num: %d, num subframe: %d\n", mac_inst->current_subframe, i, mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T);
		}
		else
		{
			UE_list_flag[i]=0;
		}
	}
	
	// Update available resource
	if(num_sched_UE_list > 0)
	{
		LOG_D(MAC,"[%d][USS_scheduling_module] extend resource\n", mac_inst->current_subframe);
		//DEBUG("[%04d][USS_scheduling_module] In extend_available_resource_DL\n", mac_inst->current_subframe);
	    extend_available_resource_DL(mac_inst, mac_inst->current_subframe +1 + max_subframe);
	}
	maintain_available_resource(mac_inst);
	// reserve resource for NPRACH
	if((abs_subframe % nprach_list->nprach_Periodicity) == rachstart[0])
	{
		//DEBUG("[%04d][USS_scheduling_module] In add_UL_Resource\n", mac_inst->current_subframe);
		add_UL_Resource();
	}

 	//The scheduling time is current subframe + 1
	convert_system_number(abs_subframe+1, &h, &f, &sf);
	// loop all USS period
	for(UE_list_index=0;UE_list_index<total_num_UE_list;++UE_list_index)
	{
		//if((scheduler_flags&(flag_uss_v<<i))>0){
		if(UE_list_flag[UE_list_index]==1)
		{
			LOG_D(MAC,"--------------[%04d][USS_scheduling_module] Schedule USS list %d------------\n", mac_inst->current_subframe, UE_list_index);
			//USS Scheduling for corresponding index
			schedule_uss_NB_IoT(0, mac_inst,sf, f, h, UE_list_index);
			LOG_D(MAC,"--------------[%04d][USS_scheduling_module] Schedule USS list %d end------------\n", mac_inst->current_subframe, UE_list_index);
			//scheduler_flags &= ~(flag_uss_v<<i);
		}
	}

	// flag for generating SIB1 and MIB message
	if(abs_subframe%10 == 0)
		MIB_flag = 1;
	if(abs_subframe%10 == 4 && mac_inst->sib1_flag[abs_subframe/10%mac_inst->sib1_period])
		SIB1_flag = 1;
	// handling output to L1
	a = output_handler(mac_inst, 0,0,h,f,sf,MIB_flag,SIB1_flag, abs_subframe);
	if(a==-1)
		LOG_D(MAC,"[%04d][USS_scheduling_module] schedule result is empty------------\n", mac_inst->current_subframe);
}

void schedule_uss_NB_IoT(module_id_t module_id, eNB_MAC_INST_NB_IoT *mac_inst, uint32_t subframe, uint32_t frame, uint32_t hypersfn, int UE_list_index)
{
	UE_SCHED_CTRL_NB_IoT_t *UE_sched_ctrl_info;
	UE_TEMPLATE_NB_IoT *UE_template_temp;
	DCIFormatN1_t *DCI_N1;
	DCIFormatN0_t *DCI_N0;

	//SCHEDULE_NB_IoT_t *scheduler =  &eNB->scheduler;
	mac_inst->scheduling_flag.flag_uss[0]=1;
	mac_inst->scheduling_flag.flag_uss[1]=0;
	mac_inst->scheduling_flag.flag_uss[2]=0;
	mac_inst->scheduling_flag.num_uss_run = 0;
	int UE_ID;

	//search space index
	//int index_ss=0;
	LOG_D(MAC,"[%04d][schedule_uss_NB_IoT] Start processing preprocessor\n", mac_inst->current_subframe);
	/***algorithm for USS scheduling***/
	preprocessor_uss_NB_IoT(module_id, mac_inst, subframe, frame, hypersfn, UE_list_index);

	LOG_D(MAC,"[%04d][schedule_uss_NB_IoT] Finish processing preprocessor\n", mac_inst->current_subframe);
	LOG_D(MAC,"[%04d][schedule_uss_NB_IoT] Do USS Final Scheduling\n", mac_inst->current_subframe);
	UE_ID = mac_inst->UE_list_spec[UE_list_index].head;
  	while(UE_ID>-1)
  	{
    	UE_template_temp = &(mac_inst->UE_list_spec[UE_list_index].UE_template_NB_IoT[UE_ID]);
   		UE_sched_ctrl_info = &(mac_inst->UE_list_spec[UE_list_index].UE_sched_ctrl_NB_IoT[UE_ID]);
    	LOG_D(MAC,"------Start Scheduling USS UE RNTI %d------\n", UE_template_temp->rnti);
		if((UE_template_temp->RRC_connected==1)&&(UE_sched_ctrl_info->flag_schedule_success==1))
    	{
	  
    		switch(UE_template_temp->direction)
      		{
		        case 1:		//	Downlink Scheduling
		        	LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] USS DL Final scheduling\n", mac_inst->current_subframe, UE_template_temp->rnti);
		        	LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] UE_sched_ctrl NPDCCH information:sf_start %d sf end %d\n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->NPDCCH_sf_start, UE_sched_ctrl_info->NPDCCH_sf_end);
					LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] UE_sched_ctrl NPDSCH information:sf_start %d sf end %d\n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->NPDSCH_sf_start, UE_sched_ctrl_info->NPDSCH_sf_end);
					LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] UE_sched_ctrl HARQ information:sf_start %d sf end %d\n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->HARQ_sf_start, UE_sched_ctrl_info->HARQ_sf_end);
	
		        	DCI_N1 = (DCIFormatN1_t*)malloc(sizeof(DCIFormatN1_t));
		        	fill_DCI_N1(DCI_N1, UE_template_temp, UE_sched_ctrl_info);
	  				generate_scheduling_result_DL(UE_sched_ctrl_info->NPDCCH_sf_end, UE_sched_ctrl_info->NPDCCH_sf_start, UE_sched_ctrl_info->NPDSCH_sf_end, UE_sched_ctrl_info->NPDSCH_sf_start, UE_sched_ctrl_info->HARQ_sf_end, UE_sched_ctrl_info->HARQ_sf_start, DCI_N1, UE_template_temp->rnti, UE_sched_ctrl_info->TBS, UE_template_temp->DLSCH_pdu.payload);
					UE_template_temp->R_dci=UE_sched_ctrl_info->R_dci;
	      			UE_template_temp->R_dl=UE_sched_ctrl_info->R_dl_data;
	      			UE_template_temp->I_mcs_dl=UE_sched_ctrl_info->dci_n1_index_mcs;
	      			UE_template_temp->DLSCH_pdu_size=UE_sched_ctrl_info->TBS;
	      			//if(UE_template_temp->HARQ_round==0)
	      				//UE_template_temp->oldNDI_DL=(UE_template_temp->oldNDI_DL+1)%2;
	      			UE_template_temp->direction = 3;
	      			break;
	    		case 0:		//	Uplink
	    			LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] USS UL Final scheduling\n", mac_inst->current_subframe, UE_template_temp->rnti);
	    			LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] UE_sched_ctrl NPDCCH information:sf_start %d sf end %d\n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->NPDCCH_sf_start, UE_sched_ctrl_info->NPDCCH_sf_end);
					LOG_D(MAC,"[%04d][schedule_uss_NB_IoT][UE%d] UE_sched_ctrl NPUSCH information:sf_start %d sf end %d\n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->NPUSCH_sf_start, UE_sched_ctrl_info->NPUSCH_sf_end);
	    			DCI_N0 = (DCIFormatN0_t*)malloc(sizeof(DCIFormatN0_t));
	    			//generate DCI-N0 content
                    fill_DCI_N0(DCI_N0, UE_template_temp, UE_sched_ctrl_info);
	    			generate_scheduling_result_UL(UE_sched_ctrl_info->NPDCCH_sf_start, UE_sched_ctrl_info->NPDCCH_sf_end,UE_sched_ctrl_info->NPUSCH_sf_start+3, UE_sched_ctrl_info->NPUSCH_sf_end+3,DCI_N0, UE_template_temp->rnti, str22, str23, 0);

	      			//sotre UE_template
	      			UE_template_temp->R_dci=UE_sched_ctrl_info->R_dci;
	      			UE_template_temp->R_ul=UE_sched_ctrl_info->R_ul_data;

      			    if(UE_template_temp->HARQ_round == 0)
				    {
				        UE_template_temp->oldNDI_UL=1-UE_template_temp->oldNDI_UL;
				    }

				    UE_template_temp->direction = -1;
	      			break;
	      		case 3:
	      			LOG_D(MAC,"This UE is already scheduled, wait for the response\n");
	      			break;
				case -1:	//	Idle
					//DEBUG("current idle.. \n");
					break;
				default:
					break;

      		}
    	}
		UE_sched_ctrl_info -> flag_schedule_success = 0;
		UE_ID = UE_template_temp->next;
	}
}


void preprocessor_uss_NB_IoT(module_id_t module_id, eNB_MAC_INST_NB_IoT *mac_inst, uint32_t subframe, uint32_t frame, uint32_t hypersfn, int UE_list_index)
{
	
	int ue_id;
	UE_TEMPLATE_NB_IoT *UE_template_temp;
	UE_SCHED_CTRL_NB_IoT_t *UE_sched_ctrl_info;
	

	ue_id = mac_inst->UE_list_spec[UE_list_index].head;
	while(ue_id>-1)
  	{
    	UE_template_temp = &(mac_inst->UE_list_spec[UE_list_index].UE_template_NB_IoT[ue_id]);
    	UE_sched_ctrl_info = &(mac_inst->UE_list_spec[UE_list_index].UE_sched_ctrl_NB_IoT[ue_id]);
		
		//determine index of MCS, TBS, R, R_max, R_dci, R_harq
		UE_sched_ctrl_info->R_dci=UE_template_temp->R_dci;
    	//Set repetition number of downlink transmission
		if(UE_template_temp->direction==1)
		{
					//UE_sched_ctrl_info->R_dci=UE_template_temp->R_dci;

			UE_sched_ctrl_info->R_dl_data=UE_template_temp->R_dl;
			UE_sched_ctrl_info->R_dl_harq=UE_template_temp->R_harq;
			UE_sched_ctrl_info->dci_n1_index_mcs=UE_template_temp->I_mcs_dl;
			LOG_N(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] Initialze R_dci %d R_data_dl %d R_harq %d \n", mac_inst->current_subframe, UE_template_temp->rnti, UE_sched_ctrl_info->R_dci=UE_template_temp->R_dci, UE_sched_ctrl_info->R_dl_data, UE_sched_ctrl_info->R_dl_harq=UE_template_temp->R_harq);
			//determine how many SF for data transmission
			//store_rlc_logical_channel_info_dl();
		}
		//Set repetition number of UL transmission
		else
		{
			UE_sched_ctrl_info->R_ul_data=UE_template_temp->R_ul;
		}
		ue_id = UE_template_temp->next;
	}
  	
  	//sort all UE regardless DL or UL UEs
	sort_UEs_uss();

	ue_id = mac_inst->UE_list_spec[UE_list_index].head;
	
	//Resource scheduling algorithm
	while(ue_id>-1)
  	{
    	UE_template_temp = &(mac_inst->UE_list_spec[UE_list_index].UE_template_NB_IoT[ue_id]);
    	UE_sched_ctrl_info = &(mac_inst->UE_list_spec[UE_list_index].UE_sched_ctrl_NB_IoT[ue_id]);
    	// UE not finish RA or finish transmission
    	if(UE_template_temp->RRC_connected!=1)
    	{
			LOG_D(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] rrc not connected\n", mac_inst->current_subframe, UE_template_temp->rnti);
		}

		// Finish RA
    	else
		{
    		//DEBUG("[%04d][preprocessor_uss_NB_IoT][UE%d] ", mac_inst->current_subframe, UE_template_temp->rnti);
    		//DEBUG("[%04d][preprocessor_uss_NB_IoT][UE%d] Start scheduling\n", mac_inst->current_subframe, UE_template_temp->rnti);
			switch(UE_template_temp->direction)
      		{
		        case 1:		//	Downlink resource allocation algorithm
					LOG_D(MAC,"uss downlink scheduling.. \n");
					//schedule_DL_NB_IoT(0, mac_inst, UE_template_temp, hypersfn, frame, subframe);
					if(0==schedule_DL_NB_IoT(0, mac_inst, UE_template_temp, hypersfn, frame, subframe, UE_sched_ctrl_info))
					{
						LOG_N(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] DL scheduling USS is successful\n", mac_inst->current_subframe, UE_template_temp->rnti);
						UE_sched_ctrl_info->flag_schedule_success=1;
					}
					else
					{
						LOG_D(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] DL scheduling USS is failed\n", mac_inst->current_subframe, UE_template_temp->rnti);
					}
					break;

        		case 0:		//	Uplink resource allocation algorithm
					LOG_D(MAC,"uss uplink scheduling.. \n");
					if(0==schedule_UL_NB_IoT(mac_inst, UE_template_temp, subframe, frame, hypersfn, UE_sched_ctrl_info))
					{
						LOG_D(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] UL scheduling USS is successful\n", mac_inst->current_subframe, UE_template_temp->rnti);
						UE_sched_ctrl_info->flag_schedule_success=1;
					}
					else
					{
						LOG_D(MAC,"[%04d][preprocessor_uss_NB_IoT][UE%d] UL scheduling USS is failed\n", mac_inst->current_subframe, UE_template_temp->rnti);
					}
					break;
          			//schedule_UL_NB_IoT(mac_inst, UE_template_temp, subframe, frame, hypersfn);
          			break;
				case -1:	//	Idle state, no data wait to send
					//sDEBUG("current idle.. \n");
				case 3:
	      			LOG_D(MAC,"This UE is already scheduled, wait for the response\n");
	      			break;
				default:
					break;
      		}
    	}
    	ue_id = UE_template_temp->next;
  	}
}

void sort_UEs_uss()
{
	//loop all UE
}
