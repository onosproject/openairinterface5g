
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


#define num_flags 2
int extend_space[num_flags] = {160, 160};
int extend_alpha_offset[num_flags] = {10, 10};

int uss_space = 320;
int uss_alpha_offset = 10;

void eNB_scheduler_computing_flag_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t abs_subframe, int *scheduler_flags, int *common_flags){
	uint32_t subframe = abs_subframe % 10;
	uint32_t frame = abs_subframe / 10;
	int i;

	if(subframe == 0){
		*common_flags |= flag_mib;
	}else if(subframe == 5){
		*common_flags |= flag_npss;
	}else if(subframe == 9 && (frame&0x1)==0){
		*common_flags |= flag_nsss;
	}else if(subframe == 4 && mac_inst->sib1_flag[frame%mac_inst->sib1_period]){
		*common_flags |= flag_sib1;
	}
	for(i=0; i<num_flags; ++i){
		if(((abs_subframe+1)%extend_space[i])==(extend_space[i]>>extend_alpha_offset[i])){
			*scheduler_flags |= (0x1<<i);
		}
	}
	//USS trigger flag
	for(i=0;i<mac_inst->num_uss_list;++i)
	{
		//printf("[eNB Computing falg] USS trigger time %d ss start time %d\n", ((abs_subframe+1)%mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T), mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.ss_start_uss);
		if(((abs_subframe+1)%mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.T)==mac_inst->UE_list_spec[i].NPDCCH_config_dedicated.ss_start_uss)
		{
			//SCHEDULE_LOG("1\n");
			*scheduler_flags |= (flag_uss_v<<i);
		}
	}
	//printf("[eNB Computing falg] scheduler_flags %X\n", *scheduler_flags);
	/*
	//USS trigger flag
	if((abs_subframe%uss_space)==(uss_space>>uss_alpha_offset)){
		*scheduler_flags |= flag_uss_v;
	}
	*/
}
/*function description:
* top level of the scheduler, this will trigger in every subframe,
* and determined if do the schedule by checking this current subframe is the start of the NPDCCH period or not
*/
void eNB_dlsch_ulsch_scheduler_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t abs_subframe){
 // eNB_MAC_INST_NB_IoT *eNB = &eNB_mac_inst_NB_IoT[module_id];
	int i, max_subframe, scheduler_flags, common_flags,MIB_flag,SIB1_flag;
	int a = 0;
	/*Check this subframe should schedule something, set the flag*/
	scheduler_flags = 0;
	common_flags = 0;
	MIB_flag = 0;
	SIB1_flag = 0;
	uint32_t h,f,sf;
	//int a;
	//DEBUG("--------------[%04d][eNB scheduler NB-IoT] Start Scheduling------------\n", mac_inst->current_subframe);
	eNB_scheduler_computing_flag_NB_IoT(mac_inst, abs_subframe, &scheduler_flags, &common_flags);
	/*Update the available resource list to current state*/
	//NB_IoT_maintain_available_resource(subframe, frame, hypersfn);
	max_subframe = 0;
	for(i=0; i<num_flags; ++i){
		if(1 == (scheduler_flags&(0x1<<i))){
			if(max_subframe < extend_space[i]){
				max_subframe = extend_space[i];
			}
		}
	}

	if(scheduler_flags > 0){
	        extend_available_resource_DL(mac_inst, mac_inst->current_subframe + 1 + max_subframe);
	}
	
	maintain_available_resource(mac_inst);

    //static int test=2;
	if((abs_subframe % rachperiod[4]) == rachstart[0]){

		   add_UL_Resource();
	}

	//Check if type2 searching space scheduling
	if((scheduler_flags&flag_css_type2)>0){
		schedule_RA_NB_IoT(mac_inst);
		scheduler_flags &= ~(flag_css_type2);
	}

	//Check if type1 searching space scheduling
	if((scheduler_flags&flag_css_type1)>0){
		scheduler_flags &= ~(flag_css_type1);
 	}
	
	// loop all USS period
	for(i=0;i<mac_inst->num_uss_list;++i)
	{
		if((scheduler_flags&(flag_uss_v<<i))>0){
			printf("--------------[%04d][SchedulerUSS] Schedule USS list %d------------\n", mac_inst->current_subframe, (scheduler_flags&(flag_uss_v<<i))>>3);
			scheduler_flags &= ~(flag_uss_v<<i);
			convert_system_number(abs_subframe, &h, &f, &sf);
			//DEBUG("=====t:%d======\n", mac_inst->current_subframe);
			//print_available_resource_DL();
			schedule_uss_NB_IoT(0, mac_inst,sf, f, h, i);
		}
	}
	/*
	//Check if UE-specific searching space scheduling
	if((scheduler_flags&flag_uss_v)>0){
		scheduler_flags &= ~(flag_uss_v);
		convert_system_number(abs_subframe, &h, &f, &sf);
		//DEBUG("=====t:%d======\n", mac_inst->current_subframe);
		//print_available_resource_DL();
		schedule_uss_NB_IoT(0, mac_inst,sf, f, h, 0);
	}
	*/
	if(common_flags ==  flag_mib)
		MIB_flag = 1;
	if(common_flags == flag_sib1)
		SIB1_flag = 1;
	convert_system_number(abs_subframe, &h, &f, &sf);
	a = output_handler(mac_inst, 0,0,h,f,sf,MIB_flag,SIB1_flag, abs_subframe);

	printf("Output_handler_return value : %d", a);
	//DEBUG("--------------[%04d][eNB scheduler NB-IoT] End Scheduling------------\n", mac_inst->current_subframe);
}

void schedule_uss_NB_IoT(module_id_t module_id, eNB_MAC_INST_NB_IoT *mac_inst, uint32_t subframe, uint32_t frame, uint32_t hypersfn, int index_ss)
{
	//int32_t i;
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
  //DEBUG("t=%d   UE ID head %d\n", mac_inst->current_subframe,UE_ID);
  while(UE_ID>-1)
  {
    
    UE_template_temp = &(mac_inst->UE_list_spec[index_ss].UE_template_NB_IoT[UE_ID]);
    printf("------Start Scheduling USS UE RNTI %d------\n", UE_template_temp->rnti);
    if(UE_template_temp->RRC_connected!=1)
    {
    	printf("[schedule_uss_NB_IoT] UE ID %d RRC not connected\n", UE_ID);
	  	printf("[%04d][USS scheduler][UE%d] rrc not connected\n", mac_inst->current_subframe, UE_template_temp->rnti);
    }
    else
    {
    	printf("t=%d*******[schedule_uss_NB_IoT] schedule UE_ID %d direction %d*******%d %d %d\n", mac_inst->current_subframe, UE_ID, UE_template_temp->direction, hypersfn, frame, subframe);
		printf("[%04d][USS scheduler][UE%d] ", mac_inst->current_subframe, UE_template_temp->rnti);
      switch(UE_template_temp->direction)
      {
        //Downlink Scheduling
        case 1:
			printf("uss downlink scheduling.. \n");
          schedule_DL_NB_IoT(0, mac_inst, UE_template_temp,  hypersfn, frame, subframe);
          break;
        //Uplink Scheduling
        case 0:
			printf("uss uplink scheduling.. \n");
          schedule_UL_NB_IoT(mac_inst, UE_template_temp, subframe, frame, hypersfn);
          break;
		case -1:
			printf("current idle.. \n");
		default:
			break;
      }
      //printf("----------------End Scheduling USS UE RNTI %d-------------------\n", UE_template_temp->rnti);
      UE_template_temp->direction = -1;
    }

    UE_ID = UE_template_temp->next;
  }

}

