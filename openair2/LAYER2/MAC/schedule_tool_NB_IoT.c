
/*! \file schedule_tool_NB_IoT.c
 * \brief scheduler helper function
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"

void init_tool_sib1(eNB_MAC_INST_NB_IoT *mac_inst){
	int i, j;

	//int repetition_pattern = 1;//	1:every2frame, 2:every4frame, 3:every8frame, 4:every16frame
	for(i=0;i<8;++i){
		mac_inst->sib1_flag[(i<<1)+mac_inst->rrc_config.sib1_NB_IoT_sched_config.starting_rf] = 1;
	}

	for(i=0, j=0;i<64;++i){
		if(mac_inst->sib1_flag[i]==1){
			++j;
		}
		mac_inst->sib1_count[i]=j;
	}

	mac_inst->sib1_period = 256 / mac_inst->rrc_config.sib1_NB_IoT_sched_config.repetitions;

	return ;
}

void init_dlsf_info(eNB_MAC_INST_NB_IoT *mac_inst, DLSF_INFO_t *DLSF_info)
{
  uint16_t dlsf_num_temp=0;
  uint16_t i;
  uint16_t j=0;

  DLSF_info->sf_to_dlsf_table=(uint16_t*)malloc(mac_inst->sib1_period*10*sizeof(uint16_t));
  for(i=0;i<mac_inst->sib1_period*10;++i)
  {
    if(is_dlsf(mac_inst, i)==1)
    {
      dlsf_num_temp++;
      DLSF_info->sf_to_dlsf_table[i]=dlsf_num_temp;
    }
    else
    {
      DLSF_info->sf_to_dlsf_table[i]=dlsf_num_temp;
    }
  }
  DLSF_info->num_dlsf_per_period = dlsf_num_temp;
  DLSF_info->dlsf_to_sf_table = (uint16_t*)malloc(dlsf_num_temp*sizeof(uint16_t));
  for(i=0;i<mac_inst->sib1_period*10;++i)
  {
    if(is_dlsf(mac_inst, i)==1)
    {
      DLSF_info->dlsf_to_sf_table[j]= i;
      j++;
    }
  }
}

int is_dlsf(eNB_MAC_INST_NB_IoT *mac_inst, int abs_subframe){
	int frame = abs_subframe/10;
	int subframe = abs_subframe%10;

	return !(subframe==0||subframe==5||((frame&0x1)==0&&subframe==9)||(mac_inst->sib1_flag[frame%mac_inst->sib1_period]==1&&subframe==4));
}

void init_dl_list(eNB_MAC_INST_NB_IoT *mac_inst){
	available_resource_DL_t *node;

	node = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
	node->next = (available_resource_DL_t *)0;
	node->prev = (available_resource_DL_t *)0;

	available_resource_DL = node;
	available_resource_DL_last = node;

	node->start_subframe = 0;
	node->end_subframe = mac_inst->rrc_config.si_window_length;
	node->DLSF_num = calculate_DLSF(mac_inst, node->start_subframe, node->end_subframe);
	mac_inst->schedule_subframe_DL = mac_inst->rrc_config.si_window_length;

	//	init sibs for first si-window
	schedule_sibs_NB_IoT(mac_inst, 0, 0);
}

void setting_nprach(void){

	nprach_list[0].nprach_Periodicity = rachperiod[4];
	nprach_list[0].nprach_StartTime = rachstart[0];
	nprach_list[0].nprach_SubcarrierOffset = rachscofst[0];
	nprach_list[0].nprach_NumSubcarriers = rachnumsc[0];
	nprach_list[0].numRepetitionsPerPreambleAttempt = rachrepeat[1];

	nprach_list[1].nprach_Periodicity = rachperiod[4];
	nprach_list[1].nprach_StartTime = rachstart[0];
	nprach_list[1].nprach_SubcarrierOffset = rachscofst[1];
	nprach_list[1].nprach_NumSubcarriers = rachnumsc[0];
	nprach_list[1].numRepetitionsPerPreambleAttempt = rachrepeat[3];

	nprach_list[2].nprach_Periodicity = rachperiod[4];
	nprach_list[2].nprach_StartTime = rachstart[0];
	nprach_list[2].nprach_SubcarrierOffset = rachscofst[2];
	nprach_list[2].nprach_NumSubcarriers = rachnumsc[1];
	nprach_list[2].numRepetitionsPerPreambleAttempt = rachrepeat[5];

	// fixed nprach configuration
}

void add_UL_Resource_node(available_resource_UL_t **head, uint32_t *end_subframe, uint32_t ce_level){
	available_resource_UL_t *new_node, *iterator;
    new_node = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));
	
	new_node->next = (available_resource_UL_t *)0;
	
	new_node->start_subframe = *end_subframe + ceil( (nprach_list+ce_level)->nprach_StartTime + 1.4*4*((nprach_list+ce_level)->numRepetitionsPerPreambleAttempt) ) ;
	
    new_node->end_subframe = *end_subframe + (nprach_list+ce_level)->nprach_Periodicity - 1;
	
	if( (available_resource_UL_t *)0 == *head){
		*head = new_node;
		new_node->prev = (available_resource_UL_t *)0;
	}else{
		iterator = *head;
		while( (available_resource_UL_t *)0 != iterator->next){
			iterator = iterator->next;
		}
		iterator->next = new_node;
		new_node->prev = iterator;
	}
	

    *end_subframe += (nprach_list+ce_level)->nprach_Periodicity;
}

/// Use to extend the UL resource grid (5 list) at the end of nprach peroid time
/// void add_UL_Resource(eNB_MAC_INST_NB_IoT *mac_inst)
void add_UL_Resource(void)
{
	
	add_UL_Resource_node(&available_resource_UL->sixtone_Head, &available_resource_UL->sixtone_end_subframe, 2);
	add_UL_Resource_node(&available_resource_UL->threetone_Head, &available_resource_UL->threetone_end_subframe, 1);
	add_UL_Resource_node(&available_resource_UL->singletone1_Head, &available_resource_UL->singletone1_end_subframe, 0);
	add_UL_Resource_node(&available_resource_UL->singletone2_Head, &available_resource_UL->singletone2_end_subframe, 0);
	add_UL_Resource_node(&available_resource_UL->singletone3_Head, &available_resource_UL->singletone3_end_subframe, 0);
	
	//print_available_UL_resource(); getchar();
}

/*when there is SIB-2 configuration coming to MAC, filled the uplink resource grid*/
void Initialize_Resource(void){

    ///memory allocate to Head
    available_resource_UL = (available_resource_tones_UL_t*)malloc(sizeof(available_resource_tones_UL_t));

    available_resource_UL->sixtone_Head = (available_resource_UL_t *)0;
    available_resource_UL->threetone_Head = (available_resource_UL_t *)0;
    available_resource_UL->singletone1_Head = (available_resource_UL_t *)0;
    available_resource_UL->singletone2_Head = (available_resource_UL_t *)0;
    available_resource_UL->singletone3_Head = (available_resource_UL_t *)0;

	available_resource_UL->sixtone_end_subframe = 0;
	available_resource_UL->threetone_end_subframe = 0;
	available_resource_UL->singletone1_end_subframe = 0;
	available_resource_UL->singletone2_end_subframe = 0;
	available_resource_UL->singletone3_end_subframe = 0;
	
	add_UL_Resource();
	add_UL_Resource();
	
    printf("Initialization of the UL Resource grid has been done\n");
}

//	extend subframe align to si-period
void extend_available_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, int max_subframe){	//	assume max_subframe is found.
    
	available_resource_DL_t *new_node;
	//int temp;
	uint32_t i, i_div_si_window;
	//uint32_t si_period_div_window;
    
    printf("%d %d\n", max_subframe, mac_inst->schedule_subframe_DL);
    
	if(max_subframe > mac_inst->schedule_subframe_DL){
		//	align to si-period

		max_subframe = ((max_subframe%mac_inst->rrc_config.si_window_length)==0)? max_subframe : (((max_subframe/mac_inst->rrc_config.si_window_length)+1)*mac_inst->rrc_config.si_window_length);
		printf("max %d last->end %p\n", max_subframe, available_resource_DL_last);
		if(mac_inst->schedule_subframe_DL == available_resource_DL_last->end_subframe){
			available_resource_DL_last->end_subframe = max_subframe;
			//available_resource_DL_last->DLSF_num += calculate_DLSF(mac_inst, mac_inst->schedule_subframe_DL+1, max_subframe);
		}else{
			new_node = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));

			available_resource_DL_last->next = new_node;
			new_node->start_subframe = mac_inst->schedule_subframe_DL+1;
			new_node->end_subframe = max_subframe;
			new_node->next = (available_resource_DL_t *)0;
			//new_node->DLSF_num = calculate_DLSF(mac_inst, mac_inst->schedule_subframe_DL+1, max_subframe);
		}

		//	do schedule sibs after extend.
		for(i=mac_inst->schedule_subframe_DL;i<max_subframe;i+=mac_inst->rrc_config.si_window_length){
			i_div_si_window = i / mac_inst->rrc_config.si_window_length;
			if(-1 != mac_inst->sibs_table[i_div_si_window]){
				printf("[sibs%d] %d\n", mac_inst->sibs_table[i_div_si_window], i);
				schedule_sibs_NB_IoT(mac_inst, mac_inst->sibs_table[i_div_si_window], i);
			}
		}

		mac_inst->schedule_subframe_DL = max_subframe;
	}
	return ;
}


void fill_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, available_resource_DL_t *node, int start_subframe, int end_subframe, schedule_result_t *new_node){
	available_resource_DL_t *temp;
	schedule_result_t *iterator, *temp1;
	//	divided into two node
	//	keep one node(align left or right)
	//	delete node

	int align_left = (node->start_subframe==start_subframe)||(calculate_DLSF(mac_inst, node->start_subframe, start_subframe-1) == 0);
	int align_right = (end_subframe==node->end_subframe)||(calculate_DLSF(mac_inst, end_subframe+1, node->end_subframe) == 0);
	//print_available_resource_DL();
	//DEBUG("[debug] align : %d %d\n", align_left, align_right);
	switch(align_left+align_right){
		case 0:
			//	divided into two node
			//	A | node | B
			//	A | temp | node | B
			temp = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
			if(node->prev){
				node->prev->next = temp;
			}else{
				available_resource_DL = temp;
			}
			temp->prev = node->prev;
			temp->next = node;
			node->prev = temp;
			//node->next don't need to change

			temp->start_subframe = node->start_subframe;
			temp->end_subframe = start_subframe - 1;

			node->start_subframe = end_subframe + 1;

			node->DLSF_num = calculate_DLSF(mac_inst, node->start_subframe, node->end_subframe);
			temp->DLSF_num = calculate_DLSF(mac_inst, temp->start_subframe, temp->end_subframe);
			break;
		case 1:
			//	keep one node
			if(align_left){
				node->start_subframe = end_subframe + 1 ;
			}else{
				node->end_subframe = start_subframe - 1 ;
			}

			node->DLSF_num = calculate_DLSF(mac_inst, node->start_subframe, node->end_subframe);
			break;
		case 2:
			//	delete
			if(node->next){
			    node->next->prev = node->prev;
            }else{
                available_resource_DL_last = node->prev;
            }
				
			if(node->prev){
				node->prev->next = node->next;
			}else{
				available_resource_DL = node->next;
			}

			free(node);
			break;
		default:
			//error
			break;
	}
	
	//	new node allocate from up-layer calling function.
	iterator = schedule_result_list_DL;
    temp1 = (schedule_result_t *)0;
	if((schedule_result_t *)0 == schedule_result_list_DL){
		schedule_result_list_DL = new_node;
	}else{
		while((schedule_result_t *)0 != iterator){
			if(start_subframe < iterator->output_subframe){
				break;
			}
			temp1 = iterator;
			iterator = iterator->next;
		}
		if((schedule_result_t *)0 == iterator){
			temp1->next = new_node;
		}else{
			new_node->next = iterator;
			if(temp1){
				temp1->next = new_node;
			}else{
				schedule_result_list_DL = new_node;
			}
		}
	}
}

available_resource_DL_t *check_sibs_resource(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t check_start_subframe, uint32_t check_end_subframe, uint32_t num_subframe, uint32_t *residual_subframe, uint32_t *out_last_subframe, uint32_t *out_first_subframe){
	available_resource_DL_t *pt;
	uint32_t num_dlsf;
	uint8_t output = 0x0;
	pt = available_resource_DL;
	//	TODO find the pt which can cover part of check_start_subframe, e.g. 1280-> 1281-1440
	while((available_resource_DL_t *)0 != pt){
		if(pt->start_subframe <= check_start_subframe && pt->end_subframe >= check_start_subframe){
			break;
		}
		pt = pt->next;
	}
	//print_available_resource_DL();
	//DEBUG("sibs %d", check_start_subframe);
	if((available_resource_DL_t *)0 == pt){
		return (available_resource_DL_t *)0;
	}
	
	num_dlsf = calculate_DLSF(mac_inst, check_start_subframe, pt->end_subframe);

	if((available_resource_DL_t *)0 == pt){
		return (available_resource_DL_t *)0;
	}else{

		if(num_subframe <= num_dlsf){

			while(num_subframe>0){
				if(is_dlsf(mac_inst, check_start_subframe)){
					--num_subframe;
					if(output == 0x0){
						*out_first_subframe = check_start_subframe;
						output = 0x1;
					}
				}
				if(num_subframe==0||check_start_subframe>=check_end_subframe){
					break;
				}else{
					++check_start_subframe;
				}
			}

			*residual_subframe = num_subframe;
			*out_last_subframe = check_start_subframe;

		}else{
			if(num_dlsf == 0){
				return (available_resource_DL_t *)0;
			}else{
				while(!is_dlsf(mac_inst, check_start_subframe)){
					++check_start_subframe;
				}
				*out_first_subframe = check_start_subframe;
			}
			*residual_subframe = num_subframe - num_dlsf;
			*out_last_subframe = pt->end_subframe;
		}
		return pt;
	}
}


uint32_t calculate_DLSF(eNB_MAC_INST_NB_IoT *mac_inst, int abs_start_subframe, int abs_end_subframe){
	int i;
	int num_dlsf=0;
	//int diff_subframe = abs_end_subframe - abs_start_subframe;

	int start_frame = abs_start_subframe / 10;
	int end_frame = abs_end_subframe / 10;
	int start_subframe = abs_start_subframe % 10;
	int end_subframe = abs_end_subframe % 10;

	int start_frame_mod_64 = start_frame & 0x0000003f;
	int end_frame_mod_64 = end_frame & 0x0000003f;
	int start_frame_div_64 = (start_frame & 0xffffffc0)>>6;
	int end_frame_div_64 = (end_frame & 0xffffffc0)>>6;

	if(start_frame > end_frame){
		return calculate_DLSF(mac_inst, abs_start_subframe, MAX_FRAME*10+9) + calculate_DLSF(mac_inst, 0, abs_end_subframe);
	}
	if(start_frame_div_64==end_frame_div_64 && start_frame==end_frame){
	    for(i=abs_start_subframe;i<=abs_end_subframe;++i){
			num_dlsf += is_dlsf(mac_inst, i);
		}
	}else{
	    num_dlsf = mac_inst->dlsf_table[end_frame_mod_64];
		num_dlsf -= (start_frame_mod_64==0)?0:mac_inst->dlsf_table[start_frame_mod_64-1];
		for(i=0;i<start_subframe;++i, --abs_start_subframe){
			num_dlsf -= is_dlsf(mac_inst, abs_start_subframe-1);
		}
		for(i=end_subframe;i<9;++i, ++abs_end_subframe){
			num_dlsf -= is_dlsf(mac_inst, abs_end_subframe+1);
		}
	    if(start_frame_div_64!=end_frame_div_64){
	        num_dlsf+= (472+(end_frame_div_64-start_frame_div_64-1)*472);
        }
    }
	return num_dlsf;
}


void maintain_available_resource(eNB_MAC_INST_NB_IoT *mac_inst){

	available_resource_DL_t *pfree;

	if(mac_inst->current_subframe >= available_resource_DL->end_subframe){
	    //DEBUG("[maintain before kill]=====t:%d=end:%d====%p\n", mac_inst->current_subframe, available_resource_DL->end_subframe, available_resource_DL->next);
		//print_available_resource_DL();
		
        pfree = available_resource_DL;
		available_resource_DL = available_resource_DL->next;
		available_resource_DL->prev = (available_resource_DL_t *)0;
		free((available_resource_DL_t *)pfree);
		//DEBUG("[maintain after kill]=====t:%d=====\n", mac_inst->current_subframe);
		//print_available_resource_DL();
	}else{
		available_resource_DL->start_subframe = mac_inst->current_subframe;
	}

	return ;
}

//	check_subframe must be DLSF, you can use is_dlsf() to check before call function
available_resource_DL_t *check_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, int check_subframe, int num_subframes, int *out_last_subframe, int *out_first_subframe){
	available_resource_DL_t *pt;
	pt = available_resource_DL;
	int end_subframe = check_subframe + num_subframes - 1;
	int diff_gap;

	while((available_resource_DL_t *)0 != pt){
		if(pt->start_subframe <= check_subframe && pt->end_subframe >= check_subframe){
			break;
		}
		pt = pt->next;
	}

	if((available_resource_DL_t *)0 == pt){
		return (available_resource_DL_t *)0;
	}else{
		if(num_subframes <= calculate_DLSF(mac_inst, check_subframe, pt->end_subframe)){

			diff_gap = num_subframes - calculate_DLSF(mac_inst, check_subframe, end_subframe);

			while(diff_gap){
				++end_subframe;
				if(is_dlsf(mac_inst, end_subframe)){
					--diff_gap;
				}
			}
			*out_last_subframe = end_subframe;
			while(!is_dlsf(mac_inst, check_subframe)){
				++check_subframe;
			}
			*out_first_subframe = check_subframe;
			return pt;
		}else{
			return (available_resource_DL_t *)0;
		}
	}
}

int get_I_TBS_NB_IoT(int x,int y)
{
    int I_TBS = 0;
    if(y==1) I_TBS=x;
    else
    {
        if(x==1)    I_TBS=2;
        else if(x==2)   I_TBS=1;
        else
        {
            I_TBS=x;
        }
    }
    return I_TBS;
}


int get_TBS_UL_NB_IoT(uint32_t mcs,uint32_t multi_tone,int Iru)
{
    int TBS;
    uint32_t I_TBS=get_I_TBS_NB_IoT(mcs,multi_tone);
    TBS=UL_TBS_Table[I_TBS][Iru];
    //if((TBS==0)||(Iru>7))
    //{
        //--Iru;
    //}
    //TBS=UL_TBS_Table[I_TBS][Iru];

    return TBS>>3;
}

void insert_schedule_result(schedule_result_t **list, int subframe, schedule_result_t *node){
	schedule_result_t *tmp, *tmp1;
	if((schedule_result_t *)0 == *list){
            *list = node;
        }else{
            tmp = *list;
            tmp1 = (schedule_result_t *)0;
            while((schedule_result_t *)0 != tmp){
                if(subframe < tmp->output_subframe){
                    break;
                }
                tmp1 = tmp;
                tmp = tmp->next;
            }
            if((schedule_result_t *)0 == tmp){
                tmp1->next = node;
            }else{
		node->next = tmp;
		if(tmp1){
			tmp1->next = node;
		}else{
			*list = node;
		}
            }
        }
}

void print_available_resource_DL(void){
	available_resource_DL_t *pt;
	pt = available_resource_DL;
	int i=0;
	printf("=== print available resource ===\n");
	while(pt){
		printf("[%2d] %p %3d-%3d\n", i, pt, pt->start_subframe, pt->end_subframe);
		pt = pt->next;
	}
}

/*Get MCS index*/
uint32_t get_I_mcs_NB_IoT(int CE_level)
{
	if(CE_level==0)
	{
		return 13;
	}
	else if(CE_level==1)
	{
		return 8;
	}
	else
	{
		return 2;
	}
}

uint32_t get_tbs(uint32_t data_size, uint32_t I_tbs, uint32_t *I_sf)
{
	for((*I_sf)=0;(*I_sf)<8;++(*I_sf))
	{
		//DEBUG("[get_tbs]TBS %d SF index %d\n", MAC_TBStable_NB_IoT[I_tbs][(*I_sf)], *I_sf);
		if(MAC_TBStable_NB_IoT[I_tbs][(*I_sf)]>=data_size*8)
		{
			return MAC_TBStable_NB_IoT[I_tbs][(*I_sf)]/8;
		}
	}
	printf("error\n");
	return 0;
}

uint32_t get_num_sf(uint32_t I_sf)
{
	if(I_sf==6)
	{
		return 8;
	}
	else if(I_sf==7)
	{
		return 10;
	}
	else
	{
		return I_sf+1;
	}
}

uint16_t find_suit_i_delay(uint32_t rmax, uint32_t r, uint32_t dci_candidate){
    uint32_t i;
	uint32_t num_candidates = rmax / r;
	uint32_t left_candidates = num_candidates - dci_candidate - 1;	// 0-7
	uint32_t resource_gap = left_candidates * r;
	resource_gap = ((resource_gap * 10)>>3);	//	x1.125
	for(i=0;i<8;++i){
		if(resource_gap <= get_scheduling_delay(i, rmax)){
			return i;
		}
	}
	return 0;
}

uint32_t get_scheduling_delay(uint32_t I_delay, uint32_t R_max)
{
	if(I_delay==0)
	{
		return 0;
	}
	else
	{
		if(R_max<128)
		{
			if(I_delay<=4)
				return 4*I_delay;
			else
				return (uint32_t)(2<<I_delay);//pow(2, I_delay);
		}
		else
		{
			return (uint32_t)(16<<(I_delay-1));//*pow(2, I_delay-1);
		}
	}
}

/*Subcarrier_spacing 0:3.75kHz \ 1 : 15kHz*/
uint32_t get_HARQ_delay(int subcarrier_spacing, uint32_t HARQ_delay_index)
{
	if(subcarrier_spacing==1)
	{
		if(HARQ_delay_index==0)
			return 13;
		else if(HARQ_delay_index==1)
			return 15;
		else if(HARQ_delay_index==2)
			return 17;
		else
			return 18;
	}
	else
	{
		if((HARQ_delay_index==0)&&(HARQ_delay_index==1))
			return 13;
		else
			return 21;
	}
}

int get_resource_field_value(int subcarrier, int k0)
{
    int value = 0;
    if (k0 == 13)
        value = subcarrier;
    else if (k0 == 15)
        value = subcarrier + 4;
    else if (k0 == 17)
        value = subcarrier + 8;
    else if (k0 == 18)
        value = subcarrier + 12;

    return value;
}

//Transfrom source into hyperSF, Frame, Subframe format
void convert_system_number(uint32_t source_sf,uint32_t *hyperSF, uint32_t *frame, uint32_t *subframe)
{
	*hyperSF = (source_sf/10)/1024;
	*frame = (source_sf/10)%1024;
	*subframe = (source_sf%1024)%10;
}

uint32_t get_max_tbs(uint32_t I_tbs)
{
	return MAC_TBStable_NB_IoT[I_tbs][7]/8;
}

//convert hyperSF, Frame, Subframe format into subframe unit
uint32_t convert_system_number_sf(uint32_t hyperSF, uint32_t frame, uint32_t subframe)
{
	return hyperSF*1024*10+frame*10+subframe;
}

/*input start position amd num_dlsf DL subframe, caculate the last subframe number*/
uint32_t cal_num_dlsf(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t hyperSF, uint32_t frame, uint32_t subframe, uint32_t* hyperSF_result, uint32_t* frame_result, uint32_t* subframe_result, uint32_t num_dlsf_require)
{
  uint16_t sf_dlsf_index;
  uint16_t dlsf_num_temp;
  uint32_t abs_sf_start = 0;
  uint32_t abs_sf_end = 0;
  uint8_t period_count=0;
  uint8_t shift_flag=0;
  uint8_t scale_flag=0;

  abs_sf_start=convert_system_number_sf(hyperSF, frame, subframe);
  sf_dlsf_index = abs_sf_start%2560%(mac_inst->sib1_period*10);
  dlsf_num_temp = DLSF_information.sf_to_dlsf_table[sf_dlsf_index];
  //DEBUG("[cal_num_dlsf]sf_dlsf_index %d dlsf_num_temp %d\n", sf_dlsf_index, dlsf_num_temp);

  while(num_dlsf_require>DLSF_information.num_dlsf_per_period)
  {
    period_count++;
    num_dlsf_require-=DLSF_information.num_dlsf_per_period;
  }
  abs_sf_end = abs_sf_start+period_count*mac_inst->sib1_period*10;
  //DEBUG("[cal_num_dlsf]abs_sf_end %d after loop\n", abs_sf_end);
  if(num_dlsf_require>DLSF_information.num_dlsf_per_period-dlsf_num_temp+1)
  {
    if(is_dlsf(mac_inst, sf_dlsf_index)==1)
    {
      num_dlsf_require-=DLSF_information.num_dlsf_per_period-dlsf_num_temp+1;
    }
    else
    {
      num_dlsf_require-=DLSF_information.num_dlsf_per_period-dlsf_num_temp;
    }
    abs_sf_end+=mac_inst->sib1_period*10-abs_sf_end%(mac_inst->sib1_period*10);
    dlsf_num_temp = 0;
    scale_flag = 1;
    //DEBUG("[cal_num_dlsf]abs_sf_end %d after scale\n", abs_sf_end);
  }
  //DEBUG("[cal_num_dlsf]num_dlsf_require remain %d\n", num_dlsf_require);

  if(num_dlsf_require!=0)
  {
    if(scale_flag!=1)
    {
      if(is_dlsf(mac_inst, abs_sf_end)==1)
      {
        shift_flag = 1;
      }
    }
    if(abs_sf_end%(mac_inst->sib1_period*10)!=0)
    {
      abs_sf_end-=abs_sf_end%(mac_inst->sib1_period*10);
      //DEBUG("[cal_num_dlsf] abs_sf_end is %d mod period =  %d\n", abs_sf_end, abs_sf_end%(mac_inst->sib1_NB_IoT_sched_config.sib1_period*10));
    }
    if(shift_flag==1)
    {
      abs_sf_end +=DLSF_information.dlsf_to_sf_table[dlsf_num_temp+num_dlsf_require-2];
    }
    else
    {
      abs_sf_end +=DLSF_information.dlsf_to_sf_table[dlsf_num_temp+num_dlsf_require-1];
    }

    //DEBUG("[cal_num_dlsf]2 DLSF_information.dlsf_to_sf_table = %d dlsf index %d\n", DLSF_information.dlsf_to_sf_table[num_dlsf_require], num_dlsf_require);
  }
  //DEBUG("[cal_num_dlsf]abs_sf_end %d\n", abs_sf_end);
  convert_system_number(abs_sf_end, hyperSF_result, frame_result, subframe_result);
  //DEBUG("[cal_num_dlsf]h %d f %d end %d\n", *hyperSF_result, *frame_result, *subframe_result);
  return abs_sf_end;
}

int get_N_REP(int CE_level)
{
    int N_rep= 0;
    if(CE_level == 0)
    {
        N_rep = (nprach_list)->numRepetitionsPerPreambleAttempt;
    }else if (CE_level == 1)
    {
        N_rep = (nprach_list+1)->numRepetitionsPerPreambleAttempt;
    }else if (CE_level == 2)
    {
        N_rep = (nprach_list+2)->numRepetitionsPerPreambleAttempt;
    }else
    {
        printf("unknown CE level!\n");
        return -1;
    }

    return N_rep;
}

int get_I_REP(int N_rep)
{
    int i;
    for(i = 0; i < 8;i++)
        {
            if(N_rep == rachrepeat[i])
                return i;
        }
    printf("unknown repetition value!\n");
    return -1;
}

int get_DCI_REP(uint32_t R,uint32_t R_max)
{
    int value = -1;
    if (R_max == 1)
    {
        if(R == 1)
        {
            value =0;
        }

    }else if (R_max == 2)
    {
        if(R == 1)
            value = 0;
        if(R == 2)
            value = 1;
     }else if (R_max == 4)
     {
        if(R == 1)
            value = 0;
        if(R == 2)
            value = 1;
        if(R == 4)
            value = 2;
     }else if (R_max >= 8)
     {
        if(R == R_max/8)
            value = 0;
        if(R == R_max/4)
            value = 1;
        if(R == R_max/2)
            value = 2;
        if(R == R_max)
            value = 3;
     }
     return value;
}

void print_available_UL_resource(void){

    int sixtone_num=0;
    int threetone_num=0;
    int singletone1_num=0;
    int singletone2_num=0;
    int singletone3_num=0;

    available_resource_UL_t *available_resource;

    ///sixtone
    available_resource = available_resource_UL->sixtone_Head;

    while(available_resource!=NULL)
    {
        sixtone_num++;
        printf("[sixtone][Node %d] start %d , end %d\n",sixtone_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///threetone
    available_resource = available_resource_UL->threetone_Head;

    while(available_resource!=NULL)
    {
        threetone_num++;
        printf("[threetone][Node %d] start %d, end %d\n",threetone_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone1_Head;

    while(available_resource!=NULL)
    {
        singletone1_num++;
        printf("[singletone1][Node %d] start %d, end %d\n",singletone1_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone2_Head;

    while(available_resource!=NULL)
    {
        singletone2_num++;
        printf("[singletone2][Node %d] start %d, end %d\n",singletone2_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone3_Head;

    while(available_resource!=NULL)
    {
        singletone3_num++;
        printf("[singletone3][Node %d] start %d, end %d\n",singletone3_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

}

//  maybe we can try to use hash table to enhance searching time.
UE_TEMPLATE_NB_IoT *get_ue_from_rnti(eNB_MAC_INST_NB_IoT *inst, rnti_t rnti){
  uint32_t i;
  for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
    if(inst->UE_list_spec->UE_template_NB_IoT[i].active == 1){
      if(inst->UE_list_spec->UE_template_NB_IoT[i].rnti == rnti){
        return &inst->UE_list_spec->UE_template_NB_IoT[i];
      }
    }
  }
  return (UE_TEMPLATE_NB_IoT *)0;
}