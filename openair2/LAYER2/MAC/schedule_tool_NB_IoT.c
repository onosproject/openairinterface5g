#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"

int rachperiod[8]={40,80,160,240,320,640,1280,2560};
int rachstart[8]={8,16,32,64,128,256,512,1024};
int rachrepeat[8]={1,2,4,8,16,32,64,128};
//int rawindow[8]={2,3,4,5,6,7,8,10}; // unit PP
//int rmax[12]={1,2,4,8,16,32,64,128,256,512,1024,2048};
//double gvalue[8]={1.5,2,4,8,16,32,48,64};
//int candidate[4]={1,2,4,8};
//double pdcchoffset[4]={0,0.125,0.25,0.375};
//int dlrepeat[16]={1,2,4,8,16,32,64,128,192,256,384,512,768,1024,1536,2048};
int rachscofst[7]={0,12,24,36,2,18,34};
int rachnumsc[4]={12,24,36,48};

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
    
	available_resource_DL_t *pt, *new_node;
	//int temp;
	uint32_t i, i_div_si_window;
	//uint32_t si_period_div_window;
	pt = available_resource_DL;
    
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

available_resource_DL_t *check_sibs_resource(eNB_MAC_INST_NB_IoT *mac_inst, int check_start_subframe, int check_end_subframe, int num_subframe, int *residual_subframe, int *out_last_subframe, int *out_first_subframe){
	available_resource_DL_t *pt, *pt_free;
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
	int diff_subframe = abs_end_subframe - abs_start_subframe;

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