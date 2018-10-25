
/*! \file eNB_scheduler_bch_NB_IoT.c
 * \brief schedule functions for SIBs transmission in NB-IoT
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"
#include "openair2/RRC/LITE/proto_NB_IoT.h"

char str[6][7] = { "SIBs_1", "SIBs_2", "SIBs_3", "SIBs_4", "SIBs_5", "SIBs_6" };

#define num_flags 2
extern int extend_space[num_flags];
extern int extend_alpha_offset[num_flags];

uint8_t get_SIB23_size(void)
{
  rrc_config_NB_IoT_t     *mac_config = &mac_inst->rrc_config;
  uint8_t size_SIB23_in_MAC = 0;
  switch(mac_config->sibs_NB_IoT_sched[0].si_tb)
  {
  	case si_TB_56:
  		size_SIB23_in_MAC = 56;
  		break;
  	case si_TB_120:
  		size_SIB23_in_MAC = 120;
  		break;
  	case si_TB_208:
  		size_SIB23_in_MAC = 208;
  		break;
  	case si_TB_256:
  		size_SIB23_in_MAC = 256;
  		break;
  	case si_TB_328:
  		size_SIB23_in_MAC = 328;
  		break; 
  	case si_TB_440:
  		size_SIB23_in_MAC = 440;
  		break;
  	case si_TB_556:
  		size_SIB23_in_MAC = 556;
  		break;
  	case SI_TB_680:
  		size_SIB23_in_MAC = 680;
  		break;
  	default:
  		LOG_E(MAC,"No index for SIB23 size from SIB1!\n");
  		break;
  	return size_SIB23_in_MAC;
  }
}

void schedule_sibs(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t sibs_order, int start_subframe1){
	
	available_resource_DL_t *pt[8] = { (available_resource_DL_t *)0 };
	int first_subframe[8] = { -1 };
	//uint32_t end_subframe[8] = { -1 };
	schedule_result_t *new_node;	
	DCIFormatN1_t *sibs_dci;
	uint32_t j, i, k;
	uint8_t SIB23_size = 0;
	uint8_t *SIB23_pdu = get_NB_IoT_SIB23();
	int residual_subframe, num_subframe, last_subframe;
	num_subframe = (mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_tb)*8;
	
	int rmax = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[0].mac_npdcch_NumRepetitions_RA_NB_IoT;
	rmax = (rmax * 10) >> 3;	//	x1.25
	


	for(k=0, i=start_subframe1; i<(start_subframe1+mac_inst->rrc_config.si_window_length); i+=si_repetition_pattern[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern], ++k){
	LOG_D(MAC,"[debug][sibs%d] subframe: %d, check %d", sibs_order, i, num_subframe);
	LOG_D(MAC,"[%d][%d][%d] [%d][%d]\n", i, start_subframe1, mac_inst->rrc_config.si_window_length, sibs_order, si_repetition_pattern[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern]);
	//system("pause");
	#if 0	//disable new feature
		//	avoid to occupied others searching space. TODO: css, uss connect with configuration module
		//	start start+rmax
		//	i i+9
		int continue_flag=0;
		for(l=0; l<num_flags; ++l){
			if((extend_space[l]>>extend_alpha_offset[l] <= i%extend_space[l] && ((extend_space[l]>>extend_alpha_offset[l])+rmax) >= i%extend_space[l]) ||
			   (extend_space[l]>>extend_alpha_offset[l] <= (i+9)%extend_space[l] && ((extend_space[l]>>extend_alpha_offset[l])+rmax) >= (i+9)%extend_space[l])){
			   		continue_flag = 1;
			   		
			   }
		}
		if(continue_flag == 1)
			continue;
	#endif	
		pt[k] = (available_resource_DL_t *)check_sibs_resource(mac_inst, i, i+9, num_subframe, &residual_subframe, &last_subframe, &first_subframe[k]);
		
		num_subframe = residual_subframe;
		LOG_D(MAC,"-- rest: %d, last: %d start: %d\n", num_subframe, last_subframe, start_subframe1);
		
		if(0==residual_subframe){LOG_D(MAC,"output\n\n");

			sibs_dci = (DCIFormatN1_t *)malloc(sizeof(DCIFormatN1_t));
			sibs_dci->type = 1;
			sibs_dci->orderIndicator = 0;
			sibs_dci->Scheddly = 0;
			sibs_dci->ResAssign = 8;
			sibs_dci->mcs = 2;
			sibs_dci->RepNum = 0;
			sibs_dci->ndi = 0;
			sibs_dci->HARQackRes = 0;
			sibs_dci->DCIRep = 0;
			
			for(k=0, j=start_subframe1;j<=i;++k, j+=si_repetition_pattern[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern]){	
				LOG_D(MAC,"for1 k=%d j=%d i=%d rep=%d\n", k, j, i, si_repetition_pattern[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern]);
				if((available_resource_DL_t *)0 != pt[k]){
					new_node = (schedule_result_t *)malloc(sizeof(schedule_result_t));
					//	fill new node
					SIB23_size = get_SIB23_size();
					new_node->output_subframe = first_subframe[k];
					new_node->end_subframe = (j==i)?last_subframe:j+9;
					new_node->sdu_length = SIB23_size;
					new_node->DLSCH_pdu = SIB23_pdu;
					new_node->direction = DL;	
					new_node->DCI_release = (j==i);
					new_node->channel = NPDSCH;
					new_node->rnti = 65535;
					new_node->rnti_type = 1;
					new_node->npusch_format = 0;	//	useless
					new_node->R_harq = 0;		//	useless
					new_node->next = (schedule_result_t *)0;
					new_node->DCI_pdu = (void *)sibs_dci;
					//new_node->debug_str = str[sibs_order];
					LOG_D(MAC,"debug: pt[k]->start_subframe:%d output_subframe:%d end_subframe:%d rep:%d\n", pt[k]->start_subframe, first_subframe[k], (j==i)?last_subframe:j+9,si_repetition_pattern[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern]);
					fill_resource_DL(mac_inst, pt[k], first_subframe[k], (j==i)?last_subframe:j+9, new_node);
					LOG_D(MAC,"for*2\n");
				}
				LOG_D(MAC,"for2\n");
			}

			return ;
		}
	}
	return ;
}
