
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

unsigned char str[6][7] = { "SIBs_1", "SIBs_2", "SIBs_3", "SIBs_4", "SIBs_5", "SIBs_6" };
unsigned char si_repetition_pattern_table[4] = { 20, 40, 80, 160};

void schedule_sibs_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t sibs_order, int start_subframe1){
	
	available_resource_DL_t *pt[8] = { (available_resource_DL_t *)0 };
	uint32_t first_subframe[8] = { -1 };
	//uint32_t end_subframe[8] = { -1 };
	schedule_result_t *new_node;	
	DCIFormatN1_t *sibs_dci;
	uint32_t j, i, k, num_subframe, last_subframe, residual_subframe;

	
	num_subframe = mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_tb;

	for(k=0, i=start_subframe1;i<(start_subframe1+mac_inst->rrc_config.si_window_length);i+=si_repetition_pattern_table[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern], ++k){
		//printf("[debug][sibs%d] subframe: %d, check %d", sibs_order, i, num_subframe);
	
		pt[k] = (available_resource_DL_t *)check_sibs_resource(mac_inst, i, i+9, num_subframe, &residual_subframe, &last_subframe, &first_subframe[k]);

		num_subframe = residual_subframe;
		//printf("-- rest: %d, last: %d start: %d\n", num_subframe, last_subframe, start_subframe1);
		
		if(0==residual_subframe){
			
			sibs_dci = (DCIFormatN1_t *)malloc(sizeof(DCIFormatN1_t));
			sibs_dci->type = 1;
			sibs_dci->orderIndicator = 0;
			sibs_dci->Scheddly = 0;
			sibs_dci->ResAssign = mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_tb;
			sibs_dci->mcs = 2;
			sibs_dci->RepNum = 0;
			sibs_dci->ndi = 0;
			sibs_dci->HARQackRes = 0;
			sibs_dci->DCIRep = 0;
			
			for(k=0, j=start_subframe1;j<=i;++k, j+=si_repetition_pattern_table[mac_inst->rrc_config.sibs_NB_IoT_sched[sibs_order].si_repetition_pattern]){	
				
				if((available_resource_DL_t *)0 != pt[k]){
					new_node = (schedule_result_t *)malloc(sizeof(schedule_result_t));
					//	fill new node
					new_node->output_subframe = first_subframe[k];
					new_node->end_subframe = (j==i)?last_subframe:j+9;
					new_node->sdu_length = 0;
					new_node->direction = DL;	
					new_node->DCI_release = (j==i);
					new_node->channel = NPDSCH;
					new_node->rnti = SI_RNTI;
					new_node->rnti_type = 3;
					new_node->npusch_format = 0;	//	useless
					new_node->R_harq = 0;		//	useless
					new_node->next = (schedule_result_t *)0;
					new_node->DCI_pdu = (void *)sibs_dci;
					new_node->debug_str = str[sibs_order];
					fill_resource_DL(mac_inst, pt[k], first_subframe[k], (j==i)?last_subframe:j+9, new_node);
				}
			}
			
			return ;
		}
	}
	return ;
}
