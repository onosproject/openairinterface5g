
/*! \file eNB_scheduler_RA_NB_IoT.c
 * \brief functions used in Random access scheduling 
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"

unsigned char str1[] = "rar_dci";
unsigned char str2[] = "rar";
unsigned char str3[] = "msg4_dci";
unsigned char str4[] = "msg4";
unsigned char str5[] = "ack_msg4";
unsigned char str6[] = "msg3_dci(retransmit)";
unsigned char str7[] = "msg3(retransmit)";
unsigned char str8[] = "msg4_dci(retransmit)";
unsigned char str9[] = "msg4(retransmit)";
unsigned char str10[] = "ack_msg4(retransmit)";
unsigned char str11[] = "msg3";
unsigned char str12[] = "msg3(retransmit)";

/*void init_RA_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint8_t preamble_index, ce_level_t ce_level, uint32_t sfn_id, uint16_t ta){

	int i;
	RA_TEMPLATE_NB_IoT *msg2_list_tail = mac_inst->RA_msg2_list.tail;
	RA_TEMPLATE_NB_IoT *migrate_node;

	static int static_count=0;
	printf("[%04d][RA scheduler][MSG1] RX %d\n", mac_inst->current_subframe, static_count++);

	for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
		if(0 == mac_inst->RA_template[i].active){
			migrate_node = &mac_inst->RA_template[i];
			break;
		}
	}

	if(i==MAX_NUMBER_OF_UE_MAX_NB_IoT){
		printf("[%04d][RA scheduler][MSG1] number of RA procedures is up to maximum..\n", mac_inst->current_subframe);
		return ;
	}

	migrate_node->active = 1;
	migrate_node->preamble_index = preamble_index;
	migrate_node->ce_level = ce_level;
	migrate_node->ra_rnti = (sfn_id>>2) + 1;
	migrate_node->ta = ta;
	migrate_node->next = (RA_template_NB_IoT *)0;
	migrate_node->prev = (RA_template_NB_IoT *)0;

	//	insert to end of list
	if((RA_template_NB_IoT *)0 == mac_inst->RA_msg2_list.head){
		mac_inst->RA_msg2_list.head = migrate_node;
	}else{
		//	not empty
		mac_inst->RA_msg2_list.tail->next = migrate_node;
		migrate_node->prev = mac_inst->RA_msg2_list.tail;
	}
	mac_inst->RA_msg2_list.tail = migrate_node;

}*/

//  7bytes
void fill_rar_NB_IoT(
	eNB_MAC_INST_NB_IoT *inst, 
	RA_TEMPLATE_NB_IoT *ra_template,
	uint8_t msg3_schedule_delay,
	uint8_t msg3_rep,
	sched_temp_UL_NB_IoT_t *schedule_template
)
//------------------------------------------------------------------------------
{
	uint8_t *dlsch_buffer = &ra_template->rar_buffer[0];
	RA_HEADER_RAPID_NB_IoT *rarh = (RA_HEADER_RAPID_NB_IoT *)dlsch_buffer;
	int i;
	//uint16_t rballoc;
	//uint8_t mcs,TPC,ULdelay,cqireq;
	
	for(i=0; i<7; ++i){
	    dlsch_buffer[i] = 0x0;
    }
	
	// subheader fixed
	rarh->E                     = 0; // First and last RAR
	rarh->T                     = 1; // 0 for E/T/R/R/BI subheader, 1 for E/T/RAPID subheader
	rarh->RAPID                 = ra_template->preamble_index; // Respond to Preamble 0 only for the moment
	
	uint8_t *rar = (uint8_t *)(dlsch_buffer+1);
	//	ta
	ra_template->ta >>= 4;
	rar[0] = (uint8_t)(ra_template->ta>>(2+4)); // 7 MSBs of timing advance + divide by 4
	rar[1] = (uint8_t)(ra_template->ta<<(4-2))&0xf0; // 4 LSBs of timing advance + divide by 4
	
	//	msg3 grant (15bits)
	//	subcarrier spacing:1 subcarrier indication:6 scheduling delay:2 msg3 repetition:3 MCS index: 3
	uint8_t subcarrier_spacing = 1;	// 1bit 15kHz
	uint8_t subcarrier_indication = schedule_template->subcarrier_indication; // 6bits
	uint8_t i_delay = msg3_schedule_delay; // 2bits
	uint8_t msg3_repetition = msg3_rep;// 3bit
	uint8_t mcs_index = 0;//3bit, msg3 88bits 3'b000
	
	rar[1] |= (subcarrier_spacing<<4) | (subcarrier_indication>>3);
	rar[2] = (uint8_t)(subcarrier_indication<<5) | (i_delay<<3) | msg3_repetition;
	rar[3] = (mcs_index<<5)&0xe0;  //  maped
	
	//	tc-rnti
	rar[4] = (uint8_t)(ra_template->ue_rnti>>8);
	rar[5] = (uint8_t)(ra_template->ue_rnti&0xff);
	
	
}


void schedule_msg3_retransimission_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst){

	RA_TEMPLATE_NB_IoT *msg3_nodes = mac_inst->RA_msg3_list.head;

	available_resource_DL_t *dci_node;
	int rmax, fail, res, r;
	int dci_subframe, dci_end_subframe, dci_first_subframe, num_dci_subframe;
	int msg3_subframe;

	int dci_candidate, num_candidate;
	int msg3_scheduling_delay;
	schedule_result_t *dci_result;//, *msg3_result;

	int rep=1;
	//sched_temp_UL_NB_IoT_t npusch_info;

	while((RA_TEMPLATE_NB_IoT *)0 != msg3_nodes){
		if(msg3_nodes->wait_msg3_ack == 0){
		    
			fail=0;
			//	check dci resource
			rmax = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg3_nodes->ce_level].mac_npdcch_NumRepetitions_RA_NB_IoT;//32;
	    num_candidate = 8;//rmax / r;
		r = rmax/num_candidate;
		num_dci_subframe = r;
		dci_subframe = mac_inst->current_subframe;
			for(dci_candidate=0; dci_candidate<8; ++dci_candidate){
                while(!is_dlsf(mac_inst, dci_subframe)){
                    ++dci_subframe;
                }
				dci_node = (available_resource_DL_t *)check_resource_DL(mac_inst, dci_subframe, num_dci_subframe, &dci_end_subframe, &dci_first_subframe);
				if((available_resource_DL_t *)0 != dci_node){
					//dci_subframe += dci_candidate*num_dci_subframe;
					//printf("[RA scheduler][%d][MSG3] [%x][HARQ][DCI]\tstart %3d, num %3d, end %3d\n", mac_inst->current_subframe, msg3_nodes->ue_rnti, dci_subframe, num_dci_subframe, dci_end_subframe);
					break;
				}
				res = num_dci_subframe;
    			while(res != 0){ //  cost lot of time to search
                    if(is_dlsf(mac_inst, dci_subframe)){
                        res--;
                    }
                    dci_subframe++;
                }
			}
			if(8==dci_candidate){
				//failed
				fail|=0x1;
			}
			//	check msg3 resource
			msg3_subframe = dci_end_subframe+8;
			rep = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg3_nodes->ce_level].mac_numRepetitionsPerPreambleAttempt_NB_IoT;
			sched_temp_UL_NB_IoT_t npusch_info;
		
        	uint32_t Iru = 0, mcs, Nru;
		uint32_t mappedMcsIndex = 4;  //  assume all ue supported multi-tone
		mcs = mapped_mcs[msg3_nodes->ce_level][mappedMcsIndex]; //  assume all ue supported multi-tone
		int TBS = get_TBS_UL_NB_IoT(mcs,1,Iru);
		while((TBS<11)&&(Iru<=7)){
            Iru++;
            TBS=get_TBS_UL_NB_IoT(mcs,1,Iru);
        }
		Nru = RU_table[Iru];
		
			for(msg3_scheduling_delay=0; msg3_scheduling_delay<4; ++msg3_scheduling_delay){
				if(0==Check_UL_resource(msg3_subframe+msg3_scheduling_delay_table[msg3_scheduling_delay]+1, Nru*rep, &npusch_info, 1, 0)){	//1: multi-tones 0: single-tone. 1: format 2(ack/nack) 0: format 1
					//msg3_subframe += (msg3_scheduling_delay<<2);
					//printf("[MSG3][%x][HARQ]\t\tstart %3d, num %3d, end %3d\n", msg3_nodes->ue_rnti, msg3_subframe, npusch_info.sf_end-msg3_subframe, npusch_info.sf_end);
					break;
				}
			}
			if(4==msg3_scheduling_delay){
				//failed
				fail|=0x2;
			}
	        
			if(0 == fail){
				
				msg3_nodes->wait_msg3_ack = 1;
				
				DCIFormatN0_t *dci_n0_msg3 = (DCIFormatN0_t *)malloc(sizeof(DCIFormatN0_t));
				//	dci entity
				dci_n0_msg3->type = 0;
				dci_n0_msg3->scind = npusch_info.subcarrier_indication;
				dci_n0_msg3->ResAssign = 0;
				dci_n0_msg3->mcs = 0;
				dci_n0_msg3->ndi = 0;	//	retrnasmit
				dci_n0_msg3->Scheddly = msg3_scheduling_delay;
				dci_n0_msg3->RepNum = rep;
				dci_n0_msg3->rv = 0;
				dci_n0_msg3->DCIRep = 1;//get_DCI_REP()
				//	for dci
				dci_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
			    dci_result->output_subframe = dci_first_subframe;//dci_subframe;
			    dci_result->end_subframe = dci_end_subframe;
			    dci_result->sdu_length = 0;
			    dci_result->direction = UL;
			    dci_result->DCI_release = 0;
			    dci_result->channel = NPDCCH;
			    dci_result->rnti = msg3_nodes->ue_rnti;
			    dci_result->rnti_type = 1;
			    dci_result->npusch_format = 0; //useless
			    dci_result->R_harq = 0;
			    dci_result->next = (schedule_result_t *)0;
			    dci_result->DCI_pdu = (void *)dci_n0_msg3;
		        dci_result->debug_str = str6;
	            
       		    //simulate_rx(&simulate_rx_msg3_list, msg3_nodes->ue_rnti, npusch_info.sf_start);

			    //	fill dci resource
			    fill_resource_DL(mac_inst, dci_node, dci_first_subframe, dci_end_subframe, dci_result);
			    //	fill msg3 resource
			    generate_scheduling_result_UL(-1, -1, npusch_info.sf_start, npusch_info.sf_end, dci_n0_msg3, msg3_nodes->ue_rnti, str12, (void *)0);	//	rnti
				adjust_UL_resource_list(&npusch_info);
	            printf("[%04d][RA scheduler][MSG3 re] MSG3DCI %d-%d MSG3 %d-%d\n", mac_inst->current_subframe, dci_first_subframe, dci_end_subframe, npusch_info.sf_start, npusch_info.sf_end );
			}else{
				printf("[%04d][RA scheduler][MSG3 re] fail vector %d\n", mac_inst->current_subframe, fail );
			}

			++msg3_nodes->msg3_retransmit_count;
		}
		msg3_nodes = msg3_nodes->next;
	}
	return ;
}

void schedule_rar_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst){

	RA_TEMPLATE_NB_IoT *msg2_nodes = mac_inst->RA_msg2_list.head;
	//RA_TEMPLATE_NB_IoT *msg3_list_tail = mac_inst->RA_msg3_list.tail;
	RA_TEMPLATE_NB_IoT *migrate_node;
	schedule_result_t *dci_result, *msg2_result;
    DCIFormatN0_t *dci_n0;
    DCIFormatN1_t *dci_n1_rar;
	available_resource_DL_t *dci_node, *msg2_node;
	int rmax, fail, r, res;
	int dci_subframe, dci_end_subframe, dci_first_subframe, num_dci_subframe;
	int msg2_subframe, msg2_end_subframe, msg2_first_subframe, num_msg2_subframe;
	int msg3_subframe;

	int dci_candidate, i, num_candidate;
	int msg2_i_delay;
	int msg3_scheduling_delay;
	static uint16_t tc_rnti = 0x0101;
	int rep=1;
	sched_temp_UL_NB_IoT_t npusch_info;
    int fail_num = 0;

	while((RA_TEMPLATE_NB_IoT *)0 != msg2_nodes){
		fail=0;
		rmax = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg2_nodes->ce_level].mac_npdcch_NumRepetitions_RA_NB_IoT;//32;
		num_candidate = 8;//rmax / r;
		r = rmax/num_candidate;
		num_dci_subframe = r;
		dci_subframe = mac_inst->current_subframe;
		for(dci_candidate=0; dci_candidate<num_candidate; ++dci_candidate){
            while(!is_dlsf(mac_inst, dci_subframe)){
                ++dci_subframe;
            }
			dci_node = (available_resource_DL_t *)check_resource_DL(mac_inst, dci_subframe, num_dci_subframe, &dci_end_subframe, &dci_first_subframe);
			if((available_resource_DL_t *)0 != dci_node){
				//dci_subframe += dci_candidate*num_dci_subframe;
				break;
			}
			res = num_dci_subframe;
			while(res != 0){ //  maybe would cost lots of time to search
                if(is_dlsf(mac_inst, dci_subframe)){
                    res--;
                }
                dci_subframe++;
            }
		}
		if(num_candidate==dci_candidate){
			fail|=0x1;
		}
		//	check msg2 resource
		uint32_t TBS, I_tbs, I_mcs, I_sf, Nrep;
		
		I_mcs = get_I_mcs_NB_IoT(msg2_nodes->ce_level);
	    I_tbs = I_mcs;
		TBS = get_tbs(7, I_tbs, &I_sf);   //  rar 7 bytes
		Nrep = dl_rep[msg2_nodes->ce_level];
		num_msg2_subframe = get_num_sf(I_sf) * Nrep;
		
		//num_msg2_subframe = 8;
		msg2_i_delay = find_suit_i_delay(rmax, r, dci_candidate);
		for(i=0; i<8; ++i, ++msg2_i_delay){
		    msg2_i_delay = (msg2_i_delay==8)?0:msg2_i_delay;
		    msg2_subframe = dci_end_subframe+4+get_scheduling_delay(msg2_i_delay, rmax);
			msg2_node = (available_resource_DL_t *)check_resource_DL(mac_inst, msg2_subframe, num_msg2_subframe, &msg2_end_subframe, &msg2_first_subframe);
			if((available_resource_DL_t *)0 != msg2_node){
				break;
			}
		}
		if(-1==msg2_i_delay){
			fail|=0x2;
		}
		//	check msg3 resource
		rep = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg2_nodes->ce_level].mac_numRepetitionsPerPreambleAttempt_NB_IoT;
		msg3_subframe = msg2_end_subframe;
		
		uint32_t Iru = 0, mcs, Nru;
		uint32_t mappedMcsIndex = 4;  //  assume all ue supported multi-tone
		mcs = mapped_mcs[msg2_nodes->ce_level][mappedMcsIndex]; //  assume all ue supported multi-tone
		
        TBS = get_TBS_UL_NB_IoT(mcs,1,Iru);
		while((TBS<11)&&(Iru<=7)){    //  88 bits
            Iru++;
            TBS=get_TBS_UL_NB_IoT(mcs,1,Iru);
        }
		Nru = RU_table[Iru];
		
        for(msg3_scheduling_delay=0; msg3_scheduling_delay<4; ++msg3_scheduling_delay){
		    //    36.213 Table 16.3.3-1 Imcs=3'b000 Nru=4
			if(0==Check_UL_resource(msg3_subframe+msg3_scheduling_delay_table[msg3_scheduling_delay]+1, Nru*rep, &npusch_info, 1, 0)){	//1: multi-tones 0: single-tone. 1: format 2(ack/nack) 0: format 1
				break;
			}
		}
		if(4==msg3_scheduling_delay){
			fail|=0x4;
		}
		if(0 < fail){
		    fail_num++;
		    printf("[%04d][RA scheduler][MSG2] fail vector %d\n", mac_inst->current_subframe, fail);
		    //SCHEDULE_LOG("[%04d][RA scheduler][MSG2] rnti: %d preamble: %d fail vector %d\n", mac_inst->current_subframe, msg2_nodes->ra_rnti, msg2_nodes->preamble_index, fail);
		    msg2_nodes = msg2_nodes->next;
		}else{
		    //SCHEDULE_LOG("[%04d][RA scheduler][MSG2] rnti: %d preamble: %d success\n", mac_inst->current_subframe, msg2_nodes->ra_rnti, msg2_nodes->preamble_index);
		    dci_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
		    msg2_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
		    dci_n0 = (DCIFormatN0_t *)malloc(sizeof(DCIFormatN0_t));
			dci_n1_rar = (DCIFormatN1_t *)malloc(sizeof(DCIFormatN1_t));
			
		    fill_rar_NB_IoT(mac_inst, msg2_nodes, msg3_scheduling_delay, rep, &npusch_info);
		    
            msg2_nodes->wait_msg3_ack = 1;
			//	dci entity
			dci_n1_rar->type = 1;
			dci_n1_rar->orderIndicator = 0;
			dci_n1_rar->Scheddly = msg2_i_delay;
			dci_n1_rar->ResAssign = 0;
			dci_n1_rar->mcs = 0;	
			dci_n1_rar->RepNum = 0;	//	36.213 table 16.4.1.3-2, 8 candidates
			dci_n1_rar->ndi = 0;	//	ndi is useless in RAR	36.212 says the feild is reserved
			dci_n1_rar->HARQackRes = 0;	//	no HARQ procedure in RAR	36.212 says the feild is reserved
			dci_n1_rar->DCIRep = 0;	//	36.213 table 16.6-1 R=Rmax/8

			//	for dci
		    dci_result->output_subframe = dci_first_subframe;//dci_subframe;
			dci_result->end_subframe = dci_end_subframe;
		    dci_result->sdu_length = 0;
		    dci_result->direction = DL;
		    dci_result->DCI_release = 0;
		    dci_result->channel = NPDCCH;
		    dci_result->rnti = msg2_nodes->ra_rnti;//ra_rnti;
		    dci_result->rnti_type = 1;
		    dci_result->npusch_format = 0; //useless
		    dci_result->R_harq = 0;
		    dci_result->next = (schedule_result_t *)0;
		    dci_result->DCI_pdu = (void *)dci_n1_rar;
		    dci_result->debug_str = &str1[0];
		    
			//	for msg2
		    msg2_result->output_subframe = msg2_first_subframe;//msg2_subframe;
		    msg2_result->end_subframe = msg2_end_subframe;
		    msg2_result->sdu_length = 56;   //  rar size
		    msg2_result->DLSCH_pdu = msg2_nodes->rar_buffer;
		    msg2_result->direction = DL;
		    msg2_result->DCI_release = 1;
		    msg2_result->channel = NPDSCH;
		    msg2_result->rnti = msg2_nodes->ra_rnti;//ra_rnti;
		    msg2_result->rnti_type = 1;
		    msg2_result->npusch_format = 0;	//useless
		    msg2_result->R_harq = 0;
		    msg2_result->next = (schedule_result_t *)0;
		    msg2_result->DCI_pdu = (void *)dci_n1_rar;
			msg2_result->debug_str = str2;
			msg2_result->rar_buffer = msg2_nodes->rar_buffer;

			//	for msg3(fake DCI N0)
			dci_n0->type = 0;
			dci_n0->scind = npusch_info.subcarrier_indication;
			dci_n0->ResAssign = 0;
			dci_n0->mcs = 0;
			dci_n0->ndi = 1;
			dci_n0->Scheddly = msg3_scheduling_delay;
			dci_n0->RepNum = rep;
			dci_n0->rv = 0;
			dci_n0->DCIRep = 1;//get_DCI_REP()
			
			msg2_nodes->ue_rnti = tc_rnti;
			
			printf("[%04d][RA scheduler][MSG2] RARDCI %d-%d RAR %d-%d MSG3 %d-%d\n", mac_inst->current_subframe, dci_first_subframe, dci_end_subframe, msg2_first_subframe, msg2_end_subframe, npusch_info.sf_start, npusch_info.sf_end);
			
            //	fill dci resource
			fill_resource_DL(mac_inst, dci_node, dci_first_subframe, dci_end_subframe, dci_result);
			//	fill msg2 resource
			fill_resource_DL(mac_inst, msg2_node, msg2_first_subframe, msg2_end_subframe, msg2_result);

			//	fill msg3 resource
			generate_scheduling_result_UL(-1, -1, npusch_info.sf_start, npusch_info.sf_end, dci_n0, tc_rnti, str11, (void *)0);
			adjust_UL_resource_list(&npusch_info);

			//simulate_rx(&simulate_rx_msg3_list, tc_rnti, npusch_info.sf_start);

			migrate_node = msg2_nodes;
			//migrate_node->ue_rnti = tc_rnti;
			tc_rnti++;
	        msg2_nodes = msg2_nodes->next;

			//	maintain list
			if((RA_TEMPLATE_NB_IoT *)0 == migrate_node->prev){
				//	first node
				mac_inst->RA_msg2_list.head = migrate_node->next;	//	including null
			}else{
				//	not first node
				migrate_node->prev->next = migrate_node->next;		//	including null
			}
			if((RA_TEMPLATE_NB_IoT *)0 == migrate_node->next){
				//	last node
				mac_inst->RA_msg2_list.tail = migrate_node->prev;	//	including null
			}else{
				//	not last node
				migrate_node->next->prev = migrate_node->prev;		//	including null
			}

			//	migrate to next list
			//	insert to end of list
			migrate_node->next = (RA_TEMPLATE_NB_IoT *)0;
			migrate_node->prev = (RA_TEMPLATE_NB_IoT *)0;
			if((RA_TEMPLATE_NB_IoT *)0 == mac_inst->RA_msg3_list.head){
				mac_inst->RA_msg3_list.head = migrate_node;
			}else{
				//	not empty
				mac_inst->RA_msg3_list.tail->next = migrate_node;
				migrate_node->prev = mac_inst->RA_msg3_list.tail;
			}
			mac_inst->RA_msg3_list.tail = migrate_node;

		}
	}
	//SCHEDULE_LOG("[%04d][RA scheduler][MSG2] failed number: %d\n", mac_inst->current_subframe, fail_num);
	return ;
}

//	msg4 scheduling: both first time or retransmit would be scheduled in this function(msg4_list).
void schedule_msg4_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst){

	RA_TEMPLATE_NB_IoT *msg4_nodes = mac_inst->RA_msg4_list.head;

	available_resource_DL_t *dci_node, *msg4_node;
	int rmax, fail, r;
	int dci_subframe, dci_end_subframe, dci_first_subframe, num_dci_subframe;
	int msg4_subframe, msg4_end_subframe, msg4_first_subframe, num_msg4_subframe;
	int harq_subframe, harq_end_subframe;

	int dci_candidate, num_candidate;
	int msg4_i_delay, i, res, rep;
	int end_flagHARQ, HARQ_delay;
	sched_temp_UL_NB_IoT_t HARQ_info;
	schedule_result_t *dci_result;
	schedule_result_t *msg4_result;
	schedule_result_t *harq_result;

	msg4_node = (available_resource_DL_t *)0;

	//print_available_resource_DL();

	while((RA_TEMPLATE_NB_IoT *)0 != msg4_nodes){

		if(msg4_nodes->wait_msg4_ack == 0){
			fail=0;
			//	check dci resource
			rmax = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg4_nodes->ce_level].mac_npdcch_NumRepetitions_RA_NB_IoT;//32;
    	    num_candidate = 8;//rmax / r;
    		r = rmax/num_candidate;
    		num_dci_subframe = r;
    		dci_subframe = mac_inst->current_subframe;
			for(dci_candidate=0; dci_candidate<num_candidate; ++dci_candidate){
	            while(!is_dlsf(mac_inst, dci_subframe)){
                    ++dci_subframe;
                }
				dci_node = (available_resource_DL_t *)check_resource_DL(mac_inst, dci_subframe, num_dci_subframe, &dci_end_subframe, &dci_first_subframe);
				if((available_resource_DL_t *)0 != dci_node){
					//dci_subframe += dci_candidate*num_dci_subframe;
					printf("%d msg4 dci %d - %d\n", mac_inst->current_subframe, dci_first_subframe, dci_end_subframe);
					break;
				}
				res = num_dci_subframe;
    			while(res != 0){ //  cost lot of time to search
                    if(is_dlsf(mac_inst, dci_subframe)){
                        res--;
                    }
                    dci_subframe++;
                }
			}
			if(num_candidate==dci_candidate){
				//failed
				fail|=1;
			}

			//	check msg4 resource
			rep = dl_rep[msg4_nodes->ce_level];
			num_msg4_subframe = 1*rep;   //  8 subframes
			msg4_i_delay = find_suit_i_delay(rmax, r, dci_candidate);
			for(i=msg4_i_delay; i<8; ++i){
				msg4_i_delay = (msg4_i_delay==8)?0:msg4_i_delay;
				msg4_subframe = dci_end_subframe+4+get_scheduling_delay(msg4_i_delay, rmax);
				msg4_node = (available_resource_DL_t *)check_resource_DL(mac_inst, msg4_subframe, num_msg4_subframe*rep, &msg4_end_subframe, &msg4_first_subframe);
				if((available_resource_DL_t *)0 != msg4_node){
					//printf("[MSG4]\t\t\tstart %3d, num %3d, end %3d\n", msg4_subframe, num_msg4_subframe, msg4_end_subframe);
					printf("%d msg4 %d - %d\n", mac_inst->current_subframe, msg4_first_subframe, msg4_end_subframe);
					break;
				}
			}
			if(8==i){
				//failed
				fail|=2;
			}

            rep = mac_inst->rrc_config.mac_NPRACH_ConfigSIB[msg4_nodes->ce_level].mac_numRepetitionsPerPreambleAttempt_NB_IoT;
            //print_available_UL_resource();
			for(HARQ_delay=0;HARQ_delay<4;++HARQ_delay){
	            end_flagHARQ=Check_UL_resource(msg4_end_subframe+get_HARQ_delay(1, HARQ_delay), rep, &HARQ_info, 0, 1);	//	RA_template->R
				if(0 == end_flagHARQ){
					harq_subframe = msg4_end_subframe + get_HARQ_delay(1, HARQ_delay);
					harq_end_subframe = harq_subframe + 2*rep -1;
					HARQ_info.ACK_NACK_resource_field=get_resource_field_value(HARQ_info.subcarrier_indication, get_scheduling_delay(HARQ_delay, rmax));
					break;
				}
			}
			if(4 == HARQ_delay){
				fail |= 4;
			}

			if(0==fail){
			    //SCHEDULE_LOG("[%04d][RA scheduler][MSG4] rnti: %d preamble: %d success\n", mac_inst->current_subframe, msg4_nodes->ra_rnti, msg4_nodes->preamble_index);
			    
			    msg4_nodes->wait_msg4_ack = 1;	
				DCIFormatN1_t *dci_n1_msg4 = (DCIFormatN1_t *)malloc(sizeof(DCIFormatN1_t));
				//	dci entity
				dci_n1_msg4->type = 1;
				dci_n1_msg4->orderIndicator = 0;
				dci_n1_msg4->Scheddly = msg4_i_delay;
				dci_n1_msg4->ResAssign = 0;
				dci_n1_msg4->mcs = 0;
				dci_n1_msg4->RepNum = 1;
				dci_n1_msg4->ndi = 1;
				dci_n1_msg4->HARQackRes = HARQ_info.ACK_NACK_resource_field;
				dci_n1_msg4->DCIRep = 1;

				//	for dci
				dci_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
				dci_result->output_subframe = dci_first_subframe;//dci_subframe;
				dci_result->end_subframe = dci_end_subframe;
				dci_result->sdu_length = 0;
				dci_result->direction = DL;
				dci_result->DCI_release = 0;
				dci_result->channel = NPDCCH;
				dci_result->rnti = msg4_nodes->ue_rnti;
				dci_result->rnti_type = 1;
				dci_result->npusch_format = 0; //useless
				dci_result->R_harq = 0;
				dci_result->next = (schedule_result_t *)0;
				dci_result->DCI_pdu = (void *)dci_n1_msg4;
				
				//	for msg4
				msg4_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
				msg4_result->output_subframe = msg4_first_subframe;// msg4_subframe;
				msg4_result->end_subframe = msg4_end_subframe;
				msg4_result->sdu_length = 0;
				msg4_result->direction = DL;
				msg4_result->DCI_release = 0;
				msg4_result->channel = NPDSCH;
				msg4_result->rnti = msg4_nodes->ue_rnti;
		    	msg4_result->rnti_type = 1;
		    	msg4_result->npusch_format = 0;	//useless
		    	msg4_result->R_harq = 0;
				msg4_result->next = (schedule_result_t *)0;
				msg4_result->DCI_pdu = (void *)dci_n1_msg4;
				

				harq_result = (schedule_result_t *)malloc(sizeof(schedule_result_t));
				harq_result->rnti = msg4_nodes->ue_rnti;
			    harq_result->output_subframe = harq_subframe;
			    harq_result->end_subframe = harq_end_subframe;
			    harq_result->sdu_length = 0;
			    harq_result->direction = UL;
			    harq_result->rnti_type = 3;
			    harq_result->DLSCH_pdu = NULL;
			    harq_result->DCI_pdu = (void *)dci_n1_msg4;
			    harq_result->DCI_release = 1;
			    harq_result->channel = NPUSCH;
			    harq_result->next = (schedule_result_t *)0;
				
				if(msg4_nodes->msg4_retransmit_count==0){
			        dci_result->debug_str = str3;
			        msg4_result->debug_str = str4;
			        harq_result->debug_str = str5;
                }else{
                    dci_result->debug_str = str8;
			        msg4_result->debug_str = str9;
			        harq_result->debug_str = str10;
                }
				
				//simulate_rx(&simulate_rx_msg4_list, msg4_nodes->ue_rnti, harq_subframe);
				
				//DEBUG("[%04d][RA scheduler][MSG4] UE:%x MSG4DCI %d-%d MSG4 %d-%d HARQ %d-%d\n", mac_inst->current_subframe, msg4_nodes->ue_rnti, dci_first_subframe, dci_end_subframe, msg4_first_subframe, msg4_end_subframe, HARQ_info.sf_start, HARQ_info.sf_end);
	            msg4_nodes->msg4_retransmit_count++;
	            
	            
				//	fill dci resource
				fill_resource_DL(mac_inst, dci_node, dci_first_subframe, dci_end_subframe, dci_result);

                //	fill msg4 resource
				fill_resource_DL(mac_inst, msg4_node, msg4_first_subframe, msg4_end_subframe, msg4_result);

				//	fill ack/nack resource
				insert_schedule_result(&schedule_result_list_UL, harq_subframe, harq_result);
				//DEBUG("check point\n");
				adjust_UL_resource_list(&HARQ_info);
                
				//	active ue_list ul/dl
				UE_list_NB_IoT_t *UE_list = mac_inst->UE_list_spec;
				for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
					if(UE_list->UE_template_NB_IoT[i].active == 1 && UE_list->UE_template_NB_IoT[i].rnti == msg4_nodes->ue_rnti){
						UE_list->UE_template_NB_IoT[i].direction = rand()&1;
					}
				}	
			}else{
			    //DEBUG("[%04d][RA scheduler][MSG4] fail vector %d\n", mac_inst->current_subframe, fail );
			    //SCHEDULE_LOG("[%04d][RA scheduler][MSG4] rnti: %d preamble: %d fail vector %d\n", mac_inst->current_subframe, msg4_nodes->ra_rnti, msg4_nodes->preamble_index, fail);
            }
		}
		
        msg4_nodes = msg4_nodes->next;
		
	}

	return ;
}

void schedule_RA_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst){

    //uint32_t current_subframe = mac_inst->current_subframe;
    
    schedule_msg3_retransimission_NB_IoT(mac_inst);
    schedule_rar_NB_IoT(mac_inst);
	schedule_msg4_NB_IoT(mac_inst);

	return ;
}






