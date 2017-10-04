
/*! \file eNB_scheduler_dlsch_NB_IoT.c
 * \brief handle DL UE-specific scheduling
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"

void schedule_DL_NB_IoT(module_id_t module_id, eNB_MAC_INST_NB_IoT *mac_inst, UE_TEMPLATE_NB_IoT *UE_info, uint32_t hyperSF_start, uint32_t frame_start, uint32_t subframe_start)
{
	//number of candidate
	int cdd_num;
	//Transport block size
	int TBS;
	//Scheduling result buffer
	sched_temp_DL_NB_IoT_t *NPDCCH_info = (sched_temp_DL_NB_IoT_t*)malloc(sizeof(sched_temp_DL_NB_IoT_t));
	sched_temp_DL_NB_IoT_t *NPDSCH_info = (sched_temp_DL_NB_IoT_t*)malloc(sizeof(sched_temp_DL_NB_IoT_t));
	sched_temp_UL_NB_IoT_t *HARQ_info = (sched_temp_UL_NB_IoT_t*)malloc(sizeof(sched_temp_UL_NB_IoT_t));
	//DCI N1
	DCIFormatN1_t *DCI_N1 = (DCIFormatN1_t*)malloc(sizeof(DCIFormatN1_t));
	//RLC Status
	//mac_rlc_status_resp_NB_IoT_t rlc_status;
	/*Index in DCI_N1*/
	uint32_t I_mcs, I_tbs, I_delay, I_sf;
	/*value for corresponding index*/
	/*Number of subframe per repetition*/
	int n_sf;
	/*flag*/
	int end_flagCCH=0;
	int end_flagSCH=0;
	int end_flagHARQ=0;
	int flag_retransmission=0;

	int HARQ_delay=0;
	uint32_t data_size;
	//uint32_t mac_sdu_size;

	//uint8_t sdu_temp[SCH_PAYLOAD_SIZE_MAX_NB_IoT];
	logical_chan_id_t logical_channel;

	uint32_t subheader_length=2;
	//uint32_t payload_offset;

	uint32_t search_space_end_sf, h_temp, f_temp, sf_temp;

	I_mcs = get_I_mcs_NB_IoT(UE_info->CE_level);
	I_tbs = I_mcs;
	//get max TBS
	TBS = get_max_tbs(I_tbs);
	printf("[%04d][DLSchedulerUSS] Max TBS %d MCS index %d TBS index %d\n", mac_inst->current_subframe, TBS, I_mcs, I_tbs);
	/*set UE data information*/
	/*New transmission*/
	if(UE_info->HARQ_round==0)
	{
		//Get RLC status
		/*
		rlc_status = mac_rlc_status_ind_NB_IoT(
										module_id,
										UE_info->rnti,
										module_id,
										frame_start,
										subframe_start,
										1,
										0,
										DCCH0_NB_IoT,
										0);
		data_size = rlc_status.bytes_in_buffer;
		*/
		data_size = 200;
	}
	/*Retransmission*/
	else
	{
		data_size = UE_info->DLSCH_pdu_size;
		flag_retransmission = 1;
		if((UE_info->HARQ_round>0)&&(TBS<data_size))
		{
			printf("[%04d][DLSchedulerUSS][Fail] TBS is not enough for retransmission\n", mac_inst->current_subframe);
			return;
		}
	}
	printf("[%04d][DLSchedulerUSS] UE data size %d\n", mac_inst->current_subframe, data_size);
	//Have DCCH data
	if(data_size == 0)
	{
		printf("[%04d][DLSchedulerUSS][Fail] No data in DCCH0_NB_IoT\n", mac_inst->current_subframe);
		return;
	}
	if(data_size>127)
	{
		subheader_length=3;
	}
	if(TBS > data_size+subheader_length)
	{
		TBS = get_tbs(data_size, I_tbs, &I_sf);
		printf("[%04d][DLSchedulerUSS] TBS change to %d because data size is smaller than previous TBS\n", mac_inst->current_subframe, TBS);
	}

  search_space_end_sf=cal_num_dlsf(mac_inst, hyperSF_start, frame_start, subframe_start, &h_temp, &f_temp, &sf_temp, UE_info->R_max);
  printf("[%04d][DLSchedulerUSS] Search_space_start_sf %d Search_space_end_sf %d\n", convert_system_number_sf(hyperSF_start, frame_start, subframe_start), mac_inst->current_subframe, search_space_end_sf);
	/*Loop all NPDCCH candidate position*/
	for(cdd_num=0;cdd_num<UE_info->R_max/UE_info->R_dci;++cdd_num)
	{
		//DEBUG("[%04d][DLSchedulerUSS] Candidate num %d DCI Rep %d\n", cdd_num, UE_info->R_dci);
		/*Check NPDCCH Resource*/
		end_flagCCH = check_resource_NPDCCH_NB_IoT(mac_inst, hyperSF_start, frame_start, subframe_start, NPDCCH_info, cdd_num, UE_info->R_dci);

		//This candidate position is available
		/*Check NPDSCH Resource*/
		if(end_flagCCH!=-1)
		{
		  //DEBUG("[%04d][DLSchedulerUSS] Candidate num %d allocate success\n", cdd_num);
			//DEBUG("[%04d][DLSchedulerUSS] Allocate NPDCCH subframe %d to subframe %d cdd index %d\n", mac_inst->current_subframe, NPDCCH_info->sf_start, NPDCCH_info->sf_end, cdd_num);
			/*
			//Max DL TBS
			if(TBS > data_size+subheader_length)
			{
				TBS = get_tbs(data_size, I_tbs, &I_sf);
				DEBUG("[%04d][DLSchedulerUSS] TBS change to %d because data size is smaller than previous TBS\n", mac_inst->current_subframe, TBS);
			}
			*/
			//Get number of subframe this UE need per repetition
			n_sf = get_num_sf(I_sf);
			//DEBUG("[%04d][DLSchedulerUSS] Number SF %d index SF %d\n", n_sf, I_sf);
			//DEBUG("[%04d][DLSchedulerUSS] Require total %d DL SF Rep %d\n", n_sf*UE_info->R_dl, UE_info->R_dl);
			//Check have enough NPDSCH resource or not
			//loop 8 scheduling delay index
			for(I_delay=0;I_delay<8;++I_delay)
			{
		        if(search_space_end_sf<NPDCCH_info->sf_end+get_scheduling_delay(I_delay, UE_info->R_max)+5)
		        {
		          end_flagSCH = check_resource_NPDSCH_NB_IoT(mac_inst, NPDSCH_info, NPDCCH_info->sf_end, I_delay, UE_info->R_max, UE_info->R_dl, n_sf);
		          //Have available resource
		          /*Check HARQ resource*/
		          if(end_flagSCH!=-1)
		          {
		            //DEBUG("[%04d][DLSchedulerUSS] Scheduling delay index: %d value: %d + 4 allocate success\n", mac_inst->current_subframe, I_delay, get_scheduling_delay(I_delay, UE_info->R_max));
		            //DEBUG("[%04d][DLSchedulerUSS] Allocate NPDSCH subframe %d to subframe %d\n", mac_inst->current_subframe, NPDSCH_info->sf_start, NPDSCH_info->sf_end);
		            for(HARQ_delay=0;HARQ_delay<4;++HARQ_delay)
		            {
		              //DEBUG("[%04d][DLSchedulerUSS] HARQ delay %d\n", mac_inst->current_subframe,get_HARQ_delay(1, HARQ_delay) );
		              end_flagHARQ=Check_UL_resource(NPDSCH_info->sf_end+get_HARQ_delay(1, HARQ_delay), UE_info->R_harq, HARQ_info, 0, 1);
		              if(end_flagHARQ!=-1)
		              {
		                //DEBUG("[%04d][DLSchedulerUSS] Allocate HARQ feedback subframe %d to subframe %d\n", mac_inst->current_subframe, HARQ_info->sf_start, HARQ_info->sf_end);
		                HARQ_info->ACK_NACK_resource_field=get_resource_field_value(HARQ_info->subcarrier_indication, get_scheduling_delay(HARQ_delay, UE_info->R_max));
		                //toggle NDI
		                if(flag_retransmission==0)
		                {
		                  UE_info->oldNDI_DL=(UE_info->oldNDI_DL+1)%2;
		                  //New transmission need to request data from RLC and generate new MAC PDU
		                  UE_info->I_mcs_dl = I_mcs;
		                  /*
		                  //Request data from RLC layer
		                  rlc_status = mac_rlc_status_ind_NB_IoT(
		                        module_id,
		                        UE_info->rnti,
		                        module_id,
		                        frame_start,
		                        subframe_start,
		                        1,
		                        0,
		                        DCCH0_NB_IoT,
		                        TBS-subheader_length);
		                  */
		                  //mac_sdu_size = mac_rlc_data_req_eNB_NB_IoT(module_id, UE_info->rnti, 0, frame_start, 0, DCCH0_NB_IoT, sdu_temp);
		                  logical_channel=DCCH0_NB_IoT;
		                  //Generate header
		                  //payload_offset = generate_dlsch_header_NB_IoT(UE_info->DLSCH_pdu.payload, 1, &logical_channel, &mac_sdu_size, 0, 0, TBS);
		                  //Complete MAC PDU
		                  //memcpy(UE_info->DLSCH_pdu.payload+payload_offset, sdu_temp, mac_sdu_size);
		                  UE_info->DLSCH_pdu.pdu_size=TBS;
		                }
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS][Success] RNTI %d complete scheduling\n", mac_inst->current_subframe, UE_info->rnti);
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS] RNTI %d\n", mac_inst->current_subframe, UE_info->rnti);
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS][Success] Allocate NPDCCH subframe %d to subframe %d cdd index %d\n", mac_inst->current_subframe, NPDCCH_info->sf_start, NPDCCH_info->sf_end, cdd_num);
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS][Success] Scheduling delay index: %d value: %d + 4\n", mac_inst->current_subframe, I_delay, get_scheduling_delay(I_delay, UE_info->R_max));
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS][Success] Allocate NPDSCH subframe %d to subframe %d\n", mac_inst->current_subframe, NPDSCH_info->sf_start, NPDSCH_info->sf_end);
		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS][Success] Allocate HARQ feedback subframe %d to subframe %d\n", mac_inst->current_subframe, HARQ_info->sf_start, HARQ_info->sf_end);
		                //DEBUG("[%04d][DLSchedulerUSS] Allocate NPDCCH subframe %d to subframe %d cdd index %d\n", mac_inst->current_subframe, NPDCCH_info->sf_start, NPDCCH_info->sf_end, cdd_num);
		                //DEBUG("[%04d][DLSchedulerUSS] Scheduling delay index: %d value: %d + 4\n", mac_inst->current_subframe, I_delay, get_scheduling_delay(I_delay, UE_info->R_max));
		                //DEBUG("[%04d][DLSchedulerUSS] Allocate NPDSCH subframe %d to subframe %d\n", mac_inst->current_subframe, NPDSCH_info->sf_start, NPDSCH_info->sf_end);
		                //DEBUG("[%04d][DLSchedulerUSS] Allocate HARQ feedback subframe %d to subframe %d\n", mac_inst->current_subframe, HARQ_info->sf_start, HARQ_info->sf_end);
		                //Store PDU in UE template for retransmission
		                fill_DCI_N1(DCI_N1, UE_info, I_delay, I_sf, HARQ_info->ACK_NACK_resource_field);
		                //DEBUG("[%04d][DLSchedulerUSS] HARQ index %d\n", HARQ_info->ACK_NACK_resource_field);
		                printf("[%04d][DLSchedulerUSS] DCI N1 type:%d order:%d MCS:%d HARQ index:%d R:%d RscAssign:%d scheddly:%d DCI_R:%d\n", mac_inst->current_subframe, DCI_N1->type, DCI_N1->orderIndicator, DCI_N1->mcs, DCI_N1->HARQackRes, DCI_N1->RepNum, DCI_N1->ResAssign, DCI_N1->Scheddly, DCI_N1->DCIRep);
		                //Generate Scheduling result for this UE
		                generate_scheduling_result_DL(NPDCCH_info->sf_start, NPDSCH_info->sf_start, HARQ_info->sf_start, DCI_N1, UE_info->rnti, TBS, UE_info->DLSCH_pdu.payload);
		                //DEBUG("[%04d][DLSchedulerUSS] finish generate scheduling result\n");
		                //matain DL avialable resource
		                maintain_resource_DL(mac_inst, NPDCCH_info, NPDSCH_info);
		                //available_resource_DL_t *temp=available_resource_DL;
		                /*
		                while(temp!=NULL)
		                {
		                  DEBUG("[%04d][DLSchedulerUSS] Available resource node subframe start %d end %d\n", mac_inst->current_subframe, temp->start_subframe, temp->end_subframe);
		                  temp=temp->next;
		                }
		                */
		                //Do maintain UL resource
		                adjust_UL_resource_list(HARQ_info);
		                printf("[%04d][DLSchedulerUSS] Complete DL scheduling\n", mac_inst->current_subframe);
		                

		                //SCHEDULE_LOG("[%04d][DLSchedulerUSS] RNTI %d complete scheduling\n", mac_inst->current_subframe, UE_info->rnti);

		                return;
		              }
		            }
		            /*harq resource fail*/
		            if(end_flagHARQ==-1)
		            {
		              //DEBUG("[%04d][DLSchedulerUSS] [Fail]HARQ_delay %d HARQ Resource fail\n", mac_inst->current_subframe, HARQ_delay);
		            }
		          }
		          //DEBUG("[%04d][DLSchedulerUSS] Scheduling delay index %d allocate fail\n", mac_inst->current_subframe, I_delay);
		        }
			}
			/*NPDSCH resource fail*/
			if(end_flagSCH==-1)
			{
				//DEBUG("[%04d][DLSchedulerUSS] [Fail]I_delay %d NPDSCH Resource fail\n", mac_inst->current_subframe, I_delay);
			}
		}
		//DEBUG("[%04d][DLSchedulerUSS] Candidate %d no resource\n", mac_inst->current_subframe, cdd_num);

	}
	/*Resource allocate fail*/
	if((end_flagCCH==-1)||(end_flagSCH==-1)||(end_flagHARQ==-1))
	{
		printf("[%04d][DLSchedulerUSS][Fail]Resource allocate fail\n", mac_inst->current_subframe);
		printf("[%04d][DLSchedulerUSS][Fail]RNTI %d resource allocate fail\n", mac_inst->current_subframe, UE_info->rnti);
	}
}

int check_resource_NPDCCH_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t hyperSF_start, uint32_t frame_start, uint32_t subframe_start, sched_temp_DL_NB_IoT_t *NPDCCH_info, uint32_t cdd_num, uint32_t dci_rep)
{
	NPDCCH_info->sf_start = cal_num_dlsf(mac_inst, hyperSF_start, frame_start, subframe_start, &(NPDCCH_info->start_h), &(NPDCCH_info->start_f), &(NPDCCH_info->start_sf), dci_rep*cdd_num+1);
	//DEBUG("[check_resource_NPDCCH_NB_IoT]abs start : %d\n", NPDCCH_info->sf_start);
	return check_resource_DL_NB_IoT(mac_inst, NPDCCH_info->start_h, NPDCCH_info->start_f, NPDCCH_info->start_sf, dci_rep, NPDCCH_info);
}

int check_resource_NPDSCH_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, sched_temp_DL_NB_IoT_t *NPDSCH_info, uint32_t sf_end, uint32_t I_delay, uint32_t R_max, uint32_t R_dl, uint32_t n_sf)
{
	int sf_temp = sf_end+get_scheduling_delay(I_delay, R_max)+5;
	while(is_dlsf(mac_inst, sf_temp)!=1)
  {
    ++sf_temp;
  }
	NPDSCH_info->sf_start = sf_temp;
	//transform sf into Hyper SF, Frame and subframe
	convert_system_number(NPDSCH_info->sf_start,&(NPDSCH_info->start_h), &(NPDSCH_info->start_f), &(NPDSCH_info->start_sf));
	//check this position available or not
	return check_resource_DL_NB_IoT(mac_inst, NPDSCH_info->start_h, NPDSCH_info->start_f, NPDSCH_info->start_sf, R_dl*n_sf, NPDSCH_info);
}

/*Check the available resource is enough or not from input starting position*/
/*return 0:success\1:fail*/
int check_resource_DL_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint32_t hyperSF_start, uint32_t frame_start, uint32_t subframe_start, uint32_t dlsf_require, sched_temp_DL_NB_IoT_t *schedule_info)
{

	uint32_t node_start_sf, node_end_sf;
	uint32_t rsc_start_sf, rsc_end_sf;

	/*calculate the last subframe number for this transmission*/
	schedule_info->sf_end= cal_num_dlsf(mac_inst, hyperSF_start, frame_start, subframe_start, &(schedule_info->end_h), &(schedule_info->end_f), &(schedule_info->end_sf), dlsf_require);
	//DEBUG("abs_end = %d\n", schedule_info->sf_end);
	rsc_start_sf = schedule_info->sf_start;
	if(schedule_info->sf_start<=schedule_info->sf_end)
	{
		rsc_end_sf = schedule_info->sf_end;
	}
	else
	{
		/*input position + Upper bound of subframe*/
		rsc_end_sf = schedule_info->sf_end+(1024*1024*10);
	}
	//DEBUG("check_resource_DL_NB_IoT flag 1\n");
	/*initialize*/
	schedule_info->node = available_resource_DL;
	//DEBUG("rsc need start subframe %d end subframe %d\n", rsc_start_sf, rsc_end_sf);
	/*Check available resource nodes to find the appropriate resource position*/
	while(schedule_info->node!=NULL)
	{
		//schedule_info->node->start_subframe <= schedule_info->sf_end
		//DEBUG("check_resource_DL_NB_IoT flag 2\n");
		node_start_sf = schedule_info->node->start_subframe;
		if(schedule_info->node->start_subframe<=schedule_info->node->end_subframe)
		{
			node_end_sf = schedule_info->node->end_subframe;
		}
		else
		{
			/*input position + Upper bound of subframe*/
			node_end_sf = schedule_info->node->end_subframe+(1024*1024*10);
		}
		//DEBUG("node start %d node end %d\n", node_start_sf, node_end_sf);
        if((node_start_sf<=rsc_start_sf)&&(node_end_sf>=rsc_end_sf))
        {
            return 0;
        }
        schedule_info->node = schedule_info->node->next;
	}
	//DEBUG("check_resource_DL_NB_IoT flag 3\n");
	return -1;
}

uint32_t generate_dlsch_header_NB_IoT(uint8_t *pdu, uint32_t num_sdu, logical_chan_id_t *logical_channel, uint32_t *sdu_length, uint8_t flag_drx, uint8_t flag_ta, uint32_t TBS)
{
	int i;
	uint32_t total_sdu_size=0;
	//number of control element
	uint32_t num_ce=0;
	uint32_t num_subheader=0;
	uint32_t num_sdu_L_15;
	int32_t padding_size;
	uint8_t flag_end_padding=0;
	SCH_SUBHEADER_FIXED_NB_IoT *mac_header=(SCH_SUBHEADER_FIXED_NB_IoT*)pdu;
	uint32_t offset=0;

	for(i=0;i<num_sdu;++i)
	{
		printf("index %d sdu size %d\n", i, sdu_length[i]);
		if(sdu_length[i]>127)
		{
			num_sdu_L_15++;
		}
		total_sdu_size+=sdu_length[i];
	}
	if(flag_drx==1)
		num_ce++;
	if(flag_ta==1)
		num_ce++;
	num_subheader=num_ce+num_sdu;
	padding_size = TBS-total_sdu_size-num_ce;
	if(padding_size<0)
	{
		printf("[ERROR]TBS less than require subheader and control element\n");
		return -1;
	}
	printf("total SDU size %d\n", total_sdu_size);
	printf("padding size %d\n", padding_size);
	if(padding_size>2)
	{
		flag_end_padding=1;
	}
	if((padding_size<=2)&&(padding_size>0))
	{
		mac_header->LCID=PADDING;
		mac_header->E=1;
		mac_header->F2=0;
		mac_header->R=0;
		mac_header++;
		offset++;
	}
	if(padding_size==2)
	{
		mac_header->LCID=PADDING;
		mac_header->E=1;
		mac_header->F2=0;
		mac_header->R=0;
		mac_header++;
		offset++;
	}
	if(flag_drx==1)
	{
		mac_header->LCID=DRX_COMMAND;
		mac_header->E=1;
		mac_header->F2=0;
		mac_header->R=0;
		mac_header++;
		num_subheader--;
		offset++;
	}
	for(i=0;i<num_sdu;++i)
	{
		if((num_subheader==1)&&(flag_end_padding!=1))
        {
            mac_header->E=0;
            mac_header->LCID = logical_channel[i];
            mac_header->F2=0;
            mac_header->R=0;
            offset++;
            printf("last sdu\n");
        }
        else
        {
            if(sdu_length[i]<128)
            {
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->LCID = logical_channel[i];
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->F2=0;
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->R=0;
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->E=1;
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->F=0;
                ((SCH_SUBHEADER_SHORT_NB_IoT*)mac_header)->L=(uint8_t)sdu_length[i];
                num_subheader--;
                mac_header+=2;
                offset+=2;
            }
            else
            {
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->LCID = logical_channel[i];
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->F2=0;
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->R=0;
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->F=1;
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->E=1;
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->L_MSB=(uint8_t)(sdu_length[i]/256);
                ((SCH_SUBHEADER_LONG_NB_IoT*)mac_header)->L_LSB=(uint8_t)(sdu_length[i]%256);
                mac_header+=3;
                num_subheader--;
                offset+=3;
            }
        }
	}
	if(flag_end_padding==1)
	{
		mac_header->LCID=PADDING;
		mac_header->E=0;
		mac_header->F2=0;
		mac_header->R=0;
		mac_header++;
		offset++;
	}
	return offset;
}

void fill_DCI_N1(DCIFormatN1_t *DCI_N1, UE_TEMPLATE_NB_IoT *UE_info, uint32_t scheddly, uint32_t I_sf, uint32_t I_harq)
{
	DCI_N1->type=1;
	DCI_N1->orderIndicator = 0;
	DCI_N1->Scheddly = scheddly;
	DCI_N1->ResAssign = I_sf;
	DCI_N1->mcs = UE_info->I_mcs_dl;
	DCI_N1->RepNum = get_index_Rep_dl(UE_info->R_dl);
	DCI_N1->HARQackRes = I_harq;
	//DCI_N1->DCIRep = 3-UE_info->R_max/UE_info->R_dci/2;
	DCI_N1->DCIRep = UE_info->R_dci;
}

void generate_scheduling_result_DL(uint32_t DCI_subframe, uint32_t NPDSCH_subframe, uint32_t HARQ_subframe, DCIFormatN1_t *DCI_pdu, rnti_t rnti, uint32_t TBS, uint8_t *DLSCH_pdu)
{
	// create the schedule result node for this time transmission
	schedule_result_t *NPDCCH_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));
	schedule_result_t *NPDSCH_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));
	schedule_result_t *HARQ_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));

	schedule_result_t *tmp, *tmp1;
	/*fill NPDCCH result*/
	NPDCCH_result->rnti=rnti;
	NPDCCH_result->output_subframe = DCI_subframe;
	NPDCCH_result->sdu_length = TBS;
	NPDCCH_result->direction = 1;
	NPDCCH_result->rnti_type = 3;
	NPDCCH_result->DLSCH_pdu = NULL;
	NPDCCH_result->DCI_pdu = DCI_pdu;
	NPDCCH_result->DCI_release = 0;
	NPDCCH_result->channel = NPDCCH;
	NPDCCH_result->next = NULL;
	/*fill NPDSCH result*/
	NPDSCH_result->rnti=rnti;
	NPDSCH_result->output_subframe = NPDSCH_subframe;
	NPDSCH_result->sdu_length = TBS;
	//NPDSCH_result->DLSCH_pdu = DLSCH_pdu;
	NPDSCH_result->DLSCH_pdu = NULL;
	NPDSCH_result->direction = 1;
	NPDSCH_result->rnti_type = 3;
	NPDSCH_result->DCI_pdu = DCI_pdu;
	NPDSCH_result->DCI_release = 0;
	NPDSCH_result->channel = NPDSCH;
	NPDSCH_result->next = NULL;
	/*fill HARQ result*/
	HARQ_result->rnti=rnti;
	HARQ_result->output_subframe = HARQ_subframe;
	HARQ_result->sdu_length = 0;
	HARQ_result->direction = 1;
	HARQ_result->rnti_type = 3;
	HARQ_result->DLSCH_pdu = NULL;
	HARQ_result->DCI_pdu = DCI_pdu;
	HARQ_result->DCI_release = 1;
	HARQ_result->channel = NPUSCH;
	HARQ_result->npusch_format = 1;
	HARQ_result->next = NULL;
  //DEBUG("[generate_scheduling_result_DL] Generate NPDCCH node\n");
	/*NPDCCH scheduling result*/
	// be the first node of the DL scheduling result
	
	tmp = NULL;
	tmp1 = NULL;
	
	if(schedule_result_list_DL == NULL)
	{
		//schedule_result_list_DL = (schedule_result_t*)malloc(sizeof(schedule_result_t));
		schedule_result_list_DL = NPDCCH_result;
		printf("[generate_scheduling_result_DL] Generate NPDCCH node at head\n");
	}
	else
	{
		tmp = schedule_result_list_DL;
		while(tmp!=NULL)
		{
			if(DCI_subframe < tmp->output_subframe)
			{
				break;
			}
			tmp1 = tmp;
			tmp = tmp->next;
			//DEBUG("[generate_scheduling_result_DL] node output subframe %d at NPDCCH part\n", tmp->output_subframe);
		}
		/*tail*/
		if(tmp==NULL)
		{
			tmp1->next = NPDCCH_result;
		}
		else
		{
			NPDCCH_result->next = tmp;
			if(tmp1)
			{
				tmp1->next = NPDCCH_result;
			}
			else
			{
				schedule_result_list_DL = NPDCCH_result;
			}
		}
	}
	//DEBUG("[generate_scheduling_result_DL] Generate NPDCSH node\n");
	/*NPDSCH scheduling result*/
	tmp1 = NULL;
	tmp = schedule_result_list_DL;
	while(tmp!=NULL)
	{
		if(NPDSCH_subframe < tmp->output_subframe)
		{
			break;
		}
		//DEBUG("[generate_scheduling_result_DL] node output subframe %d at NPDSCH part\n", tmp->output_subframe);
		tmp1 = tmp;
		tmp = tmp->next;
	}
	if(tmp==NULL)
	{
		tmp1->next = NPDSCH_result;
	}
	else
	{
		NPDSCH_result->next = tmp;
		if(tmp1)
		{
			tmp1->next = NPDSCH_result;
		}
		else
		{
			schedule_result_list_DL = NPDSCH_result;
		}
	}
  //DEBUG("[generate_scheduling_result_DL] Generate HARQ node\n");
	/*HARQ scheduling result*/
	// be the first node of UL
	// be the first node of UL
	
	tmp1 = NULL;
	if(schedule_result_list_UL == NULL)
	{
	  //schedule_result_list_UL = (schedule_result_t*)malloc(sizeof(schedule_result_t));
	  schedule_result_list_UL = HARQ_result;
	}
	else
	{
		tmp = schedule_result_list_UL;
		while(tmp!=NULL)
		{
			if(HARQ_subframe < tmp->output_subframe)
			{
				break;
			}
			//DEBUG("[generate_scheduling_result_DL] node output subframe %d at HARQ part\n", tmp->output_subframe);
			tmp1 = tmp;
			tmp = tmp->next;
		}
		if(tmp==NULL)
		{
			tmp1->next = HARQ_result;
		}
		else
		{
			HARQ_result->next = tmp;
			if(tmp1)
			{
				tmp1->next = HARQ_result;
			}else
			{
				schedule_result_list_UL = HARQ_result;
			}
		}
	}
	/*
	DEBUG("---[generate_scheduling_result_DL] schedule result after generate---\n");
	tmp = schedule_result_list_DL;
	while(tmp!=NULL)
	{
		DEBUG("[generate_scheduling_result_DL] node output subframe %d\n", tmp->output_subframe);
		tmp = tmp->next;
	}
	*/
}

void maintain_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, sched_temp_DL_NB_IoT_t *NPDCCH_info, sched_temp_DL_NB_IoT_t *NPDSCH_info)
{
	available_resource_DL_t *temp;
	uint8_t flag_same=0;
	int align_left;
	int align_right;
	uint32_t H_temp, f_temp, sf_temp;
  uint32_t H_temp_r, f_temp_r, sf_temp_r;

	if(NPDSCH_info==NULL)
	{
		/****Maintain NPDCCH node*******/
		//	divided into two node
		//	keep one node(align left or right)
		//	delete node
		convert_system_number(NPDCCH_info->node->start_subframe, &H_temp, &f_temp, &sf_temp);
		//align_left = (calculate_DLSF(mac_inst, NPDCCH_info->node->start_subframe, NPDCCH_info->sf_start) == 0);
		align_left=(cal_num_dlsf(mac_inst, H_temp, f_temp, sf_temp, &H_temp_r, &f_temp_r, &sf_temp_r, 1)==NPDCCH_info->sf_start);
		align_right = ((calculate_DLSF(mac_inst, NPDCCH_info->sf_end, NPDCCH_info->node->end_subframe) == 0)||(NPDCCH_info->sf_end==NPDCCH_info->node->end_subframe));
		//align_left = (calculate_DLSF(mac_inst, NPDCCH_info->node->start_subframe, NPDCCH_info->sf_start) == 0);
		//DEBUG("[maintain_resource_DL] align left %d align right %d\n", align_left, align_right);
		switch(align_left+align_right)
		{
			case 0:
				//	divided into two node
				temp = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
				temp->next = NPDCCH_info->node->next;
				temp->prev = NPDCCH_info->node;
				NPDCCH_info->node->next = temp;

				temp->start_subframe = NPDCCH_info->sf_end+1;
				temp->end_subframe = NPDCCH_info->node->end_subframe;

				NPDCCH_info->node->end_subframe = NPDCCH_info->sf_start - 1;
				break;
			case 1:
				//	keep one node
				if(align_left)
				{
					NPDCCH_info->node->start_subframe = NPDCCH_info->sf_end+1;
				}
				else
				{
					NPDCCH_info->node->end_subframe = NPDCCH_info->sf_start-1 ;
				}
				break;
			case 2:
				//	delete
				if(NPDCCH_info->node->prev==NULL)
        {
          available_resource_DL = NPDCCH_info->node->next;
        }
        else
        {
          NPDCCH_info->node->prev->next = NPDCCH_info->node->next;
          if(NPDCCH_info->node->next!=NULL)
          {
            NPDCCH_info->node->next->prev = NPDCCH_info->node->prev;
          }
        }
				free(NPDCCH_info->node);
				break;
			default:
				//error
				break;
      free(NPDCCH_info);
		}
	}
	else
	{
    if(NPDCCH_info->node==NPDSCH_info->node)
		{
			flag_same=1;
			printf("[%04d][maintain_resource_DL] NPDCCH and NPDSCH using the same node\n", mac_inst->current_subframe);
		}
		/****Maintain NPDCCH node*******/
		//	divided into two node
		//	keep one node(align left or right)
		//	delete node

		convert_system_number(NPDCCH_info->node->start_subframe, &H_temp, &f_temp, &sf_temp);
		//align_left = (calculate_DLSF(mac_inst, NPDCCH_info->node->start_subframe, NPDCCH_info->sf_start) == 0);
		align_left=(cal_num_dlsf(mac_inst, H_temp, f_temp, sf_temp, &H_temp_r, &f_temp_r, &sf_temp_r, 1)==NPDCCH_info->sf_start);
		align_right = ((calculate_DLSF(mac_inst, NPDCCH_info->sf_end, NPDCCH_info->node->end_subframe) == 0)||(NPDCCH_info->sf_end==NPDCCH_info->node->end_subframe));
		//DEBUG("[maintain_resource_DL] align left %d align right %d\n", align_left, align_right);
		switch(align_left+align_right)
		{
			case 0:
				//	divided into two node
				temp = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
				
				//	calvin added
				if((available_resource_DL_t *)0 == NPDCCH_info->node->next){
					available_resource_DL_last = temp;
				}
				
				temp->next = NPDCCH_info->node->next;
				temp->prev = NPDCCH_info->node;
				NPDCCH_info->node->next = temp;

				temp->start_subframe = NPDCCH_info->sf_end+1;
				temp->end_subframe = NPDCCH_info->node->end_subframe;

				NPDCCH_info->node->end_subframe = NPDCCH_info->sf_start - 1;
				if(flag_same==1)
				{
					NPDSCH_info->node = temp;
				}
				break;
			case 1:
				//	keep one node
				if(align_left)
				{
					NPDCCH_info->node->start_subframe = NPDCCH_info->sf_end+1;
					printf("[%04d][maintain_resource_DL] NPDCCH keep one node\n", mac_inst->current_subframe);
				}
				else
				{
					NPDCCH_info->node->end_subframe = NPDCCH_info->sf_start-1 ;
				}
				break;
			case 2:
				//	delete
				printf("[%04d][maintain_resource_DL] NPDCCH delete node\n", mac_inst->current_subframe);
				
				//	calvin add
				if((available_resource_DL_t *)0 == NPDCCH_info->node->next){
					available_resource_DL_last = NPDCCH_info->node->prev;
				}
				
        if(NPDCCH_info->node->prev==NULL)
        {
          available_resource_DL = NPDCCH_info->node->next;
        }
        else
        {
          NPDCCH_info->node->prev->next = NPDCCH_info->node->next;
          if(NPDCCH_info->node->next!=NULL)
          {
            NPDCCH_info->node->next->prev = NPDCCH_info->node->prev;
          }
        }
				free(NPDCCH_info->node);
				break;
			default:
				//error
				break;
		}
		/****Maintain NPDSCH node*******/
		align_left = (calculate_DLSF(mac_inst, NPDSCH_info->node->start_subframe, NPDSCH_info->sf_start) == 0);
		align_right = ((calculate_DLSF(mac_inst, NPDSCH_info->sf_end, NPDSCH_info->node->end_subframe) == 0)||(NPDSCH_info->sf_end==NPDSCH_info->node->end_subframe));
		switch(align_left+align_right)
		{
			case 0:
				//	divided into two node
				temp = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
				
				//	calvin added
				if((available_resource_DL_t *)0 == NPDSCH_info->node->next){
					available_resource_DL_last = temp;
				}
				
				temp->next = NPDSCH_info->node->next;
				temp->prev = NPDSCH_info->node;
				NPDSCH_info->node->next = temp;

				temp->start_subframe = NPDSCH_info->sf_end+1;
				temp->end_subframe = NPDSCH_info->node->end_subframe;

				NPDSCH_info->node->end_subframe = NPDSCH_info->sf_start - 1;

				break;
			case 1:
				//	keep one node
				if(align_left)
				{
					NPDSCH_info->node->start_subframe = NPDSCH_info->sf_end+1;
				}
				else
				{
					NPDSCH_info->node->end_subframe = NPDSCH_info->sf_start-1 ;
				}
				break;
			case 2:
				//	delete
				
				//	calvin added
				if((available_resource_DL_t *)0 == NPDSCH_info->node->next){
					available_resource_DL_last = NPDSCH_info->node->prev;
				}
				
				if(NPDSCH_info->node->prev==NULL)
        {
          available_resource_DL = NPDSCH_info->node->next;
        }
        else
        {
          NPDSCH_info->node->prev->next = NPDSCH_info->node->next;
          if(NPDSCH_info->node->next!=NULL)
          {
            NPDSCH_info->node->next->prev = NPDSCH_info->node->prev;
          }
        }
				free(NPDSCH_info->node);
				break;
			default:
				//error
				break;
		}
		free(NPDCCH_info);
		free(NPDSCH_info);
	}
}

uint8_t get_index_Rep_dl(uint16_t R)
{
  int i;
  if(R<=128)
  {
    for(i=0;i<16;++i)
    {
      if(R==R_dl_table[i])
      {
        return i;
      }
    }
    printf("[get_index_Rep] error\n");
  }
  return -1;
}