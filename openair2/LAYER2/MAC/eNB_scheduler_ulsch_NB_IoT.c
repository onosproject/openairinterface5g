
/*! \file eNB_scheduler_ulsch_NB_IoT.c
 * \brief handle UL UE-specific scheduling
 * \author  NTUST BMW Lab./
 * \date 2017
 * \email: 
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"


unsigned char str20[] = "DCI_uss";
unsigned char str21[] = "DATA_uss";

// scheduling UL
int schedule_UL_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst,UE_TEMPLATE_NB_IoT *UE_info,uint32_t subframe, uint32_t frame, uint32_t H_SFN){

	int i,ndi = 0,check_DCI_result = 0,check_UL_result = 0,candidate;
	uint32_t DL_end;
    //Scheduling resource temp buffer
    sched_temp_DL_NB_IoT_t *NPDCCH_info = (sched_temp_DL_NB_IoT_t*)malloc(sizeof(sched_temp_DL_NB_IoT_t));

	candidate = UE_info->R_max/UE_info->R_dci;
    uint32_t mcs = max_mcs[UE_info->multi_tone];
    uint32_t mappedMcsIndex=UE_info->PHR+(4 * UE_info->multi_tone);
    int TBS = 0;
    int Iru = 0, Nru, I_rep,N_rep,total_ru;
    int dly = 0,uplink_time = 0;

    TBS=get_TBS_UL_NB_IoT(mcs,UE_info->multi_tone,Iru);

    sched_temp_UL_NB_IoT_t *NPUSCH_info = (sched_temp_UL_NB_IoT_t*)malloc(sizeof(sched_temp_UL_NB_IoT_t));

    DCIFormatN0_t *DCI_N0 = (DCIFormatN0_t*)malloc(sizeof(DCIFormatN0_t));

    //available_resource_DL_t *node;

    // setting of the NDI
    if(UE_info->HARQ_round == 0)
    {
        ndi = 1-UE_info->oldNDI_UL;
        UE_info->oldNDI_UL=ndi;
    }

    for (i = 0; i < candidate; i++)
	{
		/*step 1 : Check DL resource is available for DCI N0 or not*/
		check_DCI_result = check_resource_NPDCCH_NB_IoT(mac_inst,H_SFN, frame, subframe, NPDCCH_info, i, UE_info->R_dci);

        //node = check_resource_DL(mac_inst,);

        //just use to check when there is no DL function
        //NPDCCH_info->sf_start = H_SFN*10240+frame*10 +subframe + i * UE_info->R_dci;
        //NPDCCH_info->sf_end = NPDCCH_info->sf_start + (i+1) * UE_info->R_dci;

        //DEBUG("UE : %5d, NPDCCH result: %d ,NPDCCH start: %d,NPDCCH end : %d\n",UE_info->rnti,check_DCI_result,NPDCCH_info->sf_start,NPDCCH_info->sf_end);
        if( check_DCI_result != -1)
		{
			/*step 2 : Determine MCS / TBS / REP / RU number*/
            /*while((mapped_mcs[UE_info->CE_level][mappedMcsIndex]< mcs)||((TBS>UE_info->ul_total_buffer)&&(mcs>=0)))
                {
                    --mcs;
                    TBS=get_TBS_UL(mcs,UE_info->multi_tone,Iru);
                }*/

            mcs = mapped_mcs[UE_info->CE_level][mappedMcsIndex];

            while((TBS<UE_info->ul_total_buffer)&&(Iru<=7))
                {
                    Iru++;
                    TBS=get_TBS_UL_NB_IoT(mcs,UE_info->multi_tone,Iru);
                }

            //DEBUG("TBS : %d UL_buffer: %d\n", TBS, UE_info->ul_total_buffer);

            Nru = RU_table[Iru];
            DL_end = NPDCCH_info->sf_end;
            N_rep = get_N_REP(UE_info->CE_level);
            I_rep = get_I_REP(N_rep);
            total_ru = Nru * N_rep;

            printf("[%04d][UL scheduler][UE:%05d] Multi-tone:%d,MCS:%d,TBS:%d,UL_buffer:%d,DL_start:%d,DL_end:%d,N_rep:%d,N_ru:%d,Total_ru:%d\n", mac_inst->current_subframe,UE_info->rnti,UE_info->multi_tone,mcs,TBS,UE_info->ul_total_buffer,NPDCCH_info->sf_start,DL_end,N_rep,Nru,total_ru);

            /*step 3 Check UL resource for Uplink data*/
			// we will loop the scheduling delay here
            for(dly=0;dly<4;dly++)
            {
                uplink_time = DL_end +scheduling_delay[dly];
                check_UL_result = Check_UL_resource(uplink_time,total_ru, NPUSCH_info, UE_info->multi_tone, 0);
                if (check_UL_result != -1)
                {
                    // step 4 : generate DCI content
                    DCI_N0->type = 0;
                    DCI_N0->scind = NPUSCH_info->subcarrier_indication;
                    DCI_N0->ResAssign = Iru;
                    DCI_N0->mcs = mcs;
                    DCI_N0->ndi = ndi;
                    DCI_N0->Scheddly = dly;
                    DCI_N0->RepNum = I_rep;
                    DCI_N0->rv = (UE_info->HARQ_round%2==0)?0:1; // rv will loop 0 & 2
                    DCI_N0->DCIRep = get_DCI_REP(UE_info->R_dci,UE_info->R_max);

                printf("[%04d][UL scheduler][UE:%05d] DCI content = scind : %d ResAssign : %d mcs : %d ndi : %d scheddly : %d RepNum : %d rv : %d DCIRep : %d\n", mac_inst->current_subframe,UE_info->rnti,DCI_N0->scind,DCI_N0->ResAssign,DCI_N0->mcs,DCI_N0->ndi,DCI_N0->Scheddly,DCI_N0->RepNum,DCI_N0->rv,DCI_N0->DCIRep);
                // step 5 resource allocation and generate scheduling result
                generate_scheduling_result_UL(NPDCCH_info->sf_start, NPDCCH_info->sf_end,NPUSCH_info->sf_start, NPUSCH_info->sf_end,DCI_N0,UE_info->rnti, str20, str21);
                //fill_resource_DL();
                maintain_resource_DL(mac_inst,NPDCCH_info,NULL);

                adjust_UL_resource_list(NPUSCH_info);

                return 0;
                }
            }

		}
        /*break now, we only loop one candidiate*/
        //break;
	}

	printf("[%04d][UL scheduler][UE:%05d] there is no available UL resource\n", mac_inst->current_subframe, UE_info->rnti);
	return -1;
}

int single_tone_ru_allocation(uint32_t uplink_time, int total_ru, sched_temp_UL_NB_IoT_t *NPUSCH_info, int fmt2_flag)
{
    available_resource_UL_t *single_node_tmp;
    uint32_t uplink_time_end;

    if(fmt2_flag == 0)
        // 16 * 0.5 (slot) = 8 subframe
        uplink_time_end = uplink_time + total_ru*8 -1;
    else
        // 4 * 0.5 (slot) = 2 subframe
        uplink_time_end = uplink_time + total_ru*2 -1;

    //check first list of single tone
    single_node_tmp = available_resource_UL->singletone1_Head;

    while(single_node_tmp!=NULL)
    {
        if (uplink_time >= single_node_tmp->start_subframe)
        {
            
            if ( uplink_time_end <= single_node_tmp->end_subframe)
            {
                NPUSCH_info->sf_end = uplink_time_end;
                NPUSCH_info->sf_start = uplink_time;
                NPUSCH_info->tone = singletone1;
                NPUSCH_info->subcarrier_indication = 0 ; // Isc when single tone : 0-2
                NPUSCH_info->node = single_node_tmp;
                printf("[UL scheduler] Use uplink resource single tone 1, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        single_node_tmp = single_node_tmp->next;
    }

    //check second list of single tone
    single_node_tmp = available_resource_UL->singletone2_Head;

    while(single_node_tmp!=NULL)
    {
        if (uplink_time >= single_node_tmp->start_subframe)
        {
            if ( uplink_time_end <= single_node_tmp->end_subframe)
            {
                NPUSCH_info->sf_end = uplink_time_end;
                NPUSCH_info->sf_start = uplink_time;
                NPUSCH_info->tone = singletone2;
                NPUSCH_info->subcarrier_indication = 1 ; // Isc when single tone : 0-2
                NPUSCH_info->node = single_node_tmp;
                printf("[UL scheduler] Use uplink resource single tone 2, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        single_node_tmp = single_node_tmp->next;
    }

    //check third list of single tone
    single_node_tmp = available_resource_UL->singletone3_Head;

    while(single_node_tmp!=NULL)
    {
        if (uplink_time >= single_node_tmp->start_subframe)
        {
            if ( uplink_time_end <= single_node_tmp->end_subframe)
            {
                NPUSCH_info->sf_end = uplink_time_end;
                NPUSCH_info->sf_start = uplink_time;
                NPUSCH_info->tone = singletone3;
                NPUSCH_info->subcarrier_indication = 2 ; // Isc when single tone : 0-2
                NPUSCH_info->node = single_node_tmp;
                printf("[UL scheduler]Use uplink resource single tone 3, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        single_node_tmp = single_node_tmp->next;
    }

    //DEBUG("[UL scheduler][singletone]no proper resource for this allocation\n");
    return -1;

}

int multi_tone_ru_allocation(uint32_t uplink_time, int total_ru, sched_temp_UL_NB_IoT_t *NPUSCH_info)
{
    available_resource_UL_t *Next_Node;
    int single_tone_result = -1;
    uint32_t uplink_time_end;
    /*This checking order may result in the different of the resource optimization*/
    /*check 6 tones first*/
    Next_Node = available_resource_UL->sixtone_Head;

    // 4 * 0.5 (slot) = 2 subframe
    uplink_time_end = uplink_time + total_ru*2 -1;
    while(Next_Node!=NULL)
    {
        if (uplink_time >= Next_Node->start_subframe)
        {
            if ( uplink_time_end <= Next_Node->end_subframe)
            {
                NPUSCH_info->sf_end = uplink_time_end;
                NPUSCH_info->sf_start = uplink_time;
                NPUSCH_info->tone = sixtone;
                NPUSCH_info->subcarrier_indication = 17 ; // Isc when 6 tone : 6 - 12
                NPUSCH_info->node = Next_Node;
                printf("[UL scheduler] Use uplink resource six tone, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        Next_Node = Next_Node->next;
    }

    /*check 3 tones*/
    Next_Node = available_resource_UL->threetone_Head;
    // 8 * 0.5 (slot) = 4 subframe
    uplink_time_end = uplink_time + total_ru * 4 -1;
    while(Next_Node!=NULL)
    {
        if (uplink_time >= Next_Node->start_subframe)
        {
            if ( uplink_time_end <= Next_Node->end_subframe)
            {
                NPUSCH_info->sf_end = uplink_time_end;
                NPUSCH_info->sf_start = uplink_time;
                NPUSCH_info->tone = threetone;
                NPUSCH_info->subcarrier_indication = 13 ; // Isc when 3 tone : 3-5
                NPUSCH_info->node = Next_Node;
                printf("[UL scheduler] Use uplink resource three tone, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        Next_Node = Next_Node->next;
    }

    /*if there is no multi-tone resource, try to allocate the single tone resource*/
    single_tone_result = single_tone_ru_allocation(uplink_time,total_ru,NPUSCH_info,0);
    if(single_tone_result == 0)
        return 0;

    //DEBUG("[UL scheduler][multi_tone]there is no available UL resource !\n");
    return -1;
}

int Check_UL_resource(uint32_t uplink_time, int total_ru, sched_temp_UL_NB_IoT_t *NPUSCH_info, int multi_tone, int fmt2_flag)
{

    int result =-1;
    if(fmt2_flag ==0)
    {
        if(multi_tone == 1)
            result = multi_tone_ru_allocation(uplink_time, total_ru, NPUSCH_info);
        else if(multi_tone == 0)
           result = single_tone_ru_allocation(uplink_time, total_ru, NPUSCH_info,0);

    }else if (fmt2_flag == 1)
        {
            result = single_tone_ru_allocation(uplink_time, total_ru, NPUSCH_info, 1);
            printf("harq result %d, time:%d total ru:%d\n", result, uplink_time, total_ru);
            //if(result == 0)
                    //NPUSCH_info->ACK_NACK_resource_field = get_resource_field_value(NPUSCH_info->subcarrier_indication,ack_nack_delay[i]);
                    //DEBUG("[UL scheduler] There is available resource for ACK / NACK\n");
        }
    if(result == 0)
    {
        return 0;
    }
    //DEBUG("[UL scheduler] no available UL resource\n");
    return -1;
}


void generate_scheduling_result_UL(int32_t DCI_subframe, int32_t DCI_end_subframe, uint32_t UL_subframe, uint32_t UL_end_subframe, DCIFormatN0_t *DCI_pdu, rnti_t rnti, uint8_t *ul_debug_str, uint8_t *dl_debug_str){

    // create the schedule result node for this time transmission
    schedule_result_t *UL_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));
    schedule_result_t *DL_result;
    schedule_result_t *tmp1, *tmp;

	UL_result->direction = UL;
    UL_result->output_subframe = UL_subframe;
	UL_result->end_subframe = UL_end_subframe;
    UL_result->DCI_pdu = DCI_pdu;
    UL_result->npusch_format = 0;
    UL_result->DCI_release = 1;
    UL_result->channel = NPUSCH;
    UL_result->rnti = rnti;
    UL_result->next = NULL;
	UL_result->debug_str = ul_debug_str;
	
    if(-1 == DCI_subframe){
        printf("[UL scheduler][UE:%05d] UL_result = output subframe : %d\n", rnti, UL_result->output_subframe);

    }else{
        DL_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));

        DL_result->output_subframe = DCI_subframe;
        DL_result->end_subframe = DCI_end_subframe;
		DL_result->DCI_pdu = DCI_pdu;
        DL_result->DCI_release = 0;
        DL_result->direction = UL;
        DL_result->channel = NPDCCH;
        DL_result->rnti = rnti;
        DL_result->next = NULL;
		DL_result->debug_str = dl_debug_str;
		
	insert_schedule_result(&schedule_result_list_DL, DCI_subframe, DL_result);
        
        printf("[UL scheduler][UE:%05d] DL_result = output subframe : %d UL_result = output subframe : %d\n", rnti, DL_result->output_subframe,UL_result->output_subframe);
    }

    tmp1 = NULL;

    // be the first node of UL
    if(schedule_result_list_UL == NULL)
    {
        //schedule_result_list_UL = (schedule_result_t*)malloc(sizeof(schedule_result_t));
        schedule_result_list_UL = UL_result;
    }else
    {
        tmp = schedule_result_list_UL;
				while(tmp!=NULL)
				{
					if(UL_subframe < tmp->output_subframe)
					{
						break;
					}
					tmp1 = tmp;
					tmp = tmp->next;
				}
				if(tmp==NULL)
				{
					tmp1->next = UL_result;
				}
				else
				{
					UL_result->next = tmp;
					if(tmp1){
						tmp1->next = UL_result;
					}else{
						schedule_result_list_UL = UL_result;
					}
				}
    }

}

void adjust_UL_resource_list(sched_temp_UL_NB_IoT_t *NPUSCH_info)
{
	available_resource_UL_t *temp;
	available_resource_UL_t *node = NPUSCH_info->node;
	//	divided into two node
	//	keep one node(align left or right)
	//	delete node
	int align_left = (node->start_subframe==NPUSCH_info->sf_start);
	int align_right = (node->end_subframe==NPUSCH_info->sf_end);

	switch(align_left+align_right){
		case 0:
			//	divided into two node
			temp = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));

			temp->next = node->next;
			node->next = temp;

			temp->start_subframe = NPUSCH_info->sf_end +1;
			temp->end_subframe = node->end_subframe;

			node->end_subframe = NPUSCH_info->sf_start - 1;

			break;
		case 1:
			//	keep one node
			if(align_left){
				node->start_subframe = NPUSCH_info->sf_end +1;
			}else{
				node->end_subframe = NPUSCH_info->sf_start - 1 ;
			}
			break;
		case 2:
			//	delete
			node->prev->next = node->next;
			node->next->prev = node->prev;
			free(node);
			break;
		default:
			//error
			break;
	}

//	free(NPUSCH_info);
}

uint8_t *parse_ulsch_header_NB_IoT( uint8_t *mac_header,
                             uint8_t *num_ce,
                             uint8_t *num_sdu,
                             uint8_t *rx_ces,
                             uint8_t *rx_lcids,
                             uint16_t *rx_lengths,
                             uint16_t tb_length ){

uint8_t not_done=1, num_ces=0, num_sdus=0, lcid,num_sdu_cnt;
uint8_t *mac_header_ptr = mac_header;
uint16_t length, ce_len=0;

  while(not_done==1){

    if(((SCH_SUBHEADER_FIXED_NB_IoT*)mac_header_ptr)->E == 0){
      not_done = 0;
    }

    lcid = ((SCH_SUBHEADER_FIXED_NB_IoT*)mac_header_ptr)->LCID;

    if(lcid < EXTENDED_POWER_HEADROOM){
      if (not_done==0) { // last MAC SDU, length is implicit
        mac_header_ptr++;
        length = tb_length-(mac_header_ptr-mac_header)-ce_len;

        for(num_sdu_cnt=0; num_sdu_cnt < num_sdus ; num_sdu_cnt++){
          length -= rx_lengths[num_sdu_cnt];
        }
      }else{
        if(((SCH_SUBHEADER_SHORT_NB_IoT *)mac_header_ptr)->F == 0){
          length = ((SCH_SUBHEADER_SHORT_NB_IoT *)mac_header_ptr)->L;
          mac_header_ptr += 2;//sizeof(SCH_SUBHEADER_SHORT);
        }else{ // F = 1
          length = ((((SCH_SUBHEADER_LONG_NB_IoT *)mac_header_ptr)->L_MSB & 0x7f ) << 8 ) | (((SCH_SUBHEADER_LONG_NB_IoT *)mac_header_ptr)->L_LSB & 0xff);
          mac_header_ptr += 3;//sizeof(SCH_SUBHEADER_LONG);
        }
      }

      rx_lcids[num_sdus] = lcid;
      rx_lengths[num_sdus] = length;
      num_sdus++;
    }else{ // This is a control element subheader POWER_HEADROOM, BSR and CRNTI
      if(lcid == SHORT_PADDING){
        mac_header_ptr++;
      }else{
        rx_ces[num_ces] = lcid;
        num_ces++;
        mac_header_ptr++;

        if(lcid==LONG_BSR){
          ce_len+=3;
        }else if(lcid==CRNTI){
          ce_len+=2;
        }else if((lcid==POWER_HEADROOM) || (lcid==TRUNCATED_BSR)|| (lcid== SHORT_BSR)) {
          ce_len++;
        }else{
          // wrong lcid
        }
      }
    }
  }

  *num_ce = num_ces;
  *num_sdu = num_sdus;

  return(mac_header_ptr);
}


void rx_sdu_NB_IoT(module_id_t module_id, int CC_id, frame_t frame, sub_frame_t subframe, uint16_t rnti, uint8_t *sdu, uint16_t  length)
{
    unsigned char  rx_ces[5], num_ce = 0, num_sdu = 0, *payload_ptr, i; // MAX Control element
    unsigned char  rx_lcids[5];//for NB_IoT-IoT, NB_IoT_RB_MAX should be fixed to 5 (2 DRB+ 3SRB) 
  unsigned short rx_lengths[5];
  //int UE_id = 0;
  int BSR_index=0;
  int DVI_index = 0;
  int PHR = 0;
  int ul_total_buffer = 0;
  //mac_NB_IoT_t *mac_inst;
  UE_TEMPLATE_NB_IoT *UE_info;

  //mac_inst = get_mac_inst(module_id);

  // note: if lcid < 25 this is sdu, otherwise this is CE
  payload_ptr = parse_ulsch_header_NB_IoT(sdu, &num_ce, &num_sdu,rx_ces, rx_lcids, rx_lengths, length);

  //printf("num_CE= %d, num_sdu= %d, rx_ces[0] = %d, rx_lcids =  %d, rx_lengths[0] = %d, length = %d\n",num_ce,num_sdu,rx_ces[0],rx_lcids[0],rx_lengths[0],length);

  for (i = 0; i < num_ce; i++)
  {
    switch(rx_ces[i])
    {
        case CRNTI:
          // find UE id again, confirm the UE, intial some ue specific parameters
          payload_ptr+=2;
            break;
        case SHORT_BSR:
            // update BSR here
        UE_info = get_ue_from_rnti(mac_inst, rnti);
        BSR_index = payload_ptr[0] & 0x3f;
        UE_info->ul_total_buffer = BSR_table[BSR_index];
            payload_ptr+=1;
            break;
        default:
        printf("Received unknown MAC header (0x%02x)\n", rx_ces[i]);
                break;
        }
    }
    for (i = 0; i < num_sdu; i++)
    {
        switch(rx_lcids[i])
        {
            case CCCH_NB_IoT:
                
                // MSG3 content: |R|R|PHR|PHR|DVI|DVI|DVI|DVI|CCCH payload
                PHR = ((payload_ptr[0] >> 5) & 0x01)*2+((payload_ptr[0]>>4) & 0x01);
                DVI_index = (payload_ptr[0] >>3 & 0x01)*8+ (payload_ptr[0] >>2 & 0x01)*4 + (payload_ptr[0] >>1 & 0x01)*2 +(payload_ptr[0] >>0 & 0x01);
          //printf("DVI_index= %d\n",DVI_index);
                ul_total_buffer = DV_table[DVI_index];
                printf("PHR = %d, ul_total_buffer = %d\n",PHR,ul_total_buffer);
                // go to payload
                payload_ptr+=1; 
                rx_lengths[i]-=1;
                printf("rx_lengths : %d\n", rx_lengths[i]);
                //NB_IoT_mac_rrc_data_ind(payload_ptr,mac_inst,rnti);
                //NB_IoT_receive_msg3(mac_inst,rnti,PHR,ul_total_buffer);
          break;
            case DCCH0_NB_IoT:
            case DCCH1_NB_IoT:
                // UE specific here
                //NB_IoT_mac_rlc_data_ind(payload_ptr,mac_inst,rnti);
            
          break;
            // all the DRBS
            case DTCH0_NB_IoT:
            default:
                //NB_IoT_mac_rlc_data_ind(payload_ptr,mac_inst,rnti);
          break;
        }
        payload_ptr+=rx_lengths[i];
    }

   
}