
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
        LOG_D(MAC,"[sixtone][Node %d] start %d , end %d\n",sixtone_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///threetone
    available_resource = available_resource_UL->threetone_Head;

    while(available_resource!=NULL)
    {
        threetone_num++;
        LOG_D(MAC,"[threetone][Node %d] start %d, end %d\n",threetone_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone1_Head;

    while(available_resource!=NULL)
    {
        singletone1_num++;
        LOG_D(MAC,"[singletone1][Node %d] start %d, end %d\n",singletone1_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone2_Head;

    while(available_resource!=NULL)
    {
        singletone2_num++;
        LOG_D(MAC,"[singletone2][Node %d] start %d, end %d\n",singletone2_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

    ///singletone1
    available_resource = available_resource_UL->singletone3_Head;

    while(available_resource!=NULL)
    {
        singletone3_num++;
        LOG_D(MAC,"[singletone3][Node %d] start %d, end %d\n",singletone3_num,available_resource->start_subframe,available_resource->end_subframe);
        available_resource = available_resource->next;
    }

}

void print_scheduling_result_UL(void)
{
    schedule_result_t *scheduling_result_tmp;

    scheduling_result_tmp = schedule_result_list_DL;

    while(scheduling_result_tmp!=NULL)
    {
        LOG_D(MAC,"[UE:%05d][%s] output subframe : %d\n", scheduling_result_tmp->rnti, ((scheduling_result_tmp->channel==NPDCCH)? "NPDCCH":"NPDSCH"), scheduling_result_tmp->output_subframe);
        scheduling_result_tmp = scheduling_result_tmp->next;
    }

    scheduling_result_tmp = schedule_result_list_UL;

    while(scheduling_result_tmp!=NULL)
    {
        LOG_D(MAC,"[UE:%05d][NPUSCH] output subframe : %d\n", scheduling_result_tmp->rnti, scheduling_result_tmp->output_subframe);
        scheduling_result_tmp = scheduling_result_tmp->next;
    }

}


void setting_nprach(){

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

void Initialize_Resource_node(available_resource_UL_t *tone_head, available_resource_UL_t *npusch_frame, int tone)
{
    
    int i=0;
    available_resource_UL_t *second_node;

    second_node = (available_resource_UL_t*)malloc(sizeof(available_resource_UL_t));

    if(tone == sixtone)
        i=2;
    else if(tone == threetone)
        i=1;
    else
        i=0;

    tone_head->start_subframe = ceil ( (nprach_list+i)->nprach_StartTime + 1.4*4*((nprach_list+i)->numRepetitionsPerPreambleAttempt) ) ;
    tone_head->end_subframe = (nprach_list+i)->nprach_StartTime-1 + (nprach_list+i)->nprach_Periodicity;

    second_node->start_subframe = tone_head->start_subframe + (nprach_list+i)->nprach_Periodicity;
    second_node->end_subframe = tone_head->end_subframe + (nprach_list+i)->nprach_Periodicity;
    second_node->next =NULL;
    tone_head->next = second_node;
    *npusch_frame = *tone_head->next;
    
////////////////////////CALVIN TIMING DIAGRAM GENERATOR///////////////////////////
#ifdef TIMING_GENERATOR
    uint32_t ii, jj;
    for(ii=(nprach_list+i)->nprach_StartTime; ii<tone_head->start_subframe; ++ii){
        if(ii == sim_end_time) break;
        for(jj=0; jj<(nprach_list+i)->nprach_NumSubcarriers; ++jj){
            ul_scheduled(ii, 0, (nprach_list+i)->nprach_SubcarrierOffset + jj, _NPRACH, 0, (char *)0);
        }
    }
    for(ii=tone_head->end_subframe+1; ii<second_node->start_subframe; ++ii){
        if(ii == sim_end_time) break;
        for(jj=0; jj<(nprach_list+i)->nprach_NumSubcarriers; ++jj){
            ul_scheduled(ii, 0, (nprach_list+i)->nprach_SubcarrierOffset + jj, _NPRACH, 0, (char *)0);
        }
    }
#endif
////////////////////////CALVIN TIMING DIAGRAM GENERATOR///////////////////////////
}

/*when there is SIB-2 configuration coming to MAC, filled the uplink resource grid*/
void Initialize_Resource(void){
    int i;
    available_resource_UL_t *new_node;
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
    //initialize first node
    if((nprach_list+2)->nprach_StartTime!=0)
    {
        new_node = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));
        new_node->next = (available_resource_UL_t *)0;
        new_node->prev = (available_resource_UL_t *)0;
        new_node->start_subframe = 0;
        new_node->end_subframe = (nprach_list+2)->nprach_StartTime-1;

        if( (available_resource_UL_t *)0 == available_resource_UL->sixtone_Head){
            available_resource_UL->sixtone_Head = new_node;
            new_node->prev = (available_resource_UL_t *)0;
        }
    }
    if((nprach_list+1)->nprach_StartTime!=0)
    {
        new_node = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));
        new_node->next = (available_resource_UL_t *)0;
        new_node->prev = (available_resource_UL_t *)0;
        new_node->start_subframe = 0;
        new_node->end_subframe = (nprach_list+1)->nprach_StartTime-1;

        if( (available_resource_UL_t *)0 == available_resource_UL->threetone_Head){
            available_resource_UL->threetone_Head = new_node;
            new_node->prev = (available_resource_UL_t *)0;
        }
    }
    for(i=0;i<3;++i)
    {
        if(nprach_list->nprach_StartTime!=0)
        {
            new_node = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));
            new_node->next = (available_resource_UL_t *)0;
            new_node->prev = (available_resource_UL_t *)0;
            new_node->start_subframe = 0;
            new_node->end_subframe = nprach_list->nprach_StartTime-1;

            if( (available_resource_UL_t *)0 == available_resource_UL->threetone_Head){
                if(i==0)
                    available_resource_UL->singletone1_Head = new_node;
                else if(i==1)
                    available_resource_UL->singletone2_Head = new_node;
                else
                    available_resource_UL->singletone3_Head = new_node;
                new_node->prev = (available_resource_UL_t *)0;
            }
        }
    }
    add_UL_Resource();
    add_UL_Resource();
    
    LOG_D(MAC,"Initialization of the UL Resource grid has been done\n");
}


void add_UL_Resource_node(available_resource_UL_t **head, uint32_t *end_subframe, uint32_t ce_level){
    available_resource_UL_t *new_node, *iterator;
    new_node = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));
    
    new_node->next = (available_resource_UL_t *)0;

    new_node->prev = (available_resource_UL_t *)0;
    
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
    
////////////////////////CALVIN TIMING DIAGRAM GENERATOR///////////////////////////
#ifdef TIMING_GENERATOR
    uint32_t ii, jj;
    for(ii=*end_subframe+(nprach_list+ce_level)->nprach_StartTime; ii<new_node->start_subframe; ++ii){
        if(ii >= sim_end_time) break;
        for(jj=0; jj<(nprach_list+ce_level)->nprach_NumSubcarriers; ++jj){
            ul_scheduled(ii, 0, (nprach_list+ce_level)->nprach_SubcarrierOffset + jj, _NPRACH, 0, (char *)0);
        }
    }
#endif
////////////////////////CALVIN TIMING DIAGRAM GENERATOR///////////////////////////

    *end_subframe += (nprach_list+ce_level)->nprach_Periodicity;
}

/// Use to extend the UL resource grid (5 list) at the end of nprach peroid time
void add_UL_Resource(void)
{   
    add_UL_Resource_node(&available_resource_UL->sixtone_Head, &available_resource_UL->sixtone_end_subframe, 2);
    add_UL_Resource_node(&available_resource_UL->threetone_Head, &available_resource_UL->threetone_end_subframe, 1);
    add_UL_Resource_node(&available_resource_UL->singletone1_Head, &available_resource_UL->singletone1_end_subframe, 0);
    add_UL_Resource_node(&available_resource_UL->singletone2_Head, &available_resource_UL->singletone2_end_subframe, 0);
    add_UL_Resource_node(&available_resource_UL->singletone3_Head, &available_resource_UL->singletone3_end_subframe, 0);
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
    return TBS>>3;
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
        //N_rep = (nprach_list+2)->numRepetitionsPerPreambleAttempt;
        N_rep = 1;
    }else
    {
        LOG_D(MAC,"unknown CE level!\n");
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
    LOG_D(MAC,"unknown repetition value!\n");
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
                LOG_D(MAC,"[UL scheduler] Use uplink resource single tone 1, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
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
                LOG_D(MAC,"[UL scheduler] Use uplink resource single tone 2, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
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
                LOG_D(MAC,"[UL scheduler]Use uplink resource single tone 3, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        single_node_tmp = single_node_tmp->next;
    }

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
                LOG_D(MAC,"[UL scheduler] Use uplink resource six tone, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
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
                LOG_D(MAC,"[UL scheduler] Use uplink resource three tone, sf_start: %d, sf_end: %d\n",NPUSCH_info->sf_start,NPUSCH_info->sf_end);
                return 0;
            }
        }
        Next_Node = Next_Node->next;
    }

    /*if there is no multi-tone resource, try to allocate the single tone resource*/
    single_tone_result = single_tone_ru_allocation(uplink_time,total_ru,NPUSCH_info,0);
    if(single_tone_result == 0)
        return 0;

    return -1;
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
            LOG_D(MAC,"harq result %d, time:%d total ru:%d\n", result, uplink_time, total_ru);
        }
    if(result == 0)
    {
        return 0;
    }
    return -1;
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

void generate_scheduling_result_UL(int32_t DCI_subframe, int32_t DCI_end_subframe, uint32_t UL_subframe, uint32_t UL_end_subframe, DCIFormatN0_t *DCI_inst, rnti_t rnti, uint8_t *ul_printf_str, uint8_t *dl_printf_str, uint8_t msg3_flag){

    // create the schedule result node for this time transmission
    schedule_result_t *UL_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));
    schedule_result_t *DL_result;
    schedule_result_t *tmp1, *tmp;

    UL_result->direction = UL;
    UL_result->output_subframe = UL_subframe;
    UL_result->end_subframe = UL_end_subframe;
    UL_result->DCI_pdu = DCI_inst;
    UL_result->npusch_format = 0;
    UL_result->DCI_release = 1;
    UL_result->channel = NPUSCH;
    UL_result->rnti = rnti;
    UL_result->msg3_flag = msg3_flag;
    UL_result->next = NULL;
    //UL_result->printf_str = ul_printf_str;
    
    if(-1 == DCI_subframe){
        LOG_D(MAC,"[UL scheduler][UE:%05d] UL_result = output subframe : %d\n", rnti, UL_result->output_subframe);

    }else{
        DL_result = (schedule_result_t*)malloc(sizeof(schedule_result_t));

        DL_result->output_subframe = DCI_subframe;
        DL_result->end_subframe = DCI_end_subframe;
        DL_result->DCI_pdu = DCI_inst;
        DL_result->DCI_release = 0;
        DL_result->direction = UL;
        DL_result->channel = NPDCCH;
        DL_result->rnti = rnti;
        DL_result->next = NULL;
        //DL_result->printf_str = dl_printf_str;
        
    insert_schedule_result(&schedule_result_list_DL, DCI_subframe, DL_result);
        
        LOG_D(MAC,"[UL scheduler][UE:%05d] DL_result = output subframe : %d UL_result = output subframe : %d\n", rnti, DL_result->output_subframe,UL_result->output_subframe);
    }

    tmp1 = NULL;

    // be the first node of UL
    if(schedule_result_list_UL == NULL)
    {
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
    //  divided into two node
    //  keep one node(align left or right)
    //  delete node
    int align_left = (node->start_subframe==NPUSCH_info->sf_start);
    int align_right = (node->end_subframe==NPUSCH_info->sf_end);

    switch(align_left+align_right){
        case 0:
            //  divided into two node
            temp = (available_resource_UL_t *)malloc(sizeof(available_resource_UL_t));

            temp->next = node->next;
            node->next = temp;

            temp->prev = node;

            temp->start_subframe = NPUSCH_info->sf_end +1;
            temp->end_subframe = node->end_subframe;

            node->end_subframe = NPUSCH_info->sf_start - 1;

            break;
        case 1:
            //  keep one node
            if(align_left){
                node->start_subframe = NPUSCH_info->sf_end +1;
            }else{
                node->end_subframe = NPUSCH_info->sf_start - 1 ;
            }
            break;
        case 2:

            if(node!=NULL)
            {
                //  delete
                if(node->prev==(available_resource_UL_t *)0)
                {
                    if(NPUSCH_info->tone==sixtone)
                        available_resource_UL->sixtone_Head = node->next;
                    else if(NPUSCH_info->tone==threetone)
                        available_resource_UL->threetone_Head = node->next;
                    else if(NPUSCH_info->tone==singletone1)
                        available_resource_UL->singletone1_Head = node->next;
                    else if(NPUSCH_info->tone==singletone2)
                        available_resource_UL->singletone2_Head = node->next;
                    else if(NPUSCH_info->tone==singletone3)
                        available_resource_UL->singletone3_Head = node->next;
                }else{

                    node->prev->next = node->next;
                }

                if(node->next!=(available_resource_UL_t *)0)
                {
                    node->next->prev = node->prev;
                }else{
                    node->prev->next = (available_resource_UL_t *)0;
                }

                free(node);
                break;
            }
        default:
            //error
            break;
    }
}

void add_ue_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst, uint16_t rnti, ce_level_t ce, uint32_t PHR, uint32_t ul_total_buffer){
    int32_t i;
    UE_list_NB_IoT_t *UE_list = (mac_inst->UE_list_spec + (uint32_t)ce);
    for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
        if(UE_list->UE_template_NB_IoT[i].active == 0){
            UE_list->UE_template_NB_IoT[i].active = 1;
            UE_list->UE_template_NB_IoT[i].rnti = rnti;
            UE_list->UE_template_NB_IoT[i].PHR = PHR;
            UE_list->UE_template_NB_IoT[i].ul_total_buffer = ul_total_buffer;
            //New UE setting start
            UE_list->UE_template_NB_IoT[i].R_dl = dl_rep[(uint32_t)ce];;
            UE_list->UE_template_NB_IoT[i].I_mcs_dl = 0;
            UE_list->UE_template_NB_IoT[i].CE_level = (uint32_t)ce;
            //assume random select direction
            UE_list->UE_template_NB_IoT[i].R_dci = dci_rep[(uint32_t)ce];
            UE_list->UE_template_NB_IoT[i].R_max = UE_list->NPDCCH_config_dedicated.R_max;
            //UE_list->UE_template_NB_IoT[i].R_max = 16;
            UE_list->UE_template_NB_IoT[i].R_harq = harq_rep[(uint32_t)ce];
            UE_list->UE_template_NB_IoT[i].HARQ_round = 0;
            UE_list->UE_template_NB_IoT[i].oldNDI_UL = 0;
            UE_list->UE_template_NB_IoT[i].oldNDI_DL = 0;

            UE_list->UE_template_NB_IoT[i].multi_tone = 0;
            //New UE setting ending
            UE_list->UE_template_NB_IoT[i].prev = -1;
            
            if(-1 == UE_list->head){
                UE_list->UE_template_NB_IoT[i].next = -1;
            }else{
                UE_list->UE_template_NB_IoT[i].next = UE_list->head;
            }
            UE_list->head = i;
            return ;
        }
    }
}

void remove_ue(eNB_MAC_INST_NB_IoT *mac_inst, uint16_t rnti, ce_level_t ce){
    int32_t i;
    UE_list_NB_IoT_t *UE_list = (mac_inst->UE_list_spec + (uint32_t)ce);
    for(i=0; i<MAX_NUMBER_OF_UE_MAX_NB_IoT; ++i){
        if(UE_list->UE_template_NB_IoT[i].active == 1 && UE_list->UE_template_NB_IoT[i].rnti == rnti){
            UE_list->UE_template_NB_IoT[i].active = 0;
            return ;
        }
    }
}

//Transfrom source into hyperSF, Frame, Subframe format
void convert_system_number(uint32_t source_sf,uint32_t *hyperSF, uint32_t *frame, uint32_t *subframe)
{
    if(source_sf>=1024*1024*10)
    {
        source_sf=source_sf%(1024*1024*10);
    }
    *hyperSF = (source_sf/10)/1024;
    *frame = (source_sf/10)%1024;
    *subframe = (source_sf%10240)%10;
}

//Trnasform hyperSF, Frame, Subframe format into subframe unit
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
  //uint8_t flag_printf=0;

  abs_sf_start=convert_system_number_sf(hyperSF, frame, subframe);
  sf_dlsf_index = abs_sf_start%2560%(mac_inst->sib1_period*10);
  dlsf_num_temp = DLSF_information.sf_to_dlsf_table[sf_dlsf_index];
  while(num_dlsf_require>DLSF_information.num_dlsf_per_period)
  {
    //flag_printf=1;
    period_count++;
    num_dlsf_require-=DLSF_information.num_dlsf_per_period;
  }
  abs_sf_end = abs_sf_start+period_count*mac_inst->sib1_period*10;
  //LOG_D(MAC,"[cal_num_dlsf]abs_sf_end %d after loop\n", abs_sf_end);
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
  }
  
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
      //LOG_D(MAC,"[cal_num_dlsf] abs_sf_end is %d mod period =  %d\n", abs_sf_end, abs_sf_end%(mac_inst->sib1_NB_IoT_sched_config.sib1_period*10));
    }
    if(shift_flag==1)
    {
      abs_sf_end +=DLSF_information.dlsf_to_sf_table[dlsf_num_temp+num_dlsf_require-2];
    }
    else
    {
      abs_sf_end +=DLSF_information.dlsf_to_sf_table[dlsf_num_temp+num_dlsf_require-1];
    }

    }
  convert_system_number(abs_sf_end, hyperSF_result, frame_result, subframe_result);
  return abs_sf_end;
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

void init_tool_sib1(eNB_MAC_INST_NB_IoT *mac_inst){
    int i, j;

    //int repetition_pattern = 1;// 1:every2frame, 2:every4frame, 3:every8frame, 4:every16frame
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

uint32_t calculate_DLSF(eNB_MAC_INST_NB_IoT *mac_inst, int abs_start_subframe, int abs_end_subframe){   //LOG_D(MAC,"calcu %p %d %d\n", mac_inst, abs_start_subframe, abs_end_subframe);
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

    if(abs_start_subframe > abs_end_subframe){
        return calculate_DLSF(mac_inst, abs_start_subframe, (MAX_FRAME*10)+9) + calculate_DLSF(mac_inst, 0, abs_end_subframe);
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
    node->end_subframe = 0;
    mac_inst->schedule_subframe_DL = 0;

    //node->end_subframe = mac_inst->rrc_config.si_window_length;
    //mac_inst->schedule_subframe_DL = mac_inst->rrc_config.si_window_length;

    //  init sibs for first si-window
    //schedule_sibs(mac_inst, 0, 0);    //  TODO, check init
}

#if 1
//  extend subframe align to si-period
void extend_available_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, int max_subframe){

    available_resource_DL_t *new_node;
    uint32_t i, i_div_si_window;
    
    LOG_D(MAC,"[extend DL] max_subframe: %d, current schedule subframe: %d\n", max_subframe, mac_inst->schedule_subframe_DL);
    print_available_resource_DL(mac_inst);
    
    if(max_subframe > mac_inst->schedule_subframe_DL){
        //  align to si-period

        max_subframe = ((max_subframe%mac_inst->rrc_config.si_window_length)==0)? max_subframe : (((max_subframe/mac_inst->rrc_config.si_window_length)+1)*mac_inst->rrc_config.si_window_length);
        
        if(mac_inst->schedule_subframe_DL == available_resource_DL_last->end_subframe){
            LOG_D(MAC,"[extend DL] last node is align to schedule_sf_dl\n");

            available_resource_DL_last->end_subframe = max_subframe;
        }else{
            
            LOG_D(MAC,"[extend DL] add new node !\n");
            new_node = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
            new_node->prev= available_resource_DL_last;
            available_resource_DL_last->next = new_node;
            new_node->start_subframe = mac_inst->schedule_subframe_DL+1;
            new_node->end_subframe = max_subframe;
            new_node->next = (available_resource_DL_t *)0;
            available_resource_DL_last = new_node;
        }
        
        //  do schedule sibs after extend.
        for(i=mac_inst->schedule_subframe_DL; i<max_subframe; i+=mac_inst->rrc_config.si_window_length){
            
            i_div_si_window = (i / mac_inst->rrc_config.si_window_length)%256;
            if(-1 != mac_inst->sibs_table[i_div_si_window]){
                LOG_D(MAC,"[sibs%d] %d\n", mac_inst->sibs_table[i_div_si_window], i + (mac_inst->rrc_config.si_radio_frame_offset*10));
                schedule_sibs(mac_inst, mac_inst->sibs_table[i_div_si_window], i + (mac_inst->rrc_config.si_radio_frame_offset*10));    //  add si-radio-frame-offset carried in SIB1
            }
        }

        mac_inst->schedule_subframe_DL = max_subframe;
    }
}
#endif
#if 0
//  extend subframe align to si-period
void extend_available_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, int max_subframe){ //  assume max_subframe is found.
    

    available_resource_DL_t *new_node;
    //int temp;
    uint32_t i, i_div_si_window;
    //uint32_t si_period_div_window;
    //pt = available_resource_DL;
    
    LOG_D(MAC,"[extend DL] max_subframe: %d, current schedule subframe: %d\n", max_subframe, mac_inst->schedule_subframe_DL);
    print_available_resource_DL(mac_inst);
    
    if(max_subframe > mac_inst->schedule_subframe_DL){
        //  align to si-period

        max_subframe = ((max_subframe%mac_inst->rrc_config.si_window_length)==0)? max_subframe : (((max_subframe/mac_inst->rrc_config.si_window_length)+1)*mac_inst->rrc_config.si_window_length);
        
        if(mac_inst->schedule_subframe_DL == available_resource_DL_last->end_subframe){
            LOG_D(MAC,"[extend DL] last node is align to schedule_sf_dl\n");

            available_resource_DL_last->end_subframe = max_subframe;
        }else{
            
            LOG_D(MAC,"[extend DL] add new node !\n");
            new_node = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));

            available_resource_DL_last->next = new_node;
            new_node->start_subframe = mac_inst->schedule_subframe_DL+1;
            new_node->end_subframe = max_subframe;
            new_node->next = (available_resource_DL_t *)0;
            available_resource_DL_last = new_node;
        }
        
        LOG_D(MAC,"sf_dl:%d max:%d siw:%d\n",mac_inst->schedule_subframe_DL,max_subframe,mac_inst->rrc_config.si_window_length);

        //  do schedule sibs after extend.
        for(i=mac_inst->schedule_subframe_DL;i<max_subframe;i+=mac_inst->rrc_config.si_window_length){
            
            i_div_si_window = (i / mac_inst->rrc_config.si_window_length)%256;
                LOG_D(MAC,"[sibs out:%d] schedule_DL:%d i_div_si_window:%d\n", mac_inst->sibs_table[i_div_si_window], i, i_div_si_window);
            if(-1 != mac_inst->sibs_table[i_div_si_window]){
                LOG_D(MAC,"[sibs%d] %d\n", mac_inst->sibs_table[i_div_si_window], i);
                schedule_sibs(mac_inst, mac_inst->sibs_table[i_div_si_window], i);
            }
        }

        mac_inst->schedule_subframe_DL = max_subframe;
    }
    
    return ;
}
#endif
void maintain_available_resource(eNB_MAC_INST_NB_IoT *mac_inst){

    available_resource_DL_t *pfree, *iterator;
    available_resource_UL_t *pfree2, *iterator2;
    schedule_result_t *iterator1;
    if(available_resource_DL != (available_resource_DL_t *)0){
    LOG_D(MAC,"[maintain]current:%d, end:%d\n",mac_inst->current_subframe,available_resource_DL->end_subframe);
    if(mac_inst->current_subframe >= available_resource_DL->end_subframe){
        pfree = available_resource_DL;

        if(available_resource_DL->next == (available_resource_DL_t *)0){
            LOG_D(MAC,"[maintain_available_resource]=====t:%d=====dl resource list next is NULL %d\n", mac_inst->current_subframe, available_resource_DL->end_subframe);
            available_resource_DL = (available_resource_DL_t *)0;
        }else{
            LOG_D(MAC,"[maintain_available_resource]=====t:%d=====dl resource list remove next:%d-%d\n", mac_inst->current_subframe, available_resource_DL->next->start_subframe, available_resource_DL->next->end_subframe);
            available_resource_DL = available_resource_DL->next;
            available_resource_DL->prev = (available_resource_DL_t *)0;
        }
        free((available_resource_DL_t *)pfree);
        
    }else{
        // only update when current subframe bigger than to start subframe
        if(mac_inst->current_subframe > available_resource_DL->start_subframe)
        {
            LOG_D(MAC,"[maintain] update from %d to current %d, ori end %d\n",available_resource_DL->start_subframe,mac_inst->current_subframe,available_resource_DL->end_subframe);
            available_resource_DL->start_subframe = mac_inst->current_subframe;
        }else
            LOG_D(MAC,"[maintain] do nothing\n");
    }
}
    //  UL 
    iterator2 = available_resource_UL->singletone1_Head;
    if(iterator2 != (available_resource_UL_t *)0){
    if(mac_inst->current_subframe > iterator2->end_subframe){
        pfree2 = iterator2;

        available_resource_UL->singletone1_Head = iterator2->next;
        available_resource_UL->singletone1_Head->prev = (available_resource_UL_t *)0;
        
        free((available_resource_UL_t *)pfree2);
        
    }else{
        if(iterator2->start_subframe<mac_inst->current_subframe)
        iterator2->start_subframe = mac_inst->current_subframe;
    }
}
    iterator2 = available_resource_UL->singletone2_Head;
    if(iterator2 != (available_resource_UL_t *)0){
    if(mac_inst->current_subframe > iterator2->end_subframe){
        pfree2 = iterator2;

        available_resource_UL->singletone2_Head = iterator2->next;
        available_resource_UL->singletone2_Head->prev = (available_resource_UL_t *)0;
        
        free((available_resource_UL_t *)pfree2);
        
    }else{
        if(iterator2->start_subframe<mac_inst->current_subframe)
        iterator2->start_subframe = mac_inst->current_subframe;
    }
}
    iterator2 = available_resource_UL->singletone3_Head;
    if(iterator2 != (available_resource_UL_t *)0){
    if(mac_inst->current_subframe > iterator2->end_subframe){
        pfree2 = iterator2;

        available_resource_UL->singletone3_Head = iterator2->next;
        available_resource_UL->singletone3_Head->prev = (available_resource_UL_t *)0;
        
        free((available_resource_UL_t *)pfree2);
        
    }else{
        if(iterator2->start_subframe<mac_inst->current_subframe)
        iterator2->start_subframe = mac_inst->current_subframe;
    }
}
    iterator2 = available_resource_UL->sixtone_Head;
    if(iterator2 != (available_resource_UL_t *)0){
    if(mac_inst->current_subframe > iterator2->end_subframe){
        pfree2 = iterator2;

        available_resource_UL->sixtone_Head = iterator2->next;
        available_resource_UL->sixtone_Head->prev = (available_resource_UL_t *)0;
        
        free((available_resource_UL_t *)pfree2);
        
    }
    else{
        if(iterator2->start_subframe<mac_inst->current_subframe)
        iterator2->start_subframe = mac_inst->current_subframe;
    }
}
    iterator2 = available_resource_UL->threetone_Head;
    if(iterator2 != (available_resource_UL_t *)0){
    if(mac_inst->current_subframe > iterator2->end_subframe){
        pfree2 = iterator2;

        available_resource_UL->threetone_Head = iterator2->next;
        available_resource_UL->threetone_Head->prev = (available_resource_UL_t *)0;
        
        free((available_resource_UL_t *)pfree2);
        
    }else{
        if(iterator2->start_subframe<mac_inst->current_subframe)
        iterator2->start_subframe = mac_inst->current_subframe;
    }
 }   
    if(mac_inst->current_subframe == 0){
        //  DL available cross zero
        iterator = available_resource_DL;
        while(iterator != (available_resource_DL_t *)0){
            if(iterator->start_subframe >= MAX_SUBFRAME)
                iterator->start_subframe -= MAX_SUBFRAME;
            if(iterator->end_subframe >= MAX_SUBFRAME)
                iterator->end_subframe -= MAX_SUBFRAME;
            iterator = iterator->next;
        }
        if(mac_inst->schedule_subframe_DL >= MAX_SUBFRAME)
            mac_inst->schedule_subframe_DL -= MAX_SUBFRAME;
        //  UL available cross zero
        iterator2 = available_resource_UL->sixtone_Head;
        while(iterator2 != (available_resource_UL_t *)0){
            if(iterator2->start_subframe >= MAX_SUBFRAME)
                iterator2->start_subframe -= MAX_SUBFRAME;
            if(iterator2->end_subframe >= MAX_SUBFRAME)
                iterator2->end_subframe -= MAX_SUBFRAME;
            iterator2 = iterator2->next;
        }
        iterator2 = available_resource_UL->threetone_Head;
        while(iterator2 != (available_resource_UL_t *)0){
            if(iterator2->start_subframe >= MAX_SUBFRAME)
                iterator2->start_subframe -= MAX_SUBFRAME;
            if(iterator2->end_subframe >= MAX_SUBFRAME)
                iterator2->end_subframe -= MAX_SUBFRAME;
            iterator2 = iterator2->next;
        }
        iterator2 = available_resource_UL->singletone3_Head;
        while(iterator2 != (available_resource_UL_t *)0){
            if(iterator2->start_subframe >= MAX_SUBFRAME)
                iterator2->start_subframe -= MAX_SUBFRAME;
            if(iterator2->end_subframe >= MAX_SUBFRAME)
                iterator2->end_subframe -= MAX_SUBFRAME;
            iterator2 = iterator2->next;
        }
        iterator2 = available_resource_UL->singletone1_Head;
        while(iterator2 != (available_resource_UL_t *)0){
            if(iterator2->start_subframe >= MAX_SUBFRAME)
                iterator2->start_subframe -= MAX_SUBFRAME;
            if(iterator2->end_subframe >= MAX_SUBFRAME)
                iterator2->end_subframe -= MAX_SUBFRAME;
            iterator2 = iterator2->next;
        }
        iterator2 = available_resource_UL->singletone2_Head;
        while(iterator2 != (available_resource_UL_t *)0){
            if(iterator2->start_subframe >= MAX_SUBFRAME)
                iterator2->start_subframe -= MAX_SUBFRAME;
            if(iterator2->end_subframe >= MAX_SUBFRAME)
                iterator2->end_subframe -= MAX_SUBFRAME;
            iterator2 = iterator2->next;
        }
        
        if(available_resource_UL->singletone1_end_subframe >= MAX_SUBFRAME)
            available_resource_UL->singletone1_end_subframe -= MAX_SUBFRAME;
        if(available_resource_UL->singletone2_end_subframe >= MAX_SUBFRAME)
            available_resource_UL->singletone2_end_subframe -= MAX_SUBFRAME;
        if(available_resource_UL->singletone3_end_subframe >= MAX_SUBFRAME)
            available_resource_UL->singletone3_end_subframe -= MAX_SUBFRAME;
        if(available_resource_UL->sixtone_end_subframe >= MAX_SUBFRAME)
            available_resource_UL->sixtone_end_subframe -= MAX_SUBFRAME;
        if(available_resource_UL->threetone_end_subframe >= MAX_SUBFRAME)
            available_resource_UL->threetone_end_subframe -= MAX_SUBFRAME;
        //  DL result cross zero
        iterator1 = schedule_result_list_DL;
        while(iterator1 != (schedule_result_t *)0){
            if(iterator1->output_subframe >= MAX_SUBFRAME)
                iterator1->output_subframe -= MAX_SUBFRAME;
            if(iterator1->end_subframe >= MAX_SUBFRAME)
                iterator1->end_subframe -= MAX_SUBFRAME;
            iterator1 = iterator1->next;
        }
        //  UL result cross zero
        iterator1 = schedule_result_list_UL;
        while(iterator1 != (schedule_result_t *)0){
            if(iterator1->output_subframe >= MAX_SUBFRAME)
                iterator1->output_subframe -= MAX_SUBFRAME;
            if(iterator1->end_subframe >= MAX_SUBFRAME)
                iterator1->end_subframe -= MAX_SUBFRAME;
            iterator1 = iterator1->next;
        }
    }
    
    
    return ;
}

void fill_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst, available_resource_DL_t *node, int start_subframe, int end_subframe, schedule_result_t *new_node){
    //printf_FUNCTION_IN("[FILL DL]");
    available_resource_DL_t *temp;
    schedule_result_t *iterator, *temp1;
    //  divided into two node
    //  keep one node(align left or right)
    //  delete node

//LOG_D(MAC,"fill dl test1\n"); 
    int align_left = (node->start_subframe==start_subframe)||(calculate_DLSF(mac_inst, node->start_subframe, start_subframe-1) == 0);
    int align_right = (end_subframe==node->end_subframe)||(calculate_DLSF(mac_inst, end_subframe+1, node->end_subframe) == 0);
//LOG_D(MAC,"fill dl test2\n");
    switch(align_left+align_right){
        case 0:
            //  divided into two node, always insert before original node, so won't happen that temp is the last node of the list.
            //  A | node | B
            //  A | temp | node | B
            LOG_D(MAC,"Case 0 [b], node : %p node_prev : %p\n",node,node->prev);
            temp = (available_resource_DL_t *)malloc(sizeof(available_resource_DL_t));
            if(node->prev){
                //LOG_I(MAC,"start_subframe : %d\n",node->prev->start_subframe);
                node->prev->next = temp;
            }else{
                available_resource_DL = temp;
            }
            
            temp->prev = node->prev;
            temp->next = node;
            node->prev = temp;
            
            temp->start_subframe = node->start_subframe;
            temp->end_subframe = start_subframe - 1;

            node->start_subframe = end_subframe + 1;
            LOG_D(MAC,"Case 0 [a], node : %p node_prev : %p\n",node,node->prev);
            break;
        case 1:
            LOG_D(MAC,"Case 1, node : %p node_prev : %p\n",node,node->prev);
            //  keep one node
            if(align_left){
                node->start_subframe = end_subframe + 1 ;
            }else{
                node->end_subframe = start_subframe - 1 ;
            }

            break;
        case 2:
            LOG_D(MAC,"Case 2 [b], node : %p node_prev : %p\n",node,node->prev);
            //  delete
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
            LOG_D(MAC,"Case 2 [a], node : %p node_prev : %p\n",node,node->prev);

            free(node);
            break;
        default:
            //error
            break;
    }
    
    //  new node allocate from up-layer calling function.
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
    
    //printf_FUNCTION_OUT("[FILL DL]");
}

//  check_subframe must be DLSF, you can use is_dlsf() to check before call function
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
            
            LOG_D(MAC,"Diff_gap : %d num_subframes : %d \n",diff_gap,num_subframes);

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

available_resource_DL_t *check_sibs_resource(eNB_MAC_INST_NB_IoT *mac_inst, int check_start_subframe, int check_end_subframe, int num_subframe, int *residual_subframe, int *out_last_subframe, int *out_first_subframe){
    available_resource_DL_t *pt;
    uint32_t num_dlsf;
    uint8_t output = 0x0;
    pt = available_resource_DL;
    
    //  TODO find the pt which can cover part of check_start_subframe, e.g. 1280-> 1281-1440
    while((available_resource_DL_t *)0 != pt){
        if(pt->start_subframe <= check_start_subframe && pt->end_subframe >= check_start_subframe){
            break;
        }
        pt = pt->next;
    }
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

void print_available_resource_DL(eNB_MAC_INST_NB_IoT *mac_inst){
    available_resource_DL_t *pt;
    pt = available_resource_DL;
    int i=0;
    LOG_D(MAC,"=== print available resource === t=%d\nsched subframe: %d, list end: %d-%d\n", mac_inst->current_subframe, mac_inst->schedule_subframe_DL, available_resource_DL_last->start_subframe, available_resource_DL_last->end_subframe);
    while(pt){
        LOG_D(MAC,"[%2d] %p %3d-%3d prev:%p\n", i, pt, pt->start_subframe, pt->end_subframe, pt->prev);
        pt = pt->next;
    }
    LOG_D(MAC,"\n");
}

void print_schedule_result(void){
    schedule_result_t *iterator_dl = schedule_result_list_DL;
    schedule_result_t *iterator_ul = schedule_result_list_UL;
    schedule_result_t *iterator;
    int i = 0;
    char str[20];
    char str1[20];
    char str2[20];
    LOG_D(MAC,"=== print schedule result ===\n");

    while((schedule_result_t *)0 != iterator_dl || (schedule_result_t *)0 != iterator_ul){
        if((schedule_result_t *)0 == iterator_dl){
            iterator = iterator_ul;
            iterator_ul = iterator_ul->next;
        }
        else if((schedule_result_t *)0 == iterator_ul){
            iterator = iterator_dl;
            iterator_dl = iterator_dl->next;
        }else{
            if(iterator_ul->output_subframe < iterator_dl->output_subframe){
                iterator = iterator_ul;
                iterator_ul = iterator_ul->next;
            }else{
                iterator = iterator_dl;
                iterator_dl = iterator_dl->next;
            }
        }


        if(iterator->rnti == P_RNTI){
            sprintf(str, " PAGING");
        }
        else if(iterator->rnti == SI_RNTI){
            sprintf(str, "SI-RNTI");
        }
        else if(iterator->rnti <= RA_RNTI_HIGH && iterator->rnti >= RA_RNTI_LOW){
            sprintf(str, "RA-RNTI");
        }else{
            sprintf(str, "UE%05d", iterator->rnti-C_RNTI_LOW);
        }

        if(iterator->direction == DL){
            sprintf(str1, "DL");
        }else{
            sprintf(str1, "UL");
        }

        switch(iterator->channel){
            case NPDCCH:
                sprintf(str2, "NPDCCH");
                break;
            case NPDSCH:
                sprintf(str2, "NPDSCH");
                break;
            case NPUSCH:
                sprintf(str2, "NPUSCH");
                break;
            default:
                break;
        }

        LOG_D(MAC,"[%2d][%s][%s][%s] output(%4d)\n", i++, str, str1, str2, iterator->output_subframe);
/*      if((uint8_t *)0 != iterator->printf_str){
            LOG_D(MAC," printf: %s\n", iterator->printf_str);
        }else{
            LOG_D(MAC,"\n");
        }*/
    }
}


void print_schedule_result_DL(void){
    schedule_result_t *iterator = schedule_result_list_DL;
    int i=0;
    char str[20];
    LOG_D(MAC,"=== print schedule result DL ===\n");
    while((schedule_result_t *)0 != iterator){
        if(iterator->rnti == P_RNTI){
            sprintf(str, " PAGE");
        }
        else if(iterator->rnti == SI_RNTI){
            sprintf(str, "   SI");
        }
        else if(iterator->rnti <= RA_RNTI_HIGH && iterator->rnti >= RA_RNTI_LOW){
            sprintf(str, "   RA");
        }else{
            sprintf(str, "UE%03d", iterator->rnti-C_RNTI_LOW);
        }
        LOG_D(MAC,"[%2d][%s][""DL""] output(%4d)\n", i++, str, iterator->output_subframe);
        /*if((uint8_t *)0 != iterator->printf_str){
            LOG_D(MAC," printf: %s\n", iterator->printf_str);
        }else{
            LOG_D(MAC,"\n");
        }*/
        iterator = iterator->next;
    }
}

void print_schedule_result_UL(void){
    schedule_result_t *iterator = schedule_result_list_UL;
    int i=0;

    char str[20];

    LOG_D(MAC,"=== print schedule result UL ===\n");
    while((schedule_result_t *)0 != iterator){

        sprintf(str, "UE%03d", iterator->rnti-C_RNTI_LOW);
        LOG_D(MAC,"[%2d][%s][""UL""] output(%4d)\n", i++, str, iterator->output_subframe);
        /*if((uint8_t *)0 != iterator->printf_str){
            LOG_D(MAC," printf: %s\tnext %p\n", iterator->printf_str, iterator->next);
        }else{
            LOG_D(MAC,"\n");
        }*/
        iterator = iterator->next;
    }
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
                return (uint32_t)(2<<I_delay);
        }
        else
        {
            return (uint32_t)(16<<(I_delay-1));
        }
    }
}

/*
uint8_t *parse_ulsch_header( uint8_t *mac_header,
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

        lcid = ((SCH_SUBHEADER_FIXED_NB_IoT *)mac_header_ptr)->LCID;

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
                    mac_header_ptr += 2;//sizeof(SCH_SUBHEADER_SHORT_NB_IoT);
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
*/

//  calvin
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
