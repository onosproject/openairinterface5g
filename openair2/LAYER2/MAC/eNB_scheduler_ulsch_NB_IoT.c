/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */
 /*! \file eNB_scheduler_ulsch_NB_IoT.c
 * \brief handle UL UE-specific scheduling
 * \author  NTUST BMW Lab./Nick HO
 * \date 2017 - 2018
 * \email: nick133371@gmail.com
 * \version 1.0
 *
 */

#include "defs_NB_IoT.h"
#include "proto_NB_IoT.h"
#include "extern_NB_IoT.h"
#include "RRC/LITE/proto.h"
#include "RRC/LITE/extern.h"
#include "RRC/L2_INTERFACE/openair_rrc_L2_interface.h"

unsigned char str20[] = "DCI_uss";
unsigned char str21[] = "DATA_uss";

// scheduling UL
int schedule_UL_NB_IoT(eNB_MAC_INST_NB_IoT *mac_inst,UE_TEMPLATE_NB_IoT *UE_info,uint32_t subframe, uint32_t frame, uint32_t H_SFN, UE_SCHED_CTRL_NB_IoT_t *UE_sched_ctrl_info){
	int i,ndi = 0,check_DCI_result = 0,check_UL_result = 0,candidate;
	uint32_t DL_end;
    //Scheduling resource temp buffer
    sched_temp_DL_NB_IoT_t *NPDCCH_info = (sched_temp_DL_NB_IoT_t*)malloc(sizeof(sched_temp_DL_NB_IoT_t));
	  candidate = UE_info->R_max/UE_sched_ctrl_info->R_dci;
    uint32_t mcs = max_mcs[UE_info->multi_tone];
    uint32_t mappedMcsIndex=UE_info->PHR+(4 * UE_info->multi_tone);
    int TBS = 0;
    int Iru = 0, Nru, I_rep,N_rep,total_ru;
    int dly = 0,uplink_time = 0;

    if(UE_info->ul_total_buffer<=0)
    {
        LOG_D(MAC,"[%04d][ULSchedulerUSS][UE:%05d] No UL data in buffer\n", mac_inst->current_subframe, UE_info->rnti);
        return -1;
    }


    TBS=get_TBS_UL_NB_IoT(mcs,UE_info->multi_tone,Iru);
    LOG_I(MAC,"Initial TBS : %d UL_buffer: %d\n", TBS, UE_info->ul_total_buffer);

    sched_temp_UL_NB_IoT_t *NPUSCH_info = (sched_temp_UL_NB_IoT_t*)malloc(sizeof(sched_temp_UL_NB_IoT_t));

    // setting of the NDI
    /*
    if(UE_info->HARQ_round == 0)
    {
        ndi = 1-UE_info->oldNDI_UL;
        UE_info->oldNDI_UL=ndi;
    }
    */

    ndi = 1;

    for (i = 0; i < candidate; i++)
	  {
		    /*step 1 : Check DL resource is available for DCI N0 or not*/
		    check_DCI_result = check_resource_NPDCCH_NB_IoT(mac_inst,H_SFN, frame, subframe, NPDCCH_info, i, UE_sched_ctrl_info->R_dci);

        if( check_DCI_result != -1)
		    {
			     /*step 2 : Determine MCS / TBS / REP / RU number*/
            /*while((mapped_mcs[UE_info->CE_level][mappedMcsIndex]< mcs)||((TBS>UE_info->ul_total_buffer)&&(mcs>=0)))
                {
                    --mcs;
                    TBS=get_TBS_UL_NB_IoT(mcs,UE_info->multi_tone,Iru);
                }*/

            mcs = mapped_mcs[UE_info->CE_level][mappedMcsIndex];

            if (UE_info->ul_total_buffer == 10)
              mcs = 8;

            if (UE_info->ul_total_buffer == 31)
              mcs = 8;

            //mcs = 2;
            while((TBS<UE_info->ul_total_buffer)&&(Iru<=7))
                {
                    Iru++;
                    TBS=get_TBS_UL_NB_IoT(mcs,UE_info->multi_tone,Iru);
                }


            LOG_D(MAC,"TBS : %d MCS %d I_RU %d\n", TBS, mcs, Iru);

            Nru = RU_table[Iru];
            DL_end = NPDCCH_info->sf_end;
            N_rep = get_N_REP(UE_info->CE_level);
            I_rep = get_I_REP(N_rep);
            total_ru = Nru * N_rep;

            LOG_D(MAC,"[%04d][ULSchedulerUSS][UE:%05d] Multi-tone:%d,MCS:%d,TBS:%d,UL_buffer:%d,DL_start:%d,DL_end:%d,N_rep:%d,N_ru:%d,Total_ru:%d,Iru:%d\n", mac_inst->current_subframe,UE_info->rnti,UE_info->multi_tone,mcs,TBS,UE_info->ul_total_buffer,NPDCCH_info->sf_start,DL_end,N_rep,Nru,total_ru,Iru);

            /*step 3 Check UL resource for Uplink data, we will loop the scheduling delay here */
            for(dly=0;dly<4;dly++)
            {
                uplink_time = DL_end +scheduling_delay[dly]+1;
                check_UL_result = Check_UL_resource(uplink_time,total_ru, NPUSCH_info, UE_info->multi_tone, 0);
                if (check_UL_result != -1)
                {

                //LOG_D(MAC,"[%04d][UL scheduler][UE:%05d] DCI content = scind : %d ResAssign : %d mcs : %d ndi : %d scheddly : %d RepNum : %d rv : %d DCIRep : %d\n", mac_inst->current_subframe,UE_info->rnti,DCI_N0->scind,DCI_N0->ResAssign,DCI_N0->mcs,DCI_N0->ndi,DCI_N0->Scheddly,DCI_N0->RepNum,DCI_N0->rv,DCI_N0->DCIRep);
                LOG_I(MAC,"[%04d][ULSchedulerUSS][%d][Success] complete scheduling with data size %d\n", mac_inst->current_subframe, UE_info->rnti, UE_info->ul_total_buffer);
                LOG_I(MAC,"[%04d][ULSchedulerUSS][%d] Multi-tone:%d,MCS:%d,TBS:%d,UL_buffer:%d,DL_start:%d,DL_end:%d,N_rep:%d,N_ru:%d,Total_ru:%d\n", mac_inst->current_subframe,UE_info->rnti,UE_info->multi_tone,mcs,TBS,UE_info->ul_total_buffer,NPDCCH_info->sf_start,DL_end,N_rep,Nru,total_ru);
                //LOG_D(MAC,"[%04d][ULSchedulerUSS][%d][Success] DCI content = scind : %d ResAssign : %d mcs : %d ndi : %d scheddly : %d RepNum : %d rv : %d DCIRep : %d\n", mac_inst->current_subframe, UE_info->rnti, DCI_N0->scind,DCI_N0->ResAssign,DCI_N0->mcs,DCI_N0->ndi,DCI_N0->Scheddly,DCI_N0->RepNum,DCI_N0->rv,DCI_N0->DCIRep);

                // step 5 resource allocation and generate scheduling result
                LOG_D(MAC,"[%04d][ULSchedulerUSS][UE:%05d] Generate result\n", mac_inst->current_subframe, UE_info->rnti);
                //generate_scheduling_result_UL(NPDCCH_info->sf_start, NPDCCH_info->sf_end,NPUSCH_info->sf_start, NPUSCH_info->sf_end,DCI_N0,UE_info->rnti, str20, str21);
                LOG_D(MAC,"[%04d][ULSchedulerUSS][UE:%05d] Maintain resource\n", mac_inst->current_subframe, UE_info->rnti);
                //fill_resource_DL();
                maintain_resource_DL(mac_inst,NPDCCH_info,NULL);

                adjust_UL_resource_list(NPUSCH_info);

                //Fill result to Output structure
                    if(UE_info->ul_total_buffer==14)
                    {
                      UE_sched_ctrl_info->NPDCCH_sf_end=NPDCCH_info->sf_end;
                      UE_sched_ctrl_info->NPDCCH_sf_start=NPDCCH_info->sf_start;
                      UE_sched_ctrl_info->NPUSCH_sf_end=NPUSCH_info->sf_end;
                      UE_sched_ctrl_info->NPUSCH_sf_start=NPUSCH_info->sf_start;
                      //UE_sched_ctrl_info->resent_flag = 1;
                      LOG_D(MAC,"Key resent \n");
                      UE_sched_ctrl_info->dci_n0_index_ndi=0;
                    }else
                    {
                      UE_sched_ctrl_info->NPDCCH_sf_end=NPDCCH_info->sf_end;
                      UE_sched_ctrl_info->NPDCCH_sf_start=NPDCCH_info->sf_start;
                      UE_sched_ctrl_info->NPUSCH_sf_end=NPUSCH_info->sf_end;
                      UE_sched_ctrl_info->NPUSCH_sf_start=NPUSCH_info->sf_start;
                     UE_sched_ctrl_info->dci_n0_index_ndi=ndi;

                    }
                    UE_sched_ctrl_info->TBS=TBS;
                    UE_sched_ctrl_info->dci_n0_index_mcs=mcs;
                    UE_sched_ctrl_info->index_tbs=mcs;
                    UE_sched_ctrl_info->dci_n0_index_ru=Iru;
                    UE_sched_ctrl_info->dci_n0_n_ru=Nru;
                    UE_sched_ctrl_info->dci_n0_index_delay=dly;
                    UE_sched_ctrl_info->dci_n0_index_subcarrier=NPUSCH_info->subcarrier_indication;
                    UE_sched_ctrl_info->dci_n0_index_R_data=I_rep;

                    LOG_D(MAC,"[%04d][ULSchedulerUSS][%d][Success] Finish UL USS scheduling \n", mac_inst->current_subframe, UE_info->rnti);
                    return 0;
                }
            }

		}
        /*break now, we only loop one candidiate*/
        //break;
	}
  //----Daniel
  UE_sched_ctrl_info->flag_schedule_success=0;
  //----Daniel
  LOG_D(MAC,"[%04d][ULSchedulerUSS][%d][Fail] UL scheduling USS fail\n", mac_inst->current_subframe, UE_info->rnti);
	LOG_D(MAC,"[%04d][UL scheduler][UE:%05d] there is no available UL resource\n", mac_inst->current_subframe, UE_info->rnti);
	return -1;
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
  UE_TEMPLATE_NB_IoT *UE_info = NULL;
  uint8_t* msg4_rrc_pdu = NULL;
  LOG_D(MAC,"RX_SDU_IN\n");

  uint8_t* first_6 = (uint8_t*) malloc(6*sizeof(uint8_t));

  for(int a = 0; a<6;a++)
    first_6[a]=sdu[a+2];

  // note: if lcid < 25 this is sdu, otherwise this is CE
  payload_ptr = parse_ulsch_header_NB_IoT(sdu, &num_ce, &num_sdu,rx_ces, rx_lcids, rx_lengths, length);

  LOG_D(MAC,"num_CE= %d, num_sdu= %d, rx_ces[0] = %d, rx_lcids =  %d, rx_lengths[0] = %d, length = %d\n",num_ce,num_sdu,rx_ces[0],rx_lcids[0],rx_lengths[0],length);

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
        LOG_I(MAC,"Update BSR, rnti : %d\n",rnti);
        UE_info = get_ue_from_rnti(mac_inst, rnti);
        BSR_index = payload_ptr[0] & 0x3f;
        if(UE_info != NULL)
        {          
          LOG_I(MAC,"Find UE in CE 2 list, update ul_total_buffer to %d bytes\n",BSR_table[BSR_index]);
          UE_info->ul_total_buffer = BSR_table[BSR_index];
        }
        else
          LOG_E(MAC,"UE info empty\n"); 
            payload_ptr+=1;
            break;
        default:
        LOG_D(MAC,"Received unknown MAC header (0x%02x)\n", rx_ces[i]);
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
                ul_total_buffer = DV_table[DVI_index];
                LOG_D(MAC,"PHR = %d, ul_total_buffer = %d\n",PHR,ul_total_buffer);
                // go to payload
                payload_ptr+=1; 
		            // Note that the first 6 byte (48 bits) of this CCCH SDU should be encoded in the MSG4 for contention resolution 
                /*printf("CCCH SDU content: ");
                  for(int a = 0; a<9;a++)
                    printf("%02x ",payload_ptr[a]);
                  printf("\n");*/
                rx_lengths[i]-=1;
                mac_rrc_data_ind(
                  module_id,
                  CC_id,
                  frame,subframe,
                  rnti,
                  CCCH,
                  (uint8_t*)payload_ptr,
                  rx_lengths[i],
                  1,
                  module_id,
                  0);
                LOG_D(MAC,"rx_lengths : %d\n", rx_lengths[i]);
                msg4_rrc_pdu = mac_rrc_msg3_ind_NB_IoT(payload_ptr,rnti,rx_lengths[i]);
                if (Valid_msg3 == 1)
                {
                  receive_msg3_NB_IoT(mac_inst,rnti,PHR,ul_total_buffer,first_6,msg4_rrc_pdu);
                }else
                {
                  LOG_E(MAC,"Not available RA here\n");
                  Valid_msg3 = 1;
                }
                LOG_D(MAC,"Contention resolution ID = %02x %02x %02x %02x %02x %02x\n",first_6[0],first_6[1],first_6[2],first_6[3],first_6[4],first_6[5]);
          break;
            case DCCH0_NB_IoT:
            case DCCH1_NB_IoT:
                LOG_I(MAC,"DCCH PDU Here\n");
                if((UE_state_machine == initial_access)||(UE_state_machine == rach_for_next))
                {
                  block_RLC = 0;
                  int x = 0;
                  LOG_D(MAC,"Length: %d\n",rx_lengths[i]);
                  /*
                    for (x=0;x<rx_lengths[i];x++)
                      printf("%02x ",payload_ptr[x]);
                    printf("\n");
                  */
                  mac_rlc_data_ind(
                    module_id,
                    rnti,
                    module_id,
                    frame,
                    1,
                    0,
                    rx_lcids[i],
                    //1,/* change channel_id equals 1 (SRB) */
                    (char *)payload_ptr,
                    rx_lengths[i],
                    1,
                    NULL);//(unsigned int*)crc_status);
                      // trigger DL scheduler
                  if (RLC_RECEIVE_MSG5_FAILED == 1)
                  {
                    payload_ptr = payload_ptr+20;

                    LOG_N(MAC,"RLC Decoded data discard because of SN wrong\n");
                    //int x = 0;
                    for (x=0;x<49;x++)
                      printf("%02x ",payload_ptr[x]);
                    printf("\n");
                    protocol_ctxt_t     ctxt;
                    PROTOCOL_CTXT_SET_BY_MODULE_ID(&ctxt, module_id, 1, rnti, frame, 0, module_id);
                    rrc_data_ind(
                    &ctxt,
                    rx_lcids[i],
                    49,
                    (uint8_t *)payload_ptr);
                    RLC_RECEIVE_MSG5_FAILED = 0;
                  }
                  if (UE_info != NULL)
                  {
                    //UE_info->direction = 1; //1 for DL scheduler
                    LOG_D(MAC,"After receive Msg5, change the UE scheduling direction to DL\n");
                  }
                }else if (UE_state_machine == rach_for_auth_rsp || UE_state_machine == rach_for_TAU)
                {
                  LOG_D(MAC,"Here we are for the DCI N0 generating \n");
                  if (UE_info != NULL)
                  {
                    UE_info->direction = 0; //1 for DL scheduler
                    LOG_D(MAC,"Change direction into 0\n");
                  }
                  UE_state_machine = rach_for_next;
                }

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

void fill_DCI_N0(DCIFormatN0_t *DCI_N0, UE_TEMPLATE_NB_IoT *UE_info, UE_SCHED_CTRL_NB_IoT_t *UE_sched_ctrl_info)
{
    DCI_N0->type = 0;
    DCI_N0->scind = UE_sched_ctrl_info->dci_n0_index_subcarrier;
    DCI_N0->ResAssign = UE_sched_ctrl_info->dci_n0_index_ru;
    DCI_N0->mcs = UE_sched_ctrl_info->dci_n0_index_mcs;
    DCI_N0->ndi = UE_sched_ctrl_info->dci_n0_index_ndi;
    DCI_N0->Scheddly = UE_sched_ctrl_info->dci_n0_index_delay;
    DCI_N0->RepNum = UE_sched_ctrl_info->dci_n0_index_R_data;
    DCI_N0->rv = 0; // rv will loop 0 & 2
    DCI_N0->DCIRep = get_DCI_REP(UE_sched_ctrl_info->R_dci,UE_info->R_max);
    //DCI_N0->DCIRep = UE_sched_ctrl_info->dci_n0_index_R_dci;
    LOG_D(MAC,"[fill_DCI_N0] Type %d scind %d I_ru %d I_mcs %d ndi %d I_delay %d I_rep %d RV %d I_dci %d\n", DCI_N0->type, DCI_N0->scind, DCI_N0->ResAssign, DCI_N0->mcs, DCI_N0->ndi, DCI_N0->Scheddly, DCI_N0->RepNum, DCI_N0->rv, DCI_N0->DCIRep);
}
