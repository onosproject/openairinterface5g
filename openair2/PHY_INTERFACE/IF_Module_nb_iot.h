
/*This is the interface module between PHY
* this will trigger the corresponding function in MAC or PHY layer according to the type of the message
*
*
*
*/


#include "platform_types.h"
#include "openair1/PHY/LTE_TRANSPORT/dci_nb_iot.h"

#define NUMBER_OF_UE_MAX 20

typedef struct{
	
//flag to show which message is
uint8_t UL_MSG_flag;

//rnti
rnti_t rntiP;
//Pointer to sdu
uint8_t *sdu;
//Pointer to sdu length 
uint16_t sdu_lenP;
//HARQ ID
int harq_pidP;
//MSG3 flag
uint8_t *msg3_flagP;
//ACK/NAK
boolean_t NAK;

}UL_SPEC_t;

//UL_IND
typedef struct{

 /*Start at the common part*/

 int test;
 //Module ID
 module_id_t module_id;
 //CC ID
 int CC_id;
 //frame 
 frame_t frame;
 //subframe
 sub_frame_t subframe;
 //Number of availble UE
 int UE_NUM;

 /*preamble part*/
 //index of the preamble
 uint16_t preamble_index;
 //timing offset by PHY
 int16_t timing_offset;

 /*UE specific part*/
 UL_SPEC_t UL_SPEC_Info[NUMBER_OF_UE_MAX]; 

 }UL_IND_t;

typedef struct{

 /*Common part*/
 module_id_t module_idP; 
 int CC_id;
 frame_t frameP;
 sub_frame_t subframeP;

 rnti_t rntiP;

 /*Downlink data*/
 //TB size for Downlink data
 uint8_t TBindex;
 //PDU for MIB,SIBs
 uint8_t *BCCH_pdu_payload;
 //PDU for DL-SCH
 uint8_t *DLSCH_pdu_payload;

 /*DCI start*/
 // Format of DCI
 uint8_t DCI_Format;
 // 
 DCIFormatN0_t DCIN0;
 //
 DCIFormatN1_t DCIN1;
 //
 DCIFormatN1_RA_t DCIN1_RA;
 //
 DCIFormatN1_RAR_t DCIN1_RAR;
 //
 DCIFormatN2_Ind_t DCIN2_Ind;
 //
 DCIFormatN2_Pag_t DCIN2_Pag;


}Sched_Rsp_t;

/*Interface for uplink, transmitting the Preamble(list), ULSCH SDU, NAK, Tick (trigger scheduler)
*/
void UL_indication(UL_IND_t UL_INFO, frame_t frame, sub_frame_t subframe, module_id_t module_id);

/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void Schedule_Response(Sched_Rsp_t Sched_INFO);

/*Interface for Configuration*/
//void Config_Request();
