
/*This is the interface module between PHY
* this will trigger the corresponding function in MAC or PHY layer according to the type of the message
*
*
*
*/



#include "platform_types.h"

#define NUMBER_OF_UE_MAX 20

typedef struct{
	
	//flag to show which message is
	uint8_t UL_MSG_flag;
    
    //preamble part
    //index of the preamble
	uint16_t preamble_index;
	//timing offset by PHY
	int16_t timing_offset;
	
	//ULSCH SDU part
	
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

	boolean_t ACK_INFO;

}UL_SPEC_t;

//UL_IND
typedef struct{
	/*Start at the common part*/

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

	/*UE specific part*/

	UL_SPEC_t UL_SPEC_Info[NUMBER_OF_UE_MAX]; 

}UL_IND_t;

/*Interface for uplink, transmitting the Preamble(list), ULSCH SDU, ACK/NACK, Tick (trigger scheduler)
*Parameters:
*Parameters:
*/
void UL_indication(UL_IND_t UL_info);

/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void Schedule_Response();

/*Interface for Configuration*/
void Config_Request();
