
/*This is the interface module between PHY
*Provided the FAPI style interface structures for P7.
*
*
*
*/

#include "openair1/PHY/LTE_TRANSPORT/defs_nb_iot.h"


#define SCH_PAYLOAD_SIZE_MAX 4096
#define BCCH_PAYLOAD_SIZE_MAX 128
#define NUMBER_OF_UE_MAX 20

// uplink subframe P7

/*UL_SPEC_t:
* A struture mainly describes the UE specific information. (for NB_rx_sdu)
* Corresponding to thhe RX_ULSCH.indication, CRC.inidcation, NB_HARQ.indication in FAPI
*/
typedef struct{

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
	//CRC indication for the message is corrected or not for the Uplink HARQ
	uint8_t crc_ind;

	//This ACK NACK is for the Downlink HARQ received by the NPUSCH from UE
	uint8_t NAK;

}UL_SPEC_t;

/*UL_IND_t:
* A structure handles all the uplink information.
*/
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

 // Downlink subframe P7

 typedef union{

 	//The length (in bytes)
 	uint16_t Length;
 	//PDU index 
 	uint16_t PDU_index;
 	//Transmission Power
 	uint16_t Trans_Power;
 	//HYPER SFN 2lsbs
 	uint16_t HyperSFN2lsbs;
 	//NPBCH pdu payload
 	uint8_t npbch_pdu_payload[4]; 

 }npbch_t;

 typedef union{

 	//The length (in bytes)
 	uint16_t Length;
 	//PDU index 
 	uint16_t PDU_index;
 	//start symbol 0-4 0 for guard-band and standalone operating
 	uint8_t start_symbol;
 	//RNTI type,0 = BCCH(SIB), 1 for DL data
 	uint8_t RNTI_type;
 	// RNTI 
 	uint16_t RNTI;
 	// SIB payload
 	uint8_t nbcch_pdu_payload[BCCH_PAYLOAD_SIZE_MAX];
 	// NDLSCH payload
 	uint8_t ndlsch_pdu_payload[SCH_PAYLOAD_SIZE_MAX];

 }npdsch_t;

 typedef union{

 	// The length (in bytes)
 	uint16_t Length;
 	// PDU index 
 	uint16_t PDU_index;
 	// NCCE index value 0 -> 1
 	uint8_t NCCE_index;
 	// Aggregation level
 	uint8_t aggregation;
 	// start symbol
 	uint8_t start_symbol;
 	// RNTI type,0 = TC-RNTI, 1 = RA-RNTI, 2 = P-RNTI 3 = other
 	uint8_t RNTI_type;
 	// RNTI 
 	uint16_t RNTI;
 	// Scrambliing re-initialization batch index from FAPI specs (1-4)
 	uint8_t batch_index;
 	// NRS antenna ports assumed by the UE from FAPI specs (1-2)
 	uint8_t num_antenna;
 	// Number of DCI
  	uint8_t Num_dci;
  	// Format of DCI
 	DCI_format_NB_t DCI_Format;
 	// Content of DCI
 	DCI_CONTENT *DCI_Content;

 }npdcch_t;


typedef struct{

 	/*Start at the common part*/

 	//Module ID
	module_id_t module_idP; 
 	//CC ID
 	int CC_id;
 	//frame
 	frame_t frameP;
 	//subframe
 	sub_frame_t subframeP;

 	npdcch_t NB_DCI;

 	npdsch_t NB_DLSCH;

 	npbch_t NB_BCH;

}Sched_Rsp_t;

//	Calvin 20170531 start

/*IF_Module_t*/
typedef struct IF_Module_s{
	void (*UL_indication)(UL_IND_t UL_INFO);
	void (*Schedule_Response)(Sched_Rsp_t Sched_INFO);
}IF_Module_t;

/*Initial */
int IF_Module_init(IF_Module_t *if_inst);

//	Calvin 20170531 end

/*Interface for uplink, transmitting the Preamble(list), ULSCH SDU, NAK, Tick (trigger scheduler)
*/
void UL_indication(UL_IND_t UL_INFO);

/*Interface for Downlink, transmitting the DLSCH SDU, DCI SDU*/
void Schedule_Response(Sched_Rsp_t Sched_INFO);

/*Interface for Configuration*/
//void Config_Request();
