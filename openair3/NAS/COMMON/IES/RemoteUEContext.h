/*
 * RemoteUEContext.h
 *
 *  Created on: Jun 11, 2019
 *      Author: nepes
 */

#ifndef OPENAIR3_NAS_COMMON_IES_REMOTEUECONTEXT_H_
#define OPENAIR3_NAS_COMMON_IES_REMOTEUECONTEXT_H_

#include "RemoteUserID.h"


#define REMOTE_UE_CONTEXT_MINIMUM_LENGTH 16
#define REMOTE_UE_CONTEXT_MAXIMUM_LENGTH 28

//typedef struct imsi_identity_s {
//uint8_t  identity_digit1:4;
//#define IMSI_EVEN  0
//#define IMSI_ODD   1
//uint8_t  oddeven:1;
//#define REMOTE_UE_MOBILE_IDENTITY_IMSI  010
//uint8_t  typeofidentity:3;
//uint8_t  identity_digit2:4;
//uint8_t  identity_digit3:4;
// uint8_t  identity_digit4:4;
//uint8_t  identity_digit5:4;
//  uint8_t  identity_digit6:4;
//  uint8_t  identity_digit7:4;
//  uint8_t  identity_digit8:4;
// uint8_t  identity_digit9:4;
//  uint8_t  identity_digit10:4;
// uint8_t  identity_digit11:4;
// uint8_t  identity_digit12:4;
//  uint8_t  identity_digit13:4;
//  uint8_t  identity_digit14:4;
//  uint8_t  identity_digit15:4;
// because of union put this extra attribute at the end
// uint8_t  num_digits;
//} imsi_identity_t;

typedef struct remote_ue_context_s {
		uint8_t  spare:4;
//#define REMOTE_UE_MOBILE_IDENTITY_IMSI  010
		//uint8_t typeofuseridentity:3;
#define NUMBER_OF_REMOTE_UE_CONTEXT_IDENTITIES 1
		uint8_t numberofuseridentity:8;
#define EVEN_IDENTITY 0
#define ODD_IDENTITY  1
		uint8_t oddevenindic:1;
		bool     flags_present;
        imsi_identity_t *imsi_identity;
}remote_ue_context_t;


//typedef union remote_ue_mobile_identity_s {
//#define REMOTE_UE_MOBILE_IDENTITY_IMSI_ENCRYPTED  001
//#define REMOTE_UE_MOBILE_IDENTITY_IMSI  010
//#define REMOTE_UE_MOBILE_IDENTITY_MSISDN  011
//#define REMOTE_UE_MOBILE_IDENTITY_IMEI  100
//#define REMOTE_UE_MOBILE_IDENTITY_IMEISV  101
//	imsi_e_remote_ue_mobile_identity_t imsi_encrypted;
//	imsi_remote_ue_mobile_identity_t imsi;
//	msisdn_remote_ue_mobile_identity_t msisdn;
//	imei_remote_ue_mobile_identity_t imei;
//	imeisv_remote_ue_mobile_identity_t imeisv;
//} remote_ue_mobile_identity_t;
//#define REMOTE_UE_MOBILE_IDENTITY    "remote_ue_identity_type"

  int encode_remote_ue_context(remote_ue_context_t *remoteuecontext, uint8_t iei, uint8_t *buffer, uint32_t len);

  int decode_remote_ue_context(remote_ue_context_t *remoteuecontext, uint8_t iei, uint8_t *buffer, uint32_t len);


#endif /* OPENAIR3_NAS_COMMON_IES_REMOTEUECONTEXT_H_ */
