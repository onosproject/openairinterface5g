/*
 * RemoteUserID.h
 *
 *  Created on: Jun 11, 2019
 *      Author: nepes
 */

#include "EpsMobileIdentity.h"
//#include "RemoteUEContext.h"

#ifndef OPENAIR3_NAS_COMMON_IES_REMOTEUSERID_H_
#define OPENAIR3_NAS_COMMON_IES_REMOTEUSERID_H_

//#include "3gpp_23.003.h"
#include "EpsMobileIdentity.h"

#define REMOTE_USER_ID_MINIMUM_LENGTH 10
//#define REMOTE_USER_ID_MAXIMUM_LENGTH 1



typedef struct imsi_identity_s {
  uint8_t  spareimsi:4;
#define IMSI_INSTANCE  0
  uint8_t  instanceimsi:4;
  uint8_t  identity_digit1:4;
#define IMSI_EVEN  0
#define IMSI_ODD   1
  uint8_t  oddeven:1;
#define REMOTE_UE_MOBILE_IDENTITY_IMSI  010
  uint8_t  typeofidentity:3;
  uint8_t  identity_digit2:4;
  uint8_t  identity_digit3:4;
  uint8_t  identity_digit4:4;
  uint8_t  identity_digit5:4;
  uint8_t  identity_digit6:4;
  uint8_t  identity_digit7:4;
  uint8_t  identity_digit8:4;
  uint8_t  identity_digit9:4;
  uint8_t  identity_digit10:4;
  uint8_t  identity_digit11:4;
  uint8_t  identity_digit12:4;
  uint8_t  identity_digit13:4;
  uint8_t  identity_digit14:4;
  uint8_t  identity_digit15:4;
  // because of union put this extra attribute at the end
  uint8_t  num_digits;
} imsi_identity_t;

//#define imsi_identity_t;


typedef struct remote_user_id_s {
  uint8_t  spare:4;
#define REMOTE_USER_ID_INSTANCE  0
  uint8_t  instance:4;
  uint8_t  spare1:6;
#define REMOTE_USER_ID_IMEIF  0
  uint8_t imeif:1;
#define REMOTE_USER_ID_MSISDN  0
  uint8_t  msisdnf:1;

  uint8_t     flags_present;
  uint8_t     spare_instance;
  imsi_identity_t *imsi_identity;
}remote_user_id_t;




//typedef union remoteuser_mobile_identity_s {
	//imsi_identity_t imsi;
//} remoteuser_mobile_identity_t;


int encode_remote_user_id(remote_user_id_t *remoteuserid, uint8_t iei, uint8_t *buffer, uint32_t len);

int decode_remote_user_id(remote_user_id_t *remoteuserid, uint8_t iei, uint8_t *buffer, uint32_t len);

#endif /* OPENAIR3_NAS_COMMON_IES_REMOTEUSERID_H_ */

