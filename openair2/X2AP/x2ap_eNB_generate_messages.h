/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2014 Eurecom

    OpenAirInterface is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.


    OpenAirInterface is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with OpenAirInterface.The full GNU General Public License is
   included in this distribution in the file called "COPYING". If not,
   see <http://www.gnu.org/licenses/>.

  Contact Information
  OpenAirInterface Admin: openair_admin@eurecom.fr
  OpenAirInterface Tech : openair_tech@eurecom.fr
  OpenAirInterface Dev  : openair4g-devel@lists.eurecom.fr

  Address      : Eurecom, Compus SophiaTech 450, route des chappes, 06451 Biot, France.

 *******************************************************************************/
#ifndef X2AP_ENB_GENERATE_MESSAGES_H_
#define X2AP_ENB_GENERATE_MESSAGES_H_

#include "x2ap_eNB_defs.h"
#include "x2ap_ies_defs.h"

int x2ap_eNB_generate_x2_setup_request(x2ap_eNB_instance_t *instance_p, 
				       x2ap_eNB_data_t *x2ap_enb_data_p);

int 
x2ap_generate_x2_setup_response (x2ap_eNB_data_t * eNB_association);

int x2ap_eNB_generate_x2_setup_failure ( uint32_t assoc_id,
					 X2ap_Cause_PR cause_type,
					 long cause_value,
					 long time_to_waitx);
#endif /*  X2AP_ENB_GENERATE_MESSAGES_H_ */
