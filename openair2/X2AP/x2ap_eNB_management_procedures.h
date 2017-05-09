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

#ifndef X2AP_ENB_MANAGEMENT_PROCEDURES_H_
#define X2AP_ENB_MANAGEMENT_PROCEDURES_H_
/*
struct x2ap_eNB_mme_data_s *s1ap_eNB_get_MME(
  s1ap_eNB_instance_t *instance_p,
  int32_t assoc_id, uint16_t cnx_id);
*/
void x2ap_eNB_insert_new_instance(x2ap_eNB_instance_t *new_instance_p);

x2ap_eNB_instance_t *x2ap_eNB_get_instance(uint8_t mod_id);

uint16_t x2ap_eNB_fetch_add_global_cnx_id(void);

void x2ap_eNB_prepare_internal_data(void);

x2ap_eNB_data_t* x2ap_is_eNB_id_in_list(uint32_t eNB_id);

x2ap_eNB_data_t* x2ap_is_eNB_assoc_id_in_list(uint32_t sctp_assoc_id);

struct x2ap_eNB_data_s *x2ap_get_eNB(x2ap_eNB_instance_t *instance_p,
				     int32_t assoc_id, 
				     uint16_t cnx_id);

#endif /* X2AP_ENB_MANAGEMENT_PROCEDURES_H_ */
