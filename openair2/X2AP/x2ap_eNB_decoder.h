/*******************************************************************************
    OpenAirInterface
    Copyright(c) 1999 - 2015 Eurecom

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

/*! \file x2ap_eNB_decoder.c
 * \brief x2ap pdu decode procedures for eNB
 * \author Navid Nikaein
 * \date 2015- 2016
 * \version 0.1
 */

#include "x2ap_common.h"
#include "x2ap_ies_defs.h"

#ifndef X2AP_ENB_DECODER_H_
#define X2AP_ENB_DECODER_H_

int x2ap_eNB_decode_pdu(x2ap_message *x2ap_message, const uint8_t * const buffer, uint32_t len);

#endif /* X2AP_ENB_DECODER_H_ */
