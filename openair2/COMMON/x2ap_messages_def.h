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

  Address      : Eurecom, Campus SophiaTech, 450 Route des Chappes, CS 50193 - 06904 Biot Sophia Antipolis cedex, FRANCE

*******************************************************************************/


/* eNB application layer -> X2AP messages */
/* ITTI LOG messages */
/* ENCODER */ 
MESSAGE_DEF(X2AP_RESET_REQUST_LOG             , MESSAGE_PRIORITY_MED, IttiMsgText                      , x2ap_reset_request_log)
MESSAGE_DEF(X2AP_RESOURCE_STATUS_RESPONSE_LOG , MESSAGE_PRIORITY_MED, IttiMsgText                      , x2ap_resource_status_response_log)
MESSAGE_DEF(X2AP_RESOURCE_STATUS_FAILURE_LOG , MESSAGE_PRIORITY_MED, IttiMsgText                      , x2ap_resource_status_failure_log)

/* Messages for X2AP logging */ 
MESSAGE_DEF(X2AP_SETUP_REQUEST_LOG           , MESSAGE_PRIORITY_MED, IttiMsgText                      , x2ap_setup_request_log)


/* eNB application layer -> X2AP messages */
MESSAGE_DEF(X2AP_REGISTER_ENB_REQ          , MESSAGE_PRIORITY_MED, x2ap_register_enb_req_t          , x2ap_register_enb_req)

/* X2AP -> eNB application layer messages */
MESSAGE_DEF(X2AP_REGISTER_ENB_CNF          , MESSAGE_PRIORITY_MED, x2ap_register_enb_cnf_t          , x2ap_register_enb_cnf)
MESSAGE_DEF(X2AP_DEREGISTERED_ENB_IND      , MESSAGE_PRIORITY_MED, x2ap_deregistered_enb_ind_t      , x2ap_deregistered_enb_ind)

/* handover messages X2AP <-> RRC */
MESSAGE_DEF(X2AP_HANDOVER_REQ              , MESSAGE_PRIORITY_MED, x2ap_handover_req_t              , x2ap_handover_req)
MESSAGE_DEF(X2AP_HANDOVER_RESP             , MESSAGE_PRIORITY_MED, x2ap_handover_resp_t             , x2ap_handover_resp)
