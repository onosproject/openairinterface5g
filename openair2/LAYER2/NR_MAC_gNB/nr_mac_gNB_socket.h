/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
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

/*! \file mac.h
* \brief MAC data structures, constant, and function prototype
* \author Navid Nikaein and Raymond Knopp, WIE-TAI CHEN
* \date 2011, 2018
* \version 0.5
* \company Eurecom, NTUST
* \email navid.nikaein@eurecom.fr, kroempa@gmail.com

*/
/** @defgroup _oai2  openair2 Reference Implementation
 * @ingroup _ref_implementation_
 * @{
 */

/*@}*/

#ifndef __LAYER2_NR_MAC_GNB_H_SOCKET__
#define __LAYER2_NR_MAC_GNB_H_SOCKET_

#define RESULT_OK  0
#define RESULT_FAILED  -1

/* structure for socket as source of MAC gNB data; copied from common/utils/itti_analyzer/libbuffer/socket.h */
typedef struct {
    pthread_t thread;
    int       sd;
    char     *ip_address;
    uint16_t  port;

    /* The pipe used between main thread (running GTK) and the socket thread */
    int       pipe_fd;

    /* Time used to avoid refreshing UI every time a new signal is incoming */
    //gint64    last_data_notification;
    //uint8_t   nb_signals_since_last_update;

    /* The last signals received which are not yet been updated in GUI */
    //GList    *signal_list;
} socket_mac_gNB_data;

/* Function to get the MAC PDU from socket data */

int get_mac_pdu_from_socket(char* pdu, uint32_t TBS, socket_mac_gNB_data *socket_data);

/* Function to open a socket and connect to the MAC data source*/
int connect_mac_socket_to_data_source(const char *remote_ip, const uint16_t port, socket_mac_gNB_data *socket_data);

/* Function to close the MAC data source socket */
int disconnect_mac_socket_from_data_source(socket_mac_gNB_data *socket_data);

#endif /*__LAYER2_NR_MAC_GNB_H_SOCKET_ */
