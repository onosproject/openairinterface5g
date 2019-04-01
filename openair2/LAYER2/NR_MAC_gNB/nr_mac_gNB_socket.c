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

#include "nr_mac_gNB.h"
#include "SCHED_NR/sched_nr.h"
#include "mac_proto.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "PHY/NR_TRANSPORT/nr_dci.h"





/* Function to get the MAC PDU from socket data */
int get_mac_pdu_from_socket(char* pdu,
		                    uint32_t TBS, 
	                    	socket_mac_gNB_data *socket_data)
{

	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST\n");
	/* Should I set the socket as non-blocking? */
	//fcntl(socket_data->sd, F_SETFL, O_NONBLOCK);
	
	//If listening to more than one socket (e.g. for TCP and UDP listening) select() should be used
	
	int recv_ret;
	//struct sockaddr *from_addr;
	struct sockaddr_in from_addr;
	socklen_t from_addr_len;
	from_addr_len = sizeof(struct sockaddr_in);

	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST1, socket_data->sd=%d\n", socket_data->sd);
	//recv_ret = recv(socket_data->sd, pdu, (size_t)TBS, 0);
	recv_ret = recvfrom(socket_data->sd, pdu, (size_t)TBS, 0, &from_addr, &from_addr_len);
	//LOG_I(MAC, "[get_mac_pdu_from_socket] TEST2\n");
	if (recv_ret == -1) {
		/* Failure case */
		
		switch (errno) {
			//case EWOULDBLOCK:
			case EAGAIN:
				return -1;
			default:
				//g_info("recv failed: %s", g_strerror(errno));
				LOG_I(MAC, "[get_mac_pdu_from_socket] ERROR. Received %d bytes in MAC socket. Errno=%d\n", recv_ret, errno);
				return -1;
				break;
			
		}
	//} else if (recv_ret == 0) {
	//	/* We lost the connection with other peer or shutdown asked */
	//	ui_pipe_write_message(socket_data->pipe_fd,
	//						  UI_PIPE_CONNECTION_LOST, NULL, 0);
	//	free(socket_data->ip_address);
	//	free(socket_data);
	//	pthread_exit(NULL);
	}else{
		//Source IP address could be checked to only transmit RTP packets for video flow
		LOG_I(MAC, "[get_mac_pdu_from_socket] Received %d bytes in MAC socket.\n", recv_ret);
	}
	

	return recv_ret;
	
}

/* Function to open a socket and connect to the MAC data source*/
int connect_mac_socket_to_data_source(const char *ip_address,
		                              const uint16_t port, socket_mac_gNB_data *socket_mac)
{
	
	//LOG_I(MAC, "[connect_mac_socket_to_data_source]TEST\n");
	//socket_mac_gNB_data *socket_data;
	//socket_data = calloc(1, sizeof(*socket_data));
	socket_mac->ip_address = strdup(ip_address);
	socket_mac->port    = port;
	socket_mac->sd      = -1;
	struct sockaddr_in  si_me;

	//g_assert(socket_data != NULL);
	
	/* Preparing the socket */
	//if ((socket_data->sd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == -1) {
	if ((socket_mac->sd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
		//g_warning("socket failed: %s", g_strerror(errno));
		LOG_I(MAC, "socket failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}
	memset((void *)&si_me, 0, sizeof(si_me));

	si_me.sin_family = AF_INET;
	si_me.sin_port = htons(socket_mac->port);
	//si_me.sin_addr.s_addr = htonl(INADDR_ANY);
	//si_me.sin_addr.s_addr = htonl(socket_data->ip_address);
	si_me.sin_addr.s_addr = inet_addr(socket_mac->ip_address);
	
	if (inet_aton(socket_mac->ip_address, &si_me.sin_addr) == 0) {
		//g_warning("inet_aton() failed\n");
		LOG_I(MAC, "inet_aton() failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}
	
	if(bind(socket_mac->sd, (struct sockaddr *)&si_me, sizeof(struct sockaddr_in)) == -1){
		//g_warning("binding socket failed\n");
		LOG_I(MAC, "binding socket failed\n");
		free(socket_mac->ip_address);
		free(socket_mac);
		return RESULT_FAILED;
	}	
	
	//Set socket as non-blocking
	fcntl(socket_mac->sd, F_SETFL, O_NONBLOCK);
	
	//If more than one socket, there should be a set of sockets to use select() with them
	//The socket should be added to the set in this function.
	
	//memcpy(socket_mac, socket_data, sizeof(socket_data));
	//&socket_mac = &socket_data;
	LOG_I(MAC, "[connect_mac_socket_to_data_source]MAC socket connected; socket_mac->sd= %d\n", socket_mac->sd);
		
	return RESULT_OK;
}

/* Function to close the MAC data source socket */
int disconnect_mac_socket_from_data_source(socket_mac_gNB_data *socket_data)
{
	close(socket_data->sd);
	free(socket_data->ip_address);
	free(socket_data);
	
	LOG_I(MAC, "[disconnect_mac_socket_from_data_source] Disconnecting MAC socket\n");
	
	return RESULT_OK;
}

