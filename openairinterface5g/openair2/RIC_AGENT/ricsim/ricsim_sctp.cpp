/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*****************************************************************************
#                                                                            *
# Copyright 2019 AT&T Intellectual Property                                  *
# Copyright 2019 Nokia                                                       *
#                                                                            *
# Licensed under the Apache License, Version 2.0 (the "License");            *
# you may not use this file except in compliance with the License.           *
# You may obtain a copy of the License at                                    *
#                                                                            *
#      http://www.apache.org/licenses/LICENSE-2.0                            *
#                                                                            *
# Unless required by applicable law or agreed to in writing, software        *
# distributed under the License is distributed on an "AS IS" BASIS,          *
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# See the License for the specific language governing permissions and        *
# limitations under the License.                                             *
#                                                                            *
******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <unistd.h>		//for close()
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/sctp.h>
#include <arpa/inet.h>	//for inet_ntop()
#include <assert.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/sctp.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "ricsim_sctp.hpp"

int sctp_start_server(const char *server_ip_str, const int server_port)
{
  if(server_port < 1 || server_port > 65535) {
      LOG_E("Invalid port number (%d). Valid values are between 1 and 65535.\n", server_port);
      exit(1);
  }

  int server_fd, af;
  struct sockaddr* server_addr;
  size_t addr_len;

  struct sockaddr_in  server4_addr;
  memset(&server4_addr, 0, sizeof(struct sockaddr_in));

  struct sockaddr_in6 server6_addr;
  memset(&server6_addr, 0, sizeof(struct sockaddr_in6));

  if(inet_pton(AF_INET, server_ip_str, &server4_addr.sin_addr) == 1)
  {
    server4_addr.sin_family = AF_INET;
    server4_addr.sin_port   = htons(server_port);

    server_addr = (struct sockaddr*)&server4_addr;
    af          = AF_INET;
    addr_len    = sizeof(server4_addr);
  }
  else if(inet_pton(AF_INET6, server_ip_str, &server6_addr.sin6_addr) == 1)
  {
    server6_addr.sin6_family = AF_INET6;
    server6_addr.sin6_port   = htons(server_port);

    server_addr = (struct sockaddr*)&server6_addr;
    af          = AF_INET6;
    addr_len    = sizeof(server6_addr);
  }
  else {
    perror("inet_pton()");
    exit(1);
  }

  if((server_fd = socket(af, SOCK_STREAM, IPPROTO_SCTP)) == -1) {
    perror("socket");
    exit(1);
  }

  //set send_buffer
  // int sendbuff = 10000;
  // socklen_t optlen = sizeof(sendbuff);
  // if(getsockopt(server_fd, SOL_SOCKET, SO_SNDBUF, &sendbuff, &optlen) == -1) {
  //   perror("getsockopt send");
  //   exit(1);
  // }
  // else
  //   LOG_D("[SCTP] send buffer size = %d\n", sendbuff);


  if(bind(server_fd, server_addr, addr_len) == -1) {
    perror("bind");
    exit(1);
  }

  if(listen(server_fd, SERVER_LISTEN_QUEUE_SIZE) != 0) {
    perror("listen");
    exit(1);
  }

  assert(server_fd != 0);

  LOG_I("[SCTP] Server started on %s:%d", server_ip_str, server_port);

  return server_fd;
}

int sctp_start_client(const char *server_ip_str, const int server_port)
{
  int client_fd, af;

  struct sockaddr* server_addr;
  size_t addr_len;

  struct sockaddr_in  server4_addr;
  memset(&server4_addr, 0, sizeof(struct sockaddr_in));

  struct sockaddr_in6 server6_addr;
  memset(&server6_addr, 0, sizeof(struct sockaddr_in6));

  if(inet_pton(AF_INET, server_ip_str, &server4_addr.sin_addr) == 1)
  {
    server4_addr.sin_family = AF_INET;
    server4_addr.sin_port   = htons(server_port);
    server_addr = (struct sockaddr*)&server4_addr;
    addr_len    = sizeof(server4_addr);
  }
  else if(inet_pton(AF_INET6, server_ip_str, &server6_addr.sin6_addr) == 1)
  {
    server6_addr.sin6_family = AF_INET6;
    server6_addr.sin6_port   = htons(server_port);
    server_addr = (struct sockaddr*)&server6_addr;
    addr_len    = sizeof(server6_addr);
  }
  else {
    perror("inet_pton()");
    exit(1);
  }

  if((client_fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_SCTP)) == -1)
  {
     perror("socket");
     exit(1);
  }

  // int sendbuff = 10000;
  // socklen_t optlen = sizeof(sendbuff);
  // if(getsockopt(client_fd, SOL_SOCKET, SO_SNDBUF, &sendbuff, &optlen) == -1) {
  //   perror("getsockopt send");
  //   exit(1);
  // }
  // else
  //   LOG_D("[SCTP] send buffer size = %d\n", sendbuff);

  //--------------------------------
  //Bind before connect
  auto optval = 1;
  if( setsockopt(client_fd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof optval) != 0 ){
    perror("setsockopt port");
    exit(1);
  }

  if( setsockopt(client_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof optval) != 0 ){
    perror("setsockopt addr");
    exit(1);
  }

  struct sockaddr_in6  client6_addr {};
  client6_addr.sin6_family = AF_INET6;
  client6_addr.sin6_port   = htons(RIC_SCTP_SRC_PORT);
  client6_addr.sin6_addr   = in6addr_any;

  LOG_I("[SCTP] Binding client socket to source port %d", RIC_SCTP_SRC_PORT);
  if(bind(client_fd, (struct sockaddr*)&client6_addr, sizeof(client6_addr)) == -1) {
    perror("bind");
    exit(1);
  }
  // end binding ---------------------

  LOG_I("[SCTP] Connecting to server at %s:%d ...", server_ip_str, server_port);
  if(connect(client_fd, server_addr, addr_len) == -1) {
    perror("connect");
    exit(1);
  }
  assert(client_fd != 0);

  LOG_I("[SCTP] Connection established");

  return client_fd;
}

int sctp_accept_connection(const char *server_ip_str, const int server_fd)
{
  LOG_I("[SCTP] Waiting for new connection...");

  struct sockaddr client_addr;
  socklen_t       client_addr_size;
  int             client_fd;

  //Blocking call
  client_fd = accept(server_fd, &client_addr, &client_addr_size);
  fprintf(stderr, "client fd is %d\n", client_fd);
  if(client_fd == -1){
    perror("accept()");
    close(client_fd);
    exit(1);
  }

  //Retrieve client IP_ADDR
  char client_ip6_addr[INET6_ADDRSTRLEN], client_ip4_addr[INET_ADDRSTRLEN];
  if(strchr(server_ip_str, ':') != NULL) //IPv6
  {
    struct sockaddr_in6* client_ipv6 = (struct sockaddr_in6*)&client_addr;
    inet_ntop(AF_INET6, &(client_ipv6->sin6_addr), client_ip6_addr, INET6_ADDRSTRLEN);
    LOG_I("[SCTP] New client connected from %s", client_ip6_addr);
  }
  else {
    struct sockaddr_in* client_ipv4 = (struct sockaddr_in*)&client_addr;
    inet_ntop(AF_INET, &(client_ipv4->sin_addr), client_ip4_addr, INET_ADDRSTRLEN);
    LOG_I("[SCTP] New client connected from %s", client_ip4_addr);
  }

  return client_fd;
}

int sctp_send_data(int &socket_fd, sctp_buffer_t &data)
{
  fprintf(stderr,"in sctp send data func\n");
  fprintf(stderr,"data.len is %d", data.len);
  int sent_len = send(socket_fd, (void*)(&(data.buffer[0])), data.len, 0);
  fprintf(stderr,"after getting sent_len\n");

  if(sent_len == -1) {
    perror("[SCTP] sctp_send_data");
    exit(1);
  }

  return sent_len;
}

int sctp_send_data_X2AP(int &socket_fd, sctp_buffer_t &data)
{
  int sent_len = sctp_sendmsg(socket_fd, (void*)(&(data.buffer[0])), data.len,
                  NULL, 0, (uint32_t) X2AP_PPID, 0, 0, 0, 0);

  if(sent_len == -1) {
    perror("[SCTP] sctp_send_data");
    exit(1);
  }

}

/*
Receive data from SCTP socket
Outcome of recv()
-1: exit the program
0: close the connection
+: new data
*/
int sctp_receive_data(int &socket_fd, sctp_buffer_t &data)
{
  //clear out the data before receiving
  fprintf(stderr, "receive data1\n");
  memset(data.buffer, 0, sizeof(data.buffer));
  fprintf(stderr, "receive data2\n");  
  data.len = 0;

  //receive data from the socket
  int recv_len = recv(socket_fd, &(data.buffer), sizeof(data.buffer), 0);
  fprintf(stderr, "receive data3\n");
  
  if(recv_len == -1)
  {
    perror("[SCTP] recv");
    exit(1);
  }
  else if (recv_len == 0)
  {
    LOG_I("[SCTP] Connection closed by remote peer");
    if(close(socket_fd) == -1)
    {
      perror("[SCTP] close");
    }
    return -1;
  }

  data.len = recv_len;

  return recv_len;
}
