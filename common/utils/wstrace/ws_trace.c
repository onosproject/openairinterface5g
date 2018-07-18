#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/errno.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <string.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <arpa/inet.h>

#include "ws_trace.h"

pthread_t pid = -1;
pthread_mutex_t ws_lock = PTHREAD_MUTEX_INITIALIZER;

static int ws_client_socket = 0;
static volatile int ws_thread_disable = 0;

static void *ws_listen_thread(void *data)
{
  struct sockaddr_in servaddr; /* the server's full addr */
  int ws_server_socket = -1;
  int ep_fd = -1;
  struct epoll_event read_event;
  struct epoll_event events[10];
  int i, ret_ev, nbytes;
  unsigned char buffer[256];

  /* prepare ws server socket and lisener */

  if ((ws_server_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("create ws server socket failed");
    goto error_handler;
  }

  bzero((char *)&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = inet_addr(WS_TRACE_ADDRESS);
  servaddr.sin_port = htons(WS_TRACE_PORT);

  if (bind(ws_server_socket, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    perror("bind ws_server_socket failed!");
    goto error_handler;
  }

  ep_fd = epoll_create1(0);
  if (ep_fd < 0) {
    perror("epoll create failed!");
    goto error_handler;
  }

  bzero((char *)&read_event, sizeof(read_event));
  read_event.events = EPOLLIN;
  read_event.data.fd = ws_server_socket;

  epoll_ctl(ep_fd, EPOLL_CTL_ADD, ws_server_socket, &read_event);

  bzero((char *)&events, sizeof(events));

  while (!ws_thread_disable) {
    ret_ev = epoll_wait(ep_fd, &events[0], sizeof(events)/sizeof(events[0]), -1);
    for (i = 0; i < ret_ev; i++) {
      if ((events[i].events & EPOLLIN) &&
          (events[i].data.fd == ws_server_socket)) {
        nbytes = recv(ws_server_socket, &buffer, sizeof(buffer), 0);
      }
    }
  }

error_handler:
  if (ws_server_socket > 0) {
    close(ws_server_socket);
  }
  if (ep_fd > 0) {
    epoll_ctl(ep_fd, EPOLL_CTL_DEL, ws_server_socket, &read_event);
    close(ep_fd);
  }

  return NULL;
}

void start_ws_trace(void)
{

  struct sockaddr_in myaddr;   /* address that client uses */
  struct sockaddr_in servaddr; /* the server's full addr */

  if (ws_client_socket) {
    return;
  }

  pthread_mutex_init(&ws_lock, NULL);
  pthread_mutex_lock(&ws_lock);

  /* prepare ws client socket */

  if ((ws_client_socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    perror("create ws client socket failed");
    goto error_handler;
  }

  bzero((char *)&myaddr, sizeof(myaddr));
  myaddr.sin_family = AF_INET;
  myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  myaddr.sin_port = htons(0);

  if (bind(ws_client_socket, (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
    perror("bind ws_client_socket failed!");
    goto error_handler;
  }

  bzero((char *)&servaddr, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = inet_addr(WS_TRACE_ADDRESS);
  servaddr.sin_port = htons(WS_TRACE_PORT);

  if (connect(ws_client_socket, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
    perror("connect to ws server failed");
    goto error_handler;
  }

  if (pthread_create(&pid, NULL, ws_listen_thread, NULL) < 0) {
    perror("ws thread create failed!");
    goto error_handler;
  }

  pthread_mutex_unlock(&ws_lock);
  return;

error_handler:
  if (ws_client_socket > 0) {
    close(ws_client_socket);
    ws_client_socket = 0;
  }
  if (pid > 0) {
    pthread_join(pid, NULL);
  }
  pthread_mutex_unlock(&ws_lock);
  return;
}

void send_ws_log(unsigned short msg_type, unsigned short rnti, const unsigned char *msg_buf, unsigned short msg_len)
{
  struct iovec iov[2];
  ssize_t nwritten;
  unsigned short header[2];

  pthread_mutex_lock(&ws_lock);
  if (ws_client_socket == 0) {
    pthread_mutex_unlock(&ws_lock);
    return;
  }

  header[0] = msg_type;
  header[1] = rnti;

  iov[0].iov_base = (void *)&header[0];
  iov[0].iov_len =  sizeof(header);
  iov[1].iov_base = (void *)msg_buf;
  iov[1].iov_len = msg_len;

  writev(ws_client_socket, iov, 2);

  pthread_mutex_unlock(&ws_lock);

  return;
}

void stop_ws_trace(void)
{
  pthread_mutex_lock(&ws_lock);
  close(ws_client_socket);
  ws_client_socket = 0;

  ws_thread_disable = 1;

  if (pid > 0) {
    pthread_join(pid, NULL);
    pid = -1;
  }
  pthread_mutex_unlock(&ws_lock);
  pthread_mutex_destroy(&ws_lock);
}

#ifdef TEST
int main(int argc, char *argv[])
{
  start_ws_trace();
  sleep(1);
  send_ws_log(LTE_RRC_DL_CCCH, 2,  "Hello World !", strlen("Hello World !"));
  stop_ws_trace();
  return 0;
}
#endif
