#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

/************************************************************************/
/* lock                                                                 */
/************************************************************************/

void init_lock(lock_t *l)
{
  if (pthread_mutex_init(&l->m, NULL) ||
      pthread_cond_init(&l->c, NULL)) {
    printf("ERROR: init_lock failed\n");
    exit(1);
  }
}

void lock(lock_t *l)
{
  if (pthread_mutex_lock(&l->m)) {
    printf("ERROR: lock failed\n");
    exit(1);
  }
}

void unlock(lock_t *l)
{
  if (pthread_mutex_unlock(&l->m)) {
    printf("ERROR: unlock failed\n");
    exit(1);
  }
}

void lock_wait(lock_t *l)
{
  if (pthread_cond_wait(&l->c, &l->m)) {
    printf("ERROR: lock_wait failed\n");
    exit(1);
  }
}

void lock_signal(lock_t *l)
{
  if (pthread_cond_broadcast(&l->c)) {
    printf("ERROR: lock_signal failed\n");
    exit(1);
  }
}

/************************************************************************/
/* thread                                                               */
/************************************************************************/

void new_thread(void *(*f)(void *), void *data)
{
  pthread_t t;
  pthread_attr_t att;

  if (pthread_attr_init(&att)) goto error;
  if (pthread_attr_setdetachstate(&att, PTHREAD_CREATE_DETACHED)) goto error;
  if (pthread_attr_setstacksize(&att, 10000000)) goto error;
  if (pthread_create(&t, &att, f, data)) goto error;
  if (pthread_attr_destroy(&att)) goto error;
  return;

error:
  printf("ERROR: new_thread failed\n");
  exit(1);
}

/************************************************************************/
/* socket                                                               */
/************************************************************************/

#include <netinet/tcp.h>

int create_listen_socket(char *addr, int port)
{
  struct sockaddr_in a;
  int s;
  int v;

  s = socket(AF_INET, SOCK_STREAM, 0);
  if (s == -1) { perror("socket"); exit(1); }
  v = 1;
  if (setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &v, sizeof(int)))
    { perror("setsockopt"); exit(1); }

  v = 1;
  if (setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &v, sizeof(v)) == -1)
    { perror("setsockopt(NODELAY)"); exit(1); }

  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = inet_addr(addr);

  if (bind(s, (struct sockaddr *)&a, sizeof(a))) { perror("bind"); exit(1); }
  if (listen(s, 5)) { perror("listen"); exit(1); }

  return s;
}

int socket_accept(int s)
{
  struct sockaddr_in a;
  socklen_t alen;
  alen = sizeof(a);
  return accept(s, (struct sockaddr *)&a, &alen);
}

int fullread(int fd, void *_buf, int count)
{
  char *buf = _buf;
  int ret = 0;
  int l;
  while (count) {
    l = read(fd, buf, count);
    if (l <= 0) return -1;
    count -= l;
    buf += l;
    ret += l;
  }
  return ret;
}

int fullwrite(int fd, void *_buf, int count)
{
  char *buf = _buf;
  int ret = 0;
  int l;
  while (count) {
    l = write(fd, buf, count);
    if (l <= 0) return -1;
    count -= l;
    buf += l;
    ret += l;
  }
  return ret;
}

uint32_t gu32(unsigned char *x)
{
  return (uint32_t)x[0] |
       (((uint32_t)x[1]) <<  8) |
       (((uint32_t)x[2]) << 16) |
       (((uint32_t)x[3]) << 24);
}

uint64_t gu64(unsigned char *x)
{
  return (uint64_t)x[0] |
       (((uint64_t)x[1]) <<  8) |
       (((uint64_t)x[2]) << 16) |
       (((uint64_t)x[3]) << 24) |
       (((uint64_t)x[4]) << 32) |
       (((uint64_t)x[5]) << 40) |
       (((uint64_t)x[6]) << 48) |
       (((uint64_t)x[7]) << 56);
}

void pu32(unsigned char *x, uint32_t v)
{
  x[0] =  v        & 255;
  x[1] = (v >>  8) & 255;
  x[2] = (v >> 16) & 255;
  x[3] = (v >> 24) & 255;
}

void pu64(unsigned char *x, uint64_t v)
{
  x[0] =  v        & 255;
  x[1] = (v >>  8) & 255;
  x[2] = (v >> 16) & 255;
  x[3] = (v >> 24) & 255;
  x[4] = (v >> 32) & 255;
  x[5] = (v >> 40) & 255;
  x[6] = (v >> 48) & 255;
  x[7] = (v >> 56) & 255;
}
