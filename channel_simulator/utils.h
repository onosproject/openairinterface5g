#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdint.h>
#include <pthread.h>

typedef struct {
  pthread_mutex_t m;
  pthread_cond_t c;
} lock_t;

void init_lock(lock_t *l);
void lock(lock_t *l);
void unlock(lock_t *l);
void lock_wait(lock_t *l);
void lock_signal(lock_t *l);

void new_thread(void *(*f)(void *), void *data);

int create_listen_socket(char *addr, int port);
int socket_accept(int s);
int fullread(int fd, void *buf, int count);
int fullwrite(int fd, void *buf, int count);

uint32_t gu32(unsigned char *x);
uint64_t gu64(unsigned char *x);
void pu32(unsigned char *x, uint32_t v);
void pu64(unsigned char *x, uint64_t v);

#endif /* _UTILS_H_ */
