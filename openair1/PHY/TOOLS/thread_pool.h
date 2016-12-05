#ifndef _THREAD_POOL_H_
#define _THREAD_POOL_H_

#include <pthread.h>

struct PHY_VARS_eNB_s;

typedef struct thread_pool {
  pthread_mutex_t *mutex;
  pthread_cond_t *cond;
  pthread_mutex_t *mutex_join;
  pthread_cond_t *cond_join;
  int number_of_threads;
  volatile int running;
  volatile int done;
  /* processing data */
  volatile int subframe;
} thread_pool;

thread_pool *new_thread_pool(
    void *(*thread_fun)(struct PHY_VARS_eNB_s *eNB),
    struct PHY_VARS_eNB_s *eNB);

void thread_pool_start(thread_pool *pool);
void thread_pool_done(thread_pool *pool);
int thread_pool_wait(thread_pool *pool);
void thread_pool_join(thread_pool *pool);

#endif /* _THREAD_POOL_H_ */
