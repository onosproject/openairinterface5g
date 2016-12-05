#include "thread_pool.h"
#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include "PHY/defs.h"

thread_pool *new_thread_pool(
    void *(*thread_fun)(PHY_VARS_eNB *arg),
    PHY_VARS_eNB *eNB)
{
  int i;
  pthread_t thread_id;
  pthread_attr_t th_att;
  thread_pool *ret;

  eNB->pool = calloc(1, sizeof(thread_pool));
  ret = eNB->pool;
  if (ret == NULL) abort();
  ret->mutex = calloc(1, sizeof(pthread_mutex_t));
  if (ret->mutex == NULL) abort();
  if (pthread_mutex_init(ret->mutex, NULL) != 0) abort();
  ret->cond = calloc(1, sizeof(pthread_cond_t));
  if (ret->cond == NULL) abort();
  if (pthread_cond_init(ret->cond, NULL) != 0) abort();
  ret->mutex_join = calloc(1, sizeof(pthread_mutex_t));
  if (ret->mutex_join == NULL) abort();
  if (pthread_mutex_init(ret->mutex_join, NULL) != 0) abort();
  ret->cond_join = calloc(1, sizeof(pthread_cond_t));
  if (ret->cond_join == NULL) abort();
  if (pthread_cond_init(ret->cond_join, NULL) != 0) abort();
  ret->number_of_threads = eNB->frame_parms.nb_antennas_tx;
  ret->running = 0;
  ret->done = ret->number_of_threads;
  for (i = 0; i < ret->number_of_threads; i++) {
    // struct sched_param th_params;
    if (pthread_attr_init(&th_att) != 0) abort();

    // if (pthread_attr_setschedpolicy(&th_att, SCHED_FIFO) != 0) abort();
    // th_params.sched_priority = 99;  /* max priority */
    // if (pthread_attr_setschedparam(&th_att, &th_params) != 0) abort();
    // if (pthread_attr_setinheritsched(&th_att, PTHREAD_EXPLICIT_SCHED) != 0) abort();

    if (pthread_create(&thread_id, &th_att, thread_fun, eNB) != 0) abort();
    if (pthread_attr_destroy(&th_att)) abort();
  }
  return ret;
}

void thread_pool_start(thread_pool *pool)
{
  if (pool->done != pool->number_of_threads) {
    fprintf(stderr, "thread_pool: ERROR: thread_pool_start called "
                    "with some threads still running\n");
    abort();
  }

  pool->done = 0;
  pool->running = pool->number_of_threads;

  if (pthread_cond_broadcast(pool->cond)) abort();
}

void thread_pool_done(thread_pool *pool)
{
  if (pthread_mutex_lock(pool->mutex_join)) abort();
  pool->done++;
  if (pthread_cond_signal(pool->cond_join)) abort();
  if (pthread_mutex_unlock(pool->mutex_join)) abort();
}

int thread_pool_wait(thread_pool *pool)
{
  int ret;
  if (pthread_mutex_lock(pool->mutex)) abort();
  while (pool->running == 0) {
    if (pthread_cond_wait(pool->cond, pool->mutex)) abort();
  }
  pool->running--;
  ret = pool->running;
  if (pthread_mutex_unlock(pool->mutex)) abort();
  return ret;
}

void thread_pool_join(thread_pool *pool)
{
  if (pthread_mutex_lock(pool->mutex_join)) abort();
  while (pool->done != pool->number_of_threads) {
    if (pthread_cond_wait(pool->cond_join, pool->mutex_join)) abort();
  }
  if (pthread_mutex_unlock(pool->mutex_join)) abort();
}
