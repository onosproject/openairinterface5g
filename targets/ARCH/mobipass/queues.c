#include "queues.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#define QSIZE 10000

struct mobipass_header {
  uint16_t flags;
  uint16_t fifo_status;
  unsigned char seqno;
  unsigned char ack;
  uint32_t word0;
  uint32_t timestamp;
} __attribute__((__packed__));

struct queue {
//  uint32_t next_timestamp;
  unsigned char buf[QSIZE][14+14+640*2];
  volatile int start;
  volatile int len;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

static struct queue to_mobipass;
static struct queue from_mobipass;

static void enqueue(void *data, struct queue *q)
{
  int pos;

  if (pthread_mutex_lock(&q->mutex)) abort();
  if (q->len == QSIZE) {
    printf("enqueue: full\n");
    goto done;
  }

  pos = (q->start + q->len) % QSIZE;
  memcpy(q->buf[pos], data, 14+14+640*2);
  q->len++;

done:
  if (pthread_cond_signal(&q->cond)) abort();
  if (pthread_mutex_unlock(&q->mutex)) abort();
}

void enqueue_to_mobipass(void *data)
{
  enqueue(data, &to_mobipass);
}

void dequeue_to_mobipass(uint32_t timestamp, void *data)
{
  if (pthread_mutex_lock(&to_mobipass.mutex)) abort();
  while (to_mobipass.len == 0) {
    if (pthread_cond_wait(&to_mobipass.cond, &to_mobipass.mutex)) abort();
  }

  memcpy(data, to_mobipass.buf[to_mobipass.start], 14+14+640*2);
  to_mobipass.len--;
  to_mobipass.start = (to_mobipass.start + 1) % QSIZE;

  if (pthread_mutex_unlock(&to_mobipass.mutex)) abort();
}

void enqueue_from_mobipass(void *data)
{
  struct mobipass_header *mh = (struct mobipass_header *)((char*)data+14);
//printf("from mobipass! timestamp %u seqno %d\n", ntohl(mh->timestamp), mh->seqno);
  enqueue(data, &from_mobipass);
}

/* to be called with lock on */
static void get_sample_from_mobipass(char *I, char *Q, uint32_t timestamp)
{
  unsigned char *b = NULL;
  unsigned char *data = NULL;
  struct mobipass_header *mh = NULL;
  uint32_t packet_timestamp = 0;

  while (from_mobipass.len) {
    b = from_mobipass.buf[from_mobipass.start];
    mh = (struct mobipass_header *)(b+14);
    data = b + 14*2;
    packet_timestamp = ntohl(mh->timestamp);
    if (timestamp < packet_timestamp) goto nodata;
    if (timestamp < packet_timestamp+640) break;
    from_mobipass.len--;
    from_mobipass.start = (from_mobipass.start+1) % QSIZE;
  }

  if (from_mobipass.len == 0) goto nodata;

  *I = data[(timestamp - packet_timestamp) * 2];
  *Q = data[(timestamp - packet_timestamp) * 2 + 1];

  return;

nodata:
  *I = 0;
  *Q = 0;
}

void dequeue_from_mobipass(uint32_t timestamp, void *data)
{
  int i;
  int ts = timestamp;
  int not_empty;

#if 0
  if (pthread_mutex_lock(&from_mobipass.mutex)) abort();
printf("want dequeue ts %u queue (start %d len %d): [", timestamp, from_mobipass.start, from_mobipass.len);
for (i = 0; i < from_mobipass.len; i++) {
  unsigned char *b = NULL;
  struct mobipass_header *mh = NULL;
  b = from_mobipass.buf[(from_mobipass.start + i) %QSIZE];
  mh = (struct mobipass_header *)(b+14);
  uint32_t packet_timestamp = ntohl(mh->timestamp);
  printf(" %d", packet_timestamp);
}
printf("]\n");
#endif

  if (from_mobipass.len == 0) {
    if (pthread_mutex_unlock(&from_mobipass.mutex)) abort();
    usleep(1000/3);
    if (pthread_mutex_lock(&from_mobipass.mutex)) abort();
  }

  not_empty = from_mobipass.len != 0;

  for (i = 0; i < 640*2; i+=2) {
    if (from_mobipass.len == 0 && not_empty) {
      if (pthread_mutex_unlock(&from_mobipass.mutex)) abort();
      usleep(1000/3);
      if (pthread_mutex_lock(&from_mobipass.mutex)) abort();
      not_empty = from_mobipass.len != 0;
    }

    get_sample_from_mobipass((char*)data + 14*2 + i, (char*)data + 14*2 + i+1, ts);
    ts++;
  }

  if (pthread_mutex_unlock(&from_mobipass.mutex)) abort();

  static int seqno = 0;

  struct mobipass_header *mh = (struct mobipass_header *)(((char *)data) + 14);
  mh->flags = 0;
  mh->fifo_status = 0;
  mh->seqno = seqno++;
  mh->ack = 0;
  mh->word0 = 0;
  mh->timestamp = htonl(timestamp);
}

void init_queues(void)
{
  if (pthread_mutex_init(&to_mobipass.mutex, NULL)) abort();
  if (pthread_mutex_init(&from_mobipass.mutex, NULL)) abort();
  if (pthread_cond_init(&to_mobipass.cond, NULL)) abort();
  if (pthread_cond_init(&from_mobipass.cond, NULL)) abort();
}
