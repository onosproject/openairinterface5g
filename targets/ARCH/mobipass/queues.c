#include "queues.h"
#include "mobipass.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>

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

//if (q == &from_mobipass) {
//struct mobipass_header *mh = (struct mobipass_header *)((char*)data+14);
//printf("recv timestamp %u in pos %d\n", ntohl(mh->timestamp), pos);
//}

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
  mh->timestamp = htonl(ntohl(mh->timestamp) % SAMPLES_PER_1024_FRAMES);
//printf("from mobipass! timestamp %u seqno %d\n", ntohl(mh->timestamp), mh->seqno);
  enqueue(data, &from_mobipass);
}

static int cmp_timestamps(uint32_t a, uint32_t b)
{
  if (a == b) return 0;
  if (a < b) {
    if (b-a > SAMPLES_PER_1024_FRAMES/2) return 1;
    return -1;
  }
  if (a-b > SAMPLES_PER_1024_FRAMES/2) return -1;
  return 1;
}

/* to be called with lock on */
static void get_sample_from_mobipass(char *I, char *Q, uint32_t timestamp)
{
  unsigned char *b = NULL;
  unsigned char *data = NULL;
  struct mobipass_header *mh = NULL;
  uint32_t packet_timestamp = 0;

uint32_t old_start = from_mobipass.start;
uint32_t old_len = from_mobipass.len;
b = from_mobipass.buf[from_mobipass.start];
mh = (struct mobipass_header *)(b+14);
uint32_t old_pts = from_mobipass.len ? ntohl(mh->timestamp) : -1;
b=NULL;
mh=NULL;

  while (from_mobipass.len) {
    b = from_mobipass.buf[from_mobipass.start];
    mh = (struct mobipass_header *)(b+14);
    data = b + 14*2;
    packet_timestamp = ntohl(mh->timestamp);
//printf("cmp A %u pt %u start %d\n", timestamp, packet_timestamp, from_mobipass.start);
    if (cmp_timestamps(timestamp, packet_timestamp) < 0) goto nodata;
//printf("cmp B %u pt %u\n", timestamp, packet_timestamp);
    if (cmp_timestamps(timestamp, (packet_timestamp+640) % SAMPLES_PER_1024_FRAMES) < 0) break;
//printf("cmp C %u pt %u\n", timestamp, packet_timestamp);
    from_mobipass.len--;
    from_mobipass.start = (from_mobipass.start+1) % QSIZE;
  }

  if (from_mobipass.len == 0) goto nodata;

  if (timestamp == (packet_timestamp + 639) % SAMPLES_PER_1024_FRAMES) {
    from_mobipass.len--;
    from_mobipass.start = (from_mobipass.start+1) % QSIZE;
  }

  if (timestamp < packet_timestamp) timestamp += SAMPLES_PER_1024_FRAMES;

  *I = data[(timestamp - packet_timestamp) * 2];
  *Q = data[(timestamp - packet_timestamp) * 2 + 1];

  return;

nodata:
  *I = 0;
  *Q = 0;
printf("no sample timestamp %u pt %u start %d old_start %d old_pt %u len %d old len %d\n", timestamp, packet_timestamp, from_mobipass.start, old_start, old_pts, from_mobipass.len, old_len);
}

/* doesn't work with delay more than 1s */
static void wait_for_data(pthread_cond_t *cond, pthread_mutex_t *mutex, int delay_us)
{
  struct timeval now;
  struct timespec target;
  gettimeofday(&now, NULL);
  target.tv_sec = now.tv_sec;
  target.tv_nsec = (now.tv_usec + delay_us) * 1000;
  if (target.tv_nsec >= 1000 * 1000 * 1000) { target.tv_nsec -= 1000 * 1000 * 1000; target.tv_sec++; }
  int err = pthread_cond_timedwait(cond, mutex, &target);
  if (err != 0 && err != ETIMEDOUT) { printf("pthread_cond_timedwait: err (%d) %s\n", err, strerror(err)); abort(); }
}

void dequeue_from_mobipass(uint32_t timestamp, void *data)
{
  int i;
//  int ts = timestamp;
  int waiting_allowed;

  if (pthread_mutex_lock(&from_mobipass.mutex)) abort();

#if 0
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
    //if (pthread_mutex_unlock(&from_mobipass.mutex)) abort();
printf("sleep 1\n");
    //usleep(1000/3);
    //if (pthread_mutex_lock(&from_mobipass.mutex)) abort();
    wait_for_data(&from_mobipass.cond, &from_mobipass.mutex, 2000); //1000/3);
  }

  waiting_allowed = from_mobipass.len != 0;

  for (i = 0; i < 640*2; i+=2) {
    if (from_mobipass.len == 0 && waiting_allowed) {
      //if (pthread_mutex_unlock(&from_mobipass.mutex)) abort();
//printf("sleep 2\n");
      //usleep(1000/3);
      //if (pthread_mutex_lock(&from_mobipass.mutex)) abort();
      wait_for_data(&from_mobipass.cond, &from_mobipass.mutex, 2000); //1000/3);
      waiting_allowed = from_mobipass.len != 0;
    }

    get_sample_from_mobipass((char*)data + 14*2 + i, (char*)data + 14*2 + i+1, timestamp % SAMPLES_PER_1024_FRAMES);
    timestamp++;
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
