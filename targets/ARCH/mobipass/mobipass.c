#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/if.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <linux/if_packet.h>
#include <netinet/ether.h>
#include <unistd.h>
#include <pthread.h>

#include "queues.h"

/******************************************************************/
/* time begin                                                     */
/******************************************************************/

#include <time.h>

#if 0
           struct timespec {
               time_t   tv_sec;        /* seconds */
               long     tv_nsec;       /* nanoseconds */
           };
#endif

static uint64_t t0;

static void init_time(void)
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &now)) abort();
  t0 = (uint64_t)now.tv_sec * (uint64_t)1000000000 + (uint64_t)now.tv_nsec;
}

static void synch_time(uint32_t ts)
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &now)) abort();

  uint64_t tnow;
  tnow = (uint64_t)now.tv_sec * (uint64_t)1000000000 + (uint64_t)now.tv_nsec;

  uint64_t cur = tnow - t0;

  /* 15360000 samples/second, in nanoseconds:
   *  = 15360000 / 1000000000 = 1536 / 100000 = 48 / 3125*/

  uint64_t ts_ns = (uint64_t)ts * (uint64_t)3125 / (uint64_t)48;

//printf("tnow %lu t0 %lu ts %u ts_ns %lu\n", tnow, t0, ts, ts_ns);
  if (cur >= ts_ns) return;

  uint64_t delta = ts_ns - cur;
  /* don't sleep more than 1 ms */
  if (delta > 1000*1000) delta = 1000*1000;
  delta = delta/1000;
printf("ts %u delta %lu\n", ts, delta);
  if (delta) usleep(delta);
}

/******************************************************************/
/* time end                                                       */
/******************************************************************/

static unsigned char receive_packet[14 + 14 + 1280];
static int sock;
static unsigned char packet[14 + 14 + 1280];

struct ethernet_header {
  unsigned char dst[6];
  unsigned char src[6];
  uint16_t packet_type;
} __attribute__((__packed__));

struct mobipass_header {
  uint16_t flags;
  uint16_t fifo_status;
  unsigned char seqno;
  unsigned char ack;
  uint32_t word0;
  uint32_t timestamp;
} __attribute__((__packed__));

static void receive(int sock, unsigned char *b)
{
  if (recv(sock, b, 14+14+1280, 0) != 14+14+1280) { perror("recv"); exit(1); }
  struct mobipass_header *mh = (struct mobipass_header *)(b+14);
  mh->timestamp = htonl(ntohl(mh->timestamp)-45378/*40120*/);
}

void mobipass_send(void *data)
{
//printf("SEND seqno %d ts %d\n", seqno, ts);
  struct ethernet_header *eh = (struct ethernet_header *)data;
  //struct mobipass_header *mh = (struct mobipass_header *)(data+14);
//printf("SEND seqno %d ts %d\n", mh->seqno, ntohl(mh->timestamp));

  eh->dst[0] = 0x00;
  eh->dst[1] = 0x21;
  eh->dst[2] = 0x5e;
  eh->dst[3] = 0x91;
  eh->dst[4] = 0x5c;
  eh->dst[5] = 0x7e;

  eh->src[0] = 0xf0;
  eh->src[1] = 0x1f;
  eh->src[2] = 0xaf;
  eh->src[3] = 0xdb;
  eh->src[4] = 0xb9;
  eh->src[5] = 0xc8;

  eh->packet_type = htons(0xbffe);

  enqueue_to_mobipass(data);
//  if (send(sock, data, 14+14+1280, 0) != 14+14+1280) { perror("recv"); exit(1); }

//usleep(1000/24+1);

#if 0
  mh->flags = 0;
  mh->fifo_status = 0;
  mh->seqno = seqno;
  mh->ack = 0;
  mh->word0 = 0;
  mh->timestamp = ntohl(ts);

  static int pos = 0;
  memcpy(packet+14+14, frame+pos, 640*2);
  pos += 640*2;
  if (pos >= 1024*10*7680*2*2) pos = 0;
  if (send(sock, packet, 14+14+1280, 0) != 14+14+1280) { perror("recv"); exit(1); }
#endif
}

static void *receiver(void *_)
{
//  unsigned char last_seqno = 0;
  while (1) {
    //struct ethernet_header *eh;
    //struct mobipass_header *mh;
    receive(sock, receive_packet);
    enqueue_from_mobipass(receive_packet);
#if 0
    eh = (struct ethernet_header *)receive_packet;
    printf("got dst %2.2x.%2.2x.%2.2x.%2.2x.%2.2x.%2.2x src %2.2x.%2.2x.%2.2x.%2.2x.%2.2x.%2.2x type %d\n",
      eh->dst[0], eh->dst[1], eh->dst[2],
      eh->dst[3], eh->dst[4], eh->dst[5],
      eh->src[0], eh->src[1], eh->src[2],
      eh->src[3], eh->src[4], eh->src[5],
      ntohs(eh->packet_type));
    eh = (struct ethernet_header *)receive_packet;
    mh = (struct mobipass_header *)(receive_packet+14);
    printf("  flag %d fifo %d seq %d ack %d w0 %d ts %d\n",
           ntohs(mh->flags),
           ntohs(mh->fifo_status),
           mh->seqno,
           mh->ack,
           ntohl(mh->word0),
           ntohl(mh->timestamp));
    last_seqno++;
    if (last_seqno != mh->seqno) printf("DISCONTINU\n");
    last_seqno = mh->seqno;
#endif
  }
  return 0;
}

void dosend(int sock, int seqno, uint32_t ts)
{
//printf("SEND seqno %d ts %d\n", seqno, ts);
  struct ethernet_header *eh = (struct ethernet_header *)packet;
  struct mobipass_header *mh = (struct mobipass_header *)(packet+14);

  eh->dst[0] = 0x00;
  eh->dst[1] = 0x21;
  eh->dst[2] = 0x5e;
  eh->dst[3] = 0x91;
  eh->dst[4] = 0x5c;
  eh->dst[5] = 0x7e;

  eh->src[0] = 0xf0;
  eh->src[1] = 0x1f;
  eh->src[2] = 0xaf;
  eh->src[3] = 0xdb;
  eh->src[4] = 0xb9;
  eh->src[5] = 0xc8;

  eh->packet_type = htons(0xbffe);

  mh->flags = 0;
  mh->fifo_status = 0;
  mh->seqno = seqno;
  mh->ack = 0;
  mh->word0 = 0;
  mh->timestamp = htonl(ts);

#if 0
  static int pos = 0;
  memcpy(packet+14+14, frame+pos, 640*2);
  pos += 640*2;
  if (pos >= 1024*10*7680*2*2) pos = 0;
#endif

  synch_time(ts);

  if (send(sock, packet, 14+14+1280, 0) != 14+14+1280) { perror("recv"); exit(1); }
  //usleep(1000/500);
  //usleep(1);
}

void *sender(void *_)
{
  uint32_t ts = 0;
  unsigned char seqno = 0;
  //int i;
  while (1) {
    dequeue_to_mobipass(ts, packet);
    dosend(sock, seqno, ts);
    seqno++;
    ts += 640;
  }
}

static void new_thread(void *(*f)(void *), void *data)
{
  pthread_t t;
  pthread_attr_t att;

  if (pthread_attr_init(&att))
    { fprintf(stderr, "pthread_attr_init err\n"); exit(1); }
  if (pthread_attr_setdetachstate(&att, PTHREAD_CREATE_DETACHED))
    { fprintf(stderr, "pthread_attr_setdetachstate err\n"); exit(1); }
  if (pthread_attr_setstacksize(&att, 10000000))
    { fprintf(stderr, "pthread_attr_setstacksize err\n"); exit(1); }
  if (pthread_create(&t, &att, f, data))
    { fprintf(stderr, "pthread_create err\n"); exit(1); }
  if (pthread_attr_destroy(&att))
    { fprintf(stderr, "pthread_attr_destroy err\n"); exit(1); }
}

void init_mobipass(void)
{
  int i;
  unsigned char data[14+14+640];
  memset(data, 0, 14+14+640);

  init_time();

  init_queues();

  for (i = 0; i < 24*4; i++) {
    uint32_t timestamp = i*640;
    unsigned char seqno = i;
    struct mobipass_header *mh = (struct mobipass_header *)(data+14);
    mh->seqno = seqno;
    mh->timestamp = htonl(timestamp);
    enqueue_to_mobipass(data);
  }

  sock = socket(AF_PACKET, SOCK_RAW, IPPROTO_RAW);
  if (sock == -1) { perror("socket"); exit(1); }

  /* get if index */
  struct ifreq if_index;
  memset(&if_index, 0, sizeof(struct ifreq));
  strcpy(if_index.ifr_name, "eth1.300");
  if (ioctl(sock, SIOCGIFINDEX, &if_index)<0) {perror("SIOCGIFINDEX");exit(1);}

  struct sockaddr_ll local_addr;
  local_addr.sll_family   = AF_PACKET;
  local_addr.sll_ifindex  = if_index.ifr_ifindex;
  local_addr.sll_protocol = htons(0xbffe);
  local_addr.sll_halen    = ETH_ALEN;
  local_addr.sll_pkttype  = PACKET_OTHERHOST;

  if (bind(sock, (struct sockaddr *)&local_addr, sizeof(struct sockaddr_ll))<0)
    { perror("bind"); exit(1); }

  new_thread(receiver, NULL);
  new_thread(sender, NULL);

#if 0

  while (1) {
    struct ethernet_header *eh;
    struct mobipass_header *mh;
    receive(sock, receive_packet);
    eh = (struct ethernet_header *)receive_packet;
    printf("got dst %2.2x.%2.2x.%2.2x.%2.2x.%2.2x.%2.2x src %2.2x.%2.2x.%2.2x.%2.2x.%2.2x.%2.2x type %d\n",
      eh->dst[0], eh->dst[1], eh->dst[2],
      eh->dst[3], eh->dst[4], eh->dst[5],
      eh->src[0], eh->src[1], eh->src[2],
      eh->src[3], eh->src[4], eh->src[5],
      ntohs(eh->packet_type));
    eh = (struct ethernet_header *)receive_packet;
    mh = (struct mobipass_header *)(receive_packet+14);
    printf("  flag %d fifo %d seq %d ack %d w0 %d ts %d\n",
           ntohs(mh->flags),
           ntohs(mh->fifo_status),
           mh->seqno,
           mh->ack,
           ntohl(mh->word0),
           ntohl(mh->timestamp));
  }
#endif
}
