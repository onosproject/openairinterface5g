#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/epoll.h>
#include <string.h>

#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>
#include "common_lib.h"
#include <openair1/PHY/defs_eNB.h>
#include "openair1/PHY/defs_UE.h"

#define PORT 4043 //TCP port for this simulator
#define CirSize 3072000 // 100ms is enough
#define sample_t uint32_t // 2*16 bits complex number
#define sampleToByte(a,b) ((a)*(b)*sizeof(sample_t))
#define byteToSample(a,b) ((a)/(sizeof(sample_t)*(b)))
#define MAGICeNB 0xA5A5A5A5A5A5A5A5
#define MAGICUE  0x5A5A5A5A5A5A5A5A

typedef struct {
  uint64_t magic;
  uint32_t size;
  uint32_t nbAnt;
  uint64_t timestamp;
} transferHeader;

typedef struct buffer_s {
  int conn_sock;
  bool alreadyWrote;
  uint64_t lastReceivedTS;
  bool headerMode;
  transferHeader th;
  char *transferPtr;
  uint64_t remainToTransfer;
  char *circularBufEnd;
  sample_t *circularBuf;
} buffer_t;


typedef struct {
  int listen_sock, epollfd;
  uint64_t nextTimestamp;
  uint64_t typeStamp;
  char *ip;
  buffer_t buf[FD_SETSIZE];
} tcp_bridge_state_t;

void allocCirBuf(tcp_bridge_state_t *bridge, int sock) {
  buffer_t *ptr=&bridge->buf[sock];
  AssertFatal ( (ptr->circularBuf=(sample_t *) malloc(sampleToByte(CirSize,1))) != NULL, "");
  ptr->circularBufEnd=((char *)ptr->circularBuf)+sampleToByte(CirSize,1);
  ptr->conn_sock=sock;
  ptr->headerMode=true;
  ptr->transferPtr=(char *)&ptr->th;
  ptr->remainToTransfer=sizeof(transferHeader);
  int sendbuff=1000*1000*10;
  AssertFatal ( setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sendbuff, sizeof(sendbuff)) == 0, "");
  struct epoll_event ev= {0};
  ev.events = EPOLLIN | EPOLLRDHUP;
  ev.data.fd = sock;
  AssertFatal(epoll_ctl(bridge->epollfd, EPOLL_CTL_ADD,  sock, &ev) != -1, "");
}

void removeCirBuf(tcp_bridge_state_t *bridge, int sock) {
  AssertFatal( epoll_ctl(bridge->epollfd, EPOLL_CTL_DEL,  sock, NULL) != -1, "");
  close(sock);
  free(bridge->buf[sock].circularBuf);
  memset(&bridge->buf[sock], 0, sizeof(buffer_t));
  bridge->buf[sock].conn_sock=-1;
}

#define helpTxt "\
\x1b[31m\
tcp_bridge: error: you have to run one UE and one eNB\n\
For this, export RFSIMULATOR=enb (eNB case) or \n\
                 RFSIMULATOR=<an ip address> (UE case)\n\
\x1b[m"

int fullwrite(int fd, void *_buf, int count) {
  char *buf = _buf;
  int ret = 0;
  int l;

  while (count) {
    l = write(fd, buf, count);

    if (l <= 0) {
      if(errno==EAGAIN || errno==EINTR)
        continue;
      else
        return -1;
    }

    count -= l;
    buf += l;
    ret += l;
  }

  return ret;
}

enum  blocking_t {
  blocking,
  notBlocking
};

void setblocking(int sock, enum blocking_t active) {
  int opts;
  AssertFatal( (opts = fcntl(sock, F_GETFL)) >= 0,"");

  if (active==blocking)
    opts = opts & ~O_NONBLOCK;
  else
    opts = opts | O_NONBLOCK;

  AssertFatal(fcntl(sock, F_SETFL, opts) >= 0, "");
}


tcp_bridge_state_t *init_bridge(openair0_device *device) {
  tcp_bridge_state_t *tcp_bridge;

  if (device->priv)
    tcp_bridge=(tcp_bridge_state_t *) device->priv;
  else
    AssertFatal(((tcp_bridge=(tcp_bridge_state_t *)calloc(sizeof(tcp_bridge_state_t),1))) != NULL, "");

  for (int i=0; i<FD_SETSIZE; i++)
    tcp_bridge->buf[i].conn_sock=-1;

  device->priv = tcp_bridge;
  AssertFatal((tcp_bridge->epollfd = epoll_create1(0)) != -1,"");
  return tcp_bridge;
}

int server_start(openair0_device *device) {
  tcp_bridge_state_t *t = init_bridge(device);
  t->typeStamp=MAGICeNB;
  AssertFatal((t->listen_sock = socket(AF_INET, SOCK_STREAM, 0)) >= 0, "");
  int enable = 1;
  AssertFatal(setsockopt(t->listen_sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) == 0, "");
  struct sockaddr_in addr = {
  sin_family: AF_INET,
  sin_port: htons(PORT),
  sin_addr: { s_addr: INADDR_ANY }
  };
  bind(t->listen_sock, (struct sockaddr *)&addr, sizeof(addr));
  AssertFatal(listen(t->listen_sock, 5) == 0, "");
  struct epoll_event ev;
  ev.events = EPOLLIN;
  ev.data.fd = t->listen_sock;
  AssertFatal(epoll_ctl(t->epollfd, EPOLL_CTL_ADD,  t->listen_sock, &ev) != -1, "");
  return 0;
}

int start_ue(openair0_device *device) {
  tcp_bridge_state_t *t = init_bridge(device);
  t->typeStamp=MAGICUE;
  int sock;
  AssertFatal((sock = socket(AF_INET, SOCK_STREAM, 0)) >= 0, "");
  struct sockaddr_in addr = {
  sin_family: AF_INET,
  sin_port: htons(PORT),
  sin_addr: { s_addr: INADDR_ANY }
  };
  addr.sin_addr.s_addr = inet_addr(t->ip);
  bool connected=false;

  while(!connected) {
    printf("tcp_bridge: trying to connect to %s:%d\n", t->ip, PORT);

    if (connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
      printf("tcp_bridge: connection established\n");
      connected=true;
    }

    perror("tcp_bridge");
    sleep(1);
  }

  setblocking(sock, notBlocking);
  allocCirBuf(t, sock);
  t->buf[sock].alreadyWrote=true;
  return 0;
}

int tcp_bridge_write(openair0_device *device, openair0_timestamp timestamp, void **samplesVoid, int nsamps, int nbAnt, int flags) {
  
  tcp_bridge_state_t *t = device->priv;
  
  for (int i=0; i<FD_SETSIZE; i++) {
    buffer_t *ptr=&t->buf[i];

    if (ptr->conn_sock >= 0 ) {
      setblocking(ptr->conn_sock, blocking);
      transferHeader header= {t->typeStamp, nsamps, nbAnt, timestamp};
      int n=-1;

      AssertFatal( fullwrite(ptr->conn_sock,&header, sizeof(header)) == sizeof(header), "");
      sample_t tmpSamples[nsamps][nbAnt];
      for(int a=0; a<nbAnt; a++) {
	sample_t* in=(sample_t*)samplesVoid[a];
	for(int s=0; s<nsamps; s++)
	  tmpSamples[s][a]=in[s];
      }
      n = fullwrite(ptr->conn_sock, (void*)tmpSamples, sampleToByte(nsamps,nbAnt));

      if (n != sampleToByte(nsamps,nbAnt) ) {
        printf("tcp_bridge: write error ret %d (wanted %ld) error %s\n", n, sampleToByte(nsamps,nbAnt), strerror(errno));
        abort();
      }

      ptr->alreadyWrote=true;
      setblocking(ptr->conn_sock, notBlocking);
    }
  }

  LOG_D(HW,"sent %d samples at time: %ld->%ld, energy in first antenna: %d\n",
        nsamps, timestamp, timestamp+nsamps, signal_energy(samplesVoid[0], nsamps) );
  return nsamps;
}

bool flushInput(tcp_bridge_state_t *t) {
  // Process all incoming events on sockets
  // store the data in lists
  bool completedABuffer=false;
  int iterations=10;

  while (!completedABuffer && iterations-- ) {
    struct epoll_event events[FD_SETSIZE]= {0};
    int nfds = epoll_wait(t->epollfd, events, FD_SETSIZE, 20);

    if ( nfds==-1 ) {
      if ( errno==EINTR || errno==EAGAIN )
        continue;
      else
        AssertFatal(false,"error in epoll_wait\n");
    }

    //printf("waited iter=%d, res %d, waiting fd %d\n", iterations, nfds, nfds>=1? events[0].data.fd:-1);

    for (int nbEv = 0; nbEv < nfds; ++nbEv) {
      int fd=events[nbEv].data.fd;

      if (events[nbEv].events & EPOLLIN && fd == t->listen_sock) {
        int conn_sock;
        AssertFatal( (conn_sock = accept(t->listen_sock,NULL,NULL)) != -1, "");
        allocCirBuf(t, conn_sock);
        LOG_I(HW,"A ue connected\n");
      } else {
        if ( events[nbEv].events & (EPOLLHUP | EPOLLERR | EPOLLRDHUP) ) {
          LOG_W(HW,"Lost socket\n");
          removeCirBuf(t, fd);

          if (t->typeStamp==MAGICUE)
            exit(1);

          continue;
        }

        buffer_t *b=&t->buf[fd];

        if ( b->circularBuf == NULL ) {
          LOG_E(HW, "received data on not connected socket %d\n", events[nbEv].data.fd);
          continue;
        }

        int blockSz;

        if ( b->headerMode)
          blockSz=b->remainToTransfer;
        else
          blockSz= b->transferPtr+b->remainToTransfer < b->circularBufEnd ?
	    b->remainToTransfer :
	    b->circularBufEnd - 1 - b->transferPtr ;

        int sz=recv(fd, b->transferPtr, blockSz, MSG_DONTWAIT);

        if ( sz < 0 ) {
          if ( errno != EAGAIN ) {
            LOG_E(HW,"socket failed %s\n", strerror(errno));
            abort();
          }
        } else if ( sz == 0 )
          continue;

        AssertFatal((b->remainToTransfer-=sz) >= 0, "");
        b->transferPtr+=sz;
	if (b->transferPtr==b->circularBufEnd - 1)
		b->transferPtr=(char*)b->circularBuf;

        // check the header and start block transfer
        if ( b->headerMode==true && b->remainToTransfer==0) {
	  AssertFatal( (t->typeStamp == MAGICUE  && b->th.magic==MAGICeNB) ||
                       (t->typeStamp == MAGICeNB && b->th.magic==MAGICUE), "Socket Error in protocol");
          b->headerMode=false;
          b->lastReceivedTS=b->th.timestamp;
          b->transferPtr=(char *)&b->circularBuf[b->lastReceivedTS%CirSize];
          b->remainToTransfer=sampleToByte(b->th.size, b->th.nbAnt);
        }

        if ( b->headerMode==false ) {
	  b->lastReceivedTS=b->th.timestamp+b->th.size-byteToSample(b->remainToTransfer,b->th.nbAnt);
	  if ( b->remainToTransfer==0) {
	    completedABuffer=true;
	    LOG_D(HW,"Completed block reception: %ld\n", b->lastReceivedTS);
	    // First block in UE, resync with the eNB current TS
	    if ( t->nextTimestamp == 0 )
	      t->nextTimestamp=b->lastReceivedTS-b->th.size;
	    b->headerMode=true;
	    b->transferPtr=(char *)&b->th;
	    b->remainToTransfer=sizeof(transferHeader);
	    b->th.magic=-1;
	  } 
        }
      }
    }
  }
  
  return completedABuffer;
}

int tcp_bridge_read(openair0_device *device, openair0_timestamp *ptimestamp, void **samplesVoid, int nsamps, int nbAnt) {
  if (nbAnt != 1) { printf("tcp_bridge: only 1 antenna tested\n"); exit(1); }

  tcp_bridge_state_t *t = device->priv;
  // deliver data from received data
  // check if a UE is connected
  int first_sock;

  for (first_sock=0; first_sock<FD_SETSIZE; first_sock++)
    if (t->buf[first_sock].circularBuf != NULL )
      break;

  if ( first_sock ==  FD_SETSIZE ) {
    // no connected device (we are eNB, no UE is connected)
    if (!flushInput(t)) {
      for (int x=0; x < nbAnt; x++) 
	memset(samplesVoid[x],0,sampleToByte(nsamps,1));
      t->nextTimestamp+=nsamps;
      LOG_W(HW,"Generated void samples for Rx: %ld\n", t->nextTimestamp);
      for (int a=0; a<nbAnt; a++) {
	sample_t *out=(sample_t *)samplesVoid[a];
	for ( int i=0; i < nsamps; i++ )
	  out[i]=0;
      }

      *ptimestamp = t->nextTimestamp-nsamps;
      return nsamps;
    }
  } else {
    bool have_to_wait;

    do {
      have_to_wait=false;

      for ( int sock=0; sock<FD_SETSIZE; sock++)
        if ( t->buf[sock].circularBuf &&
             t->buf[sock].alreadyWrote &&
             (t->nextTimestamp+nsamps) > t->buf[sock].lastReceivedTS ) {
          have_to_wait=true;
          break;
        }

      if (have_to_wait)
        /*printf("Waiting on socket, current last ts: %ld, expected at least : %ld\n",
          ptr->lastReceivedTS,
          t->nextTimestamp+nsamps);
        */
        flushInput(t);
    } while (have_to_wait);
  }

  // Clear the output buffer
  for (int a=0; a<nbAnt; a++) {
    sample_t *out=(sample_t *)samplesVoid[a];
    for ( int i=0; i < nsamps; i++ )
      out[i]=0;
  }

  // Add all input signal in the output buffer
  for (int sock=0; sock<FD_SETSIZE; sock++) {
    buffer_t *ptr=&t->buf[sock];

    if ( ptr->circularBuf && ptr->alreadyWrote ) {
      for (int a=0; a<nbAnt; a++) {
	sample_t *out=(sample_t *)samplesVoid[a];
	for ( int i=0; i < nsamps; i++ )
	  out[i]+=ptr->circularBuf[(t->nextTimestamp+(a*nbAnt+i))%CirSize]<<1;
      }
    }
  }

  *ptimestamp = t->nextTimestamp; // return the time of the first sample
  t->nextTimestamp+=nsamps;
  LOG_D(HW,"Rx to upper layer: %d from %ld to %ld, energy in first antenna %d\n",
	nsamps,
	*ptimestamp, t->nextTimestamp,
	signal_energy(samplesVoid[0], nsamps));
  return nsamps;
}


int tcp_bridge_request(openair0_device *device, void *msg, ssize_t msg_len) {
  abort();
  return 0;
}
int tcp_bridge_reply(openair0_device *device, void *msg, ssize_t msg_len) {
  abort();
  return 0;
}
int tcp_bridge_get_stats(openair0_device *device) {
  return 0;
}
int tcp_bridge_reset_stats(openair0_device *device) {
  return 0;
}
void tcp_bridge_end(openair0_device *device) {}
int tcp_bridge_stop(openair0_device *device) {
  return 0;
}
int tcp_bridge_set_freq(openair0_device *device, openair0_config_t *openair0_cfg,int exmimo_dump_config) {
  return 0;
}
int tcp_bridge_set_gains(openair0_device *device, openair0_config_t *openair0_cfg) {
  return 0;
}


__attribute__((__visibility__("default")))
int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {
  //set_log(HW,OAILOG_DEBUG);
  tcp_bridge_state_t *tcp_bridge = (tcp_bridge_state_t *)calloc(sizeof(tcp_bridge_state_t),1);

  if ((tcp_bridge->ip=getenv("RFSIMULATOR")) == NULL ) {
    printf(helpTxt);
    exit(1);
  }

  tcp_bridge->typeStamp = strncasecmp(tcp_bridge->ip,"enb",3) == 0 ?
    MAGICeNB:
    MAGICUE;
  printf("tcp_bridge: running as %s\n", tcp_bridge-> typeStamp == MAGICeNB ? "eNB" : "UE");

  /* only 25, 50 or 100 PRBs handled for the moment */
  if (openair0_cfg[0].sample_rate != 30720000 &&
      openair0_cfg[0].sample_rate != 15360000 &&
      openair0_cfg[0].sample_rate !=  7680000) {
    printf("tcp_bridge: ERROR: only 25, 50 or 100 PRBs supported\n");
    exit(1);
  }

  device->trx_start_func       = tcp_bridge->typeStamp == MAGICeNB ?
    server_start :
    start_ue;
  device->trx_get_stats_func   = tcp_bridge_get_stats;
  device->trx_reset_stats_func = tcp_bridge_reset_stats;
  device->trx_end_func         = tcp_bridge_end;
  device->trx_stop_func        = tcp_bridge_stop;
  device->trx_set_freq_func    = tcp_bridge_set_freq;
  device->trx_set_gains_func   = tcp_bridge_set_gains;
  device->trx_write_func       = tcp_bridge_write;
  device->trx_read_func      = tcp_bridge_read;
  device->priv = tcp_bridge;
  /* let's pretend to be a b2x0 */
  device->type = USRP_B200_DEV;
  device->openair0_cfg=&openair0_cfg[0];
  return 0;
}
