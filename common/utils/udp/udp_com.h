#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/if_ether.h>
#include <linux/if_ether.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

#define OK  0
#define NG  (-1)

// message head
typedef struct{
  uint32_t  msgid;        /**< message ID          */
  uint32_t  srcFbNo;      /**< source FB number      */
  uint32_t  dstFbNo;      /**< destination FB number    */
  uint32_t  allMsgLen;      /**< total message length    */
  uint32_t  replyId;      /**< reply Id          */
  uint16_t  headSeqNo;      /**< head sequence number    */
  uint16_t  next;        /**< next flag          */
  uint16_t  nowSeqNo;      /**< current sequence number  */
  uint16_t  dummy;          /**< dummy                    */
  uint32_t  msgLen;        /**< single message length    */
} T_MSGHEAD;


typedef struct{
  T_MSGHEAD   msgHead;
  char        data[15000];
} T_UDP_MSG;

// server socket create
int DataLinkSocket(int addr, int port, int* fd);

// client socket create
int SendSocket(int32_t* fd);

// sendto msg
int PacketWrite(
  int32_t         fd,
  void            *buf,
  uint32_t        len,
  uint32_t        daddr,
  uint16_t        dport);

// recvfrom msg
int PacketRead(int32_t fd, void *buf, uint32_t len);

void closeSocket_fd(int32_t fd);
