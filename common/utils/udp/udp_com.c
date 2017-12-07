#include "udp_com.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <ctype.h>

/*!
 *  @brief  Send Socket
 *  @note  create send socket
 *  @param  fd      [out] socket descriptor
 *  @return  0 : success, -1 : error
 *  @date
 */
int
SendSocket(int* fd)
{
  if(fd == NULL){
    printf(" illegal parameter (fd)\n");
    return NG;
  }
  *fd = socket(AF_INET, SOCK_DGRAM, 0);

  if (*fd > 0) {
    // nothing to do
  } else {
    //  error occured
    printf("[SendSocket][ERR]:socket() call error (%m)\n");
    return NG;
  }
  return OK;
}

/*!
 *  @brief  Data Link Socket
 *  @note  create data link socket
 *  @param  addr     [in] ip address
 *  @param  port     [in] port address
 *  @param  fd      [out] socket descriptor
 *  @retval 0 : success, -1 : error
 *  @date
 */
int
DataLinkSocket(int addr, int port, int* fd)
{
  int ret;

  if(fd == NULL){
    printf("[DataLinkSocket][ERR]: illegal parameter (fd)\n");
    return NG;
  }
  struct sockaddr_in server_addr;
  bzero(&server_addr, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = addr;
  server_addr.sin_port = htons(port);

  *fd  = socket(AF_INET, SOCK_DGRAM, 0);

  if(*fd <= 0 ){
    printf("[DataLinkSocket][ERR]:socket() call error (%d)\n", *fd);
    return NG;
  }

  ret = 0;
  ret = bind(*fd, (struct sockaddr*)&server_addr, sizeof(struct sockaddr_in));
  if(ret != OK)
  {
    printf("[DataLinkSocket][ERR]:bind error(ret=%d,errno=%d)\n", ret,errno);
    return NG;
  }

  return OK;
}

/*!
 *  @brief  Packet Write
 *  @note  send message
 *  @param  fd    [in] socket descriptor
 *  @param  buf    [in] message
 *  @param  len    [in] message size
 *  @param  daddr  [in] ip address of destination  
 *  @param  dport  [in] port address of destination
 *  @return  data length that has been sent
 *  @retval  -1  : error
 *  @date
 */
int
PacketWrite(
  int32_t            fd,
  void            *buf,
  uint32_t          len,
  uint32_t          daddr,
  uint16_t          dport)
{
  int            slen;

  struct sockaddr_in    to;

  socklen_t        tolen = sizeof(to);

  if(buf == NULL){
    printf("[PacketWrite][ERR]: illegal parameter (buf)\n");
    return NG;
  }
  
  to.sin_family  = AF_INET;
  to.sin_port    = htons(dport);
  to.sin_addr.s_addr    = daddr;

  slen = sendto(fd, buf, len, 0, (struct sockaddr *)&to, tolen);


  if (slen == -1) {
    printf("[PacketWrite][ERR]:sendmsg() call error (%m)\n");
    printf("Check message length. You must set msgLen correctly. msgLen=%d\n", len);
  }
  return (slen);
}


/*!
 *  @brief  PacketRead
 *  @note  receive message
 *  @param  fd      [in] socket descriptor
 *  @param  buf      [in/out] buffer to save received data
 *  @param  len      [in] buffer length
 *  @return  received data length
 *  @retval  -1  : error
 *  @date
 */
int
PacketRead(int32_t fd, void *buf, uint32_t len)
{
  int        rlen = 0;
  struct sockaddr  from;
  socklen_t    fromlen = sizeof(from);
  struct timeval  timeout;
  fd_set    readfds;

  if(buf == NULL)
  {
    printf("[PacketReadFile][ERR]: illegal parameter (buf)\n");
    return NG;
  }

  for (;;) {
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);
    if (select(fd + 1, &readfds, NULL, NULL, &timeout) == 0) {
      return -2;  /* Time Out  */
    }

    rlen = recvfrom(fd, buf, len, MSG_TRUNC, &from, &fromlen);

    if (rlen == -1 || rlen == 0) {
      printf("[PacketRead][ERR]:recvfrom() call error (%m)\n");
    } else {
      return (rlen);
    }
  }

  return (rlen);
}


/*!
 *  @brief  closeSocket_fd
 *  @note   close socket descriptor
 *  @param  fd    [in] socket descriptor
 *  @return
 *  @date
 */
void closeSocket_fd(int fd){
  int ret;
  if(fd < 0){
      printf("[closeSocket_fd][WARN]: illegal parameter (fds)\n");
      return;
  }

  ret = close(fd);
  if(ret < 0){
      printf("[closeSocket_fd][ERR]: close call error (fd=%d)(errno=%d)\n",fd, errno);
      return;
  }
}
