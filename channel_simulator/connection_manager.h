#ifndef _CONNECTION_MANAGER_H_
#define _CONNECTION_MANAGER_H_

#include <stdint.h>
#include "utils.h"

typedef struct {
  int      socket;
  uint64_t rx_frequency;
  uint64_t tx_frequency;
  uint32_t samplerate;
} new_connection;

typedef struct {
  char           *ip;
  int            port;
  volatile int   new_connections;
  new_connection *c;
  lock_t         l;
} connection_manager;

void init_connection_manager(connection_manager *cm, char *listen_ip, int port);
void connection_manager_lock(connection_manager *cm);
void connection_manager_unlock(connection_manager *cm);
void connection_manager_wait_connection(connection_manager *cm);
void connection_manager_clear(connection_manager *cm);

#endif /* _CONNECTION_MANAGER_H_ */
