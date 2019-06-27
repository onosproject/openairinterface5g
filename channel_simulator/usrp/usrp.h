#ifndef _USRP_H_openair
#define _USRP_H_openair

#include <stdint.h>

void usrp_init_connection(uint64_t rx_freq, uint64_t tx_freq);
void usrp_start(void);
uint64_t usrp_read(char *buf, int samples_count);
void usrp_write(char *buf, int samples_count, uint64_t timestamp);

#endif /* _USRP_H_openair */
