#ifndef ISIP_HW_LIB_H
#define ISIP_HW_LIB_H

int ISIP_device_init(openair0_device* device, openair0_config_t *openair0_cfg);

void *eNB_thread_fronthaul( void *ptr );
void console_get_command(char* command);
void command_check(char * command);

int trx_ISIP_HW_read (void **buff, int nsamps, int cc);
int trx_ISIP_HW_write(void **buff, int nsamps, int cc);

int ISIP_HW_recv(void *buff, int nsamps);
int ISIP_HW_send(void *buff, int nsamps);

void unpack_data(int* oai_stream_buff, int* unpack_buff,int nsamps);
void pack_data(int* pack_buff, int* oai_stream_buff,int nsamps);

#endif // COMMON_LIB_H
