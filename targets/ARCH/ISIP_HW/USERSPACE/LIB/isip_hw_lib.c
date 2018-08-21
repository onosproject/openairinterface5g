#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>	

//#define MULTI_LANE	   

int number_of_rrh = 4; 

static int  uplink_rd_fd = 0; 
static int  uplink_wr_fd = 0; 
static int  receive_times; 
static char *uplink_wr_filename; 

static int  downlink_wr_fd = 0;
static int  downlink_rd_fd = 0; 
extern int  debug_sent_fd=0, debug_sent_fd1=0, debug_sent_fd2=0, debug_sent_fd3=0; 

int *isip_sent_int_buff;
int *isip_sent_int_buff1;
int *isip_sent_int_buff2;
int *isip_sent_int_buff3;

void *isip_sent_buff; 
void *isip_sent_buff1;
void *isip_sent_buff2; 
void *isip_sent_buff3; 

int switch_buf = 0;
int first_sent = 1, first_received=1;

int pattern_record;
	

extern int  debug_file_fd=0, debug_file_fd1 =0; 
   
int tx_lane_sel = 0;      
int rx_lane_sel = 0;   

void pack_data(int* pack_buff, int* oai_stream_buff,int nsamps){
	for(int i = 0; i < nsamps; i++)
		pack_buff[i*number_of_rrh+tx_lane_sel] = oai_stream_buff[i];  
}  

void unpack_data(int* oai_stream_buff, int* unpack_buff,int nsamps){
	for(int i = 0; i < nsamps; i++)
		oai_stream_buff[i] = unpack_buff[i*number_of_rrh+rx_lane_sel];       
}


int ISIP_HW_send(void *buff, int nsamps) 
{ 	
	int written_bytes=0, do_bytes =0;
	int rc = 0;

	do_bytes = nsamps*4;
	if (do_bytes == 0){  		
		return 0;
	} 

	for (; do_bytes > 0; buff += written_bytes, do_bytes -= written_bytes) 
	{ 
		
		written_bytes = write(downlink_wr_fd, buff, do_bytes); 
		if ((written_bytes < 0) && (errno != EINTR)) 
		{
			perror("write() failed");
			return NULL;
		}
		if (written_bytes == 0) 
		{		  
			fprintf(stderr, "Reached write EOF (?!)\n");
			return NULL;
		}
		if (written_bytes < 0) { // errno is EINTR
			written_bytes = 0;
			continue;
		}

		// write flush
		while (1) {
			rc = write(downlink_wr_fd, NULL, 0); 
			if ((rc < 0) && (errno == EINTR))
				continue; // Interrupted. Try again.
			 
			if (rc < 0) { 
				perror("flushing failed");
				break;
			}
			break; // Flush successful
		} 			
	} 
	return 0;
}

int ISIP_HW_recv(void *buff, int nsamps)
{
	int do_bytes, read_bytes; 
 
	do_bytes = nsamps*4; 
	if (do_bytes == 0){  	 	
		return 0;
	}
	
	for (; do_bytes > 0; buff += read_bytes, do_bytes -= read_bytes) 
	{
		read_bytes = read(uplink_rd_fd, buff, do_bytes);
		if ((read_bytes < 0) && (errno != EINTR)) {
			perror("read() failed");
			return NULL;
		}

		if (read_bytes == 0) {
			// Reached EOF. Quit without complaining.
			return NULL;
		}

		if (read_bytes < 0) { // errno is EINTR
			read_bytes = 0;
			continue;
		}
	} 
	
	return (nsamps*4-do_bytes)/4; //recv bytes	
}

int trx_ISIP_HW_write(void **buff, int nsamps, int cc) 
{
	if (cc>1) {
		//ISIP unsupport
	} 
	else {     
		
#ifdef MULTI_LANE  
		if(first_sent){ 
			ISIP_HW_send(isip_sent_buff1, nsamps*number_of_rrh);  
			isip_sent_int_buff1[tx_lane_sel] = 0;
			ISIP_HW_send(isip_sent_buff1, nsamps*number_of_rrh);   
			first_sent=0;
		}  
	
		pack_data(isip_sent_buff1,buff[0],nsamps);  
		ISIP_HW_send(isip_sent_buff1, nsamps*number_of_rrh);   
		
#else
		if(first_sent){ 
			ISIP_HW_send(isip_sent_buff1, nsamps);  
			isip_sent_int_buff1[0] = 0;
			ISIP_HW_send(isip_sent_buff1, nsamps);   
			first_sent=0;
		}  
	
		memcpy(isip_sent_buff1,buff[0],nsamps*4);	 
		ISIP_HW_send(isip_sent_buff1, nsamps);    
		
		//printf("ISIP_SEND nsampes = %d\n",nsamps);  
		//memcpy(isip_sent_buff1,isip_sent_buff,nsamps*4);	
#endif 		
		/*    
		if(switch_buf == 0) { 
			ISIP_HW_send(isip_sent_buff1, nsamps);  
			memcpy(isip_sent_buff1,buff[0],nsamps*4);
		}  
		else {    
			ISIP_HW_send(isip_sent_buff2, nsamps);     
			memcpy(isip_sent_buff2,buff[0],nsamps*4);
		}   
		switch_buf = (switch_buf+1)%2; 
		*/      
	}  
	//printf("isip send ...\n"); 
	return nsamps;
}  

	
int trx_ISIP_HW_read(void **buff, int nsamps, int cc)
{
   int samples_received=0,i,j;
	
   if (cc>1) {
   // receive multiple channels (e.g. RF A and RF B) 
	 // ISIP HW unsupport
   } 
   else {
   // receive a single channel (e.g. from connector RF A)
#ifdef MULTI_LANE 	 
	int sample_advance;
	switch (nsamps) {
		case 30720:
			sample_advance = 100;  

			if(first_received){  
				samples_received = ISIP_HW_recv(isip_sent_buff2, sample_advance*number_of_rrh); 
				first_received=0; 
			}
		
			samples_received = ISIP_HW_recv(isip_sent_buff2, nsamps*number_of_rrh); 
			unpack_data(buff[0],isip_sent_buff2,nsamps);
			samples_received /= number_of_rrh;
			
			break; 
		case 7680:
			sample_advance = 15;  

			if(first_received){  
				samples_received = ISIP_HW_recv(isip_sent_buff2, sample_advance*number_of_rrh); 
				first_received=0; 
			}
		
			samples_received = ISIP_HW_recv(isip_sent_buff2, nsamps*number_of_rrh); 
			unpack_data(buff[0],isip_sent_buff2,nsamps);
			samples_received /= number_of_rrh;
			break;
			
		default:
		  printf("unsupport sampling rate %d\n",nsamps*1000);
		  exit(-1);
		  break;
	}

#else
 
	switch (nsamps) {
		case 30720:
			if(first_received){  
				samples_received = ISIP_HW_recv(buff[0], 100);  
				first_received=0; 
			}  
		
			samples_received = ISIP_HW_recv(buff[0], nsamps); 		

			break; 
		case 7680:
			if(first_received){  
				samples_received = ISIP_HW_recv(buff[0], 15); //best 25  //47	//sma:61 //dudd : 37
				first_received=0; 
			}
		
			samples_received = ISIP_HW_recv(buff[0], nsamps); 
			break;
		default:
		  printf("unsupport sampling rate %d\n",nsamps*1000);
		  exit(-1); 
		  break;
	}
 	
#endif	
   } 
    

   if (samples_received < nsamps) {
	 printf("[recv] received %d samples out of %d\n",samples_received,nsamps);
	 
   } 
	 
	//write(debug_file_fd, buff[0], samples_received*4);       
	//write(debug_file_fd1, buff[0], samples_received*4);  
  // printf("trx_ISIP_HW_read end ... \n");  
 // printf("isip received ...\n");  
   return samples_received; 
}	 


	
int ISIP_device_init(openair0_device* device, openair0_config_t *openair0_cfg){
	printf("[ISIP] ISIP_HW initalization............\n");

#ifdef MULTI_LANE  	
	downlink_wr_fd = open("/dev/xillybus_write_128", O_WRONLY);
	if (downlink_wr_fd < 0) {   
		if (errno == ENODEV)
		fprintf(stderr, "(Maybe xillybus_write a read-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}    
	 
	uplink_rd_fd = open("/dev/xillybus_read_128", O_RDONLY);
	if (uplink_rd_fd < 0) 
	{ 
		if (errno == ENODEV)
		  fprintf(stderr, "(Maybe xillybus_read a read-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	} 
#else
	downlink_wr_fd = open("/dev/xillybus_write_32", O_WRONLY);
	if (downlink_wr_fd < 0) {    
		if (errno == ENODEV)
		fprintf(stderr, "(Maybe xillybus_write a read-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}     
	 
	uplink_rd_fd = open("/dev/xillybus_read_128", O_RDONLY);
	if (uplink_rd_fd < 0) 
	{ 
		if (errno == ENODEV)
		  fprintf(stderr, "(Maybe xillybus_read a read-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	} 
#endif

/*
 
	debug_file_fd = open("/home/isip/ramdisk/rx/OAI_rx.txt", O_RDWR | O_CREAT, 0644);
	if (debug_file_fd < 0) {  
		if (errno == ENODEV)
		  fprintf(stderr, "(Maybe OAI_rx.txt a read-only file?)\n");
	  
		perror("Failed to open target file");
		exit(1);
	}
	
	debug_file_fd1 = open("/home/isip/ramdisk/rx/OAI_rx1.txt", O_RDWR | O_CREAT, 0644);
	if (debug_file_fd1 < 0) { 
		if (errno == ENODEV)
		  fprintf(stderr, "(Maybe OAI_rx.txt a read-only file?)\n");
	  
		perror("Failed to open target file");
		exit(1);
	}
	
	
	debug_sent_fd = open("/home/isip/ramdisk/tx/fiber_test.txt", O_RDONLY);
	if (debug_sent_fd < 0) { 
		if (errno == ENODEV)
			fprintf(stderr, "(Maybe fiber_test a write-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}
	
	
	debug_sent_fd1 = open("/home/isip/ramdisk/tx/fiber_test1.txt", O_RDONLY);
	if (debug_sent_fd1 < 0) { 
		if (errno == ENODEV)
			fprintf(stderr, "(Maybe fiber_test a write-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}
	
	debug_sent_fd2 = open("/home/isip/ramdisk/tx/fiber_test2.txt", O_RDONLY);
	if (debug_sent_fd2 < 0) { 
		if (errno == ENODEV)
			fprintf(stderr, "(Maybe fiber_test a write-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}
	
	debug_sent_fd3 = open("/home/isip/ramdisk/tx/fiber_test3.txt", O_RDONLY);
	if (debug_sent_fd3 < 0) { 
		if (errno == ENODEV)
			fprintf(stderr, "(Maybe fiber_test a write-only file?)\n");
	
		perror("Failed to open devfile");
		exit(1);
	}
	
	isip_sent_buff=malloc(1073741824); 
	isip_sent_int_buff = (int *)isip_sent_buff;		
	read(debug_sent_fd, isip_sent_buff, 30720*16);  
*/
	
	isip_sent_buff1=malloc(1073741824); 
	isip_sent_int_buff1 = (int *)isip_sent_buff1; 
	//read(debug_sent_fd1, isip_sent_buff1, 30720); 
	 
	isip_sent_buff2=malloc(1073741824); 
	isip_sent_int_buff2 = (int *)isip_sent_buff2; 
	//read(debug_sent_fd1, isip_sent_buff2, 30720); 
	
/*	
	isip_sent_buff3=malloc(1073741824);  
	isip_sent_int_buff3 = (int *)isip_sent_buff3; 
	read(debug_sent_fd3, isip_sent_buff3, 30720*4); 
	
	short *short_buff3 = (short *)isip_sent_buff3; 
*/	
	/*
	for(int i = 0 ; i < 7680*4 ; i++){
		isip_sent_int_buff3[i] = i;
		printf("\033[35m[CUDA] i = %d ,s_re = %d\033[0m",i,short_buff3[i*2]);
		printf("\033[35m ,s_im = %d \033[0m\n", short_buff3[i*2+1]);
	}
	*/
	int i = 0;

	switch ((int)openair0_cfg[0].sample_rate) {
		case 30720000:
			for(i=0;i<number_of_rrh;i++)
				isip_sent_int_buff1[i] = 3428174848;
			break;
		case 7680000:
			for(i=0;i<number_of_rrh;i++)
				isip_sent_int_buff1[i] = 3428174849;
			break;
		default:
			printf("Error: ISIP_HW not support this sampling rate %f\n",openair0_cfg[0].sample_rate);
			exit(-1);
			break;
	}
	
	device->priv 		   = NULL;
	device->trx_start_func = NULL;
	device->trx_write_func = NULL;
	device->trx_read_func  = NULL;
	device->trx_get_stats_func 	 = NULL;
	device->trx_reset_stats_func = NULL;
	device->trx_end_func   = NULL; 
	device->trx_stop_func  = NULL;
	device->trx_set_freq_func = NULL;
	device->trx_set_gains_func   = NULL; 
	device->openair0_cfg = openair0_cfg;	
	
	return 0;
}
