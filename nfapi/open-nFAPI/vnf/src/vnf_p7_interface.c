/*
 * Copyright 2017 Cisco Systems, Inc.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include <stdio.h>

#include "vnf_p7.h"

#define FAPI2_IP_DSCP	0
#ifdef PHY_RM
#define SF_BAND_CORRECT_200US 200000
#define SF_BAND_CORRECT_500US 500000

#define VNF_MSG_SIZE			10000000
#define VNF_MSG_BUF_SIZE		10000000
#define VNF_RX_MESSAGE_BUF_SIZE		65535
#define VNF_REASSEMBY_BUF_SIZE	1000000

uint8_t* vnf_msg_ptr;
uint8_t* vnf_msg_buf_ptr;

uint8_t vnf_reassembly_flg = 0;

//#define LOG
#endif
nfapi_vnf_p7_config_t* nfapi_vnf_p7_config_create()
{
	vnf_p7_t* _this = (vnf_p7_t*)calloc(1, sizeof(vnf_p7_t));

	if(_this == 0)
		return 0;

	// todo : initialize
	_this->_public.segment_size = 1400;
	_this->_public.max_num_segments = 8;
	_this->_public.checksum_enabled = 1;
	
	_this->_public.malloc = &malloc;
	_this->_public.free = &free;	

	_this->_public.codec_config.allocate = &malloc;
	_this->_public.codec_config.deallocate = &free;
	

	return (nfapi_vnf_p7_config_t*)_this;
}

void nfapi_vnf_p7_config_destory(nfapi_vnf_p7_config_t* config)
{
	free(config);
}


struct timespec timespec_add(struct timespec lhs, struct timespec rhs)
{
	struct timespec result;

	result.tv_sec = lhs.tv_sec + rhs.tv_sec;
	result.tv_nsec = lhs.tv_nsec + rhs.tv_nsec;

	if(result.tv_nsec > 1e9)
	{
		result.tv_sec++;
		result.tv_nsec-= 1e9;
	}

	return result;
}

struct timespec timespec_sub(struct timespec lhs, struct timespec rhs)
{
	struct timespec result;
	if ((lhs.tv_nsec-rhs.tv_nsec)<0) 
	{
		result.tv_sec = lhs.tv_sec-rhs.tv_sec-1;
		result.tv_nsec = 1000000000+lhs.tv_nsec-rhs.tv_nsec;
	} 
	else 
	{
		result.tv_sec = lhs.tv_sec-rhs.tv_sec;
		result.tv_nsec = lhs.tv_nsec-rhs.tv_nsec;
	}
	return result;
}

// monitor the p7 endpoints and the timing loop and 
// send indications to mac
int nfapi_vnf_p7_start(nfapi_vnf_p7_config_t* config)
{
	if(config == 0)
		return -1;

	NFAPI_TRACE(NFAPI_TRACE_INFO, "%s()\n", __FUNCTION__);

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;

	// Create p7 receive udp port 
	// todo : this needs updating for Ipv6
	
	NFAPI_TRACE(NFAPI_TRACE_INFO, "Initialising VNF P7 port:%u\n", config->port);

	// open the UDP socket
	if ((vnf_p7->socket = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
	{
		NFAPI_TRACE(NFAPI_TRACE_ERROR, "After P7 socket errno: %d\n", errno);
		return -1;
	}

	NFAPI_TRACE(NFAPI_TRACE_INFO, "VNF P7 socket created...\n");

	// configure the UDP socket options
	int iptos_value = FAPI2_IP_DSCP << 2;
	if (setsockopt(vnf_p7->socket, IPPROTO_IP, IP_TOS, &iptos_value, sizeof(iptos_value)) < 0)
	{
		NFAPI_TRACE(NFAPI_TRACE_ERROR, "After setsockopt (IP_TOS) errno: %d\n", errno);
		return -1;
	}
	
	NFAPI_TRACE(NFAPI_TRACE_INFO, "VNF P7 setsockopt succeeded...\n");

	// Create the address structure
	struct sockaddr_in addr;
	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(config->port);
	addr.sin_addr.s_addr = INADDR_ANY;
	// bind to the configured port
	NFAPI_TRACE(NFAPI_TRACE_INFO, "VNF P7 binding too %s:%d\n", inet_ntoa(addr.sin_addr), ntohs(addr.sin_port));
	if (bind(vnf_p7->socket, (struct sockaddr *)&addr, sizeof(struct sockaddr_in)) < 0)
	//if (sctp_bindx(config->socket, (struct sockaddr *)&addr, sizeof(struct sockaddr_in), 0) < 0)
	{
		NFAPI_TRACE(NFAPI_TRACE_ERROR, "After bind errno: %d\n", errno);
		return -1;
	}

	NFAPI_TRACE(NFAPI_TRACE_INFO, "VNF P7 bind succeeded...\n");


	//struct timespec original_pselect_timeout;
	struct timespec pselect_timeout;
	pselect_timeout.tv_sec = 0;
	pselect_timeout.tv_nsec = 1000000; // ns in a 1 us
#ifdef PHY_RM
	vnf_msg_ptr     = (uint8_t *)malloc(VNF_MSG_SIZE);
	vnf_msg_buf_ptr = (uint8_t *)malloc(VNF_MSG_BUF_SIZE);

	vnf_p7->rx_message_buffer = (uint8_t*)malloc(VNF_RX_MESSAGE_BUF_SIZE);
	vnf_p7->rx_message_buffer_size = VNF_RX_MESSAGE_BUF_SIZE;
#else
	struct timespec pselect_start;
	struct timespec pselect_stop;
	//struct timespec sf_end;
	long last_millisecond = -1;


	struct timespec sf_duration;
	sf_duration.tv_sec = 0;
	sf_duration.tv_nsec = 1e6; // We want 1ms pause

	struct timespec sf_start;
	clock_gettime(CLOCK_MONOTONIC_RAW, &sf_start);
	long millisecond = sf_start.tv_nsec / 1e6;
	sf_start = timespec_add(sf_start, sf_duration);
	NFAPI_TRACE(NFAPI_TRACE_INFO, "next subframe will start at %d.%d\n", sf_start.tv_sec, sf_start.tv_nsec);
#endif
	while(vnf_p7->terminate == 0)
	{
		fd_set rfds;
		int maxSock = 0;
		FD_ZERO(&rfds);
		int selectRetval = 0;

		// Add the p7 socket
		FD_SET(vnf_p7->socket, &rfds);
		maxSock = vnf_p7->socket;
		
		//NFAPI_TRACE(NFAPI_TRACE_INFO, "pselect_start:%d.%d sf_start:%d.%d\n", pselect_start.tv_sec, pselect_start.tv_nsec, sf_start.tv_sec, sf_start.tv_nsec);
#ifdef PHY_RM
		selectRetval = pselect(maxSock+1, &rfds, NULL, NULL, &pselect_timeout, NULL);
		nfapi_vnf_p7_connection_info_t* phy = vnf_p7->p7_connections;

		if (selectRetval==-1 && errno == 22)
		{
		    NFAPI_TRACE(NFAPI_TRACE_ERROR, "INVAL: pselect_timeout:%d.%ld adj[dur:%d adj:%d]\n",
                    pselect_timeout.tv_sec, pselect_timeout.tv_nsec,
                    phy->insync_minor_adjustment_duration, phy->insync_minor_adjustment);
		}

		if(selectRetval > 0)
		{
			if((vnf_reassembly_flg == 0) && (phy != 0))
			{
				phy->reassembly_buffer = (uint8_t*)malloc(VNF_REASSEMBY_BUF_SIZE);
				phy->reassembly_buffer_size = VNF_REASSEMBY_BUF_SIZE;

				vnf_reassembly_flg = 1;
			}

			// have a p7 message
			if(FD_ISSET(vnf_p7->socket, &rfds))
			{
				vnf_p7_read_dispatch_message(vnf_p7);
			}
		} else if(selectRetval < 0)
		{
			// pselect error
			if(selectRetval == -1 && errno == EINTR)
			{
				// a sigal was received.
			}
			else
			{
				NFAPI_TRACE(NFAPI_TRACE_INFO, "P7 select failed result %d errno %d timeout:%d.%d orginal:%d.%d\n", selectRetval, errno, pselect_timeout.tv_sec, pselect_timeout.tv_nsec, pselect_timeout.tv_sec, pselect_timeout.tv_nsec);
				// should we exit now?
				if (selectRetval == -1 && errno == 22) // invalid argument??? not sure about timeout duration
				{
					usleep(100000);
				}
			}
		}
#else
	clock_gettime(CLOCK_MONOTONIC_RAW, &pselect_start);
	//long millisecond = pselect_start.tv_nsec / 1e6;

	if((last_millisecond == -1) || (millisecond == last_millisecond) || (millisecond == (last_millisecond + 1) % 1000) )
	{
		if((pselect_start.tv_sec > sf_start.tv_sec) || ((pselect_start.tv_sec == sf_start.tv_sec) && (pselect_start.tv_nsec > sf_start.tv_nsec)))
		{
			// overran the end of the subframe we do not want to wait
			pselect_timeout.tv_sec = 0;
			pselect_timeout.tv_nsec = 0;

		} else {
			// still time before the end of the subframe wait
			pselect_timeout = timespec_sub(sf_start, pselect_start);
		}
		selectRetval = pselect(maxSock+1, &rfds, NULL, NULL, &pselect_timeout, NULL);

		clock_gettime(CLOCK_MONOTONIC_RAW, &pselect_stop);

		nfapi_vnf_p7_connection_info_t* phy = vnf_p7->p7_connections;

		if (selectRetval==-1 && errno == 22)
		{
			NFAPI_TRACE(NFAPI_TRACE_ERROR, "INVAL: pselect_timeout:%d.%ld adj[dur:%d adj:%d], sf_dur:%d.%ld\n",
					pselect_timeout.tv_sec, pselect_timeout.tv_nsec,
					phy->insync_minor_adjustment_duration, phy->insync_minor_adjustment,
					sf_duration.tv_sec, sf_duration.tv_nsec);
		}
		if(selectRetval == 0)
		{
			// calculate the start of the next subframe
			sf_start = timespec_add(sf_start, sf_duration);
			//NFAPI_TRACE(NFAPI_TRACE_INFO, "next subframe will start at %d.%d\n", sf_start.tv_sec, sf_start.tv_nsec);
			if(phy && phy->in_sync && phy->insync_minor_adjustment != 0 && phy->insync_minor_adjustment_duration > 0)
			{
				long insync_minor_adjustment_ns = (phy->insync_minor_adjustment * 1000);
				sf_start.tv_nsec -= insync_minor_adjustment_ns;
				if (sf_start.tv_nsec > 1e9)
				{
					sf_start.tv_sec++;
					sf_start.tv_nsec-=1e9;
				} else if (sf_start.tv_nsec < 0) {
					sf_start.tv_sec--;
					sf_start.tv_nsec+=1e9;
				}

				phy->insync_minor_adjustment_duration--;

				NFAPI_TRACE(NFAPI_TRACE_NOTE, "[VNF] AFTER adjustment - Subframe minor adjustment %dus sf_start.tv_nsec:%d duration:%u\n",
						phy->insync_minor_adjustment, sf_start.tv_nsec, phy->insync_minor_adjustment_duration);
				if (phy->insync_minor_adjustment_duration==0)
				{
					phy->insync_minor_adjustment = 0;
				}
			}
			millisecond ++;
		}
	} else {
		// we have overrun the subframe advance to go and collect $200
		if((millisecond - last_millisecond) > 3)
			NFAPI_TRACE(NFAPI_TRACE_WARN, "subframe overrun %ld %ld (%ld)\n", millisecond, last_millisecond, millisecond - last_millisecond + 1);

		last_millisecond = ( last_millisecond + 1 ) % 1000;
		selectRetval = 0;
	}

	if(selectRetval == 0)
	{
		vnf_p7->sf_start_time_hr = vnf_get_current_time_hr();

		// pselect timed out
		nfapi_vnf_p7_connection_info_t* curr = vnf_p7->p7_connections;

		while(curr != 0)
		{
			curr->sfn_sf = increment_sfn_sf(curr->sfn_sf);
			vnf_sync(vnf_p7, curr);
			curr = curr->next;
		}

		send_mac_subframe_indications(vnf_p7);

	} else if(selectRetval > 0) {
		// have a p7 message
		if(FD_ISSET(vnf_p7->socket, &rfds))
		{
			vnf_p7_read_dispatch_message(vnf_p7);
		}
	} else {
		// pselect error
		if(selectRetval == -1 && errno == EINTR)
		{
			// a sigal was received.
		} else {
			NFAPI_TRACE(NFAPI_TRACE_INFO, "P7 select failed result %d errno %d timeout:%d.%d orginal:%d.%d last_ms:%ld ms:%ld\n", selectRetval, errno, pselect_timeout.tv_sec, pselect_timeout.tv_nsec, pselect_timeout.tv_sec, pselect_timeout.tv_nsec, last_millisecond, millisecond);
			// should we exit now?
			if (selectRetval == -1 && errno == 22) // invalid argument??? not sure about timeout duration
			{
				usleep(100000);
			}
		}
	}
#endif

  }

  NFAPI_TRACE(NFAPI_TRACE_INFO, "Closing p7 socket\n");
  close(vnf_p7->socket);

  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() returning\n", __FUNCTION__);

  return 0;

}
#ifdef PHY_RM
int nfapi_vnf_p7_time(nfapi_vnf_p7_config_t* config){

#ifdef LOG
	#define LOGFILE_PATH "/dev/shm/vnf_sf_start_time.log"
	FILE *fp;
#endif

  vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
  //struct timespec original_pselect_timeout;
  nfapi_vnf_p7_connection_info_t* phy = vnf_p7->p7_connections;
  //int8_t ret = 0;
  struct timespec pselect_timeout;
  pselect_timeout.tv_sec = 0;
  pselect_timeout.tv_nsec = 1000000; // ns in a 1 us

  struct timespec sf_duration;
  sf_duration.tv_sec = 0;
  sf_duration.tv_nsec = 1e6; // We want 1ms pause
  struct timespec rem_time;
  struct timespec pselect_start;
  struct timespec sf_start;
  uint32_t usleep_time;
  volatile struct timespec p1,p2,p3;
  uint32_t correct_flg;
  fd_set catch_fd_set;
  uint32_t rcd;
  uint32_t data;
  uint32_t size;

  uint8_t buf_1ms[32];
  struct sockaddr_in addr;
  int sock_1ms;
  fd_set fds, readfds;
  int retval;
  int ret;
  static int sync = 0;

  sock_1ms = socket(AF_INET,SOCK_DGRAM,0);
  addr.sin_family = AF_INET;
  addr.sin_port = htons(50040);
  addr.sin_addr.s_addr = INADDR_ANY;
  bind(sock_1ms, (struct sockaddr *) &addr,sizeof(addr));

  FD_ZERO(&readfds);
  FD_SET(vnf_p7->fapi_1ms_fd_list[0], &readfds);

  clock_gettime(CLOCK_MONOTONIC_RAW, &sf_start);
  //long millisecond = sf_start.tv_nsec / 1e6;
  sf_start = timespec_add(sf_start, sf_duration);
  NFAPI_TRACE(NFAPI_TRACE_INFO, "next subframe will start at %d.%d\n", sf_start.tv_sec, sf_start.tv_nsec);
  while(vnf_p7->terminate == 0)
  {
	correct_flg = 0;

	if(vnf_p7->sf_duration_correct_flg == 1)
	{
#ifdef LOG
		printf("nfapi_vnf_p7_time1 sf_corr_tim.tv_nsec = %d\n", vnf_p7->sf_corr_tim.tv_nsec);
#endif

		if(vnf_p7->vnf_extend_flg == 1)
		{
			sf_duration = timespec_add(sf_duration, vnf_p7->sf_corr_tim);
		}
		else
		{
			sf_duration = timespec_sub(sf_duration, vnf_p7->sf_corr_tim);
		}

		vnf_p7->sf_duration_correct_flg = 0;
		correct_flg = 1;
#ifdef LOG
		printf("nfapi_vnf_p7_time2 vnf_extend_flg = %d, sf_duration.tv_nsec = %d \n", vnf_p7->vnf_extend_flg, sf_duration.tv_nsec);
#endif
	}

	if((sf_duration.tv_nsec < 0) || (sf_duration.tv_nsec > 2000000))
	{
		sf_duration.tv_nsec = 1000000;
	}

	usleep_time = sf_duration.tv_nsec / 1000;
//	usleep(usleep_time);

	clock_gettime(CLOCK_MONOTONIC_RAW, &p1);
	p3 = timespec_sub(p1,p2);
	p2=p1;

	sf_duration.tv_sec = 0;
	if ((p3.tv_nsec > 900000) && (p3.tv_nsec<1100000))
	{
		if(correct_flg == 0)
		{
			sf_duration.tv_nsec -= p3.tv_nsec - 1000000;
		}
	}
	else
	{
		sf_duration.tv_nsec = 1000000;
	}
  
  nfapi_vnf_p7_connection_info_t* curr = vnf_p7->p7_connections;

#ifndef UDP_1MS
  memcpy(&fds, &readfds, sizeof(fd_set));  
  if(!sync) {
    retval = select(vnf_p7->fapi_1ms_fd_list[0]+1, &fds, NULL, NULL, NULL);
  }
  else {
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 1200;
    retval = select(vnf_p7->fapi_1ms_fd_list[0]+1, &fds, NULL, NULL, &tv);
  }
  
  if(retval == 1) {
    while(1) {
      ret = recv(vnf_p7->fapi_1ms_fd_list[0], &data, sizeof(data), 0);
      if(ret < 1) {
        break;
      }
    }
  }
  else {
     NFAPI_TRACE(NFAPI_TRACE_ERROR, "recv timeout sfnsf = 0x%x\n", curr->sfn_sf);
  }
  sync = 1;
    
#else
	recv(sock_1ms,buf_1ms,sizeof(buf_1ms),0);
#endif

    vnf_p7->sf_start_time_hr_old = vnf_p7->sf_start_time_hr;
    vnf_p7->sf_start_time_hr = vnf_get_current_time_hr();
    // pselect timed out

#ifdef LOG
 	fp = fopen(LOGFILE_PATH, "a");
#endif

    while(curr != 0)
    {
      curr->sfn_sf = increment_sfn_sf(curr->sfn_sf);


#ifdef LOG
		printf("nfapi_vnf_p7_time = 0x%x\n", curr->sfn_sf);
		printf("1ms_boundary_time = %u, sfnsf = 0x%x\n",vnf_p7->sf_start_time_hr, curr->sfn_sf);

		fprintf(fp,"1ms_boundary_time = %u,%x, sfnsf = 0x%x, usleep_time = %u, sf_length = %ld \n",TIMEHR_SEC(vnf_p7->sf_start_time_hr)*1000000+TIMEHR_USEC(vnf_p7->sf_start_time_hr), vnf_p7->sf_start_time_hr, curr->sfn_sf, usleep_time, p3.tv_nsec/1000);
#endif

//		if((TIMEHR_USEC(vnf_p7->sf_start_time_hr) - TIMEHR_USEC(vnf_p7->sf_start_time_hr_old)) > 2000)
//		{
//			printf("sf_start_time_hr_usec = %u, sf_start_time_hr_usec_old = %u, now - old = %u\n", TIMEHR_USEC(vnf_p7->sf_start_time_hr), TIMEHR_USEC(vnf_p7->sf_start_time_hr_old), TIMEHR_USEC(vnf_p7->sf_start_time_hr) - TIMEHR_USEC(vnf_p7->sf_start_time_hr_old));
//		}


      vnf_sync(vnf_p7, curr);
      curr = curr->next;
    }

#ifdef LOG
	fclose(fp);
#endif

    send_mac_subframe_indications(vnf_p7);
  }
  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() returning\n", __FUNCTION__);
  return 0;
}
#endif
int nfapi_vnf_p7_stop(nfapi_vnf_p7_config_t* config)
{
	if(config == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	vnf_p7->terminate =1;
	return 0;
}

int nfapi_vnf_p7_add_pnf(nfapi_vnf_p7_config_t* config, const char* pnf_p7_addr, int pnf_p7_port, int phy_id)
{
	NFAPI_TRACE(NFAPI_TRACE_INFO, "%s(config:%p phy_id:%d pnf_addr:%s pnf_p7_port:%d)\n", __FUNCTION__, config, phy_id,  pnf_p7_addr, pnf_p7_port);

	if(config == 0)
        {
          return -1;
        }

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;

	nfapi_vnf_p7_connection_info_t* node = (nfapi_vnf_p7_connection_info_t*)malloc(sizeof(nfapi_vnf_p7_connection_info_t));

	memset(node, 0, sizeof(nfapi_vnf_p7_connection_info_t));
	node->phy_id = phy_id;
	node->in_sync = 0;
	node->dl_out_sync_offset = 30;
	node->dl_out_sync_period = 10;
	node->dl_in_sync_offset = 30;
#ifdef PHY_RM
        node->dl_in_sync_period = 32;
#else
	node->dl_in_sync_period = 512;
#endif
	node->sfn_sf = 0;

	node->min_sync_cycle_count = 8;

	// save the remote endpoint information
	node->remote_addr.sin_family = AF_INET;
	node->remote_addr.sin_port = htons(pnf_p7_port);
	node->remote_addr.sin_addr.s_addr = inet_addr(pnf_p7_addr);

	vnf_p7_connection_info_list_add(vnf_p7, node);

	return 0;
}

int nfapi_vnf_p7_del_pnf(nfapi_vnf_p7_config_t* config, int phy_id)
{
	NFAPI_TRACE(NFAPI_TRACE_INFO, "%s(phy_id:%d)\n", __FUNCTION__, phy_id);

	if(config == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;

	nfapi_vnf_p7_connection_info_t* to_delete = vnf_p7_connection_info_list_delete(vnf_p7, phy_id);

	if(to_delete)
	{
		NFAPI_TRACE(NFAPI_TRACE_INFO, "%s(phy_id:%d) deleting connection info\n", __FUNCTION__, phy_id);
		free(to_delete);
	}

	return 0;
}
int nfapi_vnf_p7_dl_config_req(nfapi_vnf_p7_config_t* config, nfapi_dl_config_request_t* req)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s(config:%p req:%p)\n", __FUNCTION__, config, req);

	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}

int nfapi_vnf_p7_nr_dl_config_req(nfapi_vnf_p7_config_t* config, nfapi_nr_dl_tti_request_t* req)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s(config:%p req:%p)\n", __FUNCTION__, config, req);

	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}

int nfapi_vnf_p7_ul_config_req(nfapi_vnf_p7_config_t* config, nfapi_ul_config_request_t* req)
{
	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}
int nfapi_vnf_p7_hi_dci0_req(nfapi_vnf_p7_config_t* config, nfapi_hi_dci0_request_t* req)
{
	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}
int nfapi_vnf_p7_tx_req(nfapi_vnf_p7_config_t* config, nfapi_tx_request_t* req)
{
	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}
int nfapi_vnf_p7_lbt_dl_config_req(nfapi_vnf_p7_config_t* config, nfapi_lbt_dl_config_request_t* req)
{
	if(config == 0 || req == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}
int nfapi_vnf_p7_vendor_extension(nfapi_vnf_p7_config_t* config, nfapi_p7_message_header_t* header)
{
	if(config == 0 || header == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	return vnf_p7_pack_and_send_p7_msg(vnf_p7, header);
}

int nfapi_vnf_p7_ue_release_req(nfapi_vnf_p7_config_t* config, nfapi_ue_release_request_t* req)
{
    if(config == 0 || req == 0)
        return -1;

    vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
    return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}

int nfapi_vnf_p7_release_msg(nfapi_vnf_p7_config_t* config, nfapi_p7_message_header_t* header)
{
	if(config == 0 || header == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	vnf_p7_release_msg(vnf_p7, header);

	return 0;

}

int nfapi_vnf_p7_release_pdu(nfapi_vnf_p7_config_t* config, void* pdu)
{
	if(config == 0 || pdu == 0)
		return -1;

	vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
	vnf_p7_release_pdu(vnf_p7, pdu);

	return 0;
}
#ifdef PHY_RM
int nfapi_vnf_p7_phy_rm_start_req(nfapi_vnf_p7_config_t* config, nfapi_phy_rm_start_request_t* req)
{
    if(config == 0 || req == 0)
        return -1;

    vnf_p7_t* vnf_p7 = (vnf_p7_t*)config;
    return vnf_p7_pack_and_send_p7_msg(vnf_p7, &req->header);
}
#endif
