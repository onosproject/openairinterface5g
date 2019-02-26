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

#define _GNU_SOURCE
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include "vnf_p7.h"

#define FAPI2_IP_DSCP	0

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
	

	return &(_this->_public);
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

	//struct timespec sf_end;

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

		selectRetval = pselect(maxSock+1, &rfds, NULL, NULL, &pselect_timeout, NULL);
		nfapi_vnf_p7_connection_info_t* phy = vnf_p7->p7_connections;

		if (selectRetval==-1 && errno == 22)
		{
		    NFAPI_TRACE(NFAPI_TRACE_ERROR, "INVAL: pselect_timeout:%d.%ld adj[dur:%d adj:%d], sf_dur:%d.%ld\n",
                    pselect_timeout.tv_sec, pselect_timeout.tv_nsec, 
                    phy->insync_minor_adjustment_duration, phy->insync_minor_adjustment, 
                    sf_duration.tv_sec, sf_duration.tv_nsec);
		}

   if(selectRetval > 0)
   {
     // have a p7 message
     if(FD_ISSET(vnf_p7->socket, &rfds))
     {
				vnf_p7_read_dispatch_message(vnf_p7);
     }
   }
   else if(selectRetval < 0)
   {
     // pselect error
     if(selectRetval == -1 && errno == EINTR)
     {
       // a sigal was received.
     }
     else
     {
       NFAPI_TRACE(NFAPI_TRACE_INFO, "P7 select failed result %d errno %d timeout:%d.%d orginal:%d.%d last_ms:%ld ms:%ld\n", selectRetval, errno, pselect_timeout.tv_sec, pselect_timeout.tv_nsec, pselect_timeout.tv_sec, pselect_timeout.tv_nsec, last_millisecond, millisecond);
       // should we exit now?
       if (selectRetval == -1 && errno == 22) // invalid argument??? not sure about timeout duration
       {
         usleep(100000);
       }
     }
   }

  }

  NFAPI_TRACE(NFAPI_TRACE_INFO, "Closing p7 socket\n");
  close(vnf_p7->socket);

  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() returning\n", __FUNCTION__);

  return 0;

}

int nfapi_vnf_p7_time(nfapi_vnf_p7_config_t* config){
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
#ifdef DEADLINE_SCHEDULER
  struct sched_attr attr;

  unsigned int flags = 0;

  attr.size = sizeof(attr);
  attr.sched_flags = 0;
  attr.sched_nice = 0;
  attr.sched_priority = 0;

  attr.sched_policy   = SCHED_DEADLINE;
  attr.sched_runtime  = 870000L; 
  attr.sched_deadline = 1000000L;
  attr.sched_period   = 1000000L; 

  if (sched_setattr(0, &attr, flags) < 0 ) {
    fprintf(stderr,"sched_setattr Error = %s",strerror(errno));
    exit(1);
  }
#else
  int policy, s, j;
  struct sched_param sparam;
  char cpu_affinity[1024];
  cpu_set_t cpuset;

  /* Set affinity mask to include CPUs 1 to MAX_CPUS */
  /* CPU 0 is reserved for UHD threads */
  /* CPU 1 is reserved for all RX_TX threads */
  /* Enable CPU Affinity only if number of CPUs >2 */
  CPU_ZERO(&cpuset);
#ifdef CPU_AFFINITY
  if (get_nprocs() >= 8)
  {
    CPU_SET(4, &cpuset);
  } else if (get_nprocs() > 2) {
    for (j = 1; j < get_nprocs(); j++) {
      CPU_SET(j, &cpuset);
    }
  }
  s = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0)
  {
    printf("Error setting processor affinity");
    exit(1);
  }
#endif //CPU_AFFINITY
  /* Check the actual affinity mask assigned to the thread */
  s  = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (s != 0) {
    printf("Error getting processor affinity ");
    exit(1);
  }
  memset(cpu_affinity,0,sizeof(cpu_affinity));
  for (j = 0; j < 1024; j++)
    if (CPU_ISSET(j, &cpuset)) {  
      char temp[1024];
      sprintf (temp, " CPU_%d", j);
      strcat(cpu_affinity, temp);
    }

  memset(&sparam, 0, sizeof(sparam));
  sparam.sched_priority = sched_get_priority_max(SCHED_FIFO);
  policy = SCHED_FIFO ; 
  
  s = pthread_setschedparam(pthread_self(), policy, &sparam);
  if (s != 0) {
    printf("Error setting thread priority");
    exit(1);
  }
  
  s = pthread_getschedparam(pthread_self(), &policy, &sparam);
  if (s != 0) {
    printf("Error getting thread priority");
    exit(1);
  }

  pthread_setname_np(pthread_self(), "nfapi_vnf_p7_time");

  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s()[SCHED][eNB] %s started on CPU %d, sched_policy = %s , priority = %d, CPU Affinity=%s \n", __FUNCTION__, "nfapi_vnf_p7_time", sched_getcpu(),
                   (policy == SCHED_FIFO)  ? "SCHED_FIFO" : (policy == SCHED_RR)    ? "SCHED_RR" :
                   (policy == SCHED_OTHER) ? "SCHED_OTHER" : "???", sparam.sched_priority, cpu_affinity );

#endif //LOW_LATENCY
  clock_gettime(CLOCK_MONOTONIC, &sf_start);
  //long millisecond = sf_start.tv_nsec / 1e6;
  sf_start = timespec_add(sf_start, sf_duration);
  NFAPI_TRACE(NFAPI_TRACE_INFO, "next subframe will start at %d.%d\n", sf_start.tv_sec, sf_start.tv_nsec);
  while(vnf_p7->terminate == 0)
  {
    clock_gettime(CLOCK_MONOTONIC, &pselect_start);
    if((pselect_start.tv_sec > sf_start.tv_sec) ||
      ((pselect_start.tv_sec == sf_start.tv_sec) && (pselect_start.tv_nsec > sf_start.tv_nsec)))
    {
      pselect_timeout.tv_sec = 0;
      pselect_timeout.tv_nsec = 0;
    }
    else
    {
      // still time before the end of the subframe wait
      pselect_timeout = timespec_sub(sf_start, pselect_start);
      nanosleep(&pselect_timeout,&rem_time);
    }
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
      }
      else if (sf_start.tv_nsec < 0)
      {
        sf_start.tv_sec--;
        sf_start.tv_nsec+=1e9;
      }

      //phy->insync_minor_adjustment = 0;
      phy->insync_minor_adjustment_duration--;

      if (phy->insync_minor_adjustment_duration==0)
      {
        phy->insync_minor_adjustment = 0;
      }
    }
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
  }
  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() returning\n", __FUNCTION__);
  return 0;
}

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
	node->dl_in_sync_period = 512;
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

int nfapi_vnf_p7_release_rnti_req(nfapi_vnf_p7_config_t* config, nfapi_release_rnti_request_t* req)
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
