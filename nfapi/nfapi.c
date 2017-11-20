
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <pthread.h>
#include <unistd.h>

#if 0
#if 0
//DJP
#include <mutex>
#include <queue>
#include <list>
#endif

//DJP struct phy_pdu
typedef struct
{
#if 0
	phy_pdu() : buffer_len(1500), buffer(0), len(0)
	{
		buffer = (char*) malloc(buffer_len);
	}
	
	virtual ~phy_pdu()
	{
		free(buffer);
	}
#endif

	unsigned buffer_len;
	char* buffer;
	unsigned len;
} phy_pdu;

// DJP class fapi_private 
typedef struct
{
		//std::mutex mutex;
		//std::queue<phy_pdu*> rx_buffer;

		//std::queue<phy_pdu*> free_store;
#if 0
	public:

		fapi_private()
			: byte_count(0), tick(0), first_dl_config(false)
		{
		}

		phy_pdu* allocate_phy_pdu()
		{
			phy_pdu* pdu = 0;
			mutex.lock();
			if(free_store.empty())
			{
				pdu = new phy_pdu();
			}
			else
			{
				pdu = free_store.front();
				free_store.pop();
			}
			mutex.unlock();
			return pdu;
		}

		void release_phy_pdu(phy_pdu* pdu)
		{
			mutex.lock();
			free_store.push(pdu);
			mutex.unlock();
		}

		bool rx_buffer_empty()
		{
			bool empty;
			mutex.lock();
			empty = rx_buffer.empty();
			mutex.unlock();

			return empty;
		}
		

		void push_rx_buffer(phy_pdu* buff)
		{
			mutex.lock();
			rx_buffer.push(buff);
			mutex.unlock();
		}

		phy_pdu* pop_rx_buffer()
		{
			phy_pdu* buff = 0;
			mutex.lock();
			if(!rx_buffer.empty())
			{
				buff = rx_buffer.front();
				rx_buffer.pop();
			}
			mutex.unlock();
			return buff;
		}
#endif

		uint32_t byte_count;
		uint32_t tick;
		uint8_t first_dl_config;

} fapi_private ;

#if defined(__cplusplus)
extern "C"
{
#endif
	typedef struct fapi_internal
	{
		fapi_t _public;

		fapi_cb_t callbacks;

		uint8_t state;
		fapi_config_t config;

		int rx_sock;
		int tx_sock;
		struct sockaddr_in tx_addr;

		uint32_t tx_byte_count;
		uint32_t tick;
		
		fapi_private* fapi;

	} fapi_internal_t;
#if defined(__cplusplus)
}
#endif
#endif

void set_thread_priority(int priority)
{
	//printf("%s(priority:%d)\n", __FUNCTION__, priority);
	
	pthread_attr_t ptAttr;
	
	struct sched_param schedParam;
	schedParam.__sched_priority = priority; //79;
	if(sched_setscheduler(0, SCHED_RR, &schedParam) != 0)
	{
		printf("Failed to set scheduler to SCHED_RR\n");
	}

	if(pthread_attr_setschedpolicy(&ptAttr, SCHED_RR) != 0)
	{
		printf("Failed to set pthread sched policy SCHED_RR\n");
	}

	pthread_attr_setinheritsched(&ptAttr, PTHREAD_EXPLICIT_SCHED);

	struct sched_param thread_params;
	thread_params.sched_priority = 20;
	if(pthread_attr_setschedparam(&ptAttr, &thread_params) != 0)
	{
		printf("failed to set sched param\n");
	}
}
