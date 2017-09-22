
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "nfapi_interface.h"
#include "nfapi_vnf_interface.h"
#include "nfapi.h"
#include "vendor_ext.h"

#include "nfapi_vnf.h"


#include "common/ran_context.h"
//#include "openair1/PHY/vars.h"
extern RAN_CONTEXT_t RC;

typedef struct 
{
	//public:
		uint8_t enabled;
		uint32_t rx_port;
		uint32_t tx_port;
		//std::string tx_addr;
		char tx_addr[80];
} udp_data;

//class phy_info
//DJP
typedef struct 
{
#if 0
	public:

		phy_info()
			: first_subframe_ind(0), fapi(0),
			  dl_ues_per_subframe(0), ul_ues_per_subframe(0), 
			  timing_window(0), timing_info_mode(0), timing_info_period(0)
		{
			index = 0;
			id = 0;

			local_port = 0;
			remote_addr = 0;
			remote_port = 0;
	
			duplex_mode = 0;
			dl_channel_bw_support = 0;
			ul_channel_bw_support = 0;
			num_dl_layers_supported = 0;
			num_ul_layers_supported = 0;
			release_supported = 0;
			nmm_modes_supported = 0;
		}
#endif

		uint16_t index;
		uint16_t id;
		//std::vector<uint8_t> rfs;
		//std::vector<uint8_t> excluded_rfs;
		uint8_t rfs[2];
		uint8_t excluded_rfs[2];

		udp_data udp;

		//std::string local_addr;
		char local_addr[80];
		int local_port;

		char* remote_addr;
		int remote_port;

		uint8_t duplex_mode;
		uint16_t dl_channel_bw_support;
		uint16_t ul_channel_bw_support;
		uint8_t num_dl_layers_supported;
		uint8_t num_ul_layers_supported;
		uint16_t release_supported;
		uint8_t nmm_modes_supported;

		uint8_t dl_ues_per_subframe;
		uint8_t ul_ues_per_subframe;

		uint8_t first_subframe_ind;

		// timing information recevied from the vnf
		uint8_t timing_window;
		uint8_t timing_info_mode;
		uint8_t timing_info_period;

    // DJP
		//fapi_t* fapi;

} phy_info;

//class rf_info
//DJP
typedef struct 
{
	//public:
		uint16_t index;
		uint16_t band;
		int16_t max_transmit_power;
		int16_t min_transmit_power;
		uint8_t num_antennas_supported;
		uint32_t min_downlink_frequency;
		uint32_t max_downlink_frequency;
		uint32_t max_uplink_frequency;
		uint32_t min_uplink_frequency;
} rf_info;


//class pnf_info
//DJP
typedef struct 
{
#if 0
	public:

		pnf_info() 
		: release(13), wireshark_test_mode(0),
		  max_total_power(0), oui(0)
					
		{
			release = 0;
	
			sync_mode = 0;
			location_mode = 0;
			dl_config_timing = 0;
			ul_config_timing = 0;
			tx_timing = 0;
			hi_dci0_timing = 0;
	
			max_phys = 0;
			max_total_bw = 0;
			max_total_dl_layers = 0;
			max_total_ul_layers = 0;
			shared_bands = 0;
			shared_pa = 0;
			
		}
#endif

		int release;
		//DJPstd::vector<phy_info> phys;
		//std::vector<rf_info> rfs;
		phy_info phys[2];
		rf_info rfs[2];

		uint8_t sync_mode;
		uint8_t location_mode;
		uint8_t location_coordinates[6];
		uint32_t dl_config_timing;
		uint32_t ul_config_timing;
		uint32_t tx_timing;
		uint32_t hi_dci0_timing;

		uint16_t max_phys;
		uint16_t max_total_bw;
		uint16_t max_total_dl_layers;
		uint16_t max_total_ul_layers;
		uint8_t shared_bands;
		uint8_t shared_pa;
		int16_t max_total_power;
		uint8_t oui;
		
		uint8_t wireshark_test_mode;

} pnf_info;

typedef struct mac mac_t;

typedef struct mac
{
	void* user_data;

	void (*dl_config_req)(mac_t* mac, nfapi_dl_config_request_t* req);
	void (*ul_config_req)(mac_t* mac, nfapi_ul_config_request_t* req);
	void (*hi_dci0_req)(mac_t* mac, nfapi_hi_dci0_request_t* req);
	void (*tx_req)(mac_t* mac, nfapi_tx_request_t* req);
} mac_t;

//class vnf_p7_info
typedef struct 
{
	//public:

#if 0
		vnf_p7_info()
			: thread_started(false), 
			  config(nfapi_vnf_p7_config_create(), 
				     [] (nfapi_vnf_p7_config_t* f) { nfapi_vnf_p7_config_destory(f); }),
			  mac(0)
		{
			local_port = 0;
			
			timing_window = 0;
			periodic_timing_enabled = 0;
			aperiodic_timing_enabled = 0;
			periodic_timing_period = 0;
			
			//config = nfapi_vnf_p7_config_create();
		}
		
		vnf_p7_info(const vnf_p7_info& other)  = default;
		
		vnf_p7_info(vnf_p7_info&& other) = default;
		
		vnf_p7_info& operator=(const vnf_p7_info&) = default;
		
		vnf_p7_info& operator=(vnf_p7_info&&) = default;
		
		
		
		virtual	~vnf_p7_info()
		{
			//NFAPI_TRACE(NFAPI_TRACE_INFO, "*** vnf_p7_info delete ***\n");
			
			//nfapi_vnf_p7_config_destory(config);
			
			// should we delete the mac?
		}
#endif
		

		int local_port;
		//DJP std::string local_addr;
		char local_addr[80];

		unsigned timing_window;
		unsigned periodic_timing_enabled;
		unsigned aperiodic_timing_enabled;
		unsigned periodic_timing_period;

		// This is not really the right place if we have multiple PHY, 
		// should be part of the phy struct
		udp_data udp;

		uint8_t thread_started;

		nfapi_vnf_p7_config_t* config;
		//std::shared_ptr<nfapi_vnf_p7_config_t> config;

		mac_t* mac;

} vnf_p7_info;

//class vnf_info
typedef struct
{
	//public:
	
		uint8_t wireshark_test_mode;

		//std::map<uint16_t, pnf_info> pnfs;
		pnf_info pnfs[2];

		//std::vector<vnf_p7_info> p7_vnfs;
		vnf_p7_info p7_vnfs[2];
} vnf_info;

int vnf_pack_vendor_extension_tlv(void* ve, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t* codec)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "vnf_pack_vendor_extension_tlv\n");
  nfapi_tl_t* tlv = (nfapi_tl_t*)ve;
  switch(tlv->tag)
  {
    case VENDOR_EXT_TLV_2_TAG:
      {
        //NFAPI_TRACE(NFAPI_TRACE_INFO, "Packing VENDOR_EXT_TLV_2\n");
        vendor_ext_tlv_2* ve = (vendor_ext_tlv_2*)tlv;
        if(!push32(ve->dummy, ppWritePackedMsg, end))
          return 0;
        return 1;
      }
      break;
  }
  return -1;
}
int vnf_unpack_vendor_extension_tlv(nfapi_tl_t* tl, uint8_t **ppReadPackedMessage, uint8_t *end, void** ve, nfapi_p4_p5_codec_config_t* codec)
{
  return -1;
}

void install_schedule_handlers(IF_Module_t *if_inst);
extern int single_thread_flag;
extern void init_eNB_afterRU(void);

void oai_create_enb(void)
{
  static int bodge_counter = 0;
  PHY_VARS_eNB *eNB;

  if (RC.eNB && RC.eNB[0] && RC.eNB[0][0])
  {
    eNB = RC.eNB[0][0];
    printf("[VNF] RC.eNB[0][0]. Mod_id:%d CC_id:%d\n", eNB->Mod_id, eNB->CC_id);
  }
  else
  {
    printf("[VNF] DJP ***** RC.eNB[] and RC.eNB[%d][%d] RC.nb_CC[%d]=1 MALLOCING structure and zeroing *******\n", bodge_counter, bodge_counter, bodge_counter);

    RC.eNB[bodge_counter] = (PHY_VARS_eNB **)malloc((1+MAX_NUM_CCs)*sizeof(PHY_VARS_eNB**));
    RC.eNB[bodge_counter][bodge_counter] = (PHY_VARS_eNB *)malloc(sizeof(PHY_VARS_eNB));
    memset((void*)RC.eNB[bodge_counter][bodge_counter],0,sizeof(PHY_VARS_eNB));

    eNB = RC.eNB[bodge_counter][bodge_counter];

    eNB->Mod_id  = bodge_counter;
    eNB->CC_id   = bodge_counter;
    eNB->if_inst = IF_Module_init(bodge_counter);
    eNB->abstraction_flag   = 0;
    eNB->single_thread_flag = 0;//single_thread_flag;
    eNB->td                   = ulsch_decoding_data;//(single_thread_flag==1) ? ulsch_decoding_data_2thread : ulsch_decoding_data;
    eNB->te                   = dlsch_encoding;//(single_thread_flag==1) ? dlsch_encoding_2threads : dlsch_encoding;

    RC.nb_CC[bodge_counter] = 1;
  }

  //init_eNB_proc(bodge_counter);

  // This will cause phy_config_request to be installed. That will result in RRC configuring the PHY
  // that will result in eNB->configured being set to TRUE.
  // See we need to wait for that to happen otherwise the NFAPI message exchanges won't contain the right parameter values
  if (RC.eNB[0][0]->if_inst->PHY_config_req==0 || RC.eNB[0][0]->if_inst->schedule_response==0)
  {
    printf("RC.eNB[0][0]->if_inst->PHY_config_req is not installed - install it\n");
    install_schedule_handlers(RC.eNB[0][0]->if_inst);
  }

  do {
    printf("%s() Waiting for eNB to become configured (by RRC/PHY) - need to wait otherwise NFAPI messages won't contain correct values\n", __FUNCTION__);
    usleep(50000);
  } while(eNB->configured != 1);
}


void oai_enb_init(void)
{
  init_eNB_afterRU();
}

int pnf_connection_indication_cb(nfapi_vnf_config_t* config, int p5_idx)
{
  printf("[VNF] pnf connection indication idx:%d\n", p5_idx);

  //pnf_info pnf;
  //vnf_info* vnf = (vnf_info*)(config->user_data);
  //vnf->pnfs.insert(std::pair<uint16_t, pnf_info>(p5_idx, pnf));

  oai_create_enb();

  nfapi_pnf_param_request_t req;
  memset(&req, 0, sizeof(req));
  req.header.message_id = NFAPI_PNF_PARAM_REQUEST;
  nfapi_vnf_pnf_param_req(config, p5_idx, &req);
  return 0;
}

int pnf_disconnection_indication_cb(nfapi_vnf_config_t* config, int p5_idx)
{
  printf("[VNF] pnf disconnection indication idx:%d\n", p5_idx);

  vnf_info* vnf = (vnf_info*)(config->user_data);
#if 0
  auto find_result = vnf->pnfs.find(p5_idx);

  if(find_result != vnf->pnfs.end())
  {
    pnf_info& pnf = find_result->second;

    for(phy_info& phy : pnf.phys)
    {
      vnf_p7_info& p7_vnf = vnf->p7_vnfs[0];
      nfapi_vnf_p7_del_pnf((p7_vnf.config.get()), phy.id);
    }
  }
#else
  pnf_info *pnf = vnf->pnfs;
  phy_info *phy = pnf->phys;

  vnf_p7_info* p7_vnf = vnf->p7_vnfs;
  // DJP nfapi_vnf_p7_del_pnf((p7_vnf->config.get()), phy->id);
  nfapi_vnf_p7_del_pnf((p7_vnf->config), phy->id);
#endif

  return 0;
}

int pnf_param_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_pnf_param_response_t* resp)
{
  printf("[VNF] pnf param response idx:%d error:%d\n", p5_idx, resp->error_code);

  vnf_info* vnf = (vnf_info*)(config->user_data);

#if 0
  auto find_result = vnf->pnfs.find(p5_idx);
  if(find_result != vnf->pnfs.end())
  {
    pnf_info& pnf = find_result->second;
#else
    {
      pnf_info *pnf = vnf->pnfs;
#endif

      for(int i = 0; i < resp->pnf_phy.number_of_phys; ++i)
      {
        phy_info phy;
        phy.index = resp->pnf_phy.phy[i].phy_config_index;

        printf("[VNF] (PHY:%d) phy_config_idx:%d\n", i, resp->pnf_phy.phy[i].phy_config_index);

        nfapi_vnf_allocate_phy(config, p5_idx, &(phy.id));

        for(int j = 0; j < resp->pnf_phy.phy[i].number_of_rfs; ++j)
        {
          printf("[VNF] (PHY:%d) (RF%d) %d\n", i, j, resp->pnf_phy.phy[i].rf_config[j].rf_config_index);
          phy.rfs[0] = resp->pnf_phy.phy[i].rf_config[j].rf_config_index;
        }

        pnf->phys[0] = phy;
      }

      for(int i = 0; i < resp->pnf_rf.number_of_rfs; ++i)
      {
        rf_info rf;
        rf.index = resp->pnf_rf.rf[i].rf_config_index;

        printf("[VNF] (RF:%d) rf_config_idx:%d\n", i, resp->pnf_rf.rf[i].rf_config_index);

        pnf->rfs[0] = rf;
    }

    nfapi_pnf_config_request_t req;
    memset(&req, 0, sizeof(req));
    req.header.message_id = NFAPI_PNF_CONFIG_REQUEST;

    req.pnf_phy_rf_config.tl.tag = NFAPI_PNF_PHY_RF_TAG;
    req.pnf_phy_rf_config.number_phy_rf_config_info = 2; // DJP pnf.phys.size();
    printf("DJP:Hard coded num phy rf to 2\n");

    // DJP for(unsigned i = 0; i < pnf.phys.size(); ++i)
    for(unsigned i = 0; i < 2; ++i)
    {
      req.pnf_phy_rf_config.phy_rf_config[i].phy_id = pnf->phys[i].id;
      req.pnf_phy_rf_config.phy_rf_config[i].phy_config_index = pnf->phys[i].index;
      req.pnf_phy_rf_config.phy_rf_config[i].rf_config_index = pnf->phys[i].rfs[0];
    }

    nfapi_vnf_pnf_config_req(config, p5_idx, &req);
  }
  return 0;
}

int pnf_config_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_pnf_config_response_t* resp)
{
  printf("[VNF] pnf config response idx:%d resp[header[phy_id:%u message_id:%u message_length:%u]]\n", p5_idx, resp->header.phy_id, resp->header.message_id, resp->header.message_length);

  if(1)
  {
    //vnf_info* vnf = (vnf_info*)(config->user_data);
#if 0
    auto find_result = vnf->pnfs.find(p5_idx);
    if(find_result != vnf->pnfs.end())
    {
      //pnf_info& pnf = find_result->second;
    }
#else
      nfapi_pnf_start_request_t req;
      memset(&req, 0, sizeof(req));
      req.header.phy_id = resp->header.phy_id;
      req.header.message_id = NFAPI_PNF_START_REQUEST;
      nfapi_vnf_pnf_start_req(config, p5_idx, &req);
#endif
  }
  else
  {
    // Rather than send the pnf_start_request we will demonstrate
    // sending a vendor extention message. The start request will be
    // send when the vendor extension response is received 

    //vnf_info* vnf = (vnf_info*)(config->user_data);
    vendor_ext_p5_req req;
    memset(&req, 0, sizeof(req));
    req.header.message_id = P5_VENDOR_EXT_REQ;
    req.dummy1 = 45;
    req.dummy2 = 1977;
    nfapi_vnf_vendor_extension(config, p5_idx, &req.header);
  }
  return 0;
}

int wake_eNB_rxtx(PHY_VARS_eNB *eNB, uint16_t sfn, uint16_t sf)
{
  eNB_proc_t *proc=&eNB->proc;

  eNB_rxtx_proc_t *proc_rxtx=&proc->proc_rxtx[sf&1];

  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;

  //int i;
  struct timespec wait;

  wait.tv_sec=0;
  wait.tv_nsec=5000000L;

#if 0
  /* accept some delay in processing - up to 5ms */
  for (i = 0; i < 10 && proc_rxtx->instance_cnt_rxtx == 0; i++) {
    LOG_W( PHY,"[eNB] sfn/sf:%d:%d proc_rxtx[%d]:TXsfn:%d/%d eNB RXn-TXnp4 thread busy!! (cnt_rxtx %i)\n", sfn, sf, sf&1, proc_rxtx->frame_tx, proc_rxtx->subframe_tx, proc_rxtx->instance_cnt_rxtx);
    usleep(500);
  }
  if (proc_rxtx->instance_cnt_rxtx == 0) {
    exit_fun( "TX thread busy" );
    return(-1);
  }
#endif

  // wake up TX for subframe n+4
  // lock the TX mutex and make sure the thread is ready
  if (pthread_mutex_timedlock(&proc_rxtx->mutex_rxtx,&wait) != 0) {
    LOG_E( PHY, "[eNB] ERROR pthread_mutex_lock for eNB RXTX thread %d (IC %d)\n", proc_rxtx->subframe_rx&1,proc_rxtx->instance_cnt_rxtx );
    exit_fun( "error locking mutex_rxtx" );
    return(-1);
  }

  {
    static uint16_t old_sf = 0;
    static uint16_t old_sfn = 0;

    proc->subframe_rx = old_sf;
    proc->frame_rx = old_sfn;

    // Try to be 1 frame back
    old_sf = sf;
    old_sfn = sfn;

    if (old_sf == 0 && old_sfn % 100==0) LOG_W( PHY,"[eNB] sfn/sf:%d:%d old_sfn/sf:%d:%d proc[frame_rx:%d subframe_rx:%d]\n", sfn, sf, old_sfn, old_sf, proc->frame_rx, proc->subframe_rx);
  }

  ++proc_rxtx->instance_cnt_rxtx;

  //LOG_E( PHY,"[VNF-subframe_ind] sfn/sf:%d:%d proc[frame_rx:%d subframe_rx:%d] proc_rxtx->instance_cnt_rxtx:%d \n", sfn, sf, proc->frame_rx, proc->subframe_rx, proc_rxtx->instance_cnt_rxtx);

  // We have just received and processed the common part of a subframe, say n.
  // TS_rx is the last received timestamp (start of 1st slot), TS_tx is the desired
  // transmitted timestamp of the next TX slot (first).
  // The last (TS_rx mod samples_per_frame) was n*samples_per_tti,
  // we want to generate subframe (n+4), so TS_tx = TX_rx+4*samples_per_tti,
  // and proc->subframe_tx = proc->subframe_rx+4
  proc_rxtx->timestamp_tx = proc->timestamp_rx + (4*fp->samples_per_tti);
  proc_rxtx->frame_rx     = proc->frame_rx;
  proc_rxtx->subframe_rx  = proc->subframe_rx;
  proc_rxtx->frame_tx     = (proc_rxtx->subframe_rx > 5) ? (proc_rxtx->frame_rx+1)&1023 : proc_rxtx->frame_rx;
  proc_rxtx->subframe_tx  = (proc_rxtx->subframe_rx + 4)%10;

  // the thread can now be woken up
  if (pthread_cond_signal(&proc_rxtx->cond_rxtx) != 0) {
    LOG_E( PHY, "[eNB] ERROR pthread_cond_signal for eNB RXn-TXnp4 thread\n");
    exit_fun( "ERROR pthread_cond_signal" );
    return(-1);
  }

  pthread_mutex_unlock( &proc_rxtx->mutex_rxtx );

  return(0);
}

extern pthread_cond_t nfapi_sync_cond;
extern pthread_mutex_t nfapi_sync_mutex;
extern int nfapi_sync_var;

int phy_sync_indication(struct nfapi_vnf_p7_config* config, uint8_t sync)
{
  //vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);

  printf("[VNF] SYNC %s\n", sync==1 ? "ACHIEVED" : "LOST");
  
  if (1)
  {
    if (sync==1 && nfapi_sync_var!=0)
    {

      printf("[VNF] Signal to OAI main code that it can go\n");
      pthread_mutex_lock(&nfapi_sync_mutex);
      nfapi_sync_var=0;
      pthread_cond_broadcast(&nfapi_sync_cond);
      pthread_mutex_unlock(&nfapi_sync_mutex);
    }
  }

  //if (RC.eNB && RC.eNB[0][0]->configured)
  //wake_eNB_rxtx(RC.eNB[0][0], 0, 0);

  return(0);
}

int phy_subframe_indication(struct nfapi_vnf_p7_config* config, uint16_t phy_id, uint16_t sfn_sf)
{
  static uint8_t first_time = 1;
  if (first_time)
  {
    printf("[VNF] subframe indication %d\n", sfn_sf);
    first_time = 0;
  }

  // vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
  //mac_subframe_ind(p7_vnf->mac, phy_id, sfn_sf);

#if 1
  if (RC.eNB && RC.eNB[0][0]->configured)
  {
    uint16_t sfn = NFAPI_SFNSF2SFN(sfn_sf);
    uint16_t sf = NFAPI_SFNSF2SF(sfn_sf);

    //LOG_E(PHY,"[VNF] subframe indication sfn_sf:%d sfn:%d sf:%d\n", sfn_sf, sfn, sf);

    wake_eNB_rxtx(RC.eNB[0][0], sfn, sf);
  }
  else
  {
    printf("[VNF] %s() RC.eNB:%p\n", __FUNCTION__, RC.eNB);
    if (RC.eNB) printf("RC.eNB[0][0]->configured:%d\n", RC.eNB[0][0]->configured);
  }
#endif

  return 0;
}

int phy_harq_indication(struct nfapi_vnf_p7_config* config, nfapi_harq_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_harq_ind(p7_vnf->mac, ind);
	return 1;
}

int phy_crc_indication(struct nfapi_vnf_p7_config* config, nfapi_crc_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_crc_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_rx_indication(struct nfapi_vnf_p7_config* config, nfapi_rx_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_rx_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_rach_indication(struct nfapi_vnf_p7_config* config, nfapi_rach_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_rach_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_srs_indication(struct nfapi_vnf_p7_config* config, nfapi_srs_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_srs_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_sr_indication(struct nfapi_vnf_p7_config* config, nfapi_sr_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_sr_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_cqi_indication(struct nfapi_vnf_p7_config* config, nfapi_cqi_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_cqi_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_lbt_dl_indication(struct nfapi_vnf_p7_config* config, nfapi_lbt_dl_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_lbt_dl_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_nb_harq_indication(struct nfapi_vnf_p7_config* config, nfapi_nb_harq_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_nb_harq_ind(p7_vnf->mac, ind);
	return 1;
}
int phy_nrach_indication(struct nfapi_vnf_p7_config* config, nfapi_nrach_indication_t* ind)
{
	// vnf_p7_info* p7_vnf = (vnf_p7_info*)(config->user_data);
	//mac_nrach_ind(p7_vnf->mac, ind);
	return 1;
}

void* vnf_allocate(size_t size)
{
	//return (void*)memory_pool::allocate(size);
	return (void*)malloc(size);
}

void vnf_deallocate(void* ptr)
{
	//memory_pool::deallocate((uint8_t*)ptr);
	free(ptr);
}

void vnf_trace(nfapi_trace_level_t level, const char* message, ...)
{
	va_list args;
	va_start(args, message);
	vprintf(message, args);
	va_end(args);
}

int phy_vendor_ext(struct nfapi_vnf_p7_config* config, nfapi_p7_message_header_t* msg)
{
	if(msg->message_id == P7_VENDOR_EXT_IND)
	{
		//vendor_ext_p7_ind* ind = (vendor_ext_p7_ind*)msg;
		//printf("[VNF] vendor_ext (error_code:%d)\n", ind->error_code);
	}
	else
	{
		printf("[VNF] unknown %d\n", msg->message_id);
	}
	return 0;
}

nfapi_p7_message_header_t* phy_allocate_p7_vendor_ext(uint16_t message_id, uint16_t* msg_size)
{
	if(message_id == P7_VENDOR_EXT_IND)
	{
		*msg_size = sizeof(vendor_ext_p7_ind);
		return (nfapi_p7_message_header_t*)malloc(sizeof(vendor_ext_p7_ind));
	}
	return 0;
}

void phy_deallocate_p7_vendor_ext(nfapi_p7_message_header_t* header)
{
	free(header);
}

/// maybe these should be in the mac file...
int phy_unpack_vendor_extension_tlv(nfapi_tl_t* tl, uint8_t **ppReadPackedMessage, uint8_t *end, void** ve, nfapi_p7_codec_config_t* codec)
{
	(void)tl;
	(void)ppReadPackedMessage;
	(void)ve;
	return -1;
}

int phy_pack_vendor_extension_tlv(void* ve, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p7_codec_config_t* codec)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "phy_pack_vendor_extension_tlv\n");

	nfapi_tl_t* tlv = (nfapi_tl_t*)ve;
	switch(tlv->tag)
	{
		case VENDOR_EXT_TLV_1_TAG:
			{
				//NFAPI_TRACE(NFAPI_TRACE_INFO, "Packing VENDOR_EXT_TLV_1\n");
				vendor_ext_tlv_1* ve = (vendor_ext_tlv_1*)tlv;
				if(!push32(ve->dummy, ppWritePackedMsg, end))
					return 0;
				return 1;
			}
			break;
		default:
			return -1;
			break;
	}
}

int phy_unpack_p7_vendor_extension(nfapi_p7_message_header_t* header, uint8_t** ppReadPackedMessage, uint8_t *end, nfapi_p7_codec_config_t* config)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
	if(header->message_id == P7_VENDOR_EXT_IND)
	{
		vendor_ext_p7_ind* req = (vendor_ext_p7_ind*)(header);
		if(!pull16(ppReadPackedMessage, &req->error_code, end))
			return 0;
	}
	return 1;
}

int phy_pack_p7_vendor_extension(nfapi_p7_message_header_t* header, uint8_t** ppWritePackedMsg, uint8_t *end, nfapi_p7_codec_config_t* config)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
	if(header->message_id == P7_VENDOR_EXT_REQ)
	{
		//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
		vendor_ext_p7_req* req = (vendor_ext_p7_req*)(header);
		if(!(push16(req->dummy1, ppWritePackedMsg, end) &&
			 push16(req->dummy2, ppWritePackedMsg, end)))
			return 0;
	}
	return 1;
}

int vnf_pack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t* header, uint8_t** ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t* codec)
{
	//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
	if(header->message_id == P5_VENDOR_EXT_REQ)
	{
		vendor_ext_p5_req* req = (vendor_ext_p5_req*)(header);
		//NFAPI_TRACE(NFAPI_TRACE_INFO, "%s %d %d\n", __FUNCTION__, req->dummy1, req->dummy2);
		return (!(push16(req->dummy1, ppWritePackedMsg, end) &&
				  push16(req->dummy2, ppWritePackedMsg, end)));
	}
	return 0;
}

static pthread_t vnf_start_pthread;
static pthread_t vnf_p7_start_pthread;
void* vnf_p7_start_thread(void *ptr)
{
  printf("%s()\n", __FUNCTION__);

  nfapi_vnf_p7_config_t* config = (nfapi_vnf_p7_config_t*)ptr;

  nfapi_vnf_p7_start(config);

  return config;
}

void set_thread_priority(int priority);

void* vnf_p7_thread_start(void* ptr)
{
	set_thread_priority(79);

	vnf_p7_info* p7_vnf = (vnf_p7_info*)ptr;

	p7_vnf->config->port = p7_vnf->local_port;
	p7_vnf->config->sync_indication = &phy_sync_indication;
	p7_vnf->config->subframe_indication = &phy_subframe_indication;
	p7_vnf->config->harq_indication = &phy_harq_indication;
	p7_vnf->config->crc_indication = &phy_crc_indication;
	p7_vnf->config->rx_indication = &phy_rx_indication;
	p7_vnf->config->rach_indication = &phy_rach_indication;
	p7_vnf->config->srs_indication = &phy_srs_indication;
	p7_vnf->config->sr_indication = &phy_sr_indication;
	p7_vnf->config->cqi_indication = &phy_cqi_indication;
	p7_vnf->config->lbt_dl_indication = &phy_lbt_dl_indication;
	p7_vnf->config->nb_harq_indication = &phy_nb_harq_indication;
	p7_vnf->config->nrach_indication = &phy_nrach_indication;
	p7_vnf->config->malloc = &vnf_allocate;
	p7_vnf->config->free = &vnf_deallocate;

	p7_vnf->config->trace = &vnf_trace;

	p7_vnf->config->vendor_ext = &phy_vendor_ext;
	p7_vnf->config->user_data = p7_vnf;

	p7_vnf->mac->user_data = p7_vnf;

	p7_vnf->config->codec_config.unpack_p7_vendor_extension = &phy_unpack_p7_vendor_extension;
	p7_vnf->config->codec_config.pack_p7_vendor_extension = &phy_pack_p7_vendor_extension;
	p7_vnf->config->codec_config.unpack_vendor_extension_tlv = &phy_unpack_vendor_extension_tlv;
	p7_vnf->config->codec_config.pack_vendor_extension_tlv = &phy_pack_vendor_extension_tlv;
	p7_vnf->config->codec_config.allocate = &vnf_allocate;
	p7_vnf->config->codec_config.deallocate = &vnf_deallocate;

	p7_vnf->config->allocate_p7_vendor_ext = &phy_allocate_p7_vendor_ext;
	p7_vnf->config->deallocate_p7_vendor_ext = &phy_deallocate_p7_vendor_ext;

        {
          NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] Creating VNF NFAPI start thread %s\n", __FUNCTION__);
          pthread_create(&vnf_p7_start_pthread, NULL, &vnf_p7_start_thread, p7_vnf->config);
        }
	return 0;
}

int pnf_start_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_pnf_start_response_t* resp)
{
  vnf_info* vnf = (vnf_info*)(config->user_data);
  vnf_p7_info *p7_vnf = vnf->p7_vnfs;

  printf("[VNF] pnf start response idx:%d config:%p user_data:%p p7_vnf[config:%p thread_started:%d]\n", p5_idx, config, config->user_data, vnf->p7_vnfs[0].config, vnf->p7_vnfs[0].thread_started);

  if(p7_vnf->thread_started == 0)
  {
    pthread_t vnf_p7_thread;
    pthread_create(&vnf_p7_thread, NULL, &vnf_p7_thread_start, p7_vnf);
    p7_vnf->thread_started = 1;
  }
  else
  {
    // P7 thread already running. 
  }

  // start all the phys in the pnf.

#if 0
  auto find_result = vnf->pnfs.find(p5_idx);
  if(find_result != vnf->pnfs.end())
  {
    pnf_info& pnf = find_result->second;

    for(unsigned i = 0; i < pnf.phys.size(); ++i)
    {
      pnf_info& pnf = find_result->second;
    }
  }
#else
  {
    pnf_info *pnf = vnf->pnfs;
    nfapi_param_request_t req;

    printf("[VNF] Sending NFAPI_PARAM_REQUEST phy_id:%d\n", pnf->phys[0].id);

    memset(&req, 0, sizeof(req));
    req.header.message_id = NFAPI_PARAM_REQUEST;
    req.header.phy_id = pnf->phys[0].id;

    nfapi_vnf_param_req(config, p5_idx, &req);
  }
#endif

  return 0;
}

extern uint32_t to_earfcn(int eutra_bandP,uint32_t dl_CarrierFreq,uint32_t bw);

int param_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_param_response_t* resp)
{
  printf("[VNF] Received NFAPI_PARAM_RESP idx:%d phy_id:%d\n", p5_idx, resp->header.phy_id);

  vnf_info* vnf = (vnf_info*)(config->user_data);

#if 0
  auto find_result = vnf->pnfs.find(p5_idx);
  if(find_result != vnf->pnfs.end())
  {
    pnf_info& pnf = find_result->second;

    auto found = std::find_if(pnf.phys.begin(), pnf.phys.end(), [&](phy_info& item)
        { return item.id == resp->header.phy_id; });

    if(found != pnf.phys.end())
    {
      phy_info& phy = (*found);
#else
  pnf_info *pnf = vnf->pnfs;
  phy_info *phy = pnf->phys;
  {
    {
#endif

      phy->remote_port = resp->nfapi_config.p7_pnf_port.value;

      struct sockaddr_in pnf_p7_sockaddr;
      memcpy(&pnf_p7_sockaddr.sin_addr.s_addr, &(resp->nfapi_config.p7_pnf_address_ipv4.address[0]), 4);

      phy->remote_addr = inet_ntoa(pnf_p7_sockaddr.sin_addr);

      // for now just 1
      vnf_p7_info *p7_vnf = vnf->p7_vnfs;

      printf("[VNF] %d.%d pnf p7 %s:%d timing %d %d %d %d\n", p5_idx, phy->id, phy->remote_addr, phy->remote_port, p7_vnf->timing_window, p7_vnf->periodic_timing_period, p7_vnf->aperiodic_timing_enabled, p7_vnf->periodic_timing_period);

      nfapi_config_request_t *req = &RC.mac[0]->config[0];

      //memset(&req, 0, sizeof(req));
      req->header.message_id = NFAPI_CONFIG_REQUEST;
      req->header.phy_id = phy->id;

      printf("[VNF] Send NFAPI_CONFIG_REQUEST\n");

      req->nfapi_config.p7_vnf_port.tl.tag = NFAPI_NFAPI_P7_VNF_PORT_TAG;
      req->nfapi_config.p7_vnf_port.value = p7_vnf->local_port;
      req->num_tlv++;

printf("[VNF] DJP local_port:%d\n", p7_vnf->local_port);

      req->nfapi_config.p7_vnf_address_ipv4.tl.tag = NFAPI_NFAPI_P7_VNF_ADDRESS_IPV4_TAG;
      struct sockaddr_in vnf_p7_sockaddr;
      vnf_p7_sockaddr.sin_addr.s_addr = inet_addr(p7_vnf->local_addr);
      memcpy(&(req->nfapi_config.p7_vnf_address_ipv4.address[0]), &vnf_p7_sockaddr.sin_addr.s_addr, 4);
      req->num_tlv++;
printf("[VNF] DJP local_addr:%s\n", p7_vnf->local_addr);

      req->nfapi_config.timing_window.tl.tag = NFAPI_NFAPI_TIMING_WINDOW_TAG;
      req->nfapi_config.timing_window.value = p7_vnf->timing_window;
      req->num_tlv++;

      if(p7_vnf->periodic_timing_enabled || p7_vnf->aperiodic_timing_enabled)
      {
        req->nfapi_config.timing_info_mode.tl.tag = NFAPI_NFAPI_TIMING_INFO_MODE_TAG;
        req->nfapi_config.timing_info_mode.value = (p7_vnf->aperiodic_timing_enabled << 1) | (p7_vnf->periodic_timing_enabled);
        req->num_tlv++;

        if(p7_vnf->periodic_timing_enabled)
        {
          req->nfapi_config.timing_info_period.tl.tag = NFAPI_NFAPI_TIMING_INFO_PERIOD_TAG;
          req->nfapi_config.timing_info_period.value = p7_vnf->periodic_timing_period;
          req->num_tlv++;
        }
      }

      vendor_ext_tlv_2 ve2;
      memset(&ve2, 0, sizeof(ve2));
      ve2.tl.tag = VENDOR_EXT_TLV_2_TAG;
      ve2.dummy = 2016;
      req->vendor_extension = &ve2.tl;

      nfapi_vnf_config_req(config, p5_idx, req);
      printf("[VNF] Sent NFAPI_CONFIG_REQ num_tlv:%u\n",req->num_tlv);
    }
#if 0
    else
    {
      printf("[VNF] param response failed to find pnf %d phy %d\n", p5_idx, resp->header.phy_id);
    }
#endif
  }
#if 0
  else
  {
    printf("[VNF] param response failed to find pnf %d\n", p5_idx);
  }
#endif

  return 0;
}

int config_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_config_response_t* resp)
{
  nfapi_start_request_t req;

  printf("[VNF] Received NFAPI_CONFIG_RESP idx:%d phy_id:%d\n", p5_idx, resp->header.phy_id);

  printf("[VNF] Calling oai_enb_init()\n");
  oai_enb_init();

  memset(&req, 0, sizeof(req));
  req.header.message_id = NFAPI_START_REQUEST;
  req.header.phy_id = resp->header.phy_id;
  nfapi_vnf_start_req(config, p5_idx, &req);

  printf("[VNF] Send NFAPI_START_REQUEST idx:%d phy_id:%d\n", p5_idx, resp->header.phy_id);

  return 0;
}

int start_resp_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_start_response_t* resp)
{
  printf("[VNF] Received NFAPI_START_RESP idx:%d phy_id:%d\n", p5_idx, resp->header.phy_id);

  vnf_info* vnf = (vnf_info*)(config->user_data);

#if 0
  auto find_result = vnf->pnfs.find(p5_idx);
  if(find_result != vnf->pnfs.end())
  {
    pnf_info& pnf = find_result->second;


    auto found = std::find_if(pnf.phys.begin(), pnf.phys.end(), [&](phy_info& item)
        { return item.id == resp->header.phy_id; });

    if(found != pnf.phys.end())
    {
      phy_info& phy = (*found);

      vnf_p7_info& p7_vnf = vnf->p7_vnfs[0];

      nfapi_vnf_p7_add_pnf((p7_vnf.config.get()), phy.remote_addr.c_str(), phy.remote_port, phy.id);

    }
  }
#else
  pnf_info *pnf = vnf->pnfs;
  phy_info *phy = pnf->phys;
  vnf_p7_info *p7_vnf = vnf->p7_vnfs;
  nfapi_vnf_p7_add_pnf((p7_vnf->config), phy->remote_addr, phy->remote_port, phy->id);


#if 0
  {
    extern pthread_cond_t nfapi_sync_cond;
    extern pthread_mutex_t nfapi_sync_mutex;
    extern int nfapi_sync_var;

    printf("[VNF] Signal to OAI main code that it can go\n");
    pthread_mutex_lock(&nfapi_sync_mutex);
    nfapi_sync_var=0;
    pthread_cond_broadcast(&nfapi_sync_cond);
    pthread_mutex_unlock(&nfapi_sync_mutex);
  }
#endif

#endif

  return 0;
}

int vendor_ext_cb(nfapi_vnf_config_t* config, int p5_idx, nfapi_p4_p5_message_header_t* msg)
{
  printf("[VNF] %s\n", __FUNCTION__);

  switch(msg->message_id)
  {
    case P5_VENDOR_EXT_RSP:
      {
        vendor_ext_p5_rsp* rsp = (vendor_ext_p5_rsp*)msg;
        printf("[VNF] P5_VENDOR_EXT_RSP error_code:%d\n", rsp->error_code);

        // send the start request

        nfapi_pnf_start_request_t req;
        memset(&req, 0, sizeof(req));
        req.header.message_id = NFAPI_PNF_START_REQUEST;
        nfapi_vnf_pnf_start_req(config, p5_idx, &req);
      }
      break;
  }

  return 0;
}

int vnf_unpack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t* header, uint8_t** ppReadPackedMessage, uint8_t *end, nfapi_p4_p5_codec_config_t* codec)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P5_VENDOR_EXT_RSP)
  {
    vendor_ext_p5_rsp* req = (vendor_ext_p5_rsp*)(header);
    return(!pull16(ppReadPackedMessage, &req->error_code, end));
  }
  return 0;
}

nfapi_p4_p5_message_header_t* vnf_allocate_p4_p5_vendor_ext(uint16_t message_id, uint16_t* msg_size)
{
  if(message_id == P5_VENDOR_EXT_RSP)
  {
    *msg_size = sizeof(vendor_ext_p5_rsp);
    return (nfapi_p4_p5_message_header_t*)malloc(sizeof(vendor_ext_p5_rsp));
  }
  return 0;
}

void vnf_deallocate_p4_p5_vendor_ext(nfapi_p4_p5_message_header_t* header)
{
  free(header);
}

nfapi_vnf_config_t *config = 0;

void vnf_start_thread(void* ptr)
{
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] VNF NFAPI thread - nfapi_vnf_start()%s\n", __FUNCTION__);

  config = (nfapi_vnf_config_t*)ptr;

  nfapi_vnf_start(config);
}

static vnf_info vnf;
/*------------------------------------------------------------------------------*/
void configure_nfapi_vnf(char *vnf_addr, int vnf_p5_port)
{
  memset(&vnf, 0, sizeof(vnf));

  memset(vnf.p7_vnfs, 0, sizeof(vnf.p7_vnfs));

  vnf.p7_vnfs[0].timing_window = 32;
  vnf.p7_vnfs[0].periodic_timing_enabled = 1;
  vnf.p7_vnfs[0].aperiodic_timing_enabled = 0;
  vnf.p7_vnfs[0].periodic_timing_period = 10;

  vnf.p7_vnfs[0].config = nfapi_vnf_p7_config_create();
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] %s() vnf.p7_vnfs[0].config:%p VNF ADDRESS:%s:%d\n", __FUNCTION__, vnf.p7_vnfs[0].config, vnf_addr, vnf_p5_port);

  strcpy(vnf.p7_vnfs[0].local_addr, vnf_addr);
  vnf.p7_vnfs[0].local_port = 50001;
  vnf.p7_vnfs[0].mac = (mac_t*)malloc(sizeof(mac_t));

  nfapi_vnf_config_t* config = nfapi_vnf_config_create();

  config->malloc = malloc;
  config->free = free;
  config->trace = &vnf_trace;

  config->vnf_p5_port = vnf_p5_port;
  config->vnf_ipv4 = 1;
  config->vnf_ipv6 = 0;

  config->pnf_list = 0;
  config->phy_list = 0;

  config->pnf_connection_indication = &pnf_connection_indication_cb;
  config->pnf_disconnect_indication = &pnf_disconnection_indication_cb;
  config->pnf_param_resp = &pnf_param_resp_cb;
  config->pnf_config_resp = &pnf_config_resp_cb;
  config->pnf_start_resp = &pnf_start_resp_cb;
  config->param_resp = &param_resp_cb;
  config->config_resp = &config_resp_cb;
  config->start_resp = &start_resp_cb;
  config->vendor_ext = &vendor_ext_cb;

  config->user_data = &vnf;

  // To allow custom vendor extentions to be added to nfapi
  config->codec_config.unpack_vendor_extension_tlv = &vnf_unpack_vendor_extension_tlv;
  config->codec_config.pack_vendor_extension_tlv = &vnf_pack_vendor_extension_tlv;

  config->codec_config.unpack_p4_p5_vendor_extension = &vnf_unpack_p4_p5_vendor_extension;
  config->codec_config.pack_p4_p5_vendor_extension = &vnf_pack_p4_p5_vendor_extension;
  config->allocate_p4_p5_vendor_ext = &vnf_allocate_p4_p5_vendor_ext;
  config->deallocate_p4_p5_vendor_ext = &vnf_deallocate_p4_p5_vendor_ext;
  config->codec_config.allocate = &vnf_allocate;
  config->codec_config.deallocate = &vnf_deallocate;

  {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] Creating VNF NFAPI start thread %s\n", __FUNCTION__);
    pthread_create(&vnf_start_pthread, NULL, (void*)&vnf_start_thread, config);
    NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] Created VNF NFAPI start thread %s\n", __FUNCTION__);
  }
}

int oai_nfapi_dl_config_req(nfapi_dl_config_request_t *dl_config_req)
{
  nfapi_vnf_p7_config_t *p7_config = vnf.p7_vnfs[0].config;

  dl_config_req->header.phy_id = 1; // DJP HACK TODO FIXME - need to pass this around!!!!
  //dl_config_req->header.message_id = NFAPI_DL_CONFIG_BCH_PDU_TYPE;

  //NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] %s() header message_id:%d\n", __FUNCTION__, dl_config_req->header.message_id);

  //NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] %s() p7_config:%p phy_id:%d message_id:%d sfn_sf:%d pdcch:%d dci:%d pdu:%d pdsch_rnti:%d pcfich:%d\n", __FUNCTION__, p7_config, dl_config_req->header.phy_id, dl_config_req->header.message_id, dl_config_req->sfn_sf, dl_config_req->dl_config_request_body.number_pdcch_ofdm_symbols, dl_config_req->dl_config_request_body.number_dci, dl_config_req->dl_config_request_body.number_pdu, dl_config_req->dl_config_request_body.number_pdsch_rnti, dl_config_req->dl_config_request_body.transmission_power_pcfich);

  return nfapi_vnf_p7_dl_config_req(p7_config, dl_config_req);
}

int oai_nfapi_tx_req(nfapi_tx_request_t *tx_req)
{
  nfapi_vnf_p7_config_t *p7_config = vnf.p7_vnfs[0].config;

  tx_req->header.phy_id = 1; // DJP HACK TODO FIXME - need to pass this around!!!!
  //dl_config_req->header.message_id = NFAPI_DL_CONFIG_BCH_PDU_TYPE;

  //NFAPI_TRACE(NFAPI_TRACE_INFO, "[VNF] %s() p7_config:%p phy_id:%d message_id:%d sfn_sf:%d pdcch:%d dci:%d pdu:%d pdsch_rnti:%d pcfich:%d\n", __FUNCTION__, p7_config, dl_config_req->header.phy_id, dl_config_req->header.message_id, dl_config_req->sfn_sf, dl_config_req->dl_config_request_body.number_pdcch_ofdm_symbols, dl_config_req->dl_config_request_body.number_dci, dl_config_req->dl_config_request_body.number_pdu, dl_config_req->dl_config_request_body.number_pdsch_rnti, dl_config_req->dl_config_request_body.transmission_power_pcfich);

  return nfapi_vnf_p7_tx_req(p7_config, tx_req);
}
