#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#include "debug.h"
#include "nfapi_pnf_interface.h"
#include "nfapi.h"
#include "nfapi_pnf.h"
#include "common/ran_context.h"
//#include "openair1/PHY/vars.h"
extern RAN_CONTEXT_t RC;

#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <assert.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <errno.h>

#include <vendor_ext.h>
#include "fapi_stub.h"

#define NUM_P5_PHY 2

extern void phy_init_RU(RU_t*);
extern int mac_top_init_eNB(void);



uint16_t phy_antenna_capability_values[] = { 1, 2, 4, 8, 16 };

nfapi_pnf_param_response_t g_pnf_param_resp;


nfapi_pnf_p7_config_t *p7_config_g = NULL;

void* pnf_allocate(size_t size)
{
  //DJP
  //return (void*)memory_pool::allocate(size);
  return malloc(size);
}

void pnf_deallocate(void* ptr)
{
  //DJP
  //memory_pool::deallocate((uint8_t*)ptr);
  free(ptr);
}


//class udp_data
//DJP
typedef struct 
{
  //public:
  uint8_t enabled;
  uint32_t rx_port;
  uint32_t tx_port;
  //std::string tx_addr;
  char tx_addr[80];
}udp_data;

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

    //fapi_t* fapi;

}phy_info;

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
}rf_info;


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

}pnf_info;

// DJP struct pnf_phy_user_data_t
typedef struct 
{
  uint16_t phy_id;
  nfapi_pnf_config_t* config;
  phy_info* phy;
  nfapi_pnf_p7_config_t* p7_config;
}pnf_phy_user_data_t;


void pnf_sim_trace(nfapi_trace_level_t level, const char* message, ...)
{
  va_list args;
  va_start(args, message);
  vprintf(message, args);
  va_end(args);
}

void pnf_set_thread_priority(int priority)
{
  //printf("%s(priority:%d)\n", __FUNCTION__, priority);

  pthread_attr_t ptAttr;

  struct sched_param schedParam;
  schedParam.__sched_priority = priority; //79;
  if(sched_setscheduler(0, SCHED_RR, &schedParam) != 0)
  {
    printf("failed to set SCHED_RR\n");
  }

  if(pthread_attr_setschedpolicy(&ptAttr, SCHED_RR) != 0)
  {
    printf("failed to set pthread SCHED_RR %d\n", errno);
  }

  pthread_attr_setinheritsched(&ptAttr, PTHREAD_EXPLICIT_SCHED);

  struct sched_param thread_params;
  thread_params.sched_priority = 20;
  if(pthread_attr_setschedparam(&ptAttr, &thread_params) != 0)
  {
    printf("failed to set sched param\n");
  }
}

void* pnf_p7_thread_start(void* ptr)
{
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] P7 THREAD %s\n", __FUNCTION__);

  pnf_set_thread_priority(79);

  nfapi_pnf_p7_config_t* config = (nfapi_pnf_p7_config_t*)ptr;
  nfapi_pnf_p7_start(config);

  return 0;
}



int pnf_param_request(nfapi_pnf_config_t* config, nfapi_pnf_param_request_t* req)
{
  printf("[PNF] pnf param request\n");

  nfapi_pnf_param_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_PNF_PARAM_RESPONSE;
  resp.error_code = NFAPI_MSG_OK;

  pnf_info* pnf = (pnf_info*)(config->user_data);

  resp.pnf_param_general.tl.tag = NFAPI_PNF_PARAM_GENERAL_TAG;
  resp.pnf_param_general.nfapi_sync_mode = pnf->sync_mode;
  resp.pnf_param_general.location_mode = pnf->location_mode;
  //uint8_t location_coordinates[NFAPI_PNF_PARAM_GENERAL_LOCATION_LENGTH];
  resp.pnf_param_general.dl_config_timing = pnf->dl_config_timing;
  resp.pnf_param_general.tx_timing = pnf->tx_timing;
  resp.pnf_param_general.ul_config_timing = pnf->ul_config_timing;
  resp.pnf_param_general.hi_dci0_timing = pnf->hi_dci0_timing;
  resp.pnf_param_general.maximum_number_phys = pnf->max_phys;
  resp.pnf_param_general.maximum_total_bandwidth = pnf->max_total_bw;
  resp.pnf_param_general.maximum_total_number_dl_layers = pnf->max_total_dl_layers;
  resp.pnf_param_general.maximum_total_number_ul_layers = pnf->max_total_ul_layers;
  resp.pnf_param_general.shared_bands = pnf->shared_bands;
  resp.pnf_param_general.shared_pa = pnf->shared_pa;
  resp.pnf_param_general.maximum_total_power = pnf->max_total_power;
  //uint8_t oui[NFAPI_PNF_PARAM_GENERAL_OUI_LENGTH];

  resp.pnf_phy.tl.tag = NFAPI_PNF_PHY_TAG;
  //DJP resp.pnf_phy.number_of_phys = pnf->phys.size();
  resp.pnf_phy.number_of_phys = 1;

  //for(int i = 0; i < pnf->phys.size(); ++i)
  for(int i = 0; i < 1; ++i)
  {
    resp.pnf_phy.phy[i].phy_config_index = pnf->phys[i].index; 
    resp.pnf_phy.phy[i].downlink_channel_bandwidth_supported = pnf->phys[i].dl_channel_bw_support;
    resp.pnf_phy.phy[i].uplink_channel_bandwidth_supported = pnf->phys[i].ul_channel_bw_support;
    resp.pnf_phy.phy[i].number_of_dl_layers_supported = pnf->phys[i].num_dl_layers_supported;
    resp.pnf_phy.phy[i].number_of_ul_layers_supported = pnf->phys[i].num_ul_layers_supported;
    resp.pnf_phy.phy[i].maximum_3gpp_release_supported = pnf->phys[i].release_supported;
    resp.pnf_phy.phy[i].nmm_modes_supported = pnf->phys[i].nmm_modes_supported;

    //DJP resp.pnf_phy.phy[i].number_of_rfs = pnf->phys[i].rfs.size();
    resp.pnf_phy.phy[i].number_of_rfs = 2;
    //for(int j = 0; j < pnf->phys[i].rfs.size(); ++j)
    for(int j = 0; j < 1; ++j)
    {
      resp.pnf_phy.phy[i].rf_config[j].rf_config_index = pnf->phys[i].rfs[j];
    }

    //DJP resp.pnf_phy.phy[i].number_of_rf_exclusions = pnf->phys[i].excluded_rfs.size();
    resp.pnf_phy.phy[i].number_of_rf_exclusions = 0;
    //DJP for(int j = 0; j < pnf->phys[i].excluded_rfs.size(); ++j)
    for(int j = 0; j < 0; ++j)
    {
      resp.pnf_phy.phy[i].excluded_rf_config[j].rf_config_index = pnf->phys[i].excluded_rfs[j];
    }
  }


  resp.pnf_rf.tl.tag = NFAPI_PNF_RF_TAG;
  //DJPresp.pnf_rf.number_of_rfs = pnf->rfs.size();
  resp.pnf_rf.number_of_rfs = 2;

  //for(int i = 0; i < pnf->rfs.size(); ++i)
  for(int i = 0; i < 2; ++i)
  {
    resp.pnf_rf.rf[i].rf_config_index = pnf->rfs[i].index; 
    resp.pnf_rf.rf[i].band = pnf->rfs[i].band;
    resp.pnf_rf.rf[i].maximum_transmit_power = pnf->rfs[i].max_transmit_power; 
    resp.pnf_rf.rf[i].minimum_transmit_power = pnf->rfs[i].min_transmit_power;
    resp.pnf_rf.rf[i].number_of_antennas_suppported = pnf->rfs[i].num_antennas_supported;
    resp.pnf_rf.rf[i].minimum_downlink_frequency = pnf->rfs[i].min_downlink_frequency;
    resp.pnf_rf.rf[i].maximum_downlink_frequency = pnf->rfs[i].max_downlink_frequency;
    resp.pnf_rf.rf[i].minimum_uplink_frequency = pnf->rfs[i].min_uplink_frequency;
    resp.pnf_rf.rf[i].maximum_uplink_frequency = pnf->rfs[i].max_uplink_frequency;
  }

  if(pnf->release >= 10)
  {
    resp.pnf_phy_rel10.tl.tag = NFAPI_PNF_PHY_REL10_TAG;
    //DJPresp.pnf_phy_rel10.number_of_phys = pnf->phys.size();
    resp.pnf_phy_rel10.number_of_phys = 1;

    //for(int i = 0; i < pnf->phys.size(); ++i)
    for(int i = 0; i < 1; ++i)
    {
      resp.pnf_phy_rel10.phy[i].phy_config_index = pnf->phys[i].index; 
      resp.pnf_phy_rel10.phy[i].transmission_mode_7_supported = 0;
      resp.pnf_phy_rel10.phy[i].transmission_mode_8_supported = 1;
      resp.pnf_phy_rel10.phy[i].two_antenna_ports_for_pucch = 0;
      resp.pnf_phy_rel10.phy[i].transmission_mode_9_supported = 1;
      resp.pnf_phy_rel10.phy[i].simultaneous_pucch_pusch = 0;
      resp.pnf_phy_rel10.phy[i].four_layer_tx_with_tm3_and_tm4 = 1;

    }
  }

  if(pnf->release >= 11)
  {
    resp.pnf_phy_rel11.tl.tag = NFAPI_PNF_PHY_REL11_TAG;
    //DJP resp.pnf_phy_rel11.number_of_phys = pnf->phys.size();
    resp.pnf_phy_rel11.number_of_phys = 1;

    //DJP for(int i = 0; i < pnf->phys.size(); ++i)
    for(int i = 0; i < 1; ++i)
    {
      resp.pnf_phy_rel11.phy[i].phy_config_index = pnf->phys[i].index; 
      resp.pnf_phy_rel11.phy[i].edpcch_supported = 0;
      resp.pnf_phy_rel11.phy[i].multi_ack_csi_reporting = 1;
      resp.pnf_phy_rel11.phy[i].pucch_tx_diversity = 0;
      resp.pnf_phy_rel11.phy[i].ul_comp_supported = 1;
      resp.pnf_phy_rel11.phy[i].transmission_mode_5_supported = 0;
    }
  }

  if(pnf->release >= 12)
  {
    resp.pnf_phy_rel12.tl.tag = NFAPI_PNF_PHY_REL12_TAG;
    //DJP resp.pnf_phy_rel12.number_of_phys = pnf->phys.size();
    resp.pnf_phy_rel12.number_of_phys = 1;

    //DJP for(int i = 0; i < pnf->phys.size(); ++i)
    for(int i = 0; i < 1; ++i)
    {
      resp.pnf_phy_rel12.phy[i].phy_config_index = pnf->phys[i].index; 
      resp.pnf_phy_rel12.phy[i].csi_subframe_set = 0;
      resp.pnf_phy_rel12.phy[i].enhanced_4tx_codebook = 2; // yes this is invalid
      resp.pnf_phy_rel12.phy[i].drs_supported = 0;
      resp.pnf_phy_rel12.phy[i].ul_64qam_supported = 1;
      resp.pnf_phy_rel12.phy[i].transmission_mode_10_supported = 0;
      resp.pnf_phy_rel12.phy[i].alternative_bts_indices = 1;
    }
  }

  if(pnf->release >= 13)
  {
    resp.pnf_phy_rel13.tl.tag = NFAPI_PNF_PHY_REL13_TAG;
    //DJP resp.pnf_phy_rel13.number_of_phys = pnf->phys.size();
    resp.pnf_phy_rel13.number_of_phys = 1;

    //for(int i = 0; i < pnf->phys.size(); ++i)
    for(int i = 0; i < 1; ++i)
    {
      resp.pnf_phy_rel13.phy[i].phy_config_index = pnf->phys[i].index; 
      resp.pnf_phy_rel13.phy[i].pucch_format4_supported = 0;
      resp.pnf_phy_rel13.phy[i].pucch_format5_supported = 1;
      resp.pnf_phy_rel13.phy[i].more_than_5_ca_support = 0;
      resp.pnf_phy_rel13.phy[i].laa_supported = 1;
      resp.pnf_phy_rel13.phy[i].laa_ending_in_dwpts_supported = 0;
      resp.pnf_phy_rel13.phy[i].laa_starting_in_second_slot_supported = 1;
      resp.pnf_phy_rel13.phy[i].beamforming_supported = 0;
      resp.pnf_phy_rel13.phy[i].csi_rs_enhancement_supported = 1;
      resp.pnf_phy_rel13.phy[i].drms_enhancement_supported = 0;
      resp.pnf_phy_rel13.phy[i].srs_enhancement_supported = 1;
    }

    resp.pnf_phy_rel13_nb_iot.tl.tag = NFAPI_PNF_PHY_REL13_NB_IOT_TAG;
    //DJP resp.pnf_phy_rel13_nb_iot.number_of_phys = pnf->phys.size();		
    resp.pnf_phy_rel13_nb_iot.number_of_phys = 1;

    //for(int i = 0; i < pnf->phys.size(); ++i)
    for(int i = 0; i < 1; ++i)
    {
      resp.pnf_phy_rel13_nb_iot.phy[i].phy_config_index = pnf->phys[i].index; 

      //DJP resp.pnf_phy_rel13_nb_iot.phy[i].number_of_rfs = pnf->phys[i].rfs.size();
      resp.pnf_phy_rel13_nb_iot.phy[i].number_of_rfs = 1;
      //DJP for(int j = 0; j < pnf->phys[i].rfs.size(); ++j)
      for(int j = 0; j < 1; ++j)
      {
        resp.pnf_phy_rel13_nb_iot.phy[i].rf_config[j].rf_config_index = pnf->phys[i].rfs[j];
      }

      //DJP resp.pnf_phy_rel13_nb_iot.phy[i].number_of_rf_exclusions = pnf->phys[i].excluded_rfs.size();
      resp.pnf_phy_rel13_nb_iot.phy[i].number_of_rf_exclusions = 1;
      //DJP for(int j = 0; j < pnf->phys[i].excluded_rfs.size(); ++j)
      for(int j = 0; j < 1; ++j)
      {
        resp.pnf_phy_rel13_nb_iot.phy[i].excluded_rf_config[j].rf_config_index = pnf->phys[i].excluded_rfs[j];
      }

      resp.pnf_phy_rel13_nb_iot.phy[i].number_of_dl_layers_supported = pnf->phys[i].num_dl_layers_supported;
      resp.pnf_phy_rel13_nb_iot.phy[i].number_of_ul_layers_supported = pnf->phys[i].num_ul_layers_supported;
      resp.pnf_phy_rel13_nb_iot.phy[i].maximum_3gpp_release_supported = pnf->phys[i].release_supported;
      resp.pnf_phy_rel13_nb_iot.phy[i].nmm_modes_supported = pnf->phys[i].nmm_modes_supported;

    }
  }


  nfapi_pnf_pnf_param_resp(config, &resp);

  return 0;
}

int pnf_config_request(nfapi_pnf_config_t* config, nfapi_pnf_config_request_t* req)
{
  printf("[PNF] pnf config request\n");

  pnf_info* pnf = (pnf_info*)(config->user_data);

#if 0
  for(int i = 0; i < req->pnf_phy_rf_config.number_phy_rf_config_info; ++i)
  {
    auto found = std::find_if(pnf->phys.begin(), pnf->phys.end(), [&](phy_info& item)
        { return item.index == req->pnf_phy_rf_config.phy_rf_config[i].phy_config_index; });

    if(found != pnf->phys.end())
    {
      phy_info& phy = (*found);
      phy.id = req->pnf_phy_rf_config.phy_rf_config[i].phy_id;
      printf("[PNF] pnf config request assigned phy_id %d to phy_config_index %d\n", phy.id, req->pnf_phy_rf_config.phy_rf_config[i].phy_config_index);
    }
    else
    {
      // did not find the phy
      printf("[PNF] pnf config request did not find phy_config_index %d\n", req->pnf_phy_rf_config.phy_rf_config[i].phy_config_index);
    }

  }
#endif
  //DJP
  phy_info *phy = pnf->phys;
  phy->id = req->pnf_phy_rf_config.phy_rf_config[0].phy_id;
  printf("[PNF] pnf config request assigned phy_id %d to phy_config_index %d\n", phy->id, req->pnf_phy_rf_config.phy_rf_config[0].phy_config_index);
  //DJP

  nfapi_pnf_config_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_PNF_CONFIG_RESPONSE;
  resp.error_code = NFAPI_MSG_OK;
  nfapi_pnf_pnf_config_resp(config, &resp);
  printf("[PNF] Sent pnf_config_resp\n");

  return 0;
}

void nfapi_send_pnf_start_resp(nfapi_pnf_config_t* config, uint16_t phy_id)
{
  printf("Sending NFAPI_START_RESPONSE config:%p phy_id:%d\n", config, phy_id);

  nfapi_start_response_t start_resp;
  memset(&start_resp, 0, sizeof(start_resp));
  start_resp.header.message_id = NFAPI_START_RESPONSE;
  start_resp.header.phy_id = phy_id;
  start_resp.error_code = NFAPI_MSG_OK;

  nfapi_pnf_start_resp(config, &start_resp);
}

int pnf_start_request(nfapi_pnf_config_t* config, nfapi_pnf_start_request_t* req)
{
  printf("Received NFAPI_PNF_START_REQUEST\n");

  pnf_info* pnf = (pnf_info*)(config->user_data);

  // start all phys that have been configured
  //for(phy_info& phy : pnf->phys)
  phy_info* phy = pnf->phys;
  if(phy->id != 0)
  {
    //auto found = std::find_if(pnf->phys.begin(), pnf->phys.end(), [&](phy_info& item)
    //		{ return item.id == req->header.phy_id; });
    //
    //	if(found != pnf->phys.end())
    //	{
    //		phy_info& phy = (*found);
    //}

    nfapi_pnf_start_response_t resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.message_id = NFAPI_PNF_START_RESPONSE;
    resp.error_code = NFAPI_MSG_OK;
    nfapi_pnf_pnf_start_resp(config, &resp);
    printf("[PNF] Sent NFAPI_PNF_START_RESP\n");
  }
  return 0;
}

int pnf_stop_request(nfapi_pnf_config_t* config, nfapi_pnf_stop_request_t* req)
{
  printf("[PNF] Received NFAPI_PNF_STOP_REQ\n");

  nfapi_pnf_stop_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_PNF_STOP_RESPONSE;
  resp.error_code = NFAPI_MSG_OK;
  nfapi_pnf_pnf_stop_resp(config, &resp);
  printf("[PNF] Sent NFAPI_PNF_STOP_REQ\n");

  return 0;
}

int param_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_param_request_t* req)
{
  printf("[PNF] Received NFAPI_PARAM_REQUEST phy_id:%d\n", req->header.phy_id);

  //pnf_info* pnf = (pnf_info*)(config->user_data);

  nfapi_param_response_t nfapi_resp;

  pnf_info* pnf = (pnf_info*)(config->user_data);

  memset(&nfapi_resp, 0, sizeof(nfapi_resp));
  nfapi_resp.header.message_id = NFAPI_PARAM_RESPONSE;
  nfapi_resp.header.phy_id = req->header.phy_id;
  nfapi_resp.error_code = 0; // DJP - what value???

  struct sockaddr_in pnf_p7_sockaddr;

  pnf_p7_sockaddr.sin_addr.s_addr = inet_addr(pnf->phys[0].local_addr);
  nfapi_resp.nfapi_config.p7_pnf_address_ipv4.tl.tag = NFAPI_NFAPI_P7_PNF_ADDRESS_IPV4_TAG;
  memcpy(nfapi_resp.nfapi_config.p7_pnf_address_ipv4.address, &pnf_p7_sockaddr.sin_addr.s_addr, 4);
  nfapi_resp.num_tlv++;

  // P7 PNF Port
  nfapi_resp.nfapi_config.p7_pnf_port.tl.tag = NFAPI_NFAPI_P7_PNF_PORT_TAG;
  nfapi_resp.nfapi_config.p7_pnf_port.value = 32123; // DJP - hard code alert!!!! FIXME TODO
  nfapi_resp.num_tlv++;

  nfapi_pnf_param_resp(config, &nfapi_resp);

  printf("[PNF] Sent NFAPI_PARAM_RESPONSE phy_id:%d number_of_tlvs:%u\n", req->header.phy_id, nfapi_resp.num_tlv);
#if 0
  //DJP
  auto found = std::find_if(pnf->phys.begin(), pnf->phys.end(), [&](phy_info& item)
      { return item.id == req->header.phy_id; });

  if(found != pnf->phys.end())
#endif
  {
    //DJP phy_info& phy_info = (*found);
    //phy_info *phy_info = pnf->phys;

  }
#if 0
  else
  {
    // did not find the phy
  }
#endif

  printf("[PNF] param request .. exit\n");

  return 0;
}

// From MAC config.c
extern uint32_t from_earfcn(int eutra_bandP,uint32_t dl_earfcn);
extern int32_t get_uldl_offset(int eutra_bandP);

int config_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_config_request_t* req)
{
  printf("[PNF] Received NFAPI_CONFIG_REQ phy_id:%d\n", req->header.phy_id);

  pnf_info* pnf = (pnf_info*)(config->user_data);
  uint8_t num_tlv = 0;
  struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];
  LTE_DL_FRAME_PARMS *fp = &eNB->frame_parms;

#if 0
  //DJP
  auto found = std::find_if(pnf->phys.begin(), pnf->phys.end(), [&](phy_info& item)
      { return item.id == req->header.phy_id; });

  if(found != pnf->phys.end())
  {
    phy_info& phy_info = (*found);
  }
#endif
  //DJP 
  phy_info* phy_info = pnf->phys;

  if(req->nfapi_config.timing_window.tl.tag == NFAPI_NFAPI_TIMING_WINDOW_TAG)
  {
    phy_info->timing_window = req->nfapi_config.timing_window.value;
    num_tlv++;
  }

  if(req->nfapi_config.timing_info_mode.tl.tag == NFAPI_NFAPI_TIMING_INFO_MODE_TAG)
  {
    printf("timing info mode provided\n");
    phy_info->timing_info_mode = req->nfapi_config.timing_info_mode.value;
    num_tlv++;
  }
  else 
  {
    phy_info->timing_info_mode = 0;
    printf("NO timing info mode provided\n");
  }

  if(req->nfapi_config.timing_info_period.tl.tag == NFAPI_NFAPI_TIMING_INFO_PERIOD_TAG)
  {
    printf("timing info period provided\n");
    phy_info->timing_info_period = req->nfapi_config.timing_info_period.value;
    num_tlv++;
  }
  else 
  {
    phy_info->timing_info_period = 0;
  }

  if(req->rf_config.dl_channel_bandwidth.tl.tag == NFAPI_RF_CONFIG_DL_CHANNEL_BANDWIDTH_TAG)
  {
    phy_info->dl_channel_bw_support = req->rf_config.dl_channel_bandwidth.value;
    fp->N_RB_DL = req->rf_config.dl_channel_bandwidth.value;
    num_tlv++;
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() NFAPI_RF_CONFIG_DL_CHANNEL_BANDWIDTH_TAG N_RB_DL:%u\n", __FUNCTION__, fp->N_RB_DL);
  }
  else
  {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() Missing NFAPI_RF_CONFIG_DL_CHANNEL_BANDWIDTH_TAG\n", __FUNCTION__);
  }

  if(req->rf_config.ul_channel_bandwidth.tl.tag == NFAPI_RF_CONFIG_UL_CHANNEL_BANDWIDTH_TAG)
  {
    phy_info->ul_channel_bw_support = req->rf_config.ul_channel_bandwidth.value;
    fp->N_RB_UL = req->rf_config.ul_channel_bandwidth.value;
    num_tlv++;
  }

  if(req->nfapi_config.rf_bands.tl.tag == NFAPI_NFAPI_RF_BANDS_TAG)
  {
    pnf->rfs[0].band = req->nfapi_config.rf_bands.rf_band[0];
    fp->eutra_band = req->nfapi_config.rf_bands.rf_band[0];
    num_tlv++;
  }

  if(req->nfapi_config.earfcn.tl.tag == NFAPI_NFAPI_EARFCN_TAG)
  {
    fp->dl_CarrierFreq = from_earfcn(fp->eutra_band, req->nfapi_config.earfcn.value); // DJP - TODO FIXME - hard coded to first rf
    fp->ul_CarrierFreq = fp->dl_CarrierFreq - (get_uldl_offset(fp->eutra_band) * 1e5);
    num_tlv++;

    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() earfcn:%u dl_carrierFreq:%u ul_CarrierFreq:%u band:%u N_RB_DL:%u\n", 
        __FUNCTION__, req->nfapi_config.earfcn.value, fp->dl_CarrierFreq, fp->ul_CarrierFreq, pnf->rfs[0].band, fp->N_RB_DL);
  }

  if (req->subframe_config.duplex_mode.tl.tag == NFAPI_SUBFRAME_CONFIG_DUPLEX_MODE_TAG)
  {
    fp->frame_type = req->subframe_config.duplex_mode.value==0 ? TDD : FDD;
    num_tlv++;
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() frame_type:%d\n", __FUNCTION__, fp->frame_type);
  }
  if (req->subframe_config.dl_cyclic_prefix_type.tl.tag == NFAPI_SUBFRAME_CONFIG_DL_CYCLIC_PREFIX_TYPE_TAG)
  {
    fp->Ncp = req->subframe_config.dl_cyclic_prefix_type.value;
    num_tlv++;
  }

  if (req->subframe_config.ul_cyclic_prefix_type.tl.tag == NFAPI_SUBFRAME_CONFIG_UL_CYCLIC_PREFIX_TYPE_TAG)
  {
    fp->Ncp_UL = req->subframe_config.ul_cyclic_prefix_type.value;
    num_tlv++;
  }

  fp->num_MBSFN_config = 0; // DJP - hard code alert

  if (req->sch_config.physical_cell_id.tl.tag == NFAPI_SCH_CONFIG_PHYSICAL_CELL_ID_TAG)
  {
    fp->Nid_cell = req->sch_config.physical_cell_id.value;
    fp->nushift = fp->Nid_cell%6;
    num_tlv++;
  }

  if (req->rf_config.tx_antenna_ports.tl.tag == NFAPI_RF_CONFIG_TX_ANTENNA_PORTS_TAG)
  {
    fp->nb_antennas_tx = req->rf_config.tx_antenna_ports.value;
    fp->nb_antenna_ports_eNB = 1;
    num_tlv++;
  }

  if (req->rf_config.rx_antenna_ports.tl.tag == NFAPI_RF_CONFIG_RX_ANTENNA_PORTS_TAG)
  {
    fp->nb_antennas_rx = req->rf_config.rx_antenna_ports.value;
    num_tlv++;
  }

  if (req->phich_config.phich_resource.tl.tag == NFAPI_PHICH_CONFIG_PHICH_RESOURCE_TAG)
  {
    fp->phich_config_common.phich_resource = req->phich_config.phich_resource.value;
    num_tlv++;
  }

  if (req->phich_config.phich_duration.tl.tag == NFAPI_PHICH_CONFIG_PHICH_DURATION_TAG)
  {
    fp->phich_config_common.phich_duration = req->phich_config.phich_duration.value;
    num_tlv++;
  }

  if (req->phich_config.phich_power_offset.tl.tag == NFAPI_PHICH_CONFIG_PHICH_POWER_OFFSET_TAG)
  {
    LOG_E(PHY, "%s() NFAPI_PHICH_CONFIG_PHICH_POWER_OFFSET_TAG tag:%d not supported\n", __FUNCTION__, req->phich_config.phich_power_offset.tl.tag);
    //fp->phich_config_common.phich_power_offset = req->phich_config.
    num_tlv++;
  }

  // UL RS Config
  if (req->uplink_reference_signal_config.cyclic_shift_1_for_drms.tl.tag == NFAPI_UPLINK_REFERENCE_SIGNAL_CONFIG_CYCLIC_SHIFT_1_FOR_DRMS_TAG)
  {
    fp->pusch_config_common.ul_ReferenceSignalsPUSCH.cyclicShift = req->uplink_reference_signal_config.cyclic_shift_1_for_drms.value;
    num_tlv++;
  }

  if (req->uplink_reference_signal_config.uplink_rs_hopping.tl.tag == NFAPI_UPLINK_REFERENCE_SIGNAL_CONFIG_UPLINK_RS_HOPPING_TAG)
  {
    fp->pusch_config_common.ul_ReferenceSignalsPUSCH.groupHoppingEnabled = req->uplink_reference_signal_config.uplink_rs_hopping.value;
    num_tlv++;
  }

  if (req->uplink_reference_signal_config.group_assignment.tl.tag == NFAPI_UPLINK_REFERENCE_SIGNAL_CONFIG_GROUP_ASSIGNMENT_TAG)
  {
    fp->pusch_config_common.ul_ReferenceSignalsPUSCH.groupAssignmentPUSCH = req->uplink_reference_signal_config.group_assignment.value;
    num_tlv++;
  }

  if (req->pusch_config.hopping_mode.tl.tag == NFAPI_PUSCH_CONFIG_HOPPING_MODE_TAG) { }  // DJP - not being handled?

  fp->pusch_config_common.ul_ReferenceSignalsPUSCH.sequenceHoppingEnabled = 0; // DJP - not being handled

  if (req->prach_config.configuration_index.tl.tag == NFAPI_PRACH_CONFIG_CONFIGURATION_INDEX_TAG)
  {
    fp->prach_config_common.prach_ConfigInfo.prach_ConfigIndex=req->prach_config.configuration_index.value;
    num_tlv++;
  }

  if (req->prach_config.root_sequence_index.tl.tag == NFAPI_PRACH_CONFIG_ROOT_SEQUENCE_INDEX_TAG)
  {
    fp->prach_config_common.rootSequenceIndex=req->prach_config.root_sequence_index.value;
    num_tlv++;
  }

  if (req->prach_config.zero_correlation_zone_configuration.tl.tag == NFAPI_PRACH_CONFIG_ZERO_CORRELATION_ZONE_CONFIGURATION_TAG)
  {
    fp->prach_config_common.prach_ConfigInfo.zeroCorrelationZoneConfig=req->prach_config.zero_correlation_zone_configuration.value;
    num_tlv++;
  }

  if (req->prach_config.high_speed_flag.tl.tag == NFAPI_PRACH_CONFIG_HIGH_SPEED_FLAG_TAG)
  {
    fp->prach_config_common.prach_ConfigInfo.highSpeedFlag=req->prach_config.high_speed_flag.value;
    num_tlv++;
  }

  if (req->prach_config.frequency_offset.tl.tag == NFAPI_PRACH_CONFIG_FREQUENCY_OFFSET_TAG)
  {
    fp->prach_config_common.prach_ConfigInfo.prach_FreqOffset=req->prach_config.frequency_offset.value;
    num_tlv++;
  }

  printf("[PNF] CONFIG_REQUEST[num_tlv:%d] TLVs processed:%d\n", req->num_tlv, num_tlv);

  printf("[PNF] Simulating PHY CONFIG - DJP\n");
  PHY_Config_t phy_config;
  phy_config.Mod_id = 0;
  phy_config.CC_id=0;
  phy_config.cfg = req;

  phy_config_request(&phy_config);

  dump_frame_parms(fp);

  phy_info->remote_port = req->nfapi_config.p7_vnf_port.value;

  struct sockaddr_in vnf_p7_sockaddr;
  memcpy(&vnf_p7_sockaddr.sin_addr.s_addr, &(req->nfapi_config.p7_vnf_address_ipv4.address[0]), 4);
  phy_info->remote_addr = inet_ntoa(vnf_p7_sockaddr.sin_addr);

  printf("[PNF] %d vnf p7 %s:%d timing %d %d %d\n", phy_info->id, phy_info->remote_addr, phy_info->remote_port, 
      phy_info->timing_window, phy_info->timing_info_mode, phy_info->timing_info_period);

  nfapi_config_response_t nfapi_resp;
  memset(&nfapi_resp, 0, sizeof(nfapi_resp));
  nfapi_resp.header.message_id = NFAPI_CONFIG_RESPONSE;
  nfapi_resp.header.phy_id = phy_info->id;
  nfapi_resp.error_code = 0; // DJP - some value resp->error_code;
  nfapi_pnf_config_resp(config, &nfapi_resp);
  printf("[PNF] Sent NFAPI_CONFIG_RESPONSE phy_id:%d\n", phy_info->id);

  return 0;
}

nfapi_p7_message_header_t* pnf_phy_allocate_p7_vendor_ext(uint16_t message_id, uint16_t* msg_size)
{
  if(message_id == P7_VENDOR_EXT_REQ)
  {
    (*msg_size) = sizeof(vendor_ext_p7_req);
    return (nfapi_p7_message_header_t*)malloc(sizeof(vendor_ext_p7_req));
  }

  return 0;
}

void pnf_phy_deallocate_p7_vendor_ext(nfapi_p7_message_header_t* header)
{
  free(header);
}

int pnf_phy_ul_config_req(nfapi_pnf_p7_config_t* pnf_p7, nfapi_ul_config_request_t* req)
{
  //printf("[PNF] ul config request\n");
  //phy_info* phy = (phy_info*)(pnf_p7->user_data);

  return 0;
}

int pnf_phy_hi_dci0_req(nfapi_pnf_p7_config_t* pnf_p7, nfapi_hi_dci0_request_t* req)
{
  //printf("[PNF] hi dci0 request\n");
  //phy_info* phy = (phy_info*)(pnf_p7->user_data);

  return 0;
}

nfapi_dl_config_request_pdu_t* dlsch_pdu[1023][10];

int pnf_phy_dl_config_req(nfapi_pnf_p7_config_t* pnf_p7, nfapi_dl_config_request_t* req)
{
#if 1
  if (NFAPI_SFNSF2SF(req->sfn_sf)==5)
    printf("[PNF] dl config request sfn_sf:%d(%d) pdcch:%u dci:%u pdu:%d pdsch_rnti:%d pcfich:%u\n", 
        req->sfn_sf, 
        NFAPI_SFNSF2DEC(req->sfn_sf), 
        req->dl_config_request_body.number_pdcch_ofdm_symbols, 
        req->dl_config_request_body.number_dci,
        req->dl_config_request_body.number_pdu,
        req->dl_config_request_body.number_pdsch_rnti,
        req->dl_config_request_body.transmission_power_pcfich
        );
#endif

  if (RC.ru == 0)
  {
    return -1;
  }

  if (RC.eNB == 0)
  {
    return -2;
  }

  if (RC.eNB[0][0] == 0)
  {
    return -3;
  }

  if (RC.eNB[0][0] == 0)
  {
    return -3;
  }

  int sfn = NFAPI_SFNSF2SFN(req->sfn_sf);
  int sf = NFAPI_SFNSF2SF(req->sfn_sf);

  struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];
  eNB_rxtx_proc_t *proc = &eNB->proc.proc_rxtx[0];
  nfapi_dl_config_request_pdu_t* dl_config_pdu_list = req->dl_config_request_body.dl_config_pdu_list;
  int total_number_of_pdus = req->dl_config_request_body.number_pdu;

  eNB->pdcch_vars[sf&1].num_pdcch_symbols = req->dl_config_request_body.number_pdcch_ofdm_symbols;
  eNB->pdcch_vars[sf&1].num_dci = 0;

  NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() sfn_sf:%d DCI:%d PDU:%d\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), req->dl_config_request_body.number_dci, req->dl_config_request_body.number_pdu);

  // DJP - force proc to look like current frame!
  proc->frame_tx = NFAPI_SFNSF2SFN(req->sfn_sf);
  proc->subframe_tx = NFAPI_SFNSF2SF(req->sfn_sf);

  for (int i=0;i<total_number_of_pdus;i++)
  {
    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() PDU[%d]:\n", __FUNCTION__, i);

    if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_DCI_DL_PDU_TYPE)
    {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() DCI:\n", __FUNCTION__);

      handle_nfapi_dci_dl_pdu(eNB,proc,&dl_config_pdu_list[i]);

      eNB->pdcch_vars[sf&1].num_dci++; // Is actually number of DCI PDUs
    }
    else if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_BCH_PDU_TYPE)
    {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() BCH:\n", __FUNCTION__);
    }
    else if (dl_config_pdu_list[i].pdu_type == NFAPI_DL_CONFIG_DLSCH_PDU_TYPE)
    {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() DLSCH:\n", __FUNCTION__);


      dlsch_pdu[sfn][sf] = &dl_config_pdu_list[i];
    }
    else
    {
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s() UNKNOWN:%d\n", __FUNCTION__, dl_config_pdu_list[i].pdu_type);
    }
  }

  if(req->vendor_extension)
    free(req->vendor_extension);

  return 0;
}


void common_signal_procedures (PHY_VARS_eNB *eNB,int frame, int subframe);
void pdsch_procedures(PHY_VARS_eNB *eNB,
		      eNB_rxtx_proc_t *proc,
		      int harq_pid,
		      LTE_eNB_DLSCH_t *dlsch, 
		      LTE_eNB_DLSCH_t *dlsch1,
		      LTE_eNB_UE_stats *ue_stats,
		      int ra_flag);

//int  __attribute__((optimize("O0"))) pnf_phy_tx_req(nfapi_pnf_p7_config_t* pnf_p7, nfapi_tx_request_t* req)
int  pnf_phy_tx_req(nfapi_pnf_p7_config_t* pnf_p7, nfapi_tx_request_t* req)
{
  if (RC.ru == 0)
  {
    return -1;
  }

  if (RC.eNB == 0)
  {
    return -2;
  }

  if (RC.eNB[0][0] == 0)
  {
    return -3;
  }

  if (RC.eNB[0][0] == 0)
  {
    return -3;
  }

  {
    uint16_t sfn = NFAPI_SFNSF2SFN(req->sfn_sf);
    uint16_t sf = NFAPI_SFNSF2SF(req->sfn_sf);
    LTE_DL_FRAME_PARMS *fp = &RC.ru[0]->frame_parms;
    int ONE_SUBFRAME_OF_SAMPLES = fp->ofdm_symbol_size*fp->symbols_per_tti;
    //int ONE_SUBFRAME_OF_SAMPLES = fp->symbols_per_tti;
    //int ONE_SUBFRAME_OF_SAMPLES = fp->ofdm_symbol_size*fp->symbols_per_tti*sizeof(int32_t);
    int offset = sf * ONE_SUBFRAME_OF_SAMPLES;
    struct PHY_VARS_eNB_s *eNB = RC.eNB[0][0];

    //DJP - the proc does not seem to be getting filled - so let fill it

    eNB->proc.proc_rxtx[0].frame_tx = sfn;
    eNB->proc.proc_rxtx[0].subframe_tx = sf;

    // clear the transmit data array for the current subframe
    for (int aa=0; aa<fp->nb_antenna_ports_eNB; aa++) {      
      memset(&eNB->common_vars.txdataF[aa][offset], 0, ONE_SUBFRAME_OF_SAMPLES * sizeof(int32_t));
    }

    // clear previous allocation information for all UEs
    for (int i=0; i<NUMBER_OF_UE_MAX; i++) {
      if (eNB->dlsch[i][0])
        eNB->dlsch[i][0]->subframe_tx[sf] = 0;
    }

    NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() sfn_sf:%u pdus:%u\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), req->tx_request_body.number_of_pdus);

    for(int i = 0; i < req->tx_request_body.number_of_pdus; ++i)
    {
      // DJP - TODO FIXME - work out if BCH (common_var)s or DLSCH (common.txdata)

      for(int j=0; j < req->tx_request_body.tx_pdu_list[i].num_segments; ++j)
      {
        NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() sfn_sf:%u pdu[%d] segment:%u segment_length:%u\n", __FUNCTION__, NFAPI_SFNSF2DEC(req->sfn_sf), i, j, req->tx_request_body.tx_pdu_list[i].segments[j].segment_length);

        // DJP - hack - assume tx_req segment of length 3 = bch
        if (req->tx_request_body.tx_pdu_list[i].segments[0].segment_length == 3)
        {
          eNB->pbch_pdu[2] = req->tx_request_body.tx_pdu_list[i].segments[j].segment_data[0];
          eNB->pbch_pdu[1] = req->tx_request_body.tx_pdu_list[i].segments[j].segment_data[1];
          eNB->pbch_pdu[0] = req->tx_request_body.tx_pdu_list[i].segments[j].segment_data[2];

          eNB->pbch_configured=1;

          if (
              1 
              //&& NFAPI_SFNSF2DEC(req->sfn_sf) % 500 == 0
             )
            NFAPI_TRACE(NFAPI_TRACE_INFO, "%s() [PDU:%u] len:%u pdu_index:%u num_segments:%u segment[0]_length:%u pbch_pdu:%x %x %x\n", 
                __FUNCTION__, i, req->tx_request_body.tx_pdu_list[i].pdu_length, req->tx_request_body.tx_pdu_list[i].pdu_index, req->tx_request_body.tx_pdu_list[i].num_segments,
                req->tx_request_body.tx_pdu_list[i].segments[0].segment_length,
                eNB->pbch_pdu[0],
                eNB->pbch_pdu[1],
                eNB->pbch_pdu[2]);

        }
        else
        {
          // Not bch
          handle_nfapi_dlsch_pdu(
              eNB,
              &eNB->proc.proc_rxtx[0],
              dlsch_pdu[sfn][sf], 
              dlsch_pdu[sfn][sf]->dlsch_pdu.dlsch_pdu_rel8.transport_blocks-1, 
              req->tx_request_body.tx_pdu_list[dlsch_pdu[sfn][sf]->dlsch_pdu.dlsch_pdu_rel8.pdu_index].segments[0].segment_data
              );
        }
      }
    }

    common_signal_procedures(eNB, sfn, sf);

    if (eNB->pdcch_vars[sf&1].num_dci > 0)
    {
      LOG_E(PHY,"SFN/SF:%d/%d eNB->pdcch_vars[sf&1].num_dci:%d num_pdcch_symbols:%d\n", sfn, sf, eNB->pdcch_vars[sf&1].num_dci, eNB->pdcch_vars[sf&1].num_pdcch_symbols);
    }

    generate_dci_top(
        eNB->pdcch_vars[sf&1].num_pdcch_symbols,
        eNB->pdcch_vars[sf&1].num_dci,
        &eNB->pdcch_vars[sf&1].dci_alloc[0],
        0,
        AMP,
        fp,
        eNB->common_vars.txdataF,
        sf);

#if 1
    // Now scan UE specific DLSCH
    for (int UE_id=0; UE_id<NUMBER_OF_UE_MAX; UE_id++)
    {
      LTE_eNB_DLSCH_t *dlsch0 = eNB->dlsch[(uint8_t)UE_id][0]; 
      LTE_eNB_DLSCH_t *dlsch1 = eNB->dlsch[(uint8_t)UE_id][1]; 

      if ((dlsch0)&&
          (dlsch0->rnti>0) &&
          (dlsch0->active == 1)) {

        // get harq_pid
        uint8_t harq_pid = dlsch0->harq_ids[sf];
        AssertFatal(harq_pid>=0,"harq_pid is negative\n");
        // generate pdsch
        LOG_E(PHY,"PDSCH active %d/%d\n", sfn,sf);
        pdsch_procedures(eNB,
            &eNB->proc.proc_rxtx[0],
            harq_pid,
            dlsch0,
            dlsch1,
            &eNB->UE_stats[(uint32_t)UE_id],
            0);
      }

      else if ((dlsch0)&&
          (dlsch0->rnti>0)&&
          (dlsch0->active == 0)) {

        // clear subframe TX flag since UE is not scheduled for PDSCH in this subframe (so that we don't look for PUCCH later)
        dlsch0->subframe_tx[sf]=0;
      }
    }
#endif

    if (0 && NFAPI_SFNSF2DEC(req->sfn_sf) % 500 == 0)
    {
      int32_t *txdataF = eNB->common_vars.txdataF[0];

      char *buf = malloc(fp->ofdm_symbol_size * fp->symbols_per_tti * 3);
      char *pbuf = buf;

      for (int i=0;i<10;i++)
      {
        buf[0]='\0';
        pbuf = buf;

        for (int j=0;j<fp->ofdm_symbol_size;j++)
        {
          for (int k=0;k<fp->symbols_per_tti;k++)
          {
            pbuf += sprintf(pbuf, "%2x ", txdataF[(i*fp->symbols_per_tti)+j]);
          }
        }
        NFAPI_TRACE(NFAPI_TRACE_INFO, "%s", buf);

      }
      free(buf);
    }
  }

  return 0;
}

int pnf_phy_lbt_dl_config_req(nfapi_pnf_p7_config_t* config, nfapi_lbt_dl_config_request_t* req)
{
  //printf("[PNF] lbt dl config request\n");
  return 0;
}

int pnf_phy_vendor_ext(nfapi_pnf_p7_config_t* config, nfapi_p7_message_header_t* msg)
{
  if(msg->message_id == P7_VENDOR_EXT_REQ)
  {
    //vendor_ext_p7_req* req = (vendor_ext_p7_req*)msg;
    //printf("[PNF] vendor request (1:%d 2:%d)\n", req->dummy1, req->dummy2);
  }
  else
  {
    printf("[PNF] unknown vendor ext\n");
  }
  return 0;
}

int pnf_phy_pack_p7_vendor_extension(nfapi_p7_message_header_t* header, uint8_t** ppWritePackedMsg, uint8_t *end, nfapi_p7_codec_config_t* codex)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P7_VENDOR_EXT_IND)
  {
    vendor_ext_p7_ind* ind = (vendor_ext_p7_ind*)(header);
    if(!push16(ind->error_code, ppWritePackedMsg, end))
      return 0;

    return 1;
  }
  return -1;
}

int pnf_phy_unpack_p7_vendor_extension(nfapi_p7_message_header_t* header, uint8_t** ppReadPackedMessage, uint8_t *end, nfapi_p7_codec_config_t* codec)
{
  if(header->message_id == P7_VENDOR_EXT_REQ)
  {
    //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
    vendor_ext_p7_req* req = (vendor_ext_p7_req*)(header);
    if(!(pull16(ppReadPackedMessage, &req->dummy1, end) &&
          pull16(ppReadPackedMessage, &req->dummy2, end)))
      return 0;
    return 1;
  }
  return -1;
}

int pnf_phy_unpack_vendor_extension_tlv(nfapi_tl_t* tl, uint8_t **ppReadPackedMessage, uint8_t* end, void** ve, nfapi_p7_codec_config_t* config)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "pnf_phy_unpack_vendor_extension_tlv\n");

  switch(tl->tag)
  {
    case VENDOR_EXT_TLV_1_TAG:
      *ve = malloc(sizeof(vendor_ext_tlv_1));
      if(!pull32(ppReadPackedMessage, &((vendor_ext_tlv_1*)(*ve))->dummy, end))
	return 0;

      return 1;
      break;
  }

  return -1;
}

int pnf_phy_pack_vendor_extention_tlv(void* ve, uint8_t **ppWritePackedMsg, uint8_t* end, nfapi_p7_codec_config_t* config)
{
  //printf("%s\n", __FUNCTION__);
  (void)ve;
  (void)ppWritePackedMsg;
  return -1;
}

int pnf_sim_unpack_vendor_extension_tlv(nfapi_tl_t* tl, uint8_t **ppReadPackedMessage, uint8_t *end, void** ve, nfapi_p4_p5_codec_config_t* config)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "pnf_sim_unpack_vendor_extension_tlv\n");

  switch(tl->tag)
  {
    case VENDOR_EXT_TLV_2_TAG:
      *ve = malloc(sizeof(vendor_ext_tlv_2));
      if(!pull32(ppReadPackedMessage, &((vendor_ext_tlv_2*)(*ve))->dummy, end))
	return 0;

      return 1;
      break;
  }

  return -1;
}

int pnf_sim_pack_vendor_extention_tlv(void* ve, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t* config)
{
  //printf("%s\n", __FUNCTION__);
  (void)ve;
  (void)ppWritePackedMsg;
  return -1;
}

int start_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_start_request_t* req)
{
  printf("[PNF] Received NFAPI_START_REQ phy_id:%d\n", req->header.phy_id);

  pnf_info* pnf = (pnf_info*)(config->user_data);

#if 0
  //DJP
  auto found = std::find_if(pnf->phys.begin(), pnf->phys.end(), [&](phy_info& item)
      { return item.id == req->header.phy_id; });

  if(found != pnf->phys.end())
#endif
  {
    //DJP phy_info& phy_info = (*found);
    phy_info* phy_info = pnf->phys;

    nfapi_pnf_p7_config_t* p7_config = nfapi_pnf_p7_config_create();

    p7_config->phy_id = phy->phy_id;

    p7_config->remote_p7_port = phy_info->remote_port;
    p7_config->remote_p7_addr = phy_info->remote_addr;
    p7_config->local_p7_port = 32123; // DJP - good grief cannot seem to get the right answer phy_info->local_port;
    //DJP p7_config->local_p7_addr = (char*)phy_info->local_addr.c_str();
    p7_config->local_p7_addr = phy_info->local_addr;

    printf("[PNF] P7 remote:%s:%d local:%s:%d\n", p7_config->remote_p7_addr, p7_config->remote_p7_port, p7_config->local_p7_addr, p7_config->local_p7_port);

    p7_config->user_data = phy_info;

    p7_config->malloc = &pnf_allocate;
    p7_config->free = &pnf_deallocate;
    p7_config->codec_config.allocate = &pnf_allocate;
    p7_config->codec_config.deallocate = &pnf_deallocate;

    p7_config->trace = &pnf_sim_trace;

    phy->user_data = p7_config;

    p7_config->subframe_buffer_size = phy_info->timing_window;
    if(phy_info->timing_info_mode & 0x1)
    {
      p7_config->timing_info_mode_periodic = 1;
      p7_config->timing_info_period = phy_info->timing_info_period;
    }

    if(phy_info->timing_info_mode & 0x2)
    {
      p7_config->timing_info_mode_aperiodic = 1;
    }

    p7_config->dl_config_req = &pnf_phy_dl_config_req;
    p7_config->ul_config_req = &pnf_phy_ul_config_req;
    p7_config->hi_dci0_req = &pnf_phy_hi_dci0_req;
    p7_config->tx_req = &pnf_phy_tx_req;
    p7_config->lbt_dl_config_req = &pnf_phy_lbt_dl_config_req;

    p7_config->vendor_ext = &pnf_phy_vendor_ext;

    p7_config->allocate_p7_vendor_ext = &pnf_phy_allocate_p7_vendor_ext;
    p7_config->deallocate_p7_vendor_ext = &pnf_phy_deallocate_p7_vendor_ext;

    p7_config->codec_config.unpack_p7_vendor_extension = &pnf_phy_unpack_p7_vendor_extension;
    p7_config->codec_config.pack_p7_vendor_extension = &pnf_phy_pack_p7_vendor_extension;
    p7_config->codec_config.unpack_vendor_extension_tlv = &pnf_phy_unpack_vendor_extension_tlv;
    p7_config->codec_config.pack_vendor_extension_tlv = &pnf_phy_pack_vendor_extention_tlv;

    NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating P7 thread %s\n", __FUNCTION__);
    pthread_t p7_thread;
    pthread_create(&p7_thread, NULL, &pnf_p7_thread_start, p7_config);

    //((pnf_phy_user_data_t*)(phy_info->fapi->user_data))->p7_config = p7_config;

    NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Calling l1_north_init_eNB() %s\n", __FUNCTION__);
    l1_north_init_eNB();

    NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] DJP - HACK - Set p7_config global ready for subframe ind%s\n", __FUNCTION__);
    p7_config_g = p7_config;

    // DJP - INIT PHY RELATED STUFF - this should be separate i think but is not currently...
    // Taken mostly from init_eNB_afterRU() dont think i can call it though...
    {
      printf("[PNF] %s() Calling phy_init_lte_eNB() and setting nb_antennas_rx = 1\n", __FUNCTION__);
      printf("[PNF] %s() TBD create frame_parms from NFAPI message\n", __FUNCTION__);

      phy_init_lte_eNB(RC.eNB[0][0],0,0);
      //RC.eNB[0][0]->frame_parms.nb_antennas_rx = 1;
      for (int ce_level=0;ce_level<4;ce_level++)
	RC.eNB[0][0]->prach_vars.rxsigF[ce_level] = (int16_t**)malloc16(64*sizeof(int16_t*));
#ifdef Rel14
      for (int ce_level=0;ce_level<4;ce_level++)
	RC.eNB[0][0]->prach_vars_br.rxsigF[ce_level] = (int16_t**)malloc16(64*sizeof(int16_t*));
#endif
      init_transport(RC.eNB[0][0]);
      //DJP - this crashes because RC.nb_RU is 1 but RC.ru[0] is NULL - init_precoding_weights(RC.eNB[0][0]);

      printf("[PNF] Calling mac_top_init_eNB() so that RC.mac[] is init\n");
      mac_top_init_eNB();
    }

    while(sync_var<0)
    {
      usleep(5000000);
      printf("[PNF] waiting for OAI to be started\n");
    }

    printf("[PNF] RC.nb_inst=1 DJP - this is because phy_init_RU() uses that to index and not RC.num_eNB - why the 2 similar variables?\n");
    RC.nb_inst =1; // DJP - fepc_tx uses num_eNB but phy_init_RU uses nb_inst
    printf("[PNF] About to call phy_init_RU()\n");
    phy_init_RU(RC.ru[0]);

    printf("[PNF] Sending PNF_START_RESP\n");
    nfapi_send_pnf_start_resp(config, p7_config->phy_id);

    printf("[PNF] Sending first P7 subframe ind\n");
    nfapi_pnf_p7_subframe_ind(p7_config, p7_config->phy_id, 0); // DJP - SFN_SF set to zero - correct???
  }

  return 0;
}

int measurement_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_measurement_request_t* req)
{
  nfapi_measurement_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_MEASUREMENT_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_MSG_OK;
  nfapi_pnf_measurement_resp(config, &resp);
  return 0;
}

int rssi_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_rssi_request_t* req)
{
  nfapi_rssi_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_RSSI_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_rssi_resp(config, &resp);

  nfapi_rssi_indication_t ind;
  memset(&ind, 0, sizeof(ind));
  ind.header.message_id = NFAPI_RSSI_INDICATION;
  ind.header.phy_id = req->header.phy_id;
  ind.error_code = NFAPI_P4_MSG_OK;
  ind.rssi_indication_body.tl.tag = NFAPI_RSSI_INDICATION_TAG;
  ind.rssi_indication_body.number_of_rssi = 1;
  ind.rssi_indication_body.rssi[0] = -42;
  nfapi_pnf_rssi_ind(config, &ind);
  return 0;
}

int cell_search_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_cell_search_request_t* req)
{
  nfapi_cell_search_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_CELL_SEARCH_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_cell_search_resp(config, &resp);

  nfapi_cell_search_indication_t ind;
  memset(&ind, 0, sizeof(ind));
  ind.header.message_id = NFAPI_CELL_SEARCH_INDICATION;
  ind.header.phy_id = req->header.phy_id;
  ind.error_code = NFAPI_P4_MSG_OK;

  switch(req->rat_type)
  {
    case NFAPI_RAT_TYPE_LTE:
      {
        ind.lte_cell_search_indication.tl.tag = NFAPI_LTE_CELL_SEARCH_INDICATION_TAG;
        ind.lte_cell_search_indication.number_of_lte_cells_found = 1;
        ind.lte_cell_search_indication.lte_found_cells[0].pci = 123;
        ind.lte_cell_search_indication.lte_found_cells[0].rsrp = 123;
        ind.lte_cell_search_indication.lte_found_cells[0].rsrq = 123;
        ind.lte_cell_search_indication.lte_found_cells[0].frequency_offset = 123;
      }
      break;
    case NFAPI_RAT_TYPE_UTRAN:
      {
        ind.utran_cell_search_indication.tl.tag = NFAPI_UTRAN_CELL_SEARCH_INDICATION_TAG;
        ind.utran_cell_search_indication.number_of_utran_cells_found = 1;
        ind.utran_cell_search_indication.utran_found_cells[0].psc = 89;
        ind.utran_cell_search_indication.utran_found_cells[0].rscp = 89;
        ind.utran_cell_search_indication.utran_found_cells[0].ecno = 89;
        ind.utran_cell_search_indication.utran_found_cells[0].frequency_offset = -89;

      }
      break;
    case NFAPI_RAT_TYPE_GERAN:
      {
        ind.geran_cell_search_indication.tl.tag = NFAPI_GERAN_CELL_SEARCH_INDICATION_TAG;
        ind.geran_cell_search_indication.number_of_gsm_cells_found = 1;
        ind.geran_cell_search_indication.gsm_found_cells[0].bsic = 23;
        ind.geran_cell_search_indication.gsm_found_cells[0].rxlev = 23;
        ind.geran_cell_search_indication.gsm_found_cells[0].rxqual = 23;
        ind.geran_cell_search_indication.gsm_found_cells[0].frequency_offset = -23;
        ind.geran_cell_search_indication.gsm_found_cells[0].sfn_offset = 230;

      }
      break;
  }

  ind.pnf_cell_search_state.tl.tag = NFAPI_PNF_CELL_SEARCH_STATE_TAG;
  ind.pnf_cell_search_state.length = 3;

  nfapi_pnf_cell_search_ind(config, &ind);	

  return 0;
}

int broadcast_detect_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_broadcast_detect_request_t* req)
{
  nfapi_broadcast_detect_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_BROADCAST_DETECT_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_broadcast_detect_resp(config, &resp);

  nfapi_broadcast_detect_indication_t ind;
  memset(&ind, 0, sizeof(ind));
  ind.header.message_id = NFAPI_BROADCAST_DETECT_INDICATION;
  ind.header.phy_id = req->header.phy_id;
  ind.error_code = NFAPI_P4_MSG_OK;

  switch(req->rat_type)
  {
    case NFAPI_RAT_TYPE_LTE:
      {
        ind.lte_broadcast_detect_indication.tl.tag = NFAPI_LTE_BROADCAST_DETECT_INDICATION_TAG;
        ind.lte_broadcast_detect_indication.number_of_tx_antenna = 1;
        ind.lte_broadcast_detect_indication.mib_length = 4;
        //ind.lte_broadcast_detect_indication.mib...
        ind.lte_broadcast_detect_indication.sfn_offset = 77;

      }
      break;
    case NFAPI_RAT_TYPE_UTRAN:
      {
        ind.utran_broadcast_detect_indication.tl.tag = NFAPI_UTRAN_BROADCAST_DETECT_INDICATION_TAG;
        ind.utran_broadcast_detect_indication.mib_length = 4;
        //ind.utran_broadcast_detect_indication.mib...
        // ind.utran_broadcast_detect_indication.sfn_offset; DJP - nonsense line

      }
      break;
  }

  ind.pnf_cell_broadcast_state.tl.tag = NFAPI_PNF_CELL_BROADCAST_STATE_TAG;
  ind.pnf_cell_broadcast_state.length = 3;

  nfapi_pnf_broadcast_detect_ind(config, &ind);	

  return 0;
}

int system_information_schedule_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_system_information_schedule_request_t* req)
{
  nfapi_system_information_schedule_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_SYSTEM_INFORMATION_SCHEDULE_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_system_information_schedule_resp(config, &resp);

  nfapi_system_information_schedule_indication_t ind;
  memset(&ind, 0, sizeof(ind));
  ind.header.message_id = NFAPI_SYSTEM_INFORMATION_SCHEDULE_INDICATION;
  ind.header.phy_id = req->header.phy_id;
  ind.error_code = NFAPI_P4_MSG_OK;

  ind.lte_system_information_indication.tl.tag = NFAPI_LTE_SYSTEM_INFORMATION_INDICATION_TAG;
  ind.lte_system_information_indication.sib_type = 3;
  ind.lte_system_information_indication.sib_length = 5;
  //ind.lte_system_information_indication.sib...

  nfapi_pnf_system_information_schedule_ind(config, &ind);		

  return 0;
}

int system_information_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_system_information_request_t* req)
{
  nfapi_system_information_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_SYSTEM_INFORMATION_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_system_information_resp(config, &resp);

  nfapi_system_information_indication_t ind;
  memset(&ind, 0, sizeof(ind));
  ind.header.message_id = NFAPI_SYSTEM_INFORMATION_INDICATION;
  ind.header.phy_id = req->header.phy_id;
  ind.error_code = NFAPI_P4_MSG_OK;

  switch(req->rat_type)
  {
    case NFAPI_RAT_TYPE_LTE:
      {
        ind.lte_system_information_indication.tl.tag = NFAPI_LTE_SYSTEM_INFORMATION_INDICATION_TAG;
        ind.lte_system_information_indication.sib_type = 1;
        ind.lte_system_information_indication.sib_length = 3;
        //ind.lte_system_information_indication.sib...
      }
      break;
    case NFAPI_RAT_TYPE_UTRAN:
      {
        ind.utran_system_information_indication.tl.tag = NFAPI_UTRAN_SYSTEM_INFORMATION_INDICATION_TAG;
        ind.utran_system_information_indication.sib_length = 3;
        //ind.utran_system_information_indication.sib...

      }
      break;
    case NFAPI_RAT_TYPE_GERAN:
      {
        ind.geran_system_information_indication.tl.tag = NFAPI_GERAN_SYSTEM_INFORMATION_INDICATION_TAG;
        ind.geran_system_information_indication.si_length = 3;
        //ind.geran_system_information_indication.si...

      }
      break;
  }

  nfapi_pnf_system_information_ind(config, &ind);		

  return 0;
}

int nmm_stop_request(nfapi_pnf_config_t* config, nfapi_pnf_phy_config_t* phy, nfapi_nmm_stop_request_t* req)
{
  nfapi_nmm_stop_response_t resp;
  memset(&resp, 0, sizeof(resp));
  resp.header.message_id = NFAPI_NMM_STOP_RESPONSE;
  resp.header.phy_id = req->header.phy_id;
  resp.error_code = NFAPI_P4_MSG_OK;
  nfapi_pnf_nmm_stop_resp(config, &resp);
  return 0;
}

int vendor_ext(nfapi_pnf_config_t* config, nfapi_p4_p5_message_header_t* msg)
{
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] P5 %s %p\n", __FUNCTION__, msg);

  switch(msg->message_id)
  {
    case P5_VENDOR_EXT_REQ:
      {
        vendor_ext_p5_req* req = (vendor_ext_p5_req*)msg;
        NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] P5 Vendor Ext Req (%d %d)\n", req->dummy1, req->dummy2);
        // send back the P5_VENDOR_EXT_RSP
        vendor_ext_p5_rsp rsp;
        memset(&rsp, 0, sizeof(rsp));
        rsp.header.message_id = P5_VENDOR_EXT_RSP;
        rsp.error_code = NFAPI_MSG_OK;
        nfapi_pnf_vendor_extension(config, &rsp.header, sizeof(vendor_ext_p5_rsp));
      }
      break;
  }

  return 0;
}

nfapi_p4_p5_message_header_t* pnf_sim_allocate_p4_p5_vendor_ext(uint16_t message_id, uint16_t* msg_size)
{
  if(message_id == P5_VENDOR_EXT_REQ)
  {
    (*msg_size) = sizeof(vendor_ext_p5_req);
    return (nfapi_p4_p5_message_header_t*)malloc(sizeof(vendor_ext_p5_req));
  }

  return 0;
}

void pnf_sim_deallocate_p4_p5_vendor_ext(nfapi_p4_p5_message_header_t* header)
{
  free(header);
}

int pnf_sim_pack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t* header, uint8_t** ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t* config)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P5_VENDOR_EXT_RSP)
  {
    vendor_ext_p5_rsp* rsp = (vendor_ext_p5_rsp*)(header);
    return (!push16(rsp->error_code, ppWritePackedMsg, end));
  }
  return 0;
}

int pnf_sim_unpack_p4_p5_vendor_extension(nfapi_p4_p5_message_header_t* header, uint8_t** ppReadPackedMessage, uint8_t *end, nfapi_p4_p5_codec_config_t* codec)
{
  //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s\n", __FUNCTION__);
  if(header->message_id == P5_VENDOR_EXT_REQ)
  {
    vendor_ext_p5_req* req = (vendor_ext_p5_req*)(header);
    return (!(pull16(ppReadPackedMessage, &req->dummy1, end) &&
          pull16(ppReadPackedMessage, &req->dummy2, end)));

    //NFAPI_TRACE(NFAPI_TRACE_INFO, "%s (%d %d)\n", __FUNCTION__, req->dummy1, req->dummy2);
  }
  return 0;
}

/*------------------------------------------------------------------------------*/
static pnf_info pnf;
static pthread_t pnf_start_pthread;

void* pnf_start_thread(void* ptr)
{
  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] IN PNF NFAPI start thread %s\n", __FUNCTION__);

  nfapi_pnf_config_t *config = (nfapi_pnf_config_t*)ptr;

  nfapi_pnf_start(config);

  return (void*)0;
}

void configure_nfapi_pnf(char *vnf_ip_addr, int vnf_p5_port, char *pnf_ip_addr, int pnf_p7_port, int vnf_p7_port)
{
  nfapi_pnf_config_t* config = nfapi_pnf_config_create();

  config->vnf_ip_addr = vnf_ip_addr;
  config->vnf_p5_port = vnf_p5_port;

  pnf.phys[0].udp.enabled = 1;
  pnf.phys[0].udp.rx_port = pnf_p7_port;
  pnf.phys[0].udp.tx_port = vnf_p7_port;
  strcpy(pnf.phys[0].udp.tx_addr, vnf_ip_addr);

  strcpy(pnf.phys[0].local_addr, pnf_ip_addr);

  printf("%s() VNF:%s:%d PNF_PHY[addr:%s UDP:tx_addr:%s:%d rx:%d]\n", 
      __FUNCTION__, 
      config->vnf_ip_addr, config->vnf_p5_port, 
      pnf.phys[0].local_addr,
      pnf.phys[0].udp.tx_addr, pnf.phys[0].udp.tx_port,
      pnf.phys[0].udp.rx_port);

  config->pnf_param_req = &pnf_param_request;
  config->pnf_config_req = &pnf_config_request;
  config->pnf_start_req = &pnf_start_request;
  config->pnf_stop_req = &pnf_stop_request;
  config->param_req = &param_request;
  config->config_req = &config_request;
  config->start_req = &start_request;

  config->measurement_req = &measurement_request;
  config->rssi_req = &rssi_request;
  config->cell_search_req = &cell_search_request;
  config->broadcast_detect_req = &broadcast_detect_request;
  config->system_information_schedule_req = &system_information_schedule_request;
  config->system_information_req = &system_information_request;
  config->nmm_stop_req = &nmm_stop_request;

  config->vendor_ext = &vendor_ext;

  config->trace = &pnf_sim_trace;

  config->user_data = &pnf;

  // To allow custom vendor extentions to be added to nfapi
  config->codec_config.unpack_vendor_extension_tlv = &pnf_sim_unpack_vendor_extension_tlv;
  config->codec_config.pack_vendor_extension_tlv = &pnf_sim_pack_vendor_extention_tlv;

  config->allocate_p4_p5_vendor_ext = &pnf_sim_allocate_p4_p5_vendor_ext;
  config->deallocate_p4_p5_vendor_ext = &pnf_sim_deallocate_p4_p5_vendor_ext;

  config->codec_config.unpack_p4_p5_vendor_extension = &pnf_sim_unpack_p4_p5_vendor_extension;
  config->codec_config.pack_p4_p5_vendor_extension = &pnf_sim_pack_p4_p5_vendor_extension;

  NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] Creating PNF NFAPI start thread %s\n", __FUNCTION__);
  pthread_create(&pnf_start_pthread, NULL, &pnf_start_thread, config);
}

void oai_subframe_ind(uint16_t frame, uint16_t subframe)
{
  //TODO FIXME - HACK - DJP - using a global to bodge it in 

  if (p7_config_g != NULL && sync_var==0)
  {
    uint16_t sfn = subframe>=9?frame+1:frame;
    uint16_t sf = subframe>=9?0:subframe+1;
    uint16_t sfn_sf = sfn<<4 | sf;

    if ((frame % 100 == 0) && subframe==0)
    {
      struct timespec ts;

      clock_gettime(CLOCK_MONOTONIC, &ts);

      NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] %s %d.%d (frame:%u subframe:%u) sfn_sf:%u sfn_sf(DEC):%u\n", __FUNCTION__, ts.tv_sec, ts.tv_nsec, frame, subframe, sfn_sf, NFAPI_SFNSF2DEC(sfn_sf));
    }

    int subframe_ret = nfapi_pnf_p7_subframe_ind(p7_config_g, p7_config_g->phy_id, sfn_sf);

    if (subframe_ret)
    {
      NFAPI_TRACE(NFAPI_TRACE_INFO, "[PNF] %s(frame:%u subframe:%u) sfn_sf:%u sfn_sf(DEC):%u - PROBLEM with pnf_p7_subframe_ind()\n", __FUNCTION__, frame, subframe, sfn_sf, NFAPI_SFNSF2DEC(sfn_sf));
    }
  }
  else
  {
  }
}
