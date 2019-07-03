/* g++ bug-3.13.1.0.cc -luhd -lboost_system */
#include <uhd/usrp/multi_usrp.hpp>
#include <stdio.h>
#include <stdlib.h>

static uhd::usrp::multi_usrp::sptr usrp;
static uhd::rx_streamer::sptr rx_stream;
static uhd::tx_streamer::sptr tx_stream;
static uhd::rx_metadata_t rx_md;
static uhd::tx_metadata_t tx_md;

extern "C" {

#include "usrp.h"

void usrp_init_connection(uint64_t rx_freq, uint64_t tx_freq)
{
  std::string args = "type=b200";
  uhd::device_addrs_t device_adds = uhd::device::find(args);
  if (device_adds.size() == 0) { printf("no device\n"); exit(1); }
  if (device_adds[0].get("type") != "b200") { printf("no b200\n"); exit(1); }

  double usrp_master_clock = 30.72e6;
  args += boost::str(boost::format(",master_clock_rate=%f") % usrp_master_clock);
  //args += ",num_send_frames=256,num_recv_frames=256, send_frame_size=3840, recv_frame_size=3840";
  args += ",num_send_frames=256,num_recv_frames=256";

  usrp = uhd::usrp::multi_usrp::make(args);

  usrp->set_clock_source("internal");
  usrp->set_master_clock_rate(30.72e6);

  usrp->set_rx_rate(7680000, 0);
  usrp->set_rx_freq(rx_freq, 0);
  usrp->set_rx_gain(62.2, 0);

  usrp->set_tx_rate(7680000, 0);
  usrp->set_tx_freq(tx_freq, 0);
  usrp->set_tx_gain(89.75, 0);

  uhd::stream_args_t stream_args_rx("sc16", "sc16");
  stream_args_rx.args["spp"] = str(boost::format("%d") % 768 );
  stream_args_rx.channels.push_back(0);
  rx_stream = usrp->get_rx_stream(stream_args_rx);

  uhd::stream_args_t stream_args_tx("sc16", "sc16");
  stream_args_tx.channels.push_back(0);
  tx_stream = usrp->get_tx_stream(stream_args_tx);

  usrp->set_tx_bandwidth(20e6);
  usrp->set_rx_bandwidth(20e6);
}

void usrp_start(void)
{
  uhd::stream_cmd_t cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  cmd.stream_now = false; // start at constant delay
  cmd.time_spec = usrp->get_time_now() + uhd::time_spec_t(0.05);
  rx_stream->issue_stream_cmd(cmd);

  tx_md.start_of_burst = true;
  tx_md.end_of_burst = false;
}

uint64_t usrp_read(char *buf, int samples_count)
{
    std::vector<void *> buff_ptrs;
    buff_ptrs.push_back(buf);

    int recv = rx_stream->recv(buff_ptrs, samples_count, rx_md);
    return rx_md.time_spec.to_ticks(7680000);
}

void usrp_write(char *buf, int samples_count, uint64_t timestamp)
{
  tx_md.time_spec = uhd::time_spec_t::from_ticks(timestamp, 7680000);
  tx_md.has_time_spec = true;
  int sendv = tx_stream->send(buf, samples_count, tx_md, 10 /*1e-3*/);
  tx_md.start_of_burst = false;
}

#if 0
  while (1) {
    char buf[7680*2*2];
    std::vector<void *> buff_ptrs;
    buff_ptrs.push_back(buf);

    int recv = rx_stream->recv(buff_ptrs, 7680, rx_md);
    //printf("got %d samples ret %d [%lld]\n", recv, rx_md.error_code, rx_md.time_spec.to_ticks(7680000));

    unsigned long long ts = rx_md.time_spec.to_ticks(7680000) + 4 * 7680;

    tx_md.time_spec = uhd::time_spec_t::from_ticks(ts, 7680000);
    tx_md.has_time_spec = true;

    int sendv = tx_stream->send(buf, 7680, tx_md, 1e-3);
    //printf("send %d samples\n", sendv);

    tx_md.start_of_burst = false;
  }

  return 0;
}
#endif

} /* extern "C" */
