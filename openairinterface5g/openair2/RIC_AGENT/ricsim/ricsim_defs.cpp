/*
 * SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*****************************************************************************
#                                                                            *
# Copyright 2019 AT&T Intellectual Property                                  *
# Copyright 2019 Nokia                                                       *
#                                                                            *
# Licensed under the Apache License, Version 2.0 (the "License");            *
# you may not use this file except in compliance with the License.           *
# You may obtain a copy of the License at                                    *
#                                                                            *
#      http://www.apache.org/licenses/LICENSE-2.0                            *
#                                                                            *
# Unless required by applicable law or agreed to in writing, software        *
# distributed under the License is distributed on an "AS IS" BASIS,          *
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# See the License for the specific language governing permissions and        *
# limitations under the License.                                             *
#                                                                            *
******************************************************************************/

#include "ricsim_defs.h"
#include <getopt.h>
#include <sys/time.h>
#include <time.h>

char* time_stamp(void)
{
  timeval curTime;
  gettimeofday(&curTime, NULL);
  int milli = curTime.tv_usec / 1000;

  char buffer [80];
  strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

  const int time_buffer_len = 84;
  static char currentTime[time_buffer_len] = "";
  snprintf(currentTime, time_buffer_len, "%s:%03d", buffer, milli);

  return currentTime;
}

options_t read_input_options_old(int argc, char* argv[])
{
  options_t options;

  options.server_ip         = (char*)DEFAULT_SCTP_IP;
  options.server_port       = X2AP_SCTP_PORT;

  // Parse command line options
  static struct option long_options[] =
    {
      {"ipv4", required_argument, 0, 'i'},
      {"ipv6", required_argument, 0, 'I'},
      {"port", required_argument, 0, 'p'},
      {"verbose", no_argument, 0, 'v'},
    };

    while(1)
    {
      int option_index = 0;

      char c = getopt_long(argc, argv, "i:I:p:", long_options, &option_index);

      if(c == -1)
        break;

      switch(c)
      {
        case 'i':
          options.server_ip = optarg;
          break;
        case 'I':
          break;
        case 'p':
          options.server_port = atoi(optarg);
          if(options.server_port < 1 || options.server_port > 65535)
          {
            LOG_E("Invalid port number (%d). Valid values are between 1 and 65535.\n",
                                                              options.server_port);
            exit(1);
          }
          break;

        default:
          LOG_E("Error: unknown input option: %c\n", optopt);
          exit(1);
      }
    }

    return options;
}

options_t read_input_options(int argc, char *argv[])
{
  options_t options;

  options.server_ip         = (char*)DEFAULT_SCTP_IP;
  options.server_port       = X2AP_SCTP_PORT;

  if(argc == 3) //user provided IP and PORT
  {
    options.server_ip = argv[1];
    options.server_port = atoi(argv[2]);
    if(options.server_port < 1 || options.server_port > 65535) {
      LOG_E("Invalid port number (%d). Valid values are between 1 and 65535.\n",
                                  options.server_port);
      exit(1);
    }
  }
  else if(argc == 2) //user provided only IP
  {
    options.server_ip = argv[1];
  }
  else if(argc == 1)
  {
    options.server_ip = (char*)DEFAULT_SCTP_IP;
  }
  else
  {
    LOG_I("Unrecognized option.\n");
    LOG_I("Usage: %s [SERVER IP ADDRESS] [SERVER PORT]\n", argv[0]);
    exit(1);
  }

  return options;
}
