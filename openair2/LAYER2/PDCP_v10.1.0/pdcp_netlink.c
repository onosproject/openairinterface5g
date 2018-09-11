/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

/*! \file pdcp_netlink.c
 * \brief pdcp communication with linux IP interface,
 * have a look at http://man7.org/linux/man-pages/man7/netlink.7.html for netlink.
 * Read operation from netlink should be achieved in an asynchronous way to avoid
 * subframe overload, netlink congestion...
 * \author Sebastien ROUX
 * \date 2013
 * \version 0.1
 * @ingroup pdcp
 */

#define _GNU_SOURCE // required for pthread_setname_np()
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <error.h>
#include <unistd.h>

/* Bugfix for version of GCC = 4.4.3 (Ubuntu 10.04) */
#if GCC_VERSION <= 40403
# include <sys/socket.h>
#endif

#include <linux/netlink.h>

#include "assertions.h"
#include "queue.h"
#include "liblfds611.h"

#include "common/utils/LOG/log.h"
#include "UTIL/OCG/OCG.h"
#include "UTIL/OCG/OCG_extern.h"
#include "LAYER2/MAC/mac_extern.h"

#include "pdcp.h"
#include "pdcp_primitives.h"
#include "msc.h"


#define PDCP_QUEUE_NB_ELEMENTS 200

extern char nl_rx_buf[NL_MAX_PAYLOAD];
extern struct nlmsghdr *nas_nlh_rx;
extern struct iovec nas_iov_rx;
extern int nas_sock_fd;
extern struct msghdr nas_msg_rx;

