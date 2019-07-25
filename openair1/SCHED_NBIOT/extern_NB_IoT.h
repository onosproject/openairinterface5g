/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

/*!\brief SCHED external variables */

#ifndef __SCHED_EXTERN_NB_IOT_H__
#define __SCHED_EXTERN_NB_IOT_H__

#ifndef USER_MODE
#define __NO_VERSION__


#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/types.h>

#include <asm/io.h>
#include <asm/bitops.h>

#include <asm/uaccess.h>
#include <asm/segment.h>
#include <asm/page.h>


#ifdef RTAI_ENABLED
#include <rtai.h>
//#include <rtai_posix.h>
#include <rtai_fifos.h>
#include <rtai_sched.h>
#include <rtai_sem.h>
//#include "rt_compat.h"

#else
#include <unistd.h>
#endif

#endif  /* USER_MODE */




#include "defs_NB_IoT.h"



extern uint16_t hundred_times_log10_NPRB_NB_IoT[100];


#endif /*__SCHED_EXTERN_H__ */
