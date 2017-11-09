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

/*! \file PHY_INTERFACE/defs.h
* \brief mac phy interface primitives
* \author Raymond Knopp and Navid Nikaein
* \date 2011
* \version 0.5
* \mail navid.nikaein@eurecom.fr or openair_tech@eurecom.fr
*/

#ifndef __MAC_PHY_PRIMITIVES_NB_IOT_H__
#define __MAC_PHY_PRIMITIVES_NB_IOT_H__

#include "LAYER2/MAC/defs_NB_IoT.h"
//#include "LAYER2/MAC/defs.h"

#define MAX_NUMBER_OF_MAC_INSTANCES 16

#define NULL_PDU 255
#define DCI 0
#define DLSCH 1
#define ULSCH 2

/*
#define mac_exit_wrapper(sTRING)                                                            \
do {                                                                                        \
    char temp[300];                                                                         \
    snprintf(temp, sizeof(temp), "%s in file "__FILE__" at line %d\n", sTRING, __LINE__);   \
    mac_xface->macphy_exit(temp);                                                           \
} while(0)
*/
/** @defgroup _phy_if MAC-PHY interface
 * @ingroup _oai2
 * @{
 */



#endif


/** @} */
