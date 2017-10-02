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

/*! \file main.c
 * \brief top init of Layer 2
 * \author  Navid Nikaein and Raymond Knopp, Michele Paffetti
 * \date 2010 - 2014
 * \version 1.0
 * \email: navid.nikaein@eurecom.fr, michele.paffetti@studio.unibo.it
 * @ingroup _mac

 */


//#include "asn1_constants.h"
#include "LAYER2/MAC/defs_NB_IoT.h"
#include "LAYER2/MAC/proto_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
#include "RRC/LITE/proto_NB_IoT.h"

int mac_init_global_param_NB_IoT(void)
{

  if (rlc_module_init()!=0) {
    return(-1);
 }

  LOG_I(MAC,"[MAIN] RRC NB-IoT initialization of global params\n");
  rrc_init_global_param_NB_IoT();


  LOG_I(MAC,"[MAIN] PDCP layer init\n");
#ifdef USER_MODE
  pdcp_layer_init ();
#else
  pdcp_module_init ();
#endif

  return 0;
}


// Initial function of the intialization for NB-IoT MAC
int mac_top_init_NB_IoT()
{

}


int l2_init_eNB_NB_IoT()
{

  LOG_I(MAC,"[MAIN] Mapping L2 IF-Module functions\n");
  IF_Module_init_L2();

  LOG_I(MAC,"[MAIN] MAC_INIT_GLOBAL_PARAM NB-IoT IN...\n");

  Is_rrc_registered_NB_IoT=0;
  mac_init_global_param_NB_IoT();
  Is_rrc_registered_NB_IoT=1;


  LOG_D(MAC,"[MAIN][NB-IoT] ALL INIT OK\n");

//    mac_xface->macphy_init(eMBMS_active,uecap_xer,cba_group_active,HO_active); (old mac_top_init)
  mac_top_init_NB_IoT();

  return(1);
}

