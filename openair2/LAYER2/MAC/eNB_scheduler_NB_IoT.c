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

/*! \file eNB_scheduler_nb_iot.c
 * \brief top level of the scheduler, it scheduled in pdcch period based.
 * \author  
 * \date 2017
 * \email: 
 * \version 0.5
 * @ingroup _mac
 *
 */





/*function description
* top level of the scheduler, this will trigger in every subframe, 
* and determined if do the schedule by checking this current subframe is the start of the NPDCCH period or not
*/


/* already defined in proto_NB_IoT.h
void eNB_dlsch_ulsch_scheduler_NB_IoT(module_id_t module_idP, frame_t frameP, sub_frame_t subframeP, uint16_t hypersfn)  
{

}

*/

