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

/*!
 * \file   slicing.c
 * \brief  Generic slicing helper functions and Static Slicing Implementation
 * \author Robert Schmidt
 * \date   2020
 * \email  robert.schmidt@eurecom.fr
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <dlfcn.h>

#include "assertions.h"
#include "common/utils/LOG/log.h"

#include "slicing.h"
#include "slicing_internal.h"

#include "common/ran_context.h"
extern RAN_CONTEXT_t RC;

#define RET_FAIL(ret, x...) do { LOG_E(MAC, x); return ret; } while (0)

int slicing_get_UE_slice_idx(slice_info_t *si, int UE_id) {
  return si->UE_assoc_slice[UE_id];
}

void slicing_add_UE(slice_info_t *si, int UE_id) {
  LOG_W(MAC, "Adding UE %d to slice index %d\n", UE_id, 0);
  add_ue_list(&si->s[0]->UEs, UE_id);
  si->UE_assoc_slice[UE_id] = 0;
}

void _remove_UE(slice_t **s, uint8_t *assoc, int UE_id) {
  const uint8_t i = assoc[UE_id];
  DevAssert(remove_ue_list(&s[i]->UEs, UE_id));
  assoc[UE_id] = -1;
}

void slicing_remove_UE(slice_info_t *si, int UE_id) {
  _remove_UE(si->s, si->UE_assoc_slice, UE_id);
}

void _move_UE(slice_t **s, uint8_t *assoc, int UE_id, int to) 
{
  const uint8_t i = assoc[UE_id]; /*get source slice id */
  
  /*remove UE from the source slice id */
  const int ri = remove_ue_list(&s[i]->UEs, UE_id);
  if (!ri)
  {
    LOG_W(MAC, "did not find UE %d in DL slice index %d\n", UE_id, i);
  }
  else
  {
    /* check if it was the last UE that got removed from dedicated slice i */
    if ( (dump_ue_list(&s[i]->UEs) == 0) && (i > 0) )
    {
      /* Add the time_schd value of this dedicated slice back to default slice */
      ((static_slice_param_t *)s[0]->algo_data)->timeSchd += ((static_slice_param_t *)s[i]->algo_data)->timeSchd;
      LOG_I(MAC,"Last UEID:%d removed from sliceIdx:%d sliceId:%d, def slice timeschd:%d\n",
            UE_id, i, s[i]->id, ((static_slice_param_t *)s[0]->algo_data)->timeSchd);
    }
  }

  /*add UE to the target slice id */
  add_ue_list(&s[to]->UEs, UE_id);

  /* Check if ue_id is the first UE getting assoc to dedicated slice to (target) */
  if ( (dump_ue_list(&s[to]->UEs) == 1) && (to > 0) )
  {
    /* Reduce the time_schd value from default slice */
    ((static_slice_param_t *)s[0]->algo_data)->timeSchd -= ((static_slice_param_t *)s[to]->algo_data)->timeSchd;
    LOG_I(MAC,"First UEID:%d associated with sliceIdx:%d sliceId:%d, def slice timeschd:%d ded slice timeschd:%d\n",
          UE_id, to, s[to]->id, ((static_slice_param_t *)s[0]->algo_data)->timeSchd, ((static_slice_param_t *)s[to]->algo_data)->timeSchd);
  } 

  /*update UE:Slice association */
  assoc[UE_id] = to;
  LOG_I(MAC, "UEID:%d associated with SliceId:%d\n",UE_id, to);
}

void slicing_move_UE(slice_info_t *si, int UE_id, int idx) {
  DevAssert(idx >= -1 && idx < si->num);
  if (idx >= 0)
    _move_UE(si->s, si->UE_assoc_slice, UE_id, idx);
}

int _exists_slice(uint8_t n, slice_t **s, int id) {
  for (int i = 0; i < n; ++i)
    if (s[i]->id == id)
      return i;
  return -1;
}

slice_t *_add_slice(uint8_t *n, slice_t **s) {
  s[*n] = calloc(1, sizeof(slice_t));
  if (!s[*n])
    return NULL;
  init_ue_list(&s[*n]->UEs);
  *n += 1;
  LOG_I(FLEXRAN_AGENT, "[%s] n %d\n", __func__, *n);
  return s[*n - 1];
}

slice_t *_remove_slice(uint8_t *n, slice_t **s, uint8_t *assoc, int idx) {
  if (idx >= *n)
    return NULL;

  slice_t *sr = s[idx];
  while (sr->UEs.head >= 0)
    _move_UE(s, assoc, sr->UEs.head, 0);

  for (int i = idx + 1; i < *n; ++i)
    s[i - 1] = s[i];
  *n -= 1;
  s[*n] = NULL;

  for (int i = 0; i < MAX_MOBILES_PER_ENB; ++i)
    if (assoc[i] > idx)
      assoc[i] -= 1;

  if (sr->label)
    free(sr->label);

  LOG_I(MAC, "[%s] sliceIdx:%d removed \n",__func__, idx); 

  return sr;
}

/************************ Static Slicing Implementation ************************/

int addmod_static_slice_dl(slice_info_t *si,
                           int id,
                           char *label,
                           void *algo,
                           void *slice_params_dl) 
{
  static_slice_param_t *dl = slice_params_dl;
  //uint8_t rbgMap[25] = { 0 };
  uint16_t totalTimeSchd = 0;
  
  if (dl && dl->posLow > dl->posHigh)
    RET_FAIL(-1, "%s(): slice id %d posLow > posHigh\n", __func__, id);
  
  if ( (id == 0) && (dl->timeSchd > MAX_DEF_SLICE_TIME_SCHD))
    RET_FAIL(-1, "%s(): Default slice id:%d timeSchd:%d exceeds 100\n", __func__, id, dl->timeSchd);

  if ( (id > 0) && (dl->timeSchd > MAX_DED_SLICE_TIME_SCHD))
    RET_FAIL(-1, "%s(): Dedicated slice id:%d, timeSchd:%d exceeds 80\n", __func__, id, dl->timeSchd);
  
  int index = _exists_slice(si->num, si->s, id);
  LOG_I(MAC, "[%s]Enter index:%d si->num:%d id:%d\n", __func__, index, si->num,id);
  if (index >= 0) 
  {
    /* This part of code will only execute during default slice updation or 
       during dedicated slice parameter updation */
    for (int s = 0; s < si->num; ++s) 
    {
      static_slice_param_t *sd = dl && si->s[s]->id == id ? dl : si->s[s]->algo_data;
#if 0 /*RBG slice overlap check not required */
      for (int i = sd->posLow; i <= sd->posHigh; ++i) 
      {
        if (rbgMap[i])
          RET_FAIL(-33, "%s(): overlap of slices detected at RBG %d\n", __func__, i);
        
        rbgMap[i] = 1;
      }
#endif
      if (si->s[s]->id > 0)
      {
        totalTimeSchd += sd->timeSchd;
      }
    }

    if (totalTimeSchd > MAX_DED_SLICE_TIME_SCHD)
      RET_FAIL(-1, "%s(): Total Dedicated timeSchd:%d exceeds 80, cannot create new dedicated slice id:%d\n",
                __func__, totalTimeSchd, id);

    /* no problem, can allocate */
    slice_t *s = si->s[index];
    if (label) 
    {
      if (s->label) free(s->label);
      s->label = label;
    }
    if (algo) 
    {
      s->dl_algo.unset(&s->dl_algo.data);
      s->dl_algo = *(default_sched_dl_algo_t *) algo;
      if (!s->dl_algo.data)
        s->dl_algo.data = s->dl_algo.setup();
    }
    /* Check if time_schd of dedicated slice is getting reduced */
    if ( ( ((static_slice_param_t *)s->algo_data)->timeSchd > dl->timeSchd) &&
         (dump_ue_list(&s->UEs) > 0) )     
    {
      /* In that case timeSchd difference should be added back to default slice */
      ((static_slice_param_t *)si->s[0]->algo_data)->timeSchd +=
          ( ((static_slice_param_t *)s->algo_data)->timeSchd - dl->timeSchd);
      LOG_I(MAC, "adding back %d time_sch to def slice,updated def slice time_schd:%d\n",
            ( ((static_slice_param_t *)s->algo_data)->timeSchd - dl->timeSchd),
            ((static_slice_param_t *)si->s[0]->algo_data)->timeSchd);
    }

    if (dl) 
    {
      free(s->algo_data);
      s->algo_data = dl;
    }
    LOG_I(MAC, "Updated ded slice:%d time_schd:%d\n", s->id, ((static_slice_param_t *)s->algo_data)->timeSchd);
    LOG_D(MAC, "[%s]return index:%d si->num:%d id:%d\n", __func__, index, si->num,id);
    return index;
  }

  /* Below code is executed for creating Default or Dedicated slice */
  if (!dl)
    RET_FAIL(-100, "%s(): no parameters for new slice %d, aborting\n", __func__, id);

  if (si->num >= MAX_STATIC_SLICES)
    RET_FAIL(-2, "%s(): cannot have more than %d slices\n", __func__, MAX_STATIC_SLICES);

  if (si->num > 1)
  {
    /* Check that total timeSchd of dedicated slices should not exceed 80%*/
    for(int ded_slice_idx = 1; ded_slice_idx < si->num; ded_slice_idx++)
    {
      static_slice_param_t *dedSlice = si->s[ded_slice_idx]->algo_data;
      totalTimeSchd += dedSlice->timeSchd;
    }

    if ( (totalTimeSchd + dl->timeSchd) > MAX_DED_SLICE_TIME_SCHD )
      RET_FAIL(-1, "%s(): Existing Dedicated timeSchd:%d, new dedicated timeSchd:%d, total exceeds 80, cannot create new dedicated slice id:%d\n",
                __func__, totalTimeSchd, dl->timeSchd, id);
  }

#if 0 
  /* Marking the RBG-MAP with existing slice data */  
  for (int s = 0; s < si->num; ++s) {
    static_slice_param_t *sd = si->s[s]->algo_data;
    for (int i = sd->posLow; i <= sd->posHigh; ++i)
      rbgMap[i] = 1;
  }

  /*check for the overlap with new slice params*/
  for (int i = dl->posLow; i <= dl->posHigh; ++i)
    if (rbgMap[i])
      RET_FAIL(-3, "%s(): overlap of slices detected at RBG %d\n", __func__, i);
#endif

  if (!algo)
    RET_FAIL(-14, "%s(): no scheduler algorithm provided\n", __func__);

  /*Adding new Slice */
  slice_t *ns = _add_slice(&si->num, si->s);
  if (!ns)
    RET_FAIL(-4, "%s(): could not create new slice\n", __func__);
  ns->id = id;
  ns->label = label;
  ns->dl_algo = *(default_sched_dl_algo_t *) algo;
  if (!ns->dl_algo.data)
    ns->dl_algo.data = ns->dl_algo.setup();
  ns->algo_data = dl;

  LOG_I(MAC, "[%s]New Slice added index:%d si->num:%d id:%d\n", __func__, index, si->num,ns->id);
  return si->num - 1;
}

int addmod_static_slice_ul(slice_info_t *si,
                           int id,
                           char *label,
                           void *algo,
                           void *slice_params_ul) 
{
  static_slice_param_t *ul = slice_params_ul;
  uint16_t totalTimeSchd = 0;

  /* Minimum 3RBs, because LTE stack requires this */
  if (ul && ul->posLow + 2 > ul->posHigh)
    RET_FAIL(-1, "%s(): slice id %d posLow + 2 > posHigh\n", __func__, id);

  if ( (id == 0) && (ul->timeSchd > MAX_DEF_SLICE_TIME_SCHD))
    RET_FAIL(-1, "%s():UL Default slice id:%d timeSchd:%d exceeds 100\n", __func__, id, ul->timeSchd);

  if ( (id > 0) && (ul->timeSchd > MAX_DED_SLICE_TIME_SCHD))
    RET_FAIL(-1, "%s(): Dedicated slice id:%d, timeSchd:%d exceeds 80\n", __func__, id, ul->timeSchd);

  //uint8_t rbMap[110] = { 0 };
  int index = _exists_slice(si->num, si->s, id);
  LOG_I(MAC, "[%s]Enter index:%d si->num:%d id:%d\n", __func__, index, si->num,id);

  if (index >= 0) 
  {
    for (int s = 0; s < si->num; ++s) 
    {
      static_slice_param_t *su = ul && si->s[s]->id == id && ul ? ul : si->s[s]->algo_data;
#if 0 /*RB slice overlap check not required */
      for (int i = su->posLow; i <= su->posHigh; ++i) 
      {
        if (rbMap[i])
          RET_FAIL(-33, "%s(): overlap of slices detected at RBG %d\n", __func__, i);
        rbMap[i] = 1;
      }
#endif
      if (si->s[s]->id > 0)
      {
        totalTimeSchd += su->timeSchd;
      }
    }

    if (totalTimeSchd > MAX_DED_SLICE_TIME_SCHD)
      RET_FAIL(-1, "%s(): Total Dedicated timeSchd:%d exceeds 80, cannot create new dedicated slice id:%d\n",
                __func__, totalTimeSchd, id);

    /* no problem, can allocate */
    slice_t *s = si->s[index];
    if (algo) {
      s->ul_algo.unset(&s->ul_algo.data);
      s->ul_algo = *(default_sched_ul_algo_t *) algo;
      if (!s->ul_algo.data)
        s->ul_algo.data = s->ul_algo.setup();
    }
    if (label) {
      if (s->label) free(s->label);
      s->label = label;
    }

    /* Check if time_schd of dedicated slice is getting reduced */
    if ( ( ((static_slice_param_t *)s->algo_data)->timeSchd > ul->timeSchd) &&
         (dump_ue_list(&s->UEs) > 0) )     
    {
      /* In that case timeSchd difference should be added back to default slice */
      ((static_slice_param_t *)si->s[0]->algo_data)->timeSchd +=
          ( ((static_slice_param_t *)s->algo_data)->timeSchd - ul->timeSchd);
      LOG_I(MAC, "adding back %d time_sch to UL def slice,updated def slice time_schd:%d\n",
            ( ((static_slice_param_t *)s->algo_data)->timeSchd - ul->timeSchd),
            ((static_slice_param_t *)si->s[0]->algo_data)->timeSchd);
    }

    if (ul) {
      free(s->algo_data);
      s->algo_data = ul;
    }

    LOG_I(MAC, "Updated UL ded slice:%d time_schd:%d\n", s->id, ((static_slice_param_t *)s->algo_data)->timeSchd);
    return index;
  }

  /* Below code is executed for creating Default or Dedicated slice */
  if (!ul)
    RET_FAIL(-100, "%s(): no parameters for new slice %d, aborting\n", __func__, id);

  if (si->num >= MAX_STATIC_SLICES)
    RET_FAIL(-2, "%s(): cannot have more than %d slices\n", __func__, MAX_STATIC_SLICES);

  if (si->num > 1)
  {
    /* Check that total timeSchd of dedicated slices should not exceed 80%*/
    for(int ded_slice_idx = 1; ded_slice_idx < si->num; ded_slice_idx++)
    {
      static_slice_param_t *dedSlice = si->s[ded_slice_idx]->algo_data;
      totalTimeSchd += dedSlice->timeSchd;
    }

    if ( (totalTimeSchd + ul->timeSchd) > MAX_DED_SLICE_TIME_SCHD )
      RET_FAIL(-1, "%s(): Existing UL Dedicated timeSchd:%d, new dedicated timeSchd:%d, total exceeds 80, cannot create new dedicated slice id:%d\n",
                __func__, totalTimeSchd, ul->timeSchd, id);
  }

#if 0
  for (int s = 0; s < si->num; ++s) { /* Marking the RB-MAP with existing slice data */
    static_slice_param_t *sd = si->s[s]->algo_data;
    for (int i = sd->posLow; i <= sd->posHigh; ++i)
      rbMap[i] = 1;
  }

  for (int i = ul->posLow; i <= ul->posHigh; ++i) /*check for the overlap with new slice params*/
    if (rbMap[i])
      RET_FAIL(-3, "%s(): overlap of slices detected at RBG %d\n", __func__, i);
#endif

  if (!algo)
    RET_FAIL(-14, "%s(): no scheduler algorithm provided\n", __func__);

  slice_t *ns = _add_slice(&si->num, si->s);
  if (!ns)
    RET_FAIL(-4, "%s(): could not create new slice\n", __func__);
  ns->id = id;
  ns->label = label;
  ns->ul_algo = *(default_sched_ul_algo_t *) algo;
  if (!ns->ul_algo.data)
    ns->ul_algo.data = ns->ul_algo.setup();
  ns->algo_data = ul;

  LOG_I(MAC, "[%s]New UL Slice added index:%d si->num:%d id:%d\n", __func__, index, si->num,ns->id);
  return si->num - 1;
}

int remove_static_slice_dl(slice_info_t *si, uint8_t slice_idx) {
  if (slice_idx == 0)
    return 0;
  slice_t *sr = _remove_slice(&si->num, si->s, si->UE_assoc_slice, slice_idx);
  if (!sr)
    return 0;
  free(sr->algo_data);
  sr->dl_algo.unset(&sr->dl_algo.data);
  free(sr);
  return 1;
}

int remove_static_slice_ul(slice_info_t *si, uint8_t slice_idx) {
  if (slice_idx == 0)
    return 0;
  slice_t *sr = _remove_slice(&si->num, si->s, si->UE_assoc_slice, slice_idx);
  if (!sr)
    return 0;
  free(sr->algo_data);
  sr->ul_algo.unset(&sr->ul_algo.data);
  free(sr);
  return 1;
}

uint8_t reset_bit[MAX_STATIC_SLICES] = {0xe, 0xd, 0xb, 0x7};

void static_dl(module_id_t mod_id,
               int CC_id,
               frame_t frame,
               sub_frame_t subframe) 
{
  //LOG_I(FLEXRAN_AGENT, "[%s] SF %d\n", __func__, subframe);
  UE_info_t *UE_info = &RC.mac[mod_id]->UE_info;
  static uint16_t slice_schd_time[MAX_STATIC_SLICES] = {0};
  static uint8_t slice_schd_idx = 0;
  static uint8_t slice_mask = 0;
  uint8_t iter = 0;
  int i;
  static_slice_param_t *p = NULL;

  store_dlsch_buffer(mod_id, CC_id, frame, subframe);

  for (int UE_id = UE_info->list.head; UE_id >= 0; UE_id = UE_info->list.next[UE_id]) 
  {
    UE_sched_ctrl_t *ue_sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];

    /* initialize per-UE scheduling information */
    ue_sched_ctrl->pre_nb_available_rbs[CC_id] = 0;
    ue_sched_ctrl->dl_pow_off[CC_id] = 2;
    memset(ue_sched_ctrl->rballoc_sub_UE[CC_id], 0, sizeof(ue_sched_ctrl->rballoc_sub_UE[CC_id]));
    ue_sched_ctrl->pre_dci_dl_pdu_idx = -1;
  }

  const int N_RBG = to_rbg(RC.mac[mod_id]->common_channels[CC_id].mib->message.dl_Bandwidth);
  const int RBGsize = get_min_rb_unit(mod_id, CC_id);
  uint8_t *vrb_map = RC.mac[mod_id]->common_channels[CC_id].vrb_map;
  uint8_t rbgalloc_mask[N_RBG_MAX];
  
  for (i = 0; i < N_RBG; i++) 
  {
    // calculate mask: init to one + "AND" with vrb_map:
    // if any RB in vrb_map is blocked (1), the current RBG will be 0
    rbgalloc_mask[i] = 1;
    for (int j = 0; j < RBGsize; j++)
      rbgalloc_mask[i] &= !vrb_map[RBGsize * i + j];
  }

  /* Considering slice parameters can not be updated in background */
  slice_info_t *s = RC.mac[mod_id]->pre_processor_dl.slices;
  int max_num_ue; /*Max UE to be scheduled per slice */

#if 0
  switch (s->num) 
  {
    case 1:
      max_num_ue = 4;
      break;
    case 2:
      max_num_ue = 2;
      break;
    default:
      max_num_ue = 1;
      break;
  }
#endif

  /* As single slice get scehduled per SF, hence 4UEs can be scheduled per slice*/ 
  max_num_ue = 4;

  /* check & set slice scheduling timeframe in the begining of every SFN*/
  if (subframe == 0)
  {
    for (i = 0; i < s->num; ++i)
    {
      p = s->s[i]->algo_data;
      slice_schd_time[i] = p->timeSchd;
      
      if(slice_schd_time[i])
        slice_mask |= (1 << i);
    }
  }

  slice_schd_idx = (slice_schd_idx % s->num);

#if 0
  if ( ((frame & 127) == 0) && (subframe == 0) )
  {
    LOG_I(MAC, "[%u,%u,%d] UEs-slice0:%d S#:%d sliceMask:%x def time sched:%u dl-buff:%u slice_sch_Idx:%u\n",
          frame, subframe, UE_info->list.head, s->s[0]->UEs.head, s->num, slice_mask, slice_schd_time[0], UE_info->UE_template[0][UE_info->list.head].dl_buffer_total, slice_schd_idx);
  }
#endif

  for (i = slice_schd_idx; (i < s->num) && (slice_mask != 0) && (iter < s->num); (i = (i + 1) % s->num) ) 
  {
    iter++;
    /* Check if slice has scheduling oppertunities left in timeframe */
    if (slice_schd_time[i] == 0)
      continue;

    /* Skip Slice scheduling in case no UE is associated */
    if (s->s[i]->UEs.head < 0)
    {
      continue;
    }

    uint8_t rbgalloc_slice_mask[N_RBG_MAX];
    memset(rbgalloc_slice_mask, 0, sizeof(rbgalloc_slice_mask));
    int n_rbg_sched = 0;
    
    p = s->s[i]->algo_data;
    for (int rbg = p->posLow; rbg <= p->posHigh && rbg <= N_RBG; ++rbg) 
    {
      rbgalloc_slice_mask[rbg] = rbgalloc_mask[rbg];
      n_rbg_sched += rbgalloc_mask[rbg];
    }

#if 0
    if ( (UE_info->UE_template[0][UE_info->list.head].dl_buffer_total > 0) ||
         ( ((frame & 127) == 0) && (subframe == 0) )
       )
    {
        LOG_I(MAC, "[%u,%u,%d] DL Sch SId:%d S#:%d nrbg:%d tschd:%d buff:%u\n",
          frame, subframe, UE_info->list.head, i, s->num, n_rbg_sched, slice_schd_time[i], UE_info->UE_template[0][UE_info->list.head].dl_buffer_total);
    }
#endif

    s->s[i]->dl_algo.run(mod_id,
                         CC_id,
                         frame,
                         subframe,
                         &s->s[i]->UEs,
                         max_num_ue, // max_num_ue
                         n_rbg_sched,
                         rbgalloc_slice_mask,
                         s->s[i]->dl_algo.data);

    if (s->num > 1)
      slice_schd_idx = ( (slice_schd_idx + 1) % s->num); /*all slices get schedule in RR */
    
    if (slice_schd_time[i] >= 10)
    {
      slice_schd_time[i] -= 10;

      if (slice_schd_time[i] == 0)
        slice_mask &= reset_bit[i]; /*reset slice bit */

      break; /* only single slice per subframe scheduling allowed*/
    }

  }

  // the following block is meant for validation of the pre-processor to check
  // whether all UE allocations are non-overlapping and is not necessary for
  // scheduling functionality
  char t[26] = "_________________________";
  t[N_RBG] = 0;
  for (int i = 0; i < N_RBG; i++)
    for (int j = 0; j < RBGsize; j++)
      if (vrb_map[RBGsize*i+j] != 0)
        t[i] = 'x';
  int print = 0;
  for (int UE_id = UE_info->list.head; UE_id >= 0; UE_id = UE_info->list.next[UE_id]) {
    const UE_sched_ctrl_t *ue_sched_ctrl = &UE_info->UE_sched_ctrl[UE_id];

    if (ue_sched_ctrl->pre_nb_available_rbs[CC_id] == 0)
      continue;

    LOG_D(MAC,
          "%4d.%d UE%d %d RBs allocated, pre MCS %d\n",
          frame,
          subframe,
          UE_id,
          ue_sched_ctrl->pre_nb_available_rbs[CC_id],
          UE_info->eNB_UE_stats[CC_id][UE_id].dlsch_mcs1);

    print = 1;

    for (int i = 0; i < N_RBG; i++) {
      if (!ue_sched_ctrl->rballoc_sub_UE[CC_id][i])
        continue;
      for (int j = 0; j < RBGsize; j++) {
        if (vrb_map[RBGsize*i+j] != 0) {
          LOG_I(MAC, "%4d.%d DL scheduler allocation list: %s\n", frame, subframe, t);
          LOG_E(MAC, "%4d.%d: UE %d allocated at locked RB %d/RBG %d\n", frame,
                subframe, UE_id, RBGsize * i + j, i);
        }
        vrb_map[RBGsize*i+j] = 1;
      }
      t[i] = '0' + UE_id;
    }
  }
  if (print)
    LOG_D(MAC, "%4d.%d DL scheduler allocation list: %s\n", frame, subframe, t);
}

void static_ul(module_id_t mod_id,
               int CC_id,
               frame_t frame,
               sub_frame_t subframe,
               frame_t sched_frame,
               sub_frame_t sched_subframe) 
{
  UE_info_t *UE_info = &RC.mac[mod_id]->UE_info;
  static uint16_t slice_schd_time[MAX_STATIC_SLICES] = {0};
  static uint8_t slice_schd_idx = 0;
  static uint8_t slice_mask = 0;
  uint8_t iter = 0;
  int i;
  static_slice_param_t *p = NULL;

  const int N_RB_UL = to_prb(RC.mac[mod_id]->common_channels[CC_id].ul_Bandwidth);
  COMMON_channels_t *cc = &RC.mac[mod_id]->common_channels[CC_id];

  for (int UE_id = UE_info->list.head; UE_id >= 0; UE_id = UE_info->list.next[UE_id]) 
  {
    /* initialize per-UE scheduling information */
    UE_TEMPLATE *UE_template = &UE_info->UE_template[CC_id][UE_id];
    UE_template->pre_assigned_mcs_ul = 0;
    UE_template->pre_allocated_nb_rb_ul = 0;
    UE_template->pre_allocated_rb_table_index_ul = -1;
    UE_template->pre_first_nb_rb_ul = 0;
    UE_template->pre_dci_ul_pdu_idx = -1;
  }

  slice_info_t *s = RC.mac[mod_id]->pre_processor_ul.slices;
  int max_num_ue; /*Max UE to be scheduled per slice */

#if 0
  switch (s->num) {
    case 1:
      max_num_ue = 4;
      break;
    case 2:
      max_num_ue = 2;
      break;
    default:
      max_num_ue = 1;
      break;
  }
#endif

  /* As single slice get scehduled per SF, hence 4UEs can be scheduled per slice*/ 
  max_num_ue = 4;

  /* check & set slice scheduling timeframe in the begining of every SFN*/
  if (subframe == 0)
  {
    for (i = 0; i < s->num; ++i)
    {
      p = s->s[i]->algo_data;
      slice_schd_time[i] = p->timeSchd;
      
      if(slice_schd_time[i])
        slice_mask |= (1 << i);
    }
  }

  slice_schd_idx = (slice_schd_idx % s->num);

  for (i = slice_schd_idx; (i < s->num) && (slice_mask != 0) && (iter < s->num); (i = (i + 1) % s->num) ) 
  {
    iter++;
    /* Check if slice has scheduling oppertunities left in timeframe */
    if (slice_schd_time[i] == 0)
      continue;
    
    /* Skip Slice scheduling in case no UE is associated */
    if (s->s[i]->UEs.head < 0)
      continue;

    int last_rb_blocked = 1;
    int n_contig = 0;
    contig_rbs_t rbs[2]; // up to two contig RBs for PRACH in between
    p = s->s[i]->algo_data; 
    for (int rb = p->posLow; rb <= p->posHigh && rb < N_RB_UL; ++rb) 
    {
      if (cc->vrb_map_UL[rb] == 0 && last_rb_blocked) 
      {
        last_rb_blocked = 0;
        n_contig++;
        AssertFatal(n_contig <= 2, "cannot handle more than two contiguous RB regions\n");
        rbs[n_contig - 1].start = rb;
      }
      if (cc->vrb_map_UL[rb] == 1 && !last_rb_blocked) {
        last_rb_blocked = 1;
        rbs[n_contig - 1].length = rb - rbs[n_contig - 1].start;
      }
    }
    if (!last_rb_blocked)
      rbs[n_contig - 1].length = p->posHigh - rbs[n_contig - 1].start + 1;

    s->s[i]->ul_algo.run(mod_id,
                         CC_id,
                         frame,
                         subframe,
                         sched_frame,
                         sched_subframe,
                         &s->s[i]->UEs,
                         max_num_ue, // max_num_ue
                         n_contig,
                         rbs,
                         s->s[i]->ul_algo.data);

    if (s->num > 1)
      slice_schd_idx = ( (slice_schd_idx + 1) % s->num); /*all slices get schedule in RR */
    
    if (slice_schd_time[i] >= 10)
    {
      slice_schd_time[i] -= 10;
      if (slice_schd_time[i] == 0)
        slice_mask &= reset_bit[i]; /*reset slice bit */
      break; /* only single slice per subframe scheduling allowed*/
    }
  }

  // the following block is meant for validation of the pre-processor to check
  // whether all UE allocations are non-overlapping and is not necessary for
  // scheduling functionality
  char t[101] = "__________________________________________________"
                "__________________________________________________";
  t[N_RB_UL] = 0;
  for (int j = 0; j < N_RB_UL; j++)
    if (cc->vrb_map_UL[j] != 0)
      t[j] = 'x';
  int print = 0;
  for (int UE_id = UE_info->list.head; UE_id >= 0; UE_id = UE_info->list.next[UE_id]) {
    UE_TEMPLATE *UE_template = &UE_info->UE_template[CC_id][UE_id];
    if (UE_template->pre_allocated_nb_rb_ul == 0)
      continue;

    print = 1;
    uint8_t harq_pid = subframe2harqpid(&RC.mac[mod_id]->common_channels[CC_id],
                                        sched_frame, sched_subframe);
    LOG_D(MAC, "%4d.%d UE%d %d RBs (index %d) at start %d, pre MCS %d %s\n",
          frame,
          subframe,
          UE_id,
          UE_template->pre_allocated_nb_rb_ul,
          UE_template->pre_allocated_rb_table_index_ul,
          UE_template->pre_first_nb_rb_ul,
          UE_template->pre_assigned_mcs_ul,
          UE_info->UE_sched_ctrl[UE_id].round_UL[CC_id][harq_pid] > 0 ? "(retx)" : "");

    for (int i = 0; i < UE_template->pre_allocated_nb_rb_ul; ++i) {
      /* only check if this is not a retransmission */
      if (UE_info->UE_sched_ctrl[UE_id].round_UL[CC_id][harq_pid] == 0
          && cc->vrb_map_UL[UE_template->pre_first_nb_rb_ul + i] == 1) {

        LOG_I(MAC, "%4d.%d UL scheduler allocation list: %s\n", frame, subframe, t);
        LOG_E(MAC,
              "%4d.%d: UE %d allocated at locked RB %d (is: allocated start "
              "%d/length %d)\n",
              frame, subframe, UE_id, UE_template->pre_first_nb_rb_ul + i,
              UE_template->pre_first_nb_rb_ul,
              UE_template->pre_allocated_nb_rb_ul);
      }
      cc->vrb_map_UL[UE_template->pre_first_nb_rb_ul + i] = 1;
      t[UE_template->pre_first_nb_rb_ul + i] = UE_id + '0';
    }
  }
  if (print)
    LOG_D(MAC,
          "%4d.%d UL scheduler allocation list: %s\n",
          sched_frame,
          sched_subframe,
          t);
}

void static_destroy(slice_info_t **si) 
{
  const int n = (*si)->num;
  LOG_I(FLEXRAN_AGENT, "[%s] destroying slice %d\n", __func__, n);
  (*si)->num = 0;
  for (int i = 0; i < n; ++i) {
    slice_t *s = (*si)->s[i];
    if (s->label)
      free(s->label);
    free(s->algo_data);
    free(s);
  }
  free((*si)->s);
  free(*si);
}

pp_impl_param_t static_dl_init(module_id_t mod_id, int CC_id) 
{
  slice_info_t *si = calloc(1, sizeof(slice_info_t));
  DevAssert(si);
  LOG_I(FLEXRAN_AGENT, "[%s]mod_id %d cc_id:%d\n", __func__, mod_id,CC_id);

  si->num = 0;
  si->s = calloc(MAX_STATIC_SLICES, sizeof(slice_t));
  DevAssert(si->s);
  for (int i = 0; i < MAX_MOBILES_PER_ENB; ++i)
    si->UE_assoc_slice[i] = -1;

  /* insert default slice, all resources */
  static_slice_param_t *dlp = malloc(sizeof(static_slice_param_t));
  dlp->posLow = 0;
  dlp->posHigh = to_rbg(RC.mac[mod_id]->common_channels[CC_id].mib->message.dl_Bandwidth) - 1;
  dlp->timeSchd = MAX_DEF_SLICE_TIME_SCHD;
  default_sched_dl_algo_t *algo = &RC.mac[mod_id]->pre_processor_dl.dl_algo;
  algo->data = NULL;
  
  /* Add default DL slice */
  DevAssert(0 == addmod_static_slice_dl(si, 0, strdup("default"), algo, dlp));
  
  /* Add existing UEs to Default DL slice */
  const UE_list_t *UE_list = &RC.mac[mod_id]->UE_info.list;
  for (int UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id])
    slicing_add_UE(si, UE_id);

  pp_impl_param_t sttc;
  sttc.algorithm = STATIC_SLICING;
  sttc.add_UE = slicing_add_UE;
  sttc.remove_UE = slicing_remove_UE;
  sttc.move_UE = slicing_move_UE;
  sttc.addmod_slice = addmod_static_slice_dl;
  sttc.remove_slice = remove_static_slice_dl;
  sttc.dl = static_dl;
  // current DL algo becomes default scheduler
  sttc.dl_algo = *algo;
  sttc.destroy = static_destroy;
  sttc.slices = si;

  return sttc;
}

pp_impl_param_t static_ul_init(module_id_t mod_id, int CC_id) 
{
  slice_info_t *si = calloc(1, sizeof(slice_info_t));
  DevAssert(si);
  LOG_I(FLEXRAN_AGENT, "[%s]mod_id %d cc_id:%d\n", __func__, mod_id,CC_id);

  si->num = 0;
  si->s = calloc(MAX_STATIC_SLICES, sizeof(slice_t));
  DevAssert(si->s);
  for (int i = 0; i < MAX_MOBILES_PER_ENB; ++i)
    si->UE_assoc_slice[i] = -1;

  /* insert default slice, all resources */
  static_slice_param_t *ulp = malloc(sizeof(static_slice_param_t));
  ulp->posLow = 0;
  ulp->posHigh = to_prb(RC.mac[mod_id]->common_channels[CC_id].ul_Bandwidth) - 1;
  ulp->timeSchd = MAX_DEF_SLICE_TIME_SCHD;

  default_sched_ul_algo_t *algo = &RC.mac[mod_id]->pre_processor_ul.ul_algo;
  algo->data = NULL;
  
  /* Add default UL slice */
  DevAssert(0 == addmod_static_slice_ul(si, 0, strdup("default"), algo, ulp));

  /* Add existing UEs to Default DL slice */
  const UE_list_t *UE_list = &RC.mac[mod_id]->UE_info.list;
  for (int UE_id = UE_list->head; UE_id >= 0; UE_id = UE_list->next[UE_id])
    slicing_add_UE(si, UE_id);

  pp_impl_param_t sttc;
  sttc.algorithm = STATIC_SLICING;
  sttc.add_UE = slicing_add_UE;
  sttc.remove_UE = slicing_remove_UE;
  sttc.move_UE = slicing_move_UE;
  sttc.addmod_slice = addmod_static_slice_ul;
  sttc.remove_slice = remove_static_slice_ul;
  sttc.ul = static_ul;
  // current DL algo becomes default scheduler
  sttc.ul_algo = *algo;
  sttc.destroy = static_destroy;
  sttc.slices = si;

  return sttc;
}
