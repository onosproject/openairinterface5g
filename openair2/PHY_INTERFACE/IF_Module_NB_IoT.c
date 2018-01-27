#include "openair2/PHY_INTERFACE/IF_Module_NB_IoT.h"
#include "openair2/PHY_INTERFACE/IF_Module_L2_primitives_NB_IoT.h"
#include "openair1/SCHED/IF_Module_L1_primitives_NB_IoT.h"
#include "LAYER2/MAC/extern_NB_IoT.h"
//#include "LAYER2/MAC/proto_NB_IoT.h"


#define MAX_IF_MODULES_NB_IoT 1 

IF_Module_NB_IoT_t *if_inst_NB_IoT[MAX_IF_MODULES_NB_IoT];
//#include "LAYER2/MAC/proto_NB_IoT.h"


IF_Module_NB_IoT_t *IF_Module_init_NB_IoT(int Mod_id){

  AssertFatal(Mod_id<MAX_MODULES,"Asking for Module %d > %d\n",Mod_id,MAX_IF_MODULES_NB_IoT);

  if (if_inst_NB_IoT[Mod_id]==NULL) {
    if_inst_NB_IoT[Mod_id] = (IF_Module_NB_IoT_t*)malloc(sizeof(IF_Module_NB_IoT_t));
    memset((void*)if_inst_NB_IoT[Mod_id],0,sizeof(IF_Module_NB_IoT_t));
    
    //if_inst[Mod_id]->CC_mask=0;
    if_inst_NB_IoT[Mod_id]->UL_indication = UL_indication_NB_IoT;

    /*AssertFatal(pthread_mutex_init(&if_inst[Mod_id]->if_mutex,NULL)==0,
		"allocation of if_inst[%d]->if_mutex fails\n",Mod_id);*/
  }
  return if_inst_NB_IoT[Mod_id];
}
