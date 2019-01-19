#ifndef OPENAIRINTERFACE5G_LIMITS_H_
#define OPENAIRINTERFACE5G_LIMITS_H_
#include <platform_constants.h>

#define NUMBER_OF_UE_MAX MAX_MOBILES_PER_ENB
#define NUMBER_OF_CONNECTED_eNB_MAX MAX_MANAGED_ENB_PER_MOBILE
#define NUMBER_OF_eNB_MAX MAX_eNB

#if defined(MANAGED_RF)
#        define NUMBER_OF_RU_MAX 2
#else
#        define NUMBER_OF_RU_MAX 32
#        if defined(STANDALONE) && STANDALONE==1
#                undef  NUMBER_OF_RU_MAX
#                define NUMBER_OF_RU_MAX 3
#        endif
#        if defined(LARGE_SCALE) && LARGE_SCALE
#                undef  NUMBER_OF_RU_MAX
#                define NUMBER_OF_RU_MAX 16
#        endif
#endif

#endif /* OPENAIRINTERFACE5G_LIMITS_H_ */
