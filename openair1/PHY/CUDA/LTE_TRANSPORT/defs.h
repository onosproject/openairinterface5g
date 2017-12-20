#include <stdint.h>
#include <stdio.h>
#include "assertions.h"


#ifdef __cplusplus
extern "C"
#endif
void ulsch_extract_rb_and_compensation_cu( unsigned int first_rb,
                                  unsigned int nb_rb,
				                  unsigned short first_carrier_offset,
								  unsigned short number_symbols,
				                  unsigned short sf);

								  
#ifdef __cplusplus
extern "C"
#endif
void ulsch_channel_compensation_cu( short sf, short cyclic_shift, int *out, int *out2, int *u, int *v, int Msc_RS, short const_shift);		

#ifdef __cplusplus
extern "C"
#endif
void ulsch_extract_rb_cu( unsigned int first_rb,
                          unsigned int nb_rb,
						  unsigned short number_symbols,
				          unsigned short sf);

#ifdef __cplusplus
extern "C"
#endif						  
void exrb_compen_esti_cu( unsigned int first_rb,
                          unsigned int nb_rb,
						  unsigned short number_symbols,
				          unsigned short sf);
#ifdef __cplusplus
extern "C"
#endif
void estimation_cu( unsigned int first_rb,
                    unsigned int nb_rb,
					unsigned short number_symbols,
				    unsigned short sf);

#ifdef __cplusplus
extern "C"
#endif
void compensation_cu( unsigned int first_rb,
                    unsigned int nb_rb,
					unsigned short number_symbols,
					short Qm,
				    unsigned short sf);	
#ifdef __cplusplus
extern "C"
#endif					
void idft_cu(unsigned int first_rb,
             unsigned int nb_rb,
			 unsigned short number_symbols,
			 short cl,
			 unsigned short sf
            );
