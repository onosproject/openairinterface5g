/******************************************************************************
*******************************************************************************
***                                                                         ***
***  =====================================================================  ***
***                                                                         ***
***  (C) copyright Fraunhofer IIS (2008..2018) - all rights reserved        ***
***  sfn@iis.fraunhofer.de                                      ***
***  =====================================================================  ***
***                                                                         ***
*******************************************************************************
*******************************************************************************
***  $Id: main.c 438 2017-07-18 12:00:15Z det $
******************************************************************************/

#ifndef SUDAS_TM4_H
#define SUDAS_TM4_H


#define	FHG_TM4
//#define	FHG_TM4_LOG

#define	FHG_TM4_LOG_CQI
#define FHG_LOG




// [sfn]
extern FILE *debug_sudas_LOG_PHY;
extern FILE *debug_sudas_LOG_MAC;

#ifdef FHG_LOG

	#define sudas_LOG_PHY(c,...) fprintf(c,__VA_ARGS__)
	#define sudas_LOG_MAC(c,...) fprintf(c,__VA_ARGS__)
#else
	#define sudas_LOG_PHY(c,...)
	#define sudas_LOG_MAC(c,...)
#endif


#endif
	
