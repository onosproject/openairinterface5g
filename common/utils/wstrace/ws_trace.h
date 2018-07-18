

#ifndef _WS_TRACE_H_
#define _WS_TRACE_H_

#define WS_PROTOCOL_MASK                     (0xFF00)

#define WS_WCDMA_PROTOCOL                    (0x0100)
#define WS_LTE_PROTOCOL                      (0x0200)
#define WS_NR_PROTOCOL                       (0x0300)

#define LTE_RRC_DL_CCCH                      (WS_LTE_PROTOCOL | 0x0001)
#define LTE_RRC_DL_DCCH                      (WS_LTE_PROTOCOL | 0x0002)
#define LTE_RRC_UL_CCCH                      (WS_LTE_PROTOCOL | 0x0003)
#define LTE_RRC_UL_DCCH                      (WS_LTE_PROTOCOL | 0x0004)
#define LTE_RRC_BCCH_BCH                     (WS_LTE_PROTOCOL | 0x0005)
#define LTE_RRC_BCCH_DL_SCH                  (WS_LTE_PROTOCOL | 0x0006)
#define LTE_RRC_BCCH_DL_SCH_BR               (WS_LTE_PROTOCOL | 0x0007)
#define LTE_RRC_PCCH                         (WS_LTE_PROTOCOL | 0x0008)
#define LTE_RRC_MCCH                         (WS_LTE_PROTOCOL | 0x0009)
#define LTE_RRC_HANDOVER_PREP_INFO           (WS_LTE_PROTOCOL | 0x000A)
#define LTE_RRC_SBCCH_SL_BCH                 (WS_LTE_PROTOCOL | 0x000B)
#define LTE_RRC_SBCCH_SL_BCH_V2X             (WS_LTE_PROTOCOL | 0x000C)
#define LTE_RRC_SC_MCCH                      (WS_LTE_PROTOCOL | 0x000D)
#define LTE_RRC_DL_CCCH_NB                   (WS_LTE_PROTOCOL | 0x000E)
#define LTE_RRC_DL_DCCH_NB                   (WS_LTE_PROTOCOL | 0x000F)
#define LTE_RRC_UL_CCCH_NB                   (WS_LTE_PROTOCOL | 0x0010)
#define LTE_RRC_UL_DCCH_NB                   (WS_LTE_PROTOCOL | 0x0011)
#define LTE_RRC_BCCH_BCH_NB                  (WS_LTE_PROTOCOL | 0x0012)
#define LTE_RRC_BCCH_DL_SCH_NB               (WS_LTE_PROTOCOL | 0x0013)
#define LTE_RRC_PCCH_NB                      (WS_LTE_PROTOCOL | 0x0014)
#define LTE_RRC_SC_MCCH_NB                   (WS_LTE_PROTOCOL | 0x0015)
#define LTE_RRC_BCCH_BCH_MBMS                (WS_LTE_PROTOCOL | 0x0016)
#define LTE_RRC_BCCH_DL_SCH_MBMS             (WS_LTE_PROTOCOL | 0x0017)

#define LTE_RRC_UE_RADIO_ACCESS_CAP_INFO     (WS_LTE_PROTOCOL | 0x0018)
#define LTE_RRC_UE_RADIO_PAGING_INFO         (WS_LTE_PROTOCOL | 0x0019)

#define LTE_RRC_UE_CAP_INFO                  (WS_LTE_PROTOCOL | 0x0026)
#define LTE_RRC_UE_EUTRA_CAP                 (WS_LTE_PROTOCOL | 0x0027)

#define LTE_RRC_UE_RADiO_ACCESS_CAP_INFO_NB  (WS_LTE_PROTOCOL | 0x002A)
#define LTE_RRC_UE_RADIO_PAGING_INFO_NB      (WS_LTE_PROTOCOL | 0x002B)

#define NR_RRC_BCCH_BCH                      (WS_NR_PROTOCOL | 0x0001)
#define NR_RRC_DL_DCCH                       (WS_NR_PROTOCOL | 0x0002)
#define NR_RRC_UL_DCCH                       (WS_NR_PROTOCOL | 0x0003)

#define WS_TRACE_ADDRESS                     "127.0.0.100"
#define WS_TRACE_PORT                        (9999)

void start_ws_trace(void);
void send_ws_log(unsigned short msg_type, unsigned short rnti, const unsigned char *msg_buf, unsigned short msg_len);
void stop_ws_trace(void);

#endif /* _WS_TRACE_H_ */



