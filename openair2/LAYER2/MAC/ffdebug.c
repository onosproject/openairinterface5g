#include <stdio.h>
#include <stdlib.h>

#include "ff-mac-sched-sap.h"
#include "ff-mac-csched-sap.h"

extern FILE *Q;

#define fp(l, Q, ...) do { \
  int fpp; \
  for (fpp = 0; fpp < (l); fpp++) fprintf(Q, " "); \
  fprintf(Q, __VA_ARGS__); \
  fflush(Q); \
} while (0)

static void dump_CschedUeReleaseReqParameters(const struct CschedUeReleaseReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct CschedUeReleaseReqParameters){\n");
  fp(l, Q, "  .rnti= %d,\n", p->rnti);
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}
static void dump_SrListElement_s(struct SrListElement_s s, int l)
{
  fp(l, Q, "(struct SrListElement_s){\n");
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "}");
}

static void dump_SchedUlSrInfoReqParameters(const struct SchedUlSrInfoReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct SchedUlSrInfoReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nr_srList= %d,\n", p->nr_srList);
  fp(l, Q, "  .srList= ");
  if (p->nr_srList != 0) {
    fp(0, Q, "(struct SrListElement_s []){\n");
    for (i = 0; i < p->nr_srList; i++) {
      dump_SrListElement_s(p->srList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  }");
  } else
    fp(0, Q, "NULL");
  fp(0, Q, ",\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_MacCeUlValue_u(struct MacCeUlValue_u s, int l)
{
  int i;
  fp(l, Q, "(struct MacCeUlValue_u){\n");
  fp(l, Q, "  .phr= %d,\n", s.phr);
  fp(l, Q, "  .crnti= %d,\n", s.crnti);
  fp(l, Q, "  .bufferStatus= {");
  for (i = 0; i < MAX_NR_LCG; i++)
    fp(0, Q, " %d,", s.bufferStatus[i]);
  fp(0, Q, "},\n");
  fp(l, Q, "}");
}

static void dump_MacCeUlListElement_s(struct MacCeUlListElement_s s, int l)
{
  fp(l, Q, "(struct MacCeUlListElement_s){\n");
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "  .macCeType= %d,\n", s.macCeType);
  fp(l, Q, "  .macCeValue=\n");
  dump_MacCeUlValue_u(s.macCeValue, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "}");
}

static void dump_SchedUlMacCtrlInfoReqParameters(const struct SchedUlMacCtrlInfoReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct SchedUlMacCtrlInfoReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nr_macCEUL_List= %d,\n", p->nr_macCEUL_List);
  fp(l, Q, "  .macCeUlList= ");
  if (p->nr_macCEUL_List != 0) {
    fp(0, Q, "(struct MacCeUlListElement_s[]){\n");
    for (i = 0; i < p->nr_macCEUL_List; i++) {
      dump_MacCeUlListElement_s(p->macCeUlList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  }");
  } else
    fp(0, Q, "NULL");
  fp(0, Q, ",\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_SchedDlRlcBufferReqParameters(const struct SchedDlRlcBufferReqParameters *p, int l)
{
  fp(l, Q, "(struct SchedDlRlcBufferReqParameters){\n");
  fp(l, Q, "  .rnti= %d,\n", p->rnti);
  fp(l, Q, "  .logicalChannelIdentity= %d,\n", p->logicalChannelIdentity);
  fp(l, Q, "  .rlcTransmissionQueueSize= %d,\n", p->rlcTransmissionQueueSize);
  fp(l, Q, "  .rlcTransmissionQueueHolDelay= %d,\n", p->rlcTransmissionQueueHolDelay);
  fp(l, Q, "  .rlcRetransmissionQueueSize= %d,\n", p->rlcRetransmissionQueueSize);
  fp(l, Q, "  .rlcRetransmissionHolDelay= %d,\n", p->rlcRetransmissionHolDelay);
  fp(l, Q, "  .rlcStatusPduSize= %d,\n", p->rlcStatusPduSize);
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_LogicalChannelConfigListElement_s(struct LogicalChannelConfigListElement_s s, int l)
{
  fp(l, Q, "(struct LogicalChannelConfigListElement_s){\n");
  fp(l, Q, "  .logicalChannelIdentity= %d,\n", s.logicalChannelIdentity);
  fp(l, Q, "  .logicalChannelGroup= %d,\n", s.logicalChannelGroup);
  fp(l, Q, "  .direction= %d,\n", s.direction);
  fp(l, Q, "  .qosBearerType= %d,\n", s.qosBearerType);
  fp(l, Q, "  .qci= %d,\n", s.qci);
  fp(l, Q, "  .eRabMaximulBitrateUl= %ld,\n", s.eRabMaximulBitrateUl);
  fp(l, Q, "  .eRabMaximulBitrateDl= %ld,\n", s.eRabMaximulBitrateDl);
  fp(l, Q, "  .eRabGuaranteedBitrateUl= %ld,\n", s.eRabGuaranteedBitrateUl);
  fp(l, Q, "  .eRabGuaranteedBitrateDl= %ld,\n", s.eRabGuaranteedBitrateDl);
  fp(l, Q, "}");
}

static void dump_CschedLcConfigReqParameters(const struct CschedLcConfigReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct CschedLcConfigReqParameters){\n");
  fp(l, Q, "  .rnti= %d,\n", p->rnti);
  fp(l, Q, "  .reconfigureFlag= %d,\n", p->reconfigureFlag);
  fp(l, Q, "  .nr_logicalChannelConfigList= %d,\n", p->nr_logicalChannelConfigList);
  fp(l, Q, "  .logicalChannelConfigList= ");
  if (p->nr_logicalChannelConfigList != 0) {
    fp(0, Q, "(struct LogicalChannelConfigListElement_s []){\n");
    for (i = 0; i < p->nr_logicalChannelConfigList; i++) {
      //fp(l, Q, "    &\n");
      dump_LogicalChannelConfigListElement_s(p->logicalChannelConfigList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  },\n");
  } else
    fp(0, Q, "NULL,");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_SchedDlMacBufferReqParameters(const struct SchedDlMacBufferReqParameters *p, int l)
{
  fp(l, Q, "(struct SchedDlMacBufferReqParameters){\n");
  fp(l, Q, "  .rnti= %d,\n", p->rnti);
  fp(l, Q, "  .ceBitmap= %d,\n", p->ceBitmap);
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_DrxConfig_s(struct DrxConfig_s s, int l)
{
  fp(l, Q, "(struct DrxConfig_s){\n");
  fp(l, Q, "  .onDurationTimer= %d,\n", s.onDurationTimer);
  fp(l, Q, "  .drxInactivityTimer= %d,\n", s.drxInactivityTimer);
  fp(l, Q, "  .drxRetransmissionTimer= %d,\n", s.drxRetransmissionTimer);
  fp(l, Q, "  .longDrxCycle= %d,\n", s.longDrxCycle);
  fp(l, Q, "  .longDrxCycleStartOffset= %d,\n", s.longDrxCycleStartOffset);
  fp(l, Q, "  .shortDrxCycle= %d,\n", s.shortDrxCycle);
  fp(l, Q, "  .drxShortCycleTimer= %d,\n", s.drxShortCycleTimer);
  fp(l, Q, "}");
}

static void dump_SpsConfig_s(struct SpsConfig_s s, int l)
{
  int i;
  fp(l, Q, "(struct SpsConfig_s){\n");
  fp(l, Q, "  .semiPersistSchedIntervalUl= %d,\n", s.semiPersistSchedIntervalUl);
  fp(l, Q, "  .semiPersistSchedIntervalDl= %d,\n", s.semiPersistSchedIntervalDl);
  fp(l, Q, "  .numberOfConfSpsProcesses= %d,\n", s.numberOfConfSpsProcesses);
  fp(l, Q, "  .n1PucchAnPersistentListSize= %d,\n", s.n1PucchAnPersistentListSize);
  fp(l, Q, "  .n1PucchAnPersistentList= {");
  for (i = 0; i < 4; i++)
    fp(0, Q, " %d,", s.n1PucchAnPersistentList[i]);
  fp(0, Q, "},\n");
  fp(l, Q, "  .implicitReleaseAfter= %d,\n", s.implicitReleaseAfter);
  fp(l, Q, "}");
}

static void dump_SrConfig_s(struct SrConfig_s s, int l)
{
  fp(l, Q, "(struct SrConfig_s){\n");
  fp(l, Q, "  .action= %d,\n", s.action);
  fp(l, Q, "  .schedInterval= %d,\n", s.schedInterval);
  fp(l, Q, "  .dsrTransMax= %d,\n", s.dsrTransMax);
  fp(l, Q, "}");
}

static void dump_CqiConfig_s(struct CqiConfig_s s, int l)
{
  fp(l, Q, "(struct CqiConfig_s){\n");
  fp(l, Q, "  .action= %d,\n", s.action);
  fp(l, Q, "  .cqiSchedInterval= %d,\n", s.cqiSchedInterval);
  fp(l, Q, "  .riSchedInterval= %d,\n", s.riSchedInterval);
  fp(l, Q, "}");
}

static void dump_UeCapabilities_s(struct UeCapabilities_s s, int l)
{
  fp(l, Q, "(struct UeCapabilities_s){\n");
  fp(l, Q, "  .halfDuplex= %d,\n", s.halfDuplex);
  fp(l, Q, "  .intraSfHopping= %d,\n", s.intraSfHopping);
  fp(l, Q, "  .type2Sb1= %d,\n", s.type2Sb1);
  fp(l, Q, "  .ueCategory= %d,\n", s.ueCategory);
  fp(l, Q, "  .resAllocType1= %d,\n", s.resAllocType1);
  fp(l, Q, "}");
}

static void dump_CschedUeConfigReqParameters(const struct CschedUeConfigReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct CschedUeConfigReqParameters){\n");
  fp(l, Q, "  .rnti= %d,\n", p->rnti);
  fp(l, Q, "  .reconfigureFlag= %d,\n", p->reconfigureFlag);
  fp(l, Q, "  .drxConfigPresent= %d,\n", p->drxConfigPresent);
  fp(l, Q, "  .drxConfig= ");
  dump_DrxConfig_s(p->drxConfig, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .timeAlignmentTimer= %d,\n", p->timeAlignmentTimer);
  fp(l, Q, "  .measGapConfigPattern= %d,\n", p->measGapConfigPattern);
  fp(l, Q, "  .measGapConfigSubframeOffset= %d,\n", p->measGapConfigSubframeOffset);
  fp(l, Q, "  .spsConfigPresent= %d,\n", p->spsConfigPresent);
  fp(l, Q, "  .spsConfig= ");
  dump_SpsConfig_s(p->spsConfig, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .srConfigPresent= %d,\n", p->srConfigPresent);
  fp(l, Q, "  .srConfig= ");
  dump_SrConfig_s(p->srConfig, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .cqiConfigPresent= %d,\n", p->cqiConfigPresent);
  fp(l, Q, "  .cqiConfig= ");
  dump_CqiConfig_s(p->cqiConfig, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .transmissionMode= %d,\n", p->transmissionMode);
  fp(l, Q, "  .ueAggregatedMaximumBitrateUl= %ld,\n", p->ueAggregatedMaximumBitrateUl);
  fp(l, Q, "  .ueAggregatedMaximumBitrateDl= %ld,\n", p->ueAggregatedMaximumBitrateDl);
  fp(l, Q, "  .ueCapabilities= ");
  dump_UeCapabilities_s(p->ueCapabilities, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .ueTransmitAntennaSelection= %d,\n", p->ueTransmitAntennaSelection);
  fp(l, Q, "  .ttiBundling= %d,\n", p->ttiBundling);
  fp(l, Q, "  .maxHarqTx= %d,\n", p->maxHarqTx);
  fp(l, Q, "  .betaOffsetAckIndex= %d,\n", p->betaOffsetAckIndex);
  fp(l, Q, "  .betaOffsetRiIndex= %d,\n", p->betaOffsetRiIndex);
  fp(l, Q, "  .betaOffsetCqiIndex= %d,\n", p->betaOffsetCqiIndex);
  fp(l, Q, "  .ackNackSrsSimultaneousTransmission= %d,\n", p->ackNackSrsSimultaneousTransmission);
  fp(l, Q, "  .simultaneousAckNackAndCqi= %d,\n", p->simultaneousAckNackAndCqi);
  fp(l, Q, "  .aperiodicCqiRepMode= %d,\n", p->aperiodicCqiRepMode);
  fp(l, Q, "  .tddAckNackFeedbackMode= %d,\n", p->tddAckNackFeedbackMode);
  fp(l, Q, "  .ackNackRepetitionFactor= %d,\n", p->ackNackRepetitionFactor);
  fp(l, Q, "  .extendedBSRSizes= %d,\n", p->extendedBSRSizes);
  fp(l, Q, "  .caSupport= %d,\n", p->caSupport);
  fp(l, Q, "  .crossCarrierSchedSupport= %d,\n", p->crossCarrierSchedSupport);
  fp(l, Q, "  .pcellCarrierIndex= %d,\n", p->pcellCarrierIndex);
  fp(l, Q, "  .nr_scells= %d,\n", p->nr_scells);
  if (p->nr_scells != 0) { printf("%s:%d:%s: TODO!!\n", __FILE__, __LINE__, __FUNCTION__); abort(); }
  else {
    fp(l, Q, "  .scellConfigList= {");
    for (i = 0; i < 2 /* MAX_NUM_CCs */ - 1; i++)
      fp(0, Q, " NULL,");
    fp(0, Q, " },\n");
  }
  fp(l, Q, "  .scellDeactivationTimer= %d,\n", p->scellDeactivationTimer);
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_RachListElement_s(struct RachListElement_s s, int l)
{
  fp(l, Q, "(struct RachListElement_s){\n");
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "  .estimatedSize= %d,\n", s.estimatedSize);
  fp(l, Q, "  .carrierIndex= %d,\n", s.carrierIndex);
  fp(l, Q, "}");
}

static void dump_SchedDlRachInfoReqParameters(const struct SchedDlRachInfoReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct SchedDlRachInfoReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nrrachList= %d,\n", p->nrrachList);
  if (p->nrrachList != 0) {
    fp(l, Q, "  .rachList= (struct RachListElement_s []){\n");
    for (i = 0; i < p->nrrachList; i++) {
      //fp(l, Q, "    &\n");
      dump_RachListElement_s(p->rachList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  },\n");
  } else
    fp(l, Q, "  .rachList= NULL,\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_CsiReport_s(struct CsiReport_s s, int l)
{
  static char *mode[] = { "P10", "P11", "P20", "P21", "A12", "A22", "A20", "A30", "A31" };
  int i;
  fp(l, Q, "(struct CsiReport_s){\n");
  fp(l, Q, "  .ri= %d,\n", s.ri);
  if (s.mode != A30)
    { printf("%s:%d: TODO, mode != A30\n", __FILE__, __LINE__); abort(); }
  fp(l, Q, "  .mode = %s,\n", mode[s.mode]);
  fp(l, Q, "  .report.A30Csi = {\n");
  fp(l, Q, "    .wbCqi= %d,\n", s.report.A30Csi.wbCqi);
  fp(l, Q, "    .sbCqi= {");
  for (i = 0; i < MAX_HL_SB; i++)
    fp(0, Q, " %d,", s.report.A30Csi.sbCqi[i]);
  fp(0, Q, " },\n");
  fp(l, Q, "  },\n");
  fp(l, Q, "}");
}

static void dump_CqiListElement_s(struct CqiListElement_s s, int l)
{
  fp(l, Q, "(struct CqiListElement_s){\n");
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "  .csiReport=\n");
  dump_CsiReport_s(s.csiReport, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .servCellIndex= %d,\n", s.servCellIndex);
  fp(l, Q, "}");
}

static void dump_SchedDlCqiInfoReqParameters(const struct SchedDlCqiInfoReqParameters*p, int l)
{
  int i;
  fp(l, Q, "(struct SchedDlCqiInfoReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nrcqiList= %d,\n", p->nrcqiList);
  if (p->nrcqiList != 0) {
    fp(l, Q, "  .cqiList= (struct CqiListElement_s[]){\n");
    for (i = 0; i < p->nrcqiList; i++) {
      dump_CqiListElement_s(p->cqiList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  },\n");
  } else
    fp(l, Q, "  .cqiList= NULL,\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_UlInfoListElement_s(struct UlInfoListElement_s s, int l)
{
  int i;
  fp(l, Q, "(struct UlInfoListElement_s){\n");
  fp(l, Q, "  .puschTransmissionTimestamp= %d,\n", s.puschTransmissionTimestamp);
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "  .ulReception= {");
  for (i = 0; i < MAX_LC_LIST+1; i++)
    fp(0, Q, " %d,", s.ulReception[i]);
  fp(0, Q, " },\n");
  fp(l, Q, "  .receptionStatus= %d,\n", s.receptionStatus);
  fp(l, Q, "  .tpc= %d,\n", s.tpc);
  fp(l, Q, "  .servCellIndex= %d\n", s.servCellIndex);
  fp(l, Q, "}");
}

static void dump_SchedUlTriggerReqParameters(const struct SchedUlTriggerReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct SchedUlTriggerReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nr_ulInfoList= %d,\n", p->nr_ulInfoList);
  if (p->nr_ulInfoList != 0) {
    fp(l, Q, "  .ulInfoList= (struct UlInfoListElement_s []){\n");
    for (i = 0; i < p->nr_ulInfoList; i++) {
      //fp(l, Q, "    &\n");
      dump_UlInfoListElement_s(p->ulInfoList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  },\n");
  } else
    fp(l, Q, "  .ulInfoList= NULL,\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_DlInfoListElement_s(struct DlInfoListElement_s s, int l)
{
  int i;
  fp(l, Q, "(struct DlInfoListElement_s){\n");
  fp(l, Q, "  .rnti= %d,\n", s.rnti);
  fp(l, Q, "  .harqProcessId= %d,\n", s.harqProcessId);
  fp(l, Q, "  .nr_harqStatus= %d,\n", s.nr_harqStatus);
  fp(l, Q, "  .harqStatus= {");
  for (i = 0; i < MAX_TB_LIST; i++)
    fp(0, Q, " %d,", s.harqStatus[i]);
  fp(0, Q, " },\n");
  fp(l, Q, "  .servCellIndex= %d\n", s.servCellIndex);
  fp(l, Q, "}");
}

static void dump_SchedDlTriggerReqParameters(const struct SchedDlTriggerReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct SchedDlTriggerReqParameters){\n");
  fp(l, Q, "  .sfnSf= %d,\n", p->sfnSf);
  fp(l, Q, "  .nr_dlInfoList= %d,\n", p->nr_dlInfoList);
  if (p->nr_dlInfoList != 0) {
    fp(l, Q, "  .dlInfoList= (struct DlInfoListElement_s []){\n");
    for (i = 0; i < p->nr_dlInfoList; i++) {
      //fp(l, Q, "    &\n");
      dump_DlInfoListElement_s(p->dlInfoList[i], l+4);
      fp(0, Q, ",\n");
    }
    fp(l, Q, "  },\n");
  } else
    fp(l, Q, "  .dlInfoList= NULL,\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

static void dump_SiMessageListElement_s(struct SiMessageListElement_s s, int l)
{
  fp(l, Q, "(struct SiMessageListElement_s){\n");
  fp(l, Q, "  .periodicity= %d,\n", s.periodicity);
  fp(l, Q, "  .length= %d,\n", s.length);
  fp(l, Q, "}");
}

static void dump_SiConfiguration_s(struct SiConfiguration_s s, int l)
{
  int i;
  fp(l, Q, "(struct SiConfiguration_s){\n");
  fp(l, Q, "  .sfn= %d,\n", s.sfn);
  fp(l, Q, "  .sib1Length= %d,\n", s.sib1Length);
  fp(l, Q, "  .siWindowLength= %d,\n", s.siWindowLength);
  fp(l, Q, "  .nrSI_Message_List= %d,\n", s.nrSI_Message_List);
  fp(l, Q, "  .siMessageList= (struct SiMessageListElement_s []){\n");
  for (i = 0; i < s.nrSI_Message_List; i++) {
    //fp(l, Q, "    &\n");
    dump_SiMessageListElement_s(s.siMessageList[i], l+4);
    fp(0, Q, ",\n");
  }
  fp(l, Q, "  }\n");
  fp(l, Q, "}");
}

static void dump_CschedCellConfigReqParametersListElement(struct CschedCellConfigReqParametersListElement *p, int l)
{
  int i;
  fp(l, Q, "(struct CschedCellConfigReqParametersListElement){\n");
  fp(l, Q, "  .puschHoppingOffset= %d,\n", p->puschHoppingOffset);
  fp(l, Q, "  .NcellID= %d,\n", p->NcellID);
  fp(l, Q, "  .hoppingMode= %d,\n", p->hoppingMode);
  fp(l, Q, "  .nSb= %d,\n", p->nSb);
  fp(l, Q, "  .phichResource= %d,\n", p->phichResource);
  fp(l, Q, "  .phichDuration= %d,\n", p->phichDuration);
  fp(l, Q, "  .initialNrOfPdcchOfdmSymbols= %d,\n", p->initialNrOfPdcchOfdmSymbols);
  fp(l, Q, "  .siConfiguration= \n");
  dump_SiConfiguration_s(p->siConfiguration, l+4);
  fp(0, Q, ",\n");
  fp(l, Q, "  .ulBandwidth= %d,\n", p->ulBandwidth);
  fp(l, Q, "  .dlBandwidth= %d,\n", p->dlBandwidth);
  fp(l, Q, "  .ulCyclicPrefixLength= %d,\n", p->ulCyclicPrefixLength);
  fp(l, Q, "  .dlCyclicPrefixLength= %d,\n", p->dlCyclicPrefixLength);
  fp(l, Q, "  .antennaPortsCount= %d,\n", p->antennaPortsCount);
  fp(l, Q, "  .duplexMode= %d,\n", p->duplexMode);
  fp(l, Q, "  .subframeAssignment= %d,\n", p->subframeAssignment);
  fp(l, Q, "  .specialSubframePatterns= %d,\n", p->specialSubframePatterns);
  fp(l, Q, "  .mbsfn_SubframeConfigPresent= %d,\n", p->mbsfn_SubframeConfigPresent);
  fp(l, Q, "  .mbsfnSubframeConfigRfPeriod= {");
  for (i = 0; i < MAX_MBSFN_CONFIG; i++)
    fp(0, Q, " %d,", p->mbsfnSubframeConfigRfPeriod[i]);
  fp(0, Q, "  },\n");
  fp(l, Q, "  .mbsfnSubframeConfigRfOffset= {");
  for (i = 0; i < MAX_MBSFN_CONFIG; i++)
    fp(0, Q, " %d,", p->mbsfnSubframeConfigRfOffset[i]);
  fp(0, Q, "  },\n");
  fp(l, Q, "  .mbsfnSubframeConfigSfAllocation= {");
  for (i = 0; i < MAX_MBSFN_CONFIG; i++)
    fp(0, Q, " %d,", p->mbsfnSubframeConfigSfAllocation[i]);
  fp(0, Q, "  },\n");
  fp(l, Q, "  .prachConfigurationIndex= %d,\n", p->prachConfigurationIndex);
  fp(l, Q, "  .prachFreqOffset= %d,\n", p->prachFreqOffset);
  fp(l, Q, "  .raResponseWindowSize= %d,\n", p->raResponseWindowSize);
  fp(l, Q, "  .macContentionResolutionTimer= %d,\n", p->macContentionResolutionTimer);
  fp(l, Q, "  .maxHarqMsg3Tx= %d,\n", p->maxHarqMsg3Tx);
  fp(l, Q, "  .n1PucchAn= %d,\n", p->n1PucchAn);
  fp(l, Q, "  .deltaPucchShift= %d,\n", p->deltaPucchShift);
  fp(l, Q, "  .nrbCqi= %d,\n", p->nrbCqi);
  fp(l, Q, "  .ncsAn= %d,\n", p->ncsAn);
  fp(l, Q, "  .srsSubframeConfiguration= %d,\n", p->srsSubframeConfiguration);
  fp(l, Q, "  .srsSubframeOffset= %d,\n", p->srsSubframeOffset);
  fp(l, Q, "  .srsBandwidthConfiguration= %d,\n", p->srsBandwidthConfiguration);
  fp(l, Q, "  .srsMaxUpPts= %d,\n", p->srsMaxUpPts);
  fp(l, Q, "  .enable64Qam= %d,\n", p->enable64Qam);
  fp(l, Q, "  .carrierIndex= %d,\n", p->carrierIndex);
  fp(l, Q, "}");
}

static void dump_CschedCellConfigReqParameters(const struct CschedCellConfigReqParameters *p, int l)
{
  int i;
  fp(l, Q, "(struct CschedCellConfigReqParameters){\n");
  fp(l, Q, "  .nr_carriers= %d,\n", p->nr_carriers);
  fp(l, Q, "  .ccConfigList= {\n");
  for (i = 0; i < p->nr_carriers; i++) {
    fp(l, Q, "    &\n");
    dump_CschedCellConfigReqParametersListElement(p->ccConfigList[i], l+4);
    fp(0, Q, ",\n");
  }
  fp(l, Q, "  },\n");
  fp(l, Q, "  .nr_vendorSpecificList= 0,\n");
  fp(l, Q, "  .vendorSpecificList= NULL\n");
  fp(l, Q, "}");
}

#undef CschedCellConfigReq
void _CschedCellConfigReq(void *x, const struct CschedCellConfigReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct CschedCellConfigReqParameters *p = &\n");
  dump_CschedCellConfigReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  CschedCellConfigReq(x, p);\n");
  fp(2, Q, "  CschedCellConfigCnf(NULL, NULL);\n");
  fp(2, Q, "}\n");
  CschedCellConfigReq(x, params);
}

#undef CschedUeConfigReq
void _CschedUeConfigReq(void *x, const struct CschedUeConfigReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct CschedUeConfigReqParameters *p = &\n");
  dump_CschedUeConfigReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  CschedUeConfigReq(x, p);\n");
  fp(2, Q, "  CschedUeConfigCnf(NULL, NULL);\n");
  fp(2, Q, "}\n");
  CschedUeConfigReq(x, params);
}

#undef CschedLcConfigReq
void _CschedLcConfigReq(void *x, const struct CschedLcConfigReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct CschedLcConfigReqParameters *p = &\n");
  dump_CschedLcConfigReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  CschedLcConfigReq(x, p);\n");
  fp(2, Q, "  CschedLcConfigCnf(NULL, NULL);\n");
  fp(2, Q, "}\n");
  CschedLcConfigReq(x, params);
}

#undef CschedLcReleaseReq
void _CschedLcReleaseReq(void *x, const struct CschedLcReleaseReqParameters *params)
{
  abort();
}

#undef CschedUeReleaseReq
void _CschedUeReleaseReq(void *x, const struct CschedUeReleaseReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct CschedUeReleaseReqParameters *p = &\n");
  dump_CschedUeReleaseReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  CschedUeReleaseReq(x, p);\n");
  fp(2, Q, "  CschedUeReleaseInd(NULL, NULL);\n");
  fp(2, Q, "}\n");
  CschedUeReleaseReq(x, params);
}

#undef SchedDlRlcBufferReq
void _SchedDlRlcBufferReq(void *x, const struct SchedDlRlcBufferReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedDlRlcBufferReqParameters *p = &\n");
  dump_SchedDlRlcBufferReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedDlRlcBufferReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedDlRlcBufferReq(x, params);
}

#undef SchedDlPagingBufferReq
void _SchedDlPagingBufferReq(void *x, const struct SchedDlPagingBufferReqParameters *params)
{
  abort();
}

#undef SchedDlMacBufferReq
void _SchedDlMacBufferReq(void *x, const struct SchedDlMacBufferReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedDlMacBufferReqParameters *p = &\n");
  dump_SchedDlMacBufferReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedDlMacBufferReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedDlMacBufferReq(x, params);
}

#undef SchedDlTriggerReq
void _SchedDlTriggerReq(void *x, const struct SchedDlTriggerReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedDlTriggerReqParameters *p = &\n");
  dump_SchedDlTriggerReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedDlTriggerReq(x, p);\n");
  fp(2, Q, "  SchedDlConfigInd(NULL, NULL);\n");
  fp(2, Q, "}\n");
  SchedDlTriggerReq(x, params);
}

#undef SchedDlRachInfoReq
void _SchedDlRachInfoReq(void *x, const struct SchedDlRachInfoReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedDlRachInfoReqParameters *p = &\n");
  dump_SchedDlRachInfoReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedDlRachInfoReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedDlRachInfoReq(x, params);
}

#undef SchedDlCqiInfoReq
void _SchedDlCqiInfoReq(void *x, const struct SchedDlCqiInfoReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedDlCqiInfoReqParameters *p = &\n");
  dump_SchedDlCqiInfoReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedDlCqiInfoReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedDlCqiInfoReq(x, params);
}

#undef SchedUlTriggerReq
void _SchedUlTriggerReq(void *x, const struct SchedUlTriggerReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedUlTriggerReqParameters *p = &\n");
  dump_SchedUlTriggerReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedUlTriggerReq(x, p);\n");
  fp(2, Q, "  SchedUlConfigInd(NULL, NULL);\n");
  fp(2, Q, "}\n");
  SchedUlTriggerReq(x, params);
}

#undef SchedUlNoiseInterferenceReq
void _SchedUlNoiseInterferenceReq(void *x, const struct SchedUlNoiseInterferenceReqParameters *params)
{
  abort();
}

#undef SchedUlSrInfoReq
void _SchedUlSrInfoReq(void *x, const struct SchedUlSrInfoReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedUlSrInfoReqParameters *p = &\n");
  dump_SchedUlSrInfoReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedUlSrInfoReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedUlSrInfoReq(x, params);
}

#undef SchedUlMacCtrlInfoReq
void _SchedUlMacCtrlInfoReq(void *x, const struct SchedUlMacCtrlInfoReqParameters *params)
{
  fp(2, Q, "{\n");
  fp(2, Q, "  struct SchedUlMacCtrlInfoReqParameters *p = &\n");
  dump_SchedUlMacCtrlInfoReqParameters(params, 2+4);
  fp(0, Q, ";\n");
  fp(2, Q, "  SchedUlMacCtrlInfoReq(x, p);\n");
  fp(2, Q, "}\n");
  SchedUlMacCtrlInfoReq(x, params);
}

#undef SchedUlCqiInfoReq
void _SchedUlCqiInfoReq(void *x, const struct SchedUlCqiInfoReqParameters *params)
{
  abort();
}
