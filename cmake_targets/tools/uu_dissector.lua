local proto_uu = Proto("Uu", "Uu Protocol")
local uu_dissector_switch = {
  [0x0201] = Dissector.get("lte_rrc.dl_ccch"),            -- LTE_RRC_DL_CCCH              (WS_LTE_PROTOCOL | 0x0001)
  [0x0202] = Dissector.get("lte_rrc.dl_dcch"),            -- LTE_RRC_DL_DCCH              (WS_LTE_PROTOCOL | 0x0002)
  [0x0203] = Dissector.get("lte_rrc.ul_ccch"),            -- LTE_RRC_UL_CCCH              (WS_LTE_PROTOCOL | 0x0003)
  [0x0204] = Dissector.get("lte_rrc.ul_dcch"),            -- LTE_RRC_UL_DCCH              (WS_LTE_PROTOCOL | 0x0004)
  [0x0205] = Dissector.get("lte_rrc.bcch_bch"),           -- LTE_RRC_BCCH_BCH             (WS_LTE_PROTOCOL | 0x0005)
  [0x0206] = Dissector.get("lte_rrc.bcch_dl_sch"),        -- LTE_RRC_BCCH_DL_SCH          (WS_LTE_PROTOCOL | 0x0006)
  [0x0207] = Dissector.get("lte_rrc.bcch_dl_sch_br"),     -- LTE_RRC_BCCH_DL_SCH_BR       (WS_LTE_PROTOCOL | 0x0007)
  [0x0208] = Dissector.get("lte_rrc.pcch"),               -- LTE_RRC_PCCH                 (WS_LTE_PROTOCOL | 0x0008)
  [0x0209] = Dissector.get("lte_rrc.mcch"),               -- LTE_RRC_MCCH                 (WS_LTE_PROTOCOL | 0x0009)
  [0x020A] = Dissector.get("lte_rrc.handover_prep_info"), -- LTE_RRC_HANDOVER_PREP_INFO   (WS_LTE_PROTOCOL | 0x000A)
  [0x020B] = Dissector.get("lte_rrc.sbcch_sl_bch"),       -- LTE_RRC_SBCCH_SL_BCH         (WS_LTE_PROTOCOL | 0x000B)
  [0x020C] = Dissector.get("lte_rrc.sbcch_sl_bch.v2x"),   -- LTE_RRC_SBCCH_SL_BCH_V2X     (WS_LTE_PROTOCOL | 0x000C)
  [0x020D] = Dissector.get("lte_rrc.sc_mcch"),            -- LTE_RRC_SC_MCCH              (WS_LTE_PROTOCOL | 0x000D)
  [0x020E] = Dissector.get("lte_rrc.dl_ccch.nb"),         -- LTE_RRC_DL_CCCH_NB           (WS_LTE_PROTOCOL | 0x000E)
  [0x010F] = Dissector.get("lte_rrc.dl_dcch.nb"),         -- LTE_RRC_DL_DCCH_NB           (WS_LTE_PROTOCOL | 0x000F)
  [0x0210] = Dissector.get("lte_rrc.ul_ccch.nb"),         -- LTE_RRC_UL_CCCH_NB           (WS_LTE_PROTOCOL | 0x0010)
  [0x0211] = Dissector.get("lte_rrc.ul_dcch.nb"),         -- LTE_RRC_UL_DCCH_NB           (WS_LTE_PROTOCOL | 0x0011)
  [0x0212] = Dissector.get("lte_rrc.bcch_bch.nb"),        -- LTE_RRC_BCCH_BCH_NB          (WS_LTE_PROTOCOL | 0x0012)
  [0x0213] = Dissector.get("lte_rrc.bcch_dl_sch.nb"),     -- LTE_RRC_BCCH_DL_SCH_NB       (WS_LTE_PROTOCOL | 0x0013)
  [0x0214] = Dissector.get("lte_rrc.pcch.nb"),            -- LTE_RRC_PCCH_NB              (WS_LTE_PROTOCOL | 0x0014)
  [0x0215] = Dissector.get("lte_rrc.sc_mcch.nb"),         -- LTE_RRC_SC_MCCH_NB           (WS_LTE_PROTOCOL | 0x0015)
  [0x0216] = Dissector.get("lte_rrc.bcch_bch.mbms"),      -- LTE_RRC_BCCH_BCH_MBMS        (WS_LTE_PROTOCOL | 0x0016)
  [0x0217] = Dissector.get("lte_rrc.bcch_dl_sch.mbms"),   -- LTE_RRC_BCCH_DL_SCH_MBMS     (WS_LTE_PROTOCOL | 0x0017)

  [0x0218] = Dissector.get("lte-rrc.ue_radio_access_cap_info"),    -- LTE_RRC_UE_RADIO_ACCESS_CAP_INFO     (WS_LTE_PROTOCOL | 0x0018)
  [0x0219] = Dissector.get("lte-rrc.ue_radio_paging_info"),        -- LTE_RRC_UE_RADIO_PAGING_INFO         (WS_LTE_PROTOCOL | 0x0019)
  [0x021A] = Dissector.get("lte-rrc.bcch.bch"),
  [0x021B] = Dissector.get("lte-rrc.bcch.bch.mbms"),
  [0x021C] = Dissector.get("lte-rrc.bcch.dl.sch"),
  [0x021D] = Dissector.get("lte-rrc.bcch.dl.sch.br"),
  [0x021E] = Dissector.get("lte-rrc.bcch.dl.sch.mbms"),
  [0x021F] = Dissector.get("lte-rrc.mcch"),
  [0x0220] = Dissector.get("lte-rrc.pcch"),
  [0x0221] = Dissector.get("lte-rrc.dl.ccch"),
  [0x0222] = Dissector.get("lte-rrc.dl.dcch"),
  [0x0223] = Dissector.get("lte-rrc.ul.ccch"),
  [0x0224] = Dissector.get("lte-rrc.ul.dcch"),
  [0x0225] = Dissector.get("lte-rrc.sc.mcch"),
  [0x0226] = Dissector.get("lte-rrc.ue_cap_info"),        -- LTE_RRC_UE_CAP_INFO          (WS_LTE_PROTOCOL | 0x0026)
  [0x0227] = Dissector.get("lte-rrc.ue_eutra_cap"),       -- LTE_RRC_UE_EUTRA_CAP         (WS_LTE_PROTOCOL | 0x0027)
  [0x0228] = Dissector.get("lte-rrc.sbcch.sl.bch"),
  [0x0229] = Dissector.get("lte-rrc.sbcch.sl.bch.v2x"),
  [0x022A] = Dissector.get("lte-rrc.ue_radio_access_cap_info.nb"), -- LTE_RRC_UE_RADiO_ACCESS_CAP_INFO_NB  (WS_LTE_PROTOCOL | 0x002A)
  [0x022B] = Dissector.get("lte-rrc.ue_radio_paging_info.nb"),     -- LTE_RRC_UE_RADIO_PAGING_INFO_NB      (WS_LTE_PROTOCOL | 0x002B)
  [0x022C] = Dissector.get("lte-rrc.bcch.bch.nb"),
  [0x022D] = Dissector.get("lte-rrc.bcch.dl.sch.nb"),
  [0x022E] = Dissector.get("lte-rrc.pcch.nb"),
  [0x022F] = Dissector.get("lte-rrc.dl.ccch.nb"),
  [0x0230] = Dissector.get("lte-rrc.dl.dcch.nb"),
  [0x0231] = Dissector.get("lte-rrc.ul.ccch.nb"),
  [0x0232] = Dissector.get("lte-rrc.sc.mcch.nb"),
  [0x0233] = Dissector.get("lte-rrc.ul.dcch.nb"),

  [0x0301] = Dissector.get("nr-rrc.bcch.bch"),            -- NR_RRC_BCCH_BCH              (WS_NR_PROTOCOL | 0x0001)
  [0x0302] = Dissector.get("nr-rrc.dl.dcch"),             -- NR_RRC_DL_DCCH               (WS_NR_PROTOCOL | 0x0002)
  [0x0303] = Dissector.get("nr-rrc.ul.dcch")              -- NR_RRC_UL_DCCH               (WS_NR_PROTOCOL | 0x0003)
}

function proto_uu.dissector(buffer, pinfo, tree)
  length = buffer:len()
  if length <= 2 then return end

  message_type = buffer(0,2):le_uint()
  local f = uu_dissector_switch[message_type]

    f:call(buffer(4):tvb(), pinfo, tree)

end

-- install Uu dissector at UDP port 9999
DissectorTable.get("udp.port"):add(9999, proto_uu)
