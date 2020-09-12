FROM oai-build-base:latest AS builder

LABEL autodelete="true"

WORKDIR /root

COPY ./ ./

RUN patch -p1 -d . < docker/oai-enb/patches/disable_building_nasmesh_and_rbtool.patch \
        && patch -p1 -d . < docker/oai-enb/patches/disable_sched_fifo_fail_exits.patch

RUN cd ./cmake_targets \
        && ./build_oai -c --eNB --verbose-compile

FROM ubuntu:16.04
LABEL name="oai-enb-du" \
      version="develop-onf" \
      maintainer="Shad Ansari <shad@opennetworking.org>" \
      io.k8s.description="openairinterface5g eNB DU"

WORKDIR /opt/oai

COPY --from=builder /root/cmake_targets/ran_build/build/lte-softmodem ./bin/
COPY --from=builder /root/ci-scripts/conf_files ./conf_files/

COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib /usr/local/lib/
COPY --from=builder /root/cmake_targets/ran_build/build/*.so* /usr/local/lib/
RUN ldconfig

# 2152 - S1U, GTP/UDP
EXPOSE 2152/udp
# 22100 - ?
EXPOSE 22100/tcp
# 36412 - S1C, SCTP/UDP
EXPOSE 36412/udp
# 36422 - X2C, SCTP/UDP
EXPOSE 36422/udp
# 50000 - IF5 / ORI (control)
EXPOSE 50000/udp
# 50001 - IF5 / ECPRI (data)
EXPOSE 50001/udp

EXPOSE 4043/tcp

CMD ["-O", "/opt/oai/conf_files/du.band7.tm1.25PRB.usrpb210.conf", "--basicsim"]
ENTRYPOINT ["/opt/oai/bin/lte-softmodem"]
