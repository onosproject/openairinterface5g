FROM oai-enb-cu-base:latest AS builder
LABEL name="oai-enb-cu" \
      version="develop-onf" \
      maintainer="Shad Ansari <shad@opennetworking.org>" \
      io.k8s.description="openairinterface5g eNB CU"

WORKDIR /root
COPY ./openair2/RIC_AGENT openair2/RIC_AGENT/
RUN OPENAIR_DIR=/root && \
    cmake_targets/build_oai -k --eNB -w SIMU -g --build-ric-agent && \
    mkdir -p /opt/oai/bin && \
    cp /root/cmake_targets/ran_build/build/lte-softmodem /opt/oai/bin  && \
    cp -r /root/ci-scripts/conf_files /opt/oai

#------

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

CMD ["-O", "/opt/oai/conf_files/cu.band7.tm1.25PRB.conf"]
ENTRYPOINT ["/opt/oai/bin/lte-softmodem"]
