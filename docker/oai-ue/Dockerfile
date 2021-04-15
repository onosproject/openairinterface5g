ARG OAI_ALL_VERSION=latest
FROM onosproject/oai-all:$OAI_ALL_VERSION AS builder

LABEL autodelete="true"

WORKDIR /root

COPY ./ ./

FROM ubuntu:16.04
LABEL name="oai-ue" \
      version="develop-onf" \
      maintainer="Shad Ansari <shad@opennetworking.org>" \
      io.k8s.description="openairinterface5g UE"

WORKDIR /opt/oai-ue

RUN apt-get update && apt-get install -y libblas-dev liblapack-dev liblapacke-dev iproute2 net-tools software-properties-common iputils-ping
RUN add-apt-repository ppa:ettusresearch/uhd -y
RUN apt-get update && apt-get -y install libuhd-dev libuhd003 uhd-host
RUN uhd_images_downloader

COPY --from=builder /root/cmake_targets/ran_build/build/lte-uesoftmodem ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.usim.nvram0 ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.ue.nvram0 ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.ue_emm.nvram0 ./bin/
COPY --from=builder /root/cmake_targets/nas_sim_tools/build/conf2uedata ./bin/
COPY --from=builder /root/cmake_targets/nas_sim_tools/build/nvram ./bin/
COPY --from=builder /root/cmake_targets/nas_sim_tools/build/usim ./bin/

COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib /usr/local/lib/
COPY --from=builder /root/cmake_targets/ran_build/build/*.so* /usr/local/lib/
RUN ldconfig

EXPOSE 50000/udp 50001/udp

CMD ["-C", "2625000000", "-r", "25", "--ue-rxgain", "140", "--basicsim", "--noS1"]
ENTRYPOINT ["/opt/oai-ue/bin/lte-uesoftmodem"]
