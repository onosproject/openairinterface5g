ARG BUILD_BASE_VERSION=latest
FROM onosproject/oai-build-base:$BUILD_BASE_VERSION AS builder

LABEL autodelete="true"

WORKDIR /root

COPY ./ ./

RUN patch -p1 -d . < docker/oai-enb/patches/disable_building_nasmesh_and_rbtool.patch

RUN cd ./cmake_targets \
        && ./build_oai -c --UE --verbose-compile \
        && ./nas_sim_tools/build/conf2uedata -c ../openair3/NAS/TOOLS/ue_eurecom_test_sfr.conf -o ran_build/build

FROM ubuntu:16.04
LABEL name="oai-ue" \
      version="develop-onf" \
      maintainer="Shad Ansari <shad@opennetworking.org>" \
      io.k8s.description="openairinterface5g UE"

WORKDIR /opt/oai-ue

RUN apt-get update && apt-get install -y libblas-dev liblapack-dev liblapacke-dev

COPY --from=builder /root/cmake_targets/ran_build/build/lte-uesoftmodem ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.usim.nvram0 ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.ue.nvram0 ./bin/
COPY --from=builder /root/cmake_targets/ran_build/build/.ue_emm.nvram0 ./bin/

COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/local/lib /usr/local/lib/
COPY --from=builder /root/cmake_targets/ran_build/build/*.so* /usr/local/lib/
RUN ldconfig

EXPOSE 50000/udp 50001/udp

CMD ["-C", "2625000000", "-r", "25", "--ue-rxgain", "140", "--basicsim", "--noS1"]
ENTRYPOINT ["/opt/oai-ue/bin/lte-uesoftmodem"]
