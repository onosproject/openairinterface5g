ARG BUILD_BASE_VERSION=latest
FROM onosproject/oai-build-base:$BUILD_BASE_VERSION AS builder

WORKDIR /root

COPY ./ ./

RUN patch -p1 -d . < docker/oai-all/patches/disable_building_nasmesh_and_rbtool.patch \
        && patch -p1 -d . < docker/oai-all/patches/disable_sched_fifo_fail_exits.patch \
        && patch -p1 -d . < docker/oai-all/patches/redirect_build_logs.patch

RUN cd ./cmake_targets \
        && ./build_oai -c -I --eNB --UE -w SIMU -w USRP -g Release --build-ric-agent --build-ran-slicing \
        && ./nas_sim_tools/build/conf2uedata -c ../openair3/NAS/TOOLS/ue_eurecom_test_sfr.conf -o ran_build/build
