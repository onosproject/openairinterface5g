FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y curl unzip build-essential subversion git cmake autoconf libtool pkg-config vim-common sudo protobuf-compiler libxml2-dev libyaml-dev libxslt-dev automake libtool make g++ libssl-dev libconfig-dev nettle-dev libsctp-dev libopenblas-dev libatlas-base-dev liblapack-dev libblas-dev liblapack3 libopenblas-base liblapacke-dev openssl iproute2 net-tools ethtool software-properties-common iputils-ping

COPY ./ ./

RUN docker/dev/base/scripts/build_missing_packages

RUN patch -p1 -d . < docker/dev/base/patches/disable_building_nasmesh_and_rbtool.patch \
        && patch -p1 -d . < docker/dev/base/patches/disable_sched_fifo_fail_exits.patch \
        && patch -p1 -d . < docker/dev/base/patches/redirect_build_logs.patch

RUN cmake_targets/build_oai -c -I --eNB -w SIMU -g --build-ric-agent
