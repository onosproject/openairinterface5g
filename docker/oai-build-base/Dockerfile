FROM ubuntu:16.04

WORKDIR /root

RUN apt-get update && apt-get install -y curl unzip build-essential subversion git cmake autoconf libtool pkg-config vim-common sudo protobuf-compiler libxml2-dev libyaml-dev libxslt-dev automake libtool make g++ libssl-dev libconfig-dev nettle-dev libsctp-dev libopenblas-dev libatlas-base-dev liblapack-dev libblas-dev liblapack3 libopenblas-base liblapacke-dev 
COPY docker/oai-build-base/scripts scripts/

RUN scripts/build_missing_packages
