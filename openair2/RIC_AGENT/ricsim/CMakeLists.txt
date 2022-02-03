#!/bin/bash

#
# SPDX-FileCopyrightText: 2020-present Open Networking Foundation <info@opennetworking.org>
#
# SPDX-License-Identifier: Apache-2.0
#

#/*****************************************************************************
#                                                                            *
# Copyright 2019 AT&T Intellectual Property                                  *
# Copyright 2019 Nokia                                                       *
#                                                                            *
# Licensed under the Apache License, Version 2.0 (the "License");            *
# you may not use this file except in compliance with the License.           *
# You may obtain a copy of the License at                                    *
#                                                                            *
#      http://www.apache.org/licenses/LICENSE-2.0                            *
#                                                                            *
# Unless required by applicable law or agreed to in writing, software        *
# distributed under the License is distributed on an "AS IS" BASIS,          *
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# See the License for the specific language governing permissions and        *
# limitations under the License.                                             *
#                                                                            *
#******************************************************************************/

cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 14)



project(e2sim)

set (RICSIM_ROOT ./)


include_directories("${RICSIM_ROOT}")
include_directories("${RICSIM_ROOT}/ASN1c")


set(THREADS_PREFER_PTHREAD_FLAG ON)

find_library( SCTP_STD_LIB sctp )  #needed for sctp_sendmsgf
#find_package( Threads REQUIRED )

file(GLOB RICSIM_SOURCES

  "${RICSIM_ROOT}/ricsim.cpp"
  "${RICSIM_ROOT}/encode_kpm.cpp"
  "${RICSIM_ROOT}/e2ap_asn1c_codec.c"
  "${RICSIM_ROOT}/e2ap_asn1c_codec.h"
  "${RICSIM_ROOT}/e2ap_message_handler.cpp"
  "${RICSIM_ROOT}/e2ap_message_handler.hpp"
  "${RICSIM_ROOT}/encode_e2apv1.cpp"
  "${RICSIM_ROOT}/ricsim_defs.cpp"
  "${RICSIM_ROOT}/ricsim_defs.h"
  "${RICSIM_ROOT}/ricsim_sctp.cpp"
  "${RICSIM_ROOT}/ricsim_sctp.hpp"
  "${RICSIM_ROOT}/ASN1c/*.c"
  )
add_executable(ricsim ${RICSIM_SOURCES})
target_link_libraries( ricsim ${SCTP_STD_LIB}  )
#target_link_libraries( ricsim Threads::Threads )
