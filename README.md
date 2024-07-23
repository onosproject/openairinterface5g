# OAI 5G gNB for SD-RAN #
This repository has the OAI 5G source from 2022.w51 tag [OpenAirInterface](http://www.openairinterface.org) and has O-RAN's E2AP Interface support to connect and communicate with ONF RIC (SD-RAN Release 1.4).Along with the E2AP Interface support, the OAI gNB in this repo also support Key Performance Metrics (E2SM-KPM) service model and periodically reports KPIs to the KPM xApp.

## Deployment Details ##
OAI 5G gNB with E2 Interface functionality is implemented & verified in monolithic deployment model. USPR B210 along with Intel NUC is used for deployment & testing of OAI 5G gNB. OAI 5G gNB is verified with configurations to support transmission bandwidth of 40MHz, SCS of 30kHz in TDD mode over Band 78 with SA mode. Google Pixel 5G, HOCell Dongle and APAL Dongles were used for the verificaiton of 5G Over the Air(OTA) scenarios like UE attach/detach & ping related traffic scenarios. 

## RIC Agent ##
The RIC Agent is an addition to OAI 5G gNB that adds support for interfacing the OAI gNB with a O-RAN Real-time Intelligent Controller (RIC) over the E2 interface. To build OAI 5G gNB with this support, enable the *--build-ric-agent* build option:

```shell
$ cd openairinterface5g
$ source oaienv
$ cd cmake_targets
$ sudo ./build_oai -w USRP --gNB -I --build-ric-agent
```

## OAI 5G gNB Configuration ##
Below mentioned is the OAI 5G gNB configuration file used for the execution & configuration of gNB parameters  

```shell
openairinterface5g/targets/PROJECTS/GENERIC-NR-5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf 
```
Following parameters should be modified as per the environment

**RIC IP Address**
```shell
RIC : {remote_ipv4_addr = "x.x.x.x";
       remote_port = 36401;
       enabled = "yes";
    };
```

**NUC IP Address**
```shell
local_s_address = "y.y.y.y";
```

**gNB IP Address**
```shell
NETWORK_INTERFACES :
    {
        GNB_INTERFACE_NAME_FOR_NG_AMF            = "demo-oai";
        GNB_IPV4_ADDRESS_FOR_NG_AMF              = "y.y.y.y/24";
        GNB_INTERFACE_NAME_FOR_NGU               = "demo-oai";
        GNB_IPV4_ADDRESS_FOR_NGU                 = "y.y.y.y/24";
        GNB_PORT_FOR_S1U                         = 2152; # Spec 2152
    };
 ```
 
 **AMF IP Address**
 ```shell
 ////////// AMF parameters:
    amf_ip_address      = ( { ipv4       = "z.z.z.z";
                              ipv6       = "192:168:30::17";
                              active     = "yes";
                              preference = "ipv4";
                            }
                          );
```

## Build and Execution ##
OAI 5G Build Command with RIC Agent
```shell
$ cd openairinterface5g/cmake_targets
$ sudo ./build_oai -w USRP --gNB -k --build-ric-agent
```

To run OAI 5G gNB along with RIC Agent
```shell
$ cd openairinterface5g/cmake_targets/ran_build/build
$ sudo ./nr-softmodem -O ../../../targets/PROJECTS/GENERIC-NR 5GC/CONF/gnb.sa.band78.fr1.106PRB.usrpb210.conf --sa -E --continuous-tx
```
