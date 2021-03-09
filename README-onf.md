# RIC Agent #

The RIC Agent adds support for interfacing to a O-RAN Real-time Intelligent Controller (RIC) over the E2 interface. To build OAI with this support, enable the *--build-ric-agent* build option:

```shell
$ cd openairinterface5g
$ oaienv
$ ./build_oai -c --eNB --UE -w USRP -g --build-ric-agent
```

The top-level *Makefile* builds docker images:

```shell
$ cd openairinterface5g
$ make images
``
