## oai-docker

### Docker images build
* At either ubuntu 16.04 or ubuntu 18.04 host, the following builds softmodem images for ubuntu16.04/18.04 targets
(*) centos targets still in progress

sudo make docker-build-ubuntu-1604 
sudo make docker-build-ubuntu-1804 

### Container generation

* ubuntu 16.04 target

sudo docker run --net=host  --cap-add=NET_ADMIN --device /dev/net/tun:/dev/net/tun --env RFSIMULATOR=enb -it lte-softmodem:1.0.0_ubuntu.16.04 ran_build/build/lte-softmodem -O ../ci-scripts/conf_files/lte-fdd-mbms-basic-sim.conf --rfsim  --noS1 --nokrnmod 1

sudo docker run --net=host --cap-add=NET_ADMIN --device /dev/net/tun:/dev/net/tun --env RFSIMULATOR=127.0.0.1 -it --privileged lte-uesoftmodem:1.0.0_ubuntu.16.04 ran_build/build/lte-uesoftmodem -r 25 --ue-rxgain 140 --ue-txgain 120 --rfsim --noS1 --nokrnmod 1

* ubuntu 18.04 target

sudo docker run --net=host  --cap-add=NET_ADMIN --device /dev/net/tun:/dev/net/tun --env RFSIMULATOR=enb -it lte-softmodem:1.0.0_ubuntu.18.04 ran_build/build/lte-softmodem -O ../ci-scripts/conf_files/lte-fdd-mbms-basic-sim.conf --rfsim  --noS1 --nokrnmod 1

sudo docker run --net=host --cap-add=NET_ADMIN --device /dev/net/tun:/dev/net/tun --env RFSIMULATOR=127.0.0.1 -it --privileged lte-uesoftmodem:1.0.0_ubuntu.18.04 ran_build/build/lte-uesoftmodem -r 25 --ue-rxgain 140 --ue-txgain 120 --rfsim --noS1 --nokrnmod 1

## Authors

* **Javier Morgade** (javier.morgade@ieee.org) - *Initial work* - [opencord](https://github.com/opencord/openairinterface.git)

## Acknowledgements 
* Work based on a set of dockerfiles inherited from [opencord](https://github.com/opencord/openairinterface.git)

