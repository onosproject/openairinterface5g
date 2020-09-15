# Dockerfiles for OpenAirInterface

## Building
```sh
git clone git@github.com:onosproject/openairinterface5g.git
cd openairinterface5g
docker build . -f docker/oai-build-base/Dockerfile -t onosproject/oai-build-base:latest
docker build . -f docker/oai-ue/Dockerfile -t onosproject/oai-ue:latest
docker build . -f docker/oai-enb/Dockerfile -t onosproject/oai-enb:latest
docker build . -f docker/oai-enb-cu/Dockerfile -t onosproject/oai-enb-cu:latest
docker build . -f docker/oai-enb-du/Dockerfile -t onosproject/oai-enb-du:latest
docker rmi $(docker images -q -f "dangling=true" -f "label=autodelete=true")
```

## Running
### eNB and UE
```sh
docker run -d --net=host --privileged -e ENODEB=1 onosproject/oai-enb
docker run -d --net=host --privileged -e ENODEB=1 onosproject/oai-enb-cu
docker run -d --net=host --privileged -e ENODEB=1 onosproject/oai-enb-du
docker run -d --net=host --privileged  onosproject/oai-ue
```
