# Dockerfiles for OpenAirInterface

## Building
```sh
git clone git@github.com:onosproject/openairinterface5g.git
cd openairinterface5g
docker build . -f docker/oai-build-base/Dockerfile -t oai-build-base
docker build . -f docker/oai-ue/Dockerfile -t oai-ue
docker build . -f docker/oai-enb/Dockerfile -t oai-enb
docker build . -f docker/oai-enb-cu/Dockerfile -t oai-enb-cu
docker build . -f docker/oai-enb-du/Dockerfile -t oai-enb-du
docker rmi $(docker images -q -f "dangling=true" -f "label=autodelete=true")
```

## Running
### eNB and UE
```sh
docker run -d --net=host --privileged -e ENODEB=1 oai-enb
docker run -d --net=host --privileged -e ENODEB=1 oai-enb-cu
docker run -d --net=host --privileged -e ENODEB=1 oai-enb-du
docker run -d --net=host --privileged  oai-ue
```
