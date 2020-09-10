# Dockerfiles for OpenAirInterface

### Building
```sh
git clone git@github.com:onosproject/openairinterface5g.git
cd openairinterface5g
git checkout develop-onf
docker build . -f docker/oai-build-base/Dockerfile -t oai-build-base
docker build . -f docker/oai-enb/Dockerfile -t oai-enb
```

### Running
```sh
docker run -d --net=host --privileged -e ENODEB=1 oai-enb
```
