# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Open Networking Foundation
# Copyright 2024 Intel Corporation

OAI_ALL_VERSION ?= latest

.PHONY: build

all: build docker-build

build: # @HELP build all OAI code
build:
	. ./oaienv; cd cmake_targets; ./build_oai -c -I --eNB --UE -w USRP -g --build-ric-agent --build-ran-slicing

test:
	$(info No tests exist yet)

.PHONY: oai-build-base oai-ue oai-enb oai-enb-cu oai-enb-du

docker-build-oai-build-base: # @HELP build oai build base image
	docker build . -f docker/oai-build-base/Dockerfile \
		-t onosproject/oai-build-base:${OAI_ALL_VERSION}

docker-build-oai-all: # @HELP build oai all image
	docker build . -f docker/oai-all/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-all:${OAI_ALL_VERSION}

docker-build-oai-ue: # @HELP build oai ue image
	docker build . -f docker/oai-ue/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-ue:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

docker-build-oai-enb: # @HELP build oai enb image
	docker build . -f docker/oai-enb/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-enb:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

docker-build-oai-enb-cu: # @HELP build oai enb cu image
	docker build . -f docker/oai-enb-cu/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-enb-cu:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

docker-build-oai-enb-du: # @HELP build oai enb du image
	docker build . -f docker/oai-enb-du/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-enb-du:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

docker-build: # @HELP build all Docker images
docker-build: docker-build-oai-all docker-build-oai-ue docker-build-oai-enb docker-build-oai-enb-cu docker-build-oai-enb-du

docker-push-oai-build-base: # @HELP push oai build base image
	docker push onosproject/oai-build-base:${OAI_ALL_VERSION}

docker-push-oai-all: # @HELP push oai all image
	docker push onosproject/oai-all:${OAI_ALL_VERSION}

docker-push-oai-ue: # @HELP push oai ue image
	docker push onosproject/oai-ue:${OAI_ALL_VERSION}

docker-push-oai-enb: # @HELP push oai enb image
	docker push onosproject/oai-enb:${OAI_ALL_VERSION}

docker-push-oai-enb-cu: # @HELP push oai enb cu image
	docker push onosproject/oai-enb-cu:${OAI_ALL_VERSION}

docker-push-oai-enb-du: # @HELP push oai enb du image
	docker push onosproject/oai-enb-du:${OAI_ALL_VERSION}

docker-push: # @HELP push docker images
docker-push: docker-push-oai-all docker-push-oai-ue docker-push-oai-enb docker-push-oai-enb-cu docker-push-oai-enb-du

dev-base:
	docker build . -f docker/dev/base/Dockerfile -t oai-enb-cu-base:latest --rm=false

dev:
	docker build . -f docker/dev/Dockerfile -t onosproject/oai-enb-cu:latest --rm=false

check-version: # @HELP check version is duplicated
	./build/bin/version_check.sh all
