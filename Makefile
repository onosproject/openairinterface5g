
BUILD_BASE_VERSION := latest
OAI_ALL_VERSION := latest

all: images test

test:
	$(info No tests exist yet)

images: oai-all oai-ue oai-enb oai-enb-cu oai-enb-du

.PHONY: oai-build-base oai-ue oai-enb oai-enb-cu oai-enb-du

oai-build-base:
	docker build . -f docker/oai-build-base/Dockerfile \
		-t onosproject/oai-build-base:${BUILD_BASE_VERSION}

oai-all:
	docker build . -f docker/oai-all/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-all:${OAI_ALL_VERSION}

oai-ue:
	docker build . -f docker/oai-ue/Dockerfile \
		--build-arg OAI_ALL_VERSION=${OAI_ALL_VERSION} \
		-t onosproject/oai-ue:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb:
	docker build . -f docker/oai-enb/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb-cu:
	docker build . -f docker/oai-enb-cu/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb-cu:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb-du:
	docker build . -f docker/oai-enb-du/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb-du:${OAI_ALL_VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

dev-base:
	docker build . -f docker/dev/base/Dockerfile -t oai-enb-cu-base:latest --rm=false

dev:
	docker build . -f docker/dev/Dockerfile -t onosproject/oai-enb-cu:latest --rm=false

build-tools: # @HELP install the ONOS build tools if needed
	@if [ ! -d "../build-tools" ]; then cd .. && git clone https://github.com/onosproject/build-tools.git; fi

jenkins-tools: # @HELP installs tooling needed for Jenkins
	cd .. && go get -u github.com/jstemmer/go-junit-report && go get github.com/t-yuki/gocover-cobertura

jenkins-test: images build-tools jenkins-tools
	TEST_PACKAGES=NONE ./../build-tools/build/jenkins/make-unit

publish: # @HELP publish version on github and dockerhub
	./../build-tools/publish-version ${VERSION} onosproject/oai-ue onosproject/oai-enb onosproject/oai-enb-cu onosproject/oai-enb-du

jenkins-publish: build-tools jenkins-tools
	./build/bin/push-images
	BASE_BRANCH=develop-onf ../build-tools/release-merge-commit
