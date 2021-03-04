
BUILD_BASE_VERSION := latest
OAI_ALL_VERSION := latest
VERSION := latest

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
		-t onosproject/oai-ue:${VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb:
	docker build . -f docker/oai-enb/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb:${VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb-cu:
	docker build . -f docker/oai-enb-cu/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb-cu:${VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

oai-enb-du:
	docker build . -f docker/oai-enb-du/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-enb-du:${VERSION}
	-docker rmi $$(docker images -q -f "dangling=true" -f "label=autodelete=true")

build-tools: # @HELP install the ONOS build tools if needed
	@if [ ! -d "../build-tools" ]; then cd .. && git clone https://github.com/onosproject/build-tools.git; fi

jenkins-test: images
	TEST_PACKAGES=NONE ./../build-tools/build/jenkins/make-unit

jenkins-publish: build-tools
	./build/bin/push-images
	../build-tools/release-merge-commit
	../build-tools/build/docs/push-docs
