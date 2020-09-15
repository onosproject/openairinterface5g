
BUILD_BASE_VERSION := latest
VERSION := latest

all: images test

test:
	$(info No tests exist yet)

images: oai-build-base oai-ue oai-enb oai-enb-cu oai-enb-du

.PHONY: oai-build-base oai-ue oai-enb oai-enb-cu oai-enb-du

oai-build-base:
	docker build . -f docker/oai-build-base/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
		-t onosproject/oai-build-base:${VERSION}

oai-ue:
	docker build . -f docker/oai-ue/Dockerfile \
		--build-arg BUILD_BASE_VERSION=${BUILD_BASE_VERSION} \
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
