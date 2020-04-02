#
# Copyright 2019-present Open Networking Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

VERSION                  ?= $(shell cat ./VERSION)
BASE_NAME                ?= oai-base
UE_NAME                  ?= lte-uesoftmodem
ENB_NAME                 ?= lte-softmodem
GNB_NAME                 ?= nr-softmodem

# Tag is fixed to OAI release version, v1.0.0, regardless of VERSION file
DOCKER_TAG               ?= 1.0.0
DOCKER_REGISTRY          ?=
DOCKER_REPOSITORY        ?=
DOCKER_BUILD_ARGS        ?=
BASE_IMAGENAME           ?= ${DOCKER_REGISTRY}${DOCKER_REPOSITORY}${BASE_NAME}:${DOCKER_TAG}
UE_IMAGENAME             ?= ${DOCKER_REGISTRY}${DOCKER_REPOSITORY}${UE_NAME}:${DOCKER_TAG}
ENB_IMAGENAME            ?= ${DOCKER_REGISTRY}${DOCKER_REPOSITORY}${ENB_NAME}:${DOCKER_TAG}
GNB_IMAGENAME            ?= ${DOCKER_REGISTRY}${DOCKER_REPOSITORY}${GNB_NAME}:${DOCKER_TAG}

## Docker labels. Only set ref and commit date if committed
DOCKER_LABEL_VCS_URL     ?= $(shell git remote get-url $(shell git remote))
DOCKER_LABEL_VCS_REF     ?= $(shell git diff-index --quiet HEAD -- && git rev-parse HEAD || echo "unknown")
DOCKER_LABEL_COMMIT_DATE ?= $(shell git diff-index --quiet HEAD -- && git show -s --format=%cd --date=iso-strict HEAD || echo "unknown" )
DOCKER_LABEL_BUILD_DATE  ?= $(shell date -u "+%Y-%m-%dT%H:%M:%SZ")

# https://docs.docker.com/engine/reference/commandline/build/#specifying-target-build-stage---target

#
docker-build-centos.7.6.1810:
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${BASE_NAME} \
		--tag ${BASE_IMAGENAME}_centos.7.6.1810 \
		--file Dockerfile.base.centos.7.6.1810 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${ENB_NAME} \
		--tag ${ENB_IMAGENAME}_centos.7.6.1810  \
		--file Dockerfile.enb.centos.7.6.1810 \
		--build-arg build_base=${BASE_IMAGENAME}_centos.7.6.1810 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
		.
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${UE_NAME} \
		--tag ${UE_IMAGENAME}_centos.7.6.1810 \
		--file Dockerfile.ue.centos.7.6.1810 \
		--build-arg build_base=${BASE_IMAGENAME}_centos.7.6.1810 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .
#	docker build $(DOCKER_BUILD_ARGS) \
#		--target ${GNB_NAME} \
#		--tag ${GNB_IMAGENAME} \
#		--file Dockerfile.gnb \
#		--build-arg build_base=${BASE_IMAGENAME} \
#		--build-arg org_label_schema_version="${VERSION}" \
#		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
#		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
#		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
#		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
#		.

docker-build-ubuntu-1604:
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${BASE_NAME} \
		--tag ${BASE_IMAGENAME}_ubuntu.16.04 \
		--file Dockerfile.base.ubuntu.16.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${ENB_NAME} \
		--tag ${ENB_IMAGENAME}_ubuntu.16.04 \
		--file Dockerfile.enb.ubuntu.16.04 \
		--build-arg build_base=${BASE_IMAGENAME}_ubuntu.16.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
		.
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${UE_NAME} \
		--tag ${UE_IMAGENAME}_ubuntu.16.04 \
		--file Dockerfile.ue.ubuntu.16.04 \
		--build-arg build_base=${BASE_IMAGENAME}_ubuntu.16.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .

docker-build-ubuntu-1804:
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${BASE_NAME} \
		--tag ${BASE_IMAGENAME}_ubuntu.18.04 \
		--file Dockerfile.base.ubuntu.18.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${UE_NAME} \
		--tag ${UE_IMAGENAME}_ubuntu.18.04 \
		--file Dockerfile.ue.ubuntu.18.04 \
		--build-arg build_base=${BASE_IMAGENAME}_ubuntu.18.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
                .
	docker build $(DOCKER_BUILD_ARGS) \
		--target ${ENB_NAME} \
		--tag ${ENB_IMAGENAME}_ubuntu.18.04 \
		--file Dockerfile.enb.ubuntu.18.04 \
		--build-arg build_base=${BASE_IMAGENAME}_ubuntu.18.04 \
		--build-arg org_label_schema_version="${VERSION}" \
		--build-arg org_label_schema_vcs_url="${DOCKER_LABEL_VCS_URL}" \
		--build-arg org_label_schema_vcs_ref="${DOCKER_LABEL_VCS_REF}" \
		--build-arg org_label_schema_build_date="${DOCKER_LABEL_BUILD_DATE}" \
		--build-arg org_opencord_vcs_commit_date="${DOCKER_LABEL_COMMIT_DATE}" \
		.

docker-push:
	docker push ${BASE_IMAGENAME}
	docker push ${UE_IMAGENAME}
	docker push ${ENB_IMAGENAME}

test: docker-build

.PHONY: docker-build docker-push test
