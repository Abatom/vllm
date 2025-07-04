#!/bin/bash

# This script build the Neuron docker image and run the API server inside the container.
# It serves a sanity check for compilation and basic model usage.
set -e
set -v

image_name="neuron/vllm-ci"
container_name="neuron_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"
HF_TOKEN=$(aws secretsmanager get-secret-value  --secret-id "ci/vllm-neuron/hf-token" --region us-west-2 --query 'SecretString' --output text | jq -r .VLLM_NEURON_CI_HF_TOKEN)

NEURON_COMPILE_CACHE_URL="$(realpath ~)/neuron_compile_cache"
mkdir -p "${NEURON_COMPILE_CACHE_URL}"
NEURON_COMPILE_CACHE_MOUNT="/root/.cache/neuron_compile_cache"

# Try building the docker image
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws

# prune old image and containers to save disk space, and only once a day
# by using a timestamp file in tmp.
if [ -f /tmp/neuron-docker-build-timestamp ]; then
    last_build=$(cat /tmp/neuron-docker-build-timestamp)
    current_time=$(date +%s)
    if [ $((current_time - last_build)) -gt 86400 ]; then
        # Remove dangling images (those that are not tagged and not used by any container)
        docker image prune -f
        # Remove unused volumes / force the system prune for old images as well.
        docker volume prune -f && docker system prune -f
        echo "$current_time" > /tmp/neuron-docker-build-timestamp
    fi
else
    date "+%s" > /tmp/neuron-docker-build-timestamp
fi

docker build -t "${image_name}" -f docker/Dockerfile.neuron .

# Setup cleanup
remove_docker_container() {
    docker image rm -f "${image_name}" || true;
}
trap remove_docker_container EXIT

# Run the image
docker run --rm -it --device=/dev/neuron0 --network bridge \
       -v "${HF_CACHE}:${HF_MOUNT}" \
       -e "HF_HOME=${HF_MOUNT}" \
       -e "HF_TOKEN=${HF_TOKEN}" \
       -v "${NEURON_COMPILE_CACHE_URL}:${NEURON_COMPILE_CACHE_MOUNT}" \
       -e "NEURON_COMPILE_CACHE_URL=${NEURON_COMPILE_CACHE_MOUNT}" \
       --name "${container_name}" \
       ${image_name} \
       /bin/bash -c "
            set -e; # Exit on first error
            python3 /workspace/vllm/examples/offline_inference/neuron.py;
            python3 -m pytest /workspace/vllm/tests/neuron/1_core/ -v --capture=tee-sys;
            for f in /workspace/vllm/tests/neuron/2_core/*.py; do
                echo \"Running test file: \$f\";
                python3 -m pytest \$f -v --capture=tee-sys;
            done
       "