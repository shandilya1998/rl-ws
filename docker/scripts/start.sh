#!/bin/bash

name="$2"
image="$3"
$platform=$4
gpus="$5"
if [[ "$gpus" == "all_gpus" ]]; then
    gpus="all"
elif [[ "$gpus" == "no_gpus" ]]; then
    gpus=0
else
    gpus="$4"
fi
echo "Container name: $name"
echo "Image name: $image"
echo "GPU ID: $gpus"
echo "Workspace path: $1"
XAUTH=/tmp/.docker.xauth
docker run -d -it -v $1:/ws/ \
    -v ./.ssh:/root/.ssh \
    -v /dev:/dev \
    -v ./.gemini:/root/.gemini \
    -v ./.claude:/root/.claude \
    --gpus $gpus \
    --device-cgroup-rule "c 81:* rmw" \
    --device-cgroup-rule "c 189:* rmw" \
    --device /dev/vchiq \
    -v /run/udev:/run/udev:ro \
    -e DISPLAY=$DISPLAY \
    -e XAUTHORITY=$XAUTHORITY \
    -e QT_GRAPHICSSYSTEM=native \
    -v $XAUTH:$XAUTH:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="DISPLAY" --net=host \
    --privileged --name $name $image bash
