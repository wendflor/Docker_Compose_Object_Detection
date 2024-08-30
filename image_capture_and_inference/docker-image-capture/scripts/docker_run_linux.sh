#!/bin/bash

session_name=zed_image_capture
# kill old session if exists
tmux kill-session -t "${session_name}"
sleep 1

# start new tmux session
tmux new-session -s "${session_name}" 'docker run --gpus all -it --privileged --runtime nvidia --rm  \
--mount type=bind,source=${PWD},target=/app \
 zed_image_capture:latest'