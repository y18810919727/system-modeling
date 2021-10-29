#!/usr/bin/env bash
docker build -t ai3d/uc_control .
docker run --rm --gpus all -it -p 6010:6010 --name uc_test -e algorithm=test -e cuda=1 ai3d/uc_control