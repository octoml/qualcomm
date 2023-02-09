#!/bin/bash

file_name=${1##*/}

adb push $1 /data/local/tmp
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/${file_name} --use_gpu=true
adb shell rm -f /data/local/tmp/${file_name}
