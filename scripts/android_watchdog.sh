#!/bin/bash

function watchdog() {
    # TODO: device_id=$1
    processes=`adb shell ps | grep org.apache.tvm.tvmrpc:RPCProcess | wc -l`
    if [ "$processes" -eq "0" ]; then
        adb shell am start -n org.apache.tvm.tvmrpc/org.apache.tvm.tvmrpc.MainActivity
    fi
    sleep 20
}

while true
do
    watchdog
done

