# Qualcomm TVM Evaluation Repo

Last version of TVM this was evaluated on and worked (10/1/2020): `c549239b9248371ca3eeabcd99e05e5e406fd43e`

Questions of issues using the scripts? Submit a ticket via the OctoML [helpdesk](https://octoml.atlassian.net/servicedesk/customer/portal/6).


## Setting up the host development machine

On the host machine (typically your development box) you'll need to build TVM. 

```
git clone https://github.com/apache/incubator-tvm.git --recursive
cd incubator-tvm
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_LLVM llvm-config)' >> build/config.cmake
echo 'set(USE_GRAPH_RUNTIME_DEBUG ON)' >> build/config.cmake
cd build
cmake ..
make -j8
```

## Cross compiling the C++ RPC server for Android

Refer to the documentation [here](https://github.com/apache/incubator-tvm/tree/master/apps/cpp_rpc) to cross compile the C++ RPC binary and tvm_runtime libraries for Android.

To run, use `adb` to push the cross compiled `tvm_rpc` binary and `libtvm_runtime.so` shared library to `/data/local/tmp` on the Android device. Then run the RPC server with:
```
adb shell
cd /data/local/tmp
LD_LIBRARY_PATH=. ./tvm_rpc server --tracker=<tracker IP>:<tracker port> --key=android
```

## Setting up the RPC device tracker

Once TVM is built on the host, you'll need to launch the RPC tracker service with the following command:
```
python -m tvm.exec.rpc_tracker --host=<tracker IP> --port=<tracker port> --port-end=9192
```
Where `tracker IP` is the host IP, and `tracker port` can be `9191`.

When done, you can register the Android device on the tracker with the same key used to run the on device RPC server:

```
python -m tvm.exec.rpc_server --tracker <tracker host>:<tracker port> --key android
```

Finally, make sure that the hardware is properly registered to the tracker. On the host, or any machine connected to the local network, check the devices registered on the tracker with the following command:

```
python -m tvm.exec.query_rpc_tracker --host <tracker IP> --port <tracker port>
```

## Using the experiment script

Under `scripts` you'll find a python script `evaluate.py` that can evaluate or tune a set of models:

Below is the usage for the script, which you can get with 

```
$ python3 scripts/evaluate.py -h

usage: evaluate.py [-h] -m
                   {resnet50,mobilenetv1,inceptionv3,vgg16,mobilenetv3_ssdlite,deeplabv3}
                   [-t {float32,float16}] [-l LOG] [-k RPC_KEY]
                   [-r RPC_TRACKER_HOST] [-p RPC_TRACKER_PORT] [-T TARGET]
                   [--tune TUNE] [--debug DEBUG]

Tune and/or evaluate a curated set of models

optional arguments:
  -h, --help            show this help message and exit
  -m {resnet50,mobilenetv1,inceptionv3,vgg16,mobilenetv3_ssdlite,deeplabv3}, --model {resnet50,mobilenetv1,inceptionv3,vgg16,mobilenetv3_ssdlite,deeplabv3}
                        Model to tune and/or evaluate
  -t {float32,float16}, --type {float32,float16}
                        Specify whether the model should be run with single or
                        half precision floating point values
  -l LOG, --log LOG     AutoTVM tuning logfile name
  -k RPC_KEY, --rpc_key RPC_KEY
                        RPC key to use
  -r RPC_TRACKER_HOST, --rpc_tracker_host RPC_TRACKER_HOST
                        RPC tracker host IP address
  -p RPC_TRACKER_PORT, --rpc_tracker_port RPC_TRACKER_PORT
                        RPC tracker host port
  -T TARGET, --target TARGET
                        Compilation target
  --tune TUNE           Whether or not to run autotuning
  --debug DEBUG         Use graph runtime debugger to output per layer perf.
                        data and other statistics
```

## Known issues ##
Currently running with `-m deeplabv3 -t float16` will produce an internal invariant violation in TVM. This is known and under investigation.
