# Qualcomm Adreno TVM Evaluation Repo

***Disclaimer: This is a development repository, texture memory support is currently in the upstreaming process and has been cut from the TVM subtree contained herein. See the TVM discuss forum [RFC](https://discuss.tvm.apache.org/t/rfc-texture-memory-support/9467) for more information and links to relevant PRs.

Last version of TVM this was evaluated on and worked (01/28/2021): `4abbe4902e451cc5a963b8b60a70e548d48ace62`.

For testing texture memory support, please use the tvm repository included as a subtree in this repository: [tvm](https://github.com/octoml/qualcomm/tree/master/tvm).



Questions of issues using the scripts? Submit a ticket via the OctoML [helpdesk](https://octoml.atlassian.net/servicedesk/customer/portal/6).

## Testing model performance with texture memory:
In the table below you can see the performance numbers (inference time in
milliseconds) which were achieved on the [Realme GT 5G](https://www.gsmarena.com/realme_gt_5g-10689.php).

|                      | mace_mobilenetv1 | mace_resnet50_v2 | mace_inceptionv3 | vgg16  | mace_deeplabv3 | mace_yolov3 |
|----------------------|------------------|------------------|------------------|--------|----------------|-------------|
| TVM textures FP16    |              4,8 |            27,46 |            44,45 |  78,96 |         103,99 |      175,09 |
| TVM textures FP16a32 |             5,42 |            28,51 |            44,31 |  96,64 |         110,32 |      242,34 |
| TVM textures FP32    |             7,66 |            40,56 |            69,87 | 131,99 |         154,06 |      306,27 |

The tuning log files are located in [logs/mace_models/](logs/mace_models/). You
can use the `evaluate.py` script for reproducing these numbers. Copy the name of
the model from the table and use the relevant log file with tuned statistic.
Below, you can see examples of run mobilenetv1:
```
# float16 compute, float16 accumulate
python ./scripts/evaluate.py -m mace_mobilenetv1 -t float16 -k android --target="opencl --device=adreno" -l ./logs/mace_models/mace_mobilenetv1.texture.float16.acc16.autotvm.log

# float16 compute, float32 accumulate
python ./scripts/evaluate.py -m mace_mobilenetv1 -t float16 -k android --target="opencl --device=adreno" -l ./logs/mace_models/mace_mobilenetv1.texture.float16.acc32.autotvm.log

# float32 inference
python ./scripts/evaluate.py -m mace_mobilenetv1 -t float32 -k android --target="opencl --device=adreno" -l ./logs/mace_models/mace_mobilenetv1.texture.float32.acc32.autotvm.log
```
Refer to the below instructions for running the `scripts/evaluate.py` script for more information

## Running texture.py tests:
`scripts/texture.py` is a set of compute and schedule definitions for various workloads employing texture memory cache stage when the `-m "texture"` argument is supplied. For each test, numerical comparisons are checked against numpy results. Some of the tests can be tuned with the `--tune` flag. Log files with autotvm tuning records exist in the logs/ directory for many these tunable tests. See the below for a few invocation examples on how to run a tuned schedule with texture memory.

```
usage: scripts/texture.py [-h] [-m MEMORY] [-s] [-l LOG] [-T] -t TEST
                  [-r RPC_TRACKER_HOST] [-p RPC_TRACKER_PORT] [-k RPC_KEY]

Set test arguments

optional arguments:
  -h, --help            show this help message and exit
  -m MEMORY, --memory MEMORY
                        Use global or texture
  -s, --shared          Use shared memory
  -l LOG, --log LOG     AutoTVM tuning record logfile
  -T, --tune            Whether to tune or not
  -t TEST, --test TEST  Selected test to run
  -r RPC_TRACKER_HOST, --rpc_tracker_host RPC_TRACKER_HOST
                        RPC tracker host IP address
  -p RPC_TRACKER_PORT, --rpc_tracker_port RPC_TRACKER_PORT
                        RPC tracker host port
  -k RPC_KEY, --rpc_key RPC_KEY
                        RPC key to use

```
Example invocations,
```
# ------------------------
# Conv2d VGG16 layer [3x3]
# ------------------------

# Memory hierarchy: shared->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune2 -l logs/conv2d_NCHWc_KCRSk_tx_tune2.autotvm.shared.log
> 115.4 GFLOPS

# Memory hierarchy: texture->shared->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune2 -l logs/conv2d_NCHWc_KCRSk_tx_tune2.texture.shared.autotvm.best.log -m texture -s
> 116.9 GFLOPS

# Memory hierarchy: texture->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune2 -m texture -l logs/conv2d_NCHWc_KCRSk_tx_tune2.texture.noshared.autotvm.log
> 147.6 GFLOPS

# ------------------------------
# Conv2d MobilenetV1 layer [1x1]
# ------------------------------

# Memory hierarchy: shared->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune -l logs/conv2d_NCHWc_KCRSk_tx_tune_1024.log -s
> 100.2 GFLOPS

# Memory hierarchy: texture->shared->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune -l logs/conv2d_NCHWc_KCRSk_tx_tune_1024.log -s -m "texture"
> 89.2 GFLOPS

# Memory hierarchy: texture->local
$ python scripts/texture.py -r 0.0.0.0 -p 9191 -k android --test=conv2d_NCHWc_KCRSk_tx_tune -l logs/conv2d_NCHWc_KCRSk_tx_tune.texture.noshared.log -m "texture"
> 137.5 GFLOPS

```


## Setting up the host development machine

On the host machine (typically your development box) you'll need to build TVM. 

```
git clone https://github.com/octoml/qualcomm --recursive
cd qualcomm/tvm
mkdir build
cp cmake/config.cmake build/.
echo 'set(USE_LLVM llvm-config)' >> build/config.cmake
echo 'set(USE_GRAPH_RUNTIME_DEBUG ON)' >> build/config.cmake
cd build
cmake ..
make -j8
cd ..
export TVM_HOME=$PWD
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${TVM_HOME}/build:$LD_LIBRARY_PATH
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

