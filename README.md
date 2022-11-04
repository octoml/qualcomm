# Qualcomm Adreno TVM Evaluation Repo

Last version of TVM this was evaluated on and worked (09/26/2022): `46ea2ed42ee41225141c5ed522900d340b08944d`.

Questions of issues using the scripts? Submit a ticket via the OctoML [helpdesk](https://octoml.atlassian.net/servicedesk/customer/portal/6).

## Testing model performance with texture memory:
In the table below you can see the performance numbers (inference time in
milliseconds) which were achieved on the [Realme GT 5G](https://www.gsmarena.com/realme_gt_5g-10689.php).

|                      | mace_mobilenetv1_nchw | mace_resnet50_v2 | mace_inceptionv3 | mxnet_vgg16 | mace_deeplabv3 | mace_yolov3 |
|----------------------|-----------------------|------------------|------------------|-------------|----------------|-------------|
| TVM textures FP16    |                  4,82 |            35,89 |            39,07 |       56,37 |          58,05 |      171,93 |
| TVM textures FP16a32 |                  5,15 |            38,12 |            40,18 |       65,14 |          61,85 |      192,15 |
| TVM textures FP32    |                  7,62 |            58,61 |            59,74 |        96,9 |           85,5 |      276,61 |

The tuning log files are located in [logs/](logs/). You
can use the `evaluate.py` script for reproducing these numbers. Copy the name of
the model from the table and use the relevant log file with tuned statistic.
Below, you can see examples of run mobilenetv1:
```
# float16 compute, float16 accumulate
python ./evaluate.py -m mace_mobilenetv1_nchw -t float16 -k android --target="opencl --device=adreno" -l ./logs/mace_mobilenetv1_nchw.texture.float16.acc16.autotvm.log

# float16 compute, float32 accumulate
python ./evaluate.py -m mace_mobilenetv1_nchw -t float16_acc32 -k android --target="opencl --device=adreno" -l ./logs/mace_mobilenetv1_nchw.texture.float16.acc32.autotvm.log

# float32 inference
python ./evaluate.py -m mace_mobilenetv1_nchw -t float32 -k android --target="opencl --device=adreno" -l ./logs/mace_mobilenetv1_nchw.texture.float32.autotvm.log
```
Refer to the below instructions for running the `scripts/evaluate.py` script for more information

## Setting up the host development machine

On the host machine (typically your development box) you'll need to build TVM. 

```
git clone https://github.com/apache/tvm.git --recursive
cd tvm
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

You also can use `launch_rpc.sh` script for running `tvm_rpc` on your device. To
get more information about how to use this script, run the following command:
```
./launch_rpc.sh --help
```

Typical usage is:
```
./launch_rpc.sh -d <device_hash> -k android -t <tracker_ip>:<tracker_port>
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

A python script `evaluate.py` can evaluate or tune a set of models.

Usage for the script, you can get with:

```
$ python3 scripts/evaluate.py -h
```

## Helper applications
In directory [apps/](apps/) there are two applications which can be used for
profiling separate OpenCL kernels:
- `OpenCLRunner` is an Android GUI application which can be used to run OpenCL
    kernels on Android device and collect performance metrics.

- `OpenCLCPPRunner` is an native application which can be used to run OpenCL
    kernels on Android device or on a host and collect performance metrics.
