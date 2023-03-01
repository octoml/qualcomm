# TFLite models benchmarking

## TFLite building

1. Clone TensorFLow: `git clone https://github.com/tensorflow/tensorflow.git`
2. Prepare for build: `cd tensorflow && mkdir tflite_android_build && cd tflite_android_build`
3. Modify cmake configuration for flatbuffers. Change `GIT_TAG` version from `2.0.6` to `2.0.8` here: `tensorflow/lite/tools/cmake/modules/flatbuffers.cmake`
4. Clone TensorFlow repository:
   ``` bash
   cmake -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_NATIVE_API_LEVEL=android-23 -DANDROID_STL=c++_static -DTFLITE_ENABLE_GPU=ON ..
   ```
5. Run make:
   ```bash
   make -j4
   make benchmark_model
   ```

## Run benchmarking

1. Push `benchmark_model` binary to device:
   ```bash
   adb push ./tools/benchmark/benchmark_model /data/local/tmp
   ```
2. Download model by using `download_models.py`:
   ```bash
   python3 download_models.py mobilenet
   ```
3. Run the benchmarking:
   ```bash
   ./run_benchmark.sh -m ./models/mobilenet-v1-1.0.tflite -t FP16
   ```

## Connect to the remote adb
1. Forward local `5037` port to remote machine: `ssh -L 5037:localhost:5037 -N username@host -p port`
2. Try to run `adb devices`. If there are no errors then congrats, you have
   forwarded `adb` from remote machine to local. If you see similar error:
   ```
   adb server version (39) doesn't match this client (41); killing...
   ```
   then go to the next step.
3. Copy `adb` from remote machine to local. You can find `adb` location by the
   following command: `which adb`
4. Try to run new `adb` executor locally. If there are any errors that some
   libraries were not found than go to the remote machine and run
   `ldd /path/to/adb`. It will print path to used libraries and copy them to the
   local machine into the directory with `adb` execution file.
5. Finally `adb devices` should print list of the devices which are attached to
   remote machine.

## Collect GPU per-layer statistic
0. If you use `adb` which was copied from a remote machine (as described in
   section "Connect to the remote adb"), then add directory with `adb` file to
   `PATH` and `LD_LIBRARY_PATH` environment variables.
1. Go to the directory with source code of TensorFlow.
2. Run the following command:
   ```
   tensorflow/lite/delegates/gpu/cl/testing/run_performance_profiling.sh -m /path/to/tflite/model.tflite -d <device_hash> -t FP16
   ```
