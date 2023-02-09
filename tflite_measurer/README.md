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
   ./run_benchmark.sh ./models/mobilenet-v1-1.0.tflite
   ```
