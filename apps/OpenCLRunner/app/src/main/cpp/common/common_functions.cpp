#include "common_functions.h"

#include <android/asset_manager_jni.h>
#include <android/log.h>

std::string readKernel(JNIEnv* env, jobject assetManager, const std::string& name) {
    std::string res;
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    AAsset* asset = AAssetManager_open(mgr, name.c_str(), AASSET_MODE_BUFFER);
    size_t assetLength = AAsset_getLength(asset);
    char* buffer = new char[assetLength+1];
    AAsset_read(asset, buffer, assetLength);
    AAsset_close(asset);
    buffer[assetLength] = '\0';
    res = buffer;
    delete [] buffer;

    return res;
}

void printDevInfo(cl_device_id& device_id) {
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "======================================================");
    size_t valueSize;
    int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* devName = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, devName, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_NAME: %s", devName);
    delete [] devName;

    err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* devVersion = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, valueSize, devVersion, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_VERSION: %s", devVersion);
    delete [] devVersion;

    err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* driverVersion = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, valueSize, driverVersion, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DRIVER_VERSION: %s", driverVersion);
    delete [] driverVersion;
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "***");

    cl_uint clUintVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %d", clUintVal);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "***");

    err = clGetDeviceInfo(device_id, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MEM_BASE_ADDR_ALIGN: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: %d", clUintVal);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "***");

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d", clUintVal);
    size_t max_wis[3];
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wis) * 3, max_wis, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_WORK_ITEM_SIZES: %dx%dx%d", max_wis[0], max_wis[1], max_wis[2]);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_WORK_GROUP_SIZE: %d", valueSize);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_SAMPLERS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_SAMPLERS: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_READ_IMAGE_ARGS: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_PARAMETER_SIZE: %d", valueSize);
    cl_ulong clUlongVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_MEM_ALLOC_SIZE: %d", clUlongVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %d", clUlongVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_CONSTANT_ARGS: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_COMPUTE_UNITS: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_MAX_CLOCK_FREQUENCY: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_LOCAL_MEM_SIZE: %d", clUlongVal);

    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE3D_MAX_WIDTH: %d", valueSize);
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE3D_MAX_HEIGHT: %d", valueSize);
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE3D_MAX_DEPTH: %d", valueSize);
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE2D_MAX_WIDTH: %d", valueSize);
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE2D_MAX_HEIGHT: %d", valueSize);
    cl_bool clBoolVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(clBoolVal), &clBoolVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_IMAGE_SUPPORT: %d", clBoolVal);

    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_GLOBAL_MEM_SIZE: %d", clUlongVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %d", clUintVal);
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %d", clUlongVal);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "***");

    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* ext = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, valueSize, ext, NULL);
    assert(err == CL_SUCCESS);
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "CL_DEVICE_EXTENSIONS: %s", ext);
    delete [] ext;
    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner DevInfo", "======================================================");
}

void prepareOpenCLDevice(cl_device_id& device_id, cl_context& ctx, cl_command_queue& cq, bool printDeviceInfo) {
    cl_platform_id platform_id;
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;

    int err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    assert(err == CL_SUCCESS);

    // Create context
    ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    cq = clCreateCommandQueue(ctx, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    assert(err == CL_SUCCESS);

    if (printDeviceInfo) {
        printDevInfo(device_id);
    }
}

std::string measureExecTime(Executor exec, JNIEnv* env, jobject assetManager, unsigned int repeat) {
    double cpuTime = 0;
    double kernelTime = 0;
    for (int i = 0; i < repeat; ++i) {
        auto time = exec(env, assetManager);
        cpuTime += time.cpuTime;
        kernelTime += time.kernelTime;
    }
    cpuTime /= repeat;
    kernelTime /= repeat;
    std::string str = "CPU: " + std::to_string(cpuTime) + ", OpenCL: " + std::to_string(kernelTime);
    return str;
}

int clBuildProgramWrapper(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options) {
    int err = clBuildProgram(program, num_devices, device_list, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        std::string log;
        clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        log.resize(len);
        clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
        __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner clBuildProgramWrapper", "Error: %s", log.c_str());
    }
    return err;
}