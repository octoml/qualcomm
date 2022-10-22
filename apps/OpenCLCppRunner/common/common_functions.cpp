#include "common_functions.h"

#include <iostream>
#include <fstream>
#include <sstream>

std::string readKernel(const std::string& name) {
    std::ifstream file("../kernels/" + name);
    if (!file.is_open()) {
        file.open(name);
        if (!file.is_open()) {
            std::cerr << "Error! Cannot read kernel: " << name << std::endl;
            return "";
        }
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

void printDevInfo(cl_device_id& device_id) {
    std::cout << "======================================================" << std::endl;
    size_t valueSize;
    int err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* devName = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, devName, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_NAME: " << devName << std::endl;
    delete [] devName;

    err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* devVersion = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, valueSize, devVersion, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_VERSION: " << devVersion << std::endl;
    delete [] devVersion;

    err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* driverVersion = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, valueSize, driverVersion, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DRIVER_VERSION: " << driverVersion << std::endl;
    delete [] driverVersion;
    std::cout << "***" << std::endl;

    cl_uint clUintVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: " << clUintVal << std::endl;
    std::cout << "***" << std::endl;

    err = clGetDeviceInfo(device_id, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MEM_BASE_ADDR_ALIGN: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS: " << clUintVal << std::endl;
    std::cout << "***" << std::endl;

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << clUintVal << std::endl;
    size_t max_wis[3];
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wis) * 3, max_wis, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << max_wis[0] << "x" << max_wis[1] << "x" << max_wis[2] << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << valueSize << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_SAMPLERS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_SAMPLERS: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_READ_IMAGE_ARGS: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_PARAMETER_SIZE: " << valueSize << std::endl;
    cl_ulong clUlongVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clUlongVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clUlongVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: " << clUlongVal << std::endl;

    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE3D_MAX_WIDTH: " << valueSize << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE3D_MAX_HEIGHT: " << valueSize << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE3D_MAX_DEPTH: " << valueSize << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE2D_MAX_WIDTH: " << valueSize << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(valueSize), &valueSize, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE2D_MAX_HEIGHT: " << valueSize << std::endl;
    cl_bool clBoolVal;
    err = clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(clBoolVal), &clBoolVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_IMAGE_SUPPORT: " << clBoolVal << std::endl;

    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: " << clUlongVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(clUintVal), &clUintVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: " << clUintVal << std::endl;
    err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(clUlongVal), &clUlongVal, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: " << clUlongVal << std::endl;
    std::cout << "***" << std::endl;

    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
    assert(err == CL_SUCCESS);
    char* ext = new char[valueSize];
    err = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, valueSize, ext, NULL);
    assert(err == CL_SUCCESS);
    std::cout << "CL_DEVICE_EXTENSIONS: " << ext << std::endl;
    delete [] ext;
    std::cout << "======================================================" << std::endl;
}

void prepareOpenCLDevice(cl_device_id& device_id, cl_context& ctx, cl_command_queue& cq, bool printDeviceInfo) {
    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;

    int err = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    assert(err == CL_SUCCESS);
    cl_platform_id* platforms = new cl_platform_id[ret_num_platforms];
    err = clGetPlatformIDs(ret_num_platforms, platforms, &ret_num_platforms);
    assert(err == CL_SUCCESS);
    for (int i = 0; i < ret_num_platforms; ++i) {
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
        if (err == CL_SUCCESS)
            break;
    }
    if (err != CL_SUCCESS) {
        std::cerr << "WARNING! Cannot find platform with GPU. Using CPU instead!" << std::endl;
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        assert(err == CL_SUCCESS);
    }

    // Create context
    ctx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    cq = clCreateCommandQueue(ctx, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    assert(err == CL_SUCCESS);
    delete [] platforms;

    if (printDeviceInfo) {
        printDevInfo(device_id);
    }
}

std::string measureExecTime(Executor exec, unsigned int repeat) {
    double cpuTime = 0;
    double kernelTime = 0;
    for (int i = 0; i < repeat; ++i) {
        auto time = exec();
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
        std::cerr << "clBuildProgramWrapper, error: " << log << std::endl;
    }
    return err;
}

