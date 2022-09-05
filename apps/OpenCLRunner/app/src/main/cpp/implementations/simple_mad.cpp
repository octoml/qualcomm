#include "simple_mad.h"
#include <common_functions.h>

#include <chrono>

#include <android/log.h>

ExecTime simple_mad(JNIEnv* env, jobject assetManager)
{
    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    int err;
    prepareOpenCLDevice(device_id, context, command_queue);

    std::string kernelSource = readKernel(env, assetManager, "octoml_simple.cl");
    const char* str = kernelSource.c_str();

    const int rows = 10000;
    const int cols = 1000;
    const int iters = 1000;
    const int bufSize = rows * cols;
    // Create buffers
    float *a_arr = new float[bufSize];
    float *b_arr = new float[bufSize];
    float *c_arr = new float[bufSize];

    // Create and print arrays
    /*for (int i(0); i < bufSize; ++i) {
        a_arr[i] = i + 0.5f;
        b_arr[i] = 10*i + 0.5f;
    }*/

    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufSize * sizeof(float), a_arr, &err);
    assert(err == CL_SUCCESS);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufSize * sizeof(float), b_arr, &err);
    assert(err == CL_SUCCESS);
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufSize * sizeof(float), NULL, &err);
    assert(err == CL_SUCCESS);

    cl_program program = clCreateProgramWithSource(context, 1,  &str, NULL, &err);
    assert(err == CL_SUCCESS);

    auto cpuStart = std::chrono::high_resolution_clock::now();

    //err = clBuildProgram(program, 1, &device_id, "-g -s vectorAdd.cl", NULL, NULL);
    err = clBuildProgramWrapper(program, 1, &device_id);
    assert(err == CL_SUCCESS);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "simple_mad", &err);
    assert(err == CL_SUCCESS);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&rows);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 1, sizeof(int), (void *)&cols);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&a_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&b_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&c_mem);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 5, sizeof(int), (void *)&iters);
    assert(err == CL_SUCCESS);

    // Run kernel
    size_t global_work_size[2] = { rows, cols / 4 }; // Define global size of execution
    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);
    auto cpuEnd = std::chrono::high_resolution_clock::now();
    // Read buffer with result of calculation
    err = clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0, bufSize * sizeof(float), c_arr, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    std::string res = "{";
    /*for (int i(0); i < bufSize; ++i) {
        res += std::to_string(c_arr[i]) + ", ";
    }*/
    res += "}";

    __android_log_print(ANDROID_LOG_DEBUG, "OpenCL Runner Answer", "Vector Add answer : %s", res.c_str());

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double kernelTimeMS = (time_end - time_start)   * 1e-6; // from ns to ms
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseEvent(event);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(a_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(b_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(c_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel);
    assert(err == CL_SUCCESS);
    err = clReleaseCommandQueue(command_queue);
    assert(err == CL_SUCCESS);
    err = clReleaseProgram(program);
    assert(err == CL_SUCCESS);
    err = clReleaseContext(context);
    assert(err == CL_SUCCESS);
    err = clReleaseDevice(device_id);
    assert(err == CL_SUCCESS);

    delete [] a_arr;
    delete [] b_arr;
    delete [] c_arr;
    return {cpuTimeMS, kernelTimeMS};
}
