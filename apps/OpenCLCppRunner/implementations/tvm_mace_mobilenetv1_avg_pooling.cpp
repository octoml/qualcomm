#include "tvm_mace_mobilenetv1_avg_pooling.h"

#include <chrono>
#include <string>
#include <vector>
#include <iostream>

namespace {
    struct InputShape {
        int b;
        int c;
        int h;
        int w;
        int bc;
    };
}

ExecTime tvm_mace_mobilenetv1_avg_pooling_kernel() {
    std::string kernelName = "tvm_mace_mobilenetv1_avg_pooling.cl";
    // pad, placeholder1, output, weights, bias
    InputShape is = {1, 256, 7, 7, 4}; // Output
    std::vector<float> input(is.b * is.h * is.w * is.c * is.bc);
    size_t gws0[3] = {256, 1, 1};
    size_t lws0[3] = {256, 1, 1};

    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    int err;
    prepareOpenCLDevice(device_id, context, command_queue);

    std::string kernelSource = readKernel(kernelName);
    const char* str = kernelSource.c_str();

    // ============ CREATE OpenCL IMAGES ============
    cl_image_format format;             // structure to define image format
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_RGBA;

    // init image description
    //cl_image_desc desc = { CL_MEM_OBJECT_IMAGE2D, is.w, is.h, 0, 0, 0, 0, 0, 0 };
    cl_image_desc desc = { 0 };               // structure to define image description
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = is.w;
    desc.image_height = is.h * is.c * is.b; // h * b

    // input
    cl_mem input_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            input.data(),
            &err);
    assert(err == CL_SUCCESS);

    // output
    cl_float *out_arr = new cl_float[is.c * is.bc];
    cl_mem out_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                is.c * is.bc * sizeof(cl_float), out_arr, &err);
    assert(err == CL_SUCCESS);

    cl_program program = clCreateProgramWithSource(context, 1,  &str, NULL, &err);
    assert(err == CL_SUCCESS);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    err = clBuildProgramWrapper(program, 1, &device_id);
    std::cout << "clBuildProgramWrapper ret: " <<  err << std::endl;
    assert(err == CL_SUCCESS);
    cl_kernel kernel0 = clCreateKernel(program, "fused_nn_avg_pool2d_1_kernel0", &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void *)&out_mem);
    assert(err == CL_SUCCESS);
    cl_event event0;
    err = clEnqueueNDRangeKernel(command_queue, kernel0, 1, NULL, gws0, lws0, 0, NULL, &event0);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event0);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernelTimeMS = (time_end - time_start);

    kernelTimeMS *= 1e-6;
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseMemObject(input_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(out_mem);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel0);
    assert(err == CL_SUCCESS);
    err = clReleaseCommandQueue(command_queue);
    assert(err == CL_SUCCESS);
    err = clReleaseProgram(program);
    assert(err == CL_SUCCESS);
    err = clReleaseContext(context);
    assert(err == CL_SUCCESS);
    err = clReleaseDevice(device_id);
    assert(err == CL_SUCCESS);

    return {cpuTimeMS, kernelTimeMS};
}

