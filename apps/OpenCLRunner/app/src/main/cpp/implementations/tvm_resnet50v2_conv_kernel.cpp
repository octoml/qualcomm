#include "tvm_resnet50v2_conv_kernel.h"

#include <chrono>
#include <string>
#include <vector>

namespace {
    struct InputShape {
        int b;
        int c;
        int h;
        int w;
        int bc;
    };
    struct FilterShape {
        int och;
        int ich;
        int h;
        int w;
        int bc;
    };
}

ExecTime tvm_resnet50v2_conv_kernel(JNIEnv* env, jobject assetManager) {
    std::string kernelName = "octoml_tvm.cl";
    // pad, placeholder1, output, weights, bias
    InputShape is = {1, 128, 7, 7, 4}; // Output
    FilterShape fs = {128, 512, 3, 3, 4};
    std::vector<float> input(is.b * is.h * is.w * is.c * is.bc);
    std::vector<float> filter(fs.och * fs.ich * fs.h * fs.w * fs.bc);
    std::vector<float> bias(fs.och * fs.bc);
    std::vector<float> mul(fs.och * fs.bc);
    size_t gws0[3] = {10368, 1, 1};
    size_t lws0[3] = {32, 1, 1};
    size_t gws1[3] = {7, 1, 64};
    size_t lws1[3] = {7, 1, 8};

    cl_device_id device_id;
    cl_context context;
    cl_command_queue command_queue;
    int err;
    prepareOpenCLDevice(device_id, context, command_queue);

    std::string kernelSource = readKernel(env, assetManager, kernelName);
    const char* str = kernelSource.c_str();

    // ============ CREATE OpenCL IMAGES ============
    cl_image_format format;             // structure to define image format
    format.image_channel_data_type = CL_HALF_FLOAT;
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

    cl_mem pad_img = clCreateImage(
            context,
            CL_MEM_READ_WRITE,
            &format,
            &desc,
            NULL,
            &err);
    assert(err == CL_SUCCESS);

    // filter
    desc.image_width = fs.ich * fs.h * fs.w; // cout%4 * cin
    desc.image_height = fs.och; // kh * kw * cout/4
    cl_mem filter_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            filter.data(),
            &err);
    assert(err == CL_SUCCESS);

    // bias
    desc.image_width = fs.och;
    desc.image_height = 1;
    cl_mem bias_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            bias.data(),
            &err);
    assert(err == CL_SUCCESS);

    // mult
    desc.image_width = fs.och;
    desc.image_height = 1;
    cl_mem mul_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            mul.data(),
            &err);
    assert(err == CL_SUCCESS);

    // output
    desc.image_width = is.w;
    desc.image_height = is.h * is.c * is.b; // h * b
    cl_mem output_img = clCreateImage(
            context,
            CL_MEM_WRITE_ONLY,
            &format,
            &desc,
            NULL,
            &err);
    assert(err == CL_SUCCESS);

    cl_program program = clCreateProgramWithSource(context, 1,  &str, NULL, &err);
    assert(err == CL_SUCCESS);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    err = clBuildProgramWrapper(program, 1, &device_id);
    assert(err == CL_SUCCESS);
    cl_kernel kernel0 = clCreateKernel(program, "fused_nn_conv2d_multiply_add_nn_relu_16_kernel0", &err);
    assert(err == CL_SUCCESS);
    cl_kernel kernel1 = clCreateKernel(program, "fused_nn_conv2d_multiply_add_nn_relu_16_kernel1", &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&pad_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel0, 1, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    cl_event event0;
    err = clEnqueueNDRangeKernel(command_queue, kernel0, 1, NULL, gws0, lws0, 0, NULL, &event0);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event0);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&pad_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)&filter_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&output_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&mul_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&bias_img);
    assert(err == CL_SUCCESS);


    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel1, 3, NULL, gws1, lws1, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);
    auto cpuEnd = std::chrono::high_resolution_clock::now();


    cl_ulong time_start;
    cl_ulong time_end;
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event0, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double kernelTimeMS = (time_end - time_start);

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    kernelTimeMS += (time_end - time_start); // from ns to ms
    kernelTimeMS *= 1e-6;
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseEvent(event);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(pad_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(filter_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(mul_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(bias_img);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel0);
    assert(err == CL_SUCCESS);
    err = clReleaseKernel(kernel1);
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
