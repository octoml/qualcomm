#include "mace_inceptionv3_kernel.h"

#include <chrono>
#include <string>
#include <vector>

namespace {
    struct InputShape {
        int b;
        int h;
        int w;
        int c;
    };
    struct FilterShape {
        int och;
        int ich;
        int h;
        int w;
    };
}

ExecTime mace_inceptionv3_kernel(JNIEnv* env, jobject assetManager) {
    std::string kernelName = "mace_inceptionv3_kernel.cl";
    std::string buildOptions = "-DBIAS -DCMD_DATA_TYPE=f -DDATA_TYPE=float -DNON_UNIFORM_WORK_GROUP -DUSE_RELU";
    InputShape is = {1, 147, 147, 32};
    FilterShape fs = {64, 32, 3, 3};
    std::vector<float> input(is.b * is.h * is.w * is.c);
    std::vector<float> filter(fs.och * fs.ich * fs.h * fs.w);
    std::vector<float> bias(fs.och);
    float * input_ptr = new float[is.b * is.h * is.w * is.c];
    for (size_t i = 0; i < is.b * is.h * is.w * is.c; ++i) {
        input_ptr[i] = i + 0.5f;
    }
    size_t gws[3] = {16, 30, 147};
    size_t lws[3] = {4, 30, 8};


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
    cl_image_desc desc = { 0 };               // structure to define image description
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    // TODO: Am I sure about order of w and h in the kernel?
    //desc.image_width =  is.c%4 * is.w * is.c/4; // c%4 * w * c/4
    desc.image_width = is.w * is.c / 4; //2352;
    desc.image_height = is.h * is.b; // h * b

    // input
    cl_mem input_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            input_ptr,
            &err);
    assert(err == CL_SUCCESS);

    // filter
    desc.image_width = fs.ich; // cout%4 * cin
    desc.image_height = fs.och/4 * fs.h * fs.w; // kh * kw * cout/4
    cl_mem filter_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            filter.data(),
            &err);
    assert(err == CL_SUCCESS);

    // bias
    desc.image_width = fs.och/4;
    desc.image_height = 1;
    cl_mem bias_img = clCreateImage(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            &format,
            &desc,
            bias.data(),
            &err);
    assert(err == CL_SUCCESS);

    // output
    desc.image_width = fs.och / 4 * is.w; // c%4 * w * c/4
    desc.image_height = is.h * is.b; // h * b
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
    err = clBuildProgramWrapper(program, 1, &device_id, buildOptions.c_str());
    assert(err == CL_SUCCESS);
    cl_kernel kernel = clCreateKernel(program, "conv_2d_3x3", &err);
    assert(err == CL_SUCCESS);


    int arg0 = gws[0];
    err = clSetKernelArg(kernel, 0, sizeof(int), (void *)&arg0);
    assert(err == CL_SUCCESS);
    int arg1 = gws[1];
    err = clSetKernelArg(kernel, 1, sizeof(int), (void *)&arg1);
    assert(err == CL_SUCCESS);
    int arg2 = gws[2];
    err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&arg2);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&input_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&filter_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&bias_img);
    assert(err == CL_SUCCESS);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&output_img);
    assert(err == CL_SUCCESS);
    float arg7 = 0.f;
    err = clSetKernelArg(kernel, 7, sizeof(float), (void *)&arg7);
    assert(err == CL_SUCCESS);
    float arg8 = 0.f;
    err = clSetKernelArg(kernel, 8, sizeof(float), (void *)&arg8);
    assert(err == CL_SUCCESS);
    int arg9 = is.h;
    err = clSetKernelArg(kernel, 9, sizeof(int), (void *)&arg9);
    assert(err == CL_SUCCESS);
    int arg10 = is.w;
    err = clSetKernelArg(kernel, 10, sizeof(int), (void *)&arg10);
    assert(err == CL_SUCCESS);
    int arg11 = 8;
    err = clSetKernelArg(kernel, 11, sizeof(int), (void *)&arg11);
    assert(err == CL_SUCCESS);
    int arg12 = is.h;
    err = clSetKernelArg(kernel, 12, sizeof(int), (void *)&arg12);
    assert(err == CL_SUCCESS);
    int arg13 = is.w;
    err = clSetKernelArg(kernel, 13, sizeof(int), (void *)&arg13);
    assert(err == CL_SUCCESS);
    int arg14 = 1;
    err = clSetKernelArg(kernel, 14, sizeof(int), (void *)&arg14);
    assert(err == CL_SUCCESS);
    int arg15 = 1;
    err = clSetKernelArg(kernel, 15, sizeof(int), (void *)&arg15);
    assert(err == CL_SUCCESS);
    int arg16 = 1;
    err = clSetKernelArg(kernel, 16, sizeof(int), (void *)&arg16);
    assert(err == CL_SUCCESS);
    int arg17 = 1;
    err = clSetKernelArg(kernel, 17, sizeof(int), (void *)&arg17);
    assert(err == CL_SUCCESS);
    int arg18 = 1;
    err = clSetKernelArg(kernel, 18, sizeof(int), (void *)&arg18);
    assert(err == CL_SUCCESS);
    int arg19 = 1;
    err = clSetKernelArg(kernel, 19, sizeof(int), (void *)&arg19);
    assert(err == CL_SUCCESS);

    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, gws, lws, 0, NULL, &event);
    assert(err == CL_SUCCESS);
    err = clWaitForEvents(1, &event);
    assert(err == CL_SUCCESS);
    err = clFinish(command_queue);
    assert(err == CL_SUCCESS);
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    double kernelTimeMS = (time_end - time_start) * 1e-6; // from ns to ms
    auto cpuTimeMS = std::chrono::duration_cast<std::chrono::nanoseconds>(cpuEnd - cpuStart).count() * 1e-6;

    err = clReleaseEvent(event);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(input_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(filter_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(bias_img);
    assert(err == CL_SUCCESS);
    err = clReleaseMemObject(output_img);
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

    return {cpuTimeMS, kernelTimeMS};
}
