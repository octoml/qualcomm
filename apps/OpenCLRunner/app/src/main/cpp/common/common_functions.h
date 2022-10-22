#ifndef OPENCL_RUNNER_COMMON_FUNCTIONS_H
#define OPENCL_RUNNER_COMMON_FUNCTIONS_H

#include <cassert>
#include <functional>
#include <jni.h>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct ExecTime {
    double cpuTime;
    double kernelTime;
};
using Executor = std::function<ExecTime(JNIEnv* env, jobject assetManager)>;

std::string readKernel(JNIEnv* env, jobject assetManager, const std::string& name);
void prepareOpenCLDevice(cl_device_id& device_id, cl_context& ctx, cl_command_queue& cq, bool printDeviceInfo = false);
std::string measureExecTime(Executor exec, JNIEnv* env, jobject assetManager, unsigned int repeat = 10);
int clBuildProgramWrapper(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options = NULL);

#endif //OPENCL_RUNNER_COMMON_FUNCTIONS_H
