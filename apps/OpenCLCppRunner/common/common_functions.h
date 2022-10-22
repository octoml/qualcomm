#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <cassert>
#include <functional>
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
using Executor = std::function<ExecTime()>;

std::string readKernel(const std::string& name);
void prepareOpenCLDevice(cl_device_id& device_id, cl_context& ctx, cl_command_queue& cq, bool printDeviceInfo = false);
std::string measureExecTime(Executor exec, unsigned int repeat = 10);
int clBuildProgramWrapper(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options = NULL);

#endif // COMMON_FUNCTIONS_H
