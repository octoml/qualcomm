#ifndef OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H
#define OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H

#include <jni.h>
#include <string>

#include <common_functions.h>

ExecTime mace_inceptionv3_kernel(JNIEnv* env, jobject assetManager);

#endif //OPENCL_RUNNER_MACE_INCEPTIONV3_KERNEL_H
