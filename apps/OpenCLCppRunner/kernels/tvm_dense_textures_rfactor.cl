#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

// Work_size: 1024x1x1x64x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel0(__write_only image2d_t input_pack_texture, __global half* restrict placeholder0) {
  (void)write_imageh(input_pack_texture, (int2)(((int)get_local_id(0)), ((int)get_group_id(0))), vload4(0, placeholder0 + ((((int)get_group_id(0)) * 256) + (((int)get_local_id(0)) * 4))));
}


// Work_size: 128x64x8x1x64x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel1(__read_only image2d_t input_pack_texture, __read_only image2d_t placeholder1, __global half* restrict compute) {
  __local half T_dense[4];
  half T_dense_rf[1];
  __local half red_buf0[64];
  for (int ax2 = 0; ax2 < 4; ++ax2) {
    T_dense_rf[(0)] = (half)0.000000e+00f;
    for (int in_h = 0; in_h < 4; ++in_h) {
      for (int in_w = 0; in_w < 4; ++in_w) {
        for (int kcb = 0; kcb < 4; ++kcb) {
          half4 _1 = read_imageh(input_pack_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((int)get_local_id(1)), ((in_h * 4) + in_w)));
          half4 _2 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((in_h * 1024) + (in_w * 256)) + (((int)get_local_id(1)) * 4)) + kcb), ((((int)get_group_id(0)) * 8) + ((int)get_group_id(2)))));
          T_dense_rf[(0)] = (T_dense_rf[(0)] + (((half*)&_1)[kcb] * ((half*)&_2)[ax2]));
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = T_dense_rf[(0)];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 32) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 32))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 16) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 16))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 8) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 8))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 4) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 4))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 2) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 2))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (((int)get_local_id(1)) < 1) {
      ((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] = (((volatile __local half*)red_buf0)[(((int)get_local_id(1)))] + ((volatile __local half*)red_buf0)[((((int)get_local_id(1)) + 1))]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    T_dense[(ax2)] = ((volatile __local half*)red_buf0)[(0)];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  vstore4(vload4(0, T_dense + 0), 0, compute + ((((int)get_group_id(0)) * 32) + (((int)get_group_id(2)) * 4)));
}

// Work_size: 1x4096x1x1x1x1x
__kernel void fused_nn_dense_add_nn_relu_1_kernel2(__global half* restrict compute, __global half* restrict compute1, __global half* restrict placeholder2) {
  compute[(((int)get_group_id(1)))] = max((half)(compute1[(((int)get_group_id(1)))] + placeholder2[(((int)get_group_id(1)))]), (half)(half)0.000000e+00f);
}

