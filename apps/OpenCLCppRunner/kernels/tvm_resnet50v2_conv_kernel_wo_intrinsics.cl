/*
Kernel params: fused_nn_conv2d_multiply_add_nn_relu_16_kernel0
work_dim: 1
gws: 10368x1x1
lws: 32x1x1
Kernel params!
Kernel params: fused_nn_conv2d_multiply_add_nn_relu_16_kernel1
work_dim: 3
gws: 7x1x64
lws: 7x1x8
Kernel params!
*/


#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

__kernel void fused_nn_conv2d_multiply_add_nn_relu_16_kernel0(__write_only image2d_t pad_temp_texture, __read_only image2d_t placeholder0) {
  half4 _1 = read_imageh(placeholder0, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9) - 1), ((((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) / 81) * 7) + ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81) / 9)) - 1)));
  (void)write_imageh(pad_temp_texture, (int2)((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9), (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) / 9)), (((((9 <= (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81)) && ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 81) < 72)) && (1 <= (((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9))) && ((((((int)get_group_id(0)) * 32) + ((int)get_local_id(0))) % 9) < 8)) ? _1 : ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
}


#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

__kernel void fused_nn_conv2d_multiply_add_nn_relu_16_kernel1(__read_only image2d_t pad_temp_texture, __read_only image2d_t placeholder1, __write_only image2d_t compute, __read_only image2d_t placeholder2, __read_only image2d_t placeholder3) {
  half4 compute1[14];
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 0);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 28);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 4);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 32);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 8);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 36);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 12);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 40);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 16);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 44);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 20);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 48);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 24);
  vstore4(((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f)), 0, (half*)compute1 + 52);
  for (int rc_inner = 0; rc_inner < 128; ++rc_inner) {
    for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
      for (int rx_inner = 0; rx_inner < 3; ++rx_inner) {
        for (int rc = 0; rc < 4; ++rc) {
          half4 _1 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), ((rc_inner * 9) + ry_inner)));
          half4 _2 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((rc_inner * 36) + (rc * 9)) + (ry_inner * 3)) + rx_inner), ((((int)get_group_id(2)) * 16) + ((int)get_local_id(2)))));
          vstore4((vload4(0, (half*)compute1 + 0) + (((half*)&_1)[rc] * _2)), 0, (half*)compute1 + 0);
          half4 _3 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((rc_inner * 36) + (rc * 9)) + (ry_inner * 3)) + rx_inner), (((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8)));
          vstore4((vload4(0, (half*)compute1 + 28) + (((half*)&_1)[rc] * _3)), 0, (half*)compute1 + 28);
          half4 _4 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 1)));
          vstore4((vload4(0, (half*)compute1 + 4) + (((half*)&_4)[rc] * _2)), 0, (half*)compute1 + 4);
          vstore4((vload4(0, (half*)compute1 + 32) + (((half*)&_4)[rc] * _3)), 0, (half*)compute1 + 32);
          half4 _5 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 2)));
          vstore4((vload4(0, (half*)compute1 + 8) + (((half*)&_5)[rc] * _2)), 0, (half*)compute1 + 8);
          vstore4((vload4(0, (half*)compute1 + 36) + (((half*)&_5)[rc] * _3)), 0, (half*)compute1 + 36);
          half4 _6 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 3)));
          vstore4((vload4(0, (half*)compute1 + 12) + (((half*)&_6)[rc] * _2)), 0, (half*)compute1 + 12);
          vstore4((vload4(0, (half*)compute1 + 40) + (((half*)&_6)[rc] * _3)), 0, (half*)compute1 + 40);
          half4 _7 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 4)));
          vstore4((vload4(0, (half*)compute1 + 16) + (((half*)&_7)[rc] * _2)), 0, (half*)compute1 + 16);
          vstore4((vload4(0, (half*)compute1 + 44) + (((half*)&_7)[rc] * _3)), 0, (half*)compute1 + 44);
          half4 _8 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 5)));
          vstore4((vload4(0, (half*)compute1 + 20) + (((half*)&_8)[rc] * _2)), 0, (half*)compute1 + 20);
          vstore4((vload4(0, (half*)compute1 + 48) + (((half*)&_8)[rc] * _3)), 0, (half*)compute1 + 48);
          half4 _9 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_inner), (((rc_inner * 9) + ry_inner) + 6)));
          vstore4((vload4(0, (half*)compute1 + 24) + (((half*)&_9)[rc] * _2)), 0, (half*)compute1 + 24);
          vstore4((vload4(0, (half*)compute1 + 52) + (((half*)&_9)[rc] * _3)), 0, (half*)compute1 + 52);
        }
      }
    }
  }
  half4 _10 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _11 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), ((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7))), max(((vload4(0, (half*)compute1 + 0) * _10) + _11), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _12 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _13 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 56)), max(((vload4(0, (half*)compute1 + 28) * _12) + _13), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _14 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _15 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 1)), max(((vload4(0, (half*)compute1 + 4) * _14) + _15), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _16 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _17 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 57)), max(((vload4(0, (half*)compute1 + 32) * _16) + _17), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _18 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _19 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 2)), max(((vload4(0, (half*)compute1 + 8) * _18) + _19), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _20 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _21 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 58)), max(((vload4(0, (half*)compute1 + 36) * _20) + _21), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _22 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _23 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 3)), max(((vload4(0, (half*)compute1 + 12) * _22) + _23), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _24 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _25 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 59)), max(((vload4(0, (half*)compute1 + 40) * _24) + _25), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _26 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _27 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 4)), max(((vload4(0, (half*)compute1 + 16) * _26) + _27), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _28 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _29 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 60)), max(((vload4(0, (half*)compute1 + 44) * _28) + _29), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _30 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _31 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 5)), max(((vload4(0, (half*)compute1 + 20) * _30) + _31), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _32 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _33 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 61)), max(((vload4(0, (half*)compute1 + 48) * _32) + _33), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _34 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  half4 _35 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 6)), max(((vload4(0, (half*)compute1 + 24) * _34) + _35), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
  half4 _36 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  half4 _37 = read_imageh(placeholder3, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((((int)get_group_id(2)) * 16) + ((int)get_local_id(2))) + 8), 0));
  (void)write_imageh(compute, (int2)(((int)get_local_id(0)), (((((int)get_group_id(2)) * 112) + (((int)get_local_id(2)) * 7)) + 62)), max(((vload4(0, (half*)compute1 + 52) * _36) + _37), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
}
