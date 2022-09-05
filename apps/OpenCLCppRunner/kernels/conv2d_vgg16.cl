#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
#endif

// Work_size: 115200x1x1x64x1x1x
__kernel void fused_nn_conv2d_add_nn_relu_1_kernel0(__write_only image2d_t pad_temp_texture, __read_only image2d_t placeholder0) {
  half4 _1 = read_imageh(placeholder0, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 30) - 1), ((((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) / 900) * 28) + ((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 900) / 30)) - 1)));
  (void)write_imageh(pad_temp_texture, (int2)((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 30), (((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) / 30)), (((((30 <= (((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 900)) && ((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 900) < 870)) && (1 <= (((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 30))) && ((((((int)get_group_id(0)) * 64) + ((int)get_local_id(0))) % 30) < 29)) ? _1 : ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))));
}

// Work_size: 14x14x128x14x1x32x
__kernel void fused_nn_conv2d_add_nn_relu_1_kernel1(__read_only image2d_t pad_temp_texture, __read_only image2d_t placeholder1, __global half* restrict compute, __read_only image2d_t placeholder2) {
  float4 compute1[4];
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)compute1 + 0);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)compute1 + 8);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)compute1 + 4);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)compute1 + 12);
  for (int rc_outer = 0; rc_outer < 128; ++rc_outer) {
    for (int rx_outer = 0; rx_outer < 3; ++rx_outer) {
      for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
        for (int rc = 0; rc < 4; ++rc) {
          half4 _1 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_outer), (((rc_outer * 30) + (((int)get_group_id(1)) * 2)) + ry_inner)));
          half4 _2 = read_imageh(placeholder1, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((rc_outer * 36) + (rc * 9)) + (ry_inner * 3)) + rx_outer), ((((int)get_group_id(2)) * 32) + ((int)get_local_id(2)))));
          vstore4((vload4(0, (float*)compute1 + 0) + (convert_float4((((half*)&_1)[rc] * _2)))), 0, (float*)compute1 + 0);
          half4 _3 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)((((int)get_local_id(0)) + rx_outer), ((((rc_outer * 30) + (((int)get_group_id(1)) * 2)) + ry_inner) + 1)));
          vstore4((vload4(0, (float*)compute1 + 8) + (convert_float4((((half*)&_3)[rc] * _2)))), 0, (float*)compute1 + 8);
          half4 _4 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_local_id(0)) + rx_outer) + 14), (((rc_outer * 30) + (((int)get_group_id(1)) * 2)) + ry_inner)));
          vstore4((vload4(0, (float*)compute1 + 4) + (convert_float4((((half*)&_4)[rc] * _2)))), 0, (float*)compute1 + 4);
          half4 _5 = read_imageh(pad_temp_texture, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_local_id(0)) + rx_outer) + 14), ((((rc_outer * 30) + (((int)get_group_id(1)) * 2)) + ry_inner) + 1)));
          vstore4((vload4(0, (float*)compute1 + 12) + (convert_float4((((half*)&_5)[rc] * _2)))), 0, (float*)compute1 + 12);
        }
      }
    }
  }
  half4 _6 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 32) + ((int)get_local_id(2))), 0));
  vstore4(max(((convert_half4(vload4(0, (float*)compute1 + 0))) + _6), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))), 0, compute + ((((((int)get_group_id(2)) * 100352) + (((int)get_local_id(2)) * 3136)) + (((int)get_group_id(1)) * 224)) + (((int)get_local_id(0)) * 4)));
  half4 _7 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 32) + ((int)get_local_id(2))), 0));
  vstore4(max(((convert_half4(vload4(0, (float*)compute1 + 8))) + _7), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))), 0, compute + (((((((int)get_group_id(2)) * 100352) + (((int)get_local_id(2)) * 3136)) + (((int)get_group_id(1)) * 224)) + (((int)get_local_id(0)) * 4)) + 112));
  half4 _8 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 32) + ((int)get_local_id(2))), 0));
  vstore4(max(((convert_half4(vload4(0, (float*)compute1 + 4))) + _8), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))), 0, compute + (((((((int)get_group_id(2)) * 100352) + (((int)get_local_id(2)) * 3136)) + (((int)get_group_id(1)) * 224)) + (((int)get_local_id(0)) * 4)) + 56));
  half4 _9 = read_imageh(placeholder2, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(((((int)get_group_id(2)) * 32) + ((int)get_local_id(2))), 0));
  vstore4(max(((convert_half4(vload4(0, (float*)compute1 + 12))) + _9), ((half4)((half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f, (half)0.000000e+00f))), 0, compute + (((((((int)get_group_id(2)) * 100352) + (((int)get_local_id(2)) * 3136)) + (((int)get_group_id(1)) * 224)) + (((int)get_local_id(0)) * 4)) + 168));
}
