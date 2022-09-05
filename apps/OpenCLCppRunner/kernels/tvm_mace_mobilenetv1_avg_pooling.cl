__kernel void fused_nn_avg_pool2d_1_kernel0(__read_only image2d_t placeholder0, __global float* restrict compute) {
  float tensor[4];
  float4 _1;
  for (int ax4 = 0; ax4 < 4; ++ax4) {
    tensor[(ax4)] = 0.000000e+00f;
    for (int dh = 0; dh < 7; ++dh) {
      for (int dw = 0; dw < 7; ++dw) {
        //if (((int)get_local_id(0)) < 1024) {
          _1 = read_imagef(placeholder0, CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST, (int2)(dw, ((((int)get_local_id(0)) * 7) + dh)));
          tensor[(ax4)] = (tensor[(ax4)] + ((float*)&_1)[ax4]);
        //}
      }
    }
  }
  if (((int)get_local_id(0)) < 256) {
    vstore4((vload4(0, tensor + 0) * ((float4)(2.040816e-02f, 2.040816e-02f, 2.040816e-02f, 2.040816e-02f))), 0, compute + (((int)get_local_id(0)) * 4));
  }
}
