#define WITH_INTRIN 1

#if WITH_INTRIN == 1
#define MAD(A, B, C) \
    mad(A, B, C)
#else
#define MAD(A, B, C) \
    A * B + C
#endif

/*
// ======================= 1 =======================
// Time wo intrinsics: 48.54 ms
// Time with intrinsics: 41.24 ms
__kernel void simple_mad(  const int rows,
         const int cols,
         __global float* matrixA,
         __global float* matrixB,
         __global float* MatrixSum,
         const int iters)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    // Utilize built-in function to calculate index offset
    int offset = mul24(j, cols);
    int index = mad24(i, 4, offset);

    float4 tmpA = (*((__global float4*)&matrixA[index]));
    float4 tmpB = (*((__global float4*)&matrixB[index]));
    for (int i = 0; i < iters; ++i) {
        tmpA = MAD(tmpA, tmpA, tmpB);
        tmpA = MAD(tmpB, tmpA, tmpB);
    }
    (*((__global float4*)&MatrixSum[index])) = MAD(tmpA, tmpA, tmpB);
}
*/
/*
// ======================= 2 =======================
// Time wo intrinsics: 48.53 ms
// Time with intrinsics: 41.24 ms
__kernel void simple_mad(  const int rows,
         const int cols,
         __global float* matrixA,
         __global float* matrixB,
         __global float* MatrixSum,
         const int iters)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    // Utilize built-in function to calculate index offset
    int offset = mul24(j, cols);
    int index = mad24(i, 4, offset);

    float4 tmpA, tmpB;
    vstore4(vload4(0, matrixA + index), 0, (float*)&tmpA);
    vstore4(vload4(0, matrixB + index), 0, (float*)&tmpB);
    for (int i = 0; i < iters; ++i) {
        vstore4(MAD(tmpA, tmpA, tmpB), 0, (float*)&tmpA);
        vstore4(MAD(tmpB, tmpA, tmpB), 0, (float*)&tmpA);
    }
    vstore4((float4)MAD(tmpA, tmpA, tmpB), 0, MatrixSum + index);
}
*/

// ======================= 3 =======================
// Time wo intrinsics: 41.15 ms
// Time with intrinsics: 43.68 ms
__kernel void simple_mad(  const int rows,
         const int cols,
         __global float* matrixA,
         __global float* matrixB,
         __global float* MatrixSum,
         const int iters)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    // Utilize built-in function to calculate index offset
    int offset = mul24(j, cols);
    int index = mad24(i, 4, offset);

    float4 tmpA[2];
    float4 tmpB[2];
    tmpA[0] = (*((__global float4*)&matrixA[index]));
    tmpA[1] = (*((__global float4*)&matrixA[index + 1]));
    tmpB[0] = (*((__global float4*)&matrixB[index]));
    tmpB[1] = (*((__global float4*)&matrixB[index + 1]));
    for (int i = 0; i < iters; ++i) {
        tmpA[0] = MAD(tmpA[0], tmpA[0], tmpB[0]);
        tmpA[1] = MAD(tmpB[1], tmpA[1], tmpB[1]);
    }
    (*((__global float4*)&MatrixSum[index])) = MAD(tmpA[0], tmpA[0], tmpB[0]);
    (*((__global float4*)&MatrixSum[index + 1])) = MAD(tmpA[1], tmpA[1], tmpB[1]);
}

/*
// ======================= 4 =======================
// Time wo intrinsics: 1015.72 ms
// Time with intrinsics: 1031.35 ms
__kernel void simple_mad(  const int rows,
         const int cols,
         __global float* matrixA,
         __global float* matrixB,
         __global float* MatrixSum,
         const int iters)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    // Utilize built-in function to calculate index offset
    int offset = mul24(j, cols);
    int index = mad24(i, 4, offset);

    float4 tmpA[2];
    float4 tmpB[2];
    vstore4(vload4(0, matrixA + index), 0, (float*)tmpA + 0);
    vstore4(vload4(0, matrixA + index + 1), 0, (float*)tmpA + 1);
    vstore4(vload4(0, matrixB + index), 0, (float*)tmpB + 0);
    vstore4(vload4(0, matrixB + index + 1), 0, (float*)tmpB + 1);
    for (int i = 0; i < iters; ++i) {
        vstore4(MAD(tmpA[0], tmpA[0], tmpB[0]), 0, (float*)tmpA + 0);
        vstore4(MAD(tmpB[1], tmpA[1], tmpB[1]), 0, (float*)tmpA + 1);
    }
    vstore4((float4)MAD(tmpA[0], tmpA[0], tmpB[0]), 0, MatrixSum + index);
    vstore4((float4)MAD(tmpA[1], tmpA[1], tmpB[1]), 0, MatrixSum + index + 1);
}
*/
