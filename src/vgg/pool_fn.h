#include "../common/common.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

/*
 * pool kernel 1
 * <<<(ceil(MATROW/8.), 8), 32>>>
 * each warp(8 * 4) oprerates on 8(4 * 2) sliding windows 
*/
__global__ void pool_1(float* in, float* out, uint32_t lrow, uint32_t lcol){
    uint32_t in_size = lrow * lcol; 
    uint32_t onetime_vol = gridDim.y * lrow * 4;
    uint32_t iwx = threadIdx.x % 8;
    uint32_t iwy = threadIdx.x / 8;
    uint32_t off_in = blockIdx.y * lrow * 4 + blockIdx.x * 8; 
    uint32_t off_out = blockIdx.y * lrow + blockIdx.x * 4; 
    if( blockIdx.x * 8 + iwx >= lrow){
        return;
    }      

    for(uint32_t iter = 0 ; iter < in_size/onetime_vol ; iter++){
        uint32_t total_off_in = off_in + iter * onetime_vol;
        uint32_t total_off_out = off_out + iter * onetime_vol / 4;
        float val = in[total_off_in + lrow * iwy + iwx];
        // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && iter == 0){
        //     printf("in[%d] = %f\n", total_off_in + lrow * iwy + iwx, in[total_off_in + lrow * iwy + iwx]);
        // }
        // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 1 && iter == 0){
        //     printf("in[%d] = %f\n", total_off_in + lrow * iwy + iwx, in[total_off_in + lrow * iwy + iwx]);
        // }
        // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 8 && iter == 0){
        //     printf("in[%d] = %f\n", total_off_in + lrow * iwy + iwx, in[total_off_in + lrow * iwy + iwx]);
        // }
        // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 9 && iter == 0){
        //     printf("in[%d] = %f\n", total_off_in + lrow * iwy + iwx, in[total_off_in + lrow * iwy + iwx]);
        // }
        

        float pval = __shfl_down(val, 1);
        val = val > pval? val : pval;
        pval = __shfl_down(val, 8);
        if(!(iwx & 0x1) && !(iwy & 0x1)){
            out[total_off_out + lrow/2 * iwy/2 + iwx/2] = val > pval? val : pval;
            // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && iter == 0){
            //     printf("iwx = %d\n", iwx);
            //     printf("iwy = %d\n", iwy);
            //     printf("total_off_out = %d\n", total_off_out);
            //     printf("out[%d] = %f\n", total_off_out + lrow/2 * iwy/2 + iwx/2, out[total_off_out + lrow/2 * iwy/2 + iwx/2]);
            // }
        }
    }
}

/*
 * pool kernel 2
 * <<<(ceil(MATROW/8.), ceil(MATROW/8.)), (8, 8)>>>
 * each thread calculates max per sliding window
*/
__global__ void pool_2(float* in, float* out,  uint32_t lrow, uint32_t lcol){
    uint32_t in_size = lrow * lcol; 
    uint32_t onetime_vol = gridDim.y * lrow * blockDim.y*2;
    uint32_t wx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    uint32_t wy = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    uint32_t off_in = wy * lrow + wx;  
    uint32_t off_out = wy * lrow / 4 + wx / 2;  
    if( (blockIdx.x * blockDim.x + threadIdx.x)*2 >= lrow)
        return;
    // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0){
    //     printf("in[%d] = %f\n", 0, in[0]);
    //     printf("in[%d] = %f\n", 1, in[1]);
    //     printf("in[%d] = %f\n", 224, in[224]);
    //     printf("in[%d] = %f\n", 225, in[225]);
    // }

    for(uint32_t iter = 0 ; iter < in_size/onetime_vol ; iter++){
        uint32_t total_off_in = off_in + iter * onetime_vol;
        uint32_t total_off_out = off_out + iter * onetime_vol / 4;
        float val1 = in[total_off_in];
        float val2 = in[total_off_in + 1];
        float val3 = in[total_off_in + lrow];
        float val4 = in[total_off_in + lrow + 1];
        out[total_off_out] = max(max(max(val1, val2), val3), val4);
        // if(blockIdx.y == 0 && blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && iter == 0){
        //     printf("out[%d] = %f\n", 0, out[0]);
        // }
    }
}

/*
 * fulConn kernel
 * <<<ceil(MATROW/32), 32>>>
 * each thread calculates one column of kernel times input
*/
__global__ void fn(float* in, float* out, float* kernel_weights, float* kernel_bias, uint32_t lrow, uint32_t lcol){
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lid = threadIdx.x;

    if(gid >= lrow)
        return;

    extern __shared__ float sout[];
    sout[lid] = 0;
    for(uint32_t iter = 0 ; iter < lcol ; iter++){
        sout[lid] += kernel_weights[lrow * iter + gid] * in[iter];
    }
    sout[lid] += kernel_bias[gid];
    out[gid] = sout[lid];
}

/*
 * prediction
 * <<<1, 1000>>>
*/
__global__ void predict(float* in, uint32_t* pred){
    uint32_t gid = threadIdx.x;

    __shared__ float cache[1024];
    __shared__ uint32_t idx[1024];
    cache[gid] = in[gid];
    idx[gid] = gid;
    if(gid < 24){
        cache[1000 + gid] = 0;
        idx[1000 + gid] = 1000 + gid;
    }
    __syncthreads();
    int i = blockDim.x / 2;
    while(i != 0){
        if(gid < i){
            if(cache[gid] < cache[gid + i]){
                cache[gid] = cache[gid + i];
                idx[gid] = idx[gid + i];
            }
        }
        __syncthreads();
        i = i / 2;
    }
    if(gid == 0)
        *pred = idx[0];
}