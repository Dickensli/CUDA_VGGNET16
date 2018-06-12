/*
 * conv.h
 * Copyright (C) 2018-06-01 Hanxiao <hah114@ucsd.edu>
 *
 * Distributed under terms of the MIT license.
 */

#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Conv Structure 1
 * <<<(x0,y0), (x1, y1)>>>
 * each kernel calculates whole layer
 * each thread calculates one fixed position for all filters
*/
__global__ void ConvLayer1(float *gInputLayer, float *gOutputLayer, float *gWeights, float *bias, int inputDepth, int inputSize, int outputDepth)
{
    int padding = 1;
    int filterSize = 3;
    float features = 0;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < inputSize && y < inputSize)
    {
        for (int iterOutDepth = 0; iterOutDepth < outputDepth; iterOutDepth++)
        {
            float res = 0;
            for (int iterInDepth = 0; iterInDepth < inputDepth; iterInDepth++)
            {
                float *gInputOffsetLayer = gInputLayer + iterInDepth * inputSize * inputSize;
                float *gWeightsOffsetLayer = gWeights + iterOutDepth * inputDepth * filterSize * filterSize + iterInDepth * filterSize * filterSize;
                for (int i = x - padding; i <= x + padding; i++)
                {
                    for (int j = y - padding; j <= y + padding; j++)
                    {
                        int filterIdx = i - x + padding;
                        int filterIdy = j - y + padding;
                        float *gInputOffsetPosition = gInputOffsetLayer + i * inputSize + j;
                        float *gWeightsOffsetPosition = gWeightsOffsetLayer + filterIdx * filterSize + filterIdy;
                        bool outBoundaryJudge = (i >= 0 && i < inputSize && j >= 0 && j < inputSize);
                        float featureOnePoint = outBoundaryJudge ? gInputOffsetPosition[0] : 0;
                        float weightOnePoint = outBoundaryJudge ? gWeightsOffsetPosition[0] : 0;
                        res += featureOnePoint * weightOnePoint;
                    }
                }
            }
            res += bias[iterOutDepth];
            // Relu
            res = res < 0 ? 0 : res;
            gOutputLayer[iterOutDepth * inputSize * inputSize + x * inputSize + y] = res;
        }
    }
}

/*
 * Conv Structure 2
 * <<<x0, (x1, y1)>>>
 * each kernel calculates whole layer
 * each threadblock calculates one filter
 * each thread calculates 16 fixed position for 1 filter
*/
__global__ void ConvLayer2(float *gInputLayer, float *gOutputLayer, float *gWeights, float *bias, int inputDepth, int inputSize, int outputDepth)
{
    int padding = 1;
    int filterSize = 3;
    float features = 0;

	int x = threadIdx.x;
	int y = threadIdx.y;
    
    int filterId = blockIdx.x;

    float res = 0;
    if (x < inputSize && y < inputSize)
    {
        for (int posX = 0; x + posX < inputSize; posX += 32)
        {
            for (int posY = 0; y + posY < inputSize; posY += 32)
            {
                res = 0;
                float *gInputOffsetLayer = gInputLayer + (posX) * inputSize + posY;
                for (int iterInDepth = 0; iterInDepth < inputDepth; iterInDepth++)
                {
                    float *gInputOffsetLayerPos = gInputOffsetLayer + iterInDepth * inputSize * inputSize;
                    float *gWeightsOffsetLayer = gWeights + filterId * inputDepth * filterSize * filterSize + iterInDepth * filterSize * filterSize;
                    for (int i = x - padding; i <= x + padding; i++)
                    {
                        for (int j = y - padding; j <= y + padding; j++)
                        {
                            int filterIdx = i - x + padding;
                            int filterIdy = j - y + padding;
                            float *gInputOffsetPosition = gInputOffsetLayerPos + i * inputSize + j;
                            float *gWeightsOffsetPosition = gWeightsOffsetLayer + filterIdx * filterSize + filterIdy;
                            bool outBoundaryJudge = (i + posX >= 0 && i + posX < inputSize && j + posY >= 0 && j + posY < inputSize);
                            float featureOnePoint = outBoundaryJudge ? gInputOffsetPosition[0] : 0;
                            float weightOnePoint = outBoundaryJudge ? gWeightsOffsetPosition[0] : 0;
                            res += featureOnePoint * weightOnePoint;
                        }
                    }
                }
                res += bias[filterId];
                // Relu
                res = res < 0 ? 0 : res;
                gOutputLayer[filterId * inputSize * inputSize + (x + posX) * inputSize + y + posY] = res;
            }
        }
    }
}

/*
 * Conv Structure 3
 * <<<x0, x1>>>
 * x0: SMX
 * x1: inputDepth
 * each threadblock calculates one filter
 * each thread calculates 16 fixed position for 1 filter
*/
__global__ void ConvLayer3(float *gInputLayer, float *gOutputLayer, float *gWeights, float *bias, int inputDepth, int inputSize, int outputDepth)
{    
    int padding = 1;
    int filterSize = 3;
    float features = 0;
    int SMX = gridDim.x;

	int x = threadIdx.x;

    extern __shared__ float total[]; //dynamically allocation, size = 3 * 3 * inputDepth + inputDepth / 32

    float *oneFilterWeight = &total[0]; //size = 3 * 3 * inputDepth
    float *warpReduce = &total[filterSize * filterSize * inputDepth]; //size = 16

    float *gInputOffsetLayer = gInputLayer + x;
    for (int filterId = blockIdx.x; filterId < outputDepth; filterId += SMX)
    {
        float *gWeightsOffsetLayer = gWeights + filterId * filterSize * filterSize * inputDepth + x * filterSize * filterSize;

        //read to shared memory
        for (int iterRead = 0; iterRead < filterSize * filterSize; iterRead++)
            oneFilterWeight[iterRead + x * filterSize * filterSize] = gWeightsOffsetLayer[iterRead];

        float a00, a01, a02, a10, a11, a12, a20, a21, a22;
        float res = 0;
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                res = 0;
                a00 = (i == 0 || j == 0) ? 0 : gInputOffsetLayer[((i - 1) * inputSize + j - 1) * inputDepth];
                a01 = (i == 0) ? 0 : gInputOffsetLayer[((i - 1) * inputSize + j) * inputDepth];
                a02 = (i == 0 || j == inputSize - 1) ? 0 : gInputOffsetLayer[((i - 1) * inputSize + j + 1) * inputDepth];
                a10 = (j == 0) ? 0 : gInputOffsetLayer[(i * inputSize + j - 1) * inputDepth];
                a11 = gInputOffsetLayer[(i * inputSize + j) * inputDepth];
                a12 = (j == inputSize - 1) ? 0 : gInputOffsetLayer[(i * inputSize + j + 1) * inputDepth];
                a20 = (i == inputSize - 1 || j == 0) ? 0 : gInputOffsetLayer[((i + 1) * inputSize + j - 1) * inputDepth];
                a21 = (i == inputSize - 1) ? 0 : gInputOffsetLayer[((i + 1) * inputSize + j) * inputDepth];
                a22 = (i == inputSize - 1 || j == inputSize - 1) ? 0 : gInputOffsetLayer[((i + 1) * inputSize + j + 1) * inputDepth];

                res += a00 * oneFilterWeight[x * filterSize * filterSize + 0];
                res += a01 * oneFilterWeight[x * filterSize * filterSize + 1];
                res += a02 * oneFilterWeight[x * filterSize * filterSize + 2];
                res += a10 * oneFilterWeight[x * filterSize * filterSize + 3];
                res += a11 * oneFilterWeight[x * filterSize * filterSize + 4];
                res += a12 * oneFilterWeight[x * filterSize * filterSize + 5];
                res += a20 * oneFilterWeight[x * filterSize * filterSize + 6];
                res += a21 * oneFilterWeight[x * filterSize * filterSize + 7];
                res += a22 * oneFilterWeight[x * filterSize * filterSize + 8];

                //reduce inside warp
                res += __shfl_down(res, 16);
                res += __shfl_down(res, 8);
                res += __shfl_down(res, 4);
                res += __shfl_down(res, 2);
                res += __shfl_down(res, 1);

                if (x % 32 == 0)
                    warpReduce[x / 32] = res;

                __syncthreads();

                //reduce different warps
                if (x / 32 == 0)
                {
                    res = x < ceil(inputDepth / 32.0) ? warpReduce[x % 32] : 0;
                    res += __shfl_down(res, 8);
                    res += __shfl_down(res, 4);
                    res += __shfl_down(res, 2);
                    res += __shfl_down(res, 1);

                    if (x == 0)
                    {
                        res += bias[filterId];
                        res = res < 0 ? 0 : res;
                        gOutputLayer[filterId * inputSize * inputSize + i * inputSize + j] = res;
                    }
                }
            }
        }
    }
}

/*
 * Conv Structure 3 (with bank conflict)
 * <<<x0, x1>>>
 * x0: SMX
 * x1: inputDepth
 * each threadblock calculates one filter
 * each thread calculates 16 fixed position for 1 filter
*/
__global__ void ConvLayer3_1(float *gInputLayer, float *gOutputLayer, float *gWeights, float *bias, int inputDepth, int inputSize, int outputDepth)
{    
    int padding = 1;
    int filterSize = 3;
    float features = 0;
    int SMX = gridDim.x;

	int x = threadIdx.x;

    extern __shared__ float total[]; //dynamically allocation, size = 3 * 3 * inputDepth + inputDepth / 32

    float *oneFilterWeight = &total[0]; //size = 3 * 3 * inputDepth
    float *warpReduce = &total[filterSize * filterSize * inputDepth]; //size = 16

    float *gInputOffsetLayer = gInputLayer + x * inputSize * inputSize;
    for (int filterId = blockIdx.x; filterId < outputDepth; filterId += SMX)
    {
        float *gWeightsOffsetLayer = gWeights + filterId * filterSize * filterSize * inputDepth + x * filterSize * filterSize;

        //read to shared memory
        for (int iterRead = 0; iterRead < filterSize * filterSize; iterRead++)
            oneFilterWeight[iterRead + x * filterSize * filterSize] = gWeightsOffsetLayer[iterRead];

        float a00, a01, a02, a10, a11, a12, a20, a21, a22;
        float res = 0;
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                res = 0;
                a00 = (i == 0 || j == 0) ? 0 : gInputOffsetLayer[(i - 1) * inputSize + j - 1];
                a01 = (i == 0) ? 0 : gInputOffsetLayer[(i - 1) * inputSize + j];
                a02 = (i == 0 || j == inputSize - 1) ? 0 : gInputOffsetLayer[(i - 1) * inputSize + j + 1];
                a10 = (j == 0) ? 0 : gInputOffsetLayer[i * inputSize + j - 1];
                a11 = gInputOffsetLayer[i * inputSize + j];
                a12 = (j == inputSize - 1) ? 0 : gInputOffsetLayer[i * inputSize + j + 1];
                a20 = (i == inputSize - 1 || j == 0) ? 0 : gInputOffsetLayer[(i + 1) * inputSize + j - 1];
                a21 = (i == inputSize - 1) ? 0 : gInputOffsetLayer[(i + 1) * inputSize + j];
                a22 = (i == inputSize - 1 || j == inputSize - 1) ? 0 : gInputOffsetLayer[(i + 1) * inputSize + j + 1];

                res += a00 * oneFilterWeight[x * filterSize * filterSize + 0];
                res += a01 * oneFilterWeight[x * filterSize * filterSize + 1];
                res += a02 * oneFilterWeight[x * filterSize * filterSize + 2];
                res += a10 * oneFilterWeight[x * filterSize * filterSize + 3];
                res += a11 * oneFilterWeight[x * filterSize * filterSize + 4];
                res += a12 * oneFilterWeight[x * filterSize * filterSize + 5];
                res += a20 * oneFilterWeight[x * filterSize * filterSize + 6];
                res += a21 * oneFilterWeight[x * filterSize * filterSize + 7];
                res += a22 * oneFilterWeight[x * filterSize * filterSize + 8];

                //reduce inside warp
                res += __shfl_down(res, 16);
                res += __shfl_down(res, 8);
                res += __shfl_down(res, 4);
                res += __shfl_down(res, 2);
                res += __shfl_down(res, 1);

                if (x % 32 == 0)
                    warpReduce[x / 32] = res;

                __syncthreads();

                //reduce different warps
                if (x / 32 == 0)
                {
                    res = x < ceil(inputDepth / 32.0) ? warpReduce[x % 32] : 0;
                    res += __shfl_down(res, 8);
                    res += __shfl_down(res, 4);
                    res += __shfl_down(res, 2);
                    res += __shfl_down(res, 1);

                    if (x == 0)
                    {
                        res += bias[filterId];
                        res = res < 0 ? 0 : res;
                        gOutputLayer[filterId * inputSize * inputSize + i * inputSize + j] = res;
                    }
                }
            }
        }
    }
}

/*
 * Conv Structure 4
 * <<<x0, (x1, y1)>>>
 * x0 = SM number
 * x1, y1 = 32, 32
 * shared memory version of ConvLayer2
*/
__global__ void ConvLayer4(float *gInputLayer, float *gOutputLayer, float *gWeights, float *bias, int inputDepth, int inputSize, int outputDepth)
{
    int padding = 1;
    int filterSize = 3;
    int SMX = gridDim.x;

    extern __shared__ float total[]; //dynamically allocation, size = 3 * 3 * inputDepth
    float *oneFilterWeight = &total[0]; //size = 3 * 3 * inputDepth

	int x = threadIdx.x;
	int y = threadIdx.y;

//     if (x == 0 && y == 0 && blockIdx.x == 0)
//     {
//         for (int k = 0; k < 32; k++)
//             printf("%0.1f ", gInputLayer[k]);
//         printf("\n");
//     }
//     for (int filterId = blockIdx.x; filterId < outputDepth; filterId += SMX)
    int filterId = blockIdx.x;
    while (filterId < outputDepth)
    {
        float *gWeightsOffsetLayer = gWeights + filterId * filterSize * filterSize * inputDepth;

        //read to shared memory
        for (int iterRead = x * 32 + y; iterRead < filterSize * filterSize * inputDepth; iterRead += 1024)
            oneFilterWeight[iterRead] = gWeightsOffsetLayer[iterRead];
        __syncthreads();

        if (x < inputSize && y < inputSize)
        {
            for (int posX = 0; x + posX < inputSize; posX += 32)
            {
                for (int posY = 0; y + posY < inputSize; posY += 32)
                {
                    float res = 0;
                    for (int iterInDepth = 0; iterInDepth < inputDepth; iterInDepth++)
                    {
                        for (int i = x - padding; i <= x + padding; i++)
                        {
                            for (int j = y - padding; j <= y + padding; j++)
                            {
                                int filterIdx = i - x + padding;
                                int filterIdy = j - y + padding;
                                bool outBoundaryJudge = (i + posX >= 0 && i + posX < inputSize && j + posY >= 0 && j + posY < inputSize);
                                float featureOnePoint = outBoundaryJudge ? gInputLayer[iterInDepth * inputSize * inputSize + (i + posX) * inputSize + j + posY] : 0;
                                float weightOnePoint = oneFilterWeight[iterInDepth * filterSize * filterSize + filterIdx * filterSize + filterIdy];
                                res += featureOnePoint * weightOnePoint;
                            }
                        }
                    }
                    res += bias[filterId];
                    // Relu
                    res = res < 0 ? 0 : res;
                    gOutputLayer[filterId * inputSize * inputSize + (x + posX) * inputSize + y + posY] = res;
                }
            }
        }
        filterId += SMX;
        __syncthreads();
     }
}
