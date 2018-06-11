/*
 * conv_unnitest.cu
 * Copyright (C) 2018-06-09 Hanxiao <hah114@ucsd.edu>
 *
 * Distributed under terms of the MIT license.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>
#include <cuda_runtime_api.h>
#include "conv.h"

using namespace std;

void PrintResult(float *res, int size, int dime, string name, float *gt, bool printFlag = false)
{
    float error = 0;
    cout << "--- " << name << " ---" << endl;
    for (int d = 0; d < dime * size * size; d += size * size)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                if (printFlag)
                    printf("%0.4f ", gt[d + i * size + j] - res[d + i * size + j]);
                float tmpRrror = abs(gt[d + i * size + j] - res[d + i * size + j]);
                error += tmpRrror;
            }
            if (printFlag)
                cout << endl;
        }
        if (printFlag)
            cout << "--------" << endl;
    }
    cout << "Error: " << error / (size * size * dime);
    cout << endl;
}

void CalGT(float *input, float *filter, float *bias, int size, int inDimention, int outDimention, float *res)
{
    for (int jD = 0; jD < outDimention; jD++)
    {
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                float tmpRes = 0;
                for (int iD = 0; iD < inDimention; iD++)
                {
                    for (int i_t = i - 1; i_t <= i + 1; i_t++)
                    {
                        for (int j_t = j - 1; j_t <= j + 1; j_t++)
                        {
                            if (i_t < 0 || i_t >= size || j_t < 0 || j_t >= size)
                                tmpRes += 0;
                            else
                                tmpRes += input[iD * size * size + i_t * size + j_t] * filter[jD * inDimention * 3 * 3 + iD * 3 * 3 + (i_t - i + 1) * 3 + j_t - j + 1];
                        }
                    }
                }
                tmpRes += bias[jD];
                tmpRes = tmpRes < 0 ? 0 : tmpRes;
                res[jD * size * size + i * size + j] = tmpRes;
            }
        }
    }
}

void AssignRandomValue(float *res, int size, bool randomFlag = true)
{
    for (int i = 0; i < size; i ++)
    {
        if (randomFlag)
            res[i] = float(rand() % 20001) / 10000 - 1;
        else
            res[i] = 0;
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    // timing setup
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event);
    float milliseconds = 0;

    /*
    // Test case 1
    int size = 4, inDimention = 2, outDimention = 2;
    float inputTest[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
    float filter[] = {1,2,3,4,5,6,7,8,9,9,8,7,6,5,4,3,2,1,9,8,7,6,5,4,3,2,1,1,2,3,4,5,6,7,8,9};
    float bias[] = {-1, 1};
    float outputTest[size * size * outDimention];
    float outputTestGT[size * size * outDimention];
    */

    // Test case 2
    srand(time(NULL));
    float *inputTest, *filter, *bias, *outputTest, *outputTestGT;
    int size = 14, inDimention = 512, outDimention = 512;
    inputTest = new float[size * size * inDimention];
    filter = new float[3 * 3 * inDimention * outDimention];
    bias = new float[outDimention];
    outputTest = new float[size * size * outDimention];
    outputTestGT = new float[size * size * outDimention];

    AssignRandomValue(inputTest, size * size * inDimention);
    AssignRandomValue(filter, 3 * 3 * inDimention * outDimention);
    AssignRandomValue(bias, outDimention);
    AssignRandomValue(outputTest, size * size * outDimention, false);

    CalGT(inputTest, filter, bias, size, inDimention, outDimention, outputTestGT);
    cout << "Finish calculating GT..." << endl;

    float *gInputTest, *gFilter, *gOutputTest, *gBias, *gInputTestRearrange;
    CHECK(cudaMalloc(&gInputTest, sizeof(float) * size * size * inDimention));
    CHECK(cudaMalloc(&gInputTestRearrange, sizeof(float) * size * size * inDimention));
    CHECK(cudaMalloc(&gFilter, sizeof(float) * 3 * 3 * inDimention * outDimention));
    CHECK(cudaMalloc(&gOutputTest, sizeof(float) * size * size * outDimention));
    CHECK(cudaMalloc(&gBias, sizeof(float) * outDimention));

    CHECK(cudaMemcpy(gInputTest, inputTest, sizeof(float) * size * size * inDimention, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gFilter, filter, sizeof(float) * 3 * 3 * inDimention * outDimention, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gBias, bias, sizeof(float) * outDimention, cudaMemcpyHostToDevice));

    cudaEventRecord(start_event, 0);
    ConvLayer1<<<dim3(8, 8, 1), dim3(32, 32, 1)>>>(gInputTest, gOutputTest, gFilter, gBias, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    CHECK(cudaMemcpy(outputTest, gOutputTest, sizeof(float) * size * size * outDimention, cudaMemcpyDeviceToHost));
    PrintResult(outputTest, size, outDimention, "Conv1 out", outputTestGT);
    cout << "Time: " << milliseconds / 1000 << endl;
    AssignRandomValue(outputTest, size * size * outDimention, false);

    cudaEventRecord(start_event, 0);
    ConvLayer2<<<outDimention, dim3(32, 32, 1)>>>(gInputTest, gOutputTest, gFilter, gBias, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    CHECK(cudaMemcpy(outputTest, gOutputTest, sizeof(float) * size * size * outDimention, cudaMemcpyDeviceToHost));
    PrintResult(outputTest, size, outDimention, "Conv2 out", outputTestGT, false);
    cout << "Time: " << milliseconds / 1000 << endl;
    AssignRandomValue(outputTest, size * size * outDimention, false);

    float *inputTestRearrange = new float[size * size * inDimention];
    for (int ai = 0; ai < size * size; ai++)
        for (int bi = 0; bi < inDimention; bi++)
            inputTestRearrange[ai * inDimention + bi] = inputTest[bi * size * size + ai];
    CHECK(cudaMemcpy(gInputTestRearrange, inputTestRearrange, sizeof(float) * size * size * inDimention, cudaMemcpyHostToDevice));

    cudaEventRecord(start_event, 0);
    ConvLayer3<<<8, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(gInputTestRearrange, gOutputTest, gFilter, gBias, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    CHECK(cudaMemcpy(outputTest, gOutputTest, sizeof(float) * size * size * outDimention, cudaMemcpyDeviceToHost));
    PrintResult(outputTest, size, outDimention, "Conv3 out", outputTestGT);
    cout << "Time: " << milliseconds / 1000 << endl;
    AssignRandomValue(outputTest, size * size * outDimention, false);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<8, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(gInputTest, gOutputTest, gFilter, gBias, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    CHECK(cudaMemcpy(outputTest, gOutputTest, sizeof(float) * size * size * outDimention, cudaMemcpyDeviceToHost));
    PrintResult(outputTest, size, outDimention, "Conv4 out", outputTestGT);
    cout << "Time: " << milliseconds / 1000 << endl;
    AssignRandomValue(outputTest, size * size * outDimention, false);
}
