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
#include "pool_fn.h"

#define IMAGE_WIDTH 224
#define CHANNELS 3
#define THREADBLOCK 24
#define LAYER1_PARAMS 1728 //params: (3*3*3)*64
#define LAYER1_BIAS_PARAMS 64 //bias: 64
#define LAYER2_PARAMS 36864 //params: (3*3*64)*64
#define LAYER2_BIAS_PARAMS 64 //bias: 64
#define LAYER3_PARAMS 73728 //params: (3*3*64)*128
#define LAYER3_BIAS_PARAMS 128 //bias: 128
#define LAYER4_PARAMS 147456 //params: (3*3*128)*128
#define LAYER4_BIAS_PARAMS 128 //bias: 128
#define LAYER5_PARAMS 294912 //params: (3*3*128)*256
#define LAYER5_BIAS_PARAMS 256 //bias: 256
#define LAYER6_PARAMS 589824 //params: (3*3*256)*256
#define LAYER6_BIAS_PARAMS 256 //bias: 256
#define LAYER7_PARAMS 589824 //params: (3*3*256)*256
#define LAYER7_BIAS_PARAMS 256 //bias: 256
#define LAYER8_PARAMS 1179648 //params: (3*3*256)*512
#define LAYER8_BIAS_PARAMS 512 //bias: 512
#define LAYER9_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER9_BIAS_PARAMS 512 //bias: 512
#define LAYER10_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER10_BIAS_PARAMS 512 //bias: 512
#define LAYER11_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER11_BIAS_PARAMS 512 //bias: 512
#define LAYER12_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER12_BIAS_PARAMS 512 //bias: 512
#define LAYER13_PARAMS 2359296 //params: (3*3*512)*512
#define LAYER13_BIAS_PARAMS 512 //bias: 512
#define LAYER14_PARAMS 102760448 //params: (7*7*512)*4096
#define LAYER14_BIAS_PARAMS 4096
#define LAYER15_PARAMS 16777216 //params: 4096*4096
#define LAYER15_BIAS_PARAMS 4096
#define LAYER16_PARAMS 4096000 //params: 4096*4096
#define LAYER16_BIAS_PARAMS 1000
#define MASK_WIDTH 3
#include "utils.h"

using namespace std;

void ConvertInput(float *&Data_Layer_CPU);
void LoadImageNetClass(char **image_class, char *file_path);
void LoadInput(float *Data_Layer_CPU,char* file_path);
void InitWeights_Biases(float *Weights_CPU, int size, char* file_path);


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    if(argc != 2){
        cout << "Usage ./vgg_net file_path" << endl;
        return -1;
    }
    char * file_path = argv[1];

    int x = 0;
    // timing setup
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event);
    float milliseconds = 0;
    float totalTime = 0;

    //Allocation of host memory for weights
    float *Layer1_Weights_CPU = (float*) malloc (LAYER1_PARAMS * sizeof(float)); //no. of features in nth layer
    float *Layer1_Weights_Bias_CPU = (float*) malloc (LAYER1_BIAS_PARAMS * sizeof(float));
    float *Layer2_Weights_CPU = (float*) malloc (LAYER2_PARAMS * sizeof(float)); // no. of features in nth layer
    float *Layer2_Weights_Bias_CPU = (float*) malloc (LAYER2_BIAS_PARAMS * sizeof(float));
    float *Layer3_Weights_CPU = (float*) malloc (LAYER3_PARAMS * sizeof(float));
    float *Layer3_Weights_Bias_CPU = (float*) malloc (LAYER3_BIAS_PARAMS * sizeof(float));
    float *Layer4_Weights_CPU = (float*) malloc (LAYER4_PARAMS * sizeof(float));
    float *Layer4_Weights_Bias_CPU = (float*) malloc (LAYER4_BIAS_PARAMS * sizeof(float));
    float *Layer5_Weights_CPU = (float*) malloc (LAYER5_PARAMS * sizeof(float));
    float *Layer5_Weights_Bias_CPU = (float*) malloc (LAYER5_BIAS_PARAMS * sizeof(float));
    float *Layer6_Weights_CPU = (float*) malloc (LAYER6_PARAMS * sizeof(float));
    float *Layer6_Weights_Bias_CPU = (float*) malloc (LAYER6_BIAS_PARAMS * sizeof(float));
    float *Layer7_Weights_CPU = (float*) malloc (LAYER7_PARAMS * sizeof(float));
    float *Layer7_Weights_Bias_CPU = (float*) malloc (LAYER7_BIAS_PARAMS * sizeof(float));
    float *Layer8_Weights_CPU = (float*) malloc (LAYER8_PARAMS * sizeof(float));
    float *Layer8_Weights_Bias_CPU = (float*) malloc (LAYER8_BIAS_PARAMS * sizeof(float));
    float *Layer9_Weights_CPU = (float*) malloc (LAYER9_PARAMS * sizeof(float));
    float *Layer9_Weights_Bias_CPU = (float*) malloc (LAYER9_BIAS_PARAMS * sizeof(float));
    float *Layer10_Weights_CPU = (float*) malloc (LAYER10_PARAMS * sizeof(float));
    float *Layer10_Weights_Bias_CPU = (float*) malloc (LAYER10_BIAS_PARAMS * sizeof(float));
    float *Layer11_Weights_CPU = (float*) malloc (LAYER11_PARAMS * sizeof(float));
    float *Layer11_Weights_Bias_CPU = (float*) malloc (LAYER11_BIAS_PARAMS * sizeof(float));
    float *Layer12_Weights_CPU = (float*) malloc (LAYER12_PARAMS * sizeof(float));
    float *Layer12_Weights_Bias_CPU = (float*) malloc (LAYER12_BIAS_PARAMS * sizeof(float));
    float *Layer13_Weights_CPU = (float*) malloc (LAYER13_PARAMS * sizeof(float));
    float *Layer13_Weights_Bias_CPU = (float*) malloc (LAYER13_BIAS_PARAMS * sizeof(float));
    float *Layer14_Weights_CPU = (float*) malloc (LAYER14_PARAMS * sizeof(float));
    float *Layer14_Weights_Bias_CPU = (float*) malloc (LAYER14_BIAS_PARAMS * sizeof(float));
    float *Layer15_Weights_CPU = (float*) malloc (LAYER15_PARAMS * sizeof(float));
    float *Layer15_Weights_Bias_CPU = (float*) malloc (LAYER15_BIAS_PARAMS * sizeof(float));
    float *Layer16_Weights_CPU = (float*) malloc (LAYER16_PARAMS * sizeof(float));
    float *Layer16_Weights_Bias_CPU = (float*) malloc (LAYER16_BIAS_PARAMS * sizeof(float));

    float *Data_Layer_CPU = (float*) malloc (CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(float));

    InitWeights_Biases(Layer1_Weights_CPU,LAYER1_PARAMS, (char *)"data/conv1_1_v.txt");
    InitWeights_Biases(Layer1_Weights_Bias_CPU,LAYER1_BIAS_PARAMS, (char *)"data/conv1_1_v_bias.txt");
    InitWeights_Biases(Layer2_Weights_CPU,LAYER2_PARAMS, (char *)"data/conv1_2_v.txt");
    InitWeights_Biases(Layer2_Weights_Bias_CPU,LAYER2_BIAS_PARAMS, (char *)"data/conv1_2_v_bias.txt");
    InitWeights_Biases(Layer3_Weights_CPU,LAYER3_PARAMS, (char *)"data/conv2_1_v.txt");
    InitWeights_Biases(Layer3_Weights_Bias_CPU,LAYER3_BIAS_PARAMS, (char *)"data/conv2_1_v_bias.txt");
    InitWeights_Biases(Layer4_Weights_CPU,LAYER4_PARAMS, (char *)"data/conv2_2_v.txt");
    InitWeights_Biases(Layer4_Weights_Bias_CPU,LAYER4_BIAS_PARAMS, (char *)"data/conv2_2_v_bias.txt");
    InitWeights_Biases(Layer5_Weights_CPU,LAYER5_PARAMS, (char *)"data/conv3_1_v.txt");
    InitWeights_Biases(Layer5_Weights_Bias_CPU,LAYER5_BIAS_PARAMS, (char *)"data/conv3_1_v_bias.txt");
    InitWeights_Biases(Layer6_Weights_CPU,LAYER6_PARAMS, (char *)"data/conv3_2_v.txt");
    InitWeights_Biases(Layer6_Weights_Bias_CPU,LAYER6_BIAS_PARAMS, (char *)"data/conv3_2_v_bias.txt");
    InitWeights_Biases(Layer7_Weights_CPU,LAYER7_PARAMS, (char *)"data/conv3_3_v.txt");
    InitWeights_Biases(Layer7_Weights_Bias_CPU,LAYER7_BIAS_PARAMS, (char *)"data/conv3_3_v_bias.txt");
    InitWeights_Biases(Layer8_Weights_CPU,LAYER8_PARAMS, (char *)"data/conv4_1_v.txt");
    InitWeights_Biases(Layer8_Weights_Bias_CPU,LAYER8_BIAS_PARAMS, (char *)"data/conv4_1_v_bias.txt");
    InitWeights_Biases(Layer9_Weights_CPU,LAYER9_PARAMS, (char *)"data/conv4_2_v.txt");
    InitWeights_Biases(Layer9_Weights_Bias_CPU,LAYER9_BIAS_PARAMS, (char *)"data/conv4_2_v_bias.txt");
    InitWeights_Biases(Layer10_Weights_CPU,LAYER10_PARAMS, (char *)"data/conv4_3_v.txt");
    InitWeights_Biases(Layer10_Weights_Bias_CPU,LAYER10_BIAS_PARAMS, (char *)"data/conv4_3_v_bias.txt");
    InitWeights_Biases(Layer11_Weights_CPU,LAYER11_PARAMS, (char *)"data/conv5_1_v.txt");
    InitWeights_Biases(Layer11_Weights_Bias_CPU,LAYER11_BIAS_PARAMS, (char *)"data/conv5_1_v_bias.txt");
    InitWeights_Biases(Layer12_Weights_CPU,LAYER12_PARAMS, (char *)"data/conv5_2_v.txt");
    InitWeights_Biases(Layer12_Weights_Bias_CPU,LAYER12_BIAS_PARAMS, (char *)"data/conv5_2_v_bias.txt");
    InitWeights_Biases(Layer13_Weights_CPU,LAYER13_PARAMS, (char *)"data/conv5_3_v.txt");
    InitWeights_Biases(Layer13_Weights_Bias_CPU,LAYER13_BIAS_PARAMS, (char *)"data/conv5_3_v_bias.txt");
    InitWeights_Biases(Layer14_Weights_CPU,LAYER14_PARAMS, (char *)"data/fc6_v.txt");
    InitWeights_Biases(Layer14_Weights_Bias_CPU,LAYER14_BIAS_PARAMS, (char *)"data/fc6_v_bias.txt");
    InitWeights_Biases(Layer15_Weights_CPU,LAYER15_PARAMS, (char *)"data/fc7_v.txt");
    InitWeights_Biases(Layer15_Weights_Bias_CPU,LAYER15_BIAS_PARAMS, (char *)"data/fc7_v_bias.txt");
    InitWeights_Biases(Layer16_Weights_CPU,LAYER16_PARAMS, (char *)"data/fc8_v.txt");
    InitWeights_Biases(Layer16_Weights_Bias_CPU,LAYER16_BIAS_PARAMS, (char *)"data/fc8_v_bias.txt");

    LoadInput(Data_Layer_CPU,file_path);
    ConvertInput(Data_Layer_CPU);

    //Allocate device GMEM input
    float *Data_Layer_GPU;
    cudaMalloc(&Data_Layer_GPU, CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(float));
    cudaMemcpy(Data_Layer_GPU, Data_Layer_CPU, CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    int inDimention, outDimention, size;
    //layer 1
    inDimention = 3; outDimention = 64; size = 224;
    float *Conv_Layer1_Features;
    float *Layer1_Weights_GPU, *Layer1_Weights_Bias_GPU;
    cudaMalloc(&Layer1_Weights_GPU, LAYER1_PARAMS * sizeof(float));
    cudaMalloc(&Layer1_Weights_Bias_GPU, LAYER1_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer1_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer1_Weights_GPU, Layer1_Weights_CPU, LAYER1_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer1_Weights_Bias_GPU, Layer1_Weights_Bias_CPU, LAYER1_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Data_Layer_GPU, Conv_Layer1_Features, Layer1_Weights_GPU, Layer1_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv1: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Data_Layer_GPU);

    //layer 2
    inDimention = outDimention; outDimention = 64; size = 224;
    float *Conv_Layer2_Features;
    float *Layer2_Weights_GPU, *Layer2_Weights_Bias_GPU;
    cudaMalloc(&Layer2_Weights_GPU, LAYER2_PARAMS * sizeof(float));
    cudaMalloc(&Layer2_Weights_Bias_GPU, LAYER2_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer2_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer2_Weights_GPU, Layer2_Weights_CPU, LAYER2_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer2_Weights_Bias_GPU, Layer2_Weights_Bias_CPU, LAYER2_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer1_Features, Conv_Layer2_Features, Layer2_Weights_GPU, Layer2_Weights_Bias_GPU, inDimention, size, outDimention);
    //ConvLayer2<<<outDimention, dim3(32, 32, 1)>>>(Conv_Layer1_Features, Conv_Layer2_Features, Layer2_Weights_GPU, Layer2_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer1<<<dim3(7,7,1), dim3(32, 32, 1)>>>(Conv_Layer1_Features, Conv_Layer2_Features, Layer2_Weights_GPU, Layer2_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv2: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer1_Features);

    //pool layer 2
    float * Pool_Layer2_Features ;
    cudaMalloc(&Pool_Layer2_Features, 112 * 112 * 128 * sizeof(float));
    cudaEventRecord(start_event, 0);
    pool_2<<<dim3(14, 14, 1), dim3(8, 8, 1)>>>(Conv_Layer2_Features, Pool_Layer2_Features, 224, 224 * 64);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pool2: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer2_Features);

    //layer 3
    inDimention = outDimention; outDimention = 128; size = 112;
    float *Conv_Layer3_Features;
    float *Layer3_Weights_GPU, *Layer3_Weights_Bias_GPU;
    cudaMalloc(&Layer3_Weights_GPU, LAYER3_PARAMS * sizeof(float));
    cudaMalloc(&Layer3_Weights_Bias_GPU, LAYER3_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer3_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer3_Weights_GPU, Layer3_Weights_CPU, LAYER3_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer3_Weights_Bias_GPU, Layer3_Weights_Bias_CPU, LAYER3_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Pool_Layer2_Features, Conv_Layer3_Features, Layer3_Weights_GPU, Layer3_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer2<<<outDimention, dim3(32, 32, 1)>>>(Pool_Layer2_Features, Conv_Layer3_Features, Layer3_Weights_GPU, Layer3_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv3: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Pool_Layer2_Features);

    //layer 4
    inDimention = outDimention; outDimention = 128; size = 112;
    float *Conv_Layer4_Features;
    float *Layer4_Weights_GPU, *Layer4_Weights_Bias_GPU;
    cudaMalloc(&Layer4_Weights_GPU, LAYER4_PARAMS * sizeof(float));
    cudaMalloc(&Layer4_Weights_Bias_GPU, LAYER4_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer4_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer4_Weights_GPU, Layer4_Weights_CPU, LAYER4_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer4_Weights_Bias_GPU, Layer4_Weights_Bias_CPU, LAYER4_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer3_Features, Conv_Layer4_Features, Layer4_Weights_GPU, Layer4_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer2<<<outDimention, dim3(32, 32, 1)>>>(Conv_Layer3_Features, Conv_Layer4_Features, Layer4_Weights_GPU, Layer4_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv4: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer3_Features);


    //pool layer 4
    float * Pool_Layer4_Features ;
    cudaMalloc(&Pool_Layer4_Features, 56 * 56 * 256 * sizeof(float));
    cudaEventRecord(start_event, 0);
    pool_2<<<dim3(7, 7, 1), dim3(8, 8, 1)>>>(Conv_Layer4_Features, Pool_Layer4_Features, 112, 112 * 128);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pool4: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer4_Features);

    //layer 5 
    inDimention = outDimention; outDimention = 256; size = 56;
    float *Conv_Layer5_Features;
    float *Layer5_Weights_GPU, *Layer5_Weights_Bias_GPU;
    cudaMalloc(&Layer5_Weights_GPU, LAYER5_PARAMS * sizeof(float));
    cudaMalloc(&Layer5_Weights_Bias_GPU, LAYER5_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer5_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer5_Weights_GPU, Layer5_Weights_CPU, LAYER5_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer5_Weights_Bias_GPU, Layer5_Weights_Bias_CPU, LAYER5_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Pool_Layer4_Features, Conv_Layer5_Features, Layer5_Weights_GPU, Layer5_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv5: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Pool_Layer4_Features);

    //layer 6
    inDimention = outDimention; outDimention = 256; size = 56;
    float *Conv_Layer6_Features;
    float *Layer6_Weights_GPU, *Layer6_Weights_Bias_GPU;
    cudaMalloc(&Layer6_Weights_GPU, LAYER6_PARAMS * sizeof(float));
    cudaMalloc(&Layer6_Weights_Bias_GPU, LAYER6_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer6_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer6_Weights_GPU, Layer6_Weights_CPU, LAYER6_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer6_Weights_Bias_GPU, Layer6_Weights_Bias_CPU, LAYER6_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer5_Features, Conv_Layer6_Features, Layer6_Weights_GPU, Layer6_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv6: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer5_Features);

    //layer 7
    inDimention = outDimention; outDimention = 256; size = 56;
    float *Conv_Layer7_Features;
    float *Layer7_Weights_GPU, *Layer7_Weights_Bias_GPU;
    cudaMalloc(&Layer7_Weights_GPU, LAYER7_PARAMS * sizeof(float));
    cudaMalloc(&Layer7_Weights_Bias_GPU, LAYER7_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer7_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer7_Weights_GPU, Layer7_Weights_CPU, LAYER7_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer7_Weights_Bias_GPU, Layer7_Weights_Bias_CPU, LAYER7_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer6_Features, Conv_Layer7_Features, Layer7_Weights_GPU, Layer7_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv7: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer6_Features);

    //pool layer 7
    float * Pool_Layer7_Features ;
    cudaMalloc(&Pool_Layer7_Features, 28 * 28 * 512 * sizeof(float));
    cudaEventRecord(start_event, 0);
    pool_2<<<dim3(4, 4, 1), dim3(8, 8, 1)>>>(Conv_Layer7_Features, Pool_Layer7_Features, 56, 56 * 256);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pool7: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer7_Features);

    //layer 8
    inDimention = outDimention; outDimention = 512; size = 28;
    float *Conv_Layer8_Features;
    float *Layer8_Weights_GPU, *Layer8_Weights_Bias_GPU;
    cudaMalloc(&Layer8_Weights_GPU, LAYER8_PARAMS * sizeof(float));
    cudaMalloc(&Layer8_Weights_Bias_GPU, LAYER8_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer8_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer8_Weights_GPU, Layer8_Weights_CPU, LAYER8_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer8_Weights_Bias_GPU, Layer8_Weights_Bias_CPU, LAYER8_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Pool_Layer7_Features, Conv_Layer8_Features, Layer8_Weights_GPU, Layer8_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv8: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Pool_Layer7_Features);

    //layer 9
    inDimention = outDimention; outDimention = 512; size = 28;
    float * Conv_Layer9_Features;
    float *Layer9_Weights_GPU, *Layer9_Weights_Bias_GPU;
    cudaMalloc(&Layer9_Weights_GPU, LAYER9_PARAMS * sizeof(float));
    cudaMalloc(&Layer9_Weights_Bias_GPU, LAYER9_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer9_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer9_Weights_GPU, Layer9_Weights_CPU, LAYER9_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer9_Weights_Bias_GPU, Layer9_Weights_Bias_CPU, LAYER9_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer8_Features, Conv_Layer9_Features, Layer9_Weights_GPU, Layer9_Weights_Bias_GPU, inDimention, size, outDimention);
    //ConvLayer3_1<<<THREADBLOCK, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(Conv_Layer8_Features, Conv_Layer9_Features, Layer9_Weights_GPU, Layer9_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv9: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer8_Features);

    //layer 10
    inDimention = outDimention; outDimention = 512; size = 28;
    float *Conv_Layer10_Features;
    float *Layer10_Weights_GPU, *Layer10_Weights_Bias_GPU;
    cudaMalloc(&Layer10_Weights_GPU, LAYER10_PARAMS * sizeof(float));
    cudaMalloc(&Layer10_Weights_Bias_GPU, LAYER10_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer10_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer10_Weights_GPU, Layer10_Weights_CPU, LAYER10_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer10_Weights_Bias_GPU, Layer10_Weights_Bias_CPU, LAYER10_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer9_Features, Conv_Layer10_Features, Layer10_Weights_GPU, Layer10_Weights_Bias_GPU, inDimention, size, outDimention);
    //ConvLayer3_1<<<THREADBLOCK, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(Conv_Layer9_Features, Conv_Layer10_Features, Layer10_Weights_GPU, Layer10_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv10: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer9_Features);

    //pool layer 10
    float * Pool_Layer10_Features ;
    cudaMalloc(&Pool_Layer10_Features, 14 * 14 * 512 * sizeof(float));
    cudaEventRecord(start_event, 0);
    pool_2<<<dim3(2, 2, 1), dim3(8, 8, 1)>>>(Conv_Layer10_Features, Pool_Layer10_Features, 28, 28 * 512);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pool10: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer10_Features);

    //layer 11
    inDimention = outDimention; outDimention = 512; size = 14;
    float *Conv_Layer11_Features;
    float *Layer11_Weights_GPU, *Layer11_Weights_Bias_GPU;
    cudaMalloc(&Layer11_Weights_GPU, LAYER11_PARAMS * sizeof(float));
    cudaMalloc(&Layer11_Weights_Bias_GPU, LAYER11_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer11_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer11_Weights_GPU, Layer11_Weights_CPU, LAYER11_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer11_Weights_Bias_GPU, Layer11_Weights_Bias_CPU, LAYER11_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Pool_Layer10_Features, Conv_Layer11_Features, Layer11_Weights_GPU, Layer11_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer3_1<<<THREADBLOCK, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(Pool_Layer10_Features, Conv_Layer11_Features, Layer11_Weights_GPU, Layer11_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv11: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Pool_Layer10_Features);

    //layer 12
    inDimention = outDimention; outDimention = 512; size = 14;
    float *Conv_Layer12_Features;
    float *Layer12_Weights_GPU, *Layer12_Weights_Bias_GPU;
    cudaMalloc(&Layer12_Weights_GPU, LAYER12_PARAMS * sizeof(float));
    cudaMalloc(&Layer12_Weights_Bias_GPU, LAYER12_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer12_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer12_Weights_GPU, Layer12_Weights_CPU, LAYER12_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer12_Weights_Bias_GPU, Layer12_Weights_Bias_CPU, LAYER12_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer11_Features, Conv_Layer12_Features, Layer12_Weights_GPU, Layer12_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer3_1<<<THREADBLOCK, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(Conv_Layer11_Features, Conv_Layer12_Features, Layer12_Weights_GPU, Layer12_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv12: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer11_Features);

    //layer 13
    inDimention = outDimention; outDimention = 512; size = 14;
    float *Conv_Layer13_Features;
    float *Layer13_Weights_GPU, *Layer13_Weights_Bias_GPU;
    cudaMalloc(&Layer13_Weights_GPU, LAYER13_PARAMS * sizeof(float));
    cudaMalloc(&Layer13_Weights_Bias_GPU, LAYER13_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Conv_Layer13_Features, size * size * outDimention * sizeof(float));
    cudaMemcpy(Layer13_Weights_GPU, Layer13_Weights_CPU, LAYER13_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer13_Weights_Bias_GPU, Layer13_Weights_Bias_CPU, LAYER13_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    //ConvLayer4<<<THREADBLOCK, dim3(32, 32, 1), (3 * 3 * inDimention) * sizeof(float)>>>(Conv_Layer12_Features, Conv_Layer13_Features, Layer13_Weights_GPU, Layer13_Weights_Bias_GPU, inDimention, size, outDimention);
    ConvLayer3_1<<<THREADBLOCK, inDimention, (3 * 3 * inDimention + inDimention / 32) * sizeof(float)>>>(Conv_Layer12_Features, Conv_Layer13_Features, Layer13_Weights_GPU, Layer13_Weights_Bias_GPU, inDimention, size, outDimention);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Conv13: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer12_Features);

    //pool layer 13
    float * Pool_Layer13_Features ;
    cudaMalloc(&Pool_Layer13_Features, 7 * 7 * 512 * sizeof(float));
    cudaEventRecord(start_event, 0);
    pool_2<<<dim3(1, 1, 1), dim3(8, 8, 1)>>>(Conv_Layer13_Features, Pool_Layer13_Features, 14, 14 * 512);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pool13: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cudaFree(Conv_Layer13_Features);

    //fully connected layer 14
    float *Fu_Layer14_Features;
    float *Layer14_Weights_GPU, *Layer14_Weights_Bias_GPU;
    cudaMalloc(&Layer14_Weights_GPU, LAYER14_PARAMS * sizeof(float));
    cudaMalloc(&Layer14_Weights_Bias_GPU, LAYER14_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Fu_Layer14_Features, 4096 * sizeof(float));

    cudaMemcpy(Layer14_Weights_GPU, Layer14_Weights_CPU, LAYER14_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer14_Weights_Bias_GPU, Layer14_Weights_Bias_CPU, LAYER14_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    fn<<<128, 32>>>(Pool_Layer13_Features, Fu_Layer14_Features, Layer14_Weights_GPU, Layer14_Weights_Bias_GPU, 4096, 7 * 7 * 512);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "FC14: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;

    cudaFree(Pool_Layer13_Features);
    cudaFree(Layer14_Weights_GPU);
    cudaFree(Layer14_Weights_Bias_GPU);

    //fully connected layer 15
    float *Fu_Layer15_Features;
    float *Layer15_Weights_GPU, *Layer15_Weights_Bias_GPU;
    cudaMalloc(&Layer15_Weights_GPU, LAYER15_PARAMS * sizeof(float));
    cudaMalloc(&Layer15_Weights_Bias_GPU, LAYER15_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Fu_Layer15_Features, 4096 * sizeof(float));

    cudaMemcpy(Layer15_Weights_GPU, Layer15_Weights_CPU, LAYER15_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer15_Weights_Bias_GPU, Layer15_Weights_Bias_CPU, LAYER15_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    fn<<<128, 32>>>(Fu_Layer14_Features, Fu_Layer15_Features, Layer15_Weights_GPU, Layer15_Weights_Bias_GPU, 4096, 4096);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "FC15: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;

    cudaFree(Fu_Layer14_Features);
    cudaFree(Layer15_Weights_GPU);
    cudaFree(Layer15_Weights_Bias_GPU);

    //fully connected layer 16
    float * Fu_Layer16_Features;
    float * Layer16_Weights_GPU, *Layer16_Weights_Bias_GPU;
    cudaMalloc(&Layer16_Weights_GPU, LAYER16_PARAMS * sizeof(float));
    cudaMalloc(&Layer16_Weights_Bias_GPU, LAYER16_BIAS_PARAMS * sizeof(float));
    cudaMalloc(&Fu_Layer16_Features, 1000 * sizeof(float));

    cudaMemcpy(Layer16_Weights_GPU, Layer16_Weights_CPU, LAYER16_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Layer16_Weights_Bias_GPU, Layer16_Weights_Bias_CPU, LAYER16_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(start_event, 0);
    fn<<<32, 32>>>(Fu_Layer15_Features, Fu_Layer16_Features, Layer16_Weights_GPU, Layer16_Weights_Bias_GPU, 1000, 4096);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "FC16: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;

    cudaFree(Fu_Layer15_Features);
    cudaFree(Layer16_Weights_GPU);
    cudaFree(Layer16_Weights_Bias_GPU);

    //predict layer
    uint32_t * pred = (uint32_t*)malloc(sizeof(uint32_t));
    uint32_t * d_pred;
    cudaMalloc(&d_pred, sizeof(uint32_t));
    cudaEventRecord(start_event, 0);
    predict<<<1, 1024>>>(Fu_Layer16_Features, d_pred);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);
    cout << "Pred: " << milliseconds / 1000 << endl;
    totalTime += milliseconds / 1000;
    cout << "Total Time: " << totalTime << endl;
    cudaMemcpy(pred, d_pred, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(Fu_Layer16_Features);
    cout << endl << "prediction : " << *pred << endl;

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}	
