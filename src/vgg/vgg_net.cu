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

void ConvertInput(double *Data_Layer_CPU_R, double *Data_Layer_CPU_G, double *Data_Layer_CPU_B, double *Data_Layer_CPU);
void LoadImageNetClass(char **image_class, char *file_path);
void LoadInput(double *Data_Layer_CPU,char* file_path);
void InitWeights_Biases(double *Weights_CPU, int size, char* file_path);



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
    
	// timing setup
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event) ;
    cudaEventCreate(&stop_event);
    float milliseconds = 0;

	//Allocation of host memory for weights
	double *Layer1_Weights_CPU = (double*) malloc (LAYER1_PARAMS * sizeof(double)); //no. of features in nth layer
	double *Layer1_Weights_Bias_CPU = (double*) malloc (LAYER1_BIAS_PARAMS * sizeof(double));
	double *Layer2_Weights_CPU = (double*) malloc (LAYER2_PARAMS * sizeof(double)); // no. of features in nth layer
	double *Layer2_Weights_Bias_CPU = (double*) malloc (LAYER2_BIAS_PARAMS * sizeof(double));
	double *Layer3_Weights_CPU = (double*) malloc (LAYER3_PARAMS * sizeof(double));
	double *Layer3_Weights_Bias_CPU = (double*) malloc (LAYER3_BIAS_PARAMS * sizeof(double));
	double *Layer4_Weights_CPU = (double*) malloc (LAYER4_PARAMS * sizeof(double));
	double *Layer4_Weights_Bias_CPU = (double*) malloc (LAYER4_BIAS_PARAMS * sizeof(double));
	double *Layer5_Weights_CPU = (double*) malloc (LAYER5_PARAMS * sizeof(double));
	double *Layer5_Weights_Bias_CPU = (double*) malloc (LAYER5_BIAS_PARAMS * sizeof(double));
	double *Layer6_Weights_CPU = (double*) malloc (LAYER6_PARAMS * sizeof(double));
	double *Layer6_Weights_Bias_CPU = (double*) malloc (LAYER6_BIAS_PARAMS * sizeof(double));
	double *Layer7_Weights_CPU = (double*) malloc (LAYER7_PARAMS * sizeof(double));
	double *Layer7_Weights_Bias_CPU = (double*) malloc (LAYER7_BIAS_PARAMS * sizeof(double));
	double *Layer8_Weights_CPU = (double*) malloc (LAYER8_PARAMS * sizeof(double));
	double *Layer8_Weights_Bias_CPU = (double*) malloc (LAYER8_BIAS_PARAMS * sizeof(double));
	double *Layer9_Weights_CPU = (double*) malloc (LAYER9_PARAMS * sizeof(double));
	double *Layer9_Weights_Bias_CPU = (double*) malloc (LAYER9_BIAS_PARAMS * sizeof(double));
	double *Layer10_Weights_CPU = (double*) malloc (LAYER10_PARAMS * sizeof(double));
	double *Layer10_Weights_Bias_CPU = (double*) malloc (LAYER10_BIAS_PARAMS * sizeof(double));
	double *Layer11_Weights_CPU = (double*) malloc (LAYER11_PARAMS * sizeof(double));
	double *Layer11_Weights_Bias_CPU = (double*) malloc (LAYER11_BIAS_PARAMS * sizeof(double));
	double *Layer12_Weights_CPU = (double*) malloc (LAYER12_PARAMS * sizeof(double));
	double *Layer12_Weights_Bias_CPU = (double*) malloc (LAYER12_BIAS_PARAMS * sizeof(double));
	double *Layer13_Weights_CPU = (double*) malloc (LAYER13_PARAMS * sizeof(double));
	double *Layer13_Weights_Bias_CPU = (double*) malloc (LAYER13_BIAS_PARAMS * sizeof(double));
	double *Layer14_Weights_CPU = (double*) malloc (LAYER14_PARAMS * sizeof(double));
	double *Layer14_Weights_Bias_CPU = (double*) malloc (LAYER14_BIAS_PARAMS * sizeof(double));
	double *Layer15_Weights_CPU = (double*) malloc (LAYER15_PARAMS * sizeof(double));
	double *Layer15_Weights_Bias_CPU = (double*) malloc (LAYER15_BIAS_PARAMS * sizeof(double));
	double *Layer16_Weights_CPU = (double*) malloc (LAYER16_PARAMS * sizeof(double));
	double *Layer16_Weights_Bias_CPU = (double*) malloc (LAYER16_BIAS_PARAMS * sizeof(double));

	double *Data_Layer_CPU = (double*) malloc (CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(double));
	
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
	double *Data_Layer_GPU;
	cudaMalloc(&Data_Layer_GPU, CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(double));
	cudaMemcpy(Data_Layer_GPU, Data_Layer_CPU, CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(double), cudaMemcpyHostToDevice);

	//layer 1

	//layer 2

	//pool layer 2
	float * Pool_Layer2_Features ;
	cudaMalloc(&Pool_Layer2_Features, 112, 112 * 128 * sizeof(float));
	pool_2<<<dim3(14, 14, 1), dim3(8, 8, 1)>>>(Layer2_Features, Pool_Layer2_Features, 224, 224 * 64);
	cudaFree(Layer2_Features);

	//pool layer 4
	float * Pool_Layer4_Features ;
	cudaMalloc(&Pool_Layer4_Features, 56 * 56 * 256 * sizeof(float));
	pool_2<<<dim3(7, 7, 1), dim3(8, 8, 1)>>>(Layer4_Features, Pool_Layer4_Features, 112, 112 * 128);
	cudaFree(Layer4_Features);

	//pool layer 7
	float * Pool_Layer7_Features ;
	cudaMalloc(&Pool_Layer7_Features, 28 * 28 * 512 * sizeof(float));
	pool_2<<<dim3(4, 4, 1), dim3(8, 8, 1)>>>(Layer7_Features, Pool_Layer7_Features, 56, 56 * 256);
	cudaFree(Layer7_Features);

	//pool layer 10
	float * Pool_Layer10_Features ;
	cudaMalloc(&Pool_Layer10_Features, 14 * 14 * 512 * sizeof(float));
	pool_2<<<dim3(2, 2, 1), dim3(8, 8, 1)>>>(Layer10_Features, Pool_Layer10_Features, 28, 28 * 512);
	cudaFree(Layer10_Features);

	//pool layer 13
	float * Pool_Layer13_Features ;
	cudaMalloc(&Pool_Layer13_Features, 7 * 7 * 512 * sizeof(float));
	pool_2<<<dim3(1, 1, 1), dim3(8, 8, 1)>>>(Layer13_Features, Pool_Layer13_Features, 14, 14 * 512);
	cudaFree(Layer13_Features);

	//fully connected layer 14
	float * Fu_layer14_Features;
	float * Layer14_Weights_GPU, Layer14_Weights_Bias_GPU;
	cudaMalloc(&Layer14_Weights_GPU, LAYER14_PARAMS * sizeof(float));
	cudaMalloc(&Layer14_Weights_Bias_GPU, LAYER14_BIAS_PARAMS * sizeof(float));
	cudaMalloc(&Fu_Layer14_Features, 4096 * sizeof(float));

	cudaMemcpy(Layer14_Weights_GPU, Layer14_Weights_CPU, LAYER14_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer14_Weights_Bias_GPU, Layer14_Weights_Bias_CPU, LAYER14_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

	fn<<<128, 32, 32 * sizeof(float)>>>(Pool_Layer13_Features, Fu_Layer14_Features, Layer14_Weights_GPU, Layer14_Weights_Bias_GPU, 4096, 7 * 7 * 512);

	cudaFree(Pool_Layer13_Features);
	cudaFree(Layer14_Weights_GPU);
	cudaFree(Layer14_Weights_Bias_GPU);

	//fully connected layer 15
	float * Fu_layer15_Features;
	float * Layer15_Weights_GPU, Layer15_Weights_Bias_GPU;
	cudaMalloc(&Layer15_Weights_GPU, LAYER15_PARAMS * sizeof(float));
	cudaMalloc(&Layer15_Weights_Bias_GPU, LAYER15_BIAS_PARAMS * sizeof(float));
	cudaMalloc(&Fu_Layer15_Features, 4096 * sizeof(float));

	cudaMemcpy(Layer15_Weights_GPU, Layer15_Weights_CPU, LAYER15_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer15_Weights_Bias_GPU, Layer15_Weights_Bias_CPU, LAYER15_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

	fn<<<128, 32, 32 * sizeof(float)>>>(Pool_Layer13_Features, Fu_Layer15_Features, Layer15_Weights_GPU, Layer15Weights_Bias_GPU, 4096, 4096);

	cudaFree(Fu_layer14_Features);
	cudaFree(Layer15_Weights_GPU);
	cudaFree(Layer15_Weights_Bias_GPU);

	//fully connected layer 16
	float * Fu_layer16_Features;
	float * Layer16_Weights_GPU, Layer16_Weights_Bias_GPU;
	cudaMalloc(&Layer16_Weights_GPU, LAYER16_PARAMS * sizeof(float));
	cudaMalloc(&Layer16_Weights_Bias_GPU, LAYER16_BIAS_PARAMS * sizeof(float));
	cudaMalloc(&Fu_Layer16_Features, 1000 * sizeof(float));

	cudaMemcpy(Layer16_Weights_GPU, Layer16_Weights_CPU, LAYER16_PARAMS * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Layer16_Weights_Bias_GPU, Layer16_Weights_Bias_CPU, LAYER16_BIAS_PARAMS * sizeof(float), cudaMemcpyHostToDevice);

	fn<<<32, 32, 32 * sizeof(float)>>>(Pool_Layer13_Features, Fu_Layer16_Features, Layer16_Weights_GPU, Layer16_Weights_Bias_GPU, 1000, 4096);

	cudaFree(Fu_layer15_Features);
	cudaFree(Layer16_Weights_GPU);
	cudaFree(Layer16_Weights_Bias_GPU);

	//predict layer
	uint32_t * pred = (uint32_t*)malloc(sizeof(uint32_t));
	uint32_t * d_pred;
	cudaMalloc(&d_pred, sizeof(uint32_t));
	predict<<<1, 1024>>>(Fu_layer16_Features, d_pred);
	cudaMemcpy(pred, d_pred, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(Fu_layer16_Features);
	cout << "prediction : " << *pred << endl;

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}	
