#include "../common/common.h"
#include <iostream>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <cmath>
#include <time.h>

#define PRED
#ifdef POOL
    #define MATROW (224)
    #define MATCOL (224 * 64)
    #define MATSIZE (MATROW * MATCOL)
#endif
#ifdef FN
    #define MATROW (4096)
    #define MATCOL (1024)
    #define MATSIZE (MATROW * MATCOL)
#endif
#ifdef PRED
    #define MATROW (1000)
    #define MATCOL (1000)
    #define MATSIZE (MATROW * MATCOL)
#endif

#include "pool_fn.h"
#include "utils.h"

using namespace std;

void pool_unitest(float *in, float *out, float * err, float * cpu_time);
void fn_unitest(float *kernel_weights, float *kernel_bias, float *in, float * out, float * err, float * cpu_time);
void AssignRandomValue(float *res, int size, bool randomFlag);

// pool_1<<<dim3(28, 8, 1), 32>>>(d_in, d_out, 224, 224 * 64);
// pool_1<<<dim3(14, 8, 1), 32>>>(d_in, d_out, 112, 112 * 128);
// pool_1<<<dim3(7, 8, 1), 32>>>(d_in, d_out, 56, 56 * 256);
// pool_1<<<dim3(4, 8, 1), 32>>>(d_in, d_out, 28, 28 * 512);
// pool_1<<<dim3(2, 8, 1), 32>>>(d_in, d_out, 14, 14 * 512);

// pool_2<<<dim3(14, 14, 1), dim3(8, 8, 1)>>>(d_in, d_out, 224, 224 * 64);
// pool_2<<<dim3(7, 7, 1), dim3(8, 8, 1)>>>(d_in, d_out, 112, 112 * 128);
// pool_2<<<dim3(4, 4, 1), dim3(8, 8, 1)>>>(d_in, d_out, 56, 56 * 256);
// pool_2<<<dim3(2, 2, 1), dim3(8, 8, 1)>>>(d_in, d_out, 28, 28 * 512);
// pool_2<<<dim3(1, 1, 1), dim3(8, 8, 1)>>>(d_in, d_out, 14, 14 * 512);

// fn<<<128, 32, 32 * sizeof(float)>>>(d_in, d_out, d_kernel_weights, d_kernel_bias, 4096, 7 * 7 * 512);
// fn<<<128, 32, 32 * sizeof(float)>>>(d_in, d_out,d_kernel_weights, d_kernel_bias, 4096, 4096);
// fn<<<32, 32, 32 * sizeof(float)>>>(d_in, d_out, d_kernel_weights, d_kernel_bias, 1000, 4096);


int main(int argc, char** argv) {
	    // set up device
        int dev = 0;
        cudaDeviceProp deviceProp;
        CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    
        // timing setup
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event) ;
        cudaEventCreate(&stop_event);
        float milliseconds = 0;

        srand(time(NULL));
        float err, cpu_time = 0;
        #ifdef POOL
            cout << "matrix: " << MATROW << " * " << MATCOL << endl;
            float* in = (float*)malloc(MATSIZE * sizeof(float));
            float* out = (float*)malloc(MATSIZE / 4 * sizeof(float));
            AssignRandomValue(in, MATSIZE);
            float* d_in, *d_out;
            cudaMalloc(&d_in, MATSIZE * sizeof(float));
            cudaMalloc(&d_out, MATSIZE / 4 * sizeof(float));
            cudaMemcpy(d_in, in, MATSIZE * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start_event, 0);
            pool_1<<<dim3(ceil(MATROW/8.), 8, 1), 32>>>(d_in, d_out, MATROW, MATCOL);
            CHECK(cudaGetLastError());
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&milliseconds, start_event, stop_event);

            cudaMemcpy(out, d_out, MATSIZE / 4 * sizeof(float), cudaMemcpyDeviceToHost);
            pool_unitest(in, out, &err, &cpu_time);
            cout << "pool1 err = " << err << endl;
            cout << "pool1 gpu_time = " << milliseconds << endl;

            cudaEventRecord(start_event, 0);
            pool_2<<<dim3(ceil(MATROW/8.), ceil(MATROW/8.), 1), dim3(8, 8, 1)>>>(d_in, d_out, MATROW, MATCOL);
            CHECK(cudaGetLastError());
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&milliseconds, start_event, stop_event);

            cudaMemcpy(out, d_out, MATSIZE / 4 * sizeof(float), cudaMemcpyDeviceToHost);
            pool_unitest(in, out, &err, &cpu_time);
            cout << "pool2 err = " << err << endl;
            cout << "pool2 gpu_time = " << milliseconds << endl;
            cout << "cpu_time = " << cpu_time << endl;
            cudaFree(d_in);
            cudaFree(d_out);
        #endif
        #ifdef FN
            cout << "matrix: " << MATROW << " * " << MATCOL << endl;
            float* kernel_weights = (float*)malloc(MATSIZE * sizeof(float));
            float* kernel_bias = (float*)malloc(MATROW * sizeof(float));
            float* in = (float*)malloc(MATCOL * sizeof(float));
            float* out = (float*)malloc(MATROW * sizeof(float));
            AssignRandomValue(in, MATCOL);
            AssignRandomValue(kernel_weights, MATSIZE);
            AssignRandomValue(kernel_bias, MATROW);

            float *d_kernel_weights, *d_kernel_bias, *d_in, *d_out;
            cudaMalloc(&d_kernel_weights, MATSIZE * sizeof(float));
            cudaMalloc(&d_kernel_bias, MATROW * sizeof(float));
            cudaMalloc(&d_in, MATCOL * sizeof(float));
            cudaMalloc(&d_out, MATROW * sizeof(float));
            cudaMemcpy(d_in, in, MATCOL * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel_weights, kernel_weights, MATSIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_kernel_bias, kernel_bias, MATROW * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start_event, 0);
            fn<<<ceil(MATROW/32), 32, 32 * sizeof(float)>>>(d_in, d_out, d_kernel_weights, d_kernel_bias, MATROW, MATCOL);
            CHECK(cudaGetLastError());
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&milliseconds, start_event, stop_event);

            cudaMemcpy(out, d_out, MATROW * sizeof(float), cudaMemcpyDeviceToHost);
            fn_unitest(kernel_weights, kernel_bias, in, out, &err, &cpu_time);
            cout << "fn err = " << err << endl;
            cout << "fn gpu_time = " << milliseconds << endl;
            cout << "cpu_time = " << cpu_time << endl;
            cudaFree(d_kernel_weights);
            cudaFree(d_kernel_bias);
            cudaFree(d_in);
            cudaFree(d_out);
        #endif
        #ifdef PRED
            float* in = (float*)malloc(1000 * sizeof(float));
            uint32_t * pred = (uint32_t*)malloc(sizeof(uint32_t));
            AssignRandomValue(in, 1000);

            float* d_in;
            uint32_t * d_pred;
            cudaMalloc(&d_in, 1000 * sizeof(float));
            cudaMalloc(&d_pred, sizeof(uint32_t));
            cudaMemcpy(d_in, in, 1000 * sizeof(float), cudaMemcpyHostToDevice);

            cudaEventRecord(start_event, 0);
            predict<<<1, 1024>>>(d_in, d_pred);
            CHECK(cudaGetLastError());
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
            cudaEventElapsedTime(&milliseconds, start_event, stop_event);

            cudaMemcpy(pred, d_pred, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaFree(d_in);

            const clock_t begin_time = clock();
            uint32_t idx = 0;
            float gr = in[0];
            for(uint32_t i = 1; i < 1000 ; i++){
                if(gr < in[i]){
                    gr = in[i];
                    idx = i;
                }
            }
            cpu_time = float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000; 

            cout << "cpu pred : " << idx << endl;
            cout << "gpu pred : " << *pred << endl;
            cout << "pred gpu_time = " << milliseconds << endl;
            cout << "pred cpu_time = " << cpu_time << endl;
        #endif
        return 1;
}

void pool_unitest(float *in, float * out, float * err, float * cpu_time){

    float** rei_in = new float*[MATCOL];
    for(uint32_t i = 0 ; i < MATCOL ; i++){
        rei_in[i] = new float[MATROW];
    }
    for(uint32_t i=0 ; i<MATCOL ; i++){
        for(uint32_t j=0 ; j<MATROW ; j++){
            rei_in[i][j] = in[MATROW*i+j];
        }
    }

    const clock_t begin_time = clock();
    for(uint32_t i=0 ; i<MATCOL/2 ; i++){
        for(uint32_t j=0 ; j<MATROW/2 ; j++){
            rei_in[i][j] = max(max(max(rei_in[2*i][2*j], rei_in[2*i+1][2*j]), rei_in[2*i][2*j+1]), rei_in[2*i+1][2*j+1]);
        }
    }
    *cpu_time = float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000; 
    *err  = 0;
    for(uint32_t i=0 ; i<MATCOL/2 ; i++){
        for(uint32_t j=0 ; j<MATROW/2 ; j++){
            *err += (out[MATROW/2*i + j] - rei_in[i][j]) * (out[MATROW/2*i + j] - rei_in[i][j]);
        }
    }
    for(uint32_t i = 0 ; i < MATCOL ; i++){
        free(rei_in[i]);
    }
    free(rei_in);
    return;
}

void fn_unitest(float *kernel_weights, float *kernel_bias, float *in, float * out, float * err, float * cpu_time){
    float* rei_out = new float[MATROW];
    memset(rei_out, 0, MATROW * sizeof(float));
    const clock_t begin_time = clock();
    for(uint32_t i = 0 ; i < MATROW ; i++){
        for(uint32_t j = 0 ; j < MATCOL ; j++){
            rei_out[i] += in[j] * kernel_weights[j * MATROW + i];
        }
        rei_out[i] += kernel_bias[i];
    }
    *cpu_time = float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000;
    *err  = 0;
    for(uint32_t i = 0 ; i < MATROW ; i++){
        *err += (rei_out[i] - out[i]) * (rei_out[i] - out[i]);
    }
    free(rei_out);
    return;
}