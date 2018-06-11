#include <stdlib.h>
#include <iostream>
#include <stdio.h>

void AssignRandomValue(float *res, int size, bool randomFlag = true){
    for (int i = 0; i < size; i ++)
    {
        if (randomFlag)
            res[i] = float(rand() % 20001) / 10000 - 1;
        else
            res[i] = 0;
    }
}

void InitWeights_Biases(double *Weights_CPU, int size, char* file_path) {
	//Layer Weights
	FILE * pFile = fopen (file_path,"r");
	if (!pFile) {
		printf("FAIL! INPUT WEIGHTS NOT FOUND! %s\n",file_path);
		exit(1);
	}

	long int i = 0;
	if (pFile != NULL){
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			double temp_num = atof(line);
			Weights_CPU[i] = temp_num;
			i++;
			if(i==size) {
				break;
			}
		}
		fclose(pFile);
	}
}

void LoadInput(double *Data_Layer_CPU,char* file_path) {
	FILE * pFile = fopen (file_path,"rb");
	if (!pFile) {
		printf("FAIL! INPUT FILE NOT FOUND!\n");
		exit(1);
	}
	if (pFile != NULL){
		long int i = 0;
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			double temp_num = atof(line);
			Data_Layer_CPU[i] = temp_num;
			i++;
			if(i==IMAGE_WIDTH*IMAGE_WIDTH*3) {
				printf("compeleted reading img file\n");
				break;
			}
		}
		fclose(pFile);
	}
}

void ConvertInput(double *Data_Layer_CPU)
{
    double *Data_layer_CPU_out = (double*) malloc (CHANNELS*IMAGE_WIDTH*IMAGE_WIDTH * sizeof(double));
	for(int i=0; i<IMAGE_WIDTH*IMAGE_WIDTH*CHANNELS; i+=3)
	{
		Data_Layer_CPU_out[i/3] = Data_Layer_CPU[i];
		Data_Layer_CPU_out[i/3 + IMAGE_WIDTH * IMAGE_WIDTH] = Data_Layer_CPU[i + 1];
		Data_Layer_CPU_out[i/3 + 2 * IMAGE_WIDTH * IMAGE_WIDTH] = Data_Layer_CPU[i + 2];
	}
    Data_Layer_CPU = Data_layer_CPU_out;
}

void LoadImageNetClass(char **image_class, char *file_path){
	FILE * pFile = fopen (file_path,"r");
	if (!pFile) {
		printf("FAIL! INPUT WEIGHTS NOT FOUND! %s\n",file_path);
		exit(1);
	}

	long int i = 0;
	if (pFile != NULL){
		size_t len = 99;
		char *line = NULL;
		while ((getline(&line, &len, pFile)) != -1) {
			strcpy(image_class[i],line);
			i++;
			if(i==1000) {
				break;
			}
		}
		fclose(pFile);
	}
}