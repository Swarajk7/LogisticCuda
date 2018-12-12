#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "util.h"
#include "gpu_data_handling.h"
#include "logistic_regression_kernels.cu"

// Allocate a device memory of num_elements float elements
float * AllocateDeviceArray(int num_elements)
{
    float * devArray;
    int size = num_elements * sizeof(float);
    cudaMalloc((void**)&devArray, size);
    return devArray;
}

void InitailizeDeviceArrayValues(float * devArray,int num_elements, int random = 0)
{
	int size = num_elements * sizeof(float);
	if(random==1){
		//Initialize value to some random int. Use Khadanga's function here
		float * hostArray = generate_random_weight(num_elements);
		cudaMemcpy(devArray,hostArray,size,cudaMemcpyHostToDevice);
	}
	else{
		//Initialize values to zeroes
		cudaMemset(array, 0, size);
	}
}

void SetDeviceArrayValues(float * devArray,float * hostArray, int num_elements){
	int size = num_elements*sizeof(float);
	cudaMemcpy(devArray,hostArray,size,cudaMemcpyHostToDevice);
}

 
void GPUClassificationModel::initializeWeights(int random=0){
		weights = AllocateDeviceArray(num_features);
		InitailizeDeviceArrayValues(weights,num_features,random);
	}
	
 GPUClassificationModel::GPUClassificationModel(int batch_size, int num_features = 28, int random = 0){
		batch_size = batch_size;
		//Taking care of the weight for bias by adding 1
		num_features = num_features+1;
		initializeWeights(random);
		X = AllocateDeviceArray(batch_size*(num_features));
		y = AllocateDeviceArray(batch_size);
	}
	
GPUClassificationModel::GPUClassificationModel(HIGGSItem item, int num_features = 28, int random = 0){
		batch_size = item.size;
		N = item.N;
		//Taking care of the weight for bias by adding 1
		num_features = num_features+1;
		initializeWeights(random);
		X = AllocateDeviceArray(batch_size*num_features);
		y = AllocateDeviceArray(batch_size);
		SetDeviceArrayValues(X,item.X,batch_size*num_features);
		SetDeviceArrayValues(y,item.y,batch_size);
	}
	
void GPUClassificationModel::resetWeights(int random = 0){
		InitailizeDeviceArrayValues(weights,num_features,random);
	}
	

void GPUClassificationModel::setData(HIGGSItem item){
		if(item.size != batch_size){
			printf("Data size mismatch");
			return;
		}
		SetDeviceArrayValues(X,item.X,batch_size*num_features);
		SetDeviceArrayValues(y,item.y,batch_size);
	}

void GPUClassificationModel::evaluateModel(){
		//Evaluating Kernel code here
	}

void GPUClassificationModel::trainModel(){
		//training kernel code here
}