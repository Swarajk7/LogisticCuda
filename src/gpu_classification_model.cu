#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "gpu_data_handling.h"
#include "logistic_regression_kernels.cu"

// Allocate a device memory of num_elements float elements
float *AllocateDeviceArray(int num_elements)
{
	float *devArray;
	int size = num_elements * sizeof(float);
	cudaMalloc((void **)&devArray, size);
	return devArray;
}

void InitailizeDeviceArrayValues(float *devArray, int num_elements, bool random)
{
	int size = num_elements * sizeof(float);
	if (random == 1)
	{
		//Initialize value to some random int. Use Khadanga's function here
		float *hostArray = generate_random_weight(num_elements);
		cudaMemcpy(devArray, hostArray, size, cudaMemcpyHostToDevice);
	}
	else
	{
		//Initialize values to zeroes
		cudaMemset(devArray, 0, size);
	}
}

void SetDeviceArrayValues(float *devArray, float *hostArray, int num_elements)
{
	int size = num_elements * sizeof(float);
	cudaMemcpy(devArray, hostArray, size, cudaMemcpyHostToDevice);
}

//Keeping a preTrained flag right now for future use of preTrained weights as well.
void GPUClassificationModel::initializeWeights(bool random, bool preTrained)
{
	weights = AllocateDeviceArray(num_features);
	InitailizeDeviceArrayValues(weights, num_features, random);
}

GPUClassificationModel::GPUClassificationModel(int batch_size, int num_features, bool random)
{
	batch_size = batch_size;
	//Taking care of the weight for bias by adding 1
	num_features = num_features + 1;
	initializeWeights(random);

	grad_weights = AllocateDeviceArray(num_features);
	intermediate_vector = AllocateDeviceArray(batch_size);

	X = AllocateDeviceArray(batch_size * num_features);
	y = AllocateDeviceArray(batch_size);
}

GPUClassificationModel::GPUClassificationModel(HIGGSItem item, int num_features, bool random)
{
	batch_size = item.size;
	N = item.N;
	//Taking care of the weight for bias by adding 1
	num_features = num_features + 1;
	initializeWeights(random);

	grad_weights = AllocateDeviceArray(num_features);
	intermediate_vector = AllocateDeviceArray(batch_size);

	X = AllocateDeviceArray(batch_size * num_features);
	y = AllocateDeviceArray(batch_size);
	SetDeviceArrayValues(X, item.X, batch_size * num_features);
	SetDeviceArrayValues(y, item.y, batch_size);
}

void GPUClassificationModel::resetWeights(bool random)
{
	InitailizeDeviceArrayValues(weights, num_features, random);
}

void GPUClassificationModel::setData(HIGGSItem item)
{
	if (item.size != batch_size)
	{
		printf("Data size mismatch");
		return;
	}
	N = item.N;
	SetDeviceArrayValues(X, item.X, batch_size * num_features);
	SetDeviceArrayValues(y, item.y, batch_size);
}

void GPUClassificationModel::evaluateModel()
{
	//Evaluating Kernel code here
}

void GPUClassificationModel::trainModel(bool memory_coalescing)
{
	//training kernel code here
	//We can pass the "this" item also instead of individual values
	//trainingKernel(weights,X,y,memory_coalescing);

	int num_threads_p_block = BLOCK_SIZE;
	int num_blocks = ceilf((N * 1.0f) / num_threads_p_block);

	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(num_threads_p_block, 1, 1);
	int X_dim = 32;
	int size = 10;

	if (memory_coalescing)
	{
		memory_coalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, size, N, num_features);
		externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(grad_weights, X, intermediate_vector, size, N, num_features, X_dim);
			//Subtract W with GradWeights
			for (int i = 0; i < num_features; i++)
		{
			weights[i] -= grad_weights[i];
		}
	}
	else
	{
		uncoalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, size, N, num_features);
		externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(grad_weights, X, intermediate_vector, size, N, num_features, X_dim);
			//Subtract W with GradWeights
			for (int i = 0; i < num_features; i++)
		{
			weights[i] -= grad_weights[i];
		}
	}
}