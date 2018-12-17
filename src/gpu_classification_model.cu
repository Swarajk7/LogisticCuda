#include "gpu_data_handling.h"
__constant__ float constant_weights[NUM_FEATURES];
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "logistic_regression_kernels.cu"
#include "data_reader.h"

#include "support.h"
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <sys/stat.h>
#include <sys/mman.h>
#include <iostream>

// Allocate a device memory of num_elements float elements
float *AllocateDeviceArray(int num_elements)
{
	float *devArray;
	int size = num_elements * sizeof(float);
	cudaError_t error = cudaMalloc((void **)&devArray, size);
	// if(error == cudaSuccess)
	// 	printf("AllocateDeviceArray -- 1\n");
	// else if(error ==cudaErrorInvalidValue)
	// 	printf("AllocateDeviceArray -- 2\n");
	// else if(error ==cudaErrorInvalidDevicePointer)
	// 	printf("AllocateDeviceArray -- 3\n");
	// else if(error ==cudaErrorInvalidMemcpyDirection)
	// 	printf("AllocateDeviceArray -- 4\n");
	return devArray;
}

void InitailizeDeviceArrayValues(float *devArray, int num_elements, bool random)
{
	int size = num_elements * sizeof(float);
	if (random == 1)
	{
		//Initialize value to some random int. Use Khadanga's function here
		float *hostArray = generate_random_weight(num_elements);
		//printf("%d",num_elements);
		cudaError_t error = cudaMemcpy(devArray, hostArray, size, cudaMemcpyHostToDevice);
		// if(error == cudaSuccess)
		// 	printf("InitailizeDeviceArrayValues  - 1\n");
		// else if(error ==cudaErrorInvalidValue)
		// 	printf("InitailizeDeviceArrayValues  - 2\n");
		// else if(error ==cudaErrorInvalidDevicePointer)
		// 	printf("InitailizeDeviceArrayValues  - 3\n");
		// else if(error ==cudaErrorInvalidMemcpyDirection)
		// 	printf("InitailizeDeviceArrayValues  - 4\n");
	}
	else
	{
		//Initialize values to zeroes
		cudaMemset(devArray, 0, size);
	}
}

void GPUClassificationModel::SetDeviceArrayValues(float *devArray, float *hostArray, int num_elements)
{
	int size = num_elements * sizeof(float);
	//printf("%d  ",size);
	cudaError_t error = cudaMemcpy(devArray, hostArray, size, cudaMemcpyHostToDevice);
	// if(error == cudaSuccess)
	// 	printf("SetDeviceArrayValues  - 1\n");
	// else if(error ==cudaErrorInvalidValue)
	// 	printf("SetDeviceArrayValues  - 2\n");
	// else if(error ==cudaErrorInvalidDevicePointer)
	// 	printf("SetDeviceArrayValues  - 3\n");
	// else if(error ==cudaErrorInvalidMemcpyDirection)
	// 	printf("SetDeviceArrayValues  - 4\n");

	// for(int i=0;i<29;i++) printf("%f ",hostArray[i]);
	// printf("\n");
	// printGpuData(devArray);
}

//Keeping a preTrained flag right now for future use of preTrained weights as well.
void GPUClassificationModel::initializeWeights(bool random, bool preTrained)
{
	weights = AllocateDeviceArray(num_features);
	InitailizeDeviceArrayValues(weights, num_features, random);
}

GPUClassificationModel::GPUClassificationModel(int batch_size, int num_features, bool random)
{
	this->batch_size = batch_size;
	//Taking care of the weight for bias by adding 1
	this->num_features = num_features + 1;
	initializeWeights(random);

	grad_weights = AllocateDeviceArray(this->num_features);
	intermediate_vector = AllocateDeviceArray(batch_size);

	X = AllocateDeviceArray(batch_size * this->num_features);
	X_trans = AllocateDeviceArray(batch_size * this->num_features);
	y = AllocateDeviceArray(batch_size);
	correct_val = AllocateDeviceArray(1);
	cudaMemset(correct_val, 0, sizeof(float));
	host_weights = (float *)malloc(num_features * sizeof(float));
}

GPUClassificationModel::GPUClassificationModel(HIGGSItem item, int num_features, bool random)
{
	batch_size = item.size;
	N = item.N;
	//Taking care of the weight for bias by adding 1
	this->num_features = num_features + 1;
	initializeWeights(random);

	grad_weights = AllocateDeviceArray(this->num_features);
	intermediate_vector = AllocateDeviceArray(batch_size);

	X = AllocateDeviceArray(batch_size * this->num_features);
	y = AllocateDeviceArray(batch_size);
	correct_val = AllocateDeviceArray(1);
	cudaMemset(correct_val, 0, sizeof(float));
	SetDeviceArrayValues(X, item.X, batch_size * this->num_features);
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
	//for(int i=0;i<29;i++) printf("%f ",item.y[i]);
	//printf("\n");
	//printGpuData(X);
	//printf("%d",num_features);
}

float GPUClassificationModel::evaluateModel(HIGGSItem item, bool memory_coalescing)
{
	//Evaluating Kernel code here
	cudaMemset(correct_val, 0, sizeof(float));
	int num_blocks = ceilf((N * 1.0f) / BLOCK_SIZE);
	setData(item);
	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(BLOCK_SIZE, 1, 1);
	evaluate_model<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features, correct_val);
	float *host_correct_val = (float *)malloc(sizeof(float));
	cudaMemcpy(host_correct_val, correct_val, sizeof(float), cudaMemcpyDeviceToHost);
	return *host_correct_val;
}

void GPUClassificationModel::trainModel(HIGGSItem item, bool memory_coalescing, int memory_access_type, float learning_rate)
{
	//training kernel code here
	//We can pass the "this" item also instead of individual values
	//trainingKernel(weights,X,y,memory_coalescing);

	// memory_access_type 1 => Global
	// memory_access_type 2 => shared
	// memory_access_type 3 or everything else => constant
	//int memory_access_type = 1;

	setData(item);

	this->learning_rate = learning_rate;

	//printf("Running trainmodel()\n");

	int num_blocks = ceilf((N * 1.0f) / BLOCK_SIZE);

	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(BLOCK_SIZE, 1, 1);
	const int X_dim = 32;

	if (memory_coalescing)
	{
		if (memory_access_type == 1)
		{
			//Global memory
			BlockSize.x = 800;
			GridSize.x = ceilf((N * 1.0f) / BlockSize.x);
			// TODO: Constant Memory
			//computeForward<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
			//memory_coalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
			//externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights,grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
			/*GridSize.y = NUM_FEATURES;
			BlockSize.x = 1024;
			GridSize.x = ceilf((N * 1.0f) / BlockSize.x);
			//computeGrad<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
			computeGrad_v2<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
			updateGrad<<<1, NUM_FEATURES>>>(weights, grad_weights, learning_rate, N);*/
		}
		else if (memory_access_type == 2)
		{
			//printf("Using shared");
			//shared memory
			//BLOCK_SIZE should be greater than 29
			memory_coalescedKernel_shared<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
			//externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
			GridSize.y = NUM_FEATURES;
			//computeGrad<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
			//computeGrad_v2<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
			//updateGrad<<<1, NUM_FEATURES>>>(weights, grad_weights, learning_rate, N);
		}
		else
		{
			//constant memory
			//printf("Using constant");
			/*cudaMemcpy(host_weights, weights, num_features * sizeof(float), cudaMemcpyDeviceToHost);

			cudaError_t cuda_ret = cudaMemcpyToSymbol(constant_weights, host_weights, num_features * sizeof(float));
			if (cuda_ret != cudaSuccess)
				FATAL("Unable to copy to constant memory");
			*/
			//memory_coalescedKernel<<<GridSize, BlockSize>>>(constant_weights, X, y, intermediate_vector, batch_size, N, num_features);
			BlockSize.x = GRAD_COMP_BLOCK_SIZE;
			GridSize.x = N;
			// TODO: Constant Memory
			computeForward<<<GridSize, BlockSize>>>(weights,X, y, intermediate_vector, batch_size, N, num_features);

			//cudaMemcpy(weights, &constant_weights[0], num_features * sizeof(float), cudaMemcpyDeviceToDevice);

			//externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);

			// TODO: Transpose Kernel
			dim3 transpose_grid(ceilf((NUM_FEATURES * 1.0f) / TILE_DIM), ceilf((N * 1.0f) / TILE_DIM));
			dim3 transpose_block(TILE_DIM, TILE_DIM);
			tranposeKernel<<<transpose_grid, transpose_block>>>(X_trans, X, N);

			GridSize.y = NUM_FEATURES;
			BlockSize.x = GRAD_COMP_BLOCK_SIZE;
			GridSize.x = ceilf((N * 1.0f) / BlockSize.x);
			//computeGrad<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
			computeGrad_v2<<<GridSize, BlockSize>>>(grad_weights, X_trans, intermediate_vector, batch_size, N, num_features, learning_rate);
			updateGrad<<<1, NUM_FEATURES>>>(weights, grad_weights, learning_rate, N);
		}
	}
	else
	{
		uncoalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
		externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
	}
}

void GPUClassificationModel::printWeights()
{
	printKernel<<<dim3(1, 1), dim3(1, 1)>>>(weights, intermediate_vector, num_features);
}

void GPUClassificationModel::printIntermediateValue()
{
	printKernel<<<dim3(1, 1), dim3(1, 1)>>>(weights, intermediate_vector, num_features);
}

void GPUClassificationModel::printGpuData(float *array)
{
	printKernel<<<dim3(1, 1), dim3(1, 1)>>>(array, intermediate_vector, num_features);
}

void dbl_buffer(HIGGSDataset &dataset, HIGGSDataset &valdataset, GPUClassificationModel &model, int batch_size, const char *file_name, int epoch)
{
	dataset.reset();
	//static const size_t host_buffer_size = 512 * 1024;
	int fd = -1;
	static const size_t host_buffer_size_X = batch_size * (HIGGSDataset::NUMBER_OF_FEATURE + 1) * sizeof(float);
	static const size_t host_buffer_size_y = batch_size * sizeof(float);
	struct stat file_stat;
	cudaError_t cuda_ret;
	cudaStream_t cuda_stream;
	cudaEvent_t tmp_event, active_event, passive_event;
	HIGGSItem *active, *passive, *tmp, *batch;

	batch = new HIGGSItem();
	batch->allocateMemory(batch_size);

	float *host_buffer_X, *host_buffer_y;
	float *current_X, *current_y;

	active = new HIGGSItem();
	active->size = batch_size;

	passive = new HIGGSItem();
	passive->size = batch_size;

	tmp = new HIGGSItem();

	float *device_buffer_X, *device_buffer_y;

	//float bw;
	clock_t start_time, end_time;
	double total_time, total_time_by_dataset;

	/* Open the file */
	if ((fd = open(file_name, O_RDONLY)) < 0)
		FATAL("Unable to open %s", file_name);

	if (fstat(fd, &file_stat) < 0)
		FATAL("Unable to read meta data for %s", file_name);
	/* Create CUDA stream for asynchronous copies */

	cuda_ret = cudaStreamCreate(&cuda_stream);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to create CUDA stream");
	/* Create CUDA events */
	cuda_ret = cudaEventCreate(&active_event);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to create CUDA event");
	cuda_ret = cudaEventCreate(&passive_event);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to create CUDA event");
	/* Allocate a big chunk of host pinned memory */
	cuda_ret = cudaHostAlloc(&host_buffer_X, 2 * host_buffer_size_X, cudaHostAllocDefault);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to allocate host X memory");
	cuda_ret = cudaHostAlloc(&host_buffer_y, 2 * host_buffer_size_y, cudaHostAllocDefault);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to allocate host y memory");
	// TODO: check file stat
	cuda_ret = cudaMalloc(&device_buffer_X, file_stat.st_size);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to allocate device memory");
	cuda_ret = cudaMalloc(&device_buffer_y, file_stat.st_size);

	/* Start transferring */
	/* Queue dummy first event */
	cuda_ret = cudaEventRecord(active_event, cuda_stream);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to queue CUDA event");

	active->X = host_buffer_X;
	active->y = host_buffer_y;

	passive->X = host_buffer_X + host_buffer_size_X / sizeof(float);
	passive->y = host_buffer_y + host_buffer_size_y / sizeof(float);

	for (int i = 0; i < epoch; i++)
	{
		dataset.reset();
		start_time = std::clock();
		current_X = device_buffer_X;
		current_y = device_buffer_y;
		// TODO: Update file_stat accordingly.
		//pending = file_stat.st_size;
		/* Start the copy machine */
		while (dataset.hasNext())
		{
			/* Make sure CUDA is not using the buffer */
			cuda_ret = cudaEventSynchronize(active_event);
			if (cuda_ret != cudaSuccess)
				FATAL("Unable to wait for event");
			//transfer_size = (pending > host_buffer_size_y) ? host_buffer_size_y : pending;

			//Item load using dataset.dataset.read(active);
			dataset.getNextBatch(false, active);
			/*if (read(fd, active, transfer_size) < transfer_size)
			FATAL("Unable to read data from %s", file_name);*/

			/* Send data to the device asynchronously */
			cuda_ret = cudaMemcpyAsync(current_X, active->X, host_buffer_size_X, cudaMemcpyHostToDevice, cuda_stream);
			if (cuda_ret != cudaSuccess)
				FATAL("Unable to copy data to device memory");
			cuda_ret = cudaMemcpyAsync(current_y, active->y, host_buffer_size_y, cudaMemcpyHostToDevice, cuda_stream);
			if (cuda_ret != cudaSuccess)
				FATAL("Unable to copy data to device memory");
			/*
			Kernel call in the stream. 
		*/
			model.trainBatchInStream(active->X, active->y, active->N, true, 0.0001, cuda_stream);
			/* Record event to know when the buffer is idle */
			cuda_ret = cudaEventRecord(active_event, cuda_stream);
			if (cuda_ret != cudaSuccess)
				FATAL("Unable to queue CUDA event");
			/* Update counters and buffers */
			// TODO: Size check
			current_X = current_X + batch_size * (HIGGSDataset::NUMBER_OF_FEATURE + 1);
			current_y = current_y + batch_size;
			tmp = active;
			active = passive;
			passive = tmp;
			tmp_event = active_event;
			active_event = passive_event;
			passive_event = tmp_event;
		}

		cuda_ret = cudaStreamSynchronize(cuda_stream);
		if (cuda_ret != cudaSuccess)
			FATAL("Unable to wait for device");
		end_time = std::clock();
		total_time += (end_time - start_time) / (double)CLOCKS_PER_SEC;
		total_time_by_dataset += dataset.total_time_taken;

		float total = 0;
		float corr = 0;
		valdataset.reset();
		while (valdataset.hasNext())
		{
			valdataset.getNextBatch(true, batch);
			total += batch->N;
			corr += model.evaluateModel(*batch, true);
		}
		std::cout << "Validation Accuracy From GPU: " << corr / total << std::endl;

		// bw = 1.0f * file_stat.st_size / (total_time) / (1024 * 1024);
		// fprintf(stdout, "%d MB in %f sec : %f MBps\n", file_stat.st_size / (1024 * 1024), total_time, bw);
		// printf("Time take by data processing: %f , Total Number of Batch Processed: %f\n",
		// 	   dataset.total_time_taken, dataset.total_batch_executed);
		// printf("Computation Time : %f", total_time - dataset.total_time_taken);
	}

	std::cout << "\n********************************************" << endl;
	std::cout << "Total time taken: " << total_time << endl;
	std::cout << "Total time taken by data processing: " << total_time_by_dataset << endl;
	std::cout << "Computation Time  After " << epoch << "epochs: " << total_time - total_time_by_dataset << endl;
	std::cout << "Average Computation Time GPU :" << (total_time - total_time_by_dataset) / epoch << endl;
	std::cout << "********************************************\n"
			  << endl;

	cuda_ret = cudaFree(device_buffer_X);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to free device memory");
	cuda_ret = cudaFree(device_buffer_y);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to free device memory");
	cuda_ret = cudaFreeHost(host_buffer_X);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to free host memory");
	cuda_ret = cudaFreeHost(host_buffer_y);
	if (cuda_ret != cudaSuccess)
		FATAL("Unable to free host memory");
	close(fd);

	fprintf(stdout, "File Size %d", file_stat.st_size);
}

void GPUClassificationModel::trainBatchInStream(float *X, float *y, int N, bool memory_coalescing, float learning_rate, cudaStream_t & stream)
{
	//printf("Running trainmodelInStrean() ");
	int num_threads_p_block = BLOCK_SIZE;
	int num_blocks = ceilf((N * 1.0f) / num_threads_p_block);

	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(num_threads_p_block, 1, 1);

	//cudaMemcpyAsync(host_weights, weights, num_features * sizeof(float), cudaMemcpyDeviceToHost, stream);

	//cudaError_t cuda_ret = cudaMemcpyToSymbolAsync(constant_weights, host_weights, num_features * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
	//if (cuda_ret != cudaSuccess)
	//	FATAL("Unable to copy to constant memory");

	//memory_coalescedKernel<<<GridSize, BlockSize>>>(constant_weights, X, y, intermediate_vector, batch_size, N, num_features);
	BlockSize.x = GRAD_COMP_BLOCK_SIZE;
	GridSize.x = N;
	// TODO: Constant Memory
	computeForward<<<GridSize, BlockSize, 0, stream>>>(weights,X, y, intermediate_vector, batch_size, N, num_features);

	//cudaMemcpy(weights, &constant_weights[0], num_features * sizeof(float), cudaMemcpyDeviceToDevice);

	//externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);

	// TODO: Transpose Kernel
	dim3 transpose_grid(ceilf((NUM_FEATURES * 1.0f) / TILE_DIM), ceilf((N * 1.0f) / TILE_DIM));
	dim3 transpose_block(TILE_DIM, TILE_DIM);
	tranposeKernel<<<transpose_grid, transpose_block, 0, stream>>>(X_trans, X, N);

	GridSize.y = NUM_FEATURES;
	BlockSize.x = GRAD_COMP_BLOCK_SIZE;
	GridSize.x = ceilf((N * 1.0f) / BlockSize.x);
	//computeGrad<<<GridSize, BlockSize>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, learning_rate);
	computeGrad_v2<<<GridSize, BlockSize, 0, stream>>>(grad_weights, X_trans, intermediate_vector, batch_size, N, num_features, learning_rate);
	updateGrad<<<1, NUM_FEATURES, 0, stream>>>(weights, grad_weights, learning_rate, N);
	cudaDeviceSynchronize();
}