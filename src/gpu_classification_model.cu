#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "gpu_data_handling.h"
#include "logistic_regression_kernels.cu"

#include "support.h"
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>
#include <sys/stat.h>
#include <sys/mman.h>


__constant__ float constant_weights[29];

// Allocate a device memory of num_elements float elements
float * AllocateDeviceArray(int num_elements)
{
	float * devArray;
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
	y = AllocateDeviceArray(batch_size);
	correct_val = AllocateDeviceArray(1);
	cudaMemset(correct_val, 0, sizeof(float));
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
	SetDeviceArrayValues(X, item.X, batch_size*num_features);
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
	int num_threads_p_block = BLOCK_SIZE;
	int num_blocks = ceilf((N * 1.0f) / num_threads_p_block);

	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(num_threads_p_block, 1, 1);
	evaluate_model<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features,correct_val);
	float * host_correct_val = (float *)malloc(sizeof(float));
	cudaMemcpy(host_correct_val,correct_val,sizeof(float),cudaMemcpyDeviceToHost);
	return *host_correct_val;
}

void GPUClassificationModel::trainModel(HIGGSItem item, bool memory_coalescing,int memory_access_type,float learning_rate)
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

	int num_threads_p_block = BLOCK_SIZE;
	int num_blocks = ceilf((N * 1.0f) / num_threads_p_block);

	dim3 GridSize(num_blocks, 1, 1);
	dim3 BlockSize(num_threads_p_block, 1, 1);
	const int X_dim = 32;

	if (memory_coalescing)
	{
		if(memory_access_type == 1){
			//Global memory
			memory_coalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
			externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights,grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
		}
		else if(memory_access_type == 2){
			//printf("Using shared");
			//shared memory
			//BLOCK_SIZE should be greater than 29
			memory_coalescedKernel_shared<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
			externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights,grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
		}
		else{
			//constant memory
			//printf("Using constant");
			float * host_weights = (float *)malloc(num_features*sizeof(float));
			cudaMemcpy(host_weights,weights,num_features*sizeof(float),cudaMemcpyDeviceToHost);

			cudaMemcpyToSymbol(constant_weights,host_weights,num_features*sizeof(float));

			memory_coalescedKernel<<<GridSize, BlockSize>>>(constant_weights, X, y, intermediate_vector, batch_size, N, num_features);

			cudaMemcpy(weights,&constant_weights[0],num_features*sizeof(float),cudaMemcpyDeviceToDevice);
			
			externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights,grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
			free(host_weights);
		}
	}
	else
	{
		uncoalescedKernel<<<GridSize, BlockSize>>>(weights, X, y, intermediate_vector, batch_size, N, num_features);
		externalKernel<<<dim3(1, 1, 1), dim3(X_dim, num_features, 1)>>>(weights, grad_weights, X, intermediate_vector, batch_size, N, num_features, X_dim, learning_rate);
	}
}

void GPUClassificationModel::printWeights(){	
	printKernel<<<dim3(1,1),dim3(1,1)>>>(weights,intermediate_vector,num_features);
}


void GPUClassificationModel::printIntermediateValue(){	
	printKernel<<<dim3(1,1),dim3(1,1)>>>(weights,intermediate_vector,num_features);
}

void GPUClassificationModel::printGpuData(float * array){	
	printKernel<<<dim3(1,1),dim3(1,1)>>>(array,intermediate_vector,num_features);
}


void dbl_buffer(int num,const char* file_name)
{

//static const size_t host_buffer_size = 512 * 1024;
//int main(int argc, char *argv[])
//{
    int fd = -1;
    static const size_t host_buffer_size = 1024 * 1024;
    struct stat file_stat;
    cudaError_t cuda_ret;
    cudaStream_t cuda_stream;
    cudaEvent_t tmp_event,active_event, passive_event;
    void *host_buffer, *device_buffer, *active, *passive, *tmp, *current;
    size_t pending, transfer_size;

//    Timer timer;
  //  timestamp_t start_time, end_time;
    float bw;
clock_t start_time,end_time;
        double total_time;	
	srand(2012);

    if(num < 2) FATAL("Bad argument count");
    /* Open the file */
    if((fd = open(file_name, O_RDONLY)) < 0)
        FATAL("Unable to open %s", file_name);

    if(fstat(fd, &file_stat) < 0)
        FATAL("Unable to read meta data for %s", file_name);
    /* Create CUDA stream for asynchronous copies */
   
     cuda_ret = cudaStreamCreate(&cuda_stream);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA stream");
    /* Create CUDA events */
    cuda_ret = cudaEventCreate(&active_event);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA event");
    cuda_ret = cudaEventCreate(&passive_event);
    if(cuda_ret != cudaSuccess) FATAL("Unable to create CUDA event");
    /* Allocate a big chunk of host pinned memory */
    cuda_ret = cudaHostAlloc(&host_buffer, 2 * host_buffer_size,cudaHostAllocDefault);
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate host memory");
    cuda_ret = cudaMalloc(&device_buffer, file_stat.st_size);
    if(cuda_ret != cudaSuccess) 
        FATAL("Unable to allocate device memory");

/* Start transferring */
 //   start_time = get_timestamp();
 //    startTime(&timer); 
   start_time = std::clock();
    /* Queue dummy first event */
    cuda_ret =  cudaEventRecord(active_event, cuda_stream);
    if(cuda_ret != cudaSuccess) FATAL("Unable to queue CUDA event");
   
    active = host_buffer; 
    passive = (uint8_t *)host_buffer + host_buffer_size;
    current = device_buffer; pending = file_stat.st_size;
	/* Start the copy machine */
    while(pending > 0) {
        /* Make sure CUDA is not using the buffer */
        cuda_ret = cudaEventSynchronize(active_event);
        if(cuda_ret != cudaSuccess) FATAL("Unable to wait for event");
        transfer_size = (pending > host_buffer_size) ? host_buffer_size : pending;
        if(read(fd, active, transfer_size) < transfer_size)
            FATAL("Unable to read data from %s", file_name);

/* Send data to the device asynchronously */
        cuda_ret = cudaMemcpyAsync(current, active, transfer_size,cudaMemcpyHostToDevice, cuda_stream);
        if(cuda_ret != cudaSuccess)
            FATAL("Unable to copy data to device memory");
        /* Record event to know when the buffer is idle */
        cuda_ret = cudaEventRecord(active_event, cuda_stream);
        if(cuda_ret != cudaSuccess) FATAL("Unable to queue CUDA event");
        /* Update counters and buffers */
        pending = pending - transfer_size;
        current = (uint8_t *) current + transfer_size;
        tmp = active; active = passive; passive = tmp;
        tmp_event = active_event; active_event = passive_event;
        passive_event = tmp_event;
    } 
   
    cuda_ret = cudaStreamSynchronize(cuda_stream);
    if(cuda_ret != cudaSuccess) FATAL("Unable to wait for device");
    end_time = std::clock();
    total_time = (end_time - start_time)/(double)CLOCKS_PER_SEC; 
   printf("%f s\n", total_time);   

  bw = 1.0f * file_stat.st_size / (total_time);
   fprintf(stdout, "%d bytes in %f msec : %f MBps\n", file_stat.st_size,1e-3f * total_time, bw);
    cuda_ret = cudaFree(device_buffer);
   if(cuda_ret != cudaSuccess) FATAL("Unable to free device memory");
   cuda_ret = cudaFreeHost(host_buffer);
   if(cuda_ret != cudaSuccess) FATAL("Unable to free host memory");
   close(fd);

   fprintf(stdout,"File Size %d", file_stat.st_size);
// return 0;
//}

}
