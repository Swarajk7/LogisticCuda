#define BLOCK_SIZE 32
#define NUM_FEATURES 785
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define GRAD_COMP_BLOCK_SIZE 512
#include "data_reader.h"
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include "support.h"
#include <ctime>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <sys/mman.h>

class GPUClassificationModel
{
	float *weights;
	float *X;
	float *y;
	float *X_trans;
	float *grad_weights;
	float *intermediate_vector;
	float *correct_val;
	float *host_weights;

	//__constant__ float constant_weights[29];

	int batch_size;
	int N;
	int num_features;
	float learning_rate;

private:
	void initializeWeights(bool random = false, bool preTrained = false);

public:
	GPUClassificationModel(int batch_size, int num_features = 28, bool random = false);
	GPUClassificationModel(HIGGSItem item, int num_features = 28, bool random = false);
	void resetWeights(bool random = false);
	void setData(HIGGSItem item);
	float evaluateModel(HIGGSItem item, bool memory_coalescing);
	//void trainModel(HIGGSItem item,bool memory_coalescing,int memory_access_type,float learning_rate);
	void trainModel(HIGGSItem item,bool memory_coalescing,int memory_access_type,float learning_rate);
	void printWeights();
	void printIntermediateValue();
	void printGpuData(float *array);
	void SetDeviceArrayValues(float *devArray, float *hostArray, int num_elements);
	void trainBatchInStream(float *, float *, int, bool, float, cudaStream_t&);
};

void dbl_buffer(HIGGSDataset &dataset, HIGGSDataset &valdataset, GPUClassificationModel &model, int batch_size, const char *file_name, int epoch);