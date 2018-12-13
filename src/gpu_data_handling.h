#define BLOCK_SIZE 32
#include "data_reader.h"

class GPUClassificationModel
{
	float *weights;
	float *X;
	float *y;
	float *grad_weights;
	float *intermediate_vector;
	int batch_size;
	int N;
	int num_features;

private:
	void initializeWeights(bool random = 0, bool preTrained = 0);

public:
	GPUClassificationModel(int batch_size, int num_features = 28, bool random = 0);
	GPUClassificationModel(HIGGSItem item, int num_features = 28, bool random = 0);
	void resetWeights(bool random = 0);
	void setData(HIGGSItem item);
	void evaluateModel();
	void trainModel(bool memory_coalescing);
};