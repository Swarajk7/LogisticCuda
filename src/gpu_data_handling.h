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
	float learning_rate;

  private:
	void initializeWeights(bool random = false, bool preTrained = false);

  public:
	GPUClassificationModel(int batch_size, int num_features = 28, bool random = false);
	GPUClassificationModel(HIGGSItem item, int num_features = 28, bool random = false);
	void resetWeights(bool random = false);
	void setData(HIGGSItem item);
	void evaluateModel(HIGGSItem item, bool memory_coalescing);
	void trainModel(HIGGSItem item,bool memory_coalescing,float learning_rate);
};