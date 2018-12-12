class GPUClassificationModel
{
    float * weights;
	float * X;
	float * y;
	int batch_size;
	int N;
	int num_features;

  private:
	void initializeWeights(int random=0);
	
  public:
	GPUClassificationModel(int batch_size, int num_features = 28, int random = 0);
	
	GPUClassificationModel(HIGGSItem item, int num_features = 28, int random = 0);
	
	void resetWeights(int random = 0);
	
	void setData(HIGGSItem item);

	void evaluateModel();

	void trainModel();
};