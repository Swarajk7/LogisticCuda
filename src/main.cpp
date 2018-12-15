#include <iostream>
#include "data_reader.h"
#include "logistic.h"
#include "gpu_data_handling.h"
#include <cuda.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
	// HIGGSDataset dataset("./data/sample.csv", 10);
	// HIGGSItem item = dataset.getNextBatch(false);
	// ofstream writer;
	// writer.open("./data/sample_copy.csv", std::ios::out);
	// int xindex = 0.0f;
	// for (int j = 0; j < item.N; j++)
	// {
	// 	for (int i = 0; i < HIGGSDataset::NUMBER_OF_FEATURE + 1; i++)
	// 	{
	// 		writer << item.X[xindex + i];
	// 		if (i != HIGGSDataset::NUMBER_OF_FEATURE)
	// 			writer << ",";
	// 	}
	// 	xindex += HIGGSDataset::NUMBER_OF_FEATURE + 1;
	// 	writer << endl;
	// }
	// writer.close();
	int batch_size = 100;
	if (argc >= 2)
	{
		batch_size = stoi(argv[1]);
		std::cout << "Batch Size: " << batch_size << endl;
	}
	HIGGSDataset dataset("./data/HIGGS_Sample.csv", batch_size);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	int batch_no = 0;

//Added by anand
    dbl_buffer(2,"./data/HIGGS_Sample.csv");
	HIGGSDataset valdataset("./data/HIGGS_Sample_Val.csv", batch_size);
	// std::cout << classifier.evaluate(valdataset);

	// for (int i = 0; i < 10; i++)
	// {
	// 	int correct = 0, total = 0;
	// 	dataset.reset();
	// 	while (dataset.hasNext())
	// 	{
	// 		/*
	// 		1. train logistic
	// 		2. compute accuracy for validation set
	// 		*/
	// 		++batch_no;
	// 		HIGGSItem batch = dataset.getNextBatch(false);
	// 		if (batch.N == batch_size)
	// 			correct += classifier.trainBatch(batch, 0.0001);
	// 		total += batch.N;
	// 	}
	// 	std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
	// 	std::cout << "Evaluating! Accuracy: ";
	// 	valdataset.reset();
	// 	std::cout << classifier.evaluate(valdataset) << endl;
	// }

	GPUClassificationModel model(batch_size, HIGGSDataset::NUMBER_OF_FEATURE, true);
	//model.printWeights();


	for (int i = 0; i < 10; i++)
	{
		int correct = 0, total = 0;
		dataset.reset();
		while (dataset.hasNext())
		{
			/*
			1. train logistic
			2. compute accuracy for validation set
			*/
			++batch_no;
			HIGGSItem batch = dataset.getNextBatch(true);
			//model.setData(batch);
			if (batch.N == batch_size)
				model.trainModel(batch,true,3,0.0001);
			total += batch.N;
			if(batch_no ==1) {
				//for(int i=0;i<29;i++) printf("%f ",batch.X[i]);
				printf("\n");
				//model.printWeights();
			}
			free(batch.X);
			free(batch.y);
			//printf("WEIGHTS: \n");
			//model.printWeights();

			//cudaDeviceSynchronize();
			//usleep(3000);

			//printf("Intermediate: \n");
			//model.printIntermediateValue();
		}
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
		//model.printWeights();
		std::cout << "Evaluating! Accuracy: ";
		valdataset.reset();
		
		total = 0;
		float corr=0;
		while(valdataset.hasNext()) {
			HIGGSItem item = valdataset.getNextBatch(true);
			total += item.N;
			corr+=model.evaluateModel(item,true);
			free(item.X);
			free(item.y);
		}
		std::cout << corr/total << std::endl;
	}
}
