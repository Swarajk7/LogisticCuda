#include <iostream>
#include "data_reader.h"
#include "logistic.h"
#include "gpu_data_handling.h"
#include <cuda.h>
#include <ctime>
#include <unistd.h>

int main(int argc, char *argv[])
{
	clock_t start_time, end_time;
	double total_train_time = 0, total_evaluation_time = 0;
	double total_cpu_train_time = 0;

	int batch_size = 100;
	if (argc >= 2)
	{
		batch_size = stoi(argv[1]);
		std::cout << "Batch Size: " << batch_size << endl;
	}
	HIGGSItem *batch = new HIGGSItem();
	batch->allocateMemory(batch_size);
	GPUClassificationModel model(batch_size, HIGGSDataset::NUMBER_OF_FEATURE, true);
	HIGGSDataset dataset("./data/HIGGS_Sample.csv", batch_size);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	HIGGSDataset valdataset("./data/HIGGS_Sample_Val.csv", batch_size);
	int batch_no = 0;

	// std::cout << classifier.evaluate(valdataset);

	for (int i = 0; i < 1; i++)
	{
		int correct = 0, total = 0;
		dataset.reset();

		total_cpu_train_time = 0;
		std::cout << "Strated CPU timing for the epoch\n";

		start_time = std::clock();

		while (dataset.hasNext())
		{
			/*
			1. train logistic
			2. compute accuracy for validation set
			*/
			++batch_no;
			dataset.getNextBatch(false, batch);
			correct += classifier.trainBatch(*batch, 0.0001);
			total += batch->N;
		}
		end_time = std::clock();
		total_cpu_train_time += (end_time - start_time) / (double)CLOCKS_PER_SEC;
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
		std::cout << "Total time taken: " << total_cpu_train_time << endl;
		std::cout << "Total time taken by data processing: " << dataset.total_time_taken << endl;
		std::cout << "Computation Time : " << total_cpu_train_time - dataset.total_time_taken << endl;
		valdataset.reset();
		//std::cout << "Validating Accuracy at CPU: " << classifier.evaluate(valdataset) << endl;
	}
	//model.printWeights();

	// GPU Code Starts here.
	for (int i = 0; i < 1; i++)
	{
		int correct = 0, total = 0;
		dataset.reset();

		std::cout << "Strated GPU timing for the epoch\n";

		total_train_time = 0;
		start_time = std::clock();

		while (dataset.hasNext())
		{
			/*
			1. train logistic
			2. compute accuracy for validation set
			*/
			++batch_no;
			dataset.getNextBatch(true, batch);

			model.trainModel(*batch, true, 1, 0.0001);
			total += batch->N;

			//printf("WEIGHTS: \n");
			//model.printWeights();

			//cudaDeviceSynchronize();
			//usleep(3000);

			//printf("Intermediate: \n");
			//model.printIntermediateValue();
		}

		end_time = std::clock();
		total_train_time += (end_time - start_time) / (double)CLOCKS_PER_SEC;
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
		std::cout << "Total time taken: " << total_train_time << endl;
		std::cout << "Total time taken by data processing: " << dataset.total_time_taken << endl;
		std::cout << "Computation Time : " << total_train_time - dataset.total_time_taken << endl;
		//model.printWeights();
		valdataset.reset();

		total = 0;
		float corr = 0;
		while (valdataset.hasNext())
		{
			valdataset.getNextBatch(true, batch);
			total += batch->N;
			start_time = std::clock();
			corr += model.evaluateModel(*batch, true);
			end_time = std::clock();
			total_evaluation_time = (end_time - start_time) / (double)CLOCKS_PER_SEC;
		}
		std::cout << "Validation Accuracy From GPU: " << corr / total << std::endl;
	}

	dataset.reset();
	GPUClassificationModel model2(batch_size, HIGGSDataset::NUMBER_OF_FEATURE, true);
	dbl_buffer(dataset, model2, batch_size, "./data/HIGGS_Sample.csv");
}
