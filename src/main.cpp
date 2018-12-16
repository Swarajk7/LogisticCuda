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

	double total_time_by_dataset = 0;

	int memory_type = 1;
	int batch_size = 100;
	int epoch = 10;

	if (argc >= 2)
	{
		batch_size = stoi(argv[1]);
	}
	std::cout << "Batch Size: " << batch_size << endl;

	if (argc >= 3)
	{
		memory_type = stoi(argv[2]);
	}
	std::cout << "Using memory type: " << memory_type << endl;

	if (argc >= 4)
	{
		epoch = stoi(argv[3]);
	}
	std::cout << "Running for epochs : " << epoch << endl;

	HIGGSItem *batch = new HIGGSItem();
	batch->allocateMemory(batch_size);

	HIGGSDataset dataset("./data/train_digits.csv", batch_size);
	HIGGSDataset valdataset("./data/train_digits.csv", batch_size);

	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	GPUClassificationModel model(batch_size, HIGGSDataset::NUMBER_OF_FEATURE, true);

	total_cpu_train_time = 0;
	total_time_by_dataset = 0;
	std::cout << "Strated CPU timing for the epoch\n";

	for (int i = 0; i < epoch; i++)
	{
		int correct = 0, total = 0;
		dataset.reset();
		int batch_no = 0;

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
		total_time_by_dataset += dataset.total_time_taken;
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
		valdataset.reset();
		std::cout << "Validating Accuracy at CPU: " << classifier.evaluate(valdataset, batch) << endl;
	}

	std::cout << "\n********************************************" << endl;
	std::cout << "Total time taken: " << total_cpu_train_time << endl;
	std::cout << "Total time taken by data processing: " << total_time_by_dataset << endl;
	std::cout << "Computation Time After" << epoch << " epochs: " << total_cpu_train_time - total_time_by_dataset << endl;
	std::cout << "Average Computation Time After" << (total_cpu_train_time - total_time_by_dataset) / epoch << endl;
	std::cout << "********************************************\n"
			  << endl;
	//model.printWeights();

	std::cout << "Strated GPU timing for the epoch\n";
	total_train_time = 0;
	total_time_by_dataset = 0;
	// GPU Code Starts here.
	for (int i = 0; i < epoch; i++)
	{
		int correct = 0, total = 0;
		int batch_no = 0;
		dataset.reset();

		start_time = std::clock();

		while (dataset.hasNext())
		{
			/*
			1. train logistic
			2. compute accuracy for validation set
			*/
			++batch_no;
			dataset.getNextBatch(true, batch);

			model.trainModel(*batch, true, memory_type, 0.0001);
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
		total_time_by_dataset += dataset.total_time_taken;
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
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
	std::cout << "\n********************************************" << endl;
	std::cout << "Total time taken: " << total_train_time << endl;
	std::cout << "Total time taken by data processing: " << total_time_by_dataset << endl;
	std::cout << "Computation Time  After " << epoch << "epochs: " << total_train_time - total_time_by_dataset << endl;
	std::cout << "Average Computation Time GPU :" << (total_train_time - total_time_by_dataset) / epoch << endl;
	std::cout << "********************************************\n"
			  << endl;

	// Double buffering!
	dataset.reset();
	GPUClassificationModel model2(batch_size, HIGGSDataset::NUMBER_OF_FEATURE, true);
	dbl_buffer(dataset, valdataset, model2, batch_size, "./data/HIGGS_Sample.csv", epoch);
}
