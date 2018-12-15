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
	HIGGSItem *batch = new HIGGSItem();
	batch->allocateMemory(batch_size);
	HIGGSDataset dataset("./data/HIGGS_Sample.csv", batch_size);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	int batch_no = 0;

	//Added by anand
	dbl_buffer(dataset, batch_size, "./data/HIGGS_Sample.csv");
	HIGGSDataset valdataset("./data/HIGGS_Sample_Val.csv", batch_size);
	// std::cout << classifier.evaluate(valdataset);

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
			dataset.getNextBatch(false, batch);
			if (batch->N == batch_size)
			{
				start_time = std::clock();
				correct += classifier.trainBatch(*batch, 0.0001);
				end_time = std::clock();
				total_cpu_train_time += (end_time - start_time) / (double)CLOCKS_PER_SEC;
			}
			total += batch->N;
		}
		std::cout << "Finished training one epoch, accuracy: " << correct * 1.0f / total << endl;
		std::cout << "Evaluating! Accuracy: ";
		valdataset.reset();
		//std::cout << classifier.evaluate(valdataset) << endl;
	}

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
			dataset.getNextBatch(true, batch);
			//model.setData(batch);
			if (batch->N == batch_size)
			{
				start_time = std::clock();
				model.trainModel(*batch, true, 0.0001);
				end_time = std::clock();
				total_train_time += (end_time - start_time) / (double)CLOCKS_PER_SEC;
			}
			total += batch->N;
			if (batch_no == 1)
			{
				//for(int i=0;i<29;i++) printf("%f ",batch.X[i]);
				printf("\n");
				//model.printWeights();
			}
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
		std::cout << corr / total << std::endl;
	}

	std::cout << "\n **** Total CPU Train Time: " << total_cpu_train_time << endl;
	std::cout << "\n***\n Total Train Time: " << total_train_time << "\n Total Evaluation Time: " << total_evaluation_time << endl;
}
