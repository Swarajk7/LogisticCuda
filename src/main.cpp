#include <iostream>
#include "data_reader.h"
#include "logistic.h"
int main(int argc, char *argv[])
{
	int batch_size = 100;
	if (argc >= 2)
	{
		batch_size = stoi(argv[1]);
	}
	HIGGSDataset dataset("./data/sample.csv", batch_size);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	int batch_no = 0;

	HIGGSDataset valdataset("./data/sample.csv", batch_size);
	std::cout << classifier.evaluate(valdataset);

	HIGGSDataset valdataset2("./data/sample.csv", batch_size);
	int count = 0, total = 0;
	while (valdataset2.hasNext())
	{
		HIGGSItem batch = valdataset2.getNextBatch(false);
		for (int i = 0; i < batch.N; i++)
			count += batch.y[i];
		total += batch.N;
	}
	cout << count << " " << total << endl;
	while (dataset.hasNext())
	{
		/*
		 1. train logistic
		 2. compute accuracy for validation set
		 */
		++batch_no;
		HIGGSItem batch = dataset.getNextBatch(false);
		classifier.trainBatch(batch, 0.01);
		if (batch_no % 20 == 0)
		{
			std::cout << "Finished training batch: " << batch_no << endl;
			std::cout << "Evaluating! Accuracy: ";
			HIGGSDataset valdataset("./data/HIGGS_Sample_Val.csv", batch_size);
			std::cout << classifier.evaluate(valdataset) << endl;
		}
	}
}