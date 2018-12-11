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
	HIGGSDataset dataset("./data/HIGGS_Sample.csv", batch_size);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	int batch_no = 0;

	HIGGSDataset valdataset("./data/HIGGS_Sample_Val.csv", batch_size);
	std::cout << classifier.evaluate(valdataset);

	for (int i = 0; i < 10; i++)
	{
		dataset.reset();
		while (dataset.hasNext())
		{
			/*
			1. train logistic
			2. compute accuracy for validation set
			*/
			++batch_no;
			HIGGSItem batch = dataset.getNextBatch(false);
			if (batch.N == batch_size)
				classifier.trainBatch(batch, 0.0001);
		}
		std::cout << "Finished training one epoch: " << batch_no << endl;
		std::cout << "Evaluating! Accuracy: ";
		valdataset.reset();
		std::cout << classifier.evaluate(valdataset) << endl;
	}
}