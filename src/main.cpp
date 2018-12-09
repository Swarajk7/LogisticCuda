#include <iostream>
#include "data_reader.h"
#include "logistic.h"
int main(int argc, char *argv[])
{
	HIGGSDataset dataset("../data/sample.csv", 3);
	LogisticRegression classifier(HIGGSDataset::NUMBER_OF_FEATURE);
	int batch_no = 0;
	while (dataset.hasNext())
	{
		/*
		 1. train logistic
		 2. compute accuracy for validation set
		 */
		std::cout << "Training one batch: " << ++batch_no << endl;
		classifier.trainBatch(dataset.getNextBatch(false));
	}
}