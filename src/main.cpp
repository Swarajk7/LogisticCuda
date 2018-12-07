#include <iostream>
#include "data_reader.h"
int main(int argc, char *argv[])
{
	/*HIGGSDataset dataset("../data/sample.csv", 3);
	while (dataset.hasNext())
	{
		HIGGSItem item = dataset.getNextBatch(true);
		cout << item.size << " " << item.N << endl;
		for (int i = 0; i < item.size; i++)
		{
			cout << i << " " << item.y[i] << endl;
		}
	}*/
	HIGGSDataset dataset("../data/sample.csv", 3);
	while (dataset.hasNext())
	{
		/*
		 1. train logistic
		 2. compute accuracy for validation set
		 */
	}
}