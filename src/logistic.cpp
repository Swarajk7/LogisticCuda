#include "logistic.h"
#include "utils.h"
#include "data_reader.h"
#include <iostream>
LogisticRegression::LogisticRegression(int features_count)
{
    // generate random weights for features and one bias term.
    num_features = features_count;
    weights = generate_random_weight(num_features + 1);
}

int LogisticRegression::trainBatch(HIGGSItem &item, float learning_rate)
{
    int xindex = 0;
    int correct = 0;
    if (item.N == 0)
        return 0;
    // allocate memory to store W.T*X.
    float *result = (float *)malloc(sizeof(float) * item.N);
    grad_weights = generate_zeros(num_features + 1);
    if (result == NULL)
        return -1;
    // Compute sigmoid(W.T*X)
    for (int i = 0; i < item.N; i++)
    {
        float mul = 0.0f;
        for (int j = 0; j < num_features + 1; j++)
        {
            mul += (weights[j] * item.X[xindex + j]);
        }
        float pred = sigmoid(mul) > 0.5f ? 1.0f : 0.0f;
        if (fabs(pred - item.y[i]) < 0.001f)
            correct++;
        result[i] = sigmoid(mul) - item.y[i];
        // Compute Gradient
        // Reuse result vector, assuming result is not required later. :)
        for (int j = 0; j < num_features + 1; j++)
        {
            grad_weights[j] += (result[i] * item.X[xindex + j]);
        }
        xindex += num_features + 1;
    }
    // Update Gradients after multiplying with learning rate
    float sum = 0.0f;
    for (int i = 0; i < num_features + 1; i++)
    {
        assert(item.N != 0);
        weights[i] -= (learning_rate * grad_weights[i]) / item.N;
        sum += grad_weights[i];
    }
    //std::cout << "grad sum: " << sum << endl;
    free(result);
    free(grad_weights);
    return correct;
}

float LogisticRegression::evaluate(HIGGSDataset &validationSet, HIGGSItem *batch)
{
    float correct = 0, total = 0;
    float mul = 0.0f;
    for (int j = 0; j < num_features + 1; j++)
    {
        mul += weights[j];
    }
    cout << "\n Sum: " << mul << endl;
    while (validationSet.hasNext())
    {
        int xindex = 0;
        validationSet.getNextBatch(false, batch);
        assert(batch->N != 0);
        // Compute sigmoid(W.T*X)
        for (int i = 0; i < batch->N; i++)
        {
            float mul = 0.0f;
            for (int j = 0; j < num_features + 1; j++)
            {
                mul += weights[j] * batch->X[xindex + j];
            }
            xindex += num_features + 1;
            float pred = sigmoid(mul) > 0.5f ? 1.0f : 0.0f;
            if (fabs(pred - batch->y[i]) < 0.001f)
                correct++;
            total++;
        }
    }
    return correct / total;
}