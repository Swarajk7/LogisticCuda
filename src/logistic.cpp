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
        xindex += num_features + 1;
        int pred = sigmoid(mul) > 0.5f ? 1 : 0;
        if (pred == (int)item.y[i])
            correct++;
        result[i] = sigmoid(mul) - item.y[i];
        // Compute Gradient
        // Reuse result vector, assuming result is not required later. :)
        for (int j = 0; j < num_features + 1; j++)
        {
            grad_weights[j] += ((result[i] * item.X[j]) / item.N);
        }
    }
    // Update Gradients after multiplying with learning rate
    float sum = 0.0f;
    for (int i = 0; i < num_features + 1; i++)
    {
        weights[i] -= (learning_rate * grad_weights[i]);
        sum += grad_weights[i];
    }
    //std::cout << "grad sum: " << sum << endl;
    free(result);
    free(grad_weights);
    return correct;
}

float LogisticRegression::evaluate(HIGGSDataset &validationSet)
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
        HIGGSItem item = validationSet.getNextBatch(false);
        // Compute sigmoid(W.T*X)
        for (int i = 0; i < item.N; i++)
        {
            float mul = 0.0f;
            for (int j = 0; j < num_features + 1; j++)
            {
                mul += weights[j] * item.X[xindex + j];
            }
            xindex += num_features + 1;
            int pred = sigmoid(mul) > 0.5f ? 1 : 0;
            if (pred == (int)item.y[i])
                correct++;
            total++;
        }
    }
    return correct / total;
}