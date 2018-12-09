#include "logistic.h"
#include "utils.h"
#include "data_reader.h"
#include <iostream>
LogisticRegression::LogisticRegression(int features_count)
{
    // generate random weights for features and one bias term.
    num_features = features_count;
    weights = generate_random_weight(num_features + 1);
    grad_weights = generate_zeros(num_features + 1);
}

int LogisticRegression::trainBatch(HIGGSItem item)
{
    int xindex = 0;

    // allocate memory to store W.T*X.
    float *result = (float *)malloc(sizeof(float) * item.N);
    if (result == NULL)
        return -1;
    // Compute sigmoid(W.T*X)
    for (int i = 0; i < item.N; i++)
    {
        float mul =0.0f;
        for (int j = 0; j < num_features + 1; j++)
        {
            mul += weights[j] * item.X[xindex + j];
        }
        xindex += num_features;
        result[i] = sigmoid(mul) - item.y[i];
        // Compute Gradient
        // Reuse result vector, assuming result is not required later. :)
        for (int j = 0; j < num_features; j++)
        {
            grad_weights[j] += result[i] * item.X[j];
        }
    }
    return 0;
}

void LogisticRegression::preallocateAuxMemory(int size)
{
}