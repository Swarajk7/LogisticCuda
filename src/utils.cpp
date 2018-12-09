#include "utils.h"
#include <cmath>
float *generate_random_weight(int n)
{
    float *a = (float *)malloc(sizeof(float) * n);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(R_MEAN, R_STD);
    for (int i = 0; i < n; i++)
    {
        float number = distribution(generator);
        a[i] = number;
    }
    return a;
}

float sigmoid(float f)
{
    return 1.0f / (1.0f + expf(-1.0f * f));
}

float *generate_zeros(int n)
{
    float *a = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        a[i] = 0.0f;
    return a;
}