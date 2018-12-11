#include "utils.h"
#include <cmath>
#include <iostream>
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
    float res = 1.0f / (1.0f + expf(-1.0f * f));
    if (res != res)
        std::cout << f << std::endl;
    return res;
}

float *generate_zeros(int n)
{
    float *a = (float *)malloc(sizeof(float) * n);
    for (int i = 0; i < n; i++)
        a[i] = 0.0f;
    return a;
}