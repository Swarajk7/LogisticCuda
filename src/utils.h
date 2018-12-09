#ifndef UTILS
#define UTILS
#include <random>
#define R_MEAN 0.0f
#define R_STD 1.0f
float *generate_random_weight(int n);
float *generate_zeros(int n);
float sigmoid(float a);
#endif