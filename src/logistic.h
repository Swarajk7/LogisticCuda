#ifndef LOGISTIC
#define LOGISTIC
#include "data_reader.h"
class LogisticRegression
{
    float *weights;
    float *grad_weights;
    int num_features;
  public:
    LogisticRegression(int NUMFEATURES);
    int trainBatch(HIGGSItem item);
    void preallocateAuxMemory(int batch_size);
};
#endif