#ifndef DATA_READER
#define DATA_READER
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <assert.h>
using namespace std;
struct HIGGSItem
{
    float *X;
    float *y;
    int size;
    int N;
    void allocateMemory(int sz);
};
class HIGGSDataset
{
    int batch_size;
    string file_name;
    ifstream file;
    bool has_next;

  public:
    HIGGSDataset(string fname, int batch_size);
    void getNextBatch(bool transposed, HIGGSItem *item);
    bool hasNext();
    void reset();
    static const int NUMBER_OF_FEATURE = 28;

    double total_time_taken;
    double total_batch_executed;
};
#endif