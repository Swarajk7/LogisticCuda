#include "data_reader.h"
#include <iostream>
HIGGSDataset::HIGGSDataset(string fname, int batch)
{
    file_name = fname;
    batch_size = batch;
    file.open(fname.c_str(), ios::in);
    if (!file)
    {
        // Can't open file.
        assert(false);
    }
    has_next = true;
    std::cout << "Info: Opened File: " << file_name << endl;
}

vector<string> split(string str)
{
    std::string buf;           // Have a buffer string
    std::stringstream ss(str); // Insert the string into a stream

    std::vector<std::string> tokens; // Create vector to hold our words

    while (getline(ss, buf, ','))
        tokens.push_back(buf);

    return tokens;
}

HIGGSItem HIGGSDataset::getNextBatch(bool transposed)
{
    HIGGSItem item;
    item.allocateMemory(batch_size);
    string line;
    int xindex = 0;
    int yindex = 0;
    int count = 0;
    if (transposed)
    {
        for (int i = 0; i < item.size; i++)
        {
            item.X[xindex++] = 1.0f;
        }
    }
    while (getline(file, line))
    {
        /*
            1. split line by ","
            2. convert each string element to float.
        */
        vector<string> tokens = split(line);
        assert(tokens.size() == 29);
        item.y[yindex++] = stof(tokens[0]);
        if (!transposed)
            item.X[xindex++] = 1.0f;

        for (unsigned int i = 1; i < tokens.size(); i++)
        {
            float xx = stof(tokens[i]);
            if (!transposed)
            {
                item.X[xindex++] = xx;
            }
            else
            {
                item.X[xindex + (i - 1) * item.size] = xx;
            }
        }
        if (transposed)
            xindex++;
        count++;
        if (count >= batch_size)
            break;
    }
    if (transposed)
        assert(xindex <= item.size + 1);
    item.N = yindex;
    if (item.N < item.size)
        has_next = false;
    return item;
}

bool HIGGSDataset::hasNext()
{
    return has_next;
}

void HIGGSItem::allocateMemory(int sz)
{
    size = sz;
    y = (float *)malloc(sizeof(float) * sz);
    X = (float *)malloc(sizeof(float) * sz * (HIGGSDataset::NUMBER_OF_FEATURE + 1));
}

void HIGGSDataset::reset()
{
    file.clear();
    file.seekg(0, ios::beg);
    has_next = true;
}