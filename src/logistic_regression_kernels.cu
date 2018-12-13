//Put the kernel codes here.
//Optimizations:
//Weights in memory, shared memory, constant memory.
//Instead of using value, directly use intermediate_vector[i].
//Use hardware math functions for exp.

__global__ void memory_coalescedKernel(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 0;
	float value = 0;
	//Start needs to be verified
	int start = index
	//int start = index * num_features;
	if (start < N)
	{
		for (int i; i < num_features; i++)
		{
			value += weights[i] * X[start + stride];
			stride += size;
		}
		value = exp(value) / (1 + exp(value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}

__global__ void externalKernel(float *grad_weights, float *X, float *intermediate_vector, int size, int N, int num_features, int X_dim)
{
	__shared__ float values[X_dim][num_features] = 0;
	__shared__ float intermediate_shared[X_dim];
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	for (int m = 0; m < ceilf((N * 1.0f) / X_dim); m++)
	{
		Col = tx + m * X_dim;
		Row = ty;
		if (Col < N)
		{
			if (ty == 0)
				intermediate_shared[tx] = intermediate_vector[tx + m * X_dim];
			__syncthreads();

			values[tx][ty] += X[Row * size + Col] * intermediate_shared[tx];

			__syncthreads();
		}
	}
	if (tx == 0)
	{
		for (int q = 1; q < X_dim; q++)
		{
			values[tx][ty] += values[tx + q][ty];
		}
		grad_weights[ty] = values[tx][ty]
	}
}

__global__ void uncoalescedKernel(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0;
	int start = index * num_features;
	if (start < N)
	{
		for (int i; i < num_features; i++)
		{
			value += weights[i] * X[start + i];
		}
		value = exp(value) / (1 + exp(value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}