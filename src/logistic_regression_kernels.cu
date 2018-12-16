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
	int start = index;
	//int start = index * num_features;
	if (start < N)
	{
		for (int i = 0; i < num_features; i++)
		{
			value += weights[i] * X[start + stride];
			stride += size;
		}
		value = 1 / (1 + expf(-value));
		//value = exp(value) / (1 + exp(value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}

__global__ void computeForward(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features)
{
	__shared__ float data[BLOCK_SIZE];
	// TODO : Index calculation colesced, works only for untransposed data.
	int index = blockIdx.x * blockDim.x;
	int tx = threadIdx.x;
	data[tx] = tx < NUM_FEATURES ? X[index + tx] : 0.0f;
	data[tx] *= weights[tx];
	__syncthreads();

	int stride = blockDim.x >> 1;
	while (stride >= 1)
	{
		__syncthreads();
		if (tx < stride)
		{
			data[tx] = data[tx] + data[tx + stride];
		}
		stride = stride >> 1;
	}
	__syncthreads();
	if (tx == 0)
	{
		float value = data[tx];
		value = 1 / (1 + expf(-value));
		//value = exp(value) / (1 + exp(value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}

__global__ void externalKernel(float *weights, float *grad_weights, float *X, float *intermediate_vector, int size, int N, const int num_features, const int X_dim, float learning_rate)
{
	__shared__ float values[32][29];
	__shared__ float intermediate_shared[32];
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	values[tx][ty] = 0.0f;
	__syncthreads();
	for (int m = 0; m < ceilf((N * 1.0f) / X_dim); m++)
	{
		int Col = tx + m * X_dim;
		int Row = ty;
		if (Col < N)
		{
			if (ty == 0)
				intermediate_shared[tx] = intermediate_vector[tx + m * X_dim];
			__syncthreads();

			values[tx][ty] += X[Row * size + Col] * intermediate_shared[tx];
		}
		__syncthreads();
	}
	if (tx == 0)
	{
		for (int q = 1; q < X_dim; q++)
		{
			values[tx][ty] += values[tx + q][ty];
		}
		grad_weights[ty] = values[tx][ty];
		//printf("Updating weight %f %d",grad_weights[ty], ty);
		weights[ty] -= ((learning_rate * grad_weights[ty]) / N);
	}
}

__global__ void computeGrad(float *weights, float *grad_weights, float *X, float *intermediate_vector, int size, int N, const int num_features, float learning_rate)
{
	__shared__ float values[BLOCK_SIZE][NUM_FEATURES];
	__shared__ float intermediate_shared[BLOCK_SIZE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col = tx + blockIdx.x * blockDim.x;
	int row = ty;

	values[tx][ty] = 0.0f;
	__syncthreads();

	if (col < N)
	{
		if (ty == 0)
			intermediate_shared[tx] = intermediate_vector[col];
		__syncthreads();

		// TODO: check for memory colescing in a transposed way, may be?
		values[tx][ty] += X[row * size + col] * intermediate_shared[tx];

		__syncthreads();

		int stride = blockDim.x >> 1;
		while (stride >= 1)
		{
			__syncthreads();
			if (tx < stride)
			{
				values[tx][ty] = values[tx][ty] + values[tx + stride][ty];
			}
			stride = stride >> 1;
		}
		__syncthreads();
		if (tx == 0)
		{
			atomicAdd(&grad_weights[ty], values[tx][ty]);
		}
	}
}

__global__ void computeGrad_v2(float *weights, float *grad_weights, float *X, float *intermediate_vector, int size, int N, const int num_features, float learning_rate)
{
	__shared__ float values[1024];

	int tx = threadIdx.x;
	int ty = blockIdx.y;

	int col = tx + blockIdx.x * blockDim.x;
	int row = ty;

	values[tx] = 0.0f;
	if (col < N)
	{
		// TODO: check for memory colescing in a transposed way, may be?
		values[tx] = X[row * size + col] * intermediate_vector[col];
	}

	__syncthreads();

	int stride = blockDim.x >> 1;
	while (stride >= 1)
	{
		__syncthreads();
		if (tx < stride)
		{
			values[tx] = values[tx] + values[tx + stride];
		}
		stride = stride >> 1;
	}
	__syncthreads();
	if (tx == 0)
	{
		atomicAdd(&grad_weights[ty], values[tx]);
	}
}

__global__ void updateGrad(float *weights, float *grad_weights, float learning_rate, int N)
{
	if (threadIdx.x < NUM_FEATURES)
	{
		weights[threadIdx.x] -= (learning_rate * grad_weights[threadIdx.x]) / N;
	}
}

__global__ void evaluate_model(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features, float *correct_val)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 0;
	float value = 0;
	//Start needs to be verified
	int start = index;
	//int start = index * num_features;
	if (start < N)
	{
		for (int i = 0; i < num_features; i++)
		{
			value += weights[i] * X[start + stride];
			stride += size;
		}
		value = 1 / (1 + expf(-value));
		float y_pred = value > 0.5f ? 1.0f : 0.0f;
		if (fabs(y_pred - y[index]) < 0.001)
			atomicAdd(correct_val, 1);
	}
}

__global__ void uncoalescedKernel(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0;
	int start = index * num_features;
	if (start < N)
	{
		for (int i = 0; i < num_features; i++)
		{
			value += weights[i] * X[start + i];
		}
		value = 1 / (1 + expf(-value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}

__global__ void memory_coalescedKernel_shared(float *weights, float *X, float *y, float *intermediate_vector, int size, int N, int num_features)
{
	//BLOCK_SIZE should be greater than 29
	__shared__ float weights_shared[29];
	if (threadIdx.x < num_features)
	{
		weights_shared[threadIdx.x] = weights[threadIdx.x];
	}
	__syncthreads();

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = 0;
	float value = 0;
	//Start needs to be verified
	int start = index;
	//int start = index * num_features;
	if (start < N)
	{
		for (int i = 0; i < num_features; i++)
		{
			value += weights_shared[i] * X[start + stride];
			stride += size;
		}
		value = 1 / (1 + expf(-value));
		//value = exp(value) / (1 + exp(value));
		value -= y[index];
		intermediate_vector[index] = value;
	}
}

__global__ void printKernel(float *weights, float *inter_vector, int num_features)
{
	printf("WEIGHTS\n");
	for (int i = 0; i < num_features; i++)
		printf("%f ", weights[i]);

	printf("Inter Values\n");
	for (int i = 0; i < num_features; i++)
		printf("%f ", inter_vector[i]);
}
