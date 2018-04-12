#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void matMultCuda(float *cu_C, float *cu_A, float *cu_B, unsigned int n) {

	int i = threadIdx.x;
	cu_C[i] += cu_A[i] * cu_B[i];

};

void matMultHost(float* h_A, float* h_B, float* h_C, int n) // n = m
{
	for (int row = 0; row < n; ++row)
	{
		for (int col = 0; col < n; ++col)
		{
			for (int elem = 0; elem < n; ++elem)
			{
				h_C[row * n + col] += h_A[row * n + elem] * h_B[elem * n + col];
			}
		}
	}
}

void printMatrixHost(float* h_C, int n)
{
	for (int row = 0; row < n; ++row)
	{
		for (int col = 0; col < n; ++col)
		{
			printf("%f ", h_C[row * n + col]);
		}
		printf("\n");
	}
}

int main()
{
	unsigned int const n = 3;
	float *d_A, *d_B, *d_C;

	float h_A[] = { 
		1,1,1,
		2,2,2,
		3,3,3 };

	float h_B[] = { 
		1,1,1,
		2,2,2,
		3,3,3 };

	float h_C[n * n] = {};
	matMultHost(h_A, h_B, h_C, n);
	printMatrixHost(h_C, n);


	unsigned int memorySize = (n * n) * sizeof(float);

	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_A, memorySize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_A, h_A, memorySize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_B, memorySize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, memorySize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_C, memorySize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_C, h_C, memorySize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	int num_blocks = 1;
	int num_threads = 9;
	matMultCuda <<<num_blocks, num_threads>>> (d_C, d_A, d_B, n);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
	}

	err = cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

	printMatrixHost(h_C, n);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<<1, size>>> (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
