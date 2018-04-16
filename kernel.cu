#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_functions.h>  // bringt intern helper_timer.h mit


#include <stdio.h>
#include <cstdlib>

/**
* Matrix multiplication(CUDA Kernel) on the device : C = A * B
* wA is A's width and wB is B's width
*/
template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
	a <= aEnd;
		a += aStep, b += bStep)
	{

		// Declaration of the shared memory array As used to
		// store the sub-matrix of A
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		// Declaration of the shared memory array Bs used to
		// store the sub-matrix of B
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load the matrices from device memory
		// to shared memory; each thread loads
		// one element of each matrix
		As[ty][tx] = A[a + wA * ty + tx];
		Bs[ty][tx] = B[b + wB * ty + tx];

		// Synchronize to make sure the matrices are loaded
		__syncthreads();

		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix
#pragma unroll

		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			Csub += As[ty][k] * Bs[k][tx];
		}

		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}


__global__ void matMultCuda(float *cu_C, float *cu_A, float *cu_B, unsigned int n) {

	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = (blockIdx.y * blockDim.y) + threadIdx.y;

	//Log row and col of each thread
	//printf("row : %d , col : %d \n", row, col);

	if (row < n && col < n) {
		int temp_sum = 0;

		for (int elem = 0; elem < n; elem++)
		{
			temp_sum += cu_A[row * n + elem] * cu_B[elem * n + col];
		}

		cu_C[row * n + col] = temp_sum;
	}
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
	unsigned int const n = 1024;
	float *d_A, *d_B, *d_C;

	float *h_A = new float [n * n];

	float *h_B = new float [n * n];

	float *h_C = new float [n * n];

	StopWatchInterface *t_cpu, *t_gpu, *t_gpu_cuda;
	if (!sdkCreateTimer(&t_cpu) || !sdkCreateTimer(&t_gpu) || !sdkCreateTimer(&t_gpu_cuda)) {
		printf("timercreate failed\n");
		exit(-1);
	};

	//Init matrices with random data
	for (int row = 0; row < n; row++) {
		for (int col = 0; col < n; col++) {
			h_A[row * n + col] = sin((float)rand());
			h_B[row * n + col] = cos((float)rand());
		}
	}

	//Start time measurement
	sdkStartTimer(&t_cpu);
	matMultHost(h_A, h_B, h_C, n);
	//Stop time measurement
	sdkStopTimer(&t_cpu);
	//printMatrixHost(h_C, n);
	printf("Zeitdauer - CPU : %f ms\n", sdkGetTimerValue(&t_cpu));


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

	 double const max_blocks_per_grid = 256;

	// Use a Grid with one Block containing n * n Threads
	dim3 threads_per_block(n, n, 1);
	dim3 blocks_per_grid(ceil(n/max_blocks_per_grid), ceil(n/max_blocks_per_grid), 1);
	//Start time measurement
	sdkStartTimer(&t_gpu);
	matMultCuda <<<blocks_per_grid, threads_per_block >> >(d_C, d_A, d_B, n);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
	}

	err = cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

	//Stop time measurement
	sdkStopTimer(&t_gpu);
	printf("Zeitdauer - GPU : %f ms\n", sdkGetTimerValue(&t_gpu));

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
	}

	//Setup matrixMulCuda
	sdkStartTimer(&t_gpu_cuda);

	matrixMulCUDA<32><<<blocks_per_grid, threads_per_block>>>(d_C, d_A, d_B, n, n);

	err = cudaMemcpy(h_C, d_C, memorySize, cudaMemcpyDeviceToHost);

	sdkStopTimer(&t_gpu_cuda);
	printf("Zeitdauer - GPU_Cuda_Method : %f ms\n", sdkGetTimerValue(&t_gpu_cuda));
	//printMatrixHost(h_C, n);

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}