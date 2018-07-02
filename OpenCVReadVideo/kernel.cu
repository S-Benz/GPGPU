#include "cuda_runtime.h"
#include "kernel.h"
#include "fix.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <cstdlib>
//
#define CHANNELS 3
#define REDCHANNEL 'r'
#define GREENCHANNEL 'g'
#define BLUECHANNEL 'b'
#define GRAYSCLAEREDCHANNEL 0.21
#define GRAYSCLAEGREENCHANNEL 0.71
#define GRAYSCLAEBLUECHANNEL 0.07
#define SOBEL_RADIUS 1
#define TILE_W 16
#define BLOCK_W (TILE_W + 2*SOBEL_RADIUS)
#define ANGLE 50
#define HISTOGRAMMSIZE 256

__global__ void sobelFilterKernelTiledStreams(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image, int sobel_offset)
{
	__shared__ char ds_Img[BLOCK_W][BLOCK_W];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int sobel_x[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	int sobel_y[3][3] = {
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int x = bx * TILE_W + tx - SOBEL_RADIUS; //cols
	int y = by * TILE_W + ty - SOBEL_RADIUS; //rows

											 //Make sure x/y are not negative
	if (x < 0) {
		x = 0;
	}

	if (y < 0) {
		y = 0;
	}

	//Calc index of global memory
	int global_index = sobel_offset + (y * (*cu_image_width) + x);

	//Load Data into Shared Memory
	//Insert 0 if the thread is supposed to fill the filter radius border of the tile
	if (x >= 0 && x < *cu_image_width - 1 && y >= 0 && y < *cu_image_height - 1) {
		ds_Img[ty][tx] = cu_src_image[global_index];
	}
	else {
		if (x < *cu_image_width && y < *cu_image_height) {
			ds_Img[ty][tx] = 0;
		}
	}
	__syncthreads();

	//Calc Sobel X & Y if the thread is inside the filter area
	if ((tx >= SOBEL_RADIUS) && (tx <= TILE_W) &&
		(ty >= SOBEL_RADIUS) && (ty <= TILE_W)) {
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;
		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				sobel_gradient_x += ds_Img[ty + j][tx + k] * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += ds_Img[ty + j][tx + k] * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}
		//Calc Sobel magnitude and save it to the original image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));
		cu_dest_image[global_index] = (unsigned char)sobel_magnitude;
	}
}

__global__ void sobelFilterKernelStreams(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image, int offset_sobel)
{
	int sobel_x[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	int sobel_y[3][3] = {
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows

												   //Calc index
	int global_index = offset_sobel + (y * (*cu_image_width) + x);

	if (x >= SOBEL_RADIUS && x < *cu_image_width - 1 && y >= SOBEL_RADIUS && y < *cu_image_height - 1) {
		//Calc Sobel X & Y if the thread is inside the filter area
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;

		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				int sobel_index = offset_sobel + (y + j) * (*cu_image_width) + (x + k);
				sobel_gradient_x += cu_src_image[sobel_index] * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += cu_src_image[sobel_index] * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}

		//Calc Sobel magnitude and save it to the image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));

		cu_dest_image[global_index] = (unsigned char)sobel_magnitude;
	}
	else {
		if (x < *cu_image_width && y < *cu_image_height) {
			cu_dest_image[global_index] = 0;
		}
	}
}

//Kernel rgb to grayscale function with streams
__global__ void rgbToGrayscaleKernelStream(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image, int offset_rgb, int offset_gray)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows
	unsigned char r, g, b, gray;

	if (x < *cu_image_width && y < *cu_image_height) {
		int offset = (y * (*cu_image_width) + x);
		int grayOffset =  offset + offset_gray;
		int rgbOffset = offset_rgb + offset  * CHANNELS;

		b = cu_src_image[rgbOffset];
		g = cu_src_image[rgbOffset + 1];
		r = cu_src_image[rgbOffset + 2];

		gray = 0.21 * r + 0.71 *g + 0.07 *b;

		cu_dest_image[grayOffset] = gray;
	}
}

__global__ void getHistogrammTiledKernel(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned int *cu_dest_histogramm) {

	__shared__ unsigned int smem[HISTOGRAMMSIZE];

	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows

	int shared_index = threadIdx.x + threadIdx.y * blockDim.x;
	int stride_shared = blockDim.x * blockDim.y;

	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	//Init shared memory histogramm with 0's
	for (int i = shared_index; i < HISTOGRAMMSIZE; i += stride_shared) {
		smem[i] = 0;
	}

	__syncthreads();

	//Add data to histogramm in shared memory
	while (x < *cu_image_width && y < *cu_image_height) {
		int index = y * *cu_image_width + x;

		atomicAdd(&(smem[cu_src_image[index]]), 1);

		x += stride_x;
		y += stride_y;
	
	}

	__syncthreads();
	/*
	long test = 0;
	if (threadIdx.x == 0) {
		for (int j = 0; j < HISTOGRAMMSIZE; j++) {
			test += smem[j];
		}
		printf("Smem total per block: %lu \n", test);
	}*/
	
	//Add shared memory histogramm part to global memory histogramm
	for (int i = shared_index; i < HISTOGRAMMSIZE; i+= stride_shared) {
		atomicAdd(&(cu_dest_histogramm[i]), smem[i]);
	}

	__syncthreads();
	
}

__global__ void getHistogrammKernel(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned int *cu_dest_histogramm)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows


	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	while (x < *cu_image_width && y < *cu_image_height) {
		int index = y * *cu_image_width + x;
		
		atomicAdd(&(cu_dest_histogramm[cu_src_image[index]]), 1);

		x += stride_x;
		y += stride_y;
	}
}

__global__ void sobelFilterTexture(int *cu_image_width, int *cu_image_height, unsigned char *cu_output, cudaTextureObject_t cu_texObj, float theta)
{
	int sobel_x[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	int sobel_y[3][3] = {
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < *cu_image_width - 1 &&  y < *cu_image_height - 1) {
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;

		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				//Calc normalized texture coordinates
				float u = (x + k) / (float)*cu_image_width;
				float v = (y + j) / (float)*cu_image_height;

				// Transform coordinates
				u -= 0.5f;
				v -= 0.5f;

				float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
				float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

				sobel_gradient_x += tex2D<float>(cu_texObj, tu, tv) * 255 * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += tex2D<float>(cu_texObj, tu, tv) * 255 * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}

		//Calc Sobel magnitude and save it to the image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));

		cu_output[y * *cu_image_width + x] = (unsigned char)sobel_magnitude;
	}

};

//Kernel sobel function
__global__ void sobelFilterKernelTiled(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
{

	__shared__ char ds_Img[BLOCK_W][BLOCK_W];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int sobel_x[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	int sobel_y[3][3] = {
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int x = bx * TILE_W + tx - SOBEL_RADIUS; //cols
	int y = by * TILE_W + ty - SOBEL_RADIUS; //rows

	//Make sure x/y are not negative
	if (x < 0) {
		x = 0;
	}

	if (y < 0) {
		y = 0;
	}

	//Calc index of global memory
	int global_index = (y * (*cu_image_width) + x);

	//Load Data into Shared Memory
	//Insert 0 if the thread is supposed to fill the filter radius border of the tile
	if (x >= 0 && x < *cu_image_width - 1 && y >=  0 && y < *cu_image_height - 1) {
		ds_Img[ty][tx] = cu_src_image[global_index];
	}
	else {
		ds_Img[ty][tx] = 0;
	}
	__syncthreads();

	//Calc Sobel X & Y if the thread is inside the filter area
	if ((tx >= SOBEL_RADIUS) && (tx <= TILE_W) &&
		(ty >= SOBEL_RADIUS) && (ty <= TILE_W)){
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;
		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				sobel_gradient_x += ds_Img[ty + j][tx + k] * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += ds_Img[ty + j][tx + k] * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}
		//Calc Sobel magnitude and save it to the original image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));
		cu_dest_image[global_index] = (unsigned char)sobel_magnitude;
	}
}

__global__ void sobelFilterKernel(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
{
	int sobel_x[3][3] = {
		{ 1, 0, -1 },
		{ 2, 0, -2 },
		{ 1, 0, -1 }
	};
	int sobel_y[3][3] = {
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows

	//Calc index
	int global_index = (y * (*cu_image_width) + x);

	if (x >= SOBEL_RADIUS && x < *cu_image_width - 1 && y >= SOBEL_RADIUS && y < *cu_image_height - 1) {
		//Calc Sobel X & Y if the thread is inside the filter area
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;

		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				sobel_gradient_x += cu_src_image[(y + j) * (*cu_image_width) + (x + k)] * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += cu_src_image[(y + j) * (*cu_image_width) + (x + k)] * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}

		//Calc Sobel magnitude and save it to the image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));

		cu_dest_image[global_index] = (unsigned char)sobel_magnitude;
	}
	else {
		if (x < *cu_image_width && y < *cu_image_height) {
			cu_dest_image[global_index] = 0;
		}
	}
}

//Kernel rgb to grayscale function
__global__ void rgbToGrayscaleKernel(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows
	unsigned char r, g, b, gray;

	if (x < *cu_image_width && y < *cu_image_height) {
		int grayOffset = (y * (*cu_image_width) + x);
		int rgbOffset = grayOffset * CHANNELS;

		b = cu_src_image[rgbOffset];
		g = cu_src_image[rgbOffset + 1];
		r = cu_src_image[rgbOffset + 2];

		gray = 0.21 * r + 0.71 *g + 0.07 *b;

		cu_dest_image[grayOffset] = gray;
	}
}

//Kernel ColorChannel function
__global__ void setColorChannelKernel(int *cu_image_width, int *cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image, unsigned char *cu_channel_to_keep)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows
	unsigned char r, g, b;

	if (x < *cu_image_width && y < *cu_image_height) {
		int offset = (y * (*cu_image_width) + x) * CHANNELS;

		switch (*cu_channel_to_keep)
		{
		case BLUECHANNEL:
			b = cu_src_image[offset];
			g = 0;
			r = 0;
			break;
		case GREENCHANNEL:
			b = 0;
			g = cu_src_image[offset + 1];
			r = 0;
			break;
		case REDCHANNEL:
			b = 0;
			g = 0;
			r = cu_src_image[offset + 2];
			break;
		default: //Defaults to REDCHANNEL
			b = 0;
			g = 0;
			r = cu_src_image[offset + 2];
			break;
		}

		cu_dest_image[offset] = b; //B
		cu_dest_image[offset + 1] = g; //G
		cu_dest_image[offset + 2] = r; //R
	}

};

void setColorChannel(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image, unsigned char channel_to_keep)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image, *d_dest_image, *d_channel_to_keep;

	unsigned int imgSize = (image_width * image_height) * CHANNELS * sizeof(unsigned char);

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy channel to keep to gpu
	err = cudaMalloc((void **)&d_channel_to_keep, sizeof(unsigned char));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_channel_to_keep, &channel_to_keep, sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaMalloc((void **)&d_src_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_src_image, src_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image dest to gpu
	err = cudaMalloc((void **)&d_dest_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_dest_image, dest_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Pro Grid N/16 Blöcke, n = Anzahl Threads
	dim3 blocks_per_grid((image_width - 1) / threads + 1, (image_height - 1) / threads + 1, 1);
	setColorChannelKernel << <blocks_per_grid, threads_per_block >> >(d_image_width, d_image_height, d_src_image, d_dest_image, d_channel_to_keep);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
	}

	err = cudaMemcpy(dest_image, d_dest_image, imgSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaFree(d_channel_to_keep);
	cudaFree(d_src_image);
	cudaFree(d_dest_image);
}


void rgbToGrayscale(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image, *d_dest_image;

	unsigned int imgSizeRgb = (image_width * image_height) * CHANNELS * sizeof(unsigned char);
	unsigned int imgSizeGray = (image_width * image_height) * sizeof(unsigned char);

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaMalloc((void **)&d_src_image, imgSizeRgb);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_src_image, src_image, imgSizeRgb, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image dest to gpu
	err = cudaMalloc((void **)&d_dest_image, imgSizeGray);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_dest_image, dest_image, imgSizeGray, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Pro Grid N/16 Blöcke, n = Anzahl Threads
	dim3 blocks_per_grid((image_width - 1) / threads + 1, (image_height - 1) / threads + 1, 1);
	//rgbToGrayscaleKernel << <blocks_per_grid, threads_per_block >> >(d_image_width, d_image_height, d_src_image, d_dest_image);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
	}

	err = cudaMemcpy(dest_image, d_dest_image, imgSizeGray, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaFree(d_src_image);
	cudaFree(d_dest_image);
};

void sobelFilter(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image, *d_dest_image;

	unsigned int imgSize = (image_width * image_height) * sizeof(unsigned char);

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaMalloc((void **)&d_src_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_src_image, src_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image dest to gpu
	err = cudaMalloc((void **)&d_dest_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_dest_image, dest_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((image_width - 1) / threads + 1, (image_height - 1) / threads + 1, 1);
	sobelFilterKernel <<<blocks_per_grid, threads_per_block >>>(d_image_width, d_image_height, d_src_image, d_dest_image);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(dest_image, d_dest_image, imgSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaFree(d_src_image);
	cudaFree(d_dest_image);
};

void sobelFilterShared(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image, *d_dest_image;

	unsigned int imgSize = (image_width * image_height) * sizeof(unsigned char);

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaMalloc((void **)&d_src_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_src_image, src_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image dest to gpu
	err = cudaMalloc((void **)&d_dest_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_dest_image, dest_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Use a Grid with one Block containing Block_width threads
	dim3 threads_per_block_tiled(BLOCK_W, BLOCK_W, 1);
	//Per Grid N/Tile_wisth blocks
	dim3 blocks_per_grid_tiled((image_width - 1) / TILE_W + 1, (image_height - 1) / TILE_W + 1, 1);
	sobelFilterKernelTiled << <blocks_per_grid_tiled, threads_per_block_tiled >> >(d_image_width, d_image_height, d_src_image, d_dest_image);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(dest_image, d_dest_image, imgSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaFree(d_src_image);
	cudaFree(d_dest_image);
};

void sobelFilterTexture(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image)
{
	int *d_image_width, *d_image_height;

	unsigned int imgSize = (image_width * image_height) * sizeof(unsigned char);

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Create ChannelDesc
	//Sets output format of the value when the texture is fetched i.e. float texel
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	
	//Create cuda array
	cudaArray *cuArray;
	
	//Allocate cuda array
	err = cudaMallocArray(&cuArray, &channelDesc, image_width, image_height);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image data to cuda array
	err = cudaMemcpyToArray(cuArray, 0, 0, src_image, image_height * image_width * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Set Texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	//Set Texture object params
	struct cudaTextureDesc textDesc;
	memset(&textDesc, 0, sizeof(textDesc));
	textDesc.addressMode[0] = cudaAddressModeMirror;
	textDesc.addressMode[1] = cudaAddressModeMirror;
	textDesc.filterMode = cudaFilterModeLinear;
	textDesc.readMode = cudaReadModeNormalizedFloat;
	textDesc.normalizedCoords = 1;

	//Create Texture Object
	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &textDesc, NULL);

	unsigned char *output;
	err = cudaMalloc(&output, image_height * image_width * sizeof(unsigned char));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	//

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	float angle = 0;
	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((image_width - 1) / threads + 1, (image_height - 1) / threads + 1, 1);
	sobelFilterTexture <<<blocks_per_grid, threads_per_block >>>(d_image_width, d_image_height, output, texObj, angle);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(dest_image, output, imgSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaDestroyTextureObject(texObj);
	cudaFreeArray(cuArray);
	cudaFree(output);
};


void getHistogramm(int image_width, int image_height, unsigned char *src_image)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image;
	unsigned int  *d_dest_histogramm;

	unsigned int histogramm[HISTOGRAMMSIZE] = { 0 };

	unsigned int imgSize = (image_width * image_height) * sizeof(unsigned char);

	unsigned int histogrammSize = HISTOGRAMMSIZE * sizeof(unsigned int);


	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_image_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &image_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaMalloc((void **)&d_image_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &image_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaMalloc((void **)&d_src_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_src_image, src_image, imgSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image dest to gpu
	err = cudaMalloc((void **)&d_dest_histogramm, histogrammSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	err = cudaMemcpy(d_dest_histogramm, histogramm, histogrammSize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}


	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((image_width - 1) / threads + 1, (image_height - 1) / threads + 1, 1);
	getHistogrammTiledKernel <<<blocks_per_grid, threads_per_block >>>(d_image_width, d_image_height, d_src_image, d_dest_histogramm);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(histogramm, d_dest_histogramm, histogrammSize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	long histoCount = 0;

	for (int i = 0; i < HISTOGRAMMSIZE; i++) {
		histoCount += histogramm[i];
	}

	printf("HistogrammSize: %ld \n", histoCount);
	
	cudaFree(d_dest_histogramm);
	cudaFree(d_image_width);
	cudaFree(d_image_height);
	cudaFree(d_src_image);
};


void streamAufgabe5(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image)
{
	int *d_image_width, *d_image_height;
	unsigned char *d_src_image, *d_dest_image, *d_dest_image_sobel;

	unsigned int imgSize = image_width * image_height;
	unsigned int imgSizeRgb = imgSize * CHANNELS * sizeof(unsigned char);
	unsigned int imgSizeGray = imgSize * sizeof(unsigned char);
	
	//Cuda Stream vars
	const unsigned int stream_count = 4;

	//Kernel vars
	unsigned int threads = 16;
	int stream_width = image_width;
	int stream_height = image_height / stream_count;
	int stream_size = stream_width * stream_height;
	int stream_size_gray = stream_size * sizeof(unsigned char);
	int stream_size_rgb = stream_size * CHANNELS * sizeof(unsigned char);

	//tiled sobel
	dim3 threads_per_block_tiled(BLOCK_W, BLOCK_W, 1);
	//Per Grid N/Tile_wisth blocks
	dim3 blocks_per_grid_tiled((stream_width - 1) / TILE_W + 1, (stream_height - 1) / TILE_W + 1, 1);
	
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Pro Grid N/16 Blöcke
	dim3 blocks_per_grid((stream_width - 1) / threads + 1, (stream_height - 1) / threads + 1, 1);

	cudaStream_t streams[stream_count];

	int dev_count;
	cudaDeviceProp prop;

	cudaError_t err = cudaSuccess;

	//Enable device Overlap
	err = cudaGetDeviceCount(&dev_count);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Set device to a device with overlap property
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&prop, i);

		if (prop.deviceOverlap) {
			err = cudaSetDevice(i);
			if (err != cudaSuccess) {
				printf("%s in %s at line %d\n",
					cudaGetErrorString(err), __FILE__, __LINE__);
				exit(EXIT_FAILURE);
			}

		}
	}
	
	err = cudaHostAlloc((void **)&d_image_width, sizeof(int), cudaHostAllocDefault);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_image_width, &stream_width, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image height to gpu
	err = cudaHostAlloc((void **)&d_image_height, sizeof(int), cudaHostAllocDefault);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_image_height, &stream_height, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Copy image src to gpu
	err = cudaHostAlloc((void **)&d_src_image, imgSizeRgb, cudaHostAllocDefault);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Alloc memory for grayscale image
	err = cudaHostAlloc((void **)&d_dest_image, imgSizeGray, cudaHostAllocDefault);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaHostAlloc((void **)&d_dest_image_sobel, imgSizeGray, cudaHostAllocDefault);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Create cuda Streams & memory for each stream
	for (int i = 0; i < stream_count; i++) {
		//Create cuda Streams
		err = cudaStreamCreate(&streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}
		
	}

	//fill memory
	for (int i = 0; i < stream_count; i++) {
		//calc offset for memory copy
		int offset_gray = i * stream_size;
		int offset_rgb = offset_gray * CHANNELS;

		//copy memory for each stream
		err = cudaMemcpyAsync(&d_src_image[offset_rgb], &src_image[offset_rgb], stream_size_rgb, cudaMemcpyHostToDevice, streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(&d_dest_image[offset_gray], &dest_image[offset_gray], stream_size_gray, cudaMemcpyHostToDevice, streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}

		err = cudaMemcpyAsync(&d_dest_image_sobel[offset_gray], &dest_image[offset_gray], stream_size_gray, cudaMemcpyHostToDevice, streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}


	}

	//execute kernel for grayscale
	for (int i = 0; i < stream_count; i++) {
		int offset_gray = i * stream_size;
		int offset_rgb = offset_gray * CHANNELS;

		rgbToGrayscaleKernelStream<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>(d_image_width, d_image_height, d_src_image, d_dest_image, offset_rgb, offset_gray);	
	}

	for (int i = 0; i < stream_count; i++) {
		err = cudaStreamSynchronize(streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	//execute kernel for sobel
	for (int i = 0; i < stream_count; i++) {
		int offset_sobel = i * stream_size;

		//sobelFilterKernelStreams<<<blocks_per_grid, threads_per_block, 0, streams[i]>>>(d_image_width, d_image_height, d_dest_image, d_dest_image_sobel, offset_sobel);
		sobelFilterKernelTiledStreams<<<blocks_per_grid_tiled, threads_per_block_tiled, 0, streams[i] >> >(d_image_width, d_image_height, d_dest_image, d_dest_image_sobel, offset_sobel);
	}

	for (int i = 0; i < stream_count; i++) {
		err = cudaStreamSynchronize(streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}
	}

	// Save grayscale data
	for (int i = 0; i < stream_count; i++) {
		int offset = i * stream_size;
		//printf("offset: %d\n", offset);
		err = cudaMemcpyAsync(&dest_image[offset], &d_dest_image_sobel[offset], stream_size_gray, cudaMemcpyDeviceToHost, streams[i]);
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n",
				cudaGetErrorString(err), __FILE__, __LINE__);
			exit(EXIT_FAILURE);
		}

	}

	cudaFreeHost(d_image_width);
	cudaFreeHost(d_image_height);
	cudaFreeHost(d_src_image);
	cudaFreeHost(d_dest_image);
	cudaFreeHost(d_dest_image_sobel);

	for (int i = 0; i < stream_count; i++) {
		cudaStreamDestroy(streams[i]);
	}
	
};