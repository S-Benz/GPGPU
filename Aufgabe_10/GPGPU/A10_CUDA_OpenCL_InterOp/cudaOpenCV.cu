#include "cudaKernel.h"
#ifdef WIN32
#include <windows.h>
#endif
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHANNELS 3
#define SOBEL_RADIUS 1

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

void cudaGetOpenCVImageSize(unsigned int &cols, unsigned int &rows) {
	cols = 1440;
	rows = 1080;
}

void cudaInit ( unsigned int texId, unsigned int vboId, unsigned int cols, unsigned int rows){
	
	//Device Params
	unsigned int *d_width, *d_height;
	float *d_pointer;
	size_t ptr_size = 0;
	cudaArray *texArray;
	unsigned char *d_src_image, *d_dest_image;

	unsigned int imgSize = cols * rows * sizeof(unsigned char);
	unsigned int imgSizeRGB = imgSize * CHANNELS;


	cudaError_t err = cudaSuccess;

	cudaGraphicsResource_t vboRes;
	cudaGraphicsResource_t texRes;

	//Init Cuda variables
	//Width
	err = cudaMalloc((void **)&d_width, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_width, &cols, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Height
	err = cudaMalloc((void **)&d_height, sizeof(int));
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_height, &rows, sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Image
	err = cudaMalloc((void **)&d_src_image, imgSizeRGB);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Register Gl Buffer
	err = cudaGraphicsGLRegisterBuffer(&vboRes, vboId, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Buffer Pointer
	err = cudaGraphicsMapResources(1, &vboRes, 0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsResourceGetMappedPointer((void**)&d_pointer, &ptr_size, vboRes);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Register Gl Texture
	err = cudaGraphicsGLRegisterImage(&texRes, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsMapResources(1, &texRes);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsSubResourceGetMappedArray(&texArray, texRes, 0, 0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Kernel Calls
	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((cols - 1) / threads + 1, (rows - 1) / threads + 1, 1);
	//rgbToGrayscaleKernel<<<blocks_per_grid, threads_per_block >> >(d_width, d_height, d_src_image, d_dest_image);

	//Copy data to Opengl

	//Clean up
	cudaFree(d_width);
	cudaFree(d_height);
	cudaFree(d_src_image);
	cudaGraphicsUnmapResources(1, &texRes);
	cudaGraphicsUnmapResources(1, &vboRes, 0);
	
}

int cudaExecOneStep(void) {
	return 0;
}
