#include "cudaKernel.h"
#ifdef WIN32
#include <windows.h>
#endif
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "opencv2/opencv.hpp"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHANNELS 3
#define SOBEL_RADIUS 1
#define HISTOGRAMMSIZE 256

using namespace cv;

cudaGraphicsResource_t vboRes; // cuda vertex buffer reference
cudaGraphicsResource_t texRes; // cuda texture reference
cudaGraphicsResource_t texResGray; // cuda texture reference
int *d_width, *d_height; // device memory varíables
VideoCapture cap("C:/Users/sbenz/Desktop/OpenCVReadVideo/Videos/robotica_1080.mp4"); // Opencv video capture
Mat currFrame;


__global__ void getHistogrammKernel(int cu_image_width, int cu_image_height, unsigned char *cu_src_image, unsigned int *cu_dest_histogramm)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows


	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;

	while (x < cu_image_width && y < cu_image_height) {
		int index = y * cu_image_width + x;

		atomicAdd(&(cu_dest_histogramm[cu_src_image[index]]), 1);

		x += stride_x;
		y += stride_y;
	}
}

__global__ void sobelFilterKernel(int cu_image_width, int cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
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
	int global_index = (y * cu_image_width)+ x;

	if (x >= SOBEL_RADIUS && x < cu_image_width - 1 && y >= SOBEL_RADIUS && y < cu_image_height - 1) {
		//Calc Sobel X & Y if the thread is inside the filter area
		int sobel_gradient_y = 0, sobel_gradient_x = 0, sobel_magnitude = 0;

		for (int j = -SOBEL_RADIUS; j <= SOBEL_RADIUS; j++) {
			for (int k = -SOBEL_RADIUS; k <= SOBEL_RADIUS; k++) {
				sobel_gradient_x += cu_src_image[(y + j) * cu_image_width + (x + k)] * sobel_x[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
				sobel_gradient_y += cu_src_image[(y + j) * cu_image_width + (x + k)] * sobel_y[j + SOBEL_RADIUS][k + SOBEL_RADIUS];
			}
		}

		//Calc Sobel magnitude and save it to the image
		sobel_magnitude = (int)sqrt((float)pow((float)sobel_gradient_x, 2) + (float)pow((float)sobel_gradient_y, 2));

		cu_dest_image[global_index] = (unsigned char)sobel_magnitude;
	}
	else {
		if (x < cu_image_width && y < cu_image_height) {
			cu_dest_image[global_index] = 0;
		}
	}
}

//Kernel rgb to grayscale function
__global__ void rgbToGrayscaleKernel(int cu_image_width, int cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows
	unsigned char r, g, b, gray;

	if (x < cu_image_width && y < cu_image_height) {
		int grayOffset = (y * (cu_image_width) + x);
		int rgbOffset = grayOffset * CHANNELS;

		b = cu_src_image[rgbOffset];
		g = cu_src_image[rgbOffset + 1];
		r = cu_src_image[rgbOffset + 2];

		gray = 0.21 * r + 0.71 *g + 0.07 *b;

		cu_dest_image[grayOffset] = gray;
	}
}

__global__ void colorKernel(int cu_image_width, int cu_image_height, unsigned char *cu_src_image, unsigned char *cu_dest_image)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x; //cols
	int y = blockIdx.y * blockDim.y + threadIdx.y; //rows
	unsigned char r, g, b, gray;

	if (x < cu_image_width && y < cu_image_height) {
		int grayOffset = (y * (cu_image_width) + x);
		int rgbOffset = grayOffset * CHANNELS;
		int rgbaOffset = grayOffset * (CHANNELS + 1);

		b = cu_src_image[rgbOffset];
		g = cu_src_image[rgbOffset + 1];
		r = cu_src_image[rgbOffset + 2];
		
		cu_dest_image[rgbaOffset] = r;
		cu_dest_image[rgbaOffset + 1] = g;
		cu_dest_image[rgbaOffset + 2] = b;
		cu_dest_image[rgbaOffset + 3] = 0;
	}
}

void cudaGetOpenCVImageSize(unsigned int &cols, unsigned int &rows) {
	cols = 1440;
	rows = 1080;
}

void cudaInit ( unsigned int texId, unsigned int texIdGray, unsigned int vboId, unsigned int cols, unsigned int rows){
	
	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
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

	//Register Gl Texture
	err = cudaGraphicsGLRegisterImage(&texRes, texId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsGLRegisterImage(&texResGray, texIdGray, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	//Init device memory
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

}

void cudaClearAllocatedData() {
	cudaFree(d_width);
	cudaFree(d_height);
}

void copyColorImage(unsigned char *src_image, int width, int height) {
	unsigned char *d_dest_image, *d_src_image;
	cudaArray *texArray;

	unsigned int imgSize = width * height * sizeof(unsigned char);
	unsigned int imgSizeRgb = imgSize * CHANNELS;
	unsigned int imgSizeRgba = imgSize * (CHANNELS + 1);

	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(0);
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

	err = cudaMalloc((void **)&d_dest_image, imgSizeRgba);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Texture
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


	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Pro Grid N/16 Blöcke, n = Anzahl Threads
	dim3 blocks_per_grid((width - 1) / threads + 1, (height - 1) / threads + 1, 1);

	colorKernel <<<blocks_per_grid, threads_per_block >> >(width, height, d_src_image, d_dest_image);

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToArray(texArray, 0, 0, d_dest_image, imgSizeRgba, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsUnmapResources(1, &texRes);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_dest_image);
	cudaFree(d_src_image);
	cudaFree(d_height);
	cudaFree(d_width);
}

void rgbToGrayscale(unsigned char *src_image, int width, int height) {
	unsigned char *d_src_image;
	unsigned char *d_dest_image;
	cudaArray *texArray;

	unsigned int imgSize = width * height * sizeof(unsigned char);
	unsigned int imgSizeRgb = imgSize * CHANNELS;

	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(0);
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

	err = cudaMalloc((void **)&d_dest_image, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Texture
	err = cudaGraphicsMapResources(1, &texResGray);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsSubResourceGetMappedArray(&texArray, texResGray, 0, 0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Pro Grid N/16 Blöcke, n = Anzahl Threads
	dim3 blocks_per_grid((width - 1) / threads + 1, (height - 1) / threads + 1, 1);

	rgbToGrayscaleKernel<<<blocks_per_grid, threads_per_block>>>(width, height, d_src_image, d_dest_image);

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToArray(texArray, 0, 0, d_dest_image, imgSize, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsUnmapResources(1, &texResGray);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_dest_image);
	cudaFree(d_src_image);
}

void sobelFilter(unsigned char *src_image, int width, int height)
{
	unsigned char *d_src_image, *d_dest_image_gray, *d_dest_image_sobel;
	cudaArray *texArray;

	unsigned int imgSize = width * height * sizeof(unsigned char);
	unsigned int imgSizeRgb = imgSize * CHANNELS;

	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
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
	err = cudaMalloc((void **)&d_dest_image_gray, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaMalloc((void **)&d_dest_image_sobel, imgSize);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Texture
	err = cudaGraphicsMapResources(1, &texResGray);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsSubResourceGetMappedArray(&texArray, texResGray, 0, 0);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((width - 1) / threads + 1, (height - 1) / threads + 1, 1);

	rgbToGrayscaleKernel <<<blocks_per_grid, threads_per_block >> >(width, height, d_src_image, d_dest_image_gray);

	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	sobelFilterKernel <<<blocks_per_grid, threads_per_block>>>(width, height, d_dest_image_gray, d_dest_image_sobel);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpyToArray(texArray, 0, 0, d_dest_image_sobel, imgSize, cudaMemcpyDeviceToDevice);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	err = cudaGraphicsUnmapResources(1, &texResGray);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	cudaFree(d_src_image);
	cudaFree(d_dest_image_gray);
	cudaFree(d_dest_image_sobel);
};

void getHistogramm(int width, int height, unsigned char *src_image)
{
	unsigned char *d_src_image;
	unsigned int  *d_dest_histogramm;
	float *vboPtr;

	unsigned int imgSize = (width * height) * sizeof(unsigned char);

	unsigned int histogrammSize = HISTOGRAMMSIZE * sizeof(unsigned int);

	size_t vboSize = HISTOGRAMMSIZE * 3 * sizeof(float);


	cudaError_t err = cudaSuccess;

	//Set Device
	err = cudaSetDevice(0);
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

	//VBO
	err = cudaGraphicsResourceGetMappedPointer((void **)&vboPtr, &vboSize, vboRes);
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n",
			cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}

	unsigned int threads = 16;
	// Use a Grid with one Block containing 16x16 Threads
	dim3 threads_per_block(threads, threads, 1);
	//Per Grid N/16 Blocks
	dim3 blocks_per_grid((width - 1) / threads + 1, (height - 1) / threads + 1, 1);
	getHistogrammKernel << <blocks_per_grid, threads_per_block >> >(width, height, d_src_image, d_dest_histogramm);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", err);
		exit(EXIT_FAILURE);
	}


	cudaFree(d_dest_histogramm);
	cudaFree(d_src_image);
};


void displayGrayscaleImage(void) {
	rgbToGrayscale(currFrame.data, currFrame.cols, currFrame.rows);
}

void applySobelFilter(void) {
	sobelFilter(currFrame.data, currFrame.cols, currFrame.rows);
}

void displayColorImage(void) {
	copyColorImage(currFrame.data, currFrame.cols, currFrame.rows);
}

void displayHistogramm(void) {

}

int cudaExecOneStep(void) {
	// Check if videocapture suceeded
	if (!cap.isOpened()) {
		return -1;
	}

	cap >> currFrame;

	// Check if the video is done
	if (currFrame.empty()) {
		cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	}

	return 0;
}