// OpenCVReadVideo.cpp : Definiert den Einstiegspunkt für die Konsolenanwendung.
//

#include <iostream>
#include "stdafx.h"
#include "kernel.h"

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main(int, char**)
{
//	VideoCapture cap("Z:/Videos/robotica_1080.mp4"); // open the default camera
	VideoCapture cap("C:/Users/sbenz/Desktop/OpenCVReadVideo/Videos/robotica_1080.mp4");
//	VideoCapture cap("C:/Users/fischer/Downloads/Bennu4k169Letterbox_h264.avi"); // open the default camera
//	VideoCapture cap("D:/Users/fischer/Videos/fireworks.mp4");
//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.mp4");
//	VideoCapture cap("D:/Users/fischer/Videos/Bennu4k169Letterbox_h264.avi");

	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	//namedWindow("ColorChannelFilter", 1);
	namedWindow("Grayscale", 1);
	namedWindow("Sobel", 1);

	for (;;)
	{
		int firstCall = 0;
		Mat frame;
		//Output for color channel 
		Mat output;
		cap >> frame; // get a new frame from camera
		if (frame.dims == 0) { // we're done
			break;
		}
		//Output for grayscale
		Mat grayscale(frame.rows, frame.cols, CV_8UC1, Scalar(0,0,0));
		//Output for sobel
		Mat sobel = grayscale.clone();
		
		if (firstCall++ == 0) {
			//cout << "frame: dims: " << frame.dims << ", size[0]: " << frame.size[0] << ", size[1]:" << frame.size[1] << ", step[0]: " << frame.step[0] << ", step[1]:" << frame.step[1];
			//cout << ", type: " << frame.type() << " (CV16U: " << CV_16UC1 << ", CV8UC3: " << CV_8UC3 << ")" << ", elemSize: " << frame.elemSize();
			//cout << ", rows: " << frame.rows << ", cols: " << frame.cols << ", size: " << frame.size << ", dataPtr: " << frame.data << endl;
			output = frame.clone();  // Kopie der Eingangsmatrix erzeugen, in deren Rohdaten (output.data) der CUDA-Kernel schreiben kann.
		}
		//cout << "src_size: " << frame.size << ", dest_size: " << output.size << endl;
		
		//exit(0);
//		cvtColor(frame, edges, COLOR_BGR2GRAY);
//		GaussianBlur(edges, edges, Size(7, 7), 1.5, 1.5);
//		Sobel(frame, edges, frame.depth(), 2, 2);
//		Canny(edges, edges, 0, 30, 3);
		unsigned char channel_to_keep = 'b';
		//setColorChannel(frame.cols, frame.rows,frame.data, output.data, channel_to_keep);
		rgbToGrayscale(grayscale.cols, grayscale.rows, frame.data, grayscale.data);
		sobelFilter(grayscale.cols, grayscale.rows, grayscale.data, sobel.data);
		//imshow("ColorChannelFilter", output);
		imshow("Grayscale", grayscale);
		imshow("Sobel", sobel);
		if (waitKey(1) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}