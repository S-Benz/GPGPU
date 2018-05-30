#ifndef _kernel_h
#define _kernel_h
#endif // !_kernel_h

//Keeps one color channel of an image and sets the other ones to 0
void setColorChannel(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image, unsigned char channel_to_keep);
//Converts an 3 Channel RGB Image to a 1 Channel grayscale image
void rgbToGrayscale(int image_width, int image_height, unsigned char *src_image, unsigned char *dest_image);
//Applies the 3x3 sobel filter to a 1 Channel grayscale image
void sobelFilter(int image_width, int image_height, unsigned char *src_iamge, unsigned char *dest_image);

void sobelFilterShared(int image_width, int image_height, unsigned char *src_iamge, unsigned char *dest_image);

void sobelFilterTexture(int image_width, int image_height, unsigned char *src_iamge, unsigned char *dest_image);
