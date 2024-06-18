#ifndef CONVOLUTION_H
#define CONVOLUTION_H
#include <stdlib.h>

typedef struct {
    int8_t *image;
    float *kernel;
    int8_t *output;
    int row_start;
    int row_end;
    int image_width;
    int image_height;
    int kernel_width;
    int kernel_height;
} t_Args;

// Função para realizar a convolução de uma imagem
int convolution(int8_t *image, float *kernel, int8_t *output, int image_width, int image_height, int kernel_width, int kernel_height);

// Função para realizar a convolução de uma imagem multithread
int convolution_thread(int8_t *image, float *kernel, int8_t *output, int image_width, int image_height, int kernel_width, int kernel_height, int num_threads);

void *calculate_convolution_thread(void *arg);

#endif // CONVOLUTION_H