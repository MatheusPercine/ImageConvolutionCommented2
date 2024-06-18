/* Convolution in an image
   image: input image
   kernel: convolution kernel
   output: output image
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "convolution.h"

// Function to perform convolution
int convolution(int8_t *image, float *kernel, int8_t *output, int image_width, int image_height, int kernel_width, int kernel_height) {
    int i, j, m, n, mm, nn;
    float sum;
    int kernel_center_x, kernel_center_y;
    int row, col;

    // Find center of kernel
    kernel_center_x = kernel_width / 2;
    kernel_center_y = kernel_height / 2;

    // Convolution operation on the image
    // Loop over all pixels in the image
    for (i = 0; i < image_height; i++) { // Loop over all rows
        for (j = 0; j < image_width; j++) { // Loop over all columns
            sum = 0;
            for (m = 0; m < kernel_height; m++) { // Loop over kernel rows
                mm = kernel_height - 1 - m; // row index of flipped kernel
                for (n = 0; n < kernel_width; n++) { // Loop over kernel columns
                    nn = kernel_width - 1 - n; // column index of flipped kernel

                    // Index of input signal used for checking boundary
                    row = i + m - kernel_center_y;
                    col = j + n - kernel_center_x; 

                    // Ignore input samples which are out of bound
                    if (row >= 0 && row < image_height && col >= 0 && col < image_width) {
                        sum += image[row * image_width + col] * kernel[mm * kernel_width + nn]; // Perform convolution
                    }
                }
            }
            output[i * image_width + j] = sum; // Store the result in the output image
        }
    }
    return 0;
}

void *calculate_convolution_thread(void *arg) {
    t_Args *args = (t_Args *) arg;

    int i, j, m, n, mm, nn;
    float sum;
    int kernel_center_x, kernel_center_y;
    int row, col;
    int kernel_height = args->kernel_height;
    int kernel_width = args->kernel_width;
    int image_width = args->image_width;

    // Find center of kernel
    kernel_center_x = args->kernel_width / 2;
    kernel_center_y = args->kernel_height / 2;

    // Convolution operation on the image
    // Loop over all pixels in the image

    for (i = args->row_start; i < args->row_end; i++) { // Loop over all rows
        for (j = 0; j < image_width; j++) { // Loop over all columns
            sum = 0;
            for (m = 0; m < kernel_height; m++) { // Loop over kernel rows
                mm = kernel_height - 1 - m; // row index of flipped kernel
                for (n = 0; n < kernel_width; n++) { // Loop over kernel columns
                    nn = kernel_width - 1 - n; // column index of flipped kernel

                    // Index of input signal used for checking boundary
                    row = i + m - kernel_center_y;
                    col = j + n - kernel_center_x; 

                    // Ignore input samples which are out of bound
                    if (row >= 0 && row < args->image_height && col >= 0 && col < image_width) {
                        sum += args->image[row * image_width + col] * args->kernel[mm * kernel_width + nn]; // Perform convolution
                    }
                }
            }
            args->output[i * image_width + j] = sum; // Store the result in the output image
        }
    }
    free(args);
    pthread_exit(NULL);
}

int convolution_thread(int8_t *image, float *kernel, int8_t *output, int image_width, int image_height, int kernel_width, int kernel_height, int num_threads) {
    
    // Strategy: Divide the image floato num_threads parts and assign each part to a thread
    // Each thread will perform convolution on its part of the image
    // The division of the image is done row-wise, the remaining rows are assigned to the last thread
    // printf("Threads: %d\n", num_threads);

    pthread_t *tid = (pthread_t *) malloc(sizeof(pthread_t) * num_threads);
    if (!tid) {
        fprintf(stderr, "Error in memory allocation\n");
        return -1;
    }

    int block_size = image_height / num_threads;
    int start = 0;
    int end = block_size;

    for (int i = 0; i < num_threads; i++) {
        t_Args *arg = malloc(sizeof(t_Args));
        if (!arg) {
            fprintf(stderr, "Error in memory allocation\n");
            return -2;
        }
        arg->image = image;
        arg->kernel = kernel;
        arg->output = output;
        arg->image_width = image_width;
        arg->image_height = image_height;
        arg->kernel_width = kernel_width;
        arg->kernel_height = kernel_height;
        arg->row_start = start;
        arg->row_end = end;

        if (i == num_threads - 1) {
            arg->row_end = image_height; // Assign the remaining rows to the last thread
        }

        if (pthread_create(tid + i, NULL, calculate_convolution_thread, (void *) arg)) {
            fprintf(stderr, "Erro: pthread_create()\n");
            return -3;
        }

        start = end;
        end += block_size;
    }

    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(tid[i], NULL)) {
            fprintf(stderr, "Erro: pthread_join()\n");
            return -4;
        }
    }

    free(tid);
    return 0;
}
