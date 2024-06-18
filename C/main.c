/* Main file for the convolution program with the following functions:
   Read an matrix from a file
   Write an matrix to a file
   Perform convolution on an matrix/image
   Perform convolution on an matrix/image using multiple threads
   Calculate time elapsed in a convolution operation
 */

#include "convolution.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"

// #define TEST

typedef struct {
    int linhas;
    int colunas;
    float *elementos;
} Matriz;

typedef struct {
    int linhas;
    int colunas;
    int8_t *elementos;
} Img;

Img *le_imagem_bin(const char *arquivo) {
    Img *img = (Img *) malloc(sizeof(Img));
    FILE * descritor_arquivo; //descritor do arquivo de entrada
    size_t ret; //retorno da funcao de leitura no arquivo de entrada

    if (!img) {
        fprintf(stderr, "Erro de alocação da imagem\n");
        return NULL;
    }
    
    descritor_arquivo = fopen(arquivo, "rb");
    if (!descritor_arquivo) {
        fprintf(stderr, "Erro de abertura do arquivo de imagem\n");
        return NULL;
    }

    ret = fread(&img->linhas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de leitura das dimensões da imagem\n");
        return NULL;
    }

    ret = fread(&img->colunas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de leitura das dimensões da imagem\n");
        return NULL;
    }

    img->elementos = (int8_t *) malloc(sizeof(int8_t) * img->linhas * img->colunas);
    if (!img->elementos) {
        fprintf(stderr, "Erro de alocação da imagem\n");
        return NULL;
    }

    ret = fread(img->elementos, sizeof(int8_t), img->linhas * img->colunas, descritor_arquivo);
    if (ret < img->linhas * img->colunas) {
        fprintf(stderr, "Erro de leitura dos elementos da imagem\n");
        return NULL;
    }
    
    # ifdef TEST
    printf("Imagem lida:\n");
    printf("Linhas: %d\n", img->linhas);
    printf("Colunas: %d\n", img->colunas);
    // for (int i = 0; i < img->linhas; i++) {
    //     for (int j = 0; j < img->colunas; j++) {
    //         printf("%d ", img->elementos[i * img->colunas + j]);
    //     }
    //     printf("\n");
    // }
    # endif

    return img;
}

int escreve_imagem_bin(const char *arquivo, Img *img) {
    FILE * descritor_arquivo; //descritor do arquivo de saida
    size_t ret; //retorno da funcao de escrita no arquivo de saida

    descritor_arquivo = fopen(arquivo, "wb");  // abre o arquivo para escrita em binario
    if (!descritor_arquivo) {
        fprintf(stderr, "Erro de abertura do arquivo de imagem\n");
        return -1;
    }

    ret = fwrite(&img->linhas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de escrita das dimensões da imagem\n");
        return -2;
    }

    ret = fwrite(&img->colunas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de escrita das dimensões da imagem\n");
        return -3;
    }

    ret = fwrite(img->elementos, sizeof(int8_t), img->linhas * img->colunas, descritor_arquivo); // escreve a imagem no arquivo
    if (ret < img->linhas * img->colunas) {
        fprintf(stderr, "Erro de escrita dos elementos da imagem\n");
        return -4;
    }

    fclose(descritor_arquivo);
    return 0;
}


Matriz *le_matriz_bin(const char *arquivo) {
    Matriz *matriz = (Matriz *) malloc(sizeof(Matriz));
    FILE * descritor_arquivo; //descritor do arquivo de entrada
    size_t ret; //retorno da funcao de leitura no arquivo de entrada

    if (!matriz) {
        fprintf(stderr, "Erro de alocação da matriz de kernel\n");
        return NULL;
    }
    
    descritor_arquivo = fopen(arquivo, "rb");
    if (!descritor_arquivo) {
        fprintf(stderr, "Erro de abertura do arquivo de kernel\n");
        return NULL;
    }

    ret = fread(&matriz->linhas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de leitura das dimensões da matriz\n");
        return NULL;
    }

    ret = fread(&matriz->colunas, sizeof(int), 1, descritor_arquivo);
    if (!ret) {
        fprintf(stderr, "Erro de leitura das dimensões da matriz de kernel\n");
        return NULL;
    }

    matriz->elementos = (float *) malloc(sizeof(float) * matriz->linhas * matriz->colunas);
    if (!matriz->elementos) {
        fprintf(stderr, "Erro de alocação da matriz de kernel\n");
        return NULL;
    }

    ret = fread(matriz->elementos, sizeof(float), matriz->linhas * matriz->colunas, descritor_arquivo);
    if (ret < matriz->linhas * matriz->colunas) {
        fprintf(stderr, "Erro de leitura dos elementos da matriz de kernel\n");
        return NULL;
    }
    
    # ifdef TEST
    printf("Matriz lida:\n");
    printf("Linhas: %d\n", matriz->linhas);
    printf("Colunas: %d\n", matriz->colunas);
    for (int i = 0; i < matriz->linhas; i++) {
        for (int j = 0; j < matriz->colunas; j++) {
            printf("%.2f ", matriz->elementos[i * matriz->colunas + j]);
        }
        printf("\n");
    }
    # endif

    fclose(descritor_arquivo);
    return matriz;
}


int main(int argc, char *argv[]) {

    if (argc < 5) {
        printf("Argumentos insuficientes\n - %d", argc);
        fprintf(stderr, "Uso: %s <matriz.bin> <kernel.bin> <n_threads> <matriz_saida>\n", argv[0]);
        return 1;
    }
    Matriz *kernel = le_matriz_bin(argv[2]);
    Img *imagem = le_imagem_bin(argv[1]);
    

    if (!imagem || !kernel) {
        fprintf(stderr, "Erro de leitura da imagem ou do kernel\n");
        return -2;
    }

    int n_threads = atoi(argv[3]);
    // printf("Threads: %d\n", n_threads);
    if (n_threads < 1) {
        fprintf(stderr, "Número de threads inválido\n");
        return 2;
    }

    Img *saida = (Img *) malloc(sizeof(Img));
    if (!saida) {
        fprintf(stderr, "Erro de alocação da matriz de saída\n");
        return -3;
    }

    // Alocar memória para a matriz de saída
    saida->linhas = imagem->linhas;
    saida->colunas = imagem->colunas;
    saida->elementos = (int8_t *) malloc(sizeof(int8_t) * saida->linhas * saida->colunas);
    if (!saida->elementos) {
        fprintf(stderr, "Erro de alocação da matriz de saída\n");
        return -3;
    }

    double start, finish, elapsed;

    GET_TIME(start);
    // printf("Iniciando convolução\n");
    if (n_threads == 1) {
        if (convolution(imagem->elementos, kernel->elementos, saida->elementos, imagem->linhas, imagem->colunas, kernel->linhas, kernel->colunas)) {
            fprintf(stderr, "Erro na convolução\n");
            return -4;
        }

    } else {
        if (convolution_thread(imagem->elementos, kernel->elementos, saida->elementos, imagem->linhas, imagem->colunas, kernel->linhas, kernel->colunas, n_threads)) {
            fprintf(stderr, "Erro na convolução\n");
            return -4;
        }
    }
    GET_TIME(finish);
    // printf("Convolução finalizada\n");

    elapsed = finish - start;
    printf("%.4f\n", elapsed);
    #ifdef TEST
    printf("Salvando matriz de saída\n");
    printf("Imagem de saída salva em %s\n", argv[4]);
    #endif

    if (escreve_imagem_bin(argv[4], saida)) {
        fprintf(stderr, "Erro na escrita da matriz de saída\n");
        return -5;
    }

    free(imagem->elementos);
    free(imagem);
    free(kernel->elementos);
    free(kernel);
    free(saida->elementos);
    free(saida);

    return 0;
}
