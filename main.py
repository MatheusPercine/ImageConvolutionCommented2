import sys

import convolution
from image import Image
import kernel
import time


def main(argc, argv):
    if argc != 3:
        print("Usage: python main.py <input_image> <output_image>")
        raise ValueError("Invalid arguments")

    # Load the image
    image = Image(argv[1])

    print("Image loaded successfully from the path: ", argv[1])
    print("Image dimensions: ", image.height, "x", image.width, "x", " - ", image.channels)
    print("\nSelect the kernel you want to apply: ")
    print("1. Sharpen")
    print("2. Edge Detection")
    print("3. Emboss")
    print("4. Gaussian")
    kernel_choice = int(input("Enter the kernel number: "))

    # Create a kernel based on the user's choice
    match kernel_choice:
        case 1:
            selected_kernel = kernel.sharpen_kernel
        case 2:
            """
            TAMANHO DO KERNEL
            Kernels maiores tendem a capturar mais detalhes e características das bordas na imagem, mas também podem suavizar pequenas variações. Isso ocorre porque um kernel maior abrange uma área maior da imagem, fazendo com que a média das diferenças de intensidade de pixel se espalhe por uma região maior. Portanto, um kernel maior pode detectar bordas mais gerais e menos sensíveis a pequenos detalhes. 
            Kernels maiores aumentam o tempo de processamento, pois cada ponto da imagem precisa ser convolucionado com uma matriz maior. Em aplicações de tempo real ou com imagens de alta resolução, isso pode ser uma consideração importante.
            Kernels maiores podem resultar em bordas mais espessas na imagem resultante, enquanto kernels menores tendem a produzir bordas mais finas. Isso ocorre porque a soma dos valores do kernel em um kernel maior se espalha por mais pixels, resultando em um efeito de "mancha" mais amplo.
            Kernels menores, por outro lado, são mais sensíveis a pequenas variações nos níveis de intensidade dos pixels. Eles podem detectar bordas finas e detalhes pequenos com mais precisão. No entanto, isso pode resultar em uma imagem mais "ruidosa", com mais bordas detectadas em áreas onde há pequenas variações de intensidade que não são necessariamente bordas significativas.
            """
            kernel_size = int(input("Enter the size of the kernel: "))
            selected_kernel = kernel.Kernel2D(kernel.edge_detection_kernel(kernel_size))
            # print(selected_kernel.kernel)
        case 3:
            selected_kernel = kernel.emboss_kernel
        case 4:
            """
            TAMANHO DO KERNEL
            Kernels Gaussianos maiores produzem uma suavização mais forte na imagem. Isso ocorre porque um kernel maior abrange uma área maior da imagem e, consequentemente, a média ponderada dos valores dos pixels é calculada sobre uma área maior. Isso resulta na suavização de detalhes finos e na redução de ruídos na imagem.
            Kernels Gaussianos menores proporcionam uma suavização mais leve. Eles afetam menos pixels ao redor de cada ponto, preservando mais detalhes e texturas na imagem. Isso pode ser útil quando você deseja suavizar ligeiramente uma imagem sem perder muita informação detalhada.
            Kernels maiores aumentam o tempo de processamento, pois cada ponto da imagem precisa ser convolucionado com uma matriz maior. Em aplicações de tempo real ou com imagens de alta resolução, isso pode ser uma consideração importante.
            Kernels Gaussianos maiores espalham o desfoque por uma área maior, resultando em uma aparência mais suave e borrada. Isso é útil para reduzir ruídos ou criar efeitos de desfoque artístico.
            """

            """
            DESVIO PADRÃO
            -Suavização:
            Desvio padrão pequeno: Um sigma menor resulta em uma curva Gaussiana mais estreita, o que significa que os valores do kernel caem rapidamente à medida que se afastam do centro. Isso leva a uma suavização leve, onde apenas os pixels próximos ao ponto central contribuem significativamente para o valor do pixel convoluído. Detalhes finos são preservados.
            Desvio padrão grande: Um sigma maior resulta em uma curva Gaussiana mais larga, o que significa que os valores do kernel caem mais lentamente. Isso leva a uma suavização mais forte, onde pixels mais distantes do ponto central também contribuem significativamente para o valor do pixel convoluído. Detalhes finos são suavizados ou borrados.
            -Efeito de Desfoque:
            Desvio padrão pequeno: O efeito de desfoque é menos pronunciado. A imagem resultante terá um desfoque leve, mantendo a maioria dos detalhes e texturas originais.
            Desvio padrão grande: O efeito de desfoque é mais pronunciado. A imagem resultante terá um desfoque mais forte, com a redução de detalhes finos e a suavização de texturas.
            -Amplitude da Influência:
            Desvio padrão pequeno: A influência de cada pixel no resultado da convolução é restrita a uma área menor. Os pixels fora dessa área têm pouca ou nenhuma influência.
            Desvio padrão grande: A influência de cada pixel no resultado da convolução se estende por uma área maior. Isso significa que mais pixels contribuem para o valor de cada pixel convoluído, resultando em uma mistura mais ampla dos valores dos pixels.
            """

            kernel_size = int(input("Enter the size of the kernel: "))
            sigma = float(input("Enter the standard deviation of the Gaussian distribution: "))
            selected_kernel = kernel.Kernel2D(kernel.gaussian_kernel(kernel_size, sigma))
            # print(selected_kernel.kernel)
        case _:
            # selected_kernel = kernel.identity_kernel
            raise ValueError("Invalid kernel choice")

    # Apply the selected kernel and measure the time
    print("\nSelect the model you want to use: ")
    print("1. Sequential")
    print("2. Pool")
    print("3. Block")
    print("4. Numba")
    model_choice = int(input("Enter the model number: "))

    if model_choice == 1:
        """
        1. Sequential
        Forma sequencial de fazer a convolução
        No arquivo convolution.py, temos a seguinte ordem de chamadas de função: 
        convolution() --> calculate_convolution_padless()
        Desvantagens do Modelo Sequencial:
        Desempenho:
        O processamento sequencial é lento para imagens grandes e kernels complexos devido à falta de paralelismo.
        Cada pixel é processado individualmente, o que pode levar a um tempo de execução prolongado.
        Vantagens do Modelo Sequencial:
        Simplicidade:
        A implementação é direta e fácil de entender.
        Não há necessidade de lidar com problemas de sincronização ou comunicação entre threads/processos.
        """
        start_time = time.time()
        print("Applying the kernel...")
        result_image = convolution.convolution(image.data, selected_kernel.kernel)
        end_time = time.time()
        result_image = Image.from_data(result_image)

    else:
        print("\nSelect the number of threads/processes you want to use: ")
        num_threads = int(input("Enter the number of threads/processes: "))
        start_time = time.time()
        print("Applying the kernel...")
        match model_choice:
            case 2:
                """
                2. Pool
                No arquivo convolution.py, temos a seguinte ordem de chamadas de função: 
                convolution_pool() --> calculate_convolution_padless()
                """
                result_image = convolution.convolution_pool(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case 3:
                """
                3. Block
                No arquivo convolution.py, temos a seguinte ordem de chamadas de função: 
                convolution_block() --> calculate_convolution_block()
                """
                result_image = convolution.convolution_block(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case 4:
                """
                4. Numba
                No arquivo convolution.py, temos a seguinte ordem de chamadas de função: 
                convolution_numba()
                """

                result_image = convolution.convolution_numba(image.data, selected_kernel.kernel, num_threads)
                end_time = time.time()
                result_image = Image.from_data(result_image)
            case _:
                raise ValueError("Invalid model choice")

    print("Time taken to apply the kernel: ", end_time - start_time, " seconds")

    # Save the image
    result_image.save_image(argv[2])
    print("Image saved successfully to the path: ", argv[2])

    # Display the image
    result_image.show()


if __name__ == "__main__":

    main(len(sys.argv), sys.argv)
    # main(3, ["main.py", "images/dog.jpg", "output.jpg"])