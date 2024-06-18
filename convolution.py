from numba import jit, prange, set_num_threads
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


def calculate_convolution(image: np.ndarray, x, y, c, kernel: np.ndarray, new_image: np.ndarray = None):
    """
    Calculate the convolution of a pixel
    :param image: The image data
    :param x: The x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :param new_image: The new image to store the result
    :return: The convoluted pixel
    """

    '''
    Kernel example:

    -------------
    | 1 | 2 | 3 | 
    | 4 | 5 | 6 |
    | 7 | 8 | 9 |
    -------------
    x = 2
    y = 2

    x -> 1, 2, 3 := x - kernel.center[0] + i
    y -> 1, 2, 3 := y - kernel.center[1] + j
    '''
    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)

    # Initialize the result
    result = 0

    # Iterate over the kernel
    for i in range(kernel_height):
        for j in range(kernel_width):
            x_k = x - kernel_center[0] + i
            y_k = y - kernel_center[1] + j
            result += image[x_k, y_k, c] * kernel[i, j]

    if new_image is not None:
        new_image[x, y, c] = result
    return result


def calculate_convolution_padless(image: np.ndarray, x, y, c, kernel: np.ndarray, new_image: np.ndarray = None):
    """
    Calculate the convolution of a pixel without padding
    :param image: The image data
    :param x: The x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :param new_image: The new image to store the result
    :return: The convoluted pixel
    """

    """
    A função calculate_convolution_padless no arquivo convolution.py é responsável por calcular a convolução de um pixel específico em uma imagem, aplicando um kernel sem adicionar padding (zeros ao redor da imagem). Isso significa que a convolução é calculada apenas para os pixels que têm todos os seus vizinhos necessários dentro dos limites da imagem.
    A importância do "padless" se dá porque isso evita distorções causadas pelo padding, mas também significa que a convolução não é calculada para pixels nas bordas da imagem onde o kernel se estenderia além dos limites da imagem.
    A função começa determinando as dimensões do kernel (kernel_height e kernel_width). Depois, calcula o centro do kernel (kernel_center), que é usado para alinhar o kernel corretamente com o pixel atual. As dimensões da imagem (image_height, image_width, _) são obtidas para verificar os limites da imagem durante a convolução.
    'result' é inicializado como 0. Este valor acumulará a soma ponderada dos valores dos pixels vizinhos multiplicados pelos valores correspondentes do kernel.
    Em seguida, começa a iteração sobre o kernel da seguinte forma:
    Dois loops aninhados percorrem cada elemento do kernel.
    Para cada elemento (i, j) do kernel, a posição correspondente na imagem (x_k, y_k) é calculada:
    x_k = x - kernel_center[0] + i
    y_k = y - kernel_center[1] + j
    Isso desloca o kernel ao redor do pixel atual (x, y).
    Os limites da Imagem são verificados.
    Se a posição (x_k, y_k) estiver fora dos limites da imagem (menor que 0 ou maior que as dimensões da imagem), o loop continua sem adicionar ao resultado (continue).
    Porém, caso a posição (x_k, y_k) estiver dentro dos limites da imagem, o valor do pixel correspondente na imagem image[x_k, y_k, c] é multiplicado pelo valor do kernel kernel[i, j] e somado ao result e a convolução no pixel é calculada.
    Se new_image for fornecida, o resultado é armazenado na posição (x, y, c) da nova imagem.
    A função retorna o resultado da convolução para o pixel atual.
    """

    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)
    image_height, image_width, _ = image.shape

    # Initialize the result
    result = 0

    # Iterate over the kernel
    for i in range(kernel_height):
        for j in range(kernel_width):
            x_k = x - kernel_center[0] + i
            y_k = y - kernel_center[1] + j
            if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                continue
            result += image[x_k, y_k, c] * kernel[i, j]

    if new_image is not None:
        new_image[x, y, c] = result
    return result


def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform a convolution operation on the image
    :param image:  data
    :param kernel: The kernel to perform the convolution
    :return: The convoluted image
    """

    """
    Função que performa uma operação de convolução em uma imagem
    Parametros: dados da imagem e kernel selecionado
    Primeiramente, armazena-se, em variáveis, informações relativas à forma da image, como altura, largura e número de canais.
    Depois, é criada uma matriz do mesmo formato da imagem original, porém preenchida com zeros. Esta matriz zerada se transformará na imagem convoluída.
    Em seguida, é realizada a operação de convolução para cada pixel da imagem original, isto é, um pixel para cada canal, para cada altura, para cada largura da imagem original.
    """

    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation
    for c in range(image_channels):
        for x in range(image_height):
            for y in range(image_width):
                calculate_convolution_padless(image, x, y, c, kernel, new_image)
    return new_image


def convolution_pool(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #1: using the Pool class
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """

    """
    A função convolution_pool performa uma operação de convolução na imagem utilizando multiplos processos e para isso considera a estratégia de usar a classe Pool.
    Cada processo pode tratar qualquer pixel da imagem. A tarefa de convolução de cada pixel é distribuída entre os processos disponíveis. Isso é feito através do método starmap do multiprocessing.Pool, que distribui a lista de tarefas (cada tarefa é calcular a convolução para um pixel específico) entre os processos.
    A abordagem distribui a carga de trabalho de forma equilibrada, pois cada processo recebe uma tarefa (pixel) por vez, e conforme vai concluindo, pega a próxima tarefa disponível até que todas as tarefas sejam concluídas.
    A funçaõ considera o balanceamento dinâmico: A distribuição dinâmica de tarefas ajuda a balancear a carga entre os processos. Nenhum processo fica ocioso enquanto houver tarefas pendentes.
    Primeiramente, as dimensões da imagem (image_height, image_width, image_channels) são obtidas e 'new_image' é inicializada como uma matriz de zeros com as mesmas dimensões da imagem original.
    A função cria um pool de processos usando o contexto gerenciador (with mp.Pool(num_processes) as pool), garantindo que os recursos sejam liberados automaticamente ao final da operação.
    Em seguida, Para cada canal de cor (c) da imagem, pool.starmap é usada para mapear a função calculate_convolution_padless para cada pixel da imagem em paralelo, starmap permite passar múltiplos argumentos para a função, no caso (image, x, y, c, kernel). 
    
    Uma lista de tarefas é criada, onde cada tarefa é uma tupla de argumentos a ser passada para a função calculate_convolution_padless. Cada tupla representa a convolução a ser realizada em um pixel específico (x, y) de um canal específico c. 
    starmap distribui essas tuplas de argumentos entre os processos do pool. Cada processo pega uma dessas tarefas e executa calculate_convolution_padless com os argumentos fornecidos.
    O resultado é uma lista achatada de valores convolucionados, que é então remodelada para corresponder às dimensões originais da imagem (reshape(image_height, image_width)).
    Os valores resultantes da convolução para cada canal são armazenados na new_image.
    """

    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using map
    with (mp.Pool(num_processes) as pool):
        for c in range(image_channels):
            new_image[:, :, c] = np.array(pool.starmap(calculate_convolution_padless,
                                                       [(image, x, y, c, kernel) for x in range(image_height)
                                                        for y in range(image_width)])).reshape(image_height,
                                                                                               image_width)
    return new_image


def calculate_convolution_block(image: np.ndarray, x_i, y_i, x_f, y_f, c, kernel: np.ndarray) -> np.ndarray:
    """
    Calculate the convolution of a pixel, return the block result instead of a single value
    :param image: The image data
    :param x_i: The initial x-coordinate of the block
    :param y_i: The initial y-coordinate of the block
    :param x_f: The final x-coordinate of the block
    :param y_f: The final y-coordinate of the block
    :param c: The channel of the pixel
    :param kernel: The kernel to perform the convolution
    :return: The convoluted block
    """

    """
    A função calculate_convolution_block realiza a operação de convolução em um bloco específico de uma imagem e retorna o resultado desse bloco.
    x_i, y_i são as coordenadas iniciais (superior esquerda) do bloco da imagem.
    x_f, y_f são as coordenadas finais (inferior direita) do bloco da imagem.
    A função começa determinando as dimensões do kernel (kernel_height e kernel_width). Depois, calcula o centro do kernel (kernel_center), que é usado para alinhar o kernel corretamente com o pixel atual. As dimensões da imagem (image_height, image_width, _) são obtidas para verificar os limites da imagem durante a convolução.
    Um array de zeros é criado para armazenar o resultado da convolução do bloco especificado.
    Na interação sobre o kernel, dois loops aninhados iteram sobre cada pixel do bloco especificado, dois loops aninhados iteram sobre cada elemento do kernel; calcula-se a posição do pixel da imagem a ser multiplicado pelo elemento correspondente do kernel; verifica-se se as coordenadas calculadas estão dentro dos limites da imagem. Se não estiverem, o loop continua para a próxima iteração, ignorando os valores fora dos limites; O valor do pixel da imagem é multiplicado pelo valor correspondente do kernel e somado ao resultado. 
    
    O bloco convolucionado é retornado como um array NumPy.
    """

    kernel_height, kernel_width = kernel.shape
    kernel_center = (kernel_height // 2, kernel_width // 2)
    image_height, image_width, _ = image.shape

    # Initialize the result
    result = np.zeros((x_f - x_i, y_f - y_i))

    # Iterate over the kernel
    for i in range(x_f - x_i):
        for j in range(y_f - y_i):
            for k in range(kernel_height):
                for l in range(kernel_width):
                    x_k = x_i + i - kernel_center[0] + k
                    y_k = y_i + j - kernel_center[1] + l
                    if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                        continue
                    result[i, j] += image[x_k, y_k, c] * kernel[k, l]
    return result


def convolution_block(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #2: divide the image into chunks and process them in parallel, then merge the results
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """

    """
    Nessa função, a operação de convolução segue a estratégia de dividir a imagem em blocos que são distribuídos entre diferentes processos. Cada processo trata de uma região específica (bloco) da imagem, calculando a convolução para todos os pixels dentro desse bloco. Após o processamento, os blocos são reunidos para formar a imagem convoluída final.
    Primeiramente, informações sobre a imagem são armazenadas. Uma variável chamada result é inicializada e funcionará para armazenar o resultado de convoluções. A variável new_image é criada e que é uma nova matriz com a mesma forma (dimensões) da imagem original, onde cada elemento é do tipo np.uint8 (inteiro sem sinal de 8 bits). A matriz é criada sem valores iniciais definidos, o que é eficiente em termos de tempo, já que os valores iniciais serão sobrescritos durante o processamento da convolução.
    A imagem é dividida em blocos horizontais, cada um com uma altura de chunk_size. Se o número de linhas não for divisível pelo número de processos, o último bloco inclui todas as linhas restantes.
    Um pool de processos é criado com o número especificado de processos (num_processes).
    Uma lista de tarefas é criada, onde cada tarefa é uma tupla de argumentos a ser passada para a função calculate_convolution_block. Cada tupla representa a convolução a ser realizada em um bloco específico da imagem (definido por x_i e x_f).
    Para cada canal da imagem, starmap distribui essas tuplas de argumentos entre os processos do pool. Cada processo pega uma dessas tarefas e executa calculate_convolution_block com os argumentos fornecidos.
    Os resultados dos blocos convoluídos são concatenados verticalmente (vstack) para reconstruir a imagem final para cada canal.
    """

    image_height, image_width, image_channels = image.shape

    result = []
    new_image = np.empty(image.shape, dtype=np.uint8)

    # Divide the image into chunks and the rest of the division into the last chunk
    chunk_size = image_height // num_processes
    chunks = []
    for p in range(num_processes):
        chunks.append((p * chunk_size, (p + 1) * chunk_size if p != num_processes - 1 else image_height))

    # print(chunks)

    # Perform the convolution operation in parallel using map
    with (mp.Pool(num_processes) as pool):
        for c in range(image_channels):
            result.append(pool.starmap(calculate_convolution_block,
                                       [(image, x_i, 0, x_f, image_width, c, kernel) for x_i, x_f in chunks]))
            # concatenate the results
            # concatenate the results
            for i, chunk in enumerate(result):
                new_image[:, :, c] = np.vstack(chunk)
    return new_image


@jit(nopython=True, parallel=True)
def convolution_numba(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple processes
    Strategy #5 : using numba.jit and parallel=True decorator and prange for parallelization
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """

    """
    Função implementada utilizando a biblioteca Numba, que permite acelerar o código Python ao compilá-lo para código máquina usando just-in-time (JIT). 
    A função é decorada com @jit(nopython=True, parallel=True). Isso instrui o Numba a compilar a função em modo nopython, que é a configuração mais rápida, e a paralelizar os loops onde possível.
    Primeiramente, define-se o número de threads que o Numba deve utilizar para paralelizar a execução, obtém as dimensões da imagem original e inicializa um array de zeros com o mesmo formato para armazenar a imagem convolucionada.
    Na operação de convolução, temos que, nesta ordem, para cada canal da imagem, utiliza-se prange (parallel range) para indicar que os loops devem ser paralelizados. Isso significa que cada iteração desses loops pode ser executada em paralelo, o que acelera significativamente a computação em máquinas com múltiplos núcleos; loops aninhados iteram sobre cada elemento do kernel; calcula-se a posição do pixel da imagem a ser multiplicado pelo elemento correspondente do kernel; verifica se as coordenadas calculadas estão dentro dos limites da imagem. Se não estiverem, a iteração continua, ignorando os valores fora dos limites; o valor do pixel da imagem é multiplicado pelo valor correspondente do kernel e somado ao resultado na nova imagem.
    Finalmente, a função retorna a nova imagem convolucionada.
    """

    set_num_threads(num_processes)
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using numba.jit
    for c in range(image_channels):
        for x in prange(image_height):
            for y in prange(image_width):
                for j in range(kernel.shape[0]):
                    for k in range(kernel.shape[1]):
                        x_k = x - j + kernel.shape[0] // 2
                        y_k = y - k + kernel.shape[1] // 2
                        if x_k < 0 or y_k < 0 or x_k >= image_height or y_k >= image_width:
                            continue
                        new_image[x, y, c] += image[x_k, y_k, c] * kernel[j, k]
    return new_image


def convolution_thread(image: np.ndarray, kernel: np.ndarray, num_processes: int) -> np.ndarray:
    """
    Perform a convolution operation on the image using multiple threads
    Strategy #1: using the ThreadPoolExecutor and saving directly to the new image
    :param image: The image data
    :param kernel: The kernel to perform the convolution
    :param num_processes: The number of processes to use
    :return: The convoluted image
    """
    image_height, image_width, image_channels = image.shape

    # Initialize the new image
    new_image = np.zeros(image.shape, dtype=np.uint8)

    # Perform the convolution operation in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(num_processes) as executor:
        for c in range(image_channels):
            executor.map(calculate_convolution_padless,
                         [(image, x, y, c, kernel, new_image)
                          for x in range(image_height) for y in range(image_width)])
    return new_image