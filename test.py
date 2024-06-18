import os
import unittest
import cv2
import numpy as np
import convolution
import kernel
import image
import time
import pandas as pd
from subprocess import call, Popen, PIPE


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.uint8)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

    return result


def create_bw_image(width, height, path=None):
    data = np.random.randint(0, 255, (height, width, 1), np.uint8)
    # Convert to Gray
    img = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
    if path:
        cv2.imwrite(path, img[:, :, 0])
    print("Image created with shape", img.shape[:2])
    return data


def create_color_image(width, height, path=None):
    data = np.random.randint(0, 255, (height, width, 3), np.uint8)
    img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)  # Convert to RGB
    if path:
        cv2.imwrite(path, img)
    print("Image created with shape", img.shape[:2])
    return data


def create_test_images(width: list, height: list, path: str, is_bw: bool):
    images = []
    for w, h in zip(width, height):
        path_img = f"{path}_{is_bw}_{w}_{h}.jpg"
        if is_bw:
            img = create_bw_image(w, h, path_img)
        else:
            img = create_color_image(w, h, path_img)
        images.append(img)
    return images


class Corretude(unittest.TestCase):

    def test_concurrent_color(self):
        width = [333, 500, 1024]
        height = [333, 500, 768]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/corretude/corretude_False_{w}_{h}.jpg").data)

        print('-'*50)
        print("\n Começando testes de corretude em Python com imagens coloridas")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for img in images:
            print("\nTestando imagem de tamanho", img.shape)

            for thread in threads:
                print("\nTestando com thread = ", thread)

                sequential = convolution.convolution(img, kernel.edge_detection_kernel(3))
                self.assertEqual(sequential.all(),
                                 convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -pool OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -block OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -thread OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -numba OK")

        print("\nTestes finalizados com sucesso! Todos os testes de corretude de Python em Color passaram.")

    def test_concurrent_bw(self):
        width = [333, 500, 1024]
        height = [333, 500, 768]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/corretude/corretude_True_{w}_{h}.jpg").data)

        print('-'*50)
        print("\nComeçando testes de corretude")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])

        for img in images:
            print("\nTestando imagem de tamanho", img.shape)

            for thread in threads:
                print("\nTestando com thread = ", thread)

                sequential = convolution.convolution(img, kernel.edge_detection_kernel(3))
                self.assertEqual(sequential.all(),
                                 convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -pool OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -block OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -thread OK")

                self.assertEqual(sequential.all(),
                                 convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread).all())
                print(" -numba OK")

        print("\nTestes finalizados com sucesso! Todos os testes de corretude de Python em BW passaram.")

    def test_concurrent_C(self):
        # Use the C code to test the performance

        # Compile the C code
        call(["gcc", "-o", "main", "./C/main.c", "./C/convolution.c", "-lpthread", "-Wall"])

        # Create the binary files
        image.folder_to_bin("./images/corretude", "./bin/corr")

        bin_path = []
        for root, _, files in os.walk("./bin/corr"):
            for file in files:
                if file.endswith('.bin'):
                    bin_path.append(file)
            break

        print('-'*50)
        print("\nComeçando testes de corretude")
        print("Testando binários - ", [file for file in bin_path])

        for file in bin_path:
            for thread in [1, 2, 4, 8]:
                # sequencial result
                print("\nRunning C code for", file, "with", thread, "threads")
                process_seq = Popen(["./main", f"./bin/corr/{file}", "./bin/kernel/edge_detection_kernel.bin", "1",
                                     "out_seq_0.bin"])
                process = Popen(["./main", f"./bin/corr/{file}", "./bin/kernel/edge_detection_kernel.bin", str(thread),
                                 "out_0.bin"])

                # Wait for the process to finish
                process_seq.wait()
                process_seq.kill()
                process.wait()
                process.kill()

                # Assert the results
                self.assertEqual(image.Image.read_image_bin("./out_seq", 1).data.all(),
                                 image.Image.read_image_bin("./out", 1).data.all())

                print(" -OK")

        print("\nTestes finalizados com sucesso! Todos os testes de corretude de C passaram.")

    def test_comparison_python_C(self):
        # Compare the results of the python code with the C code
        # Compile the C code
        call(["gcc", "-o", "main", "./C/main.c", "./C/convolution.c", "-lpthread", "-Wall"])

        # Create the binary files
        image.folder_to_bin("./images/corretude", "./bin/corr")

        bin_path = []
        for root, _, files in os.walk("./bin/corr"):
            for file in files:
                if file.endswith('.bin'):
                    bin_path.append(file)
            break

        print('-'*50)
        print("\nComeçando testes de comparação")
        print("Testando binários - ", [file for file in bin_path])

        for file in bin_path:
            print("\nRunning Python code for", file)

            file_python = file[:-6]  # Remove the extension and the channel number
            img = image.Image.read_image_bin(f"./bin/corr/{file_python}", 1)
            sequential = convolution.convolution(img.data, kernel.edge_detection_kernel(3))

            print("Running C code for", file)
            process = Popen(
                ["./main", f"./bin/corr/{file}", "./bin/kernel/edge_detection_kernel.bin", "1", "out.bin"],
                stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            process.wait()  # Wait for the process to finish

            print("Output:", stdout.decode())
            # Assert the results

            self.assertEqual(sequential.all(), image.Image.read_image_bin("./out", 1).data.all())
            print(" -OK")

        print("\nTestes finalizados com sucesso! A saída em C foi igual a saída em Python.")


class Performance(unittest.TestCase):

    def test_performance_bw(self):
        width = [500, 1000, 5000]
        height = [500, 1000, 5000]
        threads = [1, 2, 4, 8]
        images = []
        for w, h in zip(width, height):
            images.append(image.Image(f"./images/performance/performance_True_{w}_{h}.jpg").data)

        result_list = []

        print('-'*50)
        print("\nComeçando testes de performance em Python")
        print("Testando imagens de tamanho", [(img.shape[0], img.shape[1]) for img in images])
        for i in range(5):
            print("# Iteração - ", i)
            for img in images:
                print("\nTestando imagem de tamanho", img.shape)

                start = time.time()
                convolution.convolution(img, kernel.edge_detection_kernel(3))
                total_seq = time.time() - start
                print("Sequential:", total_seq)

                for thread in threads:
                    print("\nTestando com thread = ", thread)

                    start = time.time()
                    convolution.convolution_pool(img, kernel.edge_detection_kernel(3), thread)
                    total_pool = time.time() - start
                    print("Pool:", total_pool)

                    start = time.time()
                    convolution.convolution_block(img, kernel.edge_detection_kernel(3), thread)
                    total_block = time.time() - start
                    print("Block:", total_block)

                    # start = time.time()
                    # convolution.convolution_thread(img, kernel.edge_detection_kernel(3), thread)
                    # total_thread = time.time() - start
                    # print("Thread:", total_thread)

                    start = time.time()
                    convolution.convolution_numba(img, kernel.edge_detection_kernel(3), thread)
                    total_numba = time.time() - start
                    print("Numba:", total_numba)

                    result_list.append([total_seq, total_pool, total_block, total_numba,
                                        thread, img.shape])

        results = pd.DataFrame(result_list, columns=["Sequential", "Pool", "Block", "Numba",
                                                     "Thread Number", "Image Size"])

        results.to_csv("results.csv")

        print("\nTeste de performance em Python finalizado com sucesso!")

    def test_performance_C(self):
        # Use the C code to test the performance

        # Compile the C code
        call(["gcc", "-o", "main", "./C/main.c", "./C/convolution.c", "-lpthread", "-Wall"])

        # Create the binary files
        image.folder_to_bin("./images/performance", "./bin/perf")

        bin_path = []
        for root, _, files in os.walk("./bin/perf"):
            for file in files:
                if file.endswith('.bin'):
                    bin_path.append(file)
            break  # Only the first level

        result_list = []

        print('-'*50)
        print("\nComeçando testes de performance em C")
        print("Testando binários - ", [file for file in bin_path])

        # Run the C code for each image and thread number five times
        for i in range(5):
            print("# Iteração - ", i)
            for file in bin_path:
                for thread in [1, 2, 4, 8]:
                    print("\nRunning C code for", file, "with", thread, "threads")
                    process = Popen(
                        ["./main", f"./bin/perf/{file}", "./bin/kernel/edge_detection_kernel.bin", str(thread),
                         "out.bin"], stdout=PIPE, stderr=PIPE)

                    stdout, stderr = process.communicate()
                    process.wait()  # Wait for the process to finish

                    print("Output:", stdout.decode())
                    # Save the results
                    result_list.append([float(stdout.decode()), thread, file])

        results = pd.DataFrame(result_list, columns=["C", "Thread Number", "Image Size"])
        results.to_csv("results_c.csv")

        print("\nTeste de performance em C finalizado com sucesso!")


if __name__ == '__main__':
    create_test_images([333, 500, 1024], [333, 500, 768], "./images/corretude/corretude", True)
    create_test_images([333, 500, 1024], [333, 500, 768], "./images/corretude/corretude", False)
    create_test_images([500, 1000, 5000], [500, 1000, 5000], "./images/performance/performance", True)
    unittest.main()
