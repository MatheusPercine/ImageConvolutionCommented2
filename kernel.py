# Class for the kernel for any convolutional operation
import numpy as np
import array


class Kernel2D:
    def __init__(self, kernel: np.ndarray):
        self.kernel = kernel.astype(np.float32)
        self.height, self.width = kernel.shape
        self.center = (self.height // 2, self.width // 2)  # Floor division to get the center of the kernel
    
    @classmethod
    def read_kernel_bin(cls, path: str):
        """
        Read the kernel from a binary file
        :param path: The path to read the kernel
        """
        with open(path, "rb") as f:
            # Reading the dimensions of the kernel
            height, width = array.array('i', f.read(8))
            kernel = np.frombuffer(f.read(), dtype=np.float32).reshape((height, width))
        return cls(kernel)
    
    def save_kernel_txt(self, path: str):
        """
        Save the kernel to a text file
        :param path: The path to save the kernel
        """
        # Adding the dimensions of the kernel to the file
        with open(path, "w") as f:
            f.write(f"{self.height} {self.width}\n")
            for row in self.kernel:
                f.write(" ".join(map(str, row)) + "\n")
    
    def save_kernel_bin(self, path: str):
        """
        Save the kernel to a binary file for use in C code in float format
        :param path: The path to save the kernel
        """
        with open(path, "wb") as f:
            # Writing dimensions of the kernel
            f.write(array.array('i', [self.height, self.width]).tobytes())
            f.write(array.array('f', self.kernel.flatten()).tobytes())


def gaussian_kernel(size: int, sigma: float):
    """
    Create a Gaussian kernel
    :param size: The size of the kernel
    :param sigma: The standard deviation of the Gaussian distribution
    :return: The Gaussian kernel
    """
    # Create a 1D array of size 'size' with values from -(size - 1) / 2 to (size - 1) / 2
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))  # Apply the Gaussian function
    kernel = np.outer(gauss, gauss)  # Create a 2D Gaussian kernel
    return kernel / kernel.sum()  # Normalize the kernel


def edge_detection_kernel(size: int):
    """
    Create an edge detection kernel
    :param size: The size of the kernel. Must be an odd number.
    :return: The edge detection kernel
    """
    if size % 2 == 0:
        raise ValueError("Size of the kernel must be an odd number")

    # Create a kernel filled with -1
    kernel = np.full((size, size), -1)

    # Set the center of the kernel to a value such that the sum of all elements in the kernel is 1
    center = size // 2
    kernel[center, center] = size * size

    return kernel


# gaussian_kernel = Kernel2D(np.array([[1, 4, 6, 4, 1],
#                                      [4, 16, 24, 16, 4],
#                                      [6, 24, 36, 24, 6],
#                                      [4, 16, 24, 16, 4],
#                                      [1, 4, 6, 4, 1]]) / 256)

sharpen_kernel = Kernel2D(np.array([[0, -1, 0],
                                    [-1, 5, -1],
                                    [0, -1, 0]]))

# edge_detection_kernel = Kernel2D(np.array([[-1, -1, -1],
#                                            [-1, 9, -1],
#                                            [-1, -1, -1]]))

emboss_kernel = Kernel2D(np.array([[-1, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 1]]))

# emboss_kernel = Kernel2D(np.array([[-1, 0, 0, 0, 0],
#                                   [0, -1, 0, 0, 0],
#                                   [0, 0, 0, 0, 0],
#                                   [0, 0, 0, 1, 0],
#                                   [0, 0, 0, 0, 1]]))

identity_kernel = Kernel2D(np.array([[0, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]))

if __name__ == "__main__":
    # Save the kernels to binary files
    identity_kernel.save_kernel_bin("./bin/kernel/identity_kernel.bin")
    sharpen_kernel.save_kernel_bin("./bin/kernel/sharpen_kernel.bin")
    Kernel2D(gaussian_kernel(3, 1)).save_kernel_bin("./bin/kernel/gaussian_kernel.bin")
    Kernel2D(edge_detection_kernel(3)).save_kernel_bin("./bin/kernel/edge_detection_kernel.bin")
