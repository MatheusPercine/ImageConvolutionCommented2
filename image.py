import numpy as np
import cv2
import array


class Image:

    def __init__(self, filepath: str):
        """
        Initialize the image
        :param filepath: The path to the image
        """
        self.data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  # Read the image in the original format
        if self.data is None:
            raise FileNotFoundError(f"File {filepath} not found")

        self.height = self.data.shape[0]
        self.width = self.data.shape[1]

        if len(self.data.shape) == 2:
            self.channels = 1
            self.data = self.data[:, :, np.newaxis]
        else:
            self.channels = self.data.shape[2]

        self.filepath = filepath
        self.name = filepath.split('/')[-1].split('.')[0]
        self.pitch = self.width  # Can be used to optimize the padding

    @classmethod
    def from_data(cls, data: np.ndarray):
        """
        Initialize the image from data
        :param data: The image data
        :return: The image
        """
        image = cls.__new__(cls)
        image.data = data
        image.height, image.width, image.channels = data.shape
        image.filepath = None
        image.name = "Image"
        image.pitch = image.width
        return image

    @classmethod
    def read_image_bin(cls, path: str, channels: int):
        """
        Read the image from a binary file
        :param path: The path to read the image
        :param channels: The number of channels in the image
        """
        data = []
        for i in range(channels):
            with open(f"{path}_{i}.bin", "rb") as f:
                # Reading the dimensions of the image
                height, width = array.array('i', f.read(8))
                data.append(np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width)))
        return cls.from_data(np.stack(data, axis=2))

    def padding(self, padding_size_x: int, padding_size_y: int):
        """
        Pad the image with zeros
        :param padding_size_x: The padding size in the x-direction
        :param padding_size_y: The padding size in the y-direction
        :return: The padded image
        """
        self.data = cv2.copyMakeBorder(self.data, padding_size_x, padding_size_x, padding_size_y, padding_size_y,
                                       cv2.BORDER_REFLECT)
        self.pitch = self.width + 2 * padding_size_x

    def save_image(self, output_path: str):
        """
        Save the image to a file
        :param output_path: The path to save the image
        """
        cv2.imwrite(output_path, self.data)

    def show(self):
        """
        Display the image
        """
        cv2.imshow(self.name, self.data)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image_txt(self, path: str):
        """
        Save the image to a text file
        :param path: The path to save the image
        """
        # Save file for each channel
        for i in range(self.channels):
            with open(f"{path}_{i}.txt", "w") as f:
                for row in self.data[:, :, i]:
                    # Saving Dimensions
                    f.write(f"{self.height} {self.width}\n")
                    f.write(" ".join(map(str, row)) + "\n")

    def save_image_bin(self, path: str):
        """
        Save the image to a binary file
        :param path: The path to save the image
        """
        for i in range(self.channels):
            with open(f"{path}_{i}.bin", "wb") as f:
                # Writing dimensions of the image
                f.write(array.array('i', [self.height, self.width]).tobytes())
                f.write(array.array('B', self.data[:, :, i].flatten()).tobytes())
                print("Type of data", self.data[:, :, i].dtype)
                print(f"Saved {path}_{i}.bin")


def folder_to_bin(folder_path: str, output_path: str):
    """
    Convert a folder of images to binary files
    :param folder_path: The path to the folder
    :param output_path: The path to save the binary files
    """
    import os
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith('.jpg'):
                continue
            img = Image(os.path.join(root, file))
            img.save_image_bin(os.path.join(output_path, file.split('.')[0]))


if __name__ == "__main__":
    # img = Image('./images/dog.jpg')
    # img.save_image_bin('./bin/dog')
    # img = Image.read_image_bin('./bin/dog', 1)
    # img.show()

    # Code to test C binary file
    # c_image = Image.read_image_bin('./test', 1)
    # c_image.save_image('./test.jpg')
    # c_image.show()
    folder_to_bin('./images', './bin')
