import numpy as np

from basic_layers import Layer
from initializer import Initializer
from parameter import Parameter

from typing import Union


class Conv2d(Layer):
    """todo change to fft implementation"""
    def __init__(self, in_channels: int, out_channels: int, filter_size: Union[tuple, int],
                 stride: Union[tuple, int] = 1, padding=0):
        """

        :param in_channels: 3, if inputting image
        :param out_channels: number of filters
        :param filter_size: width/height of a square filter
        :param stride:
        :param padding:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(filter_size, int):
            self.filter_size = filter_size, filter_size

        if isinstance(stride, int):
            self.stride_size = stride, stride

        self.padding = padding

        self.X = None

        self.filters = Parameter()  # 4d array
        self.b = Parameter()

    def initialize_params(self, initializer: Initializer):
        if self.initializer is not None:
            initializer = self.initializer

        self.filters.value = initializer.initialize_array([self.out_channels, self.in_channels, *self.filter_size])

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.X = X

        # transform images into rows of patches
        X = im2col(X, self.filter_size, self.stride_size, self.padding)

        # transform filters into flattened columns
        filters = self.filters.value
        filters = filters.reshape((self.out_channels, -1))  # width and height in the same dimension

        # applying multiple filters to all patches at once
        X2 = X @ filters.T  # (channel*width*height, filters)
        b = self.b.value
        X3 = X2 + b

        oup_height, oup_width = calculate_output_size(X, self.filter_size, self.stride_size, self.padding)

        return X3.reshape((self.out_channels, oup_height, oup_width))

    def backward(self, dX: np.ndarray) -> np.ndarray:
        pass


class Conv1D(Layer):
    pass


class MaxPooling2D(Layer):
    pass


class AvgPooling2D(Layer):
    pass


def calculate_output_size(image: np.array, filter_size: tuple, stride: tuple, padding: int) -> tuple[int, int]:

    batch_size, num_channels, image_height, image_width = image.shape
    filter_height, filter_width = filter_size
    stride_height, stride_width = stride

    # output (feature map) size -> how many times dot product happens
    oup_width = (image_width + padding - filter_width) // stride_width + 1  # +1 -> placeholder for the last conv
    oup_height = (image_height + padding - filter_height) // stride_height + 1

    return oup_height, oup_width

def im2col(image: np.array, filter_size: tuple, stride: tuple, padding: int) -> np.ndarray:
    """
    Splits a batch of  images (or any 2d numpy array) into flattened patches


    :param image: input image shape with size (batch, channel, height, width)
    :param filter_size:
    :param padding: padding to be applied in both the left/right and top/bottom
    :param stride:
    :return:
    """
    # X -> (batch, channel, height, width)
    # oup -> (batch*patch, channel*patch_size)

    batch_size, num_channels, image_height, image_width = image.shape
    filter_height, filter_width = filter_size
    stride_height, stride_width = stride

    oup_height, oup_width = calculate_output_size(image, filter_size, stride, padding)

    image = np.pad(image, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')  # apply padding to height and width dims
    col = np.zeros((batch_size, num_channels, filter_height, filter_width, oup_height, oup_width))

    # get 1 element per patch at each iteration
    for y in range(filter_height):
        y_max = y + stride_height*oup_height
        for x in range(filter_width):
            x_max = x + stride_width*oup_width
            col[:, :, y, x, :, :] = image[:, :, y_max:stride_height, x:x_max:stride_width]


    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(batch_size*oup_height*oup_width, -1)  # each row = 1 flattened patch
    return col
