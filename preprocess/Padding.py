import numpy as np


class Padding:
    def __init__(self, input_shape=(512,512,75), desired_depth=128):
        self.input_shape = input_shape
        self.desired_depth = desired_depth

    def paddData(self, input):
        # Define your input volume dimensions
        input_shape = self.input_shape
        desired_depth = self.desired_depth

        # Calculate the amount of padding needed along the depth axis on each side
        depth_padding = desired_depth - input_shape[2]
        left_padding = depth_padding // 2
        right_padding = depth_padding - left_padding

        # Pad the input volume along the depth axis with zeros on both sides
        padded_volume = np.pad(input , ((0, 0), (0, 0), (left_padding, right_padding)), mode='constant')

        return padded_volume