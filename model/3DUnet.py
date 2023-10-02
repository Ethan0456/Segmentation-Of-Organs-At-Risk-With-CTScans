import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model

class UNet3D:
    def __init__(self):
        self.input_shape = (64,64,64,3)
        self.num_classes = 6
        self.model = self.build_model()
        self.kernel_size = (3,3,3)

    def oneUnit(self, num_filters, kernel_size, prev, activation='relu', padding='same'):
        conv_layer = Conv3D(num_filters, kernel_size, padding=padding)(prev)
        bn_layer = BatchNormalization()(conv_layer)
        activation_layer = Activation(activation)(bn_layer)
        return activation_layer

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        ##################### ENCODER ###################
        #################################################
        # Channels : 3 to 32
        conv_bn_relu_1d = self.oneUnit(32, self.kernel_size, inputs)
        # Channels : 32 to 64
        conv_bn_relu_2d = self.oneUnit(64, self.kernel_size, conv_bn_relu_1d)
        # Max Pooling
        pool1d = MaxPooling3D(pool_size=(2, 2, 2))(conv_bn_relu_2d)
        #################################################


        #################################################
        # Channels : 64 to 64
        conv_bn_relu_3d = self.oneUnit(64, self.kernel_size, pool1d)
        # Channels : 64 to 128
        conv_bn_relu_4d = self.oneUnit(128, self.kernel_size, conv_bn_relu_3d)
        # Max Pooling
        pool2d = MaxPooling3D(pool_size=(2,2,2))(conv_bn_relu_4d)
        #################################################


        #################################################
        # Channels : 128 to 128
        conv_bn_relu_5d = self.oneUnit(128, self.kernel_size, pool2d)
        # Channels : 128 to 256
        conv_bn_relu_6d = self.oneUnit(256, self.kernel_size, conv_bn_relu_5d)
        # Max Pooling
        pool3d = MaxPooling3D(pool_size=(2,2,2))(conv_bn_relu_6d)
        #################################################


        ##################### DECODER ###################
        #################################################
        # Channels : 256 to 256
        conv_bn_relu_last1 = self.oneUnit(256, self.kernel_size, pool3d)
        # Channels : 256 to 512
        conv_bn_relu_last2 = self.oneUnit(512, self.kernel_size, conv_bn_relu_last1)
        # UpSampling
        up3 = UpSampling3D((2,2,2))(conv_bn_relu_last2)
        #################################################
        # Merge UpSampling with conv
        concat3 = concatenate([up3, conv_bn_relu_6d], axis=-1)
        #################################################


        #################################################
        # Channels : 256+512 to 256
        conv_bn_relu_6u = self.oneUnit(256, self.kernel_size, concat3)
        # Channels : 256 to 256
        conv_bn_relu_5u = self.oneUnit(256, self.kernel_size, conv_bn_relu_6u)
        # UpSampling
        up2 = UpSampling3D((2,2,2))(conv_bn_relu_5u)
        #################################################
        # Merge UpSampling with conv
        concat2 = concatenate([up2, conv_bn_relu_4d], axis=-1)
        #################################################


        #################################################
        # Channels : 128+256 to 128
        conv_bn_relu_4u = self.oneUnit(128, self.kernel_size, concat2)
        # Channels : 128 to
        conv_bn_relu_3u = self.oneUnit(128, self.kernel_size, conv_bn_relu_4u)
        # UpSampling
        up1 = UpSampling3D((2,2,2))(conv_bn_relu_3u)
        #################################################
        # Merge UpSampling with conv
        concat1 = concatenate([up1, conv_bn_relu_2d], axis=-1)
        #################################################


        #################################################
        # Channels : 64+128 to 64
        conv_bn_relu_2u = self.oneUnit(64, self.kernel_size, concat1)
        # Channels : 64 to 64
        conv_bn_relu_1u = self.oneUnit(64, self.kernel_size, conv_bn_relu_2u)
        #################################################
        output = Conv3D(self.num_classes, self.kernel_size, activation='softmax', padding='same')(conv_bn_relu_1u)
        #################################################

        model = Model(inputs=inputs, outputs=output)
