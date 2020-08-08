from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization
from keras import layers
from keras import models
from LayerConfigHandler import LayerConfigHandler

class AutoNeuralNetwork(object):

    def __init__(self, name):
        self.name = name

    def build_model(self, inputShape, allLayers):

        # Building a sequential model
        self.model = models.Sequential()

        isFirstLayer = True

        for layer in allLayers:
            if isFirstLayer:
                newModelLayer = self.create_layer(layer, inputShape)
                isFirstLayer = False
            else:
                newModelLayer = self.create_layer(layer)
            
            self.model.add(newModelLayer)

        self.summary()

    def summary(self):
        print(self.model.summary())

    def create_layer(self, layer, inputShape=None):
        layerType = layer.split(",")[0]

        ## Convolutional Layers
        if layerType == "Conv2d":
            return self.create_conv2d_layer(layer, inputShape)

        ## Pooling Layers
        if layerType == "MaxPooling2D":
            return self.create_maxpooling2d_layer(layer)
        if layerType == "AveragePooling2D":
            return self.create_averagepooling2d_layer(layer)
        
        ## Dense layer
        if layerType == "Dense":
            return self.create_dense_layer(layer, inputShape)

        ## Activation Layers
        if layerType == "Relu":
            return self.create_relu_layer()
        if layerType == "Softmax":
            return self.create_softmax_layer()

        ## Normalization Layers
        if layerType == "BatchNormalization":
            return self.create_batch_normalization_layer()

        ## Reshaping Layers
        if layerType == "Flatten":
            return self.create_flatten_layer()

        ## Regularization Layers
        if layerType == "Dropout":
            return self.create_dropout_layer(layer)


    ## Creating Convolutional Layers
    
    def create_conv2d_layer(self, layerConfig, inputShape=None):
        filters, kernelSize, stride, padding = LayerConfigHandler.read_conv2d_layer_config(layerConfig)

        if inputShape != None:
            return Conv2D(filters=filters, kernel_size=kernelSize, strides=stride, 
                padding=padding, input_shape=eval(inputShape))
        return Conv2D(filters=filters, kernel_size=kernelSize, strides=stride, padding=padding)


    ## Creating Pooling Layers

    def create_maxpooling2d_layer(self, layer):
        poolSize = LayerConfigHandler.read_pooling2d_layer_config(layer)
        return MaxPooling2D(pool_size=poolSize)
    
    def create_averagepooling2d_layer(self, layer):
        poolSize = LayerConfigHandler.read_pooling2d_layer_config(layer)
        return AveragePooling2D(pool_size=poolSize)


    ## Creating Dense Layer

    def create_dense_layer(self, layerConfig, inputShape=None):
        neurons = LayerConfigHandler.read_dense_layer_config(layerConfig)

        if inputShape != None:
            return Dense(units=neurons, input_shape=eval(inputShape))
        return Dense(units=neurons)


    ## Creating Normalization Layers

    def create_batch_normalization_layer(self):
        return BatchNormalization()


    ## Creating Activation Layers

    def create_relu_layer(self):
        return Activation("relu")

    def create_softmax_layer(self):
        return Activation("softmax")


    ## Creating Reshaping Layers

    def create_flatten_layer(self):
        return Flatten()


    ## Creating Regularization Layers

    def create_dropout_layer(self, layer):
        dropoutRate = LayerConfigHandler.read_dropout_layer_config(layer)
        return Dropout(rate=dropoutRate)