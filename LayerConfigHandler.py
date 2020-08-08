class LayerConfigHandler(object):

    @staticmethod
    def read_dense_layer_config(layerConfig):
        layerConfigSplit = layerConfig.split(", ")
        layerNeurons = int(layerConfigSplit[1])

        return layerNeurons

    @staticmethod
    def read_conv2d_layer_config(layerConfig):
        layerConfigSplit = layerConfig.split(", ")
        layerFilters = int(layerConfigSplit[1])
        layerKernelSize = eval(layerConfigSplit[2])
        layerStride = eval(layerConfigSplit[3])
        layerPadding = str(layerConfigSplit[4])

        return layerFilters, layerKernelSize, layerStride, layerPadding

    @staticmethod
    def read_pooling2d_layer_config(layerConfig):
        layerConfigSplit = layerConfig.split(", ")
        layerPoolSize = eval(layerConfigSplit[1])

        return layerPoolSize

    @staticmethod
    def read_dropout_layer_config(layerConfig):
        layerConfigSplit = layerConfig.split(", ")
        layerDropoutRate = float(layerConfigSplit[1])

        return layerDropoutRate