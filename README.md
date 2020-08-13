# Automate-Keras-Models

This project aims to automate the process of creating machine learning models based on neural networks using Keras framework.

## Dependencies

Python and the following dependencies must be installed to run this project:

```
pip install keras
```

## Run

Command to execute the script:

```
python Main.py
```

## Configuration

The `config.json` configuration file has the following structure:

```
    "modelInput": "(90,120,3)",
    "modelLayers": 
        ["Conv2d, 20, (7,9), (1,1), valid",
        "BatchNormalization",
        "Relu",
        "MaxPooling2D, (2,2)",
        
        "Conv2d, 50, (5,5), (1,1), valid",
        "BatchNormalization",
        "Relu",
        "MaxPooling2D, (2,2)",
        
        "Conv2d, 70, (4,5), (1,1), valid",
        "BatchNormalization",
        "Relu",
        "MaxPooling2D, (2,2)",
        
        "Flatten",
        
        "Dense, 500",
        "BatchNormalization",
        "Relu",
        "Dropout, 0.5",
        
        "Dense, 4",
        "BatchNormalization",
        "Softmax"]
}
```

- modelInput: input shape of the model.
- modelLayers: list of all layers of the model.

## Supported Layers and How to Use in the Config File

### Convolutional Layers

- **Conv2d**: "Conv2d, number of filters, filters size, stride size, padding type"

Example: "Conv2d, 20, (3,3), (1,1), valid".

### Pooling Layers

- **MaxPooling2D**: "MaxPooling2D, pool size"

Example: "MaxPooling2D, (2,2)".

- **AvgPooling2D**: "AvgPooling2D, pool size"

Example: "AvgPooling2D, (2,2)".

### Dense Layers

- **Dense**: "Dense, number of neurons"

Example: "Dense, 512".

### Activation Layers

- **Relu**: "Relu"

- **Softmax**: "Softmax"

### Normalization Layers

- **BacthNormalization**: "BatchNormalization"

### Reshaping Layers

- **Flatten**: "Flatten"

### Regularization Layers

- **Dropout**: "Dropout"

## Future Improvements

- Increase the number of supported layers.
- Support functional models.
- Automate compiling and training.
