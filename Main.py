from AutoNeuralNetwork import AutoNeuralNetwork
from JsonHandler import JsonHandler
from Data import Data

config = JsonHandler.read_json("config.json")

modelInput = config["modelInput"]
modelLayers = config["modelLayers"]
dataFile = config["dataFile"]

data = Data(dataFile)

neuralNetwork = AutoNeuralNetwork("Teste")
neuralNetwork.build_model(modelInput, modelLayers)
neuralNetwork.compile_model()
neuralNetwork.train_model("KFold", data, 10, 32)