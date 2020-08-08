from AutoNeuralNetwork import AutoNeuralNetwork
from JsonHandler import JsonHandler

config = JsonHandler.read_json("config.json")

modelInput = config["modelInput"]
modelLayers = config["modelLayers"]

model = AutoNeuralNetwork("Teste")
model.build_model(modelInput, modelLayers)