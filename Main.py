from AutoNeuralNetwork import AutoNeuralNetwork
from JsonHandler import JsonHandler
from Data import Data

def main():

    config = JsonHandler.read_json("config.json")

    if "train" in config:
        dataFile = config["train"]["data"]["fileName"]
        labelType = config["train"]["data"]["labelType"]

        modelName = config["train"]["model"]["name"]
        modelInput = config["train"]["model"]["input"]
        modelLayers = config["train"]["model"]["layers"]

        loss = config["train"]["compile"]["loss"]
        optimizer = config["train"]["compile"]["optimizer"]
        metrics = config["train"]["compile"]["metrics"]

        evaluation = config["train"]["fit"]["evaluation"]
        epochs = config["train"]["fit"]["epochs"]
        batchSize = config["train"]["fit"]["batchSize"]

        data = Data(dataFile, labelType)

        neuralNetwork = AutoNeuralNetwork(modelName)
        neuralNetwork.build_model(modelInput, modelLayers)
        neuralNetwork.compile_model(loss, optimizer, metrics)
        neuralNetwork.train_model(data, evaluation, epochs, batchSize)
        neuralNetwork.save_model()

    if "test" in config:
        dataFile = config["test"]["data"]["fileName"]
        labelType = config["test"]["data"]["labelType"]

        modelName = config["train"]["model"]["name"]

        data = Data(dataFile, labelType)

        neuralNetwork = AutoNeuralNetwork(modelName)
        neuralNetwork.load_model()
        neuralNetwork.test_model(data)

main()