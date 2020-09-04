from sklearn.model_selection import KFold, StratifiedKFold
from random import shuffle
import numpy as np

class Data():

    def __init__(self, dataFile, labelType):
        self.input = []
        self.output = []
        self.process_data(dataFile, labelType)

    def process_data(self, dataFile, labelType):
        examples = np.load(dataFile)
        shuffle(examples)

        exampleShape = (examples[0][0].shape[1], examples[0][0].shape[0], 1)

        for example in examples:
            self.input.append(example[0].reshape(exampleShape))
            if labelType == "OneHot":
                example[1] = self.encode_to_one_hot(example[1])
            self.output.append(example[1])

        self.input = np.array(self.input)
        self.output = np.array(self.output)
    
    def encode_to_one_hot(self, label):
        if label == 0:
            return np.array([1,0,0,0])
        if label == 1:
            return np.array([0,1,0,0])
        if label == 2:
            return np.array([0,0,1,0])
        if label == 3:
            return np.array([0,0,0,1])

    def split_data_holdout(self, trainPercentage=0.8):
        examplesLength = self.input.shape[0]
        trainLength = int(examplesLength * trainPercentage)

        trainInput = self.input[:trainLength]
        trainOutput = self.output[:trainLength]
        testInput = self.input[trainLength:]
        testOutput = self.output[trainLength:]

        return trainInput, testInput, trainOutput, testOutput

    def create_k_folds(self, mode, foldsNumber=5):
        if mode == "KFold":
            return KFold(n_splits=foldsNumber)
        if mode == "StratifiedKFold":
            return StratifiedKFold(n_splits=foldsNumber)