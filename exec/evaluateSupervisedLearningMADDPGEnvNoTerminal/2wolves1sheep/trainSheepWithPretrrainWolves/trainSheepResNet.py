import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..', '..', '..'))
import time
import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt
import itertools as it
import pathos.multiprocessing as mp

from src.constrainedChasingEscapingEnv.envMADDPG import *
from src.constrainedChasingEscapingEnv.envNoPhysics import TransitWithInterpolation, Reset, IsTerminal, StayInBoundaryByReflectVelocity
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete, RewardFunctionWithWall
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory
from visualize.visualizeMultiAgent import *


class TrainModelForConditions:
    def __init__(self, trainIntervelIndexes, trainStepsIntervel, trainData, NNModel, getTrain, getModelSavePath):
        self.trainIntervelIndexes = trainIntervelIndexes
        self.trainStepsIntervel = trainStepsIntervel
        self.trainData = trainData
        self.NNModel = NNModel
        self.getTrain = getTrain
        self.getModelSavePath = getModelSavePath

    def __call__(self, parameters):
        print(parameters)
        miniBatchSize = parameters['miniBatchSize']
        learningRate = parameters['learningRate']

        model = self.NNModel
        train = self.getTrain(miniBatchSize, learningRate)
        parameters.update({'trainSteps': 0})
        modelSavePath = self.getModelSavePath(parameters)
        saveVariables(model, modelSavePath)

        for trainIntervelIndex in self.trainIntervelIndexes:
            parameters.update({'trainSteps': trainIntervelIndex * self.trainStepsIntervel})
            modelSavePath = self.getModelSavePath(parameters)
            if not os.path.isfile(modelSavePath + '.index'):
                trainedModel = train(model, self.trainData)
                saveVariables(trainedModel, modelSavePath)
            else:
                trainedModel = restoreVariables(model, modelSavePath)
            model = trainedModel


def trainOneCondition(manipulatedVariables):
    depth = int(manipulatedVariables['depth'])
    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', 'MADDPG2wolves1sheep', 'trainSheepWithPretrrainWolves', 'trajectories')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 50
    dataSetNumSimulations = 250
    sheepId = 0
    agentId = sheepId
    dataSetFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    numSheeps = 1
    numWolves = 2
    numOfAgent = numSheeps + numWolves
    numBlocks = 0
    numEntities = numOfAgent + numBlocks

    # accumulate rewards for trajectories
    decay = 1
    accumulateRewards = AccumulateRewards(decay)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 0.5
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    predatorPowerRatio = 0.5
    wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))

    numActionSpace = len(sheepActionSpace)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(sheepActionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, sheepId)

    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    # print(loadedTrajectories[1])

    visualize = False
    if visualize:

        sheepSize = 0.05
        wolfSize = 0.075
        blockSize = 0.2
        entitiesSizeList = [sheepSize] * numSheeps + [wolfSize] * numWolves + [blockSize] * numBlocks

        sheepColor = [0.35, 0.85, 0.35]
        wolfColor = [0.85, 0.35, 0.35]
        blockColor = [0.25, 0.25, 0.25]
        entitiesColorList = [sheepColor] * numSheeps + [wolfColor] * numWolves + [blockColor] * numBlocks
        # for i in range(len(loadedTrajectories)):
        for i in range(0, 3):
            print(np.array(loadedTrajectories[i])[:, 0])
            render = Render(entitiesSizeList, entitiesColorList, numOfAgent, getPosFromAgentState)
            render(loadedTrajectories[i])

    filterState = lambda timeStep: (timeStep[0][:numOfAgent], timeStep[1], timeStep[2], timeStep[3])
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    print(len(trajectories))

    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))

    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]
    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 4 * numEntities
    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]

    generateModel = GenerateModel(numStateSpace, numActionSpace, regularizationFactor)

    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    sheepNNModel = generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

    initTimeStep = 0
    valueIndex = 3
    trainDataMeanAccumulatedReward = np.mean([tra[initTimeStep][valueIndex] for tra in valuedTrajectories])
    print(trainDataMeanAccumulatedReward)

    # function to train NN model
    terminalThreshold = 1e-10
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)
    terminalController = lambda evalDict, numSteps: False
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)
    reportInterval = 100
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda batchSize, learningRate: Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController, trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', 'MADDPG2wolves1sheep', 'trainSheepWithPretrrainWolves', 'trainedResNNModels')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)
    NNModelSaveExtension = ''
    getNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, NNModelFixedParameters)

    # function to train models
    numOfTrainStepsIntervel = 6
    trainIntervelIndexes = list(range(numOfTrainStepsIntervel))
    trainModelForConditions = TrainModelForConditions(trainIntervelIndexes, trainStepsIntervel, trainData, sheepNNModel, getTrainNN, getNNModelSavePath)
    trainModelForConditions(manipulatedVariables)


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['depth'] = [9]
    manipulatedVariables['miniBatchSize'] = [256]
    manipulatedVariables['learningRate'] = [1e-4]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    trainPool = mp.Pool(numCpuToUse)

    startTime = time.time()
    trainOneCondition(parametersAllCondtion[0])
    #trainPool.map(trainOneCondition, parametersAllCondtion)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
