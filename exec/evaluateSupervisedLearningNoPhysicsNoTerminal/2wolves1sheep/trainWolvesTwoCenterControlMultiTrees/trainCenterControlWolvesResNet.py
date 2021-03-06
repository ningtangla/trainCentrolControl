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

from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysics, Reset, IsTerminal, StayInBoundaryByReflectVelocity
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
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction


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
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainWolvesTwoCenterControlMultiTrees', 'trajectories')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 50
    dataSetNumSimulations = 500
    killzoneRadius = 50
    agentId = 1
    wolvesId = 1
    dataSetFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    # accumulate rewards for trajectories
    numOfAgent = 3

    sheepId = 0
    wolvesId = 1

    wolfOneId = 1
    wolfTwoId = 2
    xPosIndex = [0, 1]
    xBoundary = [0, 600]
    yBoundary = [0, 600]

    getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
    getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)

    reset = Reset(xBoundary, yBoundary, numOfAgent)

    isTerminalOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
    isTerminalTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)
    playIsTerminal = lambda state: isTerminalOne(state) or isTerminalTwo(state)

    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = TransiteForNoPhysics(stayInBoundaryByReflectVelocity)

    playAliveBonus = -1 / dataSetMaxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius

    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories

    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 12
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    predatorPowerRatio = 8

    actionSpaceOne = [(10, 0), (-10, 0)]
    wolfActionOneSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))

    actionSpaceTwo = [(10, 0), (-10, 0)]
    wolfActionTwoSpace = list(map(tuple, np.array(actionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))

    numActionSpace = len(wolvesActionSpace)

    actionIndex = 1
    actionToOneHot = ActionToOneHot(wolvesActionSpace)
    getTerminalActionFromTrajectory = lambda trajectory: trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)
    processTrajectoryForNN = ProcessTrajectoryForPolicyValueNet(actionToOneHot, wolvesId)

    preProcessTrajectories = PreProcessTrajectories(addValuesToTrajectory, removeTerminalTupleFromTrajectory, processTrajectoryForNN)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    # print(loadedTrajectories[0])

    filterState = lambda timeStep: (timeStep[0][0:3], timeStep[1], timeStep[2])  # !!? magic
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    print(len(trajectories))

    preProcessedTrajectories = np.concatenate(preProcessTrajectories(trajectories))

    trainData = [list(varBatch) for varBatch in zip(*preProcessedTrajectories)]
    valuedTrajectories = [addValuesToTrajectory(tra) for tra in trajectories]

    # neural network init and save path
    numStateSpace = 6
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
    reportInterval = 10000
    trainStepsIntervel = 10000
    trainReporter = TrainReporter(trainStepsIntervel, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRateModifier = lambda learningRate: LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)
    getTrainNN = lambda batchSize, learningRate: Train(trainStepsIntervel, batchSize, sampleData, learningRateModifier(learningRate), terminalController, coefficientController, trainReporter)

    # get path to save trained models
    NNModelFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations}

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainWolvesTwoCenterControlMultiTrees', 'trainedResNNModels')
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

    trainPool.map(trainOneCondition, parametersAllCondtion)

    endTime = time.time()
    print("Time taken {} seconds".format((endTime - startTime)))


if __name__ == '__main__':
    main()
