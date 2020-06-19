import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..'))
import json
import numpy as np
from collections import OrderedDict
import pandas as pd
import pathos.multiprocessing as mp
import itertools as it

from src.constrainedChasingEscapingEnv.envMADDPG import *
from src.constrainedChasingEscapingEnv.envNoPhysics import IsTerminal
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle, DeleteUsedModel
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet, ProcessTrajectoryForPolicyValueNetMultiAgentReward
from exec.parallelComputing import GenerateTrajectoriesParallel


class PreprocessTrajectoriesForBuffer:
    def __init__(self, addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory):
        self.addMultiAgentValuesToTrajectory = addMultiAgentValuesToTrajectory
        self.removeTerminalTupleFromTrajectory = removeTerminalTupleFromTrajectory

    def __call__(self, trajectories):
        trajectoriesWithValues = [self.addMultiAgentValuesToTrajectory(trajectory) for trajectory in trajectories]
        filteredTrajectories = [self.removeTerminalTupleFromTrajectory(trajectory) for trajectory in trajectoriesWithValues]
        return filteredTrajectories


class TrainOneAgent:
    def __init__(self, numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets,
                 sampleBatchFromBuffer, trainNN):
        self.numTrainStepEachIteration = numTrainStepEachIteration
        self.numTrajectoriesToStartTrain = numTrajectoriesToStartTrain
        self.sampleBatchFromBuffer = sampleBatchFromBuffer
        self.processTrajectoryForPolicyValueNets = processTrajectoryForPolicyValueNets
        self.trainNN = trainNN

    def __call__(self, agentId, multiAgentNNmodel, updatedReplayBuffer):
        NNModel = multiAgentNNmodel[agentId]
        if len(updatedReplayBuffer) >= self.numTrajectoriesToStartTrain:
            for _ in range(self.numTrainStepEachIteration):
                sampledBatch = self.sampleBatchFromBuffer(updatedReplayBuffer)
                processedBatch = self.processTrajectoryForPolicyValueNets[agentId](sampledBatch)
                trainData = [list(varBatch) for varBatch in zip(*processedBatch)]
                updatedNNModel = self.trainNN(NNModel, trainData)
                NNModel = updatedNNModel

        return NNModel


def iterateTrainOneCondition(parameterOneCondition):

    numTrainStepEachIteration = int(parameterOneCondition['numTrainStepEachIteration'])
    numTrajectoriesPerIteration = int(parameterOneCondition['numTrajectoriesPerIteration'])
    dirName = os.path.dirname(__file__)

    numOfAgent = 2
    agentIds = list(range(numOfAgent))

# env MDP
    sheepsID = [0]
    wolvesID = [1, 2]
    blocksID = []

    numSheeps = len(sheepsID)
    numWolves = len(wolvesID)
    numBlocks = len(blocksID)

    numAgents = numWolves + numSheeps
    numEntities = numAgents + numBlocks

    sheepSize = 0.05
    wolfSize = 0.075
    blockSize = 0.2

    sheepMaxSpeed = 1.3 * 1
    wolfMaxSpeed = 1.0 * 1
    blockMaxSpeed = None

    entitiesSizeList = [sheepSize] * numSheeps + [wolfSize] * numWolves + [blockSize] * numBlocks
    entityMaxSpeedList = [sheepMaxSpeed] * numSheeps + [wolfMaxSpeed] * numWolves + [blockMaxSpeed] * numBlocks
    entitiesMovableList = [True] * numAgents + [False] * numBlocks
    massList = [1.0] * numEntities

    centralControlId = 1
    centerControlIndexList = [centralControlId]
    reshapeAction = UnpackCenterControlAction(centerControlIndexList)
    getCollisionForce = GetCollisionForce()
    applyActionForce = ApplyActionForce(wolvesID, sheepsID, entitiesMovableList)
    applyEnvironForce = ApplyEnvironForce(numEntities, entitiesMovableList, entitiesSizeList,
                                          getCollisionForce, getPosFromAgentState)
    integrateState = IntegrateState(numEntities, entitiesMovableList, massList,
                                    entityMaxSpeedList, getVelFromAgentState, getPosFromAgentState)
    interpolateState = TransitMultiAgentChasing(numEntities, reshapeAction, applyActionForce, applyEnvironForce, integrateState)

    numFramesToInterpolate = 1

    def transit(state, action):
        for frameIndex in range(numFramesToInterpolate):
            nextState = interpolateState(state, action)
            action = np.array([(0, 0)] * numAgents)
            state = nextState
        return nextState

    isTerminal = lambda state: False

    isCollision = IsCollision(getPosFromAgentState)
    collisonRewardWolf = 1
    punishForOutOfBound = PunishForOutOfBound()
    rewardWolf = RewardCentralControlPunishBond(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisonRewardWolf)
    collisonRewardSheep = -1
    rewardSheep = RewardCentralControlPunishBond(sheepsID, wolvesID, entitiesSizeList, getPosFromAgentState, isCollision, punishForOutOfBound, collisonRewardSheep)

    resetState = ResetMultiAgentChasing(numAgents, numBlocks)

    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID, blocksID, getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgents)]

# policy
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]

    preyPowerRatio = 0.5
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    predatorPowerRatio = 0.5
    wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

    wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))

    actionSpaceList = [sheepActionSpace, wolvesActionSpace]

    # neural network init
    numStateSpace = 4 * numEntities
    numSheepActionSpace = len(sheepActionSpace)
    numWolvesActionSpace = len(wolvesActionSpace)

    regularizationFactor = 1e-4
    sharedWidths = [128]
    actionLayerWidths = [128]
    valueLayerWidths = [128]

    generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)
    generateWolvesModel = GenerateModel(numStateSpace, numWolvesActionSpace, regularizationFactor)
    generateModelList = [generateSheepModel, generateWolvesModel]

    sheepDepth = 9
    wolfDepth = 9
    depthList = [sheepDepth, wolfDepth]
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    # replay buffer
    bufferSize = 20000
    saveToBuffer = SaveToBuffer(bufferSize)

    def getUniformSamplingProbabilities(buffer): return [(1 / len(buffer)) for _ in buffer]
    miniBatchSize = 512
    sampleBatchFromBuffer = SampleBatchFromBuffer(miniBatchSize, getUniformSamplingProbabilities)

    # pre-process the trajectory for replayBuffer
    rewardMultiAgents = [rewardSheep, rewardWolf]
    decay = 1
    accumulateMultiAgentRewards = AccumulateMultiAgentRewards(decay)

    addMultiAgentValuesToTrajectory = AddValuesToTrajectory(accumulateMultiAgentRewards)
    actionIndex = 1

    def getTerminalActionFromTrajectory(trajectory): return trajectory[-1][actionIndex]
    removeTerminalTupleFromTrajectory = RemoveTerminalTupleFromTrajectory(getTerminalActionFromTrajectory)

    # pre-process the trajectory for NNTraining
    sheepActionToOneHot = ActionToOneHot(sheepActionSpace)
    wolvesActionToOneHot = ActionToOneHot(wolvesActionSpace)
    actionToOneHotList = [sheepActionToOneHot, wolvesActionToOneHot]
    processTrajectoryForPolicyValueNets = [ProcessTrajectoryForPolicyValueNetMultiAgentReward(actionToOneHotList[agentId], agentId) for agentId in agentIds]

    # function to train NN model
    terminalThreshold = 1e-6
    lossHistorySize = 10
    initActionCoeff = 1
    initValueCoeff = 1
    initCoeff = (initActionCoeff, initValueCoeff)
    afterActionCoeff = 1
    afterValueCoeff = 1
    afterCoeff = (afterActionCoeff, afterValueCoeff)

    terminalController = TrainTerminalController(lossHistorySize, terminalThreshold)
    coefficientController = CoefficientCotroller(initCoeff, afterCoeff)

    reportInterval = 10000
    trainStepsIntervel = 1  # 10000

    trainReporter = TrainReporter(numTrainStepEachIteration, reportInterval)
    learningRateDecay = 1
    learningRateDecayStep = 1
    learningRate = 0.0001
    learningRateModifier = LearningRateModifier(learningRate, learningRateDecay, learningRateDecayStep)

    trainNN = Train(numTrainStepEachIteration, miniBatchSize, sampleData, learningRateModifier, terminalController, coefficientController, trainReporter)

    # load save dir

    trajectorySaveExtension = '.pickle'
    NNModelSaveExtension = ''
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'iterTrain2wolves1sheepMADDPGEnv', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'iterTrain2wolves1sheepMADDPGEnv', 'NNModelRes')
    if not os.path.exists(NNModelSaveDirectory):
        os.makedirs(NNModelSaveDirectory)

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

    startTime = time.time()

    sheepDepth = 9
    wolfDepth = 9
    depthList = [sheepDepth, wolfDepth]
    resBlockSize = 2
    dropoutRate = 0.0
    initializationMethod = 'uniform'
    multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

    preprocessMultiAgentTrajectories = PreprocessTrajectoriesForBuffer(addMultiAgentValuesToTrajectory, removeTerminalTupleFromTrajectory)
    numTrajectoriesToStartTrain = 1024

    trainOneAgent = TrainOneAgent(numTrainStepEachIteration, numTrajectoriesToStartTrain, processTrajectoryForPolicyValueNets, sampleBatchFromBuffer, trainNN)

    # restorePretrainModel
    sheepPreTrainModelPath = os.path.join(dirName, '..', '..', '..', '..', 'data', 'MADDPG2wolves1sheep', 'trainWolvesTwoCenterControlAction', 'trainedResNNModels', 'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=250_trainSteps=50000')

    wolvesPreTrainModelPath = os.path.join(dirName, '..', '..', '..', '..', 'data', 'MADDPG2wolves1sheep', 'trainSheepWithPretrrainWolves', 'trainedResNNModels', 'agentId=1_depth=9_learningRate=0.0001_maxRunningSteps=50_miniBatchSize=256_numSimulations=250_trainSteps=50000')

    pretrainModelPathList = [sheepPreTrainModelPath, wolvesPreTrainModelPath]

    trainableAgentIds = [sheepId, wolvesId]
    for agentId in trainableAgentIds:

        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], pretrainModelPathList[agentId])
        multiAgentNNmodel[agentId] = restoredNNModel

        NNModelPathParameters = {'iterationIndex': 0, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
        NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
        saveVariables(multiAgentNNmodel[agentId], NNModelSavePath)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectoriesForParallel = LoadTrajectories(generateTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesForTrainBreak = LoadTrajectories(generateTrajectorySavePath, loadFromPickle)

    # initRreplayBuffer
    replayBuffer = []
    trajectoryBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': trajectoryBeforeTrainIndex}
    trajectoriesBeforeTrain = loadTrajectoriesForParallel(trajectoryBeforeTrainPathParamters)
    preProcessedTrajectoriesBeforeTrain = preprocessMultiAgentTrajectories(trajectoriesBeforeTrain)
    replayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectoriesBeforeTrain)

    # delete used model for disk space
    fixedParametersForDelete = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
    toDeleteNNModelExtensionList = ['.meta', '.index', '.data-00000-of-00001']
    generatetoDeleteNNModelPathList = [GetSavePath(NNModelSaveDirectory, toDeleteNNModelExtension, fixedParametersForDelete) for toDeleteNNModelExtension in toDeleteNNModelExtensionList]

# restore model
    restoredIteration = 0
    for agentId in trainableAgentIds:
        modelPathForRestore = generateNNModelSavePath({'iterationIndex': restoredIteration, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
        restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPathForRestore)
        multiAgentNNmodel[agentId] = restoredNNModel

# restore buffer
    bufferTrajectoryPathParameters = {'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
    restoredIterationIndexRange = range(restoredIteration)
    restoredTrajectories = loadTrajectoriesForTrainBreak(parameters=bufferTrajectoryPathParameters, parametersWithSpecificValues={'iterationIndex': list(restoredIterationIndexRange)})
    preProcessedRestoredTrajectories = preprocessMultiAgentTrajectories(restoredTrajectories)
    print(len(preProcessedRestoredTrajectories))
    replayBuffer = saveToBuffer(replayBuffer, preProcessedRestoredTrajectories)

    modelMemorySize = 5
    modelSaveFrequency = 100
    deleteUsedModel = DeleteUsedModel(modelMemorySize, modelSaveFrequency, generatetoDeleteNNModelPathList)
    numIterations = 10000
    for iterationIndex in range(restoredIteration + 1, numIterations):
        print('iterationIndex: ', iterationIndex)

        numCpuToUseWhileTrain = int(16)
        numCmdList = min(numTrajectoriesPerIteration, numCpuToUseWhileTrain)
        sampleTrajectoryFileName = 'sampleMultiMCTSAgentCenterControlResNetTrajCondtion.py'

        generateTrajectoriesParallelWhileTrain = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesPerIteration, numCmdList)
        trajectoryPathParameters = {'iterationIndex': iterationIndex, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
        cmdList = generateTrajectoriesParallelWhileTrain(trajectoryPathParameters)

        trajectories = loadTrajectoriesForParallel(trajectoryPathParameters)
        trajectorySavePath = generateTrajectorySavePath(trajectoryPathParameters)
        saveToPickle(trajectories, trajectorySavePath)

        preProcessedTrajectories = preprocessMultiAgentTrajectories(trajectories)
        updatedReplayBuffer = saveToBuffer(replayBuffer, preProcessedTrajectories)

        for agentId in trainableAgentIds:

            updatedAgentNNModel = trainOneAgent(agentId, multiAgentNNmodel, updatedReplayBuffer)

            NNModelPathParameters = {'iterationIndex': iterationIndex, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration}
            NNModelSavePath = generateNNModelSavePath(NNModelPathParameters)
            saveVariables(updatedAgentNNModel, NNModelSavePath)
            multiAgentNNmodel[agentId] = updatedAgentNNModel
            replayBuffer = updatedReplayBuffer

            deleteUsedModel(iterationIndex, agentId)

    endTime = time.time()
    print("Time taken for {} iterations: {} seconds".format(
        numIterations, (endTime - startTime)))


def main():
    manipulatedVariables = OrderedDict()
    manipulatedVariables['numTrainStepEachIteration'] = [1]
    manipulatedVariables['numTrajectoriesPerIteration'] = [1]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    # Sample Trajectory Before Train to fill Buffer
    miniBatchSize = 256
    numTrajectoriesToStartTrain = 2 * miniBatchSize
    sampleTrajectoryFileName = 'preparePretrainedNNMCTSAgentCenterControlResNetTraj.py'
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.8 * numCpuCores)
    numCmdList = min(numTrajectoriesToStartTrain, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(sampleTrajectoryFileName, numTrajectoriesToStartTrain, numCmdList)
    iterationBeforeTrainIndex = 0
    trajectoryBeforeTrainPathParamters = {'iterationIndex': iterationBeforeTrainIndex}
    prepareBefortrainData = True
    if prepareBefortrainData:
        cmdList = generateTrajectoriesParallel(trajectoryBeforeTrainPathParamters)

    trainPool = mp.Pool(numCpuToUse)
    trainPool.map(iterateTrainOneCondition, parametersAllCondtion)


if __name__ == '__main__':
    main()
