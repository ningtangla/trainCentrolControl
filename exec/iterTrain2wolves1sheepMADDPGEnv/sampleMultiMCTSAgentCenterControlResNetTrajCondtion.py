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
from itertools import product
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envMADDPG import *
from src.constrainedChasingEscapingEnv.envNoPhysics import TransiteForNoPhysicsWithCenterControlAction, Reset, IsTerminal, StayInBoundaryByReflectVelocity, UnpackCenterControlAction, TransitWithInterpolateStateWithCenterControlAction
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueResNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateMultiAgentRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, \
    ActionToOneHot, ProcessTrajectoryForPolicyValueNet
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution, SoftMax
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, StochasticMCTS, backup, establishPlainActionDistFromMultipleTrees, Expand
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy, HeatSeekingContinuesDeterministicPolicy
from src.episode import Render, ForwardMultiAgentsOneStep, SampleTrajectory, SampleTrajectoryWithRender
from exec.parallelComputing import GenerateTrajectoriesParallel


class ComposeMultiAgentTransitInSingleAgentMCTS:
    def __init__(self, chooseAction):
        self.chooseAction = chooseAction

    def __call__(self, agentId, state, selfAction, othersPolicy, transit):
        multiAgentActions = [self.chooseAction(policy(state)) for policy in othersPolicy]
        multiAgentActions.insert(agentId, selfAction)
        transitInSelfMCTS = transit(state, multiAgentActions)
        return transitInSelfMCTS


class ComposeSingleAgentGuidedMCTS():
    def __init__(self, numTree, numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS):
        self.numTree = numTree
        self.numSimulations = numSimulations
        self.numSimulationsPerTree = int(self.numSimulations / self.numTree)
        self.actionSpaceList = actionSpaceList
        self.terminalRewardList = terminalRewardList
        self.selectChild = selectChild
        self.isTerminal = isTerminal
        self.transit = transit
        self.getStateFromNode = getStateFromNode
        self.getApproximatePolicy = getApproximatePolicy
        self.getApproximateValue = getApproximateValue
        self.composeMultiAgentTransitInSingleAgentMCTS = composeMultiAgentTransitInSingleAgentMCTS

    def __call__(self, agentId, selfNNModel, othersPolicy):
        approximateActionPrior = self.getApproximatePolicy[agentId](selfNNModel)

        def transitInMCTS(state, selfAction): return self.composeMultiAgentTransitInSingleAgentMCTS(agentId, state, selfAction, othersPolicy, self.transit)
        initializeChildren = InitializeChildren(self.actionSpaceList[agentId], transitInMCTS, approximateActionPrior)
        expand = Expand(self.isTerminal, initializeChildren)

        terminalReward = self.terminalRewardList[agentId]
        approximateValue = self.getApproximateValue[agentId](selfNNModel)
        estimateValue = EstimateValueFromNode(terminalReward, self.isTerminal, self.getStateFromNode, approximateValue)
        guidedMCTSPolicy = StochasticMCTS(self.numTree, self.numSimulationsPerTree, self.selectChild, expand, estimateValue, backup, establishPlainActionDistFromMultipleTrees)

        return guidedMCTSPolicy


class PrepareMultiAgentPolicy:
    def __init__(self, composeSingleAgentGuidedMCTS, approximatePolicy, MCTSAgentIds):
        self.composeSingleAgentGuidedMCTS = composeSingleAgentGuidedMCTS
        self.approximatePolicy = approximatePolicy
        self.MCTSAgentIds = MCTSAgentIds

    def __call__(self, multiAgentNNModel):
        multiAgentApproximatePolicy = np.array([approximatePolicy(NNModel) for approximatePolicy, NNModel in zip(self.approximatePolicy, multiAgentNNModel)])
        otherAgentPolicyForMCTSAgents = np.array([np.concatenate([multiAgentApproximatePolicy[:agentId], multiAgentApproximatePolicy[agentId + 1:]]) for agentId in self.MCTSAgentIds])
        MCTSAgentIdWithCorrespondingOtherPolicyPair = zip(self.MCTSAgentIds, otherAgentPolicyForMCTSAgents)
        MCTSAgentsPolicy = np.array([self.composeSingleAgentGuidedMCTS(agentId, multiAgentNNModel[agentId], correspondingOtherAgentPolicy)
                                     for agentId, correspondingOtherAgentPolicy in MCTSAgentIdWithCorrespondingOtherPolicyPair])
        multiAgentPolicy = np.copy(multiAgentApproximatePolicy)
        multiAgentPolicy[self.MCTSAgentIds] = MCTSAgentsPolicy

        def policy(state): return [agentPolicy(state) for agentPolicy in multiAgentPolicy]
        return policy


def main():
    DEBUG = 0
    renderOn = 0

    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 0
        endSampleIndex = 1
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
        iterationIndex = 0
        numTrainStepEachIteration = 1
        numTrajectoriesPerIteration = 1

    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
        iterationIndex = int(parametersForTrajectoryPath['iterationIndex'])
        numTrainStepEachIteration = int(parametersForTrajectoryPath['numTrainStepEachIteration'])
        numTrajectoriesPerIteration = int(parametersForTrajectoryPath['numTrajectoriesPerIteration'])

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'iterTrain2wolves1sheepMADDPGEnv', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'

    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 50
    numTree = 2
    fixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}
    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
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

        rewardMultiAgents = [rewardSheep, rewardWolf]

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
        trainableAgentIds = [sheepId, wolvesId]

        multiAgentNNmodel = [generateModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate) for depth, generateModel in zip(depthList, generateModelList)]

        otherAgentApproximatePolicy = [lambda NNmodel, : ApproximatePolicy(NNmodel, sheepActionSpace), lambda NNmodel, : ApproximatePolicy(NNmodel, wolvesActionSpace)]
        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        getApproximatePolicy = [lambda NNmodel, : ApproximatePolicy(NNmodel, sheepActionSpace), lambda NNmodel, : ApproximatePolicy(NNmodel, wolvesActionSpace)]
        getApproximateValue = [lambda NNmodel: ApproximateValue(NNmodel), lambda NNmodel: ApproximateValue(NNmodel)]

        def getStateFromNode(node): return list(node.id.values())[0]

        chooseActionInMCTS = sampleFromDistribution

        composeMultiAgentTransitInSingleAgentMCTS = ComposeMultiAgentTransitInSingleAgentMCTS(chooseActionInMCTS)
        composeSingleAgentGuidedMCTS = ComposeSingleAgentGuidedMCTS(numTree, numSimulations, actionSpaceList, terminalRewardList, selectChild, isTerminal, transit, getStateFromNode, getApproximatePolicy, getApproximateValue, composeMultiAgentTransitInSingleAgentMCTS)
        prepareMultiAgentPolicy = PrepareMultiAgentPolicy(composeSingleAgentGuidedMCTS, otherAgentApproximatePolicy, trainableAgentIds)

        # load model
        NNModelSaveExtension = ''
        NNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', 'data', 'iterTrain2wolves1sheepMADDPGEnv', 'NNModelRes')
        if not os.path.exists(NNModelSaveDirectory):
            os.makedirs(NNModelSaveDirectory)

        generateNNModelSavePath = GetSavePath(NNModelSaveDirectory, NNModelSaveExtension, fixedParameters)

        for agentId in trainableAgentIds:
            modelPath = generateNNModelSavePath({'iterationIndex': iterationIndex - 1, 'agentId': agentId, 'numTrajectoriesPerIteration': numTrajectoriesPerIteration, 'numTrainStepEachIteration': numTrainStepEachIteration})
            restoredNNModel = restoreVariables(multiAgentNNmodel[agentId], modelPath)
            multiAgentNNmodel[agentId] = restoredNNModel

        policy = prepareMultiAgentPolicy(multiAgentNNmodel)
        chooseActionList = [maxFromDistribution, maxFromDistribution]

        def sampleAction(state):
            actionDists = [sheepPolicy(state), wolfPolicy(state)]
            action = [chooseAction(actionDist) for actionDist, chooseAction in zip(actionDists, chooseActionList)]
            return action

        render = None
        forwardOneStep = ForwardMultiAgentsOneStep(transit, rewardMultiAgents)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn)

        trajectories = [sampleTrajectory(sampleAction) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
