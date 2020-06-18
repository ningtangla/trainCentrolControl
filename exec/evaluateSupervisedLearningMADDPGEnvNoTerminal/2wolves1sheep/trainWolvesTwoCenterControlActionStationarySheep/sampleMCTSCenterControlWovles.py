import time
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DIRNAME = os.path.dirname(__file__)
sys.path.append(os.path.join(DIRNAME, '..', '..', '..', '..'))

import json
import numpy as np
from collections import OrderedDict
import pandas as pd
from itertools import product
from gym import spaces

from src.constrainedChasingEscapingEnv.envMADDPG import *
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.constrainedChasingEscapingEnv.envNoPhysics import UnpackCenterControlAction
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteStochasticPolicy, HeatSeekingContinuesDeterministicPolicy
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution, SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.episode import Render, ForwardOneStep, SampleTrajectory, SampleTrajectoryWithRender
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle

def main():
    startTime = time.time()

    DEBUG = 1
    renderOn = 1
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 5
        endSampleIndex = 8
        agentId = 1
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)
    else:
        parametersForTrajectoryPath = json.loads(sys.argv[1])
        startSampleIndex = int(sys.argv[2])
        endSampleIndex = int(sys.argv[3])
        agentId = int(parametersForTrajectoryPath['agentId'])
        parametersForTrajectoryPath['sampleIndex'] = (startSampleIndex, endSampleIndex)

    # check file exists or not
    dirName = os.path.dirname(__file__)
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', 'MADDPG2wolves1sheep', 'trainWolvesTwoCenterControlAction', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 250
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations}

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

        entitiesSizeList = [sheepSize] * numSheeps + [wolfSize]* numWolves + [blockSize]* numBlocks
        entityMaxSpeedList = [sheepMaxSpeed] * numSheeps + [wolfMaxSpeed]* numWolves + [blockMaxSpeed]* numBlocks
        entitiesMovableList = [True]* numAgents + [False] * numBlocks
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

        sheepPolicy = lambda state: {(0, 0): 1}

    # MCTS
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        # prior
        getActionPrior = lambda state: {action: 1 / len(wolvesActionSpace) for action in wolvesActionSpace}

    # load chase nn policy
        chooseActionInMCTS = sampleFromDistribution

        def wolvesTransit(state, action): return transit(
            state, [chooseActionInMCTS(sheepPolicy(state)), action])

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolvesActionSpace, wolvesTransit, getActionPrior)
        isTerminal = lambda state: False
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return [sampleFromDistribution(sheepPolicy(state)), wolvesActionSpace[np.random.choice(range(numWolvesActionSpace))]]

        rolloutHeuristic = lambda state: 0
        maxRolloutSteps = 15
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transit, rewardWolf, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
        chooseActionList = [maxFromDistribution, maxFromDistribution]

        def sampleAction(state):
            actionDists = [sheepPolicy(state), wolfPolicy(state)]
            action = [chooseAction(actionDist) for actionDist, chooseAction in zip(actionDists, chooseActionList)]
            return action

        render = None
        forwardOneStep = ForwardOneStep(transit, rewardWolf)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn)

        trajectories = [sampleTrajectory(sampleAction) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


    endTime = time.time()
    #print("Time taken {} seconds".format((endTime - startTime)))
if __name__ == '__main__':
    main()
