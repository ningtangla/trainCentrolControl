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

from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, MCTS, backup, establishPlainActionDist, Expand, RollOut, establishSoftmaxActionDist
from src.constrainedChasingEscapingEnv.envNoPhysics import Reset, IsTerminal, StayInBoundaryByReflectVelocity, InterpolateOneFrame, UnpackCenterControlAction, \
        ChooseInterpolatedStateByEarlyTermination, TransitWithInterpolation
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompeteWithStateInterpolation, HeuristicDistanceToTarget
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.constrainedChasingEscapingEnv.policies import HeatSeekingDiscreteStochasticPolicy, HeatSeekingContinuesDeterministicPolicy
from src.mathTools.distribution import sampleFromDistribution, maxFromDistribution, SoftDistribution, SoftMax
from src.neuralNetwork.policyValueResNet import GenerateModel, ApproximatePolicy, restoreVariables
from src.episode import Render, ForwardOneStep, SampleTrajectory, SampleTrajectoryWithRender
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, conditionDfFromParametersDict, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle


def main():
    DEBUG = 0
    renderOn = 0
    if DEBUG:
        parametersForTrajectoryPath = {}
        startSampleIndex = 5
        endSampleIndex = 7
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
    trajectoriesSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', 'NoPhysics2wolves1sheep', 'trainWolvesTwoCenterControlAction88', 'trajectories')
    if not os.path.exists(trajectoriesSaveDirectory):
        os.makedirs(trajectoriesSaveDirectory)

    trajectorySaveExtension = '.pickle'
    maxRunningSteps = 50
    numSimulations = 250
    killzoneRadius = 150
    fixedParameters = {'agentId': agentId, 'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius}

    generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)

    trajectorySavePath = generateTrajectorySavePath(parametersForTrajectoryPath)

    if not os.path.isfile(trajectorySavePath):
        numOfAgent = 3
        xBoundary = [0, 600]
        yBoundary = [0, 600]
        resetState = Reset(xBoundary, yBoundary, numOfAgent)

        stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
        interpolateOneFrame = InterpolateOneFrame(stayInBoundaryByReflectVelocity)

        chooseInterpolatedNextState = lambda interpolatedStates: interpolatedStates[-1]
        
        sheepId = 0
        wolvesId = 1
        centerControlIndexList = [wolvesId]
        unpackCenterControlAction = UnpackCenterControlAction(centerControlIndexList)

        numFramesToInterpolate = 0
        transit = TransitWithInterpolation(numFramesToInterpolate, interpolateOneFrame, 
                chooseInterpolatedNextState, unpackCenterControlAction)

        # NNGuidedMCTS init
        cInit = 1
        cBase = 100
        calculateScore = ScoreChild(cInit, cBase)
        selectChild = SelectChild(calculateScore)

        actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
        wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]

        preyPowerRatio = 10
        sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

        predatorPowerRatio = 8
        wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
        wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))

        wolvesActionSpace = list(product(wolfActionOneSpace, wolfActionTwoSpace))

        actionSpaceList = [sheepActionSpace, wolvesActionSpace]

        # neural network init
        numStateSpace = 2 * numOfAgent
        numSheepActionSpace = len(sheepActionSpace)
        numWolvesActionSpace = len(wolvesActionSpace)

        regularizationFactor = 1e-4
        sharedWidths = [128]
        actionLayerWidths = [128]
        valueLayerWidths = [128]
        generateSheepModel = GenerateModel(numStateSpace, numSheepActionSpace, regularizationFactor)

        # load save dir
        NNModelSaveExtension = ''
        sheepNNModelSaveDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', 'NoPhysics2wolves1sheep', 'trainSheepWithTwoHeatSeekingWolves', 'trainedResNNModels')
        sheepNNModelFixedParameters = {'agentId': 0, 'maxRunningSteps': 50, 'numSimulations': 110, 'miniBatchSize': 256, 'learningRate': 0.0001, }
        getSheepNNModelSavePath = GetSavePath(sheepNNModelSaveDirectory, NNModelSaveExtension, sheepNNModelFixedParameters)

        depth = 9
        resBlockSize = 2
        dropoutRate = 0.0
        initializationMethod = 'uniform'
        initSheepNNModel = generateSheepModel(sharedWidths * depth, actionLayerWidths, valueLayerWidths, resBlockSize, initializationMethod, dropoutRate)

        sheepTrainedModelPath = getSheepNNModelSavePath({'trainSteps': 50000, 'depth': depth})
        sheepTrainedModel = restoreVariables(initSheepNNModel, sheepTrainedModelPath)
        sheepPolicy = ApproximatePolicy(sheepTrainedModel, sheepActionSpace)
        
        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        speed = 120
        #sheepPolicy = HeatSeekingContinuesDeterministicPolicy(getWolfOneXPos, getSheepXPos, speed)
    
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

        # reward function
        wolfOneId = 1
        wolfTwoId = 2
        xPosIndex = [0, 1]
        getSheepXPos = GetAgentPosFromState(sheepId, xPosIndex)
        getWolfOneXPos = GetAgentPosFromState(wolfOneId, xPosIndex)
        getWolfTwoXPos = GetAgentPosFromState(wolfTwoId, xPosIndex)
        isCollidedOne = IsTerminal(getWolfOneXPos, getSheepXPos, killzoneRadius)
        isCollidedTwo = IsTerminal(getWolfTwoXPos, getSheepXPos, killzoneRadius)

        calCollisionTimes = lambda state: np.sum([isCollidedOne(state), isCollidedTwo(state)]) # collisionTimeByAddingCollisionInAllWolves
        #calCollisionTimes = lambda state: np.max([isCollidedOne(state), isCollidedTwo(state)]) # collisionTimeByBooleanCollisionForAnyWolf
       
        calTerminationSignals = calCollisionTimes
        chooseInterpolatedStateByEarlyTermination = ChooseInterpolatedStateByEarlyTermination(calTerminationSignals)

        numFramesToInterpolateInReward = 3
        interpolateStateInReward = TransitWithInterpolation(numFramesToInterpolateInReward, interpolateOneFrame, 
                chooseInterpolatedStateByEarlyTermination, unpackCenterControlAction)
        
        aliveBonus = -1 / maxRunningSteps * 10
        deathPenalty = 1
        rewardFunction = RewardFunctionCompeteWithStateInterpolation(
            aliveBonus, deathPenalty, calCollisionTimes, interpolateStateInReward)

        # initialize children; expand
        initializeChildren = InitializeChildren(
            wolvesActionSpace, wolvesTransit, getActionPrior)
        isTerminal = lambda state: False
        expand = Expand(isTerminal, initializeChildren)

        # random rollout policy
        def rolloutPolicy(
            state): return [sampleFromDistribution(sheepPolicy(state)), wolvesActionSpace[np.random.choice(range(numWolvesActionSpace))]]

        # rollout
        #rolloutHeuristicWeight = 0
        #minDistance = 400
        #rolloutHeuristic1 = HeuristicDistanceToTarget(
        #    rolloutHeuristicWeight, getWolfOneXPos, getSheepXPos, minDistance)
        #rolloutHeuristic2 = HeuristicDistanceToTarget(
        #    rolloutHeuristicWeight, getWolfTwoXPos, getSheepXPos, minDistance)

        #rolloutHeuristic = lambda state: (rolloutHeuristic1(state) + rolloutHeuristic2(state)) / 2
        
        rolloutHeuristic = lambda state: 0
        maxRolloutSteps = 15
        rollout = RollOut(rolloutPolicy, maxRolloutSteps, transit, rewardFunction, isTerminal, rolloutHeuristic)

        wolfPolicy = MCTS(numSimulations, selectChild, expand, rollout, backup, establishSoftmaxActionDist)

        # All agents' policies
        policy = lambda state: [sheepPolicy(state), wolfPolicy(state)]
        chooseActionList = [maxFromDistribution, maxFromDistribution]

        def sampleAction(state):
            actionDists = [sheepPolicy(state), wolfPolicy(state)]
            action = [chooseAction(actionDist) for actionDist, chooseAction in zip(actionDists, chooseActionList)]
            return action

        render = None
        if renderOn:
            import pygame as pg
            from pygame.color import THECOLORS
            screenColor = THECOLORS['black']
            circleColorList = [THECOLORS['green'], THECOLORS['yellow'], THECOLORS['red']]
            circleSize = 10
            saveImage = False
            saveImageDir = os.path.join(dirName, '..', '..', '..', '..', 'data', 'demoImg')
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)
            screen = pg.display.set_mode([max(xBoundary), max(yBoundary)])
            render = Render(numOfAgent, xPosIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir)

        forwardOneStep = ForwardOneStep(transit, rewardFunction)
        sampleTrajectory = SampleTrajectoryWithRender(maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn)

        trajectories = [sampleTrajectory(sampleAction) for sampleIndex in range(startSampleIndex, endSampleIndex)]
        print([len(traj) for traj in trajectories])
        saveToPickle(trajectories, trajectorySavePath)


if __name__ == '__main__':
    main()
