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
import pygame as pg
from pygame.color import THECOLORS

from src.constrainedChasingEscapingEnv.envNoPhysics import Reset, IsTerminal, StayInBoundaryByReflectVelocity, InterpolateOneFrame, UnpackCenterControlAction, \
        ChooseInterpolatedStateByEarlyTermination, TransitWithInterpolation
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
from src.visualization.drawDemo import DrawBackground, DrawCircleOutside, DrawState, ChaseTrialWithTraj, InterpolateState

def main():
    depth = 9
    # Get dataset for training
    DIRNAME = os.path.dirname(__file__)
    dataSetDirectory = os.path.join(dirName, '..', '..', '..', '..', 'data', '2wolves1sheep', 'trainWolvesTwoCenterControlAction88', 'trajectories')

    if not os.path.exists(dataSetDirectory):
        os.makedirs(dataSetDirectory)

    dataSetExtension = '.pickle'
    dataSetMaxRunningSteps = 50
    dataSetNumSimulations = 250
    killzoneRadius = 100
    agentId = 1
    wolvesId = 1
    dataSetFixedParameters = {'agentId': agentId, 'maxRunningSteps': dataSetMaxRunningSteps, 'numSimulations': dataSetNumSimulations, 'killzoneRadius': killzoneRadius}

    getDataSetSavePath = GetSavePath(dataSetDirectory, dataSetExtension, dataSetFixedParameters)
    print("DATASET LOADED!")

    numSheep = 1
    numWolves = 2
    numOfAgent = numSheep + numWolves
    # accumulate rewards for trajectories
    decay = 1
    accumulateRewards = AccumulateRewards(decay)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

    # pre-process the trajectories
    actionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7), (0, 0)]
    preyPowerRatio = 8
    sheepActionSpace = list(map(tuple, np.array(actionSpace) * preyPowerRatio))

    predatorPowerRatio = 8
    wolfActionSpace = [(10, 0), (7, 7), (0, 10), (-7, 7), (-10, 0), (-7, -7), (0, -10), (7, -7)]
    wolfActionOneSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
    wolfActionTwoSpace = list(map(tuple, np.array(wolfActionSpace) * predatorPowerRatio))
    wolvesActionSpace = list(it.product(wolfActionOneSpace, wolfActionTwoSpace))

    numActionSpace = len(wolvesActionSpace)

    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getDataSetSavePath, loadFromPickle, fuzzySearchParameterNames)
    loadedTrajectories = loadTrajectories(parameters={})
    # print(loadedTrajectories[0])

    filterState = lambda timeStep: (timeStep[0][:numOfAgent], np.array([timeStep[1][0], timeStep[1][1][0], timeStep[1][1][1]]), 
            timeStep[2], timeStep[3])
    trajectories = [[filterState(timeStep) for timeStep in trajectory] for trajectory in loadedTrajectories]
    print(len(trajectories))


    # generate demo image
    screenWidth = 600
    screenHeight = 600
    screen = pg.display.set_mode((screenWidth, screenHeight))
    screenColor = THECOLORS['black']
    xBoundary = [0, 600]
    yBoundary = [0, 600]
    lineColor = THECOLORS['white']
    lineWidth = 4
    drawBackground = DrawBackground(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    
    FPS = 30
    circleColorSpace = [[0, 255, 0]]*numSheep + [[255, 255, 255]] * numWolves
    circleSize = 10
    positionIndex = [0, 1]
    agentIdsToDraw = list(range(numSheep + numWolves))
    saveImage = True
    imageSavePath = os.path.join(dataSetDirectory, 'picMovingSheep')
    if not os.path.exists(imageSavePath):
        os.makedirs(imageSavePath)
    imageFolderName = str('forDemo')
    saveImageDir = os.path.join(os.path.join(imageSavePath, imageFolderName))
    if not os.path.exists(saveImageDir):
        os.makedirs(saveImageDir)
    goalSpace = list(range(numSheep))
    imaginedWeIdsForInferenceSubject = list(range(numSheep, numWolves + numSheep))
    updateColorSpaceByPosterior = None
    
    #updateColorSpaceByPosterior = lambda originalColorSpace, posterior : originalColorSpace
    outsideCircleAgentIds = imaginedWeIdsForInferenceSubject
    outsideCircleColor = np.array([[255, 0, 0]] * numWolves)
    outsideCircleSize = 15 
    drawCircleOutside = DrawCircleOutside(screen, outsideCircleAgentIds, positionIndex, outsideCircleColor, outsideCircleSize)
    drawState = DrawState(FPS, screen, circleColorSpace, circleSize, agentIdsToDraw, positionIndex, 
            saveImage, saveImageDir, drawBackground, updateColorSpaceByPosterior, drawCircleOutside)
    
    # MDP Env
    xBoundary = [0,600]
    yBoundary = [0,600]
    stayInBoundaryByReflectVelocity = StayInBoundaryByReflectVelocity(xBoundary, yBoundary)
    transit = InterpolateOneFrame(stayInBoundaryByReflectVelocity)
    numFramesToInterpolate = 8
    interpolateState = InterpolateState(numFramesToInterpolate, transit)
    
    stateIndexInTimeStep = 0
    actionIndexInTimeStep = 1
    chaseTrial = ChaseTrialWithTraj(stateIndexInTimeStep, drawState, interpolateState, actionIndexInTimeStep)
    
    print(len(trajectories))
    lens = [len(trajectory) for trajectory in trajectories]
    index = np.argsort(-np.array(lens))
    print(index)
    print(trajectories[0][1])
    [chaseTrial(trajectory) for trajectory in np.array(trajectories)[1:]]
    print([len(trajectory) for trajectory in np.array(trajectories)[index[:]]])
    #[chaseTrial(trajectory) for trajectory in np.array(trajectories)[index[:5]]]



if __name__ == '__main__':
    main()
