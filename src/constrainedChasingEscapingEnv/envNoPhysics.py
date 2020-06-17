import numpy as np


class Reset():
    def __init__(self, xBoundary, yBoundary, numOfAgent, isLegal = lambda state: True):
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.numOfAgnet = numOfAgent
        self.isLegal = isLegal

    def __call__(self):
        xMin, xMax = self.xBoundary
        yMin, yMax = self.yBoundary
        initState = [[np.random.uniform(xMin, xMax),
                      np.random.uniform(yMin, yMax)]
                     for _ in range(self.numOfAgnet)]
        while np.all([self.isLegal(state) for state in initState]) is False:
            initState = [[np.random.uniform(xMin, xMax),
                          np.random.uniform(yMin, yMax)]
                         for _ in range(self.numOfAgnet)] 
        return np.array(initState)


def samplePosition(xBoundary, yBoundary):
    positionX = np.random.uniform(xBoundary[0], xBoundary[1])
    positionY = np.random.uniform(yBoundary[0], yBoundary[1])
    position = [positionX, positionY]
    return position


class RandomReset():
    def __init__(self, numOfAgent, xBoundary, yBoundary, isTerminal):
        self.numOfAgent = numOfAgent
        self.xBoundary = xBoundary
        self.yBoundary = yBoundary
        self.isTerminal = isTerminal

    def __call__(self):
        terminal = True
        while terminal:
            initState = [samplePosition(self.xBoundary, self.yBoundary) for i in range(self.numOfAgent)]
            initState = np.array(initState)
            terminal = self.isTerminal(initState)
        return initState


class FixedReset():
    def __init__(self, initPositionList):
        self.initPositionList = initPositionList

    def __call__(self, trialIndex):
        initState = self.initPositionList[trialIndex]
        initState = np.array(initState)
        return initState


class InterpolateOneFrame():
    def __init__(self, stayInBoundaryByReflectVelocity):
        self.stayInBoundaryByReflectVelocity = stayInBoundaryByReflectVelocity

    def __call__(self, positions, velocities):
        newPositions = np.array(positions) + np.array(velocities)
        checkedNewPositionsAndVelocities = [self.stayInBoundaryByReflectVelocity(
            position, velocity) for position, velocity in zip(newPositions, velocities)]
        newPositions, newVelocities = list(zip(*checkedNewPositionsAndVelocities))
        return np.array(newPositions), np.array(newVelocities)

class ChooseInterpolatedStateByEarlyTermination:
    def __init__(self, calTerminationSignal):
        self.calTerminationSignal = calTerminationSignal

    def __call__(self, interpolatedStates):
        signals = [self.calTerminationSignal(state) 
                for state in interpolatedStates]
        maxInSignals = max(signals)
        indexForTermination = -1
        if maxInSignals != 0:
            indexForTermination = signals.index(max(signals))
        choosedInterpolatedState = interpolatedStates[indexForTermination]
        return choosedInterpolatedState

class UnpackCenterControlAction:
    def __init__(self, centerControlIndexList):
        self.centerControlIndexList = centerControlIndexList

    def __call__(self, centerControlAction):
        upackedAction = []
        for index, action in enumerate(centerControlAction):
            if index in self.centerControlIndexList:
                [upackedAction.append(subAction) for subAction in action]
            else:
                upackedAction.append(action)
        return np.array(upackedAction)

class TransitWithInterpolation:
    def __init__(self, numFramesToInterpolate, interpolateOneFrame, chooseInterpolatedNextState, unpackCenterControlAction):
        self.numFramesToInterpolate = numFramesToInterpolate
        self.interpolateOneFrame = interpolateOneFrame
        self.chooseInterpolatedNextState = chooseInterpolatedNextState
        self.unpackCenterControlAction = unpackCenterControlAction

    def __call__(self, state, action):
        actionDecentralized = self.unpackCenterControlAction(action)
        actionForInterpolation = np.array(actionDecentralized) / (self.numFramesToInterpolate + 1)
        interpolatedStates = []
        for frameIndex in range(self.numFramesToInterpolate + 1):
            nextState, nextActionForInterpolation = self.interpolateOneFrame(state, actionForInterpolation)
            interpolatedStates.append(nextState)
            state = nextState
            actionForInterpolation = nextActionForInterpolation
        choosedNextState = self.chooseInterpolatedNextState(interpolatedStates)
        return np.array(choosedNextState)

class IsTerminal():
    def __init__(self, getPredatorPos, getPreyPos, minDistance):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.minDistance = minDistance

    def __call__(self, state):
        terminal = False
        preyPosition = self.getPreyPos(state)
        predatorPosition = self.getPredatorPos(state)
        L2Normdistance = np.linalg.norm((np.array(preyPosition) - np.array(predatorPosition)), ord=2)
        if L2Normdistance <= self.minDistance:
            terminal = True
        return terminal



class StayInBoundaryByReflectVelocity():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

class StayInBoundaryAndOutObstacleByReflectVelocity():
    def __init__(self, xBoundary, yBoundary, xObstacles, yObstacles):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary
        self.xObstacles = xObstacles
        self.yObstacles = yObstacles
    def __call__(self, position, velocity):
        adjustedX, adjustedY = position
        adjustedVelX, adjustedVelY = velocity
        if position[0] >= self.xMax:
            adjustedX = 2 * self.xMax - position[0]
            adjustedVelX = -velocity[0]
        if position[0] <= self.xMin:
            adjustedX = 2 * self.xMin - position[0]
            adjustedVelX = -velocity[0]
        if position[1] >= self.yMax:
            adjustedY = 2 * self.yMax - position[1]
            adjustedVelY = -velocity[1]
        if position[1] <= self.yMin:
            adjustedY = 2 * self.yMin - position[1]
            adjustedVelY = -velocity[1]
	
        for xObstacle, yObstacle in zip(self.xObstacles, self.yObstacles):
            xObstacleMin, xObstacleMax = xObstacle
            yObstacleMin, yObstacleMax = yObstacle
            if position[0] >= xObstacleMin and position[0] <= xObstacleMax and position[1] >= yObstacleMin and position[1] <= yObstacleMax:
                if position[0]-velocity[0]<=xObstacleMin:
                    adjustedVelX=-velocity[0]
                    adjustedX=2*xObstacleMin-position[0]
                if position[0]-velocity[0]>=xObstacleMax:
                    adjustedVelX=-velocity[0]
                    adjustedX=2*xObstacleMax-position[0]
                if position[1]-velocity[1]<=yObstacleMin:
                    adjustedVelY=-velocity[1]
                    adjustedY=2*yObstacleMin-position[1]
                if position[1]-velocity[1]>=yObstacleMax:
                    adjustedVelY=-velocity[1]
                    adjustedY=2*yObstacleMax-position[1]

        checkedPosition = np.array([adjustedX, adjustedY])
        checkedVelocity = np.array([adjustedVelX, adjustedVelY])
        return checkedPosition, checkedVelocity

class CheckBoundary():
    def __init__(self, xBoundary, yBoundary):
        self.xMin, self.xMax = xBoundary
        self.yMin, self.yMax = yBoundary

    def __call__(self, position):
        xPos, yPos = position
        if xPos >= self.xMax or xPos <= self.xMin:
            return False
        elif yPos >= self.yMax or yPos <= self.yMin:
            return False
        return True
