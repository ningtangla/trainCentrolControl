import numpy as np
import random


def stationaryAgentPolicy(state):
    return {(0, 0): 1}


class RandomPolicy:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        likelihood = {action: 1 / len(self.actionSpace) for action in self.actionSpace}
        return likelihood


class RandomPolicyChooseAction:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def __call__(self, state):
        action = random.choice(self.actionSpace)
        return action


class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        assert actionL2Norm != 0
        action = action / actionL2Norm
        action *= self.actionMagnitude

        actionTuple = tuple(action)
        actionDist = {actionTuple: 1}
        return actionDist

class HeatSeekingDiscreteStochasticPolicy:
    def __init__(self, assumePrecision, actionSpace, getPredatorPos, getPreyPos):
        self.assumePrecision = assumePrecision
        self.actionSpace = actionSpace
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.vecToAngle = lambda vector: np.angle(complex(vector[0], vector[1]))
        self.degreeList = [self.vecToAngle(vector) for vector in self.actionSpace]

    def __call__(self, state):
        heatseekingVector = self.getPreyPos(state) - self.getPredatorPos(state)
        heatseekingDirection = self.vecToAngle(heatseekingVector)
        pdf = np.array([stats.vonmises.pdf(heatseekingDirection - degree, self.assumePrecision) * 2 for degree in self.degreeList])
        normProb = pdf / pdf.sum()
        actionDict = {action: prob for action, prob in zip(self.actionSpace, normProb)}
        return actionDict


class HeatSeekingContinuesDeterministicPolicy:
    def __init__(self, getPredatorPos, getPreyPos, actionMagnitude):
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos
        self.actionMagnitude = actionMagnitude

    def __call__(self, state):
        action = np.array(self.getPreyPos(state)) - np.array(self.getPredatorPos(state))
        actionL2Norm = np.linalg.norm(action, ord=2)
        if actionL2Norm != 0:
            action = action / actionL2Norm
            action *= self.actionMagnitude

        actionTuple = tuple(action)
        actionDist = {actionTuple: 1}
        return actionDist


class ActHeatSeeking:
    def __init__(self, actionSpace, calculateAngle, lowerBoundAngle, upperBoundAngle):
        self.actionSpace = actionSpace
        self.calculateAngle = calculateAngle
        self.lowerBoundAngle = lowerBoundAngle
        self.upperBoundAngle = upperBoundAngle

    def __call__(self, heatSeekingDirection):
        heatActionAngle = {mvmtVector: self.calculateAngle(heatSeekingDirection, mvmtVector)
                           for mvmtVector in self.actionSpace}

        angleWithinRange = lambda angle: self.lowerBoundAngle <= angle < self.upperBoundAngle
        movementAnglePair = zip(self.actionSpace, heatActionAngle.values())

        angleFilter = {movement: angleWithinRange(angle) for movement, angle in movementAnglePair}
        chosenActions = [action for action, index in zip(angleFilter.keys(), angleFilter.values()) if index]

        unchosenFilter = [action in chosenActions for action in self.actionSpace]
        unchosenActions = [action for action, index in zip(self.actionSpace, unchosenFilter) if not index]

        return [chosenActions, unchosenActions]


class HeatSeekingDiscreteStochasticPolicy:
    def __init__(self, rationalityParam, actHeatSeeking, getPredatorPos, getPreyPos):
        self.rationalityParam = rationalityParam
        self.actHeatSeeking = actHeatSeeking
        self.getPredatorPos = getPredatorPos
        self.getPreyPos = getPreyPos

    def __call__(self, state):
        predatorPosition = self.getPredatorPos(state)
        preyPosition = self.getPreyPos(state)

        heatSeekingDirection = np.array(preyPosition) - np.array(predatorPosition)
        chosenActions, unchosenActions = self.actHeatSeeking(heatSeekingDirection)

        chosenActionsLikelihood = {action: self.rationalityParam / len(chosenActions) for action in chosenActions}
        unchosenActionsLikelihood = {action: (1 - self.rationalityParam) / len(unchosenActions) for action in unchosenActions}

        heatSeekingActionLikelihood = {**chosenActionsLikelihood, **unchosenActionsLikelihood}
        heatSeekingSampleLikelihood = list(heatSeekingActionLikelihood.values())
        heatSeekingActionIndex = list(np.random.multinomial(1, heatSeekingSampleLikelihood)).index(1)
        chasingAction = list(heatSeekingActionLikelihood.keys())[heatSeekingActionIndex]
        return chasingAction
