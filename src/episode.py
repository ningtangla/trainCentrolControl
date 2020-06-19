import numpy as np
import random
import pygame as pg


class ForwardOneStep:
    def __init__(self, transitionFunction, rewardFunction):
        self.transitionFunction = transitionFunction
        self.rewardFunction = rewardFunction

    def __call__(self, state, sampleAction):
        action = sampleAction(state)
        nextState = self.transitionFunction(state, action)
        reward = self.rewardFunction(state, action, nextState)
        return (state, action, nextState, reward)


class ForwardMultiAgentsOneStep:
    def __init__(self, transitionFunction, rewardFunctionList):
        self.transitionFunction = transitionFunction
        self.rewardFunctionList = rewardFunctionList

    def __call__(self, state, sampleAction):
        action = sampleAction(state)
        nextState = self.transitionFunction(state, action)
        rewards = [rewardFunction(state, action, nextState) for rewardFunction in self.rewardFunctionList]
        return (state, action, nextState, rewards)


class SampleTrajectory:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep

    def __call__(self, sampleAction):
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 0))
                break
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState

        return trajectory


class SampleTrajectoryWithRender:
    def __init__(self, maxRunningSteps, isTerminal, resetState, forwardOneStep, render, renderOn):
        self.maxRunningSteps = maxRunningSteps
        self.isTerminal = isTerminal
        self.resetState = resetState
        self.forwardOneStep = forwardOneStep
        self.render = render
        self.renderOn = renderOn

    def __call__(self, sampleAction):
        state = self.resetState()
        while self.isTerminal(state):
            state = self.resetState()

        trajectory = []
        for runningStep in range(self.maxRunningSteps):
            if self.isTerminal(state):
                trajectory.append((state, None, None, 0))
                break
            if self.renderOn:
                self.render(state)
            state, action, nextState, reward = self.forwardOneStep(state, sampleAction)
            trajectory.append((state, action, nextState, reward))
            state = nextState

        return trajectory


class Render():
    def __init__(self, numOfAgent, posIndex, screen, screenColor, circleColorList, circleSize, saveImage, saveImageDir):
        self.numOfAgent = numOfAgent
        self.posIndex = posIndex
        self.screen = screen
        self.screenColor = screenColor
        self.circleColorList = circleColorList
        self.circleSize = circleSize
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, state):
        for j in range(1):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
            self.screen.fill(self.screenColor)
            for i in range(self.numOfAgent):
                agentPos = state[i][self.posIndex]
                pg.draw.circle(self.screen, self.circleColorList[i], [np.int(
                    agentPos[0]), np.int(agentPos[1])], self.circleSize)
            pg.display.flip()
            pg.time.wait(100)

            if self.saveImage == True:
                filenameList = os.listdir(self.saveImageDir)
                pg.image.save(self.screen, self.saveImageDir + '/' + str(len(filenameList)) + '.png')
