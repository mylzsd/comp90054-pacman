# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'DummyAgent', second = 'DummyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        '''
        Your initialization code goes here, if you need any.
        '''
        # print "legal actions: ", gameState.getLegalActions(self.index)
        # print "agent state: ", gameState.getAgentState(self.index)
        # print "agent positions: ", gameState.getAgentPosition(self.index)
        # print "agent count: ", gameState.getNumAgents()
        # print "red food: ", gameState.getRedFood()
        # print "blue food: ", gameState.getBlueFood()
        # print "red capsules: ", gameState.getRedCapsules()
        # print "blue capsules: ", gameState.getBlueCapsules()
        # print "walls: ", gameState.getWalls()
        # print "red indices: ", gameState.getRedTeamIndices()
        # print "blue indices: ", gameState.getBlueTeamIndices()
        # print "agent distances: ", gameState.getAgentDistances()
        # print "capsules: ", gameState.getCapsules()
        # print gameState.data.layout.width, gameState.data.layout.height


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        # time.sleep(0.5)
        actions = gameState.getLegalActions(self.index)
        foodList = self.getFood(gameState).asList()
        # print self.getCurrentObservation()
        team = self.getTeam(gameState)
        closedPacman = []
        closedGhost = []
        hasPacman = False
        opIndices = self.getOpponents(gameState)
        distances = gameState.getAgentDistances()
        for index in opIndices:
            pos = gameState.getAgentPosition(index)
            sta = gameState.getAgentState(index)
            # print "opponent: ", index
            # print "Pacman: ", sta.isPacman
            if sta.isPacman: hasPacman = True
            conf = sta.configuration
            if conf != None:
                if sta.isPacman: 
                    closedPacman.append(conf.pos)
                else:
                    closedGhost.append(conf.pos) 
            #     print "position: ", conf.pos
            #     print "direction: ", conf.direction
            # print "distance: ", distances[index]

        successors = [(action, self.getSuccessor(gameState, action)) for action in actions]
        # if enemy Pacman closed
        if len(closedPacman) > 0:
            # print self.index, ": 1"
            if self.index == self.defender(gameState, team, closedPacman[0]):
                return self.bestDefAction(successors, closedPacman[0])
        # if enemy Pacman somewhere second agent go catch
        elif hasPacman and self.index == self.getTeam(gameState)[1]:
            # print self.index, ": 2"
            return self.randomDefAction(gameState, successors)
        # if enough food
        elif gameState.getAgentState(self.index).numCarrying > 5 or len(foodList) <= 0:
            # print self.index, ": 3"
            return self.bestRunAction(successors, closedGhost)
        # search food
        else:
            # print self.index, ": 4"
            return self.bestEatAction(gameState, successors, closedGhost)

        return random.choice(actions)

    def randomDefAction(self, gameState, successors):
        foodList = self.getFood(gameState).asList()
        food = random.choice(foodList)
        return self.bestDefAction(successors, food)

    def bestEatAction(self, gameState, successors, ghosts):
        minV = 9999
        foodList = self.getFood(gameState).asList()
        homes = self.closestHome(gameState, self.red)
        foodDis = []
        for f in foodList:
            minD = 9999
            for h in homes:
                minD = min(self.getMazeDistance(f, h), minD)
            foodDis.append((minD, (f)))
        sorted(foodDis)
        for a, s in successors:
            v = 0
            if a == "Stop": continue
            elif a == "West" and self.red: v = 1
            elif a == "East" and not self.red: v = 1
            myPos = s.getAgentPosition(self.index)
            v += 50 * len(foodList)
            if self.index == self.getTeam(s)[1]:
                # v += self.getMazeDistance(myPos, foodDis[-1][1])
                fDis = [self.getMazeDistance(myPos, f) for f in foodList]
                if len(fDis) > 0:
                    v += min(fDis)
            else:
                if len(foodDis) > 0:
                    v += self.getMazeDistance(myPos, foodDis[0][1])
            homeDis = [self.getMazeDistance(myPos, h) for h in homes]
            for g in ghosts:
                gDis = self.getMazeDistance(myPos, g)
                if gDis <= 4:
                    v += min(homeDis) - gDis * 3
            if v < minV:
                minV = v
                action = a
        return action

    def closestHome(self, gameState, isRed):
        l = []
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        if isRed: x = width / 2 - 1
        else: x = width / 2
        for y in range(height):
            if not gameState.hasWall(x, y):
                l.append((x, y))
        return l

    def bestRunAction(self, successors, ghosts):
        maxV = -9999
        for a, s in successors:
            if a == "Stop": continue
            myPos = s.getAgentPosition(self.index)
            v = 0
            for g in ghosts:
                v += self.getMazeDistance(myPos, g)
            backList = self.closestHome(s, self.red)
            backDist = [self.getMazeDistance(myPos, p) for p in backList]
            v -= min(backDist) * 3
            if v > maxV:
                maxV = v
                action = a
        return action

    def bestDefAction(self, successors, pos):
        minDis = 9999
        for a, s in successors:
            d = self.getMazeDistance(s.getAgentPosition(self.index), pos)
            if d < minDis:
                minDis = d
                action = a
        return action

    def defender(self, gameState, team, pos):
        minDis = 9999
        for i in team:
            d = self.getMazeDistance(gameState.getAgentPosition(i), pos)
            if d < minDis:
                minDis = d
                closer = i
        return closer

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

