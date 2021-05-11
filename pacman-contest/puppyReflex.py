from InferenceModule import InferenceModule
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    depth = 1
    time = 1200
    friends = {}
    enemies = {}
    maze_dist = {}
    safe_pt = []

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def feature_food_dist(self, gameState, food):
        pos = gameState.getAgentPosition(self.index)
        min_food_dist = 99
        max_food_dist = -1
        nf = None
        for f in food:
            d = self.getMazeDistance(pos, f)
            if d < min_food_dist:
                nf = f
                min_food_dist = d
            max_food_dist = max(max_food_dist, d)
        dist = prim(self, nf, food)
        dist /= len(food)
        return [dist, min_food_dist, max_food_dist]

    def getFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        pos = gameState.getAgentPosition(self.index)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if self.red:
            enemy_capsule = gameState.getBlueCapsules()
        else:
            enemy_capsule = gameState.getRedCapsules()

        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['capsulesLeft'] = len(enemy_capsule)
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.scaredTimer == 0 and a.getPosition() != None]

        if len(defenders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
            features['defenderDistance'] = min(min(dists), 3)
            if not myState.isPacman:
                features['defenderDistance'] = 3

        # Compute distance to the nearest food
        fl = []
        for f in foodList:
            if len(defenders) == 0:
                fl = foodList
            for d in defenders:
                if self.getMazeDistance(f, d.getPosition()) > 1:
                    fl.append(f)

        features['successorScore'] = -len(foodList)
        food_dist = self.feature_food_dist(successor, fl)

        features['distanceToFood'] = 0.1 * food_dist[0] + 0.9 * food_dist[1]
        if len(enemy_capsule) > 0:
            features['distanceToCapsule'] = min([self.getMazeDistance(myPos, c) for c in enemy_capsule])

        grid = gameState.data.layout
        width = grid.width
        height = grid.height
        mid_dist = 99
        if self.red:
            x = width / 2 - 1
        else:
            x = width / 2
        for i in range(0, height):
            if not gameState.hasWall(x, i):
                mid_dist = min(mid_dist, self.getMazeDistance(myPos, (x, i)))
        features['mid_dist'] = mid_dist

        features['food_carrying'] = gameState.getAgentState(self.index).numCarrying

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        myState = gameState.getAgentState(self.index)
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        numCarrying = gameState.getAgentState(self.index).numCarrying
        numDefenders = len([a for a in enemies if not a.isPacman and myState.isPacman])
        w = util.Counter()
        w['capsulesLeft'] = -100 * numDefenders
        w['successorScore'] = 100
        w['distanceToFood'] = -1
        w['food_carrying'] = -1
        w['mid_dist'] = -0.5 * numCarrying * numDefenders * numDefenders - 0.1 * numDefenders
        w['stop'] = -3
        w['reverse'] = -2
        w['defenderDistance'] = 10
        w['distanceToCapsule'] = -1 * numDefenders
        return w


class DefensiveReflexAgent(ReflexCaptureAgent):
    dist_matrix = util.Counter()

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.initial_dist_matrix(gameState)
        self.enemies = self.getOpponents(gameState)
        self.infer = InferenceModule(self)
        self.infer.initialize(gameState)

    def chooseAction(self, gameState):
        self.updateEnemyPosDist()
        # self.displayEnemyPosDist()
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [(i, successor.getAgentState(i)) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a[1].isPacman]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            eid = invaders[0][0]
            nc = 0
            dists = []
            enemy_infer_pos = self.getEnemyPos(gameState)
            for a in invaders:
                if a[1].getPosition() is None:
                    if a[0] < 2:
                        dists.append(self.getMazeDistance(myPos, enemy_infer_pos[0]))
                    else:
                        dists.append(self.getMazeDistance(myPos, enemy_infer_pos[1]))
                else:
                    dists.append(self.getMazeDistance(myPos, a[1].getPosition()))

                num_carrying = a[1].numCarrying
                if num_carrying > nc:
                    nc = num_carrying
                    eid = a[0]
                if eid < 2:
                    features['mostCarryEnemyDist'] = self.getMazeDistance(myPos, enemy_infer_pos[0])
                else:
                    features['mostCarryEnemyDist'] = self.getMazeDistance(myPos, enemy_infer_pos[1])
            features['invaderDistance'] = min(dists)

        food = self.getFood(gameState).asList()
        for f in food:
            if f[0] == myPos[0] and f[1] == myPos[1]:
                features['food_1_step'] = 1

        grid = gameState.data.layout
        width = grid.width
        height = grid.height
        mid_dist = 99
        if self.red:
            x = width / 2 - 1
        else:
            x = width / 2
        for i in range(0, height):
            if not gameState.hasWall(x, i):
                mid_dist = min(mid_dist, (0.8 * self.getMazeDistance(myPos, (x, i)) + 0.2 * abs(8 - i)))
        features['mid_dist'] = mid_dist

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        w = util.Counter()
        w['numInvaders'] = -1000
        w['onDefense'] = 100
        w['invaderDistance'] = -5
        w['mostCarryEnemyDist'] = -5
        w['stop'] = -100
        w['reverse'] = -2
        w['mid_dist'] = -1
        w['food_1_step'] = 0
        return w

    def initial_dist_matrix(self, gameState):
        grid = gameState.data.layout
        width = grid.width
        height = grid.height
        for i in range(0, width):
            for j in range(0, height):
                if gameState.hasWall(i, j):
                    continue
                else:
                    for p in range(0, width):
                        for q in range(0, height):
                            if gameState.hasWall(p, q):
                                continue
                            else:
                                d = self.getMazeDistance((i, j), (p, q))
                                self.dist_matrix[((i, j), (p, q))] = d

    def getEnemyPos(self, gameState):
        res = []
        for dist in self.enemy_pos_dist:
            if len(dist) == 0:
                if len(res) == 0:
                    index = self.enemies[0]
                    pos = gameState.getInitialAgentPosition(index)
                else:
                    index = self.enemies[1]
                    pos = gameState.getInitialAgentPosition(index)
            else:
                pos = max(dist.items(), key=lambda x: x[1])[0]
            res.append(pos)
        return res

    def updateEnemyPosDist(self):
        last_obs = self.getPreviousObservation()
        obs = self.getCurrentObservation()
        self.infer.updateBelief(obs, last_obs)
        self.enemy_pos_dist = []
        for enemy in self.enemies:
            self.enemy_pos_dist.append(self.infer.beliefs[enemy])

    def displayEnemyPosDist(self):
        self.displayDistributionsOverPositions(self.enemy_pos_dist)


def prim(agent, nearest, food):
    if nearest is None:
        return -1
    for i in range(0, len(food)):
        if food[i][0] == nearest[0] and food[i][1] == nearest[1]:
            start = i
            break
    edge = {}
    vnum = len(food)
    for i in range(0, vnum):
        edge[i] = {}
        for j in range(0, vnum):
            edge[i][j] = agent.getMazeDistance(food[i], food[j])
    lowcost = {}
    addvnew = {}
    adjecent = {}
    sumweight = 0
    i, j = 0, 0
    for i in range(0, vnum):
        lowcost[i] = edge[start][i]
        addvnew[i] = -1
    addvnew[start] = 0
    adjecent[start] = start
    for i in range(0, vnum - 1):
        min = 999999
        v = -1
        for j in range(0, vnum):
            if addvnew[j] == -1 and lowcost[j] < min:
                min = lowcost[j]
                v = j
        if v != -1:
            addvnew[v] = 0
            sumweight += lowcost[v]
            for j in range(0, vnum):
                if addvnew[j] == -1 and edge[v][j] < lowcost[j]:
                    lowcost[j] = edge[v][j]
                    adjecent[j] = v
    return sumweight