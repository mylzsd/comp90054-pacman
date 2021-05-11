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
            enemy_capsule = successor.getBlueCapsules()
        else:
            enemy_capsule = successor.getRedCapsules()

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
        w['capsulesLeft'] = -100 * numDefenders - 1
        w['successorScore'] = 100
        w['distanceToFood'] = -1
        w['food_carrying'] = -1
        w['mid_dist'] = -0.5 * numCarrying * numDefenders * numDefenders - 0.1 * numDefenders - 0.1 * numCarrying
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
        self.displayEnemyPosDist()
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
        w['food_1_step'] = 110
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


class InferenceModule:

    def __init__(self, me):
        self.me = me
        self.index = me.index
        self.obs = util.Counter()
        self.assumption = util.Counter()
        self.beliefs = util.Counter()
        self.legalPositions = util.Counter()
        self.dist_matrix = self.reverse_dict(me.dist_matrix)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        for enemy in self.me.enemies:
            p_notwall = gameState.getWalls().asList(False)
            self.legalPositions[enemy] = [p for p in p_notwall if p[1] > 1]
        self.elapseTime(gameState)

    def initializeUniformly(self, gameState, ghostIndex):
        emissionModel = self.getObservationDistribution(gameState)
        self_pos = gameState.getAgentPosition(self.me.index)
        self.beliefs[ghostIndex] = util.Counter()
        allPossible = util.Counter()
        for p in self.legalPositions[ghostIndex]:
            trueDistance = util.manhattanDistance(p, self_pos)
            if emissionModel[ghostIndex][trueDistance] > 0:
                allPossible[p] = emissionModel[ghostIndex][trueDistance]
        self.beliefs[ghostIndex] = allPossible
        self.beliefs[ghostIndex].normalize()

    def getObservationDistribution(self, gameState):
        result = util.Counter()
        self_pos = gameState.getAgentPosition(self.me.index)
        noisy_dist = gameState.getAgentDistances()
        for enemy in self.me.enemies:
            enemy_info = gameState.getAgentState(enemy)
            if enemy_info.configuration is not None:
                enemy_pos = gameState.getAgentPosition(enemy)
                e_dist = util.manhattanDistance(self_pos, enemy_pos)
                result[enemy] = util.Counter()
                result[enemy][e_dist] = 1.0
            else:
                result[enemy] = util.Counter()
                nd_e = noisy_dist[enemy]
                for td_e in range(nd_e - 10, nd_e + 10):
                    prob = gameState.getDistanceProb(td_e, nd_e)
                    if prob > 0:
                        result[enemy][td_e] = prob
        return result

    def getLegalPositions(self, ghostIndex):
        return self.legalPositions[ghostIndex]

    def updateBelief(self, currState, prevState):
        flag = False
        self_pos = currState.getAgentPosition(self.me.index)
        p_food = self.me.getFoodYouAreDefending(currState).asList()
        cp = currState.getCapsules()
        lost_food = []
        lost_capsules = []
        prev_belief = self.beliefs.copy()
        if self.me.red:
            s_col = currState.data.layout.width / 2 - 1
            e_col = s_col + 1
        else:
            s_col = currState.data.layout.width / 2
            e_col = s_col - 1
        if prevState is not None:
            lp_food = self.me.getFoodYouAreDefending(prevState).asList()
            lcp = prevState.getCapsules()
            lost_capsules = [c for c in lcp if c not in cp]
            if self.me.red:
                lost_capsules = filter(lambda (x, y): x < e_col, lost_capsules)
            else:
                lost_capsules = filter(lambda (x, y): x > e_col, lost_capsules)
            lost_food = [f for f in lp_food if f not in p_food]

        for enemy in self.me.enemies:
            if prevState is None:
                continue
            prev_enemy_state = prevState.getAgentState(enemy)
            curr_enemy_state = currState.getAgentState(enemy)
            # Case 1: killed an enemy pacman
            # prev, pacman, near;
            # curr, ghost, far;
            if prev_enemy_state.isPacman and not curr_enemy_state.isPacman \
                    and prev_enemy_state.configuration is not None and curr_enemy_state.configuration is None:
                state = currState.deepCopy()
                self.setGhostPositions(state, (enemy, state.getInitialAgentPosition(enemy)))
                self.elapseTime(state)
                flag = True
            # Case 2: enemy passing cross line
            if prev_enemy_state.isPacman and not curr_enemy_state.isPacman:  # leave
                for pos, value in self.beliefs[enemy].items():
                    if pos[0] != e_col:
                        self.beliefs[enemy][pos] = 0
                self.beliefs[enemy].normalize()
            elif not prev_enemy_state.isPacman and curr_enemy_state.isPacman:  # enter
                for pos, value in self.beliefs[enemy].items():
                    if pos[0] != s_col:
                        self.beliefs[enemy][pos] = 0
                self.beliefs[enemy].normalize()
        # Case 3: we lost some food
        if len(lost_food) > 0:  # some food are disappeared
            for f in lost_food:
                flag = self.observe_lostfood(prevState, currState, f, self_pos)
        if len(lost_capsules) > 0:
            for c in lost_capsules:
                flag = self.observe_lostfood(prevState, currState, c, self_pos)
        if not flag:
            self.elapseTime(currState)
        for e in self.beliefs.keys():
            if len(self.beliefs[e]) == 0:
                self.beliefs[e] = prev_belief[e]

    def observe_lostfood(self, lastState, currState, f, self_pos):
        currying_prev = map(lambda e: lastState.getAgentState(e).numCarrying, self.me.enemies)
        currying_curr = map(lambda e: currState.getAgentState(e).numCarrying, self.me.enemies)
        currying_diff = map(lambda a, b: a - b, currying_curr, currying_prev)
        currying_diff = map(lambda e, n: {e: n}, self.me.enemies, currying_diff)
        e_id = filter(lambda i: i.values()[0] > 0, currying_diff)
        if len(e_id) == 1:
            state = currState.deepCopy()
            self.setGhostPositions(state, (e_id[0].keys()[0], f))
            self.elapseTime(state)
            return True
        else:
            ef_poss = map(lambda e: self.beliefs[e][f], self.beliefs)
            ef_poss = zip(self.me.enemies, ef_poss)
            ef_poss = filter(lambda e: e[1] > 0, ef_poss)
            if len(ef_poss) == 2:
                state = currState.deepCopy()
                if ef_poss[0][1] > ef_poss[1][1]:
                    e_id = ef_poss[0][0]
                else:
                    e_id = ef_poss[1][0]
                self.setGhostPositions(state, (e_id, f))
                self.elapseTime(state)
                return True
            elif len(ef_poss) == 1:
                state = currState.deepCopy()
                self.setGhostPositions(state, (ef_poss[0][0], f))
                self.elapseTime(state)
                return True
            else:
                f_dist = util.manhattanDistance(f, self_pos)
                emissionModel = self.getObservationDistribution(currState)
                poss = map(lambda e: emissionModel[e][f_dist], emissionModel)
                poss = zip(self.me.enemies, poss)
                poss = filter(lambda e: e[1] > 0, poss)
                if len(poss) == 2:
                    state = currState.deepCopy()
                    if poss[0][1] > poss[1][1]:
                        e_id = poss[0][0]
                    elif poss[1][1] > poss[0][1]:
                        e_id = poss[1][0]
                    else:
                        return False
                    self.setGhostPositions(state, (e_id, f))
                    self.elapseTime(state)
                    return True
                elif len(poss) == 1:
                    state = currState.deepCopy()
                    self.setGhostPositions(state, (poss[0][0], f))
                    self.elapseTime(state)
                    return True
        return False

    def observe(self, gameState):
        emissionModel = self.getObservationDistribution(gameState)
        self_pos = gameState.getAgentPosition(self.me.index)
        belief = self.beliefs.copy()
        allPossible = util.Counter()
        for enemy in self.me.enemies:
            enemy_info = gameState.getAgentState(enemy)
            if enemy_info.configuration is not None:
                enemy_pos = gameState.getAgentPosition(enemy)
                allPossible[enemy] = util.Counter()
                allPossible[enemy][enemy_pos] = 1.0
            else:
                allPossible[enemy] = util.Counter()
                for p in self.legalPositions[enemy]:
                    trueDistance = util.manhattanDistance(p, self_pos)
                    if emissionModel[enemy][trueDistance] > 0:
                        prob = emissionModel[enemy][trueDistance] * self.beliefs[enemy][p]
                        if prob > 0:
                            allPossible[enemy][p] = emissionModel[enemy][trueDistance] \
                                                    * belief[enemy][p]
                allPossible[enemy].normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        state_cpy = gameState.deepCopy()
        self.observe(gameState)
        belief = self.beliefs.copy()
        allPossible = util.Counter()
        for enemy in self.me.enemies:
            if sum(belief[enemy].values()) == 0:
                self.initializeUniformly(gameState, enemy)
                belief = self.beliefs.copy()
            allPossible[enemy] = util.Counter()
            for oldPos in self.legalPositions[enemy]:
                newPosDist = self.getPositionDistribution(self.setGhostPositions(state_cpy, (enemy, oldPos)), enemy)
                for newPos, prob in newPosDist.items():
                    newPos = (int(newPos[0]), int(newPos[1]))
                    newProb = prob * belief[enemy][oldPos]
                    if newProb > 0:
                        allPossible[enemy][newPos] += prob * belief[enemy][oldPos]
            allPossible[enemy].normalize()
        self.beliefs = allPossible

    def getPositionDistribution(self, gameState, agentIndex):
        ghostPosition = self.getGhostPosition(agentIndex, gameState)  # The position you set
        actions = gameState.getLegalActions(agentIndex)
        actionDist = []
        for a in actions:
            actionDist.append((a, 1.0 / len(actions)))
        dist = util.Counter()
        for action, prob in actionDist:
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            successorPosition = (int(successorPosition[0]), int(successorPosition[1]))
            dist[successorPosition] = prob
        return dist

    def getBeliefDistribution(self):
        return self.beliefs

    def observeState(self, gameState):
        distances = gameState.getAgentDistances()
        obs = util.Counter()
        for enemy in self.me.enemies:
            obs[enemy] = distances[enemy]
        self.obs = obs
        self.observe(gameState)

    def reverse_dict(self, matrix):
        result = util.Counter()
        keys = matrix.keys()
        for k in keys:
            v = matrix[k]
            if v not in result.keys():
                result[v] = [k]
            else:
                result[v].append(k)
        return result

    def getGhostPosition(self, ghostIndex, gameState):
        return gameState.getAgentPosition(ghostIndex)

    def setGhostPositions(self, gameState, ghostPositions):
        index, pos = ghostPositions
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState
