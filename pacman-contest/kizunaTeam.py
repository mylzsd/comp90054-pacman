# baselineTeam.py
# ---------------
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


import random
import util
from compiler.ast import flatten

# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
import InferenceModule
import game
from DQN import DQN
from captureAgents import CaptureAgent
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='KizunaAI', second='KizunaAI'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


#############
# CONSTANTS #
#############
global_cnt = 0
FEATURE_SIZE = 39
ACTION_SIZE = 5
MEMORY_CAPACITY = 16384
PATH = "/home/sakuya/ramdisk/"


#############
# FUNCTIONS #
#############
def code_action(action_code):
    dic = {0: game.Directions.EAST,
           1: game.Directions.WEST,
           2: game.Directions.NORTH,
           3: game.Directions.SOUTH,
           4: game.Directions.STOP}
    return dic[action_code]


def action_code(action):
    dic = {game.Directions.EAST: 0,
           game.Directions.WEST: 1,
           game.Directions.NORTH: 2,
           game.Directions.SOUTH: 3,
           game.Directions.STOP: 4}
    return dic[action]

##########
# Agents #
##########
class DummyAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return random.choice(actions)


class KizunaAI(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """
    enemies = util.Counter()
    ep_r = 0
    time_left = 1200
    enemy_pos_dist = []
    safe_pt = []

    def registerInitialState(self, gameState):
        grid = gameState.data.layout
        width = grid.width
        height = grid.height
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.dist_matrix = util.Counter()
        tool = Tools()
        tool.initial_dist_matrix(self, gameState)
        eval_file = PATH + "kizuna_eval" + str(self.index) + ".pkl"
        tar_file = PATH + "kizuna_tar" + str(self.index) + ".pkl"
        mem_file = PATH + "kizuna_mem" + str(self.index) + ".bin"
        counter_file = PATH + "kizuna_cnt" + str(self.index) + ".txt"
        self.enemies = self.getOpponents(gameState)
        self.infer = InferenceModule.InferenceModule(self)
        self.infer.initialize(gameState)
        self.state = gameState
        self.dqn = DQN(FEATURE_SIZE, ACTION_SIZE, eval_file, tar_file, mem_file)
        self.dqn.read_counter(counter_file)
        if self.red:
            s_col = width / 2 - 1
        else:
            s_col = width / 2
        for j in range(0, height):
            if not gameState.hasWall(s_col, j):
                self.safe_pt.append((s_col, j))


    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        prev_state = self.getPreviousObservation()
        curr_state = self.getCurrentObservation()
        self.time_left = curr_state.data.timeleft
        if prev_state is None:
            prev_state = curr_state
        self.updateEnemyPosDist()
        self.displayEnemyPosDist()
        s = self.extractStates(curr_state)
        a = code_action(self.dqn.choose_action(s))
        actions = curr_state.getLegalActions(self.index)
        r = self.getReward(prev_state, curr_state, a)
        if a not in actions:
            action = random.choice(actions)
        else:
            action = a
        next_state = curr_state.generateSuccessor(self.index, action)
        s_ = self.extractStates(next_state)
        self.dqn.store_transition(s, action_code(a), r, s_)
        self.ep_r += r
        if self.dqn.memory_counter > MEMORY_CAPACITY:
            self.dqn.learn()

        eval_file = PATH + "kizuna_eval" + str(self.index) + ".pkl"
        tar_file = PATH + "kizuna_tar" + str(self.index) + ".pkl"
        mem_file = PATH + "kizuna_mem" + str(self.index) + ".bin"
        counter_file = PATH + "kizuna_cnt" + str(self.index) + ".txt"
        self.dqn.save(eval_file, tar_file, mem_file)
        self.dqn.write_counter(counter_file)

        return action

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

    def getReward(self, prev_state, curr_state, action):
        r = 0
        if action not in curr_state.getLegalActions(self.index):
            # penalty for illegal action
            r = r - 100
        prev_score = self.calcPotentialScore(prev_state)
        curr_score = self.calcPotentialScore(curr_state)
        r += (curr_score - prev_score)
        return r

    def getFeatures(self, gameState, action):
        team = self.getTeam(gameState)
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        features['actionValidity'] = self.getActionValidity(gameState)
        features['food_dist'] = self.feature_food_dist(gameState)
        features['wall_3x3'] = self.feature_wall_3x3(gameState)
        features['is_pacman'] = gameState.getAgentState(self.index).isPacman
        features['enemies'] = self.feature_enemies(gameState)
        features['time_left'] = self.time_left
        features['f_distant'] = self.feature_friend_dist(gameState)
        features['f_scared'] = [gameState.getAgentState(a).scaredTimer for a in team]
        features['safe_dist'] = self.feature_home_pos(gameState)
        features['food_cnt'] = self.feature_food_cnt(gameState)
        features['food_carrying'] = [gameState.getAgentState(a).numCarrying for a in team]
        return features

    def feature_friend_dist(self, gameState):
        team = self.getTeam(gameState)
        pos = gameState.getAgentPosition(self.index)
        for a in team:
            if a != self.index:
                p = gameState.getAgentPosition(a)
                return self.getMazeDistance(p, pos)
        return 0

    def feature_enemies(self, gameState):
        enemy_pos = self.getEnemyPos()
        pos = gameState.getAgentPosition(self.index)
        enemy_dist = [self.getMazeDistance(e_pos, pos) for e_pos in enemy_pos]
        enemy_ispacman = [gameState.getAgentState(e).isPacman for e in self.enemies]
        enemy_food_carrying = [gameState.getAgentState(e).numCarrying for e in self.enemies]
        enemy_scared = [gameState.getAgentState(e).scaredTimer for e in self.enemies]
        enemy_food_mst = []
        myfood = self.getFoodYouAreDefending(gameState).asList()
        for e_pos in enemy_pos:
            min_food_dist = 99
            nf = None
            for f in myfood:
                d = self.getMazeDistance(e_pos, f)
                if d < min_food_dist:
                    nf = f
                    min_food_dist = d
            dist = Tools.prim(Tools(), self, gameState, nf, myfood)
            enemy_food_mst.append(dist)
        return enemy_dist + enemy_ispacman + enemy_food_mst + enemy_food_carrying + enemy_scared

    def getActionValidity(self, gameState):
        res = []
        legal_actions = gameState.getLegalActions(self.index)
        for i in range(0, 5):
            if code_action(i) in legal_actions:
                res.append(1)
            else:
                res.append(0)
        return res

    def feature_food_cnt(self, gameState):
        self_food_cnt = len(self.getFood(gameState).asList())
        enemy_food_cnt = len(self.getFoodYouAreDefending(gameState).asList())
        return [self_food_cnt, enemy_food_cnt]

    def feature_food_dist(self, gameState):
        food = self.getFood(gameState).asList()
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
        dist = Tools.prim(Tools(), self, gameState, nf, food)
        return [dist, min_food_dist, max_food_dist]

    def feature_wall_3x3(self, gameState):
        self_pos = gameState.getAgentPosition(self.index)
        x, y = self_pos
        surr = []
        for a in range(x-1, x+2):
            for b in range(y-1, y+2):
                surr.append((a, b))
        iswall = [int(gameState.hasWall(posx, posy)) for (posx, posy) in surr]
        result = iswall
        return result

    def feature_home_pos(self, gameState):
        min_safe_dist = 99.0
        pos = gameState.getAgentPosition(self.index)
        for s in self.safe_pt:
            d = self.getMazeDistance(pos, s)
            min_safe_dist = min(min_safe_dist, d)
        if gameState.getAgentState(self.index).isPacman:
            return [min_safe_dist, 0]
        else:
            return [0, min_safe_dist]

    def extractStates(self, gameState):
        features = self.getFeatures(gameState, game.Directions.STOP)
        return flatten(features.values())

    def updateEnemyPosDist(self):
        last_obs = self.getPreviousObservation()
        obs = self.getCurrentObservation()
        self.infer.updateBelief(obs, last_obs)
        self.enemy_pos_dist = []
        for enemy in self.enemies:
            self.enemy_pos_dist.append(self.infer.beliefs[enemy])

    def displayEnemyPosDist(self):
        self.displayDistributionsOverPositions(self.enemy_pos_dist)

    def getEnemyPos(self):
        res = []
        for dist in self.enemy_pos_dist:
            pos = max(dist.items(), key=lambda x: x[1])[0]
            res.append(pos)
        return res

    def calcPotentialScore(self, state):
        team = self.getTeam(state)
        enemy_food_carrying = [state.getAgentState(e).numCarrying for e in self.enemies]
        enemy_food_returned = [state.getAgentState(e).numReturned for e in self.enemies]
        self_food_carrying = [state.getAgentState(a).numCarrying for a in team]
        self_food_returned = [state.getAgentState(a).numReturned for a in team]
        r_e = sum(enemy_food_carrying) + 2 * sum(enemy_food_returned)
        r_s = sum(self_food_carrying) + 2 * sum(self_food_returned)
        return r_s - r_e

class Tools:

    def initial_dist_matrix(self, agent, gameState):
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
                                d = agent.getMazeDistance((i, j), (p, q))
                                agent.dist_matrix[((i, j),(p, q))] = d

    def prim(self, agent, state, nearest, food):
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


if __name__ == "__main__":
    print "a"
