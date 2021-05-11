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
import json, ast, math


GAMMA = 0.9
EPSILON = 0.1

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, first = 'Mylzsd', second = 'Mylzsd'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
# python capture.py -r myTeam.py -b myTeam.py -l RANDOM -q -n 20 --delay-step 0
# python qLearning.py

class Mylzsd(CaptureAgent):
    dist_matrix = util.Counter()

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.initSetup(gameState)
        # infer module
        self.infer = InferenceModule(self)
        self.infer.initialize(gameState)
        # [cap_dist, food_dist, back, ghost_near, ghost_far, pacman_near, pacman_far]
        self.weights = [1.11342, 2.05454, 47.06378, 2.03915, -2.65936, 69.36933, 11.03093]
        # try:
        #     with open("weights.txt", "r") as f:
        #         self.weights = ast.literal_eval(f.read())
        # except IOError:
        #     self.weights = [0.03427, 1.59158, 28.16338, 2.24028, -1.81024, 49.92805, 11.05009]
       

    def initSetup(self, state):
        # variables for inference module
        self.initial_dist_matrix(state)
        self.enemies = self.getOpponents(state)
        # static map info
        self.init_state = state
        self.total_time = state.data.timeleft
        self.width = state.data.layout.width
        self.height = state.data.layout.height
        self.max_cap = len(self.getCapsules(state))
        self.max_food = len(self.getFood(state).asList())
        self.max_dist = (self.height - 1) * (self.width - 2) / 2
        self.static_map = [[0 for _ in range(self.height)] for _ in range(self.width)]
        self.boarder = [(self.width / 2, y) for y in range(self.height)]
        self.boarder = sorted(filter(lambda p: not state.hasWall(p[0], p[1]), self.boarder))
        # self.fillStaticMap()


    def fillStaticMap(self):
        left_homes = self.closestHome(True)
        right_homes = self.closestHome(False)
        for i in range(1, self.width / 2):
            x_l = i
            x_r = i + self.width / 2 - 1
            for y in range(1, self.height):
                self.static_map[x_l][y] = self.fillHelper((x_l, y), right_homes)
                self.static_map[x_r][y] = self.fillHelper((x_r, y), left_homes)


    def fillHelper(self, src, homes):
        if self.init_state.hasWall(src[0], src[1]):
            return 0
        neighbors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        ret = 0
        for n in neighbors:
            x = src[0] + n[0]
            y = src[1] + n[1]
            if self.init_state.hasWall(x, y): continue
            if self.fillBFS((x, y), [src], homes):
                ret += 1
        return ret


    def fillBFS(self, src, visited, tars):
        neighbors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        q = [src]
        while len(q) > 0:
            l = len(q)
            for i in range(l):
                s = q.pop(0)
                for n in neighbors:
                    t = (s[0] + n[0], s[1] + n[1])
                    if t in tars: return True
                    if t in visited: continue
                    if self.init_state.hasWall(t[0], t[1]): continue
                    q.append(t)
                    visited.append(t)
        return False


    def chooseAction(self, state):
        # self.getRecord()
        # update infer module
        self.updateEnemyPosDist()
        # select best action
        actions = state.getLegalActions(self.index)
        # with e probability choose random action, turn on when training
        # if random.random() < EPSILON:
        #     return random.choice(actions)
        best_actions = []
        best_q = float('-inf')
        # print(self.index)
        # print(state.data.timeleft)
        for action in actions:
            # stop at the first step
            # if action == "Stop": and state.data.timeleft > self.total_time - 4:
            #     return action
            if action == "Stop": continue
            state_p = self.getSuccessor(state, action)
            feature_p = self.getFeature(state_p)
            # print("\t%s" % action)
            # print("\t%s" % str(feature_p))
            reward = self.getReward(state, state_p)
            next_q = self.qFunc(feature_p)
            temp_q = reward + GAMMA * next_q
            if temp_q > best_q:
                best_q = temp_q
                best_actions = [action]
            elif temp_q == best_q:
                best_actions.append(action)
        return random.choice(best_actions)


    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            successor = successor.generateSuccessor(self.index, action)
        return successor


    def closestHome(self, is_red):
        homes = []
        if is_red: x = self.width / 2 - 1
        else: x = self.width / 2
        for y in range(self.height):
            if not self.init_state.hasWall(x, y):
                homes.append((x, y))
        return homes


    def realDistance(self, pos, tars, ghost):
        ret = 1
        neighbors = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        q = [pos]
        visited = [pos]
        while len(q) > 0:
            l = len(q)
            for i in range(l):
                src = q.pop(0)
                for n in neighbors:
                    tar = (src[0] + n[0], src[1] + n[1])
                    if tar in tars: return ret
                    if tar in visited: continue
                    if self.init_state.hasWall(tar[0], tar[1]): continue
                    # check near ghost
                    ghost_dist = [self.getMazeDistance(tar, g) for g in ghost]
                    if len(ghost) > 0 and min(ghost) <= 1: continue
                    q.append(tar)
                    visited.append(tar)
            ret += 1
        return self.max_dist


    def getTargetFood(self, state, ghost):
        my_pos = state.getAgentPosition(self.index)
        food_list = self.getFood(state).asList()
        if len(food_list) == 0: return None
        length = len(self.boarder)
        if self.index < 2:
            # first agent closed to top
            if len(food_list) == 1: return None
            index = int(length / 4 * 3)
        else:
            # second agent closed to bot
            index = int(length / 4)
        src = self.boarder[index]
        food_list = sorted(food_list, key = lambda x: self.getMazeDistance(src, x))
        for food in food_list:
            if self.realDistance(my_pos, [food], ghost) < self.max_dist:
                return food
        return None


    def getFeature(self, state):
        # [cap_dist, food_dist, back, ghost_near, ghost_far, pacman_near, pacman_far]
        features = [0.0] * 7
        my_pos = state.getAgentPosition(self.index)
        my_state = state.getAgentState(self.index)
        homes = self.closestHome(self.red)
        # fetch enemies' info
        all_ghost = []
        ghost = []
        pacman = []
        enemy_infer_pos = self.getEnemyPos(state)
        for e in self.enemies:
            s = state.getAgentState(e)
            conf = s.configuration
            if conf:
                p = conf.pos
                d = self.getMazeDistance(my_pos, conf.pos)
            else:
                p = enemy_infer_pos[0] if e < 2 else enemy_infer_pos[1]
                d = self.getMazeDistance(my_pos, p)
            if not s.isPacman:
                all_ghost.append(p)
            if s.isPacman:
                if my_state.scaredTimer <= d / 2:
                    pacman.append((d, s, p))
            elif d <= 4 and s.scaredTimer <= 5:
                if my_state.isPacman or d <= 2:
                    ghost.append((d, s, p))
        # minimum distance to home
        if my_state.isPacman:
            home_dist = self.realDistance(my_pos, homes, all_ghost)
        else:
            home_dist = 0
        # nearest capsule
        capsule_list = self.getCapsules(state)
        if len(capsule_list) > 0:
            min_dist = self.realDistance(my_pos, capsule_list, all_ghost)
            features[0] = 1.0 / min_dist
        else:
            features[0] = 1.0
        # nearest food
        target_food = self.getTargetFood(state, all_ghost)
        if target_food:
            features[1] = 1.0 / self.realDistance(my_pos, [target_food], all_ghost)
        else:
            features[1] = 1.0
        # back home
        if my_state.isPacman:
            num_carry = state.getAgentState(self.index).numCarrying
            features[2] = float(num_carry) / (self.max_food * max(1, home_dist))
        # enemy features
        near_ghost = sorted(ghost)
        near_pacman = sorted(pacman)
        # closer ghost
        if len(ghost) > 0:
            d, _, _ = ghost[0]
            # features[3] = 2 * math.log(d) - home_dist
            features[3] = float(d - 1.1 * home_dist) / self.max_dist
        else:
            features[3] = 1.0
        # farther ghost
        if len(ghost) > 1:
            d, _, _ = ghost[1]
            # features[4] = 2 * math.log(d) - home_dist
            features[4] = float(d - 1.1 * home_dist) / self.max_dist
        else:
            features[4] = 1.0
        # closer pacman
        if len(pacman) > 0:
            d, s, _ = pacman[0]
            food_amp = float(s.numCarrying) / self.max_food
            features[5] = 1.0 / max(1, d) * food_amp
        # farther pacman
        if len(pacman) > 1:
            d, s, _ = pacman[1]
            food_amp = float(s.numCarrying) / self.max_food
            features[6] = 1.0 / max(1, d) * food_amp
        return features


    def getReward(self, prev_state, curr_state):
        ret = 0.0
        prev_pos = prev_state.getAgentPosition(self.index)
        curr_pos = curr_state.getAgentPosition(self.index)
        prev_num_carry = prev_state.getAgentState(self.index).numCarrying
        curr_num_carry = curr_state.getAgentState(self.index).numCarrying
        # eat food
        prev_food_list = self.getFood(prev_state).asList()
        if curr_pos in prev_food_list:
            ret += 1
        # eat capsule
        prev_capsule_list = self.getCapsules(prev_state)
        if curr_pos in prev_capsule_list:
            ret += 10
        # carry food back
        if curr_num_carry == 0 and prev_num_carry > 0 and abs(prev_pos[0] - curr_pos[0]) == 1:
            ret += 2 * prev_num_carry
        # kill Pacman or enemy carry food back
        for i in self.getOpponents(curr_state):
            prev_op_state = prev_state.getAgentState(i)
            curr_op_state = curr_state.getAgentState(i)
            prev_op_pos = prev_state.getAgentPosition(i)
            curr_op_pos = curr_state.getAgentPosition(i)
            if prev_op_state.isPacman and not curr_op_state.isPacman:
                if prev_op_pos and self.getMazeDistance(prev_op_pos, prev_pos) <= 2:
                    # kill enemy or enemy suicide
                    kill = curr_pos == prev_op_pos
                    suicide = (self.getMazeDistance(curr_pos, prev_op_pos) <= 1 and 
                              (not curr_op_pos or self.getMazeDistance(curr_pos, curr_op_pos) > 2))
                    if kill or suicide:
                        ret += 10 + prev_op_state.numCarrying
                if not prev_op_pos:
                    # enemy carry food back in the dark
                    ret -= 2 * prev_op_state.numCarrying
                else:
                    # enemy carry food back in sight
                    if curr_op_pos:
                        # see it both before and after
                        if self.getMazeDistance(prev_op_pos, curr_op_pos) == 1:
                            ret -= 2 * prev_op_state.numCarrying
                    else:
                        # cant see it after
                        prev_team_pos = prev_state.getAgentPosition((self.index + 2) % 4)
                        team_dist = self.getMazeDistance(prev_team_pos, prev_op_pos)
                        my_dist = self.getMazeDistance(prev_pos, prev_op_pos)
                        # both teammate and I are 2 distance away, impossible to kill
                        if team_dist > 2 and my_dist > 2:
                            ret -= 2 * prev_op_state.numCarrying
        # died
        start_pos = curr_state.getAgentState(self.index).start.pos
        if curr_pos == start_pos and self.getMazeDistance(prev_pos, curr_pos) > 1:
            ret -= 10 + prev_num_carry
        # print(ret)
        return ret


    def qFunc(self, features):
        ret = 0.0
        for i in range(len(features)):
            ret += features[i] * self.weights[i]
        return ret


    # This module records state transaction and reward
    def getRecord(self):
        if len(self.observationHistory) <= 1: return
        prev_state = self.observationHistory[-2]
        curr_state = self.observationHistory[-1]
        p_state_str = json.dumps(self.getFeature(prev_state))
        c_state_str = json.dumps(self.getFeature(curr_state))
        reward = self.getReward(prev_state, curr_state)
        record = dict()
        record["pre"] = p_state_str
        record["cur"] = c_state_str
        record["r"] = reward
        record_str = json.dumps(record)
        # if reward == 0: return
        with open("record.txt", "a") as f:
            f.write(record_str + "\n")


    # Following functions serve for inference
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

