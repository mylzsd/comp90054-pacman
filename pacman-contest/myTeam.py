from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import json, ast, math

MAX_VALUE = 99999999
MIN_VALUE = -99999999

MAX_DIST = 50
ENEMY_VISIBLE_DIST = 5

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CleanAgent', second = 'CleanAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

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

class CleanAgent(CaptureAgent):

    def registerInitialState(self, initial_state):
        CaptureAgent.registerInitialState(self, initial_state)
        self.enemies = initial_state.getBlueTeamIndices() if initial_state.isOnRedTeam(self.index) else initial_state.getRedTeamIndices()
        self.dist_matrix = self.distancer._distances
        self.infer = InferenceModule(self)
        self.infer.initialize(initial_state)
        print self.DeadLaneEntrance(initial_state)
        print initial_state

    def DeadLaneEntrance(self, state):
        walls = state.getWalls().asList()
        all_pos = [(x, y)
                   for x in range(0, state.data.layout.width)
                   for y in range(0, state.data.layout.height)]
        walkables = []
        for pos in all_pos:
            if pos not in walls:
                walkables.append(pos)
        walkables_pos_wallcounts = map(lambda pos: (pos, self.getNearWallCounts(pos, walls)), walkables)
        walkable_3walls = filter(lambda pos_wc: pos_wc[1] == 3, walkables_pos_wallcounts)
        walkable_3walls_pos = map(lambda x: x[0], walkable_3walls)
        ### examine each of the walkable with 3 walls
        dead_entries = []
        for pos in walkable_3walls_pos:
            ### block with 3 walls only has one way out
            pos_out = self.getOpenings(pos, walls)[0]
            prev_pos = pos
            ### position to expand
            temp_pos = pos_out # (1, 2)
            next_openings = self.getOpenings(temp_pos, walls)
            next_openings.remove(prev_pos)
            temp_depth = 1
            while len(next_openings) == 1:
                prev_pos = temp_pos
                temp_pos = next_openings[0]
                next_openings = self.getOpenings(temp_pos, walls)
                next_openings.remove(prev_pos)
                temp_depth += 1
            dead_entries.append((prev_pos, temp_depth))
        return dead_entries

    def isNear(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return True if abs(x1 - x2) + abs(y1 - y2) == 1 else False

    def getNearPos(self, pos):
        x, y = pos
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def getNearWallCounts(self, pos, walls):
        near_pos = self.getNearPos(pos)
        wall_count = 0
        for pos in near_pos:
            if pos in walls:
                wall_count += 1
        return wall_count

    def getOpenings(self, pos, walls):
        near_pos = self.getNearPos(pos)
        openings = []
        for pos in near_pos:
            if not pos in walls:
                openings.append(pos)
        return openings

    def getEnemyPos(self, state, agent_id):
        res = []
        self.enemies = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        for dist in self.enemy_pos_dist:
            if len(dist) == 0:
                if len(res) == 0:
                    index = self.enemies[0]
                    pos = state.getInitialAgentPosition(index)
                else:
                    index = self.enemies[1]
                    pos = state.getInitialAgentPosition(index)
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

    def chooseAction(self, current_state):
        self.updateEnemyPosDist()
        #print len(self.observationHistory)
        return self.getSimulationAction(current_state, 30, 20, self.index)

    def evaluateState(self, state, agent_id):
        return self.getFeatures(state, agent_id) * self.getWeights(state, agent_id)

    def getWeights(self, state, agent_id):
        w = util.Counter()
        # feature_MazeDistToNearestFood
        w[1] = 0
        # feature_MazeDistToFarestFood
        w[2] = 0
        # feature_MazeDistToNearestCapsule
        w[3] = -2 if self.feature_MazeDistanceToNearestGhost(state, agent_id) < ENEMY_VISIBLE_DIST else 0
        # feature_MazeDistToHighestScoredFood
        w[4] = -1
        # feature_MazeDistToNearestPacman
        w[5] = -4

        # feature_MazeDistToMostCarryingPacman
        w[6] = -5
        # feature_MazeDistanceToNearestGhost
        w[7] = 5
        # feature_MazeDistToMidLine
        w[8] = -5 if self.feature_CarryingFoodsCounts(state, agent_id) > 0 and self.feature_DefenderCounts(state, agent_id) > 0 else 0
        # feature_OneStepFood
        w[9] = 0
        # feature_CarryingFoodsCounts
        w[10] = 0

        # feature_CapsuleToEatCounts
        w[11] = -200 if self.feature_MazeDistanceToNearestGhost(state, agent_id) < ENEMY_VISIBLE_DIST else 0
        # feature_FoodsToEatCounts
        w[12] = -100
        # feature_FoodsToDefCounts
        w[13] = 0
        # feature_Score
        w[14] = 0
        # feature_InvaderCounts
        w[15] = -10000

        # feature_DefenderCounts
        w[16] = 0
        # feature_AtHome
        w[17] = 0
        # feature_EnemyAtHome
        w[18] = 0
        # feature_StopAction
        w[19] = 0
        # feature_DuplicatedAction
        w[20] = 0

        # feature_OnDefensive
        w[21] = 0
        # feature_FoodEatUp
        w[22] = 1000
        # feature_IsPacman
        w[23] = 0
        # feature_feature_MazeDistanceToFarestGhost
        w[24] = 0
        # feature_MazeDistanceToFarestPacman
        w[25] = 0
        # feature_EnemyGhostInvisible
        w[26] = -10000 if self.feature_OnEnemyBoard(state, agent_id) else 0
        return w

    def getFeatures(self, state, agent_id):
        f = util.Counter()
        f[1] = self.feature_MazeDistToNearestFood(state, agent_id)
        f[2] = self.feature_MazeDistToFarestFood(state, agent_id)
        f[3] = self.feature_MazeDistToNearestCapsule(state, agent_id)
        f[4] = self.feature_MazeDistToHighestScoredFood(state, agent_id)
        f[5] = self.feature_MazeDistToNearestPacman(state, agent_id)

        f[6] = self.feature_MazeDistToMostCarryingPacman(state, agent_id)
        f[7] = self.feature_MazeDistanceToNearestGhost(state, agent_id)
        f[8] = self.feature_MazeDistToMidLine(state, agent_id)
        f[9] = self.feature_OneStepFood(state, agent_id)
        f[10] = self.feature_CarryingFoodsCounts(state, agent_id)

        f[11] = self.feature_CapsuleToEatCounts(state, agent_id)
        f[12] = self.feature_FoodsToEatCounts(state, agent_id)
        f[13] = self.feature_FoodsToDefCounts(state, agent_id)
        f[14] = self.feature_Score(state, agent_id)
        f[15] = self.feature_InvaderCounts(state, agent_id)

        f[16] = self.feature_DefenderCounts(state, agent_id)
        f[17] = self.feature_AtHome(state, agent_id)
        f[18] = self.feature_EnemyAtHome(state, agent_id)
        f[19] = self.feature_StopAction(state, agent_id)
        f[20] = self.feature_DuplicatedAction(state, agent_id)

        f[21] = self.feature_OnDefensive(state, agent_id)
        f[22] = self.feature_FoodEatUp(state, agent_id)
        f[23] = self.feature_IsPacman(state, agent_id)
        f[24] = self.feature_MazeDistanceToFarestGhost(state, agent_id)
        f[25] = self.feature_MazeDistToFarestPacman(state, agent_id)

        f[26] = self.feature_EnemyGhostInvisible(state, agent_id)
        f[27] = self.feature_OnEnemyBoard(state, agent_id)
        return f

    ### features extracted to describe a state
    def feature_MazeDistToNearestFood(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        if foods_to_eat != []:
            return min([self.getMazeDistance(agent_pos, food_pos) for food_pos in foods_to_eat])
        else:
            return 0

    def feature_MazeDistToFarestFood(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        if foods_to_eat != []:
            return max([self.getMazeDistance(agent_pos, food_pos) for food_pos in foods_to_eat])
        else:
            return 0

    def feature_MazeDistToNearestCapsule(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        cap_positions = state.getBlueCapsules() if state.isOnRedTeam(agent_id) else state.getRedCapsules()
        if cap_positions != []:
            cap_distances = map(lambda pos: self.getMazeDistance(agent_pos, pos), cap_positions)
            return min(cap_distances)
        else:
            return 0

    def feature_MazeDistToHighestScoredFood(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        if foods_to_eat != []:
            walls = state.getWalls().asList()
            oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
            oppo_positions = []
            for oppo_id in oppo_ids:
                temp_pos = state.getAgentPosition(oppo_id)
                if temp_pos != None:
                    oppo_positions.append(temp_pos)
            max_food_score = MIN_VALUE
            max_food_pos = foods_to_eat[0]
            for fx, fy in foods_to_eat:
                temp_food_score = -1 * self.getMazeDistance(agent_pos, (fx, fy))
                for tx in range(fx - 1, fx + 2):
                    for ty in range(fy - 1, fy + 2):
                        if (tx, ty) in walls:
                            temp_food_score += -5
                        elif (tx, ty) in foods_to_eat:
                            temp_food_score += 30
                for tx in range(fx - 3, fx + 4):
                    for ty in range(fy - 3, fy + 4):
                        if (tx, ty) in oppo_positions:
                            temp_food_score += -100
                if temp_food_score > max_food_score:
                    max_food_score = temp_food_score
                    max_food_pos = (fx, fy)
            return self.getMazeDistance(agent_pos, max_food_pos)
        else:
            return 0

    def feature_MazeDistToNearestPacman(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        pacman_ids = filter(lambda id: state.getAgentState(id).isPacman, oppo_ids)
        pacman_pos = map(lambda id: state.getAgentPosition(id), pacman_ids)
        pacman_id_pos = zip(pacman_ids, pacman_pos)
        visible_pacman_id_pos = filter(lambda id_pos: id_pos[1] != None, pacman_id_pos)
        if visible_pacman_id_pos != []:
            pacman_dist = map(lambda id_pos: self.getMazeDistance(agent_pos, id_pos[1]), visible_pacman_id_pos)
            min_pacman_dist = min(pacman_dist)
            return min_pacman_dist
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistToFarestPacman(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        pacman_ids = filter(lambda id: state.getAgentState(id).isPacman, oppo_ids)
        pacman_pos = map(lambda id: state.getAgentPosition(id), pacman_ids)
        pacman_id_pos = zip(pacman_ids, pacman_pos)
        visible_pacman_id_pos = filter(lambda id_pos: id_pos[1] != None, pacman_id_pos)
        if visible_pacman_id_pos != []:
            pacman_dist = map(lambda id_pos: self.getMazeDistance(agent_pos, id_pos[1]), visible_pacman_id_pos)
            max_pacman_dist = max(pacman_dist)
            return max_pacman_dist
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistToMostCarryingPacman(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        enemy_infer_id_pos = zip(oppo_ids, self.getEnemyPos(state, agent_id))
        enemy_infer_id_dist = map(lambda eidpos:(eidpos[0], self.getMazeDistance(eidpos[1], agent_pos)), enemy_infer_id_pos)
        oppo_carrying = map(lambda eid: state.getAgentState(eid).numCarrying, oppo_ids)
        max_carrying = max(oppo_carrying)
        max_carrying_ids = filter(lambda x: x[0] == max_carrying, zip(oppo_carrying, oppo_ids))
        max_id = random.choice(max_carrying_ids)[1]
        return filter(lambda eid_dist: eid_dist[0] == max_id, enemy_infer_id_dist)[0][1]

    def feature_MazeDistanceToNearestGhost(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        ghost_ids = filter(lambda x: not state.getAgentState(x).isPacman and state.getAgentState(x).scaredTimer == 0, oppo_ids)
        ghost_pos = map(lambda o_id: state.getAgentPosition(o_id), ghost_ids)  # [(x, y)/ None]
        ghost_id_pos = zip(ghost_ids, ghost_pos)
        ghost_id_not_null_pos = filter(lambda x: x[1] != None, ghost_id_pos)
        ghost_id_not_null_dist = map(lambda (id, pos): (id, self.getMazeDistance(agent_pos, pos)),
                                     ghost_id_not_null_pos)
        if ghost_id_not_null_dist != []:
            min_dist = ghost_id_not_null_dist[0][1]
            for id, dist in ghost_id_not_null_dist:
                if min_dist > dist:
                    min_dist = dist
            return min(min_dist, ENEMY_VISIBLE_DIST)
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistanceToFarestGhost(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        ghost_ids = filter(lambda x: not state.getAgentState(x).isPacman and state.getAgentState(x).scaredTimer == 0, oppo_ids)
        ghost_pos = map(lambda o_id: state.getAgentPosition(o_id), ghost_ids)  # [(x, y)/ None]
        ghost_id_pos = zip(ghost_ids, ghost_pos)
        ghost_id_not_null_pos = filter(lambda x: x[1] != None, ghost_id_pos)
        ghost_id_not_null_dist = map(lambda (id, pos): (id, self.getMazeDistance(agent_pos, pos)), ghost_id_not_null_pos)
        if ghost_id_not_null_dist != []:
            max_dist = max([dist for id, dist in ghost_id_not_null_dist])
            return max(max_dist, ENEMY_VISIBLE_DIST)
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistToMidLine(self, state, agent_id):
        x_range = state.data.layout.width
        y_range = state.data.layout.height
        agent_pos = state.getAgentPosition(agent_id)
        mid_x = x_range / 2 - 1 if state.isOnRedTeam(agent_id) else x_range / 2
        wall_pos = state.getWalls().asList()
        mid_wall_pos = filter(lambda x: x[0] == mid_x, wall_pos)
        mid_all_pos = [(mid_x, y) for y in range(1, y_range)]
        mid_not_wall_pos = filter(lambda pos: not pos in mid_wall_pos, mid_all_pos)
        mid_not_wall_dist = map(lambda pos: self.getMazeDistance(agent_pos, pos), mid_not_wall_pos)
        return min(mid_not_wall_dist)

    def feature_OneStepFood(self, state, agent_id):
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        agent_pos = state.getAgentPosition(agent_id)
        food_distances = map(lambda x: self.getMazeDistance(agent_pos, x), foods_to_eat)
        return 1 if filter(lambda x: x == 1, food_distances) != [] else 0

    def feature_CarryingFoodsCounts(self, state, agent_id):
        return state.getAgentState(agent_id).numCarrying

    def feature_CapsuleToEatCounts(self, state, agent_id):
        return len(state.getBlueCapsules()) if state.isOnRedTeam(agent_id) else len(state.getRedCapsules())

    def feature_FoodsToEatCounts(self, state, agent_id):
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        return len(foods_to_eat)

    def feature_FoodsToDefCounts(self, state, agent_id):
        foods_to_def = state.getBlueFood().asList() if not state.isOnRedTeam(
            agent_id) else state.getRedFood().asList()
        return len(foods_to_def)

    def feature_Score(self, state, agent_id):
        return state.getScore() if state.isOnRedTeam(agent_id) else -1 * state.getScore()

    def feature_InvaderCounts(self, state, agent_id):
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        inv_cnt = 0
        for oppo_id in oppo_ids:
            if state.getAgentState(oppo_id).isPacman:
                inv_cnt += 1
        return inv_cnt

    def feature_DefenderCounts(self, state, agent_id):
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        def_cnt = 0
        for oppo_id in oppo_ids:
            if not state.getAgentState(oppo_id).isPacman and state.getAgentState(oppo_id).scaredTimer < 3:
                def_cnt += 1
        return def_cnt

    def feature_AtHome(self, state, agent_id):
        home_pos = state.getInitialAgentPosition(agent_id)
        agent_pos = state.getAgentPosition(agent_id)
        return 1 if agent_pos != None and agent_pos == home_pos else 0

    def feature_EnemyAtHome(self, state, agent_id):
        enemy_at_home_count = 0
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        oppo_pos = map(lambda id: state.getAgentPosition(id), oppo_ids)
        oppo_start = map(lambda id: state.getAgentState(id).start.pos, oppo_ids)
        for oppo_p in oppo_pos:
            if oppo_p in oppo_start:
                enemy_at_home_count += 1
        return enemy_at_home_count

    def feature_StopAction(self, state, agent_id):
        return 1 if state.getAgentState(agent_id).configuration.direction == "Stop" else 0

    def feature_DuplicatedAction(self, state, agent_id):
        agent_action = state.getAgentState(agent_id).configuration.direction
        agent_prev_action = None
        try:
            agent_prev_action = self.getPreviousObservation().getAgentState().configuration.direction
        except:
            pass
        return 0 if agent_prev_action == None or agent_prev_action != agent_action else 1

    def feature_OnDefensive(self, state, agent_id):
        return 0 if state.getAgentState(agent_id).isPacman else 1

    def feature_FoodEatUp(self, state, agent_id):
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        return 1 if len(foods_to_eat) < 3 else 0

    def feature_IsPacman(self, state, agent_id):
        return 1 if state.getAgentState(agent_id).isPacman else 0

    def feature_EnemyGhostInvisible(self, state, agent_id):
        return 1 if self.feature_MazeDistanceToNearestGhost(state, agent_id) >= ENEMY_VISIBLE_DIST else 0

    def feature_OnEnemyBoard(self, state, agent_id):
        return 1 if state.getAgentState(agent_id).isPacman else 0

    ### support functions
    def randomSingleSimulation(self, depth, state, agent_id):
        copy_state = state.deepCopy()
        while depth > 0:
            actions = copy_state.getLegalActions(agent_id)
            actions.remove("Stop")
            current_direction = copy_state.getAgentState(agent_id).configuration.direction
            reversed_direction = Directions.REVERSE[current_direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            next_action = random.choice(actions)
            copy_state = copy_state.generateSuccessor(agent_id, next_action)
            depth += -1
        return self.evaluateState(copy_state, agent_id)

    def getSimulationAction(self, state, simulations_num, simulation_depth, agent_id):
        legal_actions = state.getLegalActions(agent_id)
        legal_actions.remove("Stop")
        next_states = map(lambda a: state.generateSuccessor(agent_id, a), legal_actions)
        next_eval_actions = []
        max_eval = MIN_VALUE
        for next_state, next_action in zip(next_states, legal_actions):
            temp_simulation_eval = 0
            for i in range(0, simulations_num):
                temp_simulation_eval += self.randomSingleSimulation(simulation_depth, next_state, agent_id)
            if max_eval < temp_simulation_eval:
                max_eval = temp_simulation_eval
            next_eval_actions.append((temp_simulation_eval, next_action))
        if next_eval_actions != []:
            return random.choice(filter(lambda e_a: e_a[0] == max_eval, next_eval_actions))[1]
        return random.choice(legal_actions)


class GeneralAgent(CleanAgent):
    def getWeights(self, state, agent_id):
        w = util.Counter()
        # feature_MazeDistToNearestFood
        w[1] = 0
        # feature_MazeDistToFarestFood
        w[2] = 0
        # feature_MazeDistToNearestCapsule
        w[3] = 0.03427
        # feature_MazeDistToHighestScoredFood
        w[4] = 1.59158
        # feature_MazeDistToNearestPacman
        w[5] = 0

        # feature_MazeDistToMostCarryingPacman
        w[6] = 0
        # feature_MazeDistanceToNearestGhost
        w[7] = 2.24028
        # feature_MazeDistToMidLine
        w[8] = 28.16338
        # feature_OneStepFood
        w[9] = 0
        # feature_CarryingFoodsCounts
        w[10] = 0

        # feature_CapsuleToEatCounts
        w[11] = 0
        # feature_FoodsToEatCounts
        w[12] = 0
        # feature_FoodsToDefCounts
        w[13] = 0
        # feature_Score
        w[14] = 0
        # feature_InvaderCounts
        w[15] = 0

        # feature_DefenderCounts
        w[16] = 0
        # feature_AtHome
        w[17] = 0
        # feature_EnemyAtHome
        w[18] = 0
        # feature_StopAction
        w[19] = 0
        # feature_DuplicatedAction
        w[20] = 0

        # feature_OnDefensive
        w[21] = 0
        # feature_FoodEatUp
        w[22] = 0
        # feature_IsPacman
        w[23] = 0
        # feature_feature_MazeDistanceToFarestGhost
        w[24] = -1.81024
        # feature_MazeDistanceToFarestPacman
        w[25] = 11.05009
        return w

GAMMA = 0.9
EPSILON = 0.0

class RsngAgent(CleanAgent):

    dist_matrix = util.Counter()

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.initSetup(gameState)
        # infer module
        self.infer = InferenceModule(self)
        self.infer.initialize(gameState)
        # [cap_dist, food_dist, back, ghost_near, ghost_far, pacman_near, pacman_far]
        try:
            with open("weights.txt", "r") as f:
                self.weights = ast.literal_eval(f.read())
        except:
            self.weights = [0.03427, 1.59158, 28.16338, 2.24028, -1.81024, 49.92805, 11.05009]

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

    def chooseAction2(self, state):
        self.getRecord()
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

    def chooseAction(self, state):
        self.updateEnemyPosDist()
        return self.getQSimulationAction(state, 20, 1, self.index)

    def getStateQValue(self, state, agent_id):
        self.getRecord()
        # update infer module
        self.updateEnemyPosDist()
        # select best action
        actions = state.getLegalActions(agent_id)
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
            return reward + GAMMA * next_q

    def randomQSingleSimulation(self, depth, state, agent_id):
        copy_state = state.deepCopy()
        while depth > 0:
            actions = copy_state.getLegalActions(agent_id)
            actions.remove("Stop")
            current_direction = copy_state.getAgentState(agent_id).configuration.direction
            reversed_direction = Directions.REVERSE[current_direction]
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            next_action = random.choice(actions)
            copy_state = copy_state.generateSuccessor(agent_id, next_action)
            depth += -1
        return self.getStateQValue(copy_state, agent_id)

    def getQSimulationAction(self, state, simulations_num, simulation_depth, agent_id):
        legal_actions = state.getLegalActions(agent_id)
        legal_actions.remove("Stop")
        next_states = map(lambda a: state.generateSuccessor(agent_id, a), legal_actions)
        next_eval_actions = []
        max_eval = MIN_VALUE
        for next_state, next_action in zip(next_states, legal_actions):
            temp_simulation_eval = 0
            for i in range(0, simulations_num):
                temp_simulation_eval += self.randomQSingleSimulation(simulation_depth, next_state, agent_id)
            if max_eval < temp_simulation_eval:
                max_eval = temp_simulation_eval
            next_eval_actions.append((temp_simulation_eval, next_action))
        if next_eval_actions != []:
            return random.choice(filter(lambda e_a: e_a[0] == max_eval, next_eval_actions))[1]
        return random.choice(legal_actions)

    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            successor = successor.generateSuccessor(self.index, action)
        return successor

    def closestHome(self, is_red):
        homes = []
        if is_red:
            x = self.width / 2 - 1
        else:
            x = self.width / 2
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
        food_list = sorted(food_list, key=lambda x: self.getMazeDistance(src, x))
        for food in food_list:
            if self.realDistance(my_pos, [food], ghost) < self.max_dist:
                return food
        return None

    def getWeights(self, state, agent_id):
        w = util.Counter()
        w[0] = 0.03427
        w[0] = 1.59158
        w[0] = 28.16338
        w[0] = 2.24028
        w[0] = -1.81024
        w[0] = 49.92805
        w[0] = 11.05009
        return w

    def getFeature(self, state):
        # [cap_dist, food_dist, back, ghost_near, ghost_far, pacman_near, pacman_far]
        features = [0.0] * 7
        #features = util.Counter()
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
            elif d <= 6 and s.scaredTimer <= 5:
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
            ret += 5
        # carry food back
        if curr_num_carry == 0 and prev_num_carry > 0 and abs(prev_pos[0] - curr_pos[0]) == 1:
            ret += prev_num_carry
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
                        ret += 5 + prev_op_state.numCarrying
                if not prev_op_pos:
                    # enemy carry food back in the dark
                    ret -= prev_op_state.numCarrying
                else:
                    # enemy carry food back in sight
                    if curr_op_pos:
                        # see it both before and after
                        if self.getMazeDistance(prev_op_pos, curr_op_pos) == 1:
                            ret -= prev_op_state.numCarrying
                    else:
                        # cant see it after
                        prev_team_pos = prev_state.getAgentPosition((self.index + 2) % 4)
                        team_dist = self.getMazeDistance(prev_team_pos, prev_op_pos)
                        my_dist = self.getMazeDistance(prev_pos, prev_op_pos)
                        # both teammate and I are 2 distance away, impossible to kill
                        if team_dist > 2 and my_dist > 2:
                            ret -= prev_op_state.numCarrying
        # died
        start_pos = curr_state.getAgentState(self.index).start.pos
        if curr_pos == start_pos and self.getMazeDistance(prev_pos, curr_pos) > 1:
            ret -= 5 + prev_num_carry
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