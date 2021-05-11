import game
import util

"""
Initialization:
    self.infer = InferenceModule(self)
        arg: self -> agent
    self.infer.initialize(gameState)
        arg: gameState -> state
Requirement:
    agent.dist_matrix -> a matrix storing all maze distances
    agent.enemies -> a list including indexes of all enemies
    
Usage:
    Add following code into "chooseAction" of your agent.
    
    last_obs = self.getPreviousObservation()
    obs = self.getCurrentObservation()
    self.infer.updateBelief(obs, last_obs)
    distribution = []
    for enemy in self.enemies:
        distribution.append(self.infer.beliefs[enemy])
    self.displayDistributionsOverPositions(distribution)

"""


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
