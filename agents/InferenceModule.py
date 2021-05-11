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
    
    state = self.getCurrentObservation()
    self.infer.updateBelief(state)
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
            # e_pos = gameState.getInitialAgentPosition(enemy)
            # self.beliefs[enemy] = util.Counter()
            # self.beliefs[enemy][e_pos] = 1.0
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
            if self.me.red:
                s_col = currState.data.layout.width / 2 - 1
                e_col = s_col + 1
            else:
                s_col = currState.data.layout.width / 2
                e_col = s_col - 1
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
        if not flag:
            self.elapseTime(currState)

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
