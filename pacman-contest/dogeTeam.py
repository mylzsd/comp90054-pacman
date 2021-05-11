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
from InferenceModule import InferenceModule
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='AlphaBetaAgent', second='AlphaBetaAgent'):
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

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)
        distances = gameState.getAgentDistances()
        # print gameState.getDistanceProb(distances[1] - 1, distances[1])
        # print gameState.getAgentDistances()
        obs = self.getCurrentObservation()
        food = self.getFood(obs).asList()
        p_food = self.getFoodYouAreDefending(obs).asList()
        friends = {}
        for agent in self.getTeam(gameState):
            friends[agent] = obs.getAgentState(agent)
        enemies = {}
        for agent in self.getOpponents(gameState):
            enemies[agent] = obs.getAgentState(agent)
        # print "\n"
        pos = obs.getAgentPosition(self.index)
        min_food_dist = 999999
        nf = None
        for f in food:
            d = self.getMazeDistance(pos, f)
            if d < min_food_dist:
                nf = f
                min_food_dist = d
        # Tools.prim(Tools(), self, gameState, nf)

        '''
    You should change this in your own agent.
    '''

        return random.choice(actions)


class Go1807Agent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        pos = gameState.getAgentPosition(self.index)
        actions = gameState.getLegalActions(self.index)
        small_d = 999
        action = 'STOP'
        for a in actions:
            next_state = gameState.generateSuccessor(self.index, a)
            nextpos = next_state.getAgentPosition(self.index)
            dist = self.getMazeDistance((16, 7), nextpos)
            if dist < small_d:
                small_d = dist
                action = a
        return action


class AlphaBetaAgent(CaptureAgent):
    depth = 1
    time = 1200
    friends = {}
    enemie = {}
    curr_info = {"food_eaten": 0, "food_holding": 0, "enemy_pos": None, "lostfood": []}
    maze_dist = {}
    safe_pt = []
    threaten_lv = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
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
                                self.maze_dist[(i, j), (p, q)] = d
        if self.red:
            s_col = width / 2 - 1
        else:
            s_col = width / 2
        for j in range(0, height):
            if not gameState.hasWall(s_col, j):
                self.safe_pt.append((s_col, j))
        self.dist_matrix = self.maze_dist
        self.enemies = self.getOpponents(gameState)
        self.infer = InferenceModule(self)
        self.infer.initialize(gameState)

    def chooseAction(self, gameState):
        self.last_info = self.curr_info.copy()
        last_obs = self.getPreviousObservation()
        obs = self.getCurrentObservation()
        self.infer.updateBelief(obs, last_obs)
        distribution = []
        for enemy in self.enemies:
            distribution.append(self.infer.beliefs[enemy])
        self.displayDistributionsOverPositions(distribution)
        self.update_threaten(obs)
        if last_obs is None:
            last_obs = obs
        p_food = self.getFoodYouAreDefending(obs).asList()
        lp_food = self.getFoodYouAreDefending(last_obs).asList()
        lost_food = [f for f in lp_food if f not in p_food]
        if len(lost_food) > 0:
            self.curr_info["lostfood"] = lost_food
        pos = obs.getAgentPosition(self.index)
        last_food = self.getFood(last_obs).asList()
        curr_food = self.getFood(obs).asList()
        if pos in last_food and pos not in curr_food:
            self.curr_info["food_holding"] += 1
        isPacman = gameState.getAgentState(self.index).isPacman
        if not isPacman:
            self.curr_info["food_eaten"] += self.curr_info["food_holding"]
            self.curr_info["food_holding"] = 0
        root = {"state": obs, "actions": []}
        for agent in self.getTeam(gameState):
            self.friends[agent] = obs.getAgentState(agent)
        for agent in self.getOpponents(gameState):
            self.enemie[agent] = obs.getAgentState(agent)
        self_id = self.index
        if isPacman:
            choice = self.alpha_beta(root, self.depth * gameState.getNumAgents(),
                                     -99999999, 99999999, self_id, self.pacmanEval)
        else:
            choice = self.alpha_beta(root, self.depth * gameState.getNumAgents(),
                                     -99999999, 99999999, self_id, self.ghostEval)
        self.time -= 2
        action = choice["actions"][0]
        dice = random.randint(0, 100)
        if dice < 98:
            return action
        else:
            return random.choice(gameState.getLegalActions(self.index))

    def alpha_beta(self, node, depth, alpha, beta, agent_id, eva_func):
        state = node["state"]
        n_agents = state.getNumAgents()
        if depth == 0:
            return {"actions": node["actions"], "score": eva_func(state)}
        agent_info = state.getAgentState(agent_id)
        if agent_info.configuration is not None:
            n_moves = state.getLegalActions(agent_id)
            if len(n_moves) == 0:
                return {"actions": node["actions"], "score": eva_func(state)}
        else:
            next_agent = (agent_id + 1) % n_agents
            return self.alpha_beta(node, depth - 1, alpha, beta, next_agent, eva_func)
        if agent_id == self.index:
            value = -99999999
            for action in n_moves:
                next_state = state.generateSuccessor(agent_id, action)
                next_agent = (agent_id + 1) % n_agents
                next_actions = node["actions"][:]
                next_actions.append(action)
                next_node = {"state": next_state, "actions": next_actions}
                result = self.alpha_beta(next_node, depth - 1, alpha, beta, next_agent, eva_func)
                next_value = result["score"]
                if action == "Stop":
                    next_value -= 5
                if next_value > value:
                    a = result["actions"]
                value = max(value, next_value)
                alpha = max(alpha, value)
                if beta < alpha:
                    break
        elif not self.isFriend(agent_id):  # enemy
            value = 99999999
            pos = state.getAgentPosition(self.index)
            enemy_pos = state.getAgentPosition(agent_id)
            if self.maze_dist[pos, enemy_pos] > 12:
                next_agent = (agent_id + 1) % n_agents
                return self.alpha_beta(node, depth - 1, alpha, beta, next_agent, eva_func)
            for action in n_moves:
                next_state = state.generateSuccessor(agent_id, action)
                next_agent = (agent_id + 1) % n_agents
                next_actions = node["actions"][:]
                next_actions.append(action)
                next_node = {"state": next_state, "actions": next_actions}
                result = self.alpha_beta(next_node, depth - 1, alpha, beta, next_agent, eva_func)
                next_value = result["score"]
                if next_value < value:
                    a = result["actions"]
                value = min(value, next_value)
                beta = min(beta, value)
                if beta < alpha:
                    break
        else:
            next_agent = (agent_id + 1) % n_agents
            return self.alpha_beta(node, depth - 1, alpha, beta, next_agent, eva_func)
        if len(node["actions"]) == 0:
            node["actions"] = a
        return {"actions": node["actions"], "score": value}

    def isFriend(self, a_id):
        return self.friends.has_key(a_id)

    def pacmanEval(self, state):
        food = self.getFood(state).asList()
        pos = state.getAgentPosition(self.index)
        min_food_dist = 99
        min_safe_dist = 99.0
        if state.getAgentState(self.index).isPacman:
            for s in self.safe_pt:
                d = self.maze_dist[pos, s]
                min_safe_dist = min(min_safe_dist, d)
        else:
            min_safe_dist = 0.0
        nf = None
        for f in food:
            d = self.maze_dist[pos, f]
            if d < min_food_dist:
                nf = f
                min_food_dist = d
        dist = Tools.prim(Tools(), self, state, nf)
        if len(food) == 0:
            min_food_dist = 0
            dist = 0
        else:
            dist /= len(food)
        e_dist = 99
        for enemy in self.enemie:
            enemy_info = state.getAgentState(enemy)
            isPacman = enemy_info.isPacman
            if enemy_info.configuration is not None:
                if not isPacman:
                    e_pos = enemy_info.configuration.getPosition()
                    e_dist = min(self.maze_dist[e_pos, pos], e_dist)
            else:
                if not isPacman:
                    e_dist = min(state.getAgentDistances()[enemy], e_dist)
        prize = 0
        if self.red:
            prize += state.getScore()
        else:
            prize -= state.getScore()
        prize += -0.5 * (min_food_dist + dist)
        prize += -50 * len(food)
        prize += 50 * min(e_dist, 4)
        prize += -0.2 * min_safe_dist * self.curr_info["food_holding"]
        prize += -0.5 * min_safe_dist * self.threaten_lv
        prize += 100 * self.curr_info["food_eaten"]
        return prize

    def ghostEval(self, state):
        pos = state.getAgentPosition(self.index)
        p_food = self.getFoodYouAreDefending(state).asList()
        offensive_score = self.pacmanEval(state)
        n_invaders = 0
        e_dist = 20
        eg_dist = 99
        for enemy in self.enemie:
            enemy_info = state.getAgentState(enemy)
            isPacman = enemy_info.isPacman
            if isPacman:
                n_invaders += 1
            if enemy_info.configuration is not None:
                e_pos = enemy_info.configuration.getPosition()
                self.curr_info["enemy_pos"] = e_pos
                if isPacman:
                    e_dist = min(self.maze_dist[e_pos, pos], e_dist)
                else:
                    eg_dist = min(self.maze_dist[e_pos, pos], eg_dist)
            else:
                if isPacman:
                    e_dist = min(state.getAgentDistances()[enemy], e_dist)
                else:
                    eg_dist = min(state.getAgentDistances()[enemy], eg_dist)
        lf_dist = 0
        for f in self.curr_info["lostfood"]:
            lf_dist += self.maze_dist[f, pos]
        defensive_score = 0
        defensive_score += 100 * e_dist
        defensive_score += 30 * lf_dist
        rm_ghost_score = 50 * min(e_dist, 4)
        if pos == self.last_info["enemy_pos"]:
            defensive_score -= 80000
        return offensive_score - self.threaten_lv * defensive_score - rm_ghost_score

    def update_threaten(self, state):
        p_food = self.getFoodYouAreDefending(state).asList()
        n_invaders = 0
        for enemy in self.enemie:
            enemy_info = state.getAgentState(enemy)
            isPacman = enemy_info.isPacman
            if isPacman:
                n_invaders += 1
        self.threaten_lv = 0
        self.threaten_lv += n_invaders
        self.threaten_lv += pow(20 / max(0.1, len(p_food)), 1.5)

class Tools:
    def prim(self, agent, state, nearest):
        if nearest is None:
            return -1
        food = agent.getFood(state).asList()
        for i in range(0, len(food)):
            if food[i][0] == nearest[0] and food[i][1] == nearest[1]:
                start = i
                break
        edge = {}
        vnum = len(food)
        for i in range(0, vnum):
            edge[i] = {}
            for j in range(0, vnum):
                edge[i][j] = agent.maze_dist[food[i], food[j]]
        lowcost = {}
        addvnew = {}
        adjecent = {}
        sumweight = 0
        i, j = 0, 0
        for i in range(1, vnum):
            lowcost[i] = edge[start][i]
            addvnew[i] = -1
        addvnew[start] = 0
        adjecent[start] = start
        for i in range(1, vnum - 1):
            min = 999999
            v = -1
            for j in range(1, vnum):
                if addvnew[j] == -1 and lowcost[j] < min:
                    min = lowcost[j]
                    v = j
            if v != -1:
                addvnew[v] = 0
                sumweight += lowcost[v]
                for j in range(1, vnum):
                    if addvnew[j] == -1 and edge[v][j] < lowcost[j]:
                        lowcost[j] = edge[v][j]
                        adjecent[j] = v
        return sumweight
