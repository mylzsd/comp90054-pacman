from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import Queue


MAX_VALUE = 99999999
MIN_VALUE = -99999999

MAX_DIST = 50
ENEMY_VISIBLE_DIST = 100



def createTeam(firstIndex, secondIndex, isRed,
               first = 'search_agent', second = 'search_agent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class TreeNode:
    def __init__(self, value, father):
        self.value = value
        self.father = father

    def __getitem__(self):
        return self

    def expand(self, values):
        return map(lambda value: TreeNode(value, self), values)

class search_agent(CaptureAgent):
    last_eaten_def_foods = []

    def registerInitialState(self, initial_state):
        CaptureAgent.registerInitialState(self, initial_state)
        self.enemies = initial_state.getBlueTeamIndices() if initial_state.isOnRedTeam(self.index) else initial_state.getRedTeamIndices()
        self.dist_matrix = {}
        self.initial_dist_matrix(initial_state)
        self.infer = InferenceModule(self)
        self.infer.initialize(initial_state)

    def chooseAction(self, current_state):
        self.updateEnemyPosDist()
        return self.multipleExpanding(current_state, 2)

    def multipleExpanding(self, current_state, depth):
        queue_fathers = Queue()
        queue_children = Queue()
        queue_fathers.push(TreeNode((current_state, []), None))
        if self.hasVisibleEnemy(current_state):
            print "searching"
            for i in range(0, depth):
                while not queue_fathers.isEmpty():
                    top_father = queue_fathers.pop()
                    temp_my_children = top_father.expand(self.generateMyNextStatesIdAction(top_father.value[0]))
                    temp_enemy_children = map(lambda my_child: my_child.expand(self.generateNextEnemyStatesIdAction(my_child.value[0])), temp_my_children)
                    temp_enemy_children = filter(lambda x: x!= [], temp_enemy_children)
                    #print temp_enemy_children
                    for temp_enemy_children_group in temp_enemy_children:
                        temp_group_evals = map(lambda e_state_ids_actions: self.evaluateEnemyState(e_state_ids_actions.value[0]), temp_enemy_children_group)
                        #print temp_group_evals
                        if temp_group_evals != []:
                            max_group_eval = max(temp_group_evals)
                            max_group_children = random.choice(filter(lambda x: x[0] == max_group_eval, zip(temp_group_evals, temp_enemy_children_group)))[1]
                            #print max_group_children
                            queue_children.push(max_group_children)
                while not queue_children.isEmpty():
                    queue_fathers.push(queue_children.pop())
        while not queue_fathers.isEmpty():
            last_father = queue_fathers.pop()
            temp_my_lasts = last_father.expand(self.generateMyNextStatesIdAction(last_father.value[0]))
            temp_evals = map(lambda x: self.evaluateState(x.value[0], self.index), temp_my_lasts)
            if temp_evals != []:
                max_eval = max(temp_evals)
                max_evals_my_lasts = filter(lambda x: x[0] == max_eval, zip(temp_evals, temp_my_lasts))
                best_my_last = random.choice(max_evals_my_lasts)[1]
                prev_father = None
                while best_my_last.father != None:
                    prev_father = best_my_last
                    best_my_last = best_my_last.father
                if prev_father != None:
                    return prev_father.value[1][0][1]
        return random.choice(current_state.getLegalActions(self.index))




    ### assume the enemy only make greedy decision
    def evaluateEnemyState(self, state):
        enemy_ids = self.getOpponents(state)
        enemy_visibilities = map(lambda eid: True if state.getAgentPosition(eid) != None else False, enemy_ids)
        enemy_id_visibilities = zip(enemy_ids, enemy_visibilities)
        visible_enemy_ids = filter(lambda x: x[1], enemy_id_visibilities)  # [(id, True)]
        visible_enemy_counts = len(visible_enemy_ids)
        if visible_enemy_counts != 0:
            enemy_eval = 0
            for eid, _ in visible_enemy_ids:
                enemy_eval += self.evaluateState(state, eid)
            return enemy_eval
        else:
            return None

    def generateIdActions(self, current_state):
        temp_agent_next_actions = []
        ### removing friend's id
        oppo_ids = current_state.getBlueTeamIndices() if current_state.isOnRedTeam(self.index) else current_state.getRedTeamIndices()
        for agent_id in [self.index] + oppo_ids:
            try:
                agent_next_actions = current_state.getLegalActions(agent_id)
                #agent_next_actions.remove("Stop")
            except:
                agent_next_actions = []
            temp_agent_next_actions.append((agent_id, agent_next_actions))

        not_null_agent_id_actions = filter(lambda x: x[1] != [], temp_agent_next_actions)
        agent_ids = [not_null_agent_id_action[0] for not_null_agent_id_action in not_null_agent_id_actions]
        agent_legal_actions = [not_null_agent_id_action[1] for not_null_agent_id_action in not_null_agent_id_actions]
        blend_actions = []
        if len(not_null_agent_id_actions) == 1:
            blend_actions = [(a0,)
                             for a0 in agent_legal_actions[0]]
        elif len(not_null_agent_id_actions) == 2:
            blend_actions = [(a0, a1,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]]
        elif len(not_null_agent_id_actions) == 3:
            blend_actions = [(a0, a1, a2,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]
                             for a2 in agent_legal_actions[2]]
        elif len(not_null_agent_id_actions) == 4:
            blend_actions = [(a0, a1, a2, a3,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]
                             for a2 in agent_legal_actions[2]
                             for a3 in agent_legal_actions[3]]
        not_null_agent_ids_actions = []
        for blend_action in blend_actions:
            zipped_id_action = zip(agent_ids, blend_action)
            not_null_agent_ids_actions.append(zipped_id_action)
        return not_null_agent_ids_actions
        # [[(id1, action1), (id2, action2), (id3, action3), (id4, action4)]]

    def generateNextState(self, current_state, next_ids_actions):
        # next_id_actions = [(id1, action1), (id2, action2), (id3, action3), (id4, action4)]
        next_state = current_state
        for agent_id, agent_action in next_ids_actions:
            if agent_action != None:
                next_state = next_state.generateSuccessor(agent_id, agent_action)
        return next_state

    ### return states that contains all agents' possible next positions
    def generateNextStatesIdAction(self, current_state):
        next_ids_actions = self.generateIdActions(current_state)
        next_states_id_action = []
        for next_id_action in next_ids_actions:
            try:
                temp_next_state_id_action = (self.generateNextState(current_state, next_id_action), next_id_action)
                next_states_id_action.append(temp_next_state_id_action)
            except:
                continue
        return next_states_id_action

    def getEatenDefFoodPos(self, state, agent_id):
        try:
            prev_state = self.getPreviousObservation()
        except:
            prev_state = None
        if prev_state != None:
            prev_foods_to_def = prev_state.getRedFood().asList() if state.isOnRedTeam(agent_id) else prev_state.getBlueFood().asList()
        else:
            prev_foods_to_def = []
        foods_to_def = state.getRedFood().asList() if state.isOnRedTeam(agent_id) else state.getBlueFood().asList()
        eaten_food_pos = filter(lambda food_pos: not food_pos in foods_to_def, prev_foods_to_def)
        return eaten_food_pos

    def generateEnemyIdActions(self, current_state):
        temp_agent_next_actions = []
        oppo_ids = current_state.getBlueTeamIndices() if current_state.isOnRedTeam(self.index) else current_state.getRedTeamIndices()
        for agent_id in oppo_ids:
            try:
                agent_next_actions = current_state.getLegalActions(agent_id)
                agent_next_actions.remove("Stop")
            except:
                agent_next_actions = []
            temp_agent_next_actions.append((agent_id, agent_next_actions))

        not_null_agent_id_actions = filter(lambda x: x[1] != [], temp_agent_next_actions)
        agent_ids = [not_null_agent_id_action[0] for not_null_agent_id_action in not_null_agent_id_actions]
        agent_legal_actions = [not_null_agent_id_action[1] for not_null_agent_id_action in not_null_agent_id_actions]
        blend_actions = []
        if len(not_null_agent_id_actions) == 1:
            blend_actions = [(a0,)
                             for a0 in agent_legal_actions[0]]
        elif len(not_null_agent_id_actions) == 2:
            blend_actions = [(a0, a1,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]]
        elif len(not_null_agent_id_actions) == 3:
            blend_actions = [(a0, a1, a2,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]
                             for a2 in agent_legal_actions[2]]
        elif len(not_null_agent_id_actions) == 4:
            blend_actions = [(a0, a1, a2, a3,)
                             for a0 in agent_legal_actions[0]
                             for a1 in agent_legal_actions[1]
                             for a2 in agent_legal_actions[2]
                             for a3 in agent_legal_actions[3]]
        not_null_agent_ids_actions = []
        for blend_action in blend_actions:
            zipped_id_action = zip(agent_ids, blend_action)
            not_null_agent_ids_actions.append(zipped_id_action)
        return not_null_agent_ids_actions

    ### return all enemies' next states
    def generateNextEnemyStatesIdAction(self, current_state):
        next_ids_actions = self.generateEnemyIdActions(current_state)
        next_states_id_action = []
        for next_id_action in next_ids_actions:
            try:
                temp_next_state_id_action = (self.generateNextState(current_state, next_id_action), next_id_action)
                next_states_id_action.append(temp_next_state_id_action)
            except:
                pass
        return next_states_id_action

    def generateMyNextStatesIdAction(self, current_state):
        next_my_actions = current_state.getLegalActions(self.index)
        next_my_actions.remove("Stop")
        next_my_states_id_action = []
        for next_my_action in next_my_actions:
            next_my_states_id_action.append(
                (current_state.generateSuccessor(self.index, next_my_action), [(self.index, next_my_action)]))
        return next_my_states_id_action

    def hasVisibleEnemy(self, current_state):
        enemy_ids = self.getOpponents(current_state)
        enemy_visibilities = map(lambda eid: True if current_state.getAgentPosition(eid) != None else False, enemy_ids)
        enemy_id_visibilities = zip(enemy_ids, enemy_visibilities)
        visible_enemy_ids = filter(lambda x: x[1], enemy_id_visibilities)  # [(id, True)]
        visible_enemy_dist = map(lambda eid: self.getMazeDistance(current_state.getAgentPosition(eid[0]), current_state.getAgentPosition(self.index)), visible_enemy_ids)
        #print visible_enemy_dist
        return True if visible_enemy_ids != [] else False

    ################################################################################
    ################################################################################
    def evaluateState(self, state, agent_id):
        return self.getFeatures(state, agent_id) * self.getWeights(state, agent_id)


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

        return f

    def getWeights(self, state, agent_id):
        w = util.Counter()
        w[1] = 0
        w[2] = 0
        w[3] = 0
        w[4] = 0
        w[5] = -10

        w[6] = 0
        w[7] = 0
        w[8] = -1
        w[9] = 0
        w[10] = 0

        w[11] = 0
        w[12] = 0
        w[13] = 1000
        w[14] = 0
        w[15] = -10000

        w[16] = 0
        w[17] = -100000
        w[18] = 100000
        w[19] = -4
        w[20] = -2

        w[21] = 0
        w[22] = 0
        return w

    ##########################################################################
    ##########################################################################
    ### feature should be based on state, getting rid of generating new state or new actions
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
        #pacman_pos = map(lambda id: state.getAgentPosition(id), pacman_ids)
        pacman_pos = self.inferredEnemyPos(state, agent_id)
        pacman_id_pos = zip(pacman_ids, pacman_pos)
        visible_pacman_id_pos = filter(lambda id_pos: id_pos[1] != None, pacman_id_pos)
        if visible_pacman_id_pos != []:
            pacman_dist = map(lambda id_pos: self.getMazeDistance(agent_pos, id_pos[1]), visible_pacman_id_pos)
            min_pacman_dist = min(pacman_dist)
            return min_pacman_dist
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistToMostCarryingPacman(self, state, agent_id):
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        pacman_ids = filter(lambda id: state.getAgentState(id).isPacman, oppo_ids)
        all_ids = state.getRedTeamIndices() + state.getBlueTeamIndices()
        all_ids_dist =zip(all_ids, state.getAgentDistances())
        pacman_id_dist = filter(lambda x: x[0] in pacman_ids, all_ids_dist)
        pacman_carrying = map(lambda pid_dist: state.getAgentState(pid_dist[0]).numCarrying, pacman_id_dist)
        if pacman_carrying != []:
            max_carrying = max(pacman_carrying)
            max_carrying_ids_dist = filter(lambda carrying_id_dist: carrying_id_dist[0] == max_carrying, zip(pacman_carrying, pacman_id_dist))
            max_carrying_id_dist = random.choice(max_carrying_ids_dist)
            return max_carrying_id_dist[1][1]
        return MAX_DIST

    def feature_MazeDistanceToNearestGhost(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        ghost_ids = filter(lambda x: not state.getAgentState(x).isPacman and state.getAgentState(x).scaredTimer < 4, oppo_ids)
        #ghost_pos = map(lambda o_id: state.getAgentPosition(o_id), ghost_ids)  # [(x, y)/ None]
        ghost_pos = self.inferredEnemyPos(state, self.index)
        ghost_id_pos = zip(ghost_ids, ghost_pos)
        ghost_id_not_null_pos = filter(lambda x: x[1] != None, ghost_id_pos)
        ghost_id_not_null_dist = map(lambda (id, pos): (id, self.getMazeDistance(agent_pos, pos)), ghost_id_not_null_pos)
        if ghost_id_not_null_dist != []:
            min_dist = ghost_id_not_null_dist[0][1]
            for id, dist in ghost_id_not_null_dist:
                if min_dist > dist:
                    min_dist = dist
            return min(min_dist, ENEMY_VISIBLE_DIST)
        return ENEMY_VISIBLE_DIST

    def feature_MazeDistanceToNearestGhostReversed(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        ghost_ids = filter(lambda x: not state.getAgentState(x).isPacman and state.getAgentState(x).scaredTimer < 4, oppo_ids)
        ghost_pos = map(lambda o_id: state.getAgentPosition(o_id), ghost_ids)  # [(x, y)/ None]
        ghost_id_pos = zip(ghost_ids, ghost_pos)
        ghost_id_not_null_pos = filter(lambda x: x[1] != None, ghost_id_pos)
        ghost_id_not_null_dist = map(lambda (id, pos): (id, self.getMazeDistance(agent_pos, pos)), ghost_id_not_null_pos)
        if ghost_id_not_null_dist != []:
            min_dist = ghost_id_not_null_dist[0][1]
            for id, dist in ghost_id_not_null_dist:
                if min_dist > dist:
                    min_dist = dist
            return min(min_dist, ENEMY_VISIBLE_DIST)
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
        foods_to_def = state.getBlueFood().asList() if not state.isOnRedTeam(agent_id) else state.getRedFood().asList()
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
    ##########################################################################
    ##########################################################################

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

    def inferredEnemyPos(self, state, agent_id):
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
