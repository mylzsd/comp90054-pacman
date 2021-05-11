from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import Queue


MAX_VALUE = 99999999
MIN_VALUE = -99999999

MAX_DIST = 50
ENEMY_VISIBLE_DIST = 5
#OffensiveYoo
#DefensiveYoo
#YooAgent

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveYoo', second = 'DefensiveYoo'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class YooAgent(CaptureAgent):
    grid_score = {}
    last_eaten_def_foods = []
    step_left = 300

    def registerInitialState(self, initial_state):
        CaptureAgent.registerInitialState(self, initial_state)
        ######################################################
        x_mid = initial_state.data.layout.width / 2
        walls = initial_state.getWalls().asList()
        all_pos = [(x, y)
                   for x in range(0, initial_state.data.layout.width)
                   for y in range(0, initial_state.data.layout.height)]
        walkables = filter(lambda (x, y): not (x, y) in walls and x > x_mid, all_pos)
        for wx, wy in walkables:
            wall_counts = 0
            for x in range(wx - 1, wx + 2):
                for y in range(wy - 1, wy + 2):
                    if (x, y) in walls:
                        wall_counts += 1
            self.grid_score[(wx, wy)] = 0 if wall_counts < 5 else 1
        ######################################################


    # def chooseAction(self, current_state):
    #     timer = time.time()
    #     my_legal_actions = current_state.getLegalActions(self.index)
    #     #######################################
    #     best_my_id_action_diff = self.singleExpanding(current_state)
    #     if best_my_id_action_diff != None:
    #         #print time.time() - timer
    #         return best_my_id_action_diff[0][0][1]
    #     else:
    #         my_next_state_id_actions = self.generateMyNextStatesIdAction(current_state)
    #         my_eval = []
    #         for state, id_action in my_next_state_id_actions:
    #             my_eval.append(self.evaluateState(state, self.index))
    #         if my_eval != []:
    #             best_eval = max(my_eval)
    #             best_states_id_action = filter(lambda x: x[0] == best_eval, zip(my_eval, my_next_state_id_actions))
    #             next_action = random.choice(best_states_id_action)[1][1][0][1]
    #             if next_action in my_legal_actions:
    #                 #print time.time() - timer
    #                 return next_action
    #     #######################################
    #     #print time.time() - timer
    #     return random.choice(my_legal_actions)

    def bestEnemyNextState(self, current_state):
        next_enemy_states = self.generateNextEnemyStatesIdAction(current_state)
        best_ids_actions = []
        try:
            best_id_action_1 = self.nextBestStateIdAction(next_enemy_states, 1)[1]
            best_ids_actions += filter(lambda x: x[0] == 1 ,best_id_action_1)
        except:
            pass
        try:
            best_id_action_3 = self.nextBestStateIdAction(next_enemy_states, 3)[1]
            best_ids_actions += filter(lambda x: x[0] == 3, best_id_action_3)
        except:
            pass

        if best_ids_actions != []:
            best_enemy_state = self.generateNextState(current_state, best_ids_actions)
            return best_enemy_state
        else:
            return None

    def nextBestStateIdAction(self, states_id_action, agent_id):
        eval = []
        for state, id_action in states_id_action:
            eval.append(self.evaluateState(state, agent_id))
        if eval != []:
            best_eval = max(eval)
            best_states_id_action = filter(lambda x: x[0] == best_eval, zip(eval, states_id_action))
            return random.choice(best_states_id_action)[1]
        else:
            return None

    def evaluateAverageEnemyState(self, state):
        enemy_ids = self.getOpponents(state)
        enemy_visibilities = map(lambda eid: True if state.getAgentPosition(eid) != None else False, enemy_ids)
        enemy_id_visibilities = zip(enemy_ids, enemy_visibilities)
        visible_enemy_ids = filter(lambda x: x[1], enemy_id_visibilities) #[(id, True)]
        visible_enemy_counts = len(visible_enemy_ids)
        if visible_enemy_counts != 0:
            enemy_eval = 0
            for eid, _ in visible_enemy_ids:
                enemy_eval += self.evaluateState(state, eid)
            return enemy_eval / visible_enemy_counts
        else:
            return None

    ### performing one my and one enemy's layer, return the best action of mine
    def singleExpanding(self, current_state):
        # generate my layer of state
        my_layer_states_ids_actions = self.generateMyNextStatesIdAction(current_state)
        # for each of my next possible state, generate enemy's next states
        my_id_actions_enemy_avg_diff = []
        for my_state, my_id_action in my_layer_states_ids_actions:
            temp_group_enemy_states_ids_actions = self.generateNextEnemyStatesIdAction(my_state)
            if temp_group_enemy_states_ids_actions != []:
                temp_enemy_group_evals = map(lambda e_state_id_action: self.evaluateAverageEnemyState(e_state_id_action[0]), temp_group_enemy_states_ids_actions)
                temp_enemy_group_not_null_evals = filter(lambda x: x != None, temp_enemy_group_evals)
                if temp_enemy_group_not_null_evals != []:
                    temp_avg_enemy_eval = sum(temp_enemy_group_not_null_evals) / len(temp_enemy_group_not_null_evals)
                    my_id_actions_enemy_avg_diff.append((my_id_action, self.evaluateState(my_state, self.index) - temp_avg_enemy_eval, my_state))
        # find the largest diff between my and enemy's layer of states
        if my_id_actions_enemy_avg_diff != []:
            best_diff = max(map(lambda x: x[1], my_id_actions_enemy_avg_diff))
            best_my_id_actions_enemy_avg_diff = filter(lambda x: x[1] == best_diff, my_id_actions_enemy_avg_diff)
            return random.choice(best_my_id_actions_enemy_avg_diff)
        else:
            return None

    def evaluateState(self, state, agent_id):
        return self.getFeatures(state, agent_id) * self.getWeights(state, agent_id)


    def getFeatures(self, state, agent_id):
        f = util.Counter()
        return f

    def getWeights(self, state, agent_id):
        w = util.Counter()
        return w

    ##########################################################################
    ##########################################################################
    ##########################################################################
    def evaluateOffensiveState(self, state, agent_id):
        return self.getOffensiveFeatures(state, agent_id) * self.getOffensiveWeights(state, agent_id)

    def evaluateDefensiveState(self, state, agent_id):
        return self.getDefensiveFeatures(state, agent_id) * self.getDefensiveWeights(state, agent_id)

    def getOffensiveFeatures(self, state, agent_id):
        o_f = util.Counter()
        # o_f[1] = self.feature_CapsuleToEatCounts(state, agent_id)
        # o_f[2] = self.feature_FoodsToEatCounts(state, agent_id)
        # o_f[3] = self.feature_MazeDistToNearestFood(state, agent_id) * 0.9 + 0.1 * self.feature_AvgMSTDistToFood(state, agent_id)
        # o_f[3] = self.feature_MazeDistToHighestScoredFood(state, agent_id)
        # o_f[4] = self.feature_CarryingFoodsCounts(state, agent_id)
        # o_f[5] = self.feature_MazeDistToMidLine(state, agent_id)
        # o_f[6] = self.feature_StopAction(state, agent_id)
        # # o_f[7] =
        # o_f[8] = self.feature_MazeDistanceToNearestGhost(state, agent_id)
        # o_f[9] = self.feature_MazeDistToNearestCapsule(state, agent_id)

        o_f[1] = self.feature_MazeDistToNearestFood(state, agent_id)
        o_f[2] = self.feature_MazeDistToFarestFood(state, agent_id)
        o_f[3] = self.feature_MazeDistToNearestCapsule(state, agent_id)
        o_f[4] = self.feature_MazeDistToHighestScoredFood(state, agent_id)
        o_f[5] = self.feature_MazeDistToMidLine(state, agent_id)
        #o_f[5] = self.feature_MazeDistToHome(state, agent_id)

        o_f[6] = self.feature_CarryingFoodsCounts(state, agent_id)
        o_f[7] = self.feature_CapsuleToEatCounts(state, agent_id)
        o_f[8] = self.feature_FoodsToEatCounts(state, agent_id)
        o_f[9] = self.feature_FoodsToDefCounts(state, agent_id)
        o_f[10] = self.feature_Score(state, agent_id)

        o_f[11] = self.feature_InvaderCounts(state, agent_id)
        o_f[12] = self.feature_AtHome(state, agent_id)
        o_f[13] = self.feature_DeathThreatLevel(state, agent_id)
        o_f[14] = self.feature_MazeDistanceToNearestGhost(state, agent_id)
        o_f[15] = self.feature_EnemyAtHome(state, agent_id)

        o_f[16] = self.feature_FoodEatUp(state, agent_id)
        o_f[17] = self.feature_StopAction(state, agent_id)
        o_f[18] = self.feature_GridScore(state, agent_id)
        o_f[19] = self.feature_DuplicatedAction(state, agent_id)
        o_f[20] = self.feature_StepsLessThan50(state, agent_id)
        return o_f

    def getOffensiveWeights(self, state, agent_id):
        o_w = util.Counter()
        # o_w[1] = -100 * self.feature_DefenderCounts(state, agent_id) - 1  # capsules Left counts
        # o_w[2] = -100  # food counts
        # o_w[3] = -1  # distance to Food
        # o_w[4] = -1  # food carrying
        # o_w[5] = -0.5 * self.feature_CarryingFoodsCounts(state, agent_id) * self.feature_DefenderCounts(state, agent_id) ** 2 - 0.1 * self.feature_DefenderCounts(state, agent_id) - 0.1 * self.feature_CarryingFoodsCounts(state, agent_id)  # dist to mid line
        # o_w[6] = -3  # stop action
        # o_w[7] = 0  # reverse action
        # o_w[8] = 10  # dist to enemy ghost
        # o_w[9] = -1 * self.feature_DefenderCounts(state, agent_id)  # dist to capsules

        #o_w[1] = -1 if self.feature_FoodsToEatCounts(state, agent_id) > 2  else 0
        #o_w[1] = -1 if self.feature_FoodsToEatCounts(state, agent_id) > 2 and self.feature_MazeDistanceToVisibleNearestGhost(state, agent_id) > 5 else 0
        o_w[2] = 0
        o_w[3] = -100 if self.feature_MazeDistanceToNearestGhost(state, agent_id) < ENEMY_VISIBLE_DIST else 0
        o_w[4] = -1 if self.feature_FoodsToEatCounts(state, agent_id) > 2 and self.feature_MazeDistanceToNearestGhost(state, agent_id) >= ENEMY_VISIBLE_DIST else 0
        #o_w[5] = -5 if (self.feature_CarryingFoodsCounts(state, agent_id) > 0 and self.feature_MazeDistanceToNearestGhost(state, agent_id) != MAX_DIST) or self.feature_FoodsToEatCounts(state, agent_id) < 3 or self.feature_StepsLessThan50(state, agent_id) == 1  else 0
        o_w[5] = -0.5 * self.feature_CarryingFoodsCounts(state, agent_id) * self.feature_DefenderCounts(state, agent_id) ** 2 - 0.1 * self.feature_DefenderCounts(state, agent_id) - 0.1 * self.feature_CarryingFoodsCounts(state, agent_id)  # dist to mid line

        o_w[6] = 0
        o_w[7] = -200 if self.feature_MazeDistanceToNearestGhost(state, agent_id) < ENEMY_VISIBLE_DIST else 0
        o_w[8] = -150 if self.feature_MazeDistanceToNearestGhost(state, agent_id) >= ENEMY_VISIBLE_DIST else 0
        o_w[9] = 200
        o_w[10] = 1000

        o_w[11] = -10000
        o_w[12] = -100000
        o_w[13] = 20
        o_w[14] = 0
        o_w[15] = 10000 if not state.getAgentState(agent_id).isPacman else 0

        o_w[16] = 10000
        o_w[17] = 0
        #o_w[18] = -10
        o_w[19] = -4
        o_w[20] = 0
        return o_w

    def getDefensiveFeatures(self, state, agent_id):
        d_f = util.Counter()
        agent_pos = state.getAgentPosition(agent_id)
        d_f[1] = self.feature_InvaderCounts(state, agent_id)
        d_f[2] = self.feature_OnDefensive(state, agent_id)
        d_f[3] = self.getMazeDistance(agent_pos, self.last_eaten_def_foods[0]) if self.last_eaten_def_foods != [] and self.feature_InvaderCounts(state, agent_id) > 0 and self.feature_MazeDistToNearestPacman(state, agent_id) == MAX_DIST else self.feature_MazeDistToNearestPacman(state, agent_id)
        d_f[4] = self.feature_MazeDistToMostCarryingPacman(state, agent_id)
        d_f[5] = self.feature_StopAction(state, agent_id)

        d_f[6] = self.feature_DuplicatedAction(state, agent_id)
        d_f[7] = self.feature_MazeDistToMidLine(state, agent_id)
        d_f[8] = self.feature_OneStepFood(state, agent_id)
        d_f[9] = self.feature_AtHome(state, agent_id)
        d_f[10] = self.feature_EnemyAtHome(state, agent_id)
        return d_f

    def getDefensiveWeights(self, state, agent_id):
        d_w = util.Counter()
        d_w[1] = -1000  # invader number
        d_w[2] = 100  # is on defensive
        d_w[3] = -10 if state.getAgentState(agent_id).scaredTimer == 0 else 10 # dist to enemy pacman
        d_w[4] = -0  # dist to enemy pacman with most carryings
        d_w[5] = -100  # stop action
        d_w[6] = -2  # reverse action
        d_w[7] = -1  # dist to mid line
        d_w[8] = 110  # dist to 1 step foods
        d_w[9] = -100000
        d_w[10] = 100000

        return d_w

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

    def feature_MazeDistToHome(self, state, agent_id):
        home_pos = state.getInitialAgentPosition(agent_id)
        agent_pos = state.getAgentPosition(agent_id)
        return self.getMazeDistance(home_pos, agent_pos)

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
        return MAX_DIST

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
        return MAX_DIST

    def feature_AvgMSTDistToFood(self, state, agent_id):
        foods_to_eat = state.getBlueFood().asList() if state.isOnRedTeam(agent_id) else state.getRedFood().asList()
        if foods_to_eat != []:
            agent_pos = state.getAgentPosition(agent_id)
            food_distances = map(lambda x: self.getMazeDistance(agent_pos, x), foods_to_eat)
            min_dist_to_food = min(food_distances)
            food_dist_pos = zip(food_distances, foods_to_eat)
            min_dist_food = filter(lambda x: x[0] == min_dist_to_food, food_dist_pos)
            min_food_pos = random.choice(min_dist_food)[1]
            return prim(self,  min_food_pos, foods_to_eat)/ len(foods_to_eat)
        else:
            return 0

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

    def feature_DeathThreatLevel(self, state, agent_id):
        ax, ay= state.getAgentPosition(agent_id)
        oppo_ids = state.getBlueTeamIndices() if state.isOnRedTeam(agent_id) else state.getRedTeamIndices()
        ghost_ids = filter(lambda x: not state.getAgentState(x).isPacman, oppo_ids)
        ghost_pos = map(lambda o_id: state.getAgentPosition(o_id), ghost_ids) #[(x, y)/ None]
        # see in 5x5 range of grid
        death_eval = 0
        for tx in range(ax - 2, ax + 3):
            for ty in range(ay - 2, ay + 3):
                if tx in range(ax - 1, ax + 2) and ty in range(ay - 1, ay + 2):
                    if (tx, ty) in ghost_pos:
                        death_eval += 10
                else:
                    if (tx, ty) in ghost_pos:
                        death_eval += 5
        return death_eval

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

    def feature_GridScore(self, state, agent_id):
        agent_pos = state.getAgentPosition(agent_id)
        try:
            return self.grid_score[agent_pos]
        except:
            return 0

    def feature_StepsLessThan50(self, state, agent_id):
        return 1 if self.step_left < 50 else 0


    ##########################################################################
    ##########################################################################
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
        agent_pos = state.getAgentPosition(agent_id)
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

    ##########################################################################
    ############################ enemy anticipation ##########################
    ##########################################################################
    def generateEnemyIdActions(self, current_state):
        temp_agent_next_actions = []
        ### removing friend's id
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

    ##########################################################################
    ############################# my next decision ###########################
    ##########################################################################
    def generateMyNextStatesIdAction(self, current_state):
        next_my_actions = current_state.getLegalActions(self.index)
        next_my_actions.remove("Stop")
        next_my_states_id_action = []
        for next_my_action in next_my_actions:
            next_my_states_id_action.append((current_state.generateSuccessor(self.index, next_my_action), [(self.index, next_my_action)]))
        return next_my_states_id_action

######################################################################
############################ OFFENSIVE AGENT##########################
######################################################################
class OffensiveYoo(YooAgent):
    def chooseAction(self, current_state):
        timer = time.time()
        my_legal_actions = current_state.getLegalActions(self.index)
        self.step_left += -1
        #######################################
        best_my_id_action_diff = self.singleOffensiveExpanding(current_state)
        if best_my_id_action_diff != None and self.feature_MazeDistanceToNearestGhost(current_state, self.index) < ENEMY_VISIBLE_DIST:
            #print time.time() - timer ####### <timer> #######
            #print "searching"
            return best_my_id_action_diff[0][0][1]
        else:
            my_next_state_id_actions = self.generateMyNextStatesIdAction(current_state)
            my_eval = []
            for state, id_action in my_next_state_id_actions:
                my_eval.append(self.evaluateOffensiveState(state, self.index))
            if my_eval != []:
                best_eval = max(my_eval)
                best_states_id_action = filter(lambda x: x[0] == best_eval, zip(my_eval, my_next_state_id_actions))
                next_action = random.choice(best_states_id_action)[1][1][0][1]
                if next_action in my_legal_actions:
                    #print time.time() - timer ####### <timer> #######
                    #print "non-searching"
                    return next_action
        #######################################
        #print time.time() - timer ####### <timer> #######
        print "non-searching"
        return random.choice(my_legal_actions)

    def chooseAction2(self, current_state):
        self.step_left += -1
        my_legal_actions = current_state.getLegalActions(self.index)
        my_next_state_id_actions = self.generateMyNextStatesIdAction(current_state)
        my_eval = []
        for state, id_action in my_next_state_id_actions:
            my_eval.append(self.evaluateOffensiveState(state, self.index))
        print zip(my_eval, map(lambda state_id_action: state_id_action[1], my_next_state_id_actions))
        if my_eval != []:
            best_eval = max(my_eval)
            best_states_id_action = filter(lambda x: x[0] == best_eval, zip(my_eval, my_next_state_id_actions))
            next_action = random.choice(best_states_id_action)[1][1][0][1]
            if next_action in my_legal_actions:
                print "--> " + next_action
                return next_action
        return random.choice(my_legal_actions)

    def singleOffensiveExpanding(self, current_state):
        # generate my layer of state
        my_layer_states_ids_actions = self.generateMyNextStatesIdAction(current_state)
        # for each of my next possible state, generate enemy's next states
        my_id_actions_enemy_avg_diff = []
        for my_state, my_id_action in my_layer_states_ids_actions:
            temp_group_enemy_states_ids_actions = self.generateNextEnemyStatesIdAction(my_state)
            if temp_group_enemy_states_ids_actions != []:
                temp_enemy_group_evals = map(lambda e_state_id_action: self.evaluateDefensiveAverageEnemyState(e_state_id_action[0]), temp_group_enemy_states_ids_actions)
                temp_enemy_group_not_null_evals = filter(lambda x: x != None, temp_enemy_group_evals)
                if temp_enemy_group_not_null_evals != []:
                    temp_avg_enemy_eval = sum(temp_enemy_group_not_null_evals) / len(temp_enemy_group_not_null_evals)
                    my_id_actions_enemy_avg_diff.append((my_id_action, self.evaluateState(my_state, self.index) - temp_avg_enemy_eval, my_state))
        # find the largest diff between my and enemy's layer of states
        if my_id_actions_enemy_avg_diff != []:
            best_diff = max(map(lambda x: x[1], my_id_actions_enemy_avg_diff))
            best_my_id_actions_enemy_avg_diff = filter(lambda x: x[1] == best_diff, my_id_actions_enemy_avg_diff)
            return random.choice(best_my_id_actions_enemy_avg_diff)
        else:
            return None

    def evaluateDefensiveAverageEnemyState(self, state):
        #print "off eval defensive"
        enemy_ids = self.getOpponents(state)
        enemy_visibilities = map(lambda eid: True if state.getAgentPosition(eid) != None else False, enemy_ids)
        enemy_id_visibilities = zip(enemy_ids, enemy_visibilities)
        visible_enemy_ids = filter(lambda x: x[1], enemy_id_visibilities) #[(id, True)]
        visible_enemy_counts = len(visible_enemy_ids)
        if visible_enemy_counts != 0:
            enemy_eval = 0
            for eid, _ in visible_enemy_ids:
                enemy_eval += self.evaluateDefensiveState(state, eid)
            return enemy_eval / visible_enemy_counts
        else:
            return None

######################################################################
############################ DEFENSIVE AGENT #########################
######################################################################
class DefensiveYoo(YooAgent):

    def chooseAction2(self, current_state):
        timer = time.time()
        my_legal_actions = current_state.getLegalActions(self.index)
        self.step_left += -1
        #######################################
        best_my_id_action_diff = self.singleDefensiveExpanding(current_state)
        if best_my_id_action_diff != None:
            print time.time() - timer ####### <timer> #######
            return best_my_id_action_diff[0][0][1]
        else:
            my_next_state_id_actions = self.generateMyNextStatesIdAction(current_state)
            my_eval = []
            for state, id_action in my_next_state_id_actions:
                my_eval.append(self.evaluateDefensiveState(state, self.index))
            if my_eval != []:
                best_eval = max(my_eval)
                best_states_id_action = filter(lambda x: x[0] == best_eval, zip(my_eval, my_next_state_id_actions))
                next_action = random.choice(best_states_id_action)[1][1][0][1]
                if next_action in my_legal_actions:
                    print time.time() - timer ####### <timer> #######
                    return next_action
        #######################################
        print time.time() - timer ####### <timer> #######
        return random.choice(my_legal_actions)

    def chooseAction(self, current_state):
        my_legal_actions = current_state.getLegalActions(self.index)
        eaten_food_list = self.getEatenDefFoodPos(current_state, self.index)
        self.last_eaten_def_foods = eaten_food_list if eaten_food_list != [] else self.last_eaten_def_foods
        #print self.last_eaten_def_foods
        my_next_state_id_actions = self.generateMyNextStatesIdAction(current_state)
        my_eval = []
        for state, id_action in my_next_state_id_actions:
            my_eval.append(self.evaluateDefensiveState(state, self.index))
        if my_eval != []:
            best_eval = max(my_eval)
            best_states_id_action = filter(lambda x: x[0] == best_eval, zip(my_eval, my_next_state_id_actions))
            next_action = random.choice(best_states_id_action)[1][1][0][1]
            if next_action in my_legal_actions:
                return next_action
        return random.choice(my_legal_actions)

    def singleDefensiveExpanding(self, current_state):
        # generate my layer of state
        my_layer_states_ids_actions = self.generateMyNextStatesIdAction(current_state)
        # for each of my next possible state, generate enemy's next states
        my_id_actions_enemy_avg_diff = []
        for my_state, my_id_action in my_layer_states_ids_actions:
            temp_group_enemy_states_ids_actions = self.generateNextEnemyStatesIdAction(my_state)
            if temp_group_enemy_states_ids_actions != []:
                temp_enemy_group_evals = map(lambda e_state_id_action: self.evaluateOffensiveAverageEnemyState(e_state_id_action[0]), temp_group_enemy_states_ids_actions)
                temp_enemy_group_not_null_evals = filter(lambda x: x != None, temp_enemy_group_evals)
                if temp_enemy_group_not_null_evals != []:
                    temp_avg_enemy_eval = sum(temp_enemy_group_not_null_evals) / len(temp_enemy_group_not_null_evals)
                    my_id_actions_enemy_avg_diff.append((my_id_action, self.evaluateOffensiveState(my_state, self.index) - temp_avg_enemy_eval, my_state))
        # find the largest diff between my and enemy's layer of states
        if my_id_actions_enemy_avg_diff != []:
            best_diff = max(map(lambda x: x[1], my_id_actions_enemy_avg_diff))
            best_my_id_actions_enemy_avg_diff = filter(lambda x: x[1] == best_diff, my_id_actions_enemy_avg_diff)
            return random.choice(best_my_id_actions_enemy_avg_diff)
        else:
            return None

    def evaluateOffensiveAverageEnemyState(self, state):
        #print "def eval offensive"
        enemy_ids = self.getOpponents(state)
        enemy_visibilities = map(lambda eid: True if state.getAgentPosition(eid) != None else False, enemy_ids)
        enemy_id_visibilities = zip(enemy_ids, enemy_visibilities)
        visible_enemy_ids = filter(lambda x: x[1], enemy_id_visibilities) #[(id, True)]
        visible_enemy_counts = len(visible_enemy_ids)
        if visible_enemy_counts != 0:
            enemy_eval = 0
            for eid, _ in visible_enemy_ids:
                enemy_eval += self.evaluateOffensiveState(state, eid)
            return enemy_eval / visible_enemy_counts
        else:
            return None


################################################################################
############################## support functions ###############################
################################################################################

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
