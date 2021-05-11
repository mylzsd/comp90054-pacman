from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

def createTeam(firstIndex, secondIndex, isRed,
               first = 'YooAgent', second = 'YooAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

### CONSTANT VALUES
MAX_VALUE = 99999999
MIN_VALUE = -99999999
### FEATURE TAGS
T1 = "tag1"
T2 = "tag2"
T3 = "tag3"
T4 = "tag4"
T5 = "tag5"
T6 = "tag6"
T7 = "tag7"
T8 = "tag8"
T9 = "tag9"
T10 = "tag10"
T11 = "tag11"
T12 = "tag12"
T13 = "tag13"
T14 = "tag14"
T15 = "tag15"
T16 = "tag16"
T17 = "tag17"
T18 = "tag18"
T19 = "tag19"
T20 = "tag20"


class YooAgent(CaptureAgent):

    def registerInitialState(self, current_state):
        CaptureAgent.registerInitialState(self, current_state)
        ### static tags
        self.friendly_ids = []
        self.opponent_ids = []
        self.walls = []
        self.game_width = 0
        self.game_height = 0
        self.walkable_pos = []
        self.home_pos = (0, 0)
        ### performing the initialization
        self.initializeGameTags(current_state)
        ### dynamic tags that perhaps need to update at each step/ action performed
        self.isPacman = True
        self.isThreatened = False
        self.carried_foods = 0
        self.food_to_eat_counts = 0
        self.food_to_def_counts = 0
        self.last_action = "Stop"
        self.last_def_food_eaten_pos = (-1, -1)

        ### tags that need to be updated manually
        self.food_score_dictionary = {}
        ### performe updates
        self.updateGameTags(current_state)

    def chooseAction(self, current_state):
        self.updateGameTags(current_state)
        next_action = self.nextReflexAction(current_state)
        self.last_action = next_action
        return next_action

    def nextReflexAction(self, current_state):
        next_actions = current_state.getLegalActions(self.index)
        #print "Agent " + str(self.index)
        if next_actions != []:
            max_val = MIN_VALUE
            max_action = next_actions[0]
            for next_action in next_actions:
                eva_score = self.evaluateGameState(current_state, next_action)
                if max_val < eva_score:
                    max_val = eva_score
                    max_action = next_action
            return max_action
        else:
            return "Stop"

    def evaluateGameState(self, current_state, action):
        eva_score = self.getFeatures(current_state, action) * self.getWeights()
        return eva_score

    def getFeatures(self, current_state, action):
        f = util.Counter()
        f[T1] = self.feature_MazeDistToNearestFood(current_state, action)
        f[T2] = self.feature_MazeDistToFarestFood(current_state, action)
        f[T3] = self.feature_mazeDistToHome(current_state, action) * self.carried_foods
        f[T4] = self.feature_mazeDistToEnemyPacman(current_state, action)
        f[T5] = self.feature_mazeDistToEnemyGhost(current_state, action)
        f[T6] = self.feature_FoodToEatCounts(current_state, action)
        f[T7] = self.feature_FoodToDefCounts(current_state, action)
        f[T8] = self.feature_ScoreBeatenOpponents(current_state, action)
        f[T9] = self.feature_InvaderCounts(current_state, action)
        f[T10] = self.feature_HoldingFoodCounts(current_state, action)
        f[T11] = self.feature_StopAction(current_state, action)
        f[T12] = self.feature_DeathThreatLevel(current_state, action)
        f[T13] = self.feature_DuplicatedAction(current_state, action)
        f[T14] = self.feature_MazeDistToFoodWithHighestScore(current_state, action)
        return f

    def getWeights(self):
        w = util.Counter()
        w[T1] = -5 if self.index % 2 == 0 else 0            #feature_MazeDistToNearestFood
        w[T2] = -5 if self.index % 2 == 1 else 0            #feature_MazeDistToFarestFood
        w[T3] = -10                                         #feature_mazeDistToHome
        w[T4] = -50 if not self.isThreatened else 0         #feature_mazeDistToEnemyPacman
        w[T5] = 50                                          #feature_mazeDistToEnemyGhost
        w[T6] = -10                                         #feature_FoodToEatCounts
        w[T7] = 50                                          #feature_FoodToDefCounts
        w[T8] = 10                                          #feature_ScoreBeatenOpponents
        w[T9] = -500000                                     #feature_InvaderCounts
        w[T10] = 0                                          #feature_HoldingFoodCounts
        w[T11] = -100                                       #feature_StopAction
        w[T12] = -1000000                                   #feature_DeathThreatLevel
        w[T13] = -20                                        #feature_DuplicatedAction
        w[T14] = -50                                        #feature_MazeDistToFoodWithHighestScore
        return w

    ##############################################################
    #################### feature functions #######################
    ##############################################################

    ### T1
    def feature_MazeDistToNearestFood(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        #foods_to_eat = self.getFood(next_state).asList()
        foods_to_eat = self.getFood(current_state).asList()
        next_agent_pos = next_state.getAgentPosition(self.index)
        if foods_to_eat != []:
            min_dist = MAX_VALUE
            for food_pos in foods_to_eat:
                temp_dist = self.getMazeDistance(next_agent_pos, food_pos)
                if min_dist > temp_dist:
                    min_dist = temp_dist
            return min_dist
        else:
            return 0

    ### T2
    def feature_MazeDistToFarestFood(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        #foods_to_eat = self.getFood(next_state).asList()
        foods_to_eat = self.getFood(current_state).asList()
        next_agent_pos = next_state.getAgentPosition(self.index)
        if foods_to_eat != []:
            max_dist = MIN_VALUE
            for food_pos in foods_to_eat:
                temp_dist = self.getMazeDistance(next_agent_pos, food_pos)
                if max_dist < temp_dist:
                    max_dist = temp_dist
            return max_dist
        else:
            return 0

    ### T3
    def feature_mazeDistToHome(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        next_pos = next_state.getAgentPosition(self.index)
        return self.getMazeDistance(next_pos, self.home_pos)

    ### T4
    def feature_mazeDistToEnemyPacman(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        next_pos = next_state.getAgentPosition(self.index)
        enemy_pacman_ids = []
        for opponent_id in self.opponent_ids:
            if next_state.getAgentState(opponent_id).isPacman:
                enemy_pacman_ids.append(opponent_id)
        if enemy_pacman_ids != []:
            distances = []
            for enemy_pacman_id in enemy_pacman_ids:
                temp_pos = next_state.getAgentPosition(enemy_pacman_id)
                if temp_pos != None:
                    distances.append(self.getMazeDistance(temp_pos, next_pos))
            if distances == [] and self.last_def_food_eaten_pos != (-1, -1):
                distances.append(self.getMazeDistance(self.last_def_food_eaten_pos, next_pos))
            min_dist = 9999
            for dist in distances:
                if min_dist > dist:
                    min_dist = dist
            return min_dist
        else:
            return 9999

    ### T5
    def feature_mazeDistToEnemyGhost(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        next_pos = next_state.getAgentPosition(self.index)
        enemy_ghost_ids = []
        for opponent_id in self.opponent_ids:
            if not next_state.getAgentState(opponent_id).isPacman:
                enemy_ghost_ids.append(opponent_id)
        if enemy_ghost_ids != []:
            distances = []
            for enemy_ghost_id in enemy_ghost_ids:
                temp_pos = next_state.getAgentPosition(enemy_ghost_id)
                if temp_pos != None:
                    distances.append((self.getMazeDistance(temp_pos, next_pos), enemy_ghost_id))
            min_dist = 9999
            min_ghost_id = -1
            for dist, e_id in distances:
                if min_dist > dist:
                    min_dist = dist
                    min_ghost_id = e_id
            if min_dist < 3 and current_state.getAgentState(min_ghost_id).scaredTimer == 0:
                return min_dist
            else:
                return 9999
        else:
            return 9999

    ### T6
    def feature_FoodToEatCounts(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        return self.getFoodToEatCounts(next_state)

    ### T7
    def feature_FoodToDefCounts(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        return self.getFoodToDefCounts(next_state)

    ### T8
    def feature_ScoreBeatenOpponents(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        return self.getScore(next_state)

    ### T9
    def feature_InvaderCounts(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        inv_counts = 0
        for opponent_id in self.opponent_ids:
            if next_state.getAgentState(opponent_id).isPacman:
                inv_counts += 1
        return inv_counts

    ### T10
    def feature_HoldingFoodCounts(self, current_state, action):
        return self.carried_foods

    ### T11
    def feature_StopAction(self, current_state, action):
        return 1 if action == "Stop" else 0

    ### T12
    def feature_DeathThreatLevel(self, current_state, action):
        next_state = current_state.generateSuccessor(self.index, action)
        nx, ny = next_state.getAgentPosition(self.index)
        opponent_ghost_pos = []
        for opponent_id in self.opponent_ids:
            temp_pos = next_state.getAgentPosition(opponent_id)
            if temp_pos != None and self.isPacman and not next_state.getAgentState(opponent_id).isPacman:
                opponent_ghost_pos.append(temp_pos)
        # in 3x3 grid
        ghost_counts = 0
        for x in range(nx - 1, nx + 2):
            for y in range(ny - 1, ny + 2):
                if (x, y) in opponent_ghost_pos:
                    ghost_counts += 1
        return ghost_counts

    ### T13
    def feature_DuplicatedAction(self, current_state, action):
        return 1 if action == self.getReversedAction(self.last_action) else 0

    ### T14
    def feature_MazeDistToFoodWithHighestScore(self, current_state, action):
        if self.food_score_dictionary != {}:
            next_state = current_state.generateSuccessor(self.index, action)
            next_pos = next_state.getAgentPosition(self.index)
            max_food_score = MIN_VALUE
            max_food_pos = (-1, -1)
            for k, v in self.food_score_dictionary.items():
                if max_food_score < v:
                    max_food_score = v
                    max_food_pos = k
            return self.getMazeDistance(next_pos, max_food_pos)
        else:
            return self.feature_MazeDistToNearestFood(current_state, action)
    ##############################################################
    #################### supporting functions ####################
    ##############################################################
    def initializeGameTags(self, current_state):
        self.friendly_ids = self.getTeam(current_state)
        self.opponent_ids = self.getOpponents(current_state)
        self.walls = current_state.getWalls().asList()
        self.game_width = current_state.data.layout.width
        self.game_height = current_state.data.layout.height
        self.walkable_pos = []
        for x in range(0, self.game_width):
            for y in range(0, self.game_height):
                if not (x, y) in self.walls:
                    self.walkable_pos.append((x, y))
        self.home_pos = current_state.getInitialAgentPosition(self.index)

    def updateGameTags(self, current_state):
        self.isPacman = current_state.getAgentState(self.index).isPacman
        self.isThreatened = False if current_state.getAgentState(self.index).scaredTimer == 0 else True
        self.carried_foods = current_state.getAgentState(self.index).numCarrying
        self.food_to_eat_counts = self.getFoodToEatCounts(current_state)
        self.food_to_def_counts = self.getFoodToDefCounts(current_state)
        if self.getDefFoodEatenPos() != (-1, -1):
            self.last_def_food_eaten_pos = self.getDefFoodEatenPos()
        self.food_score_dictionary.clear()
        food_pos_list = self.getFood(current_state).asList()
        #print len(food_pos_list)
        for fx, fy in food_pos_list:
            ### 3x3 grid inspection
            temp_score = -self.getMazeDistance(current_state.getAgentPosition(self.index), (fx, fy))
            for x in range(fx - 1, fx + 2):
                for y in range(fy - 1, fy + 2):
                    if (x, y) in food_pos_list and (x, y) != (fx, fy):
                        temp_score += 2
                    if (x, y) in self.walls:
                        temp_score -= 1
            self.food_score_dictionary[(fx, fy)] = temp_score
        print self.food_score_dictionary

    def getFoodToEatCounts(self, current_state):
        return len(self.getFood(current_state).asList())

    def getFoodToDefCounts(self, current_state):
        return len(self.getFoodYouAreDefending(current_state).asList())

    def getReversedAction(self, action):
        if action == "Stop":
            return "Stop"
        if action == "North":
            return "South"
        if action == "South":
            return "North"
        if action == "West":
            return "East"
        if action == "East":
            return "West"

    def getDefFoodEatenPos(self):
        try:
            last_def_foods = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()
            current_def_foods = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
            for last_pos in last_def_foods:
                if last_pos not in current_def_foods:
                    return last_pos
            return (-1, -1)
        except:
            return (-1, -1)


