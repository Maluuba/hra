import os
from copy import deepcopy
import pygame
import numpy as np
import click


# RGB colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
WALL = (80, 80, 80)


class FruitCollection(object):
    def __init__(self, game_length=300, lives=1e6, state_mode='pixel', is_fruit=True, is_ghost=True,
                 rng=None, rendering=False, image_saving=False, render_dir=None):
        self.game_length = game_length
        self.lives = lives
        self.is_fruit = is_fruit
        self.is_ghost = is_ghost
        self.legal_actions = [0, 1, 2, 3]
        self.action_meanings = ['up', 'down', 'left', 'right']
        self.reward_scheme = {'ghost': -10.0, 'fruit': +1.0, 'step': 0.0, 'wall': 0.0}
        self.nb_actions = len(self.legal_actions)
        if rng is None:
            self.rng = np.random.RandomState(1234)
        else:
            self.rng = rng
        self.player_pos_x = None
        self.player_pos_y = None
        self.agent_init_pos = None
        self.pass_wall_rows = None
        self.init_lives = deepcopy(self.lives)
        self.step_reward = 0.0
        self.possible_fruits = None
        self.state_mode = state_mode    # how the returned state look like ('pixel' or '1hot' or 'multi-head')
        self.nb_fruits = None
        self.scr_w = None
        self.scr_h = None
        self.rendering_scale = None
        self.walls = None
        self.fruits = None
        self.ghosts = None
        self.init_with_mode()
        self.nb_non_wall = self.scr_w * self.scr_h - len(self.walls)
        self.init_ghosts = deepcopy(self.ghosts)
        self._rendering = rendering
        if rendering:
            self._init_pygame()
        self.image_saving = image_saving
        self.render_dir_main = render_dir
        self.render_dir = None
        self.targets = None  # fruits + ghosts
        self.active_targets = None  # boolean list
        self.active_fruits = None
        self.nb_targets = None
        self.init_targets = None
        self.nb_ghosts = None
        self.soc_state_shape = None
        self.state_shape = None
        self.state = None
        self.step_id = 0
        self.game_over = False
        self.mini_target = []  # only is used for mini
        self.reset()

    def init_with_mode(self):
        raise NotImplementedError

    @property
    def rendering(self):
        return self._rendering

    @rendering.setter
    def rendering(self, flag):
        if flag is True:
            if self._rendering is False:
                self._init_pygame()
                self._rendering = True
        else:
            self.close()
            self._rendering = False

    def _init_pygame(self):
        pygame.init()
        size = [self.rendering_scale * self.scr_w, self.rendering_scale * self.scr_h]
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Fruit Collection")

    def _init_rendering_folder(self):
        if self.render_dir_main is None:
            self.render_dir_main = 'render'
        if not os.path.exists(os.path.join(os.getcwd(), self.render_dir_main)):
            os.mkdir(os.path.join(os.getcwd(), self.render_dir_main))
        i = 0
        while os.path.exists(os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))):
            i += 1
        self.render_dir = os.path.join(os.getcwd(), self.render_dir_main, 'render' + str(i))
        os.mkdir(self.render_dir)

    def reset(self):
        if self.image_saving:
            self._init_rendering_folder()
        self.game_over = False
        self.step_id = 0
        self._reset_targets()
        self.nb_ghosts = len(self.ghosts)
        self.targets = deepcopy(self.fruits) + deepcopy(self.ghosts)
        self.nb_targets = len(self.targets)
        self.active_targets = self.active_fruits + [True] * len(self.ghosts)
        self.lives = deepcopy(self.init_lives)
        self.soc_state_shape = [self.scr_w, self.scr_h, self.scr_w + 1, self.scr_h + 1]
        if self.state_mode == '1hot':
            self.state_shape = [self.nb_non_wall * self.nb_fruits + self.nb_ghosts * (self.nb_non_wall ** 2)]
        elif self.state_mode == 'pixel':
            self.state_shape = [4, self.scr_w, self.scr_h]
        elif self.state_mode == 'multi-head':
            self.state_shape = [3 * self.scr_w * self.scr_h]
        elif self.state_mode == 'mini':
            self.state_shape = [100 + len(self.possible_fruits)]

    def _reset_targets(self):
        raise NotImplementedError

    def close(self):
        if self.rendering:
            pygame.quit()

    def _move_player(self, action):
        assert action in self.legal_actions, 'Illegal action.'
        hit_wall = False
        if action == 3:  # right
            passed_wall = False
            if self.pass_wall_rows is not None:
                for wall_row in self.pass_wall_rows:
                    if [self.player_pos_x, self.player_pos_y] == [self.scr_w - 1, wall_row]:
                        self.player_pos_x = 0
                        passed_wall = True
                        break
            if not passed_wall:
                if [self.player_pos_x + 1, self.player_pos_y] not in self.walls and self.player_pos_x < self.scr_w - 1:
                    self.player_pos_x += 1
                else:
                    hit_wall = True
        elif action == 2:  # left
            passed_wall = False
            if self.pass_wall_rows is not None:
                for wall_row in self.pass_wall_rows:
                    if [self.player_pos_x, self.player_pos_y] == [0, wall_row]:
                        self.player_pos_x = self.scr_w - 1
                        passed_wall = True
                        break
            if not passed_wall:
                if [self.player_pos_x - 1, self.player_pos_y] not in self.walls and self.player_pos_x > 0:
                    self.player_pos_x -= 1
                else:
                    hit_wall = True
        elif action == 1:  # down
            if [self.player_pos_x, self.player_pos_y + 1] not in self.walls and self.player_pos_y < self.scr_h - 1:
                self.player_pos_y += 1
            else:
                hit_wall = True
        elif action == 0:  # up
            if [self.player_pos_x, self.player_pos_y - 1] not in self.walls and self.player_pos_y > 0:
                self.player_pos_y -= 1
            else:
                hit_wall = True
        return hit_wall

    def _check_fruit(self):
        if not self.is_fruit:
            return None
        caught_target = None
        caught_target_idx = None
        target_count = -1
        for k, target in enumerate(self.targets):
            target_count += 1
            if target['reward'] < 0:  # not fruit
                continue
            if target['location'] == [self.player_pos_x, self.player_pos_y] and target['active'] is True:
                caught_target = deepcopy([self.player_pos_y, self.player_pos_x])
                caught_target_idx = k
                target['active'] = False
                target['location'] = [self.scr_w, self.scr_h]  # null value
                break
        check = []
        for target in self.targets:
            if target['reward'] > 0:
                check.append(target['active'])
        if True not in check:
            self.game_over = True
        return caught_target, caught_target_idx

    def _check_ghost(self):
        if not self.is_ghost:
            return None
        caught_target = None
        for k, target in enumerate(self.targets):
            if target['reward'] > 0:  # not ghost
                continue
            if target['location'] == [self.player_pos_x, self.player_pos_y] and target['active'] is True:
                caught_target = k
                # target['active'] = False
                target['locations'] = [self.scr_w, self.scr_h]  # null value
                self.lives -= 1
                break
        return caught_target

    def _move_ghosts(self):
        if not self.is_ghost:
            return
        for target in self.targets:
            if target['reward'] < 0:
                loc = target['location']
                not_moved = True
                while not_moved:
                    direction = self.rng.randint(0, 4)
                    if direction == 0 and loc[0] < self.scr_w - 1 and [loc[0] + 1, loc[1]] not in self.walls:
                        loc[0] += 1
                        not_moved = False
                    elif direction == 1 and loc[0] > 0 and [loc[0] - 1, loc[1]] not in self.walls:
                        loc[0] -= 1
                        not_moved = False
                    elif direction == 2 and loc[1] < self.scr_h - 1 and [loc[0], loc[1] + 1] not in self.walls:
                        loc[1] += 1
                        not_moved = False
                    elif direction == 3 and loc[1] > 0 and [loc[0], loc[1] - 1] not in self.walls:
                        loc[1] -= 1
                        not_moved = False

    def get_state(self):
        if self.state_mode == 'pixel':
            return self.get_state_pixel()
        elif self.state_mode == '1hot':
            return self.get_1hot_features()
        elif self.state_mode == 'multi-head':
            return self.get_state_multi_head()
        elif self.state_mode == 'mini':
            return self.get_mini_state()
        else:
            raise ValueError('State-mode is not known.')

    def get_mini_state(self):
        state = np.zeros((self.scr_w * self.scr_h + len(self.possible_fruits)), dtype=np.int8)
        state[self.player_pos_y * self.scr_h + self.player_pos_x] = 1
        for target in self.targets:
            if target['active'] and target['reward'] > 0:
                offset = self.possible_fruits.index([target['location'][1], target['location'][0]])
                index = (self.scr_w * self.scr_h) + offset
                state[index] = 1
        return state

    def get_state_multi_head(self):
        # three binary heads: player, fruits, ghosts
        state = np.zeros(3 * self.scr_w * self.scr_h, dtype=np.int8)
        state[self.player_pos_y * self.scr_h + self.player_pos_x] = 1
        for target in self.targets:
            if target['active']:
                if target['reward'] > 0:
                    index = (self.scr_w * self.scr_h) + (target['location'][1] * self.scr_h + target['location'][0])
                else:
                    index = 2 * (self.scr_w * self.scr_h) + \
                            (target['location'][1] * self.scr_h + target['location'][0])
                state[index] = 1
        return state

    def get_state_pixel(self):
        state = np.zeros((self.state_shape[1], self.state_shape[2], self.state_shape[0]), dtype=np.int8)
        # walls, fruits, player, ghost
        player_pos = [self.player_pos_x, self.player_pos_y]
        fruits = []
        ghosts = []
        for target in self.targets:
            if target['active'] is True:
                if target['reward'] > 0:
                    fruits.append(target['location'])
                elif target['reward'] < 0:
                    ghosts.append(target['location'])
        for loc in fruits:
            if loc in ghosts and self.is_ghost:
                # state[tuple(loc)] = self.code['fruit+ghost']
                state[tuple(loc)][1] = 1
                state[tuple(loc)][3] = 1
                ghosts.remove(loc)
            else:
                state[tuple(loc)][1] = 1
                # state[tuple(loc)] = self.code['fruit']
        if player_pos in ghosts and self.is_ghost:
            state[tuple(player_pos)][2] = 1
            state[tuple(player_pos)][3] = 1
            ghosts.remove(player_pos)
        else:
            state[tuple(player_pos)][2] = 1
        if self.is_ghost:
            for loc in ghosts:
                state[tuple(loc)][3] = 1
                # state[tuple(loc)] = self.code['ghost']
        for loc in self.walls:
            state[tuple(loc)][0] = 1
            # state[tuple(loc)] = self.code['wall']
        return deepcopy(state.T)

    def get_soc_state(self):
        # call this after each step to get SoC state list (len = self.nb_targets)
        # returns list of 4-tuples; one 4-tuple for each target.
        state = []
        for target in self.targets:
            target_state = [self.player_pos_x, self.player_pos_y]
            if target['active'] is True:
                target_state.extend(target['location'])
            else:
                target_state.extend([self.scr_w, self.scr_h])
            state.append(target_state)
        return deepcopy(state)

    def get_1hot_features(self):
        agent_idx = self._get_idx(self.player_pos_x, self.player_pos_y)
        agent_state = np.zeros(self.nb_non_wall, dtype=np.int8)
        agent_state[agent_idx] = 1
        state = np.zeros(self.state_shape, dtype=np.int8)
        i = -1
        for target in self.targets:
            if target['reward'] > 0:
                i += 1
                if target['active'] is True:
                    state[i * self.nb_non_wall: (i + 1) * self.nb_non_wall] = agent_state.copy()
        ghost_indices = []
        for target in self.targets:
            if target['reward'] < 0:
                ghost_indices.append(self._get_idx(target['location'][0], target['location'][1]))
        last_fruit_pointer = self.nb_non_wall * self.nb_fruits
        for i, ghost_idx in enumerate(ghost_indices):
            ghost_agent_idx = agent_idx * self.nb_non_wall + ghost_idx
            state[last_fruit_pointer + i * (self.nb_non_wall ** 2) + ghost_agent_idx] = 1
        return state.copy()

    def _get_idx(self, x, y):
        assert [x, y] not in self.walls
        idx = 0
        flag = False
        for i in range(self.scr_w):
            for j in range(self.scr_h):
                if [i, j] in self.walls:
                    continue
                if [i, j] == [x, y]:
                    flag = True
                    break
                else:
                    idx += 1
            if flag:
                break
        return idx

    def step(self, action):
        # actions: [0, 1, 2, 3] == [up, down, left, right]
        if self.game_over:
            raise ValueError('Environment has already been terminated.')
        if self.step_id >= self.game_length - 1:
            self.game_over = True
            return self.get_state(), 0., self.game_over, \
                   {'ghost': None, 'fruit': None, 'hit_wall': False,
                    'head_reward': np.zeros(len(self.possible_fruits), dtype=np.float32)}
        last_player_position = deepcopy([self.player_pos_x, self.player_pos_y])
        hit_wall = self._move_player(action)
        if hit_wall:
            wall_reward = self.reward_scheme['wall']
        else:
            wall_reward = 0.0
        possible_caught_ghost = self._check_ghost()
        if possible_caught_ghost is not None:
            last_ghost_position = deepcopy(self.targets[possible_caught_ghost]['location'])
        self._move_ghosts()
        swap_flag = False  # in a T-situation it is possible that no hit happens
        if possible_caught_ghost is not None:
            if last_player_position == self.targets[possible_caught_ghost]['location'] and \
                            last_ghost_position == [self.player_pos_x, self.player_pos_y]:
                swap_flag = True
        # check for ghost hit head-to-head after the moves
        caught_ghost = self._check_ghost()
        if caught_ghost is None and swap_flag:  # if a swap occurred
            caught_ghost = possible_caught_ghost
        if caught_ghost is not None:
            ghost_reward = self.reward_scheme['ghost']
        else:
            ghost_reward = 0.
        caught_fruit, caught_fruit_idx = self._check_fruit()
        head_reward = np.zeros(len(self.possible_fruits), dtype=np.float32)
        if caught_fruit is not None:
            fruit_reward = self.reward_scheme['fruit']
            head_reward[self.possible_fruits.index(caught_fruit)] = 1.
        else:
            fruit_reward = 0.
        if self.lives == 0:
            self.game_over = True
        self.step_id += 1
        return self.get_state(), ghost_reward + fruit_reward + wall_reward, \
               self.game_over, {'fruit': caught_fruit_idx, 'ghost': caught_ghost, 'head_reward': head_reward,
                                'hit_wall': hit_wall}

    def render(self):
        if not self.rendering:
            return
        pygame.event.pump()
        self.screen.fill(BLACK)
        size = [self.rendering_scale, self.rendering_scale]
        player = pygame.Rect(self.rendering_scale * self.player_pos_x, self.rendering_scale * self.player_pos_y,
                             size[0], size[1])
        pygame.draw.rect(self.screen, WHITE, player)
        for target in self.targets:
            if target['active'] is True:
                pos = target['location']
                p = [self.rendering_scale * pos[0], self.rendering_scale * pos[1]]
                gl = pygame.Rect(p[0], p[1], size[0], size[1])
                pygame.draw.rect(self.screen, target['colour'], gl)
        for wall_pos in self.walls:
            p = [self.rendering_scale * wall_pos[0], self.rendering_scale * wall_pos[1]]
            wall = pygame.Rect(p[0], p[1], size[0], size[1])
            pygame.draw.rect(self.screen, WALL, wall)
        pygame.display.flip()

        if self.image_saving:
            self.save_image()

    def save_image(self):
        if self.rendering and self.render_dir is not None:
            pygame.image.save(self.screen, self.render_dir + '/render' + str(self.step_id) + '.jpg')
        else:
            raise ValueError('env.rendering is False and/or environment has not been reset.')


class FruitCollectionSmall(FruitCollection):
    def init_with_mode(self):
        self.nb_fruits = None
        self.scr_w = 11
        self.scr_h = 11
        self.rendering_scale = 40
        self.walls = [[4, 0], [6, 0], [1, 1], [2, 1], [4, 1], [6, 1], [8, 1], [9, 1], [1, 3], [3, 3], [4, 3],
                      [6, 3], [7, 3], [9, 3], [1, 4], [3, 4], [4, 4], [6, 4], [7, 4], [9, 4], [1, 5], [9, 5],
                      [1, 6], [2, 6], [3, 6], [5, 6], [7, 6], [8, 6], [9, 6], [0, 8], [1, 8], [2, 8], [4, 8],
                      [5, 8], [6, 8], [8, 8], [9, 8], [10, 8], [4, 9], [5, 9], [6, 9], [1, 10], [2, 10], [8, 10],
                      [9, 10]]
        if self.is_ghost:
            if self.is_fruit:
                self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 5],
                                'active': True},
                               {'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [4, 5],
                                'active': True}]
            else:
                self.ghosts = []  # will be reset
        else:
            self.ghosts = []

    def _reset_targets(self):
        if self.is_ghost and not self.is_fruit:
            if self.rng.binomial(1, 0.5):
                self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 5],
                                'active': True}]
            else:
                self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [4, 5],
                                'active': True}]
        [self.player_pos_x, self.player_pos_y] = deepcopy([self.scr_w - 1, self.scr_h - 1])
        # Targets:  Format: [ {colour: c1, reward: r1, locations: list_l1, 'active': list_a1}, ... ]
        occupied = self.walls + [[self.player_pos_x, self.player_pos_y]]
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    if [x, y] not in occupied:
                        if self.rng.binomial(1, 0.5):
                            self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                                'location': [x, y], 'active': True})
                            self.active_fruits.append(True)
                        else:
                            self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                                'location': [x, y], 'active': False})
                            self.active_fruits.append(False)
            self.nb_fruits = len(self.fruits)


class FruitCollectionMini(FruitCollection):
    def init_with_mode(self):
        self.is_ghost = False
        self.is_fruit = True
        self.nb_fruits = 5
        self.possible_fruits = [[0, 0], [0, 9], [1, 2], [3, 6], [4, 4], [5, 7], [6, 2], [7, 7], [8, 8], [9, 0]]
        self.scr_w = 10
        self.scr_h = 10
        self.rendering_scale = 50
        self.walls = []
        if self.is_ghost:
            self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 1],
                            'active': True}]
        else:
            self.ghosts = []

    def _reset_targets(self):
        while True:
            self.player_pos_x, self.player_pos_y = self.rng.randint(0, self.scr_w), self.rng.randint(0, self.scr_h)
            if [self.player_pos_x, self.player_pos_y] not in self.possible_fruits:
                break
        # Targets:  Format: [ {colour: c1, reward: r1, locations: list_l1, 'active': list_a1}, ... ]
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                        'location': [x, y], 'active': False})
                    self.active_fruits.append(False)
            fruits_idx = deepcopy(self.possible_fruits)
            self.rng.shuffle(fruits_idx)
            fruits_idx = fruits_idx[:self.nb_fruits]
            self.mini_target = [False] * len(self.possible_fruits)
            for f in fruits_idx:
                idx = f[1] * self.scr_w + f[0]
                self.fruits[idx]['active'] = True
                self.active_fruits[idx] = True
                self.mini_target[self.possible_fruits.index(f)] = True


class FruitCollectionLarge(FruitCollection):
    def init_with_mode(self):
        self.nb_fruits = None
        self.scr_w = 21
        self.scr_h = 14
        self.rendering_scale = 30
        self.pass_wall_rows = [4, 8]
        self.walls = [[0, 0], [5, 0], [15, 0], [20, 0],
                      [0, 1], [2, 1], [3, 1], [5, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1], [12, 1],
                      [13, 1], [15, 1], [17, 1], [18, 1], [20, 1],
                      [0, 2], [20, 2],
                      [0, 3], [1, 3], [3, 3], [5, 3], [6, 3], [8, 3], [9, 3], [10, 3], [11, 3], [12, 3],
                      [14, 3], [15, 3], [17, 3], [19, 3], [20, 3],
                      [3, 4], [17, 4],
                      [0, 5], [1, 5], [3, 5], [4, 5], [5, 5], [6, 5], [8, 5], [9, 5], [10, 5], [11, 5],
                      [12, 5], [14, 5], [15, 5], [16, 5], [17, 5], [19, 5], [20, 5],
                      [0, 6], [1, 6], [8, 6], [9, 6], [10, 6], [11, 6], [12, 6], [19, 6], [20, 6],
                      [0, 7], [1, 7], [3, 7], [4, 7], [5, 7], [6, 7], [8, 7], [9, 7], [10, 7], [11, 7],
                      [12, 7], [14, 7], [15, 7], [16, 7], [17, 7], [19, 7], [20, 7],
                      [3, 8], [17, 8],
                      [0, 9], [1, 9], [3, 9], [5, 9], [7, 9], [9, 9], [10, 9], [11, 9], [13, 9], [15, 9],
                      [17, 9], [19, 9], [20, 9],
                      [0, 10], [5, 10], [7, 10], [13, 10], [15, 10], [20, 10],
                      [0, 11], [2, 11], [3, 11], [5, 11], [9, 11], [10, 11], [11, 11], [15, 11], [17, 11],
                      [18, 11], [20, 11],
                      [0, 12], [2, 12], [3, 12], [5, 12], [6, 12], [7, 12], [9, 12], [10, 12], [11, 12],
                      [13, 12], [14, 12], [15, 12], [17, 12], [18, 12], [20, 12],
                      [0, 13], [20, 13]]
        if self.is_ghost:
            self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [1, 4],
                            'active': True},
                           {'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [8, 4],
                            'active': True},
                           {'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [12, 4],
                            'active': True},
                           {'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [19, 4],
                            'active': True}]
        else:
            self.ghosts = []

    def _reset_targets(self):
        self.ghosts = deepcopy(self.init_ghosts)
        [self.player_pos_x, self.player_pos_y] = [10, 8]
        # Targets:  Format: [ {colour: c1, reward: r1, locations: list_l1, 'active': list_a1}, ... ]
        occupied = self.walls + [[self.player_pos_x, self.player_pos_y]]
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    if [x, y] not in occupied:
                        if self.rng.binomial(1, 0.5):
                            self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                                'location': [x, y], 'active': True})
                            self.active_fruits.append(True)
                        else:
                            self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                                'location': [x, y], 'active': False})
                            self.active_fruits.append(False)
            self.nb_fruits = len(self.fruits)


@click.command()
@click.option('--mode', '-m', help="'small' or 'large' or 'mini'")
@click.option('--fruit/--no-fruit', default=True, help='Activates fruits.')
@click.option('--ghost/--no-ghost', default=True, help='Activates ghosts.')
@click.option('--save/--no-save', default=False, help='Saving rendering screen.')
def test(mode, fruit, ghost, save):
    if mode == 'small':
        e = FruitCollectionSmall
    elif mode == 'mini':
        e = FruitCollectionMini
    elif mode == 'large':
        e = FruitCollectionLarge
    else:
        raise ValueError('Incorrect mode.')
    env = e(rendering=True, lives=1, is_fruit=fruit, is_ghost=ghost, image_saving=save)
    print('state shape', env.state_shape)
    for _ in range(10):
        env.reset()
        env.render()
        while not env.game_over:
            action = None
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 0
                    if event.key == pygame.K_DOWN:
                        action = 1
                    if event.key == pygame.K_LEFT:
                        action = 2
                    if event.key == pygame.K_RIGHT:
                        action = 3
                    if event.key == pygame.K_q:
                        return
            if action is None:
                continue
            obs, r, term, info = env.step(action)
            env.render()
            print("\033[2J\033[H\033[2J", end="")
            print()
            print('pos: ', env.player_pos_x, env.player_pos_y)
            print('reward: ', r)
            print('state:')
            print('─' * 30)
            print(obs)
            print('─' * 30)


if __name__ == '__main__':
    test()
