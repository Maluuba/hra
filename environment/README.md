 # Fruit-Collection Environment
This environment provides the game of Fruit-Collection (a fruit-collection and ghost-avoidance game), which is a good testbed for several RL algorithms. We have used different variation of this environment in some of our papers, including:

* https://arxiv.org/abs/1706.04208
* https://arxiv.org/abs/1612.05159
* https://arxiv.org/abs/1704.00756

# About
Fruit-Collection is a vastly configurable game with very interesting properties, among which is its state-space that can be huge and completely intractable using classical tabular methods.
The game consists an arbitrary maze (with or without walls), in which the player should move and eat fixed fruits (blue blocks). Each fruit results in +1 reward. There may exist a number of ghosts (red squares) that cause the player to loose life and receive negative score.
Based on the maze shape, number of ghosts, (fixed or random) position of fruits, number of lives, and initial position of the player, the game can easily be designed in a way that serves best as a testbed for a given algorithm of interest.

# Dependencies

* Python 3.5 or higher
* pygame (pip install pygame)
* click (pip install click)
* numpy (pip install numpy -- or install Anaconda distribution)

# Getting Started 
Three particular environment are provided out-of-the-box, some of them has been used in previously published research. Nevertheless, making a new version is still very easy and only requires to identify two methods in the superclass (see the example below).

* `mini`: 10 x 10 maze + no ghost + no wall + random selection of 5 fruits from a pre-defined set of 10 possible fruits at each episode + random initial position of the player.

* `small`: 11 x 11 maze with some dead ends + two ghosts doing random walk + random fruits which are initialized at each episode with 50% chance of existence at each location + fixed initial position of the player at lower right corner.

* `large`: 21 x 14 maze with two horizontal passing corridors + 4 ghosts doing random walk + random fruits similar to `small` + initial position at [10, 8].

The main arguments and other internal properties (such as reward scheme) are straightforward.

## Arguments:

* game_length: `int` maximum number of steps at each episode.

* lives: `int` number of times that the player can hit a ghost, after which the episode will be finished.

* state_mode: `str` specifies the format of returned state (see below).

* is_fruit: `bool` whether or not the fruits appear on the maze (does not do anything if no fruit is present in the game).

* is_ghost:`bool` whether or not the ghosts appear on the maze (does not do anything if no ghost is present in the game, e.g. in `mini`)

* rendering: `bool` Does the rendering if `True`.

* image_saving: `bool` saves the rendered frame as bitmap at each `render` call.

* render_dir: `str` the directory in which rendered frames will be saved: if `image_saving=True` a new folder called `render0` will be made inside `render_dir` (with new int suffix automatically assigned for each episode). If none provided, `render` is used by default.

## Return state
Three options are available (see argument `state_mode`):

* `pixel`: includes 4 binary channels (matrices) corresponding to walls, fruits, player position, and ghosts position.

* `1hot`: include augmentation of 1hot vectors for player position and ghost positions.

* `multi-head`: Returns concatenation of three binary vectors for player position (1-hot), not-eaten fruits (binary), and ghost positions (binary).


## Main usage:
```
import numpy as np
import time
from fruit_collection import FruitCollectionSmall  # or FruitCollectionLarge or FruitCollectionMini

env = FruitCollectionSmall(rendering=True, lives=1, is_fruit=True, is_ghost=True, image_saving=False)
env.reset()
env.render()

for _ in range(50):
    action = np.random.choice(env.legal_actions)
    obs, r, term, info = env.step(action)
    env.render()
    time.sleep(.2)
```

## Human Play!
You can also play Fruit-Collection yourself:
```
python fruit_collection.py -m small
```
or to have saving and to deactivate ghosts:
```
python fruit_collection.py -m small --save --no-ghost
```

Press `Q` to finish the game.

# Making Your Own Fruit-Collection
It is still quite easy to specialize Fruit-Collection for your own experiment. You mainly need to define the two methods `init_with_mode` and `_reset_targets` in your superclass.
`init_with_mode` defines basic properties such as maze shape, location of walls (if any), etc. On the other hand, `_reset_targets` is called at the `reset` method of base class and defines properties at the begining of each episode (such as how fruits are spawned and player's init position).

See the following example (also see the three superclasses for Mini, Small, and Large):
```
class MyFruitCollection(FruitCollection):
    def init_with_mode(self):
        self.is_ghost = False
        self.is_fruit = True
        self.nb_fruits = 4
        self.scr_w = 5
        self.scr_h = 5
        self.possible_fruits = [[0, 0], [0, 4], [4, 0], [4, 4]]
        self.rendering_scale = 50
        self.walls = [[1, 0], [2, 0], [4, 1], [0, 2], [2, 2], [3, 3], [1, 4]]
        if self.is_ghost:
            self.ghosts = [{'colour': RED, 'reward': self.reward_scheme['ghost'], 'location': [0, 1],
                            'active': True}]
        else:
            self.ghosts = []

    def _reset_targets(self):
        while True:
            self.player_pos_x, self.player_pos_y = np.random.randint(0, self.scr_w), np.random.randint(0, self.scr_h)
            if [self.player_pos_x, self.player_pos_y] not in self.possible_fruits + self.walls:
                break
        self.fruits = []
        self.active_fruits = []
        if self.is_fruit:
            for x in range(self.scr_w):
                for y in range(self.scr_h):
                    self.fruits.append({'colour': BLUE, 'reward': self.reward_scheme['fruit'],
                                        'location': [x, y], 'active': False})
                    self.active_fruits.append(False)
            fruits_idx = deepcopy(self.possible_fruits)
            np.random.shuffle(fruits_idx)
            fruits_idx = fruits_idx[:self.nb_fruits]
            self.mini_target = [False] * len(self.possible_fruits)
            for f in fruits_idx:
                idx = f[1] * self.scr_w + f[0]
                self.fruits[idx]['active'] = True
                self.active_fruits[idx] = True
                self.mini_target[self.possible_fruits.index(f)] = True
```

