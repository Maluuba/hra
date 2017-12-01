import time
import numpy as np
from utils import Font, plot_and_write, create_folder


class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, history_len=1, max_start_nullops=1, replay_min_size=0,
                 score_window_size=100, rng=None, folder_location='/experiments/', folder_name='expt', testing=False):
        self.rng = rng
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_agent = 0
        self.eval_steps = []
        self.eval_scores = []
        self.env = env
        self.ai = ai
        self.history_len = history_len
        self.max_start_nullops = max_start_nullops
        if not testing:
            self.folder_name = create_folder(folder_location, folder_name)
        self.episode_max_len = episode_max_len
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = max(self.ai.minibatch_size, replay_min_size)
        self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)

    def do_training(self, total_eps=5000, eps_per_epoch=100, eps_per_test=100, is_learning=True, is_testing=True):
        while self.episode_num < total_eps:
            print(Font.yellow + Font.bold + 'Training ... ' + str(self.episode_num) + '/' + str(total_eps) + Font.end,
                  end='\n')
            self.do_episodes(number=eps_per_epoch, is_learning=is_learning)

            if is_testing:
                eval_scores, eval_steps = self.do_episodes(number=eps_per_test, is_learning=False)
                self.eval_steps.append(eval_steps)
                self.eval_scores.append(eval_scores)
                plot_and_write(plot_dict={'steps': self.eval_steps}, loc=self.folder_name + "/steps",
                               x_label="Episodes", y_label="Steps", title="", kind='line', legend=True)
                plot_and_write(plot_dict={'scores': self.eval_scores}, loc=self.folder_name + "/scores",
                               x_label="Episodes", y_label="Scores", title="", kind='line', legend=True)
                self.ai.dump_network(weights_file_path=self.folder_name + '/q_network_weights.h5',
                                     overwrite=True)

    def do_episodes(self, number=1, is_learning=True):
        scores = []
        steps = []
        for num in range(number):
            self._do_episode(is_learning=is_learning)
            scores.append(self.score_agent)
            steps.append(self.last_episode_steps)
            if not is_learning:
                self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
                self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            else:
                self.episode_num += 1
        return np.mean(scores), np.mean(steps)

    def _do_episode(self, is_learning=True):
        rewards = []
        self.env.reset()
        self._reset()
        term = False
        self.fps = 0
        start_time = time.time()
        while not term:
            reward, term = self._step(evaluate=not is_learning)
            rewards.append(reward)
            if self.ai.transitions.size >= self.replay_min_size and is_learning and \
               self.last_episode_steps % self.ai.learning_frequency == 0:
                self.ai.learn()
            self.score_agent += reward
            if not term and self.last_episode_steps >= self.episode_max_len:
                print('Reaching maximum number of steps in the current episode.')
                term = True
        self.fps = int(self.last_episode_steps / max(0.1, (time.time() - start_time)))
        return rewards

    def _step(self, evaluate=False):
        self.last_episode_steps += 1
        action = self.ai.get_action(self.last_state, evaluate)
        new_obs, reward, game_over, info = self.env.step(action)
        reward_channels = info['head_reward']
        if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
            new_obs = new_obs.reshape(self.env.state_shape)
        if not evaluate:
            self.ai.transitions.add(s=self.last_state[-1].astype('float32'), a=action, r=reward_channels, t=game_over)
            self.total_training_steps += 1
        if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
            new_obs = new_obs.reshape(self.env.state_shape)
        self._update_state(new_obs)
        return reward, game_over

    def _reset(self):
        self.last_episode_steps = 0
        self.score_agent = 0

        assert self.max_start_nullops >= self.history_len or self.max_start_nullops == 0
        if self.max_start_nullops != 0:
            num_nullops = self.rng.randint(self.history_len, self.max_start_nullops)
            for i in range(num_nullops - self.history_len):
                self.env.step(0)

        for i in range(self.history_len):
            if i > 0:
                self.env.step(0)
            obs = self.env.get_state()
            if obs.ndim == 1 and len(self.env.state_shape) == 2:
                obs = obs.reshape(self.env.state_shape)
            self.last_state[i] = obs

    def _update_state(self, new_obs):
        temp_buffer = np.empty(self.last_state.shape, dtype=np.uint8)
        temp_buffer[:-1] = self.last_state[-self.history_len + 1:]
        temp_buffer[-1] = new_obs
        self.last_state = temp_buffer

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window
