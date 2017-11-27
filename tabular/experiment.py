import pickle
import time
from copy import deepcopy
import numpy as np
from utils import Font, plot_and_write, create_folder

floatX = np.float32


class SoCExperiment(object):
    def __init__(self, ai_list, env, aggregator_epsilon, aggregator_final_epsilon, aggregator_decay_start,
                 aggregator_decay_steps, epoch_size, nb_eval_episodes, final_alpha, alpha_decay_steps,
                 alpha_decay_start, use_gvf, rng, make_folder=True, folder_location='/results/', folder_name='expt'):
        self.ai_list = ai_list
        self.env_done_for_agent = None
        self.learning_flag = None
        self.env = env
        self.use_gvf = use_gvf
        self.rng = rng
        self.aggregator_start_epsilon = aggregator_epsilon
        self.aggregator_epsilon = aggregator_epsilon
        self.aggregator_final_epsilon = aggregator_final_epsilon
        self.aggregator_decay_start = aggregator_decay_start
        self.aggregator_decay_steps = aggregator_decay_steps
        self.last_state = None
        self.action = None
        self.score_agent = 0
        self.step_in_episode = 0
        self.total_learning_steps = 0  # is not reset
        self.count_episode = 0
        self.epoch_size = epoch_size
        self.nb_eval_episodes = nb_eval_episodes
        self.eval_flag = False
        self.episode_done = False
        if make_folder:
            self.folder_name = create_folder(folder_location, folder_name)
        self.final_alpha = final_alpha
        self.alpha_decay_steps = alpha_decay_steps
        self.alpha_decay_start = alpha_decay_start
        self.reset()

    def reset(self):
        self.env.reset()
        if self.env.rendering:
            self.env.render()
        self.episode_done = False
        if self.use_gvf:
            self.env_done_for_agent = [not t for t in self.env.mini_target]
            self.last_state = [self.env.player_pos_y, self.env.player_pos_x]
        else:
            self.env_done_for_agent = [not t for t in self.env.active_targets]
            self.last_state = self.env.get_soc_state()
        self.learning_flag = deepcopy(self.env.active_targets)
        self.action = None
        self.step_in_episode = 0
        self.score_agent = 0

    def do_epochs(self, number):
        self.count_episode = 0
        eval_returns = []
        eval_steps = []
        self.eval_flag = False

        def do_eval():
            eval_return = 0
            eval_episode_steps = 0
            for eval_episode in range(self.nb_eval_episodes):
                if eval_episode in []:
                    self.env.rendering = True
                else:
                    self.env.rendering = False
                print(Font.bold + Font.blue + '>>> Eval Episode {}'.format(eval_episode) + Font.end)
                eval_return += self._do_episode(is_learning=False, rendering_sleep=None)
                eval_episode_steps += self.step_in_episode
            eval_returns.append(eval_return / self.nb_eval_episodes)
            eval_steps.append(eval_episode_steps / self.nb_eval_episodes)
            plot_and_write(plot_dict={'scores': eval_returns}, loc=self.folder_name + "/scores",
                           x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                           moving_average=True)
            plot_and_write(plot_dict={'steps': eval_steps}, loc=self.folder_name + "/steps",
                           x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)
            with open(self.folder_name + "/soc_ai_list.pkl", 'wb') as f:
                pickle.dump(self.ai_list, f)
            self.eval_flag = False

        for count_epoch in range(number):
            # Evaluation:
            do_eval()
            for count_episode in range(self.epoch_size):
                print(Font.bold + 'Epoch: {} '.format(count_epoch) +
                      Font.yellow + '>>> Episode {}'.format(self.count_episode) + Font.end)
                self._do_episode(is_learning=True, rendering_sleep=None)
                self.count_episode += 1
        do_eval()
        return eval_returns

    def demo(self, nb_episodes, rendering_sleep):
        assert self.env.rendering is True
        for k in range(nb_episodes):
            print('\nDemo Episode ', k)
            self._do_episode(is_learning=False, rendering_sleep=rendering_sleep)

    def _do_episode(self, is_learning, rendering_sleep):
        self.reset()
        episode_return = 0
        while not self.episode_done:
            r = self._step(is_learning=is_learning)
            if self.env.rendering:
                self.env.render()
                time.sleep(rendering_sleep)
            episode_return += r  # undiscounted return (for eval purposes)
        print('Aggregator eps: ', round(self.aggregator_epsilon, 2), ' | alpha: ', round(self.ai_list[0].alpha, 4),
              Font.cyan + ' | Episode Score: ' + Font.end, round(episode_return, 2))
        return episode_return

    def _get_action(self, s, explore):
        if explore and self.rng.binomial(1, self.aggregator_epsilon):  # aggregator exploration
            action = self.rng.randint(self.env.nb_actions)
        else:
            # sum all q's then select max action
            q = []
            if self.use_gvf:
                s = tuple(s)
                for gvf_idx in range(len(self.env.possible_fruits)):
                    if not self.env_done_for_agent[gvf_idx]:
                        q.append(self.ai_list[gvf_idx].get_q(s))
            else:
                for agent_idx, agent_state in enumerate(s):
                    if self.learning_flag[agent_idx] is True:
                        q.append(self.ai_list[agent_idx].get_q(agent_state))
            if self.env.rendering is True:
                print(Font.bold + Font.cyan + 'Values:' + Font.end)
                print(self.env.action_meanings)
                for kk, qq in enumerate(q):
                    print('-'*35)
                    print('alpha: ', self.ai_list[kk].alpha, 'gamma: ', self.ai_list[kk].gamma)
                    string = ' '.join('{:0.2f}'.format(i) for i in qq)
                    print(string)
                print('sum: ', np.sum(q, axis=0))
            q_aggregate = np.sum(q, axis=0)
            actions = np.where(q_aggregate == q_aggregate.max())[0]  # is biased if using np.argmax(q_aggregate)
            action = self.rng.choice(actions)
        return action

    def _step(self, is_learning=True):
        action = self._get_action(self.last_state, is_learning)
        _, r_env, self.episode_done, info = self.env.step(action)
        if self.use_gvf:
            s2 = [self.env.player_pos_y, self.env.player_pos_x]
            if is_learning:
                # Hint: ALL gvf agents learn in parallel at each transition (regardless of player's position)
                for gvf_idx, gvf_goal in enumerate(self.env.possible_fruits):
                    if gvf_goal != s2:
                        self.ai_list[gvf_idx].learn(self.last_state, action, 0., s2, False)
                    else:
                        self.ai_list[gvf_idx].learn(self.last_state, action, 1., s2, True)
                        self.env_done_for_agent[gvf_idx] = True
            else:
                for gvf_idx, gvf_goal in enumerate(self.env.possible_fruits):
                    if gvf_goal == s2:
                        self.env_done_for_agent[gvf_idx] = True
        else:
            s2 = self.env.get_soc_state()
            for k, s2_agent in enumerate(s2):
                r_base = self.env.targets[k]['reward']
                if info['fruit'] is not None:
                    if info['fruit'] == k:
                        r = r_base
                        self.env_done_for_agent[k] = True
                    else:
                        r = 0.
                else:
                    r = 0.
                if is_learning and self.learning_flag[k] is True:
                    self.ai_list[k].learn(self.last_state[k], action, r, s2_agent, self.env_done_for_agent[k])
                if self.env_done_for_agent[k] is True:  # if env terminates for agent_k it stops learning from next step
                    self.learning_flag[k] = False
        self.last_state = deepcopy(s2)
        self.score_agent += r_env
        self.step_in_episode += 1
        if is_learning:
            self.total_learning_steps += 1
            self._anneal_eps()
            self._anneal_alpha()
        if self.total_learning_steps % self.epoch_size == 0:
            self.eval_flag = True
        return r_env

    def _anneal_eps(self):
        # linear annealing
        if self.total_learning_steps < self.aggregator_decay_start:
            return
        if self.aggregator_epsilon > self.aggregator_final_epsilon:
            decay = (self.aggregator_start_epsilon - self.aggregator_final_epsilon) \
                    * (self.total_learning_steps - self.aggregator_decay_start) / self.aggregator_decay_steps
            temp = self.aggregator_start_epsilon - decay
            if temp > self.aggregator_final_epsilon:
                self.aggregator_epsilon = temp
            else:
                self.aggregator_epsilon = self.aggregator_final_epsilon

    def _anneal_alpha(self):
        # linear annealing
        if self.total_learning_steps < self.alpha_decay_start:
            return
        for ai in self.ai_list:
            if ai.alpha > self.final_alpha:
                decay = (ai.start_alpha - self.final_alpha) * (self.total_learning_steps - self.alpha_decay_start) \
                        / self.alpha_decay_steps
                temp = ai.start_alpha - decay
                if temp > ai.alpha:
                    ai.alpha = temp
                else:
                    ai.alpha = self.final_alpha
