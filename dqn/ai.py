from copy import deepcopy

import numpy as np
from keras import backend as K
from keras.optimizers import RMSprop

from utils import ExperienceReplay, slice_tensor_tensor, flatten
from dqn.model import build_dense

floatX = 'float32'


class AI(object):
    def __init__(self, state_shape, nb_actions, action_dim, reward_dim, history_len=1, gamma=.99,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.0,
                 minibatch_size=32, replay_max_size=100, update_freq=50, learning_frequency=1,
                 num_units=250, remove_features=False, use_mean=False, use_hra=True, rng=None):
        self.rng = rng
        self.history_len = history_len
        self.state_shape = [1] + state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learning_rate_start = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.minibatch_size = minibatch_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.nb_units = num_units
        self.use_mean = use_mean
        self.use_hra = use_hra
        self.remove_features = remove_features
        self.learning_frequency = learning_frequency
        self.replay_max_size = replay_max_size
        self.transitions = ExperienceReplay(max_size=self.replay_max_size, history_len=history_len, rng=self.rng,
                                            state_shape=state_shape, action_dim=action_dim, reward_dim=reward_dim)
        self.networks = [self._build_network() for _ in range(self.reward_dim)]
        self.target_networks = [self._build_network() for _ in range(self.reward_dim)]
        self.all_params = flatten([network.trainable_weights for network in self.networks])
        self.all_target_params = flatten([target_network.trainable_weights for target_network in self.target_networks])
        self.weight_transfer(from_model=self.networks, to_model=self.target_networks)
        self._compile_learning()
        print('Compiled Model and Learning.')

    def _build_network(self):
        return build_dense(self.state_shape, int(self.nb_units / self.reward_dim),
                           self.nb_actions, self.reward_dim, self.remove_features)

    def _remove_features(self, s, i):
        return K.concatenate([s[:, :, :, : -self.reward_dim],
                              K.expand_dims(s[:, :, :, self.state_shape[-1] - self.reward_dim + i], dim=-1)])

    def _compute_cost(self, q, a, r, t, q2):
        preds = slice_tensor_tensor(q, a)
        bootstrap = K.max if not self.use_mean else K.mean
        targets = r + (1 - t) * self.gamma * bootstrap(q2, axis=1)
        cost = K.sum((targets - preds) ** 2)
        return cost

    def _compile_learning(self):
        s = K.placeholder(shape=tuple([None] + [self.history_len] + self.state_shape))
        a = K.placeholder(ndim=1, dtype='int32')
        r = K.placeholder(ndim=2, dtype='float32')
        s2 = K.placeholder(shape=tuple([None] + [self.history_len] + self.state_shape))
        t = K.placeholder(ndim=1, dtype='float32')

        updates = []
        costs = 0
        qs = []
        q2s = []
        for i in range(len(self.networks)):
            local_s = s
            local_s2 = s2
            if self.remove_features:
                local_s = self._remove_features(local_s, i)
                local_s2 = self._remove_features(local_s2, i)
            qs.append(self.networks[i](local_s))
            q2s.append(self.target_networks[i](local_s2))
            if self.use_hra:
                cost = self._compute_cost(qs[-1], a, r[:, i], t, q2s[-1])
                optimizer = RMSprop(lr=self.learning_rate, rho=.95, epsilon=1e-7)
                updates += optimizer.get_updates(params=self.networks[i].trainable_weights, loss=cost, constraints={})
                costs += cost
        if not self.use_hra:
            q = sum(qs)
            q2 = sum(q2s)
            summed_reward = K.sum(r, axis=-1)
            cost = self._compute_cost(q, a, summed_reward, t, q2)
            optimizer = RMSprop(lr=self.learning_rate, rho=.95, epsilon=1e-7)
            updates += optimizer.get_updates(params=self.all_params, loss=cost, constraints={})
            costs += cost

        target_updates = []
        for network, target_network in zip(self.networks, self.target_networks):
            for target_weight, network_weight in zip(target_network.trainable_weights, network.trainable_weights):
                target_updates.append(K.update(target_weight, network_weight))

        self._train_on_batch = K.function(inputs=[s, a, r, s2, t], outputs=[costs], updates=updates)
        self.predict_network = K.function(inputs=[s], outputs=qs)
        self.update_weights = K.function(inputs=[], outputs=[], updates=target_updates)

    def update_lr(self, cur_step, total_steps):
        self.learning_rate = ((total_steps - cur_step - 1) / total_steps) * self.learning_rate_start

    def get_max_action(self, states):
        states = self._reshape(states)
        q = np.array(self.predict_network([states]))
        q = np.sum(q, axis=0)
        return np.argmax(q, axis=1)

    def get_action(self, states, evaluate):
        eps = self.epsilon if not evaluate else self.test_epsilon
        if self.rng.binomial(1, eps):
            return self.rng.randint(self.nb_actions)
        else:
            return self.get_max_action(states=states)

    def train_on_batch(self, s, a, r, s2, t):
        s = self._reshape(s)
        s2 = self._reshape(s2)
        if len(r.shape) == 1:
            r = np.expand_dims(r, axis=-1)
        return self._train_on_batch([s, a, r, s2, t])

    def learn(self):
        assert self.minibatch_size <= self.transitions.size, 'not enough data in the pool'
        s, a, r, s2, term = self.transitions.sample(self.minibatch_size)
        objective = self.train_on_batch(s, a, r, s2, term)
        if self.update_counter == self.update_freq:
            self.update_weights([])
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def dump_network(self, weights_file_path='q_network_weights.h5', overwrite=True):
        for i, network in enumerate(self.networks):
            network.save_weights(weights_file_path[:-3] + str(i) + weights_file_path[-3:], overwrite=overwrite)

    def load_weights(self, weights_file_path='q_network_weights.h5'):
        for i, network in enumerate(self.networks):
            network.load_weights(weights_file_path[:-3] + str(i) + weights_file_path[-3:])
        self.update_weights([])

    @staticmethod
    def _reshape(states):
        if len(states.shape) == 2:
            states = np.expand_dims(states, axis=0)
        if len(states.shape) == 3:
            states = np.expand_dims(states, axis=1)
        return states

    @staticmethod
    def weight_transfer(from_model, to_model):
        for f_model, t_model in zip(from_model, to_model):
            t_model.set_weights(deepcopy(f_model.get_weights()))
