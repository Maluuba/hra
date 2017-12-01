import os
import click
import yaml
import numpy as np

from utils import Font, set_params
from dqn.experiment import DQNExperiment
from environment.fruit_collection import FruitCollectionMini
from dqn.ai import AI

np.set_printoptions(suppress=True, linewidth=200, precision=2)
floatX = 'float32'


def worker(params):
    np.random.seed(seed=params['random_seed'])
    random_state = np.random.RandomState(params['random_seed'])
    env = FruitCollectionMini(rendering=False, game_length=300, state_mode='mini')
    params['reward_dim'] = len(env.possible_fruits)
    for ex in range(params['nb_experiments']):
        print('\n')
        print(Font.bold + Font.red + '>>>>> Experiment ', ex, ' >>>>>' + Font.end)
        print('\n')

        ai = AI(env.state_shape, env.nb_actions, params['action_dim'], params['reward_dim'],
                history_len=params['history_len'], gamma=params['gamma'], learning_rate=params['learning_rate'],
                epsilon=params['epsilon'], test_epsilon=params['test_epsilon'], minibatch_size=params['minibatch_size'],
                replay_max_size=params['replay_max_size'], update_freq=params['update_freq'],
                learning_frequency=params['learning_frequency'], num_units=params['num_units'], rng=random_state,
                remove_features=params['remove_features'], use_mean=params['use_mean'], use_hra=params['use_hra'])

        expt = DQNExperiment(env=env, ai=ai, episode_max_len=params['episode_max_len'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], folder_location=params['folder_location'],
                             folder_name=params['folder_name'], testing=params['test'], score_window_size=100,
                             rng=random_state)
        env.reset()
        if not params['test']:
            with open(expt.folder_name + '/config.yaml', 'w') as y:
                yaml.safe_dump(params, y)  # saving params for future reference
            expt.do_training(total_eps=params['total_eps'], eps_per_epoch=params['eps_per_epoch'],
                             eps_per_test=params['eps_per_test'], is_learning=True, is_testing=True)
        else:
            raise NotImplementedError


@click.command()
@click.option('--mode', default='all', help='Which method to run: dqn, dqn+1, hra, hra+1, all')
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(mode, options):
    valid_modes = ['dqn', 'dqn+1', 'hra', 'hra+1', 'all']
    assert mode in valid_modes
    if mode in ['all']:
        modes = valid_modes[:-1]
    else:
        modes = [mode]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    cfg_file = os.path.join(dir_path, 'config.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))
    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    for m in modes:
        params = set_params(params, m)
        worker(params)


if __name__ == '__main__':
    run()
