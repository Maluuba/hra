import pickle
import os
import yaml

import click
import numpy as np

from tabular.ai import AI
from tabular.experiment import SoCExperiment
from environment.fruit_collection import FruitCollectionMini

np.set_printoptions(suppress=True, precision=2)


def soc_agent(params):
    rng = np.random.RandomState(params['random_seed'])
    env = FruitCollectionMini(rendering=False, game_length=300)
    for mc_count in range(params['nb_experiments']):
        ai_list = []
        if not params['use_gvf']:
            for _ in range(env.nb_targets):
                fruit_ai = AI(nb_actions=env.nb_actions, init_q=params['init_q'], gamma=params['gamma'],
                              alpha=params['alpha'], learning_method=params['learning_method'], rng=rng)
                ai_list.append(fruit_ai)
        else:
            for _ in env.possible_fruits:
                gvf_ai = AI(nb_actions=env.nb_actions, init_q=params['init_q'], gamma=params['gamma'],
                            alpha=params['alpha'], learning_method=params['learning_method'], rng=rng)
                ai_list.append(gvf_ai)
        expt = SoCExperiment(ai_list=ai_list, env=env, aggregator_epsilon=params['aggregator_epsilon'],
                             aggregator_final_epsilon=params['aggregator_final_epsilon'],
                             aggregator_decay_steps=params['aggregator_decay_steps'],
                             aggregator_decay_start=params['aggregator_decay_start'], final_alpha=params['final_alpha'],
                             alpha_decay_steps=params['alpha_decay_steps'], alpha_decay_start=params['alpha_decay_start'],
                             epoch_size=params['epoch_size'], folder_name=params['folder_name'],
                             folder_location=params['folder_location'],
                             nb_eval_episodes=params['nb_eval_episodes'], use_gvf=params['use_gvf'], rng=rng)
        with open(expt.folder_name + '/config.yaml', 'w') as y:
            yaml.safe_dump(params, y)  # saving params for future reference
        expt.do_epochs(number=params['nb_epochs'])


def demo_soc(params, nb_episodes, rendering_sleep, saving):
    rng = np.random.RandomState(1234)
    env = FruitCollectionMini(rendering=True, lives=1, game_length=300, image_saving=saving, rng=rng)
    i = 0
    while os.path.exists(os.getcwd() + params['folder_location'] + params['folder_name'] + str(i)):
        i += 1
    file_name = os.getcwd() + params['folder_location'] + params['folder_name'] + str(i - 1) + '/soc_ai_list.pkl'
    print(file_name)
    with open(file_name, 'rb') as f:
        ai_list = pickle.load(f)
    expt = SoCExperiment(ai_list=ai_list, env=env, aggregator_epsilon=0., aggregator_final_epsilon=None,
                         aggregator_decay_steps=None, aggregator_decay_start=None, use_gvf=params['use_gvf'],
                         epoch_size=params['epoch_size'], make_folder=False, folder_name=params['folder_name'],
                         folder_location=params['folder_location'], nb_eval_episodes=params['nb_eval_episodes'],
                         final_alpha=None, alpha_decay_steps=None, alpha_decay_start=None, rng=rng)
    expt.demo(nb_episodes=nb_episodes, rendering_sleep=rendering_sleep)


@click.command()
@click.option('--demo/--no-demo', default=False, help='Do a demo.')
@click.option('--save/--no-save', default=False, help='Save images.')
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def main(options, demo, save):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = os.path.join(dir_path, 'config.yaml')
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    if demo:
        demo_soc(params, nb_episodes=3, rendering_sleep=0.1, saving=save)
    else:
        soc_agent(params)


if __name__ == '__main__':
    main()
