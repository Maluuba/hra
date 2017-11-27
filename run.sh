echo '====='
echo 'Tabular GVF ...'
THEANO_FLAG="device=cpu" ipython ./tabular/train.py -- -o use_gvf True -o folder_name tabular_gvf_ -o nb_experiments 5

echo '====='
echo 'Tabular NO_GVF ...'
THEANO_FLAG="device=cpu" ipython ./tabular/train.py -- -o use_gvf False -o folder_name tabular_no-gvf_ -o nb_experiments 5

echo '====='
echo 'DQN all baselines ...'
THEANO_FLAG="device=cpu" ipython ./dqn/train.py -- -o nb_experiments 5
