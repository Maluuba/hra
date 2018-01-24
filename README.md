 # Hybrid Reward Architecture
This repository hosts the code published along with the following NIPS article (Experiment 4.1: Fruit Collection Task):

* https://arxiv.org/abs/1706.04208

For more information about this article, see the following blog posts:

* https://www.microsoft.com/en-us/research/blog/hybrid-reward-architecture-achieving-super-human-ms-pac-man-performance/
* https://blogs.microsoft.com/ai/2017/06/14/divide-conquer-microsoft-researchers-used-ai-master-ms-pac-man/

 # Dependencies
 
 We strongly suggest to use [Anaconda distribution](https://www.anaconda.com/download/). 

* Python 3.5 or higher
* pygame 1.9.2+ (pip install pygame)
* click (pip install click)
* numpy (pip install numpy -- or install Anaconda distribution)
* [Keras](https://keras.io) 1.2.0+, but less than 2.0 (pip install keras==1.2)
* Theano or Tensorflow. The code is fully tested on Theano. (pip install theano)

# Usage

While any run is going on, the results as well as the **AI** models will be saved in the `./results` subfolder. For a complete run, five experiments for each method, use the following command (may take several hours depending on your machine):

```
./run.sh
```

* NOTE: Because the state-shape is relatively small, the deep RL methods of this code run faster on CPU.

Alternatively, for a single run use the following commands:

* Tabular GVF: 
```
ipython ./tabular/train.py -- -o use_gvf True -o folder_name tabular_gvf_ -o nb_experiments 1
```

* Tabular no-GVF: 
```
ipython ./tabular/train.py -- -o use_gvf False -o folder_name tabular_no-gvf_ -o nb_experiments 1
```

* DQN: 
```
THEANO_FLAG="device=cpu" ipython ./dqn/train.py -- --mode hra+1 -o nb_experiments 1
```
* `--mode` can be either of `dqn`, `dqn+1`, `hra`, `hra+1`, or `all`.

# Demo

We have also provided the code to demo Tabular GVF/NO-GVF methods. You first need to train the model using one of the above commands (Tabular GVF or no-GVF) and then run the demo. For example,
```
ipython ./tabular/train.py -- -o use_gvf True -o folder_name tabular_gvf_ -o nb_experiments 1
ipython ./tabular/train.py -- --demo -o folder_name tabular_gvf_
```

If you would like to save the results, use the `--save` option:
```
ipython ./tabular/train.py -- --demo --save -o folder_name tabular_gvf_
```
The rendered images will be saved in `./render` directory by default. 

# License

Please refer to LICENSE.txt.
