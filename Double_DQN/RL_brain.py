#! usr/bin/env python

'''
Coded by Luke Wen on 11th Oct 2017 referece :https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.1_Double_DQN/RL_brain.py

It's aiming to build the double DQN for the cartpolr Env in the gym.

The Double DQN is to overcome the overestimation problem exsited in the Natural DQN due to the Qmax which is included into the system by Q_learning.
'''

import numpy as np 
import tensorflow as tf 


np.random.seed(1)
tf.set_random_seed(1)


class DoubleDQN:
	def __init__(self,n_actions,n_features,learning_rate = 0.005,reward_decay = 0.9,e_greedy = 0.9,replace_target_iter = 200,memory)