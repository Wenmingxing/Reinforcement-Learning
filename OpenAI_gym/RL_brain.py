#! usr/bin/env python

"""
Coded by Luke wen on 11th Oct 2017, referring to https://morvanzhou.github.io/tutorials/

It's aiming to apply the DQN to the gym envs especially the CartPole and MountainCar ones.

"""

import numpy as np 
import tensorflow as tf 
import pandas as pd 
import matplotlib.pyplot as plt 


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q network off-policy, using the neural network to predict the Q value for each action under each state.
class DeepQNetwork(object):
	def __init__(self,n_actions,n_features,learning_rate = 0.01,reward_decay = 0.9,e_greedy = 0.9,replace_target_iter = 300,memory_size = 500,batch_size = 32,e_greedy_increment = None,output_graph = False):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max

		# Total learning step for epoches
		self.learn_step_counter = 0

		# initialize zero memory [s,a,r,s_]
		self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2 ))
		print(self.memory.shape)

		# build the two neural network for the eval and target 
		self._build_net()

		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')

		self.replace_target_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]


		self.sess = tf.Session()

		if output_graph:
			# showing the information in tensorboard
			tf.summary.FileWriter('log/',self.sess.graph)

		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []

	def _build_net(self):
		'''
		Build two NNs with same strucature but the parameters are not always the same, eval one is used to evaluate the action to get the Q_eval,
		target one is used to pass the action through and get the target value for the corresponding action.
		'''

		# Build the eval network
		self.s = tf.placeholder(name = 's', shape = [None,self.n_features],dtype = tf.float32)# input
		self.q_target = tf.placeholder(name = 'Q_target', shape = [None,self.n_actions],dtype = tf.float32) # for calculating loss

		with tf.variable_scope('eval_net'):
			c_names,n_l1,w_initializer,b_initializer = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],12,tf.random_normal_initializer(0,0.3),tf.constant_initializer(0.1) # config of layers

			# First layer. Collections is used later when assign to target net
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer = w_initializer,collections = c_names)
				b1 = tf.get_variable('b1',[1,n_l1],initializer = b_initializer,collections = c_names)
				l1 = tf.nn.relu(tf.matmul(self.s,w1) + b1)
			# Second layer. collections is used later when assign to target net
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer = w_initializer, collections = c_names)
				b2 = tf.get_variable('b2',[1,self.n_actions],initializer = b_initializer,collections = c_names)
				self.q_eval = tf.matmul(l1,w2) + b2
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


		# Build target net 
		self.s_ = tf.placeholder(name='s_',shape = [None,self.n_features],dtype = tf.float32) # input for the target network
		with tf.variable_scope('target_net'):
			c_names = ['target_net_params',tf.GraphKeys.GLOBAL_VARIABLES]

			# First layer 
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer = w_initializer,collections = c_names)
				b1 = tf.get_variable('b1',[1,n_l1],initializer = b_initializer,collections = c_names)
				l1 = tf.nn.relu(tf.matmul(self.s_,w1) + b1)
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer = w_initializer,collections = c_names)
				b2 = tf.get_variable('b2',[1,self.n_actions],initializer = b_initializer,collections = c_names)
				self.q_next = tf.matmul(l1,w2) + b2

	def store_transition(self,s,a,r,s_):
		"""
		store the transition for each step in the RL
		"""
		if not hasattr(self,'memory_counter'):
			self.memory_counter = 0

		transition = np.hstack((s,[a,r],s_))

		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition

		self.memory_counter += 1

	def choose_action(self,observation):
		# To have batch dimension when feed into tf placeholder
		observation = observation[np.newaxis,:]

		if np.random.uniform() < self.epsilon:
			# Forward feed the observation and get q value for every actions
			actions_value = self.sess.run(self.q_eval,feed_dict = {self.s:observation})
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0,self.n_actions)
		return action 

	def  learn(self):
		'''
		This is the most important part for the RL(DQN) algorithm, we replace the networks parameters and also run the networks for updating the parameters simutaneously
		'''
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			print('\n target_params_replaced!\n')
		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size,size = self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter,size = self.batch_size)
		batch_memory  = self.memory[sample_index,:]

		q_next,q_eval = self.sess.run([self.q_next,self.q_eval],feed_dict = {self.s_:batch_memory[:,-self.n_features:],self.s:batch_memory[:,:self.n_features]})

		# change q-target w.r.t q_eval's action
		q_target = q_eval.copy()

		batch_index = np.arange(self.batch_size,dtype=np.int32)
		eval_act_index = batch_memory[:,self.n_features].astype(int)
		reward = batch_memory[:,self.n_features + 1]

		q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next,axis=1)

		_,self.cost = self.sess.run([self._train_op,self.loss],feed_dict = {self.s:batch_memory[:,:self.n_features],self.q_target:q_target})

		self.cost_his.append(self.cost)

		# increasing epsilon

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)),self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()