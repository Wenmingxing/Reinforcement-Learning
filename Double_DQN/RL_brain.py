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
	def __init__(self,n_actions,n_features,learning_rate = 0.005,reward_decay = 0.9,e_greedy = 0.9,replace_target_iter = 200,memory_size = 3000,batch_size = 32, e_greedy_increment=None,output_graph=False,double_q = True,sess = None):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.batch_size = batch_size
		self.memory_size = memory_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		self.double_q = double_q # sign for whether using the double q network
		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size,n_features * 2 + 2))
		self._build_net()
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(e,t) for e,t in zip(e_params,t_params)]

		if sess is None: 
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
		else:
			self.sess = sess
		if output_graph:
			tf.summary.FileWriter('logs/',self.sess.graph)
		self.cost_his = []

	def _build_net(self):
		'''
		Build the evaluation and target network for training.
		'''
		def build_layers(s,c_names,n_l1,w_initializer,b_initializer):
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1',[self.n_features,n_l1],initializer = w_initializer,collections = c_names)
				b1 = tf.get_variable('b1',[1,n_l1],initializer = b_initializer,collections = c_names)
				l1 = tf.nn.relu(tf.matmul(s,w1) + b1)

			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer = w_initializer,collections = c_names)
				b2 = tf.get_variable('b2',[1,self.n_actions],initializer = b_initializer,collections = c_names)
				out = tf.matmul(l1,w2) + b2
			return out 
		# build evaluate_net
		self.s = tf.placeholder(name = 's',shape = [None,self.n_features],dtype = tf.float32) # input
		self.q_target = tf.placeholder(name = 'Q_target',shape =[None,self.n_actions],dtype =tf.float32) # for calculating loss

		with tf.variable_scope('eval_net'):
			c_names, n_l1, w_initializer, b_initializer = ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],20,tf.random_normal_initializer(0.,0.3),tf.constant_initializer(0.1) # config of layers

			self.q_eval = build_layers(self.s,c_names,n_l1,w_initializer,b_initializer)
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
		# Build target-net
		self.s_ = tf.placeholder(name = 's_',shape = [None,self.n_features],dtype =tf.float32)
		with tf.variable_scope('target_net'):
			c_names = ['target_net',tf.GraphKeys.GLOBAL_VARIABLES]
			self.q_next = build_layers(self.s_,c_names,n_l1,w_initializer,b_initializer)

	def store_transition(self,s,a,r,s_):
		'''
		Store the transition into the memory 
		'''
		if not hasattr(self,'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s,[a,r],s_))
		index = self.memory_counter % self.memory_size
		self.memory[index,:] = transition
		self.memory_counter += 1

	def choose_action(self,observation):
		'''
		Choose an action based on the current observation.
		'''
		observation = observation[np.newaxis,:]
		actions_value = self.sess.run(self.q_eval,feed_dict={self.s:observation})
		action = np.argmax(actions_value)

		if not hasattr(self,'q'): 
			self.q = []
			# record action value it gets
			self.running_q = 0
		self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value)
		self.q.append(self.running_q)

		if np.random.uniform() > self.epsilon:
			action = np.random.randint(0,self.n_actions)
		return action

	def learn(self):
		'''
		update the network and learn 
		'''
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			print('\ntraget_params_replaced\n')

		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size,size = self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter,size = self.batch_size)
		batch_memory = self.memory[sample_index,:]

		q_next,q_eval4next = self.sess.run([self.q_next,self.q_eval],feed_dict = {self.s_:batch_memory[:,-self.n_features:],self.s:batch_memory[:,-self.n_features:]})

		q_eval = self.sess.run(self.q_eval,feed_dict={self.s:batch_memory[:,:self.n_features]})

		q_target = q_eval.copy()
		batch_index = np.arange(self.batch_size,dtype = np.int32)
		eval_act_index = batch_memory[:,self.n_features].astype(int)
		reward = batch_memory[:,self.n_features + 1]


		if self.double_q:
			max_act4next = np.argmax(q_eval4next,axis =  1) # the action that brings the highest value is evaluated by q-eval
			selected_q_next = q_next[batch_index,max_act4next] # double DQN, select q_next depending on above actions
		else:
			selected_q_next = np.max(q_next,axis = 1) # the natural DQN

		q_target[batch_index,eval_act_index] = reward + self.gamma * selected_q_next

		_, self.cost = self.sess.run([self._train_op,self.loss],feed_dict={self.s:batch_memory[:,:self.n_features],self.q_target:q_target})

		self.cost_his.append(self.cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

		self.learn_step_counter += 1

