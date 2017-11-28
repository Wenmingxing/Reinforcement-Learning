#! usr/bin/env python
'''
Coded by Luke Wen MingXing on 23rd Nov 2017, referring to MorvanZhou tutorials of reinforcement learning.

This program is aiming to build a DDPG(Deep Deterministic Policy Gradient) model algorithm, which takes advantages of the DQN and Actor-Critic model.

Requirements:
python 3.5
tensorflow 1.0.1 
numpy 1.12.1
Can refer to my github link: https://github.com/Wenmingxing

'''

# import the modules we need for completing this algorithm.
import tensorflow as tf 
import numpy as np
# os is a module for manipulating the operating system 
import os 
# shutil is a module for manipulating the files 
import shutil
# car_env is another module we defined for our usage of 2D autonomous Car Environment
from car_env import CarEnv 




# set the random seed to make sure that we can get the same performance for every running of the game
np.random.seed(1)
tf.set_random_seed(1)


# the global variables for the function
MAX_EPISODES = 500 # The total number of the episodes for the training
MAX_EP_STEPS = 1000 # The total steps for each episode.
LR_A, LR_C = 1e-4,1e-4 # The learning rate for both actor and critic 
GAMMA = 0.9 # The reward discounted factor which determines how much we take the future reward into account
REPLACE_ITER_A, REPLACE_ITER_C = 800, 700 # The numer of steps to replace the target net weights with the training one.
MEMORY_CAPACITY = 2000 # Size of the memory buffer for training
BATCH_SIZE = 32 # The batch size for training the network
VAR_MIN = 0.1 
RENDER = True # Showing the visualization results
LOAD = False # whether load the traning mode from the local dir
DISCRETE_ACTION = False


# Get the information of the car env 
env = CarEnv(discrete_action=DISCRETE_ACTIOND)
STATE_DIM = env.state_dim 
ACTION_DIM = env.action_dim 
# the factor to scale the action 
ACTION_BOUND = env.action_bound 

# define the placeholder for the tf.sess
with tf.name_scope('S'):
	S = tf.placeholder(dtype=tf.float32,shape=[None,STATE_DIM],name='s')
with tf.name_scope('R'):
	R = tf.placeholder(dtype=tf.float32,shape=[None,1],name='r')
with tf.name_scope('S_'):
	S_ = tf.placeholder(dtype=tf.float32,shape=[None,STATE_DIM],name='s_')


# define the classes for both Actor and Critic 

# Class for the Actor
class Actor(object):
	# initialize the variables 
	def __init__(self,sess,action_dim,action_bound,learning_rate,t_replace_iter):
		self.sess = sess
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.lr = learning_rate
		self.t_replace_iter = t_replace_iter
		self.t_replace_counter = 0

		# build the nerwork for the Actor class
		with tf.variable_scope('Actor'):
			# Build the trianable network
			self.a = self._build_net(S,scope='eval_net',trainable = True)

			# Build the Non-trainable or the fixed target network
			self.a_ = self._build_net(S_,scope ='target_net',trainable = False)

		# Now we've had the network weights and biases, just to store them into some collections
		self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Actor/eval_net')
		self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIBALES,scope = 'Actor/target_net')
	# The private method for the class of Actor
	def _build_net(self,s,scope,trianable):
		with tf.variable_scope(scope):
			init_w = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.001)
			# The first full connected layer for the neural network,100 hidden neurons
			net = tf.layers.dense(s,100,activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name='l1',trainable = trainable)
			# the second full connected layer for the network
			net = tf.layers.dense(net,20,activation=tf.nn.relu,kernel_initializer=init_w,bias_initializer=init_b,name='l2',trainable = trainable)

			with tf.variable_scope('a'):
				# The tanh function make the output value between (-1,1)
				actions = tf.layers.dense(net,self.action_dim,activation=tf.nn.tanh,kernel_initializer=init_w,name='a',trainable = trainable)

				# scale the actions value 
				scaled_a = tf.multiply(actions,self.action_bound,name='scaled_a') # Scale the output into between(-action_bound,action_boud)
		return scaled_a                   
        
    # Define the learning method for this class                     
	def learn(self,s): 
		# batch update
		self.sess.sun(self.train_op,feed_dict={S:s}) 
		if self.t_replace_counter % self.t_replace_iter == 0:
			self.sess.run([tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)])
		self.t_replace_counter += 1

	# define the function for the actor to choose the action from the current state
	def choose_action(self,s):
		s = s[np.newaxis,:] # update the aciton for each single state
		return self.sess.run(self.a,feed_dict={S:s})[0] # Single action, the question here is why always choose the first action from the current state.

	# define the method which will add the gradients both from actor and critic to the actor network training
	def add_grad_to_graph(self,a_grad):
		with tf.variable_scope('policy_gradient'):
			self.policy_gradient = tf.gradients(ys=self.a,xs=self.e_params,grad_ys = a_grad)
		with tf.variable_scope('A_train'):
			opt = tf.train.RMSPropOptimizer(-self.lr) # gradient ascend
			self.train_op = opt.apply_gradients(zip(self.policy_gradient,self.e_params))



# Define the class for the Critic 

class Critic(object):
	# initilize the class variables
	def __init__(self,sess,state_dim,action_dim,learning_rate,gamma,t_replace_iter,a,a_):
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = learning_rate
		self.gamma = gamma
		self.t_replace_iter = t_replace_iter
		self.t_replace_counter = 0


		with tf.variable_scope('Critic'):
			# build the trainable network for the Critic neural network
			self.q = self._build_net(S,a,'eval_net',trainable = True)
			# build the target nerual network for the Critic model
			self.q_ = self._build_net(S_,a_,'target_net',trainable = False)

			self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIBALES,scope='Critic/eval_net')
			self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIBALES,scope='Critic/target_net')
		# the Target q_value for updating the weights and bias in the neural network
		with tf.variable_scope('target_q'):
			self.target_q = R + self.gamma * self.q_
		# The loss function for Critic network
		with tf.variable_scope('TD_error'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.target_q,self.q))
		# The training operation
		with tf.variable_scope('C_train'):
			self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)	

		with tf.variable_scope('a_grad'):
			self.a_grad = tf.gradients(ys = self.q,xs = a)[0] # tensor of gradients of each sample (None,a_dim)

	# Definie the neural network build 
	def _build_net(self,s,a,scope,trainable):
		with tf.variable_scope(scope):
			init_w = tf.contrib.layers.xavier_initializer()
			init_b = tf.constant_initializer(0.01)

			with tf.variable_scope('l1'):
				n_l1 = 100
				w1_s = tf.get_variable('w1_s',[self.state_dim,n_l1],initializer=init_w,trainable=trainable)
				w1_a = tf.get_variable('w1_a',[self.action_dim,n_l1],initializer=init_w,trainable=trainable)
				b1 = tf.get_variable('b1',[1,n_l1],initializer=init_b,trainable=trainable)
				net = tf.nn.relu6(tf.matmul(s,w1_s) + tf.matmul(a,w1_a) + b1)

			net = tf.layers.dense(net,20,activation  = tf.nn.relu,kernel_initializer = init_w,bias_initializer=init_b,name='l2',trainable=trainable)

			with tf.variable_scope('q'):
				q = tf.layers.dense(net,1,kernel_initializer=init_w,bias_initializer=init_b,trainable=trainable) # 	Q(S,A)
		return q


	#  The learning method for Critic neural network
	def learn(self,s,a,r,s_):
		self.sess.run(self.train_op,feed_dict={S:s,self.a:a,R:r,S_:s_})
		if self.t_replace_counter % self.t_replace_iter == 0:
			self.sess.run([tf.assign(t,e) for t,e in zip(self.t_params,self.e_params)])
		self.t_replace_counter += 1


# Define the class
class Memory(object):
	def __init__(self,capacity,dims):
		self.capacity = capacity
		self.data = np.zeros((capacity,dims))
		self.pointer = 0 

	def store_transition(self,s,a,r,s_):
		transition = np.hstack((s,a,[r],s_))
		index = self.pointer % self.capacity
		self.data[index,:] = transition
		self.pointer += 1

	def sample(self,n):
		assert self.pointer >= self.capacity, "Memory has not been fulfilled"
		indices = np.random.choice(self.capacity,size=n)
		return self.data[indices,:]


# Define the tf session for the tensorflow

sess = tf.Session()


# create actor and critic
actor = Actor(sess,action_dim,action_bound[1],LR_A,REPLACE_ITER_A)
critic = Critic(sess,state_dim,action_dim,LR_C,GAMMA,REPLACE_ITER_C,actor.a,actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY,dims = 2 * STATE_DIM + ACTION_DIM + 1 )

saver = tf.train.Saver()
path = './discrete' if DISCRETE_ACTION else './continous'

if LOAD:
	saver.restore(sess,tf.train.latest_checkpoint(path))
else:
	sess.run(tf.global_variables_initializer())


def train():
	var = 2 # control exploration
	for ep in range(MAX_EPISODES):
		s = env.reset()
		ep_step = 0

		for t in range(MAX_EP_STEPS):
			# while true
			if RENDER:
				env.render()

			# Added exploration noise
			a = actor.choose_action(s)
			a = np.clip(np.random.normal(a,var),*ACTION_BOUND) # add randomness to action selection for exploration
			s_,r,done = env.step(a)
			M.store_transition(s,a,r,s_)

			if M.pointer > MEMORY_CAPACITY:
				var = max([var * .9995,VAR_MIN]) # decay the action randomness
				b_M = M.sample(BATCH_SIZE)
				b_s = b_M[:,:STATE_DIM]
				b_a = b_M[:,STATE_DIM:STATE_DIM + ACTION_DIM]
				b_r = b_M[:,-STATE_DIM-1:-STATE_DIM]
				b_s_ = b_M[;,-STATE_DIM:]

				critic.learn(b_s,b_a,b_r,b_s_)
				actor.learn(b_s)

			s = s_
			ep_step += 1


			if done or t == MAX_EP_STEPS - 1:
				print ("EP:",ep,
					   "|Step:%i"%int(ep_step),
					   "|Explore: %.2f" % var)
				break
	if os.path.isdir(path):shutil.rmtree(path)
	os.mkdir(path)
	ckpt_path = os.path.join(path,'DDPG.ckpt')
	save_path = saver.save(sess,ckpt_path,write_meta_graph=False)
	print('\nSave Model %s \n'%save_path)


def eval():
	env.set_fps(30)
	while True:
		s = env.reset()
		while True:
			env.render()
			a = actor.choose_action(s)
			s_,r,done = env.step()
			s = s_
			if done:
				break

if __name__ == '__main__':
	if LOAD:
		eval()
	else:
		train()







