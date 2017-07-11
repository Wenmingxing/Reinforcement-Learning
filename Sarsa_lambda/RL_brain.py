#! usr/bin/env python

'''
Coded by luke on 10th July 2017 

This part is the mainly code for Sarsa_labmda algorithm.

'''

import numpy as np 
import pandas as pd 


class RL(object):
	def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9, e_greedy=0.9):
		self.actions = action_space # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy

		self.q_table = pd.DataFrame(columns=self.actions)

	def check_state_exist(self,state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state))

	
	def choose_action(self,observation):
		self.check_state_exist(observation)
		# action selection
		if np.random.rand() < self.epsilon:
			state_actions = self.q_table.ix[observation,:]
			state_actions = state_actions.reindex(np.random.permutation(state_actions.index))
			action = state_actions.argmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)
		return action

	def learn(self,*args):
		pass

# backward eligibility traces
class SarsaLambdaTable(RL):
	def __init__(self,actions,learning_rate =0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
		super(SarsaLambdaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

		# backward view, eligibility trace
		self.lambda_ = trace_decay
		self.eligibility_trace = self.q_table.copy()

	def check_state_exist(self,state):
		if state not in self.q_table.index:
			#append new state to q table
			to_be_append = pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state)
			self.q_table = self.q_table.append(to_be_append)

			#also update the eligibility trace
			self.eligibility_trace = self.eligibility_trace.append(to_be_append)

	def learn(self,s,a,r,s_,a_):
		self.check_state_exist(s_)
		q_predict = self.q_table.ix[s,a]
		if s_ != 'terminal':
			q_target = r + self.gamma*self.q_table.ix[s_,a_] # next state is not terminal
		else:
			q_target = r		
		error = q_target - q_predict

		# increase trace amount for visited state_action pair
		self.eligibility_trace.ix[s,:] *= 0
		self.eligibility_trace.ix[s,a] = 1
		# q_table
		self.q_table += self.lr*error*self.eligibility_trace

		# decay eligibility trace after update
		self.eligibility_trace *= self.gamma*self.lambda_