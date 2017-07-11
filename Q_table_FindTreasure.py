#! usr/bin/env python

'''
Coded by luke on 6th July 2017

Aiming to get familiar with the Q_table with a simple example whose name is find the treasure.


'''

import numpy as np
import pandas as pd 
import time 

N_STATE = 6 # The number of the states
ACTIONS = ['left','right'] # the actions for each states
EPSILON = 0.9 # Greedy rate to choose an action randomly
ALPHA = 0.1 # the learning rate for each actions
GAMMA = 0.9 # the discount factor for the feature q_value
MAX_EPISODES = 13 # The maximum episodes for training
FRESH_TIME = 0.1 # The time for each update


# Define the Q_table for this problem
def build_q_table(N_STATE,ACTIONS):
	table = pd.DataFrame(np.zeros((N_STATE,len(ACTIONS))),columns=ACTIONS)
	return table


# Define the function to choose the actions 
def choose_action(state,q_table):
	'this is a function about the choosing the actions'
	state_actions = q_table.iloc[state,:]
	if(np.random.uniform()>EPSILON) or (state_actions.all() == 0):
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.argmax()
	return action_name


def get_env_feedback(s,a):
	'This is how the agent will inteact with the environment'
	if a == 'right': # move to right 
		if s == N_STATE - 2: # Terminate
			s_ = 'terminal'
			r = 1
		else:
			s_ = s + 1
			r = 0
	else: # move to left
		r = 0
		if s == 0:
			s_ = s 
		else:
			s_ = s - 1
	return s_,r

def update_env(s,episode,step_count):
	'This is how the environment be updated'
	env_list = ['-'] * (N_STATE-1) + ['T'] 
	if s == 'terminal':
		interaction = 'Episode %s: total_steps = %s' %(episode+1,step_count)
		print('\r{}'.format(interaction),end='')
		time.sleep(2)
		print('\r                               ',end='')
	else:
		env_list[s] = '*'
		interaction = ''.join(env_list)
		print('\r{}'.format(interaction),end='')
		time.sleep(FRESH_TIME)
	
def rl():
	'This is the main part for RL loop'
	q_table = build_q_table(N_STATE,ACTIONS)
	for episode in range(MAX_EPISODES):
		step_count = 0
		s = 0
		is_terminated = False
		update_env(s,episode,step_count)

		while not is_terminated:
			a = choose_action(s,q_table)
			s_,r = get_env_feedback(s,a)
			q_predict = q_table.ix[s,a]
			if s_ != 'terminal':
				q_target = r + GAMMA*q_table.iloc[s_,:].max()
			else:
				q_target = r
				is_terminated = True	# terminate this episode

			q_table.ix[s,a] += ALPHA*(q_target - q_predict) # update
			s = s_ # move to the next states
			step_count += 1
			update_env(s,episode,step_count)
	return q_table


if __name__ == "__main__":
	q_table = rl()
	print('\r\nQ_table:\n')
	print(q_table)
