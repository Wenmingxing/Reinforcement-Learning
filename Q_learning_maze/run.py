#! usr/bin/env python

'''
Coded by luke on 7th July 2017

The run program for the RL maze example

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].


'''

from maze_env import Maze 
from RL_brain import QLearningTable

def update():
	for episode in range(1000):
		# initialize the observation
		observation = env.reset()
		while True:
			# fresh env
			env.render()

			# RL choose action based on observation
			action = RL.choose_action(str(observation))

			# RL take action and get next obbservation and reward
			observation_,reward,done,Goal_end = env.step(action)

			# RL learn from this transition
			RL.learn(str(observation),action,reward,str(observation_))

			# swap observation
			observation = observation_

			# break while loop when end of this episode
			if done:
				if Goal_end == False:
					print('The {} episode has stopped without reaching the Goal'.format(episode+1))
				elif Goal_end == True:
					print('The {} episode has stopped successfully reaching the Goal'.format(episode+1))
				break
	# end of the game
	print ('game over!')
	print(RL.q_table)

	#env.destroy()


if __name__=="__main__":
	env = Maze()
	RL = QLearningTable(actions=list(range(env.n_actions)))
	


	#env.after(100,update)
	#env.mainloop()	