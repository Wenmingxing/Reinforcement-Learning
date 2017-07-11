#! usr/bin/env python
'''
Coded by luke on 11th July 2017

Sarsa is an online updating method for reinforcement learning

This program is Sarsa lambda method which is more efficient for updating the table
'''

from maze_env import Maze 
from RL_brain import SarsaLambdaTable

def update():
	for episode in range(100):
		# initialize observation
		observation = env.reset()
		print(episode+1)


		# RL choose action based on observation
		action = RL.choose_action(str(observation))

		# initialize all zero eligibility trace
		RL.eligibility_trace *= 0

		while True:
			# fresh env
			env.render()

			# RL take action and get next observation and reward
			observation_,reward,done = env.step(action)

			# RL choose action based on next observation
			action_ = RL.choose_action(str(observation_))

			# RL learn from this transition (s,a,r,s_,a_) ==>> sarsa
			RL.learn(str(observation),action,reward,str(observation_),action_)

			# swap observation and action
			observation = observation_
			action = action_

			# break while loop if the end of this episode reached
			if done:
				break

	print ('game over! Luke')
	print(RL.q_table)
	env.destroy()

if __name__ == '__main__':
	env = Maze()
	RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

	env.after(100,update)
	env.mainloop()