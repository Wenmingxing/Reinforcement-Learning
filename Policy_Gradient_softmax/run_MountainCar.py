#! usr/bin/env python

"""
Coded by Luke wen on 17th Oct 2017, aiming to apply the policy gradient in MountainCar

referring to https://morvanzhou.github.io/tutorials/

"""

import gym 
import matplotlib.pyplot as plt 
from RL_brain import PolicyGradient

DISPLAY_REWARD_THRESHOLD = -2000 # renders environment if total episode reward is greater then this threshold

RENDER = False # Rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1) # reproducible, general Policy gradient has high varianve
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(n_actions = env.action_space.n,n_features = env.observation_space.shape[0],learning_rate = 0.02,reward_decay = 0.995)

for i_episode in range(1000):
	observation = env.reset()

	while True:
		if RENDER: env.render()

		action = RL.choose_action(observation)

		# reward = -1 in all cases
		observation_,reward,done,info = env.step(action)
		RL.store_transition(observation,action,reward)

		if done:
			# calculate running reward
			ep_rs_sum  = sum(RL.ep_rs)

			if "running_reward" not in globals():
				running_reward = ep_rs_sum
			else:
				running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
				if running_reward > DISPLAY_REWARD_THRESHOLD:RENDER = True # rendering

				print('episode:',i_episode," reward:",int(running_reward))

				vt = RL.learn() # train

				if i_episode == 30:
					plt.plot(vt)
					plt.xlabel('episode steps')
					plt.ylabel('normalized state-action value')
					plt.show()
				break
		observation = observation_