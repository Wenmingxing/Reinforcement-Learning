# ! /usr/bin/env python
'''
Coded by luke on 7th July 2017
Aiming to set up the env for the Q learnig maze example

Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

'''
import numpy as np
np.random.seed(1)
import tkinter as tk
import time


UNIT = 40   # pixels
MAZE_H = 10  # grid height
MAZE_W = 10 # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell1 group 1
        hell1_center = origin + np.array([UNIT * 2, UNIT * 2])
        self.hell11 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15 ,
            fill='black')
        self.hell12 = self.canvas.create_rectangle(
            hell1_center[0] - 15 + UNIT, hell1_center[1] - 15 ,
            hell1_center[0] + 15 + UNIT, hell1_center[1] + 15 ,
            fill = 'black')
        self.hell13 = self.canvas.create_rectangle(
            hell1_center[0] - 15 + UNIT * 2, hell1_center[1] - 15 ,
            hell1_center[0] + 15 + UNIT * 2, hell1_center[1] + 15 ,
            fill = 'black')
        self.hell14 = self.canvas.create_rectangle(
            hell1_center[0] - 15 + UNIT * 3, hell1_center[1] - 15 ,
            hell1_center[0] + 15 + UNIT * 3, hell1_center[1] + 15 ,
            fill = 'black')
        self.hell15 = self.canvas.create_rectangle(
            hell1_center[0] - 15 + UNIT * 4, hell1_center[1] - 15 ,
            hell1_center[0] + 15 + UNIT * 4, hell1_center[1] + 15 ,
            fill = 'black')
        # hell Group 2
        hell2_center = origin + np.array([UNIT * 7, UNIT * 5])
        self.hell21 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15 , hell2_center[1] + 15 ,
            fill='black')
        self.hell22 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT, hell2_center[1] - 15,
            hell2_center[0] + 15 + UNIT , hell2_center[1] + 15,
            fill='black')
        self.hell23 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT * 2, hell2_center[1] - 15,
            hell2_center[0] + 15 + UNIT * 2, hell2_center[1] + 15,
            fill='black')
        '''self.hell24 = self.canvas.create_rectangle(
            hell2_center[0] - 15 , hell2_center[1] - 15 + UNIT,
            hell2_center[0] + 15  , hell2_center[1] + 15 + UNIT,
            fill='black')
        self.hell25 = self.canvas.create_rectangle(
            hell2_center[0] - 15 , hell2_center[1] - 15 + UNIT * 2,
            hell2_center[0] + 15  , hell2_center[1] + 15 + UNIT * 2,
            fill='black')
        self.hell26 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT, hell2_center[1] - 15 + UNIT,
            hell2_center[0] + 15 + UNIT , hell2_center[1] + 15 + UNIT,
            fill='black')
        self.hell27 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT * 2, hell2_center[1] - 15 + UNIT,
            hell2_center[0] + 15 + UNIT * 2, hell2_center[1] + 15 + UNIT,
            fill='black')
        self.hell28 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT, hell2_center[1] - 15 + UNIT * 2,
            hell2_center[0] + 15 + UNIT , hell2_center[1] + 15 + UNIT * 2,
            fill='black')
        self.hell29 = self.canvas.create_rectangle(
            hell2_center[0] - 15 + UNIT * 2, hell2_center[1] - 15 + UNIT * 2,
            hell2_center[0] + 15 + UNIT * 2, hell2_center[1] + 15 + UNIT * 2,
            fill='black')
            '''
        # hell group 3
        hell3_center = origin + np.array([UNIT * 5, UNIT * 8])
        self.hell31 = self.canvas.create_rectangle(
            hell3_center[0] - 15,hell3_center[1] - 15,
            hell3_center[0]+15,hell3_center[1]+15,
            fill='black')
        self.hell32 = self.canvas.create_rectangle(
            hell3_center[0] - 15,hell3_center[1] - 15 + UNIT,
            hell3_center[0]+15,hell3_center[1]+15 + UNIT,
            fill='black')      
        
        '''# hell  group 4
        hell4_center = origin + np.array([UNIT * 14, UNIT * 6])
        self.hell41 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15 , hell4_center[1] + 15,
            fill='black')
        self.hell42 = self.canvas.create_rectangle(
            hell4_center[0] - 15,hell4_center[1] - 15 + UNIT,
            hell4_center[0] + 15, hell4_center[1] + 15 + UNIT,
            fill='black')
        self.hell43 = self.canvas.create_rectangle(
            hell4_center[0] - 15,hell4_center[1] - 15 + UNIT * 2,
            hell4_center[0] + 15, hell4_center[1] + 15 + UNIT * 2,
            fill='black')
        self.hell44 = self.canvas.create_rectangle(
            hell4_center[0] - 15,hell4_center[1] - 15 + UNIT * 3,
            hell4_center[0] + 15, hell4_center[1] + 15 + UNIT * 3,
            fill='black')
        '''
        # create oval
        oval_center = origin + UNIT * 9
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
            Goal_end = True
        elif s_ in [self.canvas.coords(self.hell11), self.canvas.coords(self.hell12),self.canvas.coords(self.hell13),self.canvas.coords(self.hell14),self.canvas.coords(self.hell15),self.canvas.coords(self.hell21),self.canvas.coords(self.hell22),self.canvas.coords(self.hell23),self.canvas.coords(self.hell31),self.canvas.coords(self.hell32)]:
         
            reward = -10
            done = True
            Goal_end = False
        else:
            reward = 0
            done = False
            Goal_end = False
        return s_, reward, done, Goal_end

    def render(self):
        time.sleep(0.1)
        self.update()
