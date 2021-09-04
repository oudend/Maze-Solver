import gym
import numpy as np
from colr import color
#print(color('Hello world.', fore='red', style='bright'))
from stable_baselines3 import PPO

class BasicEnv(gym.Env):
    def __init__(self):
        self.baseMaze = [1,1,1,1,1,1,1,1,1,1,1,1,
                         1,0,0,0,0,0,0,0,0,0,0,1,
                         1,0,0,1,0,1,0,1,1,1,0,1,
                         1,0,1,1,0,0,0,0,0,0,0,1,
                         1,0,0,0,0,1,0,1,0,1,0,1,
                         1,1,1,1,0,1,0,1,0,1,0,1,
                         1,0,0,0,0,1,1,1,0,1,0,1,
                         1,1,0,1,0,0,0,0,0,0,0,1,
                         1,1,0,1,1,1,0,1,1,0,1,1,
                         1,0,0,0,0,0,0,0,0,0,0,2,
                         1,1,0,1,1,0,1,1,1,1,1,1,
                         1,1,1,1,1,1,1,1,1,1,1,1,]
        self.rowLength = 12
        self.maze = list(self.baseMaze)
        self.position = 14
        #self.color = {1: '1', 0: '0', 2: '2', 3: '3'}
        char = '  '
        self.color = {1: color(char, fore='white', back='white'), 0: color(char, fore='black', back='black'), 2: color(char, fore='green', back='green'), 3: color(char, fore='red', back='red')}
        self.maze[self.position] = 3
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=
                    (len(self.maze),), dtype=np.uint8)

    def _next_observation(self):
        return self.maze

    def move(self, direction = 'w'):
        done = False
        self.maze = list(self.baseMaze)
        reward = 0
        if direction == 'w' and self.maze[self.position - self.rowLength] != 1:
            self.position -= self.rowLength
        elif direction == 's' and self.maze[self.position + self.rowLength] != 1:
            self.position += self.rowLength
        elif direction == 'd' and self.maze[self.position + 1] != 1:
            self.position += 1
        elif direction == 'a' and self.maze[self.position - 1] != 1:
            self.position -= 1
        else:
            reward = -1

        if self.maze[self.position] == 2:
            reward = 10
            done = True

        self.maze[self.position] = 3

        return reward, done
    def step(self, action):
        self.action = action
        #self.reset()
        reward = 0
        #print(action)
        if action == 0:
            reward, done = self.move('w')
        elif action == 1:
            reward, done = self.move('a')
        elif action == 2:
            reward, done = self.move('s')
        elif action == 3:
            reward, done = self.move('d')

        obs = self._next_observation()
        info = {}
        return obs, reward, done, info


    def reset(self):
        self.maze = list(self.baseMaze)

        self.position = 14
        #self.maze[self.position] = 3
        return self.maze

    def render(self):
        rowLength = self.rowLength
        baseRowLength = rowLength
        length = len(self.maze)
        res = ""
        for indx in range(length):
            if indx == rowLength:
                rowLength += baseRowLength
                res += "\n"

            value = self.maze[indx]

            res += self.color[value]
        print(res)

env = BasicEnv()

model = PPO('MlpPolicy', env).learn(100000)

obs = env.reset()
env.render()
for i in range(20):
    action, _states = model.predict(obs)
    #print(action)
    obs, rewards, dones, info = env.step(action)
    env.render()
