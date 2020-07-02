import gym
from gym import spaces
from gym.utils import seeding

import math
import numpy as np
from scipy.integrate import odeint
import requests

url = 'http://localhost:3000/api/state'

class ParallelPendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        self.viewer = None

        inf = float('inf')
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-inf, -inf, -inf, -inf, -inf, -inf]), high=np.array([inf, inf, inf, inf, inf, inf]), dtype=np.float32)

        # Phisical Parameters
        self.m1 = 0.1
        self.m2 = 0.15
        self.lg1 = 0.2
        self.lg2 = 0.3
        self.I1 = self.m1 * (2*self.lg1)**2 / 12
        self.I2 = self.m2 * (2*self.lg2)**2 / 12
        self.L = 0.5
        self.J = 0.75
        self.g = 9.8

        self.D1 = 1.5e-4
        self.D2 = 4.7e-4
        self.Dm = 1.4
        self.K1 = 0#2.5e-3
        self.K2 = 0#3.6e-3
        self.Km = 0#5.0
        
        self.time = 0.0
        self.dt = 1e-2

        self.max_step = 800 #360

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        def ode(state, t):
            theta1 = state[0]
            theta2 = state[1]
            phi = state[2]
            theta1_dot = state[3]
            theta2_dot = state[4]
            phi_dot = state[5]

            tau = action[0]

            # Coulomb Friction
            F1 = self.D1*theta1_dot + self.K1
            F2 = self.D2*theta2_dot + self.K2
            Fm = self.Dm*phi_dot + self.Km

            m1, m2, lg1, lg2, I1, I2, L, J, g = self.m1, self.m2, self.lg1, self.lg2, self.I1, self.I2, self.L, self.J, self.g

            _A = m1*lg1**2 + I1
            _B = m1*L*lg1*math.cos(theta1)
            _C = -_A*math.sin(theta1)*math.cos(theta1)*phi_dot**2 - m1*g*lg1*math.sin(theta1) + F1
            _D = m2*lg2**2 + I2
            _E = m2*L*lg2*math.cos(theta2)
            _F = -_D*math.sin(theta2)*math.cos(theta2)*phi_dot**2 - m2*g*lg2*math.sin(theta2) + F2
            _G = m1*L**2 + m2*L**2 + J + _A*math.sin(theta1)**2 + _D*math.sin(theta2)**2
            _H = (2*_A*math.sin(theta1)*math.cos(theta1)*theta1_dot + 2*_D*math.sin(theta2)*math.cos(theta2)*theta2_dot)*phi_dot \
                - m1*L*lg1*math.sin(theta1)*theta1_dot**2 - m2*L*lg2*math.sin(theta2)*theta2_dot**2
            _I = tau-Fm

            phi_ddot = (_B*_C*_D + _A*_E*_F - _A*_D*_H + _A*_D*_I) / (_A*_D*_G - _B**2*_D - _A*_E**2)
            theta1_ddot = -(_B*phi_ddot + _C)/_A
            theta2_ddot = -(_E*phi_ddot + _F)/_D

            return [theta1_dot, theta2_dot, phi_dot, theta1_ddot, theta2_ddot, phi_ddot]

        t = [0.0, self.dt]
        _, next_state = odeint(ode, self.state, t)

        self.state = next_state
        
        done = bool(self.time >= self.max_step*self.dt)

        reward = 0
        # if math.cos(self.state[0]) >= 0.8 and math.cos(self.state[1]) >= 0.8:
        #     reward += 1
        # else:
        #     if done:
        #         reward = -100.0
        reward = math.cos(self.state[0]) + math.cos(self.state[1])
        self.time += self.dt
        return self.state, reward, done, {}

    def reset(self, init=None):
        self.time = 0.0
        if init is not None:
            self.state = init
        else:
            high = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            self.state = self.np_random.uniform(low=-high, high=high)
        return self.state

    def render(self, mode='human'):
        response = requests.post(url, data={'theta1': self.state[0], 'theta2': self.state[1], 'phi': self.state[2]})

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
