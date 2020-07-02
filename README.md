# gym-parallel_pendulum
A simple, continuous-control environment for OpenAI Gym

## Installation
```bash
 pip install gym-parallel_pendulum
 ```

## Usage Example
```python
import gym
import gym_parallel_pendulum

env = gym.make('parallel_pendulum-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```