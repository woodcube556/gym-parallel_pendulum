from gym.envs.registration import register
register(
    id='parallel_pendulum-v0',
    entry_point='gym_parallel_pendulum.envs:ParallelPendulumEnv',
)