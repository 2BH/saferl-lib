from gymnasium import register
import gymnasium.spaces as spaces
from gymnasium.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np
from saferl.common.utils import interval_barrier
import torch

"""
this code is modified based on https://githetaub.com/roosephu/crabs/ and OpenAI Gym Pendulum

    ### Description
    
    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |
    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(angle)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
    ### Rewards
    The reward function is defined as:
    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
"""
class SafeClassicPendulum(PendulumEnv):
    def __init__(self,
        init_state,
        threshold,
        goal_state=(0, 0),
        max_torque=2.0,
        obs_type='observation',
        task='upright',
        violation_penalty=lambda x: [5.0],
        incl_cost_in_reward=False,
        **kwargs):

        self.init_state = np.array(init_state, dtype=np.float32)
        self.goal_state = goal_state
        self.threshold = threshold
        self.violation_penalty = violation_penalty
        self.cost_dim = 1
        self.obs_type = obs_type
        self.task = task
        self.incl_cost_in_reward = incl_cost_in_reward
        super().__init__(**kwargs)

        if obs_type == 'state':
            high = np.array([np.pi / 2, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        elif obs_type == 'observation':
            high = np.array([1, 1, self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        else:
            raise ValueError("obs_type must be either 'state' or 'observation', not {}".format(obs_type))

        self.max_torque = max_torque
        self.action_space = spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        theta, thetadot = self.state
        if self.obs_type == 'state':
            return np.array([angle_normalize(theta), thetadot], dtype=np.float32)
        else:
            return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def reset(self, **kwargs):
        self.state = self.init_state
        self.last_u = None
        return self._get_obs(), {}

    def step(self, u):
        # State is not thetae observation!
        theta, thetadot = self.state 

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthetadot = thetadot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u) * dt
        newthetadot = np.clip(newthetadot, -self.max_speed, self.max_speed)
        newtheta = theta + newthetadot * dt

        next_state = np.array([newtheta, newthetadot], np.float32)
        reward = self.reward_fn(self.state, u, next_state)
        is_next_state_safe = self.is_state_safe(next_state)
        # if not is_next_state_safe:
        #     tmp = 1
        self.state = next_state
        info = {}
        info['cost'] = [0.0] if is_next_state_safe else self.violation_penalty(next_state)
        info['state_safe'] = is_next_state_safe

        if self.incl_cost_in_reward:
            reward -= np.sum(info['cost'])

        terminated = not is_next_state_safe
        cost = np.sum(info['cost'])

        return self._get_obs(), reward, cost, terminated, False, info

    def reward_fn(self, states, actions, next_states):
        theta, thetadot = states
        goal_theta, goal_thetadot = self.goal_state
        costs = (goal_theta - theta) ** 2 + .1 * (goal_thetadot - thetadot) ** 2 + .001 * actions ** 2
        return - costs

    def parse_state(self, states):
        if self.obs_type == 'state':
            thetadot = states[..., 1]
            theta = states[..., 0]
        else:
            thetadot = states[..., 2]
            theta = np.arctan2(states[..., 1], states[..., 0])
        return theta, thetadot

    def is_state_safe(self, states):
        return self.barrier_fn(states) <= 1.0

    def barrier_fn(self, states):
        theta, thetadot = states
        return interval_barrier(theta, -self.threshold, self.threshold)
    

class SafePendulumUprightEnv(SafeClassicPendulum):
    def __init__(self, **kwargs):
        super().__init__(init_state=(0.3,-0.9), goal_state=(0,0), threshold=1.5, task='upright', **kwargs)

class SafePendulumTiltEnv(SafeClassicPendulum):
    def __init__(self, **kwargs):
        super().__init__(init_state=(0.3,-0.9), goal_state=(-0.41151684,0), threshold=1.5, task='tilt', **kwargs)

