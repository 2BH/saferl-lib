from gymnasium import register
from gymnasium.envs.mujoco.inverted_pendulum_v4 import InvertedPendulumEnv
from gymnasium.utils.ezpickle import EzPickle
from saferl.common.utils import interval_barrier
import numpy as np


class SafeInvertedPendulumEnv(InvertedPendulumEnv):
    """Inverted Pendulum Env from https://github.com/roosephu/crabs/ and OpenAI Gym
    ### Description
    This environment is the cartpole environment based on the work done by
    Barto, Sutton, and Anderson in ["Neuronlike adaptive elements that can
    solve difficult learning control problems"](https://ieeexplore.ieee.org/document/6313077),
    just like in the classic environments but now powered by the Mujoco physics simulator -
    allowing for more complex experiments (such as varying the effects of gravity).
    This environment involves a cart that can moved linearly, with a pole fixed on it
    at one end and having another end free. The cart can be pushed left or right, and the
    goal is to balance the pole on the top of the cart by applying forces on the cart.
    ### Action Space
    The agent take a 1-element vector for actions.
    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)
    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |
    ### Observation Space
    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.
    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:
    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |

    """
    def __init__(self,
        angle_threshold=0.2,
        pos_threshold=np.inf,
        task='move',
        random_reset=False,
        violation_penalty=lambda x: [5.0],
        incl_cost_in_reward=False,
        **kwargs):

        self.angle_threshold = angle_threshold
        self.pos_threshold = pos_threshold
        self.task = task
        self.random_reset = random_reset
        self.violation_penalty = violation_penalty
        self.incl_cost_in_reward = incl_cost_in_reward
        self.cost_dim = 1
        super().__init__(**kwargs)
        EzPickle.__init__(self, angle_threshold=angle_threshold, position_threshold=pos_threshold,
            task=task, random_reset=random_reset)  # deepcopy calls `get_state`
        
    def reset_model(self):
        if self.random_reset:
            qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
            qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
            self.set_state(qpos, qvel)
        else:
            self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def _get_obs(self):
        return super()._get_obs().astype(np.float32)

    def step(self, a):
        # a = np.clip(a, -1, 1)
        cur_state = self._get_obs()
        next_state, _, __, truncated, info = super().step(a)
        reward = self.reward_fn(cur_state, a, next_state)
        
        is_next_state_safe = self.is_state_safe(next_state)
        info['cost'] = [0.0] if is_next_state_safe else self.violation_penalty(next_state)
        info['state_safe'] = is_next_state_safe
        terminated = not is_next_state_safe
        cost = np.sum(info['cost'])
        if self.incl_cost_in_reward:
            reward -= np.sum(info['cost'])
        return next_state, reward, cost, terminated, truncated, info

    def is_state_safe(self, states):
        return self.barrier_fn(states) <= 1.0

    def barrier_fn(self, states):
        # The pole is upright if the angle is less than 0.2 radian and the cart is within the range [-0.9, 0.9]
        return np.maximum(
            interval_barrier(states[..., 1], -self.angle_threshold, self.angle_threshold),
            interval_barrier(states[..., 0], -self.pos_threshold, self.pos_threshold))

    def reward_fn(self, states, actions, next_states):
        if self.task == 'swing':
            reward = next_states[..., 1]**2
        elif self.task == 'move':
            reward = next_states[..., 0]**2
        else:
            assert 0
        return reward

class SafeInvertedPendulumMoveEnv(SafeInvertedPendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(angle_threshold=0.2, pos_threshold=0.9, task='move', **kwargs)

class SafeInvertedPendulumSwingEnv(SafeInvertedPendulumEnv):
    def __init__(self, **kwargs):
        super().__init__(angle_threshold=1.5, pos_threshold=0.9, task='swing', **kwargs)

# Register the environments
# register('SafeInvertedPendulum-v2', entry_point=SafeInvertedPendulumEnv, max_episode_steps=1000)
# register('SafeInvertedPendulumSwing-v2', entry_point=SafeInvertedPendulumSwingEnv, max_episode_steps=1000)
# register('SafeInvertedPendulumMove-v2', entry_point=SafeInvertedPendulumMoveEnv, max_episode_steps=1000)
