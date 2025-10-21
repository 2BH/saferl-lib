from typing import NamedTuple, Dict
import torch as th


class CostRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    states_safe_in_horizon: th.Tensor

TensorDict = Dict[str, th.Tensor]
class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    old_cost_values: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor
    states_safe_in_horizon: th.Tensor