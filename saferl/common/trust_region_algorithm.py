
from typing import List, Optional, Type, Union, Tuple
from gymnasium import spaces
import torch as th
from saferl.common.policies import ActorCriticWithCostPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from saferl.common.on_policy_algorithm import SafeOnPolicyAlgorithm
from stable_baselines3.common.distributions import kl_divergence
from sb3_contrib.common.utils import conjugate_gradient_solver, flat_grad
from stable_baselines3.common.buffers import RolloutBufferSamples

class SafeTrustRegionAlgorithm(SafeOnPolicyAlgorithm):
    """
    Base class for Trust Region Policy Optimization (TRPO) based safe algorithms

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from
    :param cg_max_steps: The maximum number of iterations for the conjugate gradient optimization
    :param cg_damping: The damping factor for the conjugate gradient optimization
    :param line_search_shrinking_factor: The factor to shrink the line search
    :param line_search_max_iter: The maximum number of iterations for the line search
    :param kl_div_step_size: The step size for the KL divergence
    :param supported_action_spaces: The action spaces supported by the algorithm
    :param kwargs: Additional keyword arguments for underlying algorithm
    """
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticWithCostPolicy]],
        env: Union[GymEnv, str],
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 15,
        kl_div_step_size: float = 0.01,
        supported_action_spaces : Optional[Tuple[Type[spaces.Space], ...]] =(
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
            ),
        **kwargs,
    ):
        super(SafeTrustRegionAlgorithm, self).__init__(
            policy,
            env,
            supported_action_spaces=supported_action_spaces,
            **kwargs,
        )

        # conjugate gradient optimization variables
        self.cg_max_steps = cg_max_steps
        self.cg_damping = cg_damping

        # line search variables
        self.line_search_shrinking_factor = line_search_shrinking_factor
        self.line_search_max_iter = line_search_max_iter
        self.max_kl = kl_div_step_size

        # logging variables for line search
        self.new_policy_loss_value = 0
        self.update_kl_divergence = 0
        self.line_search_success = False

    def train(self) -> None:
        """
        Update policy using trust region based on-policy algorithm
        """
        raise NotImplementedError

    def compute_surrogate_loss(self, advantages: th.Tensor, cost_advantages: th.Tensor, log_prob: th.Tensor, old_log_prob: th.Tensor, ep_cost_return_mean: th.Tensor) -> th.Tensor:
        """
        Compute the surrogate loss
        """
        raise NotImplementedError
    
    def matrix_vector_product_derivatives(
        self, kl_div_gradient: th.Tensor, params: List[th.Tensor], vector: th.Tensor
    ) -> th.Tensor:
        """
        Compute the product of the KL divergence gradient and a vector
        Then take the derivative of the result w.r.t. the parameters
        """
        kl_div_gradient_product = th.dot(kl_div_gradient, vector)
        return flat_grad(kl_div_gradient_product, params, retain_graph=True) + self.cg_damping * vector

    def calculate_kl_div_grad_and_policy_grad(self, kl_div: th.Tensor, policy_loss: th.Tensor) -> th.Tensor:
        """
        Calculate the gradient of the KL divergence and the gradient of the policy objective (negative gradient of the policy loss)

        :param kl_div: The KL divergence
        :param policy_loss: The policy loss (negative policy objective)

        :return: The gradient of the KL divergence, the gradient of the policy objective, the parameters and the shape of the gradients
        """
        # This is necessary because not all the parameters in the policy have gradients w.r.t. the KL divergence
        # The policy objective is also called surrogate objective (policy_loss = -policy_objective)
        policy_loss_gradients = []
        # Contains the gradients of the KL divergence
        grad_kl = []
        # Contains the shape of the gradients of the KL divergence w.r.t each parameter
        # This way the flattened gradient can be reshaped back into the original shapes and applied to
        # the parameters
        grad_shape = []
        # Contains the parameters which have non-zeros KL divergence gradients
        # The list is used during the line-search to apply the step to each parameters and for filtering gradients before fitting critics
        actor_params = []
        for name, param in self.policy.named_parameters():
            # Skip parameters related to value function (and cost function) based on name
            # this would also be filtered by the if kl_param_grad is not None
            if "value" in name or "cost" in name:
                continue
                # For each parameter we compute the gradient of the KL divergence w.r.t to that parameter
            kl_param_grad, *_ = th.autograd.grad(kl_div, param, create_graph=True, retain_graph=True, allow_unused=True)

            # Filter None gradients (no computtational path), we store the parameters in the actor_params list
            # and add the gradient and its shape to grad_kl and grad_shape respectively
            if kl_param_grad is not None:
                policy_objective_grad, *_ = th.autograd.grad(policy_loss, param, retain_graph=True, only_inputs=True)
                grad_shape.append(kl_param_grad.shape)
                grad_kl.append(kl_param_grad.reshape(-1))
                policy_loss_gradients.append(policy_objective_grad.reshape(-1))
                actor_params.append(param)
        # Gradients are concatenated before the conjugate gradient step
        policy_gradient_g = -th.cat(policy_loss_gradients)
        kl_div_gradient = th.cat(grad_kl)
        return kl_div_gradient, policy_gradient_g, actor_params, grad_shape
    
    def update_with_line_search(self, actor_params: List[th.Tensor], grad_shape: List[th.Size], step_direction: th.Tensor, rollout_data: RolloutBufferSamples, old_policy_dist: th.distributions.Distribution, policy_loss: th.Tensor, advantages: th.Tensor, cost_advantages: th.Tensor) -> None:
        """
        Update the policy with line search
        
        :param actor_params: The parameters of the actor
        :param grad_shape: The shape of the gradients
        :param step_direction: The step direction
        :param rollout_data: The rollout data
        :param old_policy_dist: The old policy distribution
        :param policy_loss: The policy loss
        :param advantages: The advantages
        :param cost_advantages: The cost advantages
        
        *Note*: This method updates the actor_params in place and logs new_policy_loss_value, update_kl_divergence, line_search_success as information about the line search

        :return: None
        """
        original_actor_params = [param.data.clone() for param in actor_params]
        # update policy with line search
        with th.no_grad():
            line_search_factor = 1.0
            line_search_success = False
            # line search for constraint satisfaction after update
            for i in range(self.line_search_max_iter):
                param_index = 0
                for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                    num_params = param.numel()
                    param.data = original_param.data + line_search_factor * step_direction[param_index : param_index + num_params].view(shape)
                    param_index += num_params

                valid, new_policy_loss, new_kl_div = self.evaluate_line_search_iterate(rollout_data, old_policy_dist, policy_loss, advantages, cost_advantages)
                
                if valid:
                    line_search_success = True
                    break
                line_search_factor *= self.line_search_shrinking_factor

            if not line_search_success:
                for param, original_param in zip(actor_params, original_actor_params):
                    param.data = original_param.data
                
                self.new_policy_loss_value = policy_loss.item()
                self.update_kl_divergence = 0
            else:
                self.new_policy_loss_value = new_policy_loss.item()
                self.update_kl_divergence = new_kl_div.item()

            # log line search result
            self.line_search_success = line_search_success
        
        self.logger.record("train/policy_loss", self.new_policy_loss_value)
        self.logger.record("train/kl_divergence", self.update_kl_divergence)
        self.logger.record("train/line_search_success", self.line_search_success)        
                        
    def evaluate_line_search_iterate(self, rollout_data: RolloutBufferSamples, old_policy_dist: th.distributions.Distribution, policy_loss: th.Tensor, advantages: th.Tensor, cost_advantages: th.Tensor):
        """
        Evaluate the line search iterate
        To be implemented in the inheriting class
        Data that is not given as parameters should be accessed through self
        """
        raise NotImplementedError
                        
    