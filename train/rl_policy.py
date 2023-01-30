from cmath import inf
from typing import Any, Dict, Optional, Tuple, Type
from train.mlp_extractor import MLPExtractor

import numpy as np
import random
import gym
import torch
from torch import long, nn,Tensor,tensor, bool
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

class RLPolicy(BasePolicy):
    """Base RL policy for the chess agent."""
    def __init__(self,
                observation_space: gym.spaces.Box,
                action_space: gym.spaces.Discrete,
                lr_schedule: Schedule = Schedule,
                log_std_init: float = 0.0,
                use_sde: bool = False,
                squash_output: bool = False,
                ortho_init: bool = True,
                features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
                optimizer_kwargs: Optional[Dict[str, Any]] = None,
                policy_kwargs: Optional[Dict[str, Any]] = None,
                ):
        super(RLPolicy,self).__init__(observation_space,
                                            action_space,
                                            features_extractor_kwargs,
                                            # features_dim,
                                            optimizer_class = optimizer_class,
                                            optimizer_kwargs = optimizer_kwargs,   
                                            squash_output = squash_output                                         
                                            )

        self.features_extractor = MLPExtractor(observation_space=observation_space, policy_kwargs=policy_kwargs) 
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

    def _predict(self, observation: Tensor, deterministic: bool = True) -> Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def _build(self):
        pass

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Checks the log distribution for actions as well as the entropy."""
        distribution, values = self.get_distribution(obs)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()


    def forward(self, obs, deterministic: bool = False):
        """obs space is 45 x 6; for now you can do a simple flatten and run it through a MLP."""
        distribution, values = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def predict_values(self,obs):
        _, values = self.get_distribution(obs)
        return values

    def get_distribution(self,obs):
        
        feature_embedding, mean_actions, values = self.extract_features(obs)

        latent_sde = feature_embedding

        if isinstance(self.action_dist, DiagGaussianDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            distribution =  self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            distribution =  self.action_dist.proba_distribution(mean_actions, self.log_std, latent_sde)
        else:
            raise ValueError("Invalid action distribution")

        return distribution, values
