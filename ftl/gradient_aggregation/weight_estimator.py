# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License

from typing import Dict, List
import numpy as np
from ftl.gradient_aggregation.reinforcement_learner import RL

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.
    Have our own implementation since scipy.special.softmax sometimes causes issues for installation

    params:
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the first non-singleton axis.

    return: softmax output, the same size as X. The sum will be unity along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class WeightEstimatorBase:
    """
    Base class for gradient weight estimator
    """
    def __init__(self, estimator_type: str, weights: "array-like list"):
        self.estimator_type = estimator_type
        self.weights = weights
        self.steps = 0

    def compute_weights(self, input_feature):
         raise NotImplementedError

    def update_model(self):
        raise NotImplementedError


class SoftmaxLWeightEstimator(WeightEstimatorBase):
    """
    Use Softmax output as a weight vector
    """
    def __init__(self, softmax_config: Dict):
        super(SoftmaxLWeightEstimator, self).__init__(estimator_type='softmax', weights=[])
        self.T = softmax_config.get("T", 1.0)  # Temperature value for softmax
        # set the index of the feature vector (softmax input vector)
        if softmax_config.get("feat_type", "loss"):
            self.offset = 0
        elif softmax_config["feat_type"] == "mean":
            self.offset = 1
        elif softmax_config["feat_type"] == "var":
            self.offset = 2
        else:
            raise ValueError("SoftmaxLWeightEstimator: unsupported feature type {}".format(softmax_config["feat_type"]))

    def compute_weights(self, input_feature: "array-like list", num_clients: int):
        """
        Compute the softmax.
        The component of the input_feature is, for example, orgnized as
        input_feature[0:no_clients] = negative loss
        input_feature[no_clients:2*no_clients] = mean
        input_feature[2*no_clients:3*no_clients] = var
        """
        return softmax(input_feature[self.offset*num_clients:(self.offset+1)*num_clients:1], theta=1.0/self.T)

    def update_model(self):
        pass


class RLWeightEstimator(WeightEstimatorBase):
    """
    This class estimates weights with a reinforcement learner
    """
    def __init__(self, rl_config: Dict):
        """
        :param rl_config: dict specifying parameters for RL-based gradient weight estimator

        :example:
        rl_config = {
                "type": "RL",
                "marginal_update_RL": True,
                "num_warmup_steps": 2,
                "delta_threshold": 0.001,
                ....
        }
        Also see `class RL` in gradient_aggregation.reinforcement_learner.py
        """
        super(RLWeightEstimator, self).__init__(estimator_type=rl_config.get('type', 'RL'), weights=[])
        self.RL = RL(rl_config=rl_config)
        self.marginal_update_rl = rl_config.get('marginal_update_RL', True)
        self.num_warmup_steps = rl_config.get('num_warmup_steps', 2)  # how many times the RL model should be updated
        self.delta_th = rl_config.get('delta_threshold', 0.001)
        self.verbose = rl_config.get('verbose_level', 1)

    def _post_process_weights(self, rl_weights: "array-like list"):
        if rl_weights.ndim > 1:
            rl_weights = rl_weights[-1, :]

        rl_weights = np.exp(rl_weights)

        # didimit: 05-30-20
        if self.verbose > 1:
            print('RL Weights BEFORE filtering: {}'.format(rl_weights))
        index = np.argwhere(np.isnan(rl_weights))
        rl_weights[index] = 0
        if self.verbose > 1:
            print('RL Weights AFTER filtering: {}'.format(rl_weights))
        return rl_weights

    def compute_weights(self, input_feature: "array-like list"):
        """
        Compute a weight for client's gradient

        :type input_feature: object
        :param input_feature: input feature vector
        :return: weight vector
        """
        assert self.RL.network_params[0] == len(input_feature), \
            "Invalid network input size in {}!={}".format(self.RL.network_params[0], len(input_feature))

        # Reinforcement Learning for estimating weights
        rl_weights = self.RL.forward(input_feature).cpu().detach().numpy()
        rl_weights = self._post_process_weights(rl_weights)

        weight_sum = np.sum(rl_weights)
        self.weights = rl_weights / weight_sum

        return self.weights

    def update_model(self, input_feature: "array-like list",
                           rl_error_rate: float,
                           org_error_rate: float):
        """
        Update the RL model and return an indicator whether the RL weight should be used or not

        :param input_feature:
        :param rl_error_rate:
        :param org_error_rate:
        :return: true if a model generating 'rl_error_rate' is ready to be used
        """
        assert len(self.weights) > 0, "Initialize or compute weights before calling this"
        # Expected structure of batch
        # state, weights, reward, state_1 = batch
        should_use_rl_model = False
        if abs(org_error_rate - rl_error_rate) < self.delta_th:
            reward = 0.1
            if self.marginal_update_rl is True and self.steps > self.num_warmup_steps:
                should_use_rl_model = True
        elif org_error_rate > rl_error_rate:
            reward = 1.0
            if self.steps > self.num_warmup_steps:
                should_use_rl_model = True
        else:
            reward = -1.0

        # Taking the policy from a game-based RL
        # The reward is 0.1 for each bird’s move without dying when it is not passing through a pipe, 1 if the bird
        # successfully pass through a pipe and -1 if the bird crashes.
        batch = (input_feature, self.weights, [reward])
        self.RL.train(batch)
        self.RL.save(self.steps, reward, rl_error_rate, self.RL.runningLoss)
        if self.verbose > 0:
            print("RL i = {}, org_er = {:0.4f}, rl_er = {:0.4f}, reward = {}".format(self.steps, org_error_rate, rl_error_rate, reward))
            print("RL Running Loss = {}".format(self.RL.runningLoss))

        self.steps += 1
        return should_use_rl_model
