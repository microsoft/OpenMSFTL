import numpy as np
from typing import Dict


class AggregationOperator:
    """
    This is the base class for all the implement GAR
    """
    def __init__(self, aggregation_config):
        self.aggregation_config = aggregation_config

    def aggregate(self, G) -> np.ndarray:
        pass

    @staticmethod
    def weighted_average(stacked_grad: np.ndarray, alphas=None):
        """
        Implements weighted average of client grads i.e. rows of G
        If no weights are supplied then its equivalent to simple average / Fed Avg
        """
        if alphas is None:
            alphas = [1.0 / stacked_grad.shape[0]] * stacked_grad.shape[0]
        else:
            assert len(alphas) == stacked_grad.shape[0]
        agg_grad = np.zeros_like(stacked_grad[0, :])
        for ix in range(0, stacked_grad.shape[0]):
            agg_grad += alphas[ix] * stacked_grad[ix, :]
        return agg_grad


class FedAvg(AggregationOperator):
    def __init__(self, aggregation_config):
        AggregationOperator.__init__(aggregation_config=aggregation_config)

