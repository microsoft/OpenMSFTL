class Aggregator:
    def __init__(self, agg_strategy):
        self.agg_strategy = agg_strategy

    def compute_grad(self):
        pass

    def fed_avg(self, clients):
        pass
