from sampler import Metric


class BayesianDTO:
    def __init__(self, metric: Metric = Metric.zero(), workload_classification: str = ''):
        self.metric = metric
        self.classification = workload_classification

    def __repr__(self):
        return f'{{"metric": {self.metric}, "classification": "{self.classification}"}}'


class EmptyBayesianDTO(BayesianDTO):
    def __init__(self):
        super(EmptyBayesianDTO, self).__init__()