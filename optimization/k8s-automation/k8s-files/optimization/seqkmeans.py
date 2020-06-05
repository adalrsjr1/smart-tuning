from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from collections import Counter
import pandas as pd
import numbers
import uuid
import dataclasses

import config
# implementation note
# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
#
def __merge_data__(data1:pd.Series, data2:pd.Series) -> (pd.Series, pd.Series, pd.Index):
    merged = pd.merge(data1, data2, how='outer', left_index=True, right_index=True)
    merged = merged.replace(float('NaN'), 0)
    columns = [column for column in merged.columns[:2]]

    return merged[columns[0]], merged[columns[1]], merged.index

def __distance__(u:pd.Series, v:pd.Series, distance=config.DISTANCE_METHOD.lower()) -> float:
    SQRT2 = np.sqrt(2)

    _distance = {
        'hellinger': lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2,
        'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        'euclidean': lambda a, b: np.linalg.norm(a-b)
    }

    return _distance[distance](u, v) or 0

@dataclass
class Metric:
    cpu: float = 0
    memory: float = 0
    throughput: float = 0
    latency: float = 0

    def __operation__(self, other, op):
        if isinstance(other, Metric):
            return Metric(cpu=op(self.cpu, other.cpu),
                          memory=op(self.memory, other.memory),
                          throughput=op(self.throughput, other.throughput),
                          latency=op(self.latency, other.latency))

        if isinstance(other, numbers.Number):
            return Metric(cpu=op(self.cpu, other),
                          memory=op(self.memory, other),
                          throughput=op(self.throughput, other),
                          latency=op(self.latency, other))

        raise TypeError(f'other is {type(other)} should be a scalar or a Metric type')

    def __add__(self, other):
        return self.__operation__(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.__operation__(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self.__operation__(other, lambda a, b: a * b)

    def __floordiv__(self, other):
        return self.__operation__(other, lambda a, b: a.__floordiv__(b))

    def __truediv__(self, other):
        return self.__operation__(other, lambda a, b: a.__truediv__(b))

    def __divmod__(self, other):
        return self.__operation__(other, lambda a, b: a.__divmod__(b))

    def objective(self, expresssion:str) -> float:
        return eval(expresssion, globals(), self.__dict__) if expresssion else 0


class Container:
    def __init__(self, label, content:pd.Series=None, metric=Metric(cpu=0,memory=0,throughput=0,latency=0),
                 similarity_threshold=config.URL_SIMILARITY_THRESHOULD):

        self.label = label
        self.content = content
        self.content.name = label
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.configuration = None

        self.start = None
        self.end = None

        self.classification = None
        self.hits = 0

    def __str__(self):
        return f'label:{self.label}, classification:{self.classification.id if self.classification else ""}, config:{self.configuration}'

    def serialize(self):
        container_dict = self.__dict__
        # casting np.array to list
        container_dict['content'] = list(self.content)
        if isinstance(self.classification, Cluster):
            container_dict['classification'] = self.classification.id
        elif isinstance(self.classification, str):
            container_dict['classification'] = self.classification
        return container_dict

    def distance(self, other:Container):
        u, v, _ = __merge_data__(self.content, other.content)
        return __distance__(u, v)

class Cluster:
    # todo: merge clusters they have save name
    def __init__(self):
        self.id = str(uuid.uuid1())
        self.hits = 0
        self.len = 0
        self.counter = Counter()
        self.center = pd.Series(name=self.id, dtype=np.float)

    def __str__(self):
        return f'[{len(self):04d}] {self.name()}: {self.counter}'

    def __len__(self):
        return self.len

    def __eq__(self, other):
        return self.id == other.id

    def name(self):
        if self.most_common():
            return self.id + ' ' + self.most_common()[0][0]
        else:
            return self.id

    def most_common(self):
        return self.counter.most_common(1)

    def add(self, container:Container):
        self.len += 1
        self.counter[container.label] += 1

        v, u, index = __merge_data__(container.content, self.center)

        center = v + (u - v) * (1/len(self))

        self.center = pd.Series(data=center, index=index, name=self.id)

        # self.mean.content = v + (u - v) * (1/len(self))
        # self.mean = self.mean  + (container - self.mean) * (1/len(self))

    def centroid(self):
        return self.center

    def inc(self):
        self.hits += 1

class KmeansContext:
    def __init__(self, k, cluster_type=Cluster):
        self.closest_cluster = None
        self.most_common_cluster = None
        self.most_common = 0
        self.min_distance = float('inf')
        self.clusters = []
        self.k = k
        self.cluster_type = cluster_type

    def cluster_by_id(self, id):
        for cluster in self.clusters:
            if id == cluster.id:
                return cluster
        # return self.clusters[np.random.randint(0, len(self.clusters))]
        return None

    def cluster(self, sample:Container):

        if len(self.clusters) < self.k:
            self.clusters.append(self.cluster_type(sample))

        self.closest_cluster = self.clusters[0]
        self.most_common_cluster = self.clusters[0]
        self.most_common = 0
        self.min_distance = float('inf')

        for cluster in self.clusters:
            distance = sample.distance(cluster.centroid())
            if distance < self.min_distance:
                self.min_distance = distance
                self.closest_cluster = cluster

            if len(cluster.most_common()):
                frequency = cluster.most_common()[0][1]
                label = cluster.most_common()[0][0]

                if label == sample.label and frequency > self.most_common:
                    self.most_common = frequency
                    self.most_common_cluster = cluster

        self.closest_cluster.add(sample)
        best_cluster = self.closest_cluster

        best_cluster.inc()

        return best_cluster