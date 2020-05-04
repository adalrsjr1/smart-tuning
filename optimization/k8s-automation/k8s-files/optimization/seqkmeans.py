from __future__ import annotations
import numpy as np
from collections import Counter
import sys
import uuid
import os

import config
# implementation note
# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
#
def mock_sampling():
    with open('letter-recognition.data') as f:
        for line in f:
            v = []
            for i, c in enumerate(line.split(',')):
                if i > 0:
                    v.append(int(c))
            _min = min(v)
            _max = max(v)
            v = np.array(v)
            v = (v - _min) / (_max - _min)
            yield Container(line[0], list(range(len(v))), np.array(v), 0)
        yield None
generator = mock_sampling()

def __distance__(u:Container, v:Container) -> float:
    u = u.content
    v = v.content
    u, v = __resize__(u, v)
    SQRT2 = np.sqrt(2)

    _distance = {
        'hellinger': lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2,
        'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        'euclidean': lambda a, b: np.linalg.norm(a-b)
    }

    return _distance[config.DISTANCE_METHOD.lower()](u, v) or 0

    # hellinger
    # return np.sqrt(np.sum((np.sqrt(u) - np.sqrt(v)) ** 2)) / np.sqrt(2)
    # cosine distance
    # return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # euclidean distance
    # return np.linalg.norm(u-v)

def __resize__(u:np.array, v:np.array) -> (np.array, np.array):
    max_lenght = max(len(u), len(v))

    u = np.append(u, [0] * (max_lenght - len(u)))
    v = np.append(v, [0] * (max_lenght - len(v)))

    return u, v

class Container:
    def __init__(self, label, content_labels, content, metric=0):
        self.label = label
        self.content_labels = content_labels
        self.content = content
        self.metric = metric
        self.configuration = None
        self.start = None
        self.end = None
        self.classification = None

    def __str__(self):
        return f'label:{self.label}, classification:{self.classification.id}, config:{self.configuration}'

    def __add__(self, other):
        u, v = __resize__(self.content, other.content)
        return Container('', [], u + v, 0)

    def __sub__(self, other):
        u, v = __resize__(self.content, other.content)
        return Container('', [], u - v, 0)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            u, v = __resize__(self.content, other.content)
            return Container(self.label, self.content_labels, u * v, self.metric * other)

    def serialize(self):
        container_dict = self.__dict__
        container_dict['content'] = list(self.content)
        container_dict['classification'] = self.classification.id
        return container_dict

    def distance(self, other:Container):
        return __distance__(self, other)

class Cluster:
    def __init__(self, container:Container):
        self.id = str(uuid.uuid1())
        self.hits = 0
        self.len = 0
        self.counter = Counter()
        self.mean = Container('', container.content_labels, container.content, 0)

    def __str__(self):
        return f'[{len(self):4d}] {self.name()}: {self.counter}'

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
        self.mean = self.mean  + (container - self.mean) * (1/len(self))

    def centroid(self):
        return self.mean

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

        return best_cluster, best_cluster.hits


def main(k=26):
    ctx = KmeansContext(k)
    while sample := next(generator):
        cluster = ctx.cluster(sample)
        # print(cluster)

    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        for cluster in ctx.clusters:
            if cluster.most_common()[0][0] == label:
                print(cluster)
