from __future__ import annotations
import numpy as np
from collections import Counter
import uuid

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
    u, v, ulabel, vlabel = __resize__(u.content, u.content_labels, v.content, v.content_labels)
    SQRT2 = np.sqrt(2)

    _distance = {
        'hellinger': lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2,
        'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        'euclidean': lambda a, b: np.linalg.norm(a-b)
    }

    return _distance[config.DISTANCE_METHOD.lower()](u, v) or 0

# resize vector preserving labels ordering
def __resize__(u:np.array, ulabels, v:np.array, vlabels) -> (np.array, np.array):
    union = sorted(set(ulabels) | set(vlabels))
    _u, _v = [], []
    _ulabels, _vlabels = [], []
    for item in union:
        _ulabels.append(item)
        _vlabels.append(item)
        if len(u) > 0 and item in ulabels:
            index = ulabels.index(item)
            # to handle a misterious bug when in production
            try:
                _u.append(u[index])
            except IndexError as e:
                _u.append(0)
        else:
            _u.append(0)

        if len(v) > 0 and item in vlabels:
            index = vlabels.index(item)
            # to handle a misterious bug when in production
            try:
                _v.append(v[index])
            except IndexError as e:
                _v.append(0)
        else:
            _v.append(0)
    u, v = _u, _v

    return np.array(u), np.array(v), _ulabels, _vlabels

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
        self.hits = 0

    def __str__(self):
        return f'label:{self.label}, classification:{self.classification.id}, config:{self.configuration}'

    def __add__(self, other):
        u, v, ulabel, vlabel = __resize__(self.content, self.content_labels, other.content, self.content_labels)
        return Container('', ulabel, u + v, max(self.metric, other))

    def __sub__(self, other):
        u, v, ulabel, vlabel = __resize__(self.content, self.content_labels, other.content, self.content_labels)
        return Container('', ulabel, u - v, min(self.metric, other))

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Container(self.label, self.content_labels, self.content * other, max(self.metric, other))

    def serialize(self):
        container_dict = self.__dict__
        # casting np.array to list
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
        v, u, vl, ul = __resize__(container.content, container.content_labels, self.mean.content, self.mean.content_labels)
        self.mean.content_labels = ul
        self.mean.content = v + (u - v) * (1/len(self))
        # self.mean = self.mean  + (container - self.mean) * (1/len(self))

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
        cluster, _ = ctx.cluster(sample)
        print(cluster)

    for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        for cluster in ctx.clusters:
            if cluster.most_common()[0][0] == label:
                print(cluster)

if __name__ == '__main__':
    main()