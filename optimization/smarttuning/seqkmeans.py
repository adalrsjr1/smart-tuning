from __future__ import  annotations
from typing import Union
import numpy as np
from collections import Counter
from sampler import Metric
import pandas as pd
from Levenshtein import StringMatcher
import logging
import uuid
import copy

import config

logger = logging.getLogger(config.KMEANS_LOGGER)
logger.setLevel(logging.DEBUG)


def __merge_data__(data1:pd.Series, data2:pd.Series) -> (pd.Series, pd.Series, pd.Index):
    merged = pd.merge(data1, data2, how='outer', left_index=True, right_index=True)
    merged = merged.replace(float('NaN'), 0)
    columns = [column for column in merged.columns[:2]]

    return merged[columns[0]], merged[columns[1]], merged.index

def __grouping_rows__(data:pd.Series, threshold) -> pd.Series:
    return __hist_reduce__(__compare__(data, threshold))

def __fuzzy_string_comparation__(u:str, v:str, threshold:float):
    """
        compare two strings u, v using Levenshtein distance
        https://en.wikipedia.org/wiki/Levenshtein_distance

        :return how many changes (%) are necessary to transform u into v
    """
    diff = StringMatcher.distance(u, v)
    return diff / len(u) < threshold

def __distance__(u:pd.Series, v:pd.Series, distance=config.DISTANCE_METHOD.lower()) -> float:
    SQRT2 = np.sqrt(2)

    _distance = {
        'hellinger': lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2,
        'cosine': lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
        'euclidean': lambda a, b: np.linalg.norm(a-b)
    }

    return _distance[distance](u, v) or 0

def __compare__(histograms:pd.Series, threshold:int):
    from collections import defaultdict
    workflows_group = defaultdict(set)
    memory = set()

    class Item:
        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __eq__(self, other):
            return self.key == other.key

        def __hash__(self):
            return hash(self.key)

        def __repr__(self):
            return f'({self.key}:{self.value})'

        def acc(self, value):
            self.value += value

    if len(histograms) > 0:
        # groups similar urls
        for hist1, i in histograms.items():
            for hist2, j in histograms.items():
                if __fuzzy_string_comparation__(hist1, hist2, threshold):
                    __group__(Item(hist1, i), Item(hist2, j), workflows_group, memory)
    return workflows_group

# TODO: optimize this in the future
def __group__(a, b, table, memory):
    if a not in memory and b not in memory:
        table[a].add(b)
    elif a in memory and b not in memory:
        if a in table:
            table[a].add(b)
        else:
            return __group__(b, a, table, memory)
    elif a not in memory and b in memory:
        for key, value in table.items():
            if b in value:
                value.add(a)
                break
    memory.add(a)
    memory.add(b)


def __hist_reduce__(table):
    data = {'path': [], 'value': []}

    for key, items in table.items():
        key.value = 0
        for value in items:
            key.acc(value.value)
        data['path'].append(key.key)
        data['value'].append(key.value)

    return pd.Series(data = data['value'], index=data['path']).sort_index()


# implementation note
# https://www.cs.princeton.edu/courses/archive/fall08/cos436/Duda/C/sk_means.htm
#
class Container:
    def __init__(self, label, content:pd.Series=None, metric=None,
                 similarity_threshold=config.URL_SIMILARITY_THRESHOLD):

        self.label = label
        self.content:pd.Series = __grouping_rows__(content, similarity_threshold)
        self.content.name = label

        self.metric = metric
        if self.metric is None:
            self.metric = Metric(cpu=0, memory=0, throughput=0, process_time=0, errors=0)

        self.configuration = None

        self.start = None
        self.end = None

        self.classification = None
        self.hits = 0

    def __str__(self):
        return f'label:{self.label}, classification:{self.classification.id if self.classification else ""}, config:{self.configuration}'

    def __lt__(self, other):
        return self.metric < other.metric

    def serialize(self):
        container_dict = copy.deepcopy(self.__dict__)
        container_dict['content_labels'] = self.content.index.to_list()
        container_dict['content'] = self.content.to_list()
        container_dict['metric'] = self.metric.serialize()

        if isinstance(self.classification, Cluster):
            container_dict['classification'] = self.classification.id
        elif isinstance(self.classification, str):
            container_dict['classification'] = self.classification
        return container_dict

    def distance(self, other:Union[pd.Series, Container]):
        if isinstance(other, Container):
            other = other.content
        u, v, _ = __merge_data__(self.content, other)
        return __distance__(u, v)

class Cluster:
    # todo: merge clusters they have save name
    def __init__(self, container:Container=None):
        self.id = str(uuid.uuid1())
        self.hits = 0
        self.len = 0
        self.counter = Counter()
        self.center = pd.Series(name=self.id, dtype=np.float)

        if container:
            self.add(container)

    def __str__(self):
        return f'[{len(self):04d}] {self.name()}: {len(self.counter)}'

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
        logging.warning('returning None cluster')
        return None

    def cluster(self, sample:Container)->(Cluster, int):
        assert isinstance(sample, Container)
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
        best_cluster:Cluster = self.closest_cluster

        best_cluster.inc()

        return best_cluster, best_cluster.hits