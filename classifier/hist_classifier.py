from common.dataaccess import MongoAccessLayer
from common.timeutil import now
import numpy as np

import os
import sys

from classifier import workload_comparision as wc


class HistClassifier:
    def __init__(self, mongo_url, mongo_port, mongo_db, mongo_collection):
        self.mongo = MongoAccessLayer(mongo_url, mongo_port, mongo_db)
        self.collection = self.mongo.collection(mongo_collection)

    def close(self):
        self.mongo.close()

    def fetch(self, application, start, end):
        result_set = self.mongo.find({'application': application, 'start': {'$gte': start}, 'end': {'$lte': end}},
                                     self.collection)

        return [np.array(list(result['histogram'].values())) for result in result_set]

    def compare(self, histograms, threshould=0):
        from collections import defaultdict
        workflows_group = defaultdict(set)
        memory = set()
        for i, hist1 in enumerate(histograms):
            for j, hist2 in enumerate(histograms):
                distance = wc.hellinger(hist1, hist2)
                if distance <= threshould:
                    self._group(i, j, workflows_group, memory)

        return workflows_group

    def _group(self, a, b, table, memory):
        if a not in memory and b not in memory:
            table[a].add(b)
        elif a in memory and b not in memory:
            if a in table:
                table[a].add(b)
            else:
                return self._group(b, a, table, memory)
        elif a not in memory and b in memory:
            for key, value in table.items():
                if b in value:
                    value.add(a)
                    break
        memory.add(a)
        memory.add(b)


def main():
    from common.timeutil import minute, day
    start = now(past=day(2))
    end = now()

    classifier = HistClassifier('localhost', 27017, 'acmeair_db_experiments', 'acmeair_collection_histogram')
    results = classifier.fetch('acmeair', start, end)
    print(len(results))
    print(results)
    print(classifier.compare(results, threshould=0))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

