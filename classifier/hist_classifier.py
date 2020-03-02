from common.dataaccess import MongoAccessLayer
from common.timeutil import now
import numpy as np

import os
import sys

from classifier import workload_comparision as wc
# data = []
# data.append({'metric': metric, 'mean': query_mean[0], 'std': query_std[0]})
# data = {
#             'metrics': {'n_samples': QUERY_STEP // SCRAP_INTERVAL, 'values': data},
#             'histogram': histogram,
#             'start': end - QUERY_STEP,
#             'end': end,
#             'step': QUERY_STEP
#         }

class HistClassifier:
    def __init__(self, application, mongo_url, mongo_port, mongo_db, histogram_collection, tuning_collection):
        self.application = application
        self.mongo = MongoAccessLayer(mongo_url, mongo_port, mongo_db)
        self.histogram_collection = self.mongo.collection(histogram_collection)
        self.tuning_collection = self.mongo.collection(tuning_collection)

    def close(self):
        self.mongo.close()

    def tunings(self, start, end):
        return self.mongo.find({'start': {'$gte': start}, 'end': {'$lte': end}},
                               self.tuning_collection)

    def histograms(self, start, end):
        return self.mongo.find({'application': self.application, 'start': {'$gte': start}, 'end': {'$lte': end}},
                               self.histogram_collection)

    def join_tuning_histogram(self, start, end):
        _tunings = self.tunings(start, end)
        _histograms = self.histograms(start, end)

        processed_tunings = []
        for tuning in _tunings:
            start = tuning['start']
            end = tuning['end']

            filtered_histograms = []
            for histogram in _histograms:
                if histogram['start'] >= start and histogram['end'] <= end:
                    filtered_histograms.append(histogram)

            tuning.update({'histograms': filtered_histograms})
            processed_tunings.append(tuning)
        return processed_tunings

    def fetch(self, start, end):
        result_set = self.mongo.find({'application': self.application, 'start': {'$gte': start}, 'end': {'$lte': end}},
                                     self.histogram_collection)

        simple_histogram = {}
        for result in result_set:
            id = str(result['_id'])
            simple_histogram[id] = np.array(list(result['histogram'].values()))

        return simple_histogram

    def compare(self, histograms, threshould=0):
        from collections import defaultdict
        workflows_group = defaultdict(set)
        memory = set()
        for i, hist1 in histograms.items():
            for j, hist2 in histograms.items():
                distance = wc.hellinger(hist1, hist2)
                if distance <= threshould:
                    self._group(i, j, workflows_group, memory)

        return workflows_group

    # TODO: optimize this in the future
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

    classifier = HistClassifier('acmeair', 'localhost', 27017, 'acmeair_db_experiments', 'acmeair_collection_histogram',
                                'acmeair_collection_tuning')

    histograms = classifier.fetch(start, end)
    print(classifier.compare(histograms, 0))

    #for hist in classifier.join_tuning_histogram(start, end):
    #    print(hist)

    # results = classifier.fetch(start, end)
    # print(len(results))
    # print(results)
    # print(classifier.compare(results, threshould=0))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
