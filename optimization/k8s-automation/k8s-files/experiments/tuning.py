import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

MONGO_PATH = '/Users/adalbertoibm.com/Coding/Dockerized_AcmeAir/smart-tuning/optimization/k8s-automation/experiments/volume/'

def mongo(filename):
    metrics = []
    starts = []
    hits = []
    classification = []
    configs = []
    contents = []
    with open(MONGO_PATH+filename) as f:
        for item in f:
            item = json.loads(item)
            print(item)
            metrics.append(item['metric'])
            starts.append(item['start'])
            hits.append(item['hits'])
            classification.append(item['classification'])
            configs.append(item['configuration'])
            contents.append(item['content'])

    return metrics, starts, hits, classification, configs, contents

def plot(xs, ys, xlabel, ylabel, title):
    fig, ax = plt.subplots()

    plt.plot(np.arange(len(xs)), ys)
    # for i, bar in enumerate(barlist):
    #     bar.set_color(f'C{i}')

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    # ax.set_title('Distribution of requests')

    # ax.set_ylim([0, 1])
    # ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    #
    # labels = [f'url-{x}' for x in xs]
    # ax.set_xticklabels(labels)


    plt.show()

if __name__ == '__main__':
    metrics, starts, hits, classification, configs, contents = mongo('mongo2.json')
    counter = Counter(classification)
    plot(starts, metrics, 'iteration', 'throughput per sampling -- (req/s)', '2 min sampling')
    for a, b, c in zip(metrics, classification, contents):
        print(a,b, c)
    print(counter)