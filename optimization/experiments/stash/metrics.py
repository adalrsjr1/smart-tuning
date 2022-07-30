import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json

def load_rawdata(filepath:str, default:dict) -> (pd.DataFrame, pd.DataFrame):
    data_p = {'cpu': [], 'memory': [], 'throughput': [], 'latency': []}
    data_t = {'cpu': [], 'memory': [], 'throughput': [], 'latency': []}
    data_d = {'cpu': [], 'memory': [], 'throuhgput': [], 'latency': []}

    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            prod = doc_parsed['prod_metric']
            for key, value in prod.items():
                data_p[key].append(value)

            train = doc_parsed['tuning_metric']
            for key, value in train.items():
                data_t[key].append(value)

    data_p, data_t = pd.DataFrame(data_p), pd.DataFrame(data_t)
    data_p['memory'] /= 2 ** 20
    data_p['objective'] = data_p['throughput'] / data_p['memory']
    data_t['memory'] /= 2 ** 20
    data_t['objective'] = data_t['throughput'] / data_t['memory']

    data_d = {}
    for k, col in default.items():
        data_d[k] = col * len(data_p)

    data_d = pd.DataFrame(data_d)

    return data_p, data_t, data_d

def plot(filename, data_p: pd.DataFrame, data_t: pd.DataFrame, data_d: pd.DataFrame):
    n_cols = data_p.count(axis=1)[1]
    n_rows = data_p.count(axis=0)[1]
    fig, axs = plt.subplots(nrows=n_cols, ncols=1, figsize=(12, 8), sharex='col',)

    data_p.plot(ax=axs, linewidth=0.7, drawstyle='steps-post', style=['b-']*n_cols, subplots=True)
    data_t.plot(ax=axs, linewidth=0.7, drawstyle='steps-post', style=['r-']*n_cols, subplots=True)
    data_d.plot(ax=axs, linewidth=0.7, drawstyle='steps-post', style=['g--']*n_cols, subplots=True)

    labels = ['cpu - millicores', 'memory - MB', 'throughput - req/s', 'latency - ms', 'req/s / MB']
    max_y_values = [6, 512, 12000, -1, 30]
    for i, ax in enumerate(axs):
        max_value = max_y_values.pop(0)
        max_value = max_value if max_value > 0 else max(data_p.iloc[:, i].max(), data_t.iloc[:, i].max(), data_d.iloc[:, i].max())
        ax.set_yticks(np.linspace(0, max_value, 5, dtype=float))
        # ax.set_yticks(np.linspace(0, max(data_p.iloc[:, i].max(), data_t.iloc[:, i].max()), 5, dtype=float))
        ax.margins(x=0)


        ax.get_legend().remove()

        ax2 = ax.twinx()
        ax.set_ylabel(labels.pop(0))
        ax2.set_yticks(np.linspace(0, max_value, 10, dtype=float))
        ax.set_yticks([])
        ax.xaxis.set_ticks(np.arange(0, n_rows, 1))
        # ax.legend(frameon=False, loc='lower right')
    ax.set_xlabel('iterations')
    ax.xaxis.set_ticks(np.arange(0, n_rows, 1))

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, ['production', 'training', 'default'], bbox_to_anchor=(0., 1.3, 1., .102), loc='upper center',
           ncol=2,  borderaxespad=0., frameon=False, )



    plt.savefig(filename)
    plt.show()

if __name__ == '__main__':
    data_p, data_t, data_d = load_rawdata('volume/multi-node/20200610-215754/mongo_metrics.json',
                                  {'cpu': [9.85], 'memory': [450], 'throughput': [7470], 'latency': [86], 'objective':[14.10]})
    plot('single-level',data_p, data_t, data_d)

    data_p, data_t, data_d = load_rawdata('volume/multi-node/20200610-174907/mongo_metrics.json',
                                          {'cpu': [10.98], 'memory': [439], 'throughput': [8030], 'latency': [21], 'objective':[16.68]})
    plot('multi-level',data_p, data_t, data_d)
    # print(data_p)