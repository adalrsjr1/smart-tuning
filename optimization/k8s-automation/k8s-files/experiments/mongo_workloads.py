import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy

def __distance__(u:list, lu:list,  v:list, lv:list) -> float:
    u, v, ulabel, vlabel = __resize__(np.array(u), lu, np.array(v), lv)
    SQRT2 = np.sqrt(2)

    hellinger = lambda a, b: np.sqrt(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)) / SQRT2


    return hellinger(u, v)

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

def load_rawdata(filepath):
    data = {'p': [], 't': [], 'distance': []}
    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            u = doc_parsed['prod_workload']['content']
            v = doc_parsed['training_workload']['content']
            lu = doc_parsed['prod_workload']['content_labels']
            lv = doc_parsed['training_workload']['content_labels']


            data['p'].append(u)
            data['t'].append(v)
            try:
                data['distance'].append( __distance__(u, lu, v, lv) )
            except:
                continue

    return pd.DataFrame(data['distance'])


if __name__ == '__main__':

    df = load_rawdata('volume/mongo/20200531-225554/mongo_workloads.json')

    print(df)

    # ax = df.plot(logy=True)
    # ax.set_ylim(0, 1)
    # plt.show()