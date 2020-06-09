import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json

def load_rawdata(filepath):
    data = {'prod. pod': [], 'train. pod': []}
    with open(filepath) as f:
        for doc in f:
            doc_parsed = json.loads(doc)
            print(doc_parsed)
            # data['prod. pod'].append(float(doc_parsed['prod_metric']))
            # data['train. pod'].append(float(doc_parsed['tuning_metric']))

    return pd.DataFrame(data)

if __name__ == '__main__':
    print(load_rawdata('volume/multi-node/20200608-041757/mongo_metrics.json'))