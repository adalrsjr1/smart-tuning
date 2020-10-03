import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename:str) -> pd.DataFrame:
    return pd.read_csv(filename)

def split_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    :param table:
    :return: metric and workload table
    """
    # metrics
    metrics_table = table[[
        'production_metric.cpu',
        'production_metric.memory',
        'production_metric.throughput',
        'production_metric.process_time',
        'production_metric.errors',
        'production_metric.objective',
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective',
        'tuned'
    ]].copy(),
    # workloads
    workload_table = table[[
        'production_workload.path',
        'production_workload.value',
        'training_workload.path',
        'training_workload.value',
        'tuned'
    ]].copy()

    return metrics_table[0], workload_table

def plot_metrics(table:pd.DataFrame):
    nrows = 6
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 8), sharex='col', gridspec_kw = {'wspace':0, 'hspace':0})
    target = [axs[0], axs[1], axs[2], axs[3], axs[4], axs[5]]

    table[[
        'production_metric.cpu',
        'production_metric.memory',
        'production_metric.throughput',
        'production_metric.process_time',
        'production_metric.errors',
        'production_metric.objective']].plot(ax=target, linewidth=0.7, drawstyle='steps-post', style=['b-']*nrows, subplots=True)

    table[[
        'training_metric.cpu',
        'training_metric.memory',
        'training_metric.throughput',
        'training_metric.process_time',
        'training_metric.errors',
        'training_metric.objective']].plot(ax=target, linewidth=0.7, drawstyle='steps-post', style=['r-']*nrows, subplots=True)

    ax2 = axs[0].twinx()
    axs[0].set_ylabel('cpu')
    ax2.set_yticks(np.linspace(0, 16, 4, dtype=float))
    axs[0].set_yticks([])
    axs[0].xaxis.set_ticks(np.arange(0, nrows, 1))

    axs[1].legend(frameon=False, )
    for i in range(1,nrows):
        axs[i].get_legend().remove()
    # ax.margins(x=0)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles, ['production', 'training'], bbox_to_anchor=(0., 1.3, 1., .102),
                  loc='upper center',
                  ncol=2, borderaxespad=0., frameon=False, )
    axs[-1].set_xlabel('iterations')
    plt.show()

if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    df = load_data('./resources/logging-202009281720.csv')
    mtable, wtable = split_table(df)

    plot_metrics(mtable)

