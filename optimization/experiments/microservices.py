import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import sys

FILEPATH='resources/logging-202009181900.csv'

def extract_data_frames(filepath) -> pd.DataFrame:
    return pd.read_csv(filepath)

def extract_metrics(df:pd.DataFrame, service_name:str, training:bool, k) -> pd.DataFrame:

    if 'nginx' in service_name:
        return extract_gw_metrics(df, service_name, training)

    prefix = 'training' if training else 'production'
    df = df.filter([

        f'{prefix}_metric.cpu',
        f'{prefix}_metric.memory',
        f'{prefix}_metric.throughput',
        f'{prefix}_metric.process_time',
        f'{prefix}_metric.errors',
        f'{prefix}_metric.objective',
        f'best_config.{service_name}-configsmarttuning.MONGO_CONNECTIONS_PER_HOST',

    ]).copy(deep=True).rename(columns={
        f'{prefix}_metric.cpu': 'cpu',
        f'{prefix}_metric.memory': 'memory',
        f'{prefix}_metric.throughput': 'throughput',
        f'{prefix}_metric.process_time': 'process_time',
        f'{prefix}_metric.errors': 'errors',
        f'{prefix}_metric.objective': 'objective',
        f'best_config.{service_name}-configsmarttuning.MONGO_CONNECTIONS_PER_HOST': 'config'
    })[k::2].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
    df['memory'] /= 1024**2
    return df

def extract_gw_metrics(df:pd.DataFrame, service_name:str, training:bool) -> pd.DataFrame:
    suffix = 'train' if training else 'prod'
    df = df.filter([

        f'overall_metrics_{suffix}.cpu',
        f'overall_metrics_{suffix}.memory',
        f'overall_metrics_{suffix}.throughput',
        f'overall_metrics_{suffix}.process_time',
        f'overall_metrics_{suffix}.errors',
        f'overall_metrics_{suffix}.objective',
    ]).copy(deep=True).rename(columns={
        f'overall_metrics_{suffix}.cpu': 'cpu',
        f'overall_metrics_{suffix}.memory': 'memory',
        f'overall_metrics_{suffix}.throughput': 'throughput',
        f'overall_metrics_{suffix}.process_time': 'process_time',
        f'overall_metrics_{suffix}.errors': 'errors',
        f'overall_metrics_{suffix}.objective': 'objective',
    })[::2].apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

    df['memory'] /= 1024 ** 2
    return df

def label_point(x, y, val, ax:plt.Axes):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(int(point['val'])), fontsize=6)

def plot_metrics(title:str, servicename:str, df:pd.DataFrame):
    table_t = extract_metrics(df, servicename, training=True)
    table_p = extract_metrics(df, servicename, training=False)

    n_plots = len(table_p.columns)
    axes = table_p.plot(subplots=True, drawstyle='steps', linewidth=0.5, style=['b-'] * n_plots, figsize=(8, 8))
    table_t.plot(ax=axes, subplots=True, drawstyle='steps', linewidth=0.5, style=['r-'] * n_plots, figsize=(8, 8))

    ax: plt.Axes
    for i, ax in enumerate(axes):
        ax.set_ylabel(table_p.keys()[i])
        # ax.get_yaxis().set_label_position('right')
        ax.get_yaxis().tick_right()
        ax.xaxis.set_ticks(np.arange(0, max(table_t.index) + 1, 5))
        ax.margins(x=0)

        ax.get_legend().remove()

    ax.set_xlabel('iterations')
    axes[0].set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, ['production', 'training'], bbox_to_anchor=(0., 1.3, 1.7, .100),
                   loc='upper center',
                   ncol=2, borderaxespad=0., frameon=False, )

    plt.savefig(f'{title}.png', dpi=300)


if __name__ == '__main__':
    df = extract_data_frames(FILEPATH)

    # acmeair-bookingservice
    # acmeair-customerservice
    # acmeair-customerservice
    # nginx

    # plot_metrics('customer', 'acmeair-customerservice', df)
    # plot_metrics('booking', 'acmeair-bookingservice', df)
    # plot_metrics('gateway', 'nginx', df)

    #
    #
    # # print(booking.columns.values)
    # # print(customer.columns.values)

    training = True
    gw = extract_metrics(df, 'nginx', training=training, k=0)
    booking = extract_metrics(df, 'acmeair-bookingservice', training=training, k=0)
    customer = extract_metrics(df, 'acmeair-customerservice', training=training, k=1)
    flight = extract_metrics(df, 'acmeair-flightservice', training=training, k=1)


    metric_name = 'throughput'
    joined = pd.DataFrame({'booking': booking[metric_name], 'customer': customer[metric_name], 'flight': flight[metric_name],
                           'gateway': gw[metric_name]})

    headers = ['booking', 'customer']
    # print(joined)
    # # #
    joined = joined.drop(joined.index[0])
    joined = joined.apply(pd.to_numeric, errors='coerce')
    joined = joined.dropna()
    joined = joined.reset_index(drop=True)
    # #
    colormap = 'summer'
    ax:plt.Axes
    # ax = joined.plot.scatter(x='booking', y='customer',  c='gateway', colormap=colormap, figsize=(12,8))
    # ax = joined.plot.scatter(ax=ax, x='booking', y='customer',  c='gateway', colormap=colormap, marker=2, figsize=(12,8))
    title = metric_name
    fig:plt.Figure
    fig, axes = plt.subplots(nrows=1, ncols=1)
    all_headers_permuations = list(combinations(headers, 2))

    # _axes = list(axes)
    _axes = [axes]
    fig.suptitle(title)
    # fig.set_tight_layout(True)
    for ax, permutation in zip(_axes, all_headers_permuations):
        x, y = permutation
        print(ax, x, y)
        _ax = joined.plot.hexbin(ax=ax, x=x, y=y, C='gateway', reduce_C_function=np.max, gridsize=20,
                           colormap=colormap)
        # _ax.set_title(title)
        # label_point(joined.booking, joined.customer, pd.Series(joined.index), _ax)
        _ax.set_xlabel(f'{x} {title}')
        _ax.set_ylabel(f'{y} {title}')
        label_point(joined.booking, joined.customer, pd.Series(joined.index), _ax)


    # ax = joined.plot.hexbin(x='booking', y='customer',  C='gateway', reduce_C_function=np.max, gridsize=20, colormap=colormap)
    # # ax = joined.plot.scatter(x='booking', y='customer',  c='gateway', colormap=colormap)
    # ax.set_title(title)
    # label_point(joined.booking, joined.customer, pd.Series(joined.index), ax)
    # ax.set_xlabel('booking throughput')
    # ax.set_ylabel('customer throughput')
    #
    plt.savefig(f'cor-{title}-{"train" if training else "prod"}.png', dpi=300)
    plt.show()
