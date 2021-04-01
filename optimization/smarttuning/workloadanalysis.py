import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import scipy
import json

def load(filepath:str)->pd.DataFrame:
    with open(filepath) as f:
        data = json.load(f)['data']
        result = data['result'][0]
        df = pd.DataFrame([(pd.Timestamp(v[0], unit='s'), np.float64(v[1])) for v in result['values']], columns=['idx', 'values'])
        # df = pd.DataFrame([(np.int(v[0]), np.float64(v[1])) for v in result['values']], columns=['idx', 'values'])
        return df.set_index('idx').fillna(1)

if __name__ == '__main__':
    df = load('resources/productionN.json')

    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    # Construct the model
    for idx, i in enumerate(range(9,len(df))):
        # result = seasonal_decompose(df['values'][:i], model='multiplicative', period=15)
        #
        # endog = result.trend
        # endog = df[:i].cumsum()
        endog = df[:i].rolling(3).mean()
        # endog['values'] = smooth(endog['values'], 3)
        # endog['values'] = sm.nonparametric.lowess(endog['values'], endog.index, frac=0.1)
        mod = sm.tsa.SARIMAX(endog, order=(1, 0, 0), trend='c')
        # Estimate the parameters
        res = mod.fit()

        fig, ax = plt.subplots(figsize=(15, 5))
        endog.plot(ax=ax, marker='^')
        # print(res.summary)
        # fcast = res.get_prediction(start=len(endog)-3, end=len(endog)+3).summary_frame()
        fcast = res.get_forecast(steps=15).summary_frame()
        fcast['mean'].plot(ax=ax, style='r--', marker='x')
        ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)

        # result = seasonal_decompose(df, model='multiplicative', period=3)
        # plt.rc("figure", figsize=(32,8))
        # result.plot()
        # plt.savefig(f'resources/figs/fig{idx:05d}.png')
        plt.show()
