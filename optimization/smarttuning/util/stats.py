from __future__ import annotations
from scipy.stats import t
import math
import heapq

class RunningStats:
    """ https://www.johndcook.com/blog/standard_deviation/ """
    def __init__(self, a=0.8):
        self._a = a
        self._m_n = 0
        self._m_oldM = 0 # normal mean
        self._m_newM = 0
        self._m_oldS = 0 # standard deviation
        self._m_newS = 0
        self._m_newE = 0 # exponential mean
        self._m_oldE = 0
        self._curr = 0
        self._last = 0
        self._acc_diff = 0

        self._max = float('-inf')
        self._min = float('inf')

        self._heap = []

    def __add__(self, other: RunningStats):
        rs = RunningStats(a=(self._a + other._a)/2)

        for stat in self._heap + other._heap:
            rs.push(stat)

        # rs.push(other.mean())
        # rs.push(other.max())
        # rs.push(other.min())
        #
        # rs._m_n = self._m_n + other._m_n - 3

        # rs._m_n = self._m_n + other._m_n
        # rs._m_oldM = self._m_oldM + other._m_oldM
        # rs._m_newM = self._m_newM + other._m_newM
        # rs._m_oldS = self._m_oldS + other._m_oldS
        # rs._m_newS = self._m_newS + other._m_newS
        # rs._m_newE = self._m_newE + other._m_newE
        # rs._m_oldE = self._m_oldE + other._m_oldE
        #
        # rs._max = max(self.max(), other.max())
        # rs._min = min(self.min(), other.min())

        return rs

    def curr(self):
        return self._curr

    def last(self):
        return self._last

    def accumulative_diff(self):
        ## goes to 0 in the long run if the values are uniform
        return self._acc_diff

    def slope(self):
        return self.curr() - self.last()

    def push(self, x):
        self._last = self.curr()
        self._curr = x

        self._acc_diff += self.curr() - self.last()

        heapq.heappush(self._heap, x)
        self._m_n += 1

        self._max = max(self._max, x)
        self._min = min(self._min, x)

        if self._m_n == 1:
            self._m_oldE = self._m_newE = self._m_oldM = self._m_newM = x
            self._m_oldS = 0
        else:
            self._m_newE = self._m_oldE * (1-self._a) + x * self._a
            self._m_newM = self._m_oldM + (x - self._m_oldM) / self._m_n
            self._m_newS = self._m_oldS + (x - self._m_oldM) * (x - self._m_newM)

            self._m_oldE = self._m_newE
            self._m_oldM = self._m_newM
            self._m_oldS = self._m_newS

    def n(self):
        return  self._m_n

    def median(self):
        pivot = self.n() // 2 + 1

        _median = float('nan')
        if self.n() > 0:
            if self.n() % 2 == 0:
                _median = sum(heapq.nsmallest(pivot, self._heap)[-2:])/2
            else:
                _median = heapq.nsmallest(pivot, self._heap)[-1]

        return _median

    def mean(self):
        return self._m_newM if self._m_n > 0 else 0

    def exp_mean(self):
        return self._m_newE if self._m_n > 0 else 0

    def variance(self):
        return self._m_newS / (self._m_n - 1) if self._m_n > 1 else 0

    def standard_deviation(self):
        return math.sqrt(self.variance())

    def max(self):
        return self._max

    def min(self):
        return self._min

    def __eq__(self, other: RunningStats):
        accept_null_hyphotesis, _ =  self.t_test(other)
        return accept_null_hyphotesis

    # def __lt__(self, other: RunningStats):
    #     accept_null_hyphotesis, stats = self.t_test(other)
    #     return stats < 0

    # def __le__(self, other: RunningStats):
    #     return self < other or self == other

    # def __gt__(self, other: RunningStats):
    #     accept_null_hyphotesis, stats = self.t_test(other)
    #     return stats > 0

    # def __ge__(self, other: RunningStats):
    #     return not self < other

    def t_test(self, other:RunningStats, alpha=0.05):
        """https://machinelearningmastery.com/how-to-code-the-students-t-test-from-scratch-in-python/"""
        # means
        mean1, mean2 = self.mean(), other.mean()
        # std deviations
        se1, se2 = self.standard_deviation(), other.standard_deviation()
        # standard error on the difference between the samples
        sed = math.sqrt(se1**2.0 + se2**2.0)

        # calculate the t statistic
        t_stat = (mean1 - mean2) / sed
        # degrees of freedom
        df = self.n() + other.n() - 2
        # calculate the critical value
        cv = t.ppf(1.0 - alpha, df)
        # calculate the p-value

        p = (1.0 - t.cdf(abs(t_stat), df)) * 2

        # # interpret via critical value
        # if abs(t_stat) <= cv:
        #     print('Accept null hypothesis that the means are equal.')
        # else:
        #     print('Reject the null hypothesis that the means are equal.')
        # # interpret via p-value
        # if p > alpha:
        #     print('Accept null hypothesis that the means are equal.')
        # else:
        #     print('Reject the null hypothesis that the means are equal.')

        return p > alpha, t_stat

    def serialize(self) -> dict:
        return {
            'median': self.median(),
            'mean': self.mean(),
            'variance': self.variance(),
            'stddev': self.standard_deviation(),
            'min': self.min(),
            'max': self.max(),
            'n': self.n()
        }
