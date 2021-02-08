from __future__ import annotations

import datetime
import heapq
import logging
import math
import time
import typing

import optuna

import config
from bayesian import BayesianDTO, EmptyBayesianDTO
from controllers.searchspace import SearchSpaceContext
from models.configuration import Configuration, EmptyConfiguration, LastConfig
from models.instance import Instance
from sampler import Metric
from util.stats import RunningStats

logger = logging.getLogger(config.PLANNER_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class Planner:
    def __init__(self, production: Instance, training: Instance, ctx: SearchSpaceContext, k: int, ratio: float = 1, when_try=1):
        self._date = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.training = training
        self.production = production
        self.ctx = ctx
        self.k = k
        self.ratio = ratio
        self.when_try = when_try

        self.heap1: list[Configuration] = []
        self.heap2: list[Configuration] = []

        self._iteration = 0
        self._first_iteration = True

    @property
    def iteration(self):
        return self._iteration

    def reinforcement_iterations(self):
        return int(round(self.k * self.ratio))

    def save_trace(self, reinforcement: typing.Union[str, bool] = True, best: list[dict] = None):
        if best is None:
            best = [{}]

        logger.info(f'saving tuning trace')
        if not config.ping(config.MONGO_ADDR, config.MONGO_PORT):
            logger.warning(f'cannot save logging -- mongo unable at {config.MONGO_ADDR}:{config.MONGO_PORT}')
            return None
        db = config.mongo()[config.MONGO_DB]
        collection = db[f'trace-{self._date}']

        try:
            collection.insert_one({
                'iteration': self.iteration,
                'best': best,
                'params_importance': {k:v for k,v in optuna.importance.get_param_importances(self.ctx.model.study, evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator()).items()},
                'reinforcement': reinforcement,
                'production': self.production.serialize(),
                'training': self.training.serialize(),
            })
        except Exception:
            logger.exception('error when saving data')
        pass

    def __next__(self) -> (Configuration, bool):
        return self.iterate()

    def iterate(self) -> (Configuration, bool):
        def restart_if_poor_perf(instance: Instance):
            logger.info(
                f'checking if {instance.name} need restart -- score:{instance.configuration.score} in mean:{instance.configuration.mean():.2f}:{instance.configuration.stddev():.2f}')
            # !!!! always minimization -- so if objective is too large (if negative close to 0) so restart !!!!
            if instance.configuration.score == 0 or instance.configuration.score > (instance.configuration.median() + instance.configuration.stddev()):
                logger.warning(
                    f'[{self.iteration}] poor perf [perf:{instance.configuration.score} > mean:{instance.configuration.mean():.2f}:{instance.configuration.stddev():.2f}] at {instance.name} -- restarting')
                instance.restart()

            # logger.info(
            #     f'checking if {instance.name} need restart -- score:{instance.configuration.score} in median:{instance.configuration.median():.2f}')
            # # !!!! always minimization -- so if objective is too large (if negative close to 0) so restart !!!!
            # if instance.configuration.score == 0 or instance.configuration.score > instance.configuration.median():
            #     logger.warning(
            #         f'[{self.iteration}] poor perf [perf:{instance.configuration.score} > median{instance.configuration.median()}] at {instance.name} -- restarting')
            #     instance.restart()

        end_of_tuning: bool = False
        logger.info(f'[{self.iteration}] iteration')
        config_to_apply = self.ctx.get_from_engine()

        if isinstance(config_to_apply, EmptyConfiguration):
            # enqueueing a empty DTO to avoid starvation into Bayesian engine
            self.ctx.put_into_engine(EmptyBayesianDTO())
            # returning an EmptyConfiguration to notify that there is no Bayesian engine running
            return config_to_apply, end_of_tuning

        if isinstance(config_to_apply, LastConfig):
            end_of_tuning = True

        self.training.configuration = config_to_apply
        logger.debug(
            f'setting new config into training "{self.training.configuration.name}":{self.training.configuration.data}')

        t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
        logger.debug(f'sampling metrics')
        logger.debug(f'[t] {t_metric.serialize()}')
        logger.debug(f'[p] {p_metric.serialize()}')

        if self._first_iteration:
            # initialize trials with the default configuration set to production replica
            # no metrics into this config
            self.production.set_default_config(p_metric)
            self._first_iteration = False

        self.production.update_configuration_score(p_metric)
        self.training.update_configuration_score(t_metric)
        logger.debug(f'updating scores')
        logger.debug(f'[t] {self.training.configuration}')
        logger.debug(f'[p] {self.production.configuration}')

        # update score of current sample at bayesian core
        if not end_of_tuning:
            self.ctx.put_into_engine(BayesianDTO(metric=t_metric, workload_classification=''))

        self.update_heap(self.heap1, self.production.configuration)
        self.update_heap(self.heap1, self.training.configuration)
        logger.debug(f'2-phase heaps')
        logger.debug(f'heap1: {self.heap1}')
        logger.debug(f'heap2: {self.heap2}')

        restart_if_poor_perf(self.production)

        best: Configuration = self.best_configuration()
        logger.debug(f'best: {best}')

        self.save_trace(best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
        if end_of_tuning or \
                (
                        best.name != self.production.configuration.name
                        and self.iteration >= self.k
                        and self.iteration % self.when_try == 0
                ):

            # ensure that only the first best config will be applied after K iterations
            # all other will be applied as soon as they pop up

            self.training.configuration = best
            curr_best: Configuration = best
            # ensure if the selected config is realy the best running it n times at training replica
            for i in range(self.reinforcement_iterations()):
                t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
                logger.debug(f'[{i}] sampling metrics dry run')
                logger.debug(f'[t] {t_metric.serialize()}')
                logger.debug(f'[p] {p_metric.serialize()}')

                self.production.update_configuration_score(p_metric)
                self.training.update_configuration_score(t_metric)

                self.update_heap(self.heap1, self.production.configuration)
                self.update_heap(self.heap1, self.training.configuration)

                self.save_trace(reinforcement='prod!=train',
                                best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])

                logger.info(f'[experimenting] old_best: {curr_best}')
                logger.info(f'                new_best: {self.best_configuration()}')

                restart_if_poor_perf(self.production)
                restart_if_poor_perf(self.training)

            logger.info(f'[p]: {self.production.configuration.name}')
            logger.info(f'[t]: {self.training.configuration.name}')

            logger.debug(f'is train.median better than prod.median? '
                         f'{self.training.configuration.median() < self.production.configuration.median()}')
            if curr_best.name != self.production.configuration.name \
                    and self.training.configuration.median() < self.production.configuration.median():
                # makes prod.config == train.config iff teh best config previous selectec remains the best
                logger.info(f'making prod.config == train.config')
                logger.debug(f'config to reinforce: {curr_best.name}:{curr_best.data}')

                old_config = self.production.configuration
                self.production.configuration = curr_best
                self.training.configuration = curr_best
                logger.info(f'[p]: {self.production.configuration.name}')
                logger.info(f'[t]: {self.production.configuration.name}')

                for i in range(self.reinforcement_iterations()):
                    logger.info(f' *** {i}th reinforcing iteration ***')
                    # reinforcing best config
                    t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
                    avg_metric = (p_metric + t_metric) / 2
                    logger.debug(f'sampling metrics')
                    logger.debug(f'[t] {t_metric.serialize()}')
                    logger.debug(f'[p] {p_metric.serialize()}')
                    logger.debug(f'[a] {avg_metric.serialize()}')

                    self.production.update_configuration_score(avg_metric)
                    self.training.update_configuration_score(avg_metric)
                    logger.debug(f'updating scores')
                    logger.debug(f'[t] {self.training.configuration}')
                    logger.debug(f'[p] {self.production.configuration}')

                    self.update_heap(self.heap1, self.production.configuration)
                    logger.debug(f'2-phase heaps')
                    logger.debug(f'heap1: {self.heap1}')
                    logger.debug(f'heap2: {self.heap2}')

                    self.save_trace(reinforcement='prod==train',
                                    best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])

                    restart_if_poor_perf(self.production)
                    restart_if_poor_perf(self.training)

                self.heap1 = []

                # update if curr config is different than prior
                if self.production.configuration.name != old_config.name:
                    # comparision using median
                    if self.production.configuration.median() <= old_config.median():
                        logger.info(f'keep reiforced config:{self.production.configuration}')
                        self.update_heap(self.heap2, self.production.configuration)
                    else:
                        logger.info(f'reverting to config:{old_config}')
                        self.production.configuration = old_config
                        self.update_heap(self.heap1, self.production.configuration)
                else:
                    logger.info(f'keeping reinforced config:{self.production.configuration}')
                    self.update_heap(self.heap2, self.production.configuration)

                logger.debug(f'heap1: {self.heap1}')
                logger.debug(f'heap2: {self.heap2}')
                self.when_try = 1 #try at least k training iterations before attempting to promote a config
                self._iteration = 0

        self._iteration += 1
        # returns best config applyed to production
        return self.production.configuration, end_of_tuning


    def best_configuration(self, n=1, return_array=False) -> typing.Union[Configuration, list[Configuration]]:
        if n == 0:
            logger.warning('n must be > 0, setting 1')
            n = 1
        tmp = list(set(self.heap1 + self.heap2))
        heapq.heapify(tmp)
        # nsmallest returns a list, so returns its head
        if n > 1 or return_array:
            # return heapq.nsmallest(n, best_concat)
            return heapq.nsmallest(n, tmp)
        # return heapq.nsmallest(n, best_concat)[0]
        return heapq.nsmallest(n, tmp)[0]

    def update_heap(self, heap: list, configuration: Configuration):
        def _update_heap(_heap: list, _configuration: Configuration, to_add=False):
            c: Configuration
            for i, c in enumerate(_heap):
                # avoid repeated configs
                if c.name == _configuration.name:
                    logger.debug(f'[heap] updating old config:{_heap[i]}')
                    logger.debug(f'[heap] updating new config:{_configuration}')
                    _heap[i] = _configuration
                    # TODO: optimize this doing a single loop -- O(n^2) -> O(n)
                    heapq.heapify(_heap)
                    return
            if to_add:
                heapq.heappush(_heap, _configuration)

        logger.debug('loopping through heap1')
        _update_heap(self.heap1, configuration, to_add=heap is self.heap1)
        logger.debug('loopping through heap2')
        _update_heap(self.heap2, configuration, to_add=heap is self.heap2)

    def wait_for_metrics(self, interval: int, n_sampling_subintervals: int = 3, logging_subinterval: float = 0.2):
        """
        wait for metrics in a given interval (s) logging at every interval * subinterval (s)

        interval: int value for wait
        n_sampling_subintervals: splits interval into n subintervals and check in sampling at each
        subinterval is worth for keeping instances running or not
        subterval: frequence of logging, default at every 20% of interval. Automatically normalize
        values between 0 and 1 if out of this range

        returns:
            production and training metrics
        """

        # safety checking for logging subinterval
        if logging_subinterval < 0:
            logging_subinterval = 0
        elif logging_subinterval > 1:
            logging_subinterval = 1

        t_metric = Metric.zero()
        p_metric = Metric.zero()
        logger.debug(f' *** waiting {(interval * n_sampling_subintervals):.2f}s *** ')
        t_running_stats = RunningStats()
        p_running_stats = RunningStats()
        for i in range(3):
            logger.info(f'[{i}] waiting {interval:.2f}s before sampling metrics')
            elapsed = 0
            now = time.time()
            while elapsed < interval:
                # waiting 20% of time before logging
                time.sleep(math.ceil(interval * logging_subinterval))
                elapsed = time.time() - now
                logger.info(
                    f'\t|- elapsed:{elapsed:.2f} < sleeptime:{interval:.2f}')

            t_metric = self.training.metrics()
            p_metric = self.production.metrics()
            t_running_stats.push(t_metric.objective())
            p_running_stats.push(p_metric.objective())
            logger.info(
                f'\t \\- prod_mean:{p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} prod_median:{p_running_stats.median()}')
            logger.info(
                f'\t \\- train_mean:{t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f} train_median: {t_running_stats.median()}')

            if i == 0:
                continue

            # TODO: Is this fail fast working as expected?
            if (t_running_stats.mean() + t_running_stats.standard_deviation()) > (
                    p_running_stats.mean() + p_running_stats.standard_deviation()):
                logger.warning(
                    f'\t |- [T] fail fast -- prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                break

            if (p_running_stats.mean() + p_running_stats.standard_deviation()) > (
                    t_running_stats.mean() - t_running_stats.standard_deviation()):
                logger.warning(
                    f'\t |- [P] fail fast -- prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                break

        return t_metric, p_metric
