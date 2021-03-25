from __future__ import annotations

import datetime
import heapq
import logging
import threading
import time
import types
import typing
from numbers import Number

import math

import config
from controllers.searchspace import SearchSpaceContext
from models.bayesiandto import BayesianDTO, EmptyBayesianDTO
from models.configuration import Configuration, EmptyConfiguration, LastConfig
from models.instance import Instance
from sampler import Metric
from util.stats import RunningStats

logger = logging.getLogger(config.PLANNER_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class Planner:
    global_counter = 0
    curr_workloads: dict[str, str] = {}
    lock = threading.RLock()

    @staticmethod
    def update_workload(key: str, workload: str):
        with Planner.lock:
            Planner.curr_workloads[key] = workload

    @staticmethod
    def get_workload(key: str):
        with Planner.lock:
            workload = Planner.curr_workloads.get(key, '')
        return workload

    def __init__(self, uid: str, production: Instance, training: Instance, ctx: SearchSpaceContext, max_iterations: int,
                 k: int, ratio: float = 1,
                 when_try: int = 1, restart_trigger: float = 1):
        # self._uid = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._uid = uid
        self.training = training
        self.production = production
        self.ctx = ctx
        self.k = k
        self.ratio = ratio
        self.when_try = when_try
        self.restart_trigger = restart_trigger

        self.heap1: list[Configuration] = []
        self.heap2: list[Configuration] = []

        self._iteration = 0
        self._iterations_performed = 0
        self._first_iteration = True
        self._max_iterations = max_iterations

        self._curr_ = self.iterate()
        self._last_ = None
        self._aux_ = None

        self.logger = logging.getLogger(f'{self.ctx.workload}.{config.PLANNER_LOGGER}')
        self.logger.setLevel(config.LOGGING_LEVEL)

    @property
    def iterations_performed(self):
        return self._iterations_performed

    @iterations_performed.setter
    def iterations_performed(self, value):
        self._iterations_performed = value

    @property
    def max_iterations(self):
        return self._max_iterations

    @property
    def iteration(self):
        return self._iteration

    def reinforcement_iterations(self):
        return int(round(self.k * self.ratio))

    def save_trace(self, reinforcement: str = 'none', pruned: bool = False, best: list[dict] = None):
        if best is None:
            best = [{}]

        self.logger.info(f'saving tuning trace into collection: trace-{self._uid}')
        if not config.ping(config.MONGO_ADDR, config.MONGO_PORT):
            self.logger.warning(f'cannot save logging -- mongo unable at {config.MONGO_ADDR}:{config.MONGO_PORT}')
            return None
        db = config.mongo()[config.MONGO_DB]
        collection = db[f'trace-{self._uid}']

        document = self.sanitize_document({
            'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pruned': pruned,
            'ctx_workload': self.ctx.workload,
            'curr_workload': Planner.get_workload(self.production.name),
            'global_iteration': Planner.global_counter,
            'iteration': self.iterations_performed,
            'best': best,
            'reinforcement': reinforcement,
            'production': self.production.serialize(),
            'training': self.training.serialize(),
        })

        try:
            collection.insert_one(document, bypass_document_validation=True)
        except Exception:
            self.logger.exception(f'error when saving data {document}')
            from pprint import pprint
            pprint(document)

    def sanitize_document(self, document):
        if isinstance(document, dict):
            memo = {}
            for k, v in document.items():
                if '.' in k:
                    new_key = k.replace('.', '_')
                    memo.update({new_key: self.sanitize_document(v)})
                else:
                    memo.update({k: self.sanitize_document(v)})
            return memo
        elif isinstance(document, list) or isinstance(document, set):
            return [self.sanitize_document(item) for item in document]
        elif isinstance(document, str):
            return document.replace('.', '_')
        else:
            return document

    def __next__(self) -> typing.Union[Configuration, Number]:
        # aux = self.iterate()
        # last = None
        # cur = None
        # while True:

        try:
            if isinstance(self._curr_, types.GeneratorType):
                self._last_ = self._aux_
                self._aux_ = self._curr_
                self._curr_ = next(self._curr_)
            else:
                self._curr_ = next(self._aux_)
        except StopIteration:
            logger.exception('Stop Iteration -- this is not an error')
            self._curr_ = self._last_
        finally:
            with Planner.lock:
                Planner.global_counter += 1
            return self._curr_ if not isinstance(self._curr_, types.GeneratorType) else None
            # return self._curr_

    def _restart_if_poor_perf(self, instance: Instance):
        self.logger.info(
            f'checking if {instance.name} need restart -- '
            f'score:{instance.configuration.score} in '
            f'mean:{instance.configuration.mean():.2f}:{instance.configuration.stddev():.2f}')

        # !!!! always minimization -- so if objective is too large (if negative close to 0) so restart !!!!
        if instance.configuration.score == 0 or \
                instance.configuration.score > (
                instance.configuration.median() + self.restart_trigger * instance.configuration.stddev()):
            self.logger.warning(
                f'[{self.iteration}] poor perf '
                f'[perf:{instance.configuration.score} > mean:{instance.configuration.mean():.2f}:'
                f'{instance.configuration.stddev():.2f}] at {instance.name} -- restarting')
            instance.restart()

    def iterate(self) -> Configuration:
        i = 0
        while i < self.max_iterations:
            self.iterations_performed = i
            end_of_tuning: bool = False
            self.logger.info(f'{{{self.iterations_performed}/{Planner.global_counter}}} iteration')
            config_to_apply = self.ctx.get_from_engine()
            # unmark configs as pruned
            config_to_apply.restore()

            if isinstance(config_to_apply, EmptyConfiguration):
                # enqueueing a empty DTO to avoid starvation into Bayesian engine
                self.ctx.put_into_engine(EmptyBayesianDTO())
                # returning an EmptyConfiguration to notify that there is no Bayesian engine running
                return config_to_apply, end_of_tuning

            if isinstance(config_to_apply, LastConfig):
                end_of_tuning = True

            self.training.configuration = config_to_apply
            self.logger.debug(
                f'setting new config into training '
                f'"{self.training.configuration.name}":{self.training.configuration.data}')

            t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
            self.logger.debug(f'sampling metrics')
            self.logger.debug(f'[t] {t_metric.serialize()}')
            self.logger.debug(f'[p] {p_metric.serialize()}')

            if self._first_iteration:
                # initialize trials with the default configuration set to production replica
                # no metrics into this config
                self.production.set_default_config(p_metric)
                self._first_iteration = False

            # update score of current sample at bayesian core
            if not end_of_tuning:
                workload_classification = Planner.get_workload(self.production.name)
                dto = BayesianDTO(metric=t_metric, workload_classification=workload_classification)
                self.training.configuration.trial.should_prune()

                # use this lock to avoid BO finish before saving trace
                self.training.configuration.semaphore.acquire()
                self.ctx.put_into_engine(dto)

            if self.training.configuration.was_pruned:
                # TODO: use metrics.ttl() to avoid prune when iteration is almost done
                self.logger.debug('pruning at planner')
                self.save_trace(pruned=True, best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
                # release lock so BO can progress
                self.training.configuration.semaphore.release()
                yield self.production.configuration
                continue
            else:
                # release lock so BO can progress
                self.training.configuration.semaphore.release()
            i = i + 1

            self.production.update_configuration_score(p_metric)
            self.training.update_configuration_score(t_metric)
            self.logger.debug(f'updating scores')
            self.logger.debug(f'[t] {self.training.configuration}')
            self.logger.debug(f'[p] {self.production.configuration}')

            self.update_heap(self.heap1, self.production.configuration)
            self.update_heap(self.heap1, self.training.configuration)

            self.logger.debug(f'2-phase heaps')
            self.logger.debug(f'heap1: {self.heap1}')
            self.logger.debug(f'heap2: {self.heap2}')

            self._restart_if_poor_perf(self.production)

            best: Configuration = self.best_configuration()
            self.logger.debug(f'best: {best}')

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

                yield self.reinforcement(curr_best)

                # for i in range(self.reinforcement_iterations()):
                #     t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
                #     self.logger.debug(f'[{i}] sampling metrics dry run')
                #     self.logger.debug(f'[t] {t_metric.serialize()}')
                #     self.logger.debug(f'[p] {p_metric.serialize()}')
                #
                #     self.production.update_configuration_score(p_metric)
                #     self.training.update_configuration_score(t_metric)
                #
                #     self.update_heap(self.heap1, self.production.configuration)
                #     self.update_heap(self.heap1, self.training.configuration)
                #
                #     self.save_trace(reinforcement='prod!=train',
                #                     best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
                #
                #     self.logger.info(f'[experimenting] old_best: {curr_best}')
                #     self.logger.info(f'                new_best: {self.best_configuration()}')
                #
                #     self._restart_if_poor_perf(self.production)
                #     self._restart_if_poor_perf(self.training)

                self.logger.info(f'[p]: {self.production.configuration.name}')
                self.logger.info(f'[t]: {self.training.configuration.name}')

                self.logger.debug(f'is train.median better than prod.median? '
                                  f'{self.training.configuration.median() < self.production.configuration.median()}')
                if curr_best.name != self.production.configuration.name \
                        and self.training.configuration.median() < self.production.configuration.median():
                    # makes prod.config == train.config iff teh best config previous selectec remains the best
                    self.logger.info(f'making prod.config == train.config')
                    self.logger.debug(f'config to reinforce: {curr_best.name}:{curr_best.data}')

                    old_config = self.production.configuration
                    self.production.configuration = curr_best
                    self.training.configuration = curr_best
                    self.logger.info(f'[p]: {self.production.configuration.name}')
                    self.logger.info(f'[t]: {self.production.configuration.name}')

                    # probation period

                    yield self.probation()

                    # for i in range(self.reinforcement_iterations()):
                    #     self.logger.info(f' *** {i}th reinforcing iteration ***')
                    #     # reinforcing best config
                    #     t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
                    #     avg_metric = (p_metric + t_metric) / 2
                    #     self.logger.debug(f'sampling metrics')
                    #     self.logger.debug(f'[t] {t_metric.serialize()}')
                    #     self.logger.debug(f'[p] {p_metric.serialize()}')
                    #     self.logger.debug(f'[a] {avg_metric.serialize()}')
                    #
                    #     self.production.update_configuration_score(avg_metric)
                    #     self.training.update_configuration_score(avg_metric)
                    #     self.logger.debug(f'updating scores')
                    #     self.logger.debug(f'[t] {self.training.configuration}')
                    #     self.logger.debug(f'[p] {self.production.configuration}')
                    #
                    #     self.update_heap(self.heap1, self.production.configuration)
                    #     self.logger.debug(f'2-phase heaps')
                    #     self.logger.debug(f'heap1: {self.heap1}')
                    #     self.logger.debug(f'heap2: {self.heap2}')
                    #
                    #     self.save_trace(reinforcement='prod==train',
                    #                     best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
                    #
                    #     self._restart_if_poor_perf(self.production)
                    #     self._restart_if_poor_perf(self.training)

                    self.heap1 = []

                    # update if curr config is different than prior
                    if self.production.configuration.name != old_config.name:
                        # comparision using median
                        if self.production.configuration.median() <= old_config.median():
                            self.logger.info(f'keep reiforced config:{self.production.configuration}')
                            self.update_heap(self.heap2, self.production.configuration)
                        else:
                            self.logger.info(f'reverting to config:{old_config}')
                            self.production.configuration = old_config
                            self.update_heap(self.heap1, self.production.configuration)
                    else:
                        self.logger.info(f'keeping reinforced config:{self.production.configuration}')
                        self.update_heap(self.heap2, self.production.configuration)

                    self.logger.debug(f'heap1: {self.heap1}')
                    self.logger.debug(f'heap2: {self.heap2}')
                    self.when_try = 1  # try at least k training iterations before attempting to promote a config
                    self._iteration = 0

            self._iteration += 1
            # returns best config applyed to production
            yield self.production.configuration
        yield self.production.configuration

    def reinforcement(self, curr_best):
        i = 0
        while i < self.reinforcement_iterations():
            # unmark configs as pruned
            self.training.configuration.restore()
            t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
            self.logger.debug(f'[{i}] sampling metrics dry run')
            self.logger.debug(f'[t] {t_metric.serialize()}')
            self.logger.debug(f'[p] {p_metric.serialize()}')

            if Planner.get_workload(self.production.name) != self.ctx.workload:
                self.training.configuration.prune()
                # TODO: use metrics.ttl() to avoid prune when iteration is almost done
                self.logger.warning(f'pruning reinforcement config: {self.training.configuration.name}')
                self.save_trace(reinforcement='reinforcement',
                                pruned=True,
                                best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
                yield self.production.configuration
                continue

            self.production.update_configuration_score(p_metric)
            self.training.update_configuration_score(t_metric)

            self.update_heap(self.heap1, self.production.configuration)
            self.update_heap(self.heap1, self.training.configuration)

            self.save_trace(reinforcement='reinforcement',
                            best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])

            self.logger.info(f'[experimenting] old_best: {curr_best}')
            self.logger.info(f'                new_best: {self.best_configuration()}')

            self._restart_if_poor_perf(self.production)
            self._restart_if_poor_perf(self.training)

            i = i + 1
            yield self.production.configuration

    def probation(self) -> bool:
        i = 0
        while i < self.reinforcement_iterations():
            # unmark configs as pruned
            self.training.configuration.restore()
            self.production.configuration.restore()
            self.logger.info(f' *** {i}th reinforcing iteration ***')
            # reinforcing best config
            t_metric, p_metric = self.wait_for_metrics(self.training.default_sample_interval)
            avg_metric = (p_metric + t_metric) / 2
            self.logger.debug(f'sampling metrics')
            self.logger.debug(f'[t] {t_metric.serialize()}')
            self.logger.debug(f'[p] {p_metric.serialize()}')
            self.logger.debug(f'[a] {avg_metric.serialize()}')

            if Planner.get_workload(self.production.name) != self.ctx.workload:
                self.training.configuration.prune()
                # TODO: use metrics.ttl() to avoid prune when iteration is almost done
                self.logger.warning(f'pruning probation config: {self.training.configuration.name}')
                self.save_trace(reinforcement='probation',
                                pruned=True,
                                best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])
                yield self.production.configuration
                continue

            self.production.update_configuration_score(avg_metric)
            self.training.update_configuration_score(avg_metric)
            self.logger.debug(f'updating scores')
            self.logger.debug(f'[t] {self.training.configuration}')
            self.logger.debug(f'[p] {self.production.configuration}')

            self.update_heap(self.heap1, self.production.configuration)
            self.logger.debug(f'2-phase heaps')
            self.logger.debug(f'heap1: {self.heap1}')
            self.logger.debug(f'heap2: {self.heap2}')

            self.save_trace(reinforcement='probation',
                            best=[_best_.serialize() for _best_ in self.best_configuration(n=3)])

            self._restart_if_poor_perf(self.production)
            self._restart_if_poor_perf(self.training)

            i = i + 1
            yield self.production.configuration

    def best_configuration(self, n=1, return_array=False) -> typing.Union[Configuration, list[Configuration]]:
        if n == 0:
            self.logger.warning('n must be > 0, setting 1')
            n = 1
        tmp = [cfg for cfg in set(self.heap1 + self.heap2) if not cfg.was_pruned]
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
                    self.logger.debug(f'[heap] updating old config:{_heap[i]}')
                    self.logger.debug(f'[heap] updating new config:{_configuration}')
                    _heap[i] = _configuration

                    # TODO: optimize this doing a single loop -- O(n^2) -> O(n)
                    heapq.heapify(_heap)
                    return
            if to_add:
                heapq.heappush(_heap, _configuration)

        self.logger.debug('loopping through heap1')
        _update_heap(self.heap1, configuration, to_add=heap is self.heap1)
        self.logger.debug('loopping through heap2')
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
        self.logger.debug(f' *** waiting {(interval * n_sampling_subintervals):.2f}s *** ')
        t_running_stats = RunningStats()
        p_running_stats = RunningStats()
        for i in range(3):
            self.logger.info(f'[{i}] waiting {interval:.2f}s before sampling metrics')
            elapsed = 0
            now = time.time()

            curr_workload = Planner.get_workload(self.production.name)
            while elapsed < interval:
                # waiting 20% of time before logging
                time.sleep(math.ceil(interval * logging_subinterval))
                elapsed = time.time() - now
                self.logger.info(
                    f'\t|- elapsed:{elapsed:.2f} < sleeptime:{interval:.2f} '
                    f'curr_workload:{curr_workload} new_worklaod: {Planner.get_workload(self.production.name)}')

                if curr_workload != Planner.get_workload(self.production.name):
                    self.logger.info(f'\t|- pruned!')
                    self.logger.warning('quick pruning config')
                    return self.training.metrics(), self.production.metrics()

            t_metric = self.training.metrics()
            p_metric = self.production.metrics()
            t_running_stats.push(t_metric.objective())
            p_running_stats.push(p_metric.objective())
            self.logger.info(
                f'\t \\- prod_mean:{p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                f'prod_median:{p_running_stats.median():.2f}')
            self.logger.info(
                f'\t \\- train_mean:{t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f} '
                f'train_median: {t_running_stats.median():.2f}')

            if config.FAIL_FAST:
                # TODO: fail fast disabled for mocking workload classification
                if i == 0:
                    continue

                # TODO: Is this fail fast working as expected?

                if (t_running_stats.mean() + t_running_stats.standard_deviation()) > (
                        p_running_stats.mean() + p_running_stats.standard_deviation()):
                    self.logger.warning(
                        f'\t |- [T] fail fast -- '
                        f'prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                        f'train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                    break

                if (p_running_stats.mean() + p_running_stats.standard_deviation()) > (
                        t_running_stats.mean() - t_running_stats.standard_deviation()):
                    self.logger.warning(
                        f'\t |- [P] fail fast -- '
                        f'prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                        f'train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                    break

        return t_metric, p_metric
