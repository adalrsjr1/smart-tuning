from __future__ import annotations

import heapq
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from numbers import Number
from typing import Union

import optuna
from kubernetes.client import ApiException
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import BaseTrial
from prometheus_client import Info, Gauge

import config
from controllers import workloadctrl
from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration
from models.instance import Instance
from models.workload import Workload
from util import prommetrics
from util.stats import RunningStats


class DriverSession:
    def __init__(self, workload: Workload, search_space: SearchSpaceModel, bayesian: bool = True,
                 n_startup_trial: int = 10, n_ei_candidates: int = 24, seed: int = 0):
        self.__workload: Workload = workload
        self.__search_space: SearchSpaceModel = search_space
        self.__nursery = []
        self.__tenured = []

        if bayesian:
            sampler = TPESampler(
                n_startup_trials=n_startup_trial,
                n_ei_candidates=n_ei_candidates,
                seed=seed
            )
        else:
            sampler = RandomSampler(seed=seed)

        self.__study: optuna.Study = optuna.create_study(sampler=sampler, study_name=self.workload.name)

        self.logger = logging.getLogger(f'{type(self).__name__}.smarttuning.ibm')
        self.logger.setLevel(logging.INFO)

    @property
    def study(self):
        return self.__study

    @property
    def workload(self) -> Workload:
        return self.__workload

    @property
    def nursery(self) -> list[Configuration]:
        return self.__nursery

    @property
    def tenured(self) -> list[Configuration]:
        return self.__tenured

    @property
    def search_space(self) -> SearchSpaceModel:
        return self.__search_space

    def n_pruned(self) -> int:
        return len([trial for trial in self.study.trials if trial.state == 'PRUNED'])

    def ask(self) -> Configuration:
        trial = self.study.ask()
        self.logger.info(f'sampling configuration')
        memo = {}
        for name, tunable in self.search_space.tunables().items():
            print(name, tunable.new_sample(trial, memo))

        self.logger.info(f'asking {trial.params} to study.{self.study.study_name}')
        return Configuration(trial, self.search_space.sample(trial, True))

    def tell(self, configuration: Configuration, state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE,
             status: str = 'training'):
        """
        status: training; reinforcement; probation; tune
        """
        self.logger.info(f'telling to study.{self.study.study_name} the config:{configuration}')
        trial = configuration.trial
        if 'training' == status:
            if optuna.trial.TrialState.COMPLETE != state:
                # try again if PRUNED/FAIL/WAITING
                self.study.tell(trial, state=state)
                self.enqueue(configuration)
            else:
                # save config into nursery
                self.study.tell(trial, configuration.final_score(), state=state)
                self.update_heap(self.__nursery, configuration)

    @staticmethod
    def __single_or_multi_best(n: int, items: Union[list[Configuration], Configuration]):
        if len(items) == 0:
            return Configuration.empty_config()

        if n > 1:
            return heapq.nsmallest(n, items)
        smallest = heapq.nsmallest(n, items)[0]
        return smallest

    def best(self, n: int = 1) -> Union[list[Configuration], Configuration]:
        cfg: Configuration
        trials = {trial.number: trial.state for trial in self.study.trials}
        tmp = [cfg for cfg in set(self.nursery + self.tenured) if
               trials[cfg.trial.number] == optuna.trial.TrialState.COMPLETE]
        heapq.heapify(tmp)

        return DriverSession.__single_or_multi_best(n, tmp)

    def best_nursery(self, n: int = 1) -> Union[list[Configuration], Configuration]:
        trials = {trial.number: trial.state for trial in self.study.trials}
        tenured_names = [tenured.name for tenured in self.tenured]
        tmp = [cfg for cfg in self.nursery if trials[cfg.trial.number] == optuna.trial.TrialState.COMPLETE and
               cfg.name not in tenured_names]
        heapq.heapify(tmp)

        return DriverSession.__single_or_multi_best(n, tmp)

    def best_tenured(self, n: int = 1) -> Union[list[Configuration], Configuration]:
        trials = {trial.number: trial.state for trial in self.study.trials}
        tmp = [cfg for cfg in self.tenured if trials[cfg.trial.number] == optuna.trial.TrialState.COMPLETE]
        heapq.heapify(tmp)

        return DriverSession.__single_or_multi_best(n, tmp)

    def add(self, configuration: Configuration):
        """
        status: training; reinforcement; probation
        """
        trial = optuna.create_trial(
            params=configuration.trial.params,
            distributions=configuration.trial.distributions,
            value=configuration.final_score(),
            state=optuna.trial.TrialState.COMPLETE
        )
        self.study.add_trial(trial)
        self.update_heap(self.__nursery, configuration)

    def enqueue(self, configuration: Configuration):
        """
        status: training; reinforcement; probation
        """
        self.study.enqueue_trial(configuration.trial.params)

    def promote_to_tenured(self, configuration: Configuration):
        if configuration:
            self.logger.info(f'promoting config:{configuration} to tenured')
            self.update_heap(self.__tenured, configuration)
            self.__nursery = []

    @staticmethod
    def update_heap(heap: list, configuration: Configuration):
        if configuration in heap:
            heapq.heapify(heap)
        else:
            heapq.heappush(heap, configuration)


class IterationDriver:
    def __init__(self, workload: Workload, search_space: SearchSpaceModel, production: Instance, training: Instance,
                 max_global_iterations: int = 50, max_local_iterations: int = 10, max_reinforcement_iterations: int = 3,
                 max_probation_iterations: int = 3, sampling_interval: float = 60, n_sampling_subintervals: int = 3,
                 logging_subinterval: float = 0.2, fail_fast: bool = False, uid: str = '', bayesian=True,
                 n_startup_trial=10, n_ei_candidates=24, seed=0):

        self.__curr_best = None
        self.__workload = workload
        self.__search_space = search_space

        self.__production = production
        self.__training = training

        self.__session: DriverSession = DriverSession(workload=workload,
                                                      search_space=self.search_space,
                                                      bayesian=bayesian,
                                                      n_startup_trial=n_startup_trial,
                                                      n_ei_candidates=n_ei_candidates,
                                                      seed=seed)

        self.__max_global_iterations = max_global_iterations
        self.__max_local_iterations = max_local_iterations
        assert self.max_local_iterations > 0, f'max_local_iterations must be > 0'
        self.__max_reinforcement_iterations = max_reinforcement_iterations
        self.__max_probation_iterations = max_probation_iterations

        self.__global_iteration = 0
        self.__local_iteration = 0
        self.__reinforcement_iteration = 0
        self.__probation_iteration = 0

        self.__sampling_interval = sampling_interval
        self.__n_sampling_subintervals = n_sampling_subintervals
        self.__logging_subinterval = logging_subinterval
        self.__fail_fast = fail_fast

        self.__curr_iteration: Iteration = None
        self.__last_iteration: Iteration = None
        self.__lookahead = TrainingIteration
        self.__prev_lookahead = TrainingIteration

        self.__extra_it = 0
        self.__last_prod: Configuration = self.production.configuration

        self.logger = logging.getLogger(f'{self.workload().name}.{type(self).__name__.lower()}.smarttuning.ibm')
        self.logger.setLevel(logging.INFO)

        self.__uid = uid

    @property
    def uid(self):
        return self.__uid

    @property
    def global_iteration(self):
        return self.__global_iteration

    @property
    def local_iteration(self):
        return self.__local_iteration

    @property
    def reinforcement_iteration(self):
        return self.__reinforcement_iteration

    @property
    def probation_iteration(self):
        return self.__probation_iteration

    @property
    def max_global_iterations(self):
        return self.__max_global_iterations

    @property
    def max_local_iterations(self):
        return self.__max_local_iterations

    @property
    def max_reinforcement_iterations(self):
        return self.__max_reinforcement_iterations

    @property
    def max_probation_iterations(self):
        return self.__max_probation_iterations

    @property
    def sampling_interval(self):
        return self.__sampling_interval

    @property
    def n_sampling_subintervals(self):
        return self.__n_sampling_subintervals

    @property
    def logging_subinterval(self):
        return self.__logging_subinterval

    @property
    def fail_fast(self):
        return self.__fail_fast

    @property
    def production(self) -> Instance:
        return self.__production

    @property
    def training(self) -> Instance:
        return self.__training

    @property
    def search_space(self):
        return self.__search_space

    @property
    def curr_iteration(self) -> Iteration:
        return self.__curr_iteration

    @property
    def lookahead(self):
        return self.__lookahead

    @lookahead.setter
    def lookahead(self, next_it_type):
        self.__prev_lookahead = self.lookahead
        self.__lookahead = next_it_type

    @property
    def prev_lookahead(self):
        return self.__prev_lookahead

    def workload(self) -> Workload:
        return self.__workload

    def curr_workload(self) -> Workload:
        return workloadctrl.workload()
        # try:
        #     client = config.hpaApi()
        #     hpa = client.read_namespaced_horizontal_pod_autoscaler(name=self.production.name,
        #                                                            namespace=self.production.namespace)
        #     status = hpa.status
        #     n_replicas = status.current_replicas
        #
        #     workload = Workload(f'workload_{n_replicas}', data=n_replicas)
        # except ApiException:
        #     self.logger.exception('cannot sample HPA info')
        # finally:
        #     return workload

    @property
    def curr_best(self) -> Configuration:
        if not self.__curr_best:
            self.__curr_best = self.production.configuration
        return self.__curr_best

    @curr_best.setter
    def curr_best(self, configuration: Configuration):
        self.__curr_best = configuration

    def session(self) -> DriverSession:
        return self.__session

    def new_training_it(self, configuration: Configuration = None) -> TrainingIteration:
        return TrainingIteration(self, self.__search_space,
                                 self.production, self.training,
                                 curr_configuration=configuration,
                                 sampling_interval=self.sampling_interval,
                                 n_sampling_subintervals=self.n_sampling_subintervals,
                                 logging_subinterval=self.logging_subinterval,
                                 fail_fast=self.fail_fast)

    def new_reinforcement_it(self, configuration: Configuration) -> ReinforcementIteration:
        return ReinforcementIteration(self, self.__search_space,
                                      self.production, self.training,
                                      curr_configuration=configuration,
                                      sampling_interval=self.sampling_interval,
                                      n_sampling_subintervals=self.n_sampling_subintervals,
                                      logging_subinterval=self.logging_subinterval,
                                      fail_fast=self.fail_fast)

    def new_probation_it(self, configuration: Configuration) -> ProbationIteration:
        return ProbationIteration(self, self.__search_space,
                                  self.production, self.training,
                                  curr_configuration=configuration,
                                  sampling_interval=self.sampling_interval,
                                  n_sampling_subintervals=self.n_sampling_subintervals,
                                  logging_subinterval=self.logging_subinterval,
                                  fail_fast=self.fail_fast)

    def new_tuned_it(self, configuration: Configuration) -> TunedIteration:
        return TunedIteration(self, self.__search_space,
                              self.production, self.training,
                              curr_configuration=configuration,
                              sampling_interval=self.sampling_interval,
                              n_sampling_subintervals=self.n_sampling_subintervals,
                              logging_subinterval=self.logging_subinterval,
                              fail_fast=self.fail_fast)

    def update_trials(self):
        configuration = self.production.configuration
        try:
            self.session().study.get_trials(deepcopy=False)[configuration.uid].value = configuration.trial.value
        except Exception:
            self.logger.exception(f'[t] error to update trial {configuration.trial.number} into {configuration}')

        configuration = self.training.configuration
        try:
            self.session().study.get_trials(deepcopy=False)[configuration.uid].value = configuration.trial.value
        except Exception:
            self.logger.exception(f'[t] error to update trial {configuration.trial.number} into {configuration}')

    def __next__(self):
        if self.global_iteration < self.max_global_iterations + self.session().n_pruned() + self.__extra_it:

            if self.lookahead == TrainingIteration:
                self.handle_training()

            elif self.lookahead == ReinforcementIteration:
                self.handle_reinforcement()

            elif self.lookahead == ProbationIteration:
                self.handle_probation()

            elif self.lookahead == Iteration:
                return self.handle_reset()

        else:
            if type(self.__curr_iteration) not in [ReinforcementIteration, ProbationIteration]:
                self.handle_tuned()
            else:
                self.__extra_it += 1
                return self.__next__()

        self.__global_iteration += 1

        self.save_trace()

        return self.__curr_iteration

    def rollback(self):
        self.logger.debug(f'add last score to cancel this wrong measurement')
        self.training.configuration.score = self.training.configuration.last_score
        self.production.configuration.score = self.production.configuration.last_score

    def handle_training(self):
        self.logger.info(f'training {self.local_iteration}/{self.max_local_iterations}/{self.global_iteration}')
        it = self.new_training_it(configuration=None)
        if not it.iterate():
            self.logger.debug(f'aborting {it}')
            return

        self.__last_iteration = self.curr_iteration

        self.__curr_iteration = it
        self.__local_iteration += 1

        curr_best = self.session().best_nursery()
        if curr_best is Configuration.empty_config():
            curr_best = it.driver.production.configuration

        self.curr_best = curr_best
        self.logger.info(f'curr_best: {self.curr_best}')
        self.__last_prod = self.production.configuration

        if self.local_iteration < self.max_local_iterations:
            self.lookahead = TrainingIteration
        else:
            if self.production.configuration.name == self.curr_best.name or \
                    self.curr_best.final_score() >= self.production.configuration.final_score():
                self.lookahead = TrainingIteration
            else:
                # progress
                self.lookahead = ReinforcementIteration

    def handle_reinforcement(self):
        self.logger.info(
            f'reinforcement {self.reinforcement_iteration}/{self.max_local_iterations}/{self.max_global_iterations}')
        it = self.new_reinforcement_it(configuration=self.curr_best)
        if not it.iterate():
            self.logger.debug(f'aborting {it}')
            return

        self.__last_iteration = self.curr_iteration

        self.__curr_iteration = it

        self.__reinforcement_iteration += 1
        self.__last_prod = self.production.configuration

        if self.reinforcement_iteration < self.max_reinforcement_iterations:
            self.lookahead = ReinforcementIteration
        else:
            if self.reinforcement_iteration >= self.max_reinforcement_iterations:
                if it.configuration.final_score() <= self.production.configuration.final_score():
                    self.lookahead = ProbationIteration
                else:
                    self.lookahead = Iteration

    def handle_probation(self):
        self.logger.info(
            f'probation {self.probation_iteration}/{self.max_local_iterations}/{self.max_global_iterations}')
        it = self.new_probation_it(configuration=self.curr_best)
        if not it.iterate():
            self.logger.debug(f'aborting {it}')
            return
        self.__last_iteration = self.curr_iteration
        it.iterate()
        self.__curr_iteration = it
        self.__probation_iteration += 1
        self.lookahead = ProbationIteration

        if self.probation_iteration < self.max_probation_iterations:
            self.lookahead = ProbationIteration
        else:
            self.lookahead = Iteration

    def handle_reset(self):
        if self.__last_prod.final_score() < self.production.configuration.final_score():
            self.logger.warning(f'reverting production cfg: {self.production.configuration.name} -> '
                                f'{self.__last_prod.name}')
            self.production.configuration = self.__last_prod
        else:
            self.logger.info(f'promoting {self.production.configuration.name} to tenured')
            self.session().promote_to_tenured(self.production.configuration)

            self.logger.info(f'reseting counters {self.global_iteration}/{self.max_global_iterations}')
        self.lookahead = TrainingIteration
        self.__local_iteration = 0
        self.__reinforcement_iteration = 0
        self.__probation_iteration = 0
        self.save_trace(reset=True)
        return self.__next__()

    def handle_tuned(self):
        it = self.new_tuned_it(self.session().best())
        if not it.iterate():
            return
        self.__last_iteration = self.__curr_iteration
        self.__curr_iteration = it

    def expose_prom_metrics(self):
        # objective metric
        prommetrics.gauge_metric({
            'app': self.production.name,
            'metric': 'objective',
        }, f'smarttuning_metric', 'metric', self.production.metrics().objective())

        prommetrics.gauge_metric({
            'app': self.training.name,
            'metric': 'objective',
        }, f'smarttuning_metric', 'metric', self.training.metrics().objective())

        # penalization metric
        prommetrics.gauge_metric({
            'app': self.production.name,
            'metric': 'penalization',
        }, f'smarttuning_metric', 'metric', self.production.metrics().penalization())

        prommetrics.gauge_metric({
            'app': self.training.name,
            'metric': 'penalization',
        }, f'smarttuning_metric', 'metric', self.training.metrics().penalization())

        import re

        def expose_config_knobs(instance: Instance):
            c = instance.configuration
            knobs_labels = dict()
            knobs_labels['app'] = instance.name
            for key, knobs_list in c.data.items():
                knobs_labels['manifest'] = key
                for knob, value in knobs_list.items():
                    knobs_labels['knob'] = knob
                    knobs_labels['value'] = ''
                    if isinstance(value, Number):
                        prommetrics.gauge_metric(knobs_labels, 'smarttuning_knob_value', 'knob value', value)
                    elif isinstance(value, str) and re.compile(r"[-+]?\d*\d+|\.\d+").match(value):
                        prommetrics.gauge_metric(knobs_labels, 'smarttuning_knob_value', 'knob value', float(value))
                    else:
                        knobs_labels['value'] = str(value)
                        prommetrics.gauge_metric(knobs_labels, 'smarttuning_knob_value', 'knob value')

        expose_config_knobs(self.production)
        expose_config_knobs(self.training)

        # config score
        prommetrics.gauge_metric({
            'app': self.training.name,
            'cfg': self.training.configuration.name,
        }, f'smarttuning_config_score', 'configuration score', self.training.configuration.score)

        prommetrics.gauge_metric({
            'app': self.production.name,
            'cfg': self.production.configuration.name,
        }, f'smarttuning_config_score', 'configuration score', self.production.configuration.score)

    def save_trace(self, reset=False):
        self.expose_prom_metrics()

        def sanitize_document(document):
            if isinstance(document, dict):
                memo = {}
                for k, v in document.items():
                    if isinstance(k, str):
                        if '.' in k:
                            new_key = k.replace('.', '_')
                            memo.update({new_key: sanitize_document(v)})
                        else:
                            memo.update({k: sanitize_document(v)})
                    else:
                        memo.update({k: sanitize_document(v)})
                return memo
            elif isinstance(document, list) or isinstance(document, set):
                return [sanitize_document(item) for item in document]
            elif isinstance(document, str):
                return document.replace('.', '_')
            else:
                return document

        if not config.ping(config.MONGO_ADDR, config.MONGO_PORT):
            self.logger.warning(f'cannot save logging -- mongo unreacheable at {config.MONGO_ADDR}:{config.MONGO_PORT}')
            return None
        db = config.mongo()[config.MONGO_DB]
        collection = db[f'trace-{self.uid}']

        local_iteration = {
            TrainingIteration: self.local_iteration,
            ReinforcementIteration: self.reinforcement_iteration,
            ProbationIteration: self.probation_iteration,
            TunedIteration: self.__extra_it,
            None: -1
        }

        trace_to_save = {
            'reset': reset,
            'date': datetime.utcnow().isoformat(),
            'pruned': self.curr_workload().name != self.workload().name,
            'status': type(self.__curr_iteration).__name__,
            'global_iteration': self.global_iteration,
            'iteration': local_iteration.get(type(self.__curr_iteration), -1),
            'ctx_workload': self.workload().serialize(),
            'curr_workload': self.curr_workload().serialize(),
            'best': self.curr_best.serialize(),
            'production': self.production.serialize(),
            'training': self.training.serialize(),
            'trials': [{'uid': c.number, 'value': c.value, 'state': c.state.name} for c in self.session().study.trials],
            'nursery': [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                        for c in heapq.nsmallest(len(self.session().nursery), self.session().nursery)],
            'tenured': [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                        for c in heapq.nsmallest(len(self.session().tenured), self.session().tenured)],
        }

        try:
            self.logger.info(f'saving trace of {self.global_iteration}/{self.max_global_iterations}')
            collection.insert_one(sanitize_document(trace_to_save), bypass_document_validation=True)
        except Exception:
            self.logger.exception(f'error when saving trace {self.global_iteration}/{self.max_global_iterations}')
            from pprint import pprint
            pprint(trace_to_save)


class Iteration(ABC):
    def __init__(self, driver: IterationDriver, search_space: SearchSpaceModel,
                 production: Instance, trainining: Instance, curr_configuration: Configuration = None,
                 sampling_interval: float = 60, n_sampling_subintervals: int = 3, logging_subinterval: float = 0.2,
                 fail_fast: bool = False):

        self.__driver: IterationDriver = driver
        self.__search_space = search_space
        self.__production = production
        self.__training = trainining

        self.__sampling_interval = sampling_interval
        self.__n_sampling_subintervals = n_sampling_subintervals
        self.__logging_subinterval = logging_subinterval
        self.__fail_fast = fail_fast

        self._curr_config: Configuration = curr_configuration

        self.logger = logging.getLogger(f'{self.workload()}.{type(self).__name__}.smarttuning.ibm')
        self.logger.setLevel(logging.DEBUG)

    def __str__(self):
        return type(self).__name__

    @property
    def driver(self):
        return self.__driver

    @property
    def production(self) -> Instance:
        return self.__production

    @property
    def training(self) -> Instance:
        return self.__training

    @property
    def sampling_interval(self):
        return self.__sampling_interval

    @property
    def n_sampling_subintervals(self):
        return self.__n_sampling_subintervals

    @property
    def logging_subinterval(self):
        return self.__logging_subinterval

    @property
    def fail_fast(self):
        return self.__fail_fast

    @property
    def configuration(self) -> Configuration:
        if not self._curr_config:
            self._curr_config = self.driver.session().ask()
            self.logger.debug(f'asking for new config {self._curr_config}')
        return self._curr_config

    def status(self):
        statuses = {
            TrainingIteration: 'training',
            ReinforcementIteration: 'reinforcemente',
            ProbationIteration: 'probation',
            TunedIteration: 'tuned',
        }
        return statuses[type(self)]

    def workload(self) -> Workload:
        return self.driver.workload()

    def curr_workload(self):
        return self.driver.curr_workload()

    def most_workload(self):
        return self.workload()

    def sample(self, trial: BaseTrial) -> Configuration:
        # sample return the correct hierarchy for ConfigMaps
        return Configuration.new(trial=trial, search_space=self.__search_space)

    def sample_best(self) -> Configuration:
        return self.driver.session().best()

    def waiting_for_metrics(self):
        """
        wait for metrics in a given interval (s) logging at every interval * logging_subinterval (s)

        interval: int value for wait
        n_sampling_subintervals: splits interval into n subintervals and check in sampling at each
        subinterval is worth for keeping instances running or not
        subterval: frequence of logging, default at every 20% of interval. Automatically normalize
        values between 0 and 1 if out of this range

        returns:
            production and training metrics
        """

        # TODO: keep counting the ticks for each workload

        # safety checking for logging subinterval
        assert 0 <= self.logging_subinterval <= 1

        self.logger.debug(f' *** waiting {(self.sampling_interval * self.n_sampling_subintervals):.2f}s *** ')
        t_running_stats = RunningStats()
        p_running_stats = RunningStats()

        t_metric = self.__training.metrics()
        p_metric = self.__production.metrics()
        for i in range(self.n_sampling_subintervals):
            self.logger.info(f'[{i}] waiting {self.sampling_interval:.2f}s before sampling metrics')
            elapsed = timedelta()
            now = datetime.utcnow()

            while elapsed.total_seconds() < self.sampling_interval:
                time.sleep(round(self.sampling_interval * self.logging_subinterval))
                elapsed = datetime.utcnow() - now

                self.logger.info(
                    f'\t|- elapsed:{elapsed.total_seconds()} < sleeptime:{self.sampling_interval:.2f} '
                    f'origin_workload:{self.workload()} curr_workload:{self.curr_workload()}')

            t_metric = self.__training.metrics()
            p_metric = self.__production.metrics()

            t_running_stats.push(t_metric.objective())
            p_running_stats.push(p_metric.objective())

            self.logger.info(
                f'\t \\- prod_mean:{p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                f'prod_median:{p_running_stats.median():.2f}')
            self.logger.info(
                f'\t \\- train_mean:{t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f} '
                f'train_median: {t_running_stats.median():.2f}')

            if self.fail_fast:
                # self.logger.warning('FAIL FAST is not implemented yet')
                # TODO: fail fast disabled for mocking workload classification
                if i == 0:
                    continue

                # TODO: Is this fail fast working as expected?
                if self.curr_workload() != self.most_workload():
                    self.logger.warning(f'\t |- [W] fail fast -- {self.most_workload()} -> {self.curr_workload()}')
                    self.logger.warning(f'\t # ')
                    break

                # if (t_running_stats.mean() + t_running_stats.standard_deviation()) > (
                #         p_running_stats.mean() + p_running_stats.standard_deviation()):
                #     self.logger.warning(
                #         f'\t |- [T] fail fast -- '
                #         f'prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                #         f'train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                #     self.logger.warning(f'\t # ')
                #     break
                #
                # if (p_running_stats.mean() + p_running_stats.standard_deviation()) > (
                #         t_running_stats.mean() - t_running_stats.standard_deviation()):
                #     self.logger.warning(
                #         f'\t |- [P] fail fast -- '
                #         f'prod: {p_running_stats.mean():.2f} ± {p_running_stats.standard_deviation():.2f} '
                #         f'train: {t_running_stats.mean():.2f} ± {t_running_stats.standard_deviation():.2f}')
                #     self.logger.warning(f'\t # ')
                #     break

        return t_metric, p_metric

    def progressing(self) -> bool:
        session = self.driver.session()
        if self.curr_workload() == self.most_workload():
            session.tell(self.configuration, status=self.status())
            return True
        else:
            self.logger.debug(f'ctx_workload: {self.workload()} curr_workload: {self.curr_workload()}')
            self.logger.debug(f'enqueue config {self.configuration.name}')
            session.tell(self.configuration, status=self.status(),
                         state=optuna.trial.TrialState.PRUNED)
            # if curr_stage == TrainingIteration:
            #     session.enqueue(self.configuration)
            return False

    @abstractmethod
    def iterate(self, best_config=False, reuse_config=False, config_to_reuse=None) -> bool:
        pass


class TrainingIteration(Iteration):
    def iterate(self, **kwargs) -> bool:
        assert self._curr_config is None, f'__curr_config must be None'
        self.training.configuration = self.configuration

        assert self.production.configuration.name != self.training.configuration.name, \
            f'[t] config {self.production.configuration.name} in prod must be different than train'

        tmetrics, pmetrics = self.waiting_for_metrics()

        self.training.configuration.score = tmetrics.objective()
        self.production.configuration.score = pmetrics.objective()

        self.logger.debug(
            f'training: [p] {self.production.configuration.name}:{self.production.configuration.final_score()} '
            f'[t] {self.training.configuration.name}:{self.training.configuration.final_score()}')

        self.driver.update_trials()
        if not self.progressing():
            self.driver.rollback()
            return False
        return True


class ReinforcementIteration(Iteration):
    def iterate(self, **kwargs) -> bool:
        best_configuration: Configuration = self.configuration
        self.training.configuration = best_configuration

        assert self.production.configuration.name != self.training.configuration.name, \
            f'[r] config {self.production.configuration.name} in prod must be different than train'

        tmetrics, pmetrics = self.waiting_for_metrics()

        self.training.configuration.score = tmetrics.objective()
        self.production.configuration.score = pmetrics.objective()

        self.logger.debug(
            f'reinforcement: [p] {self.production.configuration.name}:{self.production.configuration.final_score()} '
            f'[t] {self.training.configuration.name}:{self.training.configuration.final_score()}')

        self.driver.update_trials()
        if not self.progressing():
            self.driver.rollback()
            return False
        return True


class ProbationIteration(Iteration):
    def iterate(self, **kwargs) -> bool:
        best_configuration: Configuration = self.configuration

        self.training.configuration = best_configuration
        self.production.configuration = best_configuration
        tmetrics, pmetrics = self.waiting_for_metrics()

        self.training.configuration.score = tmetrics.objective()
        self.production.configuration.score = pmetrics.objective()

        self.logger.debug(
            f'probation: [p] {self.production.configuration.name}:{self.production.configuration.final_score()} '
            f'[t] {self.training.configuration.name}:{self.training.configuration.final_score()}')

        self.driver.update_trials()
        if not self.progressing():
            self.driver.rollback()
            return False
        return True


class TunedIteration(Iteration):
    def iterate(self, **kwargs) -> bool:
        # try:
        #     if self.traininig.active:
        #         self.production.max_replicas += 1
        #         self.traininig.shutdown()
        # except Exception:
        #     self.logger.exception(f'exception catch when deleting training: {self.traininig.name}')
        self.production.configuration = self.configuration
        self.training.configuration = self.configuration
        tmetrics, pmetrics = self.waiting_for_metrics()
        self.production.configuration.score = pmetrics.objective()

        if self.curr_workload() != self.most_workload():
            return False
        return True
