from __future__ import annotations

import heapq
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from numbers import Number
from typing import Union, Optional, Type

import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.trial import BaseTrial

import config
from controllers import workloadctrl
from controllers.searchspacemodel import SearchSpaceModel
from models.configuration import Configuration
from models.instance import Instance
from models.workload import Workload
from util import prommetrics
from util.stats import RunningStats


class DriverSession:
    def __init__(self, workload: Workload, driver: IterationDriver, search_space: SearchSpaceModel,
                 production: Instance, training: Instance,
                 bayesian: bool = True,
                 n_startup_trial: int = 10, n_ei_candidates: int = 24, seed: int = 0):
        self.__workload: Workload = workload
        self.__search_space: SearchSpaceModel = search_space
        self.__nursery = []
        self.__tenured = []
        self.__driver = driver

        if bayesian:
            sampler = TPESampler(
                n_startup_trials=n_startup_trial,
                n_ei_candidates=n_ei_candidates,
                seed=seed
            )
        else:
            sampler = RandomSampler(seed=seed)

        self.__study: optuna.Study = optuna.create_study(sampler=sampler, study_name=self.workload.name)

        self.__production = production
        assert self.__production is not None
        self.__training = training
        assert self.__training is not None

        self.__global_iteration = 0
        self.__local_iteration = 0
        self.__reinforcement_iteration = 0
        self.__probation_iteration = 0

        self.__curr_iteration: Optional[Iteration] = None
        self.__last_iteration: Optional[Iteration] = None
        self.__lookahead = TrainingIteration
        self.__prev_lookahead = TrainingIteration

        self.logger = logging.getLogger(f'{type(self).__name__}.smarttuning.ibm')
        self.logger.setLevel(logging.DEBUG)

    @property
    def production(self) -> Instance:
        return self.__production

    @production.setter
    def production(self, production: Instance):
        self.__production = production

    @property
    def training(self) -> Instance:
        return self.__training

    @training.setter
    def training(self, training: Instance):
        self.__training = training

    @property
    def curr_iteration(self) -> Optional[Iteration]:
        return self.__curr_iteration

    @curr_iteration.setter
    def curr_iteration(self, curr_iteration: Optional[Iteration]):
        self.__curr_iteration = curr_iteration

    @property
    def last_iteration(self) -> Optional[Iteration]:
        return self.__last_iteration

    @last_iteration.setter
    def last_iteration(self, last_iteration: Optional[Iteration]):
        self.__last_iteration = last_iteration

    @property
    def lookahead(self) -> Type[Iteration]:
        return self.__lookahead

    @lookahead.setter
    def lookahead(self, lookahead: Type[Iteration]):
        self.__lookahead = lookahead

    @property
    def prev_lookahead(self) -> Type[Iteration]:
        return self.__prev_lookahead

    @prev_lookahead.setter
    def prev_lookahead(self, prev_lookahead: Type[Iteration]):
        self.__prev_lookahead = prev_lookahead

    @property
    def driver(self) -> IterationDriver:
        return self.__driver

    @property
    def global_iteration(self) -> int:
        return self.__global_iteration

    @global_iteration.setter
    def global_iteration(self, value: int):
        self.__global_iteration = value

    @property
    def local_iteration(self) -> int:
        return self.__local_iteration

    @local_iteration.setter
    def local_iteration(self, value: int):
        self.__local_iteration = value

    @property
    def reinforcement_iteration(self) -> int:
        return self.__reinforcement_iteration

    @reinforcement_iteration.setter
    def reinforcement_iteration(self, value: int):
        self.__reinforcement_iteration = value

    @property
    def probation_iteration(self) -> int:
        return self.__probation_iteration

    @probation_iteration.setter
    def probation_iteration(self, value: int):
        self.__probation_iteration = value

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
        from copy import deepcopy
        new_cfg = Configuration(self.study.trials[-1], data=deepcopy(configuration.data))
        self.update_heap(self.__nursery, new_cfg)

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

    def mock_progress(self):
        """progress only if training"""
        curr_iteration = self.driver.curr_iteration
        if isinstance(curr_iteration, TrainingIteration):
            self.local_iteration += 1

        self.global_iteration += 1


class IterationDriver:
    __all_sessions: dict[str, DriverSession] = {}

    def __init__(self, workload: Workload, search_space: SearchSpaceModel, production: Instance, training: Instance,
                 max_global_iterations: int = 50, max_local_iterations: int = 10, max_reinforcement_iterations: int = 3,
                 max_probation_iterations: int = 3, sampling_interval: float = 60, n_sampling_subintervals: int = 3,
                 logging_subinterval: float = 0.2, fail_fast: bool = False, uid: str = '', bayesian=True,
                 n_startup_trial=10, n_ei_candidates=24, seed=0):

        self.__curr_best = None
        self.__workload = workload
        self.__search_space = search_space

        # self.__production = production
        # self.__training = training

        self.__session: DriverSession = DriverSession(workload=workload,
                                                      driver=self,
                                                      search_space=self.search_space,
                                                      production=production,
                                                      training=training,
                                                      bayesian=bayesian,
                                                      n_startup_trial=n_startup_trial,
                                                      n_ei_candidates=n_ei_candidates,
                                                      seed=seed)

        IterationDriver.__all_sessions[self.__session.workload.name] = self.__session

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

        # moved to driver session
        # self.curr_iteration: Optional[Iteration] = None
        # self.last_iteration: Optional[Iteration] = None
        # self.lookahead = TrainingIteration
        # self.prev_lookahead = TrainingIteration

        self.__extra_it = 0
        self.__last_prod: Configuration = self.production.configuration

        self.logger = logging.getLogger(f'{self.workload().name}.{type(self).__name__.lower()}.smarttuning.ibm')
        self.logger.setLevel(logging.INFO)

        self.__uid = uid

    @property
    def all_sessions(self) -> dict:
        return IterationDriver.__all_sessions

    @property
    def uid(self):
        return self.__uid

    @property
    def global_iteration(self) -> int:
        return self.session().global_iteration

    @global_iteration.setter
    def global_iteration(self, value: int):
        self.session().global_iteration = value

    @property
    def local_iteration(self) -> int:
        return self.session().local_iteration

    @local_iteration.setter
    def local_iteration(self, value: int):
        self.session().local_iteration = value

    @property
    def reinforcement_iteration(self) -> int:
        return self.session().reinforcement_iteration

    @reinforcement_iteration.setter
    def reinforcement_iteration(self, value: int):
        self.session().reinforcement_iteration = value

    @property
    def probation_iteration(self) -> int:
        return self.session().probation_iteration

    @probation_iteration.setter
    def probation_iteration(self, value: int):
        self.session().probation_iteration = value

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
        return self.session().production

    @property
    def training(self) -> Instance:
        return self.session().training

    @property
    def search_space(self):
        return self.__search_space

    @property
    def curr_iteration(self) -> Iteration:
        return self.session().curr_iteration

    @curr_iteration.setter
    def curr_iteration(self, curr_iteration: Iteration):
        self.session().curr_iteration = curr_iteration

    @property
    def lookahead(self) -> Type[Iteration]:
        return self.session().lookahead

    @lookahead.setter
    def lookahead(self, next_it_type: Type[Iteration]):
        self.prev_lookahead = self.lookahead
        self.session().lookahead = next_it_type

    @property
    def prev_lookahead(self) -> Type[Iteration]:
        return self.session().prev_lookahead

    @prev_lookahead.setter
    def prev_lookahead(self, prev_lookahead: Type[Iteration]):
        self.session().prev_lookahead = prev_lookahead

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

    def session(self, workload: Optional[Workload] = None) -> DriverSession:
        if workload:
            return IterationDriver.__all_sessions.get(workload.name)
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
            if type(self.curr_iteration) not in [ReinforcementIteration, ProbationIteration]:
                self.handle_tuned()
            else:
                self.__extra_it += 1
                return self.__next__()

        self.global_iteration += 1

        self.save_trace(self.session())

        prommetrics.gauge_metric({
            'app': self.production.name,
            'cfg': self.production.configuration.name,
            'workload': self.session(),
        }, f'smarttuning_cfg_timeline', 'configuration timeline', 0)

        prommetrics.gauge_metric({
            'app': self.training.name,
            'cfg': self.training.configuration.name,
            'workload': self.session().workload.name,
        }, f'smarttuning_cfg_timeline', 'configuration timeline', 0)

        return self.curr_iteration

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

        self.last_iteration = self.curr_iteration

        self.curr_iteration = it
        self.local_iteration += 1

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

        self.last_iteration = self.curr_iteration

        self.curr_iteration = it

        self.reinforcement_iteration += 1
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
        self.last_iteration = self.curr_iteration
        self.curr_iteration = it
        self.probation_iteration += 1
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
        self.local_iteration = 0
        self.reinforcement_iteration = 0
        self.probation_iteration = 0
        self.save_trace(self.session(), reset=True)
        return self.__next__()

    def handle_tuned(self):
        it = self.new_tuned_it(self.session().best())
        if not it.iterate():
            return
        self.last_iteration = self.curr_iteration
        self.curr_iteration = it

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
            'workload': self.curr_workload()
        }, f'smarttuning_config_score', 'configuration score', self.training.configuration.score)

        prommetrics.gauge_metric({
            'app': self.production.name,
            'cfg': self.production.configuration.name,
            'workload': self.curr_workload(),
        }, f'smarttuning_config_score', 'configuration score', self.production.configuration.score)

    def serialize(self, curr_session: DriverSession) -> dict:
        local_iteration = {
            TrainingIteration: self.local_iteration,
            ReinforcementIteration: self.reinforcement_iteration,
            ProbationIteration: self.probation_iteration,
            TunedIteration: self.__extra_it,
            None: -1
        }

        return {
            'reset': False,
            'date': datetime.utcnow().isoformat(),
            'pruned': self.workload() != curr_session.curr_iteration.mostly_workload() if curr_session.curr_iteration else True,
            'another_session': self.session() is curr_session,
            'status': type(curr_session.curr_iteration).__name__ if curr_session.curr_iteration else None,
            'global_iteration': self.global_iteration,
            'iteration': local_iteration.get(type(curr_session.curr_iteration), -1),
            'ctx_workload': curr_session.workload.serialize(),
            'curr_workload': self.curr_workload().serialize(),
            'mostly_workload': curr_session.curr_iteration.mostly_workload().serialize() if curr_session.curr_iteration else None,
            'workload_counter': {key.name: value for key, value in
                                 workloadctrl.list_workloads(self.n_sampling_subintervals).items()},
            'production': curr_session.production.serialize(),
            'training': curr_session.training.serialize(),
            'trials': [{'uid': c.number, 'value': c.value, 'state': c.state.name} for c in curr_session.study.trials],
            'nursery': [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                        for c in heapq.nsmallest(len(curr_session.nursery), curr_session.nursery)],
            'tenured': [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                        for c in heapq.nsmallest(len(curr_session.tenured), curr_session.tenured)],
            'all_trials': {name: [{'uid': c.number, 'value': c.value, 'state': c.state.name}
                                  for c in session.study.trials]
                           for name, session in self.all_sessions.items()},
            'all_nursery': {name: [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                                   for c in heapq.nsmallest(len(session.nursery), session.nursery)]
                            for name, session in self.all_sessions.items()},
            'all_tenured': {name: [{'name': c.name, 'uid': c.trial.number, 'value': c.trial.value}
                                   for c in heapq.nsmallest(len(session.tenured), session.tenured)]
                            for name, session in self.all_sessions.items()},

        }

    def save_trace(self, session: DriverSession, reset=False):
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

        trace_to_save = self.serialize(session)
        trace_to_save['reset'] = reset

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

        # self.__workload_counter = 0

        self.logger = logging.getLogger(f'{self.workload()}.{type(self).__name__}.smarttuning.ibm')
        self.logger.setLevel(logging.DEBUG)

    def __str__(self):
        return type(self).__name__

    # @property
    # def workload_counter(self) -> int:
    #     return self.__workload_counter
    #
    # @workload_counter.setter
    # def workload_counter(self, value: int):
    #     self.__workload_counter = int(value)

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
        statuses: dict[Type[Iteration]] = {
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

    def count_curr_workload(self):
        workloadctrl.workload_counter(self.curr_workload(), offset=1)

    def mostly_workload(self) -> Workload:
        w, c = workloadctrl.get_mostly_workload(offset=self.n_sampling_subintervals)

        return w
        #
        # if self.workload_counter > 0:
        #     return self.workload()
        # else:
        #     return self.curr_workload()

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

        prommetrics.gauge_metric({
            'app': self.production.name,
            'cfg': self.production.configuration.name,
            'workload': self.driver.session(),
        }, f'smarttuning_cfg_timeline', 'configuration timeline', 1)

        prommetrics.gauge_metric({
            'app': self.training.name,
            'cfg': self.training.configuration.name,
            'workload': self.driver.session().workload.name,
        }, f'smarttuning_cfg_timeline', 'configuration timeline', 1)

        self.logger.debug(f' *** waiting {(self.sampling_interval * self.n_sampling_subintervals):.2f}s *** ')
        t_running_stats = RunningStats()
        p_running_stats = RunningStats()

        t_metric = self.training.metrics()
        p_metric = self.production.metrics()
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

                for name, session in self.driver.all_sessions.items():
                    prommetrics.gauge_metric({
                        'app': self.production.name,
                        'workload': session.workload.name,
                    }, f'smarttuning_workload_timeline', 'workload timeline',
                        int(self.curr_workload() == session.workload))

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

            self.count_curr_workload()
            # if self.curr_workload() == self.workload():
            #     self.workload_counter += 1
            # else:
            #     self.workload_counter += -1

            if self.fail_fast:
                # self.logger.warning('FAIL FAST is not implemented yet')
                # TODO: fail fast disabled for mocking workload classification
                # TODO: reorganize this order, workload checking must be evaluated before than i==0
                if i == 0:
                    continue

                # TODO: Is this fail fast working as expected?
                if self.curr_workload() != self.mostly_workload():
                    self.logger.warning(f'\t |- [W] fail fast -- {self.mostly_workload()} -> {self.curr_workload()}')
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
        mostly_workload = self.mostly_workload()
        if self.workload() == mostly_workload:
            session.tell(self.configuration, status=self.status())
            return True
        else:
            self.logger.debug(f'ctx_workload: {self.workload()} curr_workload: {mostly_workload}')
            self.logger.debug(f'enqueue config {self.configuration.name}')
            session.tell(self.configuration, status=self.status(),
                         state=optuna.trial.TrialState.PRUNED)

            another_session = self.driver.session(mostly_workload)
            if another_session:
                another_session.add(self.configuration)
                another_session.mock_progress()
                self.driver.save_trace(another_session)
            return False

        # session = self.driver.session()
        # if self.curr_workload() == self.workload():
        #     session.tell(self.configuration, status=self.status())
        #     return True
        # else:
        #     self.logger.debug(f'ctx_workload: {self.workload()} curr_workload: {self.curr_workload()}')
        #     self.logger.debug(f'enqueue config {self.configuration.name}')
        #     session.tell(self.configuration, status=self.status(),
        #                  state=optuna.trial.TrialState.PRUNED)
        #     return False

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

        if self.curr_workload() != self.mostly_workload():
            return False
        return True
