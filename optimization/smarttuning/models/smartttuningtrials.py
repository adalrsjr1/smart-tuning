from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from hyperopt import Trials, space_eval, STATUS_OK, JOB_STATE_DONE

import config

# workaround to fix circular dependency
# https://www.stefaanlippens.net/circular-imports-type-hints-python.html
if TYPE_CHECKING:
    from models.configuration import Configuration

logger = logging.getLogger(config.SMARTTUNING_TRIALS_LOGGER)
logger.setLevel(config.LOGGING_LEVEL)


class SmartTuningTrials:
    def __init__(self, space: dict, trials: Trials):
        self._data: dict[str, Configuration] = {}
        self._space = space
        self._trials = trials

    def clean_space(self, to_eval: dict) -> dict:
        new_space = {}
        """
         {
            'manifest_name': {
                'tunable1': 'v1',
                'tunable2': 'v2',
                'tunable3': 'v3',
            },
            'another_manifest': {
                [0, 1]
            }
        """
        for k, v in self._space.items():
            for manifest_k, manifest_v in v.items():
                if manifest_k in to_eval:
                    if k in new_space:
                        new_space[k].update({manifest_k:manifest_v})
                    else:
                        new_space[k] = {manifest_k:manifest_v}

        return new_space


    def serialize(self) -> list[dict]:
        documents = []
        try:
            for i, trial in enumerate(self.wrapped_trials.trials):
                to_eval = {k: v[0] for k, v in trial['misc'].get('vals', {}).items()}
                reduced_space = self.clean_space(to_eval)
                # logger.debug(f'[{i}] serializing trials')
                # logger.debug(f'space: {reduced_space}')
                # logger.debug(f'to_eval: {to_eval}')
                # logger.debug(' ****** ')
                params = space_eval(reduced_space, to_eval)
                tid = trial['misc'].get('tid', -1)
                loss = trial['result'].get('loss', float('inf'))
                status = trial['result'].get('status', None)

                documents.append({
                    'tid': tid,
                    'params': params,
                    'loss': loss,
                    'status': status
                })
        except:
            logger.exception(f'error when retrieving trials at it: {i}')

        return documents

    def last_uid(self) -> int:
        if self.wrapped_trials:
            return self.wrapped_trials.trials[-1]['tid']
        logger.warning('no trial available')
        return -1

    @property
    def wrapped_trials(self) -> Trials:
        return self._trials

    def add_default_config(self, configuration: Configuration, score: float):
        self.new_hyperopt_trial_entry(configuration.data, score, STATUS_OK, classification=None)
        self.add_new_configuration(configuration)

    def new_hyperopt_trial_entry(self, configuration: dict, loss: float, status: int,
                                 classification: str = None) -> int:
        rval_spec = [None]
        rval_results = [{
            'loss': loss,
            'classification': classification,
            'eval_time': 0,
            'iterations': 1,
            'status': status}]

        uid = self.last_uid() + 1

        idxs = {}
        vals = {}

        def get_nested_configs(_uid, _configuration, _idxs, _vals):
            for k, v in _configuration.items():
                if isinstance(v, dict):
                    get_nested_configs(_uid, v, _idxs, _vals)
                else:
                    _idxs[k] = [_uid]
                    _vals[k] = [v]

        get_nested_configs(uid, configuration, idxs, vals)

        rval_miscs = [{
            'tid': uid,
            'idxs': idxs,
            'cmd': ('domain_attachment', 'FMinIter_Domain'),
            'vals': vals,
            'workdir': None
        }]

        hyperopt_trial = self.wrapped_trials.new_trial_docs([uid], rval_spec, rval_results, rval_miscs)[0]

        hyperopt_trial['state'] = JOB_STATE_DONE
        self.wrapped_trials.insert_trial_docs([hyperopt_trial])
        self.wrapped_trials.refresh()

        return uid

    def add_new_configuration(self, configuration: Configuration):
        if not self.get_config_by_name(configuration.name):
            self._data[configuration.name] = configuration
        return self._data[configuration.name]

    def get_config_by_name(self, name: str):
        return self._data.get(name, None)

    def get_config_by_data(self, data: dict):
        for value in self._data.values():
            if data == value.data:
                return value

    def update_hyperopt_score(self, configuration: Configuration):
        trial_to_update = self.wrapped_trials.trials[configuration.uid]
        trial_to_update['result']['loss'] = configuration.mean()
        self.wrapped_trials.refresh()


class EmptySmartTuningTrials(SmartTuningTrials):
    def __init__(self):
        super().__init__({}, Trials())
