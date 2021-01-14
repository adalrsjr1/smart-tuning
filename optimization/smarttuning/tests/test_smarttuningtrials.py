import unittest
from controllers.searchspacemodel import SearchSpaceModel, ConfigMapSearhSpaceModel
from models.smartttuningtrials import SmartTuningTrials
from models.configuration import DefaultConfiguration
from mock_searchspace import mock_acmeair_search_space
from hyperopt import Trials, space_eval


def new_smarttunint_trials():
    trials = Trials()

    ss = SearchSpaceModel(mock_acmeair_search_space())

    st_trials = SmartTuningTrials(ss.search_space(), trials)

    return st_trials


class MyTestCase(unittest.TestCase):
    def test_smarttuningtrials_clean_space(self):
        st_trials = new_smarttunint_trials()
        c = {"MONGO_MAX_CONNECTIONS": 10, "cpu": 1, "-Xtune:virtualized": True,
             "container_support": 0}
        new_space = st_trials.clean_space(c)
        self.assertIn('acmeair-config-app', new_space)
        self.assertIn('acmeair-config-jvm', new_space)
        self.assertIn('acmeair-service', new_space)
        self.assertIn('MONGO_MAX_CONNECTIONS', new_space['acmeair-config-app'])
        self.assertIn('-Xtune:virtualized', new_space['acmeair-config-jvm'])
        self.assertIn('container_support', new_space['acmeair-config-jvm'])
        self.assertIn('cpu', new_space['acmeair-service'])

        self.assertDictEqual(
            {'acmeair-config-app': {'MONGO_MAX_CONNECTIONS': 10},
             'acmeair-config-jvm': {'-Xtune:virtualized': False, "container_support": "-XX:+UseContainerSupport"},
             'acmeair-service': {'cpu': 4}}, space_eval(new_space, c))

    def test_smarttuningtrials_serialize(self):
        st_trials = new_smarttunint_trials()
        st_trials.add_default_config(DefaultConfiguration({
            "MONGO_MAX_CONNECTION": 100,
        }, {
            "MONGO_MAX_CONNECTION": 100
        }, st_trials), score=10)

        self.assertListEqual([{'tid': 0, 'params': {'MONGO_MAX_CONNECTION': 100}, 'loss': 10, 'status': 'ok'}],
                             st_trials.serialize())


if __name__ == '__main__':
    unittest.main()
