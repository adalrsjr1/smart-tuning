import unittest
from unittest import TestCase
from unittest.mock import Mock, MagicMock

from models.configuration import Configuration, EmptyConfiguration
from sampler import Metric
from smarttuning2.test_planner.mock_searchspace import MockSearchSpace, MockTrial


class TestConfiguration(TestCase):
    def test_instantiate(self):
        trial = MockTrial.create_trial()
        c = Configuration.new(trial=trial,
                              search_space=MockSearchSpace.new())

        self.assertIsInstance(c, Configuration)

    def test_instantiate_empty(self):
        c = Configuration.empty_config()
        self.assertIsInstance(c, EmptyConfiguration)

        self.assertEqual('2b94ec6adabb54cf6314edace78da81e', c.name)
        self.assertEqual(-1, c.uid)
        self.assertDictEqual({}, c.data)
        self.assertIsNotNone(c.trial)

        with self.assertRaises(NotImplementedError):
            c.score = 0

    def test_update_score(self):
        trial = MockTrial.create_trial()
        c = Configuration.new(trial=trial, search_space=MockSearchSpace.new())

        metric = Metric(to_eval='100')
        c.score = metric.objective()
        self.assertEqual(c.score, trial.value)
        self.assertEqual(c.score, 100)
        self.assertEqual(trial.value, 100)

        new_metric = Metric(to_eval='0')
        c.score = new_metric.objective()
        self.assertEqual(c.score, new_metric.objective())
        self.assertEqual(c.mean, (metric.objective() + new_metric.objective()) / 2)
        self.assertEqual(c.median, (metric.objective() + new_metric.objective()) / 2)

    def test_initial_config(self):
        search_space = MockSearchSpace.new()
        m1 = MagicMock()
        m1.name = 'manifest-a'
        m1.get_current_config = MagicMock(return_value={
            'a1': 5,
            'a2': 5.5,
        })

        m2 = MagicMock()
        m2.name = 'manifest-b'
        m2.get_current_config = MagicMock(return_value={
            'b1': 'b2_y',
        })

        m3 = MagicMock()
        m3.name = 'manifest-c'
        m3.get_current_config = MagicMock(return_value={
            'c1': True,
            'c2': 55,
        })
        search_space.manifests = [m1, m2, m3]
        c = Configuration.running_config(search_space=search_space)


if __name__ == '__main__':
    unittest.main()
