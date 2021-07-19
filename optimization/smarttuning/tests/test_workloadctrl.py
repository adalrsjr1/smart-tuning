import unittest
from unittest import TestCase

from controllers import workloadctrl
from models.workload import Workload


class Test(TestCase):
    def test_get_mostly_workload(self):
        self.assertEqual((Workload('workload_0', 0), 0), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 3))
        workloadctrl.workload_counter(Workload('workload_0', data=0), 3)
        self.assertEqual((Workload('workload_0', 0), 1), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 3))
        workloadctrl.workload_counter(Workload('workload_0', data=0), 3)
        self.assertEqual((Workload('workload_0', 0), 3), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 3))
        workloadctrl.workload_counter(Workload('workload_0', data=0), 3)
        self.assertEqual((Workload('workload_0', 0), 7), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 3))

        self.assertEqual((Workload('workload_0', 0), 0), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 0))
        self.assertEqual((Workload('workload_0', 0), 1), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 1))
        self.assertEqual((Workload('workload_0', 0), 3), workloadctrl.get_mostly_workload(Workload('workload_0', data=0), 2))


if __name__ == '__main__':
    unittest.main()
