import unittest
from ..dataaccess import MongoAccessLayer
from ..dataaccess import PrometheusAccessLayer, PrometheusResponse


class TestMongoClient(unittest.TestCase):
    def test_mongo_instantiation(self):
        self.assertNotEqual(MongoAccessLayer('localhost', 27017, 'admin'), False)


class TestPrometheusClient(unittest.TestCase):
    def test_prometheus_instantiation(self):
        PrometheusAccessLayer('localhost', 30090)

    def test_prometheus_query(self):
        p = PrometheusAccessLayer('localhost', 30090)
        r = p.query('up')

        self.assertEqual(r.status(), PrometheusResponse.SUCCESS)

    def test_prometheus_query_data(self):
        p = PrometheusAccessLayer('localhost', 30090)
        r = p.query('up').data()

        self.assertTrue(r)

    def test_prometheus_query_result_type(self):
        p = PrometheusAccessLayer('localhost', 30090)
        response = p.query('up')
        r = response.data().result_type

        self.assertEqual(PrometheusResponse.VECTOR, r)

    def test_prometheus_query_values(self):
        p = PrometheusAccessLayer('localhost', 30090)
        response = p.query('up')
        r = response.value()

        self.assertTrue(len(r) > 0)



if __name__ == '__main__':
    unittest.main()
