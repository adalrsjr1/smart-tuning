import unittest
from ..dataaccess import MongoAccessLayer


class TestMongoClient(unittest.TestCase):
    def test_mongo_instantiation(self):
        self.assertNotEqual(MongoAccessLayer('localhost', 27017, 'admin'), False)


if __name__ == '__main__':
    unittest.main()
