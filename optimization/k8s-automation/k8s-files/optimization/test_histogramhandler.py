import unittest
import histogramhandler as hh


class TestHistogramHandler(unittest.TestCase):

    def load_data(self):
        data = {}
        with open("requests.histogram") as raw_data:
            for row in raw_data:
                key, value = row.split()
                data[key] = value

        return data

    def test_split_uri(self):
        self.assertEqual(1, len([item for item in hh.split_uri("/acmeair-webapp/rest/api/bookings/byuser/uid3@email.com") if not item is None ]))
        self.assertEqual(2, len([item for item in hh.split_uri("/acmeair-webapp/rest/api/login/logout?login=uid3%40email.com") if not item is None ]))
        self.assertEqual(1, len([item for item in hh.split_uri("/") if not item is None ]))
        self.assertEqual(1, len([item for item in hh.split_uri("") if not item is None ]))

    def test_split_path(self):
        self.assertListEqual(["acmeair-webapp","rest","api","bookings","byuser","uid3@email.com"], hh.split_path("/acmeair-webapp/rest/api/bookings/byuser/uid3@email.com"))
        self.assertListEqual(["acmeair-webapp","rest","api","login","logout?login=uid3%40email.com"], hh.split_path("/acmeair-webapp/rest/api/login/logout?login=uid3%40email.com"))
        self.assertListEqual([""], hh.split_path("/"))
        self.assertListEqual([], hh.split_path(""))

    def test_fuzzy_string_comparation(self):
        self.assertGreaterEqual(hh.fuzzy_string_comparation("test", "test"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid3@email.com", "uid3@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid2@email.com", "uid3@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid3@email.com", "uid30@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid3@email.com", "uid31@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid3@email.com", "uid312@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("uid3@email.com", "uid3120@email.com"), .8)
        self.assertGreaterEqual(hh.fuzzy_string_comparation("casa", "bola"), 1)


    def test_something(self):
        self.load_data()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
