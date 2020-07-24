import unittest
from app import sanitize_dict_to_save, sanitize_token


class AppTestCase(unittest.TestCase):
    def test_sanitize_data_to_save(self):
        data = {}
        sanitize_dict_to_save(data)

    def test_sanitize_token(self):
        token = '.$\\'
        self.assertEqual('\_\$\\\\', sanitize_token(token))

        token2 = '.$\\.$\\'
        self.assertEqual('\_\$\\\\\_\$\\\\', sanitize_token(token2))

        token3 = {'.$\\.$\\': {'.$\\': '.'}}
        self.assertDictEqual({'\_\$\\\\\_\$\\\\': {'\_\$\\\\': '\_'}}, sanitize_token(token3))

        token4 = ['.$\\.$\\','.$\\', '.']
        self.assertListEqual(['\_\$\\\\\_\$\\\\', '\_\$\\\\', '\_'], sanitize_token(token4))

        class Tdd:
            def __init__(self):
                self.a = 0
                self.b = 'a.b'

        self.assertDictEqual({'a': 0, 'b': 'a\_b'}, sanitize_token(Tdd()))


if __name__ == '__main__':
    unittest.main()
