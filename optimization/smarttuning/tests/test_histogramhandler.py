import unittest
import histogramhandler as hh
from prometheus_pandas import query
import sampler

class TestHistogramHandler(unittest.TestCase):

    def load_data(self):
        data = {}
        with open("requests.histogram") as raw_data:
            for row in raw_data:
                key, weight = row.split()
                data[key] = weight

        return data

    def test_split_uri(self):
        self.assertEqual(1, len([item for item in hh.split_uri("/acmeair-webapp/rest/api/bookings/byuser/uid3@email.com") if not item is None ]))
        self.assertEqual(2, len([item for item in hh.split_uri("/acmeair-webapp/rest/api/login/logout?login=uid3%40email.com") if not item is None ]))
        self.assertEqual(1, len([item for item in hh.split_uri("/") if not item is None ]))
        self.assertEqual(1, len([item for item in hh.split_uri("") if not item is None ]))

    def test_split_path(self):
        self.assertListEqual(["/","acmeair-webapp","rest","api","bookings","byuser","uid3@email.com"], hh.split_path("/acmeair-webapp/rest/api/bookings/byuser/uid3@email.com"))
        self.assertListEqual(["/","acmeair-webapp","rest","api","login","logout?login=uid3%40email.com"], hh.split_path("/acmeair-webapp/rest/api/login/logout?login=uid3%40email.com"))
        self.assertListEqual(["/", ""], hh.split_path("/"))
        self.assertListEqual([], hh.split_path(""))

    def test_fuzzy_string_comparation(self):
        self.assertTrue(hh.fuzzy_string_comparation("test", "test", 0))
        self.assertTrue(hh.fuzzy_string_comparation("uid3@email.com", "uid3@email.com", 0))
        self.assertTrue(hh.fuzzy_string_comparation("uid2@email.com", "uid3@email.com", .8))
        self.assertTrue(hh.fuzzy_string_comparation("uid3@email.com", "uid30@email.com", .8))
        self.assertTrue(hh.fuzzy_string_comparation("uid3@email.com", "uid31@email.com", .8))
        self.assertTrue(hh.fuzzy_string_comparation("uid3@email.com", "uid312@email.com", .8))
        self.assertTrue(hh.fuzzy_string_comparation("uid3@email.com", "uid3120@email.com", .8))
        self.assertFalse(hh.fuzzy_string_comparation("uid3@email.com", "uid317204@email.com", .35))
        self.assertTrue(hh.fuzzy_string_comparation("uid3", "uid4", .8))


    def test_insert_node_threshould0(self):
        root = hh.Node('/', 0)

        hh.insert(['/', 'a', 'b'], 1, root)
        self.assertListEqual([hh.Node('a', 0)], root.children)
        self.assertListEqual([hh.Node('b', 1)], root.children[0].children)
        hh.insert(['/', 'b', 'c'], 1, root)
        self.assertListEqual([hh.Node('a', 0), hh.Node('b', 0)], root.children)
        self.assertListEqual([hh.Node('c', 1)], root.children[1].children)
        hh.insert(['/', 'b', 'a'], 1, root)
        self.assertListEqual([hh.Node('a', 0), hh.Node('b', 0)], root.children)
        self.assertListEqual([hh.Node('a', 1), hh.Node('c', 1)], root.children[1].children)
        hh.insert(['/', 'a'], 1, root)
        self.assertListEqual([hh.Node('a', 1), hh.Node('b', 0)], root.children)
        hh.print_tree(root)

    def test_insert_node_threshould1(self):
        root = hh.Node('/', 0)

        hh.insert(['/', 'a', 'b'], 1, root, 1)
        self.assertListEqual([hh.Node('a', 0)], root.children)
        self.assertListEqual([hh.Node('b', 1)], root.children[0].children)
        self.assertEqual(1, root.children[0].children[0].weight)

        hh.print_tree(root)

        hh.insert(['/', 'b', 'c'], 1, root, 1)
        self.assertListEqual([hh.Node('a', 0)], root.children)
        self.assertListEqual([hh.Node('b', 2)], root.children[0].children)
        self.assertEqual(2, root.children[0].children[0].weight)

        hh.print_tree(root)

        hh.insert(['/', 'b', 'a'], 1, root, 1)
        hh.print_tree(root)
        self.assertListEqual([hh.Node('a', 0)], root.children)
        self.assertListEqual([hh.Node('b', 3)], root.children[0].children)
        self.assertEqual(3, root.children[0].children[0].weight)

        hh.print_tree(root)

        hh.insert(['/', 'a'], 1, root, 1)
        self.assertListEqual([hh.Node('a', 1)], root.children)
        self.assertListEqual([hh.Node('b', 3)], root.children[0].children)
        self.assertEqual(3, root.children[0].children[0].weight)

        hh.print_tree(root)

    def test_insert_node_threshould0_8(self):
        root = hh.Node('/', 0)
        threshould = 0.8

        hh.insert(['/'], 1, root, threshould)

        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byuser/uid3@email.com'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byuser/uid2@email.com'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byuser/uid37@email.com'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byuser/uid356@email.com'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byuser/uid3241@email.com'), 1, root, threshould)
        hh.insert(['/'], 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byid/uid3'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byid/uid2'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byid/uid30'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byid/uid300'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/bookings/byid/uid3001'), 1, root, threshould)

        hh.insert(hh.split_path('/acmeair-webapp/rest/api/byid/uid3/bookings'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/byid/uid2/bookings'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/byid/uid30/bookings'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/byid/uid300/bookings'), 1, root, threshould)
        hh.insert(hh.split_path('/acmeair-webapp/rest/api/byid/uid3001/bookings'), 1, root, threshould)

        hh.print_tree(root)

    def test_compare_tree(self):
        root1 = hh.Node('', 0)
        hh.insert(hh.split_path('/a0/b0/c0'), 0, root1)
        hh.insert(hh.split_path('/a1/b1'), 0, root1)
        hh.insert(hh.split_path('/a2'), 0, root1)

        root2 = hh.Node('', 0)
        hh.insert(hh.split_path('/a0/b0/c0'), 0, root2)
        hh.insert(hh.split_path('/a1/b1'), 0, root2)
        hh.insert(hh.split_path('/a2'), 0, root2)

        result, _ = root1.compare_subtree(root2)
        self.assertTrue(result)

        root3 = hh.Node('/',0)
        result, _ = root1.compare_subtree(root3)
        self.assertFalse(result)

        root4 = hh.Node('',0)
        hh.insert(hh.split_path('/a0/b0/c2'), 0, root4)
        hh.insert(hh.split_path('/a1/b1'), 0, root4)
        hh.insert(hh.split_path('/a2'), 0, root4)

        result, _ = root1.compare_subtree(root4)
        print(root1.compare_subtree(root4))
        self.assertFalse(result)

        root5 = hh.Node('', 0)
        hh.insert(hh.split_path('/a0/b0/'), 0, root5)
        hh.insert(hh.split_path('/a1/b1'), 0, root5)
        hh.insert(hh.split_path('/a2'), 0, root5)

        result, _ = root1.compare_subtree(root5)
        print(root1.compare_subtree(root5))
        self.assertFalse(result)

    def test_clone(self):
        node = hh.Node('x',7)
        clone = node.clone()
        self.assertEqual(node.key, clone.key)
        self.assertEqual(node.weight, clone.weight)

        node.children.append(hh.Node('a',1))

        b = hh.Node('b', 2)
        m = hh.Node('m', 10)
        m.children.append(hh.Node('p', 20))
        b.children.append(m)
        m.children.append(hh.Node('n', 15))
        node.children.append(b)
        node.children.append(hh.Node('c',3))

        clone = node.clone()
        for item1, item2 in zip(node.children, clone.children):

            if 'b' == item1.key:
                for b1, b2 in zip(item1.children, item2.children):
                    print('b')
                    self.assertEqual(b1.key, b2.key)
                    self.assertEqual(b1.weight, b2.weight)
                    if 'm' == b1.key:
                        for m1, m2 in zip(b1.children, b2.children):
                            print('m')
                            self.assertEqual(m1.key, m2.key)
                            self.assertEqual(m1.weight, m2.weight)

            self.assertEqual(item1.key, item2.key)
            self.assertEqual(item1.weight, item2.weight)


        hh.print_tree(node)
        hh.print_tree(clone)
    def test_node_in_node(self):

        root = hh.Node('a', 1)
        root.children.append(hh.Node('b',2))
        root.children.append(hh.Node('c',3))
        root.children.append(hh.Node('d',4))

        self.assertTrue(hh.Node('b',2) in root.children)
        self.assertTrue(hh.Node('b',3) in root.children)
        self.assertFalse(hh.Node('x',0) in root.children)

    def test_compare_tree_T(self):
        root = hh.Node('a', 1)
        b = hh.Node('b', 2)
        m = hh.Node('m', 10)
        m.children.append(hh.Node('p', 20))
        b.children.append(m)
        b.children.append(hh.Node('n', 15))
        root.children.append(b)
        node = hh.Node('y', 3)
        node.children.append(hh.Node('z', 4))
        root.children.append(node)
        root.children.append(hh.Node('d', 4))

        self.assertTrue(hh.compare_trees(root, root.clone(mark=False)))

    def test_compare_tree_F(self):
        root1 = hh.Node('a', 1)
        root1.children.append(hh.Node('b', 2))
        root1.children.append(hh.Node('c', 3))
        root1.children.append(hh.Node('d', 4))

        root2 = hh.Node('a', 1)
        b = hh.Node('b', 2)
        m = hh.Node('m', 10)
        m.children.append(hh.Node('p', 20))
        b.children.append(m)
        b.children.append(hh.Node('n', 15))
        root2.children.append(b)
        node = hh.Node('y', 3)
        node.children.append(hh.Node('z', 4))
        root2.children.append(node)
        root2.children.append(hh.Node('d', 4))

        self.assertFalse(hh.compare_trees(root1, root2))

    def test_insert_node(self):
        root1 = hh.Node('/', 0)
        hh.insert(hh.split_path('/a/b'), 2, root1)
        hh.insert(hh.split_path('/a/c'), 3, root1)
        hh.insert(hh.split_path('/a/d'), 4, root1)

        root2 = hh.Node('/', 1)
        hh.insert(hh.split_path('/a/b'), 3, root2)
        hh.insert(hh.split_path('/a/c'), 4, root2)
        hh.insert(hh.split_path('/a/d'), 5, root2)

        nroot1 = root1.clone(mark=False)
        hh.expand_trees(root2, nroot1)
        self.assertTrue(hh.compare_trees(root1, root2))
        self.assertTrue(hh.compare_trees(nroot1, root1))
        self.assertTrue(hh.compare_trees(nroot1, root2))


        root3 = hh.Node('/', 0)
        hh.insert(hh.split_path('/a'), 1, root3)
        hh.insert(hh.split_path('/a/b'), 2, root3)
        hh.insert(hh.split_path('/a/b/m'), 10, root3)
        hh.insert(hh.split_path('/a/b/m/p'), 20, root3)
        hh.insert(hh.split_path('/a/y'), 3, root3)
        hh.insert(hh.split_path('/a/y/z'), 4, root3)
        hh.insert(hh.split_path('/a/d'), 4, root3)

        root4 = hh.Node('/', 0)
        a = hh.Node('a', 1)
        root4.children.append(a)
        b = hh.Node('b', 2)
        m = hh.Node('m', 10)
        m.children.append(hh.Node('p', 20))
        b.children.append(m)
        b.children.append(hh.Node('n', 15))
        a.children.append(b)
        node = hh.Node('y', 3)
        node.children.append(hh.Node('z', 4))
        a.children.append(node)
        a.children.append(hh.Node('d', 4))

        self.assertTrue(hh.compare_trees(root3, root4))
        self.assertFalse(hh.compare_trees(root1, root3))

    def test_tree_to_list(self):
        root = hh.Node('/', 100)
        hh.insert(hh.split_path('/a'), 1, root)
        hh.insert(hh.split_path('/a/b'), 2, root)
        hh.insert(hh.split_path('/a/b/m'), 10, root)
        hh.insert(hh.split_path('/a/b/m/p'), 20, root)
        hh.insert(hh.split_path('/a/y'), 3, root)
        hh.insert(hh.split_path('/a/y/z'), 4, root)
        hh.insert(hh.split_path('/a/d'), 4, root)

        l = hh.tree_to_list(root, [])
        self.assertListEqual([20, 10, 2, 4, 3, 4, 1,100], l)

    def test_pandasseries_to_tree(self):
        prometheus = query.Prometheus(f'http://localhost:30090')
        hist = sampler.workload('acmeair-tuning.*', 600)
        # print(hist.result())
        hh.pandas_to_tree(hist.result())

if __name__ == '__main__':
    unittest.main()
