import heapq
import pandas as pd
import re
from Levenshtein import StringMatcher


class Node():
    def __init__(self, key:str, value:float):
        self.key = key
        self.value = value
        self.is_clone = False

        self.children = []

    

    def is_endpoint(self):
        return 0 == len(self.children) or self.value > 0

    def __lt__(self, other):
        return self.key < other.key

    def __eq__(self, other):
        return self.key == other.key

    def __repr__(self):
        return f'({self.key}, {self.value}, e:{self.is_endpoint()}, c:{self.is_clone})'

    def clone(self, mark=True):
        node = Node(self.key, self.value)
        node.is_clone = mark

        for child in self.children:
            node.children.append(child.clone(mark))

        return node


def split_uri(uri:str) -> tuple:
    uri_splitted = uri.split('?')

    path = uri_splitted[0]
    query = uri_splitted[1] if len(uri_splitted) > 1 else None

    return path, query

def split_path(path:str) -> list:
    if path:
        path_splitted = path.split('/')
        path_splitted[0] = '/'
        return path_splitted
    return []

def group_data(keys, values):
    root = Node("/", 0)

    for key, value in zip(keys, values):
        insert(key, value, root)

def insert(path:list, value:float, node:Node, threshould=0):

    if len(path) > 1:
        for child in node.children:
            if fuzzy_string_comparation(child.key, path[1], threshould):
                return insert(path[1:], value, child, threshould)
        child = Node(path[1], 0)
        heapq.heappush(node.children, child)
        return insert(path[1:], value, child, threshould)

    node.value += value
    return node

def compare_trees(tree1, tree2):
    if tree1 != tree2:
        return False

    stack = []
    for child1 in tree1.children:
        if not child1 in tree2.children:
            return False
        else:
            stack.append(child1)

    while stack:
        index = tree2.children.index(stack.pop())
        return compare_trees(child1, tree2.children[index])

    return True

def fuzzy_string_comparation(u:str, v:str, threshould=0):
    smaller_str = min(len(u), len(v))

    diff = StringMatcher.distance(u, v)

    return diff / smaller_str <= threshould

def print_tree(node:Node, space=0):
    print('\t' * (space + 1), node)
    for child in node.children:
        print_tree(child, space=space+1)

def merge_trees(node1:Node, node2:Node) -> Node:
    new_node = node1.clone(mark=False)

    aux = [node2]

    while aux:
        item = aux.pop(0)

        insert(item.)

def expand_trees(root1:Node, root2:Node, merge=True):

    aux1 = [root1]
    aux2 = [root2]
    while aux1 or aux2:
        _aux1 = []
        _aux2 = []

        level1, level2 = [], []

        # transversing
        _item1 = None
        if aux1:
            _item1 = aux1.pop(0)
            for child in _item1.children:
                level2.append(child)
                _aux1.append(child)

        _item2 = None
        if aux2:
            _item2 = aux2.pop(0)
            for child in _item2.children:

                level1.append(child)
                _aux2.append(child)
        # end transversing

        # copying
        for item in level1:
            equals = False
            for child in _item1.children:
                if child.key == item.key:
                    equals = True
            if not equals:
                new_item = item.clone()
                if not merge:
                    new_item.value = 0
                heapq.heappush(_item1.children, new_item)

        for item in level2:
            equals = False
            for child in _item2.children:
                if child.key == item.key:
                    equals = True
            if not equals:
                new_item = item.clone()
                if not merge:
                    new_item.value = 0
                heapq.heappush(_item2.children, new_item)
        # end copying

def tree_to_list(node:Node, l:list):
    for child in node.children:
        tree_to_list(child, l)

    if node.is_endpoint():
        if node.is_clone:
            l.append(0)
        else:
            l.append(node.value)

    return l

def pandas_to_tree(serie:pd.Series, similarity_threshould=0.8) -> Node:
    regex = "(?<=\")(.*)(?=\")"

    root = Node('/', 0)
    for index, item in zip(serie.index, serie):
        path = re.search(regex, index).group()
        path = split_path(split_uri(path)[0])
        insert(path, item, root, similarity_threshould)

    return root

def create_comparable_histograms(urls1:list, urls2:list):
    root1 = Node('/', 0)
    root2 = Node('/', 0)

    _len1 = len(urls1)
    _len2 = len(urls2)
    length = max(len(urls1), len(urls2))

    for i in range(length):
        if i < _len1:
            insert(split_path(split_path(urls1[i])[0]))
        if i < _len2:
            insert(split_path(split_path(urls2[i])[0]))

    nroot1 = expand_trees(root2, root1.clone(mark=False))
    nroot2 = expand_trees(root1, root2.clone(mark=False))

    histogram1, histogram2 = [], []
    tree_to_list(nroot1, histogram1)
    tree_to_list(nroot2, histogram2)

    return histogram1, histogram2