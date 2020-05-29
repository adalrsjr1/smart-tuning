import numpy as np

class Node():
    def __init__(self, key:str, value:float):
        self.key = key
        self.value = value

        self.children = []

def split_uri(uri:str) -> tuple:
    uri_splitted = uri.split('?')

    path = uri_splitted[0]
    query = uri_splitted[1] if len(uri_splitted) > 1 else None

    return path, query

def split_path(path:str) -> list:
    path_splitted = path.split('/')
    return path_splitted[1:] if path_splitted[0] == "" else path_splitted

def group_data(keys, values):
    root = Node("/", 0)

    for key, value in zip(keys, values):
        insert(key, value, root)

def insert(path:list, value:float, node:Node):
    if not node.children:
        node.children.append(Node(path[0], value))
    else:
        for child in node.children:
            if fuzzy_string_comparation(child.key, path[0], ???):
                return insert(path[1:], value, child)
        node.children.append(Node(path[0], value))

def fuzzy_string_comparation(u, v, threshoud):
    if u == v:
        return True

    len_u = len(u)
    len_v = len(v)
    u = np.array([ord(c) for c in u] + [0] * (max(len_u, len_v) - len_u))
    v = np.array([ord(c) for c in v] + [0] * (max(len_u, len_v) - len_v))

    # to implement Levenshtein distance
    # cosine distance
    # return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return np.linalg.norm(u-v)

def create_histogram(keys, values):
    pass