"""
Implementation of various utilities used throughout sevnet.
"""

import torch


def check_graph_input(graph):
    """
    Check that the given graph input `graph` is a valid graph to analyze.
    """
    if not isinstance(graph, torch.Tensor):
        raise TypeError("`graph` must be a torch.Tensor.")
    if len(graph.shape) != 2:
        raise ValueError("`graph` must be represented as an adjacency matrix.")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("`graph` must be a square adjacency matrix.")
    return None


def extract_min(_dict, _list):
    """
    Extract the key from the dictionary `_dict` that has the minimum value and
    that is in the list `_list`.
    """
    min = torch.tensor([float("inf")])
    for key in _list:
        if _dict[key] < min:
            min = _dict[key]
            min_index = key
    return min_index


def find_path(prev_node, source, target):
    """
    Find the shortest path from `source` to `target` by back tracking through
    the `prev_node` dictionary.
    """
    path = []
    u = target
    if (prev_node[u] is not None) or (u == source):
        while u is not None:
            path.insert(0, u)
            u = prev_node[u]
    return path


def get_edges(graph):
    """
    Extract from `graph` a torch.Tensor (|E|, 2) containing the edges of
    `graph` as index pairs.
    """
    edges = (graph != 0.0).nonzero()
    return edges


class DisjointSet:
    """
    Implementation of a simple disjoint-set data structure.

    Parameters
    ----------
    xs : list of hashable
        The elements of the disjoint sets.
    """

    def __init__(self, xs):
        self._parent = dict([(x, x) for x in xs])

    def union(self, x1, x2):
        """
        Merge the sets containing `x1` and `x2`.
        """
        root1 = self.find(x=x1)
        root2 = self.find(x=x2)
        self._parent[root1] = root2
        return None

    def find(self, x):
        """
        Find the set, i.e. the root, containing `x`.
        """
        if self._parent[x] == x:
            return x
        else:
            return self.find(self._parent[x])
