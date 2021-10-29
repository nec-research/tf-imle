"""
Implementation of Kruskal's algorithm.
"""

import torch

from sevnet.utils import check_graph_input, extract_min
from sevnet.utils import DisjointSet


def kruskal(graph):
    """
    Determine a minimum spanning tree (MST) for the given graph `graph`.

    Parameters
    ----------
    graph : torch.Tensor (n, n)
        The graph to compute the MST for expressed as an (nxn) weighted
        adjacency matrix.

    Returns
    -------
    tuple : (list, float)
        The first element of the tuple is a list of the edges of the MST, and
        the second element is the weight of the MST.

    See Also
    --------
    For more information on Kruskal's algorithm:
        https://en.wikipedia.org/wiki/Kruskal%27s_algorithm.

    Examples
    --------
    >>> import torch
    >>> from sevnet import kruskal
    >>> G = torch.tensor([[0, 5, 10],
    ...                   [5, 0, 1],
    ...                   [10, 1, 0]])
    >>> kruskal(graph=G)
    ([(1, 2), (0, 1)], 6)
    """
    check_graph_input(graph=graph)

    n = graph.shape[0]
    ds = DisjointSet(xs=[i for i in range(n)])
    edges = list(map(lambda x: tuple(x), (graph != 0.0).nonzero().tolist()))
    weights = dict([(e, graph[e].item()) for e in edges])

    tree = []
    while len(tree) < (n-1):
        e = extract_min(weights, edges)
        edges.remove(e)
        u, v = e
        if ds.find(u) != ds.find(v):
            tree.append(e)
            ds.union(u, v)

    weight = sum([graph[e] for e in tree]).item()

    return tree, weight
