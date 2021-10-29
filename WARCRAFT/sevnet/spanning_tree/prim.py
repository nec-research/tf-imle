"""
Implementation of Prim's algorithm.
"""

import torch

from sevnet.utils import check_graph_input, extract_min


def prim(graph):
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
    For more information on Prim's algorithm:
        https://en.wikipedia.org/wiki/Prim%27s_algorithm.

    Examples
    --------
    >>> import torch
    >>> from sevnet import prim
    >>> G = torch.tensor([[0, 5, 10],
    ...                   [5, 0, 1],
    ...                   [10, 1, 0]])
    >>> prim(graph=G)
    ([(0, 1), (1, 2)], 6)
    """
    check_graph_input(graph=graph)

    n = graph.shape[0]
    root = 0

    nodes_outside_cloud = [i for i in range(n)]
    dist_from_cloud = dict([(i, torch.tensor([float("inf")]))
                             for i in nodes_outside_cloud])
    best_edge = dict([(i, None) for i in nodes_outside_cloud])

    dist_from_cloud[root] = 0.0
    tree = []
    while nodes_outside_cloud:
        nearest_node = extract_min(dist_from_cloud, nodes_outside_cloud)
        nodes_outside_cloud.remove(nearest_node)

        if best_edge[nearest_node] is not None:
            tree.append(best_edge[nearest_node])

        for node, weight in enumerate(graph[nearest_node, :]):
            if weight != 0.0:
                if weight < dist_from_cloud[node]:
                    dist_from_cloud[node] = weight
                    best_edge[node] = (nearest_node, node)

    weight = sum([graph[e] for e in tree]).item()

    return tree, weight
