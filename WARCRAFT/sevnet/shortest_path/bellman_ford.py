"""
Implementation of the Bellman-Ford algorithm.
"""

import torch

from sevnet.utils import check_graph_input, find_path, get_edges


def bellman_ford(graph, source=0):
    """
    Determine the shortest path from `source` to every other node in the graph
    `graph`. The weight/length of each path is also returned.

    Parameters
    ----------
    graph : torch.Tensor (n, n)
        The graph to search expressed as an (nxn) weighted adjacency matrix.
    source : int, optional
        The starting node of the paths.

    Returns
    -------
    tuple : (dict, dict)
        The first element is a dictionary where the value is the cost
        associated with the shortest path through `graph` from `source` to key.
        The second element is also a dictionary where the value is the actual
        shortest path from `source` to key.

    See Also
    --------
    For more information on the Bellman-Ford algorithm:
        https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm.

    Examples
    --------
    >>> import torch
    >>> from sevnet import bellman_ford
    >>> graph = torch.tensor([[0.0, 1.0, 10.0, 0.0],
    ...                       [0.0, 0.0, 0.0, 1.0],
    ...                       [0.0, 0.0, 0.0, -10.0],
    ...                       [0.0, 0.0, 0.0, 0.0]])
    >>> dist, path = bellman_ford(graph, 0)
    >>> dist
    {0: 0.0, 1: tensor(1.), 2: tensor(10.), 3: tensor(0.)}
    """
    check_graph_input(graph)

    n = graph.shape[0]
    edges = get_edges(graph)

    prev_node = dict([(i, None) for i in range(n)])
    dist_from_source = dict([(i, torch.tensor([float("inf")]))
                             for i in range(n)])

    dist_from_source[source] = 0.0

    for _ in range(n - 1):
        for e in edges:
            u, v = e[0].item(), e[1].item()
            w = graph[u, v]
            if (dist_from_source[u] + w) < dist_from_source[v]:
                dist_from_source[v] = dist_from_source[u] + w
                prev_node[v] = u

    for e in edges:
        u, v = e[0].item(), e[1].item()
        w = graph[u, v]
        if (dist_from_source[u] + w) < dist_from_source[v]:
            raise ValueError("`graph` contains a negative-weight cycle.")

    paths = {}
    for i in range(n):
        paths[i] = find_path(prev_node, source, i)

    return (dist_from_source, paths)
