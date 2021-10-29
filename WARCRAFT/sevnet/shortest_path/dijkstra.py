"""
Implementation of Dijkstra's algorithm.
"""

import torch

from sevnet.utils import check_graph_input, extract_min, find_path


def dijkstra(graph, source, target):
    """
    Determine the shortest path, which begins at the given source node `source`
    and terminates at the given target node `target`, through the given graph
    `graph`.

    Parameters
    ----------
    graph : torch.Tensor (n, n)
        The graph to search expressed as an (nxn) weighted adjacency matrix.
    source : int
        The starting node of the path.
    target : int
        The terminating node of the path.

    Returns
    -------
    tuple : (torch.Tensor (1,), list)
        The first element is the cost associated with the shortest path through
        `graph` from `source` to `target`. The second element is the actual
        shortest path from `source` to `target`.

    See Also
    --------
    For more information on Dijkstra's algorithm:
        https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm.

    Examples
    --------
    >>> import torch
    >>> from sevnet import dijkstra
    >>> graph = torch.tensor([[0.0, 1.0, 10.0, 0.0],
    ...                       [0.0, 0.0, 0.0, 1.0],
    ...                       [0.0, 0.0, 0.0, 10.0],
    ...                       [0.0, 0.0, 0.0, 0.0]])
    >>> dijkstra(graph, 0, 3)
    (tensor(2.), [0, 1, 3])
    """
    check_graph_input(graph)
    if not (graph >= 0.0).all():
        raise ValueError("`graph` must have non-negative edge weights")

    remaining_nodes = [n for n in range(graph.shape[0])]
    dist_from_source = dict([(n, torch.tensor([float("inf")]))
                             for n in remaining_nodes])
    prev_node = dict([(n, None) for n in remaining_nodes])

    dist_from_source[source] = 0.0

    while remaining_nodes:
        nearest_node = extract_min(dist_from_source, remaining_nodes)
        remaining_nodes.remove(nearest_node)

        if nearest_node == target:
            break

        for node, weight in enumerate(graph[nearest_node, :]):
            if graph[nearest_node, node] > 0.0:
                curr_dist_from_source = dist_from_source[node]
                alt_dist_from_source = dist_from_source[nearest_node] + weight

                if alt_dist_from_source < curr_dist_from_source:
                    dist_from_source[node] = alt_dist_from_source
                    prev_node[node] = nearest_node

    path = find_path(prev_node, source, target)

    return (dist_from_source[target], path)
