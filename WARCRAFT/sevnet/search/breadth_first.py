"""
Implementation of the breadth first search (BFS) algorithm.
"""

import queue

import torch

from sevnet.utils import check_graph_input


def breadth_first_search(graph, source=0):
    """
    Perform breadth first search of the given graph `graph` starting from the
    node `source`.

    Parameters
    ----------
    graph : torch.Tensor (n, n)
        The graph to search expressed as an (nxn) adjacency matrix.
    source : int, optional
        The starting node of the search.

    Returns
    -------
    reached_nodes : torch.Tensor (k,)
        The nodes that are reached/found via breadth first search starting
        from the node `source`.

    See Also
    --------
    For more information on the breadth first search algorithm:
        https://en.wikipedia.org/wiki/Breadth-first_search.

    Examples
    --------
    >>> import torch
    >>> from sevnet import breadth_first_search
    >>> G = torch.tensor([[False, True, True, False],
    ...                   [False, False, True, False],
    ...                   [True, False, False, True],
    ...                   [False, False, False, True]])
    >>> breadth_first_search(graph=G, source=2)
    tensor([2, 0, 3, 1])
    """
    check_graph_input(graph)

    reached_nodes = []
    reached_nodes.append(source)

    Q = queue.Queue()
    Q.put(source)

    while not Q.empty():
        current_node = Q.get()

        for node, is_edge in enumerate(graph[current_node, :]):
            if (is_edge) and (node not in reached_nodes):
                reached_nodes.append(node)
                Q.put(node)

    reached_nodes = torch.tensor(reached_nodes)

    return reached_nodes
