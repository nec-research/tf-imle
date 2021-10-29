"""
Implementation of the depth first search (DFS) algorithm.
"""

import torch

from sevnet.utils import check_graph_input


def depth_first_search(graph, source=0):
    """
    Perform depth first search of the given graph `graph` starting from the
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
        The nodes that are reached/found via depth first search starting
        from the node `source`.

    See Also
    --------
    For more information on the depth first search algorithm:
        https://en.wikipedia.org/wiki/Depth-first_search.

    Examples
    --------
    >>> import torch
    >>> from sevnet import depth_first_search
    >>> G = torch.tensor([[False, True, True, False],
    ...                   [False, False, True, False],
    ...                   [True, False, False, True],
    ...                   [False, False, False, True]])
    >>> depth_first_search(graph=G, source=0)
    tensor([0, 2, 3, 1])
    """
    check_graph_input(graph)

    reached_nodes = []

    stack = []
    stack.append(source)

    while stack:
        current_node = stack.pop()

        if (current_node not in reached_nodes):
            reached_nodes.append(current_node)

            for node, is_edge in enumerate(graph[current_node, :]):
                if (is_edge):
                    stack.append(node)

    reached_nodes = torch.tensor(reached_nodes)

    return reached_nodes
