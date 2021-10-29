# -*- coding: utf-8 -*-

import numpy as np

from maprop.decorators import input_to_numpy, none_if_missing_arg
from maprop.utils import all_accuracies
from maprop.blackbox.dijkstra import dijkstra


@none_if_missing_arg
def perfect_match_accuracy(true_paths, suggested_paths):
    matching_correct = np.sum(np.abs(true_paths - suggested_paths), axis=-1)
    avg_matching_correct = (matching_correct < 0.5).mean()
    return avg_matching_correct


@none_if_missing_arg
def cost_ratio(vertex_costs, true_paths, suggested_paths):
    suggested_paths_costs = suggested_paths * vertex_costs
    true_paths_costs = true_paths * vertex_costs
    return (np.sum(suggested_paths_costs, axis=1) / np.sum(true_paths_costs, axis=1)).mean()


@input_to_numpy
def compute_metrics(true_paths, suggested_paths, true_vertex_costs):
    batch_size = true_vertex_costs.shape[0]
    metrics = {
        "perfect_match_accuracy": perfect_match_accuracy(true_paths.reshape(batch_size,-1), suggested_paths.reshape(batch_size,-1)),
        "cost_ratio_suggested_true": cost_ratio(true_vertex_costs, true_paths, suggested_paths),
        **all_accuracies(true_paths, suggested_paths, true_vertex_costs, is_valid_label_fn, 6)
    }
    return metrics


def is_valid_label_fn(suggested_path):
    inverted_path = 1.-suggested_path
    shortest_path, _, _ = dijkstra(inverted_path)
    is_valid = (shortest_path * inverted_path).sum() == 0
    return is_valid
