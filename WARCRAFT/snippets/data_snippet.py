#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import maprop.warcraft_shortest_path.data_utils as warcraft_shortest_path_data

train_iterator, eval_iterator, metadata = warcraft_shortest_path_data.load_dataset(
    data_dir="data/warcraft_shortest_path/12x12",
    evaluate_with_extra=False,
    normalize=True,
    use_local_path=False,
    use_test_set=True)

print(train_iterator)
print(eval_iterator)
print(metadata)
