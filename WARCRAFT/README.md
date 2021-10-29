# map-backprop

Download and prepare the blackbox-backprop datasets.

```bash
$ ./download.sh
$ cd data/
$ ./merge.py
```

### Experiments 

- Warcraft (12x12) shortest path with mean squared loss (computed using the true edges cost): `settings/warcraft_shortest_path/12x12_map_mse.json`. Default update mode: 2; only MAP. 
  Notes: loss is reduced fo a couple of epochs but after that there seems to be slow divergence... should to a small HP sweep, maybe it's because of the optimizer. We should also implement early stopping!
