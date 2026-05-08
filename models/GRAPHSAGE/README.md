# GraphSAGE Baseline02

GraphSAGE paper-style training with neighbor-sampled mini-batches on Planetoid datasets:

- `cora`
- `citeseer`
- `pubmed`

## Run

From project root:

```bash
python graphsage_baseline02/train_graphsage.py --dataset pubmed
```

## Notes

- Uses `NeighborLoader` for sampled training (`--num_neighbors`, `--batch_size`).
- Keeps full-graph evaluation on the standard train/val/test masks.
- Saves summaries to:
  - `graphsage_baseline02/results_graphsage02.txt`
