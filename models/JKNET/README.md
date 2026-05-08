# JK-Net Baseline02

JK-Net baseline on Planetoid datasets:

- `cora`
- `citeseer`
- `pubmed`

## Run

```bash
python jknet_baseline02/train_jknet.py --dataset pubmed
```

## Notes

- GCN backbone + `JumpingKnowledge` (`cat`/`max`/`lstm`).
- Results are written to:
  - `jknet_baseline02/results_jknet02.txt`
