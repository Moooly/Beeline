# SLINGSHOT — pseudotime estimator

A self-contained Docker tool that estimates a pseudotime trajectory for a
single-cell expression matrix, so datasets uploaded **without** a `PseudoTime.csv`
can still run the trajectory-based GRN methods (SCODE, SINCERITIES, SCRIBE,
SINGE, LEAP, GRISLI, GRNVBEM).

It mirrors BEELINE's preprocessing pipeline: **log-transform → PCA → cluster →
Slingshot**, and writes a BEELINE-compatible `PseudoTime.csv`. Unlike the other
`Algorithms/` folders (which run GRN inference), this is a **preprocessing /
trajectory-inference** tool that *produces* an input the GRN methods consume.

> ⚠️ Estimated pseudotime is only meaningful for a **continuous biological
> process** (e.g. differentiation, cell cycle). For discrete, unrelated cell
> types it is not interpretable. Always prefer a user-provided `PseudoTime.csv`
> when one exists, and label generated pseudotime as *estimated*.

## Interface (the contract later phases depend on)

```
Rscript estimate_pseudotime.R \
  --expression ExpressionData.csv \   # required
  [--clusters ClusterLabels.csv] \    # optional
  [--startCluster <label>] \          # optional trajectory root
  [--nPCs 20] \                       # optional, PCA components
  [--nClusters 10] \                  # optional, k for k-means when no clusters
  [--seed 1] \                        # optional
  --output PseudoTime.csv             # default: PseudoTime.csv
```

| Flag | Required | Default | Meaning |
|------|----------|---------|---------|
| `-e`, `--expression` | yes | — | `ExpressionData.csv`, genes × cells (first column = gene names, header = cell names). |
| `-c`, `--clusters` | no | — | `ClusterLabels.csv`: cell name in the first column, cluster label in the next. If omitted, cells are clustered by k-means on the PCA embedding. |
| `-s`, `--startCluster` | no | auto | Cluster label to use as the trajectory root. If omitted, Slingshot picks the root (direction is then arbitrary — prefer setting this). |
| `--nPCs` | no | 20 | Principal components used for the embedding (clamped to the data size). |
| `--nClusters` | no | 10 | k for k-means, used only when `--clusters` is not given. |
| `--seed` | no | 1 | Random seed (affects k-means; keeps runs reproducible). |
| `-o`, `--output` | no | `PseudoTime.csv` | Output path. |

### Input formats

- **ExpressionData.csv** — genes × cells. Row 1 is the header of cell names
  (with an empty first cell); column 1 is gene names. Same file the GRN
  algorithms consume.
- **ClusterLabels.csv** — first column = cell name (must match the expression
  cell names), the following column = cluster label. Extra columns are ignored.

### Output format

`PseudoTime.csv` in BEELINE format — cells × lineages:

```
PseudoTime1,PseudoTime2
CELL_A,114.76,NA
CELL_B,NA,78.73
```

- Header lists one column per recovered Slingshot lineage (`PseudoTime1`, `PseudoTime2`, …).
- Each row is `<cell name>,<value|NA>,…`; `NA` means the cell is not on that lineage.
- Read back with `pandas.read_csv(path, index_col=0)`.

### Exit codes

- `0` — success, `PseudoTime.csv` written.
- `1` — a handled error (bad input, too few clusters, unknown start cluster,
  no variable genes, etc.). A human-readable reason is printed to `stderr`;
  progress/info goes to `stdout` for the job log.

## Build & run

```bash
# from this directory
docker build -t grnbeeline/slingshot:0.1.0 .

docker run --rm -v "$PWD:/data" grnbeeline/slingshot:0.1.0 \
  --expression /data/ExpressionData.csv \
  --clusters   /data/ClusterLabels.csv \
  --startCluster 0 \
  --output     /data/PseudoTime.csv
```

## Notes for the GRNScope backend integration (Phase 2)

- The container follows the same "mount the run dir, produce a CSV" pattern as
  the algorithm images, so it can be launched by a service analogous to the
  algorithm runner (async job, progress from stdout, stop by killing the
  container).
- On success, the produced `PseudoTime.csv` becomes the project's
  `pseudotime_path`, flagged as **estimated** in the manifest, which flips the
  trajectory algorithms to "available".
- Reuse the project's `ClusterLabels.csv` (already collected for CellOracle) as
  `--clusters` when present; otherwise the container clusters automatically.
