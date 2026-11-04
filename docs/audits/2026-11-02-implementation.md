# BEELINE audit implementation — 2 November 2026

Baseline: `52851d6b3421ea1c1e398c223937466c7cd21823`. Changes apply to future runs; existing project results have not been rewritten.

## Correctness and reliability

- **SCODE direction:** `A[target, source]` now produces `source → target`, for both full and bounded output and across trajectories. This follows the pinned upstream SCODE model `dx/dt = A*x`. Old capped SCODE results need a fresh inference run; merely reversing their edge names cannot recover candidates discarded under the old cap.
- **Regulator selection:** self-links and nonfinite scores are excluded before every affected early cap. Duplicate edges consume one slot. Across trajectories, the largest absolute score wins while retaining its sign. Exact magnitude ties use source/target names; an equal positive/negative observation of the same edge selects the positive value. Final files are globally ranked by decreasing magnitude.
- **GRISLI:** parsers use the retained `GeneList.txt` order and size, including uncapped and matrix-fallback output. Matrix dimensions are checked. The compiled algorithm's direction convention has not been changed without executable-level scientific evidence.
- **SCRIBE/SINGE/SCODE:** bounded cross-trajectory merging no longer constructs the complete edge union. SCRIBE and SINGE no longer retain duplicate trajectory maxima. LEAP's full merge uses the same signed-score rule.
- **CellOracle:** early caps rank the emitted `coef_mean` magnitude, matching the website's ranking. Previously `coef_abs` could select a different set before signed coefficients reached the backend. Self-links and duplicates are removed first.
- **Failed runs:** missing trajectory/raw output raises an error. A new attempt removes its stale final result; failure cannot leave that result marked completed. Shared writers publish atomically. A genuinely empty, header-only network remains valid output.
- **Configuration/retries:** scalar and singleton-list booleans work. Invalid path segments and duplicate output destinations are rejected before runner construction. Worker requests never prompt on their JSON input stream. CLI retries use `--overwrite`; the website explicitly sets overwrite for its owned temporary runtime outputs.
- **Resource allocation:** trajectory worker count respects both CPUs and total memory, including minimum container allowances. SCRIBE/GRISLI preflight calculations share that allocation. Logs are consolidated even after a parallel failure. Summed container elapsed time is no longer subtracted from wall time for parallel execution; RSS is labelled as a process/lifetime high-water mark, not aggregate memory.
- **SINCERITIES:** requested and effective quantile bins must be at least five. Collapsed bins produce a useful error. Bootstrap gene limits consider distinct source cells. Constant regression responses report a clear solver limitation. Zero/nonpositive normalization divisors no longer create infinities or flip the sign of an all-negative network.
- **Pearson:** nonfinite expression input is explicitly rejected; constant genes continue to produce no correlations. Boundary ties are deterministic. The blockwise BLAS calculation is preserved.
- **Evaluation:** PathStats sorts scores itself before applying the reference-edge-count cutoff and deduplicates by edge identity.
- **Inactive methods:** JUMP3 accepts a canonically named single pseudotime lineage, orders cells by time, and rejects ambiguous multiple lineages. scSGL no longer requires an unused ground-truth file. Neither method has been enabled in the website.

## Performance and reproducibility

The AWK compactors use indexed heaps instead of scanning every retained edge for every candidate. PIDC writes bounded output directly from its inferred network using heaps, avoiding the full temporary TSV write/read cycle. Arboreto applies top-K on Dask workers before transferring results to the client, with a final reduction that remains correct if a target crosses partitions. SCODE gets dimensions from cached inputs rather than rereading its generated matrices.

GENIE3 and GRNBoost2 expose an inference seed, default **1**. This intentionally replaces unseeded model fitting; bootstrap input sampling remains separate. The existing GENIE3 default remains **400 trees**, rather than upstream's 1,000; no further tree-count reduction was made and identical rankings at different tree counts are not claimed.

`phase_timings.json` records the source-content signature, Git revision when available, Docker image ID/digests, host Python/NumPy/pandas versions, effective parameters, and hashes of source/selected/generated inputs and final output. Image IDs identify the installed container dependency set; this is not a claim that every upstream Dockerfile is fully pinned.

### Optional faster PCA

Slingshot accepts `--pcaSolver full|truncated|auto`. **Full PCA remains the default.** The optional iterative solver restores the clustering RNG state, checks eigenpair residuals, and falls back to full PCA on missing dependencies, warnings, nonconvergence or excessive residuals. `auto` attempts truncation only for larger matrices with relatively few requested components.

The Slingshot Dockerfile pins `irlba 2.3.5.1`, tested with the current R/Matrix image. A trial of 2.4.1 failed with a native NULL-argument error in this image. The local `grnbeeline/slingshot:0.1.0` image was updated with the tested package and entrypoint; the previous image is retained as `grnbeeline/slingshot:before-audit-20261102`.

On three 500-cell × 600-gene fixtures, full/truncated PCA gave the same lineage counts and membership. Maximum pseudotime deviations were approximately `1.5e-9` (linear), `1.9e-7` (branching), and `0.0651` (diffuse). The diffuse case is why truncation is **not automatically enabled for website analyses**. Small end-to-end fixtures showed little total-runtime benefit because Slingshot and startup dominate.

A separate 1,000 × 1,500 dense PCA microbenchmark gave median full-SVD time **1.045 s**, versus **0.156 s** for iterative PCA (three repetitions, including one cold iterative call of 0.820 s). Subspace cosine errors were below `3e-15`. This measures the PCA kernel, not total pseudotime runtime or a universal speedup.

### Other measured results

- AWK, 100,000 edges, K=20, three-run medians: **1.442 s → 1.170 s**, about **19% less time**, with identical selected edges/scores on the untied fixture. Randomized tied/duplicate/self-link fixtures are checked against an independent full-sort oracle.
- GENIE3 and GRNBoost2, 100 genes × 180 cells, fixed seed 17, 30 trees/rounds, K=5: full and worker-reduced output matched **exactly**. Single-run timings were 7.71→6.84 s and 4.79→4.92 s respectively. No universal inference-time speedup is claimed; the main benefit is reducing transferred/materialized output.
- Pearson's finite 1,000-gene × 500-cell fixture preserved exactly the same selected edges and signed scores as the pre-audit optimized implementation. Correct tie/deduplication handling adds work; the existing BLAS acceleration relative to original upstream remains in place.

Measurements used the available local Docker images, mostly amd64 under emulation on an Apple Silicon host. They are small-fixture checks, not production capacity estimates.

## Validation

- BEELINE: **86 tests passed**, including independent asymmetric SCODE fixtures, signs/ties/duplicate streams, retained GRISLI labels, missing-output retries, memory caps, failure logs, invalid bins, finite Pearson equivalence, CellOracle early ranking, and PathStats order independence.
- GRNScope backend: **229 tests and 65 subtests passed**.
- Fresh synthetic runs exercised all **13 active algorithms** (Pearson locally; the other methods in Docker). A separate scSGL container run also passed.
- On a sampled copy of the bundled 19-gene GSD dataset with both lineages, PPCOR, LEAP, SINCERITIES and PIDC full versus capped outputs matched within `1e-12`. SCODE, GRISLI and SCRIBE completed bounded multi-trajectory runs. SINGE produced a valid empty network on this small GSD fixture; it produced nonempty output on the independent synthetic fixture. Empty output is not fabricated into edges.
- Website integration: upload, two bootstrap runs, Pearson/GENIE3/CellOracle inference, ensemble/results, and a STAT3 perturbation completed on a disposable synthetic project.
- All comparisons used temporary data copies. Original datasets and existing projects were not modified.

Reproducible opt-in harnesses:

```sh
python utils/audit_smoke.py
python utils/audit_equivalence.py
python utils/audit_gsd.py
python utils/audit_pca.py
```

Normal regression suite:

```sh
python -m pytest -q tests Algorithms/CELLORACLE/test_perturbation_distributions.py
```

## Deliberate limits

PIDC's shared-array inference loops have not been made multiprocessing simply by adding Julia workers: upstream shared score increments can race. The safe output optimization is implemented; parallel inference still needs a separate deterministic reduction and scientific validation. Likewise, compiled GRISLI internals and cross-bootstrap converted-input reuse were not changed without proof of equivalent data/order/results.

The inactive JUMP3 Docker executable started eight MATLAB pool workers despite a two-CPU quota, approaching its 5 GiB memory limit without finishing pool startup. The isolated test was stopped. Its input-format regression is covered, but its legacy runtime is **not validated for release** and remains disabled. This is separate from the 13 active algorithms' successful checks.
