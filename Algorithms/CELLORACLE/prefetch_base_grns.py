"""Pre-download CellOracle's built-in base GRNs at image-build time.

Running the loaders once caches the base-GRN data inside the image so that
`docker run --rm` invocations need no network access at runtime (each run gets a
fresh container, so an un-cached base GRN would otherwise be re-downloaded).
"""
import celloracle as co

LOADERS = [
    co.data.load_human_promoter_base_GRN,
    co.data.load_mouse_promoter_base_GRN,
    co.data.load_rat_promoter_base_GRN,
    co.data.load_Pig_promoter_base_GRN,
    co.data.load_chicken_promoter_base_GRN,
    co.data.load_zebrafish_promoter_base_GRN,
    co.data.load_xenopus_tropicalis_promoter_base_GRN,
    co.data.load_drosophila_promoter_base_GRN,
    co.data.load_Celegans_promoter_base_GRN,
    co.data.load_Scerevisiae_promoter_base_GRN,
    co.data.load_mouse_scATAC_atlas_base_GRN,
]

failures = []
for loader in LOADERS:
    try:
        loader()
        print(f"cached: {loader.__name__}")
    except Exception as exc:  # noqa: BLE001 - collect and report below
        print(f"ERROR: failed to pre-cache {loader.__name__}: {exc}")
        failures.append(loader.__name__)

# Fail the build if any base GRN could not be cached. Otherwise the image ships
# with a missing/empty base-GRN cache and CellOracle later dies at runtime with
# a generic "No columns to parse from file" (the --rm containers have no network
# to re-download). Prefetch failures are usually transient — re-run the build.
if failures:
    raise SystemExit(
        "Base-GRN prefetch failed for: "
        + ", ".join(failures)
        + ". This is usually a transient network error — re-run the build."
    )
print("All base GRNs cached successfully.")
