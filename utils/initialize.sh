#!/bin/bash

set -e # abandon script on error
BASEDIR="$(dirname "$(readlink -f "$0")")" # utils/
ROOTDIR="$(dirname "$BASEDIR")"           # repo root

BUILD_ALL=false
HELP=false
REMOVE_IMAGES=false
VERBOSE_VALUE="-q "

# ---------------------------------------------------------------------------
# Tested published images to PULL, for algorithms we did NOT modify inside the
# image. Any changes to these algorithms live host-side (BLRun/*.py + GRNScope),
# NOT in the image, so the published DockerHub builds are correct and reliable.
# (arboreto is excluded — we build it locally; celloracle is not published.)
# ---------------------------------------------------------------------------
PULL_IMAGES=(
    grnbeeline/grisli:base
    grnbeeline/grnvbem:base
    grnbeeline/leap:base
    grnbeeline/pidc:base
    grnbeeline/ppcor:base
    grnbeeline/scode:base
    grnbeeline/scribe:base
    grnbeeline/sincerities:base
    grnbeeline/singe:0.4.1
)

# Images that MUST be built locally. The default run builds exactly these:
#   ARBORETO   — contains our modified runArboreto.py (GENIE3 / GRNBOOST2)
#   CELLORACLE — new algorithm, not published to DockerHub
LOCAL_ONLY_TARGETS=(
    "ARBORETO=grnbeeline/arboreto:base"
    "CELLORACLE=grnbeeline/celloracle:base"
)

# Full set for --build-all (rebuild EVERY image from source).
# WARNING: some local Dockerfiles do not reproduce the published images (e.g. the
# SINGE Dockerfile lacks octave), so prefer the default for real deployments.
ALL_BUILD_TARGETS=(
    "ARBORETO=grnbeeline/arboreto:base"
    "CELLORACLE=grnbeeline/celloracle:base"
    "GRISLI=grnbeeline/grisli:base"
    "GRNVBEM=grnbeeline/grnvbem:base"
    "JUMP3=jump3:base"
    "LEAP=grnbeeline/leap:base"
    "PIDC=grnbeeline/pidc:base"
    "PNI=pni:base"
    "PPCOR=grnbeeline/ppcor:base"
    "SCNS=scns:base"
    "SCODE=grnbeeline/scode:base"
    "SCRIBE=grnbeeline/scribe:base"
    "SCSGL=scsgl:base"
    "SINCERITIES=grnbeeline/sincerities:base"
    "SINGE=grnbeeline/singe:0.4.1"
)

show_help() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo "Set up BEELINE docker images for GRNScope."
  echo ""
  echo "Default (no options) — the recommended, correct setup:"
  echo "  1. Pull the tested published images for unchanged algorithms."
  echo "  2. Build ONLY the images that must be local:"
  echo "       - arboreto   (modified: GENIE3 / GRNBOOST2)"
  echo "       - celloracle (new, not published)"
  echo "  It does not rebuild images that don't need it (avoids e.g. breaking"
  echo "  SINGE, whose local Dockerfile lacks octave)."
  echo ""
  echo "Options:"
  echo "  -h, --help        Display this help and exit."
  echo "  --build-all       Rebuild EVERY image from source locally (advanced)."
  echo "                    WARNING: local Dockerfiles may not reproduce the"
  echo "                    published images (e.g. SINGE needs octave)."
  echo "  -v, --verbose     Enable verbose docker output."
  echo "  --remove-images   Remove all managed BEELINE images, then exit (combine"
  echo "                    with --build-all to remove then rebuild)."
  echo ""
  echo "Examples:"
  echo "  $(basename "$0")               # correct setup: pull + build arboreto/celloracle"
  echo "  $(basename "$0") --build-all   # rebuild everything from source"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --build-all|-b)
      BUILD_ALL=true
      ;;
    -v|--verbose)
      VERBOSE_VALUE=""
      ;;
    -h|--help)
      HELP=true
      ;;
    --remove-images|--remove-local-images|--remove-grnbeeline-images)
      REMOVE_IMAGES=true
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
  shift
done

if [[ "$HELP" = true ]]; then
    show_help
    exit 0
fi

# Build a list of "DIR=tag" targets from Algorithms/, tagged for the registry.
build_targets() {
    for target in "$@"; do
        local dir="${target%%=*}"
        local image="${target#*=}"
        local algo_dir="$ROOTDIR/Algorithms/$dir"

        if [ ! -d "$algo_dir" ]; then
            echo "Skipping $dir: directory not found ($algo_dir)"
            continue
        fi

        echo "----- Building $dir -> $image -----"
        pushd "$algo_dir" > /dev/null
        # '|| true' so one failing build does not abort the whole run (set -e).
        docker build ${VERBOSE_VALUE}-t "$image" . || true
        if [ "$(docker images -q "$image" 2>/dev/null)" != "" ]; then
            echo "Docker image for $dir is built and tagged as $image"
        else
            echo "Oops! Unable to build Docker image for $dir"
        fi
        popd > /dev/null
    done
}

if [[ "$REMOVE_IMAGES" = true ]]; then
    echo "Removing BEELINE docker images managed by this script..."
    REMOVE_TAGS=("${PULL_IMAGES[@]}")
    for target in "${ALL_BUILD_TARGETS[@]}"; do
        REMOVE_TAGS+=("${target#*=}")
    done
    for image in "${REMOVE_TAGS[@]}"; do
        if [ "$(docker images -q "$image" 2>/dev/null)" != "" ]; then
            docker rmi "$image" || true
            echo "Removed $image"
        fi
    done
    echo "Done removing images."

    # Exit after removal unless a rebuild was explicitly requested.
    if [[ "$BUILD_ALL" = false ]]; then
        exit 0
    fi
fi

if [[ "$BUILD_ALL" = true ]]; then
    echo "Building ALL images from source locally (this may take a while)..."
    echo "WARNING: local Dockerfiles may not reproduce the published images."
    build_targets "${ALL_BUILD_TARGETS[@]}"
else
    echo "Pulling tested published images for unchanged algorithms..."
    for image in "${PULL_IMAGES[@]}"; do
        docker image pull $VERBOSE_VALUE "$image" || echo "WARNING: failed to pull $image"
    done
    echo ""
    echo "Building the images that must be local (arboreto = modified, celloracle = new)..."
    build_targets "${LOCAL_ONLY_TARGETS[@]}"
fi

echo ""
echo "Done. GRNScope BEELINE images are ready."
