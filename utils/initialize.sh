#!/bin/bash

set -e # abandon script on error
BASEDIR="$(dirname "$(readlink -f "$0")")" # set env variable for current directory (utils/)
ROOTDIR="$(dirname "$BASEDIR")"           # parent directory of BASEDIR (repo root)

BUILD=false
HELP=false
REMOVE_LOCAL=false
REMOVE_GRNBEELINE=false
VERBOSE_VALUE="-q "

# Images pulled from DockerHub (grnbeeline organisation).
# Referenced in both the --remove-grnbeeline-images block and the pull block.
# NOTE: CellOracle is NOT here — it is a local-only image (not published to
# DockerHub), so it can only be produced with --build.
DOCKERHUB_IMAGES=(
    grnbeeline/arboreto:base
    grnbeeline/grisli:base
    grnbeeline/grnvbem:base
    grnbeeline/leap:base
    grnbeeline/pidc:base
    grnbeeline/ppcor:base
    grnbeeline/scinge:base
    grnbeeline/scns:base
    grnbeeline/scode:base
    grnbeeline/scribe:base
    grnbeeline/sincerities:base
    grnbeeline/singe:0.4.1
)

# Local build targets: "<Algorithms subdirectory>=<image tag>".
# The tags MATCH the docker_image values in the GRNScope algorithm registry, so
# `--build` produces exactly the images GRNScope runs (previously the build
# tagged e.g. arboreto:base while the registry references grnbeeline/arboreto:base).
# This same list drives --remove-local-images.
BUILD_TARGETS=(
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
  echo "Usage: $(basename "$0") [OPTIONS] [ARGUMENTS]"
  echo "This script creates docker containers for BEELINE."
  echo ""
  echo "Options:"
  echo "  -h, --help                   Display this help message and exit."
  echo "  -b, --build                  Instead of pulling images from docker hub, build them manually locally."
  echo "                               Images are tagged to match the GRNScope registry (grnbeeline/*)."
  echo "  -v, --verbose                Enable verbose output."
  echo "  --remove-local-images        Remove locally built BEELINE docker images. If combined with --build,"
  echo "                               images are removed first then rebuilt. If used alone, exits after removal."
  echo "  --remove-grnbeeline-images   Remove DockerHub (grnbeeline) BEELINE docker images. If combined with --build,"
  echo "                               images are removed first then rebuilt. If used alone, exits after removal."
  echo ""
  echo "Requirements:"
  echo "  docker (last version tested 28.5.1, build e180ab8)"
  echo "  conda (last version tested 25.9.0)"
  echo "  git"
  echo ""
  echo "Examples:"
  echo "  $(basename "$0")"
  echo "  $(basename "$0") -b --verbose"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -b|--build)
      BUILD=true
      ;;
    -v|--verbose)
      VERBOSE_VALUE=""
      ;;
    -h|--help)
      HELP=true
      ;;
    --remove-local-images)
      REMOVE_LOCAL=true
      ;;
    --remove-grnbeeline-images)
      REMOVE_GRNBEELINE=true
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help
      exit 1
      ;;
  esac
  shift # Consume the current argument (flag or value)
done

if [[ "$HELP" = true ]]; then
    show_help
    exit 0
fi

if [[ "$REMOVE_GRNBEELINE" = true ]]; then
    echo "Removing grnbeeline DockerHub images..."
    for image in "${DOCKERHUB_IMAGES[@]}"; do
        if [ "$(docker images -q "$image" 2>/dev/null)" != "" ]; then
            docker rmi "$image"
            echo "Removed $image"
        fi
    done
    echo "Done removing grnbeeline images."

    # Exit here unless --build was also specified, in which case fall through
    # to rebuild the images immediately after removal.
    if [[ "$BUILD" = false ]]; then
        exit 0
    fi
fi

if [[ "$REMOVE_LOCAL" = true ]]; then
    echo "Removing locally built BEELINE docker images..."
    for target in "${BUILD_TARGETS[@]}"; do
        image="${target#*=}"
        if [ "$(docker images -q "$image" 2>/dev/null)" != "" ]; then
            docker rmi "$image"
            echo "Removed $image"
        fi
    done
    echo "Done removing local images."

    # Exit here unless --build was also specified, in which case fall through
    # to rebuild the images immediately after removal.
    if [[ "$BUILD" = false ]]; then
        exit 0
    fi
fi

if [[ "$BUILD" = true ]]; then
    # Build every algorithm image from Algorithms/, tagged to match the registry.
    # This may take a while.
    echo "Building BEELINE docker images (this may take a while)..."

    for target in "${BUILD_TARGETS[@]}"; do
        dir="${target%%=*}"       # Algorithms subdirectory
        image="${target#*=}"      # image tag (matches the GRNScope registry)
        algo_dir="$ROOTDIR/Algorithms/$dir"

        if [ ! -d "$algo_dir" ]; then
            echo "Skipping $dir: directory not found ($algo_dir)"
            continue
        fi

        echo "----- Building $dir -> $image -----"
        pushd "$algo_dir" > /dev/null
        # '|| true' so one failing build does not abort the whole run (set -e);
        # the image check below reports per-algorithm success/failure.
        docker build ${VERBOSE_VALUE}-t "$image" . || true
        if [ "$(docker images -q "$image" 2>/dev/null)" != "" ]; then
            echo "Docker container for $dir is built and tagged as $image"
        else
            echo "Oops! Unable to build Docker container for $dir"
        fi
        popd > /dev/null
    done
else
    echo "Pulling docker images from https://hub.docker.com/u/grnbeeline..."
    echo "NOTE: CellOracle is local-only; use --build to produce grnbeeline/celloracle:base."
    for image in "${DOCKERHUB_IMAGES[@]}"; do
        docker image pull $VERBOSE_VALUE "$image"
    done
fi
