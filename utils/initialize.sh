#!/bin/bash
# One-command setup for the Docker environments used by GRNScope.
set -euo pipefail
BASEDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(dirname "$BASEDIR")"

BUILD_ALL=false
DRY_RUN=false
VERBOSE=false
REMOVE_SCOPE=""
# Published BEELINE images and the legacy MATLAB runtimes require amd64.
# Docker Desktop on Apple Silicon must have x86 emulation enabled.
PLATFORM=linux/amd64
FAILURES=()
WARNINGS=()

# Current runner scripts are mounted from this checkout at execution time.
# These published images supply the dependency environments.
PULL_TARGETS=(
    "GRISLI=grnbeeline/grisli:base"
    "GRNVBEM=grnbeeline/grnvbem:base"
    "LEAP=grnbeeline/leap:base"
    "PIDC=grnbeeline/pidc:base"
    "PPCOR=grnbeeline/ppcor:base"
    "SCODE=grnbeeline/scode:base"
    "SINCERITIES=grnbeeline/sincerities:base"
)

# Keep our tested custom environments. Upstream ARBORETO/SCRIBE/SINGE
# environments have not been validated as replacements for these Dockerfiles.
LOCAL_TARGETS=(
    "ARBORETO=grnbeeline/arboreto:base"
    "CELLORACLE=grnbeeline/celloracle:base"
    "SCRIBE=grnbeeline/scribe:base"
    "SINGE=grnbeeline/singe:0.4.1"
    "SLINGSHOT=grnbeeline/slingshot:0.1.0"
)
LEGACY_TARGETS=(
    "JUMP3=jump3:base"
    "PNI=pni:base"
    "SCNS=scns:base"
    "SCSGL=scsgl:base"
)
ALL_TARGETS=("${PULL_TARGETS[@]}" "${LOCAL_TARGETS[@]}" "${LEGACY_TARGETS[@]}")

show_help() {
    cat <<'HELP'
Usage: bash utils/initialize.sh [OPTIONS]

Default: pull 7 published images and build 5 custom environments, including
CellOracle and Slingshot. GENIE3 and GRNBoost2 share ARBORETO; Pearson needs
no Docker image. Builds reuse Docker's layer cache on subsequent runs.

All images use linux/amd64. ARM hosts need Docker x86 emulation.

Options:
  -h, --help                    Show this help.
  --dry-run                     Preview commands without requiring Docker.
  -b, --build, --build-all       Build all 16 managed images from source,
                                including legacy methods outside the website.
  -v, --verbose                 Use plain Docker build progress output.
  --remove-images               Remove all 16 managed image tags.
  --remove-local-images         Remove the 5 custom image tags only.
  --remove-grnbeeline-images     Remove the 7 published image tags only.

Removal exits after cleanup unless combined with --build. Only one removal
scope may be selected. Images in use are not forcibly removed.

Examples:
  bash utils/initialize.sh              # recommended website setup
  bash utils/initialize.sh --dry-run    # preview that setup
  bash utils/initialize.sh --build-all  # source builds, including legacy methods
HELP
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build|--build-all|-b) BUILD_ALL=true ;;
        --dry-run) DRY_RUN=true ;;
        --verbose|-v) VERBOSE=true ;;
        --help|-h) show_help; exit 0 ;;
        --remove-images|--remove-local-images|--remove-grnbeeline-images)
            if [[ -n "$REMOVE_SCOPE" && "$REMOVE_SCOPE" != "$1" ]]; then
                echo "ERROR: choose only one removal scope." >&2
                exit 1
            fi
            REMOVE_SCOPE="$1"
            ;;
        *) echo "ERROR: unknown option: $1" >&2; show_help >&2; exit 1 ;;
    esac
    shift
done

run_command() {
    if [[ "$DRY_RUN" = true ]]; then
        printf '  '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ "$DRY_RUN" = false ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: install Docker and ensure it is available in PATH." >&2
        exit 1
    fi
    if ! docker info >/dev/null 2>&1; then
        echo "ERROR: start Docker before running this script." >&2
        exit 1
    fi
fi

image_ready() {
    local actual
    actual="$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$1" 2>/dev/null)" || return 1
    [[ "$actual" = "$PLATFORM" ]]
}

build_target() {
    local target="$1" dir image algo_dir
    dir="${target%%=*}"
    image="${target#*=}"
    algo_dir="$ROOTDIR/Algorithms/$dir"
    local build_args=(--platform "$PLATFORM" --tag "$image")
    if [[ "$VERBOSE" = true ]]; then
        build_args+=(--progress=plain)
    fi
    echo "Building $dir -> $image (cached layers reused)"
    if [[ ! -f "$algo_dir/Dockerfile" ]]; then
        echo "ERROR: missing $algo_dir/Dockerfile" >&2
        FAILURES+=("build: $image (missing Dockerfile)")
    elif run_command docker build "${build_args[@]}" "$algo_dir"; then
        if [[ "$DRY_RUN" = false ]] && ! image_ready "$image"; then
            FAILURES+=("verify: $image (expected $PLATFORM image)")
        fi
    else
        FAILURES+=("build: $image")
    fi
}

finish() {
    local message
    if [[ ${#WARNINGS[@]} -gt 0 ]]; then
        for message in "${WARNINGS[@]}"; do
            echo "WARNING: $message" >&2
        done
    fi
    if [[ ${#FAILURES[@]} -gt 0 ]]; then
        echo "Setup completed with failures (see Docker output above):" >&2
        for message in "${FAILURES[@]}"; do
            echo "  $message" >&2
        done
        exit 1
    fi
}

if [[ -n "$REMOVE_SCOPE" ]]; then
    case "$REMOVE_SCOPE" in
        --remove-images) REMOVE_TARGETS=("${ALL_TARGETS[@]}") ;;
        --remove-local-images) REMOVE_TARGETS=("${LOCAL_TARGETS[@]}") ;;
        --remove-grnbeeline-images) REMOVE_TARGETS=("${PULL_TARGETS[@]}") ;;
    esac
    for target in "${REMOVE_TARGETS[@]}"; do
        image="${target#*=}"
        if [[ "$DRY_RUN" = true ]] || docker image inspect "$image" >/dev/null 2>&1; then
            if ! run_command docker image rm "$image"; then
                FAILURES+=("remove: $image")
            fi
        fi
    done
    # A failed cleanup must not be reported as successful, even with --build.
    finish
    if [[ "$BUILD_ALL" = false ]]; then
        echo "Cleanup completed (preview only when --dry-run is set)."
        exit 0
    fi
fi

if [[ "$BUILD_ALL" = true ]]; then
    echo "Building all 16 managed images, including legacy methods ($PLATFORM)."
    for target in "${ALL_TARGETS[@]}"; do
        build_target "$target"
    done
else
    echo "Setting up 12 images for all website methods and Slingshot ($PLATFORM)."
    for target in "${PULL_TARGETS[@]}"; do
        image="${target#*=}"
        echo "Pulling $image"
        if run_command docker image pull --platform "$PLATFORM" "$image"; then
            if [[ "$DRY_RUN" = false ]] && ! image_ready "$image"; then
                FAILURES+=("verify: $image (expected $PLATFORM image)")
            fi
        elif image_ready "$image"; then
            WARNINGS+=("could not refresh $image; using the existing $PLATFORM image.")
        else
            WARNINGS+=("could not pull $image; attempting its local Dockerfile.")
            build_target "$target"
        fi
    done
    for target in "${LOCAL_TARGETS[@]}"; do
        build_target "$target"
    done
fi

finish
if [[ "$DRY_RUN" = true ]]; then
    echo "Preview complete. No images were pulled, built, or removed."
else
    echo "Done. All requested images are present for $PLATFORM."
    echo "Keep this BEELINE checkout on the server: runners mount its current scripts."
fi
