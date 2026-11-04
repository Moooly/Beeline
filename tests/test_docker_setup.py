"""Exercise setup orchestration without downloading or changing Docker images."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "initialize.sh"
CELLORACLE = "grnbeeline/celloracle:base"
GRISLI = "grnbeeline/grisli:base"


@pytest.fixture
def setup_cli(tmp_path):
    log = tmp_path / "calls.jsonl"
    docker = tmp_path / "docker"
    implementation = tmp_path / "fake_docker.py"
    implementation.write_text(r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ["DOCKER_TEST_LOG"])
previous = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
with log.open("a") as output:
    output.write(json.dumps(args) + "\n")
config = json.loads(os.environ["DOCKER_TEST_CONFIG"])
if args == ["info"]:
    sys.exit(1 if config.get("daemon_down") else 0)
if args[:2] == ["image", "pull"]:
    sys.exit(1 if args[-1] in config.get("pull_fail", []) else 0)
if args[0] == "build":
    tag = args[args.index("--tag") + 1]
    sys.exit(1 if tag in config.get("build_fail", []) else 0)
if args[:2] == ["image", "inspect"]:
    tag = args[-1]
    built = any(call[0] == "build" and tag in call for call in previous)
    if tag in config.get("missing", []) and not built:
        sys.exit(1)
    arch = "linux/arm64" if tag in config.get("wrong_arch", []) and not built else "linux/amd64"
    print(arch)
    sys.exit(0)
if args[:2] == ["image", "rm"]:
    sys.exit(1 if args[-1] in config.get("remove_fail", []) else 0)
raise SystemExit("unexpected Docker command: " + repr(args))
''')
    docker.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} '
                      f'{shlex.quote(str(implementation))} "$@"\n')
    docker.chmod(0o755)

    def run(*args, **config):
        if log.exists():
            log.unlink()
        env = dict(os.environ, PATH=f"{tmp_path}:{os.environ['PATH']}",
                   DOCKER_TEST_LOG=str(log), DOCKER_TEST_CONFIG=json.dumps(config))
        result = subprocess.run(["/bin/bash", str(SCRIPT), *args], cwd=tmp_path,
                                env=env, text=True, capture_output=True, timeout=15)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls
    return run


def builds(calls):
    return [call[call.index("--tag") + 1] for call in calls if call[0] == "build"]


def test_default_includes_celloracle_and_all_website_images(setup_cli):
    result, calls = setup_cli()
    assert result.returncode == 0, result.stderr
    assert len(builds(calls)) == 5
    assert CELLORACLE in builds(calls)
    assert "grnbeeline/slingshot:0.1.0" in builds(calls)
    assert sum(call[:2] == ["image", "pull"] for call in calls) == 7
    for call in calls:
        if call[0] == "build" or call[:2] == ["image", "pull"]:
            assert call[call.index("--platform") + 1] == "linux/amd64"
        if call[0] == "build":
            assert (Path(call[-1]) / "Dockerfile").is_file()


def test_preview_never_contacts_docker(setup_cli):
    result, calls = setup_cli("--dry-run", daemon_down=True)
    assert result.returncode == 0
    assert calls == []
    assert CELLORACLE in result.stdout
    assert "No images were pulled" in result.stdout


def test_source_build_includes_legacy_without_pulls(setup_cli):
    result, calls = setup_cli("--build-all")
    assert result.returncode == 0
    assert len(set(builds(calls))) == 16
    assert "jump3:base" in builds(calls)
    assert not any(call[:2] == ["image", "pull"] for call in calls)


def test_pull_failure_uses_compatible_cache(setup_cli):
    result, calls = setup_cli(pull_fail=[GRISLI])
    assert result.returncode == 0
    assert GRISLI not in builds(calls)
    assert "using the existing linux/amd64 image" in result.stderr


@pytest.mark.parametrize("condition", ["missing", "wrong_arch"])
def test_pull_failure_without_compatible_cache_builds_source(setup_cli, condition):
    result, calls = setup_cli(pull_fail=[GRISLI], **{condition: [GRISLI]})
    assert result.returncode == 0
    assert GRISLI in builds(calls)
    assert "attempting its local Dockerfile" in result.stderr


def test_failed_build_does_not_mask_failure_with_old_image(setup_cli):
    result, calls = setup_cli(build_fail=[CELLORACLE])
    assert result.returncode == 1
    assert "grnbeeline/slingshot:0.1.0" in builds(calls)
    assert f"build: {CELLORACLE}" in result.stderr
    assert "Done." not in result.stdout


def test_failed_fallback_build_is_reported(setup_cli):
    result, calls = setup_cli(pull_fail=[GRISLI], missing=[GRISLI], build_fail=[GRISLI])
    assert result.returncode == 1
    assert CELLORACLE in builds(calls)
    assert f"build: {GRISLI}" in result.stderr


@pytest.mark.parametrize("flag,count", [("--remove-images", 16),
                                         ("--remove-local-images", 5),
                                         ("--remove-grnbeeline-images", 7)])
def test_removal_scopes_are_distinct_and_no_duplicate_tags(setup_cli, flag, count):
    result, calls = setup_cli(flag)
    assert result.returncode == 0
    removed = [call[-1] for call in calls if call[:2] == ["image", "rm"]]
    assert len(removed) == len(set(removed)) == count
    assert not builds(calls)


def test_removal_failure_blocks_rebuild(setup_cli):
    result, calls = setup_cli("--remove-images", "--build", remove_fail=[CELLORACLE])
    assert result.returncode == 1
    assert f"remove: {CELLORACLE}" in result.stderr
    assert not builds(calls)


def test_wrong_architecture_after_pull_fails_verification(setup_cli):
    result, calls = setup_cli(wrong_arch=[GRISLI])
    assert result.returncode == 1
    assert f"verify: {GRISLI}" in result.stderr
    assert CELLORACLE in builds(calls)


def test_unavailable_daemon_exits_before_mutations(setup_cli):
    result, calls = setup_cli(daemon_down=True)
    assert result.returncode == 1
    assert calls == [["info"]]


def test_conflicting_removal_flags_fail_before_docker(setup_cli):
    result, calls = setup_cli("--remove-images", "--remove-local-images")
    assert result.returncode == 1
    assert calls == []
