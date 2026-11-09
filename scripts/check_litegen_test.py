"""The external platform must match the reviewed revision without changing CANN's environment."""

import json
import os
import subprocess

import pytest

import check_litegen


@pytest.fixture
def source(tmp_path, monkeypatch):
    bundle = tmp_path / "cosmos"
    root = tmp_path / "LiteGen.dev"
    (bundle / "config").mkdir(parents=True)
    (root / "litegen").mkdir(parents=True)
    (root / "litegen/__init__.py").write_text("")
    env = {k: v for k, v in os.environ.items() if k not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}}
    for args in (("init",), ("add", "."), ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture")):
        subprocess.run(["/usr/bin/git", "-C", str(root), *args], env=env, check=True, capture_output=True)
    commit = subprocess.check_output(["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"], env=env, text=True).strip()
    (bundle / "config/litegen.lock.json").write_text(json.dumps({"commit": commit}))
    monkeypatch.setattr(check_litegen, "BUNDLE", bundle)
    monkeypatch.setenv("LITEGEN_ROOT", str(root))
    return bundle, root, commit


def test_verified_external_checkout_preserves_cann_environment(source, monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/test/cann/libraries")
    assert check_litegen.check_source()["commit"] == source[2]
    assert os.environ["LD_LIBRARY_PATH"] == "/test/cann/libraries"


def test_revision_mismatch_is_rejected(source):
    (source[0] / "config/litegen.lock.json").write_text(json.dumps({"commit": "0" * 40}))
    with pytest.raises(ValueError, match="revision mismatch"):
        check_litegen.check_source()


@pytest.mark.parametrize("path", ["litegen/__init__.py", "litegen/new.py"])
def test_changed_or_untracked_runtime_is_rejected(source, path):
    (source[1] / path).write_text("# changed\n")
    with pytest.raises(ValueError, match="uncommitted"):
        check_litegen.check_source()


def test_vendored_checkout_is_rejected(source, monkeypatch):
    monkeypatch.setenv("LITEGEN_ROOT", str(source[0] / "LiteGen.dev"))
    with pytest.raises(ValueError, match="outside this project"):
        check_litegen.check_source()


def test_missing_checkout_is_rejected(source, monkeypatch):
    monkeypatch.setenv("LITEGEN_ROOT", str(source[1] / "missing"))
    with pytest.raises(ValueError, match="Missing LiteGen checkout"):
        check_litegen.check_source()
