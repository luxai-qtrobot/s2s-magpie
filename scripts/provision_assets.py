#!/usr/bin/env python3
"""Provision every network-fetched runtime asset at its immutable lock revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = PROJECT_ROOT / "src" / "speech_to_speech" / "model_assets.lock.json"
MARKER_FILE = ".luxai-asset-sha256"


def load_lock() -> dict[str, Any]:
    with LOCK_FILE.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported asset lock schema in {LOCK_FILE}")
    return lock


def validate_package_versions(lock: dict[str, Any]) -> None:
    problems: list[str] = []
    for package, expected in lock["python_packages"].items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            problems.append(f"{package} is not installed (expected {expected})")
            continue
        if installed != expected:
            problems.append(f"{package}=={installed} is installed (expected {expected})")
    if problems:
        raise RuntimeError("Model runtime package check failed:\n  - " + "\n  - ".join(problems))


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if os.path.commonpath((str(root), str(target))) != str(root):
            raise RuntimeError(f"Unsafe path in NLTK archive: {member.filename}")
    archive.extractall(destination)


def provision_nltk_asset(name: str, asset: dict[str, Any], nltk_data: Path, *, verify_only: bool) -> None:
    resource_dir = nltk_data / str(asset["resource"])
    marker = resource_dir / MARKER_FILE
    expected_sha = str(asset["sha256"])
    if marker.is_file() and marker.read_text(encoding="ascii").strip() == expected_sha:
        print(f"verified nltk:{name} -> {resource_dir}")
        return
    if verify_only:
        raise RuntimeError(f"NLTK asset {name!r} is absent or not lock-verified under {nltk_data}")

    nltk_data.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"luxai-{name}-") as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / f"{name}.zip"
        with urllib.request.urlopen(str(asset["url"]), timeout=120) as response, archive_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        actual_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"Checksum mismatch for {name}: expected {expected_sha}, got {actual_sha}")

        extracted = temporary_dir / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, extracted)
        source = extracted / resource_dir.name
        if not source.is_dir():
            raise RuntimeError(f"NLTK archive for {name} did not contain {resource_dir.name}/")
        resource_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, resource_dir, dirs_exist_ok=True)
        marker.write_text(expected_sha + "\n", encoding="ascii")
    print(f"provisioned nltk:{name} -> {resource_dir}")


def provision_huggingface(lock: dict[str, Any], cache_dir: Path | None, *, verify_only: bool) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_value = str(cache_dir) if cache_dir else None
    for name, asset in lock["huggingface"].items():
        common = {
            "repo_id": str(asset["repo_id"]),
            "revision": str(asset["revision"]),
            "cache_dir": cache_value,
            "local_files_only": verify_only,
        }
        files = asset.get("files")
        if files:
            resolved = [hf_hub_download(filename=str(filename), **common) for filename in files]
        else:
            resolved = [snapshot_download(**common)]
        print(f"{'verified' if verify_only else 'provisioned'} hf:{name} -> {', '.join(resolved)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-cache", type=Path, help="Hugging Face cache directory (default: library cache)")
    parser.add_argument(
        "--nltk-data",
        type=Path,
        default=Path(os.environ.get("NLTK_DATA", "").split(os.pathsep)[0])
        if os.environ.get("NLTK_DATA")
        else Path.home() / ".cache" / "luxai-s2s-magpie" / "nltk_data",
        help="NLTK data root",
    )
    parser.add_argument("--verify-only", action="store_true", help="Forbid downloads and verify the local caches")
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Skip exact model-loader package version validation",
    )
    args = parser.parse_args()

    lock = load_lock()
    hf_cache = args.hf_cache.expanduser().resolve() if args.hf_cache else None
    nltk_data = args.nltk_data.expanduser().resolve()
    if not args.skip_package_check:
        validate_package_versions(lock)
    provision_huggingface(lock, hf_cache, verify_only=args.verify_only)
    for name, asset in lock["nltk"].items():
        provision_nltk_asset(name, asset, nltk_data, verify_only=args.verify_only)

    print("\nRuntime environment:")
    if hf_cache:
        print(f"  export HF_HUB_CACHE={hf_cache}")
    print(f"  export NLTK_DATA={nltk_data}")
    print("  export HF_HUB_OFFLINE=1")
    print("  export TRANSFORMERS_OFFLINE=1")


if __name__ == "__main__":
    main()
