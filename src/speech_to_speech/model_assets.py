"""Resolve immutable model assets without performing implicit downloads."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any


LOCK_FILE = files("speech_to_speech").joinpath("model_assets.lock.json")


@lru_cache(maxsize=1)
def load_asset_lock() -> dict[str, Any]:
    """Load and minimally validate the packaged model/data lock."""

    with LOCK_FILE.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if lock.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported model asset lock schema in {LOCK_FILE}")
    return lock


def locked_huggingface_asset(name: str) -> dict[str, Any]:
    try:
        return dict(load_asset_lock()["huggingface"][name])
    except KeyError as exc:
        raise KeyError(f"Unknown locked Hugging Face asset: {name}") from exc


def locked_revision_for_repo(repo_id: str) -> str | None:
    for asset in load_asset_lock()["huggingface"].values():
        if asset["repo_id"] == repo_id:
            return str(asset["revision"])
    return None


def _expanded_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(str(value))).expanduser()


def _looks_like_local_path(value: str | Path) -> bool:
    raw = os.path.expandvars(str(value))
    candidate = Path(raw).expanduser()
    return (
        candidate.exists()
        or candidate.is_absolute()
        or raw.startswith((".", "~"))
        or "\\" in raw
    )


def _provisioning_error(kind: str, model: str, revision: str | None) -> RuntimeError:
    revision_text = f" at revision {revision}" if revision else ""
    return RuntimeError(
        f"Locked {kind} asset {model!r}{revision_text} is not available locally. "
        "Run `luxai-s2s-magpie-provision` with network access, then start "
        "the service with the same Hugging Face cache."
    )


def resolve_huggingface_file(
    model: str | Path,
    *,
    filename: str,
    revision: str | None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
) -> Path:
    """Resolve one pinned Hub file or an explicitly supplied local checkpoint."""

    if _looks_like_local_path(model):
        path = _expanded_path(model)
        if path.is_dir():
            path = path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Local model checkpoint not found: {path}")
        return path.resolve()

    resolved_revision = revision or locked_revision_for_repo(str(model))
    if resolved_revision is None:
        raise ValueError(
            f"No immutable revision is configured for Hugging Face model {model!r}; "
            "set model_revision or provide a local path."
        )

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=str(model),
            filename=filename,
            revision=resolved_revision,
            cache_dir=str(_expanded_path(cache_dir)) if cache_dir else None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise _provisioning_error("file", str(model), resolved_revision) from exc
    return Path(path).resolve()


def resolve_huggingface_snapshot(
    model: str | Path,
    *,
    revision: str | None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = True,
) -> Path:
    """Resolve a complete pinned Hub snapshot or an explicit local directory."""

    if _looks_like_local_path(model):
        path = _expanded_path(model)
        if not path.is_dir():
            raise FileNotFoundError(f"Local model snapshot not found: {path}")
        return path.resolve()

    resolved_revision = revision or locked_revision_for_repo(str(model))
    if resolved_revision is None:
        raise ValueError(
            f"No immutable revision is configured for Hugging Face model {model!r}; "
            "set model_revision or provide a local directory."
        )

    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=str(model),
            revision=resolved_revision,
            cache_dir=str(_expanded_path(cache_dir)) if cache_dir else None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise _provisioning_error("snapshot", str(model), resolved_revision) from exc
    return Path(path).resolve()


def require_nltk_assets() -> None:
    """Fail fast when the explicitly provisioned NLTK datasets are absent."""

    import nltk

    missing: list[str] = []
    for name, asset in load_asset_lock()["nltk"].items():
        try:
            nltk.data.find(str(asset["resource"]))
        except (LookupError, OSError):
            missing.append(name)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required NLTK assets: {joined}. Run "
            "`luxai-s2s-magpie-provision` before starting the service."
        )


__all__ = [
    "LOCK_FILE",
    "load_asset_lock",
    "locked_huggingface_asset",
    "locked_revision_for_repo",
    "require_nltk_assets",
    "resolve_huggingface_file",
    "resolve_huggingface_snapshot",
]
