from __future__ import annotations

from luxai.s2s_magpie import provision


def test_load_lock_uses_packaged_model_asset_lock(monkeypatch) -> None:
    expected = {"schema_version": 1}
    monkeypatch.setattr(provision, "load_asset_lock", lambda: expected)

    assert provision.load_lock() is expected


def test_main_preserves_provisioning_options(monkeypatch, tmp_path, capsys) -> None:
    lock = {
        "python_packages": {},
        "huggingface": {},
        "nltk": {"punkt_tab": {"resource": "tokenizers/punkt_tab"}},
    }
    calls: list[tuple] = []
    hf_cache = tmp_path / "hf"
    nltk_data = tmp_path / "nltk"

    monkeypatch.setattr(provision, "load_lock", lambda: lock)
    monkeypatch.setattr(
        provision,
        "validate_package_versions",
        lambda value: calls.append(("packages", value)),
    )
    monkeypatch.setattr(
        provision,
        "provision_huggingface",
        lambda value, cache, *, verify_only, include_optional: calls.append(
            ("hf", value, cache, verify_only, include_optional)
        ),
    )
    monkeypatch.setattr(
        provision,
        "provision_nltk_asset",
        lambda name, asset, root, *, verify_only: calls.append(("nltk", name, asset, root, verify_only)),
    )

    result = provision.main(
        [
            "--hf-cache",
            str(hf_cache),
            "--nltk-data",
            str(nltk_data),
            "--verify-only",
        ]
    )

    assert result == 0
    assert calls == [
        ("packages", lock),
        ("hf", lock, hf_cache.resolve(), True, False),
        ("nltk", "punkt_tab", lock["nltk"]["punkt_tab"], nltk_data.resolve(), True),
    ]
    output = capsys.readouterr().out
    assert f"export HF_HUB_CACHE={hf_cache.resolve()}" in output
    assert f"export NLTK_DATA={nltk_data.resolve()}" in output


def test_main_can_skip_package_validation(monkeypatch, tmp_path) -> None:
    lock = {"python_packages": {}, "huggingface": {}, "nltk": {}}
    monkeypatch.setattr(provision, "load_lock", lambda: lock)
    monkeypatch.setattr(
        provision,
        "validate_package_versions",
        lambda _lock: (_ for _ in ()).throw(AssertionError("package validation was not skipped")),
    )
    monkeypatch.setattr(provision, "provision_huggingface", lambda *_args, **_kwargs: None)

    result = provision.main(
        [
            "--nltk-data",
            str(tmp_path),
            "--skip-package-check",
        ]
    )

    assert result == 0


def test_main_can_include_optional_models(monkeypatch, tmp_path) -> None:
    lock = {"python_packages": {}, "huggingface": {}, "nltk": {}}
    calls: list[bool] = []
    monkeypatch.setattr(provision, "load_lock", lambda: lock)
    monkeypatch.setattr(provision, "validate_package_versions", lambda _lock: None)
    monkeypatch.setattr(
        provision,
        "provision_huggingface",
        lambda _lock, _cache, *, verify_only, include_optional: calls.append(include_optional),
    )

    result = provision.main(
        [
            "--nltk-data",
            str(tmp_path),
            "--include-optional",
        ]
    )

    assert result == 0
    assert calls == [True]
