"""Shared SHA256 file-integrity helpers for the retrospective blinded study
(current_tasks/double_blind_exp.md Parts III, XXII, XXIV).

Used to prove, after the fact, that:
  * frozen model/protocol artifacts were not touched between freeze
    (freeze_final_model.py) and blinded prediction (predict_blinded.py);
  * prediction files were not touched between prediction and unblinding
    (unblind_and_evaluate.py).

Nothing here reads target/patient identity -- these are pure byte-level
file-integrity utilities.
"""
from __future__ import annotations

import hashlib
import json
import os


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(file_paths: list[str], base_dir: str | None = None) -> dict:
    """{relative_or_given_path: sha256_hex} for every path in `file_paths`."""
    manifest = {}
    for p in file_paths:
        key = os.path.relpath(p, base_dir) if base_dir else p
        manifest[key] = sha256_file(p)
    return manifest


def write_manifest(file_paths: list[str], out_path: str, base_dir: str | None = None, extra: dict | None = None) -> dict:
    manifest = {"files": build_manifest(file_paths, base_dir=base_dir)}
    if extra:
        manifest.update(extra)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


class IntegrityError(RuntimeError):
    """Raised when a SHA256 check against a recorded manifest fails."""


def verify_manifest(manifest_path: str, base_dir: str | None = None) -> None:
    """Re-hash every file listed in `manifest_path`'s "files" section and raise
    IntegrityError with a clear message listing every mismatch/missing file if
    anything does not match what was recorded at freeze/prediction time."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest.get("files", manifest)  # tolerate a bare {path: hash} manifest too
    problems = []
    for rel_path, expected_hash in files.items():
        full_path = os.path.join(base_dir, rel_path) if base_dir else rel_path
        if not os.path.exists(full_path):
            problems.append(f"MISSING: {full_path}")
            continue
        actual_hash = sha256_file(full_path)
        if actual_hash != expected_hash:
            problems.append(f"HASH MISMATCH: {full_path} (expected {expected_hash}, got {actual_hash})")
    if problems:
        raise IntegrityError(
            f"SHA256 verification against {manifest_path!r} failed for {len(problems)} file(s):\n"
            + "\n".join(problems)
        )
