#!/usr/bin/env python3
"""Apply an explicitly approved bucket-clothing plan with rollback.

The default mode is preflight-only.  ``--apply`` requires the approved plan
SHA-256, takes the vault-wide exclusive maintenance lease, backs up every
target file, and changes only the frontmatter ``name`` line plus one trailing
``[检索钥匙: ...]`` line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import frontmatter

from maintenance_barrier import MaintenanceBarrier
from storage_safety import atomic_write_text


SCHEMA = "ombre.bucket-clothing-apply/v1"
PLAN_SCHEMA = "ombre.bucket-clothing-plan/v1"
LIVE_ROOTS = frozenset({"permanent", "dynamic", "feel"})
KEY_LINE_RE = re.compile(r"(?m)^\s*\[检索钥匙\s*[:：].*?\]\s*$")
FILE_SUFFIX_RE = re.compile(r"\.(?:json|py|md)(?:\b|$)", re.IGNORECASE)
PURE_VERSION_RE = re.compile(r"^v\d+(?:\.\d+)+$", re.IGNORECASE)
PURE_TECH_IDENTIFIER_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+$"
)


@dataclass(frozen=True)
class Operation:
    bucket_id: str
    relative_path: str
    target: Path
    original_text: str
    original_sha256: str
    new_text: str
    new_sha256: str
    old_name: str
    new_name: str
    name_changed: bool
    kept_keys: tuple[str, ...]
    filtered_keys: tuple[dict, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _key_rejection_reason(value: str) -> str:
    key = value.strip()
    if not key:
        return "empty"
    if any(character in key for character in ("\r", "\n", "]", "/")):
        return "unsafe_key_line_character"
    if FILE_SUFFIX_RE.search(key):
        return "file_suffix"
    if PURE_VERSION_RE.fullmatch(key):
        return "pure_version"
    if PURE_TECH_IDENTIFIER_RE.fullmatch(key):
        return "pure_english_tech_identifier"
    return ""


def _filter_keys(items: list[dict]) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    kept: list[str] = []
    filtered: list[dict] = []
    seen: set[str] = set()
    for item in items:
        key = str(item.get("key") or "").strip()
        reason = _key_rejection_reason(key)
        if reason:
            filtered.append({"key": key, "reason": reason})
            continue
        if key in seen:
            filtered.append({"key": key, "reason": "duplicate"})
            continue
        seen.add(key)
        kept.append(key)
    return tuple(kept), tuple(filtered)


def _frontmatter_bounds(text: str) -> tuple[int, int]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError("bucket is missing frontmatter")
    offset = len(lines[0])
    for line in lines[1:]:
        next_offset = offset + len(line)
        if line.rstrip("\r\n") == "---":
            return len(lines[0]), next_offset
        offset = next_offset
    raise ValueError("bucket frontmatter is not closed")


def _replace_name_line(text: str, new_name: str) -> str:
    if not new_name or any(character in new_name for character in ("\r", "\n")):
        raise ValueError("suggested name is unsafe")
    header_start, body_start = _frontmatter_bounds(text)
    header = text[header_start:body_start]
    pattern = re.compile(r"(?m)^name:[^\r\n]*(?:\r?\n|$)")
    matches = list(pattern.finditer(header))
    if len(matches) != 1:
        raise ValueError("bucket must have exactly one single-line name field")
    match = matches[0]
    old_line = match.group(0)
    ending = (
        "\r\n" if old_line.endswith("\r\n")
        else "\n" if old_line.endswith("\n")
        else ""
    )
    encoded_name = json.dumps(new_name, ensure_ascii=False)
    replacement = f"name: {encoded_name}{ending}"
    new_header = header[:match.start()] + replacement + header[match.end():]
    updated = text[:header_start] + new_header + text[body_start:]
    _, updated_body_start = _frontmatter_bounds(updated)
    if updated[updated_body_start:] != text[body_start:]:
        raise AssertionError("name replacement changed bucket body")
    return updated


def _append_key_line(text: str, keys: tuple[str, ...]) -> str:
    if not keys:
        raise ValueError("cannot append an empty retrieval-key line")
    _, body_start = _frontmatter_bounds(text)
    body = text[body_start:]
    if KEY_LINE_RE.search(body):
        raise ValueError("bucket already has a retrieval-key line")
    eol = "\r\n" if "\r\n" in text else "\n"
    separator = "" if text.endswith(("\n", "\r")) else eol
    key_line = f"[检索钥匙: {'/'.join(keys)}]{eol}"
    updated = text + separator + key_line
    _, updated_body_start = _frontmatter_bounds(updated)
    expected_body = body + separator + key_line
    if updated[updated_body_start:] != expected_body:
        raise AssertionError("retrieval-key append changed existing body")
    return updated


def _safe_target(vault: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] not in LIVE_ROOTS
    ):
        raise ValueError(f"unsafe plan path: {relative_path}")
    target = vault.joinpath(*relative.parts)
    resolved_vault = vault.resolve(strict=True)
    resolved_target = target.resolve(strict=True)
    if not resolved_target.is_relative_to(resolved_vault):
        raise ValueError(f"plan path escapes vault: {relative_path}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"target is not a regular bucket: {relative_path}")
    return target


def _load_approved_plan(
    plan_path: Path,
    approved_sha256: str,
    expected_proposals: int,
) -> tuple[dict, str]:
    actual_sha256 = _sha256_file(plan_path)
    if actual_sha256 != approved_sha256:
        raise ValueError(
            f"approved plan hash mismatch: {actual_sha256} != {approved_sha256}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA or plan.get("mode") != "dry_run_only":
        raise ValueError("unsupported or non-dry-run plan")
    if not plan.get("source", {}).get("unchanged"):
        raise ValueError("plan did not prove source stability")
    proposals = [
        item for item in plan.get("items", [])
        if item.get("status") == "propose"
    ]
    if len(proposals) != expected_proposals:
        raise ValueError(
            f"proposal count changed: {len(proposals)} != {expected_proposals}"
        )
    return plan, actual_sha256


def _domains(metadata: dict) -> list[str]:
    value = metadata.get("domain") or []
    return [value] if isinstance(value, str) else list(value)


def build_operations(
    vault: Path,
    plan: dict,
) -> tuple[list[Operation], list[dict], dict]:
    operations: list[Operation] = []
    skipped: list[dict] = []
    filtered_reasons: dict[str, int] = {}
    proposals = [
        item for item in plan["items"]
        if item.get("status") == "propose"
    ]
    for item in proposals:
        bucket_id = str(item["bucket_id"])
        target = _safe_target(vault, str(item["path"]))
        raw = target.read_text(encoding="utf-8")
        post = frontmatter.loads(raw)
        body = post.content or ""
        current_name = str(post.metadata.get("name") or "")
        if "未分类" not in _domains(dict(post.metadata)):
            raise ValueError(f"{bucket_id}: domain drifted")
        if current_name != str(item.get("current_name") or ""):
            raise ValueError(f"{bucket_id}: current name drifted")
        if _sha256_text(body) != item["body_sha256"]:
            raise ValueError(f"{bucket_id}: body drifted")
        if KEY_LINE_RE.search(body):
            raise ValueError(f"{bucket_id}: retrieval-key line appeared after review")
        for basis in item.get("name_basis") or []:
            if str(basis) not in body:
                raise ValueError(f"{bucket_id}: name basis is not literal")
        for key_item in item.get("retrieval_keys") or []:
            key = str(key_item.get("key") or "")
            evidence = str(key_item.get("evidence") or "")
            if key not in body or key not in evidence:
                raise ValueError(f"{bucket_id}: key evidence is not literal")

        kept_keys, filtered_keys = _filter_keys(item.get("retrieval_keys") or [])
        for filtered in filtered_keys:
            reason = filtered["reason"]
            filtered_reasons[reason] = filtered_reasons.get(reason, 0) + 1
        if not kept_keys:
            skipped.append({
                "bucket_id": bucket_id,
                "path": item["path"],
                "reason": "all_keys_filtered",
                "filtered_keys": list(filtered_keys),
            })
            continue

        name_action = item.get("name_action")
        suggested_name = str(item.get("suggested_name") or "")
        if name_action == "replace":
            with_name = _replace_name_line(raw, suggested_name)
            new_name = suggested_name
        elif name_action == "keep":
            with_name = raw
            new_name = current_name
        else:
            raise ValueError(f"{bucket_id}: unsupported name action")
        updated = _append_key_line(with_name, kept_keys)
        operations.append(Operation(
            bucket_id=bucket_id,
            relative_path=str(item["path"]),
            target=target,
            original_text=raw,
            original_sha256=_sha256_text(raw),
            new_text=updated,
            new_sha256=_sha256_text(updated),
            old_name=current_name,
            new_name=new_name,
            name_changed=current_name != new_name,
            kept_keys=kept_keys,
            filtered_keys=filtered_keys,
        ))

    summary = {
        "approved_proposals": len(proposals),
        "operations": len(operations),
        "skipped_after_filter": len(skipped),
        "name_changes": sum(operation.name_changed for operation in operations),
        "key_lines_to_append": len(operations),
        "keys_kept": sum(len(operation.kept_keys) for operation in operations),
        "keys_filtered": sum(filtered_reasons.values()),
        "keys_filtered_in_operations": sum(
            len(operation.filtered_keys) for operation in operations
        ),
        "keys_filtered_with_skipped": sum(
            len(item["filtered_keys"]) for item in skipped
        ),
        "filtered_reasons": filtered_reasons,
    }
    return operations, skipped, summary


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_manifest(path: Path, manifest: dict) -> None:
    atomic_write_text(
        path,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    os.chmod(path, 0o600)


def apply_operations(
    vault: Path,
    backup_root: Path,
    plan_sha256: str,
    operations: list[Operation],
    skipped: list[dict],
    summary: dict,
) -> dict:
    resolved_vault = vault.resolve(strict=True)
    resolved_backup_parent = backup_root.parent.resolve(strict=True)
    if resolved_backup_parent.is_relative_to(resolved_vault):
        raise ValueError("backup root must be outside the live vault")
    if backup_root.exists():
        raise FileExistsError(f"backup root already exists: {backup_root}")
    backup_root.mkdir(mode=0o700, parents=False)
    files_root = backup_root / "files"
    files_root.mkdir(mode=0o700)

    manifest = {
        "schema": SCHEMA,
        "status": "backing_up",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vault": str(vault),
        "backup_root": str(backup_root),
        "plan_sha256": plan_sha256,
        "summary": summary,
        "skipped": skipped,
        "operations": [],
    }
    for operation in operations:
        backup_path = files_root.joinpath(*PurePosixPath(
            operation.relative_path
        ).parts)
        _write_exclusive(
            backup_path,
            operation.original_text.encode("utf-8"),
        )
        if _sha256_file(backup_path) != operation.original_sha256:
            raise RuntimeError(f"{operation.bucket_id}: backup verification failed")
        manifest["operations"].append({
            "bucket_id": operation.bucket_id,
            "path": operation.relative_path,
            "backup_path": str(backup_path.relative_to(backup_root)),
            "before_sha256": operation.original_sha256,
            "after_sha256": operation.new_sha256,
            "old_name": operation.old_name,
            "new_name": operation.new_name,
            "name_changed": operation.name_changed,
            "kept_keys": list(operation.kept_keys),
            "filtered_keys": list(operation.filtered_keys),
        })

    manifest_path = backup_root / "manifest.json"
    manifest["status"] = "backed_up"
    _write_manifest(manifest_path, manifest)

    written: list[Operation] = []
    try:
        for operation in operations:
            if _sha256_file(operation.target) != operation.original_sha256:
                raise RuntimeError(
                    f"{operation.bucket_id}: raw file drifted before apply"
                )
            atomic_write_text(operation.target, operation.new_text)
            if _sha256_file(operation.target) != operation.new_sha256:
                raise RuntimeError(
                    f"{operation.bucket_id}: post-apply verification failed"
                )
            written.append(operation)
    except Exception as exc:
        rollback_errors: list[str] = []
        for operation in reversed(written):
            try:
                atomic_write_text(operation.target, operation.original_text)
                if _sha256_file(operation.target) != operation.original_sha256:
                    raise RuntimeError("restored hash mismatch")
            except Exception as rollback_exc:
                rollback_errors.append(
                    f"{operation.bucket_id}: {rollback_exc}"
                )
        manifest["status"] = (
            "rollback_failed" if rollback_errors else "rolled_back"
        )
        manifest["apply_error"] = repr(exc)
        manifest["rollback_errors"] = rollback_errors
        _write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "applied"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["applied_count"] = len(written)
    _write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an approved bucket-clothing plan with backups"
    )
    parser.add_argument("--buckets", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approved-plan-sha256", required=True)
    parser.add_argument("--expected-proposals", type=int, required=True)
    parser.add_argument("--backup-root")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    vault = Path(args.buckets).resolve()
    plan_path = Path(args.plan).resolve()
    if not vault.is_dir() or not plan_path.is_file():
        raise SystemExit("vault or plan is unavailable")
    plan, plan_sha256 = _load_approved_plan(
        plan_path,
        args.approved_plan_sha256,
        args.expected_proposals,
    )

    if args.apply:
        if not args.backup_root:
            raise SystemExit("--backup-root is required with --apply")
        backup_root = Path(args.backup_root).resolve()
        with MaintenanceBarrier(vault).exclusive():
            operations, skipped, summary = build_operations(vault, plan)
            manifest = apply_operations(
                vault,
                backup_root,
                plan_sha256,
                operations,
                skipped,
                summary,
            )
        result = {
            "mode": "apply",
            "status": manifest["status"],
            "manifest": str(backup_root / "manifest.json"),
            "summary": summary,
        }
    else:
        operations, skipped, summary = build_operations(vault, plan)
        result = {
            "mode": "preflight_only",
            "status": "ready",
            "summary": summary,
            "skipped_bucket_ids": [
                item["bucket_id"] for item in skipped
            ],
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
