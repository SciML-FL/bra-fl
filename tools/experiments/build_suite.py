"""Build a complete experiment suite from a tracked manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import yaml

from tools.experiments.build_configs import build_experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_manifest(path: Path) -> dict:
    """Load and minimally validate a suite manifest."""
    with path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Suite manifest must be a YAML mapping: {path}")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported suite schema in {path}")
    if not isinstance(manifest.get("experiments"), list):
        raise ValueError(f"Manifest has no experiment list: {path}")
    return manifest


def resolve_output_root(manifest: dict, override: str | None) -> Path:
    configured = override or manifest.get("default_output_root")
    if not configured:
        raise ValueError("Manifest has no default_output_root; pass --output-root")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def build_suite(
    manifest_path: Path,
    *,
    output_root: str | None = None,
    dry_run: bool = False,
    overwrite_existing: bool = False,
    selected: set[str] | None = None,
    quiet: bool = False,
) -> int:
    """Build selected manifest entries and validate their expected ID ranges."""
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    manifest_root = manifest_path.parent
    destination_root = resolve_output_root(manifest, output_root)
    prefix = manifest.get("prefix", "exp")
    offset = int(manifest.get("start_offset", 0))
    built = 0

    known_entry_ids = {entry["id"] for entry in manifest["experiments"]}
    unknown_entry_ids = (selected or set()) - known_entry_ids
    if unknown_entry_ids:
        unknown = ", ".join(sorted(unknown_entry_ids))
        available = ", ".join(sorted(known_entry_ids))
        raise ValueError(
            f"Unknown manifest entry ID(s): {unknown}. Available: {available}"
        )

    for entry in manifest["experiments"]:
        entry_id = entry["id"]
        expected = entry["expected"]
        expected_first = int(expected["first_id"])
        expected_last = int(expected["last_id"])
        expected_count = int(expected["count"])

        if offset + 1 != expected_first:
            raise ValueError(
                f"{entry_id}: manifest offset gives {offset + 1}, "
                f"expected first ID {expected_first}"
            )

        if selected and entry_id not in selected:
            offset = expected_last
            continue

        args = SimpleNamespace(
            base_template=manifest_root / entry["template"],
            build_configs=manifest_root / entry["sweep"],
            output_path=destination_root / entry["output"],
            offset=offset,
            dry_run=dry_run,
            overwrite_existing=overwrite_existing,
            quiet=quiet,
        )
        final_id = build_experiment_configs(args, prefix=prefix)
        actual_count = final_id - offset
        if actual_count != expected_count or final_id != expected_last:
            raise ValueError(
                f"{entry_id}: generated {actual_count} configs through {final_id}; "
                f"expected {expected_count} through {expected_last}"
            )
        offset = final_id
        built += actual_count

    print(
        f"Suite {manifest['suite_id']}: "
        f"{'would build' if dry_run else 'built'} {built} config(s)"
    )
    return offset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and validate a tracked experiment-suite manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only suite/grid summaries, not every parameter combination.",
    )
    parser.add_argument(
        "--only",
        action="append",
        dest="selected",
        help="Build one manifest entry by ID; repeat to select multiple entries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_suite(
        args.manifest,
        output_root=args.output_root,
        dry_run=args.dry_run,
        overwrite_existing=args.overwrite_existing,
        selected=set(args.selected) if args.selected else None,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
