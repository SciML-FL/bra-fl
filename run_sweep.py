"""Expand and run the Paper 2 sweeps locally, one experiment at a time."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from tools.experiments.build_suite import build_suite, load_manifest
from fedml.configs import parse_configs

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "papers/p02_bayesian_aggregation/experiments/2026_bayesian/suite.yaml"
DATA_FOLDERS = {
    "CIFAR-10": "cifar", "CIFAR-100": "cifar",
    "TINY-IMAGENET": "tiny-imagenet-200", "UCIML-PARKINSONS": "parkinsons",
}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_result(path: Path, config: dict, weights_path: Path | None = None) -> dict:
    """Verify this run's full-round schema before marking completion."""
    import numpy as np
    rounds = int(config["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"])
    server = config["SERVER_CONFIGS"]
    clients = max(int(server["MIN_NUM_CLIENTS"] * server["TRAINING_SAMPLE_FRACTION"]),
                  int(server["MIN_TRAINING_SAMPLE_SIZE"]))
    with np.load(path, allow_pickle=True) as archive:
        results = archive["results"].item()
    sampled = results.get("sampled", [])
    if len(sampled) != rounds:
        raise RuntimeError(f"Expected {rounds} completed fit rounds, found {len(sampled)}")
    if any(len(ids) != clients or len(set(ids)) != clients for ids in sampled):
        raise RuntimeError("A fit round has missing or duplicate client results")
    finite_losses = True
    finite_metrics = True
    if server["EVALUATE_SERVER"]:
        for key in ("centralized_loss", "centralized_accu"):
            values = np.asarray(results.get(key, []), dtype=float)
            if values.shape != (rounds,):
                raise RuntimeError(f"{key} does not cover every round")
            finite_metrics = finite_metrics and bool(np.isfinite(values).all())
            if key == "centralized_loss":
                finite_losses = bool(np.isfinite(values).all())
    finite_weights = None
    if weights_path is not None:
        import torch
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        finite_weights = bool(torch.isfinite(weights).all())
    return {"rounds": rounds, "clients_per_round": clients,
            "finite_centralized_losses": finite_losses,
            "finite_centralized_metrics": finite_metrics,
            "finite_final_weights": finite_weights,
            "finite_results": finite_metrics and finite_weights is not False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", help="Suite entry ID; repeat to select several")
    parser.add_argument("--dry-run", action="store_true", help="Validate the grid and show counts without writing or training")
    parser.add_argument("--limit", type=int, help="Run only the first N selected configurations; training settings stay unchanged")
    parser.add_argument("--data-root", default="work/data", help="Dataset root (relative paths are relative to this package)")
    parser.add_argument("--output-root", default="work/local-run", help="New or empty output directory; existing runs are never overwritten")
    parser.add_argument("--device", default="cpu", help="cpu (default), auto, cuda, or cuda:N")
    parser.add_argument("--workers", type=int, default=1, help="Client processes per experiment (default: 1)")
    parser.add_argument("--download", action="store_true", help="Permit CIFAR downloads and fetching an uncached UCI Parkinsons dataset")
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    if args.device not in {"cpu", "auto", "cuda"} and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("--device must be cpu, auto, cuda, or cuda:N")
    selected = set(args.only) if args.only else None
    manifest = load_manifest(SUITE)
    known = {entry["id"] for entry in manifest["experiments"]}
    if selected and selected - known:
        parser.error("Unknown --only value. Available: " + ", ".join(sorted(known)))
    entries = [entry for entry in manifest["experiments"] if not selected or entry["id"] in selected]
    count = sum(int(entry["expected"]["count"]) for entry in entries)
    queued = min(count, args.limit) if args.limit else count
    output = resolve_path(args.output_root)
    data = resolve_path(args.data_root)
    if args.dry_run:
        build_suite(SUITE, output_root=str(output / "configs"), dry_run=True,
                    selected=selected, quiet=True)
        print(f"Would run {queued} of {count} selected configurations on {args.device}; "
              f"ProcessPool workers={args.workers}.")
        print(f"Data: {data}\nOutput: {output}")
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be absent or empty: {output}. Choose a new --output-root.")
    output.mkdir(parents=True, exist_ok=True)
    build_suite(SUITE, output_root=str(output / "configs"), selected=selected, quiet=True)
    paths = sorted((output / "configs").rglob("*.yaml"), key=lambda p: p.stem)
    if args.limit:
        paths = paths[:args.limit]
    runtime_files = sorted((ROOT / "fedml").rglob("*.py")) + [Path(__file__)]
    runtime_hash = hashlib.sha256("\n".join(
        p.relative_to(ROOT).as_posix() + ":" + sha256(p) for p in runtime_files
    ).encode()).hexdigest()
    status = {"suite": manifest["suite_id"], "selected_entries": [e["id"] for e in entries],
              "selected_grid_count": count, "queued_count": len(paths), "device": args.device,
              "executor": "ProcessPool", "workers": args.workers,
              "runtime_sha256": runtime_hash, "suite_sha256": sha256(SUITE),
              "status": "running", "runs": []}
    status_path = output / "run-status.json"
    write_json(status_path, status)
    for index, config_path in enumerate(paths, 1):
        run_dir = output / "results" / config_path.stem
        run_dir.mkdir(parents=True)
        config = parse_configs(config_path)
        dataset = config["DATASET_CONFIGS"]
        dataset["DATASET_PATH"] = str(data / DATA_FOLDERS[dataset["DATASET_NAME"]])
        dataset["DATASET_DOWN"] = args.download
        config["CLIENT_CONFIGS"]["RUN_DEVICE"] = args.device
        config["SERVER_CONFIGS"]["RUN_DEVICE"] = args.device
        config["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"] = str(run_dir) + os.sep
        config["OUTPUT_CONFIGS"]["WANDB_LOGGING"] = False
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        record = {"config": str(config_path.relative_to(output)),
                  "config_sha256": sha256(config_path), "status": "running"}
        status["runs"].append(record)
        write_json(status_path, status)
        started = time.monotonic()
        try:
            dataset_path = Path(dataset["DATASET_PATH"])
            if dataset["DATASET_NAME"] == "UCIML-PARKINSONS" and not args.download:
                if not (dataset_path / "uciml_parkinsons.joblib").is_file():
                    raise RuntimeError("Parkinsons cache missing; use --download once to fetch it")
            if dataset["DATASET_NAME"] == "TINY-IMAGENET":
                if not all((dataset_path / name).is_dir() for name in ("train", "val")):
                    raise RuntimeError("Prepared Tiny ImageNet train/ and val/ directories are missing; see README")
            command = [sys.executable, "-m", "fedml.run_federated", "--config-file", str(config_path),
                       "--device", args.device, "--executor-type", "ProcessPool", "--max-workers", str(args.workers)]
            print(f"[{index}/{len(paths)}] {config_path.stem}: {args.device}, {args.workers} worker(s)", flush=True)
            with (run_dir / "process.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            record["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"Experiment exited {result.returncode}; inspect {run_dir / 'process.log'}")
            result_path = run_dir / (config_path.stem + ".npz")
            weights_path = run_dir / ("weights-" + config_path.stem + ".pt")
            if not result_path.is_file() or not weights_path.is_file() or weights_path.stat().st_size == 0:
                raise RuntimeError("Expected result and final-weight artifacts are missing")
            checks = verify_result(result_path, config, weights_path)
            record.update(checks)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            record["results_sha256"] = sha256(result_path)
            record["status"] = "complete" if checks["finite_results"] else "diverged"
            if record["status"] == "complete":
                write_json(run_dir / "complete.json", {**record, "runtime_sha256": runtime_hash})
            else:
                write_json(run_dir / "diverged.json", record)
        except Exception as exc:
            record.update(status="failed", error=str(exc), elapsed_seconds=round(time.monotonic() - started, 3))
            status["status"] = "failed"
            write_json(status_path, status)
            print(str(exc), file=sys.stderr)
            return 1
        write_json(status_path, status)
    status["status"] = "complete" if all(r["status"] == "complete" for r in status["runs"]) else "finished_with_divergence"
    write_json(status_path, status)
    print(f"{status['status']}: {len(paths)} experiment(s). Details: {status_path}")
    return 0 if status["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
