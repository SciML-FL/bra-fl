"""Expand compact sweep definitions into validated experiment YAML files."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
from functools import reduce
from itertools import product
from pathlib import Path
import sys


# Make the repository package importable regardless of the caller's cwd.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fedml import configs
from fedml.configs.parser import LiteralList


class SkipConfig(Exception):
    """Veto an invalid or redundant Cartesian-product combination."""


# Maps the Bayesian ablation level to the argument used by each strategy.
# Known parameterless strategies are listed separately and retain only the
# level-zero representative; unknown strategies fail validation.
ABLATION_PARAM = {
    "FED-KRUM": "num_malicious_clients",
    "FED-MDA": "num_malicious_clients",
    "FED-MEAMED": "num_malicious_clients",
    "FED-CAF": "num_malicious_clients",
    "FED-SMEA": "num_malicious_clients",
    "FED-MIXING": "num_malicious_clients",
    "FED-TRIMAVG": "beta",
}

PARAMETERLESS_ABLATION_STRATEGIES = {
    "FED-AVERAGE",
    "FED-MEDIAN",
    "FED-GEOMED",
    "FED-BAYESIAN",
}


def parse_mapping(config_path, description: str) -> dict:
    """Parse a YAML file and require a mapping at its root."""
    parsed = configs.parse_configs(config_path)
    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"{description} must parse to a YAML mapping: {config_path}"
        )
    return dict(parsed)


def find_keypaths(nested_dict: dict, key_to_find: str, prepath=()):
    """Collect every path to ``key_to_find`` below ``prepath``."""
    sub_dict = reduce(lambda in_dict, key: in_dict[key], prepath, nested_dict)
    matches = []
    for key, value in sub_dict.items():
        path = prepath + (key,)
        if key == key_to_find:
            matches.append(path)
        elif hasattr(value, "items"):
            matches.extend(find_keypaths(nested_dict, key_to_find, path))
    return matches


def resolve_keypath(template: dict, key: str):
    """Resolve one slash-qualified builder key or fail on missing ambiguity."""
    *prefix, leaf = key.split("/")
    matches = find_keypaths(template, leaf, tuple(prefix))
    if not matches:
        raise KeyError(
            f"'{key}': leaf '{leaf}' not found under {tuple(prefix) or '<root>'}"
        )
    if len(matches) > 1:
        raise KeyError(
            f"'{key}' is ambiguous -- matches {matches}. "
            "Add a path prefix to disambiguate."
        )
    return matches[0]


def update_dict(nested_dict: dict, keypath, value):
    """Update a value at ``keypath`` in a nested dictionary."""
    if len(keypath) == 1:
        nested_dict[keypath[0]] = value
    else:
        update_dict(nested_dict[keypath[0]], keypath[1:], value)
    return nested_dict


def post_processing(config_dict: dict):
    """Derive dependent parameters and prune invalid combinations."""
    server_configs = config_dict["SERVER_CONFIGS"]
    experiment_configs = config_dict.get("EXPERIMENT_CONFIGS", {})

    # A benign experiment is exactly (fraction == 0, attack type is null).
    # A malicious experiment is exactly (fraction > 0, attack type is set).
    if (
        "MAL_CLIENT_FRAC" in experiment_configs
        and "MAL_CLIENT_TYPE" in experiment_configs
    ):
        benign = experiment_configs["MAL_CLIENT_FRAC"] == 0.0
        has_attack_type = experiment_configs["MAL_CLIENT_TYPE"] is not None
        if benign == has_attack_type:
            raise SkipConfig()

    # Route the C2 ablation level into the strategy-specific argument.
    if server_configs.get("_ABLATION_NMAL") is not None:
        level = server_configs["_ABLATION_NMAL"]
        strategy = server_configs["AGGREGATE_STRAT"]
        total_clients = server_configs["MIN_TRAINING_SAMPLE_SIZE"]
        parameter = ABLATION_PARAM.get(strategy)

        if parameter is None:
            if strategy not in PARAMETERLESS_ABLATION_STRATEGIES:
                raise ValueError(
                    f"Unknown ablation behavior for strategy {strategy!r}. "
                    "Add it to ABLATION_PARAM or "
                    "PARAMETERLESS_ABLATION_STRATEGIES."
                )
            if level != 0:
                raise SkipConfig()
        else:
            strategy_args = dict(server_configs["AGGR_STRAT_ARGS"])
            if parameter == "num_malicious_clients":
                strategy_args["num_malicious_clients"] = level
                if strategy == "FED-KRUM":
                    strategy_args["num_clients_to_keep"] = total_clients - level
            elif parameter == "beta":
                strategy_args["beta"] = round(level / total_clients, 4)
            server_configs["AGGR_STRAT_ARGS"] = strategy_args

    # Scratch builder fields must never leak into runnable configurations.
    server_configs.pop("_ABLATION_NMAL", None)


def build_experiment_configs(args, prefix: str = "exp") -> int:
    """Build one grid and return its final cumulative experiment ID.

    An existing destination filename is rejected by default, while cumulative
    calls may safely append non-overlapping ID ranges to the same directory.
    """
    base_template = parse_mapping(args.base_template, "Base template")
    build_configs = parse_mapping(args.build_configs, "Build configuration")

    config_keys = []
    config_keypaths = []
    config_axes = []

    for key, value in build_configs.items():
        config_keys.append(key)
        if isinstance(key, str):
            config_keypaths.append(resolve_keypath(base_template, key))
            config_axes.append(
                [value]
                if isinstance(value, LiteralList)
                else (value if isinstance(value, list) else [value])
            )
        elif isinstance(key, tuple):
            keypaths = [resolve_keypath(base_template, item) for item in key]
            coupled_values = [tuple(item) for item in value]
            for coupled_value in coupled_values:
                if len(coupled_value) != len(keypaths):
                    raise ValueError(
                        f"Coupled builder key {key!r} has {len(keypaths)} fields, "
                        f"but row {coupled_value!r} has {len(coupled_value)} values."
                    )
            config_keypaths.append(keypaths)
            config_axes.append(coupled_values)
        else:
            raise TypeError(f"Unsupported builder key type: {key!r}")

    raw_count = reduce(lambda total, axis: total * len(axis), config_axes, 1)
    quiet = bool(getattr(args, "quiet", False))
    if not quiet:
        print(
            f"EXP-ID - up to {raw_count} configs | "
            + " x ".join(
                f"{key}({len(axis)})" for key, axis in zip(config_keys, config_axes)
            )
        )

    dry_run = bool(getattr(args, "dry_run", False))
    overwrite_existing = bool(getattr(args, "overwrite_existing", False))
    output_path = Path(args.output_path)

    written = 0
    skipped = 0
    generated_configs = []
    for combination in product(*config_axes):
        new_config = copy.deepcopy(base_template)
        for index, value in enumerate(combination):
            keypath = config_keypaths[index]
            if isinstance(keypath, tuple):
                update_dict(new_config, keypath, value)
            elif isinstance(keypath, list):
                for coupled_index, coupled_keypath in enumerate(keypath):
                    update_dict(new_config, coupled_keypath, value[coupled_index])
            else:
                raise TypeError(
                    f"Unexpected keypath for {config_keys[index]!r}: {keypath!r}"
                )

        try:
            post_processing(new_config)
        except SkipConfig:
            skipped += 1
            continue

        written += 1
        experiment_id = args.offset + written
        server_configs = new_config.get("SERVER_CONFIGS", {})
        summary = (
            server_configs.get("AGGREGATE_STRAT"),
            server_configs.get("AGGR_STRAT_ARGS"),
        )
        if not quiet:
            print(f"{experiment_id:04d} - {combination} -> {summary}")

        if not dry_run:
            destination = output_path / f"{prefix}_{experiment_id:04d}.yaml"
            generated_configs.append((destination, new_config))

    if not dry_run:
        collisions = [
            destination
            for destination, _ in generated_configs
            if destination.exists()
        ]
        if collisions and not overwrite_existing:
            preview = ", ".join(str(path) for path in collisions[:3])
            suffix = " ..." if len(collisions) > 3 else ""
            raise FileExistsError(
                f"Refusing to overwrite {len(collisions)} existing config(s): "
                f"{preview}{suffix}. Pass --overwrite-existing deliberately."
            )

        output_path.mkdir(parents=True, exist_ok=True)
        for destination, generated_config in generated_configs:
            configs.store_configs(generated_config, destination)

    action = "would write" if dry_run else "wrote"
    print(f"  -> {action} {written}, pruned {skipped} (of {raw_count})")
    return args.offset + written


def parse_args():
    parser = argparse.ArgumentParser(
        description="Expand one experiment sweep into runnable YAML configs."
    )
    parser.add_argument("--base-template", required=True)
    parser.add_argument("--build-configs", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prefix", default="exp")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final count instead of every parameter combination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview the grid without creating files.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacement of matching YAML files in the output directory.",
    )
    return parser.parse_args()


def main(args=None, prefix: str | None = None) -> int:
    if args is None:
        args = parse_args()
    return build_experiment_configs(args, prefix or getattr(args, "prefix", "exp"))


if __name__ == "__main__":
    main()
