"""Experiment manager — tracks hyperparameters, results, and serialization."""

import os
import numpy as np


class ExperimentManager:
    """Stores and manages hyperparameters and results for a single experiment."""

    def __init__(self, *, experiment_id: str, hyperparameters: dict = None) -> None:
        self.experiment_id = experiment_id
        self.hyperparameters = hyperparameters or {}
        self.results = {}
        self.parameters = {}

    def __str__(self) -> str:
        lines = ["Hyperparameters:"]
        for key, value in self.hyperparameters.items():
            lines.append(f"  {key:<24}{value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def log(self, update_dict: dict, nested: bool = False, printout: bool = False, override: bool = False) -> None:
        """Append values to the results store."""
        for key, value in update_dict.items():
            if key not in self.results or override:
                if nested and isinstance(value, dict):
                    self.results[key] = {k: [v] for k, v in value.items()}
                else:
                    self.results[key] = [value]
            else:
                if nested and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in self.results[key]:
                            self.results[key][sub_key] = [sub_value]
                        else:
                            self.results[key][sub_key].append(sub_value)
                else:
                    self.results[key].append(value)

        if printout:
            print(update_dict)

    def save_parameters(self, parameters) -> None:
        self.parameters = parameters

    def to_dict(self) -> dict:
        return {
            "hyperparameters": self.hyperparameters,
            "parameters": self.parameters,
            "results": self.results,
        }

    def from_dict(self, input_dict: dict) -> None:
        self.hyperparameters = input_dict["hyperparameters"][np.newaxis][0]
        self.parameters = input_dict["parameters"][np.newaxis][0]
        self.results = input_dict["results"][np.newaxis][0]

    def save_to_disc(self, path: str, filename: str, verbose: bool = False) -> None:
        """Save experiment results to a .npz file."""
        results_numpy = {key: np.array(value) for key, value in self.to_dict().items()}
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, filename), **results_numpy)
        if verbose:
            print(f"Saved results to {path}{filename}.npz")

    def load_from_disc(self, path: str, filename: str, verbose: bool = False) -> None:
        """Load experiment results from a .npz file."""
        self.from_dict(np.load(os.path.join(path, filename), allow_pickle=True))
        if verbose:
            print(f"Loaded results from {path}{filename}")