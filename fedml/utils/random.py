"""Utilities for reproducible random seed setup."""

import random
import numpy as np
import torch


def setup_random_seeds(seed_value: int = 333, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    When ``deterministic`` is True, also forces cuDNN into deterministic mode.
    This is left off by default because it can slow down GPU training, and the
    per-(client, round) seeding is enough for statistical stability. Enable it
    when bit-level GPU reproducibility is required.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, client_id: int, server_round: int, salt: int = 0) -> int:
    """Derive a deterministic 32-bit seed from a base seed and context.

    The same (base_seed, client_id, server_round, salt) always maps to the
    same seed, regardless of which worker process or thread runs it, or in
    what order results arrive. This is what makes per-client local work
    reproducible even though ``spawn``-ed ProcessPool workers never inherit
    the parent's RNG state.

    ``salt`` separates independent random streams that share the same context
    (e.g. salt=0 for local training, salt=1 for the attack on/off decision).
    """
    seed_sequence = np.random.SeedSequence(
        [int(base_seed), int(client_id), int(server_round), int(salt)]
    )
    return int(seed_sequence.generate_state(1)[0])
