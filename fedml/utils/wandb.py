"""WandB logging utilities."""

import random


def log_to_wandb(
    user_configs: dict,
    experiment_name: str,
    experiment_manager,
) -> None:
    """Log experiment results to Weights & Biases (optional dependency)."""
    import wandb
    wandb_configs = user_configs["OUTPUT_CONFIGS"]["WANDB_CONFIGS"]

    wandb_run = wandb.init(
        project=wandb_configs["PROJECT"],
        name=f"fed-{experiment_name}-{random.randint(1000, 9999)}",
        config=user_configs,
        dir=wandb_configs["DIR"],
    )

    rounds = list(range(user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"]))

    for key, value in experiment_manager.results.items():
        if isinstance(value, dict):
            columns = list(value.keys())
            data = list(value.values())
            wandb_run.log({
                f"Client-Stats/{key}": wandb.plot.line_series(
                    xs=rounds,
                    ys=data,
                    keys=columns,
                    xname="Server Round",
                    title=key,
                    split_table=True,
                )
            })
        else:
            wandb_run.log({
                f"Centralized-Stats/{key}": wandb.plot.line(
                    table=wandb.Table(
                        data=[[x, y] for x, y in zip(rounds, value)],
                        columns=["server_round", key],
                    ),
                    x="Server Round",
                    y=key,
                    title=key,
                    split_table=True,
                )
            })

    wandb_run.finish()