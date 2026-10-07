"""Parallel execution utilities for federated training and evaluation rounds."""

import concurrent.futures
from logging import ERROR
from typing import Optional

from fedml.utils.typing import Code
from fedml.utils.logger import log


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------

def fit_clients(executor, client_instructions, max_workers: Optional[int], group_id: int):
    """Refine parameters concurrently on all selected clients."""
    results, failures = [], []

    if executor is None:
        for client_ins in client_instructions:
            result = fit_client(*client_ins, group_id)
            _, res = result
            if res.status.code == Code.OK:
                results.append(result)
            else:
                failures.append(result)
    else:
        submitted_fs = {
            executor.submit(fit_client, *client_ins, group_id)
            for client_ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)
        for future in finished_fs:
            _handle_finished_future(future=future, results=results, failures=failures)

    # Sort by client_id so aggregation order is deterministic regardless of
    # task completion order (otherwise float-sum order drifts run-to-run).
    results.sort(key=lambda r: r[1].metrics["client_id"])
    return results, failures


def fit_client(client, fit_ins, local_model, model_as_fn, run_device, group_id: int):
    """Refine parameters on a single client."""
    fit_res = client.fit(fit_ins, local_model, model_as_fn, run_device)
    return client, fit_res


# ------------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------------

def evaluate_clients(executor, client_instructions, max_workers: Optional[int], group_id: int):
    """Evaluate parameters concurrently on all selected clients."""
    results, failures = [], []

    if executor is None:
        for client_ins in client_instructions:
            result = evaluate_client(*client_ins, group_id)
            _, res = result
            if res.status.code == Code.OK:
                results.append(result)
            else:
                failures.append(result)
    else:
        submitted_fs = {
            executor.submit(evaluate_client, *client_ins, group_id)
            for client_ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)
        for future in finished_fs:
            _handle_finished_future(future=future, results=results, failures=failures)

    # Sort by client_id so aggregation order is deterministic regardless of
    # task completion order.
    results.sort(key=lambda r: r[1].metrics["client_id"])
    return results, failures


def evaluate_client(client, eval_ins, local_model, model_as_fn, run_device, group_id: int):
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(eval_ins, local_model, model_as_fn, run_device)
    return client, evaluate_res


# ------------------------------------------------------------------
# Post-training callbacks
# ------------------------------------------------------------------

def post_training(executor, client_instructions, results, failures):
    """Run post-training callbacks concurrently on all selected clients."""
    post_results, post_failures = [], []

    if executor is None:
        for client, *_ in client_instructions:
            result = callback_client(client, results, failures)
            _, res = result
            if res.status.code == Code.OK:
                post_results.append(result)
            else:
                post_failures.append(result)
    else:
        submitted_fs = {
            executor.submit(callback_client, client, results, failures)
            for client, *_ in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(fs=submitted_fs, timeout=None)
        for future in finished_fs:
            _handle_finished_future(future=future, results=post_results, failures=post_failures)

    # Sort by client_id so aggregation order is deterministic regardless of
    # task completion order.
    post_results.sort(key=lambda r: r[1].metrics["client_id"])
    return post_results, post_failures


def callback_client(client, results, failures):
    """Run post-training callback on a single client."""
    fit_res = client.post_training_callback(results, failures)
    return client, fit_res


# ------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------

def _handle_failed_future(future: concurrent.futures.Future) -> Optional[BaseException]:
    """Convert a failed future into a failure."""
    failure = future.exception()
    if failure is not None:
        import traceback
        log(ERROR, str(failure) + "\n" + "".join(traceback.format_exception(failure)))
    return failure

def _handle_finished_future(future: concurrent.futures.Future, results, failures) -> None:
    """Convert a finished future into either a result or a failure."""

    # Check for exceptions raised during execution of the future
    failure = _handle_failed_future(future=future)
    if failure is not None:
        failures.append(failure)
        return

    result = future.result()
    _, res = result

    if res.status.code == Code.OK:
        results.append(result)
        return

    failures.append(result)
