"""
run_grid_search.py
"""

import torch
import itertools
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import shutil

from methods import *

torch.set_default_dtype(torch.float64)

# It seems necessary to limit the number of threads sooner even if it set higher later in the code
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# -------------------------
# Core computation function
# -------------------------
def core_computations(
    u0, seed, tau, kappa, init, n_samples, n_iter=3000, num_threads=None
):
    if num_threads is None:
        import os

        num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)

    init_torch = torch.tensor(init)

    # Target GMM
    means_target = torch.tensor([[-1.0], [1.0]])
    sigmas_target = torch.tensor([[u0], [u0]])
    positions_target = torch.cat([means_target, sigmas_target], dim=-1)
    weights_target = 0.5 * torch.ones(2)

    target_gmm = make_diagonal_gmm(weights_target, means_target, sigmas_target)
    torch.manual_seed(seed)
    samples = target_gmm.sample((n_samples,))

    # Optimization
    model = DiagonalGMMBlasso(kappa=kappa, y=samples, tau=tau)
    last_loss, last_estimate = adagrad_conic_optim(
        model=model,
        positions_init=init_torch[:, :-1],
        weights_init=init_torch[:, -1],
        n_iter=n_iter,
    )

    # Compute scores
    scores = {
        dist: compute_W1_score(
            positions_target,
            weights_target,
            last_estimate[:, :-1],
            last_estimate[:, -1],
            distance=dist,
        )
        for dist in ["semi-distance", "Fisher-Rao", "Euclidean"]
    }

    return (u0, seed, tau, kappa, init, n_samples), (scores, last_loss)


# -------------------------
# Wrapper for checkpointing
# -------------------------
def make_key(hyperparams):
    u0, seed, tau, kappa, init, n_samples = hyperparams
    return (u0, seed, tau, kappa, tuple(map(tuple, init)), n_samples)


def skip_completed(iterable, existing_keys):
    for h in iterable:
        if make_key(h) not in existing_keys:
            yield h


# =========================
# Grid search parameters
# =========================
tab_u0 = torch.logspace(-3, 1, steps=10).tolist()
tab_n_samples = torch.logspace(2, 5, steps=6).to(torch.int).tolist()

tab_tau = torch.logspace(-5, 0, steps=20).tolist()
tab_kappa = torch.logspace(-14, -1, steps=20).tolist()
tab_seed = [
    1214521201542216,
    1216194316,
    12251612046419,
    171214186,
    3412517412193,
    4212051621,
    223121453123,
    1216123512191918,
    9541663434,
    43128231216,
]

tab_inits = [
    torch.tensor([[2.0, 1.0, 1 / 3], [-2.0, 1.0, 1 / 3], [0.0, 1.0, 1 / 3]]),
    torch.tensor([[1.0, 1.0, 1 / 3], [-1.0, 1.0, 1 / 3], [0.0, 1.0, 1 / 3]]),
    torch.tensor([[-2.0, 2.0, 2 / 5], [-2.0, 2.0, 2 / 5], [0.0, 1.0, 1 / 5]]),
]
tab_inits = [init.numpy() for init in tab_inits]

# Prevent POT from using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    checkpoint_file = "results_grid_search.pkl"
    num_processes = 48  # adapt to number of CPU cores

    # Load previous results
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            results = pickle.load(f)
        existing_keys = {make_key(r[0]) for r in results}
        print(f"Resuming from checkpoint: {len(existing_keys)} tasks completed.")
    else:
        results = []
        existing_keys = set()

    # Lazy iterator for hyperparameters
    hyperparams_iter = itertools.product(
        tab_u0, tab_seed, tab_tau, tab_kappa, tab_inits, tab_n_samples
    )
    hyperparams_iter = skip_completed(hyperparams_iter, existing_keys)

    # Run tasks in parallel with checkpointing
    with ProcessPoolExecutor(
        max_workers=num_processes, max_tasks_per_child=1
    ) as executor:
        # Submit tasks lazily to avoid huge memory usage
        futures = {}
        for h in tqdm(hyperparams_iter, desc="Submitting tasks"):
            future = executor.submit(core_computations, *h, n_iter=3000, num_threads=1)
            futures[future] = h

        for i, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Computing tasks")
        ):
            result = future.result()
            results.append(result)

            # Save checkpoint after 1000 completed tasks
            if (i + 1) % 1000 == 0:
                with open(checkpoint_file + ".tmp", "wb") as f:
                    pickle.dump(results, f)
                shutil.move(checkpoint_file + ".tmp", checkpoint_file)

        # Last update of file
        with open(checkpoint_file + ".tmp", "wb") as f:
            pickle.dump(results, f)
            shutil.move(checkpoint_file + ".tmp", checkpoint_file)

    print(f"Grid search completed. Total results stored: {len(results)}")
