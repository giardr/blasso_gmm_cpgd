"""
export_raw_results.py
"""

from typing import TypeAlias
import numpy as np
from numpy.typing import NDArray
import pickle

DataRow: TypeAlias = tuple[
    tuple[float, int, float, float, NDArray[np.float64], int],
    tuple[dict[str, float], float],
]

def save_data(data: list[DataRow], file_name: str, dtype: type = np.float64) -> None:
    """ Save experiment's data in the given file
    
    It flatten all values in one Numpy array and store it using numpy.save_compressed.
    The advantage compare to pickle is to have a better long-term compatibility.

    Additionally, the floating-point precision can be reduced to get a smaller file.
    """
    flatten_fdata = np.empty((len(data), 16), dtype=dtype)
    flatten_idata = np.empty((len(data), 2), dtype=np.uint64)

    for row, flatten_frow, flatten_irow in zip(data, flatten_fdata, flatten_idata):
        hparams, (scores, last_loss) = row
        flatten_frow[0] = hparams[0] # u0
        flatten_irow[0] = hparams[1] # seed
        flatten_frow[1] = hparams[2] # tau
        flatten_frow[2] = hparams[3] # kappa
        flatten_frow[3:12] = hparams[4].flatten() # init
        flatten_irow[1] = hparams[5] # n_samples
        flatten_frow[12] = scores['semi-distance']
        flatten_frow[13] = scores['Fisher-Rao']
        flatten_frow[14] = scores['Euclidean']
        flatten_frow[15] = last_loss

    np.savez_compressed(file_name, fdata=flatten_fdata, idata=flatten_idata)

def load_data(file_name: str) -> list[DataRow]:
    loaded_data = np.load(file_name)
    flatten_fdata = loaded_data["fdata"]
    flatten_idata = loaded_data["idata"]
    data: list[DataRow] = []
    for flatten_frow, flatten_irow in zip(flatten_fdata, flatten_idata):
        init: NDArray[np.float64] = flatten_frow[3:12].reshape(3, 3).astype(np.float64)
        hparams = (
            float(flatten_frow[0]), # u0
            int(flatten_irow[0]), # seed
            float(flatten_frow[1]), # tau
            float(flatten_frow[2]), # kappa
            init,
            int(flatten_irow[1]), # n_samples
        )
        scores = {
            'semi-distance': float(flatten_frow[12]),
            'Fisher-Rao': float(flatten_frow[13]),
            'Euclidean': float(flatten_frow[14]),
        }
        last_loss = float(flatten_frow[15])
        row: DataRow = (hparams, (scores, last_loss))
        data.append(row)
    return data

# =========================
# Main execution
# =========================
if __name__ == "__main__":
    checkpoint_file = "results_grid_search.pkl"
    with open(checkpoint_file, "rb") as f:
        results = pickle.load(f)
        print("Recovered", len(results), "entries")
    save_data(results, "results_grid_search_f64.npz")
