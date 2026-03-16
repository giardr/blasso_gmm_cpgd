# CPGD

We provide the code associated with the paper *Riemannian Gradient Descent for Gaussian Mixture Models with unknown diagonal covariances*, by Romane Giard, Yohann De Castro, Roland Denis and Clément Marteau.

## Requirements

The code was tested with Python 3.12.

Required Python packages:

* pytorch
* numpy
* scipy
* matplotlib
* scikit-learn
* sympy
* tqdm
* jaxtyping
* POT (Python Optimal Transport)
* jupyter


## Files

| File                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `cpgd_sympy.ipynb`        | Symbolic computations to verify the calculations presented in the proofs of the article,<br>and to obtain formulas used in `methods.py`. |
| `cpgd_experiments.ipynb`  | Numerical experiments with the CPGD algorithm;<br>results discussed in Section 4 of the article. |
| `methods.py`              | Core algorithm implementations.                                             |
| `run_grid_search.py`      | Script used to generate raw results for Section 4.6 of the article (influence of separation and sample size on performance).<br>Results are processed in `cpgd_experiments.ipynb`.<br>The file containing the raw results is available upon request. |