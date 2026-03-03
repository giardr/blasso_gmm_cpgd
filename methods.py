from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Self, TypeVar, Generic


import torch
from torch import Tensor
import torch.distributions as D
from jaxtyping import Float
import ot

# -------------------------
# Optimization utilities
# -------------------------

class Particles:
    r"""
    A class for encoding discrete measure
    $$
        \mu_omega = \sum_j \omega_j \delta_{x_j}
    $$
    where $omega_j$ is a weight and $x_j$ a position.

    """
    type Positions = Float[Tensor, "n_particles dim"]
    type Weights = Float[Tensor, " n_particles"]
    positions: Positions
    weights: Weights
    gradient_evaluation: Optional["Evaluation"]

    def __init__(
        self,
        positions: Positions,
        weights: Weights,
    ):
        n_particles, _ = positions.shape
        assert weights.shape == (n_particles,)

        super().__init__()
        self.positions = positions 
        self.weights = weights
        self.gradient_evaluation = None

    def tv_norm(self) -> Float[Tensor, "1"]:
        r"""
        Returns the TV-norm of the discrete particle measure, i.e. the sum
        of the absolute values of its weights:
        $$
            \Vert \mu_omega \Vert_{TV} = \sum_j | omega_j |
        $$
        """
        return self.weights.abs().sum()

    def clone(self, *, detach: bool = True) -> "Particles":
        """
        Return a copy of Particles

        If detach=True, the returned Particles are detached from the current computation graph
        """
        if detach:
            positions = self.positions.detach().clone()
            weights = self.weights.detach().clone()
        else:
            positions = self.positions.clone()
            weights = self.weights.clone()

        return Particles(
            positions=positions,
            weights=weights,
        )
    
TProblem = TypeVar("TProblem")

@dataclass
class Evaluation(Generic[TProblem]):
    problem: TProblem
    particles: "Particles"

@dataclass
class DiagonalGMMBlasso:
    """
    Class for encoding a BLASSO problem for a GMM with diagonal covariances.
    """
    kappa: float
    y: Tensor # in the code, it is the sample (X_1,..X_n). 
    tau: float

    def __post_init__(self):
        _, self.d = self.y.shape
        self.loss_offset = 1 / (self.tau**self.d * (2 * torch.pi) ** (self.d / 2))
        # upper bound on 1/2 * ||y||^2, to keep the loss positive.
        # Avoid complexity in n^2 for the computation of ||y||^2
        # The formula for 1/2 * ||y||^2 is
        # 1/2 * (torch.exp(-((self.y[:, None, :] - self.y[None, :, :])**2).sum(-1) / (2 * self.tau**2)).mean()
        # /(2 * torch.pi * self.tau**2)**(self.d/2))
    
    def __call__(self, particles: Particles) -> Evaluation[Self]:
        return self.Evaluation(self, particles)

    class Evaluation(Evaluation["DiagonalGMMBlasso"]):
        """
        Define an evaluation context when computing the loss associated to a problem
        on a specific particles configuration.

        The goal is to allow performance optimization and behavior customization
        through caching of the terms involved during the computation of the loss.
        """
        # all the expressions for derivatives, etc are available in closed form. See `cpgd_sympy.ipynb`, section II.
        def __init__(self, problem, particles):
            self.problem = problem
            self.particles = particles
                
        def loss(self):
            return self.proximity + self.lasso

        @cached_property
        def lasso(self):
            return self.problem.kappa * self.particles.tv_norm()

        @cached_property
        def proximity(self):
            return self.l2_cost
        
        @cached_property
        def l2_cost(self):
            w = self.particles.weights
            K = self.kernel
            k = self.scalar_products
            return 0.5 * w.matmul(K.matmul(w)) - w @ k + self.problem.loss_offset
        
        def backward(self) -> None:
            """Associate the current evaluation context to the particles.
            """
            self.particles.gradient_evaluation = self
        
        @cached_property
        def Jp(self) -> Float[Tensor, " n_particles"]:
            """The J' function applied to the particle positions"""
            return (
                self.problem.kappa
                + self.kernel @ self.particles.weights
                - self.scalar_products
            )

        @cached_property
        def inv_diag_metric(self):
            sigmas2 = self.particles.positions[:, self.problem.d :] ** 2
            denom = 2 * self.sigmas2_plus_tau2_over_2
            return torch.cat([denom, denom**2 / (2 * sigmas2)], dim=1)

        @cached_property
        def Jp_grad(self) -> Float[Tensor, "n_particles dim"]:
            """The Riemannian gradient of J' applied to the particle positions"""
            return self.inv_diag_metric * self.Jp_euclidean_grad

        @cached_property
        def Jp_euclidean_grad(self) -> Float[Tensor, "n_particles dim"]:
            """The gradient of J' applied to the particle positions"""
            sigmas = self.particles.positions[:, self.problem.d :]

            # compute \sum_{j=1}^n_particles omega_j nabla_t K(x,x_j) for x=x_1,...x_{n_particles}
            frac = -self.dmeans / (
                self.sigmas2_plus_tau2_over_2[:, None, :]
                + self.sigmas2_plus_tau2_over_2[None, :, :]
            )  # (n_particles, n_particles, d)
            Kw = self.kernel * self.particles.weights[None, :]
            grad_kernel_means = ((Kw[:, :, None]) * frac).sum(dim=1)  # (n_particles, d)

            # compute \sum_{j=1}^n_particles omega_j nabla_u K(x,x_j) for x=x_1,...x_{n_particles}
            part1 = sigmas[:, None, :] * frac**2
            part2 = -sigmas[:, None, :] / (
                self.sigmas2_plus_tau2_over_2[:, None, :]
                + self.sigmas2_plus_tau2_over_2[None, :, :]
            ) + sigmas[:, None, :] / (2 * self.sigmas2_plus_tau2_over_2[:, None, :])
            grad_kernel_sigmas = ((Kw[:, :, None]) * (part1 + part2)).sum(
                dim=1
            )  # (n_particles, d)

            # euclidean gradient of scalar_product, wrt means
            scalar_products_per_obs = self.scalar_products_per_obs
            per_observation = (
                -self.dmeans_samples / (self.sigmas2_plus_tau2[:, None, :])
            )  # (n_particles, n, d)
            grad_scalar_products_means = (
                scalar_products_per_obs[:, :, None] * per_observation
            ).mean(dim=1)  # (n_particles, d)

            # euclidean gradient of scalar_product, wrt sigmas
            part1 = sigmas[:, None, :] * per_observation**2
            part2 = sigmas / (2 * self.sigmas2_plus_tau2_over_2) - sigmas / (
                self.sigmas2_plus_tau2
            )
            grad_scalar_products_sigmas = (
                scalar_products_per_obs[:, :, None] * (part1 + part2[:, None, :])
            ).mean(dim=1)  # (n_particles, d)

            return torch.cat(
                (
                    grad_kernel_means - grad_scalar_products_means,
                    grad_kernel_sigmas - grad_scalar_products_sigmas,
                ),
                dim=1,
            )

        @cached_property
        def dmeans(self) -> Float[Tensor, "n_particles n_particles d"]:
            """pairwise differences mean_i - mean_j"""
            means = self.particles.positions[:, : self.problem.d]
            return means[:, None, :] - means[None, :, :]

        @cached_property
        def dmeans2(self) -> Float[Tensor, "n_particles n_particles d"]:
            return self.dmeans**2

        @cached_property
        def sigmas2_plus_tau2_over_2(self) -> Float[Tensor, "n_particles d"]:
            """sigma_j^2 + tau^2/2"""
            sigmas = self.particles.positions[:, self.problem.d :]
            return sigmas**2 + self.problem.tau**2 / 2

        @cached_property
        def sigmas2_plus_tau2(self) -> Float[Tensor, "n_particles d"]:
            """sigma_j^2 + tau^2"""
            return self.sigmas2_plus_tau2_over_2 + self.problem.tau**2 / 2

        @cached_property
        def kernel(self) -> Float[Tensor, "n_particles n_particles"]:
            # pairwise sigma_i^2 + sigma_j^2 + tau^2
            sigma2_i_plus_sigma2_j_plus_tau2 = (
                self.sigmas2_plus_tau2_over_2[:, None, :]
                + self.sigmas2_plus_tau2_over_2[None, :, :]
            )

            # factor = (2sigma_k^2 + tau^2)^{1/4}
            factor = (2 * self.sigmas2_plus_tau2_over_2) ** 0.25  # (n_particles,d)

            # exponent term
            num = torch.exp(-self.dmeans2 / (2 * sigma2_i_plus_sigma2_j_plus_tau2))
            # denominator sqrt(u_i^2+u_j^2+tau^2)
            denom = torch.sqrt(sigma2_i_plus_sigma2_j_plus_tau2)

            per_dim = (
                factor[:, None, :] * factor[None, :, :] * num / denom
            )  # (n_particles,n_particles,d)

            # kernel is product over dimensions
            return per_dim.prod(dim=-1)

        @cached_property
        def dmeans_samples(self) -> Float[Tensor, " n_particles n d"]:
            means = self.particles.positions[:, : self.problem.d]
            return means[:, None, :] - self.problem.y[None, :, :]

        @cached_property
        def dmeans_samples2(self) -> Float[Tensor, " n_particles n d"]:
            return self.dmeans_samples**2

        @cached_property
        def scalar_products_per_obs(self) -> Float[Tensor, " n_particles"]:
            denom = self.sigmas2_plus_tau2[:, None, :]  # (n_particles, 1, d)

            per_dim = (
                (2 * self.sigmas2_plus_tau2_over_2[:, None, :]) ** 0.25
                * torch.exp(-self.dmeans_samples2 / (2 * denom))
                / (torch.sqrt(denom) * (2 * torch.pi) ** 0.25)
            )  # (n_particles, n , d)

            return per_dim.prod(dim=2)  # (n_particles, n)

        @cached_property
        def scalar_products(self) -> Float[Tensor, " n_particles"]:
            return self.scalar_products_per_obs.mean(dim=1)  # (n_particles,)

        @cached_property
        def Jp2(self) -> Float[Tensor, " n_particles"]:
            return self.Jp**2

        @cached_property
        def Jp_grad2(self) -> Float[Tensor, " n_particles dim"]:
            return self.Jp_grad * self.Jp_euclidean_grad

        def return_weights_omega_to_a(self) -> Float[Tensor, " n_particles"]:
            """
            To go from omega_j to a_j.
            No need to keep the gradients as we only do the optimization on omega_j.

            """
            weights = self.particles.weights.detach()
            sigmas2_plus_tau2_over_2 = self.sigmas2_plus_tau2_over_2.detach()
            return weights * ((4 * torch.pi * sigmas2_plus_tau2_over_2) ** 0.25).prod(
                dim=-1
            )

        def transform_weights_a_to_omega(self) -> None:
            """
            To go from a_j to omega_j.
            """
            weights = self.particles.weights.detach()
            sigmas2_plus_tau2_over_2 = self.sigmas2_plus_tau2_over_2
            with torch.no_grad():
                self.particles.weights.copy_(
                    weights
                    / ((4 * torch.pi * sigmas2_plus_tau2_over_2) ** 0.25).prod(dim=-1)
                )

        def Jp_on_x(self,x):
            """
            x of size m * 2d
            """
            d=self.problem.d
            tau2 = self.problem.tau**2
            sigma2_plus_tau2_over_2=x[:,d:]**2 + tau2/2
            sigma2_mu_plus_tau2_over_2=self.particles.positions[:,d:]**2 + tau2/2
            dmeans2=(self.particles.positions[None,:,:d]-x[:,None,:d])**2

            #kernel
            sigma2_mu_plus_sigma2_plus_tau2 = (
                sigma2_plus_tau2_over_2[:,None,:] + sigma2_mu_plus_tau2_over_2[None,:,:]
            )
            factor = (2 * sigma2_mu_plus_tau2_over_2[None,:,:]) ** 0.25 * (2 * sigma2_plus_tau2_over_2[:,None,:])** 0.25
            num = torch.exp(-dmeans2 / (2 * sigma2_mu_plus_sigma2_plus_tau2))
            denom = torch.sqrt(sigma2_mu_plus_sigma2_plus_tau2)
            per_dim = (
                factor * num / denom
            )
            kernel = per_dim.prod(dim=-1)

            #scalar product
            denom = sigma2_plus_tau2_over_2[:, None, :] + tau2/2
            dmeans_samples2 = (x[:,None,:d]-self.problem.y[None,:,:])**2
            per_dim = (
                (2 * sigma2_plus_tau2_over_2[:, None, :]) ** 0.25
                * torch.exp(-dmeans_samples2 / (2 * denom))
                / (torch.sqrt(denom) * (2 * torch.pi) ** 0.25)
            ) 
            scalar_product = per_dim.prod(dim=2).mean(dim=1)  # (n_particles, n)

            return (self.problem.kappa
                    + kernel @ self.particles.weights
                    - scalar_product)


@dataclass
class AdagradConicOptimizer:
    """
    Optimizer applied to Particles. AdaGrad with some modifications. 
    """
    particles: Particles
    eta: float
    eps: float = 1e-8

    def __post_init__(self):
        # Initialize accumulators with same shape as parameters
        self.v_positions = torch.zeros_like(self.particles.weights)
        self.v_weights = torch.zeros_like(self.particles.weights)

    def step(self):
        positions = self.particles.positions
        n_particles, total_dim = positions.shape
        d = total_dim // 2
        weights = self.particles.weights

        eta = self.eta

        with torch.no_grad():
            eval = self.particles.gradient_evaluation
            self.v_positions.add_(torch.mean(eval.Jp_grad2, dim=-1))
            self.v_weights.add_(eval.Jp2)

            # Scaled steps
            dpositions = (
                -eta
                * eval.Jp_grad
                * 1
                / (self.v_positions.sqrt().unsqueeze(-1) + self.eps)
            )
            dweights = -eta * eval.Jp * 1 / (self.v_weights.sqrt() + self.eps)

            # Retract
            weights *= torch.exp(dweights)
            positions += dpositions

            # Take absolute values for u_j
            positions[:, d:].copy_(positions[:, d:].abs())


def adagrad_conic_optim(model, positions_init, weights_init, n_iter, store_history=False, eta=1.0):
    """
    Runs several optimization steps using AdaGradConicOptimizer for a given model.

    The optimization is performed on a model of type `DiagonalGMM`,
    starting from the provided initial positions and weights.
    """
    #outside this function, we use the parametrization a for the weights (not omega)
    n_particles, total_dim = positions_init.shape
    assert model.d == total_dim // 2

    particles = Particles(
        weights=weights_init.detach().clone(), positions=positions_init.detach().clone()
    )
    model_eval_init = model(particles)
    model_eval_init.transform_weights_a_to_omega()

    optim = AdagradConicOptimizer(particles=particles, eta=eta)

    if store_history:
        iter_loss = torch.zeros(n_iter + 1)
        iter_particles = torch.zeros(
            n_iter + 1, n_particles, particles.positions.shape[1] + 1
        )

    for it in range(n_iter):
        model_eval = model(particles)
        current_loss = model_eval.loss()

        last_estimate = torch.cat(
                [
                    particles.positions.detach().clone(),
                    model_eval.return_weights_omega_to_a().unsqueeze(1),
                ],
                dim=1,
            )
            
        if torch.isnan(current_loss):
            print("NaN detected — stopping.")
            if store_history:
                iter_loss[it] = torch.inf
                iter_particles[it] = last_estimate
                return iter_loss, iter_particles
            return torch.inf, last_estimate  # break
        
        if store_history:
            iter_particles[it] = last_estimate
            iter_loss[it] = current_loss

        model_eval.backward()
        optim.step()

    model_eval = model(particles)
    last_loss = model_eval.loss()
    last_estimate = torch.cat(
        [
            particles.positions.detach().clone(),
            model_eval.return_weights_omega_to_a().unsqueeze(1),
        ],
        dim=1,
    )
    if store_history:
        iter_loss[-1] = last_loss
        iter_particles[-1] = last_estimate
        return iter_loss, iter_particles
    else:
        return last_loss, last_estimate

# -------------------------
# GMM utilities, score
# -------------------------

def make_diagonal_gmm(weights, means, sigmas):
    mix = D.Categorical(weights)
    comp = D.MultivariateNormal(
        loc=means, covariance_matrix=torch.diag_embed(sigmas**2)
    )
    return D.MixtureSameFamily(mix, comp)


def compute_W1_score(
    positions_target,
    weights_target,
    positions_estimate,
    weights_estimate,
    distance="semi-distance",
):
    """
    Compute the W1 distance between the estimate and the target. The weights of the estimate are renormalized.
    The distance used can be the semi-distance, the Fisher-Rao distance, or the Euclidean distance.
    """
    if torch.isnan(positions_estimate).any() or torch.isnan(weights_estimate).any():
        return torch.inf

    if weights_estimate.sum() < 1e-16:
        return torch.inf

    n_particles, total_dim = positions_target.shape
    d = total_dim // 2
    means_target = positions_target[:, :d]
    sigmas_target = positions_target[:, d:]
    means_estimate = positions_estimate[:, :d]
    sigmas_estimate = positions_estimate[:, d:]

    match distance:
        case "semi-distance":
            sum_sigmas2 = (
                sigmas_target[:, None, :] ** 2 + sigmas_estimate[None, :, :] ** 2
            )
            distance_matrix = torch.sqrt(
                torch.sum(
                    (means_target[:, None, :] - means_estimate[None, :, :]) ** 2
                    / sum_sigmas2
                    + torch.log(
                        sum_sigmas2
                        / (2 * sigmas_target[:, None, :] * sigmas_estimate[None, :, :])
                    ),
                    dim=-1,
                )
            )

            return ot.emd2(
                weights_target,
                weights_estimate / torch.sum(weights_estimate),
                distance_matrix,
            ).item()
        case "Fisher-Rao":
            delta_means2 = (means_target[:, None, :] - means_estimate[None, :, :]) ** 2
            distance_matrix = torch.sqrt(
                torch.sum(
                    2
                    * torch.log(
                        (
                            torch.sqrt(
                                delta_means2
                                + (
                                    sigmas_target[:, None, :]
                                    - sigmas_estimate[None, :, :]
                                )
                                ** 2
                            )
                            + torch.sqrt(
                                delta_means2
                                + (
                                    sigmas_target[:, None, :]
                                    + sigmas_estimate[None, :, :]
                                )
                                ** 2
                            )
                        )
                        / (
                            2
                            * torch.sqrt(
                                sigmas_target[:, None, :] * sigmas_estimate[None, :, :]
                            )
                        )
                    )
                    ** 2,
                    dim=-1,
                )
            )
        case "Euclidean":
            distance_matrix = ot.dist(
                positions_target, positions_estimate, metric="euclidean"
            )

        case _:
            raise NotImplementedError(
                f"Unsupported distance: {distance}. Try 'semi-distance', 'Fisher-Rao' or 'Euclidean'"
            )

    return ot.emd2(
        weights_target, weights_estimate / torch.sum(weights_estimate), distance_matrix
    ).item()