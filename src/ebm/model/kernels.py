"""RBF kernel and SVGD update for particle-based inference."""

import torch
from torch import Tensor


def rbf_kernel(particles: Tensor, bandwidth: float | None = None) -> tuple[Tensor, Tensor]:
    """
    Compute the RBF kernel matrix and its gradients for batched particles.

    Args:
        particles: Particle positions with shape ``(B, N, D)``.
        bandwidth: RBF bandwidth (sigma-squared). When ``None``, uses the
            median heuristic ``h = median(sq_dists) / (2 * log(N + 1))``,
            clamped to a minimum of ``1e-3``.

    Returns:
        Tuple of ``(K, grad_K)`` where ``K`` has shape ``(B, N, N)`` and
        ``grad_K`` has shape ``(B, N, N, D)``. ``grad_K[:, i, j]`` is the
        gradient of ``K[:, i, j]`` with respect to particle ``i``.

    """
    # (B, N, 1, D) - (B, 1, N, D) -> (B, N, N, D)
    diff = particles.unsqueeze(2) - particles.unsqueeze(1)
    sq_dists = (diff**2).sum(dim=-1)  # (B, N, N)

    if bandwidth is None:
        n = particles.shape[1]
        # Median over all pairwise distances per batch
        median_sq = torch.median(sq_dists.reshape(particles.shape[0], -1), dim=1).values  # (B,)
        h = median_sq / (2.0 * torch.log(torch.tensor(n + 1.0, device=particles.device)))
        h = torch.clamp(h, min=1e-3)
        h = h[:, None, None]  # (B, 1, 1) for broadcasting
    else:
        h = bandwidth

    K = torch.exp(-sq_dists / (2.0 * h))  # (B, N, N)

    # grad_K[:, i, j, :] = K[:, i, j] * (x_j - x_i) / h = K * (-diff) / h
    grad_K = K.unsqueeze(-1) * (-diff) / (h if isinstance(h, float) else h.unsqueeze(-1))  # (B, N, N, D)

    return K, grad_K


def svgd_update(
    grad_energy: Tensor,
    kernel_matrix: Tensor,
    kernel_grads: Tensor,
    repulsion_weight: float = 1.0,
) -> Tensor:
    """
    Compute the SVGD update direction for a batch of particles.

    Combines an attraction term (kernel-weighted negative energy gradients)
    with a repulsion term (kernel gradient, encouraging diversity).

    Args:
        grad_energy: Energy gradients with shape ``(B, N, D)``, pointing
            uphill (positive energy direction).
        kernel_matrix: RBF kernel matrix ``K`` with shape ``(B, N, N)``.
        kernel_grads: Kernel gradients with shape ``(B, N, N, D)``.
        repulsion_weight: Scalar in ``[0, 1]`` multiplied onto the repulsion
            term. Set to ``1.0`` for standard SVGD, anneal toward ``0.0``
            to let particles converge independently in late steps.

    Returns:
        SVGD update direction with shape ``(B, N, D)``, to be used as
        ``particles -= lr * update``.

    """
    n = grad_energy.shape[1]
    # Attraction: K @ (-grad_energy) -> (B, N, D)
    attraction = torch.bmm(kernel_matrix, -grad_energy)
    # Repulsion: sum over source particles j -> (B, N, D)
    repulsion = kernel_grads.sum(dim=2)
    return (attraction + repulsion_weight * repulsion) / n
