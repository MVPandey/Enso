"""Unit tests for RBF kernel and SVGD update functions."""

import torch

from ebm.model.kernels import rbf_kernel, svgd_update


def test_rbf_kernel_shape():
    """K is (B, N, N) and grad_K is (B, N, N, D)."""
    particles = torch.randn(3, 5, 8)
    K, grad_K = rbf_kernel(particles)
    assert K.shape == (3, 5, 5)
    assert grad_K.shape == (3, 5, 5, 8)


def test_rbf_kernel_diagonal_ones():
    """Self-similarity should be 1: K[:, i, i] = 1."""
    particles = torch.randn(2, 4, 6)
    K, _ = rbf_kernel(particles)
    for b in range(2):
        for i in range(4):
            assert torch.isclose(K[b, i, i], torch.tensor(1.0)), f'K[{b},{i},{i}] = {K[b, i, i]}'


def test_rbf_kernel_symmetric():
    """K[:, i, j] = K[:, j, i]."""
    particles = torch.randn(2, 4, 6)
    K, _ = rbf_kernel(particles)
    assert torch.allclose(K, K.transpose(1, 2))


def test_rbf_kernel_gradient_antisymmetric():
    """grad_K[:, i, j] should be approximately -grad_K[:, j, i]."""
    particles = torch.randn(2, 4, 6)
    _, grad_K = rbf_kernel(particles)
    assert torch.allclose(grad_K, -grad_K.transpose(1, 2), atol=1e-5)


def test_svgd_single_particle_reduces_to_gradient_descent():
    """With n=1, repulsion vanishes and attraction = -grad_energy."""
    grad_energy = torch.randn(2, 1, 8)
    K = torch.ones(2, 1, 1)
    grad_K = torch.zeros(2, 1, 1, 8)
    update = svgd_update(grad_energy, K, grad_K)
    # With single particle: update = (K @ (-grad_energy) + 0) / 1 = -grad_energy
    assert torch.allclose(update, -grad_energy)
