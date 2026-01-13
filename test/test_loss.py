import pytest
from pytest import approx

import jax
import netket as nk
import jax.numpy as jnp
import numpy as np
from functools import partial
import flax.linen as nn

from tnqg.operator import L2Loss
from tnqg.models import PsiT
from tnqg.sampling import LocalRuleT, FixedRule
from tnqg.utils import CustomHilbert

from .exact_loss import exact_loss
from ._finite_diff import central_diff_grad, same_derivatives


def _setup():
    # System size
    L = 2
    N = L**2

    # Hilbert space
    hi = nk.hilbert.Spin(s=0.5, N=N)

    # Number of basis variational states
    n_models = 4

    # Lattice
    graph = nk.graph.Square(length=L, pbc=True)

    # Hamiltonian
    h = 2.0
    J = 1.0
    H = nk.operator.IsingJax(hi, graph, h=h, J=-J)

    # Time parameters
    T = 0.2
    n_times = 254 + 1
    ts = np.linspace(0, T, n_times)

    # Extended Hilbert space of time and space
    local_states = nk.utils.StaticRange(start=0, step=T / (n_times - 1), length=n_times)
    hit = nk.hilbert.TensorHilbert(CustomHilbert(local_states=local_states, N=1), hi)

    # Sampling parameters
    n_chains_per_time = 8
    n_chains_per_rank = n_times * n_chains_per_time

    n_samples_per_time = 1024
    n_samples = n_times * n_samples_per_time

    n_samples_psi0 = 8192
    n_chains_per_rank_hi = 2048

    n_discard_per_chain = 100

    # Samplers
    sampler = nk.sampler.MetropolisSampler(
        hit,
        LocalRuleT(hit, (FixedRule(), nk.sampler.rules.LocalRule())),
        n_chains_per_rank=n_chains_per_rank,
    )

    sampler_hi = nk.sampler.MetropolisSampler(
        hi, nk.sampler.rules.LocalRule(), n_chains_per_rank=n_chains_per_rank_hi
    )

    # Number of basis functions for the time-dependent coefficients
    n_basis = 16

    # Fourier basis functions for the time-dependent coefficients
    @jax.jit
    def fourier_basis(x, gammas, omegas):
        @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        def fourier_element(x, omega):
            return jnp.exp(1j * omega * x)

        basis = fourier_element(x, omegas)

        return jnp.einsum("ij, j... -> i...", gammas, basis)

    basis_funcs = fourier_basis

    # Model for the initial state
    model_psi0 = nk.models.RBM(alpha=1, param_dtype=complex)

    # Model evaluating all the basis variational states
    class LogPhis(nn.Module):
        @nn.compact
        def __call__(self, xs):
            model_vmap = nn.vmap(
                partial(nk.models.RBM, alpha=1, param_dtype=complex),
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=0,
            )

            return model_vmap(name="model_vmap")(xs)

    logphis_model = LogPhis()

    # Initial state
    psi0 = nk.vqs.MCState(
        sampler=sampler_hi,
        model=model_psi0,
        n_samples=n_samples_psi0,
        n_discard_per_chain=n_discard_per_chain,
    )
    psi0_afun = lambda x: psi0._apply_fun(psi0.variables, x)

    # Model for the time-dependent state
    psit_model = PsiT(
        psi0_afun=psi0_afun,
        logphis_model=logphis_model,
        n_models=n_models,
        n_times=n_times,
        T=T,
        n_basis=n_basis,
        basis_funcs=basis_funcs,
        n_states_fixed=0,
        coeffs_fixed=nk.utils.HashableArray(jnp.array([1])),
    )

    # Time-dependent state
    psit_exact = nk.vqs.FullSumState(
        hilbert=hit,
        model=psit_model,
    )

    # Time-dependent state
    psit = nk.vqs.MCState(
        sampler=sampler,
        model=psit_model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        variables=psit_exact.variables,
    )

    return psit, psit_exact, H, hit, ts, n_times, T


def test_MCState():
    psit, psit_exact, H, hit, ts, n_times, T = _setup()

    window = n_times // 5
    op = L2Loss(hit=hit, ham=H, T=T, n_times=n_times, window=window)

    L2_exact = exact_loss(psit, H.to_dense(), n_times, ts, T)

    params, unravel = nk.jax.tree_ravel(psit.parameters)

    def _exact_loss(params):
        psit.parameters = unravel(params)
        return exact_loss(psit, H.to_dense(), n_times, ts, T)

    L2_grad_exact = central_diff_grad(_exact_loss, params, 1.0e-5)

    L2_stats1 = psit.expect(op)
    L2_stats, L2_grad = psit.expect_and_grad(op)
    L2_grad, _ = nk.jax.tree_ravel(L2_grad)

    assert L2_stats1.mean.real == approx(L2_stats.mean.real, abs=1e-5)
    assert np.asarray(L2_stats1.error_of_mean) == approx(
        np.asarray(L2_stats.error_of_mean), abs=1e-5
    )

    np.testing.assert_allclose(
        L2_exact, L2_stats1.mean.real, atol=3 * L2_stats1.error_of_mean
    )
    np.testing.assert_allclose(
        L2_exact, L2_stats.mean.real, atol=3 * L2_stats.error_of_mean
    )

    same_derivatives(L2_grad_exact, L2_grad, abs_eps=1e-4)


def test_FullSumState():
    psit, psit_exact, H, hit, ts, n_times, T = _setup()

    window = n_times // 5
    op = L2Loss(hit=hit, ham=H, T=T, n_times=n_times, window=window)

    L2_exact = exact_loss(psit_exact, H.to_dense(), n_times, ts, T)

    params, unravel = nk.jax.tree_ravel(psit_exact.parameters)

    def _exact_loss(params):
        psit_exact.parameters = unravel(params)
        return exact_loss(psit_exact, H.to_dense(), n_times, ts, T)

    L2_grad_exact = central_diff_grad(_exact_loss, params, 1.0e-5)

    L2_stats1 = psit_exact.expect(op)
    L2_stats, L2_grad = psit_exact.expect_and_grad(op)
    L2_grad, _ = nk.jax.tree_ravel(L2_grad)

    assert L2_stats1.mean.real == approx(L2_stats.mean.real, abs=1e-5)
    assert np.asarray(L2_stats1.error_of_mean) == approx(
        np.asarray(L2_stats.error_of_mean), abs=1e-5
    )

    np.testing.assert_allclose(
        L2_exact, L2_stats1.mean.real, atol=3 * L2_stats1.error_of_mean
    )
    np.testing.assert_allclose(
        L2_exact, L2_stats.mean.real, atol=3 * L2_stats.error_of_mean
    )

    same_derivatives(L2_grad_exact, L2_grad, abs_eps=1e-4)
