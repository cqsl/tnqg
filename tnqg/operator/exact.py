from functools import partial
import jax
import jax.numpy as jnp

import netket as nk
from netket.jax.sharding import (
    shard_along_axis,
)
from netket.vqs import FullSumState, expect, expect_and_grad
from netket.stats import Stats

from tnqg.operator.operator import L2Loss


@expect.dispatch
def expect(vstate: FullSumState, op: L2Loss):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    ts = op.hilbert.subspaces[0].all_states()
    σ = op.hilbert.subspaces[1].all_states()

    return expect_and_grad_inner_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.n_times,
        ts,
        σ,
        op.ham,
        op.T,
        return_grad=False,
    )


@expect_and_grad.dispatch
def expect_and_grad(
    vstate: FullSumState,
    op: L2Loss,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    ts = op.hilbert.subspaces[0].all_states()
    σ = op.hilbert.subspaces[1].all_states()

    return expect_and_grad_inner_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.n_times,
        ts,
        σ,
        op.ham,
        op.T,
        return_grad=True,
    )


@partial(
    jax.jit,
    static_argnames=("afun", "return_grad", "n_times"),
)
def expect_and_grad_inner_fs(
    afun,
    params,
    model_state,
    n_times,
    ts,
    σ,
    ham,
    T,
    return_grad,
):
    σ = shard_along_axis(σ, axis=0)

    def afun_t(params, σ, t):
        t = jnp.expand_dims(t, axis=-1)
        tσ = jnp.concatenate((t, σ), axis=-1)
        return afun({"params": params, **model_state}, tσ)

    # defined for one time (one chain)
    def loss_loc(params, t_single, σ):
        t_single = t_single[-1]
        t = jnp.tile(t_single, reps=σ.shape[:-1])
        logpsi_tσ = afun_t(params, σ, t)
        fun_t = lambda _t: afun_t(params, σ, jnp.full(t.shape, _t))
        O_t = jax.jacfwd(fun_t)(t_single)

        σp, mels = ham.get_conn_padded(σ)

        _t = jnp.tile(jnp.expand_dims(t, axis=-1), reps=(1,) * t.ndim + (σp.shape[-2],))

        ns = σp.shape[0]
        nconn = σp.shape[-2]
        σp = σp.reshape(-1, σp.shape[-1])
        _t = _t.reshape(-1)

        logpsi_σpt = afun_t(params, σp, _t)
        logpsi_σpt = logpsi_σpt.reshape(ns, nconn)

        Eloc = jnp.sum(
            mels * jnp.exp(logpsi_σpt - jnp.expand_dims(logpsi_tσ, axis=-1)), axis=-1
        )

        P = jnp.exp(2 * logpsi_tσ.real)
        P = P / jnp.sum(P, axis=-1, keepdims=True)

        O_t_mean = jnp.sum(P * O_t, axis=-1, keepdims=True)
        Eloc_mean = jnp.sum(P * Eloc, axis=-1, keepdims=True)

        O_t = O_t - O_t_mean
        Eloc = Eloc - Eloc_mean

        return jnp.abs(O_t + 1j * Eloc) ** 2, P

    weights_simpson = 4 / 3 * jnp.ones(n_times)
    weights_simpson = weights_simpson.at[::2].set(2 / 3)
    weights_simpson = weights_simpson.at[0].set(1 / 3)
    weights_simpson = weights_simpson.at[-1].set(1 / 3)

    dt = T / (n_times - 1)

    if not return_grad:

        def compute_loss_values(i, carry):
            loss_values, P = loss_loc(params, ts[i], σ)
            carry = carry.at[i].set(jnp.sum(P * loss_values, axis=-1))
            return carry

        loss_means = jax.lax.fori_loop(
            0, ts.shape[0], compute_loss_values, jnp.zeros(ts.shape[0], dtype=complex)
        )

        loss_mean = jnp.sum(dt * loss_means * weights_simpson / T)

        loss_stats = Stats(
            mean=loss_mean,
            error_of_mean=0,
            variance=0,
        )

        return loss_stats

    def get_grad(params, t_single, σ):
        loss_values, vjp_loss, P = nk.jax.vjp(
            loss_loc, params, t_single, σ, has_aux=True, conjugate=True
        )

        t = jnp.tile(t_single, reps=σ.shape[:-1] + (1,))
        tσ = jnp.concatenate((t, σ), axis=-1)

        log_pdf_fun = (
            lambda params, tσ: 2 * afun({"params": params, **model_state}, tσ).real
        )
        log_pdf, vjp_pdf = nk.jax.vjp(log_pdf_fun, params, tσ, conjugate=True)

        loss_mean = jnp.sum(P * loss_values, axis=-1, keepdims=True)
        loss_values -= loss_mean

        grad1 = vjp_pdf(P * loss_values)[0]

        grad2 = vjp_loss(P)[0]

        grad = jax.tree_util.tree_map(lambda x, y: x + y, grad1, grad2)

        return loss_mean, grad

    def compute_loss_grads(i, carry):
        loss_means, grads = carry
        loss_mean, grad = get_grad(params, ts[i], σ)
        loss_mean = loss_mean[..., -1]
        loss_means = loss_means.at[i].set(loss_mean)
        grads = jax.tree_util.tree_map(lambda x, y: x.at[i].set(y), grads, grad)
        return loss_means, grads

    loss_means, grads = jax.lax.fori_loop(
        0,
        ts.shape[0],
        compute_loss_grads,
        (
            jnp.zeros(ts.shape[0], dtype=complex),
            jax.tree_util.tree_map(
                lambda x: jnp.zeros((ts.shape[0],) + x.shape, dtype=complex), params
            ),
        ),
    )

    loss_mean = jnp.sum(dt * loss_means * weights_simpson / T)

    grad = jax.tree_util.tree_map(
        lambda y: jnp.sum(
            dt
            * y
            * weights_simpson.reshape(weights_simpson.shape + (1,) * (y.ndim - 1))
            / T,
            axis=0,
        ),
        grads,
    )

    loss_stats = Stats(
        mean=loss_mean,
        error_of_mean=0,
        variance=0,
    )

    return loss_stats, grad
