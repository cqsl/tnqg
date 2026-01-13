from functools import partial
import jax
import jax.numpy as jnp

import netket as nk
from netket.jax.sharding import (
    shard_along_axis,
)
from netket.vqs import MCState, expect, expect_and_grad
from netket.stats import Stats

from tnqg.operator.operator import L2Loss


@expect.dispatch
def expect(vstate: MCState, op: L2Loss, chunk_size: None):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    tσ = vstate.samples

    return expect_and_grad_inner(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.n_times,
        tσ,
        op.ham,
        op.T,
        op.window,
        return_grad=False,
    )


@expect_and_grad.dispatch
def expect_and_grad(
    vstate: MCState,
    op: L2Loss,
    chunk_size: None,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    tσ = vstate.samples

    return expect_and_grad_inner(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.n_times,
        tσ,
        op.ham,
        op.T,
        op.window,
        return_grad=True,
    )


@partial(
    jax.jit,
    static_argnames=("afun", "return_grad", "n_times", "window"),
)
def expect_and_grad_inner(
    afun,
    params,
    model_state,
    n_times,
    tσ,
    ham,
    T,
    window,
    return_grad,
):
    tσ = tσ.reshape(n_times, -1, tσ.shape[-1])

    tσ = shard_along_axis(tσ, axis=1)

    def afun_t(params, σ, t):
        t = jnp.expand_dims(t, axis=-1)
        tσ = jnp.concatenate((t, σ), axis=-1)
        return afun({"params": params, **model_state}, tσ)

    # defined for one time (one chain)
    def loss_loc(params, tσ):
        t = tσ[..., 0]
        # Here we assume that all those t are the same, so we take a single one
        t_single = t[0]
        σ = tσ[..., 1:]

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

        O_t_mean = jnp.mean(O_t, axis=-1, keepdims=True)
        Eloc_mean = jnp.mean(Eloc, axis=-1, keepdims=True)

        O_t = O_t - O_t_mean
        Eloc = Eloc - Eloc_mean

        return jnp.abs(O_t + 1j * Eloc) ** 2

    weights_simpson = 4 / 3 * jnp.ones(n_times)
    weights_simpson = weights_simpson.at[::2].set(2 / 3)
    weights_simpson = weights_simpson.at[0].set(1 / 3)
    weights_simpson = weights_simpson.at[-1].set(1 / 3)

    dt = T / (n_times - 1)

    if not return_grad:
        loss_mean = 0

        def fill_many_losses(i, carry):
            loss_mean = carry

            samples = jax.lax.dynamic_slice(
                tσ, (i * window, 0, 0), (window, tσ.shape[-2], tσ.shape[-1])
            )

            loss_values = jax.vmap(loss_loc, in_axes=(None, 0), out_axes=(0))(
                params, samples
            )
            loss_means = jnp.mean(loss_values, axis=-1)

            weights_simpson_ = jax.lax.dynamic_slice(
                weights_simpson, (i * window,), (window,)
            )

            loss_mean += jnp.sum(dt * loss_means * weights_simpson_ / T)

            return loss_mean

        loss_mean = jax.lax.fori_loop(
            0, n_times // window, fill_many_losses, (loss_mean)
        )

        loss_stats = Stats(
            mean=loss_mean,
            error_of_mean=0,
            variance=0,
        )

        return loss_stats

    def get_grad(params, tσ):
        loss_values, vjp_loss = nk.jax.vjp(loss_loc, params, tσ, conjugate=True)

        log_pdf_fun = (
            lambda params, tσ: 2 * afun({"params": params, **model_state}, tσ).real
        )
        log_pdf, vjp_pdf = nk.jax.vjp(log_pdf_fun, params, tσ, conjugate=True)

        loss_mean = jnp.mean(loss_values, axis=-1, keepdims=True)
        loss_values -= loss_mean

        grad1 = vjp_pdf(loss_values)[0]
        grad1 = jax.tree_util.tree_map(lambda x: x / loss_values.shape[-1], grad1)

        grad2 = vjp_loss(jnp.ones_like(loss_values))[0]
        grad2 = jax.tree_util.tree_map(lambda x: x / loss_values.shape[-1], grad2)

        grad = jax.tree_util.tree_map(lambda x, y: x + y, grad1, grad2)

        return loss_mean, grad

    loss_mean = 0
    grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=complex), params)

    def fill_many_losses_grads(i, carry):
        loss_mean, grad = carry

        samples = jax.lax.dynamic_slice(
            tσ, (i * window, 0, 0), (window, tσ.shape[-2], tσ.shape[-1])
        )

        loss_means, grads = jax.vmap(get_grad, in_axes=(None, 0), out_axes=(0))(
            params, samples
        )
        loss_means = loss_means[..., -1]

        weights_simpson_ = jax.lax.dynamic_slice(
            weights_simpson, (i * window,), (window,)
        )

        loss_mean += jnp.sum(dt * loss_means * weights_simpson_ / T)

        grad = jax.tree_util.tree_map(
            lambda x, y: x
            + jnp.sum(
                dt
                * y
                * weights_simpson_.reshape(weights_simpson_.shape + (1,) * (y.ndim - 1))
                / T,
                axis=0,
            ),
            grad,
            grads,
        )

        return loss_mean, grad

    loss_mean, grad = jax.lax.fori_loop(
        0, n_times // window, fill_many_losses_grads, (loss_mean, grad)
    )

    loss_stats = Stats(
        mean=loss_mean,
        error_of_mean=0,
        variance=0,
    )

    return loss_stats, grad
