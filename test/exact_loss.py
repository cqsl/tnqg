import jax.numpy as jnp
import jax


def exact_loss(psit, H_mtrx, n_times, ts, T):
    psit_state = psit.to_array(normalize=False)
    psit_state = psit_state.reshape(n_times, -1)  # (Nt, d)

    σ = psit.hilbert.all_states()
    hilb_size = σ.shape[0] // n_times
    σ = σ[:hilb_size]
    σ = σ[..., 1:]

    def afun_t(params, σ, t):
        t = jnp.expand_dims(t, axis=-1)
        tσ = jnp.concatenate((t, σ), axis=-1)
        return psit._apply_fun({"params": params, **psit.model_state}, tσ)

    def get_dlogpsit(t_single):
        t = jnp.tile(t_single, reps=σ.shape[:-1])
        fun_t = lambda _t: afun_t(psit.parameters, σ, jnp.full(t.shape, _t))
        O_t = jax.jacfwd(fun_t)(t_single)
        return O_t

    O_t = jax.vmap(get_dlogpsit)(ts)  # (Nt, d)
    dpsit_state = psit_state * O_t  # (Nt, d)

    Hpsit_state = jnp.einsum("ik, ...k -> ...i", H_mtrx, psit_state)  # (Nt, d)

    norm = jnp.linalg.norm(psit_state, axis=-1, keepdims=True)  # (Nt, 1)

    t1 = dpsit_state / norm  # (Nt, d)
    t2 = 1j * Hpsit_state / norm  # (Nt, d)
    t3 = (
        jnp.sum(psit_state.conj() * dpsit_state, axis=-1, keepdims=True)
        * psit_state
        / norm**3
    )
    t4 = (
        1j
        * jnp.sum(psit_state.conj() * Hpsit_state, axis=-1, keepdims=True)
        * psit_state
        / norm**3
    )

    Lt = jnp.linalg.norm(t1 + t2 - t3 - t4, axis=-1) ** 2  # (Nt,)

    weights_simpson = 4 / 3 * jnp.ones(n_times)
    weights_simpson = weights_simpson.at[::2].set(2 / 3)
    weights_simpson = weights_simpson.at[0].set(1 / 3)
    weights_simpson = weights_simpson.at[-1].set(1 / 3)

    dt = T / (n_times - 1)

    L = jnp.sum(dt * Lt * weights_simpson / T)

    return L
