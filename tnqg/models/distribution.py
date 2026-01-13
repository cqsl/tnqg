from flax import linen as nn
import jax.numpy as jnp
from typing import Any, Callable
import jax

from jax.nn.initializers import normal
from netket.utils.types import NNInitFunc
from tnqg.utils.has_leaf_extra import has_leaf_extra

default_kernel_init = normal(stddev=0.01)


class Distribution(nn.Module):
    psi0_afun: Any
    logphis_model: Any
    n_models: int
    n_times: int
    T: int
    n_basis: int
    basis_funcs: Callable
    n_states_fixed: Any
    coeffs_fixed: Any
    window: bool = True
    kernel_init: NNInitFunc = default_kernel_init

    def setup(self):
        self.gammas = self.param(
            "gammas", self.kernel_init, (self.n_models, self.n_basis), jnp.complex128
        )
        self.omegas = self.param(
            "omegas", self.kernel_init, (self.n_basis,), jnp.float64
        )
        self.n_group_fixed = self.n_states_fixed // self.n_models

    def _build_logphis(self, x):
        x = jnp.atleast_2d(x)

        tot_n_models = self.n_models + self.n_states_fixed
        logphis = jnp.zeros((tot_n_models + 1,) + x.shape[:-1], dtype=complex)

        xs = jnp.tile(x[None, ...], (self.n_models,) + (1,) * x.ndim)

        logphis = logphis.at[0].set(self.psi0_afun(x))

        if self.is_initializing():
            for i in range(self.n_group_fixed):
                variables = self.logphis_model.init(self.make_rng("params"), xs)
                self.scope.put_variable(
                    f"extra", f"params_fixed_{i}", variables["params"]
                )

                if has_leaf_extra(variables):
                    self.scope.put_variable(
                        "extra", f"noparams_fixed_{i}", variables["extra"]
                    )

            variables = self.logphis_model.init(self.make_rng("params"), xs)

            self.scope.put_variable("params", "var_states", variables["params"])

            if has_leaf_extra(variables):
                self.scope.put_variable("noparams", "var_states", variables["extra"])

        for i in range(self.n_group_fixed):
            pars_fixed = self.scope.get_variable("extra", f"params_fixed_{i}")

            if self.scope.get_variable("extra", f"noparams_fixed_{i}") is not None:
                nopars_fixed = self.scope.get_variable("extra", f"noparams_fixed_{i}")
                vmaplogwf = self.logphis_model.bind(
                    {"params": pars_fixed, "extra": nopars_fixed}
                )
            else:
                vmaplogwf = self.logphis_model.bind({"params": pars_fixed})

            logphis = logphis.at[
                i * self.n_models + 1 : (i + 1) * self.n_models + 1
            ].set(vmaplogwf(xs))

        pars = self.scope.get_variable("params", "var_states")

        if self.scope.get_variable("noparams", "var_states") is not None:
            nopars = self.scope.get_variable("noparams", "var_states")
            vmaplogwf = self.logphis_model.bind({"params": pars, "extra": nopars})
        else:
            vmaplogwf = self.logphis_model.bind({"params": pars})

        logphis = logphis.at[-self.n_models :].set(vmaplogwf(xs))
        return x, logphis

    def __call__(self, x):
        x, logphis = self._build_logphis(x)

        if self.window:
            if self.coeffs_fixed.wrapped.size > 1:
                coeffs_fixed = self.coeffs_fixed.wrapped
                coeffs_fixed = coeffs_fixed.reshape(
                    coeffs_fixed.shape + (1,) * (x[..., 0].ndim)
                )
                coeffs_fixed = coeffs_fixed * jnp.ones_like(x[..., 0])
                coeffs = coeffs_fixed
            else:
                coeffs = jnp.ones((1,) + x[..., 0].shape)

            logpsi0 = jax.scipy.special.logsumexp(
                logphis[: -self.n_models], b=coeffs, axis=0
            )

            logphisq = jax.scipy.special.logsumexp(
                2 * logphis[-self.n_models :].real, axis=0
            )
            return 0.5 * jax.scipy.special.logsumexp(
                jnp.array([2 * logpsi0.real, logphisq]), axis=0
            )

        return 0.5 * jax.scipy.special.logsumexp(2 * logphis.real, axis=0)
