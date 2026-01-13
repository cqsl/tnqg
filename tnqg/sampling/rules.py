import netket as nk
from netket.utils import struct
import jax
import jax.numpy as jnp


@struct.dataclass
class LocalRuleT(nk.sampler.rules.TensorRule):
    def random_state(
        rule,
        sampler,
        machine,
        parameters,
        state,
        key,
    ):
        key, subkey = jax.random.split(key)

        tN = sampler.hilbert.size
        times = sampler.hilbert.subspaces[0].all_states()
        n_times = times.size

        initial_state = jax.random.choice(
            key,
            a=jnp.array([-1.0, 1.0]),
            shape=(len(jax.devices()) * sampler.n_chains_per_rank, tN),
        )
        n_chains_per_time = len(jax.devices()) * sampler.n_chains_per_rank // n_times
        initial_state = initial_state.at[:, 0].set(jnp.repeat(times, n_chains_per_time))

        return initial_state

    def __repr__(self):
        return "LocalRuleT"


@nk.utils.struct.dataclass
class FixedRule(nk.sampler.rules.MetropolisRule):
    def transition(self, sampler, machine, parameters, state, key, σ):
        return σ, None
