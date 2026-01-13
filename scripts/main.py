import os
import jax
import netket as nk
import jax.numpy as jnp
import numpy as np
import flax
from netket.operator.spin import sigmax
from functools import partial
import flax.linen as nn
import scipy
import optax

# Import useful functions
from tnqg.operator import L2Loss
from tnqg.models import PsiT, Psi0, Distribution
from tnqg.sampling import LocalRuleT, FixedRule
from tnqg.utils import CustomHilbert

from estimate_matrices import estimate_matrices

# System size
L = 3
N = L**2

# Hilbert space
hi = nk.hilbert.Spin(s=0.5, N=N)

# Number of basis variational states
n_models = 8

# Lattice
graph = nk.graph.Square(length=L, pbc=True)

# Hamiltonian
hc = 3.044
h = 2.0
J = 1.0
H = nk.operator.IsingJax(hi, graph, h=h, J=-J)
H2 = H @ H

# Observable to monitor
obs = (sum([sigmax(hi, i) for i in range(N)]) / N).to_jax_operator()

# Time parameters
T = 0.2
n_times = 254 + 1
ts = np.linspace(0, T, n_times)
dt = ts[1] - ts[0]

# Extended Hilbert space of time and space
local_states = nk.utils.StaticRange(start=0, step=T / (n_times - 1), length=n_times)
hit = nk.hilbert.TensorHilbert(CustomHilbert(local_states=local_states, N=1), hi)

# Sampling parameters
n_chains_per_time = 8
n_chains_per_rank = n_times * n_chains_per_time

n_samples_per_time = 256
n_samples = n_times * n_samples_per_time

n_samples_psi0 = 8192
n_samples_distribution = 16384
n_chains_per_rank_hi = 2048

n_discard_per_chain = 5

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
n_basis = 128

# Frequencies for the time-dependent coefficients (initialized to evenly spaced energies in the spectrum of the Hamiltonian)
E_min = np.loadtxt(f"Emin.txt", dtype=complex).real
E_max = -E_min
omegas = jnp.linspace(E_min, E_max, n_basis)


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
with open(f"psi0_N={L}x{L}.mpack", "rb") as file:
    psi0.variables = flax.serialization.from_bytes(psi0.variables, file.read())

psi0_afun = lambda x: psi0._apply_fun(psi0.variables, x)

# Number of time sub-intervals
n_reps = 10

# Parameters and coefficients from previous time sub-intervals
coeffs_fixed = jnp.array([1])
params_fixed = ()

# Output directory
folder_out = f"output/"
if not os.path.exists(folder_out):
    os.makedirs(folder_out)

# Collect loss values throughout optimizations
learning_curves = []

# Loop over the time sub-intervals
for k in range(n_reps):
    # Number of fixed states from previous time sub-intervals
    n_states_fixed = k * n_models

    # Model for the time-dependent state
    psit_model = PsiT(
        psi0_afun=psi0_afun,
        logphis_model=logphis_model,
        n_models=n_models,
        n_times=n_times,
        T=T,
        n_basis=n_basis,
        basis_funcs=basis_funcs,
        n_states_fixed=n_states_fixed,
        coeffs_fixed=nk.utils.HashableArray(coeffs_fixed),
    )

    # Time-dependent state
    psit = nk.vqs.MCState(
        sampler=sampler,
        model=psit_model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
    )

    # Model for the initial state in the k-th time sub-interval
    psi0_restart_model = Psi0(
        psi0_afun=psi0_afun,
        logphis_model=logphis_model,
        n_models=n_models,
        n_times=n_times,
        T=T,
        n_basis=n_basis,
        basis_funcs=basis_funcs,
        n_states_fixed=n_states_fixed,
        coeffs_fixed=nk.utils.HashableArray(coeffs_fixed),
    )

    # Initial state in the k-th time sub-interval
    psi0_restart = nk.vqs.MCState(
        sampler=sampler_hi,
        model=psi0_restart_model,
        n_samples=n_samples_psi0,
        n_discard_per_chain=n_discard_per_chain,
    )

    # Set the frequencies
    variables = flax.core.unfreeze(psit.variables)
    variables["params"]["omegas"] = omegas

    # Load the parameters from previous sub-intervals
    if k > 0:
        for j in range(k):
            variables["extra"][f"params_fixed_{j}"] = params_fixed[j]
        variables["params"]["var_states"] = variables["extra"][f"params_fixed_{k-1}"]
    psit.variables = variables

    # L2 loss operator
    window = n_times // 5
    op = L2Loss(hit=hit, ham=H, T=T, n_times=n_times, window=window)

    # Optimization parameters
    n_steps = 20000
    lr = 1e-3
    min_value = 1e-4
    cos_dec = optax.cosine_decay_schedule(
        init_value=lr - min_value, decay_steps=n_steps
    )
    lr_func = lambda t: cos_dec(t) + min_value
    optimizer = nk.optimizer.Adam(learning_rate=lr_func)

    # Optimization driver
    driver = nk.driver.VMC(hamiltonian=op, optimizer=optimizer, variational_state=psit)

    # Loggers
    log = nk.logging.RuntimeLog()

    # Run the driver
    driver.run(n_iter=n_steps, out=log)

    # Saved the optimized state
    with open(
        folder_out + f"params_{k}.mpack",
        "wb",
    ) as file:
        file.write(flax.serialization.to_bytes(psit.variables))

    # Save the L2 loss values
    data = log.data
    learning_curves.append(data["Energy"].Mean)
    np.savetxt(
        folder_out + "learning_curves.txt",
        np.column_stack(learning_curves),
        delimiter=" ",
    )

    # Initialize the initial state in the k-th time sub-interval
    psi0_restart.variables = psit.variables

    # Model for the distribution to sample the matrices for the linear variational method
    distr_model = Distribution(
        psi0_afun=psi0_afun,
        logphis_model=logphis_model,
        n_models=n_models,
        n_times=n_times,
        T=T,
        n_basis=n_basis,
        basis_funcs=basis_funcs,
        n_states_fixed=n_states_fixed,
        coeffs_fixed=nk.utils.HashableArray(coeffs_fixed),
    )

    # Distribution to sample the matrices
    distr = nk.vqs.MCState(
        sampler=sampler_hi,
        model=distr_model,
        n_samples=n_samples_distribution,
        n_discard_per_chain=n_discard_per_chain,
    )

    distr.variables = psit.variables

    # Compute the matrices
    variables = {
        "params": psit.variables["params"]["var_states"],
    }
    S, F = estimate_matrices(
        distr,
        psi0_restart,
        logphis_model,
        variables,
        n_models,
        H,
    )

    # Invert S and compute S^-1 @ F
    S_inv = np.linalg.pinv(S)
    A = S_inv @ F

    # Solve the linear variational method equations
    c0 = np.zeros(n_models + 1)
    c0[0] = 1.0
    cs = scipy.sparse.linalg.expm_multiply(
        -1j * A, c0, start=ts[0], stop=ts[-1], num=ts.size, endpoint=True
    )  # (Nt, M+1)
    c = cs[-1]

    # Collect the optimal coefficients and parameters at the end of the k-th time sub-interval to move to the next sub-interval
    coeffs_fixed = coeffs_fixed * c[0]
    coeffs_fixed = jnp.concatenate((coeffs_fixed, c[1:]))

    params_fixed = params_fixed + (psit.variables["params"]["var_states"],)
