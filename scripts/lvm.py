import jax
import netket as nk
import jax.numpy as jnp
import numpy as np
import flax
from netket.operator.spin import sigmax
from functools import partial
import flax.linen as nn
import scipy

# Import useful functions
from tnqg.models import Distribution
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

# Extended Hilbert space of time and space
local_states = nk.utils.StaticRange(start=0, step=T / (n_times - 1), length=n_times)
hit = nk.hilbert.TensorHilbert(CustomHilbert(local_states=local_states, N=1), hi)

# Sampling parameters
n_chains_per_time = 8
n_chains_per_rank = n_times * n_chains_per_time

n_samples_per_time = 256
n_samples = n_times * n_samples_per_time

n_samples_psi0 = 8192
n_samples_distribution = 2**20
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
E_min = np.loadtxt(f"Emin_N{L}x{L}_h{h:.2f}.txt", dtype=complex).real
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

# Number of states from previous time sub-intervals and corresponding coefficients
n_states_fixed = (n_reps - 1) * n_models
coeffs_fixed = jnp.array([1.0])

# Input directory
folder_out = f"output/"

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


# Read the optimized parameters
with open(folder_out + f"params_{n_reps - 1}.mpack", "rb") as file:
    distr.variables = flax.serialization.from_bytes(distr.variables, file.read())

# Extract the variables
variables = distr.variables["params"]["var_states"]
for j in range(n_reps - 1)[::-1]:
    variables_fixed = distr.variables["extra"]["params_fixed_" + str(j)]
    variables = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate((x, y), axis=0),
        variables_fixed,
        variables,
    )
variables = {
    "params": variables,
}

# Compute the matrices
S, [O, F2], F = estimate_matrices(
    distr,
    psi0,
    logphis_model,
    variables,
    n_models * n_reps,
    H,
    [obs, H2],
)

# Invert S and compute S^-1 @ F
S_inv = np.linalg.pinv(S)
A = S_inv @ F

# Solve the linear variational method equations
ts = np.linspace(0, n_reps * T, n_reps * n_times)
c0 = np.zeros(n_reps * n_models + 1)
c0[0] = 1.0
cs = scipy.sparse.linalg.expm_multiply(
    -1j * A, c0, start=ts[0], stop=ts[-1], num=ts.size, endpoint=True
)  # (Nt, M+1)

# Compute and save the observable
obs_mean = np.einsum("ki, ij, kj -> k", cs.conj(), O, cs) / np.einsum(
    "ki, ij, kj -> k", cs.conj(), S, cs
)

np.savetxt(
    folder_out + f"observable.txt",
    obs_mean,
)

# Compute and save the loss function L
cs_dot = jnp.einsum("ij, kj -> ki", -1j * A, cs)

psi_psi = jnp.einsum("ki, ij, kj -> k", cs.conj(), S, cs)
psidot_psidot = jnp.einsum("ki, ij, kj -> k", cs_dot.conj(), S, cs_dot)
psidot_psi = jnp.einsum("ki, ij, kj -> k", cs_dot.conj(), S, cs)
psi_H_psi = jnp.einsum("ki, ij, kj -> k", cs.conj(), F, cs)
psi_H2_psi = jnp.einsum("ki, ij, kj -> k", cs.conj(), F2, cs)
psidot_H_psi = jnp.einsum("ki, ij, kj -> k", cs_dot.conj(), F, cs)

term1 = psidot_psidot / psi_psi
term2 = (psidot_psi / psi_psi) * (psidot_psi / psi_psi).conj()
term3 = psi_H2_psi / psi_psi
term4 = psi_H_psi / psi_psi
term5 = psidot_H_psi / psi_psi
term6 = (psidot_psi / psi_psi) * (psi_H_psi / psi_psi)

L = term1 - term2 + term3 - term4**2 + 2 * jnp.real(1j * term5 - 1j * term6)

np.savetxt(folder_out + f"loss.txt", L)

# FFT of the coefficients and its derivatives
gammas = []
for j in range(cs.shape[-1]):
    gamma = np.fft.fft(cs[:, j])
    gamma = gamma / gamma.size
    gammas.append(gamma)
gammas = np.array(gammas)  # (M + 1, Nf)

gammas_dot = []
for j in range(cs_dot.shape[-1]):
    gamma_dot = np.fft.fft(cs_dot[:, j])
    gamma_dot = gamma_dot / gamma_dot.size
    gammas_dot.append(gamma_dot)
gammas_dot = np.array(gammas_dot)  # (M + 1, Nf)

# Compute and save the observable at infinite time
obs_inft = (jnp.einsum("ij, kj, ik -> ", gammas.conj(), gammas, O)) / (
    jnp.einsum("ij, kj, ik -> ", gammas.conj(), gammas, S)
)

np.savetxt(
    folder_out + f"observable_inft.txt",
    [obs_inft],
)

# Compute and save the observable at infinite time
psi_psi = jnp.einsum("ij, kj, ik -> ", gammas.conj(), gammas, S)
psidot_psidot = jnp.einsum("ij, kj, ik -> ", gammas_dot.conj(), gammas_dot, S)
psidot_psi = jnp.einsum("ij, kj, ik -> ", gammas_dot.conj(), gammas, S)
psi_H_psi = jnp.einsum("ij, kj, ik -> ", gammas.conj(), gammas, F)
psi_H2_psi = jnp.einsum("ij, kj, ik -> ", gammas.conj(), gammas, F2)
psidot_H_psi = jnp.einsum("ij, kj, ik -> ", gammas_dot.conj(), gammas, F)

term1 = psidot_psidot / psi_psi
term2 = (psidot_psi / psi_psi) * (psidot_psi / psi_psi).conj()
term3 = psi_H2_psi / psi_psi
term4 = psi_H_psi / psi_psi
term5 = psidot_H_psi / psi_psi
term6 = (psidot_psi / psi_psi) * (psi_H_psi / psi_psi)

L_inft = term1 - term2 + term3 - term4**2 + 2 * jnp.real(1j * term5 - 1j * term6)

np.savetxt(folder_out + f"loss_inft.txt", np.array([L_inft]))
