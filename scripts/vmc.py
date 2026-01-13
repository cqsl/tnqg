import netket as nk
import flax

# System size
L = 3
N = L**2

# Transverse field and coupling
h = 2.0
# J = -1.0
J = 0.0

# Lattice
graph = nk.graph.Square(length=L, pbc=True)

# Sample parameters
n_samples = 8192
n_chains_per_rank = 512
n_discard_per_chain = 10

# Hilbert space
hi = nk.hilbert.Spin(s=1 / 2, N=N)

# Hamiltonian
H = nk.operator.Ising(hi, graph, h=h, J=J)

# Optimization parameters
learning_rate = 0.01
diag_shift = 0.001
n_iter = 1000
optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
preconditioner = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True)

# Sampler
sampler = nk.sampler.MetropolisLocal(hi, n_chains_per_rank=n_chains_per_rank)

# NQS model
alpha = 1
model = nk.models.RBM(alpha=alpha, param_dtype=complex)

# Variational state
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=model,
    n_samples=n_samples,
    n_discard_per_chain=n_discard_per_chain,
)

# Variational Monte Carlo driver
vmc = nk.driver.VMC(
    H, optimizer, preconditioner=preconditioner, variational_state=vstate
)
vmc.run(n_iter=n_iter)

with open(
    f"psi0_N={L}x{L}.mpack",
    "wb",
) as file:
    file.write(flax.serialization.to_bytes(vstate.variables))
