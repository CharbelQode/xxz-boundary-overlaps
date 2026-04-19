"""
XXZ Open Spin Chain — Ground-State Overlap vs. Left Boundary Field
==================================================================
Computes the squared overlap  |<ψ₀(h_left_ref) | ψ₀(h_left)|²  between the
ground state of a reference Hamiltonian and the ground states obtained by
scanning the left boundary field h_left over a range of values.

Model: XXZ spin-1/2 chain of length L with open boundary conditions (OBC)
and diagonal (z-direction) boundary magnetic fields.

    H = Σᵢ [ Jxy (Sˣᵢ Sˣᵢ₊₁ + Sʸᵢ Sʸᵢ₊₁) + Jzz Sᶻᵢ Sᶻᵢ₊₁ ]
        + h_left  Sᶻ₀  +  h_right  Sᶻ_{L-1}

Parameters fixed by the anisotropy parameter η:
    Jzz  = cosh(η)
    Jxy  = 1   (sets the energy scale)

Dependencies: quspin, numpy, matplotlib, scipy
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from scipy import sparse


# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
L        = 8           # chain length
eta      = 1.5          # XXZ anisotropy parameter
Jxy      = 1.0          # XY coupling (energy scale)
Jzz      = math.cosh(eta)  # ZZ coupling fixed by η

h_left_ref = -1.0       # left boundary field for the reference Hamiltonian
h_right    =  2.0       # right boundary field (fixed throughout)

# Range of left boundary fields to scan
h_left_range = np.linspace(-5, 1, 20)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def build_hamiltonian(L: int, Jxy: float, Jzz: float,
                      h_left: float, h_right: float) -> hamiltonian:
    """Build the XXZ Hamiltonian with OBC and z-boundary fields.

    Parameters
    ----------
    L       : chain length
    Jxy     : XY coupling strength
    Jzz     : ZZ coupling strength
    h_left  : magnetic field on site 0  (z-direction)
    h_right : magnetic field on site L-1 (z-direction)

    Returns
    -------
    H : quspin hamiltonian object (dtype float64, Pauli convention)
    """
    basis = spin_basis_1d(L, pauli=True)

    J_zz_list = [[Jzz, i, i + 1] for i in range(L - 1)]
    J_xy_list = [[Jxy, i, i + 1] for i in range(L - 1)]

    # Boundary fields only at the two edges; bulk sites get zero field
    boundary_fields = [[h_left, 0]] \
                    + [[0.0, i] for i in range(1, L - 1)] \
                    + [[h_right, L - 1]]

    static = [
        ["xx", J_xy_list],
        ["yy", J_xy_list],
        ["zz", J_zz_list],
        ["z",  boundary_fields],
    ]

    return hamiltonian(static, [], basis=basis, dtype=np.float64)


def ground_state(H: hamiltonian) -> np.ndarray:
    """Return the ground-state eigenvector of H."""
    _, vecs = H.eigsh(k=2, which="BE", maxiter=int(1e5), return_eigenvectors=True)
    # eigsh with which="BE" returns the two extremal states;
    # the ground state (lowest energy) is the one at index 0
    return vecs[:, 0]


def squared_overlap(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Return the squared normalised overlap |<ψ_a|ψ_b>|² / (<ψ_a|ψ_a> <ψ_b|ψ_b>)."""
    psi_a_row = sparse.csr_array(psi_a.reshape(1, -1))
    psi_b_col = sparse.csr_array(psi_b.reshape(-1, 1))

    inner_ab = float(psi_a_row.dot(psi_b_col).toarray())
    norm_a    = float(psi_a_row.dot(sparse.csr_array(psi_a.reshape(-1, 1))).toarray())
    norm_b    = float(sparse.csr_array(psi_b.reshape(1, -1)).dot(psi_b_col).toarray())

    return (inner_ab ** 2) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Reference ground state  (h_left = h_left_ref)
# ---------------------------------------------------------------------------
H_ref    = build_hamiltonian(L, Jxy, Jzz, h_left_ref, h_right)
psi_ref  = ground_state(H_ref)

# ---------------------------------------------------------------------------
# Scan h_left and accumulate squared overlaps
# ---------------------------------------------------------------------------
overlaps_sq = np.zeros(len(h_left_range))

for idx, h_left in enumerate(h_left_range):
    H_scan              = build_hamiltonian(L, Jxy, Jzz, h_left, h_right)
    psi_scan            = ground_state(H_scan)
    overlaps_sq[idx]    = squared_overlap(psi_ref, psi_scan)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(h_left_range, overlaps_sq, marker="o", linewidth=1.5)
ax.set_xlabel(r"$h_\mathrm{left}$")
ax.set_ylabel(r"$|\langle \psi_0(h_\mathrm{ref}) | \psi_0(h) \rangle|^2$")
ax.set_title(
    rf"Squared ground-state overlap — XXZ chain, $L={L}$, $\eta={eta}$, "
    rf"$h_\mathrm{{right}}={h_right}$"
)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("xxz_overlap_vs_boundary_field.pdf")
plt.show()
