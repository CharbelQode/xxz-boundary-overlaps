# Overlap Computation in the Open Boundary XXZ Spin Chain

Numerical computation of ground-state overlaps in the XXZ spin-1/2 chain with open boundary conditions and diagonal (z-direction) boundary magnetic fields, using exact diagonalisation via [QuSpin](https://weinbe58.github.io/QuSpin/).

## Model

The Hamiltonian is:

$$H = \sum_{i=0}^{L-2} \left[ J_{xy} (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + J_{zz} S^z_i S^z_{i+1} \right] + h_\text{left} S^z_0 + h_\text{right} S^z_{L-1}$$

with couplings fixed by the anisotropy parameter $\eta$:
- $J_{xy} = 1$ (energy scale)
- $J_{zz} = \cosh(\eta)$

The code computes the squared overlap $|\langle \psi_0(h_\text{ref}) | \psi_0(h) \rangle|^2$ between the ground state of a reference Hamiltonian and ground states obtained by scanning the left boundary field $h_\text{left}$ over a range of values.

## Requirements

- Python 3.9
- QuSpin 0.3.7
- NumPy 1.25
- SciPy
- Matplotlib

## Installation

```bash
conda env create -f environment.yml
conda activate quspin-env
```

## Usage

Run the script directly:

```bash
python xxz_overlap_boundary_fields.py
```

Or open it in Jupyter Notebook — make sure to select the **Python (quspin-env)** kernel.

## Output

A plot of the squared ground-state overlap as a function of the left boundary field, saved as `xxz_overlap_vs_boundary_field.pdf`.

## Author

Charbel
