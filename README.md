# stochasticnst

Anisotropic Stochastic Cell Rescaling (SCR) NPT barostat for [ASE](https://wiki.fysik.dtu.dk/ase/).

Implements the fully anisotropic formulation of Del Tatto (2021) with a time-reversible
Trotterized integrator, extended to support a per-component compressibility tensor for
independent coupling of shear and diagonal degrees of freedom.

## Background

The SCR barostat is a first-order stochastic method that samples the isothermal-isobaric
(NPT) ensemble without the damped oscillations of deterministic second-order barostats
(Parrinello-Rahman, MTK). Cell fluctuations are governed by:

```
dh = -(β / 3τ_p) [(P_target − P_int) − (k_BT/V) I] h dt
     + sqrt(2β k_BT / (3V τ_p)) dW h
```

where `h` is the cell matrix, `β` the isothermal compressibility, `τ_p` the barostat
time constant, and `dW` a matrix of Wiener increments.

The compressibility can be supplied as a scalar (isotropic, recovering Ref. [1]) or as a
symmetric tensor applied element-wise (Ref. [1], Eq. 2.55), allowing stronger coupling for
shear components — reducing their autocorrelation time without affecting the volume
coupling or the sampled ensemble. The diagonal elements of the tensor must be equal;
unequal diagonal values would bias the sampled cell shape and are rejected.

## Installation

```bash
pip install stochasticnst
```

or with [uv](https://docs.astral.sh/uv/):

```bash
uv add stochasticnst
```

## Usage

```python
import numpy as np
import ase.build
import ase.units
from ase.calculators.emt import EMT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from stochasticnst import BussiSCR, IsotropicBussiSCR

# Build system
atoms = ase.build.bulk("Cu", crystalstructure="fcc", a=3.63).repeat((4, 4, 4))
atoms.calc = EMT()

# Initialise velocities
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
Stationary(atoms)

# Isotropic NPT at 300 K, 1 GPa
timestep = 2.0 * ase.units.fs
compressibility = 1.0 / (140.0 * ase.units.GPa)   # Cu bulk modulus ~140 GPa

md = BussiSCR(
    atoms,
    timestep=timestep,
    temperature_K=300.0,
    pressure_au=1.0 * ase.units.GPa,
    tdamp=100 * timestep,
    pdamp=1000 * timestep,
    compressibility_au=compressibility,
)
md.run(10_000)
```

### Isotropic cell fluctuations (liquids)

A liquid has no shear stiffness, so in a fully flexible cell its shape random-walks until
the box collapses into a slab or needle. Pass `isotropic=True` to rescale the cell
isotropically: only the volume fluctuates and the shape stays fixed (the isotropic SCR of
Ref. [2]). This requires a hydrostatic `pressure_au`. `IsotropicBussiSCR` is the same thing
as a class, mirroring ASE's `IsotropicMTKNPT`.

```python
md = BussiSCR(
    atoms,
    timestep=timestep,
    temperature_K=300.0,
    pressure_au=1.0 * ase.units.GPa,
    tdamp=100 * timestep,
    pdamp=1000 * timestep,
    compressibility_au=compressibility,
    isotropic=True,
)
```

### Anisotropic target pressure

Pass a symmetric 3×3 (or Voigt (6,)) target stress tensor (eV/Å³, compressive-positive
convention). It is applied as a constant target Cauchy stress; Ref. [1] instead uses a
reference-cell strain energy (Eq. 2.20), which agrees at the reference cell but differs at
finite strain.

```python
P_target = np.array([
    [1.0, 0.1, 0.0],
    [0.1, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]) * ase.units.GPa

md = BussiSCR(
    atoms,
    timestep=timestep,
    temperature_K=300.0,
    pressure_au=P_target,
    tdamp=100 * timestep,
    pdamp=1000 * timestep,
    compressibility_au=compressibility,
)
```

### Compressibility tensor — faster shear equilibration

Shear stress components relax more slowly than diagonal ones because they require
collective lattice distortion. Supplying a compressibility tensor with larger off-diagonal
values increases the shear coupling strength without affecting the volume coupling:

```python
# Scalar diagonal compressibility, 5x stronger shear coupling
beta_diag  = 1.0 / (140.0 * ase.units.GPa)
beta_shear = 5.0 * beta_diag

# Voigt order: [β_xx, β_yy, β_zz, β_yz, β_xz, β_xy]
compressibility_tensor = np.array([
    beta_diag, beta_diag, beta_diag,
    beta_shear, beta_shear, beta_shear,
])

md = BussiSCR(
    atoms,
    timestep=timestep,
    temperature_K=300.0,
    pressure_au=P_target,
    tdamp=100 * timestep,
    pdamp=1000 * timestep,
    compressibility_au=compressibility_tensor,
)
```

Or pass a full 3×3 array for complete control. A zero off-diagonal element freezes that
shear component.

## API

### `BussiSCR` / `IsotropicBussiSCR`

Keywords follow ASE's `MTKNPT` / `IsotropicMTKNPT` where they overlap
(`temperature_K`, `pressure_au`, `tdamp`, `pdamp`, `**kwargs` to `MolecularDynamics`), so
switching barostats is mostly a matter of changing the class and adding
`compressibility_au`. `BussiSCR` is the fully flexible cell (like `MTKNPT`);
`isotropic=True`, or equivalently `IsotropicBussiSCR`, rescales the cell isotropically
(like `IsotropicMTKNPT`). Unlike MTK,
`tdamp` and `pdamp` are relaxation times of first-order stochastic dynamics rather than
oscillation periods, so the cell relaxes without overshooting.

| Parameter | Type | Description |
|---|---|---|
| `atoms` | `Atoms` | ASE Atoms object with attached calculator |
| `timestep` | `float` | Timestep in ASE time units |
| `temperature_K` | `float` | Target temperature (K) |
| `pressure_au` | scalar, (6,), or (3,3) | Target pressure (eV/Å³, compressive-positive); a tensor gives a target stress (`BussiSCR` only) |
| `tdamp` | `float` | Bussi thermostat relaxation time (ASE time units), typically 100 × `timestep` |
| `pdamp` | `float` | Barostat relaxation time (ASE time units), typically 1000 × `timestep` |
| `compressibility_au` | scalar, (6,), or (3,3) | Isothermal compressibility (Å³/eV); diagonal elements must be equal |
| `rng` | `Generator` or None | NumPy random generator (shared with thermostat) |
| `isotropic` | `bool` | Volume-only rescaling, fixed cell shape (like `IsotropicMTKNPT`); default `False`. `IsotropicBussiSCR` sets it to `True` |

## Pressure convention

`pressure_au` uses the **compressive-positive** convention, as in ASE's MTK barostats:
`P_int = −sigma`, where `sigma = atoms.get_stress(voigt=False, include_ideal_gas=True)`.
This is the **negative** of ASE's default stress convention.

## Validation

Errors are block-averaging standard errors; σ is the deviation from the reference in units
of that error.

### Ideal gas against the exact result

ASE's `IdealGas` calculator (32 Ar atoms, 300 K, 5 fs timestep, `tdamp = 20 dt`,
`pdamp = 100 dt`, 300k steps) has an exact NPT volume distribution
p(V) ∝ Vᴺ exp(−P₀V/kT) for both the isotropic and the fully flexible cell, so
⟨V⟩ = (N+1)kT/P₀ = 1000 Å³ and std(V) = √(N+1) kT/P₀ = 174.1 Å³. This exercises the
full MD step (thermostat, half kicks, momentum rescaling, kinetic pressure).

| Barostat | ⟨V⟩ (Å³) | std(V) (Å³) |
|---|---|---|
| `BussiSCR` (4 seeds) | 1004.5 ± 2.5 (+1.8σ) | 173.0 ± 1.4 (−0.8σ) |
| `BussiSCR(isotropic=True)` | 999.9 ± 5.2 (0.0σ) | 177.6 ± 3.4 (+1.0σ) |
| ASE `IsotropicMTKNPT` | 998.4 ± 0.7 (−2.4σ) | 173.5 ± 2.5 (−0.3σ) |
| ASE `MTKNPT` | 998.8 ± 0.7 (−1.8σ) | 180.5 ± 7.0 (+0.9σ) |

The flexible-cell mean may sit up to ~0.5 % high at this short `pdamp`, consistent with the
τ_p-dependent integration error reported for the anisotropic integrator in Ref. [1]
(Fig. 3.1); it is not statistically conclusive.

### Cell-only model against Monte Carlo

The barostat step alone, driven by an elastic cell energy U = (K/2)‖hᵀh − G₀‖² plus an
average kinetic pressure NkT/V (N = 4, which makes the kT/V corrections large), samples
p(h) ∝ Vᴺ⁻² exp(−(U + P₀V)/kT) (Ref. [1], Eq. 2.2). 800k barostat steps at
`pdamp = 50 dt` compared with Metropolis Monte Carlo of the same density
(G = hᵀh):

| Compressibility | ⟨V⟩ | ⟨G_xy²⟩ | ⟨(G_xx − G_zz)²⟩ |
|---|---|---|---|
| Monte Carlo reference | 26.262 ± 0.003 | 0.2532 ± 0.0005 | 1.018 ± 0.002 |
| scalar β | 26.285 ± 0.026 | 0.258 ± 0.004 | 1.046 ± 0.017 |
| β with 5× shear elements | 26.255 ± 0.031 | 0.257 ± 0.002 | 1.014 ± 0.016 |
| unequal diagonal β (now rejected) | 26.267 ± 0.024 | 0.250 ± 0.004 | 1.102 ± 0.017 (+5σ) |

With `isotropic=True` the cell shape stays fixed and the volume matches the exact
one-dimensional distribution p(V) ∝ Vᴺ exp(−(U(V) + P₀V)/kT): ⟨V⟩ = 26.863 ± 0.027
against 26.891 (−1.1σ), ⟨V²⟩ = 724.9 ± 1.4 against 726.4 (−1.0σ).

### Cu (EMT) against ASE's MTK barostats

108 Cu atoms, 600 K, 1 GPa, 2 fs timestep, `pdamp = 100 dt`, 75k sampled steps:

| Barostat | ⟨V/atom⟩ (Å³) | std(V/atom) (Å³) |
|---|---|---|
| `BussiSCR` | 11.909 ± 0.005 | 0.088 ± 0.002 |
| `BussiSCR(isotropic=True)` | 11.916 ± 0.005 | 0.091 ± 0.003 |
| ASE `IsotropicMTKNPT` | 11.910 ± 0.002 | 0.093 ± 0.003 |
| ASE `MTKNPT` | 11.918 ± 0.002 | 0.120 ± 0.004 |

All mean volumes agree. For a cubic crystal the volume spread should not depend on
whether the shape can change; both SCR variants and `IsotropicMTKNPT` agree on it. Given
that `MTKNPT` reproduces the exact ideal-gas spread, its larger value here most likely
reflects slow convergence of its nine-variable cell dynamics rather than a wrong ensemble.

## References

[1] V. Del Tatto, *A Fully Anisotropic Formulation of Stochastic Cell Rescaling*,
    Master's Thesis, University of Trento (2021). <https://arxiv.org/abs/2111.06402>

[2] M. Bernetti and G. Bussi, *Pressure control using stochastic cell rescaling*,
    J. Chem. Phys. **153**, 114107 (2020). <https://doi.org/10.1063/5.0020514>

[3] G. Bussi, D. Donadio and M. Parrinello, *Canonical sampling through velocity
    rescaling*, J. Chem. Phys. **126**, 014101 (2007). <https://doi.org/10.1063/1.2408420>
