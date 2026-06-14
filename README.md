# stochasticnst

Anisotropic Stochastic Cell Rescaling (SCR) NPT barostat for [ASE](https://wiki.fysik.dtu.dk/ase/).

Implements the fully anisotropic formulation of Del Tao (2021) with a time-reversible
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
full 3×3 symmetric tensor, allowing stronger coupling for shear components — reducing
their autocorrelation time without affecting the volume coupling or the sampled ensemble.

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
from stochasticnst import BussiSCR

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
    taut=100 * timestep,
    taup=1000 * timestep,
    compressibility_au=compressibility,
)
md.run(10_000)
```

### Anisotropic target pressure

Pass a symmetric 3×3 pressure tensor (eV/Å³, compressive-positive convention):

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
    pressure_au=0.0,                        # overridden by tensor
    pressure_tensor_au=P_target,
    taut=100 * timestep,
    taup=1000 * timestep,
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
    pressure_au=0.0,
    pressure_tensor_au=P_target,
    taut=100 * timestep,
    taup=1000 * timestep,
    compressibility_au=compressibility_tensor,
)
```

Or pass a full 3×3 array for complete control.

## API

### `BussiSCR`

| Parameter | Type | Description |
|---|---|---|
| `atoms` | `Atoms` | ASE Atoms object with attached calculator |
| `timestep` | `float` | Timestep in ASE time units |
| `temperature_K` | `float` | Target temperature (K) |
| `pressure_au` | `float` | Target isotropic pressure (eV/Å³) |
| `taut` | `float` | Bussi thermostat time constant (ASE time units) |
| `taup` | `float` | Barostat relaxation time constant (ASE time units) |
| `compressibility_au` | scalar, (6,), or (3,3) | Isothermal compressibility (Å³/eV) |
| `pressure_tensor_au` | (3,3) array or None | Anisotropic target pressure tensor (eV/Å³); overrides `pressure_au` |
| `rng` | `Generator` or None | NumPy random generator (shared with thermostat) |

## Pressure convention

`pressure_au` and `pressure_tensor_au` use the **compressive-positive** convention:
`P_int = −sigma`, where `sigma = atoms.get_stress(voigt=False, include_ideal_gas=True)`.
This is the **negative** of ASE's default stress convention.

## References

[1] V. Del Tao, *A Fully Anisotropic Formulation of Stochastic Cell Rescaling*,
    Master's Thesis, University of Trento (2021). <https://arxiv.org/abs/2111.06402>

[2] M. Bernetti and G. Bussi, *Pressure control using stochastic cell rescaling*,
    J. Chem. Phys. **153**, 114107 (2020). <https://doi.org/10.1063/5.0020514>
