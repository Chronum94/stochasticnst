# fmt: off

"""Stochastic Cell Rescaling (SCR) NPT barostat.

Anisotropic formulation with time-reversible integrator.

References
----------
.. [1] V. Del Tatto, "A Fully Anisotropic Formulation of Stochastic Cell
       Rescaling", Master's Thesis, University of Trento (2021),
       https://arxiv.org/abs/2111.06402

.. [2] M. Bernetti and G. Bussi, "Pressure control using stochastic cell
       rescaling", J. Chem. Phys. 153, 114107 (2020),
       https://doi.org/10.1063/5.0020514

.. [3] G. Bussi, D. Donadio and M. Parrinello, "Canonical sampling through
       velocity rescaling", J. Chem. Phys. 126, 014101 (2007),
       https://doi.org/10.1063/1.2408420
"""

import numpy as np
from scipy.linalg import expm

from ase.md.bussi import Bussi
from ase.md.verlet import VelocityVerlet


class BussiSCR(VelocityVerlet):
    """Anisotropic Stochastic Cell Rescaling (SCR) NPT molecular dynamics.

    Samples the isothermal-isobaric (NPT) ensemble using the anisotropic SCR
    barostat [1]_ combined with the Bussi stochastic velocity rescaling
    thermostat [3]_.

    The barostat propagates the simulation cell via a first-order stochastic
    equation that is the high-friction limit of the Parrinello-Rahman equations
    augmented with Langevin noise.  The time-reversible integrator from
    Algorithm 7 of Ref. [1]_ is used:

    1. Propagate the volume degree of freedom (lam = sqrt(V)) for half a
       timestep.
    2. Propagate the nine cell-matrix shape degrees of freedom for a full
       timestep, using a matrix-exponential rescaling with the noise prefactor
       evaluated at the intermediate volume from step 1.
    3. Propagate lam for the remaining half timestep.

    The resulting rescaling matrix mu is applied to both atomic positions and
    momenta:

        q  ->  mu q
        p  ->  mu^{-T} p

    This preserves the canonical conjugacy of positions and momenta under the
    cell rescaling transformation.

    Parameters
    ----------
    atoms : Atoms
        The atoms to simulate.
    timestep : float
        Integration timestep in ASE time units.
    temperature_K : float
        Target temperature in Kelvin.
    pressure_au : scalar, (6,) array, or (3,3) array
        Target pressure in eV/Ang^3, as in :class:`~ase.md.nose_hoover_chain.MTKNPT`.
        A tensor gives a target stress, using the *pressure* convention
        (compressive positive).  Note this is the *negative* of ASE's stress
        tensor convention (``atoms.get_stress`` returns compressive-negative).

        Three forms are accepted:

        * **scalar** — isotropic hydrostatic pressure; equivalent to
          ``scalar * np.eye(3)``.
        * **(6,) Voigt array** ``[P_xx, P_yy, P_zz, P_yz, P_xz, P_xy]`` —
          full symmetric stress tensor in Voigt notation.
        * **(3,3) array** — full symmetric stress tensor.

    tdamp : float
        Characteristic time scale of the Bussi thermostat (its relaxation
        time) in ASE time units.  Typically 100 times ``timestep``.
    pdamp : float
        Characteristic time scale of the barostat (the SCR relaxation time
        tau_p of the volume) in ASE time units.  Typically 1000 times
        ``timestep``.
    compressibility_au : scalar, (6,) array, or (3,3) array
        Isothermal compressibility of the system in Ang^3/eV.

        Three forms are accepted (same as ``pressure_au``):

        * **scalar** — isotropic compressibility; used uniformly for all
          nine components, i.e. ``scalar * np.ones((3, 3))`` (the original
          formulation of Ref. [1]_, Eq. 2.17).
        * **(6,) Voigt array** ``[β_xx, β_yy, β_zz, β_yz, β_xz, β_xy]`` —
          full symmetric compressibility tensor in Voigt notation.
        * **(3,3) array** — full symmetric compressibility tensor.

        The diagonal elements must be equal (see Notes).  A zero
        off-diagonal element freezes that shear component.

        The compressibility tensor controls the per-component coupling strength
        between the internal pressure deviation and the cell update rate:
        larger β_ij means stronger coupling for component (i,j).  For the
        volume (isotropic) degree of freedom the trace-mean ``Tr(β)/3`` is
        used.  For shear components, setting β_ij > β_diag (i≠j) is
        equivalent to using a shorter effective barostat time constant for
        those components, reducing their autocorrelation time without
        affecting the diagonal coupling or the sampled ensemble (the
        fluctuation-dissipation balance is maintained per-component).

        Typical values (bulk modulus → β = 1/B):
          - Diamond: B ≈ 440 GPa,  shear ≈ 440–480 GPa
          - Osmium:  B ≈ 460 GPa,  shear ≈ 220 GPa
          - Copper:  B ≈ 140 GPa,  shear ≈  50 GPa
          - Gold:    B ≈ 180 GPa,  shear ≈  27 GPa

    rng : numpy.random.Generator or None, optional
        Random number generator shared by both the thermostat and barostat.
        Defaults to ``numpy.random``.
    **kwargs
        Additional keyword arguments passed to
        :class:`~ase.md.verlet.VelocityVerlet`.

    Notes
    -----
    Sign convention
    ~~~~~~~~~~~~~~~
    Throughout this implementation *pressure* convention is used:

        P_int = -sigma,
        where sigma = atoms.get_stress(voigt=False, include_ideal_gas=True)

    So P_int[i,i] > 0 means the system is under compression along axis i.
    ``pressure_au`` follows the same convention.

    SCR equation
    ~~~~~~~~~~~~
    The SCR equations (Ref. [1]_, Eq. 2.17) read

        dh = -(beta_T / 3 tau_p) [(P_target - P_int) - (k_BT / V) I] h dt
             + sqrt(2 beta_T k_BT / (3 V tau_p)) dW h

    where h is the box matrix (columns = cell vectors), beta_T is the
    isothermal compressibility, tau_p the barostat time constant (= ``pdamp``),
    P_target the target stress tensor (= ``pressure_au``), P_int the instantaneous
    internal pressure tensor, V the volume, and dW a 3x3 matrix of
    independent Wiener increments.

    When a compressibility *tensor* beta_ij is provided, the scalar beta_T
    in the above is replaced element-wise: the coupling strength for
    component (i,j) becomes beta_ij / tau_p, and the noise amplitude scales
    as sqrt(beta_ij) (Ref. [1]_, Eq. 2.55).  The diagonal elements must be
    equal: otherwise the volume and shape noise are correlated, which the
    independent volume/shape steps of Algorithm 7 cannot reproduce, and the
    sampled shape distribution is biased.  With equal diagonal elements the
    off-diagonal (shear) elements only change relaxation rates, not the
    sampled ensemble.

    A non-hydrostatic ``pressure_au`` is applied as a constant target Cauchy
    stress, not via the reference-cell strain energy (h Sigma h^T) / V of
    Ref. [1]_, Eq. 2.20.  The two agree at the reference cell and differ at
    finite strain.

    Box rotations are eliminated by symmetrising both the deterministic drift
    (P_int -> (P_int + P_int.T) / 2) and the noise before computing the
    matrix exponential.

    The matrix exponential for the shape step is computed with
    ``scipy.linalg.expm``.
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        pressure_au,
        tdamp,
        pdamp,
        compressibility_au,
        rng=None,
        **kwargs,
    ):
        VelocityVerlet.__init__(self, atoms, timestep, **kwargs)
        self.thermostat = Bussi(
            atoms,
            timestep,
            temperature_K=temperature_K,
            taut=tdamp,
            rng=rng,
        )
        self.pdamp = pdamp
        self.set_pressure(pressure_au)
        self.compressibility_mat = self._parse_compressibility(compressibility_au)
        # Share the rng instance that Bussi resolved (handles rng=None case)
        self.rng = self.thermostat.rng

    # Shared tensor parser
    @staticmethod
    def _parse_symmetric_tensor(value, name):
        """Return a (3,3) symmetric matrix from scalar, Voigt (6,), or (3,3) input.

        Voigt order: [xx, yy, zz, yz, xz, xy]
        """
        v = np.asarray(value, dtype=float)
        if v.ndim == 0:
            return float(v) * np.eye(3)
        if v.shape == (6,):
            mat = np.zeros((3, 3))
            mat[0, 0], mat[1, 1], mat[2, 2] = v[0], v[1], v[2]
            mat[1, 2] = mat[2, 1] = v[3]
            mat[0, 2] = mat[2, 0] = v[4]
            mat[0, 1] = mat[1, 0] = v[5]
            return mat
        if v.shape == (3, 3):
            return v.copy()
        raise ValueError(
            f"{name} must be a scalar, a (6,) Voigt array, or a (3,3) array; "
            f"got shape {v.shape}"
        )

    @classmethod
    def _parse_compressibility(cls, value):
        """Return the (3,3) compressibility matrix used element-wise in the barostat.

        A scalar couples all nine components (beta * ones, Ref. [1]_ Eq. 2.17);
        beta * eye would zero every shear drift and noise term.
        """
        v = np.asarray(value, dtype=float)
        if v.ndim == 0:
            beta = float(v) * np.ones((3, 3))
        else:
            beta = cls._parse_symmetric_tensor(v, "compressibility_au")
        if not np.allclose(beta, beta.T):
            raise ValueError("compressibility_au must be symmetric")
        if np.any(beta < 0):
            raise ValueError("compressibility_au must be non-negative")
        if not np.allclose(np.diag(beta), beta[0, 0]):
            raise ValueError(
                "compressibility_au must have equal diagonal elements; unequal "
                "values bias the sampled cell shape distribution"
            )
        return beta

    def set_pdamp(self, pdamp):
        self.pdamp = pdamp

    def get_pdamp(self):
        return self.pdamp

    def set_pressure(self, pressure_au):
        self.pressure = self._parse_symmetric_tensor(pressure_au, "pressure_au")

    def get_pressure(self):
        return self.pressure

    def set_compressibility(self, compressibility_au):
        self.compressibility_mat = self._parse_compressibility(compressibility_au)

    def get_compressibility(self):
        return self.compressibility_mat

    # ------------------------------------------------------------------
    # Core barostat implementation
    # ------------------------------------------------------------------

    def _get_p_int(self):
        """Return the internal pressure tensor (3x3, eV/Ang^3, compressive positive).

        ASE get_stress returns the Cauchy stress sigma (compressive negative),
        so P_int = -sigma.
        """
        sigma = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        return -sigma

    def _propagate_lambda(self, lam, p_int_scalar, dt_half):
        """Euler half-step for lam = sqrt(V) (Ref. [1]_, Eq. 2.43, dt -> dt/2).

        Uses the isotropic mean compressibility Tr(beta_mat)/3 so that the
        volume degree of freedom always couples at a well-defined rate
        regardless of whether an anisotropic compressibility tensor was given.

        Parameters
        ----------
        lam : float
            Current value of lam = sqrt(V).
        p_int_scalar : float
            Isotropic internal pressure Tr(P_int) / 3.
        dt_half : float
            Half timestep.

        Returns
        -------
        float
            Updated lam after the half-step.
        """
        k_bt   = self.thermostat.temp
        # Isotropic mean compressibility for the volume DOF
        beta_t = np.trace(self.compressibility_mat) / 3.0
        tau_p  = self.pdamp
        # target mean pressure = Tr(P_target) / 3 (pressure convention)
        p0_scalar = np.trace(self.pressure) / 3.0
        vol = lam * lam

        # Deterministic drift (Eq. 1.108b, dt -> dt/2)
        drift = -(beta_t * lam / (2.0 * tau_p)) * (
            p0_scalar - p_int_scalar - k_bt / (2.0 * vol)
        ) * dt_half

        # Stochastic term: amplitude is sqrt(k_BT beta_T / (2 tau_p)) * sqrt(dt/2)
        noise_amp = np.sqrt(k_bt * beta_t / (2.0 * tau_p) * dt_half)
        r = self.rng.standard_normal()

        return lam + drift + noise_amp * r

    def scale_positions_and_cell(self):
        """Apply one SCR barostat step (Algorithm 7 of Ref. [1]_).

        Implements the time-reversible Trotterized integrator:

        1. Volume half-step  (propagate lam for dt/2)
        2. Shape full-step   (propagate cell-matrix shape for dt)
        3. Volume half-step  (propagate lam for dt/2)

        The total rescaling matrix mu = (V_new / V_old)^(1/3) * mu_s is then
        applied to atomic positions and momenta.
        """
        atoms  = self.atoms
        dt     = self.dt
        k_bt   = self.thermostat.temp   # eV
        tau_p  = self.pdamp

        # ---- Current state ----
        vol = atoms.get_volume()    # Ang^3
        lam = np.sqrt(vol)          # lam = sqrt(V)

        # Internal pressure tensor (3x3, eV/Ang^3, compressive positive).
        # Symmetrised to eliminate box rotations (Section 2.6 of Ref. [1]_).
        p_int = self._get_p_int()
        p_int = 0.5 * (p_int + p_int.T)

        p_int_scalar = np.trace(p_int) / 3.0   # isotropic part

        # ---- Element-wise compressibility matrix for the shape step ----
        # For a scalar compressibility this is beta_T * ones((3,3)) and the
        # equations reduce exactly to Ref. [1]_.  For a tensor, each (i,j)
        # component couples with its own strength beta_ij / tau_p while the
        # fluctuation-dissipation balance is maintained per-component.
        beta_mat = self.compressibility_mat

        # ---- Deterministic drift matrix A (traceless part) ----
        # A_ij = -(beta_ij / (3 tau_p)) (P_target - P_int)_ij
        a_mat = -(beta_mat / (3.0 * tau_p)) * (self.pressure - p_int)
        a_traceless = a_mat - (np.trace(a_mat) / 3.0) * np.eye(3)

        # ================================================================
        # Step 1: volume half-step  (uses isotropic Tr(beta)/3 internally)
        # ================================================================
        lam1 = self._propagate_lambda(lam, p_int_scalar, dt * 0.5)
        vol1 = lam1 * lam1

        # ================================================================
        # Step 2: shape full-step with b evaluated at intermediate volume vol1
        # ================================================================
        # Element-wise noise amplitude: b_ij = sqrt(2 beta_ij k_BT / (3 V1 tau_p) dt)
        # For uniform beta this reduces to the original scalar b_dt * ones.
        b_mat = np.sqrt(2.0 * beta_mat * k_bt / (3.0 * vol1 * tau_p) * dt)

        # 3x3 Gaussian noise for shape, symmetrised (no rotations) and made
        # traceless so that delta_eps stays in sl(3) (volume-preserving).
        # b_mat * w2 is symmetric, and removing its trace is independent of
        # the volume noise because the diagonal of b_mat is uniform (enforced
        # in _parse_compressibility).
        w2 = self.rng.standard_normal((3, 3))
        w2 = 0.5 * (w2 + w2.T)

        noise = b_mat * w2
        noise -= (np.trace(noise) / 3.0) * np.eye(3)

        # Argument of the matrix exponential  (Ref. [1]_, Eq. 2.39, generalised)
        delta_eps = a_traceless * dt + noise

        # Matrix exponential -> shape rescaling matrix (det = 1)
        mu_s = expm(delta_eps)

        # ================================================================
        # Step 3: volume half-step
        # ================================================================
        lam2 = self._propagate_lambda(lam1, p_int_scalar, dt * 0.5)
        vol2 = lam2 * lam2

        # ================================================================
        # Full rescaling matrix mu = (V2 / V0)^(1/3) * mu_s  (Ref. [1]_, Eq. 2.40)
        # ================================================================
        mu = (vol2 / vol) ** (1.0 / 3.0) * mu_s

        # ================================================================
        # Apply mu to cell, positions and momenta
        # ================================================================
        # ASE cell convention: cell[i] = i-th lattice vector (row).
        # Thesis convention:   h = cell.T  (columns = lattice vectors).
        #   h_new = mu h  =>  cell_new = (mu cell.T).T = cell mu.T
        # set_cell(scale_atoms=True) maps positions via fractional coordinates:
        #   pos_new[i] = pos[i] @ mu.T
        cell = atoms.get_cell()
        atoms.set_cell(cell @ mu.T, scale_atoms=True)

        # Momentum rescaling: p_new = mu^{-T} p_old
        # In ASE row-vector notation: mom_new[i] = mom_old[i] @ inv(mu)
        momenta = atoms.get_momenta()
        atoms.set_momenta(momenta @ np.linalg.inv(mu))

    def step(self, forces=None):
        """Move one timestep forward using SCR NPT + Bussi NVT dynamics.

        Follows Algorithm 8 of Ref. [1]_: the barostat acts between the
        velocity-Verlet half kick and the drift, so the stress is taken where
        forces are already known and there is one force evaluation per step.
        """
        atoms = self.atoms

        # Apply Bussi stochastic velocity rescaling (thermostat)
        self.thermostat.scale_velocities()

        if forces is None:
            forces = atoms.get_forces(md=True)

        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * forces)

        # Apply SCR barostat (cell + position + momentum rescaling)
        self.scale_positions_and_cell()

        p = atoms.get_momenta()
        atoms.set_positions(
            atoms.get_positions() + self.dt * p / atoms.get_masses()[:, np.newaxis]
        )

        forces = atoms.get_forces(md=True)
        atoms.set_momenta(atoms.get_momenta() + 0.5 * self.dt * forces)

        return forces
