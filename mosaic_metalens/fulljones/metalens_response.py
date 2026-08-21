from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def polar_angle_from_axis_tilts(
    theta_x_rad: torch.Tensor,
    theta_y_rad: torch.Tensor,
) -> torch.Tensor:
    """Return the physical polar angle represented by separable x/y tilts.

    ``theta_x_rad`` and ``theta_y_rad`` are the angles whose tangents are the
    transverse ray slopes.  Consequently the polar slope is
    ``hypot(tan(theta_x), tan(theta_y))``.  The often-used Euclidean angle
    approximation ``hypot(theta_x, theta_y)`` is only paraxial and is already
    off by about 0.46 degrees for the paper's diagonal field.
    """

    return torch.atan(torch.hypot(torch.tan(theta_x_rad), torch.tan(theta_y_rad)))


class BaseResponseModel(nn.Module, ABC):
    response_contract = "legacy_scalar_or_diagonal"

    @abstractmethod
    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def validate_final_exact_protocol(self) -> None:
        """Reject legacy scalar/phase-only responses in final-exact runs."""
        raise RuntimeError(
            f"{self.__class__.__name__} exposes response contract "
            f"{self.response_contract!r}; final-exact propagation requires "
            "a fail-closed full complex Jones LUT"
        )


class IdentityResponseModel(BaseResponseModel):
    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        del width_map_um, wavelengths_um, theta_y_rad
        return torch.ones_like(theta_x_rad, dtype=torch.complex64)


class LookupResponseModel(BaseResponseModel):
    response_contract = "legacy_scalar_phase_lut"

    def __init__(
        self,
        width_grid_um: torch.Tensor,
        wavelength_grid_um: torch.Tensor,
        angle_grid_deg: torch.Tensor,
        phase_grid_rad: torch.Tensor,
        amplitude_grid: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if phase_grid_rad.shape != (
            width_grid_um.numel(),
            wavelength_grid_um.numel(),
            angle_grid_deg.numel(),
        ):
            raise ValueError("phase_grid_rad must be [Nw,Nlambda,Ntheta].")
        if amplitude_grid is not None and amplitude_grid.shape != phase_grid_rad.shape:
            raise ValueError("amplitude_grid must match phase_grid_rad.")

        self.register_buffer("width_grid_um", width_grid_um.to(torch.float32))
        self.register_buffer("wavelength_grid_um", wavelength_grid_um.to(torch.float32))
        self.register_buffer("angle_grid_deg", angle_grid_deg.to(torch.float32))
        self.register_buffer("phase_grid_rad", phase_grid_rad.to(torch.float32))
        if amplitude_grid is None:
            amplitude_grid = torch.ones_like(phase_grid_rad, dtype=torch.float32)
        self.register_buffer("amplitude_grid", amplitude_grid.to(torch.float32))

    def _weights(self, values: torch.Tensor, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = values.clamp(grid[0], grid[-1])
        hi = torch.bucketize(values, grid)
        hi = hi.clamp(1, grid.numel() - 1)
        lo = hi - 1
        g_lo = grid[lo]
        g_hi = grid[hi]
        t = (values - g_lo) / (g_hi - g_lo).clamp_min(1e-12)
        return lo, hi, t

    def _gather_trilinear(
        self,
        table: torch.Tensor,
        width_values: torch.Tensor,
        wavelength_values: torch.Tensor,
        angle_values: torch.Tensor,
    ) -> torch.Tensor:
        nw, nl, na = table.shape
        flat = table.reshape(-1)

        iw0, iw1, tw = self._weights(width_values, self.width_grid_um)
        il0, il1, tl = self._weights(wavelength_values, self.wavelength_grid_um)
        ia0, ia1, ta = self._weights(angle_values, self.angle_grid_deg)

        def gather(iw: torch.Tensor, il: torch.Tensor, ia: torch.Tensor) -> torch.Tensor:
            idx = (iw * nl + il) * na + ia
            return flat[idx]

        c000 = gather(iw0, il0, ia0)
        c001 = gather(iw0, il0, ia1)
        c010 = gather(iw0, il1, ia0)
        c011 = gather(iw0, il1, ia1)
        c100 = gather(iw1, il0, ia0)
        c101 = gather(iw1, il0, ia1)
        c110 = gather(iw1, il1, ia0)
        c111 = gather(iw1, il1, ia1)

        one = torch.ones_like(tw)
        w000 = (one - tw) * (one - tl) * (one - ta)
        w001 = (one - tw) * (one - tl) * ta
        w010 = (one - tw) * tl * (one - ta)
        w011 = (one - tw) * tl * ta
        w100 = tw * (one - tl) * (one - ta)
        w101 = tw * (one - tl) * ta
        w110 = tw * tl * (one - ta)
        w111 = tw * tl * ta

        return (
            w000 * c000
            + w001 * c001
            + w010 * c010
            + w011 * c011
            + w100 * c100
            + w101 * c101
            + w110 * c110
            + w111 * c111
        )

    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        theta_deg = torch.rad2deg(
            polar_angle_from_axis_tilts(theta_x_rad, theta_y_rad)
        )
        # pipeline passes theta as [chunk, 1, H, W] (lambda broadcast implied);
        # expand all three inputs to the common broadcast shape
        shape = torch.broadcast_shapes(
            theta_deg.shape,
            (1, wavelengths_um.numel(), 1, 1),
            (1, 1) + tuple(width_map_um.shape),
        )
        theta_deg = theta_deg.expand(shape)
        full_width = width_map_um.view(1, 1, *width_map_um.shape).expand(shape)
        full_lambda = wavelengths_um.view(1, -1, 1, 1).expand(shape)

        phase = self._gather_trilinear(
            self.phase_grid_rad,
            full_width.reshape(-1),
            full_lambda.reshape(-1),
            theta_deg.reshape(-1),
        ).reshape_as(theta_deg)
        amplitude = self._gather_trilinear(
            self.amplitude_grid,
            full_width.reshape(-1),
            full_lambda.reshape(-1),
            theta_deg.reshape(-1),
        ).reshape_as(theta_deg)
        return amplitude.to(torch.complex64) * torch.exp(1j * phase.to(torch.complex64))


    @classmethod
    def from_rcwa_anglesweep(
        cls,
        txt_path: str,
        wl_min_nm: float = 400.0,
        wl_max_nm: float = 700.0,
    ) -> "LookupResponseModel":
        """Load full RCWA angle-sweep data (P290nm_H500nm format).

        The txt file has columns: P, H, W, angle, phase@wl1, phase@wl2, ...
        Returns LookupResponseModel with full width range (no truncation).
        """
        import numpy as np
        data = np.loadtxt(txt_path)
        widths = np.unique(data[:, 2])  # nm
        angles = np.unique(data[:, 3])  # deg
        # Format validation (2026-07-02 review): every txt currently in
        # data/rcwa uses [P, H, W, phase@51λ] — col 3 is a WRAPPED PHASE,
        # not an angle. Loading such a file silently produced a 98.7%-zeros
        # phase grid. Require a true angle-sweep layout.
        n_expected = len(widths) * len(angles)
        looks_like_angles = (angles.min() >= 0.0 and angles.max() <= 90.0
                             and len(angles) >= 2)
        if data.shape[0] != n_expected or not looks_like_angles:
            raise ValueError(
                f"from_rcwa_anglesweep: '{txt_path}' does not match the "
                f"[P, H, W, angle, phase@λ...] angle-sweep layout "
                f"(rows={data.shape[0]}, widths×angles={n_expected}, "
                f"col3 range=[{angles.min():.3g},{angles.max():.3g}]). "
                f"The P290nm_H*nm matrices are [P,H,W,phase@51λ] normal-"
                f"incidence tables — use HybridLUTResponseModel instead.")
        n_wl = data.shape[1] - 4
        wls = np.linspace(wl_min_nm, wl_max_nm, n_wl)  # nm

        phase_grid = np.zeros((len(widths), n_wl, len(angles)))
        for row in data:
            iw = np.searchsorted(widths, row[2])
            ia = np.searchsorted(angles, row[3])
            phase_grid[iw, :, ia] = row[4:]
        # unwrap along width and angle axes before interpolation (wrapped
        # phase must never be linearly interpolated across a 2π seam)
        phase_grid = np.unwrap(np.unwrap(phase_grid, axis=0), axis=2)

        return cls(
            width_grid_um=torch.from_numpy(widths / 1000.0),
            wavelength_grid_um=torch.from_numpy(wls / 1000.0),
            angle_grid_deg=torch.from_numpy(angles),
            phase_grid_rad=torch.from_numpy(phase_grid),
        )


class FullJonesLUTResponseModel(BaseResponseModel):
    """Four-dimensional complex Jones LUT in the local incidence basis.

    The Jones matrix is indexed as ``J[out_pol, in_pol]`` with polarization
    order ``(s, p)`` and table shape
    ``[Nwidth, Nlambda, Npolar, Nazimuth, 2, 2]``. Its four entries are
    ``[[t_ss, t_sp], [t_ps, t_pp]]`` and are power-normalized scattering
    amplitudes in the unit-vector local basis used by the pipeline,
    ``s = z×k/|z×k|`` and ``p = k×s``. The model interpolates the
    complex entries themselves.  The forward assembler converts physical
    incident E to modal power amplitude with ``sqrt(n_in cos(theta_in))`` and
    converts the Jones output back to E with
    ``1/sqrt(n_out cos(theta_out))``; input and output p vectors are built from
    their respective wavevectors under conserved tangential k. Thus a
    power-normalized coefficient is never silently treated as a raw E-field
    transmission coefficient;
    phase is never unwrapped or interpolated as a standalone scalar.

    NPZ schema ``engine2.full_jones_lut.v1``
    ------------------------------------------------
    Required arrays are:

    - ``widths_nm``: strictly increasing ``[Nwidth]``;
    - ``wavelengths_nm``: strictly increasing ``[Nlambda]``;
    - ``polar_angles_deg``: strictly increasing ``[Npolar]``;
    - ``azimuth_angles_deg``: strictly increasing ``[Nazimuth]`` with no
      implicit C4/mirror folding;
    - ``jones_real`` and ``jones_imag``: finite arrays of shape
      ``[Nwidth,Nlambda,Npolar,Nazimuth,2,2]``;
    - ``metadata_json``: scalar JSON string satisfying
      :meth:`validate_metadata`.

    Angles are derived from the separable ray tilts as
    ``polar=atan(hypot(tan(theta_x),tan(theta_y)))`` and
    ``azimuth=atan2(tan(theta_y),tan(theta_x))`` in ``[-180,180]`` degrees.
    No reciprocity or point-group symmetry is imposed on coefficients. If a
    generator used either operation to transform/expand the table, that fact
    and its validation status must be declared in metadata.
    """

    response_contract = "full_complex_jones_local_sp_v1"
    SCHEMA_VERSION = "engine2.full_jones_lut.v1"
    BASIS = "local_sp"
    MATRIX_ORDER = "output_sp_by_input_sp"
    # torcwa ``S_parameters(..., power_norm=True)`` returns modal scattering
    # amplitudes whose squared magnitudes are power ratios.  Calling these raw
    # electric-field amplitudes is incorrect when the input and output media
    # differ, and would make the substrate-to-air orientation ambiguous.
    COEFFICIENT_CONVENTION = (
        "power_normalized_scattering_amplitude_in_unit_sp_basis"
    )
    ANGLE_CONVENTION = "polar_atan_hypot_tan__azimuth_atan2_tan_deg"
    PRODUCTION_ORIENTATION = "substrate_to_posts_to_air_direct"
    PRODUCTION_INCIDENT_MEDIUM = "fused_silica"
    PRODUCTION_ANGLE_MEDIUM = "input_fused_silica"
    PRODUCTION_PATTERNED_LAYER = "stoichiometric_sin_square_posts_in_air"
    # Admissible pillar dielectrics for the final-exact stack.  Every entry
    # shares the identical fused-silica -> square-posts-in-air -> air geometry
    # and differs only in the pillar material.  The SiN production label stays
    # accepted; the amorphous-TiO2 and amorphous-SiO2 generality libraries are
    # additionally admitted.
    ACCEPTED_PATTERNED_LAYERS = frozenset(
        {
            PRODUCTION_PATTERNED_LAYER,
            "amorphous_tio2_square_posts_in_air",
            "amorphous_sio2_square_posts_in_air",
        }
    )
    PRODUCTION_OUTPUT_MEDIUM = "air"
    PRODUCTION_BASIS_CONVERSION = (
        "torcwa_sp_to_pipeline_sp__flip_input_and_output_p_axes"
    )
    PRODUCTION_MODAL_E_CONVERSION = (
        "E_to_power_sqrt_ncostheta__Jpower__power_to_E_inv_sqrt_ncostheta"
    )
    VALIDATED_ARTIFACT_STATUS = "validated"
    MODEL_CONDITIONAL_ARTIFACT_STATUS = "model_conditional_validated"
    MODEL_CONDITIONAL_VALIDATION_MODE = "declared_exact_c4_rcwa_model"
    MODEL_CONDITIONAL_AUDIT_STATUS = "model_conditional"
    DECLARED_EXACT_SYMMETRY_STATUS = "declared_exact_model"
    MODEL_BASIS_SCHEMA = "engine2.full_jones_model_basis.v1"
    MODEL_BASIS_SCOPE = "direct_full_jones_rcwa_c4_projected"
    MODEL_BASIS_PHYSICS = "square_post_square_lattice_c4_rotation_exact"
    PRODUCTION_AUDIT_KEYS = (
        "stack_orientation",
        "coefficient_normalization",
        "basis_conversion",
        "grid_coverage",
        "rcwa_convergence",
        "spectral_convergence",
        "angular_convergence",
        "interpolation_holdout",
        "energy_balance",
    )
    NPZ_REQUIRED_KEYS = frozenset(
        {
            "widths_nm",
            "wavelengths_nm",
            "polar_angles_deg",
            "azimuth_angles_deg",
            "jones_real",
            "jones_imag",
            "metadata_json",
        }
    )
    _AUDIT_STATUSES = frozenset(
        {
            "validated",
            "declared_exact_model",
            "not_used",
            "not_assumed",
            "not_evaluated",
            "failed",
        }
    )

    def __init__(
        self,
        width_grid_um: torch.Tensor,
        wavelength_grid_um: torch.Tensor,
        polar_angle_grid_deg: torch.Tensor,
        azimuth_grid_deg: torch.Tensor,
        jones_grid: torch.Tensor,
        *,
        metadata: dict[str, object],
        fail_closed: bool = True,
    ) -> None:
        super().__init__()
        self.validate_metadata(metadata)
        width = self._validated_axis(width_grid_um, "width_grid_um")
        wavelength = self._validated_axis(
            wavelength_grid_um, "wavelength_grid_um"
        )
        polar = self._validated_axis(
            polar_angle_grid_deg, "polar_angle_grid_deg"
        )
        azimuth = self._validated_axis(azimuth_grid_deg, "azimuth_grid_deg")
        expected = (
            width.numel(),
            wavelength.numel(),
            polar.numel(),
            azimuth.numel(),
            2,
            2,
        )
        jones = torch.as_tensor(jones_grid)
        if tuple(jones.shape) != expected:
            raise ValueError(
                f"jones_grid must have shape {expected}, got {tuple(jones.shape)}"
            )
        if not (torch.isfinite(jones.real).all() and torch.isfinite(jones.imag).all()):
            raise ValueError("jones_grid must contain only finite complex values")

        self.register_buffer("width_grid_um", width)
        self.register_buffer("wavelength_grid_um", wavelength)
        self.register_buffer("polar_angle_grid_deg", polar)
        self.register_buffer("azimuth_grid_deg", azimuth)
        self.register_buffer("jones_grid", jones.to(torch.complex64))
        self.fail_closed = bool(fail_closed)
        # JSON round-trip prevents callers from mutating nested metadata after
        # construction and keeps the contract serializable without pickle.
        self.lut_metadata = json.loads(json.dumps(metadata))

    @staticmethod
    def _validated_axis(values: torch.Tensor, name: str) -> torch.Tensor:
        axis = torch.as_tensor(values, dtype=torch.float32).detach().clone()
        if axis.ndim != 1 or axis.numel() < 2:
            raise ValueError(f"{name} must be one-dimensional with at least 2 values")
        if not bool(torch.isfinite(axis).all()):
            raise ValueError(f"{name} must contain only finite values")
        if not bool((axis[1:] > axis[:-1]).all()):
            raise ValueError(f"{name} must be strictly increasing")
        return axis

    @classmethod
    def validate_metadata(cls, metadata: dict[str, object]) -> None:
        if not isinstance(metadata, dict):
            raise TypeError("Jones LUT metadata must be a dictionary")
        exact_values = {
            "schema_version": cls.SCHEMA_VERSION,
            "basis": cls.BASIS,
            "matrix_order": cls.MATRIX_ORDER,
            "coefficient_convention": cls.COEFFICIENT_CONVENTION,
            "angle_convention": cls.ANGLE_CONVENTION,
        }
        for key, expected in exact_values.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Jones LUT metadata {key!r} must be {expected!r}, "
                    f"got {metadata.get(key)!r}"
                )
        orientation = metadata.get("orientation")
        if not isinstance(orientation, str) or not orientation.strip():
            raise ValueError("Jones LUT metadata requires a non-empty orientation")
        for name, use_key in (
            ("reciprocity", "used_to_transform"),
            ("symmetry", "used_to_expand"),
        ):
            audit = metadata.get(name)
            if not isinstance(audit, dict):
                raise ValueError(f"Jones LUT metadata requires a {name!r} object")
            if audit.get("status") not in cls._AUDIT_STATUSES:
                raise ValueError(
                    f"Jones LUT metadata {name}.status must be one of "
                    f"{sorted(cls._AUDIT_STATUSES)}"
                )
            if not isinstance(audit.get(use_key), bool):
                raise ValueError(f"Jones LUT metadata {name}.{use_key} must be boolean")

    @classmethod
    def final_exact_artifact_mode(cls, metadata: dict[str, object]) -> str:
        """Classify an artifact accepted by the final-exact runtime contract.

        The ordinary ``validated`` route retains the historical promotion
        semantics.  A second, deliberately narrow route records the square-cell
        C4 identity as part of the *declared physical model*, rather than
        relabelling unevaluated numerical-convergence audits as validated.  The
        latter is therefore model-conditional and is kept distinguishable in
        pins and downstream provenance.
        """

        status = metadata.get("artifact_status")
        if status == cls.VALIDATED_ARTIFACT_STATUS:
            return cls.VALIDATED_ARTIFACT_STATUS
        if status != cls.MODEL_CONDITIONAL_ARTIFACT_STATUS:
            raise ValueError(
                "final-exact Jones LUT artifact_status must be 'validated' or "
                f"{cls.MODEL_CONDITIONAL_ARTIFACT_STATUS!r}; candidate artifacts "
                "remain non-promotable"
            )

        basis = metadata.get("validation_basis")
        required_basis = {
            "schema": cls.MODEL_BASIS_SCHEMA,
            "scope": cls.MODEL_BASIS_SCOPE,
            "physical_basis": cls.MODEL_BASIS_PHYSICS,
            "direct_full_jones_rcwa_samples": True,
            "numerical_convergence_certified": False,
            "heldout_audit_gating": False,
        }
        if not isinstance(basis, dict) or any(
            basis.get(key) != value for key, value in required_basis.items()
        ):
            raise ValueError(
                "model-conditional Jones LUT lacks the exact declared C4 RCWA "
                "validation_basis contract"
            )

        reciprocity = metadata.get("reciprocity")
        if not isinstance(reciprocity, dict) or (
            reciprocity.get("used_to_transform") is not False
        ):
            raise ValueError(
                "model-conditional Jones LUT must use direct RCWA samples, not "
                "a reciprocity transform"
            )
        symmetry = metadata.get("symmetry")
        required_symmetry = {
            "status": cls.DECLARED_EXACT_SYMMETRY_STATUS,
            "used_to_expand": True,
            "kind": "c4_rotation_90deg_local_sp",
            "reflection_used": False,
            "heldout_azimuths_deg": [105.0, 195.0, 285.0],
        }
        if not isinstance(symmetry, dict) or any(
            symmetry.get(key) != value for key, value in required_symmetry.items()
        ):
            raise ValueError(
                "model-conditional Jones LUT must be an explicit rotation-only "
                "C4 expansion in the declared local-s/p basis"
            )

        production_audit = metadata.get("production_audit")
        incomplete = [
            key
            for key in cls.PRODUCTION_AUDIT_KEYS
            if not isinstance(production_audit, dict)
            or not isinstance(production_audit.get(key), dict)
            or production_audit[key].get("status")
            != cls.MODEL_CONDITIONAL_AUDIT_STATUS
        ]
        if incomplete:
            raise ValueError(
                "model-conditional Jones LUT must label every non-certified "
                "production audit as model_conditional: " + ", ".join(incomplete)
            )
        return cls.MODEL_CONDITIONAL_VALIDATION_MODE

    @classmethod
    def npz_schema(cls) -> dict[str, object]:
        """Return the generator-facing, JSON-serializable NPZ contract."""
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "required_keys": sorted(cls.NPZ_REQUIRED_KEYS),
            "jones_shape": "[Nwidth,Nlambda,Npolar,Nazimuth,2,2]",
            "jones_order": "[[t_ss,t_sp],[t_ps,t_pp]]",
            "basis": cls.BASIS,
            "matrix_order": cls.MATRIX_ORDER,
            "coefficient_convention": cls.COEFFICIENT_CONVENTION,
            "angle_convention": cls.ANGLE_CONVENTION,
            "azimuth_symmetry": "none_implicit",
            "metadata_required": {
                "orientation": cls.PRODUCTION_ORIENTATION,
                "artifact_status": [
                    cls.VALIDATED_ARTIFACT_STATUS,
                    cls.MODEL_CONDITIONAL_ARTIFACT_STATUS,
                ],
                "stack": {
                    "incident_medium": cls.PRODUCTION_INCIDENT_MEDIUM,
                    "object_medium": "semi_infinite_fused_silica",
                    "entrance_interface": "none_by_construction",
                    "substrate_thickness": "semi_infinite",
                    "patterned_layer": cls.PRODUCTION_PATTERNED_LAYER,
                    "output_medium": cls.PRODUCTION_OUTPUT_MEDIUM,
                },
                "solver": {
                    "direct_stack_solve": True,
                    "power_norm": True,
                    "phase_only": False,
                    "all_four_jones_terms": True,
                    "angle_medium": cls.PRODUCTION_ANGLE_MEDIUM,
                    "object_angle_refracted_by_snell": False,
                    "basis_conversion": cls.PRODUCTION_BASIS_CONVERSION,
                    "modal_e_conversion": cls.PRODUCTION_MODAL_E_CONVERSION,
                },
                "reciprocity": {
                    "status": sorted(cls._AUDIT_STATUSES),
                    "used_to_transform": "boolean",
                },
                "symmetry": {
                    "status": sorted(cls._AUDIT_STATUSES),
                    "used_to_expand": "boolean",
                },
                "production_audit": {
                    key: "status=validated" for key in cls.PRODUCTION_AUDIT_KEYS
                },
            },
            "final_exact_rule": (
                "fail_closed=true; direct semi-infinite fused-silica to SiN-"
                "posts-in-air to air stack; power-normalized full Jones; all "
                "production audits validated, or the explicitly labelled "
                "model-conditional square-cell C4 physical basis; any other "
                "reciprocity/symmetry construction is rejected"
            ),
        }

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        fail_closed: bool = True,
    ) -> "FullJonesLUTResponseModel":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            missing = cls.NPZ_REQUIRED_KEYS.difference(archive.files)
            if missing:
                raise ValueError(
                    f"Jones LUT NPZ is missing required keys: {sorted(missing)}"
                )
            raw_metadata = np.asarray(archive["metadata_json"])
            if raw_metadata.shape != ():
                raise ValueError("metadata_json must be a scalar JSON string")
            metadata_item = raw_metadata.item()
            if isinstance(metadata_item, bytes):
                metadata_item = metadata_item.decode("utf-8")
            if not isinstance(metadata_item, str):
                raise ValueError("metadata_json must decode to a string")
            try:
                metadata = json.loads(metadata_item)
            except json.JSONDecodeError as exc:
                raise ValueError("metadata_json is not valid JSON") from exc

            jones_real = np.asarray(archive["jones_real"], dtype=np.float32)
            jones_imag = np.asarray(archive["jones_imag"], dtype=np.float32)
            if jones_real.shape != jones_imag.shape:
                raise ValueError("jones_real and jones_imag must have identical shapes")
            jones = torch.from_numpy(jones_real + 1j * jones_imag)
            model = cls(
                width_grid_um=torch.from_numpy(
                    np.asarray(archive["widths_nm"], dtype=np.float32) / 1000.0
                ),
                wavelength_grid_um=torch.from_numpy(
                    np.asarray(archive["wavelengths_nm"], dtype=np.float32) / 1000.0
                ),
                polar_angle_grid_deg=torch.from_numpy(
                    np.asarray(archive["polar_angles_deg"], dtype=np.float32)
                ),
                azimuth_grid_deg=torch.from_numpy(
                    np.asarray(archive["azimuth_angles_deg"], dtype=np.float32)
                ),
                jones_grid=jones,
                metadata=metadata,
                fail_closed=fail_closed,
            )
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        # Runtime reflection-fold authorization must bind the in-memory model
        # to the exact promoted NPZ.  These immutable provenance attributes
        # survive ``nn.Module.to`` because it mutates and returns the module.
        model.source_npz_path = str(source.resolve())
        model.source_npz_sha256 = digest.hexdigest()
        return model

    def _axis_weights(
        self,
        values: torch.Tensor,
        grid: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The tolerance admits only floating-point representation noise at an
        # endpoint. It is not an extrapolation allowance.
        tol = 8.0 * torch.finfo(values.dtype).eps * torch.maximum(
            grid.abs().max(), torch.ones((), device=grid.device, dtype=grid.dtype)
        )
        below = values < grid[0] - tol
        above = values > grid[-1] + tol
        if self.fail_closed and bool((below | above).any()):
            low = float(values.min().detach())
            high = float(values.max().detach())
            raise ValueError(
                f"{name} query range [{low:.7g}, {high:.7g}] is outside LUT "
                f"range [{float(grid[0]):.7g}, {float(grid[-1]):.7g}]"
            )
        bounded = values.clamp(grid[0], grid[-1])
        hi = torch.bucketize(bounded.contiguous(), grid).clamp(1, grid.numel() - 1)
        lo = hi - 1
        fraction = (bounded - grid[lo]) / (grid[hi] - grid[lo])
        return lo, hi, fraction

    def _gather_quadrilinear(
        self,
        width: torch.Tensor,
        wavelength: torch.Tensor,
        polar: torch.Tensor,
        azimuth: torch.Tensor,
    ) -> torch.Tensor:
        iw = self._axis_weights(width, self.width_grid_um, "width")
        il = self._axis_weights(wavelength, self.wavelength_grid_um, "wavelength")
        it = self._axis_weights(polar, self.polar_angle_grid_deg, "polar angle")
        ia = self._axis_weights(azimuth, self.azimuth_grid_deg, "azimuth")
        _, nl, nt, na, _, _ = self.jones_grid.shape
        flat = self.jones_grid.reshape(-1, 2, 2)
        result = torch.zeros(
            (width.numel(), 2, 2),
            dtype=self.jones_grid.dtype,
            device=width.device,
        )
        axes = (iw, il, it, ia)
        for bw in (0, 1):
            for bl in (0, 1):
                for bt in (0, 1):
                    for ba in (0, 1):
                        indices = (
                            ((axes[0][bw] * nl + axes[1][bl]) * nt + axes[2][bt])
                            * na
                            + axes[3][ba]
                        )
                        weights = (
                            (axes[0][2] if bw else 1.0 - axes[0][2])
                            * (axes[1][2] if bl else 1.0 - axes[1][2])
                            * (axes[2][2] if bt else 1.0 - axes[2][2])
                            * (axes[3][2] if ba else 1.0 - axes[3][2])
                        )
                        result = result + weights[:, None, None] * flat[indices]
        return result

    def incident_refractive_index(
        self, wavelengths_um: torch.Tensor
    ) -> torch.Tensor:
        """Malitson fused-silica index for the declared input half-space.

        Final-exact camera B places the object/source in a semi-infinite fused-
        silica medium.  The object-to-pupil spherical phase must therefore use
        ``n(lambda) k0``; applying this LUT to the legacy vacuum incident field
        would mix two different physical stacks.
        """

        wavelength = torch.as_tensor(wavelengths_um)
        squared = wavelength.square()
        return torch.sqrt(
            1.0
            + 0.6961663 * squared / (squared - 0.0684043**2)
            + 0.4079426 * squared / (squared - 0.1162414**2)
            + 0.8974794 * squared / (squared - 9.896161**2)
        )

    def output_refractive_index(
        self, wavelengths_um: torch.Tensor
    ) -> torch.Tensor:
        """Index of the declared air output half-space."""

        return torch.ones_like(torch.as_tensor(wavelengths_um))

    def validate_final_exact_protocol(self) -> None:
        if not self.fail_closed:
            raise RuntimeError("final-exact Jones interpolation must be fail-closed")
        metadata = self.lut_metadata
        if metadata.get("orientation") != self.PRODUCTION_ORIENTATION:
            raise RuntimeError(
                "final-exact Jones LUT orientation must be direct "
                f"{self.PRODUCTION_ORIENTATION!r}, got "
                f"{metadata.get('orientation')!r}"
            )
        try:
            validation_mode = self.final_exact_artifact_mode(metadata)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        model_conditional = (
            validation_mode == self.MODEL_CONDITIONAL_VALIDATION_MODE
        )
        stack = metadata.get("stack")
        expected_stack = {
            "incident_medium": self.PRODUCTION_INCIDENT_MEDIUM,
            "object_medium": "semi_infinite_fused_silica",
            "entrance_interface": "none_by_construction",
            "substrate_thickness": "semi_infinite",
            "output_medium": self.PRODUCTION_OUTPUT_MEDIUM,
        }
        # The pillar dielectric is the one admissible material degree of freedom
        # across otherwise identical stacks, so patterned_layer is checked against
        # the accept-list rather than a single label; every other stack, solver,
        # symmetry and audit check below is unchanged.
        if (
            not isinstance(stack, dict)
            or any(stack.get(key) != value for key, value in expected_stack.items())
            or stack.get("patterned_layer") not in self.ACCEPTED_PATTERNED_LAYERS
        ):
            raise RuntimeError(
                "final-exact Jones LUT stack must be fused-silica -> square "
                "dielectric posts in air -> air with a recognised pillar material"
            )
        solver = metadata.get("solver")
        required_solver = {
            "direct_stack_solve": True,
            "power_norm": True,
            "phase_only": False,
            "all_four_jones_terms": True,
            "angle_medium": self.PRODUCTION_ANGLE_MEDIUM,
            "object_angle_refracted_by_snell": False,
            "basis_conversion": self.PRODUCTION_BASIS_CONVERSION,
            "modal_e_conversion": self.PRODUCTION_MODAL_E_CONVERSION,
        }
        if not isinstance(solver, dict) or any(
            solver.get(key) != value for key, value in required_solver.items()
        ):
            raise RuntimeError(
                "final-exact Jones LUT solver metadata does not prove a "
                "direct, power-normalized, full-complex substrate-to-air solve"
            )
        for name, use_key in (
            ("reciprocity", "used_to_transform"),
            ("symmetry", "used_to_expand"),
        ):
            audit = metadata[name]
            expected_status = self.VALIDATED_ARTIFACT_STATUS
            if model_conditional and name == "symmetry":
                expected_status = self.DECLARED_EXACT_SYMMETRY_STATUS
            if audit[use_key] and audit["status"] != expected_status:
                raise RuntimeError(
                    f"final-exact Jones LUT used {name} to construct data but "
                    f"its metadata status is {audit['status']!r}, not "
                    f"{expected_status!r}"
                )
        symmetry = metadata["symmetry"]
        if symmetry["used_to_expand"]:
            if (
                symmetry.get("kind") != "c4_rotation_90deg_local_sp"
                or symmetry.get("reflection_used") is not False
                or symmetry.get("heldout_azimuths_deg") != [105.0, 195.0, 285.0]
            ):
                raise RuntimeError(
                    "final-exact symmetry expansion must use validated C4 "
                    "rotations only, with direct 105/195/285-degree holdouts; "
                    "reflection folding is forbidden"
                )
        production_audit = metadata.get("production_audit")
        if not isinstance(production_audit, dict):
            raise RuntimeError(
                "final-exact Jones LUT requires a production_audit object"
            )
        expected_audit_status = (
            self.MODEL_CONDITIONAL_AUDIT_STATUS
            if model_conditional
            else self.VALIDATED_ARTIFACT_STATUS
        )
        incomplete = [
            key
            for key in self.PRODUCTION_AUDIT_KEYS
            if not isinstance(production_audit.get(key), dict)
            or production_audit[key].get("status") != expected_audit_status
        ]
        if incomplete:
            raise RuntimeError(
                "final-exact Jones LUT has audits inconsistent with its "
                f"{validation_mode!r} validation mode: "
                + ", ".join(incomplete)
            )

        # Mechanical coverage is checked in addition to the external audit so
        # a copied validation label cannot promote a cropped table.
        coverage = (
            (self.width_grid_um, 0.100, 0.240, "width"),
            (self.wavelength_grid_um, 0.420, 0.670, "wavelength"),
            (self.polar_angle_grid_deg, 0.0, 24.0, "polar angle"),
            (self.azimuth_grid_deg, -180.0, 180.0, "azimuth"),
        )
        for axis, lower, upper, name in coverage:
            tolerance = 16.0 * torch.finfo(axis.dtype).eps * max(
                1.0, abs(lower), abs(upper)
            )
            if float(axis[0]) > lower + tolerance or float(axis[-1]) < upper - tolerance:
                raise RuntimeError(
                    f"final-exact Jones LUT {name} axis does not cover "
                    f"[{lower}, {upper}]"
                )

        # For this square-post platform a table with both cross-polarized
        # entries identically zero is the legacy diagonal/phase-only model in a
        # six-dimensional container, not a measured full-Jones response.
        cross_max = max(
            float(self.jones_grid[..., 0, 1].abs().max()),
            float(self.jones_grid[..., 1, 0].abs().max()),
        )
        if cross_max == 0.0:
            raise RuntimeError(
                "final-exact Jones LUT has identically zero cross-polarized "
                "terms and is indistinguishable from a diagonal legacy response"
            )

    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        if width_map_um.ndim != 2:
            raise ValueError("width_map_um must be a two-dimensional pupil map")
        slopes_x = torch.tan(theta_x_rad)
        slopes_y = torch.tan(theta_y_rad)
        polar_deg = torch.rad2deg(torch.atan(torch.hypot(slopes_x, slopes_y)))
        azimuth_deg = torch.rad2deg(torch.atan2(slopes_y, slopes_x))
        azimuth_deg = torch.where(
            polar_deg < 1e-7, torch.zeros_like(azimuth_deg), azimuth_deg
        )
        shape = torch.broadcast_shapes(
            polar_deg.shape,
            (1, wavelengths_um.numel(), 1, 1),
            (1, 1) + tuple(width_map_um.shape),
        )
        polar = polar_deg.expand(shape)
        azimuth = azimuth_deg.expand(shape)
        width = width_map_um.view(1, 1, *width_map_um.shape).expand(shape)
        wavelength = wavelengths_um.view(1, -1, 1, 1).expand(shape)
        values = self._gather_quadrilinear(
            width.reshape(-1),
            wavelength.reshape(-1),
            polar.reshape(-1),
            azimuth.reshape(-1),
        ).reshape(*shape, 2, 2)
        rank = len(shape)
        return values.permute(rank, rank + 1, *range(rank))


class HybridLUTResponseModel(BaseResponseModel):
    """RCWA lookup (normal incidence) x corrected-EMT oblique phase factor.

    S(W, lam, theta) = A_lut(W, lam) * exp(i * [phi_lut(W, lam)
                        + kappa(f) * Phi0_emt(W, lam) * sin^2(theta)])

    - phi_lut/A_lut: bilinear interpolation of a rigorous RCWA sweep
      (torcwa order-7 anti-aliased, normal incidence). Valid over the full
      design width range (50-270 nm) — including W > 200 nm where the EMT
      effective-medium form breaks (near-field pillar coupling).
    - oblique correction: the RCWA-validated linear-in-fill-factor kappa
      (kappa(f) = a + b*f, default the o7 refit (-0.562, 0.625)) applied to
      the EMT phase thickness Phi0 = 2*pi*h*n_eff(f, lam)/lam. RCWA angle
      sweeps showed this reproduces 0-30 deg phase to ~0.10 rad RMS.

    Differentiable in the width map (bilinear interp + smooth n_eff).
    Returns a 4D tensor (scalar engine path), TE/TM-degenerate treatment.
    """

    response_contract = "legacy_phase_only_diagonal_local_sp"

    def __init__(
        self,
        npz_path: str,
        h_um: float = 1.0,
        P_um: float = 0.29,
        kappa: tuple[float, float] = (-0.562, 0.625),
        polarization: str = "unpolarized",
        oblique_lut_path: str | None = None,
        amp_angle_npz: str | None = None,
    ) -> None:
        super().__init__()
        if polarization not in ("te", "unpolarized"):
            raise ValueError("HybridLUT supports 'te' (4D scalar path) or "
                             "'unpolarized' (5D vectorial path, TE==TM)")
        self.polarization = polarization
        import numpy as np
        z = np.load(npz_path)
        w = torch.from_numpy(np.asarray(z["widths_nm"], dtype=np.float32) / 1000.0)
        lam = torch.from_numpy(np.asarray(z["wavelengths_nm"], dtype=np.float32) / 1000.0)
        # unwrap along both axes so bilinear interpolation never crosses a 2pi seam
        ph = np.unwrap(np.unwrap(np.asarray(z["phase"], dtype=np.float64), axis=1), axis=0)
        self.register_buffer("width_grid_um", w)
        self.register_buffer("wavelength_grid_um", lam)
        self.register_buffer("phase_grid", torch.from_numpy(ph.astype(np.float32)))
        self.register_buffer("amp_grid", torch.from_numpy(
            np.asarray(z["t_abs"], dtype=np.float32)))
        self.h_um = h_um
        self.P_um = P_um
        self.kappa = (float(kappa[0]), float(kappa[1]))
        # EMT internals reused only for n_eff (Phi0 of the analytic oblique term)
        self._emt = EMTResponseModel(h_um=h_um, P_um=P_um, include_fp=False,
                                     fp_convention="corrected", kappa=0.0)
        # Optional: replace the analytic kappa(f)*Phi0*sin^2(theta) oblique term
        # with a direct RCWA angle-correction lookup dphi(W, lambda, theta) =
        # phi_rcwa(theta) - phi_rcwa(0) (unpolarized, incl. the physical layer
        # piston; from a conical order-7 s-p sweep, azi=0). Trilinear in
        # (W, lambda, theta), differentiable in W; theta comes from geometry.
        self.use_oblique_lut = oblique_lut_path is not None
        if self.use_oblique_lut:
            zo = np.load(oblique_lut_path)
            self.register_buffer("ob_width_grid_um", torch.from_numpy(
                np.asarray(zo["widths_nm"], dtype=np.float32) / 1000.0))
            self.register_buffer("ob_wl_grid_um", torch.from_numpy(
                np.asarray(zo["wavelengths_nm"], dtype=np.float32) / 1000.0))
            ang = np.asarray(zo["angles_deg"], dtype=np.float32)
            self.register_buffer("ob_angle_grid_deg", torch.from_numpy(ang))
            self.ob_angle_step = float(ang[1] - ang[0])   # uniform grid
            # dphi table [Nw, Nl, Ntheta]; a smooth continuous correction (not a
            # wrapped phase), so no 2pi unwrap needed.
            self.register_buffer("ob_dphi_grid", torch.from_numpy(
                np.asarray(zo["dphi"], dtype=np.float32)))

        # OPTIONAL, OFF BY DEFAULT: angle-dependent transmission AMPLITUDE.
        # The LUT amplitude above is the normal-incidence |t_00|, applied
        # unchanged at every field angle. Above the Rayleigh threshold of the
        # 290 nm lattice in silica (425.6 nm on axis, up to 499.4 nm at the
        # largest quadrature field) a +/-1 order becomes propagating and drains
        # the zeroth order increasingly with angle, which the normal-incidence
        # amplitude cannot follow. Enabling this multiplies the amplitude by
        #     sqrt( T0(W, lambda, theta, azi) / T0(W, lambda, 0, 0) )
        # from an order-resolved RCWA table.
        #
        # NO ANGULAR INTERPOLATION IS PERFORMED. The table is measured at
        # exactly the (theta, azimuth) pairs of the sealed 25-field quadrature;
        # each call's chief ray is matched to one of them and must agree to
        # within ANGLE_MATCH_TOL_DEG or a RuntimeError is raised. This keeps the
        # sensitivity check free of any interpolation choice in angle.
        # Width is interpolated on the table's 2.5 nm grid, matching the
        # resolution of the normal-incidence library itself.
        self.use_amp_angle = amp_angle_npz is not None
        if self.use_amp_angle:
            za = np.load(amp_angle_npz)
            self.register_buffer("aa_width_grid_um", torch.from_numpy(
                np.asarray(za["widths_nm"], dtype=np.float32) / 1000.0))
            self.register_buffer("aa_wl_grid_um", torch.from_numpy(
                np.asarray(za["wavelengths_nm"], dtype=np.float32) / 1000.0))
            cfg = np.asarray(za["configs"], dtype=np.float64)   # [Nc, 2]
            self.register_buffer("aa_config_deg", torch.from_numpy(
                cfg.astype(np.float32)))
            t0 = np.asarray(za["T0"], dtype=np.float64)         # [Nw, Nl, Nc]
            i0 = int(np.argmin(cfg[:, 0] ** 2 + cfg[:, 1] ** 2))
            if abs(cfg[i0, 0]) > 1e-9:
                raise ValueError("amp_angle table has no theta=0 reference row")
            # amplitude ratio, not power; clamp guards a divide by a numerically
            # dead zeroth order (none occur in the shipped table: min T0 = 0.02)
            ratio = np.sqrt(t0 / np.clip(t0[:, :, i0:i0 + 1], 1e-12, None))
            self.register_buffer("aa_ratio_grid", torch.from_numpy(
                ratio.astype(np.float32)))                      # [Nw, Nl, Nc]

    # A chief ray must land on a tabulated configuration this closely (degrees).
    ANGLE_MATCH_TOL_DEG = 0.01

    def _match_config(self, theta_x_rad, theta_y_rad):
        """Chief-ray (theta, azimuth) -> index into the measured config table.

        theta_x/theta_y are per-pupil-point. Across the 208 um aperture at the
        47697 um object distance the chief ray varies by only ~0.12 deg, so the
        pupil-centre value defines the field. Because tan(theta_x) is exactly
        linear in the pupil coordinate, its mean over a centred grid equals the
        centre value, which is how the chief ray is recovered here.

        NOTE the polar angle is atan(hypot(tan tx, tan ty)), NOT the engine's
        usual sqrt(tx^2 + ty^2); the two differ by up to 0.46 deg at azimuth 45
        and only the former is the angle an RCWA solve is parameterised by.
        """
        tx = torch.tan(theta_x_rad).mean(dim=(-2, -1)).reshape(-1)
        ty = torch.tan(theta_y_rad).mean(dim=(-2, -1)).reshape(-1)
        theta = torch.rad2deg(torch.atan(torch.hypot(tx, ty)))
        azi = torch.rad2deg(torch.atan2(ty, tx)) % 90.0
        azi = torch.minimum(azi, 90.0 - azi)              # C4v fundamental domain
        azi = torch.where(theta < 1e-6, torch.zeros_like(azi), azi)
        tab = self.aa_config_deg.to(theta.dtype)
        d = ((theta[:, None] - tab[None, :, 0]) ** 2
             + (azi[:, None] - tab[None, :, 1]) ** 2).sqrt()
        best, idx = d.min(dim=1)
        if bool((best > self.ANGLE_MATCH_TOL_DEG).any()):
            b = int(best.argmax())
            raise RuntimeError(
                "angle-dependent amplitude is enabled but the chief ray "
                f"(theta={float(theta[b]):.4f} deg, azi={float(azi[b]):.4f} deg) "
                f"is {float(best[b]):.4f} deg from the nearest tabulated "
                "configuration. The table is measured only at the sealed "
                "25-field quadrature; extend it rather than interpolating.")
        return idx

    def _interp_1d_weights(self, values, grid):
        values = values.clamp(grid[0], grid[-1])
        hi = torch.bucketize(values, grid).clamp(1, grid.numel() - 1)
        lo = hi - 1
        t = (values - grid[lo]) / (grid[hi] - grid[lo]).clamp_min(1e-12)
        return lo, hi, t

    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        theta_rad = polar_angle_from_axis_tilts(theta_x_rad, theta_y_rad)
        sin_sq = torch.sin(theta_rad).square()                    # [chunk,1,H,W]

        # theta-independent quantities (LUT interp + oblique prefactor) depend
        # only on (width map, wavelengths) -> cache per version, like the EMT
        # model. Scene forwards call this tens of thousands of times per map.
        try:
            w_ver = width_map_um._version
            l_ver = wavelengths_um._version
        except AttributeError:
            w_ver = l_ver = 0
        # NOTE: id() alone is unsafe — freed tensors let CPython reuse the
        # address, so a *different* width map can silently hit the cache
        # (observed 2026-06-11 in rank_candidates). Weakrefs pin identity:
        # if the cached tensor died, the ref returns None and we recompute.
        key = (w_ver, l_ver, str(width_map_um.device))
        cache = getattr(self, "_static_cache", None)
        wm_ref = getattr(self, "_static_wm_ref", None)
        wl_ref = getattr(self, "_static_wl_ref", None)
        same_tensors = (wm_ref is not None and wm_ref() is width_map_um
                        and wl_ref is not None and wl_ref() is wavelengths_um)
        if getattr(self, "_static_key", None) != key or cache is None \
                or not same_tensors:
            ctx = torch.enable_grad() if width_map_um.requires_grad else torch.no_grad()
            with ctx:
                Nl = wavelengths_um.numel()
                W = width_map_um
                wlo, whi, tw = self._interp_1d_weights(W.reshape(-1), self.width_grid_um)
                llo, lhi, tl = self._interp_1d_weights(
                    wavelengths_um.to(self.wavelength_grid_um.dtype), self.wavelength_grid_um)

                def gather2(table):
                    c00 = table[wlo][:, llo]
                    c01 = table[wlo][:, lhi]
                    c10 = table[whi][:, llo]
                    c11 = table[whi][:, lhi]
                    twc = tw.unsqueeze(1)
                    out = ((1 - twc) * ((1 - tl) * c00 + tl * c01)
                           + twc * ((1 - tl) * c10 + tl * c11))
                    return out.transpose(0, 1)

                H_, W_ = W.shape
                phi0_lut = gather2(self.phase_grid).reshape(1, Nl, H_, W_)
                amp_c = gather2(self.amp_grid).reshape(1, Nl, H_, W_).to(torch.complex64)
                kPhi0 = None
                dphi_Wl = None
                if self.use_oblique_lut:
                    # Interpolate the RCWA angle-correction table over (W, lambda),
                    # keeping the theta axis: dphi_Wl[Ntheta, Nl, H, W]. theta is
                    # applied per-call (it varies across the pupil).
                    wlo2, whi2, tw2 = self._interp_1d_weights(
                        W.reshape(-1), self.ob_width_grid_um)
                    llo2, lhi2, tl2 = self._interp_1d_weights(
                        wavelengths_um.to(self.ob_wl_grid_um.dtype), self.ob_wl_grid_um)
                    tab = self.ob_dphi_grid                       # [Nw, Nl, Na]
                    c00 = tab[wlo2][:, llo2]; c01 = tab[wlo2][:, lhi2]
                    c10 = tab[whi2][:, llo2]; c11 = tab[whi2][:, lhi2]
                    twc = tw2.view(-1, 1, 1); tlc = tl2.view(1, -1, 1)
                    ob = ((1 - twc) * ((1 - tlc) * c00 + tlc * c01)
                          + twc * ((1 - tlc) * c10 + tlc * c11))  # [Npix, Nl, Na]
                    Na = self.ob_angle_grid_deg.numel()
                    dphi_Wl = ob.reshape(H_, W_, Nl, Na).permute(3, 2, 0, 1).contiguous()
                else:
                    f = (W / self.P_um).clamp(1e-6, 1 - 1e-6)
                    lam64 = wavelengths_um.to(torch.float64).view(1, -1, 1, 1)
                    nc = self._emt._sellmeier_SiN(lam64)
                    n_eff = self._emt._compute_n_eff(
                        f.to(torch.float64).unsqueeze(0).unsqueeze(0), nc, lam64)
                    Phi0 = (2.0 * torch.pi * self.h_um / lam64) * n_eff
                    kmap = (self.kappa[0] + self.kappa[1] * f).unsqueeze(0).unsqueeze(0)
                    kPhi0 = (kmap.to(torch.float64) * Phi0).to(torch.float32)
                aa_w = None
                if self.use_amp_angle:
                    # width-interpolation weights on the amplitude table's own
                    # 2.5 nm grid, plus the exact wavelength row mapping
                    aw_lo, aw_hi, aw_t = self._interp_1d_weights(
                        W.reshape(-1), self.aa_width_grid_um)
                    lam_q = wavelengths_um.to(self.aa_wl_grid_um.dtype)
                    dl = (lam_q[:, None] - self.aa_wl_grid_um[None, :]).abs()
                    lam_off, lam_idx = dl.min(dim=1)
                    if bool((lam_off > 1e-6).any()):
                        raise RuntimeError(
                            "angle-dependent amplitude is enabled but the "
                            "engine wavelengths do not match the table rows "
                            f"(worst offset {float(lam_off.max())*1000:.3f} nm). "
                            "The table is tabulated at the design wavelengths "
                            "exactly; no spectral interpolation is applied.")
                    aa_w = (aw_lo, aw_hi, aw_t, lam_idx)
            cache = {"phi0_lut": phi0_lut, "amp_c": amp_c,
                     "kPhi0": kPhi0, "dphi_Wl": dphi_Wl, "aa_w": aa_w}
            self._static_cache = cache
            self._static_key = key
            import weakref
            self._static_wm_ref = weakref.ref(width_map_um)
            self._static_wl_ref = weakref.ref(wavelengths_um)
        if self.use_oblique_lut:
            # theta varies across the pupil -> interpolate the cached
            # dphi_Wl[Na, Nl, H, W] table over theta (deg) with linear
            # (triangular) weights on the uniform angle grid.
            theta_deg = torch.rad2deg(theta_rad)                    # [chunk,1,H,W]
            t = (theta_deg / self.ob_angle_step).clamp(
                0.0, float(self.ob_angle_grid_deg.numel() - 1))
            dphi_Wl = cache["dphi_Wl"]
            acc = 0.0
            for a in range(dphi_Wl.shape[0]):
                w_a = (1.0 - (a - t).abs()).clamp(min=0.0)          # [chunk,1,H,W]
                acc = acc + w_a * dphi_Wl[a].unsqueeze(0)           # [chunk,Nl,H,W]
            phi = cache["phi0_lut"] + acc
        else:
            phi = cache["phi0_lut"] + cache["kPhi0"] * sin_sq
        amp_c = cache["amp_c"]
        if self.use_amp_angle:
            aw_lo, aw_hi, aw_t, lam_idx = cache["aa_w"]
            H_, W_ = width_map_um.shape
            # [Nw, Nl, Nc] -> rows for the requested wavelengths only
            cfg = self._match_config(theta_x_rad, theta_y_rad)     # [chunk]
            # select wavelengths and this chunk's configurations while the
            # table is still small ([57, 9, 16]), then expand over pixels
            tab = self.aa_ratio_grid.index_select(1, lam_idx).index_select(2, cfg)
            twc = aw_t.view(-1, 1, 1)
            r = ((1.0 - twc) * tab[aw_lo] + twc * tab[aw_hi])      # [Npix, Nl, chunk]
            r = r.permute(2, 1, 0).reshape(cfg.numel(), -1, H_, W_)
            amp_c = amp_c * r.to(amp_c.dtype)
        S = amp_c * torch.exp(1j * phi.to(torch.complex64))
        if self.polarization == "unpolarized":
            # normal-incidence LUT: TE/TM degenerate; oblique kappa factor is
            # the polarization-averaged RCWA fit -> identical channels.
            # P8: expand() view instead of stack() — avoids a full copy of the
            # response (1.7 GB per chunk at d500); consumers only index/multiply.
            return S.unsqueeze(0).expand(2, *S.shape)
        return S


class SurrogatePhaseResponseModel(BaseResponseModel):
    """
    Scalar phase-only surrogate derived from the reduced pillar phase model.

    The local response is modeled as
        S(W, lambda, theta) = exp(i * phi(W, lambda, theta))
    with
        phi(W, lambda, theta) = alpha(theta) * (a(W) / lambda + b(W)) + A(theta) + C(theta) * lambda

    Width inputs use micrometers at the engine boundary and are internally
    converted to nanometers because the fitted model was built in nanometers.
    """

    def __init__(
        self,
        a_w: torch.Tensor,
        b_w: torch.Tensor,
        alpha_th: torch.Tensor,
        A_th: torch.Tensor,
        C_th: torch.Tensor,
        widths_nm: torch.Tensor,
        angles_deg: torch.Tensor,
        wavelengths_nm: torch.Tensor,
        *,
        max_angle_deg: float | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("a_w", a_w.to(torch.float32))
        self.register_buffer("b_w", b_w.to(torch.float32))
        self.register_buffer("alpha_th", alpha_th.to(torch.float32))
        self.register_buffer("A_th", A_th.to(torch.float32))
        self.register_buffer("C_th", C_th.to(torch.float32))
        self.register_buffer("widths_nm", widths_nm.to(torch.float32))
        self.register_buffer("angles_deg", angles_deg.to(torch.float32))
        self.register_buffer("wavelengths_nm", wavelengths_nm.to(torch.float32))
        self.max_angle_deg = float(max_angle_deg) if max_angle_deg is not None else None

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        *,
        max_angle_deg: float | None = None,
    ) -> "SurrogatePhaseResponseModel":
        data = np.load(Path(path))
        return cls(
            a_w=torch.from_numpy(data["a_w"]),
            b_w=torch.from_numpy(data["b_w"]),
            alpha_th=torch.from_numpy(data["alpha_th"]),
            A_th=torch.from_numpy(data["A_th"]),
            C_th=torch.from_numpy(data["C_th"]),
            widths_nm=torch.from_numpy(data["widths"]),
            angles_deg=torch.from_numpy(data["angles"]),
            wavelengths_nm=torch.from_numpy(data["wl"]),
            max_angle_deg=max_angle_deg,
        )

    @staticmethod
    def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32).clamp(xp[0], xp[-1])
        hi = torch.bucketize(x, xp)
        hi = hi.clamp(1, xp.numel() - 1)
        lo = hi - 1
        x0 = xp[lo]
        x1 = xp[hi]
        t = (x - x0) / (x1 - x0).clamp_min(1e-12)
        return fp[lo] * (1.0 - t) + fp[hi] * t

    def phase_rad(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        theta_deg = torch.rad2deg(
            polar_angle_from_axis_tilts(theta_x_rad, theta_y_rad)
        )
        if self.max_angle_deg is not None:
            theta_deg = theta_deg.clamp(max=self.max_angle_deg)

        width_nm = 1000.0 * width_map_um.to(torch.float32)
        wavelength_nm = 1000.0 * wavelengths_um.to(torch.float32)

        full_width_nm = width_nm.view(1, 1, *width_nm.shape).expand_as(theta_deg)
        full_wavelength_nm = wavelength_nm.view(1, -1, 1, 1).expand_as(theta_deg)

        a_interp = self._interp1d(full_width_nm, self.widths_nm, self.a_w)
        b_interp = self._interp1d(full_width_nm, self.widths_nm, self.b_w)
        alpha_interp = self._interp1d(theta_deg, self.angles_deg, self.alpha_th)
        A_interp = self._interp1d(theta_deg, self.angles_deg, self.A_th)
        C_interp = self._interp1d(theta_deg, self.angles_deg, self.C_th)
        phi0 = a_interp / full_wavelength_nm + b_interp
        return alpha_interp * phi0 + A_interp + C_interp * full_wavelength_nm

    def reference_phase_curve(
        self,
        wavelength_um: float,
        angle_deg: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wavelength_nm = torch.full_like(self.widths_nm, float(wavelength_um) * 1000.0)
        angle_deg_t = torch.full_like(self.widths_nm, float(angle_deg))
        a_interp = self.a_w
        b_interp = self.b_w
        alpha_interp = self._interp1d(angle_deg_t, self.angles_deg, self.alpha_th)
        A_interp = self._interp1d(angle_deg_t, self.angles_deg, self.A_th)
        C_interp = self._interp1d(angle_deg_t, self.angles_deg, self.C_th)
        phi0 = a_interp / wavelength_nm + b_interp
        phase = alpha_interp * phi0 + A_interp + C_interp * wavelength_nm
        return self.widths_nm / 1000.0, phase

    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        phase = self.phase_rad(width_map_um, wavelengths_um, theta_x_rad, theta_y_rad)
        return torch.exp(1j * phase.to(torch.complex64))


class EMTResponseModel(BaseResponseModel):
    """
    Effective Medium Theory response model for dielectric nanopillar metalenses.

    Computes the complex transmission S(W, lambda, theta) from first principles
    using a Fabry-Perot slab model with second-order EMT for the effective index.

    Parameters
    ----------
    h_um : float
        Pillar height in micrometers.
    P_um : float
        Period (pitch) in micrometers.
    n_cl : float
        Cladding refractive index (1.0 for air).
    h_slab_um : float
        Substrate slab thickness in micrometers (0 to disable).
    kappa : float
        Oblique-incidence correction factor for the propagation phase.
    include_fp : bool
        If True, include Fabry-Perot reflections at top/bottom interfaces.
    """

    def __init__(
        self,
        h_um: float = 0.5,
        P_um: float = 0.29,
        n_cl: float = 1.0,
        h_slab_um: float = 0.0,
        kappa: float | tuple[float, float] | None = None,
        include_fp: bool = True,
        polarization: str = "unpolarized",
        rytov_order: int = 2,
        fp_convention: str = "legacy",
        o4_coeffs: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.h_um = h_um
        self.P_um = P_um
        self.n_cl = n_cl
        self.h_slab_um = h_slab_um
        # convention default: legacy keeps the historical 0.47; corrected uses
        # the RCWA-refit fill-factor-linear kappa(f) (sign-corrected). Explicit
        # kappa= always wins (opt-out).
        if kappa is None:
            kappa = (-0.562, 0.625) if fp_convention == "corrected" else 0.47
        self.kappa = kappa
        self.include_fp = include_fp
        # Rytov order-4 asymmetric-correction coefficients
        # (beta_hi, p_hi, q_hi, beta_lo, p_lo, q_lo). None -> the legacy calibrated
        # values hardcoded in _compute_n_eff (preserves v-series reproducibility).
        # Override (e.g. an o7/corrected refit) via this argument, not by editing
        # the default — see docs/ENGINE_CHANGELOG.md.
        self.o4_coeffs = tuple(o4_coeffs) if o4_coeffs is not None else None
        # Oblique-incidence kappa. Scalar (default) -> constant correction, exactly
        # as before. A 2-tuple/list (a, b) -> fill-factor-linear kappa(f) = a + b*f
        # (f = W/P), applied per-pixel in the corrected Track-B model. The scalar
        # path is untouched; only the linear branch is new (see ENGINE_CHANGELOG).
        if isinstance(kappa, (tuple, list)):
            if len(kappa) != 2:
                raise ValueError("kappa tuple must be (a, b) for kappa(f)=a+b*f, "
                                 f"got length {len(kappa)}")
            self.kappa_linear = (float(kappa[0]), float(kappa[1]))
        else:
            self.kappa_linear = None
        if polarization not in ("te", "tm", "unpolarized"):
            raise ValueError(f"polarization must be 'te', 'tm', or 'unpolarized', got '{polarization}'")
        self.polarization = polarization
        if rytov_order not in (2, 4):
            raise ValueError(f"rytov_order must be 2 or 4, got {rytov_order}")
        self.rytov_order = rytov_order
        # "legacy"    : reproduces all v-series results bit-for-bit (FP denominator
        #               sign flipped vs exact slab TMM; |S| ~ power transmittance)
        # "corrected" : exact Airy slab — denominator 1 + rho_t*rho_b*e^{2iPhi} for
        #               this code's rho sign conventions (rho_t external, rho_b
        #               internal) and field-amplitude numerator sqrt(tau_t*tau_b)
        #               so that |S|^2 = power transmittance; verified against
        #               transfer-matrix slab to 4e-16 rad. Bottom-interface Snell
        #               chain also uses conserved transverse momentum.
        if fp_convention not in ("legacy", "corrected"):
            raise ValueError(f"fp_convention must be 'legacy' or 'corrected', got '{fp_convention}'")
        self.fp_convention = fp_convention
        # Lazy precompute cache for theta-independent quantities.
        # Keyed on (width_map_um identity+version, wavelengths_um identity+version).
        # Invalidated automatically when either input is mutated or replaced.
        self._cache_key: tuple | None = None
        self._cache: dict | None = None

    def clear_cache(self) -> None:
        """Force the next forward() call to rebuild the precompute cache."""
        self._cache_key = None
        self._cache = None

    def _build_static_cache(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
    ) -> dict:
        """Precompute all theta-independent quantities used by _compute_S_polarized.

        Inference path (width_map_um.requires_grad=False): build under
        torch.no_grad() so cached tensors are leaves and do not retain an
        autograd graph across chunks → lower GPU memory and lower per-chunk
        PyTorch overhead.

        Optimization path (width_map_um.requires_grad=True): build with grad
        tracking so the cache participates in the autograd graph. The cache is
        invalidated whenever width_map_um._version changes (i.e. on every
        optimizer.step() in-place update), so each outer step builds fresh
        grad-tracked tensors and reuses them across all chunks within the step.
        """
        # Decide grad mode based on whether width_map needs gradients.
        ctx = torch.enable_grad() if width_map_um.requires_grad else torch.no_grad()
        with ctx:
            W = width_map_um.to(torch.float64).unsqueeze(0).unsqueeze(0)
            lam = wavelengths_um.to(torch.float64).view(1, -1, 1, 1)
            f = (W / self.P_um).clamp(1e-6, 1.0 - 1e-6)
            nc = self._sellmeier_SiN(lam)
            n_s = self._sellmeier_SiO2(lam)
            n_eff = self._compute_n_eff(f, nc, lam)
            # Phase prefactor (multiplied by (1 + kappa·sin²θ) per chunk).
            Phi0 = (2.0 * torch.pi * self.h_um / lam) * n_eff
            # Pre-squared ratios for cos_t terms (n_s/n_eff)² and (n_eff/n_cl)².
            ratio_sub = (n_s / n_eff).square()
            if self.n_cl > 0:
                ratio_top_air = (n_eff / self.n_cl).square()
            else:
                ratio_top_air = None
            # Substrate slab: precompute n_s² and prefactor (depends on θ via sin²θ at runtime).
            if self.h_slab_um > 0.0:
                n_s_sq = n_s * n_s
                sub_coef = (2.0 * torch.pi * self.h_slab_um / lam)
            else:
                n_s_sq = None
                sub_coef = None
            # Fill-factor-linear kappa(f) = a + b*f map [1,1,1,Nw] (only when enabled);
            # theta-independent so it lives in the cache like Phi0.
            if self.kappa_linear is not None:
                a_k, b_k = self.kappa_linear
                kappa_map = a_k + b_k * f
            else:
                kappa_map = None
        return {
            "n_eff": n_eff,
            "n_s": n_s,
            "lam": lam,
            "Phi0": Phi0,
            "ratio_sub": ratio_sub,
            "ratio_top_air": ratio_top_air,
            "n_s_sq": n_s_sq,
            "sub_coef": sub_coef,
            "kappa_map": kappa_map,
        }

    def _get_cache(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
    ) -> dict:
        """Return the precompute cache, rebuilding if inputs changed."""
        try:
            w_ver = width_map_um._version
            l_ver = wavelengths_um._version
        except AttributeError:
            w_ver = 0
            l_ver = 0
        # weakref identity guard — see HybridLUTResponseModel.forward: bare
        # id() keys can collide after tensor reuse and serve stale responses.
        key = (
            w_ver, tuple(width_map_um.shape),
            l_ver, tuple(wavelengths_um.shape),
            str(width_map_um.device),
        )
        wm_ref = getattr(self, "_cache_wm_ref", None)
        wl_ref = getattr(self, "_cache_wl_ref", None)
        same_tensors = (wm_ref is not None and wm_ref() is width_map_um
                        and wl_ref is not None and wl_ref() is wavelengths_um)
        if self._cache_key != key or self._cache is None or not same_tensors:
            self._cache = self._build_static_cache(width_map_um, wavelengths_um)
            self._cache_key = key
            import weakref
            self._cache_wm_ref = weakref.ref(width_map_um)
            self._cache_wl_ref = weakref.ref(wavelengths_um)
        return self._cache

    @staticmethod
    def _sellmeier_SiN(lam_um: torch.Tensor) -> torch.Tensor:
        """Si3N4 refractive index from Sellmeier equation. Returns n (not n^2)."""
        lam2 = lam_um * lam_um
        n_sq = (
            1.0
            + 3.0249 * lam2 / (lam2 - 0.01832)
            + 40314.0 * lam2 / (lam2 - 1537462.0)
        )
        return torch.sqrt(n_sq)

    @staticmethod
    def _sellmeier_SiO2(lam_um: torch.Tensor) -> torch.Tensor:
        """SiO2 refractive index from Sellmeier equation. Returns n (not n^2)."""
        lam2 = lam_um * lam_um
        n_sq = (
            1.0
            + 0.6962 * lam2 / (lam2 - 0.00468)
            + 0.4079 * lam2 / (lam2 - 0.01351)
            + 0.8975 * lam2 / (lam2 - 97.934)
        )
        return torch.sqrt(n_sq)

    @classmethod
    def default_SiN_on_SiO2(cls, polarization: str = "unpolarized") -> "EMTResponseModel":
        """Factory for a standard SiN-on-SiO2 metalens configuration."""
        return cls(
            h_um=0.5,
            P_um=0.29,
            n_cl=1.0,
            h_slab_um=0.0,
            kappa=0.47,
            include_fp=True,
            polarization=polarization,
        )

    def _compute_n_eff(
        self,
        f: torch.Tensor,
        nc: torch.Tensor,
        lam_um: torch.Tensor,
    ) -> torch.Tensor:
        """Rytov EMT effective index for 2D C4-symmetric square pillars.

        Order 2 (default, Lalanne 1996 Eq. 9, TE form valid for 2D C4 by symmetry):
            n_eff^2 = <eps> + (pi*P/lam)^2 * f^2(1-f)^2 * (n_c^2-1)^2 / 3

        Order 4: extends with a *waveguide-confinement asymmetric correction*
        that biases n_eff toward the bulk material at high fill factor and
        toward the cladding at low fill factor -- physically capturing the
        thick-pillar Bloch-mode behavior that pure Rytov 2nd-order misses.
        The correction has the form
            delta_4(f, lam) = Q * (n_c^2 - 1) * [
                beta_hi * f^p_hi * (1-f)^q_hi
              - beta_lo * f^p_lo * (1-f)^q_lo ],
        where Q = (pi*P/lam)^2 and the six coefficients are calibrated
        against rigorous RCWA reference data for SiN/SiO2 square pillars
        at P=290 nm, h=1 um. Calibration values:
            beta_hi=3.378, p_hi=3.784, q_hi=2.417
            beta_lo=1.892, p_lo=1.774, q_lo=1.663
        These reduce the EMT-vs-RCWA residual from RMS=1.26 rad (order 2)
        to RMS=0.39 rad (order 4) in the operating zone W<=200, lam>=420 nm.
        """
        nc2 = nc * nc
        eps_par = f * nc2 + (1.0 - f)
        delta_eps_sq = (nc2 - 1.0) ** 2
        Q = (torch.pi * self.P_um / lam_um) ** 2  # (P*pi/lam)^2
        # 2nd-order Rytov (Lalanne 1996, Eq. 9)
        n_eff_sq = (
            eps_par
            + Q * f ** 2 * (1.0 - f) ** 2 * delta_eps_sq / 3.0
        )
        if self.rytov_order >= 4:
            # Asymmetric waveguide-confinement correction (calibrated)
            if self.o4_coeffs is not None:
                beta_hi, p_hi, q_hi, beta_lo, p_lo, q_lo = self.o4_coeffs
            else:
                beta_hi, p_hi, q_hi = 3.378, 3.784, 2.417
                beta_lo, p_lo, q_lo = 1.892, 1.774, 1.663
            asym = (
                beta_hi * (f ** p_hi) * ((1.0 - f) ** q_hi)
                - beta_lo * (f ** p_lo) * ((1.0 - f) ** q_lo)
            )
            n_eff_sq = n_eff_sq + Q * (nc2 - 1.0) * asym
        return torch.sqrt(n_eff_sq.clamp_min(1e-12))

    def _compute_S_polarized(
        self,
        n_eff: torch.Tensor,
        n_s: torch.Tensor,
        lam: torch.Tensor,
        sin_theta: torch.Tensor,
        pol: str,
        *,
        Phi0: torch.Tensor | None = None,
        ratio_sub: torch.Tensor | None = None,
        ratio_top_air: torch.Tensor | None = None,
        exp_iPhi: torch.Tensor | None = None,
        exp_2iPhi: torch.Tensor | None = None,
        cos_theta: torch.Tensor | None = None,
        sin_theta_sq: torch.Tensor | None = None,
        n_s_sq: torch.Tensor | None = None,
        sub_coef: torch.Tensor | None = None,
        sub_phase: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute complex transmission S with polarization-dependent Fresnel.

        At normal incidence (theta=0), TE and TM give identical results.
        At oblique incidence, the Fresnel reflection coefficients differ:
          TE (s-pol): rho = (n1 cos_i - n2 cos_t) / (n1 cos_i + n2 cos_t)
          TM (p-pol): rho = (n2 cos_i - n1 cos_t) / (n2 cos_i + n1 cos_t)

        Optional precomputed prefactors (Phi0, ratio_sub, ratio_top_air,
        exp_iPhi, exp_2iPhi, cos_theta, sin_theta_sq, n_s_sq, sub_coef,
        sub_phase) can be supplied from the EMTResponseModel cache to avoid
        recomputation when called twice (once for TE, once for TM) within the
        same chunk. When omitted, the function falls back to recomputing them
        locally (preserves the original numerical recipe bit-for-bit).
        """
        if sin_theta_sq is None:
            sin_theta_sq = sin_theta * sin_theta
        if cos_theta is None:
            cos_theta = torch.sqrt((1.0 - sin_theta_sq).clamp_min(1e-12))
        if Phi0 is None:
            Phi0 = (2.0 * torch.pi * self.h_um / lam) * n_eff
        # Phi is only needed if the caller did not precompute exp(iPhi)/exp(2iPhi)
        # (forward() always passes them). Compute lazily so the linear-kappa path
        # — where self.kappa is a tuple and the f-dependent factor lives in the
        # cache, not here — never multiplies a tuple by a tensor. Scalar standalone
        # callers get the identical Phi as before.
        def _phi():
            return Phi0 * (1.0 + self.kappa * sin_theta_sq)

        if self.include_fp:
            if getattr(self, "fp_convention", "legacy") == "corrected":
                # Exact flux-normalized Airy slab. theta = AIR-side angle
                # (matches pipeline field angles and torcwa inc_ang); conserved
                # transverse momentum kt = sin(theta) with n_air = 1.
                cos_sub = torch.sqrt((1.0 - sin_theta_sq / (n_s * n_s)).clamp_min(1e-12))
                cos_pil = torch.sqrt((1.0 - sin_theta_sq / (n_eff * n_eff)).clamp_min(1e-12))
                cos_air = cos_theta
                if pol == "te":
                    rho_t = (n_s * cos_sub - n_eff * cos_pil) / (n_s * cos_sub + n_eff * cos_pil)
                    rho_b = (n_eff * cos_pil - self.n_cl * cos_air) / (n_eff * cos_pil + self.n_cl * cos_air)
                else:
                    rho_t = (n_eff * cos_sub - n_s * cos_pil) / (n_eff * cos_sub + n_s * cos_pil)
                    rho_b = (self.n_cl * cos_pil - n_eff * cos_air) / (self.n_cl * cos_pil + n_eff * cos_air)
                # flux-normalized amplitude: |S|^2 = power transmittance exactly;
                # denominator sign for (rho_t external, rho_b internal) conventions
                tau = torch.sqrt(((1.0 - rho_t * rho_t) * (1.0 - rho_b * rho_b)).clamp_min(0.0))
                if exp_iPhi is None:
                    exp_iPhi = torch.exp(1j * _phi())
                if exp_2iPhi is None:
                    exp_2iPhi = torch.exp(2j * _phi())
                S = (tau * exp_iPhi) / (1.0 + rho_t * rho_b * exp_2iPhi)
            else:
                if ratio_sub is None:
                    ratio_sub = (n_s / n_eff).square()
                cos_t_top = torch.sqrt((1.0 - ratio_sub * sin_theta_sq).clamp_min(1e-12))
                if self.n_cl > 0:
                    if ratio_top_air is None:
                        ratio_top_air = (n_eff / self.n_cl).square()
                    cos_t_bot = torch.sqrt((1.0 - ratio_top_air * sin_theta_sq).clamp_min(1e-12))
                else:
                    cos_t_bot = cos_theta

                if pol == "te":
                    # s-polarization Fresnel
                    rho_t = (n_s * cos_theta - n_eff * cos_t_top) / (n_s * cos_theta + n_eff * cos_t_top)
                    rho_b = (n_eff * cos_t_top - self.n_cl * cos_t_bot) / (n_eff * cos_t_top + self.n_cl * cos_t_bot)
                else:
                    # p-polarization Fresnel
                    rho_t = (n_eff * cos_theta - n_s * cos_t_top) / (n_eff * cos_theta + n_s * cos_t_top)
                    rho_b = (self.n_cl * cos_t_top - n_eff * cos_t_bot) / (self.n_cl * cos_t_top + n_eff * cos_t_bot)

                tau_t = 1.0 - rho_t * rho_t
                tau_b = 1.0 - rho_b * rho_b
                if exp_iPhi is None:
                    exp_iPhi = torch.exp(1j * _phi())
                if exp_2iPhi is None:
                    exp_2iPhi = torch.exp(2j * _phi())
                S = (tau_t * tau_b * exp_iPhi) / (
                    1.0 - rho_t * rho_b * exp_2iPhi
                )
        else:
            if exp_iPhi is None:
                exp_iPhi = torch.exp(1j * _phi())
            S = exp_iPhi

        if self.h_slab_um > 0.0:
            if sub_phase is None:
                if sub_coef is None:
                    sub_coef = (2.0 * torch.pi * self.h_slab_um / lam)
                if n_s_sq is None:
                    n_s_sq = n_s * n_s
                Phi_sub = sub_coef * torch.sqrt(
                    (n_s_sq - sin_theta_sq).clamp_min(1e-12)
                )
                sub_phase = torch.exp(1j * Phi_sub)
            S = S * sub_phase

        return S

    def forward(
        self,
        width_map_um: torch.Tensor,
        wavelengths_um: torch.Tensor,
        theta_x_rad: torch.Tensor,
        theta_y_rad: torch.Tensor,
    ) -> torch.Tensor:
        # Precompute cache: theta-independent quantities (n_eff, Sellmeier,
        # phase prefactor Phi0, ratio_sub, ratio_top_air, sub_coef, n_s_sq)
        # are computed once per (width_map, wavelengths) version and reused
        # across all chunks. Within one chunk, Phi/exp(iPhi)/exp(2iPhi) are
        # also computed once and shared between TE and TM.
        cache = self._get_cache(width_map_um, wavelengths_um)
        n_eff = cache["n_eff"]
        n_s = cache["n_s"]
        lam = cache["lam"]
        Phi0 = cache["Phi0"]
        ratio_sub = cache["ratio_sub"]
        ratio_top_air = cache["ratio_top_air"]
        n_s_sq = cache["n_s_sq"]
        sub_coef = cache["sub_coef"]

        tx = theta_x_rad.to(torch.float64)
        ty = theta_y_rad.to(torch.float64)
        theta = torch.sqrt(tx * tx + ty * ty)
        sin_theta = torch.sin(theta)
        sin_theta_sq = sin_theta * sin_theta
        cos_theta = torch.sqrt((1.0 - sin_theta_sq).clamp_min(1e-12))

        # Phi and its complex exponentials are polarization-independent,
        # so compute them once and pass to both TE and TM branches.
        if self.kappa_linear is not None:
            Phi = Phi0 * (1.0 + cache["kappa_map"] * sin_theta_sq)
        else:
            Phi = Phi0 * (1.0 + self.kappa * sin_theta_sq)
        if self.include_fp:
            exp_iPhi = torch.exp(1j * Phi)
            exp_2iPhi = exp_iPhi * exp_iPhi
        else:
            exp_iPhi = torch.exp(1j * Phi)
            exp_2iPhi = None

        # Substrate-slab phase (if any) is also polarization-independent
        # — compute once and pass through.
        if self.h_slab_um > 0.0:
            Phi_sub = sub_coef * torch.sqrt((n_s_sq - sin_theta_sq).clamp_min(1e-12))
            sub_phase = torch.exp(1j * Phi_sub)
        else:
            sub_phase = None

        kw = dict(
            Phi0=Phi0,
            ratio_sub=ratio_sub,
            ratio_top_air=ratio_top_air,
            exp_iPhi=exp_iPhi,
            exp_2iPhi=exp_2iPhi,
            cos_theta=cos_theta,
            sin_theta_sq=sin_theta_sq,
            n_s_sq=n_s_sq,
            sub_coef=sub_coef,
            sub_phase=sub_phase,
        )

        if self.polarization in ("te", "tm"):
            S = self._compute_S_polarized(n_eff, n_s, lam, sin_theta, self.polarization, **kw)
            return S.to(torch.complex64)
        else:
            # unpolarized: return [2, ...] with TE at index 0, TM at index 1
            S_te = self._compute_S_polarized(n_eff, n_s, lam, sin_theta, "te", **kw)
            S_tm = self._compute_S_polarized(n_eff, n_s, lam, sin_theta, "tm", **kw)
            return torch.stack([S_te.to(torch.complex64), S_tm.to(torch.complex64)], dim=0)

    def reference_phase_curve(
        self,
        wavelength_um: float,
        angle_deg: float = 0.0,
        polarization: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute phase vs width at a single wavelength and angle.

        Parameters
        ----------
        polarization : str or None
            Override for phase curve computation. None uses 'te' (standard for
            width-map design). When self.polarization is 'unpolarized', the TE
            phase curve is used for design by default.

        Returns
        -------
        widths_um : Tensor [Nw]
            Pillar widths from 20 nm to (P - 20 nm) in 1 nm steps.
        phases_rad : Tensor [Nw]
            Corresponding transmission phases in radians.
        """
        w_min = 0.02  # 20 nm
        w_max = self.P_um - 0.02
        N = max(int(round((w_max - w_min) / 0.001)) + 1, 2)
        widths_um = torch.linspace(w_min, w_max, N, dtype=torch.float64)

        lam_t = torch.tensor([wavelength_um], dtype=torch.float64)
        theta_rad = torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float64))

        H_pup, W_pup = 1, N
        tx = theta_rad.expand(1, 1, H_pup, W_pup)
        ty = torch.zeros(1, 1, H_pup, W_pup, dtype=torch.float64)
        width_map = widths_um.unsqueeze(0)

        # Temporarily override polarization for design phase curve
        pol_save = self.polarization
        self.polarization = polarization or "te"
        S = self.forward(width_map, lam_t, tx, ty)  # [1, 1, 1, N]
        self.polarization = pol_save

        phases = torch.angle(S[0, 0, 0, :])
        return widths_um.float(), phases.float()


def build_response_model(config) -> BaseResponseModel:
    """Factory: create a response model from a ResponseModelConfig.

    Parameters
    ----------
    config : ResponseModelConfig
        Specifies model type and parameters.
    """
    from .config import ResponseModelType

    if config.model_type == ResponseModelType.EMT:
        return EMTResponseModel(
            h_um=config.h_um,
            P_um=config.P_um,
            n_cl=config.n_cl,
            h_slab_um=config.h_slab_um,
            kappa=config.kappa,
            include_fp=config.include_fp,
            polarization=config.polarization,
        )
    elif config.model_type == ResponseModelType.RCWA_LOOKUP:
        if config.rcwa_data_path is None:
            raise ValueError("rcwa_data_path is required for RCWA_LOOKUP model type.")
        return LookupResponseModel.from_rcwa_anglesweep(
            config.rcwa_data_path,
            wl_min_nm=config.rcwa_wl_min_nm,
            wl_max_nm=config.rcwa_wl_max_nm,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def project_reference_phase_to_width(
    target_phase_rad: torch.Tensor,
    widths_um: torch.Tensor,
    reference_phase_rad: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete circular projection from a reference target phase map to widths.

    Parameters
    ----------
    target_phase_rad:
        Target phase map [...], typically on the pupil grid.
    widths_um:
        Width samples [Nw].
    reference_phase_rad:
        Reference phase curve [Nw] evaluated at one wavelength/angle condition.
    """
    target_wrapped = torch.remainder(target_phase_rad, 2.0 * torch.pi)
    phase_wrapped = torch.remainder(reference_phase_rad, 2.0 * torch.pi)
    diff = phase_wrapped.view(-1, 1, 1) - target_wrapped.unsqueeze(0)
    circ_err = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
    best_idx = circ_err.argmin(dim=0)
    return widths_um[best_idx]


def soft_project_reference_phase_to_width(
    target_phase_rad: torch.Tensor,
    widths_um: torch.Tensor,
    reference_phase_rad: torch.Tensor,
    *,
    temperature_rad: float = 0.20,
) -> torch.Tensor:
    """
    Soft circular projection from target phase to widths.

    This is intended for phase-space optimisation: the forward model is still
    evaluated with physical widths, but gradients flow through a smooth
    soft-assignment over the reference phase curve instead of a hard nearest
    neighbour lookup.
    """
    target_wrapped = torch.remainder(target_phase_rad, 2.0 * torch.pi)
    phase_wrapped = torch.remainder(reference_phase_rad, 2.0 * torch.pi)
    diff = phase_wrapped.view(-1, 1, 1) - target_wrapped.unsqueeze(0)
    circ_err = torch.abs(torch.atan2(torch.sin(diff), torch.cos(diff)))
    logits = -(circ_err.square()) / max(float(temperature_rad) ** 2, 1e-8)
    weights = torch.softmax(logits, dim=0)
    return (weights * widths_um.view(-1, 1, 1)).sum(dim=0)
