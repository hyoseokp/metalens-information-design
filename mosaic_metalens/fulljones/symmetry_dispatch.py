"""Runtime authorization for production full-Jones reflection folds.

This module does not generate or score evidence.  It accepts only an immutable
dispatch manifest emitted by the independent production symmetry gate, binds
that decision to the loaded LUT, the exact width tensor, current forward source
hashes, and the camera contracts, and returns an opaque authorization token.
The full-Jones public forward refuses D2/D4 acceleration without that token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


DISPATCH_MANIFEST_SCHEMA = "fable_full_jones_reflection_fold_dispatch_v1"
REPO = Path(__file__).resolve().parent.parent
EVIDENCE_SCHEMA = "fable_full_jones_reflection_fold_evidence_v1"
EXPECTED_CONTRACTS = {
    "camera_forward_contract": "fable_final_exact_forward_v1",
    "response_artifact_contract": "fable_full_jones_oblique_response_v1",
    "response_model_contract": "full_complex_jones_local_sp_v1",
    "pixel_stack_spatial_contract": "planar_bsi_unit_fill_v1",
    "terminal_detector_contract": "lossless_terminal_interface_flux_proxy_v1",
    "field_coordinate_contract": (
        "field_local_common_chief_ray_540nm_conserved_k_parallel_snell_v1"
    ),
}
EXPECTED_THRESHOLDS = {
    "width_symmetry_max_abs_um": 1.0e-7,
    "lut_relative_l2_max": 2.0e-3,
    "lut_peak_normalized_max_abs": 5.0e-3,
    "psf_charge_relative_max": 2.0e-5,
    "psf_normalized_l1_max": 2.0e-4,
    "psf_normalized_linf_max": 2.0e-5,
}
DESIGNS = ("hyp600", "information_design", "matched_mtf")


class ReflectionFoldAuthorizationError(RuntimeError):
    """A runtime fold request is not bound to passing immutable evidence."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_python_tree(directory: str | Path) -> str:
    """Hash every Python source in one package with path delimiters.

    Individual hashes provide useful failure messages, while this aggregate
    closes accidental omissions when a lower-level forward helper gains a new
    dependency before the production source freeze.
    """

    root = Path(directory).resolve(strict=True)
    files = sorted(
        (path for path in root.rglob("*.py") if "__pycache__" not in path.parts),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ReflectionFoldAuthorizationError(
            f"reflection-fold Python source tree is empty: {root}"
        )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Match the production checkpoint's canonical tensor hash exactly."""

    value = torch.as_tensor(tensor).detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def current_source_hashes() -> dict[str, str]:
    engine_dir = Path(__file__).resolve().parent
    repo = engine_dir.parent
    paths = {
        "pipeline_forward_sha256": engine_dir / "pipeline_forward.py",
        "symmetry_dispatch_sha256": engine_dir / "symmetry_dispatch.py",
        "metalens_response_sha256": engine_dir / "metalens_response.py",
        "object_to_pupil_sha256": engine_dir / "object_to_pupil.py",
        "propagation_sha256": engine_dir / "propagation.py",
        "grids_sha256": engine_dir / "grids.py",
        "pixel_stack_sha256": engine_dir / "pixel_stack.py",
        "sensor_stack_sha256": engine_dir / "sensor_stack.py",
        "detector_surface_flux_sha256": engine_dir / "detector_surface_flux.py",
        "image_formation_sha256": engine_dir / "image_formation.py",
        "producer_sha256": (
            repo / "experiments/fable_aopt/produce_full_jones_symmetry_evidence.py"
        ),
        "exact_landscape_raw_producer_sha256": (
            repo / "experiments/fable_aopt/produce_exact_landscape_raw_bundle.py"
        ),
        "fable_optimize_dinfo_sha256": (
            repo / "experiments/fable_aopt/fable_optimize_dinfo.py"
        ),
        "optimize_v27_sha256": (
            repo / "experiments/phase3_v27/optimize_v27.py"
        ),
        "gate_code_sha256": (
            repo / "experiments/fable_aopt/production_symmetry_gates.py"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ReflectionFoldAuthorizationError(
            f"reflection-fold source inventory is incomplete: {missing}"
        )
    result = {name: sha256_file(path) for name, path in paths.items()}
    result["engine2_python_tree_sha256"] = sha256_python_tree(engine_dir)
    return result


@dataclass(frozen=True)
class ReflectionFoldAuthorization:
    """Opaque content-bound permission for one LUT and one width map."""

    manifest_path: str
    manifest_sha256: str
    manifest_content_sha256: str
    evidence_path: str
    evidence_sha256: str
    lut_sha256: str
    width_sha256: str
    design: str
    four_image_allowed: bool
    eight_image_allowed: bool
    contracts_sha256: str
    source_hashes_sha256: str


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve(strict=True)
    if not source.is_file() or source.suffix.lower() != ".json":
        raise ReflectionFoldAuthorizationError(
            f"reflection-fold dispatch manifest is not JSON: {source}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold dispatch manifest is unreadable"
        ) from exc
    if not isinstance(payload, dict):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold dispatch manifest must be a JSON object"
        )
    return source, payload


def _engine_contracts(engine: Any) -> dict[str, str]:
    response_contract = str(getattr(engine.response_model, "response_contract", ""))
    contracts = {
        "camera_forward_contract": "fable_final_exact_forward_v1",
        "response_artifact_contract": "fable_full_jones_oblique_response_v1",
        "response_model_contract": response_contract,
        "pixel_stack_spatial_contract": str(
            getattr(engine, "pixel_stack_spatial_contract", "")
        ),
        "terminal_detector_contract": str(
            getattr(engine, "terminal_detector_contract", "")
        ),
        "field_coordinate_contract": (
            "field_local_common_chief_ray_540nm_conserved_k_parallel_snell_v1"
        ),
    }
    return contracts


def load_reflection_fold_authorization(
    manifest: str | Path,
    *,
    engine: Any,
    width_map: torch.Tensor,
    design: str,
) -> ReflectionFoldAuthorization:
    """Validate one dispatch manifest and return a runtime token.

    Validation is intentionally repeated against the live engine instead of
    trusting the manifest's labels.  The loaded response must expose the NPZ
    SHA-256 recorded by :class:`FullJonesLUTResponseModel.from_npz`.
    """

    if design not in DESIGNS:
        raise ReflectionFoldAuthorizationError(
            f"unknown reflection-fold production design {design!r}"
        )
    source, payload = _load_manifest(manifest)
    unsigned = dict(payload)
    claimed_content = unsigned.pop("content_sha256", None)
    if payload.get("schema") != DISPATCH_MANIFEST_SCHEMA or (
        claimed_content != canonical_sha256(unsigned)
    ):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold dispatch manifest content binding drift"
        )
    if payload.get("producer_receipt_alone_authorizes_dispatch") is not False or (
        payload.get("independent_gate_recomputed_from_raw_arrays") is not True
    ):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest lacks an independent raw-array gate"
        )
    if payload.get("thresholds") != EXPECTED_THRESHOLDS:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest numerical thresholds are invalid"
        )

    contracts = payload.get("contracts")
    if not isinstance(contracts, Mapping) or dict(contracts) != EXPECTED_CONTRACTS:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest camera contracts are invalid"
        )
    live_contracts = _engine_contracts(engine)
    if live_contracts != EXPECTED_CONTRACTS:
        raise ReflectionFoldAuthorizationError(
            "live engine does not satisfy the final-exact fold contracts"
        )
    if not bool(getattr(engine, "_require_full_jones_response", False)):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold authorization requires the final-exact engine"
        )
    engine._validate_final_exact_response()

    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest has no source bindings"
        )
    source_hashes = current_source_hashes()
    for key, current_sha256 in source_hashes.items():
        if key == "gate_code_sha256":
            continue
        if bindings.get(key) != current_sha256:
            raise ReflectionFoldAuthorizationError(
                f"reflection-fold source hash is stale: {key}"
            )
    if payload.get("gate_code_sha256") != source_hashes["gate_code_sha256"]:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold gate source hash is stale"
        )

    lut_sha256 = getattr(engine.response_model, "source_npz_sha256", None)
    if not _is_sha256(lut_sha256) or bindings.get("lut_sha256") != lut_sha256:
        raise ReflectionFoldAuthorizationError(
            "loaded full-Jones LUT is not the manifest-bound NPZ"
        )
    width_sha256 = sha256_tensor(width_map)
    designs = payload.get("designs")
    if not isinstance(designs, Mapping) or set(designs) != set(DESIGNS):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest design inventory is invalid"
        )
    row = designs.get(design)
    if not isinstance(row, Mapping) or row.get("width_sha256") != width_sha256:
        raise ReflectionFoldAuthorizationError(
            "live width map is not the manifest-bound selected tensor"
        )
    raw_four_allowed = row.get("four_image_d2_reflection_fold_allowed")
    raw_eight_allowed = row.get("eight_image_d4_orbit_fold_allowed")
    if type(raw_four_allowed) is not bool or type(raw_eight_allowed) is not bool or (
        raw_eight_allowed and not raw_four_allowed
    ):
        raise ReflectionFoldAuthorizationError(
            f"reflection-fold decisions are invalid for {design}"
        )
    four_allowed = raw_four_allowed
    eight_allowed = raw_eight_allowed
    if not four_allowed:
        raise ReflectionFoldAuthorizationError(
            f"dispatch manifest does not authorize four-image reuse for {design}"
        )
    evidence = payload.get("evidence")
    evidence_sha256 = evidence.get("sha256") if isinstance(evidence, Mapping) else None
    evidence_path_value = evidence.get("path") if isinstance(evidence, Mapping) else None
    if (
        not _is_sha256(evidence_sha256)
        or evidence.get("schema") != EVIDENCE_SCHEMA
        or not isinstance(evidence_path_value, str)
    ):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold manifest lacks an immutable evidence binding"
        )
    try:
        evidence_path = Path(evidence_path_value)
        if not evidence_path.is_absolute():
            evidence_path = REPO / evidence_path
        evidence_path = evidence_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold evidence artifact is unavailable"
        ) from exc
    if not evidence_path.is_file() or evidence_path.suffix.lower() != ".npz" or (
        sha256_file(evidence_path) != evidence_sha256
    ):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold evidence artifact hash is stale"
        )
    return ReflectionFoldAuthorization(
        manifest_path=str(source),
        manifest_sha256=sha256_file(source),
        manifest_content_sha256=str(claimed_content),
        evidence_path=str(evidence_path),
        evidence_sha256=str(evidence_sha256),
        lut_sha256=str(lut_sha256),
        width_sha256=width_sha256,
        design=design,
        four_image_allowed=True,
        eight_image_allowed=eight_allowed,
        contracts_sha256=canonical_sha256(dict(contracts)),
        source_hashes_sha256=canonical_sha256(source_hashes),
    )


def validate_runtime_authorization(
    authorization: ReflectionFoldAuthorization | None,
    *,
    engine: Any,
    width_map: torch.Tensor,
    eightfold: bool,
) -> None:
    """Rebind an opaque token to the live engine and requested fold order."""

    if not isinstance(authorization, ReflectionFoldAuthorization):
        raise ReflectionFoldAuthorizationError(
            "production full-Jones reflection-fold acceleration requires a "
            "validated symmetry-dispatch authorization"
        )
    if eightfold and not authorization.eight_image_allowed:
        raise ReflectionFoldAuthorizationError(
            f"eight-image reflection fold is not authorized for {authorization.design}"
        )
    if not authorization.four_image_allowed:
        raise ReflectionFoldAuthorizationError(
            "four-image reflection fold authorization is false"
        )
    if sha256_tensor(width_map) != authorization.width_sha256:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold width map changed after authorization"
        )
    lut_sha256 = getattr(engine.response_model, "source_npz_sha256", None)
    if lut_sha256 != authorization.lut_sha256:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold LUT changed after authorization"
        )
    if _engine_contracts(engine) != EXPECTED_CONTRACTS or not bool(
        getattr(engine, "_require_full_jones_response", False)
    ):
        raise ReflectionFoldAuthorizationError(
            "reflection-fold engine contracts changed after authorization"
        )
    if canonical_sha256(current_source_hashes()) != authorization.source_hashes_sha256:
        raise ReflectionFoldAuthorizationError(
            "reflection-fold source code changed after authorization"
        )


__all__ = [
    "ReflectionFoldAuthorization",
    "ReflectionFoldAuthorizationError",
    "load_reflection_fold_authorization",
    "validate_runtime_authorization",
    "sha256_file",
    "sha256_python_tree",
    "sha256_tensor",
]
