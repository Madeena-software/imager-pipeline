"""
Browser Pipeline Processor.

Imports pipeline functions from complete_pipeline and imagej_replicator
(both live alongside this file inside public/).

cv2 (OpenCV) is installed via micropip as Pyodide's pre-built Wasm package.
If any import still fails, errors are tracked in IMPORT_ERRORS and shown in the UI.
No local duplicate implementations are created.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# ─── Import tracking ─────────────────────────────────────────────────────────

IMPORT_ERRORS: Dict[str, str] = {}

# Try importing functions from complete_pipeline (lives in public/ alongside this file).
# cv2 is provided by Pyodide's Wasm opencv-python package installed via micropip.

_repo_denoise_wavelet = None
_repo_apply_threshold_separation = None
_repo_auto_threshold_detection = None
_repo_invert_image = None
_repo_normalize_to_max_value = None
_repo_apply_advanced_median_filter = None
_repo_crop_and_rotate_by_detector = None
_repo_flat_field_correction = None

try:
    from complete_pipeline import (
        denoise_wavelet as _repo_denoise_wavelet,
        apply_threshold_separation as _repo_apply_threshold_separation,
        auto_threshold_detection as _repo_auto_threshold_detection,
        invert_image as _repo_invert_image,
        normalize_to_max_value as _repo_normalize_to_max_value,
        apply_advanced_median_filter as _repo_apply_advanced_median_filter,
        crop_and_rotate_by_detector as _repo_crop_and_rotate_by_detector,
        flat_field_correction as _repo_flat_field_correction,
    )
except Exception as exc:
    IMPORT_ERRORS["complete_pipeline"] = str(exc)

# Try importing ImageJ replicator (for enhance_contrast and CLAHE).
_ImageJReplicator = None

try:
    from imagej_replicator import ImageJReplicator as _ImageJReplicator
except Exception as exc:
    IMPORT_ERRORS["imagej_replicator"] = str(exc)


# ─── Status ──────────────────────────────────────────────────────────────────


def get_import_status() -> str:
    """Return human-readable import status."""
    if not IMPORT_ERRORS:
        return "All pipeline functions imported from repo successfully."
    lines = []
    for mod, err in IMPORT_ERRORS.items():
        lines.append(f"✗ {mod}: {err}")
    return " | ".join(lines)


# ─── Helpers (not duplicates — just dtype conversion utilities) ──────────────


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float32 [0, 1]."""
    if image.dtype in (np.float32, np.float64):
        return np.clip(image, 0.0, 1.0).astype(np.float32)
    return image.astype(np.float32) / 255.0


def _from_float01(image: np.ndarray) -> np.ndarray:
    """Convert float32 [0, 1] image to uint8."""
    return (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


# ─── Pipeline runner ─────────────────────────────────────────────────────────


def _import_err(label: str, module_key: str) -> str:
    """Format an import error message for a pipeline node."""
    return f"{label}: not available — {IMPORT_ERRORS.get(module_key, 'import failed')}"


def run_pipeline(
    image_u8: np.ndarray,
    nodes: List[Dict[str, Any]],
    dark_image: np.ndarray | None = None,
    gain_image: np.ndarray | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Run the image processing pipeline using functions imported from the repo.

    Supports two modes:
    1. Single-image mode (no FFC node): processes image_u8 through all nodes.
    2. Multi-image FFC mode: maintains dark/gain/raw as separate images,
       applies pre-FFC nodes to selected targets (apply_to), then FFC merges
       them into one image, and post-FFC nodes process that single result.

    If a required function could not be imported, the node is skipped and an
    error message is recorded — no local fallback duplicates are used.
    """
    has_ffc = any(n.get("type") == "flat_field_correction" for n in nodes)

    if has_ffc and dark_image is not None and gain_image is not None:
        return _run_pipeline_ffc(image_u8, nodes, dark_image, gain_image)
    else:
        return _run_pipeline_single(image_u8, nodes)


def _apply_node_to_image(
    image: np.ndarray, node_type: str, params: dict, label: str
) -> Tuple[np.ndarray, str | None]:
    """Apply a single processing node to one image. Returns (result, error_or_None)."""
    try:
        if node_type == "denoise_wavelet":
            if _repo_denoise_wavelet is None:
                return image, _import_err(label, "complete_pipeline")
            return (
                _repo_denoise_wavelet(
                    image,
                    wavelet=params.get("wavelet", "sym4"),
                    level=int(params.get("level", 3)),
                    method=params.get("method", "BayesShrink"),
                    mode=params.get("mode", "soft"),
                ).astype(np.float32),
                None,
            )

        elif node_type == "crop_rotate":
            detector_type = str(params.get("detector_type", "BED")).upper()
            top = int(params.get("top", 0))
            bottom = int(params.get("bottom", 0))
            left = int(params.get("left", 0))
            right = int(params.get("right", 0))

            h, w = image.shape
            y_end = max(top + 1, h - bottom)
            x_end = max(left + 1, w - right)
            cropped = image[top:y_end, left:x_end]

            if detector_type == "TRX":
                return np.rot90(cropped, 1).astype(np.float32), None
            return cropped.astype(np.float32), None

        elif node_type == "threshold_auto":
            if (
                _repo_auto_threshold_detection is None
                or _repo_apply_threshold_separation is None
            ):
                return image, _import_err(label, "complete_pipeline")
            threshold = _repo_auto_threshold_detection(image)
            return (
                _repo_apply_threshold_separation(image, threshold).astype(np.float32),
                None,
            )

        elif node_type == "invert":
            if _repo_invert_image is None:
                return image, _import_err(label, "complete_pipeline")
            return _repo_invert_image(image).astype(np.float32), None

        elif node_type == "enhance_contrast":
            if _ImageJReplicator is None:
                return image, _import_err(label, "imagej_replicator")
            ij = _ImageJReplicator()
            # ImageJ enhance_contrast expects uint8/uint16, not float32
            img_u8 = (
                _from_float01(image)
                if image.dtype in (np.float32, np.float64)
                else image
            )
            result = ij.enhance_contrast(
                img_u8,
                saturated_pixels=float(params.get("saturated_pixels", 5.0)),
                normalize=bool(params.get("normalize", True)),
                equalize=bool(params.get("equalize", True)),
            )
            return _to_float01(result), None

        elif node_type == "clahe":
            if _ImageJReplicator is None:
                return image, _import_err(label, "imagej_replicator")
            ij = _ImageJReplicator()
            # CLAHE expects uint8/uint16, not float32
            img_u8 = (
                _from_float01(image)
                if image.dtype in (np.float32, np.float64)
                else image
            )
            result = ij.apply_clahe(
                img_u8,
                blocksize=int(params.get("blocksize", 127)),
                histogram_bins=int(params.get("histogram_bins", 256)),
                max_slope=float(params.get("max_slope", 0.6)),
            )
            return _to_float01(result), None

        elif node_type == "normalize":
            if _repo_normalize_to_max_value is None:
                return image, _import_err(label, "complete_pipeline")
            return (
                _repo_normalize_to_max_value(
                    image,
                    saturated_pixels=float(params.get("saturated_pixels", 0.35)),
                ).astype(np.float32),
                None,
            )

        elif node_type == "median_filter":
            if _repo_apply_advanced_median_filter is None:
                return image, _import_err(label, "complete_pipeline")
            return (
                _repo_apply_advanced_median_filter(
                    image,
                    filter_type=params.get("filter_type", "hybrid_imagej"),
                    radius=int(params.get("radius", 2)),
                ).astype(np.float32),
                None,
            )

        else:
            return image, f"{label}: unknown node type"

    except Exception as exc:
        return image, f"{label}: runtime error — {exc}"


def _run_pipeline_ffc(
    raw_u8: np.ndarray,
    nodes: List[Dict[str, Any]],
    dark_u8: np.ndarray,
    gain_u8: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Run pipeline with FFC: 3 images → pre-FFC → FFC → post-FFC → result."""
    dark = _to_float01(dark_u8)
    gain = _to_float01(gain_u8)
    raw = _to_float01(raw_u8)
    errors: List[str] = []

    images = {"dark": dark, "gain": gain, "raw": raw}

    for idx, node in enumerate(nodes):
        node_type = node.get("type")
        params = node.get("params") or {}
        label = f"Node #{idx + 1} ({node_type})"

        if node_type == "flat_field_correction":
            # FFC: merge 3 images into 1
            if _repo_flat_field_correction is None:
                errors.append(_import_err(label, "complete_pipeline"))
                # Fallback: just use raw
                raw = images["raw"]
                break
            try:
                raw = _repo_flat_field_correction(
                    images["raw"], images["dark"], images["gain"]
                ).astype(np.float32)
            except Exception as exc:
                errors.append(f"{label}: runtime error — {exc}")
                raw = images["raw"]
            # After FFC, continue with single image
            current = raw
            remaining_nodes = nodes[idx + 1 :]
            for idx2, node2 in enumerate(remaining_nodes):
                node_type2 = node2.get("type")
                params2 = node2.get("params") or {}
                label2 = f"Node #{idx + idx2 + 2} ({node_type2})"
                current, err = _apply_node_to_image(
                    current, node_type2, params2, label2
                )
                if err:
                    errors.append(err)
            return _from_float01(current), errors

        else:
            # Pre-FFC node: apply to selected images
            apply_to = node.get("apply_to", ["dark", "gain", "raw"])
            for target in apply_to:
                if target in images:
                    result, err = _apply_node_to_image(
                        images[target], node_type, params, f"{label} [{target}]"
                    )
                    images[target] = result
                    if err:
                        errors.append(err)

    # If we get here without hitting FFC node (shouldn't happen but be safe)
    return _from_float01(images["raw"]), errors


def _run_pipeline_single(
    image_u8: np.ndarray, nodes: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[str]]:
    """Run pipeline in single-image mode (no FFC)."""
    current = _to_float01(image_u8)
    errors: List[str] = []

    for idx, node in enumerate(nodes):
        node_type = node.get("type")
        params = node.get("params") or {}
        label = f"Node #{idx + 1} ({node_type})"

        if node_type == "flat_field_correction":
            errors.append(
                f"{label}: FFC membutuhkan 3 gambar (dark, gain, raw) — dilewati"
            )
            continue

        current, err = _apply_node_to_image(current, node_type, params, label)
        if err:
            errors.append(err)

    return _from_float01(current), errors
