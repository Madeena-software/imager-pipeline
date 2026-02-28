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

try:
    from complete_pipeline import (
        denoise_wavelet as _repo_denoise_wavelet,
        apply_threshold_separation as _repo_apply_threshold_separation,
        auto_threshold_detection as _repo_auto_threshold_detection,
        invert_image as _repo_invert_image,
        normalize_to_max_value as _repo_normalize_to_max_value,
        apply_advanced_median_filter as _repo_apply_advanced_median_filter,
        crop_and_rotate_by_detector as _repo_crop_and_rotate_by_detector,
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
    image_u8: np.ndarray, nodes: List[Dict[str, Any]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Run the image processing pipeline using functions imported from the repo.

    If a required function could not be imported, the node is skipped and an
    error message is recorded — no local fallback duplicates are used.
    """
    current = _to_float01(image_u8)
    errors: List[str] = []

    for idx, node in enumerate(nodes):
        node_type = node.get("type")
        params = node.get("params") or {}
        label = f"Node #{idx + 1} ({node_type})"

        try:
            if node_type == "denoise_wavelet":
                if _repo_denoise_wavelet is None:
                    errors.append(_import_err(label, "complete_pipeline"))
                    continue
                current = _repo_denoise_wavelet(
                    current,
                    wavelet=params.get("wavelet", "sym4"),
                    level=int(params.get("level", 3)),
                    method=params.get("method", "BayesShrink"),
                    mode=params.get("mode", "soft"),
                ).astype(np.float32)

            elif node_type == "crop_rotate":
                # crop_rotate uses per-node params (top/bottom/left/right).
                # The repo's crop_and_rotate_by_detector reads from global CONFIG,
                # so we use basic numpy slicing here (not a function duplication).
                detector_type = str(params.get("detector_type", "BED")).upper()
                top = int(params.get("top", 0))
                bottom = int(params.get("bottom", 0))
                left = int(params.get("left", 0))
                right = int(params.get("right", 0))

                h, w = current.shape
                y_end = max(top + 1, h - bottom)
                x_end = max(left + 1, w - right)
                cropped = current[top:y_end, left:x_end]

                if detector_type == "TRX":
                    current = np.rot90(cropped, 1).astype(np.float32)
                else:
                    current = cropped.astype(np.float32)

            elif node_type == "threshold_auto":
                if (
                    _repo_auto_threshold_detection is None
                    or _repo_apply_threshold_separation is None
                ):
                    errors.append(_import_err(label, "complete_pipeline"))
                    continue
                threshold = _repo_auto_threshold_detection(current)
                current = _repo_apply_threshold_separation(current, threshold).astype(
                    np.float32
                )

            elif node_type == "invert":
                if _repo_invert_image is None:
                    errors.append(_import_err(label, "complete_pipeline"))
                    continue
                current = _repo_invert_image(current).astype(np.float32)

            elif node_type == "enhance_contrast":
                if _ImageJReplicator is None:
                    errors.append(_import_err(label, "imagej_replicator"))
                    continue
                ij = _ImageJReplicator()
                current = ij.enhance_contrast(
                    current,
                    saturated_pixels=float(params.get("saturated_pixels", 5.0)),
                    normalize=bool(params.get("normalize", True)),
                    equalize=bool(params.get("equalize", True)),
                ).astype(np.float32)

            elif node_type == "clahe":
                if _ImageJReplicator is None:
                    errors.append(_import_err(label, "imagej_replicator"))
                    continue
                ij = _ImageJReplicator()
                current = ij.apply_clahe(
                    current,
                    blocksize=int(params.get("blocksize", 127)),
                    histogram_bins=int(params.get("histogram_bins", 256)),
                    max_slope=float(params.get("max_slope", 0.6)),
                ).astype(np.float32)

            elif node_type == "normalize":
                if _repo_normalize_to_max_value is None:
                    errors.append(_import_err(label, "complete_pipeline"))
                    continue
                current = _repo_normalize_to_max_value(
                    current,
                    saturated_pixels=float(params.get("saturated_pixels", 0.35)),
                ).astype(np.float32)

            elif node_type == "median_filter":
                if _repo_apply_advanced_median_filter is None:
                    errors.append(_import_err(label, "complete_pipeline"))
                    continue
                current = _repo_apply_advanced_median_filter(
                    current,
                    filter_type=params.get("filter_type", "hybrid_imagej"),
                    radius=int(params.get("radius", 2)),
                ).astype(np.float32)

            else:
                errors.append(f"{label}: unknown node type")

        except Exception as exc:
            errors.append(f"{label}: runtime error — {exc}")

    return _from_float01(current), errors
