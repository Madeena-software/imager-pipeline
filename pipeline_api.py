import json
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from complete_pipeline import (
    MAX_16BIT,
    apply_advanced_median_filter,
    apply_threshold_separation,
    auto_threshold_detection,
    crop_and_rotate_by_detector,
    denoise_wavelet,
    invert_image,
    normalize_to_max_value,
)
from imagej_replicator import ImageJReplicator


NODE_SPECS = {
    "denoise_wavelet": {
        "params": {
            "wavelet": "sym4",
            "level": 3,
            "method": "BayesShrink",
            "mode": "soft",
        }
    },
    "crop_rotate": {"params": {"detector_type": "BED"}},
    "threshold_auto": {"params": {}},
    "invert": {"params": {}},
    "enhance_contrast": {
        "params": {
            "saturated_pixels": 5.0,
            "normalize": True,
            "equalize": True,
            "classic_equalization": False,
        }
    },
    "clahe": {
        "params": {
            "blocksize": 127,
            "histogram_bins": 256,
            "max_slope": 0.6,
            "fast": False,
            "composite": True,
        }
    },
    "normalize": {"params": {"saturated_pixels": 0.35}},
    "median_filter": {"params": {"filter_type": "hybrid_imagej", "radius": 2}},
}


app = FastAPI(title="Image Pipeline Node API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_float01(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.float32:
        return np.clip(image, 0.0, 1.0)
    if image.dtype == np.uint16:
        return image.astype(np.float32) / MAX_16BIT
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    img = image.astype(np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return np.clip(img, 0.0, 1.0)


def _to_uint16(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint16:
        return image
    if image.dtype == np.float32:
        return (np.clip(image, 0.0, 1.0) * MAX_16BIT).astype(np.uint16)
    if image.dtype == np.uint8:
        return ((image.astype(np.float32) / 255.0) * MAX_16BIT).astype(np.uint16)
    img = image.astype(np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return (np.clip(img, 0.0, 1.0) * MAX_16BIT).astype(np.uint16)


def _to_png8_bytes(image: np.ndarray) -> bytes:
    if image.dtype == np.float32:
        img8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        img8 = (
            (image.astype(np.float32) / MAX_16BIT * 255).clip(0, 255).astype(np.uint8)
        )
    elif image.dtype == np.uint8:
        img8 = image
    else:
        img = image.astype(np.float32)
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        img8 = (img * 255).clip(0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".png", img8)
    if not ok:
        raise HTTPException(status_code=500, detail="Gagal encode output ke PNG")
    return encoded.tobytes()


def _apply_pipeline(image: np.ndarray, nodes: List[Dict[str, Any]]) -> np.ndarray:
    current = _to_float01(image)

    for index, node in enumerate(nodes):
        node_type = node.get("type")
        params = node.get("params") or {}

        if node_type not in NODE_SPECS:
            raise HTTPException(
                status_code=400,
                detail=f"Node tidak dikenali di index {index}: {node_type}",
            )

        if node_type == "denoise_wavelet":
            current = denoise_wavelet(
                _to_float01(current),
                wavelet=params.get("wavelet", "sym4"),
                level=int(params.get("level", 3)),
                method=params.get("method", "BayesShrink"),
                mode=params.get("mode", "soft"),
            ).astype(np.float32)

        elif node_type == "crop_rotate":
            detector_type = str(params.get("detector_type", "BED")).upper()
            current = crop_and_rotate_by_detector(_to_float01(current), detector_type)

        elif node_type == "threshold_auto":
            current_float = _to_float01(current)
            threshold = auto_threshold_detection(current_float)
            current = apply_threshold_separation(current_float, threshold).astype(
                np.float32
            )

        elif node_type == "invert":
            current = invert_image(_to_float01(current)).astype(np.float32)

        elif node_type == "enhance_contrast":
            current_u16 = _to_uint16(current)
            current = ImageJReplicator.enhance_contrast(
                current_u16,
                saturated_pixels=float(params.get("saturated_pixels", 5.0)),
                normalize=bool(params.get("normalize", True)),
                equalize=bool(params.get("equalize", True)),
                classic_equalization=bool(params.get("classic_equalization", False)),
            )

        elif node_type == "clahe":
            current_u16 = _to_uint16(current)
            current = ImageJReplicator.apply_clahe(
                current_u16,
                blocksize=int(params.get("blocksize", 127)),
                histogram_bins=int(params.get("histogram_bins", 256)),
                max_slope=float(params.get("max_slope", 0.6)),
                fast=bool(params.get("fast", False)),
                composite=bool(params.get("composite", True)),
            )

        elif node_type == "normalize":
            current_u16 = _to_uint16(current)
            current = normalize_to_max_value(
                current_u16,
                saturated_pixels=float(params.get("saturated_pixels", 0.35)),
            )

        elif node_type == "median_filter":
            current_u16 = _to_uint16(current)
            current = apply_advanced_median_filter(
                current_u16,
                filter_type=params.get("filter_type", "hybrid_imagej"),
                radius=int(params.get("radius", 2)),
            )

    return current


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/nodes")
def available_nodes() -> Dict[str, Any]:
    return {"nodes": NODE_SPECS}


@app.post("/api/run")
async def run_pipeline(
    image: UploadFile = File(...),
    pipeline_json: str = Form(...),
):
    try:
        payload = json.loads(pipeline_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400, detail=f"pipeline_json bukan JSON valid: {exc}"
        )

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise HTTPException(
            status_code=400,
            detail="Format pipeline tidak valid: field 'nodes' harus list",
        )

    raw = await image.read()
    np_bytes = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(np_bytes, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Gagal membaca file gambar")

    if len(decoded.shape) == 3:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)

    result = _apply_pipeline(decoded, nodes)
    output_png = _to_png8_bytes(result)

    return Response(content=output_png, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("pipeline_api:app", host="0.0.0.0", port=8000, reload=True)
