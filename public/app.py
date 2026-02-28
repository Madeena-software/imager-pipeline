"""
Image Pipeline Node Editor — PyScript Frontend.

Installs dependencies via micropip with loading feedback (including opencv-python
from Pyodide's pre-built Wasm packages). Imports pipeline functions from
complete_pipeline and imagej_replicator (both inside public/).
If imports fail, errors are shown prominently in the UI.
"""

from js import document, window
import json

# ─── Loading helpers ─────────────────────────────────────────────────────────


def _update_loading(msg: str):
    """Update the loading overlay status text."""
    el = document.querySelector("#loading-status")
    if el:
        el.textContent = msg


def _hide_loading():
    """Hide loading overlay and reveal the main app container."""
    overlay = document.querySelector("#loading-overlay")
    if overlay:
        overlay.style.display = "none"
    app_el = document.querySelector("#app-container")
    if app_el:
        app_el.style.display = "block"


def _show_error_banner(html: str):
    """Show the import error banner with the given HTML content."""
    banner = document.querySelector("#import-error-banner")
    if banner:
        banner.innerHTML = html
        banner.style.display = "block"


# ─── Phase 1: Install packages via micropip ──────────────────────────────────
# opencv-python is available as a Pyodide pre-built Wasm package.
# Note: cv2.imshow / HighGUI won't work in browser, but all array
# operations (threshold, filter, bitwise_not, etc.) work fine.

import micropip

_PACKAGES = [
    "numpy",
    "Pillow",
    "opencv-python",  # Pyodide Wasm build — NOT a regular PyPI wheel
    "scipy",
    "scikit-image",
    "PyWavelets",
    "matplotlib",  # optional — used only for debug histograms
]

for _pkg in _PACKAGES:
    _update_loading(f"Installing {_pkg}...")
    try:
        await micropip.install(_pkg)
    except Exception as _e:
        # Non-fatal: will surface later as import error if needed
        _update_loading(f"Warning: {_pkg} install failed ({_e})")

_update_loading("Packages installed. Loading modules...")

# ─── Phase 2: Import core modules ────────────────────────────────────────────

from io import BytesIO
from pyscript import when

import numpy as np
from PIL import Image

# ─── Phase 3: Import pipeline processor ──────────────────────────────────────

_update_loading("Importing pipeline processor...")

_pipeline_load_error = ""
IMPORT_ERRORS = {}

try:
    from browser_processor import get_import_status, run_pipeline, IMPORT_ERRORS
except Exception as exc:
    _pipeline_load_error = str(exc)

# ─── Phase 4: Hide loading, reveal app, show errors ─────────────────────────

_hide_loading()

if _pipeline_load_error:
    _show_error_banner(
        f"<strong>Failed to load pipeline module:</strong> {_pipeline_load_error}"
    )
elif IMPORT_ERRORS:
    _items = "".join(
        f"<li><code>{mod}</code>: {err}</li>" for mod, err in IMPORT_ERRORS.items()
    )
    _show_error_banner(
        "<strong>Some pipeline imports failed (affected nodes will show errors when run):</strong>"
        f"<ul>{_items}</ul>"
    )

# ─── Node definitions ────────────────────────────────────────────────────────

NODE_DEFS = {
    "denoise_wavelet": {
        "label": "Denoise Wavelet",
        "default": {
            "wavelet": "sym4",
            "level": 3,
            "method": "BayesShrink",
            "mode": "soft",
        },
    },
    "crop_rotate": {
        "label": "Crop + Rotate Detector",
        "default": {
            "detector_type": "BED",
            "top": 0,
            "bottom": 0,
            "left": 0,
            "right": 0,
        },
    },
    "threshold_auto": {
        "label": "Auto Threshold + Separation",
        "default": {},
    },
    "invert": {
        "label": "Invert",
        "default": {},
    },
    "enhance_contrast": {
        "label": "Enhance Contrast (ImageJ)",
        "default": {
            "saturated_pixels": 5.0,
            "normalize": True,
            "equalize": True,
            "classic_equalization": False,
        },
    },
    "clahe": {
        "label": "CLAHE (ImageJ)",
        "default": {
            "blocksize": 127,
            "histogram_bins": 256,
            "max_slope": 0.6,
            "fast": False,
            "composite": True,
        },
    },
    "normalize": {
        "label": "Normalize Max Value",
        "default": {"saturated_pixels": 0.35},
    },
    "median_filter": {
        "label": "Advanced Median Filter",
        "default": {
            "filter_type": "hybrid_imagej",
            "radius": 2,
        },
    },
}

pipeline_nodes = []
result_data_url = None


def status(msg):
    document.querySelector("#status").textContent = msg


def update_download_link(url=None):
    anchor = document.querySelector("#download-result")
    if url:
        anchor.href = url
        anchor.classList.remove("disabled")
    else:
        anchor.href = "#"
        anchor.classList.add("disabled")


async def load_image_as_u8(file_obj):
    array_buffer = await file_obj.arrayBuffer()
    u8_array = window.Uint8Array.new(array_buffer)
    raw_bytes = bytes(u8_array.to_py())

    image = Image.open(BytesIO(raw_bytes)).convert("L")
    return np.array(image, dtype=np.uint8)


def uint8_to_png_data_url(image_u8):
    output = BytesIO()
    Image.fromarray(image_u8, mode="L").save(output, format="PNG")
    png_bytes = output.getvalue()

    from base64 import b64encode

    encoded = b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_pipeline_payload():
    nodes = []
    for index, node in enumerate(pipeline_nodes):
        textarea = document.querySelector(f"#node-params-{index}")
        try:
            params = json.loads(textarea.value or "{}")
        except Exception:
            raise ValueError(f"Params JSON tidak valid di node #{index + 1}")

        nodes.append(
            {
                "type": node["type"],
                "params": params,
            }
        )

    return {
        "version": 1,
        "nodes": nodes,
    }


def render_node_types():
    select = document.querySelector("#node-type")
    select.innerHTML = ""
    for key, cfg in NODE_DEFS.items():
        option = document.createElement("option")
        option.value = key
        option.textContent = f"{cfg['label']} ({key})"
        select.appendChild(option)


def render_pipeline():
    container = document.querySelector("#pipeline-list")
    empty_state = document.querySelector("#pipeline-empty")

    if not pipeline_nodes:
        empty_state.style.display = "block"
        container.innerHTML = ""
        return

    empty_state.style.display = "none"
    container.innerHTML = ""

    for index, node in enumerate(pipeline_nodes):
        card = document.createElement("div")
        card.className = "node-card"

        head = document.createElement("div")
        head.className = "node-head"

        title = document.createElement("strong")
        title.textContent = f"{index + 1}. {NODE_DEFS[node['type']]['label']}"

        actions = document.createElement("div")
        actions.className = "node-actions"

        up_btn = document.createElement("button")
        up_btn.textContent = "↑"
        up_btn.onclick = lambda _evt, i=index: move_node_up(i)

        down_btn = document.createElement("button")
        down_btn.textContent = "↓"
        down_btn.onclick = lambda _evt, i=index: move_node_down(i)

        remove_btn = document.createElement("button")
        remove_btn.textContent = "Hapus"
        remove_btn.onclick = lambda _evt, i=index: remove_node(i)

        actions.appendChild(up_btn)
        actions.appendChild(down_btn)
        actions.appendChild(remove_btn)

        head.appendChild(title)
        head.appendChild(actions)

        textarea = document.createElement("textarea")
        textarea.id = f"node-params-{index}"
        textarea.value = json.dumps(node["params"], indent=2)

        card.appendChild(head)
        card.appendChild(textarea)

        if index < len(pipeline_nodes) - 1:
            arrow = document.createElement("div")
            arrow.className = "node-arrow"
            arrow.textContent = "↓ connect"
            card.appendChild(arrow)

        container.appendChild(card)


def add_node(node_type):
    pipeline_nodes.append(
        {
            "type": node_type,
            "params": dict(NODE_DEFS[node_type]["default"]),
        }
    )
    render_pipeline()


def remove_node(index):
    if 0 <= index < len(pipeline_nodes):
        pipeline_nodes.pop(index)
        render_pipeline()


def move_node_up(index):
    if index <= 0:
        return
    pipeline_nodes[index - 1], pipeline_nodes[index] = (
        pipeline_nodes[index],
        pipeline_nodes[index - 1],
    )
    render_pipeline()


def move_node_down(index):
    if index >= len(pipeline_nodes) - 1:
        return
    pipeline_nodes[index + 1], pipeline_nodes[index] = (
        pipeline_nodes[index],
        pipeline_nodes[index + 1],
    )
    render_pipeline()


@when("click", "#add-node-btn")
def on_add_node(_evt):
    node_type = document.querySelector("#node-type").value
    add_node(node_type)


@when("click", "#save-pipeline-btn")
def on_save_pipeline(_evt):
    try:
        payload = build_pipeline_payload()
    except Exception as exc:
        status(str(exc))
        return

    content = json.dumps(payload, indent=2)
    blob = window.Blob.new([content], {"type": "application/json"})
    url = window.URL.createObjectURL(blob)

    a = document.createElement("a")
    a.href = url
    a.download = "pipeline.json"
    a.click()
    window.URL.revokeObjectURL(url)

    status("Pipeline JSON berhasil disimpan.")


@when("change", "#load-pipeline-file")
def on_load_pipeline(evt):
    files = evt.target.files
    if not files or files.length == 0:
        return

    file = files.item(0)
    reader = window.FileReader.new()

    def onload(_event):
        global pipeline_nodes
        try:
            raw = reader.result
            data = json.loads(raw)
            nodes = data.get("nodes", [])

            parsed = []
            for node in nodes:
                ntype = node.get("type")
                if ntype not in NODE_DEFS:
                    continue
                parsed.append(
                    {
                        "type": ntype,
                        "params": node.get("params", {}),
                    }
                )

            pipeline_nodes = parsed
            render_pipeline()
            status("Pipeline JSON berhasil dimuat.")
        except Exception as exc:
            status(f"Gagal load pipeline: {exc}")

    reader.onload = onload
    reader.readAsText(file)


@when("click", "#run-pipeline-btn")
async def on_run_pipeline(_evt):
    global result_data_url

    # Block run if pipeline module failed to load entirely
    if _pipeline_load_error:
        status(f"Pipeline tidak tersedia: {_pipeline_load_error}")
        return

    try:
        payload = build_pipeline_payload()
    except Exception as exc:
        status(str(exc))
        return

    image_input = document.querySelector("#image-file")
    if not image_input.files or image_input.files.length == 0:
        status("Pilih file gambar dulu.")
        return

    status("Menjalankan pipeline di browser...")

    try:
        image_u8 = await load_image_as_u8(image_input.files.item(0))
        result_u8, errors = run_pipeline(image_u8, payload.get("nodes", []))
        result_data_url = uint8_to_png_data_url(result_u8)

        img = document.querySelector("#result-preview")
        img.src = result_data_url
        img.style.display = "block"

        update_download_link(result_data_url)

        if errors:
            status("Pipeline selesai dengan error: " + " | ".join(errors[:3]))
        else:
            status("Pipeline selesai. Hasil siap di-preview dan di-download.")

    except Exception as exc:
        status(f"Gagal menjalankan pipeline: {exc}")


# ─── Init ─────────────────────────────────────────────────────────────────────

render_node_types()
render_pipeline()

if _pipeline_load_error:
    status(f"Pipeline module error: {_pipeline_load_error}")
elif IMPORT_ERRORS:
    status("Ready (with import warnings — see banner above). " + get_import_status())
else:
    status("Ready. " + get_import_status())
