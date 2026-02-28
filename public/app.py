"""
Image Pipeline Node Editor — PyScript Frontend.

All heavy packages (numpy, opencv, scipy, scikit-image, PyWavelets, etc.) are
installed by PyScript from pyscript.toml *before* this script executes.
Pipeline modules (complete_pipeline, imagej_replicator, etc.) are fetched
via the [files] section of pyscript.toml and available as regular imports.
"""

from js import document, window
from pyscript import when
from io import BytesIO
import json

import numpy as np
from PIL import Image

# ─── Loading helpers ─────────────────────────────────────────────────────────


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


# ─── Import pipeline processor ───────────────────────────────────────────────

_pipeline_load_error = None
IMPORT_ERRORS = {}

try:
    from browser_processor import get_import_status, run_pipeline, IMPORT_ERRORS
except Exception as exc:
    _pipeline_load_error = str(exc)

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
        "note": "Wavelet: sym4/db4/haar | Method: BayesShrink (adaptive) / VisuShrink (universal) | Mode: soft (less artifact) / hard (sharper)",
        "pre_ffc": True,
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
        "note": "detector_type: BED (tanpa rotasi) / TRX (rotasi 90° CCW) | top/bottom/left/right: pixel crop",
        "pre_ffc": True,
    },
    "flat_field_correction": {
        "label": "Flat-Field Correction (FFC)",
        "default": {},
        "note": "Rumus: (raw - dark) / (gain - dark) × mean(gain - dark). Butuh 3 gambar: dark, gain, raw. Node ini menggabungkan 3 gambar menjadi 1.",
        "pre_ffc": False,
        "is_ffc": True,
    },
    "threshold_auto": {
        "label": "Auto Threshold + Separation",
        "default": {},
        "note": "Deteksi threshold otomatis lalu pisahkan foreground/background. Tidak ada parameter.",
        "pre_ffc": False,
    },
    "invert": {
        "label": "Invert",
        "default": {},
        "note": "Membalik intensitas gambar (putih ↔ hitam). Tidak ada parameter.",
        "pre_ffc": False,
    },
    "enhance_contrast": {
        "label": "Enhance Contrast (ImageJ)",
        "default": {
            "saturated_pixels": 5.0,
            "normalize": True,
            "equalize": True,
            "classic_equalization": False,
        },
        "note": "saturated_pixels: % piksel yang di-saturate (0.1-10) | normalize: stretch histogram ke full range | equalize: equalisasi histogram | classic: metode equalisasi klasik",
        "pre_ffc": False,
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
        "note": "blocksize: ukuran blok (ganjil, 1-999) | histogram_bins: jumlah bin histogram (2-65536) | max_slope: batas kontras/slope (0.1-10) | fast: mode cepat | composite: gabungkan hasil",
        "pre_ffc": False,
    },
    "normalize": {
        "label": "Normalize Max Value",
        "default": {"saturated_pixels": 0.35},
        "note": "saturated_pixels: % piksel saturasi untuk histogram stretch (0.01-5). Makin kecil, makin lebar stretch.",
        "pre_ffc": False,
    },
    "median_filter": {
        "label": "Advanced Median Filter",
        "default": {
            "filter_type": "hybrid_imagej",
            "radius": 2,
        },
        "note": "filter_type: standard/bilateral/adaptive/nlm/morphological/hybrid_imagej/circular_imagej | radius: ukuran kernel (1-10). hybrid_imagej terbaik untuk X-ray.",
        "pre_ffc": False,
    },
}

pipeline_nodes = []
result_data_url = None

# Default apply_to targets for pre-FFC nodes
DEFAULT_APPLY_TO = ["dark", "gain", "raw"]


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

        node_data = {
            "type": node["type"],
            "params": params,
        }

        # For pre-FFC nodes, read apply_to checkboxes
        node_def = NODE_DEFS.get(node["type"], {})
        if node_def.get("pre_ffc", False):
            apply_to = []
            for target in DEFAULT_APPLY_TO:
                cb = document.querySelector(f"#node-apply-{index}-{target}")
                if cb and cb.checked:
                    apply_to.append(target)
            node_data["apply_to"] = apply_to if apply_to else list(DEFAULT_APPLY_TO)

        nodes.append(node_data)

    return {
        "version": 2,
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


def _pipeline_has_ffc() -> bool:
    """Check if any node in current pipeline is FFC."""
    return any(n["type"] == "flat_field_correction" for n in pipeline_nodes)


def render_pipeline():
    container = document.querySelector("#pipeline-list")
    empty_state = document.querySelector("#pipeline-empty")

    if not pipeline_nodes:
        empty_state.style.display = "block"
        container.innerHTML = ""
        return

    empty_state.style.display = "none"
    container.innerHTML = ""

    has_ffc = _pipeline_has_ffc()

    for index, node in enumerate(pipeline_nodes):
        node_def = NODE_DEFS[node["type"]]
        card = document.createElement("div")
        card.className = "node-card"

        head = document.createElement("div")
        head.className = "node-head"

        title = document.createElement("strong")
        title.textContent = f"{index + 1}. {node_def['label']}"

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
        card.appendChild(head)

        # Show apply_to checkboxes for pre-FFC nodes (only when FFC is in pipeline)
        if node_def.get("pre_ffc", False) and has_ffc:
            apply_label = document.createElement("div")
            apply_label.className = "apply-to-label"
            apply_label.textContent = "Terapkan pada gambar:"
            card.appendChild(apply_label)

            apply_row = document.createElement("div")
            apply_row.className = "apply-to-row"

            current_apply = node.get("apply_to", list(DEFAULT_APPLY_TO))

            for target in DEFAULT_APPLY_TO:
                lbl = document.createElement("label")
                cb = document.createElement("input")
                cb.type = "checkbox"
                cb.id = f"node-apply-{index}-{target}"
                cb.checked = target in current_apply
                span = document.createElement("span")
                span.textContent = target.capitalize()
                lbl.appendChild(cb)
                lbl.appendChild(span)
                apply_row.appendChild(lbl)

            card.appendChild(apply_row)

        # Show FFC badge
        if node_def.get("is_ffc", False):
            ffc_badge = document.createElement("div")
            ffc_badge.className = "apply-to-label"
            ffc_badge.textContent = (
                "⚡ Menggabungkan 3 gambar (dark, gain, raw) → 1 output"
            )
            card.appendChild(ffc_badge)

        # Params textarea (skip for FFC which has no params)
        if node["type"] != "flat_field_correction":
            textarea = document.createElement("textarea")
            textarea.id = f"node-params-{index}"
            textarea.value = json.dumps(node["params"], indent=2)
            card.appendChild(textarea)
        else:
            # Hidden empty textarea for payload building consistency
            textarea = document.createElement("textarea")
            textarea.id = f"node-params-{index}"
            textarea.value = "{}"
            textarea.style.display = "none"
            card.appendChild(textarea)

        # Show note
        note_text = node_def.get("note", "")
        if note_text:
            note_el = document.createElement("div")
            note_el.className = "node-note"
            note_el.textContent = f"💡 {note_text}"
            card.appendChild(note_el)

        if index < len(pipeline_nodes) - 1:
            arrow = document.createElement("div")
            arrow.className = "node-arrow"
            arrow.textContent = "↓ connect"
            card.appendChild(arrow)

        container.appendChild(card)


def add_node(node_type):
    node_data = {
        "type": node_type,
        "params": dict(NODE_DEFS[node_type]["default"]),
    }
    if NODE_DEFS[node_type].get("pre_ffc", False):
        node_data["apply_to"] = list(DEFAULT_APPLY_TO)
    pipeline_nodes.append(node_data)
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
                node_data = {
                    "type": ntype,
                    "params": node.get("params", {}),
                }
                if NODE_DEFS[ntype].get("pre_ffc", False):
                    node_data["apply_to"] = node.get("apply_to", list(DEFAULT_APPLY_TO))
                parsed.append(node_data)

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

    nodes = payload.get("nodes", [])

    # Check if pipeline contains FFC node
    has_ffc = any(n["type"] == "flat_field_correction" for n in nodes)

    if has_ffc:
        # FFC mode: need 3 images
        dark_input = document.querySelector("#image-dark")
        gain_input = document.querySelector("#image-gain")
        raw_input = document.querySelector("#image-raw")

        if (
            not dark_input.files
            or dark_input.files.length == 0
            or not gain_input.files
            or gain_input.files.length == 0
            or not raw_input.files
            or raw_input.files.length == 0
        ):
            status(
                "FFC membutuhkan 3 gambar: Dark, Gain, dan Raw. Pastikan semua ter-upload."
            )
            return

        status("Memuat 3 gambar untuk FFC...")
        try:
            dark_u8 = await load_image_as_u8(dark_input.files.item(0))
            gain_u8 = await load_image_as_u8(gain_input.files.item(0))
            raw_u8 = await load_image_as_u8(raw_input.files.item(0))
        except Exception as exc:
            status(f"Gagal memuat gambar: {exc}")
            return

        status("Menjalankan pipeline (FFC mode) di browser...")
        try:
            result_u8, errors = run_pipeline(
                raw_u8,
                nodes,
                dark_image=dark_u8,
                gain_image=gain_u8,
            )
            result_data_url = uint8_to_png_data_url(result_u8)

            img = document.querySelector("#result-preview")
            img.src = result_data_url
            img.style.display = "block"

            update_download_link(result_data_url)

            if errors:
                status("Pipeline selesai dengan error: " + " | ".join(errors[:3]))
            else:
                status("Pipeline selesai (FFC). Hasil siap di-preview dan di-download.")
        except Exception as exc:
            status(f"Gagal menjalankan pipeline: {exc}")

    else:
        # Non-FFC mode: only need raw image
        raw_input = document.querySelector("#image-raw")
        if not raw_input.files or raw_input.files.length == 0:
            status("Pilih minimal gambar Raw.")
            return

        status("Menjalankan pipeline di browser...")
        try:
            image_u8 = await load_image_as_u8(raw_input.files.item(0))
            result_u8, errors = run_pipeline(image_u8, nodes)
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

render_node_types()
render_pipeline()

if _pipeline_load_error:
    status(f"Pipeline module error: {_pipeline_load_error}")
elif IMPORT_ERRORS:
    status("Ready (with import warnings — see banner above). " + get_import_status())
else:
    status("Ready. " + get_import_status())
