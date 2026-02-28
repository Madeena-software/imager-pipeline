# Image Pipeline — PyScript Web App

## Ringkasan

Web app untuk merangkai dan menjalankan image processing pipeline langsung di browser menggunakan **PyScript** (Pyodide/Wasm). Tidak ada backend/API — semua proses berjalan 100% di sisi client.

Fitur:
- Node-based pipeline builder (drag & reorder)
- Load/Save pipeline JSON
- Flat-Field Correction (FFC) dengan 3 gambar: dark, gain, raw
- Semua image processing (denoise, CLAHE, threshold, dll.) jalan di browser

Frontend ada di folder `public/`.

---

## Cara Menjalankan

PyScript perlu disajikan via HTTP (tidak bisa buka `file://` langsung). Jalankan static server:

```bash
python -m http.server 5500 --directory public
```

Buka: http://127.0.0.1:5500

---

## Format Pipeline JSON

```json
{
  "version": 2,
  "nodes": [
    { "type": "denoise_wavelet", "params": { "wavelet": "sym4", "level": 3, "method": "BayesShrink", "mode": "soft" } },
    { "type": "threshold_auto", "params": {} },
    { "type": "invert", "params": {} },
    { "type": "clahe", "params": { "blocksize": 127, "histogram_bins": 256, "max_slope": 0.6, "fast": false, "composite": true } }
  ]
}
```

Node yang tersedia:
- `denoise_wavelet` — Wavelet denoising (sym4/db4/haar)
- `crop_rotate` — Crop dan rotasi untuk detektor (BED/TRX)
- `flat_field_correction` — FFC (butuh 3 gambar)
- `threshold_auto` — Auto threshold + separation
- `invert` — Balik intensitas
- `enhance_contrast` — Enhance contrast (ImageJ-style)
- `clahe` — CLAHE (ImageJ-style)
- `normalize` — Normalize intensitas
- `median_filter` — Advanced median filter

---

## Deploy ke Firebase Hosting

```bash
firebase deploy --only hosting
```
