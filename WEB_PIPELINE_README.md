# Node-based Image Pipeline Web (Frontend-only PyScript)

## Ringkasan
Implementasi ini menyediakan:
- Web UI berbasis PyScript untuk merangkai pipeline model node (urutan node).
- Load/Save pipeline JSON (tanpa login).
- Eksekusi pipeline langsung di browser (tanpa backend wajib).

Frontend ada di folder `public/` (siap untuk Firebase Hosting).

> Catatan: app akan mencoba import fungsi dari repo jika runtime browser mendukung.
> Jika gagal (umumnya karena dependency native seperti `cv2`), app otomatis fallback ke processor kompatibel browser.

---

## 1) Jalankan frontend lokal

Karena PyScript berjalan di browser, frontend bisa disajikan dari folder `public`:

```bash
python -m http.server 5500 --directory public
```

Buka:

```bash
http://127.0.0.1:5500
```

Default API URL di UI adalah `http://127.0.0.1:8000`.

Tidak perlu menjalankan API backend untuk mode ini.

---

## 2) Format pipeline JSON

```json
{
  "version": 1,
  "nodes": [
    { "type": "denoise_wavelet", "params": { "wavelet": "sym4", "level": 3, "method": "BayesShrink", "mode": "soft" } },
    { "type": "threshold_auto", "params": {} },
    { "type": "invert", "params": {} },
    { "type": "clahe", "params": { "blocksize": 127, "histogram_bins": 256, "max_slope": 0.6, "fast": false, "composite": true } }
  ]
}
```

Node yang tersedia:
- `denoise_wavelet`
- `crop_rotate`
- `threshold_auto`
- `invert`
- `enhance_contrast`
- `clahe`
- `normalize`
- `median_filter`

---

## 3) Deploy ke Firebase Hosting

Frontend static bisa langsung deploy:

```bash
firebase deploy --only hosting
```

### Catatan kompatibilitas
- Mode frontend-only cocok untuk demo, kolaborasi, dan eksplorasi pipeline tim.
- Beberapa node dijalankan dengan pendekatan kompatibel browser (approximation), terutama ketika dependency native dari repo tidak tersedia di runtime PyScript.
