"""
Replicator untuk fungsi pemrosesan citra ImageJ di Python.

Termasuk implementasi:
- ContrastEnhancer (Enhance Contrast)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Hybrid 2D Median Filter (3x3, 5x5, 7x7)
- Fast Temporal Median Filter (running median subtraction untuk stack)
- Median Filter (circular kernel, Process > Filters > Median...)

CLAHE Implementation Reference:
    Zuiderveld, Karel. "Contrast limited adaptive histogram equalization."
    Graphics gems IV. Academic Press Professional, Inc., 1994. 474-485.

License: GPL v2 (mengikuti lisensi asli ImageJ CLAHE plugin)
"""

import cv2
import numpy as np
import math
from typing import Optional, Tuple, Union
from scipy.ndimage import gaussian_filter1d, median_filter
from concurrent.futures import ThreadPoolExecutor
import warnings

# Konstanta bit depth
MAX_UINT8 = 255
MAX_UINT16 = 65535
BINS_UINT8 = 256
BINS_UINT16 = 65536


class ImageJReplicator:
    """
    Kelas utilitas untuk mereplikasi fungsi pemrosesan citra ImageJ
    di lingkungan Python/OpenCV dengan presisi tinggi.

    Implementasi ini mengikuti logika ContrastEnhancer.java dari ImageJ
    untuk memastikan hasil yang identik.
    """

    @staticmethod
    def _get_min_and_max_imagej(
        histogram: np.ndarray, saturated: float, pixel_count: int
    ) -> Tuple[int, int]:
        """
        Mereplikasi metode getMinAndMax() dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan pendekatan berbasis histogram dengan threshold counting,
        bukan percentile-based seperti implementasi umum lainnya.

        Args:
            histogram: Array histogram (256 bins untuk 8-bit, 65536 untuk 16-bit)
            saturated: Persentase piksel tersaturasi (0-100)
            pixel_count: Total jumlah piksel dalam gambar

        Returns:
            Tuple (hmin, hmax): Indeks histogram untuk min dan max
        """
        hsize = len(histogram)

        # ImageJ: threshold = (pixelCount * saturated / 200.0)
        # Ini membagi saturated menjadi setengah untuk low dan high
        if saturated > 0.0:
            threshold = int(pixel_count * saturated / 200.0)
        else:
            threshold = 0

        # Cari hmin: scan dari kiri sampai count melebihi threshold
        i = -1
        found = False
        count = 0
        maxindex = hsize - 1

        while not found and i < maxindex:
            i += 1
            count += histogram[i]
            found = count > threshold
        hmin = i

        # Cari hmax: scan dari kanan sampai count melebihi threshold
        i = hsize
        count = 0
        found = False

        while not found and i > 0:
            i -= 1
            count += histogram[i]
            found = count > threshold
        hmax = i

        return hmin, hmax

    @staticmethod
    def _normalize_imagej(
        image: np.ndarray, min_val: float, max_val: float
    ) -> np.ndarray:
        """
        Mereplikasi metode normalize() dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan LUT (Look-Up Table) untuk normalisasi, yang memberikan
        hasil yang sedikit berbeda dari linear scaling biasa.

        Args:
            image: Citra input grayscale
            min_val: Nilai minimum dari histogram stretching
            max_val: Nilai maksimum dari histogram stretching

        Returns:
            Citra yang dinormalisasi dengan tipe data yang sama
        """
        original_dtype = image.dtype

        # Tentukan range berdasarkan bit depth
        if original_dtype == np.uint16:
            max2 = 65535
            range_val = 65536
        else:
            max2 = 255
            range_val = 256

        # Buat LUT seperti ImageJ
        lut = np.zeros(range_val, dtype=np.float64)

        for i in range(range_val):
            if i <= min_val:
                lut[i] = 0
            elif i >= max_val:
                lut[i] = max2
            else:
                # Formula ImageJ: (int)(((double)(i-min)/(max-min))*max2)
                lut[i] = int(((i - min_val) / (max_val - min_val)) * max2)

        # Terapkan LUT
        lut = lut.astype(original_dtype)
        return lut[image]

    @staticmethod
    def enhance_contrast(
        image: np.ndarray,
        saturated_pixels: float = 0.35,
        equalize: bool = False,
        normalize: bool = True,
        classic_equalization: bool = False,
    ) -> np.ndarray:
        """
        Mereplikasi ImageJ 'Enhance Contrast' (ContrastEnhancer.java).

        Args:
            image (np.ndarray): Citra input (Grayscale atau RGB).
            saturated_pixels (float): Persentase piksel tersaturasi (Default ImageJ: 0.35).
            equalize (bool): Jika True, lakukan Histogram Equalization (varian ImageJ).
            normalize (bool): Jika True, lakukan stretching dengan LUT (Data diubah).
            classic_equalization (bool): Jika True, gunakan HE klasik.
                Jika False (default), gunakan sqrt-weighted HE seperti ImageJ.

        Returns:
            np.ndarray: Citra hasil pemrosesan (preserves input bit depth).

        Raises:
            ValueError: Jika input tidak valid.
            TypeError: Jika tipe data input tidak sesuai.
        """
        # Validasi input
        if image is None:
            raise ValueError("Citra input tidak boleh kosong")

        if not isinstance(image, np.ndarray):
            raise TypeError("Input harus berupa numpy array")

        if image.size == 0:
            raise ValueError("Array citra tidak boleh kosong")

        # Clamp saturated_pixels seperti ImageJ
        if saturated_pixels < 0.0:
            saturated_pixels = 0.0
        if saturated_pixels > 100.0:
            saturated_pixels = 100.0

        # Simpan tipe data asli untuk preservasi bit depth
        original_dtype = image.dtype

        # ---------------------------------------------------------
        # MODE 1: EQUALIZE HISTOGRAM (Varian ImageJ)
        # ---------------------------------------------------------
        if equalize:
            try:
                if len(image.shape) == 3:
                    # Konversi ke LAB, proses luminance channel (L)
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l_eq = ImageJReplicator._equalize_imagej_variant(
                        l, classic_equalization
                    )
                    res_lab = cv2.merge((l_eq, a, b))
                    return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
                else:
                    return ImageJReplicator._equalize_imagej_variant(
                        image, classic_equalization
                    )
            except cv2.error as e:
                raise ValueError(f"Gagal melakukan konversi color space: {e}")

        # ---------------------------------------------------------
        # MODE 2: STRETCH HISTOGRAM (stretchHistogram dari ImageJ)
        # ---------------------------------------------------------
        # Jika normalize=False di ImageJ, hanya display range yang berubah (metadata).
        # Di Python, kita mengembalikan gambar asli tanpa perubahan.
        if not normalize:
            return image

        # Proses berdasarkan tipe gambar
        if len(image.shape) == 3:
            # Untuk RGB, proses luminance di LAB space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_stretched = ImageJReplicator._stretch_histogram_imagej(
                l, saturated_pixels, normalize
            )
            res_lab = cv2.merge((l_stretched, a, b))
            return cv2.cvtColor(res_lab, cv2.COLOR_LAB2BGR)
        else:
            return ImageJReplicator._stretch_histogram_imagej(
                image, saturated_pixels, normalize
            )

    @staticmethod
    def _stretch_histogram_imagej(
        image: np.ndarray, saturated: float, normalize: bool = True
    ) -> np.ndarray:
        """
        Implementasi internal stretchHistogram dari ImageJ.

        Mengikuti logika exact dari ContrastEnhancer.java:
        1. Hitung histogram
        2. Cari hmin dan hmax menggunakan threshold counting
        3. Hitung min dan max dari bin positions
        4. Terapkan normalisasi dengan LUT

        Args:
            image: Citra grayscale input
            saturated: Persentase saturasi (0-100)
            normalize: Jika True, terapkan LUT normalisasi

        Returns:
            Citra yang di-stretch
        """
        original_dtype = image.dtype

        # Tentukan bins berdasarkan bit depth
        if original_dtype == np.uint16:
            num_bins = BINS_UINT16
        else:
            num_bins = BINS_UINT8

        # Hitung histogram
        histogram, _ = np.histogram(image.flatten(), bins=num_bins, range=(0, num_bins))
        histogram = histogram.astype(np.int64)

        pixel_count = image.size

        # Dapatkan hmin dan hmax menggunakan metode ImageJ
        hmin, hmax = ImageJReplicator._get_min_and_max_imagej(
            histogram, saturated, pixel_count
        )

        if hmax <= hmin:
            return image  # Tidak ada stretching yang diperlukan

        # Untuk 8-bit dan 16-bit, min dan max langsung dari bin index
        min_val = float(hmin)
        max_val = float(hmax)

        if normalize:
            return ImageJReplicator._normalize_imagej(image, min_val, max_val)
        else:
            # Tanpa normalize, ImageJ hanya mengubah display range
            # Di Python kita kembalikan asli
            return image

    @staticmethod
    def _equalize_imagej_variant(
        gray_image: np.ndarray, classic_equalization: bool = False
    ) -> np.ndarray:
        """
        Implementasi exact Histogram Equalization dari ImageJ ContrastEnhancer.java.

        ImageJ menggunakan integrasi trapesium dengan opsi weighted (sqrt) atau classic.
        Algoritma ini menggunakan formula:
        - sum = getWeightedValue(histogram, 0)
        - for i=1 to max-1: sum += 2 * getWeightedValue(histogram, i)
        - sum += getWeightedValue(histogram, max)
        - scale = range/sum
        - lut[0] = 0, lut[max] = max
        - for i=1 to max-1: lut[i] = round(cumulative_sum * scale)

        Args:
            gray_image: Citra grayscale (uint8 atau uint16)
            classic_equalization: Jika True, gunakan histogram langsung.
                Jika False (default), gunakan sqrt(histogram) untuk hasil lebih halus.

        Returns:
            Citra hasil ekualisasi dengan tipe data yang sama dengan input.
        """
        original_dtype = gray_image.dtype

        # Tentukan range berdasarkan bit depth
        if original_dtype == np.uint16:
            max_val = 65535
            range_val = 65535
        else:
            max_val = 255
            range_val = 255

        # Hitung histogram
        histogram = np.bincount(gray_image.flatten(), minlength=max_val + 1)
        histogram = histogram.astype(np.float64)

        def get_weighted_value(hist: np.ndarray, i: int, classic: bool) -> float:
            """Replikasi getWeightedValue dari ImageJ."""
            h = hist[i]
            if h < 2 or classic:
                return float(h)
            return math.sqrt(float(h))

        # Hitung sum menggunakan formula ImageJ (integrasi trapesium)
        total_sum = get_weighted_value(histogram, 0, classic_equalization)
        for i in range(1, max_val):
            total_sum += 2 * get_weighted_value(histogram, i, classic_equalization)
        total_sum += get_weighted_value(histogram, max_val, classic_equalization)

        # Edge case: jika sum sangat kecil
        if total_sum < 1e-10:
            return gray_image

        scale = range_val / total_sum

        # Buat LUT
        lut = np.zeros(range_val + 1, dtype=np.int64)
        lut[0] = 0

        cumsum = get_weighted_value(histogram, 0, classic_equalization)
        for i in range(1, max_val):
            delta = get_weighted_value(histogram, i, classic_equalization)
            cumsum += delta
            lut[i] = int(round(cumsum * scale))
            cumsum += delta

        lut[max_val] = max_val

        # Clip LUT ke range valid
        lut = np.clip(lut, 0, max_val).astype(original_dtype)

        # Terapkan LUT
        return lut[gray_image]

    # ---------------------------------------------------------
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ---------------------------------------------------------

    @staticmethod
    def apply_clahe(
        image: np.ndarray,
        blocksize: int = 127,
        histogram_bins: int = 256,
        max_slope: float = 3.0,
        mask: Optional[np.ndarray] = None,
        fast: bool = True,
        composite: bool = True,
    ) -> np.ndarray:
        """
        Menerapkan CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Mereplikasi plugin CLAHE dari ImageJ/Fiji (mpicbg.ij.clahe)
        yang dikembangkan oleh Stephan Saalfeld.

        Reference:
            Zuiderveld, Karel. "Contrast limited adaptive histogram equalization."
            Graphics gems IV. Academic Press Professional, Inc., 1994. 474-485.

        Args:
            image: Input image (grayscale atau RGB)
            blocksize: Ukuran blok dalam pixel (default: 127)
            histogram_bins: Jumlah histogram bins (default: 256)
            max_slope: Maximum slope untuk contrast limiting (default: 3.0)
            mask: Optional mask (ByteProcessor equivalent)
            fast: Gunakan metode cepat yang kurang akurat (default: True)
            composite: Untuk RGB, proses setiap channel terpisah (default: True)

        Returns:
            Processed image dengan tipe data yang sama

        Example:
            >>> result = ImageJReplicator.apply_clahe(image, blocksize=127, histogram_bins=256, max_slope=3.0)
        """
        block_radius = (blocksize - 1) // 2
        bins = histogram_bins - 1

        if fast:
            return ImageJReplicator._clahe_fast(
                image, block_radius, bins, max_slope, mask, composite
            )
        else:
            return ImageJReplicator._clahe_precise(
                image, block_radius, bins, max_slope, mask, composite
            )

    @staticmethod
    def _clahe_create_histogram_lut(
        histogram: np.ndarray, slope: float, bins: int, n_pixels: int, max_val: int
    ) -> np.ndarray:
        """
        Buat LUT dari histogram dengan contrast limiting.
        """
        clip_limit = int(slope * n_pixels / (bins + 1))
        if clip_limit < 1:
            clip_limit = 1

        clipped_hist = histogram.copy().astype(np.float64)
        excess = 0

        for i in range(len(clipped_hist)):
            if clipped_hist[i] > clip_limit:
                excess += clipped_hist[i] - clip_limit
                clipped_hist[i] = clip_limit

        redistribution = excess / (bins + 1)
        clipped_hist += redistribution

        residual = excess - redistribution * (bins + 1)
        if residual > 0:
            step = max(1, (bins + 1) // int(residual + 1))
            for i in range(0, len(clipped_hist), step):
                if residual <= 0:
                    break
                clipped_hist[i] += 1
                residual -= 1

        cdf = np.cumsum(clipped_hist)
        cdf_min = cdf[0]
        cdf_max = cdf[-1]

        if cdf_max - cdf_min > 0:
            lut = ((cdf - cdf_min) / (cdf_max - cdf_min) * max_val).astype(np.uint8)
        else:
            lut = np.arange(bins + 1, dtype=np.uint8)

        return lut

    @staticmethod
    def _clahe_compute_block_histogram(
        image: np.ndarray,
        row: int,
        col: int,
        block_radius: int,
        bins: int,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Hitung histogram untuk blok di sekitar pixel (row, col).
        """
        height, width = image.shape[:2]

        r_min = max(0, row - block_radius)
        r_max = min(height, row + block_radius + 1)
        c_min = max(0, col - block_radius)
        c_max = min(width, col + block_radius + 1)

        block = image[r_min:r_max, c_min:c_max]

        if mask is not None:
            mask_block = mask[r_min:r_max, c_min:c_max]
            block = block[mask_block > 0]

        n_pixels = block.size

        if n_pixels == 0:
            return np.zeros(bins + 1, dtype=np.int64), 0

        if image.dtype == np.uint16:
            quantized = (block.astype(np.float64) / MAX_UINT16 * bins).astype(np.int32)
        else:
            quantized = (block.astype(np.float64) / MAX_UINT8 * bins).astype(np.int32)

        quantized = np.clip(quantized, 0, bins)
        histogram = np.bincount(quantized.flatten(), minlength=bins + 1)

        return histogram.astype(np.int64), n_pixels

    @staticmethod
    def _clahe_fast(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
        composite: bool,
    ) -> np.ndarray:
        """
        Implementasi CLAHE cepat menggunakan OpenCV sebagai basis.
        """
        original_dtype = image.dtype

        if len(image.shape) == 3:
            if composite:
                channels = cv2.split(image)
                processed = []
                for ch in channels:
                    processed.append(
                        ImageJReplicator._clahe_apply_single(
                            ch, block_radius, bins, slope, mask
                        )
                    )
                return cv2.merge(processed)
            else:
                if original_dtype == np.uint16:
                    img_8bit = (image / 256).astype(np.uint8)
                    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

                l, a, b = cv2.split(lab)
                l_processed = ImageJReplicator._clahe_apply_single(
                    l, block_radius, bins, slope, mask
                )
                result_lab = cv2.merge([l_processed, a, b])
                result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

                if original_dtype == np.uint16:
                    result = result.astype(np.uint16) * 256
                return result
        else:
            return ImageJReplicator._clahe_apply_single(
                image, block_radius, bins, slope, mask
            )

    @staticmethod
    def _clahe_apply_single(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Terapkan CLAHE ke single-channel image.
        """
        original_dtype = image.dtype

        if original_dtype == np.uint16:
            work_image = image.copy()
        else:
            work_image = image.copy()

        block_size = block_radius * 2 + 1
        height, width = work_image.shape

        tiles_x = max(1, width // block_size)
        tiles_y = max(1, height // block_size)

        clahe_obj = cv2.createCLAHE(clipLimit=slope, tileGridSize=(tiles_x, tiles_y))
        result = clahe_obj.apply(work_image)

        if mask is not None:
            mask_binary = (mask > 0).astype(np.uint8)
            result = np.where(mask_binary, result, work_image)

        return result.astype(original_dtype)

    @staticmethod
    def _clahe_precise(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
        composite: bool,
    ) -> np.ndarray:
        """
        Implementasi CLAHE presisi tinggi yang lebih dekat ke ImageJ.
        """
        original_dtype = image.dtype

        if len(image.shape) == 3:
            if composite:
                channels = cv2.split(image)
                processed = []
                for ch in channels:
                    processed.append(
                        ImageJReplicator._clahe_apply_precise(
                            ch, block_radius, bins, slope, mask
                        )
                    )
                return cv2.merge(processed)
            else:
                if original_dtype == np.uint16:
                    img_8bit = (image / 256).astype(np.uint8)
                    lab = cv2.cvtColor(img_8bit, cv2.COLOR_BGR2LAB)
                else:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

                l, a, b = cv2.split(lab)
                l_processed = ImageJReplicator._clahe_apply_precise(
                    l, block_radius, bins, slope, mask
                )
                result_lab = cv2.merge([l_processed, a, b])
                result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

                if original_dtype == np.uint16:
                    result = result.astype(np.uint16) * 256
                return result
        else:
            return ImageJReplicator._clahe_apply_precise(
                image, block_radius, bins, slope, mask
            )

    @staticmethod
    def _clahe_apply_precise(
        image: np.ndarray,
        block_radius: int,
        bins: int,
        slope: float,
        mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Implementasi CLAHE presisi untuk single channel dengan interpolasi bilinear.
        """
        original_dtype = image.dtype
        height, width = image.shape

        if original_dtype == np.uint16:
            max_val = MAX_UINT16
            scale_factor = MAX_UINT16 / bins
        else:
            max_val = MAX_UINT8
            scale_factor = MAX_UINT8 / bins

        block_size = block_radius * 2 + 1

        n_blocks_y = max(1, (height + block_size - 1) // block_size)
        n_blocks_x = max(1, (width + block_size - 1) // block_size)

        if n_blocks_y > 1:
            step_y = (height - 1) / (n_blocks_y - 1)
        else:
            step_y = height

        if n_blocks_x > 1:
            step_x = (width - 1) / (n_blocks_x - 1)
        else:
            step_x = width

        luts = np.zeros((n_blocks_y, n_blocks_x, bins + 1), dtype=np.float64)

        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                cy = int(by * step_y) if n_blocks_y > 1 else height // 2
                cx = int(bx * step_x) if n_blocks_x > 1 else width // 2

                hist, n_pixels = ImageJReplicator._clahe_compute_block_histogram(
                    image, cy, cx, block_radius, bins, mask
                )

                if n_pixels > 0:
                    luts[by, bx] = ImageJReplicator._clahe_create_histogram_lut(
                        hist, slope, bins, n_pixels, bins
                    )
                else:
                    luts[by, bx] = np.arange(bins + 1)

        result = np.zeros_like(image, dtype=np.float64)
        quantized = (image.astype(np.float64) / max_val * bins).astype(np.int32)
        quantized = np.clip(quantized, 0, bins)

        for y in range(height):
            for x in range(width):
                fy = y / step_y if n_blocks_y > 1 else 0
                fx = x / step_x if n_blocks_x > 1 else 0

                by0 = int(fy)
                bx0 = int(fx)
                by1 = min(by0 + 1, n_blocks_y - 1)
                bx1 = min(bx0 + 1, n_blocks_x - 1)

                wy = fy - by0
                wx = fx - bx0

                pixel = quantized[y, x]

                v00 = luts[by0, bx0, pixel]
                v01 = luts[by0, bx1, pixel]
                v10 = luts[by1, bx0, pixel]
                v11 = luts[by1, bx1, pixel]

                v0 = v00 * (1 - wx) + v01 * wx
                v1 = v10 * (1 - wx) + v11 * wx
                v = v0 * (1 - wy) + v1 * wy

                result[y, x] = v * scale_factor

        if mask is not None:
            mask_binary = mask > 0
            result = np.where(mask_binary, result, image.astype(np.float64))

        result = np.clip(result, 0, max_val).astype(original_dtype)

        return result

    # ---------------------------------------------------------
    # Hybrid 2D Median Filter
    # ---------------------------------------------------------

    @staticmethod
    def hybrid_median_filter_2d(
        image: np.ndarray,
        kernel_size: int = 3,
        repetitions: int = 1,
    ) -> np.ndarray:
        """
        Mereplikasi plugin Hybrid 2D Median Filter dari ImageJ.

        Implementasi referensi: Hybrid_2D_Median_Filter.java
        oleh Christopher Philip Mauer (cpmauer@northwestern.edu).

        Filter hybrid median menghitung:
        1. Median dari piksel kernel berbentuk Plus (+)
        2. Median dari piksel kernel berbentuk X
        3. Kemudian mengambil median dari [median_plus, median_x, piksel_pusat]

        Kernel Plus (+) mengambil piksel sepanjang arah kardinal (atas, bawah,
        kiri, kanan) dari piksel pusat, sedangkan kernel X mengambil piksel
        sepanjang arah diagonal. Pendekatan hybrid ini memberikan smoothing
        yang mempertahankan tepi (edge-preserving) lebih baik dibanding
        filter median standar.

        Boundary handling menggunakan 'edge' mode (replicate) yang mereplikasi
        perilaku fallback cascading dari implementasi Java asli, di mana piksel
        di luar batas diganti dengan piksel terdekat pada arah yang sama.

        Args:
            image: Citra input grayscale (uint8 atau uint16).
                Untuk citra RGB/multi-channel, setiap channel diproses terpisah.
            kernel_size: Ukuran kernel - 3, 5, atau 7 (default: 3).
                Sesuai opsi "3x3", "5x5", "7x7" di dialog ImageJ.
            repetitions: Jumlah pengulangan filter (default: 1).
                Sesuai "Number of Repetitions" di dialog ImageJ.

        Returns:
            np.ndarray: Citra terfilter dengan dtype yang sama dengan input.

        Raises:
            ValueError: Jika kernel_size bukan 3, 5, atau 7, atau input kosong.
            TypeError: Jika tipe data input tidak sesuai.

        Example:
            >>> filtered = ImageJReplicator.hybrid_median_filter_2d(
            ...     image, kernel_size=5, repetitions=2
            ... )
        """
        # Validasi input
        if image is None:
            raise ValueError("Citra input tidak boleh kosong")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input harus berupa numpy array")
        if image.size == 0:
            raise ValueError("Array citra tidak boleh kosong")
        if kernel_size not in (3, 5, 7):
            raise ValueError(
                "kernel_size harus 3, 5, atau 7 "
                "(sesuai opsi '3x3', '5x5', '7x7' di ImageJ)"
            )
        if repetitions < 1:
            warnings.warn(
                "Nilai repetitions tidak valid. "
                "Harus >= 1. Menggunakan 1 pengulangan.",
                stacklevel=2,
            )
            repetitions = 1

        original_dtype = image.dtype
        radius = kernel_size // 2  # 1, 2, atau 3

        # Handle citra multi-channel (RGB dsb.)
        if len(image.shape) == 3:
            channels = cv2.split(image)
            processed = [
                ImageJReplicator.hybrid_median_filter_2d(ch, kernel_size, repetitions)
                for ch in channels
            ]
            return cv2.merge(processed)

        # Kerja dengan float64 untuk presisi (mereplikasi double[] di Java)
        result = image.astype(np.float64)
        height, width = result.shape

        # Definisi offset kernel Plus (+): piksel pada arah kardinal
        # 3x3: 5 piksel, 5x5: 9 piksel, 7x7: 13 piksel
        #
        # Contoh 5x5 Plus:
        #   . . X . .
        #   . . X . .
        #   X X * X X
        #   . . X . .
        #   . . X . .
        plus_offsets = [(0, 0)]  # pusat
        for r in range(1, radius + 1):
            plus_offsets.extend([(-r, 0), (r, 0), (0, -r), (0, r)])

        # Definisi offset kernel X: piksel pada arah diagonal
        # 3x3: 5 piksel, 5x5: 9 piksel, 7x7: 13 piksel
        #
        # Contoh 5x5 X:
        #   X . . . X
        #   . X . X .
        #   . . * . .
        #   . X . X .
        #   X . . . X
        x_offsets = [(0, 0)]  # pusat
        for r in range(1, radius + 1):
            x_offsets.extend([(-r, -r), (-r, r), (r, -r), (r, r)])

        n_plus = len(plus_offsets)
        n_x = len(x_offsets)

        for _ in range(repetitions):
            # Pad citra dengan 'edge' mode (replicate boundary pixels)
            # Mereplikasi perilaku fallback try/catch cascading di Java:
            #   try { pixel[j-2*m] } catch { try { pixel[j-m] } catch { pixel[j] } }
            # yang secara efektif mengganti piksel OOB dengan piksel terdekat di tepi.
            padded = np.pad(result, radius, mode="edge")

            # Ekstrak nilai kernel Plus secara vectorized
            plus_values = np.empty((n_plus, height, width), dtype=np.float64)
            for i, (dr, dc) in enumerate(plus_offsets):
                plus_values[i] = padded[
                    radius + dr : radius + dr + height,
                    radius + dc : radius + dc + width,
                ]

            # Ekstrak nilai kernel X secara vectorized
            x_values = np.empty((n_x, height, width), dtype=np.float64)
            for i, (dr, dc) in enumerate(x_offsets):
                x_values[i] = padded[
                    radius + dr : radius + dr + height,
                    radius + dc : radius + dc + width,
                ]

            # Hitung median Plus dan median X
            median_plus = np.median(plus_values, axis=0)
            median_x = np.median(x_values, axis=0)
            center = result.copy()

            # Median dari tiga nilai: median(median_plus, median_x, center)
            # Optimasi: median(a,b,c) = max(min(a,b), min(max(a,b), c))
            # Lebih cepat daripada np.median untuk tepat 3 nilai.
            result = np.maximum(
                np.minimum(median_plus, median_x),
                np.minimum(np.maximum(median_plus, median_x), center),
            )

        # Konversi kembali ke dtype asli
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            return np.clip(result, info.min, info.max).astype(original_dtype)
        return result.astype(original_dtype)

    # ---------------------------------------------------------
    # Fast Temporal Median Filter
    # ---------------------------------------------------------

    @staticmethod
    def fast_temporal_median(
        stack: np.ndarray,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        window_size: int = 27,
        intensity_normalization: bool = False,
    ) -> np.ndarray:
        """
        Mereplikasi plugin Fast Temporal Median dari ImageJ.

        Implementasi referensi: Fast_Temporal_Median.java
        oleh Marcelo Augusto Cordeiro, Milstein Lab, University of Toronto.

        Algoritma ini menghitung running temporal median pada setiap posisi
        piksel sepanjang dimensi waktu (frame) dari image stack, kemudian
        mengurangi median tersebut dari frame asli. Berguna untuk menghilangkan
        latar belakang statis/slowly-varying pada data time-lapse microscopy.

        Plugin menggunakan Forward Window: untuk frame ke-k, jendela median
        diambil dari frame [k] sampai [k + window_size - 1].

        Jika intensity_normalization diaktifkan, nilai piksel dinormalisasi
        terhadap rata-rata intensitas frame sebelum menghitung median, lalu
        di-denormalisasi saat pengurangan. Ini mengkompensasi fluktuasi
        intensitas global (misalnya photobleaching).

        Args:
            stack: Image stack 3D (frames, height, width), uint8 atau uint16.
            start_frame: Frame awal (1-based, default: 1).
                Sesuai "Start Frame" di dialog ImageJ.
            end_frame: Frame akhir (1-based, inklusif, default: jumlah frame).
                Sesuai "End Frame" di dialog ImageJ.
            window_size: Ukuran jendela temporal (default: 27).
                Sesuai "Window Size" di dialog ImageJ. Harus >= 2.
            intensity_normalization: Jika True, normalisasi intensitas
                digunakan (default: False). Sesuai "Intensity Normalization"
                di dialog ImageJ.

        Returns:
            np.ndarray: Stack hasil filter dengan median temporal dikurangi.
                Shape: (n_output_frames, height, width).
                Jumlah frame output = end_frame - window_size - start_frame + 1.
                Tipe data sama dengan input.

        Raises:
            ValueError: Jika dimensi atau parameter tidak valid.
            TypeError: Jika tipe data input tidak sesuai.

        Note:
            Implementasi Java asli menggunakan varian algoritma Huang untuk
            update histogram inkremental secara O(1) per piksel per frame.
            Implementasi Python ini menggunakan ``np.partition`` (introselect,
            O(n) per jendela) yang memberikan hasil identik dengan performa
            yang baik berkat vektorisasi NumPy.

            Median yang digunakan adalah *true median*: elemen ke-
            ``(window_size - 1) // 2`` (0-based) dari jendela terurut.
            Ini sedikit berbeda dari implementasi Java yang mengambil elemen
            ke- ``window_size // 2 - 1`` (off-by-one), namun secara statistik
            lebih benar. Untuk window_size >= 20 (tipikal), perbedaannya
            kurang dari 1 intensity level.

        Example:
            >>> result = ImageJReplicator.fast_temporal_median(
            ...     stack, window_size=27, intensity_normalization=False
            ... )
        """
        # Validasi input
        if stack is None:
            raise ValueError("Stack input tidak boleh kosong")
        if not isinstance(stack, np.ndarray):
            raise TypeError("Input harus berupa numpy array")
        if stack.ndim != 3:
            raise ValueError(
                "Stack harus berupa 3D array dengan shape (frames, height, width)"
            )
        if stack.size == 0:
            raise ValueError("Array stack tidak boleh kosong")

        original_dtype = stack.dtype
        n_frames, height, width = stack.shape

        if end_frame is None:
            end_frame = n_frames

        # Validasi parameter (sesuai dialog Java)
        if window_size < 2 or window_size > n_frames:
            raise ValueError(
                f"window_size harus antara 2 dan {n_frames}, "
                f"diberikan: {window_size}"
            )
        if start_frame < 1 or start_frame > (n_frames - window_size + 1):
            raise ValueError(
                f"start_frame harus antara 1 dan {n_frames - window_size + 1}, "
                f"diberikan: {start_frame}"
            )
        if end_frame > n_frames or end_frame < window_size:
            raise ValueError(
                f"end_frame harus antara {window_size} dan {n_frames}, "
                f"diberikan: {end_frame}"
            )

        # Jumlah frame output
        # Java: for k=start to (end-window), inclusive (1-based)
        # Python 0-based: k dari start_frame-1 sampai end_frame-window_size-1
        n_output = end_frame - window_size - start_frame + 1
        if n_output <= 0:
            raise ValueError(
                "Tidak ada frame output yang dihasilkan. "
                "Periksa kombinasi start_frame, end_frame, dan window_size."
            )

        result = np.empty((n_output, height, width), dtype=original_dtype)

        # True median rank (0-based index dalam jendela terurut)
        # Untuk window_size=27: rank=13 → elemen ke-14 (median sejati)
        # Untuk window_size=28: rank=13 → lower-median dari dua tengah
        median_rank = (window_size - 1) // 2

        # Tentukan max value berdasarkan dtype
        if np.issubdtype(original_dtype, np.integer):
            max_val = np.iinfo(original_dtype).max
        else:
            max_val = np.finfo(original_dtype).max

        if not intensity_normalization:
            # ----------------------------------------------------------
            # Mode tanpa normalisasi intensitas
            # ----------------------------------------------------------
            for idx, k in enumerate(
                range(start_frame - 1, end_frame - window_size)
            ):
                # Ekstrak jendela temporal: frame k sampai k+window_size-1
                window = stack[k : k + window_size]

                # np.partition menempatkan elemen ke-median_rank pada posisi
                # yang benar (partial sort, O(n)), lebih cepat dari full sort
                partitioned = np.partition(window, median_rank, axis=0)
                temporal_median = partitioned[median_rank].astype(np.float64)

                # Kurangi median dari frame saat ini, clamp ke >= 0
                frame = stack[k].astype(np.float64) - temporal_median
                np.clip(frame, 0, max_val, out=frame)
                result[idx] = frame.astype(original_dtype)
        else:
            # ----------------------------------------------------------
            # Mode dengan normalisasi intensitas
            # ----------------------------------------------------------
            # Hitung rata-rata intensitas setiap frame yang digunakan
            # Java: mean[i-1] = sum(pixels) / dimension untuk frame i
            frame_means = np.mean(
                stack[start_frame - 1 : end_frame].astype(np.float64),
                axis=(1, 2),
            )

            for idx, k in enumerate(
                range(start_frame - 1, end_frame - window_size)
            ):
                window = stack[k : k + window_size].astype(np.float64)

                # Indeks ke dalam frame_means: offset dari start_frame-1
                means_offset = k - (start_frame - 1)
                w_means = frame_means[
                    means_offset : means_offset + window_size
                ]

                # Hindari pembagian dengan nol
                safe_means = np.where(w_means > 0, w_means, 1.0)

                # Normalisasi: value_norm = int(pixel / mean_frame * 1000)
                # Mereplikasi: (int)(((float)pixels[j]/(float)mean[...])*1000)
                normalized = (
                    window
                    / safe_means[:, np.newaxis, np.newaxis]
                    * 1000.0
                ).astype(np.int32)

                # Hitung median dari nilai ternormalisasi
                partitioned = np.partition(
                    normalized, median_rank, axis=0
                )
                norm_median = partitioned[median_rank].astype(np.float64)

                # De-normalisasi median dan kurangi dari frame asli
                # Java: pixels[j] -= (median[j] * ((float)mean[k-1]/(float)1000))
                current_mean = frame_means[means_offset]
                denorm_median = norm_median * (current_mean / 1000.0)

                frame = stack[k].astype(np.float64) - denorm_median
                np.clip(frame, 0, max_val, out=frame)
                result[idx] = frame.astype(original_dtype)

        return result

    # ---------------------------------------------------------
    # Median Filter (Process > Filters > Median...)
    # ---------------------------------------------------------

    @staticmethod
    def _make_circular_kernel_imagej(radius: float) -> np.ndarray:
        """
        Buat kernel lingkaran sesuai algoritma ImageJ RankFilters.java.

        ImageJ menghitung batas lingkaran menggunakan:
            r2 = int(radius * radius) + 1
            kRadius = int(sqrt(r2))
            Untuk setiap baris y: dx = int(sqrt(r2 - y*y))

        Ini memberikan bentuk lingkaran diskrit yang sedikit lebih besar
        dari lingkaran sempurna berradius 'radius', karena penambahan +1
        pada r2.

        Args:
            radius: Radius dalam piksel (bisa float, misal 5.0).

        Returns:
            np.ndarray: Boolean 2D array sebagai footprint kernel.
        """
        r2 = int(radius * radius) + 1
        k_radius = int(math.sqrt(r2 + 1e-10))
        size = 2 * k_radius + 1
        kernel = np.zeros((size, size), dtype=bool)

        for y in range(-k_radius, k_radius + 1):
            dx = int(math.sqrt(r2 - y * y + 1e-10))
            for x in range(-dx, dx + 1):
                kernel[y + k_radius, x + k_radius] = True

        return kernel

    @staticmethod
    def median_filter_imagej(
        image: np.ndarray,
        radius: float = 5.0,
    ) -> np.ndarray:
        """
        Mereplikasi filter Median dari ImageJ (Process > Filters > Median...).

        Implementasi referensi: RankFilters.java dari ImageJ.

        Filter median standar ImageJ menggunakan kernel berbentuk lingkaran
        (circular disk) yang didefinisikan oleh radius dalam piksel. Untuk
        setiap piksel, semua piksel tetangga di dalam lingkaran dikumpulkan,
        dan nilai median menggantikan piksel pusat.

        Berbeda dari `cv2.medianBlur` yang menggunakan kernel persegi,
        implementasi ini menggunakan kernel lingkaran yang identik dengan
        ImageJ, memberikan hasil yang cocok dengan output ImageJ.

        Boundary handling menggunakan mode 'nearest' (duplikat piksel tepi),
        mereplikasi PADDING_DUPLICATE dari RankFilters.java.

        Args:
            image: Citra input grayscale (uint8 atau uint16).
                Untuk citra RGB/multi-channel, setiap channel diproses
                terpisah.
            radius: Radius kernel dalam piksel (default: 5.0).
                Sesuai field "Radius" di dialog ImageJ Median.
                Nilai float diperbolehkan (misal 2.5, 5.0).

        Returns:
            np.ndarray: Citra terfilter dengan dtype yang sama dengan input.

        Raises:
            ValueError: Jika radius <= 0 atau input tidak valid.
            TypeError: Jika tipe data input tidak sesuai.

        Example:
            >>> filtered = ImageJReplicator.median_filter_imagej(
            ...     image, radius=5.0
            ... )
        """
        # Validasi input
        if image is None:
            raise ValueError("Citra input tidak boleh kosong")
        if not isinstance(image, np.ndarray):
            raise TypeError("Input harus berupa numpy array")
        if image.size == 0:
            raise ValueError("Array citra tidak boleh kosong")
        if radius <= 0:
            raise ValueError(f"Radius harus > 0, diberikan: {radius}")

        original_dtype = image.dtype

        # Handle citra multi-channel (RGB dsb.)
        if len(image.shape) == 3:
            channels = cv2.split(image)
            processed = [
                ImageJReplicator.median_filter_imagej(ch, radius)
                for ch in channels
            ]
            return cv2.merge(processed)

        # Buat kernel lingkaran sesuai ImageJ
        footprint = ImageJReplicator._make_circular_kernel_imagej(radius)

        # Terapkan median filter dengan kernel lingkaran
        # mode='nearest' mereplikasi PADDING_DUPLICATE dari ImageJ
        result = median_filter(
            image, footprint=footprint, mode='nearest'
        )

        return result.astype(original_dtype)
