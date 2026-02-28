#!/usr/bin/env python3
"""
Camera Calibration Module for Fish-eye Distortion Correction

This module provides functionality to calibrate camera distortion using a circle grid pattern
and save the calibration parameters to an NPZ file for later use in image correction.

Supports both black circles on white background and white circles on black background.
Configuration is loaded from .env file, similar to complete_pipeline.py.
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os


# Load environment variables from .env file
def load_calibration_config():
    """Load calibration configuration from .env file."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    config = {
        # Default values
        "CALIBRATION_IMAGE_PATH": "",
        "CALIBRATION_OUTPUT_NPZ": "camera_calibration.npz",
        "CALIBRATION_PATTERN_COLS": 44,
        "CALIBRATION_PATTERN_ROWS": 35,
        "CALIBRATION_CIRCLE_DIAMETER": 1.0,
        "CALIBRATION_CUSTOM_ROI_X": None,
        "CALIBRATION_CUSTOM_ROI_Y": None,
        "CALIBRATION_CUSTOM_ROI_W": None,
        "CALIBRATION_CUSTOM_ROI_H": None,
        "CALIBRATION_TEST_ENABLED": True,
        "CALIBRATION_TEST_OUTPUT": "",
    }

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    if key in config:
                        # Convert to appropriate type
                        if key in [
                            "CALIBRATION_PATTERN_COLS",
                            "CALIBRATION_PATTERN_ROWS",
                            "CALIBRATION_CUSTOM_ROI_X",
                            "CALIBRATION_CUSTOM_ROI_Y",
                            "CALIBRATION_CUSTOM_ROI_W",
                            "CALIBRATION_CUSTOM_ROI_H",
                        ]:
                            try:
                                config[key] = int(value) if value else None
                            except ValueError:
                                pass
                        elif key == "CALIBRATION_CIRCLE_DIAMETER":
                            try:
                                config[key] = float(value) if value else 1.0
                            except ValueError:
                                pass
                        elif key == "CALIBRATION_TEST_ENABLED":
                            config[key] = value.lower() in ["true", "yes", "1"]
                        else:
                            config[key] = value

    return config


# Global config loaded once
CALIBRATION_CONFIG = load_calibration_config()


class CameraCalibrator:
    """Camera calibration class for fish-eye distortion correction."""

    CIRCLE_DIAMETER_TOLERANCE = 0.10

    def __init__(
        self,
        pattern_size=(44, 35),
        circle_diameter=1.0,
    ):
        """
        Initialize the calibrator.

        Args:
            pattern_size: Tuple (cols, rows) of circles in the calibration pattern
            circle_diameter: Real-world diameter of each circle (in mm or any unit)
        """
        self.pattern_size = pattern_size
        self.circle_diameter = circle_diameter
        self.objp = self._create_object_points()
        self.blob_detectors = self._create_blob_detectors()

    def _try_find_grid(self, img, label):
        """Try ordered circle-grid extraction on one image."""
        attempts = [
            (cv2.CALIB_CB_SYMMETRIC_GRID, "symmetric"),
            (
                cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                "symmetric+cluster",
            ),
            (cv2.CALIB_CB_ASYMMETRIC_GRID, "asymmetric"),
            (
                cv2.CALIB_CB_ASYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING,
                "asymmetric+cluster",
            ),
        ]

        for detector, detector_name in self.blob_detectors:
            for flags, grid_name in attempts:
                ret, centers = cv2.findCirclesGrid(
                    img,
                    self.pattern_size,
                    flags=flags,
                    blobDetector=detector,
                )
                if ret:
                    print(
                        f"✓ Circles detected with {grid_name} grid ({label}, detector={detector_name})"
                    )
                    return True, centers

        return False, None

    def _build_blob_detector(self):
        """Build blob detector using circle diameter with ±10% tolerance."""
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 5
        params.maxThreshold = 255
        params.thresholdStep = 5

        effective_circle_px = (
            self.circle_diameter
            if self.circle_diameter and self.circle_diameter >= 6
            else 40.0
        )

        if effective_circle_px > 0:
            min_circle_px = effective_circle_px * (1.0 - self.CIRCLE_DIAMETER_TOLERANCE)
            max_circle_px = effective_circle_px * (1.0 + self.CIRCLE_DIAMETER_TOLERANCE)
            params.minArea = max(8, np.pi * (min_circle_px * 0.5) ** 2)
            params.maxArea = max(9, np.pi * (max_circle_px * 0.5) ** 2)
            params.maxArea = max(params.maxArea, params.minArea + 1.0)
            params.minDistBetweenBlobs = max(2.0, effective_circle_px * 0.4)
        else:
            params.minArea = 8
            params.maxArea = 5000

        params.filterByArea = True

        params.filterByCircularity = True
        params.minCircularity = 0.35

        params.filterByInertia = False

        params.filterByConvexity = False

        params.filterByColor = False

        return cv2.SimpleBlobDetector_create(params)

    def _create_blob_detectors(self):
        """Create detector list."""
        return [(self._build_blob_detector(), "circle-diameter±10%")]

    def _try_detect_on_image(self, img, label):
        """Try circle-grid detection with both symmetric and asymmetric flags."""
        scales = [0.5, 0.75, 1.0]
        h, w = img.shape[:2]
        for scale in scales:
            if scale == 1.0:
                test_img = img
            else:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                test_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            ret, centers = self._try_find_grid(test_img, f"{label}, scale={scale:.2f}")
            if ret:
                if scale != 1.0:
                    centers = centers / scale
                return True, centers

        return False, None

    def _try_hough_guided_detection(self, img, label):
        """Use Hough circles to create a synthetic clean circle image, then run findCirclesGrid."""
        expected_d = (
            self.circle_diameter
            if self.circle_diameter and self.circle_diameter >= 6
            else 40.0
        )
        min_d = expected_d * (1.0 - self.CIRCLE_DIAMETER_TOLERANCE)
        max_d = expected_d * (1.0 + self.CIRCLE_DIAMETER_TOLERANCE)

        scales = [0.5, 0.75, 1.0]
        h, w = img.shape[:2]

        for scale in scales:
            if scale == 1.0:
                test_img = img
            else:
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                test_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            blur = cv2.GaussianBlur(test_img, (9, 9), 1.5)
            min_r = max(2, int((min_d * 0.5) * scale))
            max_r = max(min_r + 1, int((max_d * 0.5) * scale))
            min_dist = max(8.0, min_d * scale)

            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=120,
                param2=12,
                minRadius=min_r,
                maxRadius=max_r,
            )

            if circles is None:
                continue

            circles = np.round(circles[0]).astype(np.int32)
            expected_points = self.pattern_size[0] * self.pattern_size[1]
            if len(circles) < int(expected_points * 0.5):
                continue

            synthetic = np.zeros_like(test_img, dtype=np.uint8)
            for x, y, r in circles:
                draw_r = max(2, int(max(2, r) * 0.7))
                cv2.circle(synthetic, (x, y), draw_r, 255, -1)

            synthetic = cv2.GaussianBlur(synthetic, (5, 5), 0)

            ret, centers = self._try_find_grid(
                synthetic,
                f"hough-guided/{label}, scale={scale:.2f}, circles={len(circles)}",
            )
            if ret:
                if scale != 1.0:
                    centers = centers / scale
                return True, centers

        return False, None

    def _build_preprocessed_variants(self, img):
        """Build robust preprocessing variants for noisy circle-grid images."""
        variants = [(img, "original")]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img)
        variants.append((clahe_img, "clahe"))

        blur = cv2.GaussianBlur(clahe_img, (5, 5), 0)
        variants.append((blur, "clahe+gaussian"))

        return variants

    def _create_object_points(self):
        """Create 3D object points for the circle grid pattern."""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        objp *= self.circle_diameter  # Scale by actual circle spacing
        return objp

    def detect_circles(self, image_path, invert_if_needed=True):
        """
        Detect circle grid in calibration image.

        Args:
            image_path: Path to calibration image
            invert_if_needed: Try both normal and inverted image if first attempt fails

        Returns:
            Tuple (success, centers) where success is bool and centers are detected points
        """
        # Load image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = image_path

        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return False, None

        print(f"Image size: {img.shape}")
        print(
            f"Looking for {self.pattern_size[0]}x{self.pattern_size[1]} circle grid..."
        )

        # Fixed sequence: fastest to slowest
        # 1) Direct grid detection (normal image variants)
        # 2) Direct grid detection (inverted variants)
        # 3) Hough-guided grid detection (normal variants)
        # 4) Hough-guided grid detection (inverted variants)
        variants = self._build_preprocessed_variants(img)
        inv_variants = []
        if invert_if_needed:
            img_inv = cv2.bitwise_not(img)
            inv_variants = self._build_preprocessed_variants(img_inv)

        for variant_img, label in variants:
            ret, centers = self._try_detect_on_image(variant_img, label)
            if ret:
                return True, centers

        if invert_if_needed:
            print("Trying inverted image variants...")
            for variant_img, label in inv_variants:
                ret, centers = self._try_detect_on_image(
                    variant_img, f"inverted/{label}"
                )
                if ret:
                    return True, centers

        print("Trying Hough-circle guided detection...")
        for variant_img, label in variants:
            ret, centers = self._try_hough_guided_detection(variant_img, label)
            if ret:
                return True, centers

        if invert_if_needed:
            for variant_img, label in inv_variants:
                ret, centers = self._try_hough_guided_detection(
                    variant_img, f"inverted/{label}"
                )
                if ret:
                    return True, centers

        print("✗ No circles detected")
        return False, None

    def calibrate_from_image(self, image_path, output_npz_path, roi_crop=None):
        """
        Perform camera calibration from a single calibration image.

        Args:
            image_path: Path to calibration image with circle grid pattern
            output_npz_path: Path to save calibration parameters (NPZ format)
            roi_crop: Optional ROI as tuple (x, y, w, h) to crop the corrected image

        Returns:
            bool: True if calibration successful, False otherwise
        """
        print(f"Starting camera calibration from: {image_path}")

        # Detect circles
        ret, centers = self.detect_circles(image_path)
        if not ret:
            print("Error: Could not detect circle pattern in calibration image")
            return False

        # Load image to get dimensions
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_size = img.shape[::-1]  # (width, height)

        # Prepare object and image points
        objpoints = [self.objp]  # 3D points in real world
        imgpoints = [centers]  # 2D points in image plane

        print("Performing camera calibration...")

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )

        if not ret:
            print("Error: Camera calibration failed")
            return False

        print("✓ Camera calibration successful")
        print(f"Camera matrix:\n{mtx}")
        print(f"Distortion coefficients: {dist.flatten()}")

        # Calculate optimal camera matrix and ROI
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Use custom ROI if provided, otherwise use calculated ROI
        final_roi = roi_crop if roi_crop is not None else roi

        print(f"Optimal new camera matrix:\n{newcameramtx}")
        print(f"ROI for cropping: {final_roi}")

        # Save calibration parameters
        np.savez(
            output_npz_path,
            mtx=mtx,
            dist=dist,
            rvecs=rvecs,
            tvecs=tvecs,
            roi=final_roi,
            newcameramtx=newcameramtx,
            pattern_size=self.pattern_size,
            circle_diameter=self.circle_diameter,
            image_size=img_size,
        )

        print(f"✓ Calibration parameters saved to: {output_npz_path}")

        # Verify file was created
        if Path(output_npz_path).exists():
            return True
        else:
            print("Error: Failed to save calibration file")
            return False

    def test_calibration(self, image_path, npz_path, output_test_path=None):
        """
        Test the calibration by undistorting the calibration image.

        Args:
            image_path: Path to original calibration image
            npz_path: Path to calibration NPZ file
            output_test_path: Optional path to save undistorted test image

        Returns:
            numpy.ndarray: Undistorted image
        """
        print(f"Testing calibration on: {image_path}")

        # Load calibration parameters
        with np.load(npz_path) as params:
            mtx = params["mtx"]
            dist = params["dist"]
            roi = params["roi"]
            newcameramtx = params.get("newcameramtx", None)

        # Load test image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Could not load test image from {image_path}")
            return None

        # Undistort image
        if newcameramtx is not None:
            undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        else:
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Crop to ROI
        x, y, w, h = roi
        cropped = undistorted[y : y + h, x : x + w]

        print(f"Original size: {img.shape}, Undistorted size: {cropped.shape}")

        # Save test result if requested
        if output_test_path:
            cv2.imwrite(output_test_path, cropped)
            print(f"✓ Test result saved to: {output_test_path}")

        return cropped


def undistort_image(image, npz_path):
    """
    Undistort an image using saved calibration parameters.

    Args:
        image: Input image as numpy array or path to image file
        npz_path: Path to calibration NPZ file

    Returns:
        numpy.ndarray: Undistorted and cropped image
    """
    # Load calibration parameters
    with np.load(npz_path) as params:
        mtx = params["mtx"]
        dist = params["dist"]
        roi = params["roi"]
        newcameramtx = params.get("newcameramtx", None)

    # Load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    else:
        img = image.copy()

    if img is None:
        raise ValueError(f"Could not load image")

    # Get new camera matrix if not saved
    if newcameramtx is None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Crop to ROI
    x, y, w, h = roi
    cropped = undistorted[y : y + h, x : x + w]

    return cropped


def main():
    """Main calibration function using configuration from .env file."""
    print("=" * 70)
    print("CAMERA CALIBRATION FOR FISH-EYE CORRECTION")
    print("=" * 70)

    # Load configuration
    config = CALIBRATION_CONFIG

    # Check required parameters
    calibration_image = config["CALIBRATION_IMAGE_PATH"]
    if not calibration_image:
        print("Error: CALIBRATION_IMAGE_PATH not set in .env file")
        print("Please set the path to your calibration image with circle grid pattern")
        print("Example: CALIBRATION_IMAGE_PATH=path/to/calibration_image.tiff")
        return False

    if not Path(calibration_image).exists():
        print(f"Error: Calibration image not found: {calibration_image}")
        return False

    output_npz = config["CALIBRATION_OUTPUT_NPZ"]

    # Print configuration
    print(f"Configuration:")
    print(f"  Calibration image: {calibration_image}")
    print(f"  Output NPZ file:   {output_npz}")
    print(
        f"  Pattern size:      {config['CALIBRATION_PATTERN_COLS']}x{config['CALIBRATION_PATTERN_ROWS']} circles"
    )
    print(f"  Circle diameter:   {config['CALIBRATION_CIRCLE_DIAMETER']}")
    print(
        f"  Circle tolerance:  ±{int(CameraCalibrator.CIRCLE_DIAMETER_TOLERANCE * 100)}%"
    )

    # Build custom ROI if all values are provided
    custom_roi = None
    roi_values = [
        config["CALIBRATION_CUSTOM_ROI_X"],
        config["CALIBRATION_CUSTOM_ROI_Y"],
        config["CALIBRATION_CUSTOM_ROI_W"],
        config["CALIBRATION_CUSTOM_ROI_H"],
    ]
    if all(v is not None for v in roi_values):
        custom_roi = tuple(roi_values)
        print(f"  Custom ROI:        {custom_roi}")
    else:
        print(f"  Custom ROI:        Auto-calculated")

    print(f"  Test enabled:      {config['CALIBRATION_TEST_ENABLED']}")
    print("=" * 70)

    # Create calibrator
    calibrator = CameraCalibrator(
        pattern_size=(
            config["CALIBRATION_PATTERN_COLS"],
            config["CALIBRATION_PATTERN_ROWS"],
        ),
        circle_diameter=config["CALIBRATION_CIRCLE_DIAMETER"],
    )

    # Perform calibration
    success = calibrator.calibrate_from_image(
        calibration_image, output_npz, roi_crop=custom_roi
    )

    if not success:
        print("\nCalibration failed!")
        return False

    print("\nCalibration completed successfully!")

    # Test calibration if enabled
    if config["CALIBRATION_TEST_ENABLED"]:
        test_output = config["CALIBRATION_TEST_OUTPUT"]
        if not test_output:
            # Auto-generate test output filename
            base_name = Path(output_npz).stem
            test_output = f"{base_name}_test_undistorted.tiff"

        print(f"\nTesting calibration...")
        result = calibrator.test_calibration(calibration_image, output_npz, test_output)

        if result is not None:
            print("\n✓ Calibration test completed successfully!")
        else:
            print("\n✗ Calibration test failed!")

    print("\nNext steps:")
    print("1. Update your .env file:")
    print(f"   USE_CALIBRATION=True")
    print(f"   CALIBRATION_NPZ_PATH={output_npz}")
    print("2. Run your complete pipeline:")
    print("   python complete_pipeline.py")
    print("\nThe pipeline will now include fish-eye correction as step 2.5")

    return True


if __name__ == "__main__":
    main()
