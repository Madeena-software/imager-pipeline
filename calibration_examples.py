#!/usr/bin/env python3
"""
Camera Calibration Usage Examples for X-ray Image Processing Pipeline

This file demonstrates how to use the camera calibration feature to correct
fish-eye distortion in X-ray images using the configuration-based approach.

The calibration process consists of two main steps:
1. Generate calibration parameters from a calibration image (circle grid pattern)
2. Apply the calibration during image processing

Requirements:
- Calibration image with circle grid pattern (TIFF format)
- Circle grid can be black circles on white background or vice versa
- The pattern should be detectable by OpenCV's findCirclesGrid function
"""

import os
from pathlib import Path


def example_1_generate_calibration():
    """
    Example 1: Generate calibration parameters from a calibration image
    
    This creates an NPZ file containing camera calibration parameters
    that can be used to correct distortion in all subsequent images.
    """
    print("=== Example 1: Generate Calibration Parameters ===")
    
    # Step 1: Update your .env file with calibration image path
    print("1. Update your .env file with calibration settings:")
    
    env_settings = '''
# Camera Calibration Generation parameters
CALIBRATION_IMAGE_PATH=path/to/your/calibration_image.tiff
CALIBRATION_OUTPUT_NPZ=camera_calibration.npz
CALIBRATION_PATTERN_COLS=44
CALIBRATION_PATTERN_ROWS=35
CALIBRATION_CIRCLE_DIAMETER=1.0
CALIBRATION_TEST_ENABLED=True
'''
    
    print(env_settings)
    print("   Adjust CALIBRATION_PATTERN_COLS and CALIBRATION_PATTERN_ROWS to match your grid")
    print("   Set CALIBRATION_IMAGE_PATH to your actual calibration image")
    
    # Step 2: Run calibration
    print("\\n2. Run calibration:")
    print("   python camera_calibration.py")
    
    print("\\n3. The script will:")
    print("   - Load your calibration image")
    print("   - Detect the circle grid pattern")
    print("   - Generate calibration parameters")
    print("   - Save to NPZ file")
    print("   - Test the calibration (if enabled)")
    
    print("\\n✓ If successful, you'll have a calibration file ready to use!")
    
    return True
        

def example_2_use_calibration_in_pipeline():
    """
    Example 2: Enable calibration in the complete pipeline
    
    This shows how to configure the .env file and use calibration
    in your X-ray image processing pipeline.
    """
    print("\\n=== Example 2: Use Calibration in Complete Pipeline ===")
    
    # Step 1: Generate calibration file (from Example 1)
    calibration_npz = "camera_calibration.npz"
    
    if not os.path.exists(calibration_npz):
        print(f"Please generate calibration file first: {calibration_npz}")
        print("Run example_1_generate_calibration() first")
        return
    
    # Step 2: Update your .env file
    env_content = '''
# Camera Calibration parameters
# Enable fish-eye distortion correction using pre-computed NPZ calibration file
USE_CALIBRATION=True
# Path to NPZ file containing camera calibration parameters (mtx, dist, roi, etc.)
CALIBRATION_NPZ_PATH=camera_calibration.npz
'''
    
    print("Add these lines to your .env file:")
    print(env_content)
    
    # Step 3: Run your normal pipeline
    print("\\nThen run your complete pipeline normally:")
    print("python complete_pipeline.py")
    print("")
    print("The pipeline will now include calibration as step 2.5:")
    print("  [1/10] Loading images...")
    print("  [2/10] Cropping and rotating...")
    print("  [2.5/10] Applying camera calibration (fish-eye correction)...")
    print("  [3/10] Denoising images...")
    print("  [4/10] Applying Flat-Field Correction...")
    print("  ... (rest of pipeline)")


def example_3_standalone_calibration():
    """
    Example 3: Use calibration as a standalone function
    
    This shows how to apply calibration to individual images
    without running the complete pipeline.
    """
    print("\\n=== Example 3: Standalone Calibration ===\\n")
    
    try:
        from camera_calibration import undistort_image
    except ImportError:
        print("Error: camera_calibration module not found")
        return
    
    calibration_npz = "camera_calibration.npz"
    input_image = "path/to/your/xray_image.tiff"
    output_image = "corrected_image.tiff"
    
    if not os.path.exists(calibration_npz):
        print(f"Please generate calibration file first: {calibration_npz}")
        return
        
    if not os.path.exists(input_image):
        print(f"Please provide a valid input image: {input_image}")
        return
    
    try:
        # Apply calibration to single image
        corrected_image = undistort_image(input_image, calibration_npz)
        
        # Save result
        import cv2
        cv2.imwrite(output_image, corrected_image)
        
        print(f"✓ Calibration applied successfully!")
        print(f"  Input: {input_image}")
        print(f"  Output: {output_image}")
        print(f"  Calibration: {calibration_npz}")
        
    except Exception as e:
        print(f"✗ Calibration failed: {e}")


def configuration_usage():
    """
    Example 4: Configuration-based usage of camera_calibration.py
    """
    print("\\n=== Example 4: Configuration-Based Usage ===\\n")
    
    print("Setup calibration (edit .env file):") 
    print("")
    print("# Add these lines to your .env file:")
    print("CALIBRATION_IMAGE_PATH=calibration_image.tiff")
    print("CALIBRATION_OUTPUT_NPZ=camera_calibration.npz")
    print("CALIBRATION_PATTERN_COLS=44")
    print("CALIBRATION_PATTERN_ROWS=35")
    print("CALIBRATION_CIRCLE_DIAMETER=1.0")
    print("CALIBRATION_TEST_ENABLED=True")
    print("")
    
    print("For custom ROI (optional):")
    print("CALIBRATION_CUSTOM_ROI_X=100")
    print("CALIBRATION_CUSTOM_ROI_Y=100")
    print("CALIBRATION_CUSTOM_ROI_W=3000") 
    print("CALIBRATION_CUSTOM_ROI_H=2000")
    print("")
    
    print("Run calibration:")
    print("python camera_calibration.py")
    print("")
    
    print("The script will:")
    print("- Read configuration from .env file")
    print("- Load and process your calibration image")
    print("- Generate NPZ calibration file")
    print("- Test calibration if enabled")
    print("- Provide next steps for using in pipeline")


def troubleshooting_tips():
    """
    Common issues and solutions when using camera calibration
    """
    print("\\n=== Troubleshooting Tips ===\\n")
    
    tips = [
        "1. Circle Detection Issues:",
        "   - Make sure your calibration image has good contrast",
        "   - Try both black circles on white and white circles on black",
        "   - Adjust CALIBRATION_PATTERN_COLS/ROWS to match your actual grid",
        "   - Ensure circles are clearly visible and not distorted",
        "",
        "2. Calibration Quality:",
        "   - Use a high-resolution calibration image",
        "   - Ensure the pattern covers most of the image area",
        "   - Avoid shadows or uneven lighting",
        "   - The pattern should be as flat as possible",
        "",
        "3. Integration Issues:",
        "   - Verify CALIBRATION_NPZ_PATH is correct in .env",
        "   - Make sure the NPZ file exists and is readable",
        "   - Check that USE_CALIBRATION=True in .env",
        "   - Ensure camera_calibration.py is in the same directory",
        "",
        "4. Performance Considerations:",
        "   - Calibration adds processing time to each image",
        "   - The calibrated image may be smaller than the original",
        "   - ROI cropping removes distorted edge areas",
        "   - Consider calibrating once and reusing the NPZ file",
        "",
        "5. Configuration Issues:",
        "   - Make sure .env file is in the same directory as the Python files",
        "   - Check that all configuration values are properly set",
        "   - Verify image paths are absolute or relative to the script location",
        "   - Ensure pattern size matches your actual calibration grid"
    ]
    
    for tip in tips:
        print(tip)


if __name__ == "__main__":
    print("Camera Calibration Usage Examples")
    print("===================================\\n")
    
    # Run examples (commented out - uncomment to test)
    # example_1_generate_calibration()
    # example_2_use_calibration_in_pipeline()
    # example_3_standalone_calibration()
    
    # Show usage information
    configuration_usage()
    troubleshooting_tips()
    
    print("\\n=== Next Steps ===\\n")
    print("1. Prepare your calibration image (circle grid pattern)")
    print("2. Update .env file with calibration settings")
    print("3. Run python camera_calibration.py to create NPZ file")
    print("4. Run your complete pipeline with calibration enabled")
    print("\\nFor more help, see the comments in this file")


def example_1_generate_calibration():
    """
    Example 1: Generate calibration parameters from a calibration image
    
    This creates an NPZ file containing camera calibration parameters
    that can be used to correct distortion in all subsequent images.
    """
    print("=== Example 1: Generate Calibration Parameters ===")
    
    # Step 1: Update your .env file with calibration image path
    print("1. Update your .env file with calibration settings:")
    
    env_settings = '''
# Camera Calibration Generation parameters
CALIBRATION_IMAGE_PATH=path/to/your/calibration_image.tiff
CALIBRATION_OUTPUT_NPZ=camera_calibration.npz
CALIBRATION_PATTERN_COLS=44
CALIBRATION_PATTERN_ROWS=35
CALIBRATION_CIRCLE_DIAMETER=1.0
CALIBRATION_TEST_ENABLED=True
'''
    
    print(env_settings)
    print("   Adjust CALIBRATION_PATTERN_COLS and CALIBRATION_PATTERN_ROWS to match your grid")
    print("   Set CALIBRATION_IMAGE_PATH to your actual calibration image")
    
    # Step 2: Run calibration
    print("\\n2. Run calibration:")
    print("   python camera_calibration.py")
    
    print("\\n3. The script will:")
    print("   - Load your calibration image")
    print("   - Detect the circle grid pattern")
    print("   - Generate calibration parameters")
    print("   - Save to NPZ file")
    print("   - Test the calibration (if enabled)")
    
    print("\\n✓ If successful, you'll have a calibration file ready to use!")
    
    return True
        

def example_2_use_calibration_in_pipeline():
    """
    Example 2: Enable calibration in the complete pipeline
    
    This shows how to configure the .env file and use calibration
    in your X-ray image processing pipeline.
    """
    print("\n=== Example 2: Use Calibration in Complete Pipeline ===")
    
    # Step 1: Generate calibration file (from Example 1)
    calibration_npz = "camera_calibration.npz"
    
    if not os.path.exists(calibration_npz):
        print(f"Please generate calibration file first: {calibration_npz}")
        print("Run example_1_generate_calibration() first")
        return
    
    # Step 2: Update your .env file
    env_content = \"\"\"\n# Camera Calibration parameters\n# Enable fish-eye distortion correction using pre-computed NPZ calibration file\nUSE_CALIBRATION=True\n# Path to NPZ file containing camera calibration parameters (mtx, dist, roi, etc.)\nCALIBRATION_NPZ_PATH=camera_calibration.npz\n\"\"\"\n    \n    print(\"Add these lines to your .env file:\")\n    print(env_content)\n    \n    # Step 3: Run your normal pipeline\n    print(\"\\nThen run your complete pipeline normally:\")\n    print(\"python complete_pipeline.py\")\n    print(\"\")\n    print(\"The pipeline will now include calibration as step 2.5:\")\n    print(\"  [1/10] Loading images...\")\n    print(\"  [2/10] Cropping and rotating...\")\n    print(\"  [2.5/10] Applying camera calibration (fish-eye correction)...\")\n    print(\"  [3/10] Denoising images...\")\n    print(\"  [4/10] Applying Flat-Field Correction...\")\n    print(\"  ... (rest of pipeline)\")\n\n\ndef example_3_standalone_calibration():    \"\"\"\n    Example 3: Use calibration as a standalone function\n    \n    This shows how to apply calibration to individual images\n    without running the complete pipeline.\n    \"\"\"\n    print(\"\\n=== Example 3: Standalone Calibration ===\\n\")\n    \n    calibration_npz = \"camera_calibration.npz\"\n    input_image = \"path/to/your/xray_image.tiff\"\n    output_image = \"corrected_image.tiff\"\n    \n    if not os.path.exists(calibration_npz):\n        print(f\"Please generate calibration file first: {calibration_npz}\")\n        return\n        \n    if not os.path.exists(input_image):\n        print(f\"Please provide a valid input image: {input_image}\")\n        return\n    \n    try:\n        # Apply calibration to single image\n        corrected_image = undistort_image(input_image, calibration_npz)\n        \n        # Save result\n        import cv2\n        cv2.imwrite(output_image, corrected_image)\n        \n        print(f\"✓ Calibration applied successfully!\")\n        print(f\"  Input: {input_image}\")\n        print(f\"  Output: {output_image}\")\n        print(f\"  Calibration: {calibration_npz}\")\n        \n    except Exception as e:\n        print(f\"✗ Calibration failed: {e}\")\n\n\ndef command_line_usage():\n    \"\"\"\n    Example 4: Command line usage of camera_calibration.py\n    \"\"\"\n    print(\"\\n=== Example 4: Command Line Usage ===\\n\")\n    \n    print(\"Generate calibration parameters:\")\n    print(\"python camera_calibration.py calibration_image.tiff camera_calibration.npz\")\n    print(\"\")\n    \n    print(\"With custom pattern size:\")\n    print(\"python camera_calibration.py calibration_image.tiff camera_calibration.npz --pattern-size 30 20\")\n    print(\"\")\n    \n    print(\"With custom ROI:\")\n    print(\"python camera_calibration.py calibration_image.tiff camera_calibration.npz --roi 100 100 3000 2000\")\n    print(\"\")\n    \n    print(\"Generate and test calibration:\")\n    print(\"python camera_calibration.py calibration_image.tiff camera_calibration.npz --test --test-output test_result.tiff\")\n\n\ndef troubleshooting_tips():\n    \"\"\"\n    Common issues and solutions when using camera calibration\n    \"\"\"\n    print(\"\\n=== Troubleshooting Tips ===\\n\")\n    \n    tips = [\n        \"1. Circle Detection Issues:\",\n        \"   - Make sure your calibration image has good contrast\",\n        \"   - Try both black circles on white and white circles on black\",\n        \"   - Adjust pattern_size to match your actual grid\",\n        \"   - Ensure circles are clearly visible and not distorted\",\n        \"\",\n        \"2. Calibration Quality:\",\n        \"   - Use a high-resolution calibration image\",\n        \"   - Ensure the pattern covers most of the image area\",\n        \"   - Avoid shadows or uneven lighting\",\n        \"   - The pattern should be as flat as possible\",\n        \"\",\n        \"3. Integration Issues:\",\n        \"   - Verify CALIBRATION_NPZ_PATH is correct in .env\",\n        \"   - Make sure the NPZ file exists and is readable\",\n        \"   - Check that USE_CALIBRATION=True in .env\",\n        \"   - Ensure camera_calibration.py is in the same directory\",\n        \"\",\n        \"4. Performance Considerations:\",\n        \"   - Calibration adds processing time to each image\",\n        \"   - The calibrated image may be smaller than the original\",\n        \"   - ROI cropping removes distorted edge areas\",\n        \"   - Consider calibrating once and reusing the NPZ file\"\n    ]\n    \n    for tip in tips:\n        print(tip)\n\n\nif __name__ == \"__main__\":\n    print(\"Camera Calibration Usage Examples\")\n    print(\"===================================\\n\")\n    \n    # Run examples (commented out - uncomment to test)\n    # example_1_generate_calibration()\n    # example_2_use_calibration_in_pipeline()\n    # example_3_standalone_calibration()\n    \n    # Show usage information\n    command_line_usage()\n    troubleshooting_tips()\n    \n    print(\"\\n=== Next Steps ===\\n\")\n    print(\"1. Prepare your calibration image (circle grid pattern)\")\n    print(\"2. Run example_1_generate_calibration() to create NPZ file\")\n    print(\"3. Update your .env file with calibration settings\")\n    print(\"4. Run your complete pipeline with calibration enabled\")\n    print(\"\\nFor more help, see the comments in this file or run:\")\n    print(\"python camera_calibration.py --help\")\n