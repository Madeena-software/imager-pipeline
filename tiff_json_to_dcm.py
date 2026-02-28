import os
import json
import uuid
import datetime
import cv2

try:
    import pydicom
except ImportError as exc:
    raise ImportError(
        "pydicom is not installed. Install it with: pip install pydicom"
    ) from exc
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian


def tiff_json_to_dcm(tiff_path, json_path, output_path):
    # Load image
    image = cv2.imread(tiff_path, -1)
    if image is None:
        raise ValueError(f"Could not read TIFF image: {tiff_path}")
    imHeight, imWidth = image.shape[:2]
    pixel_bytes = image.tobytes()

    # Load metadata
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Generate UIDs
    instanceUID = str(uuid.uuid4()).replace("-", "")
    studyUID = str(uuid.uuid4()).replace("-", "")
    seriesUID = str(uuid.uuid4()).replace("-", "")
    requestUID = str(uuid.uuid4()).replace("-", "")

    # File meta info
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.1.1.1")
    file_meta.MediaStorageSOPInstanceUID = UID(instanceUID)
    file_meta.ImplementationClassUID = UID("1.2.826.0.1.3680043.10.1356.2.1.0.1")
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationVersionName = "TIFF2DCM_1.0.1"

    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "DX"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = studyUID
    ds.SeriesInstanceUID = seriesUID
    ds.SecondaryCaptureDeviceManufacturer = "Python"
    ds.StudyDescription = meta.get("StudyDescription", "Study")
    ds.SeriesDescription = meta.get("SeriesDescription", "Series")
    ds.InstitutionName = "PT. Madeena Karya Indonesia"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    # Convert PixelSpacing from micrometers to millimeters if needed
    scale_x = meta.get("Scale X", 1.0)
    scale_y = meta.get("Scale Y", 1.0)
    # Assume input is in micrometers, convert to millimeters
    ds.PixelSpacing = [float(scale_x) / 1000.0, float(scale_y) / 1000.0]
    ds.PlanarConfiguration = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = 0
    ds.LargestImagePixelValue = 65535
    ds.WindowCenter = 32768
    ds.WindowWidth = 65536
    ds.Columns = imWidth
    ds.Rows = imHeight
    ds.NumberOfFrames = 1
    ds.PatientName = meta.get("Patient Name", "Unknown")
    ds.PatientID = meta.get("NIK", "Unknown")
    gender = meta.get("Gender", "").lower()
    if gender == "male":
        ds.PatientSex = "M"
    elif gender == "female":
        ds.PatientSex = "F"
    else:
        ds.PatientSex = "O"
    ds.PatientBirthDate = meta.get("Birthdate", "")
    ds.AccessionNumber = requestUID
    # Use Time from JSON if available (YYMMDDhhmmss)
    time_str = meta.get("Time", "").strip()
    if len(time_str) >= 12:
        # Parse as YYMMDDhhmmss
        year = int(time_str[0:2])
        # Assume 2000+ for years < 70, else 1900+
        year += 2000 if year < 70 else 1900
        month = int(time_str[2:4])
        day = int(time_str[4:6])
        hour = int(time_str[6:8])
        minute = int(time_str[8:10])
        second = int(time_str[10:12])
        dt = datetime.datetime(year, month, day, hour, minute, second)
        ds.StudyDate = ds.ContentDate = dt.strftime("%Y%m%d")
        ds.StudyTime = ds.ContentTime = dt.strftime("%H%M%S")
    else:
        dt = datetime.datetime.now()
        ds.StudyDate = ds.ContentDate = dt.strftime("%Y%m%d")
        ds.StudyTime = ds.ContentTime = dt.strftime("%H%M%S")
    ds.PixelData = pixel_bytes
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(output_path, write_like_original=False)
    print(f"DICOM file created: {output_path}")


if __name__ == "__main__":
    tiff_path = r"E:\test mini xray\preprocessing\output-test\Image_20260227152717925_trx_raw_processed.tiff"
    json_path = r"E:\test mini xray\preprocessing\dicom\01-GBS_Thorax_PA 70kV8mA0,50s -27_2_2025-9.48 AM [Administrator]_processed.json"
    output_path = r"E:\test mini xray\preprocessing\dicom\01-GBS_Thorax_PA 70kV8mA0,50s -27_2_2025-15.58_processed.dcm"

    if not os.path.exists(tiff_path):
        raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    tiff_json_to_dcm(tiff_path, json_path, output_path)
