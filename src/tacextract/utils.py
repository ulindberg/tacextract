"""Utility functions for dynamic PET image processing."""

from pathlib import Path
import numpy as np
from pydicom import dcmread
from pydicom.misc import is_dicom

### DICOM Functions ###
def create_filelist_from_series_path(input_path: Path, strict: bool = False)->list:
    """List all DICOM files in directory.

    Args:
        input_path: Aboslute path to directory containing DICOM files (.ima/.dcm).
        strict: Determine if each file is validated as being a DICOM file (default: False).

    Returns:
        filelist: List containing paths to all DICOM files in input_path.
    """

    assert input_path.is_dir()

    if strict:
        filelist  = [s for s in input_path.iterdir() if is_dicom(s)] # Slower but more general
    else:
        # Assuming file ends on .ima/.dcm
        filelist  = [s for s in input_path.iterdir() if s.name.endswith(('.dcm','.ima'))] # Fast
    
    return filelist

def get_midframe_time(mlist: list)->np.ndarray:
    tlist = list(zip(*mlist))
    
    # Get frame time info
    _,frameidx = np.unique(tlist[1], return_index=True)
    FrameTimesStart = np.array(tlist[2])[frameidx]
    FrameDuration = np.array(tlist[3])[frameidx]

    # Handle bug in Siemens DICOM meta data
    if FrameTimesStart[0] == -1:
        print(f'Correcting Siemens Bug')
        FrameTimesStart += 1
    
    # Returning MidFrameTime
    return FrameTimesStart+FrameDuration/2

def dcminfo(dcmfile: Path)->list:
    """ Read selected meta tags from DICOM file
    Enhanced DICOM: https://www.dicomstandard.org/News-dir/ftsup/docs/sups/sup117.pdf

    Args:
        dcmfile: Absolute path of DICOM file.

    Returns:
        List containing meta data of selected tags.
    """
    # Read DICOM meta data
    ds = dcmread(dcmfile, stop_before_pixels=True)

    # Assert data is Dynamic PET data
    # assert True

    # Number Of Slices (0054, 0081)
    NumberOfSlices = ds.get('NumberOfSlices',0)

    # FROM DICOM MANUAL
    # ImageIndex = ((Time Slice Index - 1) * (NumberOfSlices)) + Slice Index
    Frame, Slice = divmod(ds.ImageIndex-1,NumberOfSlices)
    # Change from zero idx to 1 idx
    Frame += 1
    Slice += 1

    # Get actual frame duration in milliseconds (0018,1242)
    ActualFrameDuration = ds.get('ActualFrameDuration',0)

    # Get Frame Reference Time in milliseconds (0054, 1300)
    FrameReferenceTime = ds.get('FrameReferenceTime',0)

    # Compute FrameTimeStart in seconds
    FrameTimeStart = round((FrameReferenceTime-ActualFrameDuration/2)*1e-3)
    
    # Convert Actual Frame Duration to seconds and possibly int
    ActualFrameDuration *= 1e-3 
    if ActualFrameDuration.is_integer():
        ActualFrameDuration = int(ActualFrameDuration)
    
    # Return list of selected tags
    return [
        Slice,
        Frame,
        FrameTimeStart,
        ActualFrameDuration,
        NumberOfSlices,
        ds.get('ImagePositionPatient',[]),
        str(dcmfile)
    ]

def change_array(mlistrow):
    """ Change Array updates the RawArray in buffer with pixel data
    Args:
        mlistrow: list containing needed DICOM tags
    """

    dcmfile = mlistrow[-1]

    # Read DICOM file including pixel array
    ds = dcmread(dcmfile, stop_before_pixels=False)

    # Get Frame and Slice Number from Imageindex
    NumberOfSlices = mlistrow[-3]

    # Convert from 1-indexing to 0-indexing
    Slice = mlistrow[0]-1
    Frame = mlistrow[1]-1

    # Update data array in buffer with scaled pixel_array
    X_np = np.frombuffer(array, dtype='float64')
    nvoxels = ds.Columns*ds.Rows
    X_np[nvoxels*(Slice+(Frame*NumberOfSlices)):nvoxels*(Slice+1+(Frame*NumberOfSlices))] = ds.pixel_array.ravel(order='F')*ds.get('RescaleSlope',1) + ds.get('RescaleIntercept',0)

def compute_affine(ImageOrientationPatient:np.ndarray, FirstImagePositionPatient:np.ndarray, LastImagePositionPatient:np.ndarray, NumberOfSlices:int = 1, PixelSpacing:np.ndarray = [1, 1], SliceThickness: float = 1.0)->np.ndarray:
    """ Compute affine matrix in NIfTI convention (RAS+)
    Args:
        ImageOrientationPatient:
        FirstImagePositionPatient:
        LastImagePositionPatient:
        NumberOfSlices:
        PixelSpacing:
        SliceThickness:

    Returns:
        4x4 Affine matrix in NIfTI convention.
    """
    F11, F21, F31 = ImageOrientationPatient[3:]
    F12, F22, F32 = ImageOrientationPatient[:3]

    n = np.cross(ImageOrientationPatient[3:],ImageOrientationPatient[:3])

    dr, dc = PixelSpacing
    ds = SliceThickness
    Sx, Sy, Sz = FirstImagePositionPatient

    if NumberOfSlices == 1:
        # Single case
        k1, k2, k3 = n * ds
    else:
        # Multi case
        k1, k2, k3 = (LastImagePositionPatient-FirstImagePositionPatient)/(NumberOfSlices-1)

    patient_to_tal   = np.diag([-1, -1, 1, 1,])
    dicom_to_patient = np.array(
        [
        [F11 * dr, F12 * dc, k1, Sx],
        [F21 * dr, F22 * dc, k2, Sy],
        [F31 * dr, F32 * dc, k3, Sz],
        [0, 0, 0, 1]
        ]
    )

    return np.matmul(patient_to_tal,dicom_to_patient)