""" Example use """
from tacextract.utils import create_filelist_from_series_path, dcminfo, compute_affine, change_array, initpool
from tacextract.totalsegmentator import get_regionidx
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from multiprocessing import Pool, RawArray, cpu_count
from pydicom import dcmread
from tqdm import tqdm
from matplotlib import pyplot as plt

def main(dcm_path: Path, totalsegmentatorfile: Path, outdir: Path, region: str = 'brain', threads: int = 2*cpu_count()//3):
    ncpus = cpu_count()

    # Create a list of all DICOM files in Series
    filelist  = create_filelist_from_series_path(dcm_path)

    nfiles = len(filelist)
    if nfiles == 0:
        raise Exception('No DICOM files found in given path')
    print(f'Found {nfiles} DICOM files in {dcm_path}')

    # Read meta data into mlist from all files in filelist
    print('Reading selected meta data')
    print(f' Using {threads} of {ncpus} CPU cores')
    with Pool(processes=threads) as pool:
        mlist = pool.map(dcminfo, filelist)
    
    tlist = list(zip(*mlist))
    
    # Assert that all entries in the mlist - columns Plane and Frame are unique
    assert np.unique(np.array(tlist[:2]).T,axis=0).shape[0] == nfiles

    # Read first DICOM file from list
    ds = dcmread(filelist[0], stop_before_pixels=True)

    ImageOrientationPatient = ds.get('ImageOrientationPatient')
    PixelSpacing = ds.get('PixelSpacing')
    SliceThickness = ds.get('SliceThickness')
    voxdim = list(PixelSpacing) + [SliceThickness]

    ## Temporal info
    # Get frame time info
    _,frameidx = np.unique(tlist[1], return_index=True)
    FrameTimesStart = np.array(tlist[2])[frameidx]
    FrameDuration = np.array(tlist[3])[frameidx]

    # Handle bug in Siemens DICOM meta data
    if FrameTimesStart[0] == -1:
        print(f'Correcting Siemens Bug')
        FrameTimesStart += 1
    
    # Compute MidFrameTime
    MidFrameTime = FrameTimesStart+FrameDuration/2
    NumberOfFrames = MidFrameTime.size

    ## Slice info
    slices,sliceidx = np.unique(tlist[0], return_index=True)
    ImagePositionPatient = np.array(tlist[-2])[sliceidx]
    NumberOfSlices = slices.size
    
    # Compute Affine matrix in NIfTI convention
    affine = compute_affine(
        ImageOrientationPatient, # Always [1,0,0,0,1,0]
        ImagePositionPatient[0],
        ImagePositionPatient[-1],
        NumberOfSlices, 
        PixelSpacing, # Assume constant for all slices
        SliceThickness # Assume constant for all slices
    )

    imadim = np.array([ds.get('Columns'), ds.get('Rows'), NumberOfSlices, NumberOfFrames])

    # Load and resample totalsegmentator mask to dynamic pet
    totalseg = nib.load(totalsegmentatorfile)
    totalseg = resample_from_to(totalseg,(imadim[:3],affine),order=0)

    # Get label from region
    label = get_regionidx(region)
    seg = totalseg.get_fdata()==label
    seg = seg > 0
  
    ## Create Bounding Box
    # Step 1: Get indices of True values
    indices = np.argwhere(seg)

    # Step 2: Find min and max indices along each axis (x, y, z)
    xmin, ymin, zmin = indices.min(axis=0)
    xmax, ymax, zmax = indices.max(axis=0)

    # Update Number of Slices to selected
    NumberOfSlices = int(1+zmax-zmin)

    # Select only required slices, subtract zmin from slice (column 0) and replace NumberOfSlices (column -3)  
    mlist = [[x[0]-int(zmin)] + x[1:-3] + [NumberOfSlices] + x[-2:] for x in mlist if zmin+1 <= x[0] <= zmax+1]

    # Allocate array in buffer and read Data
    X_shape = (ds.Columns, ds.Rows, NumberOfSlices, NumberOfFrames)
    print(f'Allocating data array: ({X_shape}) (float64)')
    X = RawArray('d', X_shape[0] * X_shape[1] * X_shape[2] * X_shape[3])

    print(f'Reading data using {threads} processes')
    with Pool(processes=threads, initializer=initpool, initargs=(X,)) as pool, tqdm(total=len(mlist)) as pbar:
        for _ in pool.imap(change_array, mlist):
            pbar.update()
            pbar.refresh()
    
    # Get Data Array from buffer
    deck = np.frombuffer(X).reshape(X_shape, order='F')

    img_masked = deck[seg[...,zmin:zmax+1],:]
    tac_mean = img_masked.mean(axis=0)
    tac_std = img_masked.std(axis=0)
    n_voxels = np.array([seg.sum()]*len(tac_mean))

    # Create Maximum intensity projections in Saggital and Coronal views
    pad = 5
    # Weighted Average (FrameDuration)
    background1 = np.average(deck[xmin-pad:xmax+1+pad,ymin-pad:ymax+1+pad,...], axis=-1, weights=FrameDuration).max(axis=0).transpose((1,0))
    background2 = np.average(deck[xmin-pad:xmax+1+pad,ymin-pad:ymax+1+pad,...], axis=-1, weights=FrameDuration).max(axis=1).transpose((1,0))
    background = np.concatenate((background1, background2), axis=1)
    # Segmentation
    overlay1 = seg[xmin-pad:xmax+1+pad,ymin-pad:ymax+1+pad,zmin:zmax+1].max(axis=0).transpose((1,0))
    overlay2 = seg[xmin-pad:xmax+1+pad,ymin-pad:ymax+1+pad,zmin:zmax+1].max(axis=1).transpose((1,0))
    overlay = np.concatenate((overlay1, overlay2), axis=1)

    # Check if slice orientation should be inverted for showing
    if nib.orientations.aff2axcodes(affine)[-1] == 'S':
        print(f"Flipping slice direction for displaying")
        background = background[::-1,:]
        overlay = overlay[::-1,:]

    # Create figures
    fig, ax = plt.subplots()
    ax.imshow(background, vmin=0, vmax=np.percentile(background[overlay],99), cmap='gray_r', aspect=voxdim[2]/voxdim[1])
    ax.contour(overlay, levels=[0.5], colors='r')
    ax.axis('off')
    figfile = outdir / f'label-{region}_rec-mean_mip.png'
    plt.savefig(figfile,dpi=300, bbox_inches='tight')
    print(figfile)
    plt.close(fig)
    
    plt.figure()
    plt.plot(MidFrameTime,tac_mean,'.-',label=region)
    plt.xlabel("time [s]")
    plt.ylabel(f"Concentraion [{ds.Units}]")
    plt.legend()
    plt.title("Time-activity curve")
    figfile = outdir / f'label-{region}_rec-mean_tac.png'
    plt.savefig(figfile,dpi=300, bbox_inches='tight')
    print(figfile)