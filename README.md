# TAC Extract
Extract Time Activity Curves from Dynamic PET DICOM data.

> Only works for PET/CT/MR data acquired on the same scanner as reslicing is done based on scanner coordinates.

## Installation
To install this library
```cmd
pip install git+https://github.com/ulindberg/tacextract.git
```

## Usage
```python
from tacextract import main
from pathlib import Path

main(Path(".data/dicom_series_folder"), Path(".data/TotalSegmentator.nii.gz"), Path(".data/"), region="brain")
```