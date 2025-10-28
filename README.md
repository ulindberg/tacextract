# TAC Extract

## Installation
To install this library
```cmd
pip install git+https://github.com/ulindberg/tacextract.git
```

## Usage
```cmd
from tacextract import main
from pathlib import Path

main(Path(".data/dicom_series_folder"), Path(".data/TotalSegmentator.nii.gz"), Path(".data/"))
```